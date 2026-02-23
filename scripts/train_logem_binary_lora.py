import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import socket
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

PROFILE_DEFAULTS = {
    "cpu_fast": {
        "max_length": 256,
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "num_train_epochs": 1.0,
        "eval_during_train": False,
        "save_steps": 1000,
    },
    "cpu_balanced": {
        "max_length": 384,
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "num_train_epochs": 1.5,
        "eval_during_train": True,
        "save_steps": 800,
    },
    "cpu_quality": {
        "max_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 48,
        "num_train_epochs": 2.0,
        "eval_during_train": True,
        "save_steps": 600,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for binary SIEM classification (CPU optimized)")
    parser.add_argument("--model_name", type=str, default="HassanShehata/logem")
    parser.add_argument("--dataset_csv", type=str, default="data/processed/training_dataset_binary.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/logem-binary-lora")

    parser.add_argument("--profile", type=str, choices=["cpu_fast", "cpu_balanced", "cpu_quality"], default="cpu_fast")

    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--interop_threads", type=int, default=1)
    parser.add_argument("--tokenize_workers", type=int, default=None)

    parser.add_argument("--cache_tokenized", action="store_true")
    parser.add_argument("--no_cache_tokenized", action="store_true")
    parser.add_argument("--eval_during_train", action="store_true")
    parser.add_argument("--no_eval_during_train", action="store_true")
    parser.add_argument("--run_lock", action="store_true")
    parser.add_argument("--no_run_lock", action="store_true")
    parser.set_defaults(cache_tokenized=True, run_lock=True, eval_during_train=None)

    parser.add_argument("--eval_subset_size", type=int, default=512)
    parser.add_argument("--eval_samples_final", type=int, default=1000)
    parser.add_argument("--max_train_rows", type=int, default=None)

    parser.add_argument("--balance_train", action="store_true")
    parser.add_argument("--no_balance_train", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--no_use_4bit", action="store_true")
    parser.set_defaults(balance_train=True, use_4bit=True)

    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("text"):
                rows.append(row)
    return rows


def split_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split", "train")
        out[split if split in out else "train"].append(row)
    return out


def balance_binary(rows: List[Dict[str, str]], seed: int) -> List[Dict[str, str]]:
    groups: Dict[str, List[Dict[str, str]]] = {"normal": [], "suspicious": []}
    for row in rows:
        label = row.get("binary_label", "")
        if label in groups:
            groups[label].append(row)

    if not groups["normal"] or not groups["suspicious"]:
        return rows

    random.seed(seed)
    target_count = min(len(groups["normal"]), len(groups["suspicious"]))
    sampled = random.sample(groups["normal"], target_count) + random.sample(groups["suspicious"], target_count)
    random.shuffle(sampled)
    return sampled


def print_distribution(name: str, rows: List[Dict[str, str]]) -> None:
    counts = Counter([r.get("binary_label", "missing") for r in rows])
    print(f"{name} count: {len(rows)}")
    print(f"{name} label distribution: {dict(counts)}")


def build_prompt(text: str) -> str:
    return (
        "You are a SOC analyst. Classify the following log as suspicious or normal. "
        "Reply using exactly one word: suspicious or normal.\n\n"
        f"Log:\n{text}\n\n"
        "Label:"
    )


def build_completion(label: str) -> str:
    return f" {label}"


@dataclass
class TokenizeConfig:
    tokenizer: AutoTokenizer
    max_length: int


def tokenize_batch(batch: Dict[str, List[str]], cfg: TokenizeConfig) -> Dict[str, List[List[int]]]:
    input_ids_out: List[List[int]] = []
    attention_mask_out: List[List[int]] = []
    labels_out: List[List[int]] = []

    texts = batch["text"]
    labels = batch["binary_label"]

    for text, label in zip(texts, labels):
        prompt = build_prompt(text)
        completion = build_completion(label)

        prompt_ids = cfg.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = cfg.tokenizer(
            prompt + completion + (cfg.tokenizer.eos_token or ""),
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_length,
        )["input_ids"]

        attention_mask = [1] * len(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        label_ids = full_ids.copy()
        for i in range(prompt_len):
            label_ids[i] = -100

        input_ids_out.append(full_ids)
        attention_mask_out.append(attention_mask)
        labels_out.append(label_ids)

    return {
        "input_ids": input_ids_out,
        "attention_mask": attention_mask_out,
        "labels": labels_out,
    }


def estimate_steps(num_train_rows: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    effective_batch = max(1, batch_size * grad_accum)
    steps_per_epoch = math.ceil(num_train_rows / effective_batch)
    return max(1, int(math.ceil(steps_per_epoch * epochs)))


def estimate_tokens_per_epoch(tokenized_train: Dataset) -> int:
    return sum(len(x) for x in tokenized_train["input_ids"])


def score_completion_logprob(model, tokenizer, prompt: str, completion: str, max_length: int, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    keep_prompt = max(1, max_length - len(completion_ids) - 1)
    prompt_ids = prompt_ids[-keep_prompt:]
    full_ids = prompt_ids + completion_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    if start >= len(full_ids):
        return float("-inf")

    total = 0.0
    count = 0
    for pos in range(start, len(full_ids)):
        prev_pos = pos - 1
        tok_id = full_ids[pos]
        total += float(log_probs[prev_pos, tok_id].item())
        count += 1
    return total / max(1, count)


def sample_rows(rows: List[Dict[str, str]], n: int, seed: int) -> List[Dict[str, str]]:
    if n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def evaluate_binary_classifier(model, tokenizer, rows: List[Dict[str, str]], max_length: int) -> Dict[str, float]:
    if not rows:
        return {"samples": 0, "accuracy": 0.0}

    device = next(model.parameters()).device
    labels = ["normal", "suspicious"]

    tp = tn = fp = fn = 0
    for row in rows:
        prompt = build_prompt(row["text"])
        scores = {
            label: score_completion_logprob(model, tokenizer, prompt, build_completion(label), max_length, device)
            for label in labels
        }
        pred = max(scores.items(), key=lambda x: x[1])[0]
        true = row["binary_label"]

        if true == "suspicious" and pred == "suspicious":
            tp += 1
        elif true == "normal" and pred == "normal":
            tn += 1
        elif true == "normal" and pred == "suspicious":
            fp += 1
        elif true == "suspicious" and pred == "normal":
            fn += 1

    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "samples": tp + tn + fp + fn,
        "accuracy": accuracy,
        "precision_suspicious": precision,
        "recall_suspicious": recall,
        "f1_suspicious": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_trainable_stats(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


def default_cpu_threads() -> int:
    logical = os.cpu_count() or 1
    return int(min(24, max(8, math.floor(logical * 0.5))))


def default_tokenize_workers(cpu_threads: int) -> int:
    return int(min(12, max(2, math.floor(cpu_threads / 2))))


def apply_profile_defaults(args: argparse.Namespace) -> None:
    prof = PROFILE_DEFAULTS[args.profile]

    if args.max_length is None:
        args.max_length = prof["max_length"]
    if args.batch_size is None:
        args.batch_size = prof["batch_size"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = prof["gradient_accumulation_steps"]
    if args.num_train_epochs is None:
        args.num_train_epochs = prof["num_train_epochs"]
    if args.save_steps is None:
        args.save_steps = prof["save_steps"]
    if args.eval_during_train is None:
        args.eval_during_train = prof["eval_during_train"]

    if args.cpu_threads is None:
        args.cpu_threads = default_cpu_threads()
    if args.tokenize_workers is None:
        args.tokenize_workers = default_tokenize_workers(args.cpu_threads)


def resolve_flags(args: argparse.Namespace) -> None:
    if args.no_balance_train:
        args.balance_train = False
    if args.no_use_4bit:
        args.use_4bit = False
    if args.no_cache_tokenized:
        args.cache_tokenized = False
    if args.no_run_lock:
        args.run_lock = False
    if args.no_eval_during_train:
        args.eval_during_train = False
    if args.eval_during_train:
        args.eval_during_train = True


def configure_cpu_runtime(args: argparse.Namespace) -> None:
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.interop_threads)


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(lock_path: Path) -> None:
    hostname = socket.gethostname()
    pid = os.getpid()

    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        lock_pid = int(data.get("pid", -1))
        lock_host = str(data.get("hostname", ""))
        if lock_host == hostname and is_process_alive(lock_pid):
            raise RuntimeError(
                f"Another training run is active for this output_dir (pid={lock_pid}, host={lock_host}). "
                f"Remove {lock_path} only if you are sure the process is stale."
            )
        print(f"Stale lock found, replacing: {lock_path}")

    payload = {
        "pid": pid,
        "hostname": hostname,
        "start_time": int(time.time()),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def release_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def cache_key(args: argparse.Namespace, csv_path: Path) -> str:
    stat = csv_path.stat()
    payload = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "profile": args.profile,
        "dataset_path": str(csv_path.resolve()),
        "dataset_mtime": stat.st_mtime,
        "dataset_size": stat.st_size,
        "balance_train": args.balance_train,
        "seed": args.seed,
        "max_train_rows": args.max_train_rows,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def tokenize_train_or_load(
    args: argparse.Namespace,
    train_rows: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    csv_path: Path,
) -> Dataset:
    token_cfg = TokenizeConfig(tokenizer=tokenizer, max_length=args.max_length)

    if not args.cache_tokenized:
        train_raw = Dataset.from_list(train_rows)
        train_ds = train_raw.map(
            lambda b: tokenize_batch(b, token_cfg),
            batched=True,
            batch_size=512,
            num_proc=max(1, args.tokenize_workers),
            remove_columns=train_raw.column_names,
            desc="Tokenizing train",
        )
        return train_ds

    key = cache_key(args, csv_path)
    base = Path(".cache") / "tokenized" / key
    train_dir = base / "train"

    if train_dir.exists():
        print(f"Using tokenization cache: {base}")
        return load_from_disk(str(train_dir))

    train_raw = Dataset.from_list(train_rows)
    train_ds = train_raw.map(
        lambda b: tokenize_batch(b, token_cfg),
        batched=True,
        batch_size=512,
        num_proc=max(1, args.tokenize_workers),
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )

    base.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(train_dir))
    print(f"Saved tokenization cache: {base}")
    return train_ds


def tokenize_rows(rows: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int, workers: int, desc: str) -> Dataset:
    token_cfg = TokenizeConfig(tokenizer=tokenizer, max_length=max_length)
    raw = Dataset.from_list(rows)
    if len(raw) == 0:
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
    return raw.map(
        lambda b: tokenize_batch(b, token_cfg),
        batched=True,
        batch_size=512,
        num_proc=max(1, workers),
        remove_columns=raw.column_names,
        desc=desc,
    )


def build_training_args(args: argparse.Namespace, eval_dataset: Optional[Dataset]) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "dataloader_num_workers": max(1, min(4, args.tokenize_workers)),
        "load_best_model_at_end": False,
        "seed": args.seed,
        "disable_tqdm": False,
    }

    if args.eval_during_train and eval_dataset is not None and len(eval_dataset) > 0 and args.eval_steps > 0:
        kwargs["eval_steps"] = args.eval_steps
    sig = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch" if args.eval_during_train else "no"
    else:
        kwargs["evaluation_strategy"] = "epoch" if args.eval_during_train else "no"

    return TrainingArguments(**kwargs)


def main() -> int:
    args = parse_args()
    resolve_flags(args)
    apply_profile_defaults(args)

    configure_cpu_runtime(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logical_cpu = os.cpu_count() or 1
    print(f"profile: {args.profile}")
    print(f"logical_cpus: {logical_cpu}, cpu_threads: {args.cpu_threads}, interop_threads: {args.interop_threads}, tokenize_workers: {args.tokenize_workers}")

    csv_path = Path(args.dataset_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    lock_path = Path(args.output_dir) / ".train.lock"
    if args.run_lock:
        acquire_lock(lock_path)

    try:
        rows = load_rows(csv_path)
        splits = split_rows(rows)

        train_rows = splits["train"]
        if args.balance_train:
            train_rows = balance_binary(train_rows, args.seed)
        if args.max_train_rows is not None and args.max_train_rows > 0:
            train_rows = train_rows[: args.max_train_rows]

        val_rows = splits["val"]
        val_rows_for_train = sample_rows(val_rows, args.eval_subset_size, args.seed) if args.eval_during_train else []

        print_distribution("train", train_rows)
        print_distribution("val", val_rows)
        if args.eval_during_train:
            print(f"train-time eval subset size: {len(val_rows_for_train)}")
        else:
            print("train-time eval: disabled")

        est = estimate_steps(len(train_rows), args.batch_size, args.gradient_accumulation_steps, args.num_train_epochs)
        eff_batch = args.batch_size * args.gradient_accumulation_steps
        print(f"estimated optimizer steps: {est} (epochs={args.num_train_epochs}, effective_batch={eff_batch})")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if "qwen3" not in CONFIG_MAPPING:
            raise RuntimeError(
                "Your transformers version does not support model_type='qwen3'. "
                "Upgrade first with: pip install --upgrade git+https://github.com/huggingface/transformers.git"
            )

        quant_config = None
        model_kwargs = {"trust_remote_code": True}
        if args.use_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)

        has_existing_peft = hasattr(model, "peft_config") and bool(getattr(model, "peft_config", {}))
        if has_existing_peft:
            for name, param in model.named_parameters():
                param.requires_grad = "lora_" in name
            print("Detected existing PEFT adapter in checkpoint. Reusing it for training.")
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
            else:
                print_trainable_stats(model)
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
            else:
                print_trainable_stats(model)

        train_ds = tokenize_train_or_load(args, train_rows, tokenizer, csv_path)
        val_ds_train = tokenize_rows(
            val_rows_for_train,
            tokenizer=tokenizer,
            max_length=args.max_length,
            workers=args.tokenize_workers,
            desc="Tokenizing eval subset",
        )

        tokens_epoch = estimate_tokens_per_epoch(train_ds)
        print(f"estimated train tokens/epoch: {tokens_epoch:,}")

        eval_ds_for_train = val_ds_train if args.eval_during_train else None
        print(
            f"train-time eval overhead: {'on' if args.eval_during_train else 'off'} "
            f"(eval rows={len(eval_ds_for_train) if eval_ds_for_train is not None else 0})"
        )

        training_args = build_training_args(args, eval_ds_for_train)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds_for_train,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        model.eval()
        final_val_rows = sample_rows(val_rows, args.eval_samples_final, args.seed)
        final_test_rows = sample_rows(splits["test"], args.eval_samples_final, args.seed)
        val_metrics = evaluate_binary_classifier(model, tokenizer, final_val_rows, args.max_length)
        test_metrics = evaluate_binary_classifier(model, tokenizer, final_test_rows, args.max_length)

        print("validation metrics:", val_metrics)
        print("test metrics:", test_metrics)
        print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")

    finally:
        if args.run_lock:
            release_lock(lock_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
