
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import socket
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

CPU_PROFILE_DEFAULTS = {
    "cpu_fast": {"max_length": 256, "batch_size": 1, "gradient_accumulation_steps": 32, "num_train_epochs": 1.0, "eval_during_train": False, "save_steps": 1000},
    "cpu_balanced": {"max_length": 384, "batch_size": 1, "gradient_accumulation_steps": 32, "num_train_epochs": 1.5, "eval_during_train": True, "save_steps": 800},
    "cpu_quality": {"max_length": 512, "batch_size": 1, "gradient_accumulation_steps": 48, "num_train_epochs": 2.0, "eval_during_train": True, "save_steps": 600},
}

GPU_PROFILE_DEFAULTS = {
    "gpu_fast": {"max_length": 256, "batch_size": 4, "gradient_accumulation_steps": 8, "num_train_epochs": 1.0, "eval_during_train": True, "save_steps": 1000},
    "gpu_balanced": {"max_length": 384, "batch_size": 4, "gradient_accumulation_steps": 8, "num_train_epochs": 1.5, "eval_during_train": True, "save_steps": 800},
    "gpu_quality": {"max_length": 512, "batch_size": 2, "gradient_accumulation_steps": 16, "num_train_epochs": 2.0, "eval_during_train": True, "save_steps": 600},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for binary SIEM classification (GPU-first with CPU fallback)")
    p.add_argument("--model_name", type=str, default="HassanShehata/logem")
    p.add_argument("--dataset_csv", type=str, default="data/processed/training_dataset_binary.csv")
    p.add_argument("--output_dir", type=str, default="outputs/logem-binary-lora")

    p.add_argument("--profile", type=str, choices=["auto", "gpu_fast", "gpu_balanced", "gpu_quality", "cpu_fast", "cpu_balanced", "cpu_quality"], default="auto")
    p.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--precision", type=str, choices=["auto", "fp16", "bf16", "fp32"], default="auto")

    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--cpu_threads", type=int, default=None)
    p.add_argument("--interop_threads", type=int, default=1)
    p.add_argument("--tokenize_workers", type=int, default=None)

    p.add_argument("--cache_tokenized", action="store_true")
    p.add_argument("--no_cache_tokenized", action="store_true")
    p.add_argument("--eval_during_train", action="store_true")
    p.add_argument("--no_eval_during_train", action="store_true")
    p.add_argument("--run_lock", action="store_true")
    p.add_argument("--no_run_lock", action="store_true")
    p.set_defaults(cache_tokenized=True, run_lock=True, eval_during_train=None)

    p.add_argument("--eval_subset_size", type=int, default=512)
    p.add_argument("--eval_samples_final", type=int, default=1000)
    p.add_argument("--max_train_rows", type=int, default=None)

    p.add_argument("--balance_train", action="store_true")
    p.add_argument("--no_balance_train", action="store_true")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--no_use_4bit", action="store_true")
    p.set_defaults(balance_train=True, use_4bit=True)

    p.add_argument("--report_path", type=str, default=None)
    p.add_argument("--quality_gate", action="store_true")
    p.add_argument("--no_quality_gate", action="store_true")
    p.add_argument("--min_f1_suspicious", type=float, default=0.80)
    p.add_argument("--min_recall_suspicious", type=float, default=0.80)
    p.set_defaults(quality_gate=True)
    return p.parse_args()


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
    if args.no_quality_gate:
        args.quality_gate = False


def default_cpu_threads() -> int:
    logical = os.cpu_count() or 1
    return int(min(24, max(8, math.floor(logical * 0.5))))


def default_tokenize_workers(cpu_threads: int) -> int:
    return int(min(12, max(2, math.floor(cpu_threads / 2))))


def resolve_runtime(args: argparse.Namespace) -> Dict[str, object]:
    cuda_available = torch.cuda.is_available()
    device = "cuda" if (args.device == "auto" and cuda_available) else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    profile = args.profile
    if profile == "auto":
        profile = "gpu_balanced" if device == "cuda" else "cpu_balanced"

    if args.precision == "auto":
        precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device == "cuda" else "fp32")
    else:
        precision = args.precision

    return {"device": device, "profile": profile, "precision": precision, "use_4bit": bool(args.use_4bit and device == "cuda"), "fallback_events": []}


def apply_profile_defaults(args: argparse.Namespace, runtime: Dict[str, object]) -> None:
    prof = GPU_PROFILE_DEFAULTS[str(runtime["profile"])] if str(runtime["profile"]).startswith("gpu_") else CPU_PROFILE_DEFAULTS[str(runtime["profile"])]
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


def switch_runtime_to_cpu(args: argparse.Namespace, runtime: Dict[str, object], reason: str) -> None:
    runtime["fallback_events"].append({"event": "gpu_to_cpu_fallback", "reason": reason, "time": int(time.time())})
    runtime["device"] = "cpu"
    runtime["precision"] = "fp32"
    runtime["use_4bit"] = False
    if args.profile == "auto":
        runtime["profile"] = "cpu_balanced"


def configure_runtime_threads(args: argparse.Namespace, runtime: Dict[str, object]) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if runtime["device"] == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.interop_threads)

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

    for text, label in zip(batch["text"], batch["binary_label"]):
        prompt = build_prompt(text)
        completion = build_completion(label)
        prompt_ids = cfg.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = cfg.tokenizer(prompt + completion + (cfg.tokenizer.eos_token or ""), add_special_tokens=False, truncation=True, max_length=cfg.max_length)["input_ids"]

        attention_mask = [1] * len(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        label_ids = full_ids.copy()
        for i in range(prompt_len):
            label_ids[i] = -100

        input_ids_out.append(full_ids)
        attention_mask_out.append(attention_mask)
        labels_out.append(label_ids)

    return {"input_ids": input_ids_out, "attention_mask": attention_mask_out, "labels": labels_out}


def cache_key(args: argparse.Namespace, runtime: Dict[str, object], csv_path: Path) -> str:
    stat = csv_path.stat()
    payload = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "profile": runtime["profile"],
        "dataset_path": str(csv_path.resolve()),
        "dataset_mtime": stat.st_mtime,
        "dataset_size": stat.st_size,
        "balance_train": args.balance_train,
        "seed": args.seed,
        "max_train_rows": args.max_train_rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def tokenize_train_or_load(args: argparse.Namespace, runtime: Dict[str, object], train_rows: List[Dict[str, str]], tokenizer: AutoTokenizer, csv_path: Path) -> Dataset:
    token_cfg = TokenizeConfig(tokenizer=tokenizer, max_length=args.max_length)

    if not args.cache_tokenized:
        train_raw = Dataset.from_list(train_rows)
        return train_raw.map(lambda b: tokenize_batch(b, token_cfg), batched=True, batch_size=512, num_proc=max(1, args.tokenize_workers), remove_columns=train_raw.column_names, desc="Tokenizing train")

    key = cache_key(args, runtime, csv_path)
    base = Path(".cache") / "tokenized" / key
    train_dir = base / "train"

    if train_dir.exists():
        print(f"Using tokenization cache: {base}")
        return load_from_disk(str(train_dir))

    train_raw = Dataset.from_list(train_rows)
    train_ds = train_raw.map(lambda b: tokenize_batch(b, token_cfg), batched=True, batch_size=512, num_proc=max(1, args.tokenize_workers), remove_columns=train_raw.column_names, desc="Tokenizing train")
    base.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(train_dir))
    print(f"Saved tokenization cache: {base}")
    return train_ds


def tokenize_rows(rows: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int, workers: int, desc: str) -> Dataset:
    raw = Dataset.from_list(rows)
    if len(raw) == 0:
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})
    token_cfg = TokenizeConfig(tokenizer=tokenizer, max_length=max_length)
    return raw.map(lambda b: tokenize_batch(b, token_cfg), batched=True, batch_size=512, num_proc=max(1, workers), remove_columns=raw.column_names, desc=desc)


def estimate_steps(num_train_rows: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    effective_batch = max(1, batch_size * grad_accum)
    return max(1, int(math.ceil(math.ceil(num_train_rows / effective_batch) * epochs)))


def estimate_tokens_per_epoch(tokenized_train: Dataset) -> int:
    return sum(len(x) for x in tokenized_train["input_ids"])


def sample_rows(rows: List[Dict[str, str]], n: int, seed: int) -> List[Dict[str, str]]:
    if n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)

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
        total += float(log_probs[prev_pos, full_ids[pos]].item())
        count += 1
    return total / max(1, count)


def evaluate_binary_classifier(model, tokenizer, rows: List[Dict[str, str]], max_length: int) -> Dict[str, float]:
    if not rows:
        return {"samples": 0, "accuracy": 0.0}
    device = next(model.parameters()).device

    tp = tn = fp = fn = 0
    for row in rows:
        prompt = build_prompt(row["text"])
        normal_score = score_completion_logprob(model, tokenizer, prompt, build_completion("normal"), max_length, device)
        suspicious_score = score_completion_logprob(model, tokenizer, prompt, build_completion("suspicious"), max_length, device)
        pred = "suspicious" if suspicious_score > normal_score else "normal"
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
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "samples": tp + tn + fp + fn,
        "accuracy": (tp + tn) / total,
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


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(lock_path: Path, runtime: Dict[str, object]) -> None:
    hostname = socket.gethostname()
    pid = os.getpid()

    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        lock_pid = int(data.get("pid", -1))
        lock_host = str(data.get("hostname", ""))
        lock_time = data.get("start_time", "unknown")
        if lock_host == hostname and is_process_alive(lock_pid):
            raise RuntimeError(
                f"Another training run is active for this output_dir (pid={lock_pid}, host={lock_host}, start_time={lock_time}). "
                f"If stale, remove {lock_path} and retry."
            )
        print(f"Stale lock found, replacing: {lock_path}")

    payload = {
        "pid": pid,
        "hostname": hostname,
        "start_time": int(time.time()),
        "resolved_device": runtime.get("device"),
        "argv": sys.argv,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def release_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def can_fallback_to_cpu(exc: Exception) -> bool:
    msg = str(exc).lower()
    for pattern in ["cuda", "cudnn", "out of memory", "driver", "nvml", "device-side"]:
        if pattern in msg:
            return True
    return False


def get_adapter_base_model(repo_id: str) -> Optional[str]:
    # If repo contains a PEFT adapter, load base model explicitly and attach adapter manually.
    try:
        path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json")
    except Exception:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    base = data.get("base_model_name_or_path")
    if isinstance(base, str) and base.strip():
        return base.strip()
    return None

def load_model_with_fallback(args: argparse.Namespace, runtime: Dict[str, object]):
    adapter_base = get_adapter_base_model(args.model_name)

    def _load_base(device: str, precision: str, use_4bit: bool):
        quant_config = None
        kwargs = {"trust_remote_code": True}

        if device == "cuda" and use_4bit:
            compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = "auto"
        else:
            if device == "cuda":
                kwargs["device_map"] = "auto"
                if precision == "bf16":
                    kwargs["dtype"] = torch.bfloat16
                elif precision == "fp16":
                    kwargs["dtype"] = torch.float16
                else:
                    kwargs["dtype"] = torch.float32
            else:
                kwargs["dtype"] = torch.float32

        base_model_id = adapter_base or args.model_name
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)
        return model, quant_config

    def _load(device: str, precision: str, use_4bit: bool):
        base_model, quant_config = _load_base(device, precision, use_4bit)
        if adapter_base:
            # Avoid transformers auto-adapter loading path that can fail with quantized repos.
            if device == "cpu" and "bnb-4bit" in adapter_base.lower():
                raise RuntimeError(
                    "Adapter base model is 4-bit quantized and not suitable for CPU fallback. "
                    "Run on CUDA GPU or choose a non-4bit base model."
                )
            peft_model = PeftModel.from_pretrained(base_model, args.model_name, is_trainable=True)
            return peft_model, quant_config
        return base_model, quant_config

    if runtime["device"] == "cuda":
        try:
            return _load("cuda", str(runtime["precision"]), bool(runtime["use_4bit"]))
        except Exception as exc:
            if can_fallback_to_cpu(exc):
                print(f"GPU init failed, retrying on CPU. reason={exc}")
                switch_runtime_to_cpu(args, runtime, reason=str(exc))
                configure_runtime_threads(args, runtime)
                return _load("cpu", "fp32", False)
            raise

    return _load("cpu", "fp32", False)


def build_training_args(args: argparse.Namespace, runtime: Dict[str, object], eval_dataset: Optional[Dataset]) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": 10,
        "logging_first_step": True,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": runtime["device"] == "cuda" and runtime["precision"] == "fp16",
        "bf16": runtime["device"] == "cuda" and runtime["precision"] == "bf16",
        "dataloader_num_workers": max(1, min(4, args.tokenize_workers)),
        "load_best_model_at_end": False,
        "seed": args.seed,
        "disable_tqdm": False,
    }

    if args.eval_during_train and eval_dataset is not None and len(eval_dataset) > 0 and args.eval_steps > 0:
        kwargs["eval_steps"] = args.eval_steps

    sig = inspect.signature(TrainingArguments.__init__)
    strategy = "epoch" if args.eval_during_train else "no"
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = strategy
    else:
        kwargs["evaluation_strategy"] = strategy

    return TrainingArguments(**kwargs)


def build_quality_gate(test_metrics: Dict[str, float], args: argparse.Namespace) -> Dict[str, object]:
    if not args.quality_gate:
        return {"enabled": False, "pass": True, "reasons": []}

    reasons: List[str] = []
    if float(test_metrics.get("f1_suspicious", 0.0)) < args.min_f1_suspicious:
        reasons.append(f"f1_suspicious<{args.min_f1_suspicious}")
    if float(test_metrics.get("recall_suspicious", 0.0)) < args.min_recall_suspicious:
        reasons.append(f"recall_suspicious<{args.min_recall_suspicious}")

    return {
        "enabled": True,
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "thresholds": {
            "min_f1_suspicious": args.min_f1_suspicious,
            "min_recall_suspicious": args.min_recall_suspicious,
        },
    }


def quality_label(test_metrics: Dict[str, float]) -> str:
    f1 = float(test_metrics.get("f1_suspicious", 0.0))
    recall = float(test_metrics.get("recall_suspicious", 0.0))
    if f1 >= 0.80 and recall >= 0.80:
        return "Hyva"
    if f1 < 0.70 or recall < 0.70:
        return "Heikko"
    return "Kaytettava varoen"


def write_reports(args: argparse.Namespace, runtime: Dict[str, object], history: List[Dict[str, object]], val_metrics: Dict[str, float], test_metrics: Dict[str, float], gate: Dict[str, object]) -> None:
    report_path = Path(args.report_path) if args.report_path else Path(args.output_dir) / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "runtime": {
            "device": runtime["device"],
            "profile": runtime["profile"],
            "precision": runtime["precision"],
            "use_4bit": runtime["use_4bit"],
            "fallback_events": runtime.get("fallback_events", []),
        },
        "config": {
            "model_name": args.model_name,
            "dataset_csv": args.dataset_csv,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_length": args.max_length,
            "cpu_threads": args.cpu_threads,
            "interop_threads": args.interop_threads,
            "tokenize_workers": args.tokenize_workers,
            "eval_during_train": args.eval_during_train,
            "eval_subset_size": args.eval_subset_size,
            "eval_samples_final": args.eval_samples_final,
        },
        "train_history": history,
        "final_metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "confusion_matrix_test": {"tp": test_metrics.get("tp", 0), "tn": test_metrics.get("tn", 0), "fp": test_metrics.get("fp", 0), "fn": test_metrics.get("fn", 0)},
        },
        "quality_gate": gate,
        "quality_interpretation": quality_label(test_metrics),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_path = Path(args.output_dir) / "metrics_summary.txt"
    summary = [
        f"runtime.device={runtime['device']}",
        f"runtime.profile={runtime['profile']}",
        f"runtime.precision={runtime['precision']}",
        f"runtime.use_4bit={runtime['use_4bit']}",
        f"test.accuracy={test_metrics.get('accuracy', 0.0):.4f}",
        f"test.f1_suspicious={test_metrics.get('f1_suspicious', 0.0):.4f}",
        f"test.recall_suspicious={test_metrics.get('recall_suspicious', 0.0):.4f}",
        f"test.precision_suspicious={test_metrics.get('precision_suspicious', 0.0):.4f}",
        f"test.confusion_matrix=TP:{test_metrics.get('tp', 0)} TN:{test_metrics.get('tn', 0)} FP:{test_metrics.get('fp', 0)} FN:{test_metrics.get('fn', 0)}",
        f"quality_gate.enabled={gate.get('enabled')}",
        f"quality_gate.pass={gate.get('pass')}",
        f"quality_gate.reasons={gate.get('reasons')}",
        f"quality_interpretation={quality_label(test_metrics)}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

def main() -> int:
    args = parse_args()
    resolve_flags(args)

    runtime = resolve_runtime(args)
    apply_profile_defaults(args, runtime)
    configure_runtime_threads(args, runtime)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logical_cpu = os.cpu_count() or 1
    print(
        f"runtime: device={runtime['device']} profile={runtime['profile']} precision={runtime['precision']} "
        f"quantization={'4bit' if runtime['use_4bit'] else 'none'}"
    )
    print(
        f"logical_cpus={logical_cpu} cpu_threads={args.cpu_threads} interop_threads={args.interop_threads} "
        f"tokenize_workers={args.tokenize_workers}"
    )

    csv_path = Path(args.dataset_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    lock_path = Path(args.output_dir) / ".train.lock"
    if args.run_lock:
        acquire_lock(lock_path, runtime)

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
        print(f"train-time eval: {'enabled' if args.eval_during_train else 'disabled'} (rows={len(val_rows_for_train)})")

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

        model, quant_config = load_model_with_fallback(args, runtime)
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
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
            else:
                print_trainable_stats(model)

        train_ds = tokenize_train_or_load(args, runtime, train_rows, tokenizer, csv_path)
        val_ds_train = tokenize_rows(val_rows_for_train, tokenizer=tokenizer, max_length=args.max_length, workers=args.tokenize_workers, desc="Tokenizing eval subset")

        tokens_epoch = estimate_tokens_per_epoch(train_ds)
        print(f"estimated train tokens/epoch: {tokens_epoch:,}")

        eval_ds_for_train = val_ds_train if args.eval_during_train else None
        print(f"train-time eval overhead: {'on' if args.eval_during_train else 'off'} (eval rows={len(eval_ds_for_train) if eval_ds_for_train is not None else 0})")

        training_args = build_training_args(args, runtime, eval_ds_for_train)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds_for_train, data_collator=data_collator)

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        model.eval()
        final_val_rows = sample_rows(val_rows, args.eval_samples_final, args.seed)
        final_test_rows = sample_rows(splits["test"], args.eval_samples_final, args.seed)
        val_metrics = evaluate_binary_classifier(model, tokenizer, final_val_rows, args.max_length)
        test_metrics = evaluate_binary_classifier(model, tokenizer, final_test_rows, args.max_length)

        gate = build_quality_gate(test_metrics, args)
        write_reports(args, runtime, trainer.state.log_history, val_metrics, test_metrics, gate)

        print("validation metrics:", val_metrics)
        print("test metrics:", test_metrics)
        print("quality gate:", gate)
        print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")
        print(f"Saved report to: {args.report_path or str(Path(args.output_dir) / 'training_report.json')}")

    finally:
        if args.run_lock:
            release_lock(lock_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
