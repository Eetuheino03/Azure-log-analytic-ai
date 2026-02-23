import argparse
import csv
import inspect
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for binary SIEM classification")
    parser.add_argument("--model_name", type=str, default="HassanShehata/logem")
    parser.add_argument("--dataset_csv", type=str, default="data/processed/training_dataset_binary.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/logem-binary-lora")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
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
            if not row.get("text"):
                continue
            rows.append(row)
    return rows


def split_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split", "train")
        if split not in out:
            split = "train"
        out[split].append(row)
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


def tokenize_row(row: Dict[str, str], cfg: TokenizeConfig) -> Dict[str, List[int]]:
    prompt = build_prompt(row["text"])
    completion = build_completion(row["binary_label"])

    prompt_ids = cfg.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = cfg.tokenizer(
        prompt + completion + (cfg.tokenizer.eos_token or ""),
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.max_length,
    )["input_ids"]

    attention_mask = [1] * len(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = full_ids.copy()
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def print_distribution(name: str, rows: List[Dict[str, str]]) -> None:
    counts = Counter([r.get("binary_label", "missing") for r in rows])
    print(f"{name} count: {len(rows)}")
    print(f"{name} label distribution: {dict(counts)}")


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": 20,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "dataloader_num_workers": 0,
        "load_best_model_at_end": False,
        "seed": args.seed,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def print_trainable_stats(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")


def main() -> int:
    args = parse_args()

    if args.no_balance_train:
        args.balance_train = False
    if args.no_use_4bit:
        args.use_4bit = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_path = Path(args.dataset_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    rows = load_rows(csv_path)
    splits = split_rows(rows)

    train_rows = splits["train"]
    if args.balance_train:
        train_rows = balance_binary(train_rows, args.seed)

    val_rows = splits["val"]

    print_distribution("train", train_rows)
    print_distribution("val", val_rows)

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

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except ValueError as exc:
        message = str(exc)
        if "model type `qwen3`" in message:
            raise RuntimeError(
                "Failed to load qwen3 model. Upgrade transformers to latest main branch:\n"
                "pip install --upgrade git+https://github.com/huggingface/transformers.git"
            ) from exc
        raise

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    has_existing_peft = hasattr(model, "peft_config") and bool(getattr(model, "peft_config", {}))
    if has_existing_peft:
        # This checkpoint already ships with LoRA weights. Continue training the existing adapter.
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

    token_cfg = TokenizeConfig(tokenizer=tokenizer, max_length=args.max_length)
    train_raw = Dataset.from_list(train_rows)
    val_raw = Dataset.from_list(val_rows)
    train_ds = train_raw.map(lambda r: tokenize_row(r, token_cfg), remove_columns=train_raw.column_names)
    val_ds = val_raw.map(lambda r: tokenize_row(r, token_cfg), remove_columns=val_raw.column_names)

    training_args = build_training_args(args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
