"""
Experiment 013: Fine-Tuned Small LLM with QLoRA

Fine-tunes Phi-3-mini-4k-instruct (3.8B) or Mistral-7B using QLoRA
4-bit quantization for CVR anomaly classification.

Requirements:
    pip install peft bitsandbytes trl accelerate

Hardware: RTX 4080 16GB sufficient for Phi-3-mini with 4-bit.

Usage:
    cd experiments/013_finetuned_llm
    python run.py
    python run.py --model mistralai/Mistral-7B-Instruct-v0.3
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.safety_metrics import compute_all_safety_metrics, format_metrics_table

console = Console()

LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]

PROMPT_TEMPLATE = """Classify the following cockpit voice recorder (CVR) communication sequence into one of four severity levels: NORMAL, EARLY_WARNING, ELEVATED, or CRITICAL.

CVR Sequence:
{text}

Classification:"""


def create_sequences_from_df(df, window_size=10, stride=5,
                              text_col="cvr_message", label_col="label",
                              case_col="case_id"):
    """Create formatted text sequences and labels."""
    texts, labels = [], []
    for case_id, group in df.groupby(case_col):
        group = group.sort_values(
            "turn_number" if "turn_number" in group.columns else group.index
        ).reset_index(drop=True)
        utterances = group[text_col].fillna("").tolist()
        case_labels = group[label_col].tolist()
        if len(utterances) < 3:
            continue
        for i in range(0, len(utterances) - window_size + 1, stride):
            window = utterances[i:i + window_size]
            combined = "\n".join(f"[{j+1}] {str(t)}" for j, t in enumerate(window))
            texts.append(combined)
            seq_label = case_labels[i + window_size - 1]
            if isinstance(seq_label, str):
                seq_label = LABEL_MAP.get(seq_label, 0)
            labels.append(seq_label)
    return texts, labels


def format_for_training(texts, labels):
    """Format data for instruction-tuning."""
    formatted = []
    for text, label in zip(texts, labels):
        label_name = LABEL_NAMES[label]
        formatted.append({
            "text": PROMPT_TEMPLATE.format(text=text) + f" {label_name}",
            "label": label,
            "label_name": label_name,
        })
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Override base model")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    console.print("\n[bold cyan]Experiment 013: Fine-Tuned LLM (QLoRA)[/bold cyan]")
    console.print("=" * 60)

    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    base_model = args.model or config["model"]["base_model"]
    console.print(f"[cyan]Base model: {base_model}[/cyan]")

    # Check dependencies
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Install: pip install peft bitsandbytes trl accelerate[/yellow]")
        return

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/cyan]")

    if device != "cuda":
        console.print("[red]QLoRA requires CUDA GPU. Exiting.[/red]")
        return

    # Load data
    data_path = PROJECT_ROOT / config["data"]["source"]
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    df = pd.read_csv(data_path)
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

    if df[label_col].dtype == object:
        df["label_id"] = df[label_col].map(LABEL_MAP)
    else:
        df["label_id"] = df[label_col]

    texts, labels = create_sequences_from_df(
        df, window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        text_col=text_col, label_col="label_id",
        case_col=config["data"]["case_id_column"],
    )
    labels = np.array(labels)
    console.print(f"[green]{len(texts):,} sequences[/green]")

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"], stratify=labels,
    )
    adj = config["data"]["val_split"] / (1 - config["data"]["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adj,
        random_state=config["data"]["random_seed"], stratify=y_temp,
    )
    console.print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Quantization config
    qlora_cfg = config["model"]["qlora"]
    quant_cfg = config["model"]["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    # Load model
    console.print(f"[cyan]Loading {base_model} with 4-bit quantization...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=4,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["lora_alpha"],
        lora_dropout=qlora_cfg["lora_dropout"],
        target_modules=qlora_cfg["target_modules"],
        bias=qlora_cfg["bias"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)[/green]")

    # Create datasets
    from torch.utils.data import Dataset as TorchDataset

    class CVRDataset(TorchDataset):
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.encodings = tokenizer(
                texts, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    max_len = config["model"]["max_length"]
    train_ds = CVRDataset(X_train, y_train, tokenizer, max_len)
    val_ds = CVRDataset(X_val, y_val, tokenizer, max_len)
    test_ds = CVRDataset(X_test, y_test, tokenizer, max_len)

    # Training arguments
    ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=config["training"]["max_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"] * 2,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
    )

    # Custom compute_metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    if not args.eval_only:
        console.print("\n[bold yellow]Starting QLoRA fine-tuning...[/bold yellow]")
        start = time.time()
        trainer.train()
        elapsed = time.time() - start
        console.print(f"[green]Training completed in {elapsed/60:.1f} minutes[/green]")

        # Save adapter
        model.save_pretrained(str(ckpt_dir / "best_adapter"))
        tokenizer.save_pretrained(str(ckpt_dir / "best_adapter"))

    # Evaluate on test
    console.print("\n[bold cyan]Evaluating on test set...[/bold cyan]")
    predictions = trainer.predict(test_ds)
    y_pred = np.argmax(predictions.predictions, axis=-1)

    metrics = compute_all_safety_metrics(y_test, y_pred)
    console.print(format_metrics_table(metrics))

    # Save
    np.save(output_dir / "y_true.npy", y_test)
    np.save(output_dir / "y_pred.npy", y_pred)

    results = {
        "experiment": config["experiment"],
        "base_model": base_model,
        "trainable_params": trainable,
        "total_params": total,
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
        "training_time_minutes": elapsed / 60 if not args.eval_only else None,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
