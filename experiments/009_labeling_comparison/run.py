"""
Experiment 009: Labeling Strategy Comparison

Main contribution #1: Compares three labeling strategies using the
same BERT+LSTM model architecture, isolating the effect of labeling:

1. Position-based: Labels from utterance position (last 5% = CRITICAL)
2. Content-based: Labels from LLM content annotation
3. Hybrid: Position + content-based with confidence-based overrides

Research question: "Does content-aware labeling improve anomaly
detection compared to simple position-based labeling?"

Usage:
    cd experiments/009_labeling_comparison
    python run.py
    python run.py --strategy position_based
    python run.py --strategy content_based
    python run.py --strategy hybrid
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
import torch.nn as nn
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import CVRPreprocessor
from src.models.bert_lstm import BertLSTMClassifier
from src.models.focal_loss import FocalLoss
from src.evaluate.safety_metrics import compute_all_safety_metrics, format_metrics_table

console = Console()

LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]


def create_sequences_from_df(df, window_size=10, stride=5,
                              text_col="cvr_message", label_col="label",
                              case_col="case_id"):
    """Create sequences using sliding window."""
    sequences, labels = [], []
    for case_id, group in df.groupby(case_col):
        group = group.sort_values(
            "turn_number" if "turn_number" in group.columns else group.index
        ).reset_index(drop=True)
        utterances = group[text_col].tolist()
        case_labels = group[label_col].tolist()
        if len(utterances) < 3:
            continue
        for i in range(0, len(utterances) - window_size + 1, stride):
            seq = utterances[i:i + window_size]
            seq_label = case_labels[i + window_size - 1]
            sequences.append(seq)
            labels.append(seq_label)
        if len(utterances) >= window_size and (len(utterances) - window_size) % stride != 0:
            sequences.append(utterances[-window_size:])
            labels.append(case_labels[-1])
    return sequences, labels


class SequentialCVRDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_utterances=10, max_length=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_utterances = max_utterances
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = [str(s) for s in self.sequences[idx] if pd.notna(s) and str(s).strip()]
        label = self.labels[idx]
        if not sequence:
            sequence = ["[empty]"]
        if len(sequence) > self.max_utterances:
            sequence = sequence[-self.max_utterances:]

        encoded = self.tokenizer(
            sequence, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        n = len(sequence)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        utterance_mask = torch.ones(self.max_utterances)
        if n < self.max_utterances:
            pad = self.max_utterances - n
            input_ids = torch.cat([input_ids, torch.zeros(pad, self.max_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad, self.max_length, dtype=torch.long)])
            utterance_mask[n:] = 0

        if isinstance(label, str):
            label = LABEL_MAP.get(label, 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "utterance_mask": utterance_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        output = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["utterance_mask"].to(device),
        )
        all_preds.extend(torch.argmax(output["logits"], dim=1).cpu().numpy())
        all_labels.extend(batch["labels"].numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    metrics = compute_all_safety_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def apply_labeling_strategy(df, strategy, config):
    """Apply a labeling strategy to the dataset."""
    preprocessor = CVRPreprocessor()

    if strategy == "position_based":
        console.print("[cyan]Using position-based labeling[/cyan]")
        df = preprocessor.assign_temporal_labels_by_position(df)

    elif strategy == "content_based":
        console.print("[cyan]Using content-based labeling (LLM annotations)[/cyan]")
        annotation_dir = PROJECT_ROOT / config["data"]["annotation_dir"]
        try:
            df = preprocessor.assign_content_based_labels(
                df, annotation_path=annotation_dir / "cvr_annotated_deepseek.csv"
            )
        except FileNotFoundError:
            console.print("[yellow]Content annotations not found. Run llm_annotate.py first.[/yellow]")
            console.print("[yellow]Falling back to position-based labeling.[/yellow]")
            df = preprocessor.assign_temporal_labels_by_position(df)

    elif strategy == "hybrid":
        console.print("[cyan]Using hybrid labeling (position + content override)[/cyan]")
        annotation_dir = PROJECT_ROOT / config["data"]["annotation_dir"]
        try:
            df = preprocessor.assign_hybrid_labels(
                df, annotation_path=annotation_dir / "cvr_annotated_deepseek.csv"
            )
        except FileNotFoundError:
            console.print("[yellow]Content annotations not found. Falling back to position-based.[/yellow]")
            df = preprocessor.assign_temporal_labels_by_position(df)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def train_and_evaluate(strategy, config, device):
    """Train BERT+LSTM with a specific labeling strategy and evaluate."""
    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]Strategy: {strategy}[/bold blue]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")

    # Load raw data
    data_path = PROJECT_ROOT / config["data"]["source"]
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    df = pd.read_csv(data_path)
    text_col = config["data"]["text_column"]
    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

    # Apply labeling strategy
    df = apply_labeling_strategy(df, strategy, config)

    # Map labels to integers
    if df["label"].dtype == object:
        df["label_id"] = df["label"].map(LABEL_MAP)
    else:
        df["label_id"] = df["label"]

    # Show distribution
    console.print(f"\nLabel distribution ({strategy}):")
    for label in LABEL_NAMES:
        count = (df["label"] == label).sum()
        pct = count / len(df) * 100
        console.print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Create sequences
    sequences, labels = create_sequences_from_df(
        df, window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        text_col=text_col, label_col="label_id",
        case_col=config["data"]["case_id_column"],
    )
    console.print(f"[green]{len(sequences):,} sequences[/green]")

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"], stratify=labels,
    )
    adj = config["data"]["val_split"] / (1 - config["data"]["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adj,
        random_state=config["data"]["random_seed"], stratify=y_temp,
    )

    # Tokenizer & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"])
    max_utt = config["data"]["max_utterances"]
    max_len = config["data"]["max_utterance_length"]
    bs = config["training"]["batch_size"]

    train_ds = SequentialCVRDataset(X_train, y_train, tokenizer, max_utt, max_len)
    val_ds = SequentialCVRDataset(X_val, y_val, tokenizer, max_utt, max_len)
    test_ds = SequentialCVRDataset(X_test, y_test, tokenizer, max_utt, max_len)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BertLSTMClassifier(
        model_name=config["model"]["encoder"],
        num_labels=config["model"]["num_labels"],
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        max_utterances=max_utt, max_length=max_len,
    )
    model.to(device)

    class_weights = torch.tensor([
        config["training"]["class_weights"][l] for l in LABEL_NAMES
    ], dtype=torch.float32).to(device)
    criterion = FocalLoss(num_classes=4, gamma=2.0, class_weights=class_weights.tolist())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_steps = len(train_loader) * config["training"]["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["training"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # Train
    best_val_f1 = 0.0
    patience = 0
    ckpt_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"] / strategy
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            try:
                output = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["utterance_mask"].to(device),
                )
                loss = criterion(output["logits"], batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

        val_m = evaluate(model, val_loader, device)
        console.print(
            f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f} | "
            f"Val F1={val_m['macro_f1']:.4f} | CRIT Recall={val_m['critical_recall']:.2%}"
        )

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            patience = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
        else:
            patience += 1
        if patience >= config["training"]["early_stopping_patience"]:
            console.print(f"  [yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break

    # Test
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", weights_only=True))
    test_m = evaluate(model, test_loader, device)
    console.print(format_metrics_table(test_m))

    return {
        "strategy": strategy,
        "metrics": {k: v for k, v in test_m.items() if k not in ("y_true", "y_pred")},
        "y_true": test_m["y_true"].tolist(),
        "y_pred": test_m["y_pred"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None,
                        choices=["position_based", "content_based", "hybrid"],
                        help="Run only one strategy")
    args = parser.parse_args()

    console.print("\n[bold cyan]Experiment 009: Labeling Strategy Comparison[/bold cyan]")
    console.print("=" * 60)

    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/cyan]")

    strategies = [args.strategy] if args.strategy else ["position_based", "content_based", "hybrid"]
    all_results = []

    for strategy in strategies:
        result = train_and_evaluate(strategy, config, device)
        all_results.append(result)

    # Summary comparison
    if len(all_results) > 1:
        console.print("\n" + "=" * 80)
        console.print("[bold green]LABELING STRATEGY COMPARISON[/bold green]")
        console.print("=" * 80)

        table = Table(title="Strategy Comparison (Same BERT+LSTM Model)")
        table.add_column("Strategy", style="cyan")
        table.add_column("Accuracy", style="yellow", justify="right")
        table.add_column("Macro F1", style="yellow", justify="right")
        table.add_column("Safety F1", style="green", justify="right")
        table.add_column("EDS", style="green", justify="right")
        table.add_column("CRITICAL Recall", style="red", justify="right")

        for r in all_results:
            m = r["metrics"]
            table.add_row(
                r["strategy"],
                f"{m['accuracy']:.4f}",
                f"{m['macro_f1']:.4f}",
                f"{m['safety_weighted_f1']:.4f}",
                f"{m['early_detection_score']:.4f}",
                f"{m['critical_recall']:.2%}",
            )
        console.print(table)

    # Save
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        np.save(output_dir / f"y_pred_{r['strategy']}.npy", np.array(r["y_pred"]))
        np.save(output_dir / f"y_true_{r['strategy']}.npy", np.array(r["y_true"]))

    results_save = {
        "experiment": config["experiment"],
        "strategies": [
            {"strategy": r["strategy"], "metrics": r["metrics"]}
            for r in all_results
        ],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_save, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
