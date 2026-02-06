"""
Experiment 011: DeBERTa-v3 + LSTM

Replaces bert-base-uncased with microsoft/deberta-v3-base in the
BERT+LSTM architecture. DeBERTa-v3 uses disentangled attention and
enhanced mask decoder, consistently outperforming BERT on NLU tasks.

The codebase uses AutoModel.from_pretrained() so the swap is config-only.

Usage:
    cd experiments/011_deberta_lstm
    python run.py
"""
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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier
from src.models.focal_loss import FocalLoss
from src.evaluate.safety_metrics import compute_all_safety_metrics, format_metrics_table

console = Console()

LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]


def create_sequences_from_df(
    df: pd.DataFrame,
    window_size: int = 10,
    stride: int = 5,
    text_col: str = "cvr_message",
    label_col: str = "label",
    case_col: str = "case_id",
) -> Tuple[List[List[str]], List[int]]:
    """Create sequences using sliding window."""
    sequences = []
    labels = []

    for case_id, group in df.groupby(case_col):
        group = group.sort_values(
            "turn_number" if "turn_number" in group.columns else group.index
        ).reset_index(drop=True)

        utterances = group[text_col].tolist()
        case_labels = group[label_col].tolist()

        if len(utterances) < 3:
            continue

        for i in range(0, len(utterances) - window_size + 1, stride):
            seq = utterances[i : i + window_size]
            seq_label = case_labels[i + window_size - 1]
            sequences.append(seq)
            labels.append(seq_label)

        if len(utterances) >= window_size and (len(utterances) - window_size) % stride != 0:
            seq = utterances[-window_size:]
            seq_label = case_labels[-1]
            sequences.append(seq)
            labels.append(seq_label)

    return sequences, labels


class SequentialCVRDataset(Dataset):
    """Dataset for sequential CVR utterances."""

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
    """Evaluate model and return metrics + predictions."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        utterance_mask = batch["utterance_mask"].to(device)

        output = model(input_ids, attention_mask, utterance_mask)
        preds = torch.argmax(output["logits"], dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(batch["labels"].numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_all_safety_metrics(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred

    return metrics


def main():
    console.print("\n[bold cyan]Experiment 011: DeBERTa-v3 + LSTM[/bold cyan]")
    console.print("=" * 60)

    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    encoder = config["model"]["encoder"]
    console.print(f"[cyan]Encoder: {encoder}[/cyan]")

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/cyan]")

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
        df["label"] = df[label_col].map(LABEL_MAP)
    else:
        df["label"] = df[label_col]

    console.print(f"[green]Loaded {len(df):,} utterances[/green]")

    # Create sequences
    sequences, labels = create_sequences_from_df(
        df, window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        text_col=text_col, label_col="label",
        case_col=config["data"]["case_id_column"],
    )
    console.print(f"[green]{len(sequences):,} sequences created[/green]")

    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, labels, test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"], stratify=labels,
    )
    adj_val = config["data"]["val_split"] / (1 - config["data"]["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adj_val,
        random_state=config["data"]["random_seed"], stratify=y_temp,
    )

    console.print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Tokenizer & datasets
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    max_utt = config["data"]["max_utterances"]
    max_len = config["data"]["max_utterance_length"]

    train_ds = SequentialCVRDataset(X_train, y_train, tokenizer, max_utt, max_len)
    val_ds = SequentialCVRDataset(X_val, y_val, tokenizer, max_utt, max_len)
    test_ds = SequentialCVRDataset(X_test, y_test, tokenizer, max_utt, max_len)

    bs = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BertLSTMClassifier(
        model_name=encoder,
        num_labels=config["model"]["num_labels"],
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        max_utterances=max_utt,
        max_length=max_len,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Parameters: {total_params:,}[/green]")

    # Loss
    class_weights = torch.tensor([
        config["data"]["class_weights"][l] for l in LABEL_NAMES
    ], dtype=torch.float32).to(device)
    criterion = FocalLoss(num_classes=4, gamma=2.0, class_weights=class_weights.tolist())

    # Optimizer with gradient accumulation
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_steps = (len(train_loader) // grad_accum) * config["training"]["max_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["training"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # Training loop
    console.print("\n[bold yellow]Starting training...[/bold yellow]")
    best_val_f1 = 0.0
    patience_counter = 0

    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                utterance_mask = batch["utterance_mask"].to(device)
                labels_t = batch["labels"].to(device)

                output = model(input_ids, attention_mask, utterance_mask)
                loss = criterion(output["logits"], labels_t)
                loss = loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip"]
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * grad_accum

                if device == "cuda" and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                console.print(f"[red]OOM at batch {batch_idx}, skipping[/red]")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        cr = val_metrics["critical_recall"]

        console.print(
            f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f} | "
            f"Val F1={val_f1:.4f} | CRITICAL Recall={cr:.2%}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            console.print(f"  [green]New best (F1={val_f1:.4f})[/green]")
        else:
            patience_counter += 1

        if patience_counter >= config["training"]["early_stopping_patience"]:
            console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break

    # Test
    console.print("\n[bold cyan]Testing best model...[/bold cyan]")
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    console.print(format_metrics_table(test_metrics))

    # Compare with Exp 002 (BERT+LSTM)
    console.print("\n[bold cyan]Comparison with Exp 002 (BERT+LSTM):[/bold cyan]")
    bert_acc, bert_f1 = 0.7917, 0.6589
    acc_diff = test_metrics["accuracy"] - bert_acc
    f1_diff = test_metrics["macro_f1"] - bert_f1
    console.print(f"  Accuracy: {test_metrics['accuracy']:.4f} vs {bert_acc:.4f} ({acc_diff:+.4f})")
    console.print(f"  Macro F1: {test_metrics['macro_f1']:.4f} vs {bert_f1:.4f} ({f1_diff:+.4f})")

    # Save results
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "y_true.npy", test_metrics["y_true"])
    np.save(output_dir / "y_pred.npy", test_metrics["y_pred"])

    results = {
        "experiment": config["experiment"],
        "encoder": encoder,
        "metrics": {k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred")},
        "comparison_with_bert_lstm": {
            "accuracy_diff": float(acc_diff),
            "f1_diff": float(f1_diff),
        },
        "config": config,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
