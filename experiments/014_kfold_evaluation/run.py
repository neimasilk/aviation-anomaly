"""
Experiment 014: K-Fold Cross-Validation

5-fold stratified cross-validation by case_id (no data leakage).
Reports mean +/- std for all metrics. Enables paired t-tests.

Usage:
    cd experiments/014_kfold_evaluation
    python run.py
    python run.py --n-folds 5
    python run.py --fold 0  # run only fold 0 (for parallelization)
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
from sklearn.model_selection import StratifiedKFold
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


def create_sequences_from_df(df, window_size=10, stride=5,
                              text_col="cvr_message", label_col="label",
                              case_col="case_id"):
    """Create sequences using sliding window, returns case_ids too."""
    sequences, labels, case_ids = [], [], []
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
            case_ids.append(case_id)
        if len(utterances) >= window_size and (len(utterances) - window_size) % stride != 0:
            sequences.append(utterances[-window_size:])
            labels.append(case_labels[-1])
            case_ids.append(case_id)
    return sequences, labels, case_ids


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
    return np.array(all_labels), np.array(all_preds)


def train_single_fold(
    fold_idx: int,
    train_sequences, train_labels,
    val_sequences, val_labels,
    config: Dict, device: str,
) -> Dict:
    """Train and evaluate one fold."""
    console.print(f"\n[bold blue]--- Fold {fold_idx + 1} ---[/bold blue]")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"])
    max_utt = config["data"]["max_utterances"]
    max_len = config["data"]["max_utterance_length"]
    bs = config["training"]["batch_size"]

    train_ds = SequentialCVRDataset(train_sequences, train_labels, tokenizer, max_utt, max_len)
    val_ds = SequentialCVRDataset(val_sequences, val_labels, tokenizer, max_utt, max_len)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    model = BertLSTMClassifier(
        model_name=config["model"]["encoder"],
        num_labels=config["model"]["num_labels"],
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        max_utterances=max_utt, max_length=max_len,
    ).to(device)

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

    best_val_f1 = 0.0
    patience = 0
    best_state = None

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fold {fold_idx+1} Ep{epoch+1}", leave=False):
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

        # Validate
        y_true_val, y_pred_val = evaluate(model, val_loader, device)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(y_true_val, y_pred_val, average="macro", zero_division=0)

        console.print(
            f"  Ep{epoch+1}: Loss={train_loss/len(train_loader):.4f} | Val F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= config["training"]["early_stopping_patience"]:
            break

    # Load best and evaluate
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    y_true, y_pred = evaluate(model, val_loader, device)
    metrics = compute_all_safety_metrics(y_true, y_pred)

    console.print(
        f"  [green]Fold {fold_idx+1}: Acc={metrics['accuracy']:.4f} | "
        f"F1={metrics['macro_f1']:.4f} | CRIT Recall={metrics['critical_recall']:.2%}[/green]"
    )

    # Cleanup GPU memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache() if device == "cuda" else None

    return {
        "fold": fold_idx,
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray, list)) or k == "confusion_matrix"},
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Run single fold (for parallelization)")
    args = parser.parse_args()

    console.print("\n[bold cyan]Experiment 014: K-Fold Cross-Validation[/bold cyan]")
    console.print("=" * 60)

    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

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
        df["label_id"] = df[label_col].map(LABEL_MAP)
    else:
        df["label_id"] = df[label_col]

    # Create sequences WITH case_id tracking
    sequences, labels, case_ids = create_sequences_from_df(
        df, window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        text_col=text_col, label_col="label_id",
        case_col=config["data"]["case_id_column"],
    )
    labels = np.array(labels)
    case_ids = np.array(case_ids)
    console.print(f"[green]{len(sequences):,} sequences from {len(np.unique(case_ids))} cases[/green]")

    # Stratified K-fold by case_id
    unique_cases = np.unique(case_ids)
    # Assign majority label per case for stratification
    case_majority_label = {}
    for case in unique_cases:
        mask = case_ids == case
        case_labels = labels[mask]
        case_majority_label[case] = np.bincount(case_labels, minlength=4).argmax()

    case_labels_for_strat = np.array([case_majority_label[c] for c in unique_cases])

    n_folds = args.n_folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["data"]["random_seed"])

    fold_results = []
    folds_to_run = [args.fold] if args.fold is not None else range(n_folds)

    for fold_idx, (train_case_idx, val_case_idx) in enumerate(skf.split(unique_cases, case_labels_for_strat)):
        if fold_idx not in folds_to_run:
            continue

        train_cases = set(unique_cases[train_case_idx])
        val_cases = set(unique_cases[val_case_idx])

        train_mask = np.array([c in train_cases for c in case_ids])
        val_mask = np.array([c in val_cases for c in case_ids])

        train_seqs = [sequences[i] for i in np.where(train_mask)[0]]
        train_labs = labels[train_mask].tolist()
        val_seqs = [sequences[i] for i in np.where(val_mask)[0]]
        val_labs = labels[val_mask].tolist()

        console.print(f"\n[cyan]Fold {fold_idx+1}: Train {len(train_seqs):,} | Val {len(val_seqs):,}[/cyan]")

        result = train_single_fold(
            fold_idx, train_seqs, train_labs, val_seqs, val_labs, config, device,
        )
        fold_results.append(result)

    if len(fold_results) < 2:
        console.print("[yellow]Single fold completed. Run all folds for summary statistics.[/yellow]")
    else:
        # Aggregate results
        console.print("\n" + "=" * 70)
        console.print("[bold green]K-FOLD CROSS-VALIDATION SUMMARY[/bold green]")
        console.print("=" * 70)

        metric_keys = ["accuracy", "macro_f1", "safety_weighted_f1",
                        "early_detection_score", "critical_recall", "safety_cost"]

        table = Table(title=f"{n_folds}-Fold Cross-Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="yellow", justify="right")
        table.add_column("Std", style="yellow", justify="right")
        table.add_column("Min", style="dim", justify="right")
        table.add_column("Max", style="dim", justify="right")

        summary = {}
        for key in metric_keys:
            values = [r["metrics"][key] for r in fold_results if key in r["metrics"]]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": [float(v) for v in values],
                }
                table.add_row(
                    key,
                    f"{np.mean(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                )

        console.print(table)

        # Stability check
        f1_std = summary.get("macro_f1", {}).get("std", 0)
        if f1_std < 0.03:
            console.print("[green]Model is STABLE (F1 std < 3%)[/green]")
        elif f1_std < 0.05:
            console.print("[yellow]Model stability is ACCEPTABLE (F1 std < 5%)[/yellow]")
        else:
            console.print("[red]Model is UNSTABLE (F1 std >= 5%)[/red]")

    # Save
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in fold_results:
        np.save(output_dir / f"y_true_fold{r['fold']}.npy", np.array(r["y_true"]))
        np.save(output_dir / f"y_pred_fold{r['fold']}.npy", np.array(r["y_pred"]))

    results_save = {
        "experiment": config["experiment"],
        "n_folds": n_folds,
        "fold_results": [
            {"fold": r["fold"], "metrics": r["metrics"]}
            for r in fold_results
        ],
        "summary": summary if len(fold_results) > 1 else None,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_save, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
