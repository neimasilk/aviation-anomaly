"""
Experiment 003: Ensemble - Baseline BERT + BERT+LSTM

Usage:
    cd experiments/003_ensemble
    python run.py

Or from project root:
    python -m experiments.003_ensemble.run

This experiment combines two trained models via soft voting:
- Exp 001: Baseline BERT (per-utterance classification)
- Exp 002: BERT+LSTM (sequential modeling)

Expected improvement: +5-7% F1 over best single model
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score
)
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier, create_tokenizer
from src.utils.config import config as global_config

console = Console()


# ========== Baseline BERT Model (Exp 001) ==========

class BaselineBertClassifier(nn.Module):
    """Baseline BERT classifier for per-utterance classification."""

    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 4):
        super().__init__()
        from transformers import BertModel, BertConfig

        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}


# ========== Sequential Dataset (same as Exp 002) ==========

class SequentialCVRDataset(Dataset):
    """Dataset for sequential CVR utterances."""

    def __init__(
        self,
        sequences: List[List[str]],
        labels: List[int],
        tokenizer,
        max_utterances: int = 10,
        max_length: int = 64,
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_utterances = max_utterances
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Truncate or pad sequence
        if len(sequence) > self.max_utterances:
            sequence = sequence[-self.max_utterances:]

        # Tokenize all utterances
        encoded = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad to max_utterances
        n_utterances = len(sequence)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        utterance_mask = torch.ones(self.max_utterances)
        if n_utterances < self.max_utterances:
            pad_size = self.max_utterances - n_utterances
            input_ids = torch.cat([
                input_ids,
                torch.zeros(pad_size, self.max_length, dtype=torch.long)
            ], dim=0)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_size, self.max_length, dtype=torch.long)
            ], dim=0)
            utterance_mask[n_utterances:] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "utterance_mask": utterance_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ========== Per-Utterance Dataset (for Baseline) ==========

class PerUtteranceDataset(Dataset):
    """Dataset for per-utterance baseline classification."""

    def __init__(self, sequences: List[List[str]], labels: List[int], tokenizer, max_length: int = 64):
        # Flatten sequences for per-utterance classification
        self.all_texts = []
        self.all_labels = []

        for seq, seq_label in zip(sequences, labels):
            # Use the last utterance's label for all utterances in sequence
            # (simplified - in practice might use different strategy)
            for utterance in seq:
                self.all_texts.append(utterance)
                self.all_labels.append(seq_label)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.all_texts)

    def __getitem__(self, idx):
        text = self.all_texts[idx]
        label = self.all_labels[idx]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ========== Sequence Creation ==========

def create_sequences_from_df(
    df: pd.DataFrame,
    window_size: int = 10,
    stride: int = 5,
    text_col: str = "cvr_message",
    label_col: str = "label",
    case_col: str = "case_id",
    min_utterances: int = 3,
) -> Tuple[List[List[str]], List[int]]:
    """Create sequences from DataFrame using sliding window approach."""
    sequences = []
    labels = []

    for case_id, group in df.groupby(case_col):
        group = group.sort_values("turn_number" if "turn_number" in group.columns else group.index).reset_index(drop=True)

        utterances = group[text_col].tolist()
        case_labels = group[label_col].tolist()

        if len(utterances) < min_utterances:
            continue

        # Create sliding windows
        for i in range(0, len(utterances) - window_size + 1, stride):
            seq = utterances[i:i + window_size]
            seq_label = case_labels[i + window_size - 1]
            sequences.append(seq)
            labels.append(seq_label)

        # Handle remaining utterances
        if len(utterances) >= window_size and (len(utterances) - window_size) % stride != 0:
            last_start = len(utterances) - window_size
            if last_start // stride * stride != last_start:
                seq = utterances[-window_size:]
                seq_label = case_labels[-1]
                sequences.append(seq)
                labels.append(seq_label)

    return sequences, labels


# ========== Ensemble Predictor ==========

class EnsemblePredictor:
    """Ensemble predictor combining Baseline BERT and BERT+LSTM."""

    def __init__(
        self,
        baseline_model: BaselineBertClassifier,
        lstm_model: BertLSTMClassifier,
        baseline_weight: float = 1.0,
        lstm_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.baseline_model = baseline_model
        self.lstm_model = lstm_model
        self.baseline_weight = baseline_weight
        self.lstm_weight = lstm_weight
        self.device = device

        self.baseline_model.eval()
        self.lstm_model.eval()

    def predict_sequential(
        self,
        dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using ensemble on sequential data.

        Returns:
            predictions: Array of predicted class IDs
            probabilities: Array of probability distributions (n_samples, n_classes)
        """
        all_probs = []
        all_preds_baseline = []
        all_preds_lstm = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                utterance_mask = batch["utterance_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                all_labels.extend(labels.cpu().numpy())

                # LSTM model prediction (sequential)
                lstm_outputs = self.lstm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    utterance_mask=utterance_mask
                )
                lstm_probs = torch.softmax(lstm_outputs["logits"], dim=-1)
                all_preds_lstm.append(lstm_probs)

                # For baseline, we need to process each utterance separately
                # and aggregate (use last utterance prediction for sequence)
                batch_size = input_ids.shape[0]
                baseline_batch_probs = []

                for i in range(batch_size):
                    # Get actual utterances for this sample
                    actual_utterances = int(utterance_mask[i].sum().item())

                    # Process each utterance through baseline
                    utterance_probs = []
                    for j in range(actual_utterances):
                        # Shape: [1, max_length] - baseline expects 2D input
                        utt_input_ids = input_ids[i, j:j+1, :]
                        utt_attention_mask = attention_mask[i, j:j+1, :]

                        baseline_out = self.baseline_model(
                            input_ids=utt_input_ids,
                            attention_mask=utt_attention_mask
                        )
                        utt_prob = torch.softmax(baseline_out["logits"], dim=-1)
                        utterance_probs.append(utt_prob)

                    # Average predictions across utterances (or use last)
                    if utterance_probs:
                        # Use last utterance prediction (most recent)
                        # Shape: [n_classes] - squeeze batch dim
                        seq_prob = utterance_probs[-1].squeeze(0)
                    else:
                        # Fallback to uniform if no valid utterances
                        seq_prob = torch.ones(4, device=self.device) / 4

                    baseline_batch_probs.append(seq_prob)

                all_preds_baseline.append(torch.stack(baseline_batch_probs))

        # Stack predictions
        lstm_probs = torch.cat(all_preds_lstm, dim=0)  # (n_samples, n_classes)
        baseline_probs = torch.cat(all_preds_baseline, dim=0)

        # Weighted average
        total_weight = self.baseline_weight + self.lstm_weight
        ensemble_probs = (
            self.baseline_weight * baseline_probs +
            self.lstm_weight * lstm_probs
        ) / total_weight

        predictions = torch.argmax(ensemble_probs, dim=-1).cpu().numpy()

        return predictions, ensemble_probs.cpu().numpy(), all_labels


# ========== Experiment Runner ==========

class Experiment003Runner:
    """Runner for Experiment 003: Ensemble."""

    def __init__(self):
        self.exp_dir = Path(__file__).parent
        self.config = self._load_config()
        self.results: Dict[str, Any] = {}
        self.label2id = {label: i for i, label in enumerate(self.config["data"]["labels"])}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def _load_config(self) -> Dict:
        """Load experiment config."""
        config_path = self.exp_dir / "config.yaml"
        with open(config_path) as f:
            import yaml
            return yaml.safe_load(f)

    def _get_device(self) -> torch.device:
        """Get device for training."""
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = global_config.device
        if device_cfg == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_cfg)

    def setup(self):
        """Setup experiment directories."""
        exp_id = self.config["experiment"]["id"]
        output_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
        log_dir = PROJECT_ROOT / "logs" / exp_id

        for dir_path in [output_dir, log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.log_dir = log_dir

        console.print(f"[green]Directories created[/green]")

    def print_info(self):
        """Print experiment info."""
        exp = self.config["experiment"]

        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment {exp['id']}: {exp['title']}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan", width=25)
        table.add_column("Value", style="yellow")

        table.add_row("Description", exp.get("description", ""))
        table.add_row("Ensemble Type", self.config["ensemble"]["type"])
        table.add_row("Voting Strategy", self.config["ensemble"]["voting_strategy"])
        table.add_row("", "")
        table.add_row("Base Models:", "")
        for model in self.config["ensemble"]["base_models"]:
            table.add_row(f"  - {model['name']}", f"weight={model['weight']}")
        table.add_row("", "")
        table.add_row("Expected Acc:", f">= {self.config['expected']['target_acc']}")
        table.add_row("Expected F1:", f">= {self.config['expected']['target_f1']}")

        console.print(table)

    def load_data(self) -> tuple:
        """Load and create sequential data."""
        data_path = PROJECT_ROOT / self.config["data"]["source"]

        console.print(f"\n[cyan]Loading data from: {data_path}[/cyan]")
        df = pd.read_csv(data_path)

        text_col = self.config["data"]["text_column"]
        label_col = self.config["data"]["label_column"]
        case_col = self.config["data"]["case_id_column"]

        # Filter out empty texts
        df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

        console.print(f"[green]Loaded {len(df):,} utterances from {df[case_col].nunique()} cases[/green]")

        # Show label distribution
        console.print("\n[bold]Label Distribution:[/bold]")
        label_counts = df[label_col].value_counts()
        for label in self.config["data"]["labels"]:
            if label in label_counts:
                count = label_counts[label]
                pct = count / len(df) * 100
                console.print(f"  {label}: {count:,} ({pct:.1f}%)")

        # Create sequences
        window_size = self.config["data"]["window_size"]
        stride = self.config["data"]["stride"]

        console.print(f"\n[cyan]Creating sequences (window={window_size}, stride={stride})...[/cyan]")

        sequences, label_strings = create_sequences_from_df(
            df,
            window_size=window_size,
            stride=stride,
            text_col=text_col,
            label_col=label_col,
            case_col=case_col,
        )

        console.print(f"[green]Created {len(sequences):,} sequences[/green]")

        # Encode labels as integers
        labels = [self.label2id[lbl] for lbl in label_strings]

        # Split data
        test_split = self.config["data"]["test_split"]
        val_split = self.config["data"]["val_split"]
        random_seed = self.config["data"]["random_seed"]

        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels
        )

        adjusted_val_split = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_split,
            random_state=random_seed,
            stratify=y_temp
        )

        console.print(f"\n[cyan]Data splits:[/cyan]")
        console.print(f"  Train: {len(X_train):,} sequences")
        console.print(f"  Val: {len(X_val):,} sequences")
        console.print(f"  Test: {len(X_test):,} sequences")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def load_models(self) -> Tuple[BaselineBertClassifier, BertLSTMClassifier]:
        """Load both trained models."""
        device = self._get_device()
        model_cfg = self.config["models"]

        console.print(f"\n[cyan]Loading models on {device}...[/cyan]")

        # Load Baseline BERT (Exp 001)
        console.print("[cyan]  - Loading Baseline BERT (Exp 001)...[/cyan]")
        baseline_model = BaselineBertClassifier(
            model_name=model_cfg["encoder"],
            num_labels=model_cfg["num_labels"]
        )
        baseline_checkpoint = PROJECT_ROOT / self.config["ensemble"]["base_models"][0]["checkpoint"]
        baseline_state = torch.load(baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(baseline_state)
        baseline_model.to(device)
        console.print(f"[green]    Loaded from: {baseline_checkpoint}[/green]")

        # Load BERT+LSTM (Exp 002)
        console.print("[cyan]  - Loading BERT+LSTM (Exp 002)...[/cyan]")
        lstm_model = BertLSTMClassifier(
            model_name=model_cfg["encoder"],
            num_labels=model_cfg["num_labels"],
            lstm_hidden=model_cfg["bert_lstm"]["lstm_hidden"],
            lstm_layers=model_cfg["bert_lstm"]["lstm_layers"],
            dropout=model_cfg["bert_lstm"]["dropout"],
            max_utterances=model_cfg["bert_lstm"]["max_utterances"],
        )
        lstm_checkpoint = PROJECT_ROOT / self.config["ensemble"]["base_models"][1]["checkpoint"]
        lstm_state = torch.load(lstm_checkpoint, map_location=device)
        lstm_model.load_state_dict(lstm_state)
        lstm_model.to(device)
        console.print(f"[green]    Loaded from: {lstm_checkpoint}[/green]")

        return baseline_model, lstm_model

    def create_data_loaders(self, train_data, val_data, test_data, tokenizer):
        """Create DataLoaders for all splits."""
        batch_size = 16  # Smaller batch for ensemble
        max_utterances = self.config["data"]["max_utterances"]
        max_length = self.config["data"]["max_utterance_length"]

        # Sequential dataset for LSTM
        train_dataset = SequentialCVRDataset(
            train_data[0], train_data[1], tokenizer, max_utterances, max_length
        )
        val_dataset = SequentialCVRDataset(
            val_data[0], val_data[1], tokenizer, max_utterances, max_length
        )
        test_dataset = SequentialCVRDataset(
            test_data[0], test_data[1], tokenizer, max_utterances, max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader

    def evaluate_ensemble(self, predictor: EnsemblePredictor, test_loader: DataLoader) -> Dict:
        """Evaluate ensemble predictor."""
        predictions, probs, labels = predictor.predict_sequential(test_loader)

        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_per_class = f1_score(labels, predictions, average=None)
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)

        cm = confusion_matrix(labels, predictions)

        return {
            "predictions": predictions,
            "probabilities": probs,
            "labels": labels,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_per_class": f1_per_class.tolist(),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "confusion_matrix": cm.tolist(),
        }

    def tune_weights(self, baseline_model, lstm_model, val_loader, device):
        """Find optimal weights on validation set."""
        console.print("\n[yellow]Tuning ensemble weights on validation set...[/yellow]")

        if not self.config["ensemble"]["tune_weights"]:
            return self.config["ensemble"]["base_models"][0]["weight"], self.config["ensemble"]["base_models"][1]["weight"]

        weight_range = self.config["ensemble"]["weight_range"]
        weight_steps = self.config["ensemble"]["weight_steps"]

        best_f1 = 0
        best_w1, best_w2 = 1.0, 1.0

        # Grid search for optimal weights
        n_steps = int((weight_range[1] - weight_range[0]) / weight_steps) + 1

        with Progress() as progress:
            task = progress.add_task("[cyan]Tuning weights...[/cyan]", total=n_steps * n_steps)

            for w1_idx in range(n_steps):
                w1 = weight_range[0] + w1_idx * weight_steps
                for w2_idx in range(n_steps):
                    w2 = weight_range[0] + w2_idx * weight_steps

                    if w1 == 0 and w2 == 0:
                        continue

                    predictor = EnsemblePredictor(
                        baseline_model, lstm_model,
                        baseline_weight=w1,
                        lstm_weight=w2,
                        device=device,
                    )

                    val_preds, _, val_labels = predictor.predict_sequential(val_loader)
                    val_f1 = f1_score(val_labels, val_preds, average="macro")

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_w1, best_w2 = w1, w2

                    progress.update(task, advance=1)

        console.print(f"[green]Optimal weights found: baseline={best_w1:.2f}, lstm={best_w2:.2f} (val_f1={best_f1:.4f})[/green]")

        return best_w1, best_w2

    def run(self):
        """Run ensemble experiment."""
        self.print_info()
        self.setup()

        # Check data
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        if not data_path.exists():
            console.print(f"\n[red]Data not found: {data_path}[/red]")
            console.print("\n[yellow]Run: python scripts/add_temporal_labels.py[/yellow]")
            return

        # Load data
        train_data, val_data, test_data = self.load_data()

        # Setup device
        device = self._get_device()
        console.print(f"\n[cyan]Using device: {device}[/cyan]")

        if device.type == "cuda":
            console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")

        # Load tokenizer
        console.print(f"\n[cyan]Loading tokenizer: {self.config['models']['encoder']}[/cyan]")
        tokenizer = create_tokenizer(self.config['models']['encoder'])

        # Create dataloaders
        console.print("[cyan]Creating dataloaders...[/cyan]")
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_data, val_data, test_data, tokenizer
        )

        # Load models
        baseline_model, lstm_model = self.load_models()

        # Tune weights on validation set
        opt_w1, opt_w2 = self.tune_weights(baseline_model, lstm_model, val_loader, device)

        # Create ensemble predictor with optimal weights
        console.print(f"\n[cyan]Creating ensemble predictor (weights: baseline={opt_w1:.2f}, lstm={opt_w2:.2f})...[/cyan]")
        predictor = EnsemblePredictor(
            baseline_model, lstm_model,
            baseline_weight=opt_w1,
            lstm_weight=opt_w2,
            device=device,
        )

        # Evaluate on test set
        console.print("\n[bold]Evaluating ensemble on test set...[/bold]")
        test_metrics = self.evaluate_ensemble(predictor, test_loader)

        # Print results
        console.print("\n[bold cyan]Ensemble Test Results:[/bold cyan]")
        console.print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        console.print(f"  Macro F1: {test_metrics['f1_macro']:.4f}")

        console.print("\n[bold]Per-Class Results:[/bold]")
        console.print(f"{'Class':<15} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        console.print("-" * 45)
        for label, f1, prec, rec in zip(
            self.config["data"]["labels"],
            test_metrics["f1_per_class"],
            test_metrics["precision_per_class"],
            test_metrics["recall_per_class"]
        ):
            console.print(f"{label:<15} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f}")

        # Classification report
        console.print("\n[bold]Classification Report:[/bold]")
        report = classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=self.config["data"]["labels"],
            digits=4
        )
        console.print(report)

        # Compare with base models
        console.print("\n[bold cyan]Comparison with Base Models:[/bold cyan]")

        baseline_acc = self.config["expected"]["baseline_acc"]
        baseline_f1 = self.config["expected"]["baseline_f1"]
        bert_lstm_acc = self.config["expected"]["bert_lstm_acc"]
        bert_lstm_f1 = self.config["expected"]["bert_lstm_f1"]

        console.print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Macro':<12} {'vs Baseline':<15} {'vs BERT+LSTM':<15}")
        console.print("-" * 75)

        # Exp 001
        console.print(f"{'Baseline BERT (001)':<20} {baseline_acc:<12.4f} {baseline_f1:<12.4f} {'-':<15} {'-':<15}")

        # Exp 002
        acc_diff_002 = test_metrics['accuracy'] - bert_lstm_acc
        f1_diff_002 = test_metrics['f1_macro'] - bert_lstm_f1
        acc_str_002 = f"{acc_diff_002:+.4f}" if acc_diff_002 >= 0 else f"{acc_diff_002:.4f}"
        f1_str_002 = f"{f1_diff_002:+.4f}" if f1_diff_002 >= 0 else f"{f1_diff_002:.4f}"
        console.print(f"{'BERT+LSTM (002)':<20} {bert_lstm_acc:<12.4f} {bert_lstm_f1:<12.4f} {'-':<15} {'-':<15}")

        # Ensemble
        acc_diff_001 = test_metrics['accuracy'] - baseline_acc
        f1_diff_001 = test_metrics['f1_macro'] - baseline_f1
        acc_str_001 = f"[green]{acc_diff_001:+.4f}[/green]" if acc_diff_001 >= 0 else f"[red]{acc_diff_001:.4f}[/red]"
        f1_str_001 = f"[green]{f1_diff_001:+.4f}[/green]" if f1_diff_001 >= 0 else f"[red]{f1_diff_001:.4f}[/red]"

        acc_diff_002 = test_metrics['accuracy'] - bert_lstm_acc
        f1_diff_002 = test_metrics['f1_macro'] - bert_lstm_f1
        acc_str_002 = f"[green]{acc_diff_002:+.4f}[/green]" if acc_diff_002 >= 0 else f"[red]{acc_diff_002:.4f}[/red]"
        f1_str_002 = f"[green]{f1_diff_002:+.4f}[/green]" if f1_diff_002 >= 0 else f"[red]{f1_diff_002:.4f}[/red]"

        console.print(f"{'Ensemble (003)':<20} {test_metrics['accuracy']:<12.4f} {test_metrics['f1_macro']:<12.4f} {acc_str_001:<15} {acc_str_002:<15}")

        # Check if targets met
        target_acc = self.config["expected"]["target_acc"].replace(">=", "").replace(" ", "")
        target_f1 = self.config["expected"]["target_f1"].replace(">=", "").replace(" ", "")

        console.print("\n[bold]Target Achievement:[/bold]")
        acc_target_met = test_metrics['accuracy'] >= float(target_acc)
        f1_target_met = test_metrics['f1_macro'] >= float(target_f1)

        console.print(f"  Accuracy >= {target_acc}: {'[green]YES[/green]' if acc_target_met else '[red]NO[/red]'} ({test_metrics['accuracy']:.4f})")
        console.print(f"  F1 Macro >= {target_f1}: {'[green]YES[/green]' if f1_target_met else '[red]NO[/red]'} ({test_metrics['f1_macro']:.4f})")

        # Save results
        results = {
            "experiment": self.config["experiment"]["id"],
            "ensemble_type": self.config["ensemble"]["type"],
            "voting_strategy": self.config["ensemble"]["voting_strategy"],
            "weights": {
                "baseline": opt_w1,
                "bert_lstm": opt_w2,
            },
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "per_class_f1": {
                label: float(f1)
                for label, f1 in zip(self.config["data"]["labels"], test_metrics["f1_per_class"])
            },
            "per_class_precision": {
                label: float(prec)
                for label, prec in zip(self.config["data"]["labels"], test_metrics["precision_per_class"])
            },
            "per_class_recall": {
                label: float(rec)
                for label, rec in zip(self.config["data"]["labels"], test_metrics["recall_per_class"])
            },
            "classification_report": classification_report(
                test_metrics["labels"],
                test_metrics["predictions"],
                target_names=self.config["data"]["labels"],
                digits=4
            ),
            "comparison": {
                "baseline_001": {
                    "accuracy_diff": float(test_metrics["accuracy"] - baseline_acc),
                    "f1_diff": float(test_metrics["f1_macro"] - baseline_f1),
                },
                "bert_lstm_002": {
                    "accuracy_diff": float(test_metrics["accuracy"] - bert_lstm_acc),
                    "f1_diff": float(test_metrics["f1_macro"] - bert_lstm_f1),
                },
            },
            "targets_met": {
                "accuracy": acc_target_met,
                "f1_macro": f1_target_met,
            },
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to: {results_path}[/green]")

        # Save confusion matrix
        cm = np.array(test_metrics["confusion_matrix"])
        cm_path = self.output_dir / "confusion_matrix.npy"
        np.save(cm_path, cm)
        console.print(f"[green]Confusion matrix saved to: {cm_path}[/green]")

        console.print("\n[bold green]Experiment 003 complete![/bold green]")

        return results


def main():
    runner = Experiment003Runner()
    runner.run()


if __name__ == "__main__":
    main()
