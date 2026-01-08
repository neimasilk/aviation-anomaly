"""
Experiment 004: Hierarchical Transformer - Sequential Pattern Modeling

Usage:
    cd experiments/004_hierarchical
    python run.py

Or from project root:
    python -m experiments.004_hierarchical.run

This experiment implements Model B - Hierarchical Transformer:
- Token-level: BERT encodes each utterance
- Utterance-level: Transformer layers model temporal patterns
- Uses self-attention for sequence modeling instead of LSTM
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from rich.table import Table
import yaml

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hierarchical_transformer import (
    HierarchicalTransformerClassifier,
    HierarchicalPredictor,
    create_tokenizer
)
from src.utils.config import config as global_config

console = Console()


class SequentialCVRDataset(Dataset):
    """Dataset for sequential CVR utterances."""

    def __init__(
        self,
        sequences: List[List[str]],
        labels: List[int],
        tokenizer,
        max_utterances: int = 20,
        max_length: int = 128,
    ):
        """
        Args:
            sequences: List of utterance sequences (each sequence is a list of strings)
            labels: List of integer labels
            tokenizer: BERT tokenizer
            max_utterances: Maximum utterances per sequence
            max_length: Maximum tokens per utterance
        """
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
            sequence = sequence[-self.max_utterances:]  # Keep most recent

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

        # Create padding mask
        utterance_mask = torch.ones(self.max_utterances)
        if n_utterances < self.max_utterances:
            pad_size = self.max_utterances - n_utterances
            # Pad with zeros (empty utterances)
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


def create_sequences_from_df(
    df: pd.DataFrame,
    window_size: int = 10,
    stride: int = 5,
    text_col: str = "cvr_message",
    label_col: str = "label",
    case_col: str = "case_id",
    min_utterances: int = 3,
) -> Tuple[List[List[str]], List[int]]:
    """
    Create sequences from DataFrame using sliding window approach.

    Args:
        df: Input DataFrame
        window_size: Number of utterances per sequence
        stride: Step size for sliding window
        text_col: Column name for text
        label_col: Column name for label
        case_col: Column name for case ID
        min_utterances: Minimum utterances to create a sequence

    Returns:
        Tuple of (sequences, labels)
    """
    sequences = []
    labels = []

    # Group by case
    for case_id, group in df.groupby(case_col):
        # Sort by position (assuming DataFrame has position info or index)
        if "position" in group.columns:
            group = group.sort_values("position")
        else:
            group = group.sort_values("turn_number" if "turn_number" in group.columns else group.index)
        group = group.reset_index(drop=True)

        utterances = group[text_col].tolist()
        case_labels = group[label_col].tolist()

        # Skip if too few utterances
        if len(utterances) < min_utterances:
            continue

        # Create sliding windows
        for i in range(0, len(utterances) - window_size + 1, stride):
            seq = utterances[i:i + window_size]
            seq_label = case_labels[i + window_size - 1]  # Label of last utterance
            sequences.append(seq)
            labels.append(seq_label)

        # Handle remaining utterances (last partial window)
        if len(utterances) >= window_size and (len(utterances) - window_size) % stride != 0:
            last_start = len(utterances) - window_size
            if last_start // stride * stride != last_start:
                seq = utterances[-window_size:]
                seq_label = case_labels[-1]
                sequences.append(seq)
                labels.append(seq_label)

    return sequences, labels


class Experiment004Runner:
    """Runner for Experiment 004."""

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
        checkpoint_dir = PROJECT_ROOT / "models" / exp_id
        log_dir = PROJECT_ROOT / "logs" / exp_id

        for dir_path in [output_dir, checkpoint_dir, log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
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
        table.add_row("Tags", ", ".join(exp.get("tags", [])))
        table.add_row("", "")
        table.add_row("Model", self.config["model"]["type"])
        table.add_row("Encoder", self.config["model"]["encoder"])
        table.add_row("D-Model", str(self.config["model"]["d_model"]))
        table.add_row("Heads", str(self.config["model"]["n_heads"]))
        table.add_row("Layers", str(self.config["model"]["n_layers"]))
        table.add_row("", "")
        table.add_row("Window Size", str(self.config["data"]["window_size"]))
        table.add_row("Max Utterances", str(self.config["data"]["max_utterances"]))
        table.add_row("Batch Size", str(self.config["training"]["batch_size"]))
        table.add_row("Learning Rate", str(self.config["training"]["learning_rate"]))
        table.add_row("Max Epochs", str(self.config["training"]["max_epochs"]))

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
        console.print("\n[bold]Original Label Distribution:[/bold]")
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

        # Show sequence label distribution
        console.print("\n[bold]Sequence Label Distribution:[/bold]")
        label_ids = [self.label2id[lbl] for lbl in label_strings]
        unique, counts = np.unique(label_ids, return_counts=True)
        for label_id, count in zip(unique, counts):
            label = self.id2label[label_id]
            pct = count / len(label_ids) * 100
            console.print(f"  {label}: {count:,} ({pct:.1f}%)")

        # Split data (stratified by label)
        test_split = self.config["data"]["test_split"]
        val_split = self.config["data"]["val_split"]
        random_seed = self.config["data"]["random_seed"]

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, label_ids,
            test_size=test_split,
            random_state=random_seed,
            stratify=label_ids
        )

        # Second split: train vs val
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

    def create_data_loaders(self, train_data, val_data, test_data, tokenizer):
        """Create DataLoaders for all splits."""
        batch_size = self.config["training"]["batch_size"]
        max_utterances = self.config["data"]["max_utterances"]
        max_length = self.config["data"]["max_utterance_length"]

        train_dataset = SequentialCVRDataset(
            train_data[0], train_data[1], tokenizer, max_utterances, max_length
        )
        val_dataset = SequentialCVRDataset(
            val_data[0], val_data[1], tokenizer, max_utterances, max_length
        )
        test_dataset = SequentialCVRDataset(
            test_data[0], test_data[1], tokenizer, max_utterances, max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader

    def train_epoch(self, model, dataloader, optimizer, scheduler, device, class_weights=None):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            utterance_mask = batch["utterance_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, utterance_mask=utterance_mask)
            logits = outputs["logits"]

            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["training"]["gradient_clip"])

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def evaluate(self, model, dataloader, device):
        """Evaluate model."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                utterance_mask = batch["utterance_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, utterance_mask=utterance_mask)
                logits = outputs["logits"]

                loss = criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_per_class = f1_score(all_labels, all_preds, average=None)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_per_class": f1_per_class.tolist(),
            "predictions": all_preds,
            "labels": all_labels,
        }

    def run(self):
        """Run experiment."""
        self.print_info()
        self.setup()

        # Check data
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        if not data_path.exists():
            console.print(f"\n[red]Data not found: {data_path}[/red]")
            return

        # Load data
        train_data, val_data, test_data = self.load_data()

        # Setup device
        device = self._get_device()
        console.print(f"\n[cyan]Using device: {device}[/cyan]")

        if device.type == "cuda":
            console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")
            console.print(f"[green]Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB[/green]")

        # Load tokenizer
        console.print(f"\n[cyan]Loading tokenizer: {self.config['model']['encoder']}[/cyan]")
        tokenizer = create_tokenizer(self.config['model']['encoder'])

        # Create dataloaders
        console.print("[cyan]Creating dataloaders...[/cyan]")
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_data, val_data, test_data, tokenizer
        )

        # Load model
        console.print(f"[cyan]Creating Hierarchical Transformer model...[/cyan]")
        model = HierarchicalTransformerClassifier(
            model_name=self.config["model"]["encoder"],
            num_labels=len(self.config["data"]["labels"]),
            d_model=self.config["model"]["d_model"],
            n_heads=self.config["model"]["n_heads"],
            n_layers=self.config["model"]["n_layers"],
            dim_feedforward=self.config["model"]["dim_feedforward"],
            dropout=self.config["model"]["dropout"],
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
        )
        model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"[green]Total parameters: {total_params:,}[/green]")
        console.print(f"[green]Trainable parameters: {trainable_params:,}[/green]")

        # Setup training
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"])
        )

        num_epochs = self.config["training"]["max_epochs"]
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config["training"]["warmup_ratio"]),
            num_training_steps=total_steps
        )

        # Class weights (if specified)
        class_weights = None
        if "class_weights" in self.config["data"]:
            weights = [self.config["data"]["class_weights"][lbl] for lbl in self.config["data"]["labels"]]
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            console.print(f"[cyan]Using class weights: {weights}[/cyan]")

        # Check for existing checkpoint (resume capability)
        checkpoint_path = self.checkpoint_dir / "checkpoint.pt"
        start_epoch = 0
        best_val_f1 = 0

        if checkpoint_path.exists():
            console.print(f"[yellow]Found checkpoint at {checkpoint_path}[/yellow]")
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"]
                best_val_f1 = checkpoint["best_val_f1"]
                history = checkpoint["history"]
                console.print(f"[green]Resumed from epoch {start_epoch}, best F1: {best_val_f1:.4f}[/green]")
            except Exception as e:
                console.print(f"[red]Failed to load checkpoint: {e}[/red]")
                console.print("[yellow]Starting from scratch...[/yellow]")
                start_epoch = 0
                history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
        else:
            console.print("[yellow]No checkpoint found, starting from scratch[/yellow]")
            history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

        # Training loop
        console.print(f"\n[bold]Starting training for {num_epochs} epochs...[/bold]\n")

        patience_counter = 0

        for epoch in range(start_epoch, num_epochs):
            console.print(f"[bold]Epoch {epoch + 1}/{num_epochs}[/bold]")

            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, scheduler, device, class_weights
            )
            val_metrics = self.evaluate(model, val_loader, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_macro"])

            console.print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            console.print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")

            # Early stopping & checkpoint save
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.checkpoint_dir / "best_model.pt")
                console.print(f"  [green]New best F1: {best_val_f1:.4f} - Best model saved[/green]")
            else:
                patience_counter += 1
                if patience_counter >= self.config["training"]["early_stopping_patience"]:
                    console.print(f"  [yellow]Early stopping triggered[/yellow]")
                    break

            # Save checkpoint after each epoch (for resume capability)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'history': history,
            }, checkpoint_path)
            console.print(f"  [dim]Checkpoint saved (epoch {epoch + 1})[/dim]")

        # Load best model and evaluate on test
        console.print("\n[bold]Loading best model for test evaluation...[/bold]")
        model.load_state_dict(torch.load(self.checkpoint_dir / "best_model.pt", weights_only=False))
        test_metrics = self.evaluate(model, test_loader, device)

        # Print final results
        console.print("\n[bold cyan]Final Test Results:[/bold cyan]")
        console.print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        console.print(f"  Macro F1: {test_metrics['f1_macro']:.4f}")

        console.print("\n[bold]Per-Class F1:[/bold]")
        for label, f1 in zip(self.config["data"]["labels"], test_metrics["f1_per_class"]):
            console.print(f"  {label}: {f1:.4f}")

        # Classification report
        console.print("\n[bold]Classification Report:[/bold]")
        report = classification_report(
            test_metrics["labels"],
            test_metrics["predictions"],
            target_names=self.config["data"]["labels"],
            digits=4
        )
        console.print(report)

        # Compare with previous experiments
        console.print("\n[bold cyan]Comparison with Previous Experiments:[/bold cyan]")
        comparisons = [
            ("Baseline BERT (001)", 0.6482, 0.4734),
            ("BERT+LSTM (002)", 0.7917, 0.6589),
            ("Ensemble (003)", 0.8604, 0.7668),
        ]

        table = Table(show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", style="yellow")
        table.add_column("F1 Macro", style="yellow")
        table.add_column("Acc Diff", style="green")
        table.add_column("F1 Diff", style="green")

        current_acc = test_metrics['accuracy']
        current_f1 = test_metrics['f1_macro']

        for name, acc, f1 in comparisons:
            acc_diff = current_acc - acc
            f1_diff = current_f1 - f1
            acc_str = f"{acc:+.4f}" if acc_diff >= 0 else f"[red]{acc_diff:+.4f}[/red]"
            f1_str = f"{f1:+.4f}" if f1_diff >= 0 else f"[red]{f1_diff:+.4f}[/red]"
            table.add_row(name, f"{acc:.4f}", f"{f1:.4f}", acc_str, f1_str)

        table.add_row("[bold]Hierarchical (004)[/bold]", f"[bold]{current_acc:.4f}[/bold]", f"[bold]{current_f1:.4f}[/bold]", "-", "-")
        console.print(table)

        # Save results
        results = {
            "experiment": self.config["experiment"]["id"],
            "model_type": self.config["model"]["type"],
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "per_class_f1": {
                label: float(f1)
                for label, f1 in zip(self.config["data"]["labels"], test_metrics["f1_per_class"])
            },
            "classification_report": classification_report(
                test_metrics["labels"],
                test_metrics["predictions"],
                target_names=self.config["data"]["labels"],
                digits=4
            ),
            "history": history,
            "comparisons": {
                "vs_baseline_001": {
                    "accuracy_diff": float(current_acc - 0.6482),
                    "f1_diff": float(current_f1 - 0.4734),
                },
                "vs_bert_lstm_002": {
                    "accuracy_diff": float(current_acc - 0.7917),
                    "f1_diff": float(current_f1 - 0.6589),
                },
                "vs_ensemble_003": {
                    "accuracy_diff": float(current_acc - 0.8604),
                    "f1_diff": float(current_f1 - 0.7668),
                },
            },
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to: {results_path}[/green]")

        # Save confusion matrix
        cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"])
        cm_path = self.output_dir / "confusion_matrix.npy"
        np.save(cm_path, cm)
        console.print(f"[green]Confusion matrix saved to: {cm_path}[/green]")

        console.print("\n[bold green]Experiment complete![/bold green]")

        return results


def main():
    runner = Experiment004Runner()
    runner.run()


if __name__ == "__main__":
    main()
