"""
Experiment 001: Baseline BERT - Static Per-Utterance Classification

Usage:
    cd experiments/001_baseline_bert
    python run.py

Or from project root:
    python -m experiments.001_baseline_bert.run
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import config as global_config

console = Console()


class CVRDataset(Dataset):
    """Simple dataset for CVR utterances."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class Experiment001Runner:
    """Runner for Experiment 001."""

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
        table.add_column("Key", style="cyan", width=20)
        table.add_column("Value", style="yellow")

        table.add_row("Description", exp.get("description", ""))
        table.add_row("Status", exp.get("status", ""))
        table.add_row("", "")
        table.add_row("Model", self.config["model"]["type"])
        table.add_row("Encoder", self.config["model"]["encoder"])
        table.add_row("Batch Size", str(self.config["training"]["batch_size"]))
        table.add_row("Learning Rate", str(self.config["training"]["learning_rate"]))
        table.add_row("Max Epochs", str(self.config["training"]["max_epochs"]))

        console.print(table)

    def load_data(self) -> tuple:
        """Load and split data."""
        data_path = PROJECT_ROOT / self.config["data"]["source"]

        console.print(f"\n[cyan]Loading data from: {data_path}[/cyan]")
        df = pd.read_csv(data_path)

        text_col = self.config["data"]["text_column"]
        label_col = self.config["data"]["label_column"]

        # Filter out empty texts
        df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

        console.print(f"[green]Loaded {len(df):,} utterances[/green]")

        # Show label distribution
        console.print("\n[bold]Label Distribution:[/bold]")
        label_counts = df[label_col].value_counts()
        for label in self.config["data"]["labels"]:
            if label in label_counts:
                count = label_counts[label]
                pct = count / len(df) * 100
                console.print(f"  {label}: {count:,} ({pct:.1f}%)")

        # Encode labels
        texts = df[text_col].tolist()
        labels = [self.label2id[lbl] for lbl in df[label_col].tolist()]

        # Split data (stratified by label)
        test_split = self.config["data"]["test_split"]
        val_split = self.config["data"]["val_split"]
        random_seed = self.config["data"]["random_seed"]

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels
        )

        # Second split: train vs val (adjust val_split for remaining data)
        adjusted_val_split = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_split,
            random_state=random_seed,
            stratify=y_temp
        )

        console.print(f"\n[cyan]Data splits:[/cyan]")
        console.print(f"  Train: {len(X_train):,}")
        console.print(f"  Val: {len(X_val):,}")
        console.print(f"  Test: {len(X_test):,}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def create_data_loaders(self, train_data, val_data, test_data, tokenizer):
        """Create DataLoaders for all splits."""
        batch_size = self.config["training"]["batch_size"]
        max_length = self.config["data"]["max_utterance_length"]

        train_dataset = CVRDataset(train_data[0], train_data[1], tokenizer, max_length)
        val_dataset = CVRDataset(val_data[0], val_data[1], tokenizer, max_length)
        test_dataset = CVRDataset(test_data[0], test_data[1], tokenizer, max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train_epoch(self, model, dataloader, optimizer, scheduler, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["training"]["gradient_clip"])

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
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

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
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
            console.print("\n[yellow]Run: python scripts/add_temporal_labels.py[/yellow]")
            return

        # Load data
        train_data, val_data, test_data = self.load_data()

        # Setup device
        device = self._get_device()
        console.print(f"\n[cyan]Using device: {device}[/cyan]")

        # Load tokenizer
        console.print(f"\n[cyan]Loading tokenizer: {self.config['model']['encoder']}[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["encoder"])

        # Create dataloaders
        console.print("[cyan]Creating dataloaders...[/cyan]")
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_data, val_data, test_data, tokenizer
        )

        # Load model
        console.print(f"[cyan]Loading model: {self.config['model']['encoder']}[/cyan]")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model"]["encoder"],
            num_labels=len(self.config["data"]["labels"])
        )
        model.to(device)

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

        # Training loop
        console.print(f"\n[bold]Starting training for {num_epochs} epochs...[/bold]\n")

        best_val_f1 = 0
        patience_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(num_epochs):
            console.print(f"[bold]Epoch {epoch + 1}/{num_epochs}[/bold]")

            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, scheduler, device)
            val_metrics = self.evaluate(model, val_loader, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_macro"])

            console.print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            console.print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")

            # Early stopping
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.checkpoint_dir / "best_model.pt")
                console.print(f"  [green]New best F1: {best_val_f1:.4f} - Model saved[/green]")
            else:
                patience_counter += 1
                if patience_counter >= self.config["training"]["early_stopping_patience"]:
                    console.print(f"  [yellow]Early stopping triggered[/yellow]")
                    break

        # Load best model and evaluate on test
        console.print("\n[bold]Loading best model for test evaluation...[/bold]")
        model.load_state_dict(torch.load(self.checkpoint_dir / "best_model.pt"))
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

        # Save results
        results = {
            "experiment": self.config["experiment"]["id"],
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
            "config": self.config,
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


def main():
    runner = Experiment001Runner()
    runner.run()


if __name__ == "__main__":
    main()
