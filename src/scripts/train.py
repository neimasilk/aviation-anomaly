"""
Training script for aviation anomaly detection models.

Usage:
    python -m src.scripts.train --model bert_lstm --data data/processed/
    python -m src.scripts.train --model hierarchical_transformer --epochs 50
"""
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import typer
from rich.console import Console
from rich.progress import track

from ..utils.config import config
from ..data.preprocessing import CVRPreprocessor
from ..models.bert_lstm import BertLSTMClassifier, create_model, create_tokenizer

app = typer.Typer()
console = Console()


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in track(dataloader, description="Training", console=console):
        optimizer.zero_grad()

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        utterance_mask = batch.get("utterance_mask")
        if utterance_mask is not None:
            utterance_mask = utterance_mask.to(device)
        labels = batch["label"].to(device)

        # Forward
        outputs = model(input_ids, attention_mask, utterance_mask)
        logits = outputs["logits"]

        # Loss
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        utterance_mask = batch.get("utterance_mask")
        if utterance_mask is not None:
            utterance_mask = utterance_mask.to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, utterance_mask)
        logits = outputs["logits"]

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


@app.command()
def main(
    model_type: str = typer.Option("bert_lstm", "--model", "-m", help="Model architecture"),
    data_dir: Path = typer.Option(None, "--data", "-d", help="Data directory"),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Resume from checkpoint"),
):
    """
    Train a model for aviation anomaly detection.
    """
    console.print(f"[bold blue]Training {model_type} model[/bold blue]")

    # Set paths
    if data_dir is None:
        data_dir = config.data_dir / "processed"
    if output_dir is None:
        output_dir = config.models_dir / "checkpoints"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = config.device
    console.print(f"Using device: [yellow]{device}[/yellow]")

    # Create model
    console.print("Creating model...")
    model = create_model(model_name="bert-base-uncased", num_labels=4)
    model.to(device)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    console.print("[green]Setup complete. Ready to train.[/green]")
    console.print(f"Model: {model_type}")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {learning_rate}")

    # Note: Actual data loading to be implemented when dataset is available
    console.print("\n[yellow]Note: Data loading pipeline will be implemented once dataset is available.[/yellow]")


if __name__ == "__main__":
    app()
