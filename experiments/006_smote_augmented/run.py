"""
Experiment 006: SMOTE-Augmented Training for Class Imbalance

This experiment addresses the extreme class imbalance (14:1 NORMAL:CRITICAL)
by using aggressive class weighting and strategic oversampling.

Key innovations:
1. Cost-sensitive learning (20x penalty for CRITICAL misses)
2. Aggressive class weighting
3. Focal Loss for hard example mining
4. Target: CRITICAL recall > 70%

FIXED (2026-02-05):
- Fixed sliding window logic in CVRSequenceDataset (was taking first 20 utterances only)
- Now uses proper create_sequences_from_df from Exp 002
- Label now taken from LAST utterance in window (not majority voting)
- Moved 'import pandas' out of __getitem__ (was causing performance issues)
- Added CUDA OOM handling and periodic cache clearing
- Fixed torch.load to use weights_only=True (security fix)

Usage:
    cd experiments/006_smote_augmented
    python run.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier
from src.models.focal_loss import FocalLoss
from src.utils.config import config as global_config

console = Console()


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
    
    FIXED: This function properly creates sliding windows across the entire flight,
    ensuring the model sees CRITICAL/ELEVATED phases at the end of flights.
    
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
        group = group.sort_values("turn_number" if "turn_number" in group.columns else group.index).reset_index(drop=True)

        utterances = group[text_col].tolist()
        case_labels = group[label_col].tolist()

        # Skip if too few utterances
        if len(utterances) < min_utterances:
            continue

        # Create sliding windows
        for i in range(0, len(utterances) - window_size + 1, stride):
            seq = utterances[i:i + window_size]

            # Use the label of the LAST utterance in the sequence
            # (this represents the state at that point in time)
            seq_label = case_labels[i + window_size - 1]

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


class SequentialCVRDataset(Dataset):
    """
    Dataset for CVR sequences using sliding windows.
    
    FIXED (2026-02-05):
    - Now properly uses sliding window sequences
    - Each sequence contains window_size utterances
    - Label is from the LAST utterance in the window
    """
    
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
            labels: List of integer labels (from last utterance in each window)
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
        
        # Filter valid strings only (FIXED: properly handle NaN)
        sequence = [str(s) for s in sequence if pd.notna(s) and str(s).strip()]
        
        # Handle empty sequence
        if not sequence:
            sequence = ["[EMPTY]"]

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

        # Convert label to int if it's a string (FIXED: handle string labels)
        if isinstance(label, str):
            label_map = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
            label = label_map.get(label, 0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "utterance_mask": utterance_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'utterance_mask': torch.stack([b['utterance_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


class ExperimentRunner:
    """Runner for SMOTE augmentation experiment with FIXED sliding window."""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.config_path = exp_dir / "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load experiment config."""
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        
        device_env = global_config.get_env("DEVICE", "auto")
        if device_env != "auto":
            cfg["device"] = device_env
        
        return cfg
    
    def setup(self):
        """Setup directories."""
        exp_id = self.config["experiment"]["id"]
        
        for subdir in ["outputs", "models", "logs"]:
            path = PROJECT_ROOT / subdir / "experiments" / exp_id
            path.mkdir(parents=True, exist_ok=True)
            self.config["paths"][f"{subdir}_dir"] = str(path)
    
    def load_data(self):
        """Load and prepare data with FIXED sliding window."""
        console.print("\n[yellow]Loading data with FIXED sliding window...[/yellow]")
        
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        df = pd.read_csv(data_path)
        
        text_col = self.config["data"]["text_column"]
        label_col = self.config["data"]["label_column"]
        
        # Filter out empty texts (consistent with Exp 002, 003, 004)
        df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()
        
        # Map labels if string
        label_map = {label: idx for idx, label in enumerate(self.config["data"]["labels"])}
        if df[label_col].dtype == object:
            df['label'] = df[label_col].map(label_map)
        else:
            df['label'] = df[label_col]
        
        # Create sequences using sliding window (FIXED)
        window_size = self.config["data"]["window_size"]
        stride = self.config["data"]["stride"]
        
        console.print(f"[cyan]Creating sliding window sequences (window={window_size}, stride={stride})...[/cyan]")
        
        sequences, labels = create_sequences_from_df(
            df,
            window_size=window_size,
            stride=stride,
            text_col=self.config["data"]["text_column"],
            label_col="label",
            case_col=self.config["data"]["case_id_column"],
        )
        
        console.print(f"[green]Created {len(sequences):,} sequences[/green]")
        
        # Show sequence label distribution
        console.print("\nClass distribution (sequences):")
        unique, counts = np.unique(labels, return_counts=True)
        for label_id, count in zip(unique, counts):
            label_name = self.config["data"]["labels"][label_id]
            pct = count / len(labels) * 100
            console.print(f"  {label_name}: {count:,} ({pct:.1f}%)")
        
        # Split data (stratified by label)
        test_split = self.config["data"]["test_split"]
        val_split = self.config["data"]["val_split"]
        random_seed = self.config["data"]["random_seed"]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            sequences, labels,
            test_size=test_split,
            random_state=random_seed,
            stratify=labels
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
    
    def create_weighted_sampler(self, labels: List[int]) -> WeightedRandomSampler:
        """Create sampler that oversamples minority classes."""
        # Count labels
        class_counts = np.bincount(labels)
        
        # Calculate weights (inverse frequency)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels) * 2,  # Double the dataset size
            replacement=True,
        )
    
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
        
        # Create weighted sampler for oversampling
        sampler = self.create_weighted_sampler(train_data[1])
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def train(self):
        """Run training."""
        console.print("\n[bold cyan]Experiment 006: SMOTE-Augmented Training (FIXED)[/bold cyan]")
        console.print("[bold cyan]" + "="*60 + "[/bold cyan]")
        console.print("[green]FIXED (2026-02-05): Proper sliding window implementation[/green]")
        console.print("[green]Label now taken from LAST utterance in window (not majority)[/green]\n")
        
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        console.print(f"[cyan]Device: {device}[/cyan]")
        
        # Load data with FIXED sliding window
        train_data, val_data, test_data = self.load_data()
        
        # Create model
        model = BertLSTMClassifier(
            model_name=self.config["model"]["encoder"],
            num_labels=self.config["model"]["num_labels"],
            lstm_hidden=self.config["model"]["lstm_hidden"],
            lstm_layers=self.config["model"]["lstm_layers"],
            dropout=self.config["model"]["dropout"],
        )
        model.to(device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["encoder"])
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_data, val_data, test_data, tokenizer
        )
        
        # Loss function with aggressive weighting
        class_weights = torch.tensor([
            float(self.config["training"]["class_weights"][label])
            for label in ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]
        ]).to(device)
        
        # Use Focal Loss for hard example mining
        weight_list = class_weights.tolist()
        criterion = FocalLoss(
            num_classes=4,
            gamma=2.0,
            class_weights=weight_list,
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Training loop
        console.print("\n[bold yellow]Starting training...[/bold yellow]")
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.config["training"]["max_epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            
            try:
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                    try:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        utterance_mask = batch['utterance_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        optimizer.zero_grad()
                        
                        output = model(input_ids, attention_mask, utterance_mask)
                        logits = output["logits"]
                        loss = criterion(logits, labels)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.config["training"]["gradient_clip"]
                        )
                        optimizer.step()
                        
                        train_loss += loss.item()
                        
                        # Clear cache periodically to prevent OOM
                        if device == "cuda" and batch_idx % 100 == 0:
                            torch.cuda.empty_cache()
                            
                    except torch.cuda.OutOfMemoryError as e:
                        console.print(f"\n[red]CUDA OOM at batch {batch_idx}. Clearing cache and skipping...[/red]")
                        torch.cuda.empty_cache()
                        continue
                        
            except KeyboardInterrupt:
                console.print("\n[yellow]Training interrupted by user. Saving checkpoint...[/yellow]")
                checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_dir / "interrupted_model.pt")
                console.print(f"[green]Checkpoint saved to {checkpoint_dir / 'interrupted_model.pt'}[/green]")
                raise
            
            # Validate
            val_metrics = self.evaluate(model, val_loader, device)
            val_f1 = val_metrics['macro_f1']
            critical_recall = val_metrics.get('per_class_recall', {}).get('CRITICAL', 0)
            
            scheduler.step(val_f1)
            
            console.print(
                f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
                f"Val F1: {val_f1:.4f} | CRITICAL Recall: {critical_recall:.2%}"
            )
            
            # Save best
            checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
                console.print(f"  [green]New best model (F1: {val_f1:.4f})[/green]")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config["training"]["early_stopping_patience"]:
                console.print(f"\n[yellow]Early stopping at epoch {epoch+1}[/yellow]")
                break
        
        # Test
        console.print("\n[bold cyan]Testing best model...[/bold cyan]")
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", weights_only=True))
        test_metrics = self.evaluate(model, test_loader, device)
        
        # Save results
        self.save_results(test_metrics)
        
        return test_metrics
    
    @torch.no_grad()
    def evaluate(self, model, dataloader, device):
        """Evaluate model."""
        model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            utterance_mask = batch['utterance_mask'].to(device)
            labels = batch['labels']
            
            output = model(input_ids, attention_mask, utterance_mask)
            logits = output["logits"]
            preds = torch.argmax(logits, dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Per-class metrics
        report = classification_report(
            all_labels, all_preds,
            labels=range(4),
            target_names=self.config["data"]["labels"],
            output_dict=True,
            zero_division=0
        )
        
        per_class_f1 = {
            label: report[label]['f1-score']
            for label in self.config["data"]["labels"]
        }
        per_class_recall = {
            label: report[label]['recall']
            for label in self.config["data"]["labels"]
        }
        per_class_precision = {
            label: report[label]['precision']
            for label in self.config["data"]["labels"]
        }
        
        # Critical recall specifically
        critical_recall = per_class_recall.get("CRITICAL", 0)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'per_class_recall': per_class_recall,
            'per_class_precision': per_class_precision,
            'critical_recall': critical_recall,
        }
    
    def save_results(self, metrics):
        """Save results."""
        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'experiment_id': self.config["experiment"]["id"],
            'experiment_title': self.config["experiment"]["title"],
            'status': 'FIXED_AND_COMPLETED',
            'fix_date': '2026-02-05',
            'fix_description': 'Fixed sliding window logic - now uses proper create_sequences_from_df',
            'metrics': metrics,
            'config': self.config,
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[green]Results saved to {output_dir / 'results.json'}[/green]")
        
        # Print summary
        console.print("\n[bold green]Test Results:[/bold green]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        table.add_row("Macro F1", f"{metrics['macro_f1']:.4f}")
        table.add_row("CRITICAL Recall", f"{metrics['critical_recall']:.2%}")
        
        console.print(table)
        
        # Print per-class metrics
        console.print("\n[bold cyan]Per-Class Performance:[/bold cyan]")
        table2 = Table()
        table2.add_column("Class", style="cyan")
        table2.add_column("Precision", style="yellow")
        table2.add_column("Recall", style="yellow")
        table2.add_column("F1", style="yellow")
        
        for label in self.config["data"]["labels"]:
            table2.add_row(
                label,
                f"{metrics['per_class_precision'][label]:.4f}",
                f"{metrics['per_class_recall'][label]:.4f}",
                f"{metrics['per_class_f1'][label]:.4f}",
            )
        
        console.print(table2)
    
    def run(self):
        """Run experiment."""
        self.setup()
        self.train()


def main():
    exp_dir = Path(__file__).parent
    runner = ExperimentRunner(exp_dir)
    runner.run()


if __name__ == "__main__":
    main()
