"""
Experiment 008: Ablation Study - Window Size Effect

Systematically evaluates how sequence window size affects model performance.
Tests window sizes: 5, 10, 15, 20

Usage:
    cd experiments/008_ablation_window_size
    python run.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier
from src.utils.config import config as global_config

console = Console()


class CVRWindowDataset(Dataset):
    """Dataset with configurable window size."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        window_size: int = 10,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_length = max_length
        self.label_map = {
            "NORMAL": 0,
            "EARLY_WARNING": 1,
            "ELEVATED": 2,
            "CRITICAL": 3,
        }

        # Build sequences with specified window size
        self.sequences = []
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('turn_number') if 'turn_number' in group.columns else group

            utterances = []
            labels = []

            for _, row in group.iterrows():
                if pd.notna(row['cvr_message']):
                    utterances.append(str(row['cvr_message']))
                    label = row['label']
                    if isinstance(label, str):
                        label = self.label_map[label]
                    labels.append(label)

            if len(utterances) >= 3:  # Min 3 utterances
                # Use majority label for sequence
                from collections import Counter
                majority_label = Counter(labels).most_common(1)[0][0]

                self.sequences.append({
                    'case_id': case_id,
                    'utterances': utterances[:window_size],  # Truncate to window
                    'label': majority_label,
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        utterances = seq['utterances']

        # Tokenize
        encoded = self.tokenizer(
            utterances,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        # Pad to window size
        num_utterances = len(utterances)
        if num_utterances < self.window_size:
            pad_size = self.window_size - num_utterances
            input_ids = torch.cat([
                encoded['input_ids'],
                torch.zeros(pad_size, self.max_length, dtype=torch.long)
            ], dim=0)
            attention_mask = torch.cat([
                encoded['attention_mask'],
                torch.zeros(pad_size, self.max_length, dtype=torch.long)
            ], dim=0)
            utterance_mask = torch.cat([
                torch.ones(num_utterances),
                torch.zeros(pad_size)
            ], dim=0)
        else:
            input_ids = encoded['input_ids'][:self.window_size]
            attention_mask = encoded['attention_mask'][:self.window_size]
            utterance_mask = torch.ones(self.window_size)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'utterance_mask': utterance_mask,
            'label': torch.tensor(seq['label'], dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'utterance_mask': torch.stack([b['utterance_mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


def train_single_config(
    window_size: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
    device: str,
) -> Dict:
    """Train and evaluate single window size configuration."""
    console.print(f"\n[bold blue]Training with window_size={window_size}[/bold blue]")

    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"])

    train_dataset = CVRWindowDataset(train_df, tokenizer, window_size=window_size)
    val_dataset = CVRWindowDataset(val_df, tokenizer, window_size=window_size)
    test_dataset = CVRWindowDataset(test_df, tokenizer, window_size=window_size)

    console.print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )

    # Create model
    model = BertLSTMClassifier(
        model_name=config["model"]["encoder"],
        num_labels=config["model"]["num_labels"],
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        max_utterances=window_size,
    )
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    # Training
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            utterance_mask = batch['utterance_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, utterance_mask)
            logits = output["logits"]
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"])
            optimizer.step()

            train_loss += loss.item()

        # Validate
        val_metrics = evaluate(model, val_loader, device, config)
        val_f1 = val_metrics['macro_f1']

        console.print(
            f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
            f"Val F1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= config["training"]["early_stopping_patience"]:
            console.print(f"  [yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break

    # Evaluate best model on test set
    if best_model_state:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device, config)

    console.print(f"  [green]Test Results: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['macro_f1']:.4f}[/green]")

    return {
        'window_size': window_size,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
    }


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str, config: Dict) -> Dict:
    """Evaluate model."""
    model.eval()

    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        utterance_mask = batch['utterance_mask'].to(device)
        labels = batch['label']

        output = model(input_ids, attention_mask, utterance_mask)
        logits = output["logits"]
        preds = torch.argmax(logits, dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Per-class metrics
    labels_present = np.unique(np.concatenate([all_labels, all_preds]))
    report = classification_report(
        all_labels, all_preds,
        labels=range(4),
        target_names=config["data"]["labels"],
        output_dict=True,
        zero_division=0
    )

    per_class_f1 = {label: report[label]['f1-score'] for label in config["data"]["labels"]}
    per_class_recall = {label: report[label]['recall'] for label in config["data"]["labels"]}

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1,
        'per_class_recall': per_class_recall,
        'critical_recall': per_class_recall.get('CRITICAL', 0),
    }


def main():
    """Run ablation study."""
    console.print("\n[bold cyan]Experiment 008: Ablation Study - Window Size[/bold cyan]")

    exp_dir = Path(__file__).parent
    config_path = exp_dir / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Device
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/cyan]")

    # Load data
    console.print("\n[yellow]Loading data...[/yellow]")
    data_path = PROJECT_ROOT / config["data"]["source"]
    df = pd.read_csv(data_path)

    # Map labels
    if df[config["data"]["label_column"]].dtype == object:
        label_map = {label: idx for idx, label in enumerate(config["data"]["labels"])}
        df['label'] = df[config["data"]["label_column"]].map(label_map)
    else:
        df['label'] = df[config["data"]["label_column"]]

    # Split by case
    cases = df['case_id'].unique()
    train_cases, temp_cases = train_test_split(
        cases,
        test_size=config["data"]["test_split"] + config["data"]["val_split"],
        random_state=config["data"]["random_seed"]
    )
    val_ratio = config["data"]["val_split"] / (config["data"]["test_split"] + config["data"]["val_split"])
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=1-val_ratio,
        random_state=config["data"]["random_seed"]
    )

    train_df = df[df['case_id'].isin(train_cases)]
    val_df = df[df['case_id'].isin(val_cases)]
    test_df = df[df['case_id'].isin(test_cases)]

    console.print(f"[green]Train: {len(train_cases)} cases, Val: {len(val_cases)}, Test: {len(test_cases)}[/green]")

    # Run ablation
    window_sizes = config["data"]["window_sizes"]
    results = []

    for window_size in window_sizes:
        result = train_single_config(
            window_size=window_size,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config,
            device=device,
        )
        results.append(result)

    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]Ablation Study Results[/bold green]")
    console.print("="*60)

    table = Table(title="Window Size Ablation")
    table.add_column("Window", style="cyan")
    table.add_column("Val F1", style="yellow")
    table.add_column("Test Acc", style="green")
    table.add_column("Test F1", style="green")
    table.add_column("CRITICAL Recall", style="red")

    for r in results:
        table.add_row(
            str(r['window_size']),
            f"{r['best_val_f1']:.4f}",
            f"{r['test_metrics']['accuracy']:.4f}",
            f"{r['test_metrics']['macro_f1']:.4f}",
            f"{r['test_metrics']['critical_recall']:.2%}",
        )

    console.print(table)

    # Find best
    best = max(results, key=lambda x: x['test_metrics']['macro_f1'])
    console.print(f"\n[bold green]Best window size: {best['window_size']} (F1={best['test_metrics']['macro_f1']:.4f})[/bold green]")

    # Save results
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        'experiment_id': config["experiment"]["id"],
        'experiment_title': config["experiment"]["title"],
        'timestamp': datetime.now().isoformat(),
        'ablation_results': results,
        'best_window_size': best['window_size'],
        'config': config,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir / 'results.json'}[/green]")


if __name__ == "__main__":
    main()
