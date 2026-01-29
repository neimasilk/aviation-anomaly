"""
Experiment 006: SMOTE-Augmented Training for Class Imbalance

This experiment addresses the extreme class imbalance (14:1 NORMAL:CRITICAL)
by using aggressive class weighting and strategic oversampling.

Key innovations:
1. Cost-sensitive learning (20x penalty for CRITICAL misses)
2. Aggressive class weighting
3. Focal Loss for hard example mining
4. Target: CRITICAL recall > 70%

Usage:
    cd experiments/006_smote_augmented
    python run.py
"""
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

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


class CVRSequenceDataset(Dataset):
    """Dataset for CVR sequences with label mapping."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_utterances: int = 20,
        max_length: int = 128,
        label_map: Dict[str, int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_utterances = max_utterances
        self.max_length = max_length
        self.label_map = label_map or {
            "NORMAL": 0,
            "EARLY_WARNING": 1,
            "ELEVATED": 2,
            "CRITICAL": 3,
        }
        
        # Group by case
        self.sequences = []
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('turn_number')
            
            utterances = []
            labels = []
            
            for _, row in group.iterrows():
                if pd.notna(row['cvr_message']):
                    utterances.append(str(row['cvr_message']))
                    labels.append(row['label'])
            
            if utterances:
                # Use majority label for sequence
                from collections import Counter
                majority_label = Counter(labels).most_common(1)[0][0]
                
                # If already integer, use directly; otherwise map
                if isinstance(majority_label, int):
                    label_idx = majority_label
                else:
                    label_idx = self.label_map.get(majority_label, 0)
                
                self.sequences.append({
                    'case_id': case_id,
                    'utterances': utterances,
                    'label': label_idx,
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        utterances = seq['utterances'][:self.max_utterances]
        
        # Tokenize
        encoded = self.tokenizer(
            utterances,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Pad utterances
        num_utterances = len(utterances)
        if num_utterances < self.max_utterances:
            pad_size = self.max_utterances - num_utterances
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
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            utterance_mask = torch.ones(self.max_utterances)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'utterance_mask': utterance_mask,
            'label': torch.tensor(seq['label'], dtype=torch.long),
            'case_id': seq['case_id'],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'utterance_mask': torch.stack([b['utterance_mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'case_id': [b['case_id'] for b in batch],
    }


class ExperimentRunner:
    """Runner for SMOTE augmentation experiment."""
    
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
        """Load and prepare data."""
        console.print("\n[yellow]Loading data...[/yellow]")
        
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        df = pd.read_csv(data_path)
        
        # Map labels if string
        if df[self.config["data"]["label_column"]].dtype == object:
            label_map = {label: idx for idx, label in enumerate(self.config["data"]["labels"])}
            df['label'] = df[self.config["data"]["label_column"]].map(label_map)
        else:
            df['label'] = df[self.config["data"]["label_column"]]
        
        # Split by case
        cases = df['case_id'].unique()
        train_cases, temp_cases = train_test_split(
            cases,
            test_size=self.config["data"]["test_split"] + self.config["data"]["val_split"],
            random_state=self.config["data"]["random_seed"]
        )
        val_ratio = self.config["data"]["val_split"] / (self.config["data"]["test_split"] + self.config["data"]["val_split"])
        val_cases, test_cases = train_test_split(
            temp_cases,
            test_size=1-val_ratio,
            random_state=self.config["data"]["random_seed"]
        )
        
        train_df = df[df['case_id'].isin(train_cases)]
        val_df = df[df['case_id'].isin(val_cases)]
        test_df = df[df['case_id'].isin(test_cases)]
        
        console.print(f"[green]Train: {len(train_cases)} cases, Val: {len(val_cases)}, Test: {len(test_cases)}[/green]")
        
        # Show class distribution
        console.print("\nClass distribution (original):")
        for label, count in train_df['label'].value_counts().sort_index().items():
            label_name = self.config["data"]["labels"][label]
            pct = count / len(train_df) * 100
            console.print(f"  {label_name}: {count} ({pct:.1f}%)")
        
        # Create datasets
        tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["encoder"])
        
        train_dataset = CVRSequenceDataset(
            train_df, tokenizer,
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
        )
        val_dataset = CVRSequenceDataset(
            val_df, tokenizer,
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
        )
        test_dataset = CVRSequenceDataset(
            test_df, tokenizer,
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
        )
        
        return train_dataset, val_dataset, test_dataset, tokenizer
    
    def create_weighted_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        """Create sampler that oversamples minority classes."""
        # Count labels
        labels = [seq['label'] for seq in dataset.sequences]
        class_counts = np.bincount(labels)
        
        # Calculate weights (inverse frequency)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels) * 2,  # Double the dataset size
            replacement=True,
        )
    
    def train(self):
        """Run training."""
        console.print("\n[bold cyan]Experiment 006: SMOTE-Augmented Training[/bold cyan]")
        
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        console.print(f"[cyan]Device: {device}[/cyan]")
        
        # Load data
        train_dataset, val_dataset, test_dataset, tokenizer = self.load_data()
        
        # Create model
        model = BertLSTMClassifier(
            model_name=self.config["model"]["encoder"],
            num_labels=self.config["model"]["num_labels"],
            lstm_hidden=self.config["model"]["lstm_hidden"],
            lstm_layers=self.config["model"]["lstm_layers"],
            dropout=self.config["model"]["dropout"],
        )
        model.to(device)
        
        # Create weighted sampler for oversampling
        sampler = self.create_weighted_sampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            sampler=sampler,
            collate_fn=collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        # Loss function with aggressive weighting
        # Class weights: [1.0, 1.5, 3.0, 10.0]
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
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                utterance_mask = batch['utterance_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, attention_mask, utterance_mask)

                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["training"]["gradient_clip"]
                )
                optimizer.step()
                
                train_loss += loss.item()
            
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
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
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
            labels = batch['label']
            
            logits = model(input_ids, attention_mask, utterance_mask)
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
            target_names=self.config["data"]["labels"],
            output_dict=True
        )
        
        per_class_f1 = {
            label: report[label]['f1-score']
            for label in self.config["data"]["labels"]
        }
        per_class_recall = {
            label: report[label]['recall']
            for label in self.config["data"]["labels"]
        }
        
        # Critical recall specifically
        critical_idx = self.config["data"]["labels"].index("CRITICAL")
        critical_recall = per_class_recall.get("CRITICAL", 0)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'per_class_recall': per_class_recall,
            'critical_recall': critical_recall,
        }
    
    def save_results(self, metrics):
        """Save results."""
        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'experiment_id': self.config["experiment"]["id"],
            'experiment_title': self.config["experiment"]["title"],
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
