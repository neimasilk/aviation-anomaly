"""
Experiment 005: Change Point Detection
Anomaly Onset Detection via Distribution Shift Analysis

Usage:
    cd experiments/005_change_point
    python run.py

Or from project root:
    python -m experiments.005_change_point.run
"""
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import yaml
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.model_selection import train_test_split
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.change_point_detector import ChangePointDetector, create_model, create_tokenizer
from src.utils.config import config as global_config

console = Console()


class CVRChangePointDataset(Dataset):
    """Dataset for change point detection."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_utterances: int = 50,
        max_length: int = 128,
        window_size: int = 10,
        anomaly_labels: List[int] = [1, 2, 3],
    ):
        self.tokenizer = tokenizer
        self.max_utterances = max_utterances
        self.max_length = max_length
        self.window_size = window_size
        self.anomaly_labels = set(anomaly_labels)
        
        # Group by case
        self.cases = []
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('turn_number')
            
            # Filter out NaN utterances and ensure all are strings
            utterances = []
            raw_labels = []
            for _, row in group.iterrows():
                msg = row['cvr_message']
                if pd.notna(msg) and str(msg).strip():
                    utterances.append(str(msg))
                    raw_labels.append(row.get('label', 0))
            
            if len(utterances) == 0:
                continue  # Skip empty cases
            
            # Find change point (first transition to anomaly)
            change_point = self._find_change_point(raw_labels)
            
            self.cases.append({
                'case_id': case_id,
                'utterances': utterances,
                'labels': raw_labels,
                'change_point': change_point,  # In utterance index
            })
    
    def _find_change_point(self, labels: List[int]) -> int:
        """Find first transition from normal (0) to anomaly."""
        for i, label in enumerate(labels):
            if label in self.anomaly_labels:
                return max(0, i - self.window_size)  # Adjust for window
        return len(labels) - 1  # Default to end if no anomaly
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        utterances = case['utterances'][:self.max_utterances]
        
        # Tokenize
        encoded = self.tokenizer(
            utterances,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Pad utterances if needed
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
            'change_point': torch.tensor(case['change_point'], dtype=torch.float),
            'num_utterances': num_utterances,
            'case_id': case['case_id'],
        }


def collate_fn(batch):
    """Custom collate for variable length sequences."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'utterance_mask': torch.stack([b['utterance_mask'] for b in batch]),
        'change_point': torch.stack([b['change_point'] for b in batch]),
        'num_utterances': [b['num_utterances'] for b in batch],
        'case_id': [b['case_id'] for b in batch],
    }


class ChangePointLoss(nn.Module):
    """Combined loss for change point detection."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        early_detection_weight: float = 0.5,
        smoothness_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.early_detection_weight = early_detection_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        pred_change_point: torch.Tensor,
        true_change_point: torch.Tensor,
        dissimilarity_curve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred_change_point: (batch,) predicted change points
            true_change_point: (batch,) ground truth change points
            dissimilarity_curve: (batch, num_windows) for smoothness constraint
        """
        # MSE loss
        mse_loss = F.mse_loss(pred_change_point, true_change_point)
        
        # Early detection reward (penalize late detection more)
        diff = pred_change_point - true_change_point
        early_penalty = torch.where(
            diff > 0,  # Late detection
            diff * 2.0,  # Penalize 2x for late
            diff.abs() * 0.5  # Penalize 0.5x for early
        ).mean()
        
        # Smoothness loss (penalize jagged dissimilarity curves)
        smoothness_loss = 0.0
        if dissimilarity_curve is not None and self.smoothness_weight > 0:
            diff = dissimilarity_curve[:, 1:] - dissimilarity_curve[:, :-1]
            smoothness_loss = diff.pow(2).mean()
        
        total_loss = (
            self.mse_weight * mse_loss +
            self.early_detection_weight * early_penalty +
            self.smoothness_weight * smoothness_loss
        )
        
        return total_loss


class ExperimentRunner:
    """Runner for change point detection experiments."""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.config_path = exp_dir / "config.yaml"
        self.config = self._load_config()
        self.results: Dict[str, Any] = {}
        self.best_val_mae = float('inf')
        
    def _load_config(self) -> Dict:
        """Load experiment config."""
        if not self.config_path.exists():
            console.print(f"[red]Config not found: {self.config_path}[/red]")
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        
        device_env = global_config.get_env("DEVICE", "auto")
        if device_env != "auto":
            cfg["device"] = device_env
        
        return cfg
    
    def _get_device(self) -> str:
        """Get device for training."""
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
        return device_cfg
    
    def setup(self):
        """Setup experiment directories."""
        exp_id = self.config["experiment"]["id"]
        
        output_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
        checkpoint_dir = PROJECT_ROOT / "models" / exp_id
        log_dir = PROJECT_ROOT / "logs" / exp_id
        
        for dir_path in [output_dir, checkpoint_dir, log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.config["paths"]["output_dir"] = str(output_dir)
        self.config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
        self.config["paths"]["log_dir"] = str(log_dir)
        
        console.print(f"[green]Directories created:[/green]")
        console.print(f"  Output: {output_dir}")
        console.print(f"  Checkpoints: {checkpoint_dir}")
        console.print(f"  Logs: {log_dir}")
    
    def print_info(self):
        """Print experiment info."""
        exp = self.config["experiment"]
        
        console.print(f"\n[bold blue]{'='*70}[/bold blue]")
        console.print(f"[bold blue]Experiment {exp['id']}: {exp['title']}[/bold blue]")
        console.print(f"[bold blue]{'='*70}[/bold blue]\n")
        
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Description", exp.get("description", "N/A"))
        table.add_row("Tags", ", ".join(exp.get("tags", [])))
        table.add_row("Status", exp.get("status", "unknown"))
        table.add_row("", "")
        table.add_row("Model", self.config["model"]["type"])
        table.add_row("Shift Metric", self.config["model"]["shift_metric"])
        table.add_row("Window Size", str(self.config["data"]["window_size"]))
        table.add_row("", "")
        table.add_row("Batch Size", str(self.config["training"]["batch_size"]))
        table.add_row("Learning Rate", str(self.config["training"]["learning_rate"]))
        table.add_row("Max Epochs", str(self.config["training"]["max_epochs"]))
        table.add_row("", "")
        table.add_row("Device", self._get_device())
        
        console.print(table)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data."""
        console.print("\n[yellow]Loading data...[/yellow]")
        
        data_source = self.config["data"]["source"]
        data_path = PROJECT_ROOT / data_source
        
        if not data_path.exists():
            console.print(f"[red]Data not found: {data_path}[/red]")
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        # Load CSV
        df = pd.read_csv(data_path)
        console.print(f"[green]Loaded {len(df)} utterances from {len(df['case_id'].unique())} cases[/green]")
        
        # Create label mapping
        label_map = {label: idx for idx, label in enumerate(self.config["data"]["labels"])}
        df['label'] = df[self.config["data"]["label_column"]].map(label_map)
        
        # Split by case
        cases = df['case_id'].unique()
        train_cases, temp_cases = train_test_split(
            cases, test_size=self.config["data"]["test_split"] + self.config["data"]["val_split"],
            random_state=self.config["data"]["random_seed"]
        )
        val_ratio = self.config["data"]["val_split"] / (self.config["data"]["test_split"] + self.config["data"]["val_split"])
        val_cases, test_cases = train_test_split(
            temp_cases, test_size=1-val_ratio,
            random_state=self.config["data"]["random_seed"]
        )
        
        train_df = df[df['case_id'].isin(train_cases)]
        val_df = df[df['case_id'].isin(val_cases)]
        test_df = df[df['case_id'].isin(test_cases)]
        
        console.print(f"[green]Train: {len(train_cases)} cases, Val: {len(val_cases)} cases, Test: {len(test_cases)} cases[/green]")
        
        # Create tokenizer and datasets
        tokenizer = create_tokenizer(self.config["model"]["encoder"])
        
        train_dataset = CVRChangePointDataset(
            train_df, tokenizer,
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
            window_size=self.config["data"]["window_size"],
            anomaly_labels=self.config["data"]["anomaly_labels"],
        )
        val_dataset = CVRChangePointDataset(
            val_df, tokenizer,
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
            window_size=self.config["data"]["window_size"],
            anomaly_labels=self.config["data"]["anomaly_labels"],
        )
        test_dataset = CVRChangePointDataset(
            test_df, tokenizer,
            max_utterances=self.config["data"]["max_utterances"],
            max_length=self.config["data"]["max_utterance_length"],
            window_size=self.config["data"]["window_size"],
            anomaly_labels=self.config["data"]["anomaly_labels"],
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
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
        
        return train_loader, val_loader, test_loader, tokenizer
    
    def build_model(self) -> ChangePointDetector:
        """Build model from config."""
        model_cfg = self.config["model"]
        
        model = create_model(
            model_name=model_cfg["encoder"],
            embedding_dim=model_cfg["embedding_dim"],
            freeze_encoder=model_cfg.get("freeze_encoder", True),
            window_size=self.config["data"]["window_size"],
            shift_metric=model_cfg["shift_metric"],
            smoothing_window=model_cfg["smoothing_window"],
            use_learnable_detector=model_cfg.get("use_learnable_detector", True),
            detector_hidden=model_cfg.get("detector_hidden", 256),
            detector_layers=model_cfg.get("detector_layers", 2),
            max_utterances=self.config["data"]["max_utterances"],
            dropout=model_cfg.get("dropout", 0.3),
        )
        
        console.print(f"[green]Model created: {model_cfg['type']}[/green]")
        console.print(f"  Shift metric: {model_cfg['shift_metric']}")
        console.print(f"  Learnable detector: {model_cfg.get('use_learnable_detector', True)}")
        return model
    
    def evaluate(
        self,
        model: ChangePointDetector,
        dataloader: DataLoader,
        device: str,
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        all_preds = []
        all_targets = []
        all_num_utts = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                utterance_mask = batch['utterance_mask'].to(device)
                change_point = batch['change_point'].to(device)
                
                output = model(input_ids, attention_mask, utterance_mask)
                pred_cp = output['change_point']
                
                all_preds.extend(pred_cp.cpu().numpy())
                all_targets.extend(change_point.cpu().numpy())
                all_num_utts.extend(batch['num_utterances'])
        
        # Convert to arrays
        preds = np.array(all_preds)
        targets = np.array(all_targets)
        num_utts = np.array(all_num_utts)
        
        # Compute metrics
        mae = np.abs(preds - targets).mean()
        
        # Within tolerance
        acc_at_1 = (np.abs(preds - targets) <= 1).mean()
        acc_at_3 = (np.abs(preds - targets) <= 3).mean()
        acc_at_5 = (np.abs(preds - targets) <= 5).mean()
        
        # Early detection (predict before actual)
        early = preds < targets
        early_rate = early.mean()
        mean_early_margin = (targets[early] - preds[early]).mean() if early.any() else 0.0
        
        # Late detection
        late_rate = (preds > targets).mean()
        
        return {
            'mae': mae,
            'mae_time_minutes': mae * self.config["evaluation"]["utterance_to_minutes"],
            'accuracy_at_1': acc_at_1,
            'accuracy_at_3': acc_at_3,
            'accuracy_at_5': acc_at_5,
            'early_detection_rate': early_rate,
            'late_detection_rate': late_rate,
            'mean_early_margin': mean_early_margin,
        }
    
    def train_epoch(
        self,
        model: ChangePointDetector,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: ChangePointLoss,
        device: str,
    ) -> float:
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            utterance_mask = batch['utterance_mask'].to(device)
            change_point = batch['change_point'].to(device)
            
            optimizer.zero_grad()
            
            output = model(input_ids, attention_mask, utterance_mask)
            loss = criterion(
                output['change_point'],
                change_point,
                output.get('dissimilarity_curve'),
            )
            
            loss.backward()
            
            # Gradient clipping
            if self.config["training"].get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["training"]["gradient_clip"]
                )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self):
        """Run training loop."""
        console.print("\n[bold yellow]Starting training...[/bold yellow]")
        
        device = self._get_device()
        console.print(f"[cyan]Using device: {device}[/cyan]")
        
        model = self.build_model()
        model.to(device)
        
        # Load data
        train_loader, val_loader, test_loader, tokenizer = self.load_data()
        
        # Setup training
        criterion = ChangePointLoss(
            mse_weight=self.config["training"]["mse_weight"],
            early_detection_weight=self.config["training"]["early_detection_weight"],
            smoothness_weight=self.config["training"]["smoothness_weight"],
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Training loop
        best_val_mae = float('inf')
        patience_counter = 0
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        
        for epoch in range(self.config["training"]["max_epochs"]):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate
            val_metrics = self.evaluate(model, val_loader, device)
            val_mae = val_metrics['mae']
            
            scheduler.step(val_mae)
            
            # Print progress
            console.print(
                f"Epoch {epoch+1}/{self.config['training']['max_epochs']} | "
                f"Loss: {train_loss:.4f} | "
                f"Val MAE: {val_mae:.2f} utterances ({val_metrics['mae_time_minutes']:.1f} min) | "
                f"Early detection: {val_metrics['early_detection_rate']:.2%}"
            )
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
                console.print(f"  [green]New best model saved (MAE: {val_mae:.2f})[/green]")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config["training"]["early_stopping_patience"]:
                console.print(f"\n[yellow]Early stopping triggered after {epoch+1} epochs[/yellow]")
                break
        
        # Load best model and evaluate on test
        console.print("\n[bold cyan]Evaluating best model on test set...[/bold cyan]")
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
        test_metrics = self.evaluate(model, test_loader, device)
        
        console.print("\n[bold green]Test Results:[/bold green]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("MAE (utterances)", f"{test_metrics['mae']:.2f}")
        table.add_row("MAE (minutes)", f"{test_metrics['mae_time_minutes']:.2f}")
        table.add_row("Accuracy @ ±1 utt", f"{test_metrics['accuracy_at_1']:.2%}")
        table.add_row("Accuracy @ ±3 utt", f"{test_metrics['accuracy_at_3']:.2%}")
        table.add_row("Accuracy @ ±5 utt", f"{test_metrics['accuracy_at_5']:.2%}")
        table.add_row("Early Detection Rate", f"{test_metrics['early_detection_rate']:.2%}")
        table.add_row("Late Detection Rate", f"{test_metrics['late_detection_rate']:.2%}")
        table.add_row("Mean Early Margin", f"{test_metrics['mean_early_margin']:.2f} utt")
        
        console.print(table)
        
        # Save results
        self._save_results(test_metrics)
        
        return test_metrics
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save experiment results."""
        exp_id = self.config["experiment"]["id"]
        results_path = PROJECT_ROOT / "outputs" / "experiments" / exp_id / "results.json"
        
        results = {
            "experiment_id": exp_id,
            "experiment_title": self.config["experiment"]["title"],
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "metrics": metrics,
        }
        
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[green]Results saved to: {results_path}[/green]")
    
    def run(self):
        """Run full experiment."""
        self.print_info()
        self.setup()
        
        try:
            metrics = self.train()
            self.config["experiment"]["status"] = "completed"
        except Exception as e:
            console.print(f"\n[red]Experiment failed: {e}[/red]")
            self.config["experiment"]["status"] = "failed"
            raise


def main():
    """Main entry point."""
    exp_dir = Path(__file__).parent
    runner = ExperimentRunner(exp_dir)
    runner.run()


if __name__ == "__main__":
    main()
