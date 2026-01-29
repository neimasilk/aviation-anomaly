"""
Experiment 007: Cost-Sensitive Cascade Model with Robust Checkpointing

Two-stage architecture:
  Stage 1: Binary anomaly detector (NORMAL vs ANOMALY) - 95% recall target
  Stage 2: 4-class cost-sensitive classifier - Explicit 20x penalty for CRITICAL misses

Checkpoint Features:
  - Auto-save every epoch
  - Resume from interruption automatically
  - Save best models separately
  - Progress tracking in JSON
  - Memory and time tracking

Usage:
    cd experiments/007_cost_sensitive_cascade
    python run.py
    
    # If interrupted, simply run again - will auto-resume
    python run.py

Author: Research Assistant
Date: 2026-01-29
"""
import json
import sys
import os
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier
from src.utils.config import config as global_config

console = Console()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    console.print("\n\n[yellow]Shutdown requested. Saving checkpoint...[/yellow]")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class CheckpointManager:
    """
    Robust checkpoint manager with auto-resume capability.
    """
    
    def __init__(self, checkpoint_dir: Path, config: Dict):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.checkpoint_file = self.checkpoint_dir / config["checkpoint"].get("checkpoint_file", "checkpoint.pt")
        self.progress_file = self.checkpoint_dir / "progress.json"
        
    def save_checkpoint(
        self,
        stage: str,  # 'stage1', 'stage2', or 'cascade'
        epoch: int,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Optional[Dict],
        metrics: Dict,
        best_metrics: Dict,
        extra_state: Optional[Dict] = None,
        is_best: bool = False,
    ):
        """Save comprehensive checkpoint."""
        checkpoint = {
            'stage': stage,
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'metrics': metrics,
            'best_metrics': best_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'extra_state': extra_state or {},
        }
        
        # Save main checkpoint
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        torch.save(checkpoint, temp_file)
        temp_file.replace(self.checkpoint_file)
        
        # Save best model separately
        if is_best:
            best_file = self.checkpoint_dir / f"{stage}_best.pt"
            torch.save(checkpoint, best_file)
            console.print(f"  [green]Best {stage} model saved (epoch {epoch})[/green]")
        
        # Update progress tracking
        self._update_progress(stage, epoch, metrics, best_metrics)
        
        return checkpoint
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            checkpoint = torch.load(self.checkpoint_file, map_location='cpu')
            console.print(f"[green]Checkpoint found: {checkpoint.get('stage', 'unknown')} epoch {checkpoint.get('epoch', 0)}[/green]")
            console.print(f"[green]Resuming from {checkpoint.get('timestamp', 'unknown')}[/green]")
            return checkpoint
        except Exception as e:
            console.print(f"[red]Error loading checkpoint: {e}[/red]")
            return None
    
    def _update_progress(self, stage: str, epoch: int, metrics: Dict, best_metrics: Dict):
        """Update progress tracking file."""
        progress = {
            'last_updated': datetime.now().isoformat(),
            'current_stage': stage,
            'current_epoch': epoch,
            'current_metrics': metrics,
            'best_metrics': best_metrics,
        }
        
        temp_file = self.progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
        temp_file.replace(self.progress_file)
    
    def load_progress(self) -> Optional[Dict]:
        """Load progress tracking."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        return None
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Clean up old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for checkpoint in checkpoints[keep_last:]:
            checkpoint.unlink()
            console.print(f"[yellow]Cleaned up old checkpoint: {checkpoint.name}[/yellow]")


class CostSensitiveLoss(nn.Module):
    """
    Cost-Sensitive Cross Entropy Loss.
    
    Uses explicit cost matrix where misclassifying CRITICAL has high cost.
    """
    
    def __init__(self, cost_matrix: torch.Tensor, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('cost_matrix', cost_matrix)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # (batch, num_classes)
        
        # Get costs for each sample
        # cost_matrix[targets] gives the cost row for each true class
        costs = self.cost_matrix[targets]  # (batch, num_classes)
        
        # Weighted cross-entropy
        # For each sample: -sum_j (cost[j] * prob[j] * log(prob[j]))
        log_probs = F.log_softmax(logits, dim=1)
        
        # Element-wise cost * log_prob
        weighted_log_probs = costs * log_probs  # (batch, num_classes)
        
        # Sum over classes, mean over batch
        loss = -weighted_log_probs.sum(dim=1).mean()
        
        return loss


class CVRBinaryDataset(Dataset):
    """Dataset for Stage 1: Binary classification (NORMAL vs ANOMALY)."""
    
    def __init__(self, df: pd.DataFrame, tokenizer, config: Dict, is_binary: bool = True):
        self.tokenizer = tokenizer
        self.max_utterances = config["data"]["max_utterances"]
        self.max_length = config["data"]["max_utterance_length"]
        self.is_binary = is_binary
        
        # Anomaly classes mapping
        self.anomaly_classes = set(config["data"]["anomaly_classes"])
        self.label_map = {label: idx for idx, label in enumerate(config["data"]["labels"])}
        
        # Build sequences
        self.sequences = []
        for case_id, group in df.groupby(config["data"]["case_id_column"]):
            group = group.sort_values('turn_number') if 'turn_number' in group.columns else group
            
            utterances = []
            labels = []
            
            for _, row in group.iterrows():
                msg = row[config["data"]["text_column"]]
                if pd.notna(msg):
                    utterances.append(str(msg))
                    label = row['label']
                    if isinstance(label, str):
                        label = self.label_map[label]
                    labels.append(label)
            
            if utterances:
                # For binary classification, check if ANY utterance is anomalous
                from collections import Counter
                label_counts = Counter(labels)
                
                if is_binary:
                    # Binary strategy: If ANY utterance is not NORMAL (0), mark as ANOMALY (1)
                    # This ensures we capture cases with any anomaly signal
                    has_anomaly = any(l > 0 for l in labels)  # l > 0 means EARLY, ELEVATED, or CRITICAL
                    binary_label = 1 if has_anomaly else 0
                else:
                    # For multi-class, use majority label
                    majority_label = label_counts.most_common(1)[0][0]
                    binary_label = majority_label
                
                self.sequences.append({
                    'case_id': case_id,
                    'utterances': utterances,
                    'label': binary_label,
                    'original_label': majority_label,
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
    """Runner for cost-sensitive cascade experiment with checkpointing."""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.config_path = exp_dir / "config.yaml"
        self.config = self._load_config()
        self.device = self._get_device()
        
        # Setup checkpoint manager
        checkpoint_dir = PROJECT_ROOT / self.config["paths"]["checkpoint_dir"]
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, self.config)
        
        # State tracking
        self.current_stage = None
        self.current_epoch = 0
        self.best_metrics = {}
        
    def _load_config(self) -> Dict:
        """Load experiment config."""
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
        """Setup directories."""
        exp_id = self.config["experiment"]["id"]
        
        for subdir in ["outputs", "models", "logs"]:
            path = PROJECT_ROOT / subdir / "experiments" / exp_id
            path.mkdir(parents=True, exist_ok=True)
            self.config["paths"][f"{subdir}_dir"] = str(path)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data."""
        console.print("\n[yellow]Loading data...[/yellow]")
        
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        df = pd.read_csv(data_path)
        
        # Map labels if needed
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
            temp_cases, test_size=1-val_ratio,
            random_state=self.config["data"]["random_seed"]
        )
        
        train_df = df[df['case_id'].isin(train_cases)]
        val_df = df[df['case_id'].isin(val_cases)]
        test_df = df[df['case_id'].isin(test_cases)]
        
        console.print(f"[green]Train: {len(train_cases)} cases, Val: {len(val_cases)}, Test: {len(test_cases)}[/green]")
        
        return train_df, val_df, test_df
    
    def train_stage1(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> BertLSTMClassifier:
        """Train Stage 1: Binary anomaly detector."""
        console.print("\n[bold blue]=== Stage 1: Binary Anomaly Detector ===[/bold blue]")
        
        # Check for checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint and checkpoint.get('stage') == 'stage1':
            console.print("[green]Resuming Stage 1 from checkpoint...[/green]")
            return self._resume_stage1(checkpoint, train_df, val_df)
        
        # Create datasets
        tokenizer = AutoTokenizer.from_pretrained(self.config["stage1"]["model"]["encoder"])
        train_dataset = CVRBinaryDataset(train_df, tokenizer, self.config, is_binary=True)
        val_dataset = CVRBinaryDataset(val_df, tokenizer, self.config, is_binary=True)
        
        # Show distribution
        train_labels = [seq['label'] for seq in train_dataset.sequences]
        console.print(f"Binary distribution - NORMAL: {train_labels.count(0)}, ANOMALY: {train_labels.count(1)}")
        
        # Create weighted sampler
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels) * 2,
            replacement=True,
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["stage1"]["training"]["batch_size"],
            sampler=sampler, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["stage1"]["training"]["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        # Create model
        model = BertLSTMClassifier(
            model_name=self.config["stage1"]["model"]["encoder"],
            num_labels=2,  # Binary
            lstm_hidden=self.config["stage1"]["model"]["lstm_hidden"],
            lstm_layers=self.config["stage1"]["model"]["lstm_layers"],
            dropout=self.config["stage1"]["model"]["dropout"],
        )
        model.to(self.device)
        
        # Loss with class weights
        class_weights_tensor = torch.tensor([
            float(self.config["stage1"]["class_weights"]["NORMAL"]),
            float(self.config["stage1"]["class_weights"]["ANOMALY"])
        ]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["stage1"]["training"]["learning_rate"]),
            weight_decay=float(self.config["stage1"]["training"]["weight_decay"]),
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Training loop
        best_recall = 0.0
        patience_counter = 0
        start_epoch = 0
        
        console.print("\n[bold yellow]Training Stage 1...[/bold yellow]")
        
        for epoch in range(start_epoch, self.config["stage1"]["training"]["max_epochs"]):
            if shutdown_requested:
                console.print("[yellow]Shutdown requested. Saving checkpoint...[/yellow]")
                self.checkpoint_manager.save_checkpoint(
                    'stage1', epoch, model.state_dict(), optimizer.state_dict(),
                    scheduler.state_dict() if scheduler else None,
                    {'recall': best_recall}, {'best_recall': best_recall}
                )
                break
            
            # Train
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                output = model(input_ids, attention_mask, utterance_mask)
                logits = output["logits"]
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["stage1"]["training"]["gradient_clip"]
                )
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            val_metrics = self._evaluate_binary(model, val_loader)
            anomaly_recall = val_metrics['recall']  # Recall for anomaly class
            
            scheduler.step(anomaly_recall)
            
            console.print(
                f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
                f"Anomaly Recall: {anomaly_recall:.2%} (target: 95%)"
            )
            
            # Save checkpoint
            is_best = anomaly_recall > best_recall
            if is_best:
                best_recall = anomaly_recall
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.checkpoint_manager.save_checkpoint(
                'stage1', epoch, model.state_dict(), optimizer.state_dict(),
                scheduler.state_dict() if scheduler else None,
                val_metrics, {'best_recall': best_recall}, is_best=is_best
            )
            
            # Early stopping
            if patience_counter >= self.config["stage1"]["training"]["early_stopping_patience"]:
                console.print(f"\n[yellow]Stage 1 early stopping at epoch {epoch+1}[/yellow]")
                break
        
        # Load best model
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        best_checkpoint = torch.load(checkpoint_dir / "stage1_best.pt", map_location=self.device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        console.print(f"\n[green]Stage 1 complete. Best anomaly recall: {best_recall:.2%}[/green]")
        
        return model
    
    def _resume_stage1(self, checkpoint: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> BertLSTMClassifier:
        """Resume Stage 1 from checkpoint."""
        # Implementation for resuming
        console.print(f"[green]Resuming from epoch {checkpoint['epoch']}...[/green]")
        
        # Create datasets and model
        tokenizer = AutoTokenizer.from_pretrained(self.config["stage1"]["model"]["encoder"])
        train_dataset = CVRBinaryDataset(train_df, tokenizer, self.config, is_binary=True)
        val_dataset = CVRBinaryDataset(val_df, tokenizer, self.config, is_binary=True)
        
        train_labels = [seq['label'] for seq in train_dataset.sequences]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels) * 2,
            replacement=True,
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["stage1"]["training"]["batch_size"],
            sampler=sampler, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["stage1"]["training"]["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        model = BertLSTMClassifier(
            model_name=self.config["stage1"]["model"]["encoder"],
            num_labels=2,
            lstm_hidden=self.config["stage1"]["model"]["lstm_hidden"],
            lstm_layers=self.config["stage1"]["model"]["lstm_layers"],
            dropout=self.config["stage1"]["model"]["dropout"],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Restore optimizer
        class_weights_tensor = torch.tensor([
            float(self.config["stage1"]["class_weights"]["NORMAL"]),
            float(self.config["stage1"]["class_weights"]["ANOMALY"])
        ]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["stage1"]["training"]["learning_rate"]),
            weight_decay=float(self.config["stage1"]["training"]["weight_decay"]),
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Continue training
        start_epoch = checkpoint['epoch'] + 1
        best_recall = checkpoint['best_metrics'].get('best_recall', 0.0)
        patience_counter = 0
        
        console.print(f"[green]Resuming training from epoch {start_epoch}...[/green]")
        
        for epoch in range(start_epoch, self.config["stage1"]["training"]["max_epochs"]):
            if shutdown_requested:
                console.print("[yellow]Shutdown requested. Saving checkpoint...[/yellow]")
                self.checkpoint_manager.save_checkpoint(
                    'stage1', epoch, model.state_dict(), optimizer.state_dict(),
                    scheduler.state_dict() if scheduler else None,
                    {'recall': best_recall}, {'best_recall': best_recall}
                )
                break
            
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                output = model(input_ids, attention_mask, utterance_mask)
                logits = output["logits"]
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["stage1"]["training"]["gradient_clip"]
                )
                optimizer.step()
                
                train_loss += loss.item()
            
            val_metrics = self._evaluate_binary(model, val_loader)
            anomaly_recall = val_metrics['recall']
            
            scheduler.step(anomaly_recall)
            
            console.print(
                f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
                f"Anomaly Recall: {anomaly_recall:.2%}"
            )
            
            is_best = anomaly_recall > best_recall
            if is_best:
                best_recall = anomaly_recall
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.checkpoint_manager.save_checkpoint(
                'stage1', epoch, model.state_dict(), optimizer.state_dict(),
                scheduler.state_dict() if scheduler else None,
                val_metrics, {'best_recall': best_recall}, is_best=is_best
            )
            
            if patience_counter >= self.config["stage1"]["training"]["early_stopping_patience"]:
                console.print(f"\n[yellow]Stage 1 early stopping at epoch {epoch+1}[/yellow]")
                break
        
        # Load best
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        best_checkpoint = torch.load(checkpoint_dir / "stage1_best.pt", map_location=self.device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return model
    
    @torch.no_grad()
    def _evaluate_binary(self, model: nn.Module, dataloader: DataLoader) -> Dict:
        """Evaluate binary classifier."""
        model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            utterance_mask = batch['utterance_mask'].to(self.device)
            labels = batch['label']
            
            output = model(input_ids, attention_mask, utterance_mask)
            logits = output["logits"]
            preds = torch.argmax(logits, dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Binary metrics (class 1 = anomaly)
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
        }
    
    def train_stage2(self, train_df: pd.DataFrame, val_df: pd.DataFrame, stage1_model: nn.Module) -> BertLSTMClassifier:
        """Train Stage 2: Cost-sensitive 4-class classifier."""
        console.print("\n[bold blue]=== Stage 2: Cost-Sensitive Classifier ===[/bold blue]")
        
        # Filter to only ANOMALY samples (for efficiency)
        # In practice, we use all samples but with cost-sensitive loss
        
        # Check for checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint and checkpoint.get('stage') == 'stage2':
            console.print("[green]Resuming Stage 2 from checkpoint...[/green]")
            return self._resume_stage2(checkpoint, train_df, val_df)
        
        # Create datasets (4-class)
        tokenizer = AutoTokenizer.from_pretrained(self.config["stage2"]["model"]["encoder"])
        train_dataset = CVRBinaryDataset(train_df, tokenizer, self.config, is_binary=False)
        val_dataset = CVRBinaryDataset(val_df, tokenizer, self.config, is_binary=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["stage2"]["training"]["batch_size"],
            shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["stage2"]["training"]["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        # Create model
        model = BertLSTMClassifier(
            model_name=self.config["stage2"]["model"]["encoder"],
            num_labels=4,
            lstm_hidden=self.config["stage2"]["model"]["lstm_hidden"],
            lstm_layers=self.config["stage2"]["model"]["lstm_layers"],
            dropout=self.config["stage2"]["model"]["dropout"],
        )
        model.to(self.device)
        
        # Cost-sensitive loss
        cost_matrix = torch.tensor([
            self.config["stage2"]["cost_matrix"]["NORMAL"],
            self.config["stage2"]["cost_matrix"]["EARLY"],
            self.config["stage2"]["cost_matrix"]["ELEVATED"],
            self.config["stage2"]["cost_matrix"]["CRITICAL"],
        ]).to(self.device)
        
        criterion = CostSensitiveLoss(cost_matrix)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["stage2"]["training"]["learning_rate"]),
            weight_decay=float(self.config["stage2"]["training"]["weight_decay"]),
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        
        # Training loop
        best_critical_recall = 0.0
        patience_counter = 0
        
        console.print("\n[bold yellow]Training Stage 2 (Cost-Sensitive)...[/bold yellow]")
        
        for epoch in range(self.config["stage2"]["training"]["max_epochs"]):
            if shutdown_requested:
                console.print("[yellow]Shutdown requested. Saving checkpoint...[/yellow]")
                self.checkpoint_manager.save_checkpoint(
                    'stage2', epoch, model.state_dict(), optimizer.state_dict(),
                    scheduler.state_dict() if scheduler else None,
                    {'critical_recall': best_critical_recall}, {'best_critical_recall': best_critical_recall}
                )
                break
            
            # Train
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                output = model(input_ids, attention_mask, utterance_mask)
                logits = output["logits"]
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["stage2"]["training"]["gradient_clip"]
                )
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            val_metrics = self._evaluate_multiclass(model, val_loader)
            critical_recall = val_metrics.get('per_class_recall', {}).get(3, 0)  # Class 3 = CRITICAL
            
            scheduler.step(critical_recall)
            
            console.print(
                f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
                f"CRITICAL Recall: {critical_recall:.2%} (target: 90%) | "
                f"Macro F1: {val_metrics.get('macro_f1', 0):.4f}"
            )
            
            # Save checkpoint
            is_best = critical_recall > best_critical_recall
            if is_best:
                best_critical_recall = critical_recall
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.checkpoint_manager.save_checkpoint(
                'stage2', epoch, model.state_dict(), optimizer.state_dict(),
                scheduler.state_dict() if scheduler else None,
                val_metrics, {'best_critical_recall': best_critical_recall}, is_best=is_best
            )
            
            # Early stopping
            if patience_counter >= self.config["stage2"]["training"]["early_stopping_patience"]:
                console.print(f"\n[yellow]Stage 2 early stopping at epoch {epoch+1}[/yellow]")
                break
        
        # Load best
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        best_checkpoint = torch.load(checkpoint_dir / "stage2_best.pt", map_location=self.device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        console.print(f"\n[green]Stage 2 complete. Best CRITICAL recall: {best_critical_recall:.2%}[/green]")
        
        return model
    
    def _resume_stage2(self, checkpoint: Dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> BertLSTMClassifier:
        """Resume Stage 2 from checkpoint."""
        console.print(f"[green]Resuming Stage 2 from epoch {checkpoint['epoch']}...[/green]")
        
        # Similar to _resume_stage1 but for 4-class
        tokenizer = AutoTokenizer.from_pretrained(self.config["stage2"]["model"]["encoder"])
        train_dataset = CVRBinaryDataset(train_df, tokenizer, self.config, is_binary=False)
        val_dataset = CVRBinaryDataset(val_df, tokenizer, self.config, is_binary=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["stage2"]["training"]["batch_size"],
            shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["stage2"]["training"]["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        model = BertLSTMClassifier(
            model_name=self.config["stage2"]["model"]["encoder"],
            num_labels=4,
            lstm_hidden=self.config["stage2"]["model"]["lstm_hidden"],
            lstm_layers=self.config["stage2"]["model"]["lstm_layers"],
            dropout=self.config["stage2"]["model"]["dropout"],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        cost_matrix = torch.tensor([
            self.config["stage2"]["cost_matrix"]["NORMAL"],
            self.config["stage2"]["cost_matrix"]["EARLY"],
            self.config["stage2"]["cost_matrix"]["ELEVATED"],
            self.config["stage2"]["cost_matrix"]["CRITICAL"],
        ]).to(self.device)
        criterion = CostSensitiveLoss(cost_matrix)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["stage2"]["training"]["learning_rate"]),
            weight_decay=float(self.config["stage2"]["training"]["weight_decay"]),
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_critical_recall = checkpoint['best_metrics'].get('best_critical_recall', 0.0)
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config["stage2"]["training"]["max_epochs"]):
            if shutdown_requested:
                self.checkpoint_manager.save_checkpoint(
                    'stage2', epoch, model.state_dict(), optimizer.state_dict(),
                    scheduler.state_dict() if scheduler else None,
                    {'critical_recall': best_critical_recall}, {'best_critical_recall': best_critical_recall}
                )
                break
            
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                utterance_mask = batch['utterance_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                output = model(input_ids, attention_mask, utterance_mask)
                logits = output["logits"]
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["stage2"]["training"]["gradient_clip"])
                optimizer.step()
                
                train_loss += loss.item()
            
            val_metrics = self._evaluate_multiclass(model, val_loader)
            critical_recall = val_metrics.get('per_class_recall', {}).get(3, 0)
            
            scheduler.step(critical_recall)
            
            console.print(
                f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
                f"CRITICAL Recall: {critical_recall:.2%}"
            )
            
            is_best = critical_recall > best_critical_recall
            if is_best:
                best_critical_recall = critical_recall
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.checkpoint_manager.save_checkpoint(
                'stage2', epoch, model.state_dict(), optimizer.state_dict(),
                scheduler.state_dict() if scheduler else None,
                val_metrics, {'best_critical_recall': best_critical_recall}, is_best=is_best
            )
            
            if patience_counter >= self.config["stage2"]["training"]["early_stopping_patience"]:
                console.print(f"\n[yellow]Stage 2 early stopping at epoch {epoch+1}[/yellow]")
                break
        
        checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"])
        best_checkpoint = torch.load(checkpoint_dir / "stage2_best.pt", map_location=self.device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return model
    
    @torch.no_grad()
    def _evaluate_multiclass(self, model: nn.Module, dataloader: DataLoader) -> Dict:
        """Evaluate 4-class classifier."""
        model.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            utterance_mask = batch['utterance_mask'].to(self.device)
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
        per_class_recall = {}
        per_class_precision = {}
        for i in range(4):
            per_class_recall[i] = recall_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
            per_class_precision[i] = precision_score(all_labels, all_preds, labels=[i], average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_recall': per_class_recall,
            'per_class_precision': per_class_precision,
        }
    
    @torch.no_grad()
    def evaluate_cascade(self, stage1_model: nn.Module, stage2_model: nn.Module, test_df: pd.DataFrame) -> Dict:
        """Evaluate full cascade on test set."""
        console.print("\n[bold cyan]=== Cascade Evaluation ===[/bold cyan]")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config["stage1"]["model"]["encoder"])
        test_dataset = CVRBinaryDataset(test_df, tokenizer, self.config, is_binary=False)
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["stage2"]["training"]["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        stage1_model.eval()
        stage2_model.eval()
        
        all_preds = []
        all_labels = []
        stage1_flags = []
        
        for batch in test_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            utterance_mask = batch['utterance_mask'].to(self.device)
            labels = batch['label']
            
            # Stage 1: Binary detection
            output1 = stage1_model(input_ids, attention_mask, utterance_mask)
            logits1 = output1["logits"]
            probs1 = F.softmax(logits1, dim=1)
            anomaly_scores = probs1[:, 1]  # Probability of ANOMALY
            
            # Stage 2: 4-class classification (for all samples, but we'll mask)
            output2 = stage2_model(input_ids, attention_mask, utterance_mask)
            logits2 = output2["logits"]
            preds2 = torch.argmax(logits2, dim=1).cpu()
            
            # Cascade logic
            batch_preds = []
            for i, score in enumerate(anomaly_scores):
                if score > self.config["cascade"]["stage1_threshold"]:
                    # Stage 1 flags anomaly, use Stage 2 prediction
                    batch_preds.append(preds2[i].item())
                    stage1_flags.append(1)
                else:
                    # Stage 1 says NORMAL
                    batch_preds.append(0)  # NORMAL
                    stage1_flags.append(0)
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        stage1_flags = np.array(stage1_flags)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # CRITICAL recall (class 3)
        critical_recall = recall_score(all_labels, all_preds, labels=[3], average='macro', zero_division=0)
        
        # Stage 1 stats
        stage1_recall = recall_score((all_labels > 0).astype(int), stage1_flags, zero_division=0)
        
        console.print(f"\n[bold green]Cascade Results:[/bold green]")
        console.print(f"  Accuracy: {accuracy:.4f}")
        console.print(f"  Macro F1: {macro_f1:.4f}")
        console.print(f"  [red]CRITICAL Recall: {critical_recall:.2%}[/red] (target: 90%)")
        console.print(f"  Stage 1 Anomaly Recall: {stage1_recall:.2%}")
        console.print(f"  Samples flagged by Stage 1: {stage1_flags.sum()}/{len(stage1_flags)} ({stage1_flags.mean():.1%})")
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'critical_recall': critical_recall,
            'stage1_recall': stage1_recall,
            'flag_rate': stage1_flags.mean(),
        }
    
    def save_results(self, cascade_metrics: Dict, stage1_metrics: Dict, stage2_metrics: Dict):
        """Save final results."""
        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'experiment_id': self.config["experiment"]["id"],
            'experiment_title': self.config["experiment"]["title"],
            'cascade_metrics': cascade_metrics,
            'stage1_metrics': stage1_metrics,
            'stage2_metrics': stage2_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[green]Results saved to {output_dir / 'results.json'}[/green]")

    def run(self):
        """Run full cascade experiment."""
        console.print("\n[bold cyan]Experiment 007: Cost-Sensitive Cascade[/bold cyan]")
        console.print("[cyan]Robust checkpointing enabled - interrupt anytime to save progress[/cyan]\n")
        
        self.setup()
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Stage 1: Train binary detector
        stage1_model = self.train_stage1(train_df, val_df)
        
        # Stage 2: Train cost-sensitive classifier
        stage2_model = self.train_stage2(train_df, val_df, stage1_model)
        
        # Evaluate cascade
        cascade_metrics = self.evaluate_cascade(stage1_model, stage2_model, test_df)
        
        # Get individual stage metrics
        console.print("\n[yellow]Computing individual stage metrics...[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained(self.config["stage1"]["model"]["encoder"])
        test_dataset = CVRBinaryDataset(test_df, tokenizer, self.config, is_binary=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
        stage1_metrics = self._evaluate_binary(stage1_model, test_loader)
        
        test_dataset_4class = CVRBinaryDataset(test_df, tokenizer, self.config, is_binary=False)
        test_loader_4class = DataLoader(test_dataset_4class, batch_size=8, shuffle=False, collate_fn=collate_fn)
        stage2_metrics = self._evaluate_multiclass(stage2_model, test_loader_4class)
        
        # Save results
        self.save_results(cascade_metrics, stage1_metrics, stage2_metrics)
        
        # Final summary
        console.print("\n" + "="*60)
        console.print("[bold green]Experiment 007 Complete![/bold green]")
        console.print("="*60)
        console.print(f"\n[bold]Key Results:[/bold]")
        console.print(f"  Stage 1 Anomaly Recall: {stage1_metrics['recall']:.2%}")
        console.print(f"  Stage 2 CRITICAL Recall: {stage2_metrics['per_class_recall'].get(3, 0):.2%}")
        console.print(f"  [red]Cascade CRITICAL Recall: {cascade_metrics['critical_recall']:.2%}[/red]")
        console.print(f"  Cascade Accuracy: {cascade_metrics['accuracy']:.4f}")
        console.print(f"  Cascade Macro F1: {cascade_metrics['macro_f1']:.4f}")
        
        if cascade_metrics['critical_recall'] >= 0.90:
            console.print(f"\n[bold green]🎉 TARGET ACHIEVED: CRITICAL recall > 90%![/bold green]")
        elif cascade_metrics['critical_recall'] >= 0.80:
            console.print(f"\n[bold yellow]⚠ Good progress: CRITICAL recall > 80%[/bold yellow]")
        else:
            console.print(f"\n[bold red]❌ Below target: CRITICAL recall < 80%[/bold red]")


def main():
    """Main entry point."""
    exp_dir = Path(__file__).parent
    runner = ExperimentRunner(exp_dir)
    runner.run()


if __name__ == "__main__":
    main()
