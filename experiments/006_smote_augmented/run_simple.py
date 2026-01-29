"""
Experiment 006: Simplified - Just Weighted CrossEntropy
"""
import json
import sys
from pathlib import Path
from typing import Dict
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier
from src.utils.config import config as global_config

console = Console()


class CVRSequenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_utterances=20, max_length=128):
        self.tokenizer = tokenizer
        self.max_utterances = max_utterances
        self.max_length = max_length
        
        self.sequences = []
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('turn_number')
            utterances = [str(row['cvr_message']) for _, row in group.iterrows() if pd.notna(row['cvr_message'])]
            if utterances:
                from collections import Counter
                majority_label = Counter(group['label']).most_common(1)[0][0]
                self.sequences.append({
                    'case_id': case_id,
                    'utterances': utterances,
                    'label': majority_label,
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        utterances = seq['utterances'][:self.max_utterances]
        
        encoded = self.tokenizer(
            utterances,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
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
            utterance_mask = torch.cat([torch.ones(num_utterances), torch.zeros(pad_size)], dim=0)
        else:
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            utterance_mask = torch.ones(self.max_utterances)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'utterance_mask': utterance_mask,
            'label': torch.tensor(seq['label'], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'utterance_mask': torch.stack([b['utterance_mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


def main():
    console.print("\n[bold cyan]Experiment 006: Weighted Training[/bold cyan]")
    
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Device: {device}[/cyan]")
    
    # Load data
    data_path = PROJECT_ROOT / config["data"]["source"]
    df = pd.read_csv(data_path)
    
    # Map labels
    label_map = {label: idx for idx, label in enumerate(config["data"]["labels"])}
    if df[config["data"]["label_column"]].dtype == object:
        df['label'] = df[config["data"]["label_column"]].map(label_map)
    else:
        df['label'] = df[config["data"]["label_column"]]
    
    # Split
    cases = df['case_id'].unique()
    train_cases, temp_cases = train_test_split(
        cases, test_size=config["data"]["test_split"] + config["data"]["val_split"],
        random_state=config["data"]["random_seed"]
    )
    val_ratio = config["data"]["val_split"] / (config["data"]["test_split"] + config["data"]["val_split"])
    val_cases, test_cases = train_test_split(temp_cases, test_size=1-val_ratio, random_state=config["data"]["random_seed"])
    
    train_df = df[df['case_id'].isin(train_cases)]
    val_df = df[df['case_id'].isin(val_cases)]
    test_df = df[df['case_id'].isin(test_cases)]
    
    console.print(f"[green]Train: {len(train_cases)} cases, Val: {len(val_cases)}, Test: {len(test_cases)}[/green]")
    
    # Show distribution
    console.print("\nClass distribution:")
    for label, count in train_df['label'].value_counts().sort_index().items():
        print(f"  Class {label}: {count}")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"])
    train_dataset = CVRSequenceDataset(train_df, tokenizer, max_utterances=20, max_length=128)
    val_dataset = CVRSequenceDataset(val_df, tokenizer, max_utterances=20, max_length=128)
    test_dataset = CVRSequenceDataset(test_df, tokenizer, max_utterances=20, max_length=128)
    
    # Create sampler for oversampling
    labels = [seq['label'] for seq in train_dataset.sequences]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels) * 2,
        replacement=True,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = BertLSTMClassifier(
        model_name=config["model"]["encoder"],
        num_labels=4,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
    )
    model.to(device)
    
    # Loss with class weights
    # CRITICAL gets 10x weight
    class_weights_tensor = torch.tensor([1.0, 1.5, 3.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training
    console.print("\n[bold yellow]Training...[/bold yellow]")
    checkpoint_dir = PROJECT_ROOT / "models" / "006"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            utterance_mask = batch['utterance_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask, utterance_mask)
            logits = output["logits"]
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                utterance_mask = batch['utterance_mask'].to(device)
                labels = batch['label']
                
                output = model(input_ids, attention_mask, utterance_mask)
                logits = output["logits"]
                preds = torch.argmax(logits, dim=1).cpu()
                
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        from sklearn.metrics import accuracy_score, f1_score, recall_score
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        critical_recall = recall_score(all_labels, all_preds, labels=[3], average='macro', zero_division=0)
        
        scheduler.step(macro_f1)
        
        console.print(
            f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | "
            f"Val F1: {macro_f1:.4f} | CRITICAL Recall: {critical_recall:.2%}"
        )
        
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            console.print(f"  [green]New best (F1: {macro_f1:.4f})[/green]")
        else:
            patience_counter += 1
        
        if patience_counter >= 7:
            console.print(f"\n[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break
    
    # Test
    console.print("\n[bold cyan]Testing...[/bold cyan]")
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
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
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    report = classification_report(all_labels, all_preds, target_names=["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"], output_dict=True)
    critical_recall = report['CRITICAL']['recall']
    
    console.print("\n[bold green]Test Results:[/bold green]")
    console.print(f"Accuracy: {accuracy:.4f}")
    console.print(f"Macro F1: {macro_f1:.4f}")
    console.print(f"CRITICAL Recall: {critical_recall:.2%}")
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "experiments" / "006"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment_id': '006',
        'experiment_title': 'Weighted Training',
        'metrics': {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'critical_recall': critical_recall,
        }
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]Results saved![/green]")


if __name__ == "__main__":
    main()
