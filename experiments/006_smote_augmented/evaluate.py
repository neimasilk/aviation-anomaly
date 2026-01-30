"""
Evaluate Experiment 006: SMOTE-Augmented Model
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier
from src.utils.config import config as global_config
from run import CVRSequenceDataset, collate_fn

console = Console()


def evaluate():
    """Evaluate trained model."""
    console.print("\n[bold cyan]Evaluating Experiment 006: SMOTE-Augmented Model[/bold cyan]")
    
    exp_dir = Path(__file__).parent
    config_path = exp_dir / "config.yaml"
    
    # Load config
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"[cyan]Device: {device}[/cyan]")
    
    # Load data
    data_path = PROJECT_ROOT / config["data"]["source"]
    df = pd.read_csv(data_path)
    
    if df[config["data"]["label_column"]].dtype == object:
        label_map = {label: idx for idx, label in enumerate(config["data"]["labels"])}
        df['label'] = df[config["data"]["label_column"]].map(label_map)
    else:
        df['label'] = df[config["data"]["label_column"]]
    
    # Split
    cases = df['case_id'].unique()
    train_cases, temp_cases = train_test_split(
        cases,
        test_size=config["data"]["test_split"] + config["data"]["val_split"],
        random_state=config["data"]["random_seed"]
    )
    val_ratio = config["data"]["val_split"] / (config["data"]["test_split"] + config["data"]["val_split"])
    val_cases, test_cases = train_test_split(
        temp_cases,
        test_size=1-val_ratio,
        random_state=config["data"]["random_seed"]
    )
    
    test_df = df[df['case_id'].isin(test_cases)]
    
    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"])
    test_dataset = CVRSequenceDataset(
        test_df, tokenizer,
        max_utterances=config["data"]["max_utterances"],
        max_length=config["data"]["max_utterance_length"],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Load model
    model = BertLSTMClassifier(
        model_name=config["model"]["encoder"],
        num_labels=config["model"]["num_labels"],
        lstm_hidden=config["model"]["lstm_hidden"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
    )
    
    checkpoint_path = PROJECT_ROOT / "models" / "006" / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    console.print(f"[green]Loaded model from {checkpoint_path}[/green]")
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            utterance_mask = batch['utterance_mask'].to(device)
            labels = batch['label']
            
            logits = model(input_ids, attention_mask, utterance_mask)
            preds = torch.argmax(logits, dim=1).cpu()
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    report = classification_report(
        all_labels, all_preds,
        target_names=config["data"]["labels"],
        output_dict=True
    )
    
    per_class_f1 = {label: report[label]['f1-score'] for label in config["data"]["labels"]}
    per_class_recall = {label: report[label]['recall'] for label in config["data"]["labels"]}
    per_class_precision = {label: report[label]['precision'] for label in config["data"]["labels"]}
    
    critical_recall = per_class_recall.get("CRITICAL", 0)
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'critical_recall': critical_recall,
        'per_class_f1': per_class_f1,
        'per_class_recall': per_class_recall,
        'per_class_precision': per_class_precision,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "experiments" / "006"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment_id': config["experiment"]["id"],
        'experiment_title': config["experiment"]["title"],
        'metrics': metrics,
        'config': config,
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
    
    console.print("\n[bold]Per-Class Metrics:[/bold]")
    class_table = Table()
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Precision", style="green")
    class_table.add_column("Recall", style="yellow")
    class_table.add_column("F1", style="magenta")
    
    for label in config["data"]["labels"]:
        class_table.add_row(
            label,
            f"{per_class_precision[label]:.4f}",
            f"{per_class_recall[label]:.4f}",
            f"{per_class_f1[label]:.4f}",
        )
    
    console.print(class_table)
    
    return metrics


if __name__ == "__main__":
    evaluate()
