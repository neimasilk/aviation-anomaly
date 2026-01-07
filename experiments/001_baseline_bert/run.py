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
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
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

    def _load_config(self) -> Dict:
        """Load experiment config."""
        config_path = self.exp_dir / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _get_device(self) -> str:
        """Get device for training."""
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = global_config.device
        return device_cfg

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

        console.print(f"[green]✓ Directories created[/green]")

    def print_info(self):
        """Print experiment info."""
        exp = self.config["experiment"]

        console.print(f"\n[bold cyan]{'═'*60}[/bold cyan]")
        console.print(f"[bold cyan]Experiment {exp['id']}: {exp['title']}[/bold cyan]")
        console.print(f"[bold cyan]{'═'*60}[/bold cyan]\n")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan", width=20)
        table.add_column("Value", style="yellow")

        table.add_row("Description", exp.get("description", ""))
        table.add_row("Tags", ", ".join(exp.get("tags", [])))
        table.add_row("Status", exp.get("status", ""))
        table.add_row("", "")
        table.add_row("Model", self.config["model"]["type"])
        table.add_row("Encoder", self.config["model"]["encoder"])
        table.add_row("Batch Size", str(self.config["training"]["batch_size"]))
        table.add_row("Learning Rate", str(self.config["training"]["learning_rate"]))
        table.add_row("Max Epochs", str(self.config["training"]["max_epochs"]))
        table.add_row("", "")
        table.add_row("Expected Accuracy", self.config["expected"]["accuracy"])
        table.add_row("Device", self._get_device())

        console.print(table)

    def check_data(self) -> bool:
        """Check if data exists."""
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        if not data_path.exists():
            console.print(f"\n[red]✗ Data not found: {data_path}[/red]")
            console.print("\n[yellow]Next steps:[/yellow]")
            console.print("  1. Download Noort et al. (2021) dataset:")
            console.print("     https://doi.org/10.1016/j.dib.2021.107602")
            console.print("  2. Extract to data/raw/")
            console.print("  3. Run preprocessing:")
            console.print("     python -m src.data.preprocessing")
            return False
        return True

    def run(self):
        """Run experiment."""
        self.print_info()
        self.setup()

        if not self.check_data():
            return

        console.print("\n[green]✓ All checks passed![/green]")
        console.print("\n[yellow]Training implementation will be added after data is available.[/yellow]")

        # Save placeholder results
        self._save_results({
            "status": "pending",
            "message": "Waiting for dataset",
            "experiment": self.config["experiment"]["id"],
        })

    def _save_results(self, results: Dict[str, Any]):
        """Save results."""
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]✓ Results saved to: {results_path}[/green]")


def main():
    runner = Experiment001Runner()
    runner.run()


if __name__ == "__main__":
    main()
