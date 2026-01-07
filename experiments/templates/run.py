"""
Experiment {XXX}: {TITLE}

Usage:
    cd experiments/{XXX}
    python run.py

Or from project root:
    python -m experiments.{XXX}.run
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml
from rich.console import Console
from rich.table import Table

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import CVRPreprocessor
from src.models.bert_lstm import BertLSTMClassifier, create_tokenizer
from src.utils.config import config as global_config

console = Console()


class ExperimentRunner:
    """Runner for training experiments."""

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.config_path = exp_dir / "config.yaml"
        self.config = self._load_config()
        self.results: Dict[str, Any] = {}

    def _load_config(self) -> Dict:
        """Load experiment config."""
        if not self.config_path.exists():
            console.print(f"[red]Config not found: {self.config_path}[/red]")
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        # Override device with env var if set
        device_env = global_config.get_env("DEVICE", "auto")
        if device_env != "auto":
            cfg["paths"]["device"] = device_env
            cfg["device"] = device_env

        return cfg

    def _get_device(self) -> str:
        """Get device for training."""
        device_cfg = self.config.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = global_config.device
        return device_cfg

    def setup(self):
        """Setup experiment directories."""
        exp_id = self.config["experiment"]["id"]

        # Create directories
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

        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]Experiment {exp['id']}: {exp['title']}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

        # Create info table
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Description", exp.get("description", "N/A"))
        table.add_row("Tags", ", ".join(exp.get("tags", [])))
        table.add_row("Status", exp.get("status", "unknown"))
        table.add_row("", "")
        table.add_row("Model", self.config["model"]["type"])
        table.add_row("Encoder", self.config["model"]["encoder"])
        table.add_row("Batch Size", str(self.config["training"]["batch_size"]))
        table.add_row("Learning Rate", str(self.config["training"]["learning_rate"]))
        table.add_row("Max Epochs", str(self.config["training"]["max_epochs"]))
        table.add_row("", "")
        table.add_row("Device", self._get_device())

        console.print(table)

    def load_data(self):
        """Load and prepare data."""
        console.print("\n[yellow]Loading data...[/yellow]")

        data_source = self.config["data"]["source"]
        data_path = PROJECT_ROOT / data_source

        if not data_path.exists():
            console.print(f"[red]Data not found: {data_path}[/red]")
            console.print("[yellow]Please run preprocessing first:[/yellow]")
            console.print(f"  python -m src.data.preprocessing")
            return None, None, None

        # TODO: Implement actual data loading
        console.print(f"[green]Data found: {data_path}[/green]")
        console.print("[yellow]Data loading to be implemented when dataset is available[/yellow]")

        return None, None, None

    def build_model(self) -> BertLSTMClassifier:
        """Build model from config."""
        model_cfg = self.config["model"]

        model = BertLSTMClassifier(
            model_name=model_cfg["encoder"],
            num_labels=model_cfg["num_labels"],
            lstm_hidden=model_cfg.get("lstm_hidden", 256),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            dropout=model_cfg.get("dropout", 0.3),
            max_utterances=model_cfg.get("max_utterances", 20),
        )

        console.print(f"[green]Model created: {model_cfg['type']}[/green]")
        return model

    def train(self):
        """Run training loop."""
        console.print("\n[bold yellow]Starting training...[/bold yellow]")

        # Setup
        device = self._get_device()
        model = self.build_model()
        model.to(device)

        tokenizer = create_tokenizer(self.config["model"]["encoder"])

        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        if train_loader is None:
            console.print("[red]Data loading failed. Exiting.[/red]")
            return

        # TODO: Implement actual training loop
        console.print("[yellow]Training loop to be implemented when dataset is available[/yellow]")
        console.print(f"\n[cyan]Would train on: {device}[/cyan]")

        # Save placeholder results
        self._save_results({
            "status": "not_implemented",
            "message": "Training not implemented yet - waiting for dataset"
        })

    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        exp_id = self.config["experiment"]["id"]
        results_path = PROJECT_ROOT / "outputs" / "experiments" / exp_id / "results.json"

        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"[green]Results saved to: {results_path}[/green]")

    def run(self):
        """Run full experiment."""
        self.print_info()
        self.setup()

        # Check prerequisites
        data_path = PROJECT_ROOT / self.config["data"]["source"]
        if not data_path.exists():
            console.print(f"\n[red]Data not found: {data_path}[/red]")
            console.print("\n[yellow]Next steps:[/yellow]")
            console.print("  1. Download Noort et al. (2021) dataset")
            console.print("  2. Place in data/raw/")
            console.print("  3. Run: python -m src.data.preprocessing")
            return

        self.train()


def main():
    """Main entry point."""
    # Get experiment directory from script location
    exp_dir = Path(__file__).parent

    runner = ExperimentRunner(exp_dir)
    runner.run()


if __name__ == "__main__":
    main()
