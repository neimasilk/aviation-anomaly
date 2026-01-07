"""
Experiment {XXX}: {TITLE}

Run with: python experiments/{XXX}/run.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from rich.console import Console

from src.core.data.preprocessing import CVRPreprocessor
from src.core.models.bert_lstm import create_model, create_tokenizer
from src.core.utils.config import config

console = Console()


def load_experiment_config(exp_id: str) -> dict:
    """Load experiment config."""
    config_path = project_root / "experiments" / exp_id / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(exp_id: str):
    """Run experiment."""
    console.print(f"[bold blue]Running Experiment {exp_id}[/bold blue]")

    # Load config
    cfg = load_experiment_config(exp_id)

    # Print experiment info
    console.print(f"\n[bold]Experiment:[/bold] {cfg['experiment']['title']}")
    console.print(f"[bold]Description:[/bold] {cfg['experiment']['description']}")
    console.print(f"[bold]Tags:[/bold] {', '.join(cfg['experiment']['tags'])}")

    # TODO: Implement actual training loop
    console.print("\n[yellow]Experiment template - implement actual logic[/yellow]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="XXX", help="Experiment ID")
    args = parser.parse_args()

    run_experiment(args.exp)
