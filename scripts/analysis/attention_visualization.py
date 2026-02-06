"""
Attention Visualization for BERT+LSTM Model.

Generates attention heatmaps showing which utterances the model
focuses on when making predictions. Visualizes 10 interesting cases.

Usage:
    python scripts/analysis/attention_visualization.py
    python scripts/analysis/attention_visualization.py --model-path models/002/best_model.pt
    python scripts/analysis/attention_visualization.py --n-cases 10
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bert_lstm import BertLSTMClassifier

LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]
LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}


def load_model_and_tokenizer(
    model_path: Path,
    encoder: str = "bert-base-uncased",
    max_utterances: int = 10,
    max_length: int = 128,
    device: str = "cpu",
):
    """Load trained BERT+LSTM model."""
    model = BertLSTMClassifier(
        model_name=encoder,
        num_labels=4,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
        max_utterances=max_utterances,
        max_length=max_length,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(encoder)
    return model, tokenizer


@torch.no_grad()
def get_prediction_with_attention(
    model, tokenizer, utterances: List[str],
    max_utterances: int = 10, max_length: int = 128, device: str = "cpu",
) -> Dict:
    """Get prediction, probabilities, and attention weights for one sequence."""
    # Truncate/pad
    utterances = utterances[-max_utterances:]
    n = len(utterances)

    encoded = tokenizer(
        utterances, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    utterance_mask = torch.ones(max_utterances)

    if n < max_utterances:
        pad = max_utterances - n
        input_ids = torch.cat([input_ids, torch.zeros(pad, max_length, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad, max_length, dtype=torch.long)])
        utterance_mask[n:] = 0

    # Add batch dim
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    utterance_mask = utterance_mask.unsqueeze(0).to(device)

    output = model(input_ids, attention_mask, utterance_mask)
    logits = output["logits"][0]
    attn = output["attention_weights"][0][:n].cpu().numpy()

    probs = torch.softmax(logits, dim=0).cpu().numpy()
    pred = int(logits.argmax())

    return {
        "prediction": LABEL_NAMES[pred],
        "pred_idx": pred,
        "probabilities": {LABEL_NAMES[i]: float(probs[i]) for i in range(4)},
        "attention_weights": attn,
        "n_utterances": n,
    }


def plot_attention_heatmap(
    utterances: List[str],
    attention_weights: np.ndarray,
    true_label: str,
    pred_label: str,
    case_id: str,
    output_path: Path,
    max_display_chars: int = 60,
):
    """Plot attention heatmap for a single case."""
    n = len(utterances)
    attn = attention_weights[:n]

    # Truncate long utterances for display
    labels = []
    for i, u in enumerate(utterances):
        display = u[:max_display_chars] + "..." if len(u) > max_display_chars else u
        labels.append(f"[{i+1}] {display}")

    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.4)))

    # Horizontal bar chart of attention weights
    colors = plt.cm.YlOrRd(attn / attn.max() if attn.max() > 0 else attn)
    bars = ax.barh(range(n), attn, color=colors)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Attention Weight")
    ax.set_title(
        f"Case: {case_id}\n"
        f"True: {true_label} | Predicted: {pred_label}",
        fontsize=11,
    )

    # Highlight highest attention
    max_idx = attn.argmax()
    bars[max_idx].set_edgecolor("red")
    bars[max_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_grid(cases: List[Dict], output_path: Path):
    """Plot a grid of attention visualizations."""
    n_cases = len(cases)
    fig, axes = plt.subplots(2, min(5, (n_cases + 1) // 2), figsize=(20, 10))
    axes = axes.flatten() if n_cases > 1 else [axes]

    for i, case in enumerate(cases[:len(axes)]):
        ax = axes[i]
        attn = case["attention_weights"]
        n = case["n_utterances"]

        ax.barh(range(n), attn[:n], color=plt.cm.YlOrRd(attn[:n] / max(attn[:n].max(), 1e-6)))
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"U{j+1}" for j in range(n)], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(
            f"{case['case_id']}\nT:{case['true_label']} P:{case['prediction']}",
            fontsize=8,
        )

    # Hide unused axes
    for i in range(len(cases), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Attention Weight Analysis (Top Cases)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Path to model checkpoint")
    parser.add_argument("--encoder", default="bert-base-uncased")
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    print("\n=== Attention Visualization ===\n")

    # Find model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Try Exp 002 checkpoint
        for candidate in [
            PROJECT_ROOT / "models" / "002" / "best_model.pt",
            PROJECT_ROOT / "models" / "011" / "best_model.pt",
        ]:
            if candidate.exists():
                model_path = candidate
                break
        else:
            print("No model checkpoint found. Provide --model-path")
            return

    print(f"Model: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(model_path, args.encoder, device=device)

    # Load data
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = PROJECT_ROOT / "data" / "cvr_labeled.csv"
        if not data_path.exists():
            data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    df = pd.read_csv(data_path)
    text_col = "cvr_message"
    label_col = "label"
    case_col = "case_id"

    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

    # Select interesting cases (diverse labels, different lengths)
    output_dir = PROJECT_ROOT / "outputs" / "analysis" / "attention_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_analyzed = []
    for case_id in df[case_col].unique()[:args.n_cases * 3]:
        group = df[df[case_col] == case_id].sort_values(
            "turn_number" if "turn_number" in df.columns else df.index
        )
        utterances = group[text_col].tolist()

        if len(utterances) < 5:
            continue

        # Take last window (closest to crash)
        window = utterances[-10:]
        true_labels = group[label_col].tolist()
        true_label = true_labels[-1] if isinstance(true_labels[-1], str) else LABEL_NAMES[true_labels[-1]]

        result = get_prediction_with_attention(model, tokenizer, window, device=device)
        result["case_id"] = str(case_id)
        result["true_label"] = true_label
        result["utterances"] = window

        cases_analyzed.append(result)

        # Individual heatmap
        plot_attention_heatmap(
            window, result["attention_weights"],
            true_label, result["prediction"],
            str(case_id),
            output_dir / f"attention_{case_id}.png",
        )

        if len(cases_analyzed) >= args.n_cases:
            break

    # Grid plot
    if cases_analyzed:
        plot_attention_grid(cases_analyzed, output_dir / "attention_grid.png")

    # Save metadata
    meta = []
    for c in cases_analyzed:
        meta.append({
            "case_id": c["case_id"],
            "true_label": c["true_label"],
            "prediction": c["prediction"],
            "probabilities": c["probabilities"],
            "max_attention_utterance": int(c["attention_weights"].argmax()),
            "correct": c["true_label"] == c["prediction"],
        })

    with open(output_dir / "attention_analysis.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(cases_analyzed)} attention maps to {output_dir}")
    print(f"Grid plot: {output_dir / 'attention_grid.png'}")


if __name__ == "__main__":
    main()
