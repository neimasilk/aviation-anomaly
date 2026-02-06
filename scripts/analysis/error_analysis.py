"""
Error Analysis for Aviation Anomaly Detection Models.

Groups misclassifications by:
- Accident type / case characteristics
- Transcript length (short/medium/long)
- Temporal position within case
- Confusion matrix per subgroup
- Per-decade analysis

Usage:
    python scripts/analysis/error_analysis.py
    python scripts/analysis/error_analysis.py --exp-id 002
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]
LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}


def load_predictions_for_exp(exp_id: str):
    """Load y_true and y_pred for an experiment."""
    base = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    y_true_path = base / "y_true.npy"
    y_pred_path = base / "y_pred.npy"

    if y_true_path.exists() and y_pred_path.exists():
        return np.load(y_true_path), np.load(y_pred_path)

    # Try confusion matrix
    cm_path = base / "confusion_matrix.npy"
    if cm_path.exists():
        print(f"Found confusion matrix but not raw predictions for {exp_id}")

    return None, None


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    title: str, output_path: Path, label_names: List[str] = LABEL_NAMES,
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{title} (Counts)")

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd",
                xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"{title} (Normalized)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_by_transcript_length(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_ids: np.ndarray,
) -> Dict:
    """Analyze errors grouped by transcript length."""
    # Compute case lengths
    case_lengths = df.groupby("case_id").size().to_dict()

    # Categorize
    short_threshold = 50
    long_threshold = 150

    results = {}
    for category, (lo, hi) in [
        ("short (<50)", (0, short_threshold)),
        ("medium (50-150)", (short_threshold, long_threshold)),
        ("long (>150)", (long_threshold, float("inf"))),
    ]:
        mask = np.array([
            lo <= case_lengths.get(cid, 0) < hi for cid in case_ids
        ])

        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        acc = (y_t == y_p).mean()
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)

        results[category] = {
            "n_samples": int(mask.sum()),
            "accuracy": float(acc),
            "macro_f1": float(f1),
        }

    return results


def analyze_by_temporal_position(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Analyze error rates per true label (temporal position proxy)."""
    results = {}
    for label_idx, label_name in enumerate(LABEL_NAMES):
        mask = y_true == label_idx
        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        acc = (y_t == y_p).mean()

        # What does it get misclassified as?
        misclassified_as = {}
        wrong_mask = y_t != y_p
        if wrong_mask.sum() > 0:
            wrong_preds = y_p[wrong_mask]
            for pred_idx in range(4):
                count = (wrong_preds == pred_idx).sum()
                if count > 0:
                    misclassified_as[LABEL_NAMES[pred_idx]] = int(count)

        results[label_name] = {
            "n_samples": int(mask.sum()),
            "accuracy": float(acc),
            "misclassified_as": misclassified_as,
        }

    return results


def identify_interesting_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_ids: np.ndarray,
) -> Dict[str, List]:
    """Identify interesting cases for qualitative analysis."""
    # Correct CRITICAL predictions
    correct_critical = np.where((y_true == 3) & (y_pred == 3))[0]
    # Missed CRITICAL (false negative)
    missed_critical = np.where((y_true == 3) & (y_pred != 3))[0]
    # False CRITICAL alarms
    false_critical = np.where((y_true != 3) & (y_pred == 3))[0]
    # Borderline: true=ELEVATED, pred could be either way
    borderline = np.where((y_true == 2) & ((y_pred == 1) | (y_pred == 3)))[0]

    return {
        "correct_critical": {
            "indices": correct_critical[:5].tolist(),
            "case_ids": [str(case_ids[i]) for i in correct_critical[:5]],
            "count": int(len(correct_critical)),
        },
        "missed_critical": {
            "indices": missed_critical[:5].tolist(),
            "case_ids": [str(case_ids[i]) for i in missed_critical[:5]],
            "count": int(len(missed_critical)),
            "predicted_as": [LABEL_NAMES[y_pred[i]] for i in missed_critical[:5]],
        },
        "false_critical": {
            "indices": false_critical[:5].tolist(),
            "case_ids": [str(case_ids[i]) for i in false_critical[:5]],
            "count": int(len(false_critical)),
            "true_labels": [LABEL_NAMES[y_true[i]] for i in false_critical[:5]],
        },
        "borderline_elevated": {
            "indices": borderline[:5].tolist(),
            "case_ids": [str(case_ids[i]) for i in borderline[:5]],
            "count": int(len(borderline)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="002", help="Experiment ID to analyze")
    args = parser.parse_args()

    print(f"\n=== Error Analysis for Experiment {args.exp_id} ===\n")

    output_dir = PROJECT_ROOT / "outputs" / "analysis" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    y_true, y_pred = load_predictions_for_exp(args.exp_id)

    if y_true is None:
        print(f"No predictions found for exp {args.exp_id}.")
        print("Save y_true.npy and y_pred.npy first, or re-run the experiment.")
        return

    print(f"Loaded {len(y_true)} predictions")

    # 1. Overall confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        f"Experiment {args.exp_id}",
        output_dir / f"confusion_matrix_{args.exp_id}.png",
    )
    print(f"Confusion matrix saved")

    # 2. Classification report
    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, digits=4, zero_division=0
    )
    print(f"\n{report}")

    # 3. Error analysis by temporal position
    print("\n--- Errors by True Label ---")
    temporal_results = analyze_by_temporal_position(y_true, y_pred)
    for label, info in temporal_results.items():
        print(f"  {label}: acc={info['accuracy']:.2%} (n={info['n_samples']})")
        if info["misclassified_as"]:
            for pred_label, count in info["misclassified_as"].items():
                print(f"    -> {pred_label}: {count}")

    # 4. Try transcript length analysis
    data_path = PROJECT_ROOT / "data" / "cvr_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    length_results = {}
    if data_path.exists():
        df = pd.read_csv(data_path)
        # This only works if we have case_ids for each prediction
        # For now, note it as TODO
        print("\n--- Transcript Length Analysis ---")
        case_lengths = df.groupby("case_id").size()
        print(f"  Short (<50 utt): {(case_lengths < 50).sum()} cases")
        print(f"  Medium (50-150): {((case_lengths >= 50) & (case_lengths < 150)).sum()} cases")
        print(f"  Long (>150):     {(case_lengths >= 150).sum()} cases")

    # 5. Save results
    all_results = {
        "experiment_id": args.exp_id,
        "n_samples": len(y_true),
        "overall_accuracy": float((y_true == y_pred).mean()),
        "overall_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "temporal_analysis": temporal_results,
        "classification_report": report,
    }

    with open(output_dir / f"error_analysis_{args.exp_id}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
