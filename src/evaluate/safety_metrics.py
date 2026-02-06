"""
Safety-Aware Evaluation Metrics for Aviation Anomaly Detection.

Novel metrics designed for safety-critical applications where:
- Missing a CRITICAL event is far worse than a false alarm
- Earlier detection is more valuable than later detection
- Different severity levels deserve different evaluation weights

Metrics:
1. Early Detection Score (EDS) - rewards earlier correct predictions
2. Safety-Weighted F1 - class-weighted F1 emphasizing CRITICAL
3. Detection Latency - how many windows before first non-NORMAL prediction
4. Safety Cost - asymmetric cost matrix for misclassifications
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# Label ordering: 0=NORMAL, 1=EARLY_WARNING, 2=ELEVATED, 3=CRITICAL
LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]

# Default safety weights (higher = more important to detect correctly)
DEFAULT_SAFETY_WEIGHTS = {
    "NORMAL": 1.0,
    "EARLY_WARNING": 2.0,
    "ELEVATED": 3.0,
    "CRITICAL": 4.0,
}


def early_detection_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_before_crash: Optional[np.ndarray] = None,
    label_names: List[str] = LABEL_NAMES,
) -> float:
    """
    Early Detection Score (EDS) - rewards correct predictions made earlier.

    If time_before_crash is provided, uses actual time values.
    Otherwise, uses severity level as a proxy (CRITICAL=1, ELEVATED=2, etc.)
    to weight correct predictions — correct non-NORMAL predictions at higher
    severity levels are worth more.

    EDS = sum(correct_prediction * weight) / sum(weight)

    where weight = severity_level for non-NORMAL, 1 for NORMAL.

    Args:
        y_true: Ground truth labels (integers 0-3)
        y_pred: Predicted labels (integers 0-3)
        time_before_crash: Optional time values for weighting
        label_names: Label name mapping

    Returns:
        EDS score in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if time_before_crash is not None:
        time_before_crash = np.asarray(time_before_crash)
        # Normalize time to [0, 1], higher time = earlier detection = better
        max_time = time_before_crash.max() if time_before_crash.max() > 0 else 1.0
        weights = time_before_crash / max_time
    else:
        # Use severity as proxy weight: NORMAL=1, EW=2, ELEVATED=3, CRITICAL=4
        weights = np.array([y + 1 for y in y_true], dtype=np.float64)

    correct = (y_true == y_pred).astype(np.float64)
    weighted_correct = correct * weights

    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    return float(weighted_correct.sum() / total_weight)


def safety_weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    safety_weights: Optional[Dict[str, float]] = None,
    label_names: List[str] = LABEL_NAMES,
) -> float:
    """
    Safety-Weighted F1 Score.

    Computes per-class F1 and weights them by safety importance.
    CRITICAL gets 4x weight, ELEVATED 3x, EARLY_WARNING 2x, NORMAL 1x.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        safety_weights: Dict mapping label names to weights
        label_names: Label name mapping

    Returns:
        Safety-weighted F1 score
    """
    if safety_weights is None:
        safety_weights = DEFAULT_SAFETY_WEIGHTS

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_class_f1 = f1_score(
        y_true, y_pred,
        labels=range(len(label_names)),
        average=None,
        zero_division=0,
    )

    weighted_sum = 0.0
    total_weight = 0.0
    for i, label in enumerate(label_names):
        w = safety_weights.get(label, 1.0)
        weighted_sum += per_class_f1[i] * w
        total_weight += w

    return float(weighted_sum / total_weight) if total_weight > 0 else 0.0


def detection_latency(
    sequence_predictions: List[List[int]],
    sequence_labels: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """
    Detection Latency - measures how many windows before the model
    first predicts a non-NORMAL class for each case/sequence.

    Lower latency = earlier detection = better for safety.

    Args:
        sequence_predictions: List of prediction sequences per case.
            Each inner list is chronologically ordered predictions for one case.
        sequence_labels: Optional ground truth sequences (same structure).

    Returns:
        Dict with mean, median, std of detection latency (in windows from end)
    """
    latencies = []

    for i, preds in enumerate(sequence_predictions):
        preds = np.asarray(preds)
        # Find first non-NORMAL prediction (label > 0)
        non_normal = np.where(preds > 0)[0]

        if len(non_normal) > 0:
            first_detection = non_normal[0]
            # Latency = distance from end (how early was detection)
            latency_from_end = len(preds) - first_detection
            latencies.append(latency_from_end)
        else:
            # Never detected — latency = 0 (worst case)
            latencies.append(0)

    latencies = np.array(latencies, dtype=np.float64)

    return {
        "mean_latency_windows": float(latencies.mean()) if len(latencies) > 0 else 0.0,
        "median_latency_windows": float(np.median(latencies)) if len(latencies) > 0 else 0.0,
        "std_latency_windows": float(latencies.std()) if len(latencies) > 0 else 0.0,
        "min_latency": float(latencies.min()) if len(latencies) > 0 else 0.0,
        "max_latency": float(latencies.max()) if len(latencies) > 0 else 0.0,
        "zero_detection_rate": float((latencies == 0).mean()) if len(latencies) > 0 else 1.0,
    }


def safety_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Safety Cost - asymmetric cost function for misclassifications.

    Default cost matrix penalizes:
    - Missing CRITICAL (predicting NORMAL when true=CRITICAL) = cost 20
    - Missing ELEVATED (predicting NORMAL when true=ELEVATED) = cost 10
    - False alarm (predicting CRITICAL when true=NORMAL) = cost 2

    Lower cost = better.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        cost_matrix: 4x4 cost matrix [true_label, pred_label]

    Returns:
        Mean safety cost per prediction
    """
    if cost_matrix is None:
        # Rows = true label, Cols = predicted label
        # [NORMAL, EARLY_WARNING, ELEVATED, CRITICAL]
        cost_matrix = np.array([
            #  N   EW   EL   CR   (predicted)
            [0.0, 1.0, 2.0, 2.0],   # true = NORMAL
            [3.0, 0.0, 1.0, 1.0],   # true = EARLY_WARNING
            [10., 5.0, 0.0, 1.0],   # true = ELEVATED
            [20., 15., 5.0, 0.0],   # true = CRITICAL
        ])

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    costs = np.array([cost_matrix[t, p] for t, p in zip(y_true, y_pred)])
    return float(costs.mean())


def compute_all_safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_before_crash: Optional[np.ndarray] = None,
    sequence_predictions: Optional[List[List[int]]] = None,
    label_names: List[str] = LABEL_NAMES,
) -> Dict[str, any]:
    """
    Compute all safety metrics in one call.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        time_before_crash: Optional time values
        sequence_predictions: Optional per-case prediction sequences
        label_names: Label names

    Returns:
        Dict with all metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1_vals = f1_score(
        y_true, y_pred, labels=range(len(label_names)),
        average=None, zero_division=0
    )
    per_class_recall_vals = recall_score(
        y_true, y_pred, labels=range(len(label_names)),
        average=None, zero_division=0
    )
    per_class_precision_vals = precision_score(
        y_true, y_pred, labels=range(len(label_names)),
        average=None, zero_division=0
    )

    per_class_f1 = {label_names[i]: float(v) for i, v in enumerate(per_class_f1_vals)}
    per_class_recall = {label_names[i]: float(v) for i, v in enumerate(per_class_recall_vals)}
    per_class_precision = {label_names[i]: float(v) for i, v in enumerate(per_class_precision_vals)}

    # Safety metrics
    eds = early_detection_score(y_true, y_pred, time_before_crash, label_names)
    sw_f1 = safety_weighted_f1(y_true, y_pred, label_names=label_names)
    s_cost = safety_cost(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))

    result = {
        # Standard
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "critical_recall": float(per_class_recall.get("CRITICAL", 0)),
        "confusion_matrix": cm.tolist(),
        # Safety-specific
        "early_detection_score": eds,
        "safety_weighted_f1": sw_f1,
        "safety_cost": s_cost,
    }

    # Detection latency (if per-case sequences provided)
    if sequence_predictions is not None:
        latency = detection_latency(sequence_predictions)
        result["detection_latency"] = latency

    return result


def format_metrics_table(metrics: Dict, label_names: List[str] = LABEL_NAMES) -> str:
    """Format metrics as a readable table string for logging."""
    lines = []
    lines.append("=" * 60)
    lines.append("SAFETY-AWARE EVALUATION RESULTS")
    lines.append("=" * 60)

    # Standard metrics
    lines.append(f"\nAccuracy:              {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1:              {metrics['macro_f1']:.4f}")

    # Safety metrics
    lines.append(f"\nEarly Detection Score: {metrics['early_detection_score']:.4f}")
    lines.append(f"Safety-Weighted F1:    {metrics['safety_weighted_f1']:.4f}")
    lines.append(f"Safety Cost:           {metrics['safety_cost']:.4f}")

    # Per-class
    lines.append(f"\n{'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    lines.append("-" * 48)
    for label in label_names:
        p = metrics["per_class_precision"].get(label, 0)
        r = metrics["per_class_recall"].get(label, 0)
        f = metrics["per_class_f1"].get(label, 0)
        lines.append(f"{label:<18} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    # Detection latency
    if "detection_latency" in metrics:
        dl = metrics["detection_latency"]
        lines.append(f"\nDetection Latency:")
        lines.append(f"  Mean:   {dl['mean_latency_windows']:.1f} windows")
        lines.append(f"  Median: {dl['median_latency_windows']:.1f} windows")
        lines.append(f"  Zero-detection rate: {dl['zero_detection_rate']:.2%}")

    lines.append("=" * 60)
    return "\n".join(lines)
