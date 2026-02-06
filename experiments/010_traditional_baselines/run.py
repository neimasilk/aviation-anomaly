"""
Experiment 010: Traditional ML Baselines

Comprehensive benchmark of non-deep-learning methods:
1. Random baseline
2. Majority class baseline
3. Keyword heuristic (rule-based)
4. TF-IDF + Logistic Regression
5. TF-IDF + SVM (RBF)
6. TF-IDF + XGBoost
7. TF-IDF + Random Forest
8. Linguistic Features + XGBoost
9. Sentence-BERT embeddings + XGBoost

All use scikit-learn (no GPU required except Sentence-BERT encoding).

Usage:
    cd experiments/010_traditional_baselines
    python run.py
"""
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.safety_metrics import compute_all_safety_metrics, format_metrics_table

console = Console()

LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]

# Aviation urgency keywords for heuristic baseline
URGENCY_KEYWORDS = {
    "critical": ["mayday", "emergency", "crash", "fire", "terrain", "pull up",
                  "stall", "windshear", "wind shear", "evacuate", "ditching"],
    "elevated": ["warning", "caution", "abort", "go around", "missed approach",
                 "failure", "malfunction", "problem", "trouble", "urgent",
                 "help", "unable", "lost", "confused", "don't know"],
    "early_warning": ["check", "verify", "unusual", "strange", "different",
                      "notice", "watch", "careful", "attention", "review",
                      "concern", "worry", "hmm", "uh", "wait"],
}


def create_sequences_from_df(
    df: pd.DataFrame,
    window_size: int = 10,
    stride: int = 5,
    text_col: str = "cvr_message",
    label_col: str = "label",
    case_col: str = "case_id",
) -> Tuple[List[str], List[int]]:
    """Create concatenated window texts and labels for traditional ML."""
    texts = []
    labels = []

    for case_id, group in df.groupby(case_col):
        group = group.sort_values(
            "turn_number" if "turn_number" in group.columns else group.index
        ).reset_index(drop=True)

        utterances = group[text_col].fillna("").tolist()
        case_labels = group[label_col].tolist()

        for i in range(0, len(utterances) - window_size + 1, stride):
            window_texts = utterances[i : i + window_size]
            # Concatenate for bag-of-words models
            combined = " [SEP] ".join(str(t) for t in window_texts)
            texts.append(combined)

            # Label from last utterance
            seq_label = case_labels[i + window_size - 1]
            if isinstance(seq_label, str):
                seq_label = LABEL_MAP.get(seq_label, 0)
            labels.append(seq_label)

        # Handle remaining
        if len(utterances) >= window_size:
            last_start = len(utterances) - window_size
            if (len(utterances) - window_size) % stride != 0:
                window_texts = utterances[-window_size:]
                combined = " [SEP] ".join(str(t) for t in window_texts)
                texts.append(combined)
                seq_label = case_labels[-1]
                if isinstance(seq_label, str):
                    seq_label = LABEL_MAP.get(seq_label, 0)
                labels.append(seq_label)

    return texts, labels


def extract_linguistic_features(text: str) -> Dict[str, float]:
    """Extract linguistic features from concatenated window text."""
    utterances = text.split(" [SEP] ")
    n_utterances = len(utterances)

    word_counts = [len(u.split()) for u in utterances]
    char_counts = [len(u) for u in utterances]

    features = {
        "n_utterances": n_utterances,
        "total_words": sum(word_counts),
        "mean_word_count": np.mean(word_counts) if word_counts else 0,
        "std_word_count": np.std(word_counts) if len(word_counts) > 1 else 0,
        "min_word_count": min(word_counts) if word_counts else 0,
        "max_word_count": max(word_counts) if word_counts else 0,
        "total_chars": sum(char_counts),
        "mean_char_count": np.mean(char_counts) if char_counts else 0,
        "question_marks": sum(u.count("?") for u in utterances),
        "exclamations": sum(u.count("!") for u in utterances),
        "ellipsis": sum(u.count("...") for u in utterances),
        "repetition_ratio": _repetition_ratio(utterances),
    }

    # Urgency keyword counts
    text_lower = text.lower()
    features["critical_keywords"] = sum(
        text_lower.count(kw) for kw in URGENCY_KEYWORDS["critical"]
    )
    features["elevated_keywords"] = sum(
        text_lower.count(kw) for kw in URGENCY_KEYWORDS["elevated"]
    )
    features["early_warning_keywords"] = sum(
        text_lower.count(kw) for kw in URGENCY_KEYWORDS["early_warning"]
    )
    features["total_urgency_keywords"] = (
        features["critical_keywords"]
        + features["elevated_keywords"]
        + features["early_warning_keywords"]
    )

    # Word length variance (proxy for speech disruption)
    all_words = text.split()
    if len(all_words) > 1:
        word_lengths = [len(w) for w in all_words]
        features["word_length_variance"] = np.var(word_lengths)
    else:
        features["word_length_variance"] = 0

    # Utterance length trend (increasing/decreasing — proxy for escalation)
    if len(word_counts) >= 3:
        x = np.arange(len(word_counts))
        slope = np.polyfit(x, word_counts, 1)[0]
        features["length_trend_slope"] = slope
    else:
        features["length_trend_slope"] = 0

    return features


def _repetition_ratio(utterances: List[str]) -> float:
    """Fraction of repeated words across utterances."""
    all_words = []
    for u in utterances:
        all_words.extend(u.lower().split())
    if not all_words:
        return 0
    unique = set(all_words)
    return 1.0 - len(unique) / len(all_words)


def keyword_heuristic_predict(texts: List[str]) -> np.ndarray:
    """Rule-based classification using keyword matching."""
    predictions = []
    for text in texts:
        text_lower = text.lower()

        # Check critical keywords
        critical_count = sum(
            text_lower.count(kw) for kw in URGENCY_KEYWORDS["critical"]
        )
        elevated_count = sum(
            text_lower.count(kw) for kw in URGENCY_KEYWORDS["elevated"]
        )
        warning_count = sum(
            text_lower.count(kw) for kw in URGENCY_KEYWORDS["early_warning"]
        )

        if critical_count >= 2:
            predictions.append(3)  # CRITICAL
        elif critical_count >= 1 or elevated_count >= 3:
            predictions.append(2)  # ELEVATED
        elif elevated_count >= 1 or warning_count >= 3:
            predictions.append(1)  # EARLY_WARNING
        else:
            predictions.append(0)  # NORMAL

    return np.array(predictions)


def run_model(
    name: str,
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    texts_train: List[str] = None,
    texts_test: List[str] = None,
    config: Dict = None,
) -> Dict:
    """Train and evaluate a single model."""
    console.print(f"\n[bold blue]Training: {name}[/bold blue]")
    start = time.time()

    if model_type == "random":
        model = DummyClassifier(strategy="uniform", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == "majority":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == "heuristic":
        # Keyword-based, no training needed
        y_pred = keyword_heuristic_predict(texts_test)

    elif model_type == "tfidf_lr":
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", C=1.0, random_state=42
            )),
        ])
        pipe.fit(texts_train, y_train)
        y_pred = pipe.predict(texts_test)

    elif model_type == "tfidf_svm":
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", SVC(kernel="rbf", class_weight="balanced", C=1.0, random_state=42)),
        ])
        pipe.fit(texts_train, y_train)
        y_pred = pipe.predict(texts_test)

    elif model_type == "tfidf_xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            console.print("[yellow]XGBoost not installed. Skipping.[/yellow]")
            return None

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42,
            )),
        ])
        pipe.fit(texts_train, y_train)
        y_pred = pipe.predict(texts_test)

    elif model_type == "tfidf_rf":
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42,
            )),
        ])
        pipe.fit(texts_train, y_train)
        y_pred = pipe.predict(texts_test)

    elif model_type == "ling_xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            console.print("[yellow]XGBoost not installed. Skipping.[/yellow]")
            return None

        # Extract linguistic features
        console.print("  Extracting linguistic features...")
        X_train_feat = pd.DataFrame([extract_linguistic_features(t) for t in texts_train])
        X_test_feat = pd.DataFrame([extract_linguistic_features(t) for t in texts_test])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42,
            )),
        ])
        pipe.fit(X_train_feat, y_train)
        y_pred = pipe.predict(X_test_feat)

    elif model_type == "sbert_xgb":
        try:
            from sentence_transformers import SentenceTransformer
            from xgboost import XGBClassifier
        except ImportError:
            console.print("[yellow]sentence-transformers or xgboost not installed. Skipping.[/yellow]")
            return None

        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        console.print("  Encoding train set with Sentence-BERT...")
        X_train_emb = sbert_model.encode(texts_train, show_progress_bar=True, batch_size=64)
        console.print("  Encoding test set with Sentence-BERT...")
        X_test_emb = sbert_model.encode(texts_test, show_progress_bar=True, batch_size=64)

        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42,
        )
        clf.fit(X_train_emb, y_train)
        y_pred = clf.predict(X_test_emb)

    else:
        console.print(f"[red]Unknown model type: {model_type}[/red]")
        return None

    elapsed = time.time() - start

    # Compute metrics
    metrics = compute_all_safety_metrics(y_test, y_pred)
    metrics["training_time_seconds"] = elapsed

    console.print(f"  [green]Done in {elapsed:.1f}s | Acc: {metrics['accuracy']:.4f} | F1: {metrics['macro_f1']:.4f}[/green]")

    return {
        "name": name,
        "type": model_type,
        "metrics": metrics,
        "predictions": y_pred.tolist(),
        "training_time": elapsed,
    }


def main():
    console.print("\n[bold cyan]Experiment 010: Traditional ML Baselines[/bold cyan]")
    console.print("=" * 60)

    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Load data
    data_path = PROJECT_ROOT / config["data"]["source"]
    if not data_path.exists():
        # Try alternate path
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    console.print(f"[cyan]Loading data: {data_path}[/cyan]")
    df = pd.read_csv(data_path)

    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]
    case_col = config["data"]["case_id_column"]

    # Filter empty
    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

    # Map labels
    if df[label_col].dtype == object:
        df["label_id"] = df[label_col].map(LABEL_MAP)
    else:
        df["label_id"] = df[label_col]

    console.print(f"[green]Loaded {len(df):,} utterances from {df[case_col].nunique()} cases[/green]")

    # Create sequences
    window_size = config["data"]["window_size"]
    stride = config["data"]["stride"]

    console.print(f"[cyan]Creating sequences (window={window_size}, stride={stride})...[/cyan]")
    texts, labels = create_sequences_from_df(
        df, window_size=window_size, stride=stride,
        text_col=text_col, label_col="label_id", case_col=case_col,
    )
    labels = np.array(labels)
    console.print(f"[green]Created {len(texts):,} sequences[/green]")

    # Show distribution
    unique, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(unique, counts):
        console.print(f"  {LABEL_NAMES[label_id]}: {count:,} ({count/len(labels)*100:.1f}%)")

    # Split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels,
        test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"],
        stratify=labels,
    )

    console.print(f"\n[cyan]Train: {len(X_train_texts):,} | Test: {len(X_test_texts):,}[/cyan]")

    # Dummy features for models that need arrays (not text)
    X_train_dummy = np.zeros((len(X_train_texts), 1))
    X_test_dummy = np.zeros((len(X_test_texts), 1))

    # Run all models
    all_results = []

    model_configs = [
        ("Random Baseline", "random"),
        ("Majority Baseline", "majority"),
        ("Keyword Heuristic", "heuristic"),
        ("TF-IDF + Logistic Regression", "tfidf_lr"),
        ("TF-IDF + SVM (RBF)", "tfidf_svm"),
        ("TF-IDF + XGBoost", "tfidf_xgb"),
        ("TF-IDF + Random Forest", "tfidf_rf"),
        ("Linguistic Features + XGBoost", "ling_xgb"),
        ("Sentence-BERT + XGBoost", "sbert_xgb"),
    ]

    for name, model_type in model_configs:
        try:
            result = run_model(
                name=name,
                model_type=model_type,
                X_train=X_train_dummy,
                y_train=y_train,
                X_test=X_test_dummy,
                y_test=y_test,
                texts_train=X_train_texts,
                texts_test=X_test_texts,
                config=config,
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()

    # Summary table
    console.print("\n" + "=" * 80)
    console.print("[bold green]TRADITIONAL BASELINES SUMMARY[/bold green]")
    console.print("=" * 80)

    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan", width=35)
    table.add_column("Accuracy", style="yellow", justify="right")
    table.add_column("Macro F1", style="yellow", justify="right")
    table.add_column("Safety F1", style="green", justify="right")
    table.add_column("EDS", style="green", justify="right")
    table.add_column("CRITICAL Recall", style="red", justify="right")
    table.add_column("Time (s)", style="dim", justify="right")

    for r in sorted(all_results, key=lambda x: x["metrics"]["macro_f1"], reverse=True):
        m = r["metrics"]
        table.add_row(
            r["name"],
            f"{m['accuracy']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{m['safety_weighted_f1']:.4f}",
            f"{m['early_detection_score']:.4f}",
            f"{m['critical_recall']:.2%}",
            f"{r['training_time']:.1f}",
        )

    console.print(table)

    # Save results
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions for statistical testing
    for r in all_results:
        pred_path = output_dir / f"predictions_{r['type']}.npy"
        np.save(pred_path, np.array(r["predictions"]))

    # Save ground truth
    np.save(output_dir / "y_true.npy", y_test)

    # Save full results
    results_save = {
        "experiment_id": config["experiment"]["id"],
        "experiment_title": config["experiment"]["title"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {
            "n_train": len(X_train_texts),
            "n_test": len(X_test_texts),
            "window_size": window_size,
            "stride": stride,
        },
        "models": [
            {
                "name": r["name"],
                "type": r["type"],
                "metrics": {k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
                "training_time": r["training_time"],
            }
            for r in all_results
        ],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_save, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir / 'results.json'}[/green]")
    console.print(f"[green]Predictions saved for statistical testing[/green]")


if __name__ == "__main__":
    main()
