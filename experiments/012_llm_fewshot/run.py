"""
Experiment 012: Few-Shot LLM Classification

Tests whether a general-purpose LLM (DeepSeek/OpenAI) can classify
CVR communication windows into anomaly severity levels without training.

Uses 4 examples per class as few-shot demonstrations.

Research question: "Can an LLM perform temporal anomaly detection in
aviation communications without task-specific training?"

Usage:
    cd experiments/012_llm_fewshot
    python run.py
    python run.py --provider openai --model gpt-4o-mini
    python run.py --zero-shot
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate.safety_metrics import compute_all_safety_metrics, format_metrics_table

console = Console()

LABEL_MAP = {"NORMAL": 0, "EARLY_WARNING": 1, "ELEVATED": 2, "CRITICAL": 3}
LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]

SYSTEM_PROMPT = """You are an aviation safety expert analyzing Cockpit Voice Recorder (CVR) transcripts.

Your task: Given a sequence of cockpit utterances, classify the overall communication state into one of four categories:

- NORMAL: Routine, standard aviation communication. No stress indicators.
- EARLY_WARNING: Subtle changes — hesitation, slight confusion, non-standard phrasing, mild uncertainty.
- ELEVATED: Clear stress indicators — urgency, repeated requests, confusion, workload overload.
- CRITICAL: Emergency — explicit emergency calls, panic, loss of situational awareness, crew coordination breakdown.

Respond with ONLY the label (one of: NORMAL, EARLY_WARNING, ELEVATED, CRITICAL). No explanation."""

FEW_SHOT_EXAMPLES = {
    "NORMAL": [
        "United 472 heavy, cleared for ILS runway 28 left approach.\nRoger, cleared ILS 28 left, United 472 heavy.\nLanding gear down.\nGear down, three green.\nFlaps 30.\nFlaps 30, set.\nApproach checklist complete.\nTower, United 472, runway 28 left in sight.\nUnited 472, cleared to land 28 left, wind 270 at 8.\nCleared to land 28 left, United 472.",
        "Set heading 270.\nHeading 270, roger.\nDescending through flight level 250.\nCenter, Delta 319, level 250.\nDelta 319, descend and maintain flight level 180.\nDescending to 180, Delta 319.\nAltimeter 29.92.\nCheck.\nCrossing SMITH at 180.\nRoger, expect vectors for the ILS.",
    ],
    "EARLY_WARNING": [
        "Hmm, airspeed seems a bit low.\nWhat's our target speed?\nShould be 145.\nI'm showing 138... let me check.\nWind might be shifting on us.\nYeah, I noticed the heading drift.\nLet's keep an eye on it.\nAdding a few knots.\nGood idea.\nApproach, American 215, we're getting some airspeed fluctuations.",
        "That doesn't look right.\nWhat?\nThe altitude readout, it jumped.\nWhich one? I show normal.\nThe standby is different from primary.\nBy how much?\nAbout 200 feet.\nHmm, could be an instrument issue.\nLet me cross-check with GPS.\nOkay, keep monitoring it.",
    ],
    "ELEVATED": [
        "We're not stabilized, should we go around?\nKeep going, we can make it.\nAirspeed is decaying, 128 knots.\nAdd power, add power!\nI'm at full thrust already.\nGo around, go around now!\nGoing around, TOGA power.\nPositive rate.\nGear up.\nApproach, we're going around due to unstabilized approach.",
        "We've lost the right engine.\nConfirm, right engine failure.\nRight engine, confirm failure.\nRunning the checklist.\nSpeed is dropping.\nMaintain heading, I'll handle the engine.\nAre we losing altitude?\nSlightly, but controllable.\nDeclare an emergency?\nYes, declare emergency. Mayday, mayday.",
    ],
    "CRITICAL": [
        "Pull up! Pull up!\nTerrain! Terrain ahead!\nGPWS! GPWS warning!\nClimbing, max power!\nWe're too low! Trees!\nPull up! Pull up!\nI can't see anything!\nClimb! Climb!\nMaximum power!\nOh God, oh God!",
        "We've lost all hydraulics!\nWhat?! All three systems?\nNothing is responding!\nI can't control it!\nMayday mayday mayday!\nTrying alternate controls!\nWe're in a descent we can't stop!\nBrace brace brace!\nMayday, we've lost flight controls!\nOh no, we're going down!",
    ],
}


def create_sequences_from_df(
    df: pd.DataFrame,
    window_size: int = 10,
    stride: int = 5,
    text_col: str = "cvr_message",
    label_col: str = "label",
    case_col: str = "case_id",
) -> Tuple[List[str], List[int]]:
    """Create concatenated window texts and labels."""
    texts = []
    labels = []

    for case_id, group in df.groupby(case_col):
        group = group.sort_values(
            "turn_number" if "turn_number" in group.columns else group.index
        ).reset_index(drop=True)

        utterances = group[text_col].fillna("").tolist()
        case_labels = group[label_col].tolist()

        for i in range(0, len(utterances) - window_size + 1, stride):
            window = utterances[i : i + window_size]
            combined = "\n".join(str(t) for t in window)
            texts.append(combined)

            seq_label = case_labels[i + window_size - 1]
            if isinstance(seq_label, str):
                seq_label = LABEL_MAP.get(seq_label, 0)
            labels.append(seq_label)

    return texts, labels


def build_few_shot_prompt(
    window_text: str,
    n_examples: int = 4,
    zero_shot: bool = False,
) -> str:
    """Build few-shot or zero-shot classification prompt."""
    if zero_shot:
        return f"Classify the following cockpit communication sequence:\n\n{window_text}\n\nLabel:"

    # Build few-shot examples
    examples = []
    for label_name, example_list in FEW_SHOT_EXAMPLES.items():
        for ex in example_list[:n_examples // 2 + 1]:  # Use available examples
            examples.append(f"Sequence:\n{ex}\nLabel: {label_name}")

    examples_str = "\n\n---\n\n".join(examples)

    return f"""Here are examples of cockpit communication sequences with their severity labels:

{examples_str}

---

Now classify this sequence:

{window_text}

Label:"""


class LLMClassifier:
    """LLM-based classifier for CVR sequences."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = None,
        temperature: float = 0.1,
        rate_limit_delay: float = 0.3,
    ):
        from openai import OpenAI

        self.provider = provider
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay

        if provider == "deepseek":
            self.model = model or "deepseek-chat"
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not api_key:
            raise ValueError(f"Set {provider.upper()}_API_KEY in .env")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def classify(self, prompt: str) -> str:
        """Classify a single window."""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=20,
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip().upper()

                # Parse label
                for label in LABEL_NAMES:
                    if label in text:
                        return label

                # Fuzzy match
                if "CRITICAL" in text or "EMERGENCY" in text:
                    return "CRITICAL"
                elif "ELEVATED" in text or "STRESS" in text:
                    return "ELEVATED"
                elif "WARNING" in text or "EARLY" in text:
                    return "EARLY_WARNING"
                else:
                    return "NORMAL"

            except Exception as e:
                if attempt < 2:
                    time.sleep(self.rate_limit_delay * (attempt + 1) * 2)
                    continue
                console.print(f"[red]API error: {e}[/red]")
                return "NORMAL"

    def classify_batch(
        self,
        texts: List[str],
        n_examples: int = 4,
        zero_shot: bool = False,
    ) -> List[int]:
        """Classify a batch of windows."""
        predictions = []
        errors = 0

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Classifying ({self.provider}/{self.model})...",
                total=len(texts),
            )

            for i, text in enumerate(texts):
                prompt = build_few_shot_prompt(text, n_examples, zero_shot)
                label_str = self.classify(prompt)
                predictions.append(LABEL_MAP.get(label_str, 0))

                progress.advance(task)
                time.sleep(self.rate_limit_delay)

                if (i + 1) % 100 == 0:
                    console.print(f"  [dim]Progress: {i+1}/{len(texts)} | Errors: {errors}[/dim]")

        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples for cost control")
    parser.add_argument("--rate-limit", type=float, default=0.3)
    args = parser.parse_args()

    console.print("\n[bold cyan]Experiment 012: Few-Shot LLM Classification[/bold cyan]")
    console.print("=" * 60)

    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Load data
    data_path = PROJECT_ROOT / config["data"]["source"]
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    df = pd.read_csv(data_path)
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]

    df = df[df[text_col].notna() & (df[text_col].str.len() > 0)].copy()

    if df[label_col].dtype == object:
        df["label_id"] = df[label_col].map(LABEL_MAP)
    else:
        df["label_id"] = df[label_col]

    # Create sequences
    texts, labels = create_sequences_from_df(
        df, window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
        text_col=text_col, label_col="label_id",
        case_col=config["data"]["case_id_column"],
    )
    labels = np.array(labels)

    # Use only test split
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        texts, labels,
        test_size=config["data"]["test_split"],
        random_state=config["data"]["random_seed"],
        stratify=labels,
    )

    if args.max_samples and args.max_samples < len(X_test):
        # Stratified subsample
        indices = np.arange(len(X_test))
        sampled, _, sampled_y, _ = train_test_split(
            indices, y_test,
            train_size=args.max_samples,
            random_state=42,
            stratify=y_test,
        )
        X_test = [X_test[i] for i in sampled]
        y_test = sampled_y

    console.print(f"[green]Test set: {len(X_test)} sequences[/green]")
    console.print(f"[cyan]Provider: {args.provider} | Zero-shot: {args.zero_shot}[/cyan]")

    # Estimate cost
    avg_tokens = sum(len(t.split()) for t in X_test) / len(X_test)
    est_cost = len(X_test) * avg_tokens * 2 / 1_000_000 * 0.15  # rough estimate
    console.print(f"[yellow]Estimated cost: ~${est_cost:.2f}[/yellow]")

    # Classify
    classifier = LLMClassifier(
        provider=args.provider,
        model=args.model,
        rate_limit_delay=args.rate_limit,
    )

    mode = "zero_shot" if args.zero_shot else "few_shot"
    n_examples = config["llm"]["n_examples_per_class"] if not args.zero_shot else 0

    start = time.time()
    y_pred = classifier.classify_batch(X_test, n_examples=n_examples, zero_shot=args.zero_shot)
    elapsed = time.time() - start

    y_pred = np.array(y_pred)

    # Metrics
    metrics = compute_all_safety_metrics(y_test, y_pred)
    console.print(format_metrics_table(metrics))

    console.print(f"\n[cyan]Classification time: {elapsed:.1f}s ({elapsed/len(X_test):.2f}s/sample)[/cyan]")

    # Save
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"y_true_{mode}.npy", y_test)
    np.save(output_dir / f"y_pred_{mode}_{args.provider}.npy", y_pred)

    results = {
        "experiment": config["experiment"],
        "provider": args.provider,
        "model": classifier.model,
        "mode": mode,
        "n_test_samples": len(X_test),
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
        "elapsed_seconds": elapsed,
        "cost_estimate_usd": est_cost,
    }

    with open(output_dir / f"results_{mode}_{args.provider}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
