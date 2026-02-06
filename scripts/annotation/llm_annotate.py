"""
LLM-Assisted Content Annotation for CVR Transcripts.

Main contribution #1: Content-aware labeling using LLMs, compared with
position-based labeling. Uses two different LLMs for inter-annotator
agreement (Cohen's kappa).

Pipeline:
1. For each utterance + 5 context utterances → LLM rates severity 1-5
2. Boolean flags: urgency_keyword, speech_disruption, topic_shift
3. Map scores to labels: 1-2=NORMAL, 3=EARLY_WARNING, 4=ELEVATED, 5=CRITICAL
4. Run with 2 LLMs → compute Cohen's kappa
5. Export content-labeled dataset

Usage:
    python scripts/annotation/llm_annotate.py
    python scripts/annotation/llm_annotate.py --provider deepseek --batch-size 50
    python scripts/annotation/llm_annotate.py --provider openai --model gpt-4o-mini
    python scripts/annotation/llm_annotate.py --compute-agreement
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

# Score to label mapping
SCORE_TO_LABEL = {
    1: "NORMAL",
    2: "NORMAL",
    3: "EARLY_WARNING",
    4: "ELEVATED",
    5: "CRITICAL",
}

ANNOTATION_PROMPT = """You are an aviation safety expert analyzing Cockpit Voice Recorder (CVR) transcripts from accident investigations.

Rate the following utterance on a scale of 1-5 for communication anomaly severity:

1 = ROUTINE: Normal, standard aviation communication. No stress indicators.
2 = ROUTINE-ALERT: Normal but with slight deviation from standard phraseology. No concern.
3 = EARLY WARNING: Subtle changes emerging — hesitation, slight confusion, non-standard phrasing, mild uncertainty.
4 = ELEVATED: Clear stress indicators — urgency in language, repeated requests, confusion about situation, workload overload signs.
5 = CRITICAL: Emergency communication — explicit emergency calls, panic indicators, loss of situational awareness, breakdown in crew coordination.

CONTEXT (previous {context_size} utterances):
{context}

CURRENT UTTERANCE TO RATE:
Speaker: {speaker}
Text: "{utterance}"

Respond with ONLY a JSON object (no other text):
{{"score": <1-5>, "urgency_keyword": <true/false>, "speech_disruption": <true/false>, "topic_shift": <true/false>, "reasoning": "<brief 1-sentence reason>"}}"""


class LLMAnnotator:
    """Annotate CVR utterances using LLM API."""

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        rate_limit_delay: float = 0.2,
    ):
        """
        Args:
            provider: 'deepseek' or 'openai'
            model: Model name (auto-detected from provider if None)
            api_key: API key (from env if None)
            base_url: API base URL (auto from provider if None)
            temperature: Low temperature for consistent annotations
            max_retries: Retries on API failure
            rate_limit_delay: Seconds between API calls
        """
        from openai import OpenAI

        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        if provider == "deepseek":
            self.model = model or "deepseek-chat"
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        elif provider == "openai":
            self.model = model or "gpt-4o-mini"
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            base_url = base_url or None  # Use default
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'deepseek' or 'openai'.")

        if not api_key:
            raise ValueError(
                f"API key for {provider} not found. "
                f"Set {provider.upper()}_API_KEY in .env file."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        console.print(f"[green]Initialized {provider} annotator (model={self.model})[/green]")

    def annotate_utterance(
        self,
        utterance: str,
        context: List[str],
        speaker: str = "Unknown",
        context_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Annotate a single utterance.

        Args:
            utterance: The utterance to annotate
            context: Previous utterances for context
            speaker: Speaker role
            context_size: Number of context utterances to include

        Returns:
            Dict with score, flags, reasoning
        """
        # Build context string
        ctx_lines = context[-context_size:]
        ctx_str = "\n".join(f"  [{i+1}] {line}" for i, line in enumerate(ctx_lines))
        if not ctx_str:
            ctx_str = "  (No previous context — beginning of recording)"

        prompt = ANNOTATION_PROMPT.format(
            context_size=context_size,
            context=ctx_str,
            speaker=speaker,
            utterance=utterance,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()

                # Parse JSON response
                # Handle cases where LLM wraps in ```json ... ```
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                result = json.loads(text)

                # Validate
                score = int(result.get("score", 1))
                score = max(1, min(5, score))  # Clamp to 1-5

                return {
                    "score": score,
                    "label": SCORE_TO_LABEL[score],
                    "urgency_keyword": bool(result.get("urgency_keyword", False)),
                    "speech_disruption": bool(result.get("speech_disruption", False)),
                    "topic_shift": bool(result.get("topic_shift", False)),
                    "reasoning": str(result.get("reasoning", "")),
                    "raw_response": text,
                }

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay)
                    continue
                # Fallback: try to extract score from text
                return self._fallback_parse(text if 'text' in dir() else "", e)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.rate_limit_delay * (attempt + 1) * 2
                    console.print(f"[yellow]API error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...[/yellow]")
                    time.sleep(wait)
                    continue
                return {
                    "score": 1,
                    "label": "NORMAL",
                    "urgency_keyword": False,
                    "speech_disruption": False,
                    "topic_shift": False,
                    "reasoning": f"API error: {e}",
                    "raw_response": "",
                    "error": str(e),
                }

    def _fallback_parse(self, text: str, error: Exception) -> Dict[str, Any]:
        """Try to extract score from malformed response."""
        import re

        score = 1
        # Try to find a digit after "score"
        match = re.search(r'"?score"?\s*[:=]\s*(\d)', text)
        if match:
            score = max(1, min(5, int(match.group(1))))

        return {
            "score": score,
            "label": SCORE_TO_LABEL[score],
            "urgency_keyword": "urgency" in text.lower() and "true" in text.lower(),
            "speech_disruption": "disruption" in text.lower() and "true" in text.lower(),
            "topic_shift": "topic" in text.lower() and "shift" in text.lower() and "true" in text.lower(),
            "reasoning": f"Fallback parse (original error: {error})",
            "raw_response": text,
            "parse_error": str(error),
        }

    def annotate_dataset(
        self,
        df: pd.DataFrame,
        text_col: str = "cvr_message",
        speaker_col: str = "cvr_speaker_source",
        case_col: str = "case_id",
        context_size: int = 5,
        batch_size: int = 100,
        save_every: int = 500,
        output_path: Optional[Path] = None,
        resume_from: int = 0,
    ) -> pd.DataFrame:
        """
        Annotate entire dataset.

        Args:
            df: Input DataFrame
            text_col: Text column name
            speaker_col: Speaker column name
            case_col: Case ID column name
            context_size: Context window size
            batch_size: Logging interval
            save_every: Save checkpoint every N utterances
            output_path: Path for intermediate saves
            resume_from: Resume from this index

        Returns:
            DataFrame with annotation columns added
        """
        df = df.copy()
        n_total = len(df)

        # Initialize annotation columns
        for col in ["llm_score", "llm_label", "llm_urgency", "llm_disruption",
                     "llm_topic_shift", "llm_reasoning"]:
            if col not in df.columns:
                df[col] = None

        console.print(f"\n[bold]Annotating {n_total:,} utterances with {self.provider}/{self.model}[/bold]")
        console.print(f"Context size: {context_size} utterances")
        console.print(f"Starting from index: {resume_from}")

        errors = 0
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Annotating ({self.provider})...",
                total=n_total - resume_from,
            )

            for case_id, group in df.groupby(case_col):
                group = group.sort_index()
                context_buffer = []

                for idx, row in group.iterrows():
                    row_num = df.index.get_loc(idx)
                    if row_num < resume_from:
                        # Still need to build context
                        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                        if text:
                            context_buffer.append(text)
                        continue

                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    speaker = str(row[speaker_col]) if speaker_col in row and pd.notna(row.get(speaker_col)) else "Unknown"

                    if not text.strip():
                        progress.advance(task)
                        continue

                    # Annotate
                    result = self.annotate_utterance(
                        utterance=text,
                        context=context_buffer,
                        speaker=speaker,
                        context_size=context_size,
                    )

                    # Store results
                    df.at[idx, "llm_score"] = result["score"]
                    df.at[idx, "llm_label"] = result["label"]
                    df.at[idx, "llm_urgency"] = result.get("urgency_keyword", False)
                    df.at[idx, "llm_disruption"] = result.get("speech_disruption", False)
                    df.at[idx, "llm_topic_shift"] = result.get("topic_shift", False)
                    df.at[idx, "llm_reasoning"] = result.get("reasoning", "")

                    if "error" in result:
                        errors += 1

                    # Update context
                    context_buffer.append(text)

                    progress.advance(task)

                    # Rate limiting
                    time.sleep(self.rate_limit_delay)

                    # Periodic save
                    if output_path and (row_num + 1) % save_every == 0:
                        df.to_csv(output_path, index=False)
                        elapsed = time.time() - start_time
                        rate = (row_num - resume_from + 1) / elapsed
                        remaining = (n_total - row_num - 1) / rate if rate > 0 else 0
                        console.print(
                            f"  [dim]Checkpoint saved ({row_num+1}/{n_total}). "
                            f"Rate: {rate:.1f}/s. ETA: {remaining/60:.1f}min. "
                            f"Errors: {errors}[/dim]"
                        )

        elapsed = time.time() - start_time
        console.print(f"\n[green]Annotation complete: {n_total:,} utterances in {elapsed/60:.1f} minutes[/green]")
        console.print(f"[yellow]Errors: {errors}[/yellow]")

        return df


def compute_inter_annotator_agreement(
    df: pd.DataFrame,
    col1: str = "llm_score_deepseek",
    col2: str = "llm_score_openai",
) -> Dict[str, float]:
    """
    Compute inter-annotator agreement between two LLM annotations.

    Args:
        df: DataFrame with both annotation columns
        col1: First annotator's score column
        col2: Second annotator's score column

    Returns:
        Dict with Cohen's kappa, agreement rate, per-class agreement
    """
    from sklearn.metrics import cohen_kappa_score

    # Filter rows where both annotations exist
    mask = df[col1].notna() & df[col2].notna()
    scores1 = df.loc[mask, col1].astype(int).values
    scores2 = df.loc[mask, col2].astype(int).values

    n_valid = mask.sum()
    console.print(f"\n[cyan]Computing agreement on {n_valid:,} utterances[/cyan]")

    # Cohen's kappa (exact agreement)
    kappa_exact = cohen_kappa_score(scores1, scores2)

    # Cohen's kappa (linear weighted — for ordinal scale)
    kappa_weighted = cohen_kappa_score(scores1, scores2, weights="linear")

    # Quadratic weighted kappa (common for ordinal)
    kappa_quadratic = cohen_kappa_score(scores1, scores2, weights="quadratic")

    # Simple agreement rate
    agreement_rate = (scores1 == scores2).mean()

    # Adjacent agreement (within ±1)
    adjacent_agreement = (np.abs(scores1 - scores2) <= 1).mean()

    # Map to labels and compute label-level kappa
    labels1 = np.array([SCORE_TO_LABEL[s] for s in scores1])
    labels2 = np.array([SCORE_TO_LABEL[s] for s in scores2])
    kappa_label = cohen_kappa_score(labels1, labels2)

    results = {
        "n_samples": int(n_valid),
        "cohen_kappa_exact": float(kappa_exact),
        "cohen_kappa_linear_weighted": float(kappa_weighted),
        "cohen_kappa_quadratic_weighted": float(kappa_quadratic),
        "cohen_kappa_label_level": float(kappa_label),
        "exact_agreement_rate": float(agreement_rate),
        "adjacent_agreement_rate": float(adjacent_agreement),
    }

    # Print
    console.print(f"\n[bold]Inter-Annotator Agreement:[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    for k, v in results.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)

    # Interpretation
    if kappa_weighted >= 0.8:
        interp = "Almost perfect agreement"
    elif kappa_weighted >= 0.6:
        interp = "Substantial agreement"
    elif kappa_weighted >= 0.4:
        interp = "Moderate agreement"
    elif kappa_weighted >= 0.2:
        interp = "Fair agreement"
    else:
        interp = "Poor agreement"

    console.print(f"\n[bold]Interpretation: {interp}[/bold]")
    results["interpretation"] = interp

    return results


def create_hybrid_labels(
    df: pd.DataFrame,
    position_col: str = "label",
    content_col: str = "llm_label",
    content_score_col: str = "llm_score",
    confidence_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Create hybrid labels combining position-based and content-based.

    Strategy:
    - Start with position-based labels
    - Override when LLM confidence is high (score 1 or 5 = high confidence)
    - Captures: early panic (position=NORMAL but content=ELEVATED/CRITICAL)
    - Captures: late calm (position=CRITICAL but content=NORMAL)

    Args:
        df: DataFrame with both label columns
        position_col: Position-based label column
        content_col: Content-based label column
        content_score_col: Raw LLM score column (1-5)
        confidence_threshold: Score extremity threshold for override

    Returns:
        DataFrame with 'hybrid_label' column added
    """
    df = df.copy()

    # Start with position-based labels
    df["hybrid_label"] = df[position_col]

    if content_col not in df.columns or content_score_col not in df.columns:
        console.print("[yellow]Content labels not found. Using position labels only.[/yellow]")
        return df

    # Override logic
    overrides = 0
    for idx, row in df.iterrows():
        if pd.isna(row[content_score_col]):
            continue

        score = int(row[content_score_col])
        position_label = row[position_col]
        content_label = row[content_col]

        # High-confidence overrides:
        # 1. Score 5 (very high stress) but position says NORMAL → upgrade
        if score == 5 and position_label in ["NORMAL", "EARLY_WARNING"]:
            df.at[idx, "hybrid_label"] = content_label
            overrides += 1
        # 2. Score 4 and position says NORMAL → upgrade to at least EARLY_WARNING
        elif score == 4 and position_label == "NORMAL":
            df.at[idx, "hybrid_label"] = content_label
            overrides += 1
        # 3. Score 1 and position says CRITICAL → downgrade (false alarm reduction)
        elif score == 1 and position_label == "CRITICAL":
            df.at[idx, "hybrid_label"] = "ELEVATED"  # Softer downgrade
            overrides += 1

    console.print(f"[green]Hybrid labels created: {overrides} overrides out of {len(df)} utterances ({overrides/len(df)*100:.1f}%)[/green]")

    return df


def manual_validation_sample(
    df: pd.DataFrame,
    n_samples: int = 200,
    stratify_col: str = "llm_label",
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a stratified sample for manual validation by researchers.

    Args:
        df: Annotated DataFrame
        n_samples: Total samples to select
        stratify_col: Column to stratify by
        random_seed: Random seed
        output_path: Where to save the sample

    Returns:
        DataFrame sample for manual annotation
    """
    np.random.seed(random_seed)

    # Stratified sample
    sample_frames = []
    labels = df[stratify_col].dropna().unique()
    per_class = max(1, n_samples // len(labels))

    for label in labels:
        subset = df[df[stratify_col] == label]
        n = min(per_class, len(subset))
        sample_frames.append(subset.sample(n=n, random_state=random_seed))

    sample = pd.concat(sample_frames).sample(frac=1, random_state=random_seed)

    # Add manual annotation columns
    sample["manual_score"] = None
    sample["manual_label"] = None
    sample["manual_notes"] = None

    console.print(f"\n[green]Manual validation sample: {len(sample)} utterances[/green]")
    console.print("Label distribution in sample:")
    print(sample[stratify_col].value_counts())

    if output_path:
        # Save as CSV for easy annotation
        cols = ["case_id", "cvr_message", "cvr_speaker_source", "label",
                stratify_col, "llm_score", "llm_reasoning",
                "manual_score", "manual_label", "manual_notes"]
        existing = [c for c in cols if c in sample.columns]
        sample[existing].to_csv(output_path, index=False)
        console.print(f"[green]Saved to: {output_path}[/green]")

    return sample


def main():
    parser = argparse.ArgumentParser(description="LLM-Assisted CVR Annotation")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"])
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--context-size", type=int, default=5)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--compute-agreement", action="store_true")
    parser.add_argument("--create-hybrid", action="store_true")
    parser.add_argument("--create-validation-sample", action="store_true")
    parser.add_argument("--rate-limit", type=float, default=0.2, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Paths
    data_path = PROJECT_ROOT / "data" / "cvr_labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"

    if not data_path.exists():
        console.print(f"[red]Data not found: {data_path}[/red]")
        return

    output_dir = PROJECT_ROOT / "data" / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print(f"[cyan]Loading data from: {data_path}[/cyan]")
    df = pd.read_csv(data_path)
    console.print(f"[green]Loaded {len(df):,} utterances from {df['case_id'].nunique()} cases[/green]")

    if args.compute_agreement:
        # Load both annotations and compute agreement
        deepseek_path = output_dir / "cvr_annotated_deepseek.csv"
        openai_path = output_dir / "cvr_annotated_openai.csv"

        if deepseek_path.exists() and openai_path.exists():
            df_ds = pd.read_csv(deepseek_path)
            df_oa = pd.read_csv(openai_path)

            # Merge
            df_merged = df_ds.copy()
            df_merged["llm_score_deepseek"] = df_ds["llm_score"]
            df_merged["llm_score_openai"] = df_oa["llm_score"]
            df_merged["llm_label_deepseek"] = df_ds["llm_label"]
            df_merged["llm_label_openai"] = df_oa["llm_label"]

            agreement = compute_inter_annotator_agreement(df_merged)

            # Save
            with open(output_dir / "inter_annotator_agreement.json", "w") as f:
                json.dump(agreement, f, indent=2)
            console.print(f"[green]Agreement saved to {output_dir / 'inter_annotator_agreement.json'}[/green]")
        else:
            console.print("[red]Need both deepseek and openai annotations. Run annotation first.[/red]")
        return

    if args.create_hybrid:
        # Create hybrid labels from existing annotations
        annotated_path = output_dir / f"cvr_annotated_{args.provider}.csv"
        if annotated_path.exists():
            df_annotated = pd.read_csv(annotated_path)
            df_hybrid = create_hybrid_labels(df_annotated)
            hybrid_path = output_dir / "cvr_hybrid_labeled.csv"
            df_hybrid.to_csv(hybrid_path, index=False)
            console.print(f"[green]Hybrid labels saved to {hybrid_path}[/green]")
        else:
            console.print(f"[red]Annotated data not found: {annotated_path}[/red]")
        return

    if args.create_validation_sample:
        annotated_path = output_dir / f"cvr_annotated_{args.provider}.csv"
        if annotated_path.exists():
            df_annotated = pd.read_csv(annotated_path)
            manual_validation_sample(
                df_annotated,
                n_samples=200,
                output_path=output_dir / "manual_validation_sample.csv",
            )
        else:
            console.print(f"[red]Annotated data not found: {annotated_path}[/red]")
        return

    # Run annotation
    output_path = output_dir / f"cvr_annotated_{args.provider}.csv"

    annotator = LLMAnnotator(
        provider=args.provider,
        model=args.model,
        rate_limit_delay=args.rate_limit,
    )

    df_annotated = annotator.annotate_dataset(
        df,
        context_size=args.context_size,
        batch_size=args.batch_size,
        save_every=500,
        output_path=output_path,
        resume_from=args.resume_from,
    )

    # Save final
    df_annotated.to_csv(output_path, index=False)
    console.print(f"\n[green]Annotated dataset saved to: {output_path}[/green]")

    # Summary statistics
    console.print("\n[bold]Annotation Summary:[/bold]")
    if "llm_label" in df_annotated.columns:
        console.print("\nLLM Label Distribution:")
        print(df_annotated["llm_label"].value_counts())

        console.print("\nPosition vs Content Label Comparison:")
        if "label" in df_annotated.columns:
            cross = pd.crosstab(
                df_annotated["label"], df_annotated["llm_label"],
                margins=True,
            )
            print(cross)


if __name__ == "__main__":
    main()
