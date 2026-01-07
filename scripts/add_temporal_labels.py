"""
Add temporal labels to CVR dataset using position-based approach.

Since the Noort et al. (2021) dataset doesn't have explicit timestamps,
we use utterance position as a proxy for time before crash.

Usage:
    python scripts/add_temporal_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import CVRPreprocessor


def main():
    input_file = PROJECT_ROOT / "data" / "processed" / "cvr_transcripts.csv"
    output_file = PROJECT_ROOT / "data" / "processed" / "cvr_labeled.csv"

    print("=" * 60)
    print("Adding Temporal Labels to CVR Dataset")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} utterances from {df['case_id'].nunique()} cases")

    # Add turn_number if not present (needed for windowing)
    if "turn_number" not in df.columns:
        print("\nAdding turn_number column...")
        df["turn_number"] = df.groupby("case_id").cumcount()

    # Assign temporal labels
    print("\nAssigning temporal labels (position-based)...")
    preprocessor = CVRPreprocessor()
    df_labeled = preprocessor.assign_temporal_labels_by_position(
        df,
        group_column="case_id",
        # Default ratios: 5% CRITICAL, 10% ELEVATED, 20% EARLY_WARNING
        # This roughly corresponds to time-before-crash buckets
        critical_ratio=0.05,      # Last 5% = CRITICAL
        elevated_ratio=0.15,      # 5-15% from end = ELEVATED
        early_warning_ratio=0.35, # 15-35% from end = EARLY_WARNING
    )

    # Show label distribution
    print("\nLabel Distribution:")
    print("-" * 40)
    label_counts = df_labeled["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df_labeled) * 100
        bar = "=" * int(pct / 2)
        print(f"  {label:15s}: {count:5,} ({pct:5.1f}%) {bar}")

    # Show sample by label
    print("\nSample Utterances by Label:")
    print("-" * 40)
    for label in ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]:
        sample_df = df_labeled[df_labeled["label"] == label]
        if len(sample_df) > 0:
            sample_row = sample_df.iloc[0]
            msg = str(sample_row["cvr_message"])[:80]
            print(f"\n[{label}]:")
            print(f"  Speaker: {sample_row.get('cvr_speaker_source', 'Unknown')}")
            print(f"  Message: {msg}...")

    # Save labeled data
    df_labeled.to_csv(output_file, index=False)
    print(f"\n{'=' * 60}")
    print(f"Labeled data saved to: {output_file}")
    print(f"{'=' * 60}")

    # Show what's next
    print("\nNext steps:")
    print("  1. Upload to Google Drive: .\\scripts\\sync_drive.bat upload")
    print("  2. Run experiment 001: cd experiments/001_baseline_bert && python run.py")


if __name__ == "__main__":
    main()
