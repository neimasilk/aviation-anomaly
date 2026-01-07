"""
Data preprocessing module for CVR transcript analysis.

Handles:
- Loading and cleaning CVR transcripts
- Temporal labeling based on time-before-crash
- Window segmentation
- Feature extraction
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.config import config


class CVRPreprocessor:
    """Preprocessor for Cockpit Voice Recorder transcripts."""

    # Aviation urgency keywords
    URGENCY_KEYWORDS = [
        "mayday", "emergency", "terrain", "pull up", "warning",
        "caution", "abort", "help", "fire", "failure", "disconnect"
    ]

    # Label thresholds (in minutes before crash)
    LABEL_THRESHOLDS = {
        "CRITICAL": (0, 1),      # 0-1 minutes before
        "ELEVATED": (1, 5),      # 1-5 minutes before
        "EARLY_WARNING": (5, 10), # 5-10 minutes before
        "NORMAL": (10, None),    # >10 minutes before
    }

    def __init__(self):
        self.data_dir = config.data_dir
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_transcripts(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load CVR transcripts from file.

        Args:
            filepath: Path to transcript file. If None, looks in default location.

        Returns:
            DataFrame with transcript data
        """
        if filepath is None:
            filepath = self.raw_dir / "cvr_transcripts.csv"

        # Support different formats
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        elif filepath.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        elif filepath.suffix == ".json":
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        return self.clean_transcripts(df)

    def clean_transcripts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize transcript data.

        Args:
            df: Raw transcript DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Standardize column names (Noort et al. dataset uses specific naming)
        column_mapping = {
            "cvr_message": "utterance",
            "cvr_speaker_role": "speaker_role",
            "cvr_turn_number": "turn_number",
            "case_id": "case_id",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Clean text
        if "utterance" in df.columns:
            df["utterance"] = df["utterance"].apply(self._clean_text)

        # Remove empty utterances
        if "utterance" in df.columns:
            df = df[df["utterance"].str.len() > 0].reset_index(drop=True)

        return df

    def _clean_text(self, text: str) -> str:
        """Clean individual utterance text."""
        if pd.isna(text):
            return ""

        text = str(text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove timestamps in common formats
        text = re.sub(r"\[\d+:\d+:\d+\]", "", text)
        text = re.sub(r"\(\d+:\d+:\d+\)", "", text)
        # Remove speaker tags if embedded in text
        text = re.sub(r"^(Captain|First Officer|FO|CAP|FEC):", "", text)
        return text.strip()

    def assign_temporal_labels(
        self,
        df: pd.DataFrame,
        time_column: str = "time_to_crash",
        time_unit: str = "minutes"
    ) -> pd.DataFrame:
        """
        Assign temporal labels based on time before crash.

        Args:
            df: DataFrame with time_to_crash column
            time_column: Column name for time values
            time_unit: Unit of time (minutes, seconds)

        Returns:
            DataFrame with added 'label' column
        """
        df = df.copy()

        # Convert to minutes if needed
        if time_unit == "seconds":
            df[time_column] = df[time_column] / 60

        def get_label(minutes: float) -> str:
            for label, (min_min, max_min) in self.LABEL_THRESHOLDS.items():
                if max_min is None:
                    if minutes >= min_min:
                        return label
                else:
                    if min_min <= minutes < max_min:
                        return label
            return "NORMAL"

        df["label"] = df[time_column].apply(get_label)
        return df

    def assign_temporal_labels_by_position(
        self,
        df: pd.DataFrame,
        group_column: str = "case_id",
        critical_ratio: float = 0.05,
        elevated_ratio: float = 0.15,
        early_warning_ratio: float = 0.35,
    ) -> pd.DataFrame:
        """
        Assign temporal labels based on relative position within each case.

        Uses utterance position as a proxy for time before crash, since CVR
        transcripts are recorded chronologically and the crash occurs at the
        end of the recording.

        Args:
            df: DataFrame with utterances
            group_column: Column to group by (e.g., case_id)
            critical_ratio: Last X% of utterances labeled CRITICAL (<1 min)
            elevated_ratio: Last X% of utterances labeled ELEVATED (1-5 min)
            early_warning_ratio: Last X% of utterances labeled EARLY_WARNING (5-10 min)
                               Remaining utterances labeled NORMAL (>10 min)

        Returns:
            DataFrame with added 'label' and 'relative_position' columns
        """
        df = df.copy()

        # Calculate position within each case
        df["utterance_position"] = df.groupby(group_column).cumcount()
        df["case_utterance_count"] = df.groupby(group_column)[group_column].transform("count")
        df["relative_position"] = df["utterance_position"] / df["case_utterance_count"]

        def get_position_label(rel_pos: float) -> str:
            """Assign label based on relative position (0.0 = start, 1.0 = end/crash)."""
            # CRITICAL: Last critical_ratio% (e.g., last 5% = final minute)
            if rel_pos >= (1 - critical_ratio):
                return "CRITICAL"
            # ELEVATED: From (1 - elevated_ratio)% to (1 - critical_ratio)%
            elif rel_pos >= (1 - elevated_ratio):
                return "ELEVATED"
            # EARLY_WARNING: From (1 - early_warning_ratio)% to (1 - elevated_ratio)%
            elif rel_pos >= (1 - early_warning_ratio):
                return "EARLY_WARNING"
            # NORMAL: Everything else (early conversation)
            else:
                return "NORMAL"

        df["label"] = df["relative_position"].apply(get_position_label)

        # Drop helper columns
        df = df.drop(columns=["utterance_position", "case_utterance_count", "relative_position"])

        return df

    def create_windows(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        stride: int = 5,
        group_column: str = "case_id"
    ) -> pd.DataFrame:
        """
        Create sliding windows of utterances.

        Args:
            df: DataFrame with utterances
            window_size: Number of utterances per window
            stride: Step size for sliding window
            group_column: Column to group by (e.g., case_id)

        Returns:
            DataFrame with windowed data
        """
        windows = []

        for case_id, group in df.groupby(group_column):
            group = group.sort_values("turn_number").reset_index(drop=True)

            for i in range(0, len(group) - window_size + 1, stride):
                window = group.iloc[i:i + window_size].copy()
                window["window_id"] = f"{case_id}_w{i}"
                window["window_start"] = i
                window["window_end"] = i + window_size
                windows.append(window)

        if windows:
            return pd.concat(windows, ignore_index=True)
        return pd.DataFrame()

    def extract_linguistic_features(self, utterance: str) -> Dict[str, float]:
        """
        Extract linguistic features from an utterance.

        Args:
            utterance: Text utterance

        Returns:
            Dictionary of features
        """
        features = {
            "word_count": len(utterance.split()),
            "char_count": len(utterance),
            "question_mark_count": utterance.count("?"),
            "exclamation_count": utterance.count("!"),
            "has_urgency_keyword": any(
                kw.lower() in utterance.lower() for kw in self.URGENCY_KEYWORDS
            ),
            "urgency_keyword_count": sum(
                1 for kw in self.URGENCY_KEYWORDS if kw.lower() in utterance.lower()
            ),
        }

        # Speech rate proxy (words per character - rough estimate)
        if features["char_count"] > 0:
            features["word_density"] = features["word_count"] / features["char_count"]
        else:
            features["word_density"] = 0

        return features

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        group_column: str = "case_id",
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.

        Splits by group_column to ensure no data leakage across cases.

        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            group_column: Column to group by
            random_seed: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Get unique groups
        unique_groups = df[group_column].unique()
        np.random.seed(random_seed)
        np.random.shuffle(unique_groups)

        # Calculate split points
        n_train = int(len(unique_groups) * train_ratio)
        n_val = int(len(unique_groups) * (train_ratio + val_ratio))

        train_groups = unique_groups[:n_train]
        val_groups = unique_groups[n_train:n_val]
        test_groups = unique_groups[n_val:]

        train_df = df[df[group_column].isin(train_groups)]
        val_df = df[df[group_column].isin(val_groups)]
        test_df = df[df[group_column].isin(test_groups)]

        return train_df, val_df, test_df

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        """Save processed DataFrame."""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        return filepath


def preprocess_pipeline(
    input_file: Path,
    output_prefix: str = "cvr_processed"
) -> Tuple[Path, Path, Path]:
    """
    Run the full preprocessing pipeline.

    Args:
        input_file: Path to raw transcript file
        output_prefix: Prefix for output files

    Returns:
        Tuple of (train_path, val_path, test_path)
    """
    preprocessor = CVRPreprocessor()

    # Load and clean
    print("Loading transcripts...")
    df = preprocessor.load_transcripts(input_file)
    print(f"Loaded {len(df)} utterances from {df['case_id'].nunique()} cases")

    # Assign temporal labels
    print("Assigning temporal labels...")
    if "time_to_crash" in df.columns:
        # Use explicit time values if available
        df = preprocessor.assign_temporal_labels(df)
        print("Using explicit time_to_crash values")
    else:
        # Use position-based labeling as proxy (CV Rs are chronological)
        df = preprocessor.assign_temporal_labels_by_position(df)
        print("Using position-based labeling (utterance order as time proxy)")

    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nPercentage distribution:")
    print(df["label"].value_counts(normalize=True) * 100)

    # Create windows
    print("Creating sliding windows...")
    windows_df = preprocessor.create_windows(df)
    print(f"Created {len(windows_df)} windows")

    # Split data
    print("Splitting data...")
    train_df, val_df, test_df = preprocessor.split_data(windows_df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save
    train_path = preprocessor.save_processed(train_df, f"{output_prefix}_train.csv")
    val_path = preprocessor.save_processed(val_df, f"{output_prefix}_val.csv")
    test_path = preprocessor.save_processed(test_df, f"{output_prefix}_test.csv")

    print(f"\nSaved processed files to {preprocessor.processed_dir}")

    return train_path, val_path, test_path


if __name__ == "__main__":
    # Example usage
    from ..utils.config import config

    # Update the path to point to actual data when available
    input_path = config.data_dir / "raw" / "cvr_transcripts.csv"

    if input_path.exists():
        preprocess_pipeline(input_path)
    else:
        print(f"No data file found at {input_path}")
        print("Please place the Noort et al. (2021) dataset in data/raw/")
