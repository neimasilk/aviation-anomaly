"""
Load and convert Noort et al. (2021) CVR dataset from SPSS to CSV.

Usage:
    python -m src.data.load_data
"""
import sys
from pathlib import Path

import pandas as pd
import pyreadstat
from rich.console import Console
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


def load_spss_file(filepath: Path) -> pd.DataFrame:
    """Load SPSS .sav file."""
    console.print(f"[cyan]Loading SPSS file: {filepath}[/cyan]")

    df, meta = pyreadstat.read_sav(filepath)

    console.print(f"[green]✓ Loaded {len(df):,} rows, {len(df.columns)} columns[/green]")
    return df


def show_summary(df: pd.DataFrame):
    """Show dataset summary."""
    console.print("\n[bold]Dataset Summary[/bold]")

    # Basic info
    console.print(f"\n[cyan]Shape:[/cyan] {df.shape[0]:,} rows × {df.shape[1]} columns")
    console.print(f"[cyan]Unique cases:[/cyan] {df['case_id'].nunique()} accidents")

    # Columns table
    console.print("\n[bold]Key Columns:[/bold]")
    key_cols = [
        'case_id', 'case_name', 'case_date', 'cvr_message',
        'cvr_speaker_source', 'coding_safety_concern'
    ]

    table = Table(show_header=True)
    table.add_column("Column", style="cyan")
    table.add_column("Non-Null", style="green")
    table.add_column("Sample", style="yellow")

    for col in key_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            sample = str(df[col].dropna().iloc[0])[:50] if non_null > 0 else "N/A"
            table.add_row(col, f"{non_null:,}", sample)

    console.print(table)

    # Speaker distribution
    if 'cvr_speaker_source' in df.columns:
        console.print("\n[bold]Speaker Distribution:[/bold]")
        speaker_counts = df['cvr_speaker_source'].value_counts().head(10)
        for speaker, count in speaker_counts.items():
            console.print(f"  {speaker}: {count:,}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset."""
    console.print("\n[cyan]Cleaning data...[/cyan]")

    # Select relevant columns
    cols_to_keep = [
        'case_id',
        'case_name',
        'case_date',
        'case_source_URL',
        'cvr_message',
        'cvr_speaker_source',
        'cvr_time_stamp',
        'coding_safety_concern',
        'coding_hazard_type',
        'coding_safety_voice',
        'coding_safety_listening',
    ]

    # Keep only columns that exist
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df_clean = df[existing_cols].copy()

    # Clean text columns
    if 'cvr_message' in df_clean.columns:
        df_clean['cvr_message'] = df_clean['cvr_message'].fillna('').astype(str)

    if 'cvr_speaker_source' in df_clean.columns:
        df_clean['cvr_speaker_source'] = df_clean['cvr_speaker_source'].fillna('Unknown').astype(str)

    console.print(f"[green]✓ Cleaned to {len(df_clean):,} rows, {len(df_clean.columns)} columns[/green]")
    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: Path):
    """Save processed data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df.to_csv(output_path, index=False)
    console.print(f"\n[green]✓ Saved to: {output_path}[/green]")

    # Also save as parquet (faster, smaller)
    parquet_path = output_path.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    console.print(f"[green]✓ Saved to: {parquet_path}[/green]")


def main():
    """Main processing function."""
    # Paths
    raw_file = PROJECT_ROOT / "data" / "raw" / "mmc4.sav"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_file = output_dir / "cvr_transcripts.csv"

    console.print("[bold blue]" + "="*60 + "[/bold blue]")
    console.print("[bold blue]Noort et al. (2021) CVR Dataset Loader[/bold blue]")
    console.print("[bold blue]" + "="*60 + "[/bold blue]")

    # Load
    df = load_spss_file(raw_file)

    # Show summary
    show_summary(df)

    # Clean
    df_clean = clean_data(df)

    # Save
    save_processed_data(df_clean, output_file)

    # Sample utterances
    console.print("\n[bold]Sample Utterances:[/bold]")
    sample_df = df_clean[df_clean['cvr_message'].str.len() > 20].head(5)
    for idx, row in sample_df.iterrows():
        console.print(f"\n  [yellow]{row.get('cvr_speaker_source', 'Unknown')}:[/yellow]")
        console.print(f"  {row['cvr_message'][:100]}...")

    console.print(f"\n[green]✓ Processing complete![/green]")
    console.print(f"\n[cyan]Next:[/cyan] python -m src.data.preprocessing")


if __name__ == "__main__":
    main()
