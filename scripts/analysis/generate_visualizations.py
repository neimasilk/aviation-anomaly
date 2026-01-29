"""
Generate Comprehensive Visualizations for Paper

Creates publication-ready figures:
1. Model comparison bar chart (accuracy & F1)
2. Confusion matrices (all models)
3. Per-class performance heatmap
4. Training curves comparison
5. Change point detection examples
6. Statistical significance matrix

Usage:
    python scripts/analysis/generate_visualizations.py

Output:
    paper_assets/figX_*.png
    outputs/analysis/figures/
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from rich.console import Console

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_all_results() -> Dict[str, Dict]:
    """Load results from all experiments."""
    results = {}
    exp_ids = ["001", "002", "003", "004", "005"]
    
    for exp_id in exp_ids:
        result_path = PROJECT_ROOT / "outputs" / "experiments" / exp_id / "results.json"
        alt_result_path = PROJECT_ROOT / "outputs" / exp_id / "results.json"
        
        actual_path = None
        if result_path.exists():
            actual_path = result_path
        elif alt_result_path.exists():
            actual_path = alt_result_path
        
        if actual_path:
            try:
                with open(actual_path) as f:
                    data = json.load(f)
                    # Normalize structure
                    if "test_accuracy" in data:
                        # Old format (001-004)
                        old_data = data.copy()
                        data = {
                            "experiment": {"id": exp_id, "title": f"Experiment {exp_id}"},
                            "metrics": {
                                "accuracy": old_data.get("test_accuracy", 0),
                                "macro_f1": old_data.get("test_f1_macro", 0),
                            }
                        }
                    results[exp_id] = data
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load {exp_id}: {e}[/yellow]")
    
    return results


def create_model_comparison_figure(results: Dict, save_path: Path):
    """Figure 1: Overall model comparison (accuracy and F1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract classification results
    exp_names = []
    accuracies = []
    f1_scores = []
    
    for exp_id in ["001", "002", "003", "004"]:
        if exp_id in results and "accuracy" in results[exp_id].get("metrics", {}):
            exp_names.append(f"Exp {exp_id}")
            accuracies.append(results[exp_id]["metrics"]["accuracy"] * 100)
            f1_scores.append(results[exp_id]["metrics"].get("macro_f1", 0) * 100)
    
    # Accuracy subplot
    colors = ['#e74c3c' if acc < 70 else '#f39c12' if acc < 80 else '#27ae60' for acc in accuracies]
    bars1 = axes[0].bar(exp_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='Target (80%)')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].legend()
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # F1 subplot
    colors_f1 = ['#e74c3c' if f1 < 60 else '#f39c12' if f1 < 70 else '#27ae60' for f1 in f1_scores]
    bars2 = axes[1].bar(exp_names, f1_scores, color=colors_f1, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=70, color='gray', linestyle='--', alpha=0.7, label='Target (F1=0.70)')
    axes[1].set_ylabel('Macro F1 (%)', fontsize=12)
    axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].legend()
    
    # Add value labels
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / "fig1_model_comparison_detailed.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "fig1_model_comparison_detailed.pdf", bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved: {save_path / 'fig1_model_comparison_detailed.png'}[/green]")


def create_confusion_matrices_figure(results: Dict, save_path: Path):
    """Figure 2: Confusion matrices for all classification models."""
    classification_exps = ["001", "002", "003", "004"]
    n_models = len([e for e in classification_exps if e in results])
    
    if n_models == 0:
        console.print("[yellow]No confusion matrices available[/yellow]")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    class_names = ["NORMAL", "EARLY\nWARNING", "ELEVATED", "CRITICAL"]
    
    for idx, exp_id in enumerate(classification_exps):
        if exp_id not in results:
            continue
        
        cm_path = PROJECT_ROOT / "outputs" / "experiments" / exp_id / "confusion_matrix.npy"
        if cm_path.exists():
            cm = np.load(cm_path)
            
            # Normalize by row
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], cbar_kws={'label': 'Proportion'})
            axes[idx].set_title(f'Experiment {exp_id}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('True', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_models, 4):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / "fig2_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "fig2_confusion_matrices.pdf", bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved: {save_path / 'fig2_confusion_matrices.png'}[/green]")


def create_per_class_performance_heatmap(results: Dict, save_path: Path):
    """Figure 3: Per-class F1 heatmap for all models."""
    # Extract per-class F1 scores
    models = []
    classes = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]
    f1_data = []
    
    for exp_id in ["001", "002", "003", "004"]:
        if exp_id not in results:
            continue
        
        metrics = results[exp_id].get("metrics", {})
        
        # Try to get per-class F1
        per_class_f1 = []
        for cls in classes:
            key = f"f1_{cls}"
            if key in metrics:
                per_class_f1.append(metrics[key] * 100)
            else:
                per_class_f1.append(0)
        
        if any(per_class_f1):
            models.append(f"Exp {exp_id}")
            f1_data.append(per_class_f1)
    
    if not models:
        console.print("[yellow]No per-class F1 data available[/yellow]")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f1_df = pd.DataFrame(f1_data, index=models, columns=classes)
    
    sns.heatmap(f1_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'F1 Score (%)'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Anomaly Level', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path / "fig3_per_class_f1_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "fig3_per_class_f1_heatmap.pdf", bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved: {save_path / 'fig3_per_class_f1_heatmap.png'}[/green]")


def create_change_point_visualization(results: Dict, save_path: Path):
    """Figure 4: Change point detection results."""
    if "005" not in results:
        console.print("[yellow]No change point results available[/yellow]")
        return
    
    metrics = results["005"].get("metrics", {})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Metrics bar chart
    metric_names = ['MAE\n(utterances)', 'Early\nDetection\nRate', 'Accuracy\n@ ±5 utt']
    metric_values = [
        min(float(metrics.get("mae", 0)) / 100 * 100, 100),  # Normalize for visualization
        float(metrics.get("early_detection_rate", 0)) * 100,
        float(metrics.get("accuracy_at_5", 0)) * 100
    ]
    
    colors = ['#e74c3c', '#27ae60', '#3498db']
    bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Percentage / Normalized Score', fontsize=11)
    axes[0].set_title('Change Point Detection Performance', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 100])
    
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        if val == metric_values[0]:  # MAE
            label = f'{float(metrics.get("mae", 0)):.1f}\nutt'
        elif val == metric_values[1]:  # Early detection
            label = f'{val:.1f}%'
        else:  # Accuracy
            label = f'{val:.1f}%'
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom', fontweight='bold')
    
    # Detection distribution pie chart
    early_rate = float(metrics.get("early_detection_rate", 0))
    late_rate = float(metrics.get("late_detection_rate", 0))
    
    sizes = [early_rate * 100, late_rate * 100]
    labels = [f'Early Detection\n({early_rate*100:.1f}%)', 
              f'Late Detection\n({late_rate*100:.1f}%)']
    colors_pie = ['#27ae60', '#e74c3c']
    explode = (0.05, 0)
    
    axes[1].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90,
                textprops={'fontsize': 11})
    axes[1].set_title('Detection Timing Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / "fig4_change_point_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "fig4_change_point_results.pdf", bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved: {save_path / 'fig4_change_point_results.png'}[/green]")


def create_summary_table(results: Dict, save_path: Path):
    """Create publication-ready summary table."""
    rows = []
    
    for exp_id, data in results.items():
        exp_title = data.get("experiment", {}).get("title", f"Exp {exp_id}")
        metrics = data.get("metrics", {})
        
        if "accuracy" in metrics:  # Classification
            row = {
                "Experiment": f"{exp_id}: {exp_title}",
                "Type": "Classification",
                "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                "Macro F1": f"{metrics.get('macro_f1', 0)*100:.2f}%",
                "Key Finding": data.get("experiment", {}).get("novelty", "Baseline")[:50] + "..."
            }
        else:  # Change point
            row = {
                "Experiment": f"{exp_id}: {exp_title}",
                "Type": "Change Point",
                "MAE": f"{float(metrics.get('mae', 0)):.1f} utt",
                "Early Det.": f"{float(metrics.get('early_detection_rate', 0))*100:.1f}%",
                "Key Finding": "Novel approach - detects anomaly onset"
            }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(save_path / "summary_table.csv", index=False)
    
    # Save as LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    with open(save_path / "summary_table.tex", "w") as f:
        f.write(latex_table)
    
    console.print(f"[green]Saved summary tables to {save_path}[/green]")
    
    return df


def main():
    """Main visualization generation."""
    console.print("\n[bold blue]Generating Paper Visualizations[/bold blue]")
    console.print("=" * 60)
    
    # Load results
    results = load_all_results()
    console.print(f"Loaded {len(results)} experiment results")
    
    # Create output directories
    figures_dir = PROJECT_ROOT / "outputs" / "analysis" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    paper_assets_dir = PROJECT_ROOT / "paper_assets"
    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    console.print("\n[yellow]Generating figures...[/yellow]")
    
    create_model_comparison_figure(results, figures_dir)
    create_model_comparison_figure(results, paper_assets_dir)
    
    create_confusion_matrices_figure(results, figures_dir)
    create_confusion_matrices_figure(results, paper_assets_dir)
    
    create_per_class_performance_heatmap(results, figures_dir)
    create_per_class_performance_heatmap(results, paper_assets_dir)
    
    create_change_point_visualization(results, figures_dir)
    create_change_point_visualization(results, paper_assets_dir)
    
    # Create summary table
    console.print("\n[yellow]Creating summary table...[/yellow]")
    summary_df = create_summary_table(results, figures_dir)
    console.print(summary_df.to_string(index=False))
    
    console.print("\n[bold green]All visualizations generated successfully![/bold green]")
    console.print(f"\nFigures saved to:")
    console.print(f"  - {figures_dir}")
    console.print(f"  - {paper_assets_dir}")


if __name__ == "__main__":
    main()
