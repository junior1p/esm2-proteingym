"""Visualization and report generation for ESM-2 mutation scoring results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Optional


# Demo protein: GFP (Green Fluorescent Protein) — UniProt P42212
# Sarkisyan et al. 2016 GFP DMS is the most complete assay in ProteinGym
DEMO_SEQUENCE = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
    "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHY"
    "QQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
)


def generate_heatmap(
    predictions: list[dict],
    sequence: str,
    output_path: str = "mutation_heatmap.png"
) -> None:
    """Generate positional mutation heatmap (positions × amino acids).
    
    Color = ESM-2 LLR score:
      - Red = stabilizing/beneficial (positive score)
      - Blue = destabilizing/deleterious (negative score)
      - White = neutral (score ≈ 0)
    
    Black dots mark wild-type amino acids at each position.
    """
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    df = pd.DataFrame(predictions)
    
    # Pivot: rows = positions, columns = mutant amino acids
    pivot = df.pivot_table(
        index="position",
        columns="mut_aa",
        values="esm2_score"
    ).reindex(columns=AMINO_ACIDS)
    
    fig, ax = plt.subplots(figsize=(max(12, len(sequence) // 5), 6))
    sns.heatmap(
        pivot.T,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "ESM-2 Masked Marginal LLR\n(positive = beneficial)"}
    )
    ax.set_title("Zero-Shot Mutation Fitness Landscape (ESM-2)", fontsize=14)
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Mutant Amino Acid")
    
    # Mark wild-type positions with black dots
    for i, aa in enumerate(sequence):
        if aa in AMINO_ACIDS:
            col_idx = AMINO_ACIDS.index(aa)
            ax.plot(i + 0.5, col_idx + 0.5, 'k.', markersize=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def generate_correlation_plot(
    merged_df: pd.DataFrame,
    output_path: str = "correlation_plot.png",
    metrics: dict = None
) -> None:
    """Scatter plot of ESM-2 scores vs. experimental DMS scores.
    
    Includes Spearman correlation annotation and trend line.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        merged_df["esm2_score"],
        merged_df["DMS_score"],
        alpha=0.3, s=8, color="#2196F3"
    )
    ax.set_xlabel("ESM-2 Masked Marginal Score (predicted)", fontsize=12)
    ax.set_ylabel("DMS Experimental Score (ground truth)", fontsize=12)
    ax.set_title("ESM-2 Zero-Shot Predictions vs. ProteinGym Experimental Data", fontsize=12)
    
    # Add trend line
    z = np.polyfit(merged_df["esm2_score"], merged_df["DMS_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_df["esm2_score"].min(), merged_df["esm2_score"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5)
    
    # Add Spearman annotation
    if metrics:
        ax.text(
            0.05, 0.95,
            f"Spearman ρ = {metrics['spearman_r']:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Correlation plot saved to: {output_path}")


def generate_report(
    sequence: str,
    predictions: list[dict],
    metrics: dict,
    assay_name: str,
    model_name: str,
    output_path: str = "mutation_report.txt"
) -> None:
    """Generate human-readable text report with top hits and metrics."""
    df = pd.DataFrame(predictions)
    top_beneficial = df.nlargest(10, "esm2_score")
    top_deleterious = df.nsmallest(10, "esm2_score")
    
    lines = [
        "=" * 65,
        "ESM-2 ZERO-SHOT MUTATION FITNESS REPORT",
        "=" * 65,
        f"Model: {model_name}",
        f"Sequence length: {len(sequence)} aa",
        f"Total mutants scored: {len(predictions)}",
        f"ProteinGym assay: {assay_name if assay_name else 'Not matched'}",
        "",
        "--- TOP 10 PREDICTED BENEFICIAL MUTATIONS ---",
        top_beneficial[["mutant", "esm2_score"]].to_string(index=False),
        "",
        "--- TOP 10 PREDICTED DELETERIOUS MUTATIONS ---",
        top_deleterious[["mutant", "esm2_score"]].to_string(index=False),
        "",
        "--- BENCHMARK VALIDATION (vs. ProteinGym DMS) ---",
        f"  Matched mutants:   {metrics['n_mutants']}",
        f"  Spearman ρ:        {metrics['spearman_r']} (p={metrics['spearman_p']:.2e})",
        f"  Pearson r:         {metrics['pearson_r']}",
        f"  Kendall tau:       {metrics['kendall_tau']}",
        f"  AUC-ROC:           {metrics['auc_roc']}",
        "",
        "  Context: ProteinGym top models achieve Spearman ~0.45–0.55.",
        "  ESM-1v baseline: ~0.44. ESM-2 650M: typically ~0.44–0.50.",
        "",
        "=" * 65,
    ]
    
    report_text = "\n".join(lines)
    print(report_text)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {output_path}")


def save_results(
    predictions: list[dict],
    output_path: str = "mutation_scores.csv"
) -> pd.DataFrame:
    """Save all scored mutants to CSV, sorted by ESM-2 score (descending)."""
    df = pd.DataFrame(predictions)
    df = df.sort_values("esm2_score", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Full results saved to: {output_path}")
    return df
