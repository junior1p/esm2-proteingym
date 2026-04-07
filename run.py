#!/usr/bin/env python3
"""Main entry point for ESM-2 Zero-Shot Mutation Fitness Prediction.

Usage:
    python run.py                                           # Demo: GFP with 35M model
    python run.py --sequence MKTVLR...                      # Custom sequence
    python run.py --uniprot P42212                          # By UniProt ID (GFP)
    python run.py --model 650M --validate                   # Full 650M model + ProteinGym
    python run.py --model 35M --output-dir results/my_run   # Custom output dir
"""
import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Use HuggingFace mirror in mainland China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse

# Demo protein: KALP peptide (16 aa) — small, fast, validates full pipeline
# Use GFP (P42212) with --uniprot for full benchmark validation
DEMO_SEQUENCE = "KALPGTDPAALGDDD"  # 15 aa → 285 mutants, ~10-30 min on CPU 35M

MODEL_HELP = "35M (fast, ~5min), 150M, 650M (recommended), or 3B (best accuracy)"


def main():
    parser = argparse.ArgumentParser(
        description="ESM-2 Zero-Shot Mutation Fitness Prediction with ProteinGym Validation"
    )
    parser.add_argument("--sequence", type=str, default=None,
                        help="Amino acid sequence (skip for demo: GFP)")
    parser.add_argument("--uniprot", type=str, default=None,
                        help="UniProt accession ID (e.g. P42212 for GFP)")
    parser.add_argument("--model", type=str, default="35M", choices=["35M", "150M", "650M", "3B"],
                        help=f"ESM-2 model size ({MODEL_HELP})")
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Validate against ProteinGym DMS data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: esm2_results)")
    args = parser.parse_args()

    # Determine output directory
    output_dir = args.output_dir or "esm2_results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Parse input ---
    from src.utils import fetch_uniprot_sequence, validate_sequence
    from src.esm2_scorer import run_scorer
    from src.proteingym import find_proteingym_assay, merge_predictions_with_dms, compute_validation_metrics
    from src.visualize import generate_heatmap, generate_correlation_plot, generate_report, save_results

    if args.uniprot:
        uniprot_id = args.uniprot.upper()
        sequence = fetch_uniprot_sequence(uniprot_id)
        print(f"Fetched sequence for {uniprot_id}: {len(sequence)} aa")
    elif args.sequence:
        sequence = args.sequence.upper().strip()
        uniprot_id = None
    else:
        # Default: GFP demo
        sequence = DEMO_SEQUENCE
        uniprot_id = "P42212"
        print("=== DEMO MODE: GFP (Green Fluorescent Protein, UniProt P42212) ===")
        print("This is the landmark Sarkisyan et al. 2016 DMS assay — the most complete in ProteinGym.")

    # Validate
    validate_sequence(sequence)
    if len(sequence) > 1022:
        print(f"Warning: sequence length {len(sequence)} exceeds ESM-2 context limit 1022. Truncating.")
        sequence = sequence[:1022]
    if len(sequence) > 500:
        print(f"Note: sequence {len(sequence)} aa is long. CPU scoring may take time.")
        if args.model == "650M" and not args.validate:
            print("Hint: use --model 35M for faster results on long sequences.")

    # --- Run ESM-2 scoring ---
    print(f"\n=== Scoring with ESM-2 {args.model} ===")
    predictions, model_name, device = run_scorer(sequence, args.model)

    # --- Save CSV ---
    csv_path = os.path.join(output_dir, "mutation_scores.csv")
    save_results(predictions, csv_path)

    # --- Generate heatmap ---
    heatmap_path = os.path.join(output_dir, "mutation_heatmap.png")
    generate_heatmap(predictions, sequence, heatmap_path)

    # --- ProteinGym validation ---
    metrics = None
    assay_name = None
    if args.validate or args.model == "35M":  # Auto-validate on 35M demo
        dms_df, assay_name = find_proteingym_assay(sequence, uniprot_id)
        if dms_df is not None:
            merged_df = merge_predictions_with_dms(predictions, dms_df)
            metrics = compute_validation_metrics(merged_df)
            corr_path = os.path.join(output_dir, "correlation_plot.png")
            generate_correlation_plot(merged_df, corr_path, metrics)
            merged_df.to_csv(os.path.join(output_dir, "merged_with_dms.csv"), index=False)
        else:
            print("No ProteinGym match found — skipping validation.")
            metrics = {
                "n_mutants": len(predictions),
                "spearman_r": "N/A", "spearman_p": "N/A",
                "pearson_r": "N/A", "pearson_p": "N/A",
                "kendall_tau": "N/A", "auc_roc": "N/A"
            }
            assay_name = "No match"

    # --- Generate report ---
    report_path = os.path.join(output_dir, "mutation_report.txt")
    generate_report(
        sequence=sequence,
        predictions=predictions,
        metrics=metrics,
        assay_name=assay_name,
        model_name=model_name,
        output_path=report_path
    )

    print(f"\n✅ Complete. All outputs in: {output_dir}/")
    print("Output files:")
    print("  mutation_scores.csv      — All L×19 mutants ranked by ESM-2 score")
    print("  mutation_heatmap.png    — Positional fitness landscape")
    if metrics and metrics.get("spearman_r") != "N/A":
        print("  correlation_plot.png    — ESM-2 vs. DMS experimental scatter")
        print("  merged_with_dms.csv    — Merged predictions + experimental DMS")
    print("  mutation_report.txt     — Summary with top hits and metrics")


if __name__ == "__main__":
    main()
