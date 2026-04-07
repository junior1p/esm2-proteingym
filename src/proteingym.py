"""ProteinGym DMS benchmark integration for ESM-2 validation.

ProteinGym contains 217+ DMS assays covering ~2.7M single-point and multi-site
mutations across 186+ proteins. Publicly available via AWS Open Data Registry.
Reference: Notin et al. (2023) ProteinGym: Large-Scale Benchmarks for Protein Fitness.
"""

import pandas as pd
from datasets import load_dataset
from typing import Optional, Tuple


def find_proteingym_assay(
    sequence: str,
    uniprot_id: str = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Find matching ProteinGym DMS assay for the given protein.
    
    Match by UniProt ID or by exact sequence match.
    Returns DMS DataFrame and assay_id if found, None otherwise.
    
    Uses AWS S3 public dataset — no API key required.
    """
    print("Loading ProteinGym DMS substitution benchmark...")
    
    # Load reference file to find matching assay
    ref_url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/references/DMS_substitution_benchmark.csv"
    ref_df = pd.read_csv(ref_url)
    
    # Try matching by UniProt ID
    if uniprot_id:
        matches = ref_df[ref_df["UniProt_ID"] == uniprot_id]
        if len(matches) > 0:
            assay_name = matches.iloc[0]["DMS_id"]
            print(f"Found ProteinGym assay: {assay_name}")
            # Load from AWS S3 (public dataset, no auth needed)
            s3_url = f"https://proteingym.s3.amazonaws.com/DMS_ProteinGym_substitutions/{assay_name}.csv"
            dms_df = pd.read_csv(s3_url)
            return dms_df, assay_name
    
    # Fallback: try HuggingFace dataset for sequence match
    try:
        dataset = load_dataset(
            "OATML-Markslab/ProteinGym_v1",
            name="DMS_substitutions",
            trust_remote_code=True
        )
        df = dataset.to_pandas()
        seq_matches = df[df["target_seq"] == sequence]
        if len(seq_matches) > 0:
            assay_id = seq_matches.iloc[0]["DMS_id"]
            print(f"Matched by sequence: {assay_id}")
            return seq_matches[["mutant", "DMS_score", "DMS_score_bin"]], assay_id
    except Exception as e:
        print(f"HuggingFace dataset access failed: {e}")
    
    print("No matching ProteinGym assay found. Will output predictions only.")
    return None, None


def merge_predictions_with_dms(
    predictions: list[dict],
    dms_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge ESM-2 scores with ProteinGym experimental DMS scores."""
    pred_df = pd.DataFrame(predictions)
    merged = pred_df.merge(
        dms_df[["mutant", "DMS_score", "DMS_score_bin"]],
        on="mutant",
        how="inner"
    )
    print(f"Merged {len(merged)} mutants with DMS experimental data")
    return merged


def compute_validation_metrics(merged_df: pd.DataFrame) -> dict:
    """Compute standard ProteinGym validation metrics.
    
    Primary metric: Spearman correlation (rank-based, robust to outliers).
    Context: ProteinGym top models achieve Spearman ~0.45–0.55.
    ESM-1v baseline: ~0.44; ESM-2 650M: typically ~0.44–0.50.
    """
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    
    valid = merged_df.dropna(subset=["esm2_score", "DMS_score"])
    if len(valid) < 10:
        print("Warning: fewer than 10 matched mutants. Validation results may be unreliable.")
    
    spearman_r, spearman_p = stats.spearmanr(valid["esm2_score"], valid["DMS_score"])
    pearson_r, pearson_p = stats.pearsonr(valid["esm2_score"], valid["DMS_score"])
    kendall_tau, kendall_p = stats.kendalltau(valid["esm2_score"], valid["DMS_score"])
    
    auc = None
    if "DMS_score_bin" in merged_df.columns:
        try:
            auc = roc_auc_score(
                merged_df.dropna()["DMS_score_bin"],
                merged_df.dropna()["esm2_score"]
            )
        except Exception:
            pass
    
    metrics = {
        "n_mutants": len(valid),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": spearman_p,
        "pearson_r": round(pearson_r, 4),
        "pearson_p": pearson_p,
        "kendall_tau": round(kendall_tau, 4),
        "auc_roc": round(auc, 4) if auc else "N/A"
    }
    
    print("\n=== Validation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    return metrics
