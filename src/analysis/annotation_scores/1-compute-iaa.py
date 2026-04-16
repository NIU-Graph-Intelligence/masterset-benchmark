"""
compute_iaa.py

Computes Inter-Annotator Agreement (IAA) for the human validation study.

Primary metrics (as reported in the paper):
  - Type I  (binary 0/1):      Fleiss' kappa
  - Type II (ordinal 1-5):     Krippendorff's alpha with ordinal weights

Supporting metrics (for appendix / robustness):
  - Pairwise Cohen's kappa (Type I)
  - Pairwise weighted Cohen's kappa, quadratic weights (Type II)
  - % exact agreement
  - % within +/-1 (Type II only)
  - Krippendorff's alpha nominal (Type I, for cross-check)
  - Label distributions per annotator

Outputs:
  - Console summary
  - JSON file with all metrics (for reproducibility)
  - Long-format CSV with merged annotations (for downstream LLM-vs-human)
"""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters


# ============================================================
# CONFIG
# ============================================================

OVERLAP_DIR = Path("data/annotations/overlap_sets")

ANNOTATORS = ["ratul", "dr_zhang", "tanish", "samikshya", "ibrahim", "shafayet"]

TYPE1_COL = "Baseline? (0/1)"
TYPE2_COL = "Relevance Score (1-5)"

# Dropdown string -> numeric mapping
TYPE1_MAP = {
    "Not a Baseline - 0": 0,
    "Baseline - 1": 1,
}
TYPE2_MAP = {
    "1 Background": 1,
    "2 Broad Area": 2,
    "3 Conceptually Related": 3,
    "4 Task or Component": 4,
    "5 Core Relevance": 5,
}

OUTPUT_JSON = Path("output/analysis/annotation_scores/iaa_results.json")
OUTPUT_MERGED_CSV = Path("output/analysis/annotation_scores/overlap_merged.csv")


# ============================================================
# Loading & normalisation
# ============================================================

def load_annotator(name: str) -> pd.DataFrame:
    path = OVERLAP_DIR / f"overlap_{name}.csv"
    df = pd.read_csv(path)
    # Map dropdown strings -> ints
    df[TYPE1_COL] = df[TYPE1_COL].map(TYPE1_MAP)
    df[TYPE2_COL] = df[TYPE2_COL].map(TYPE2_MAP)
    # Sanity: no unmapped values
    assert df[TYPE1_COL].notna().all(), f"{name}: unmapped Type I values"
    assert df[TYPE2_COL].notna().all(), f"{name}: unmapped Type II values"
    return df


def build_matrices(annotations: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames indexed by instance_id, columns = annotator names.
    One for Type I, one for Type II.
    """
    # Use the first annotator as the instance order reference
    first = annotations[ANNOTATORS[0]]
    instance_ids = first["instance_id"].tolist()

    t1 = pd.DataFrame(index=instance_ids)
    t2 = pd.DataFrame(index=instance_ids)

    for name in ANNOTATORS:
        df = annotations[name].set_index("instance_id")
        # Reindex to the reference order; any missing instance_ids become NaN
        t1[name] = df.loc[instance_ids, TYPE1_COL].values
        t2[name] = df.loc[instance_ids, TYPE2_COL].values

    # Sanity: every annotator must have labelled every instance
    assert t1.isna().sum().sum() == 0, f"Type I has missing labels:\n{t1.isna().sum()}"
    assert t2.isna().sum().sum() == 0, f"Type II has missing labels:\n{t2.isna().sum()}"

    return t1.astype(int), t2.astype(int)


# ============================================================
# Agreement metrics
# ============================================================

def compute_fleiss_kappa(matrix: pd.DataFrame, categories: list) -> float:
    """
    matrix: rows = instances, columns = annotators, values = category labels.
    statsmodels' fleiss_kappa needs a (n_items, n_categories) count matrix.
    """
    counts = np.zeros((len(matrix), len(categories)), dtype=int)
    for i, cat in enumerate(categories):
        counts[:, i] = (matrix == cat).sum(axis=1).values
    return fleiss_kappa(counts, method="fleiss")


def compute_krippendorff(matrix: pd.DataFrame, level: str) -> float:
    """
    krippendorff.alpha expects a (n_raters, n_items) array with numeric values.
    level: 'nominal', 'ordinal', or 'interval'.
    """
    data = matrix.T.values  # (n_raters, n_items)
    return krippendorff.alpha(reliability_data=data, level_of_measurement=level)


def percent_agreement(matrix: pd.DataFrame) -> float:
    """Proportion of instances where ALL annotators agree."""
    all_same = (matrix.nunique(axis=1) == 1).mean()
    return float(all_same)


def percent_within_one(matrix: pd.DataFrame) -> float:
    """For ordinal: proportion of instances where max - min <= 1 across all annotators."""
    spread = matrix.max(axis=1) - matrix.min(axis=1)
    return float((spread <= 1).mean())


def pairwise_kappas(matrix: pd.DataFrame, weights: str | None = None) -> dict:
    """
    Returns {(a, b): kappa} for all unique pairs.
    weights: None for unweighted (Cohen's kappa), 'quadratic' for weighted.
    """
    result = {}
    for a, b in itertools.combinations(matrix.columns, 2):
        k = cohen_kappa_score(matrix[a], matrix[b], weights=weights)
        result[f"{a}__vs__{b}"] = float(k)
    return result


def summary_stats(pairwise: dict) -> dict:
    vals = list(pairwise.values())
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def label_distribution(matrix: pd.DataFrame) -> dict:
    """Distribution of labels per annotator."""
    return {
        ann: matrix[ann].value_counts().sort_index().to_dict()
        for ann in matrix.columns
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Loading annotations...")
    print("=" * 70)
    annotations = {}
    for name in ANNOTATORS:
        df = load_annotator(name)
        annotations[name] = df
        print(f"  {name:12s}: {len(df)} instances")

    t1, t2 = build_matrices(annotations)
    n_instances = len(t1)
    n_annotators = len(ANNOTATORS)
    print(f"\nMatrix shape: {n_instances} instances x {n_annotators} annotators")

    # =====================================================
    # TYPE I (binary 0/1)
    # =====================================================
    print("\n" + "=" * 70)
    print("TYPE I - Baseline classification (binary)")
    print("=" * 70)

    t1_fleiss = compute_fleiss_kappa(t1, categories=[0, 1])
    t1_alpha_nom = compute_krippendorff(t1, level="nominal")
    t1_pct_agree = percent_agreement(t1)
    t1_pairwise = pairwise_kappas(t1, weights=None)
    t1_pairwise_stats = summary_stats(t1_pairwise)
    t1_dist = label_distribution(t1)

    print(f"  Fleiss' kappa:             {t1_fleiss:.4f}")
    print(f"  Krippendorff's alpha:      {t1_alpha_nom:.4f}  (nominal)")
    print(f"  % exact agreement (all 6): {t1_pct_agree * 100:.2f}%")
    print(f"  Pairwise Cohen's kappa:    mean={t1_pairwise_stats['mean']:.4f}, "
          f"median={t1_pairwise_stats['median']:.4f}, "
          f"min={t1_pairwise_stats['min']:.4f}, "
          f"max={t1_pairwise_stats['max']:.4f}")
    print(f"\n  Label distribution (Type I):")
    for ann, dist in t1_dist.items():
        total = sum(dist.values())
        print(f"    {ann:12s}: 0={dist.get(0, 0):3d}, 1={dist.get(1, 0):3d}  "
              f"(baseline rate: {dist.get(1, 0)/total:.2%})")

    # =====================================================
    # TYPE II (ordinal 1-5)
    # =====================================================
    print("\n" + "=" * 70)
    print("TYPE II - Relevance score (ordinal 1-5)")
    print("=" * 70)

    t2_alpha_ord = compute_krippendorff(t2, level="ordinal")
    t2_alpha_int = compute_krippendorff(t2, level="interval")
    t2_fleiss = compute_fleiss_kappa(t2, categories=[1, 2, 3, 4, 5])
    t2_pct_exact = percent_agreement(t2)
    t2_pct_within1 = percent_within_one(t2)
    t2_pairwise = pairwise_kappas(t2, weights="quadratic")
    t2_pairwise_stats = summary_stats(t2_pairwise)
    t2_dist = label_distribution(t2)

    print(f"  Krippendorff's alpha (ordinal):   {t2_alpha_ord:.4f}   <-- primary")
    print(f"  Krippendorff's alpha (interval):  {t2_alpha_int:.4f}")
    print(f"  Fleiss' kappa (nominal):          {t2_fleiss:.4f}   (cross-check)")
    print(f"  % exact agreement (all 6):        {t2_pct_exact * 100:.2f}%")
    print(f"  % within +/-1 (max-min across 6): {t2_pct_within1 * 100:.2f}%")
    print(f"  Pairwise weighted kappa (quad):   mean={t2_pairwise_stats['mean']:.4f}, "
          f"median={t2_pairwise_stats['median']:.4f}, "
          f"min={t2_pairwise_stats['min']:.4f}, "
          f"max={t2_pairwise_stats['max']:.4f}")
    print(f"\n  Label distribution (Type II):")
    for ann, dist in t2_dist.items():
        row = "  ".join(f"{k}:{dist.get(k, 0):3d}" for k in [1, 2, 3, 4, 5])
        mean_score = t2[ann].mean()
        print(f"    {ann:12s}: {row}   (mean: {mean_score:.2f})")

    # =====================================================
    # Save JSON
    # =====================================================
    results = {
        "n_instances": int(n_instances),
        "n_annotators": int(n_annotators),
        "annotators": ANNOTATORS,
        "type_1": {
            "fleiss_kappa": float(t1_fleiss),
            "krippendorff_alpha_nominal": float(t1_alpha_nom),
            "pct_exact_agreement": float(t1_pct_agree),
            "pairwise_cohen_kappa": t1_pairwise,
            "pairwise_summary": t1_pairwise_stats,
            "label_distribution": {ann: {str(k): int(v) for k, v in d.items()}
                                   for ann, d in t1_dist.items()},
        },
        "type_2": {
            "krippendorff_alpha_ordinal": float(t2_alpha_ord),
            "krippendorff_alpha_interval": float(t2_alpha_int),
            "fleiss_kappa_nominal": float(t2_fleiss),
            "pct_exact_agreement": float(t2_pct_exact),
            "pct_within_one": float(t2_pct_within1),
            "pairwise_weighted_kappa_quadratic": t2_pairwise,
            "pairwise_summary": t2_pairwise_stats,
            "label_distribution": {ann: {str(k): int(v) for k, v in d.items()}
                                   for ann, d in t2_dist.items()},
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {OUTPUT_JSON}")

    # =====================================================
    # Save merged long CSV for downstream steps (LLM-vs-human)
    # =====================================================
    first = annotations[ANNOTATORS[0]]
    merged = first[["instance_id", "paper_id", "venue", "year",
                    "Current Paper", "Ref Paper", "Contexts"]].copy()
    for ann in ANNOTATORS:
        merged[f"type1_{ann}"] = t1[ann].values
        merged[f"type2_{ann}"] = t2[ann].values

    # Majority vote / median (adjudicated gold)
    merged["type1_majority"] = t1.mode(axis=1)[0].values.astype(int)
    merged["type2_median"] = t2.median(axis=1).values

    merged.to_csv(OUTPUT_MERGED_CSV, index=False)
    print(f"Saved merged: {OUTPUT_MERGED_CSV}")

    # =====================================================
    # LaTeX-ready summary
    # =====================================================
    print("\n" + "=" * 70)
    print("FOR THE PAPER (Table \\ref{tab:agreement})")
    print("=" * 70)
    print(f"  Type I  inter-annotator (Fleiss' kappa):         {t1_fleiss:.3f}")
    print(f"  Type II inter-annotator (Krippendorff ordinal):  {t2_alpha_ord:.3f}")


if __name__ == "__main__":
    main()