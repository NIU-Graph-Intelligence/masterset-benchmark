"""
compute_iaa.py

Computes Inter-Annotator Agreement (IAA) across the full human validation
set of 510 citation instances, where each instance is independently labelled
by either 2 or 6 annotators.

Design notes
------------
Because the number of raters varies per item (2 vs 6), Fleiss' kappa is no
longer applicable as a single headline metric -- it assumes a fixed number of
raters. Krippendorff's alpha is the natural replacement: it handles missing
data and variable rater counts per item natively. We therefore use alpha as
the primary IAA statistic on the combined 510-instance set.

Primary metrics (reported in Table \\ref{tab:agreement}):
  - Type I  (binary 0/1):      Krippendorff's alpha (nominal)
  - Type II (ordinal 1-5):     Krippendorff's alpha (ordinal weights)

Supporting metrics (for appendix):
  - % exact agreement per item (unanimous across the item's raters)
  - % within +/-1 across all raters (Type II only)
  - Pairwise Cohen's kappa (Type I) and quadratic-weighted kappa (Type II),
    computed on each pair's co-rated subset with per-pair n reported
  - Per-annotator label distributions

Outputs:
  - Console summary
  - iaa_results.json              (all metrics, for reproducibility)
  - all_annotations_long.csv      (one row per (instance, annotator) with
                                   numeric labels -- used by downstream steps)
"""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score


# ============================================================
# CONFIG
# ============================================================

OVERLAP_DIR  = Path("data/annotations/overlap_sets")
DISJOINT_DIR = Path("data/annotations/disjoint_sets")

# All six annotators; each appears in the overlap set AND in exactly one
# disjoint block.
OVERLAP_ANNOTATORS = ["ratul", "dr_zhang", "tanish", "samikshya", "ibrahim", "shafayet"]

# (block_name, [annotator, annotator]) -- the two raters who labelled the block.
# NB: the file on disk uses 'drzhang' (no underscore) in the block/filename,
# but the annotator's canonical name (column and rater id) is 'dr_zhang'.
DISJOINT_BLOCKS = [
    ("drzhang",   [("drzhang",   "dr_zhang"),  ("ratul",     "ratul")]),
    ("samikshya", [("samikshya", "samikshya"), ("tanish",    "tanish")]),
    ("shafayet",  [("shafayet",  "shafayet"),  ("ibrahim",   "ibrahim")]),
]

TYPE1_COL = "Baseline? (0/1)"
TYPE2_COL = "Relevance Score (1-5)"
TYPE1_MAP = {"Not a Baseline - 0": 0, "Baseline - 1": 1}
TYPE2_MAP = {
    "1 Background": 1, "2 Broad Area": 2, "3 Conceptually Related": 3,
    "4 Task or Component": 4, "5 Core Relevance": 5,
}

OUTPUT_JSON     = Path("output/analysis/annotation_scores/iaa_results.json")
OUTPUT_LONG_CSV = Path("output/analysis/annotation_scores/all_annotations_long.csv")


# ============================================================
# Loading
# ============================================================

def load_and_normalise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TYPE1_COL] = df[TYPE1_COL].map(TYPE1_MAP)
    df[TYPE2_COL] = df[TYPE2_COL].map(TYPE2_MAP)
    if df[TYPE1_COL].isna().any() or df[TYPE2_COL].isna().any():
        raise ValueError(f"{path.name}: unmapped labels. Check the dropdown strings.")
    return df


def load_all_annotations() -> pd.DataFrame:
    """
    Returns a long-format DataFrame with columns:
        instance_id, block, annotator, type1, type2
    One row per (instance, annotator). Expected total rows:
        120 overlap * 6 + 130 * 3 blocks * 2 = 720 + 780 = 1500
    corresponding to 510 unique instances, each with 2 or 6 labels.
    """
    long_rows = []

    # -- overlap (120 x 6) --
    for ann in OVERLAP_ANNOTATORS:
        path = OVERLAP_DIR / f"overlap_{ann}.csv"
        df = load_and_normalise(path)
        for _, r in df.iterrows():
            long_rows.append({
                "instance_id": r["instance_id"],
                "block":       "overlap",
                "annotator":   ann,
                "type1":       int(r[TYPE1_COL]),
                "type2":       int(r[TYPE2_COL]),
            })

    # -- disjoint blocks (130 x 2, three of them) --
    for block_name, raters in DISJOINT_BLOCKS:
        for file_tag, rater_name in raters:
            path = DISJOINT_DIR / f"set_{block_name}_by_{file_tag}.csv"
            df = load_and_normalise(path)
            for _, r in df.iterrows():
                long_rows.append({
                    "instance_id": r["instance_id"],
                    "block":       block_name,
                    "annotator":   rater_name,
                    "type1":       int(r[TYPE1_COL]),
                    "type2":       int(r[TYPE2_COL]),
                })

    out = pd.DataFrame(long_rows)
    return out


# ============================================================
# Reliability-data construction for Krippendorff
# ============================================================

def build_reliability_matrix(long: pd.DataFrame, label_col: str,
                             all_annotators: list[str]) -> np.ndarray:
    """
    Returns a (n_raters, n_items) numpy array with NaN for missing
    (rater, item) pairs -- the exact format krippendorff.alpha expects.
    """
    items = sorted(long["instance_id"].unique())
    item_idx = {iid: j for j, iid in enumerate(items)}
    rater_idx = {r: i for i, r in enumerate(all_annotators)}

    mat = np.full((len(all_annotators), len(items)), np.nan, dtype=float)
    for _, r in long.iterrows():
        i = rater_idx[r["annotator"]]
        j = item_idx[r["instance_id"]]
        mat[i, j] = float(r[label_col])
    return mat, items


# ============================================================
# Supporting statistics
# ============================================================

def per_item_exact_agreement(long: pd.DataFrame, label_col: str) -> float:
    """Proportion of items where every rater who labelled the item agreed."""
    g = long.groupby("instance_id")[label_col].nunique()
    return float((g == 1).mean())


def per_item_within_one(long: pd.DataFrame, label_col: str) -> float:
    """Proportion of items where max - min across raters is <= 1 (Type II only)."""
    g = long.groupby("instance_id")[label_col].agg(lambda s: s.max() - s.min())
    return float((g <= 1).mean())


def pairwise_kappa(long: pd.DataFrame, label_col: str,
                   weights: str | None) -> dict:
    """
    Compute Cohen's kappa for every annotator pair that co-rated at least
    MIN_PAIR_N items. Returns {pair_key: {"kappa": ..., "n": ...}}.
    """
    MIN_PAIR_N = 10  # don't compute on tiny samples; shouldn't trigger here

    raters = sorted(long["annotator"].unique())
    wide = long.pivot(index="instance_id", columns="annotator", values=label_col)

    out = {}
    for a, b in itertools.combinations(raters, 2):
        mask = wide[a].notna() & wide[b].notna()
        n = int(mask.sum())
        if n < MIN_PAIR_N:
            continue
        ya = wide.loc[mask, a].astype(int)
        yb = wide.loc[mask, b].astype(int)
        k = cohen_kappa_score(ya, yb, weights=weights)
        out[f"{a}__vs__{b}"] = {"kappa": float(k), "n": n}
    return out


def pair_summary(pairwise: dict) -> dict:
    vals = [v["kappa"] for v in pairwise.values()]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None,
                "std": None, "n_pairs": 0}
    return {
        "mean":    float(np.mean(vals)),
        "median":  float(np.median(vals)),
        "min":     float(np.min(vals)),
        "max":     float(np.max(vals)),
        "std":     float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n_pairs": len(vals),
    }


def label_distribution(long: pd.DataFrame, label_col: str) -> dict:
    out = {}
    for ann, sub in long.groupby("annotator"):
        vc = sub[label_col].value_counts().sort_index().to_dict()
        out[ann] = {int(k): int(v) for k, v in vc.items()}
    return out


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Loading annotations (overlap + three disjoint blocks) ...")
    print("=" * 72)
    long = load_all_annotations()

    # Sanity checks
    n_instances = long["instance_id"].nunique()
    n_rows      = len(long)
    raters_per_item = long.groupby("instance_id")["annotator"].nunique()
    print(f"  Total rows (instance-rater pairs): {n_rows}")
    print(f"  Unique instances:                  {n_instances}")
    print(f"  Raters per instance:               "
          f"{dict(raters_per_item.value_counts().sort_index())}")
    print(f"  Annotators:                        {sorted(long['annotator'].unique())}")

    assert n_instances == 510, f"Expected 510 instances, got {n_instances}"
    assert set(raters_per_item.unique()) == {2, 6}, \
        f"Expected {{2, 6}} raters/instance, got {set(raters_per_item.unique())}"

    all_annotators = sorted(long["annotator"].unique())

    # =====================================================
    # Type I (binary)
    # =====================================================
    print("\n" + "=" * 72)
    print("TYPE I  ---  Baseline classification (binary)")
    print("=" * 72)

    t1_mat, _ = build_reliability_matrix(long, "type1", all_annotators)
    t1_alpha = krippendorff.alpha(reliability_data=t1_mat,
                                  level_of_measurement="nominal")
    t1_pct_exact = per_item_exact_agreement(long, "type1")
    t1_pairwise  = pairwise_kappa(long, "type1", weights=None)
    t1_pair_sum  = pair_summary(t1_pairwise)
    t1_dist      = label_distribution(long, "type1")

    print(f"  Krippendorff's alpha (nominal):      {t1_alpha:.4f}  <-- headline")
    print(f"  % exact agreement (per item):        {t1_pct_exact * 100:.2f}%")
    print(f"  Pairwise Cohen's kappa ({t1_pair_sum['n_pairs']} pairs): "
          f"mean={t1_pair_sum['mean']:.4f}, "
          f"median={t1_pair_sum['median']:.4f}, "
          f"min={t1_pair_sum['min']:.4f}, "
          f"max={t1_pair_sum['max']:.4f}")
    print(f"\n  Per-annotator label distribution (Type I):")
    for ann in all_annotators:
        d = t1_dist.get(ann, {})
        total = sum(d.values())
        rate1 = d.get(1, 0) / total if total else 0.0
        print(f"    {ann:10s}: 0={d.get(0, 0):4d}, 1={d.get(1, 0):4d}   "
              f"(n={total}, baseline rate={rate1:.2%})")

    # =====================================================
    # Type II (ordinal 1-5)
    # =====================================================
    print("\n" + "=" * 72)
    print("TYPE II  ---  Relevance score (ordinal 1-5)")
    print("=" * 72)

    t2_mat, _ = build_reliability_matrix(long, "type2", all_annotators)
    t2_alpha_ord = krippendorff.alpha(reliability_data=t2_mat,
                                      level_of_measurement="ordinal")
    t2_alpha_int = krippendorff.alpha(reliability_data=t2_mat,
                                      level_of_measurement="interval")
    t2_pct_exact   = per_item_exact_agreement(long, "type2")
    t2_pct_within1 = per_item_within_one(long, "type2")
    t2_pairwise    = pairwise_kappa(long, "type2", weights="quadratic")
    t2_pair_sum    = pair_summary(t2_pairwise)
    t2_dist        = label_distribution(long, "type2")

    print(f"  Krippendorff's alpha (ordinal):      {t2_alpha_ord:.4f}  <-- headline")
    print(f"  Krippendorff's alpha (interval):     {t2_alpha_int:.4f}")
    print(f"  % exact agreement (per item):        {t2_pct_exact * 100:.2f}%")
    print(f"  % within +/-1 across all raters:     {t2_pct_within1 * 100:.2f}%")
    print(f"  Pairwise weighted kappa ({t2_pair_sum['n_pairs']} pairs): "
          f"mean={t2_pair_sum['mean']:.4f}, "
          f"median={t2_pair_sum['median']:.4f}, "
          f"min={t2_pair_sum['min']:.4f}, "
          f"max={t2_pair_sum['max']:.4f}")
    print(f"\n  Per-annotator label distribution (Type II):")
    for ann in all_annotators:
        d = t2_dist.get(ann, {})
        total = sum(d.values())
        row = "  ".join(f"{k}:{d.get(k, 0):3d}" for k in [1, 2, 3, 4, 5])
        # Mean across this annotator's items
        sub = long[long["annotator"] == ann]["type2"]
        mean_s = sub.mean() if len(sub) else 0.0
        print(f"    {ann:10s}: {row}   (n={total}, mean={mean_s:.2f})")

    # =====================================================
    # Save outputs
    # =====================================================
    results = {
        "n_instances": int(n_instances),
        "n_rater_item_pairs": int(n_rows),
        "annotators": all_annotators,
        "raters_per_item_distribution": {int(k): int(v) for k, v
                                         in raters_per_item.value_counts().items()},
        "type_1": {
            "krippendorff_alpha_nominal": float(t1_alpha),
            "pct_exact_agreement": float(t1_pct_exact),
            "pairwise_cohen_kappa": t1_pairwise,
            "pairwise_summary": t1_pair_sum,
            "label_distribution": t1_dist,
        },
        "type_2": {
            "krippendorff_alpha_ordinal":  float(t2_alpha_ord),
            "krippendorff_alpha_interval": float(t2_alpha_int),
            "pct_exact_agreement": float(t2_pct_exact),
            "pct_within_one": float(t2_pct_within1),
            "pairwise_weighted_kappa_quadratic": t2_pairwise,
            "pairwise_summary": t2_pair_sum,
            "label_distribution": t2_dist,
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {OUTPUT_JSON}")

    long.to_csv(OUTPUT_LONG_CSV, index=False)
    print(f"Saved long CSV: {OUTPUT_LONG_CSV}  ({len(long)} rows)")

    # =====================================================
    # Paper-ready summary
    # =====================================================
    print("\n" + "=" * 72)
    print("FOR THE PAPER (Table \\ref{tab:agreement}, inter-annotator column)")
    print("=" * 72)
    print(f"  n instances = {n_instances} (2-6 annotators per instance)")
    print(f"  Type I  Krippendorff's alpha (nominal): {t1_alpha:.3f}")
    print(f"  Type II Krippendorff's alpha (ordinal): {t2_alpha_ord:.3f}")


if __name__ == "__main__":
    main()