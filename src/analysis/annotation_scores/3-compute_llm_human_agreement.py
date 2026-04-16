"""
compute_llm_human_agreement.py

Compares LLM-generated labels (Gemini 2.5 Flash, and optionally other models)
against the human-adjudicated gold standard produced by build_gold_standard.py.

Primary metrics (Table 2 of the paper):
  Type I  (binary):  accuracy
  Type II (ordinal): mean absolute error (MAE)

Supporting metrics (for appendix):
  Type I:   precision, recall, F1, Cohen's kappa, confusion matrix
  Type II:  MAE, RMSE, Pearson, Spearman, quadratic-weighted Cohen's kappa,
            exact-match accuracy, within-+/-1 accuracy, confusion matrix

Coercion rules
--------------
Type 1 (lenient):
  Any value that cannot be parsed to 0 or 1 (including NaN, out-of-range
  numbers, un-parseable strings, missing field) is coerced to 0.
  Strings like "1.0", "0.0", "1", "0" are accepted. Type 1 rows are NEVER
  excluded from metrics.

Type 2 (strict):
  Only values that cleanly parse to an integer in {1, 2, 3, 4, 5} are valid.
  Anything else (NaN, out-of-range, un-parseable, missing field) -> None,
  and that row is excluded from Type 2 metrics only. The corresponding
  Type 1 label is still used for Type 1 metrics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, cohen_kappa_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
)


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path("/home/ratul/masterset-recommendation/")
GOLD_PATH = Path("output/analysis/annotation_scores/human_gold.csv")

# Each entry: (llm_name, parquet_path, type1_col, type2_col)
LLM_SOURCES = [
    ("gemini_2.5_flash",
     ROOT_DIR / "data/train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet",
     "type_1_output", "type_2_output"),
    # ("mistral_small", ROOT_DIR / "data/train_eval_set/vX.X/<path>.parquet",
    #  "type_1_output", "type_2_output"),
]

OUTPUT_DIR = Path("output/analysis/annotation_scores")
OUTPUT_RESULTS_JSON = OUTPUT_DIR / "llm_vs_human_results.json"
OUTPUT_COMBINED_CSV = OUTPUT_DIR / "llm_vs_human_combined.csv"


# ============================================================
# Coercion helpers
# ============================================================

def _is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _coerce_type1(x) -> int:
    """
    Lenient coercion to {0, 1}. Anything that cannot be parsed cleanly into
    0 or 1 (NaN, out-of-range, un-parseable strings, etc.) returns 0.
    Accepts int, float, numpy numerics, strings like '1', '1.0', '0', '0.0'.
    Never returns None.
    """
    if _is_nan(x):
        return 0
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return 0
    return v if v in (0, 1) else 0


def _coerce_type2(x):
    """
    Strict coercion to {1, 2, 3, 4, 5}. Anything else -> None.
    Accepts int, float, numpy numerics, strings like '3', '3.0'.
    """
    if _is_nan(x):
        return None
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 5 else None


# ============================================================
# LLM label lookup
# ============================================================

def build_llm_label_map(parquet_path: Path, type1_col: str, type2_col: str):
    """
    Returns {(paper_id, target): (type1, type2)} where:
      - type1 is always 0 or 1 (invalid / missing -> 0, lenient)
      - type2 is 1..5 or None (invalid / missing -> None, strict)
    """
    print(f"  Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} papers in dataframe")

    lookup: dict[tuple[str, str], tuple[int, int | None]] = {}
    t1_coerced_to_default = 0
    t2_invalid = 0

    for _, row in df.iterrows():
        paper_id = row["paper_id"]
        refs = row["references"]
        if isinstance(refs, str):
            refs = json.loads(refs)
        if not refs:
            continue
        for ref in refs:
            target = ref.get("target")
            if target is None:
                continue
            raw_t1 = ref.get(type1_col)
            raw_t2 = ref.get(type2_col)

            t1_val = _coerce_type1(raw_t1)
            # Count only "surprising" coercions (i.e. value was present and
            # looked numeric-ish but still failed the 0/1 check)
            if not _is_nan(raw_t1):
                try:
                    if int(float(raw_t1)) not in (0, 1):
                        t1_coerced_to_default += 1
                except (TypeError, ValueError):
                    t1_coerced_to_default += 1

            t2_val = _coerce_type2(raw_t2)
            if t2_val is None and not _is_nan(raw_t2):
                t2_invalid += 1

            lookup[(paper_id, target)] = (t1_val, t2_val)

    print(f"  {len(lookup)} (paper_id, target) entries indexed")
    if t1_coerced_to_default:
        print(f"  [INFO] Type 1: {t1_coerced_to_default} out-of-range/invalid "
              f"values coerced to 0 (lenient)")
    if t2_invalid:
        print(f"  [INFO] Type 2: {t2_invalid} out-of-range/invalid values "
              f"marked NaN (excluded from T2 metrics only)")
    return lookup


# ============================================================
# Metric computation
# ============================================================

def compute_type1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "precision_pos": float(p),
        "recall_pos": float(r),
        "f1_pos": float(f1),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm,
        "support_gold": {"0": int((y_true == 0).sum()),
                         "1": int((y_true == 1).sum())},
        "support_pred": {"0": int((y_pred == 0).sum()),
                         "1": int((y_pred == 1).sum())},
    }


def compute_type2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    w_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    exact = float((y_true == y_pred).mean())
    within1 = float((np.abs(y_true - y_pred) <= 1).mean())
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5]).tolist()
    return {
        "n": int(len(y_true)),
        "mae": float(mae),
        "rmse": rmse,
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "weighted_kappa_quadratic": float(w_kappa),
        "exact_match_accuracy": exact,
        "within_1_accuracy": within1,
        "confusion_matrix": cm,
        "support_gold": {str(k): int((y_true == k).sum()) for k in range(1, 6)},
        "support_pred": {str(k): int((y_pred == k).sum()) for k in range(1, 6)},
    }


def pretty_print_results(name: str, t1: dict, t2: dict):
    print(f"\n{'-' * 70}")
    print(f"{name}")
    print(f"{'-' * 70}")
    print("Type I (binary baseline classification):")
    print(f"  N                  = {t1['n']}")
    print(f"  Accuracy           = {t1['accuracy']:.4f}   <-- headline")
    print(f"  F1 (positive=1)    = {t1['f1_pos']:.4f}")
    print(f"  Precision / Recall = {t1['precision_pos']:.4f} / {t1['recall_pos']:.4f}")
    print(f"  Macro F1           = {t1['f1_macro']:.4f}")
    print(f"  Cohen's kappa      = {t1['cohen_kappa']:.4f}")
    print(f"  Confusion matrix (rows=gold, cols=pred):")
    print(f"    [[TN {t1['confusion_matrix'][0][0]:4d}  FP {t1['confusion_matrix'][0][1]:4d}]")
    print(f"     [FN {t1['confusion_matrix'][1][0]:4d}  TP {t1['confusion_matrix'][1][1]:4d}]]")
    print()
    print("Type II (ordinal 1-5 relevance):")
    print(f"  N                     = {t2['n']}")
    print(f"  MAE                   = {t2['mae']:.4f}   <-- headline")
    print(f"  RMSE                  = {t2['rmse']:.4f}")
    print(f"  Pearson r             = {t2['pearson_r']:.4f}")
    print(f"  Spearman rho          = {t2['spearman_r']:.4f}")
    print(f"  Weighted kappa (quad) = {t2['weighted_kappa_quadratic']:.4f}")
    print(f"  Exact match           = {t2['exact_match_accuracy']:.4f}")
    print(f"  Within +/-1           = {t2['within_1_accuracy']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("LLM vs Human Agreement")
    print("=" * 70)

    print(f"\nLoading gold standard from {GOLD_PATH}")
    gold = pd.read_csv(GOLD_PATH)
    print(f"  {len(gold)} gold rows")

    all_results = {}
    combined_rows = []

    for llm_name, parquet_path, t1_col, t2_col in LLM_SOURCES:
        print(f"\n{'=' * 70}")
        print(f"Model: {llm_name}")
        print("=" * 70)

        llm_map = build_llm_label_map(parquet_path, t1_col, t2_col)

        # Attach predictions to gold.
        # If (paper_id, target) not found in the parquet at all:
        #   - Type 1 defaults to 0 (lenient; never excluded)
        #   - Type 2 defaults to None (strict; excluded from T2 metrics)
        g = gold.copy()
        g[f"{llm_name}_t1"] = g.apply(
            lambda r: llm_map.get((r["paper_id"], r["target"]), (0, None))[0],
            axis=1,
        )
        g[f"{llm_name}_t2"] = g.apply(
            lambda r: llm_map.get((r["paper_id"], r["target"]), (0, None))[1],
            axis=1,
        )

        # Coverage reporting
        not_in_parquet = g.apply(
            lambda r: (r["paper_id"], r["target"]) not in llm_map, axis=1
        ).sum()
        excluded_t2 = g[f"{llm_name}_t2"].isna().sum()
        if not_in_parquet:
            print(f"  [WARN] {not_in_parquet} gold rows not found in {llm_name} "
                  f"parquet (T1 defaulted to 0; T2 excluded)")
        if excluded_t2:
            print(f"  [INFO] {excluded_t2} rows excluded from T2 metrics "
                  f"(missing or invalid T2 value)")

        # ---- Type I: NEVER excluded. Gold NaN would be the only reason to drop,
        # but type1_gold should be 0/1 in the gold standard. ----
        t1_mask = g["type1_gold"].notna()
        y_true_t1 = g.loc[t1_mask, "type1_gold"].astype(int).values
        y_pred_t1 = g.loc[t1_mask, f"{llm_name}_t1"].astype(int).values

        # ---- Type II: exclude rows where LLM t2 is None or gold t2 is NaN. ----
        t2_mask = g["type2_gold"].notna() & g[f"{llm_name}_t2"].notna()
        y_true_t2 = g.loc[t2_mask, "type2_gold"].astype(int).values
        y_pred_t2 = g.loc[t2_mask, f"{llm_name}_t2"].astype(int).values

        t1_metrics = compute_type1_metrics(y_true_t1, y_pred_t1)
        t2_metrics = compute_type2_metrics(y_true_t2, y_pred_t2)
        pretty_print_results(llm_name, t1_metrics, t2_metrics)

        all_results[llm_name] = {
            "parquet": str(parquet_path),
            "type_1": t1_metrics,
            "type_2": t2_metrics,
            "coverage": {
                "gold_rows_total": int(len(g)),
                "not_found_in_parquet": int(not_in_parquet),
                "t2_excluded": int(excluded_t2),
            },
        }

        merged = g[["instance_id", "paper_id", "target", "venue", "year",
                    "block", "type1_gold", "type2_gold",
                    f"{llm_name}_t1", f"{llm_name}_t2"]].copy()
        combined_rows.append(merged)

    # ---- Paper-ready summary ----
    print("\n" + "=" * 70)
    print("FOR THE PAPER (Table \\ref{tab:agreement}, LLM--human column)")
    print("=" * 70)
    for name, res in all_results.items():
        print(f"  {name}:")
        print(f"     Type I  accuracy = {res['type_1']['accuracy'] * 100:.1f}%")
        print(f"     Type II MAE      = {res['type_2']['mae']:.3f}")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_RESULTS_JSON}")

    base = combined_rows[0][["instance_id", "paper_id", "target", "venue",
                             "year", "block", "type1_gold", "type2_gold"]]
    for merged in combined_rows:
        llm_cols = [c for c in merged.columns if c not in base.columns]
        base = base.merge(merged[["instance_id"] + llm_cols], on="instance_id", how="left")
    base.to_csv(OUTPUT_COMBINED_CSV, index=False)
    print(f"Saved: {OUTPUT_COMBINED_CSV}")


if __name__ == "__main__":
    main()