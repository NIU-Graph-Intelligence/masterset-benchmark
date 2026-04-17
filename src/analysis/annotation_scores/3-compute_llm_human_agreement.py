"""
compute_llm_human_agreement.py

Compares LLM-generated labels (Gemini 2.5 Flash, and optionally other models)
against the adjudicated human gold standard produced by build_gold_standard.py.

Primary metrics (Table \\ref{tab:agreement}):
  Type I  (binary) : accuracy
  Type II (ordinal): quadratic-weighted Cohen's kappa     <-- headline

Why weighted kappa for Type II?
  Gold Type II values may be half-integers (e.g., 3.5 from 2-rater
  blocks where one annotator said 3 and the other said 4). Weighted
  kappa treats the scale ordinally, penalises disagreements by their
  squared distance (same weighting as the inter-annotator statistic
  reported for Type II), and is directly comparable to the IAA numbers
  in the same table. MAE, RMSE, Pearson r, Spearman rho, exact-match
  accuracy, and +/-1 tolerant accuracy are reported alongside it.

Note on scale of weighted kappa:
  Cohen's weighted kappa requires integer labels. We therefore compute
  it on the subset of rows whose gold Type II value is an integer
  (i.e. excluding the 2-rater half-integer cases). The non-integer
  gold rows still contribute to MAE, RMSE, correlations, and the
  +/-1 tolerant accuracy. The script reports both n's.

LLM coercion rules (unchanged from prior version):
  Type 1 (lenient): anything not cleanly 0 or 1 -> 0 (never excluded).
  Type 2 (strict) : anything not cleanly in {1..5} -> None (excluded
                    from T2 metrics only).
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

ROOT_DIR  = Path("/home/ratul/masterset-recommendation/")
GOLD_PATH = Path("output/analysis/annotation_scores/human_gold.csv")

# (llm_name, parquet_path, type1_column, type2_column)
LLM_SOURCES = [
    ("gemini_2.5_flash",
     ROOT_DIR / "data/train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet",
     "type_1_output", "type_2_output"),
]

OUTPUT_DIR           = Path("output/analysis/annotation_scores")
OUTPUT_RESULTS_JSON  = OUTPUT_DIR / "llm_vs_human_results.json"
OUTPUT_COMBINED_CSV  = OUTPUT_DIR / "llm_vs_human_combined.csv"


# ============================================================
# Coercion
# ============================================================

def _is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def _coerce_type1(x) -> int:
    if _is_nan(x):
        return 0
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return 0
    return v if v in (0, 1) else 0


def _coerce_type2(x):
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

def build_llm_label_map(parquet_path: Path, t1_col: str, t2_col: str):
    """{(paper_id, target): (type1_int, type2_int_or_None)}"""
    print(f"  Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} papers in dataframe")

    lookup: dict[tuple[str, str], tuple[int, int | None]] = {}
    t1_coerced = 0
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
            raw_t1 = ref.get(t1_col)
            raw_t2 = ref.get(t2_col)

            t1_val = _coerce_type1(raw_t1)
            if not _is_nan(raw_t1):
                try:
                    if int(float(raw_t1)) not in (0, 1):
                        t1_coerced += 1
                except (TypeError, ValueError):
                    t1_coerced += 1

            t2_val = _coerce_type2(raw_t2)
            if t2_val is None and not _is_nan(raw_t2):
                t2_invalid += 1

            lookup[(paper_id, target)] = (t1_val, t2_val)

    print(f"  {len(lookup)} (paper_id, target) entries indexed")
    if t1_coerced:
        print(f"  [INFO] T1: {t1_coerced} out-of-range/invalid coerced to 0")
    if t2_invalid:
        print(f"  [INFO] T2: {t2_invalid} out-of-range/invalid set to NaN")
    return lookup


# ============================================================
# Metrics
# ============================================================

def compute_type1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
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
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f1_m),
        "cohen_kappa": float(kappa),
        "confusion_matrix": cm,
        "support_gold": {"0": int((y_true == 0).sum()),
                         "1": int((y_true == 1).sum())},
        "support_pred": {"0": int((y_pred == 0).sum()),
                         "1": int((y_pred == 1).sum())},
    }


def compute_type2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true may contain half-integers (e.g. 3.5). Weighted kappa and
    exact-match accuracy are computed on the integer-gold subset only;
    all other metrics use the full set.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Metrics that tolerate half-integer gold
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    pearson_r, _  = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    within1 = float((np.abs(y_true - y_pred) <= 1.0).mean())

    # Metrics that require integer gold: use the integer-gold subset
    int_mask = np.isclose(y_true, np.round(y_true))
    y_true_i = y_true[int_mask].astype(int)
    y_pred_i = y_pred[int_mask].astype(int)

    if len(y_true_i):
        w_kappa = cohen_kappa_score(y_true_i, y_pred_i, weights="quadratic")
        exact   = float((y_true_i == y_pred_i).mean())
        cm      = confusion_matrix(y_true_i, y_pred_i, labels=[1, 2, 3, 4, 5]).tolist()
        support_gold_int = {str(k): int((y_true_i == k).sum()) for k in range(1, 6)}
        support_pred_int = {str(k): int((y_pred_i == k).sum()) for k in range(1, 6)}
    else:
        w_kappa = float("nan"); exact = float("nan"); cm = None
        support_gold_int = None; support_pred_int = None

    return {
        "n": int(len(y_true)),
        "n_integer_gold": int(int_mask.sum()),
        "n_halfinteger_gold": int((~int_mask).sum()),
        "weighted_kappa_quadratic": float(w_kappa),
        "mae": float(mae),
        "rmse": rmse,
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "exact_match_accuracy": float(exact) if not np.isnan(exact) else None,
        "within_1_accuracy": within1,
        "confusion_matrix_integer_gold": cm,
        "support_gold_integer": support_gold_int,
        "support_pred_integer": support_pred_int,
    }


def pretty_print(name: str, t1: dict, t2: dict):
    print(f"\n{'-' * 72}")
    print(f"{name}")
    print(f"{'-' * 72}")
    print("Type I (binary baseline classification):")
    print(f"  N                  = {t1['n']}")
    print(f"  Accuracy           = {t1['accuracy']:.4f}   <-- headline")
    print(f"  F1 (positive=1)    = {t1['f1_pos']:.4f}")
    print(f"  Precision / Recall = {t1['precision_pos']:.4f} / {t1['recall_pos']:.4f}")
    print(f"  Macro F1           = {t1['f1_macro']:.4f}")
    print(f"  Cohen's kappa      = {t1['cohen_kappa']:.4f}")
    print(f"  Confusion matrix (rows=gold, cols=pred):")
    cm = t1['confusion_matrix']
    print(f"    [[TN {cm[0][0]:4d}  FP {cm[0][1]:4d}]")
    print(f"     [FN {cm[1][0]:4d}  TP {cm[1][1]:4d}]]")

    print("\nType II (ordinal 1-5 relevance):")
    print(f"  N (all)                    = {t2['n']}")
    print(f"    of which integer-gold    = {t2['n_integer_gold']}")
    print(f"    of which half-integer    = {t2['n_halfinteger_gold']}")
    print(f"  Weighted kappa (quadratic) = {t2['weighted_kappa_quadratic']:.4f}   "
          f"<-- headline  (on integer-gold subset)")
    print(f"  MAE                        = {t2['mae']:.4f}")
    print(f"  RMSE                       = {t2['rmse']:.4f}")
    print(f"  Pearson r                  = {t2['pearson_r']:.4f}")
    print(f"  Spearman rho               = {t2['spearman_r']:.4f}")
    if t2['exact_match_accuracy'] is not None:
        print(f"  Exact match (integer-gold) = {t2['exact_match_accuracy']:.4f}")
    print(f"  Within +/-1 (full set)     = {t2['within_1_accuracy']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("LLM vs Human Agreement")
    print("=" * 72)

    print(f"\nLoading gold standard from {GOLD_PATH}")
    gold = pd.read_csv(GOLD_PATH)
    print(f"  {len(gold)} gold rows")
    t1_avail = gold["type1_gold"].notna().sum()
    t2_avail = gold["type2_gold"].notna().sum()
    print(f"  type1_gold available: {t1_avail} "
          f"(excluded from T1 metrics: {len(gold) - t1_avail})")
    print(f"  type2_gold available: {t2_avail} "
          f"(excluded from T2 metrics: {len(gold) - t2_avail})")

    all_results = {}
    combined_rows = []

    for llm_name, parquet_path, t1_col, t2_col in LLM_SOURCES:
        print(f"\n{'=' * 72}")
        print(f"Model: {llm_name}")
        print("=" * 72)

        llm_map = build_llm_label_map(parquet_path, t1_col, t2_col)

        g = gold.copy()
        g[f"{llm_name}_t1"] = g.apply(
            lambda r: llm_map.get((r["paper_id"], r["target"]), (0, None))[0],
            axis=1,
        )
        g[f"{llm_name}_t2"] = g.apply(
            lambda r: llm_map.get((r["paper_id"], r["target"]), (0, None))[1],
            axis=1,
        )

        not_in_parquet = g.apply(
            lambda r: (r["paper_id"], r["target"]) not in llm_map, axis=1
        ).sum()
        excluded_t2 = g[f"{llm_name}_t2"].isna().sum()
        if not_in_parquet:
            print(f"  [WARN] {not_in_parquet} gold rows not found in {llm_name} "
                  f"parquet (T1 defaulted to 0; T2 excluded)")
        if excluded_t2:
            print(f"  [INFO] {excluded_t2} rows excluded from T2 metrics "
                  f"(missing or invalid LLM T2 value)")

        # ---- Type I : excluded only when human gold is missing ----
        t1_mask = g["type1_gold"].notna()
        y_true_t1 = g.loc[t1_mask, "type1_gold"].astype(int).to_numpy()
        y_pred_t1 = g.loc[t1_mask, f"{llm_name}_t1"].astype(int).to_numpy()

        # ---- Type II : excluded if gold OR LLM is missing ----
        t2_mask = g["type2_gold"].notna() & g[f"{llm_name}_t2"].notna()
        y_true_t2 = g.loc[t2_mask, "type2_gold"].astype(float).to_numpy()
        y_pred_t2 = g.loc[t2_mask, f"{llm_name}_t2"].astype(int).to_numpy()

        t1_metrics = compute_type1_metrics(y_true_t1, y_pred_t1)
        t2_metrics = compute_type2_metrics(y_true_t2, y_pred_t2)
        pretty_print(llm_name, t1_metrics, t2_metrics)

        all_results[llm_name] = {
            "parquet": str(parquet_path),
            "type_1": t1_metrics,
            "type_2": t2_metrics,
            "coverage": {
                "gold_rows_total":      int(len(g)),
                "t1_gold_resolved":     int(t1_mask.sum()),
                "t2_gold_resolved":     int(g["type2_gold"].notna().sum()),
                "not_found_in_parquet": int(not_in_parquet),
                "t2_excluded_invalid":  int(excluded_t2),
                "t2_final_n":           int(t2_mask.sum()),
            },
        }

        merged = g[["instance_id", "paper_id", "target", "venue", "year",
                    "block", "n_annotators", "type1_gold", "type2_gold",
                    f"{llm_name}_t1", f"{llm_name}_t2"]].copy()
        combined_rows.append(merged)

    # ---- Paper-ready summary ----
    print("\n" + "=" * 72)
    print("FOR THE PAPER (Table \\ref{tab:agreement}, LLM--human column)")
    print("=" * 72)
    for name, res in all_results.items():
        print(f"  {name}:")
        print(f"     Type I  accuracy            = "
              f"{res['type_1']['accuracy'] * 100:.1f}%   (n={res['type_1']['n']})")
        print(f"     Type II weighted kappa      = "
              f"{res['type_2']['weighted_kappa_quadratic']:.3f}  "
              f"(n={res['type_2']['n_integer_gold']} integer-gold)")
        print(f"     Type II MAE                 = "
              f"{res['type_2']['mae']:.3f}   (n={res['type_2']['n']})")

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_RESULTS_JSON}")

    base_cols = ["instance_id", "paper_id", "target", "venue", "year",
                 "block", "n_annotators", "type1_gold", "type2_gold"]
    base = combined_rows[0][base_cols]
    for merged in combined_rows:
        llm_cols = [c for c in merged.columns if c not in base_cols]
        base = base.merge(merged[["instance_id"] + llm_cols],
                          on="instance_id", how="left")
    base.to_csv(OUTPUT_COMBINED_CSV, index=False)
    print(f"Saved: {OUTPUT_COMBINED_CSV}")


if __name__ == "__main__":
    main()