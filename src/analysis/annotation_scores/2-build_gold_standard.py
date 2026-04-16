"""
build_gold_standard.py

Builds the adjudicated human gold standard from:
  - 120 six-way overlap annotations (majority vote for Type I, median for Type II)
  - 390 two-way disjoint annotations across 3 blocks, each annotated by 2 people
    (adjudication strategy configurable per TIE_BREAKER_STRATEGY)

For the overlap set, adjudication is unambiguous:
  Type I  -> majority vote across 6 annotators
  Type II -> median across 6 annotators (integer-valued since always an integer
             lies within the six scores for a 1-5 ordinal scale)

For disjoint 2-annotator blocks, when the two annotators agree, their shared
label is used. When they disagree, the tiebreaker strategy decides:

  'exclude'    -> the instance is dropped from the gold set
  'annotator1' -> use the label of the annotator named in the set name
                  (e.g., set_drzhang_*  -> whatever dr_zhang labelled)
  'annotator2' -> use the label of the other annotator
                  (e.g., set_drzhang_by_ratul -> whatever ratul labelled)

For Type II, we also support treating differences of 1 as 'agreement' via
DISAGREEMENT_TOLERANCE = 1 (the two raters are within 1 point; take their mean
rounded to the nearest integer). Set to 0 to treat any disagreement as a tie.

Outputs (in output/analysis/annotation_scores/):
  - human_gold.csv              : 510 rows with adjudicated type1_gold, type2_gold
  - gold_build_report.json      : per-block stats (agreement rates, excluded rows, etc.)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path("/home/ratul/masterset-recommendation/")
OVERLAP_DIR = Path("data/annotations/overlap_sets")
DISJOINT_DIR = Path("data/annotations/disjoint_sets")

ALL_SAMPLES_PATH = ROOT_DIR / "data/annotation/annotation_all_samples.xlsx"

OUTPUT_DIR = Path("output/analysis/annotation_scores")
OUTPUT_GOLD_CSV = OUTPUT_DIR / "human_gold.csv"
OUTPUT_REPORT = OUTPUT_DIR / "gold_build_report.json"

# --- Adjudication strategy for disjoint (2-annotator) blocks ---
# Options: 'exclude' | 'annotator1' | 'annotator2'
#   'annotator1' = the primary annotator (the one the set is named after)
#   'annotator2' = the other (secondary) annotator
TIE_BREAKER_STRATEGY = "annotator1"

# For Type II, treat |a - b| <= DISAGREEMENT_TOLERANCE as agreement, using
# round((a+b)/2) as the adjudicated label. Set to 0 for strict equality.
DISAGREEMENT_TOLERANCE_TYPE2 = 1

# --- Column / label mappings ---
TYPE1_COL = "Baseline? (0/1)"
TYPE2_COL = "Relevance Score (1-5)"
TYPE1_MAP = {"Not a Baseline - 0": 0, "Baseline - 1": 1}
TYPE2_MAP = {
    "1 Background": 1, "2 Broad Area": 2, "3 Conceptually Related": 3,
    "4 Task or Component": 4, "5 Core Relevance": 5,
}

# --- Overlap set: 6 annotators ---
OVERLAP_ANNOTATORS = ["ratul", "dr_zhang", "tanish", "samikshya", "ibrahim", "shafayet"]

# --- Disjoint blocks: (primary_annotator, secondary_annotator) pairs
#     primary = the person the set is named after (annotator1 in tiebreaker terms)
#     secondary = the other annotator (annotator2) ---
DISJOINT_BLOCKS = [
    # (block_name, primary, secondary)
    ("drzhang",   "drzhang",   "ratul"),
    ("samikshya", "samikshya", "tanish"),
    ("shafayet",  "shafayet",  "ibrahim"),
]

# ============================================================
# Helpers
# ============================================================

def load_and_normalise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TYPE1_COL] = df[TYPE1_COL].map(TYPE1_MAP)
    df[TYPE2_COL] = df[TYPE2_COL].map(TYPE2_MAP)
    missing1 = df[TYPE1_COL].isna().sum()
    missing2 = df[TYPE2_COL].isna().sum()
    if missing1 or missing2:
        raise ValueError(
            f"{path.name}: unmapped labels (Type I missing={missing1}, "
            f"Type II missing={missing2}). Check for typos in the dropdown."
        )
    return df


def load_all_samples_lookup() -> pd.DataFrame:
    """Return a DataFrame indexed by instance_id with columns we need to
    recover (paper_id, target, venue, year, title)."""
    df = pd.read_excel(ALL_SAMPLES_PATH)
    keep = ["instance_id", "paper_id", "venue", "year", "title", "target",
            "ref_paper_title", "cited_as"]
    df = df[keep].copy()
    df.set_index("instance_id", inplace=True)
    return df


def resolve_target(instance_id: str, all_samples: pd.DataFrame) -> dict | None:
    """
    Given an instance_id from an annotator CSV (e.g. 'OVL_0066' or 'DR._0001'),
    find the matching row in all_samples. The numeric suffix aligns; the prefix
    does not. We index by numeric suffix since generate_annotation_sheet.py uses
    the same 0-padded 4-digit counter across overlap / disjoint sets, but the
    counter is RESET per sheet. So we fall back to paper_id-based disambiguation.
    """
    # Try direct index first (works if instance_id matches 'ALL_NNNN')
    if instance_id in all_samples.index:
        return all_samples.loc[instance_id].to_dict()
    return None


def resolve_target_by_paper(paper_id: str, ref_paper: str,
                            all_samples: pd.DataFrame) -> dict | None:
    """Fallback: match by (paper_id, ref_paper_title) when instance_id differs.
    The 'Ref Paper' column in annotator CSVs has form '<cited_as>: <ref_title>'
    so we extract the title after the first ': '."""
    if ": " in ref_paper:
        ref_title = ref_paper.split(": ", 1)[1].strip()
    else:
        ref_title = ref_paper.strip()
    matches = all_samples[
        (all_samples["paper_id"] == paper_id)
        & (all_samples["ref_paper_title"] == ref_title)
    ]
    if len(matches) == 1:
        return matches.iloc[0].to_dict() | {"instance_id": matches.index[0]}
    return None


def attach_target(df: pd.DataFrame, all_samples: pd.DataFrame,
                  label: str) -> pd.DataFrame:
    """Add a 'target' column by looking up each row in all_samples."""
    targets = []
    unresolved = 0
    for _, row in df.iterrows():
        info = resolve_target_by_paper(row["paper_id"], row["Ref Paper"],
                                       all_samples)
        if info is None:
            unresolved += 1
            targets.append(None)
        else:
            targets.append(info["target"])
    if unresolved:
        print(f"  [WARN] {label}: {unresolved} rows could not be matched to "
              f"annotation_all_samples.xlsx")
    df = df.copy()
    df["target"] = targets
    return df


# ============================================================
# Adjudication
# ============================================================

def adjudicate_overlap(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    dfs: {annotator_name: DataFrame with columns including instance_id,
          paper_id, Ref Paper, TYPE1_COL, TYPE2_COL, target}
    Returns one row per instance with type1_gold, type2_gold.
    """
    # Line up on instance_id
    ref = dfs[OVERLAP_ANNOTATORS[0]].set_index("instance_id")
    order = ref.index.tolist()

    t1 = pd.DataFrame(index=order)
    t2 = pd.DataFrame(index=order)
    for name in OVERLAP_ANNOTATORS:
        d = dfs[name].set_index("instance_id").loc[order]
        t1[name] = d[TYPE1_COL].astype(int).values
        t2[name] = d[TYPE2_COL].astype(int).values

    # Type I: majority vote. With 6 raters, ties (3v3) possible; break via mean >= 0.5
    t1_mean = t1.mean(axis=1)
    type1_gold = (t1_mean >= 0.5).astype(int)

    # Type II: median across 6. Median of 6 integers on 1-5 scale -> may be x.5;
    # round to nearest even (banker's rounding avoids upward bias on ties).
    type2_gold = np.round(t2.median(axis=1)).astype(int)

    out = pd.DataFrame({
        "instance_id": order,
        "paper_id":    ref["paper_id"].values,
        "venue":       ref["venue"].values,
        "year":        ref["year"].values,
        "title":       ref["Current Paper"].values,
        "target":      ref["target"].values,
        "type1_gold":  type1_gold.values,
        "type2_gold":  type2_gold.values,
        "n_annotators": 6,
        "block":        "overlap",
        "adjudication": "majority_vote (T1) / median (T2)",
    })
    return out


def adjudicate_disjoint(df_primary: pd.DataFrame, df_secondary: pd.DataFrame,
                        block_name: str, primary: str, secondary: str,
                        strategy: str, tol_type2: int) -> tuple[pd.DataFrame, dict]:
    """
    Returns (gold_df, report_dict).
    Strategy is applied only to disagreements.
    """
    a = df_primary.set_index("instance_id")
    b = df_secondary.set_index("instance_id")
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        print(f"  [WARN] block '{block_name}': primary has {len(a)} instances, "
              f"secondary has {len(b)}, intersection {len(common)}")
    a = a.loc[common]; b = b.loc[common]

    rows = []
    stats = {
        "block": block_name, "primary": primary, "secondary": secondary,
        "strategy": strategy, "tol_type2": tol_type2,
        "total": len(common),
        "type1_agree": 0, "type1_disagree": 0, "type1_excluded": 0,
        "type2_agree": 0, "type2_within_tol": 0, "type2_disagree": 0,
        "type2_excluded": 0,
    }

    for iid in common:
        pa_t1 = int(a.at[iid, TYPE1_COL]); pb_t1 = int(b.at[iid, TYPE1_COL])
        pa_t2 = int(a.at[iid, TYPE2_COL]); pb_t2 = int(b.at[iid, TYPE2_COL])

        # ---- Type I adjudication ----
        if pa_t1 == pb_t1:
            t1_gold = pa_t1
            stats["type1_agree"] += 1
            t1_excl = False
        else:
            stats["type1_disagree"] += 1
            if strategy == "exclude":
                t1_gold = None
                stats["type1_excluded"] += 1
                t1_excl = True
            elif strategy == "annotator1":
                t1_gold = pa_t1; t1_excl = False
            elif strategy == "annotator2":
                t1_gold = pb_t1; t1_excl = False
            else:
                raise ValueError(f"Unknown TIE_BREAKER_STRATEGY: {strategy}")

        # ---- Type II adjudication ----
        diff = abs(pa_t2 - pb_t2)
        if diff == 0:
            t2_gold = pa_t2; stats["type2_agree"] += 1; t2_excl = False
        elif diff <= tol_type2:
            # Within tolerance -> take rounded mean
            t2_gold = int(round((pa_t2 + pb_t2) / 2))
            stats["type2_within_tol"] += 1
            t2_excl = False
        else:
            stats["type2_disagree"] += 1
            if strategy == "exclude":
                t2_gold = None
                stats["type2_excluded"] += 1
                t2_excl = True
            elif strategy == "annotator1":
                t2_gold = pa_t2; t2_excl = False
            elif strategy == "annotator2":
                t2_gold = pb_t2; t2_excl = False

        # Skip the row entirely only if BOTH labels got excluded
        if t1_gold is None and t2_gold is None:
            continue

        rows.append({
            "instance_id": iid,
            "paper_id":    a.at[iid, "paper_id"],
            "venue":       a.at[iid, "venue"],
            "year":        a.at[iid, "year"],
            "title":       a.at[iid, "Current Paper"],
            "target":      a.at[iid, "target"],
            "type1_gold":  t1_gold,
            "type2_gold":  t2_gold,
            "n_annotators": 2,
            "block":        block_name,
            "adjudication": f"primary={primary} secondary={secondary} strategy={strategy}",
        })

    return pd.DataFrame(rows), stats


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(f"Building gold standard   |   tiebreaker={TIE_BREAKER_STRATEGY}   |   "
          f"type2 tolerance={DISAGREEMENT_TOLERANCE_TYPE2}")
    print("=" * 70)

    print("\nLoading annotation_all_samples.xlsx ...")
    all_samples = load_all_samples_lookup()
    print(f"  {len(all_samples)} instances indexed")

    # --- Overlap ---
    print("\n[1/2] Loading overlap set (120 instances x 6 annotators) ...")
    overlap = {}
    for name in OVERLAP_ANNOTATORS:
        path = OVERLAP_DIR / f"overlap_{name}.csv"
        df = load_and_normalise(path)
        df = attach_target(df, all_samples, label=f"overlap/{name}")
        overlap[name] = df
        print(f"  {name:10s}: {len(df)} rows")

    overlap_gold = adjudicate_overlap(overlap)
    print(f"  -> overlap gold: {len(overlap_gold)} rows")

    # --- Disjoint blocks ---
    print("\n[2/2] Loading disjoint blocks (3 x 130 instances x 2 annotators) ...")
    disjoint_golds = []
    block_reports = []
    for block_name, primary, secondary in DISJOINT_BLOCKS:
        path_p = DISJOINT_DIR / f"set_{block_name}_by_{primary}.csv"
        path_s = DISJOINT_DIR / f"set_{block_name}_by_{secondary}.csv"
        df_p = load_and_normalise(path_p)
        df_s = load_and_normalise(path_s)
        df_p = attach_target(df_p, all_samples, label=f"{block_name}/{primary}")
        df_s = attach_target(df_s, all_samples, label=f"{block_name}/{secondary}")

        gold, stats = adjudicate_disjoint(
            df_p, df_s, block_name, primary, secondary,
            strategy=TIE_BREAKER_STRATEGY,
            tol_type2=DISAGREEMENT_TOLERANCE_TYPE2,
        )
        disjoint_golds.append(gold)
        block_reports.append(stats)
        print(f"  block '{block_name}': {stats['total']} instances | "
              f"T1 agree {stats['type1_agree']}, disagree {stats['type1_disagree']} "
              f"(excl {stats['type1_excluded']}) | "
              f"T2 exact {stats['type2_agree']}, within tol {stats['type2_within_tol']}, "
              f"disagree {stats['type2_disagree']} (excl {stats['type2_excluded']})")

    # --- Combine ---
    gold = pd.concat([overlap_gold] + disjoint_golds, ignore_index=True)
    print(f"\nTotal gold rows: {len(gold)}")
    print(f"  type1_gold missing: {gold['type1_gold'].isna().sum()}")
    print(f"  type2_gold missing: {gold['type2_gold'].isna().sum()}")

    # Cast to nullable int for clean CSV output
    gold["type1_gold"] = gold["type1_gold"].astype("Int64")
    gold["type2_gold"] = gold["type2_gold"].astype("Int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold.to_csv(OUTPUT_GOLD_CSV, index=False)
    print(f"\nSaved: {OUTPUT_GOLD_CSV}")

    # --- Distribution summary ---
    print("\n--- Gold label distributions ---")
    print(f"Type I : {dict(gold['type1_gold'].value_counts().sort_index())}")
    print(f"Type II: {dict(gold['type2_gold'].value_counts().sort_index())}")

    report = {
        "config": {
            "tiebreaker_strategy": TIE_BREAKER_STRATEGY,
            "type2_tolerance": DISAGREEMENT_TOLERANCE_TYPE2,
        },
        "overlap": {
            "n_instances": len(overlap_gold),
            "n_annotators": 6,
            "type1_dist": {str(k): int(v) for k, v in
                           overlap_gold["type1_gold"].value_counts().sort_index().items()},
            "type2_dist": {str(k): int(v) for k, v in
                           overlap_gold["type2_gold"].value_counts().sort_index().items()},
        },
        "disjoint_blocks": block_reports,
        "total_gold_rows": len(gold),
        "missing_type1": int(gold["type1_gold"].isna().sum()),
        "missing_type2": int(gold["type2_gold"].isna().sum()),
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
