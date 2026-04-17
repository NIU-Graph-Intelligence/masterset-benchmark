"""
build_gold_standard.py

Builds the adjudicated human gold standard on all 510 instances using the
rules specified for this release.

Adjudication rules
------------------
Each instance is independently labelled by either 2 or 6 annotators. Gold
labels for Type I and Type II are computed INDEPENDENTLY -- an instance
excluded from one task can still contribute to the other.

Type I (binary):
  * 2 raters  -> gold = shared label if both agree; else exclude from T1.
  * 6 raters  -> gold = majority label if one label has strictly more votes
                 (i.e. >= 4 of 6); else exclude from T1 (3-3 tie).

Type II (ordinal 1-5):
  * 2 raters  -> gold = mean of the two labels if |a - b| <= 1 (preserves
                 the ordinal information; the mean may be a half-integer
                 such as 3.5); else exclude from T2.
  * 6 raters  -> gold = modal label if one value has strictly more votes
                 than any other; else exclude from T2 (tie).

No LLM labels are used anywhere in adjudication. The LLM is the system
under evaluation; using it to break human ties would be circular.

Inputs (relative to the project root):
  data/annotations/overlap_sets/overlap_<annotator>.csv            (x6)
  data/annotations/disjoint_sets/set_<block>_by_<annotator>.csv    (x6)
  data/annotation/annotation_all_samples.xlsx                      (for target)

Outputs (in output/analysis/annotation_scores/):
  human_gold.csv           : 510 rows; type1_gold/type2_gold are nullable ints
                             (or floats for T2, because 2-rater adjudicated
                             Type II may be a half-integer such as 3.5)
  gold_build_report.json   : per-block and overall adjudication statistics
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR     = Path("/home/ratul/masterset-recommendation/")
OVERLAP_DIR  = Path("data/annotations/overlap_sets")
DISJOINT_DIR = Path("data/annotations/disjoint_sets")

ALL_SAMPLES_PATH = ROOT_DIR / "data/annotation/annotation_all_samples.xlsx"

OUTPUT_DIR      = Path("output/analysis/annotation_scores")
OUTPUT_GOLD_CSV = OUTPUT_DIR / "human_gold.csv"
OUTPUT_REPORT   = OUTPUT_DIR / "gold_build_report.json"

TYPE1_COL = "Baseline? (0/1)"
TYPE2_COL = "Relevance Score (1-5)"
TYPE1_MAP = {"Not a Baseline - 0": 0, "Baseline - 1": 1}
TYPE2_MAP = {
    "1 Background": 1, "2 Broad Area": 2, "3 Conceptually Related": 3,
    "4 Task or Component": 4, "5 Core Relevance": 5,
}

OVERLAP_ANNOTATORS = ["ratul", "dr_zhang", "tanish", "samikshya", "ibrahim", "shafayet"]

# (block_name, file_tag_of_primary, file_tag_of_secondary)
# file_tag is what appears in "set_<block>_by_<tag>.csv"
DISJOINT_BLOCKS = [
    ("drzhang",   "drzhang",   "ratul"),
    ("samikshya", "samikshya", "tanish"),
    ("shafayet",  "shafayet",  "ibrahim"),
]


# ============================================================
# Loading
# ============================================================

def load_and_normalise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TYPE1_COL] = df[TYPE1_COL].map(TYPE1_MAP)
    df[TYPE2_COL] = df[TYPE2_COL].map(TYPE2_MAP)
    if df[TYPE1_COL].isna().any() or df[TYPE2_COL].isna().any():
        raise ValueError(f"{path.name}: unmapped labels found.")
    return df


def load_all_samples_lookup() -> pd.DataFrame:
    df = pd.read_excel(ALL_SAMPLES_PATH)
    keep = ["instance_id", "paper_id", "venue", "year", "title", "target",
            "ref_paper_title", "cited_as"]
    df = df[keep].copy()
    df.set_index("instance_id", inplace=True)
    return df


def resolve_target_by_paper(paper_id: str, ref_paper: str,
                            all_samples: pd.DataFrame) -> dict | None:
    """Match a row in annotation_all_samples.xlsx by (paper_id, ref_title)."""
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

def _majority_label(votes: list[int]) -> int | None:
    """
    Return the label with strictly the most votes, or None if there is a
    tie for the top position.
    """
    c = Counter(votes)
    top = c.most_common()
    if len(top) == 1:
        return top[0][0]
    if top[0][1] > top[1][1]:
        return top[0][0]
    return None  # tie for first place


def adjudicate_overlap(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """
    6-rater adjudication on the 120 overlap instances.

    Type I : majority (>=4/6) else excluded.
    Type II: modal label if unique else excluded.
    """
    ref = dfs[OVERLAP_ANNOTATORS[0]].set_index("instance_id")
    order = ref.index.tolist()

    t1_votes = {iid: [] for iid in order}
    t2_votes = {iid: [] for iid in order}
    for name in OVERLAP_ANNOTATORS:
        d = dfs[name].set_index("instance_id").loc[order]
        for iid in order:
            t1_votes[iid].append(int(d.at[iid, TYPE1_COL]))
            t2_votes[iid].append(int(d.at[iid, TYPE2_COL]))

    rows = []
    stats = {
        "n_instances": len(order),
        "n_annotators": 6,
        "type1_resolved": 0, "type1_excluded_tie": 0,
        "type2_resolved": 0, "type2_excluded_tie": 0,
    }

    for iid in order:
        t1 = _majority_label(t1_votes[iid])
        if t1 is None:
            stats["type1_excluded_tie"] += 1
        else:
            stats["type1_resolved"] += 1

        t2 = _majority_label(t2_votes[iid])
        if t2 is None:
            stats["type2_excluded_tie"] += 1
        else:
            stats["type2_resolved"] += 1

        rows.append({
            "instance_id": iid,
            "paper_id":    ref.at[iid, "paper_id"],
            "venue":       ref.at[iid, "venue"],
            "year":        ref.at[iid, "year"],
            "title":       ref.at[iid, "Current Paper"],
            "target":      ref.at[iid, "target"],
            "type1_gold":  t1,
            "type2_gold":  float(t2) if t2 is not None else None,
            "n_annotators": 6,
            "block":        "overlap",
            "adjudication": "majority_vote",
        })

    return pd.DataFrame(rows), stats


def adjudicate_disjoint(df_a: pd.DataFrame, df_b: pd.DataFrame,
                        block_name: str, ann_a: str, ann_b: str
                        ) -> tuple[pd.DataFrame, dict]:
    """
    2-rater adjudication on a 130-instance disjoint block.

    Type I : both agree -> that label; else excluded.
    Type II: |a-b| <= 1 -> mean (may be half-integer); else excluded.
    """
    a = df_a.set_index("instance_id")
    b = df_b.set_index("instance_id")
    common = a.index.intersection(b.index)
    if len(common) != len(a) or len(common) != len(b):
        print(f"  [WARN] block '{block_name}': a has {len(a)}, b has {len(b)}, "
              f"intersection {len(common)}")
    a = a.loc[common]; b = b.loc[common]

    rows = []
    stats = {
        "block": block_name, "annotator_a": ann_a, "annotator_b": ann_b,
        "n_instances": len(common),
        "type1_agree": 0, "type1_excluded_disagree": 0,
        "type2_exact": 0, "type2_within_one": 0, "type2_excluded_gt_one": 0,
    }

    for iid in common:
        a_t1 = int(a.at[iid, TYPE1_COL]); b_t1 = int(b.at[iid, TYPE1_COL])
        a_t2 = int(a.at[iid, TYPE2_COL]); b_t2 = int(b.at[iid, TYPE2_COL])

        # ---- Type I ----
        if a_t1 == b_t1:
            t1 = a_t1; stats["type1_agree"] += 1
        else:
            t1 = None; stats["type1_excluded_disagree"] += 1

        # ---- Type II ----
        d = abs(a_t2 - b_t2)
        if d == 0:
            t2 = float(a_t2); stats["type2_exact"] += 1
        elif d == 1:
            t2 = (a_t2 + b_t2) / 2.0      # half-integer kept
            stats["type2_within_one"] += 1
        else:
            t2 = None; stats["type2_excluded_gt_one"] += 1

        rows.append({
            "instance_id": iid,
            "paper_id":    a.at[iid, "paper_id"],
            "venue":       a.at[iid, "venue"],
            "year":        a.at[iid, "year"],
            "title":       a.at[iid, "Current Paper"],
            "target":      a.at[iid, "target"],
            "type1_gold":  t1,
            "type2_gold":  t2,
            "n_annotators": 2,
            "block":        block_name,
            "adjudication": f"2-rater: {ann_a} & {ann_b}",
        })

    return pd.DataFrame(rows), stats


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 72)
    print("Building human gold standard (510 instances, 2-6 annotators each)")
    print("=" * 72)

    print("\nLoading annotation_all_samples.xlsx ...")
    all_samples = load_all_samples_lookup()
    print(f"  {len(all_samples)} instances indexed")

    # ---- Overlap ----
    print("\n[1/2] Overlap (120 x 6 annotators) ...")
    overlap_dfs = {}
    for name in OVERLAP_ANNOTATORS:
        path = OVERLAP_DIR / f"overlap_{name}.csv"
        df = load_and_normalise(path)
        df = attach_target(df, all_samples, label=f"overlap/{name}")
        overlap_dfs[name] = df
        print(f"  {name:12s}: {len(df)} rows")

    overlap_gold, overlap_stats = adjudicate_overlap(overlap_dfs)
    print(f"  T1 resolved: {overlap_stats['type1_resolved']}   "
          f"excluded (3-3 tie): {overlap_stats['type1_excluded_tie']}")
    print(f"  T2 resolved: {overlap_stats['type2_resolved']}   "
          f"excluded (tie):     {overlap_stats['type2_excluded_tie']}")

    # ---- Disjoint blocks ----
    print("\n[2/2] Disjoint (3 x 130 x 2 annotators) ...")
    disjoint_golds = []
    block_stats = []
    for block_name, file_tag_a, file_tag_b in DISJOINT_BLOCKS:
        # canonical annotator name = file tag with possible underscore restoration
        # for 'drzhang' -> 'dr_zhang'.
        ann_a = "dr_zhang" if file_tag_a == "drzhang" else file_tag_a
        ann_b = "dr_zhang" if file_tag_b == "drzhang" else file_tag_b

        path_a = DISJOINT_DIR / f"set_{block_name}_by_{file_tag_a}.csv"
        path_b = DISJOINT_DIR / f"set_{block_name}_by_{file_tag_b}.csv"
        df_a = load_and_normalise(path_a)
        df_b = load_and_normalise(path_b)
        df_a = attach_target(df_a, all_samples, label=f"{block_name}/{ann_a}")
        df_b = attach_target(df_b, all_samples, label=f"{block_name}/{ann_b}")

        gold, stats = adjudicate_disjoint(df_a, df_b, block_name, ann_a, ann_b)
        disjoint_golds.append(gold)
        block_stats.append(stats)
        print(f"  '{block_name}' ({ann_a} & {ann_b}): "
              f"T1 agree {stats['type1_agree']}/{stats['n_instances']} "
              f"(excl {stats['type1_excluded_disagree']}) | "
              f"T2 exact {stats['type2_exact']}, "
              f"within-1 {stats['type2_within_one']}, "
              f"excl {stats['type2_excluded_gt_one']}")

    # ---- Combine ----
    gold = pd.concat([overlap_gold] + disjoint_golds, ignore_index=True)
    print(f"\nTotal gold rows:         {len(gold)}")
    print(f"  type1_gold available:  {gold['type1_gold'].notna().sum()} "
          f"(excluded: {gold['type1_gold'].isna().sum()})")
    print(f"  type2_gold available:  {gold['type2_gold'].notna().sum()} "
          f"(excluded: {gold['type2_gold'].isna().sum()})")

    # T1 is always integer; T2 may be half-integer (2-rater), so keep as float.
    gold["type1_gold"] = gold["type1_gold"].astype("Int64")
    # type2_gold stays as float (NaN-aware); easy to cast downstream

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gold.to_csv(OUTPUT_GOLD_CSV, index=False)
    print(f"\nSaved: {OUTPUT_GOLD_CSV}")

    # ---- Distribution summary ----
    print("\n--- Gold label distributions (non-excluded rows only) ---")
    print(f"Type I : {dict(gold['type1_gold'].value_counts().sort_index())}")
    t2_counts = gold["type2_gold"].value_counts().sort_index().to_dict()
    print(f"Type II: {t2_counts}")

    report = {
        "rules": {
            "type_1": {
                "2_raters": "agree -> gold; disagree -> excluded",
                "6_raters": "majority (>=4/6) -> gold; 3-3 tie -> excluded",
            },
            "type_2": {
                "2_raters": "|a-b|<=1 -> mean (may be half-integer); else excluded",
                "6_raters": "modal label if unique -> gold; tie -> excluded",
            },
            "tasks_independent": True,
            "llm_used_in_adjudication": False,
        },
        "overlap": overlap_stats,
        "disjoint_blocks": block_stats,
        "totals": {
            "rows":                len(gold),
            "type1_resolved":      int(gold["type1_gold"].notna().sum()),
            "type1_excluded":      int(gold["type1_gold"].isna().sum()),
            "type2_resolved":      int(gold["type2_gold"].notna().sum()),
            "type2_excluded":      int(gold["type2_gold"].isna().sum()),
        },
        "label_distributions": {
            "type_1": {str(k): int(v) for k, v
                       in gold["type1_gold"].value_counts().sort_index().items()},
            "type_2": {str(k): int(v) for k, v
                       in gold["type2_gold"].value_counts().sort_index().items()},
        },
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()