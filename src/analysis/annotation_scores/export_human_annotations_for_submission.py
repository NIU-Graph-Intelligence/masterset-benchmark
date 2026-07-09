"""
export_human_annotations_for_submission.py

Produces a SINGLE CSV of the 510-instance human annotation set suitable for
release in an anonymous submission repository.

Design
------
* Annotators are anonymized to stable pseudonyms A1..A6 via an alphabetical
  ordering of their real usernames. The mapping is deterministic and
  reproducible from this script alone; the real-name -> pseudonym table is
  NEVER written to disk.
* The CSV is rectangular: every row has six type_1 columns and six type_2
  columns (A1..A6). Cells are left empty for annotators who did not label
  that instance. Every instance has either 2 or 6 filled entries per task.
* Adjudicated gold labels (produced by build_gold_standard.py) are included,
  with empty cells where the instance was excluded from a task.
* Paper titles and citation contexts are intentionally omitted: paper_id and
  target are sufficient to reconstruct each instance against the public
  corpus, and titles are identifying enough to risk deanonymization during
  review.

Columns (left to right)
-----------------------
  instance_id      : stable id from the annotation sheet
  paper_id         : MasterSet paper id of the citing (query) paper
  target           : reference id within that paper's parsed reference list
  venue, year      : venue and publication year of the citing paper
  n_annotators     : number of humans who labelled this instance (2 or 6)
  A1_type1 .. A6_type1 : binary Type-I labels (0/1), empty where unrated
  A1_type2 .. A6_type2 : ordinal Type-II labels (1-5), empty where unrated

Usage
-----
Run from the project root. Reads the same annotator CSVs used by the IAA
and gold-standard scripts; writes a single CSV to OUTPUT_CSV.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OVERLAP_DIR  = Path("data/annotations/overlap_sets")
DISJOINT_DIR = Path("data/annotations/disjoint_sets")
GOLD_PATH    = Path("output/analysis/annotation_scores/human_gold.csv")

OUTPUT_CSV = Path("output/analysis/annotation_scores/"
                  "masterset_human_annotations_anonymized.csv")

# Real annotator names. Pseudonyms are assigned by sorted() order of this list,
# so the mapping is deterministic:
#   dr_zhang   -> A1
#   ibrahim    -> A2
#   ratul      -> A3
#   samikshya  -> A4
#   shafayet   -> A5
#   tanish     -> A6
REAL_ANNOTATORS = ["ratul", "dr_zhang", "tanish", "samikshya", "ibrahim", "shafayet"]

# Disjoint block layout: (block_tag, [file_tag_of_rater_a, file_tag_of_rater_b])
# file_tag is what appears in "set_<block>_by_<file_tag>.csv"
# The canonical annotator name differs for dr_zhang (file tag 'drzhang').
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

# ============================================================
# Pseudonym mapping (deterministic, not persisted)
# ============================================================

PSEUDONYM = {real: f"A{i+1}" for i, real in enumerate(sorted(REAL_ANNOTATORS))}


# ============================================================
# Loading helpers
# ============================================================

def load_and_normalise(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TYPE1_COL] = df[TYPE1_COL].map(TYPE1_MAP)
    df[TYPE2_COL] = df[TYPE2_COL].map(TYPE2_MAP)
    if df[TYPE1_COL].isna().any() or df[TYPE2_COL].isna().any():
        raise ValueError(f"{path.name}: unmapped labels found.")
    return df


def collect_all_labels() -> pd.DataFrame:
    """
    Returns long-format DataFrame:
        instance_id, paper_id, venue, year, annotator (real), type1, type2
    """
    rows = []

    # Overlap: 120 instances x 6 annotators
    for real in REAL_ANNOTATORS:
        path = OVERLAP_DIR / f"overlap_{real}.csv"
        df = load_and_normalise(path)
        for _, r in df.iterrows():
            rows.append({
                "instance_id": r["instance_id"],
                "paper_id":    r["paper_id"],
                "venue":       r["venue"],
                "year":        r["year"],
                "annotator":   real,
                "type1":       int(r[TYPE1_COL]),
                "type2":       int(r[TYPE2_COL]),
            })

    # Disjoint: 3 blocks x 130 instances x 2 annotators
    for block_tag, raters in DISJOINT_BLOCKS:
        for file_tag, real_name in raters:
            path = DISJOINT_DIR / f"set_{block_tag}_by_{file_tag}.csv"
            df = load_and_normalise(path)
            for _, r in df.iterrows():
                rows.append({
                    "instance_id": r["instance_id"],
                    "paper_id":    r["paper_id"],
                    "venue":       r["venue"],
                    "year":        r["year"],
                    "annotator":   real_name,
                    "type1":       int(r[TYPE1_COL]),
                    "type2":       int(r[TYPE2_COL]),
                })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    print("Loading all annotator files ...")
    long = collect_all_labels()
    n_instances = long["instance_id"].nunique()
    print(f"  {len(long)} rater-item rows, {n_instances} unique instances")
    assert n_instances == 510, f"Expected 510, got {n_instances}"

    # Map to pseudonyms
    long["pseudonym"] = long["annotator"].map(PSEUDONYM)
    if long["pseudonym"].isna().any():
        missing = long.loc[long["pseudonym"].isna(), "annotator"].unique()
        raise ValueError(f"No pseudonym for: {list(missing)}")

    # Pivot to wide: one row per instance, six type1 + six type2 columns.
    # pandas .pivot() gives us NaN for un-rated cells automatically.
    wide_t1 = long.pivot(index="instance_id", columns="pseudonym", values="type1")
    wide_t2 = long.pivot(index="instance_id", columns="pseudonym", values="type2")

    # Ensure all six pseudonym columns exist and are ordered A1..A6
    all_pseudos = sorted(PSEUDONYM.values())
    for p in all_pseudos:
        if p not in wide_t1.columns: wide_t1[p] = pd.NA
        if p not in wide_t2.columns: wide_t2[p] = pd.NA
    wide_t1 = wide_t1[all_pseudos].rename(columns=lambda c: f"{c}_type1")
    wide_t2 = wide_t2[all_pseudos].rename(columns=lambda c: f"{c}_type2")

    # Per-instance metadata from the first occurrence of each instance_id
    meta = (long.groupby("instance_id")
                .agg(paper_id=("paper_id", "first"),
                     venue=("venue", "first"),
                     year=("year", "first"),
                     n_annotators=("annotator", "nunique"))
                .reset_index())

    # Merge metadata + wide tables
    out = meta.merge(wide_t1, on="instance_id") \
              .merge(wide_t2, on="instance_id")

    # Attach target from the adjudicated gold file (needed for linking
    # citation instances back to the public corpus). We do NOT include the
    # gold labels themselves; reviewers can reproduce those from this CSV.
    print(f"Loading target ids from {GOLD_PATH} ...")
    gold = pd.read_csv(GOLD_PATH)
    out = out.merge(gold[["instance_id", "target"]], on="instance_id", how="left")

    # Column order
    t1_cols = [f"{p}_type1" for p in all_pseudos]
    t2_cols = [f"{p}_type2" for p in all_pseudos]
    col_order = (
        ["instance_id", "paper_id", "target", "venue", "year", "n_annotators"]
        + t1_cols + t2_cols
    )
    out = out[col_order]

    # Cast numeric label columns to nullable integer so NaNs stay as empty
    # cells (not '0.0') and ints print cleanly as '1', '0', '3' etc.
    for c in t1_cols + t2_cols:
        out[c] = out[c].astype("Int64")

    # Write
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # NaN -> empty string so reviewers see blank cells in spreadsheet tools
    out.to_csv(OUTPUT_CSV, index=False, na_rep="")
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"  rows: {len(out)}")
    print(f"  cols: {len(out.columns)}")

    # Quick sanity summary
    n2 = (out["n_annotators"] == 2).sum()
    n6 = (out["n_annotators"] == 6).sum()
    print(f"  2-annotator instances: {n2}")
    print(f"  6-annotator instances: {n6}")


if __name__ == "__main__":
    main()