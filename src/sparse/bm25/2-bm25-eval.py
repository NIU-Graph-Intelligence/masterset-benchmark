"""
BM25 Baseline — Evaluation Script
===================================
Evaluates BM25 retrieval on the eval set using the prebuilt index.

For each eval paper, tokenizes its text (title + abstract),
scores it against all train paper candidates using BM25, retrieves top-K,
and computes metrics for all 3 label types independently:

Label Types:
  - Type 1 (binary):      label == 1 → relevant
  - Type 2 (usefulness):  label >= 4 → relevant
  - Type 3 (relatedness): label >= 3 → relevant

Metrics (per label type):
  - MAP (Mean Average Precision)
  - MRR (Mean Reciprocal Rank)
  - Recall@{10, 50, 100, 500}
  - nDCG@{10, 20, 30, 50}
  - HR@{10, 20} (Hit Rate)

Input:
  - output/bm25/index/bm25_index.pkl                              (from index building script)
  - data/train_eval_set/v2.0/eval_v2.0.parquet                    (eval queries + ground truth)

Output:
  - output/sparse/bm25/evaluation_results/bm25_evaluation_metrics.txt
  - output/sparse/bm25/evaluation_results/bm25_evaluation_metrics.json
"""

import os
import sys
import time
import json
import math
import pickle
import re
from pathlib import Path
from collections import Counter

import polars as pl
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from bm25_utils import load_paper_text_map, tokenize, build_inverted_index

# -------- Config --------
load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "all_papers_with_refs_and_labels_v2.0.parquet"

INDEX_DIR = OUTPUT_DIR / "sparse/bm25/index/"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = INDEX_DIR / "bm25_index.pkl"
METADATA_FILE = INDEX_DIR / "index_metadata.json"

RESULTS_DIR = OUTPUT_DIR / "sparse/bm25/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "bm25_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "bm25_evaluation_metrics.json"

# Metrics to compute (matching existing evaluation convention)
EVAL_K_VALUES = {
    "recall": [10, 50, 100, 500],
    "ndcg": [10, 20, 30, 50],
    "hr": [10, 20]
}

SEP = "[SEP]"

# ---- Label thresholds ----
# Type 1: binary (0 or 1), so 1 = relevant
# Type 2: usefulness score, >= 4 = relevant
# Type 3: relatedness score, >= 3 = relevant
TYPE_2_THRESHOLD = 4.0
TYPE_3_THRESHOLD = 3.0

# Label type definitions for iteration
LABEL_TYPES = {
    "type_1": {
        "name": "Type 1 (Binary Relevance)",
        "field": "type_1_output",
        "threshold_fn": lambda v: float(v) == 1.0,
        "description": "binary, label == 1",
    },
    "type_2": {
        "name": "Type 2 (Usefulness)",
        "field": "type_2_output",
        "threshold_fn": lambda v: float(v) >= TYPE_2_THRESHOLD,
        "description": f"usefulness >= {TYPE_2_THRESHOLD}",
    },
    "type_3": {
        "name": "Type 3 (Relatedness)",
        "field": "type_3_output",
        "threshold_fn": lambda v: float(v) >= TYPE_3_THRESHOLD,
        "description": f"relatedness >= {TYPE_3_THRESHOLD}",
    },
}


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ======================================================================
# BM25 Scoring (fast, inverted-index based)
# ======================================================================

def bm25_score_query_fast(query_tokens, index, inverted_index, top_k=500):
    """
    Optimized BM25 scoring using an inverted index.
    Only visits documents that contain at least one query term.
    """
    N = index["N"]
    avgdl = index["avgdl"]
    k1 = index["k1"]
    b = index["b"]
    df_dict = index["df"]
    tf_list = index["tf"]
    doc_lengths = index["doc_lengths"]
    paper_ids = index["paper_ids"]

    query_tf = Counter(query_tokens)

    # Accumulate scores only for documents that share terms with query
    doc_scores = Counter()

    for term in query_tf:
        doc_freq = df_dict.get(term, 0)
        if doc_freq == 0:
            continue
        idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        # Iterate only over documents containing this term
        for doc_idx in inverted_index.get(term, []):
            tf_val = tf_list[doc_idx].get(term, 0)
            dl = doc_lengths[doc_idx]
            numerator = tf_val * (k1 + 1.0)
            denominator = tf_val + k1 * (1.0 - b + b * dl / avgdl)
            doc_scores[doc_idx] += idf * numerator / denominator

    # Get top-k
    top_docs = doc_scores.most_common(top_k)
    return [paper_ids[idx] for idx, score in top_docs if score > 0]


# ======================================================================
# Metric Functions
# ======================================================================

def compute_map(relevant, retrieved):
    """Mean Average Precision"""
    if not relevant:
        return 0.0
    score = 0.0
    num_hits = 0
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant)


def compute_mrr(relevant, retrieved):
    """Mean Reciprocal Rank"""
    for i, paper_id in enumerate(retrieved):
        if paper_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(relevant, retrieved, k):
    """Recall@k"""
    if not relevant:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / len(relevant)


def compute_hit_rate_at_k(relevant, retrieved, k):
    """Hit Rate@k — 1 if any relevant doc is in top-k, else 0"""
    top_k = set(retrieved[:k])
    return 1.0 if any(item in top_k for item in relevant) else 0.0


def compute_dcg_at_k(relevant, retrieved, k):
    """Discounted Cumulative Gain at k"""
    dcg = 0.0
    for i, paper_id in enumerate(retrieved[:k]):
        if paper_id in relevant:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def compute_ndcg_at_k(relevant, retrieved, k):
    """Normalized Discounted Cumulative Gain at k"""
    if not relevant:
        return 0.0
    dcg = compute_dcg_at_k(relevant, retrieved, k)
    ideal_retrieved = list(relevant) + [0] * k
    idcg = compute_dcg_at_k(relevant, ideal_retrieved, k)
    return dcg / idcg if idcg > 0 else 0.0


# ======================================================================
# Per-query metric computation helper
# ======================================================================

def compute_all_metrics(relevant, retrieved):
    """Compute all metrics for a single query, returns a dict."""
    results = {}
    results["map"] = compute_map(relevant, retrieved)
    results["mrr"] = compute_mrr(relevant, retrieved)
    results["recall"] = {k: compute_recall_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["recall"]}
    results["ndcg"] = {k: compute_ndcg_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["ndcg"]}
    results["hr"] = {k: compute_hit_rate_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["hr"]}
    return results


def zero_metrics():
    """Return a zero-valued metrics dict (for empty retrieval)."""
    return {
        "map": 0.0,
        "mrr": 0.0,
        "recall": {k: 0.0 for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: 0.0 for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: 0.0 for k in EVAL_K_VALUES["hr"]},
    }


# ======================================================================
# Ground truth extraction
# ======================================================================

def extract_relevant_sets(references):
    """
    Given a list of reference dicts, extract the relevant paper_id sets
    for each label type.

    Returns dict: label_type_key -> set of relevant matched_paper_ids
    """
    relevant = {key: set() for key in LABEL_TYPES}

    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id:
            continue

        for key, cfg in LABEL_TYPES.items():
            field = cfg["field"]
            raw_val = ref.get(field)
            if raw_val is None:
                continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError):
                continue

    return relevant


# ======================================================================
# Main Evaluation
# ======================================================================

def main():
    log("=" * 80)
    log("BM25 Baseline — Evaluation (3 Label Types)")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")

    # ---- Load index ----
    if not INDEX_FILE.exists():
        log(f"ERROR: BM25 index not found at {INDEX_FILE}")
        log("Please run the index building script first!")
        return

    log("\nLoading BM25 index...")
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)

    log(f"  Candidates: {index['N']:,}")
    log(f"  Vocabulary: {len(index['df']):,} terms")
    log(f"  Avg doc length: {index['avgdl']:.1f}")
    log(f"  BM25 k1={index['k1']}, b={index['b']}")

    # ---- Build inverted index for fast scoring ----
    inverted_index = build_inverted_index(index)

    # ---- Load eval paper texts ----
    log("\nLoading eval paper texts...")
    eval_map = load_paper_text_map(EVAL_PARQUET)
    train_map = load_paper_text_map(TRAIN_PARQUET)
    log(f"  Eval papers with text: {len(eval_map):,}")

    # ---- Load eval data for ground truth ----
    log("\nLoading evaluation ground truth...")
    eval_df = pl.read_parquet(EVAL_PARQUET)
    num_eval = eval_df.height
    log(f"  Total evaluation papers: {num_eval}")

    # ---- Initialize per-label-type metric accumulators ----
    # Each label type gets its own independent set of metric lists
    accumulators = {}
    for key in LABEL_TYPES:
        accumulators[key] = {
            "map_scores": [],
            "mrr_scores": [],
            "recall_scores": {k: [] for k in EVAL_K_VALUES["recall"]},
            "ndcg_scores": {k: [] for k in EVAL_K_VALUES["ndcg"]},
            "hr_scores": {k: [] for k in EVAL_K_VALUES["hr"]},
            "num_queries_with_relevant": 0,
        }

    max_k = max(EVAL_K_VALUES["recall"])

    # ---- Evaluate ----
    log("\nEvaluating...")
    skipped_no_text = 0
    skipped_no_tokens = 0

    for eval_idx in tqdm(range(num_eval), desc="Processing evaluation papers"):
        try:
            row = eval_df.row(eval_idx, named=True)
        except IndexError:
            continue

        qid = row["paper_id"]
        references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]

        # Extract relevant sets for all 3 label types
        relevant_sets = extract_relevant_sets(references)

        # Skip if no label type has any relevant papers
        if all(len(v) == 0 for v in relevant_sets.values()):
            continue

        # Get query text
        query_text = eval_map.get(qid, "")
        if not query_text.strip():
            skipped_no_text += 1
            continue

        # Tokenize query
        query_tokens = tokenize(query_text)
        if not query_tokens:
            skipped_no_tokens += 1
            continue

        # Retrieve using BM25 (done ONCE, shared across all label types)
        retrieved = bm25_score_query_fast(query_tokens, index, inverted_index, top_k=max_k)

        # Exclude self from results
        retrieved = [pid for pid in retrieved if pid != qid]

        # Compute metrics for each label type independently
        for key in LABEL_TYPES:
            relevant = relevant_sets[key]

            # Only evaluate if this label type has relevant papers for this query
            if not relevant:
                continue

            acc = accumulators[key]
            acc["num_queries_with_relevant"] += 1

            if not retrieved:
                # Count this query but with zero scores
                acc["map_scores"].append(0.0)
                acc["mrr_scores"].append(0.0)
                for k in EVAL_K_VALUES["recall"]:
                    acc["recall_scores"][k].append(0.0)
                for k in EVAL_K_VALUES["ndcg"]:
                    acc["ndcg_scores"][k].append(0.0)
                for k in EVAL_K_VALUES["hr"]:
                    acc["hr_scores"][k].append(0.0)
                continue

            metrics = compute_all_metrics(relevant, retrieved)
            acc["map_scores"].append(metrics["map"])
            acc["mrr_scores"].append(metrics["mrr"])
            for k in EVAL_K_VALUES["recall"]:
                acc["recall_scores"][k].append(metrics["recall"][k])
            for k in EVAL_K_VALUES["ndcg"]:
                acc["ndcg_scores"][k].append(metrics["ndcg"][k])
            for k in EVAL_K_VALUES["hr"]:
                acc["hr_scores"][k].append(metrics["hr"][k])

    if skipped_no_text > 0:
        log(f"  Skipped {skipped_no_text} papers with no text")
    if skipped_no_tokens > 0:
        log(f"  Skipped {skipped_no_tokens} papers with no tokens after tokenization")

    # ---- Compute means and print results per label type ----
    all_results = {}

    for key, cfg in LABEL_TYPES.items():
        acc = accumulators[key]
        n_q = acc["num_queries_with_relevant"]

        mean_map = np.mean(acc["map_scores"]) if acc["map_scores"] else 0.0
        mean_mrr = np.mean(acc["mrr_scores"]) if acc["mrr_scores"] else 0.0
        mean_recall = {k: np.mean(v) if v else 0.0 for k, v in acc["recall_scores"].items()}
        mean_ndcg = {k: np.mean(v) if v else 0.0 for k, v in acc["ndcg_scores"].items()}
        mean_hr = {k: np.mean(v) if v else 0.0 for k, v in acc["hr_scores"].items()}

        all_results[key] = {
            "num_queries": n_q,
            "mean_map": float(mean_map),
            "mean_mrr": float(mean_mrr),
            "mean_recall": {k: float(v) for k, v in mean_recall.items()},
            "mean_ndcg": {k: float(v) for k, v in mean_ndcg.items()},
            "mean_hr": {k: float(v) for k, v in mean_hr.items()},
            "map_std": float(np.std(acc["map_scores"])) if acc["map_scores"] else 0.0,
            "mrr_std": float(np.std(acc["mrr_scores"])) if acc["mrr_scores"] else 0.0,
            "map_min": float(np.min(acc["map_scores"])) if acc["map_scores"] else 0.0,
            "map_max": float(np.max(acc["map_scores"])) if acc["map_scores"] else 0.0,
            "mrr_min": float(np.min(acc["mrr_scores"])) if acc["mrr_scores"] else 0.0,
            "mrr_max": float(np.max(acc["mrr_scores"])) if acc["mrr_scores"] else 0.0,
        }

        # Print
        print(f"\n{'=' * 80}")
        print(f"  {cfg['name']}  ({cfg['description']})")
        print(f"{'=' * 80}")
        print(f"  Queries evaluated: {n_q}")
        print(f"\n  Mean Average Precision (MAP): {mean_map:.6f}")
        print(f"  Mean Reciprocal Rank (MRR):   {mean_mrr:.6f}")

        print(f"\n  Normalized Discounted Cumulative Gain:")
        for k in EVAL_K_VALUES["ndcg"]:
            print(f"    nDCG@{k:3d}: {mean_ndcg[k]:.6f}")

        print(f"\n  Recall Metrics:")
        for k in EVAL_K_VALUES["recall"]:
            print(f"    Recall@{k:3d}: {mean_recall[k]:.6f}")

        print(f"\n  Hit Rate Metrics:")
        for k in EVAL_K_VALUES["hr"]:
            print(f"    HR@{k:3d}: {mean_hr[k]:.6f}")

    print(f"\n{'=' * 80}")

    # ---- Save results to text file ----
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BM25 BASELINE EVALUATION RESULTS (3 Label Types)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Method: BM25 (Best Match 25)\n")
        f.write(f"  - k1={index['k1']}, b={index['b']}\n")
        f.write(f"  - Text: title + abstract\n")
        f.write(f"  - Tokenization: lowercase + punctuation removal + stopword filtering\n")
        f.write(f"  - Candidates: {index['N']:,} train papers\n")
        f.write(f"  - Vocabulary: {len(index['df']):,} terms\n\n")

        for key, cfg in LABEL_TYPES.items():
            res = all_results[key]
            f.write("-" * 80 + "\n")
            f.write(f"  {cfg['name']}  ({cfg['description']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Queries evaluated: {res['num_queries']}\n\n")

            f.write(f"  Mean Average Precision (MAP): {res['mean_map']:.6f}\n")
            f.write(f"  Mean Reciprocal Rank (MRR):   {res['mean_mrr']:.6f}\n\n")

            f.write("  Normalized Discounted Cumulative Gain:\n")
            for k in EVAL_K_VALUES["ndcg"]:
                f.write(f"    nDCG@{k:3d}: {res['mean_ndcg'][k]:.6f}\n")
            f.write("\n")

            f.write("  Recall Metrics:\n")
            for k in EVAL_K_VALUES["recall"]:
                f.write(f"    Recall@{k:3d}: {res['mean_recall'][k]:.6f}\n")
            f.write("\n")

            f.write("  Hit Rate Metrics:\n")
            for k in EVAL_K_VALUES["hr"]:
                f.write(f"    HR@{k:3d}: {res['mean_hr'][k]:.6f}\n")
            f.write("\n")

            f.write("  Detailed Statistics:\n")
            f.write(f"    MAP - Min: {res['map_min']:.6f}, Max: {res['map_max']:.6f}, Std: {res['map_std']:.6f}\n")
            f.write(f"    MRR - Min: {res['mrr_min']:.6f}, Max: {res['mrr_max']:.6f}, Std: {res['mrr_std']:.6f}\n")
            f.write("\n")

    # ---- Save results to JSON ----
    results_dict = {
        "model_info": {
            "method": "BM25",
            "description": "BM25 (Best Match 25) term-frequency baseline",
            "k1": index["k1"],
            "b": index["b"],
            "text_source": "title + abstract",
            "tokenization": "lowercase + punctuation removal + stopword filtering",
            "num_candidates": index["N"],
            "vocab_size": len(index["df"]),
        },
        "label_thresholds": {
            "type_1": "binary, label == 1",
            "type_2": f"usefulness >= {TYPE_2_THRESHOLD}",
            "type_3": f"relatedness >= {TYPE_3_THRESHOLD}",
        },
    }

    for key, cfg in LABEL_TYPES.items():
        res = all_results[key]
        results_dict[key] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "num_queries": res["num_queries"],
            "metrics": {
                "map": res["mean_map"],
                "mrr": res["mean_mrr"],
                "ndcg": {f"ndcg@{k}": res["mean_ndcg"][k] for k in EVAL_K_VALUES["ndcg"]},
                "recall": {f"recall@{k}": res["mean_recall"][k] for k in EVAL_K_VALUES["recall"]},
                "hit_rate": {f"hr@{k}": res["mean_hr"][k] for k in EVAL_K_VALUES["hr"]},
            },
            "statistics": {
                "map_std": res["map_std"],
                "mrr_std": res["mrr_std"],
            }
        }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Text: {RESULTS_FILE}")
    print(f"  JSON: {RESULTS_JSON}")


if __name__ == "__main__":
    main()