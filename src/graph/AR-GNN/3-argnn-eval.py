"""
AR-GNN (AR-GATK) — Evaluation Script
=======================================
Evaluates each model on its label type using:
  1. Cosine similarity on learned GAT embeddings
  2. Quality-score re-ranking (0.7*cos + 0.3*quality)

Retrieval pool: all papers in graph (train set papers are the candidates).
"""

import os
import sys
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from argnn_utils import (
    log, extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

ARGNN_DIR = OUTPUT_DIR / "graph/AR-GNN/"
EMBEDDINGS_BASE_DIR = ARGNN_DIR / "embeddings/"

RESULTS_DIR = ARGNN_DIR / "evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "argnn_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "argnn_evaluation_metrics.json"


def evaluate_label_type(label_key, eval_df, train_pids_set):
    cfg = LABEL_TYPES[label_key]
    emb_dir = EMBEDDINGS_BASE_DIR / label_key
    emb_pt = emb_dir / "argnn_embeddings.pt"

    if not emb_pt.exists():
        log(f"  Missing embeddings for {cfg['name']} — skipping.")
        return None

    data = torch.load(emb_pt, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"].numpy()
    paper_ids = data["paper_ids"]
    quality_scores = data["quality_scores"]

    pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    log(f"  Total papers in embeddings: {len(paper_ids)}")

    # Candidate indices (train papers only)
    train_indices = [pid_to_idx[pid] for pid in paper_ids if pid in train_pids_set]
    train_embs = embeddings[train_indices]
    train_qs = quality_scores[train_indices]
    train_ids = [paper_ids[i] for i in train_indices]
    train_id_set = set(train_ids)

    # Normalize
    train_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_norm = train_embs / train_norms

    # Normalize quality scores
    qs_max = train_qs.max()
    if qs_max > 0:
        train_qs_norm = train_qs / qs_max
    else:
        train_qs_norm = train_qs

    # Ground truth
    ground_truth = {}
    for row in eval_df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in pid_to_idx: continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant = extract_relevant_sets(refs)[label_key]
        if relevant:
            ground_truth[pid] = relevant

    log(f"  Queries with relevant: {len(ground_truth)}")
    if not ground_truth:
        return None

    max_k = max(EVAL_K_VALUES["recall"])
    acc = {"map": [], "mrr": [],
           "recall": {k: [] for k in EVAL_K_VALUES["recall"]},
           "ndcg": {k: [] for k in EVAL_K_VALUES["ndcg"]},
           "hr": {k: [] for k in EVAL_K_VALUES["hr"]}}

    for qid in tqdm(ground_truth, desc=f"  Eval {label_key}"):
        if qid not in pid_to_idx: continue
        q_emb = embeddings[pid_to_idx[qid]]
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0: continue
        q_emb_norm = q_emb / q_norm

        # Cosine similarities to train papers
        cos_sims = train_norm @ q_emb_norm

        # Combined score: 0.7 * cos + 0.3 * quality
        combined = 0.7 * cos_sims + 0.3 * train_qs_norm

        # Exclude self
        for i, tid in enumerate(train_ids):
            if tid == qid:
                combined[i] = -1.0

        top_idx = np.argsort(-combined)[:max_k]
        retrieved = [train_ids[i] for i in top_idx]

        metrics = compute_all_metrics(ground_truth[qid], retrieved)
        acc["map"].append(metrics["map"])
        acc["mrr"].append(metrics["mrr"])
        for k in EVAL_K_VALUES["recall"]: acc["recall"][k].append(metrics["recall"][k])
        for k in EVAL_K_VALUES["ndcg"]: acc["ndcg"][k].append(metrics["ndcg"][k])
        for k in EVAL_K_VALUES["hr"]: acc["hr"][k].append(metrics["hr"][k])

    n_q = len(acc["map"])
    return {
        "num_queries": n_q,
        "mean_map": float(np.mean(acc["map"])),
        "mean_mrr": float(np.mean(acc["mrr"])),
        "mean_recall": {k: float(np.mean(v)) for k, v in acc["recall"].items()},
        "mean_ndcg": {k: float(np.mean(v)) for k, v in acc["ndcg"].items()},
        "mean_hr": {k: float(np.mean(v)) for k, v in acc["hr"].items()},
        "map_std": float(np.std(acc["map"])),
        "mrr_std": float(np.std(acc["mrr"])),
    }


def main():
    log("=" * 80)
    log("AR-GNN — Evaluation")
    log("=" * 80)

    eval_df = pl.read_parquet(EVAL_PARQUET)
    train_df = pl.read_parquet(TRAIN_PARQUET)
    train_pids_set = set(train_df["paper_id"].to_list())

    all_results = {}
    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Evaluating: {label_cfg['name']}")
        log(f"{'=' * 80}")
        result = evaluate_label_type(label_key, eval_df, train_pids_set)
        if result:
            all_results[label_key] = result
            log(f"  MAP: {result['mean_map']:.6f}  MRR: {result['mean_mrr']:.6f}")

    if not all_results:
        log("No label types evaluated.")
        return

    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\nAR-GNN (AR-GATK) EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write("Method: AR-GATK (2-layer GAT + FCN + distance loss + quality re-ranking)\n")
        f.write("Retrieval: 0.7*cosine + 0.3*quality_score\n\n")
        for key, cfg in LABEL_TYPES.items():
            if key not in all_results: continue
            res = all_results[key]
            f.write(f"--- {cfg['name']} ({cfg['description']}) ---\n")
            f.write(f"Queries: {res['num_queries']}\nMAP: {res['mean_map']:.6f}  MRR: {res['mean_mrr']:.6f}\n")
            for k in EVAL_K_VALUES["ndcg"]: f.write(f"  nDCG@{k}: {res['mean_ndcg'][k]:.6f}\n")
            for k in EVAL_K_VALUES["recall"]: f.write(f"  Recall@{k}: {res['mean_recall'][k]:.6f}\n")
            for k in EVAL_K_VALUES["hr"]: f.write(f"  HR@{k}: {res['mean_hr'][k]:.6f}\n")
            f.write(f"  MAP std: {res['map_std']:.6f}  MRR std: {res['mrr_std']:.6f}\n\n")

    results_dict = {
        "model_info": {
            "method": "AR-GATK (GAT + distance loss + quality re-ranking)",
            "features": "10D (age, citation, venue_quality, topic_influence, 6 keyword communities)",
            "retrieval": "0.7*cosine + 0.3*quality_score",
        },
        "label_thresholds": {"type_1": "label == 1", "type_2": f">= {TYPE_2_THRESHOLD}", "type_3": f">= {TYPE_3_THRESHOLD}"},
    }
    for key in LABEL_TYPES:
        if key in all_results:
            res = all_results[key]
            results_dict[key] = {
                "name": LABEL_TYPES[key]["name"], "num_queries": res["num_queries"],
                "metrics": {
                    "map": res["mean_map"], "mrr": res["mean_mrr"],
                    "ndcg": {f"ndcg@{k}": res["mean_ndcg"][k] for k in EVAL_K_VALUES["ndcg"]},
                    "recall": {f"recall@{k}": res["mean_recall"][k] for k in EVAL_K_VALUES["recall"]},
                    "hit_rate": {f"hr@{k}": res["mean_hr"][k] for k in EVAL_K_VALUES["hr"]},
                },
            }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE} and {RESULTS_JSON}")


if __name__ == "__main__":
    main()
