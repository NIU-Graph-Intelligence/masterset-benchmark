"""
LitFM Graph Retriever — Evaluation Script
============================================
Evaluates each fine-tuned LitFM model on its corresponding label type.
Uses precomputed query (eval) and candidate (train) embeddings.
Cosine similarity retrieval — candidates are train papers only.

Resume: skips label types whose embeddings don't exist.
"""

import os
import sys
import json
import torch
import numpy as np
import polars as pl
import faiss
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from litfm_utils import (
    log, extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "graph/LitFM/embeddings/"

RESULTS_DIR = OUTPUT_DIR / "graph/LitFM/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "litfm_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "litfm_evaluation_metrics.json"


def evaluate_label_type(label_key, eval_df):
    cfg = LABEL_TYPES[label_key]
    emb_dir = EMBEDDINGS_BASE_DIR / label_key

    cand_path = emb_dir / "candidate_embeddings.pt"
    query_path = emb_dir / "query_embeddings.pt"

    if not cand_path.exists() or not query_path.exists():
        log(f"  Missing embeddings for {cfg['name']} — skipping.")
        return None

    cand_data = torch.load(cand_path, map_location="cpu", weights_only=False)
    cand_embs = cand_data["embeddings"].numpy()
    cand_pids = cand_data["paper_ids"]

    query_data = torch.load(query_path, map_location="cpu", weights_only=False)
    query_embs = query_data["embeddings"].numpy()
    query_pids = query_data["paper_ids"]

    query_pid_to_idx = {pid: i for i, pid in enumerate(query_pids)}

    log(f"  Candidates: {cand_embs.shape}, Queries: {query_embs.shape}")

    # Ground truth for THIS label type
    ground_truth = {}
    for row in eval_df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in query_pid_to_idx: continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant = extract_relevant_sets(refs)[label_key]
        if relevant:
            ground_truth[pid] = relevant

    log(f"  Queries with relevant: {len(ground_truth)}")
    if not ground_truth:
        return None

    # Normalize & FAISS
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1.0
    cand_norm = cand_embs / cand_norms

    query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_norms[query_norms == 0] = 1.0
    query_norm = query_embs / query_norms

    faiss_index = faiss.IndexFlatIP(cand_norm.shape[1])
    faiss_index.add(cand_norm.astype("float32"))

    # Retrieve
    q_embs_arr = []
    q_ids = []
    for pid in ground_truth:
        q_embs_arr.append(query_norm[query_pid_to_idx[pid]])
        q_ids.append(pid)
    q_embs_arr = np.array(q_embs_arr).astype("float32")

    max_k = max(EVAL_K_VALUES["recall"])
    distances, indices = faiss_index.search(q_embs_arr, max_k)

    # Metrics
    acc = {"map": [], "mrr": [],
           "recall": {k: [] for k in EVAL_K_VALUES["recall"]},
           "ndcg": {k: [] for k in EVAL_K_VALUES["ndcg"]},
           "hr": {k: [] for k in EVAL_K_VALUES["hr"]}}

    for i, qid in enumerate(tqdm(q_ids, desc=f"  Eval {label_key}")):
        retrieved = [cand_pids[idx] for idx in indices[i]]
        retrieved = [pid for pid in retrieved if pid != qid]
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
    log("LitFM — Evaluation")
    log("=" * 80)

    eval_df = pl.read_parquet(EVAL_PARQUET)

    all_results = {}
    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Evaluating: {label_cfg['name']}")
        log(f"{'=' * 80}")
        result = evaluate_label_type(label_key, eval_df)
        if result:
            all_results[label_key] = result
            log(f"  MAP: {result['mean_map']:.6f}  MRR: {result['mean_mrr']:.6f}")

    if not all_results:
        log("No label types evaluated.")
        return

    # Save txt
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\nLitFM GRAPH RETRIEVER EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write("Method: LitFM Graph Retriever (InfoNCE + L1 Reconstruction)\n")
        f.write("Base embeddings: BERT-base-uncased (frozen)\n")
        f.write("Retrieval pool: train set only\n\n")
        for key, cfg in LABEL_TYPES.items():
            if key not in all_results: continue
            res = all_results[key]
            f.write(f"--- {cfg['name']} ({cfg['description']}) ---\n")
            f.write(f"Queries: {res['num_queries']}\nMAP: {res['mean_map']:.6f}  MRR: {res['mean_mrr']:.6f}\n")
            for k in EVAL_K_VALUES["ndcg"]: f.write(f"  nDCG@{k}: {res['mean_ndcg'][k]:.6f}\n")
            for k in EVAL_K_VALUES["recall"]: f.write(f"  Recall@{k}: {res['mean_recall'][k]:.6f}\n")
            for k in EVAL_K_VALUES["hr"]: f.write(f"  HR@{k}: {res['mean_hr'][k]:.6f}\n")
            f.write(f"  MAP std: {res['map_std']:.6f}  MRR std: {res['mrr_std']:.6f}\n\n")

    # Save json
    results_dict = {
        "model_info": {
            "method": "LitFM Graph Retriever (InfoNCE + L1 Reconstruction)",
            "base_embeddings": "bert-base-uncased (frozen)",
            "text_source": "title + abstract", "retrieval_pool": "train set only",
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
                "statistics": {"map_std": res["map_std"], "mrr_std": res["mrr_std"]},
            }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE} and {RESULTS_JSON}")


if __name__ == "__main__":
    main()
