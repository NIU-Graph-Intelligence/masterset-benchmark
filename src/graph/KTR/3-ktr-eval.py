"""
KTR — Evaluation Script
=========================
Evaluates each trained KTR model on its corresponding label type.
FAISS cosine similarity retrieval from train set only.

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

from ktr_utils import (
    log, extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

KTR_DIR = OUTPUT_DIR / "graph/KTR/"
EMBEDDINGS_BASE_DIR = KTR_DIR / "embeddings/"

RESULTS_DIR = KTR_DIR / "evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "ktr_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "ktr_evaluation_metrics.json"


def evaluate_label_type(label_key, eval_df):
    cfg = LABEL_TYPES[label_key]
    emb_dir = EMBEDDINGS_BASE_DIR / label_key

    train_path = emb_dir / "train_embeddings.pt"
    eval_path = emb_dir / "eval_embeddings.pt"

    if not train_path.exists() or not eval_path.exists():
        log(f"  Missing embeddings for {cfg['name']} — skipping.")
        return None

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)
    train_embs = train_data["embeddings"].numpy()
    train_pids = train_data["paper_ids"]

    eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)
    eval_embs = eval_data["embeddings"].numpy()
    eval_pids = eval_data["paper_ids"]

    eval_pid_to_idx = {pid: i for i, pid in enumerate(eval_pids)}
    log(f"  Train: {train_embs.shape}, Eval: {eval_embs.shape}")

    # Ground truth
    ground_truth = {}
    for row in eval_df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in eval_pid_to_idx: continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant = extract_relevant_sets(refs)[label_key]
        if relevant:
            ground_truth[pid] = relevant

    log(f"  Queries with relevant: {len(ground_truth)}")
    if not ground_truth:
        return None

    # Normalize & FAISS
    t_norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    t_norms[t_norms == 0] = 1.0
    train_norm = train_embs / t_norms

    e_norms = np.linalg.norm(eval_embs, axis=1, keepdims=True)
    e_norms[e_norms == 0] = 1.0
    eval_norm = eval_embs / e_norms

    faiss_index = faiss.IndexFlatIP(train_norm.shape[1])
    faiss_index.add(train_norm.astype("float32"))

    q_embs, q_ids = [], []
    for pid in ground_truth:
        q_embs.append(eval_norm[eval_pid_to_idx[pid]])
        q_ids.append(pid)
    q_embs = np.array(q_embs).astype("float32")

    max_k = max(EVAL_K_VALUES["recall"])
    distances, indices = faiss_index.search(q_embs, max_k)

    acc = {"map": [], "mrr": [],
           "recall": {k: [] for k in EVAL_K_VALUES["recall"]},
           "ndcg": {k: [] for k in EVAL_K_VALUES["ndcg"]},
           "hr": {k: [] for k in EVAL_K_VALUES["hr"]}}

    for i, qid in enumerate(tqdm(q_ids, desc=f"  Eval {label_key}")):
        retrieved = [train_pids[idx] for idx in indices[i]]
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
    log("KTR — Evaluation")
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

    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\nKTR EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write("Method: KTR (ConceptNet Topic Reasoning + dual GCN + GRU paths)\n")
        f.write("Base embeddings: SciBERT (frozen)\n")
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

    results_dict = {
        "model_info": {
            "method": "KTR (Knowledge-Aware Topic Reasoning)",
            "base_embeddings": "SciBERT (frozen)", "knowledge_source": "ConceptNet",
            "modules": "KCF + KTC + CSA (TopicGCN + StructureGCN) + RCR (PathGRU)",
            "retrieval_pool": "train set only",
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
