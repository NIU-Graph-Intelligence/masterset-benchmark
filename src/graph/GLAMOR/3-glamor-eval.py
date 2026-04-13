"""
GLAMOR — Evaluation Script
=============================
Evaluates the single GLAMOR model on all 3 label types.
GLAMOR is unsupervised (MLM), so one embedding set is evaluated
against all 3 ground truth label definitions.

FAISS cosine retrieval from train set only.
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

from glamor_utils import (
    log, load_paper_title_map, extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

GLAMOR_DIR = OUTPUT_DIR / "graph/GLAMOR/"
EMBED_DIR = GLAMOR_DIR / "embeddings/"

RESULTS_DIR = GLAMOR_DIR / "evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "glamor_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "glamor_evaluation_metrics.json"


def main():
    log("=" * 80)
    log("GLAMOR — Evaluation (3 Label Types)")
    log("=" * 80)

    # Load embeddings
    required = ["train_embeddings.pt", "eval_embeddings.pt",
                 "train_paper_id_to_index.json", "eval_paper_id_to_index.json"]
    for f in required:
        if not (EMBED_DIR / f).exists():
            log(f"Missing {f} — run embedding generation first.")
            return

    train_data = torch.load(EMBED_DIR / "train_embeddings.pt", map_location="cpu", weights_only=False)
    train_embeddings = train_data["embeddings"].numpy()
    train_paper_ids = train_data["paper_ids"]

    eval_data = torch.load(EMBED_DIR / "eval_embeddings.pt", map_location="cpu", weights_only=False)
    eval_embeddings = eval_data["embeddings"].numpy()

    with open(EMBED_DIR / "eval_paper_id_to_index.json", "r") as f:
        eval_id_to_idx = json.load(f)

    log(f"  Train: {len(train_paper_ids)}, Eval: {eval_embeddings.shape[0]}")

    # Load eval ground truth
    eval_df = pl.read_parquet(EVAL_PARQUET)

    # Build ground truth for all 3 label types
    ground_truth = {key: {} for key in LABEL_TYPES}
    for row in eval_df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in eval_id_to_idx: continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        rel_sets = extract_relevant_sets(refs)
        for key in LABEL_TYPES:
            if rel_sets[key]:
                ground_truth[key][pid] = rel_sets[key]

    # Normalize & FAISS
    train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_norm = train_embeddings / train_norms

    eval_norms = np.linalg.norm(eval_embeddings, axis=1, keepdims=True)
    eval_norms[eval_norms == 0] = 1.0
    eval_norm = eval_embeddings / eval_norms

    faiss_index = faiss.IndexFlatIP(train_norm.shape[1])
    faiss_index.add(train_norm.astype("float32"))

    # Collect all query IDs across all label types
    all_query_ids = set()
    for key in LABEL_TYPES:
        all_query_ids.update(ground_truth[key].keys())
    all_query_ids = sorted(all_query_ids)

    # Single FAISS retrieval for all queries
    query_embs = np.array([eval_norm[eval_id_to_idx[pid]] for pid in all_query_ids]).astype("float32")
    max_k = max(EVAL_K_VALUES["recall"])
    distances, indices = faiss_index.search(query_embs, max_k)

    # Build retrieved lists
    retrieved_map = {}
    for i, qid in enumerate(all_query_ids):
        retrieved = [train_paper_ids[idx] for idx in indices[i]]
        retrieved = [pid for pid in retrieved if pid != qid]
        retrieved_map[qid] = retrieved

    # Evaluate per label type
    all_results = {}
    for key, cfg in LABEL_TYPES.items():
        gt = ground_truth[key]
        if not gt:
            continue

        acc = {"map": [], "mrr": [],
               "recall": {k: [] for k in EVAL_K_VALUES["recall"]},
               "ndcg": {k: [] for k in EVAL_K_VALUES["ndcg"]},
               "hr": {k: [] for k in EVAL_K_VALUES["hr"]}}

        for qid in tqdm(gt, desc=f"  Eval {key}"):
            if qid not in retrieved_map: continue
            metrics = compute_all_metrics(gt[qid], retrieved_map[qid])
            acc["map"].append(metrics["map"])
            acc["mrr"].append(metrics["mrr"])
            for k in EVAL_K_VALUES["recall"]: acc["recall"][k].append(metrics["recall"][k])
            for k in EVAL_K_VALUES["ndcg"]: acc["ndcg"][k].append(metrics["ndcg"][k])
            for k in EVAL_K_VALUES["hr"]: acc["hr"][k].append(metrics["hr"][k])

        n_q = len(acc["map"])
        all_results[key] = {
            "num_queries": n_q,
            "mean_map": float(np.mean(acc["map"])),
            "mean_mrr": float(np.mean(acc["mrr"])),
            "mean_recall": {k: float(np.mean(v)) for k, v in acc["recall"].items()},
            "mean_ndcg": {k: float(np.mean(v)) for k, v in acc["ndcg"].items()},
            "mean_hr": {k: float(np.mean(v)) for k, v in acc["hr"].items()},
            "map_std": float(np.std(acc["map"])),
            "mrr_std": float(np.std(acc["mrr"])),
        }
        log(f"  {cfg['name']}: MAP={all_results[key]['mean_map']:.6f} MRR={all_results[key]['mean_mrr']:.6f}")

    # Save txt
    with open(RESULTS_FILE, "w") as f:
        f.write("=" * 80 + "\nGLAMOR EVALUATION RESULTS (3 Label Types)\n" + "=" * 80 + "\n\n")
        f.write("Method: GLAMOR (DistilRoBERTa + KG Random Walks + MLM)\n")
        f.write("Note: Single unsupervised model evaluated on all label types\n\n")
        for key, cfg in LABEL_TYPES.items():
            if key not in all_results: continue
            res = all_results[key]
            f.write(f"--- {cfg['name']} ({cfg['description']}) ---\n")
            f.write(f"Queries: {res['num_queries']}\nMAP: {res['mean_map']:.6f}  MRR: {res['mean_mrr']:.6f}\n")
            for k in EVAL_K_VALUES["ndcg"]: f.write(f"  nDCG@{k}: {res['mean_ndcg'][k]:.6f}\n")
            for k in EVAL_K_VALUES["recall"]: f.write(f"  Recall@{k}: {res['mean_recall'][k]:.6f}\n")
            for k in EVAL_K_VALUES["hr"]: f.write(f"  HR@{k}: {res['mean_hr'][k]:.6f}\n")
            f.write("\n")

    # Save json
    results_dict = {
        "model_info": {
            "method": "GLAMOR (DistilRoBERTa + KG Random Walks + MLM)",
            "base_model": "distilroberta-base", "training": "unsupervised MLM",
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
                "statistics": {"map_std": res["map_std"], "mrr_std": res["mrr_std"]},
            }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results_dict, f, indent=2)

    log(f"\nResults saved to {RESULTS_FILE} and {RESULTS_JSON}")


if __name__ == "__main__":
    main()
