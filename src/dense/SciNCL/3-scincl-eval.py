"""
SciNCL — Evaluation Script
=============================
Evaluates each fine-tuned SciNCL model on its corresponding label type.
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

from scincl_utils import (
    log, load_paper_title_map,
    extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "dense/SciNCL/embeddings/"

RESULTS_DIR = OUTPUT_DIR / "dense/SciNCL/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "scincl_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "scincl_evaluation_metrics.json"


def evaluate_label_type(label_key, eval_df):
    cfg = LABEL_TYPES[label_key]
    emb_dir = EMBEDDINGS_BASE_DIR / label_key

    required = ["train_embeddings.pt", "eval_embeddings.pt",
                 "train_paper_id_to_index.json", "eval_paper_id_to_index.json"]
    for fname in required:
        if not (emb_dir / fname).exists():
            log(f"  Missing {fname} — skipping {cfg['name']}.")
            return None

    train_data = torch.load(emb_dir / "train_embeddings.pt", map_location="cpu", weights_only=False)
    train_embeddings = train_data["embeddings"].numpy()
    train_paper_ids = train_data["paper_ids"]

    eval_data = torch.load(emb_dir / "eval_embeddings.pt", map_location="cpu", weights_only=False)
    eval_embeddings = eval_data["embeddings"].numpy()

    with open(emb_dir / "eval_paper_id_to_index.json", "r") as f:
        eval_id_to_idx = json.load(f)

    log(f"  Train: {len(train_paper_ids)}, Eval: {eval_embeddings.shape[0]}")

    ground_truth = {}
    for row in eval_df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in eval_id_to_idx:
            continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant = extract_relevant_sets(refs)[label_key]
        if relevant:
            ground_truth[pid] = relevant

    log(f"  Queries with relevant: {len(ground_truth)}")
    if not ground_truth:
        return None

    # Normalize & FAISS
    train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_norm = train_embeddings / train_norms

    eval_norms = np.linalg.norm(eval_embeddings, axis=1, keepdims=True)
    eval_norms[eval_norms == 0] = 1.0
    eval_norm = eval_embeddings / eval_norms

    faiss_index = faiss.IndexFlatIP(train_norm.shape[1])
    faiss_index.add(train_norm.astype("float32"))

    query_embs, query_ids = [], []
    for pid in ground_truth:
        query_embs.append(eval_norm[eval_id_to_idx[pid]])
        query_ids.append(pid)
    query_embs = np.array(query_embs).astype("float32")

    max_k = max(EVAL_K_VALUES["recall"])
    distances, indices = faiss_index.search(query_embs, max_k)

    acc = {"map_scores": [], "mrr_scores": [],
           "recall_scores": {k: [] for k in EVAL_K_VALUES["recall"]},
           "ndcg_scores": {k: [] for k in EVAL_K_VALUES["ndcg"]},
           "hr_scores": {k: [] for k in EVAL_K_VALUES["hr"]}}

    for i, qid in enumerate(tqdm(query_ids, desc=f"  Eval {label_key}")):
        retrieved = [train_paper_ids[idx] for idx in indices[i]]
        retrieved = [pid for pid in retrieved if pid != qid]
        metrics = compute_all_metrics(ground_truth[qid], retrieved)
        acc["map_scores"].append(metrics["map"])
        acc["mrr_scores"].append(metrics["mrr"])
        for k in EVAL_K_VALUES["recall"]: acc["recall_scores"][k].append(metrics["recall"][k])
        for k in EVAL_K_VALUES["ndcg"]: acc["ndcg_scores"][k].append(metrics["ndcg"][k])
        for k in EVAL_K_VALUES["hr"]: acc["hr_scores"][k].append(metrics["hr"][k])

    n_q = len(acc["map_scores"])
    return {
        "num_queries": n_q,
        "mean_map": float(np.mean(acc["map_scores"])),
        "mean_mrr": float(np.mean(acc["mrr_scores"])),
        "mean_recall": {k: float(np.mean(v)) for k, v in acc["recall_scores"].items()},
        "mean_ndcg": {k: float(np.mean(v)) for k, v in acc["ndcg_scores"].items()},
        "mean_hr": {k: float(np.mean(v)) for k, v in acc["hr_scores"].items()},
        "map_std": float(np.std(acc["map_scores"])),
        "mrr_std": float(np.std(acc["mrr_scores"])),
    }


def main():
    log("=" * 80)
    log("SciNCL — Evaluation")
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
        f.write("=" * 80 + "\nSciNCL EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write("Method: SciNCL (SciBERT + KNN Neighborhood Contrastive Learning)\n")
        f.write("Triplet sampling: k+=25 positives, k_hard=4000 hard negatives, 3 easy negatives\n")
        f.write("Loss: Triplet Margin Loss (margin=1.0)\n")
        f.write("Retrieval pool: train set only\n\n")
        for key, cfg in LABEL_TYPES.items():
            if key not in all_results: continue
            res = all_results[key]
            f.write(f"--- {cfg['name']} ({cfg['description']}) ---\n")
            f.write(f"Queries: {res['num_queries']}\n")
            f.write(f"MAP: {res['mean_map']:.6f}  MRR: {res['mean_mrr']:.6f}\n")
            for k in EVAL_K_VALUES["ndcg"]: f.write(f"  nDCG@{k}: {res['mean_ndcg'][k]:.6f}\n")
            for k in EVAL_K_VALUES["recall"]: f.write(f"  Recall@{k}: {res['mean_recall'][k]:.6f}\n")
            for k in EVAL_K_VALUES["hr"]: f.write(f"  HR@{k}: {res['mean_hr'][k]:.6f}\n")
            f.write(f"  MAP std: {res['map_std']:.6f}  MRR std: {res['mrr_std']:.6f}\n\n")

    results_dict = {
        "model_info": {
            "method": "SciNCL (Neighborhood Contrastive Learning with Citation Embeddings)",
            "base_model": "allenai/scibert_scivocab_uncased",
            "representation": "[CLS] token",
            "loss": "Triplet Margin Loss (margin=1.0)",
            "sampling": "KNN: k+=25 positives, k_hard=4000 hard negatives, 3 filtered random easy negatives",
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
