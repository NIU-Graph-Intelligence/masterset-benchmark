"""
HAtten — Evaluation Script
============================
Evaluates each fine-tuned HAtten model on its corresponding label type.
Uses cosine similarity for retrieval from train set only.

Resume: skips label types whose embeddings don't exist.

Output:
  - output/dense/HAtten-RR/evaluation_results/hatten_evaluation_metrics.txt
  - output/dense/HAtten-RR/evaluation_results/hatten_evaluation_metrics.json
"""

import os
import sys
import json
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from hatten_utils import (
    log, load_paper_title_map,
    extract_relevant_sets, compute_all_metrics,
    LABEL_TYPES, EVAL_K_VALUES, TYPE_2_THRESHOLD, TYPE_3_THRESHOLD,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "dense/HAtten-RR/embeddings/"

RESULTS_DIR = OUTPUT_DIR / "dense/HAtten-RR/evaluation_results/"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "hatten_evaluation_metrics.txt"
RESULTS_JSON = RESULTS_DIR / "hatten_evaluation_metrics.json"


def evaluate_label_type(label_key, eval_df):
    cfg = LABEL_TYPES[label_key]
    emb_dir = EMBEDDINGS_BASE_DIR / label_key

    required = ["train_embeddings.npy", "train_paper_ids.npy",
                 "eval_embeddings.npy", "eval_paper_ids.npy"]
    for fname in required:
        if not (emb_dir / fname).exists():
            log(f"  Missing {fname} — skipping {cfg['name']}.")
            return None

    # Load embeddings
    train_embeddings = np.load(emb_dir / "train_embeddings.npy")
    train_paper_ids = np.load(emb_dir / "train_paper_ids.npy", allow_pickle=True)
    eval_embeddings = np.load(emb_dir / "eval_embeddings.npy")
    eval_paper_ids = np.load(emb_dir / "eval_paper_ids.npy", allow_pickle=True)

    log(f"  Train: {train_embeddings.shape}, Eval: {eval_embeddings.shape}")

    # Build eval paper_id to index mapping
    eval_id_to_idx = {str(pid): i for i, pid in enumerate(eval_paper_ids)}

    # Build ground truth for THIS label type
    ground_truth = {}
    for row in eval_df.iter_rows(named=True):
        paper_id = row["paper_id"]
        if paper_id not in eval_id_to_idx:
            continue
        references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        relevant = extract_relevant_sets(references)[label_key]
        if relevant:
            ground_truth[paper_id] = relevant

    log(f"  Queries with relevant papers: {len(ground_truth)}")
    if not ground_truth:
        return None

    # Normalize (HAtten embeddings are already normalized, but be safe)
    train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_embeddings_norm = train_embeddings / train_norms

    # Build train paper_id lookup
    train_pid_list = [str(pid) for pid in train_paper_ids]

    max_k = max(EVAL_K_VALUES["recall"])

    acc = {
        "map_scores": [], "mrr_scores": [],
        "recall_scores": {k: [] for k in EVAL_K_VALUES["recall"]},
        "ndcg_scores": {k: [] for k in EVAL_K_VALUES["ndcg"]},
        "hr_scores": {k: [] for k in EVAL_K_VALUES["hr"]},
    }

    for qid in tqdm(ground_truth, desc=f"  Eval {label_key}"):
        idx = eval_id_to_idx[qid]
        query_emb = eval_embeddings[idx:idx+1]
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm

        # Cosine similarity with all train papers
        similarities = cosine_similarity(query_emb, train_embeddings_norm)[0]
        top_indices = np.argsort(-similarities)

        # Convert to paper IDs, remove self
        retrieved = []
        for tidx in top_indices:
            pid = train_pid_list[tidx]
            if pid != qid:
                retrieved.append(pid)
            if len(retrieved) >= max_k:
                break

        relevant = ground_truth[qid]
        metrics = compute_all_metrics(relevant, retrieved)

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
    log("HAtten — Evaluation")
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
        f.write("=" * 80 + "\nHATTEN EVALUATION RESULTS\n" + "=" * 80 + "\n\n")
        f.write("Method: HAtten (GloVe + Hierarchical Attention + Triplet Loss)\n")
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

    # Save json
    results_dict = {
        "model_info": {
            "method": "HAtten (Hierarchical Attention + Triplet Loss)",
            "embeddings": "GloVe 200d", "hidden_dim": 256,
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
