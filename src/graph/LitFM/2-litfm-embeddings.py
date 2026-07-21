"""
LitFM Graph Retriever — Embedding Generation Script
=====================================================
For each fine-tuned LitFM model:
  - Candidate embeddings: c_j + p_j for all TRAIN papers
  - Query embeddings: q_i for all EVAL papers

Resume: skips label types without models, skips existing embeddings.
"""

import os
import sys
import json
import random
import torch
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from litfm_utils import (
    log, load_paper_text_map, build_citation_graph, compute_bert_embeddings,
    LitFMRetriever, LABEL_TYPES, INPUT_DIM, HIDDEN_DIM, NEIGH_NUM,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = ROOT_DIR / "data/train_eval_set/v1.0/train.parquet"
EVAL_PARQUET = ROOT_DIR / "data/train_eval_set/v1.0/eval.parquet"
ALL_PAPERS_PARQUET = ROOT_DIR / "data/train_eval_set/v1.0/all_papers_with_refs_and_labels.parquet"

MODELS_DIR = OUTPUT_DIR / "graph/LitFM/fine_tuned_models/"
BERT_CACHE_DIR = OUTPUT_DIR / "graph/LitFM/bert_cache/"
EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "graph/LitFM/embeddings/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    log("=" * 80)
    log("LitFM — Embedding Generation")
    log("=" * 80)

    # Load paper text + BERT embeddings (shared)
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map

    bert_cache = BERT_CACHE_DIR / "bert_embeddings_cache.pkl"
    all_pids = set(paper_text_map.keys())
    bert_embeddings = compute_bert_embeddings(paper_text_map, all_pids, cache_path=bert_cache, device=DEVICE)

    zero_emb = torch.zeros(INPUT_DIM)

    # Train/eval paper IDs
    train_df = pl.read_parquet(TRAIN_PARQUET)
    eval_df = pl.read_parquet(EVAL_PARQUET)
    train_pids = train_df["paper_id"].to_list()
    eval_pids = eval_df["paper_id"].to_list()

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Processing: {label_cfg['name']}")
        log(f"{'=' * 80}")

        model_path = MODELS_DIR / label_key / "litfm_best.pt"
        if not model_path.exists():
            log(f"  Model not found — skipping.")
            continue

        emb_dir = EMBEDDINGS_BASE_DIR / label_key
        emb_dir.mkdir(parents=True, exist_ok=True)

        cand_path = emb_dir / "candidate_embeddings.pt"
        query_path = emb_dir / "query_embeddings.pt"

        if cand_path.exists() and query_path.exists():
            log(f"  Embeddings already exist — skipping.")
            continue

        # Load model
        model = LitFMRetriever(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()

        # Build adjacency for neighbor lookup (ALL references)
        adjacency, _, _ = build_citation_graph(TRAIN_PARQUET, label_key)

        # Generate candidate embeddings (train papers)
        log(f"  Generating candidate embeddings for {len(train_pids)} train papers...")
        cand_embs = []
        with torch.no_grad():
            for pid in tqdm(train_pids, desc="  Candidates"):
                z_c = bert_embeddings.get(pid, zero_emb).unsqueeze(0).to(DEVICE)
                neighbors = list(adjacency.get(pid, set()))
                if neighbors:
                    sampled = random.choices(neighbors, k=NEIGH_NUM) if len(neighbors) < NEIGH_NUM else random.sample(neighbors, NEIGH_NUM)
                    z_n = torch.stack([bert_embeddings.get(n, zero_emb) for n in sampled]).unsqueeze(0).to(DEVICE)
                else:
                    z_n = torch.zeros(1, NEIGH_NUM, INPUT_DIM, device=DEVICE)
                c_emb = model.get_candidate_embedding(z_c, z_n)
                cand_embs.append(c_emb.cpu())

        cand_embs = torch.cat(cand_embs, dim=0)
        torch.save({"embeddings": cand_embs, "paper_ids": train_pids}, cand_path)
        log(f"  Candidate embeddings: {cand_embs.shape}")

        # Generate query embeddings (eval papers)
        log(f"  Generating query embeddings for {len(eval_pids)} eval papers...")
        query_embs = []
        with torch.no_grad():
            for pid in tqdm(eval_pids, desc="  Queries"):
                z_q = bert_embeddings.get(pid, zero_emb).unsqueeze(0).to(DEVICE)
                q_emb = model.get_query_embedding(z_q)
                query_embs.append(q_emb.cpu())

        query_embs = torch.cat(query_embs, dim=0)
        torch.save({"embeddings": query_embs, "paper_ids": eval_pids}, query_path)
        log(f"  Query embeddings: {query_embs.shape}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()
