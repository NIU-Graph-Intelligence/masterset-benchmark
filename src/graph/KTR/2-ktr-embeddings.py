"""
KTR — Embedding Generation Script
====================================
For each trained KTR model, generates paper embeddings using the
content branch (topic GCN → mean pooled topic embeddings).

Both train and eval papers are embedded.

Resume: skips label types without models, skips existing embeddings.
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from ktr_utils import (
    log, load_paper_text_map, KTRModel,
    load_conceptnet_graph, preprocess_papers, build_topic_clusters,
    compute_scibert_embeddings,
    LABEL_TYPES, EMBED_DIM, HIDDEN_DIM, NUM_TOPICS, MAX_CONCEPTS_PER_PAPER,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
CONCEPTNET_DB = Path(os.getenv("CONCEPTNET_DB", "conceptnet.db"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

KTR_DIR = OUTPUT_DIR / "graph/KTR/"
PREPROCESS_DIR = KTR_DIR / "preprocessed/"
MODELS_DIR = KTR_DIR / "fine_tuned_models/"
EMBEDDINGS_BASE_DIR = KTR_DIR / "embeddings/"
EMBEDDINGS_BASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_papers(model, paper_ids, paper_concepts, paper_topics, cn_graph, scibert_embs, device):
    """Generate embeddings for a list of papers using content branch."""
    model.eval()
    embeddings = []
    valid_pids = []
    zero_emb = torch.zeros(EMBED_DIM)

    with torch.no_grad():
        for pid in tqdm(paper_ids, desc="  Embedding"):
            concepts = paper_concepts.get(pid, [])
            if not concepts:
                # Fallback: use SciBERT embedding projected through concept_proj
                se = scibert_embs.get(pid, zero_emb).unsqueeze(0).to(device)
                emb = model.concept_proj(se).squeeze(0)
                embeddings.append(emb.cpu())
                valid_pids.append(pid)
                continue

            concept_embs = torch.stack([scibert_embs.get(pid, zero_emb) for _ in concepts]).to(device)
            topics = paper_topics.get(pid, {"main": [], "side": []})

            topic_indices = []
            topic_adj = []
            for cluster in topics.get("main", [])[:NUM_TOPICS]:
                indices = [i for i, c in enumerate(concepts) if c in cluster]
                topic_indices.append(indices)
                topic_adj.append([])
            while len(topic_indices) < NUM_TOPICS:
                topic_indices.append([]); topic_adj.append([])

            projected = model.concept_proj(concept_embs)
            topic_embs = model.get_topic_embeddings(projected, topic_indices, topic_adj)
            emb = topic_embs.mean(dim=0)
            embeddings.append(emb.cpu())
            valid_pids.append(pid)

    return valid_pids, torch.stack(embeddings)


def main():
    log("=" * 80)
    log("KTR — Embedding Generation")
    log("=" * 80)

    # Load shared preprocessing
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map

    cn_graph = load_conceptnet_graph(CONCEPTNET_DB) if CONCEPTNET_DB.exists() else {}
    cn_concepts = set(cn_graph.keys())

    paper_concepts = preprocess_papers(paper_text_map, cn_concepts,
                                        cache_path=PREPROCESS_DIR / "paper_concepts.pkl")
    paper_topics = build_topic_clusters(paper_concepts, cn_graph,
                                         cache_path=PREPROCESS_DIR / "paper_topics.pkl")
    scibert_embs = compute_scibert_embeddings(paper_text_map,
                                              cache_path=PREPROCESS_DIR / "scibert_embeddings.pkl",
                                              device=DEVICE)

    train_pids = pl.read_parquet(TRAIN_PARQUET)["paper_id"].to_list()
    eval_pids = pl.read_parquet(EVAL_PARQUET)["paper_id"].to_list()

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Processing: {label_cfg['name']}")
        log(f"{'=' * 80}")

        best_path = MODELS_DIR / label_key / "ktr_best" / "model.pt"
        if not best_path.exists():
            log(f"  Model not found — skipping.")
            continue

        emb_dir = EMBEDDINGS_BASE_DIR / label_key
        emb_dir.mkdir(parents=True, exist_ok=True)

        train_emb_path = emb_dir / "train_embeddings.pt"
        eval_emb_path = emb_dir / "eval_embeddings.pt"

        if train_emb_path.exists() and eval_emb_path.exists():
            log(f"  Embeddings exist — skipping.")
            continue

        log(f"  Loading model...")
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        config = ckpt.get("config", {})
        model = KTRModel(
            num_papers=config.get("num_papers", len(paper_text_map)),
            embed_dim=config.get("embed_dim", EMBED_DIM),
            hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Train embeddings
        if not train_emb_path.exists():
            log(f"  Embedding train papers...")
            t_pids, t_embs = embed_papers(model, train_pids, paper_concepts, paper_topics, cn_graph, scibert_embs, DEVICE)
            torch.save({"embeddings": t_embs, "paper_ids": t_pids}, train_emb_path)
            log(f"  Train: {t_embs.shape}")

        # Eval embeddings
        if not eval_emb_path.exists():
            log(f"  Embedding eval papers...")
            e_pids, e_embs = embed_papers(model, eval_pids, paper_concepts, paper_topics, cn_graph, scibert_embs, DEVICE)
            torch.save({"embeddings": e_embs, "paper_ids": e_pids}, eval_emb_path)
            log(f"  Eval: {e_embs.shape}")

        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()
