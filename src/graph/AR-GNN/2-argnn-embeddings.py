"""
AR-GNN (AR-GATK) — Embedding Generation Script
=================================================
Loads each trained AR-GATK model and generates paper-only embeddings
(venue/topic node embeddings are discarded). Also computes quality scores.

Resume: skips label types without models, skips existing embeddings.
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from argnn_utils import (
    log, HeterogeneousGraph, ARGATK, compute_quality_scores,
    LABEL_TYPES, IN_DIM, HIDDEN, OUT_DIM, GAT_HEADS,
)

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARGNN_DIR = OUTPUT_DIR / "graph/AR-GNN/"
GRAPH_CACHE = ARGNN_DIR / "graph_cache.pkl"
MODELS_DIR = ARGNN_DIR / "fine_tuned_models/"
EMBEDDINGS_BASE_DIR = ARGNN_DIR / "embeddings/"
EMBEDDINGS_BASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    log("=" * 80)
    log("AR-GNN — Embedding Generation")
    log("=" * 80)

    if not GRAPH_CACHE.exists():
        log("ERROR: Graph cache not found. Run training first.")
        return

    log(f"Loading graph cache...")
    with open(GRAPH_CACHE, "rb") as f:
        cache = pickle.load(f)

    graph = cache["graph"]
    X = cache["X"].to(DEVICE)
    edge_index = cache["edge_index"].to(DEVICE)

    # Paper ID lists
    paper_ids_sorted = sorted(graph.paper_to_nid.keys())
    paper_nids = [graph.paper_to_nid[pid] for pid in paper_ids_sorted]

    # Quality scores (shared across label types — base scores with no query topics)
    _, qs_arr = compute_quality_scores(graph, query_topics=set())

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Processing: {label_cfg['name']}")
        log(f"{'=' * 80}")

        model_path = MODELS_DIR / label_key / "argnn_final.pt"
        if not model_path.exists():
            log(f"  Model not found — skipping.")
            continue

        emb_dir = EMBEDDINGS_BASE_DIR / label_key
        emb_dir.mkdir(parents=True, exist_ok=True)

        emb_pt = emb_dir / "argnn_embeddings.pt"
        if emb_pt.exists():
            log(f"  Embeddings already exist — skipping.")
            continue

        log(f"  Loading model...")
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model = ARGATK(
            in_dim=ckpt.get("in_dim", IN_DIM), hidden=ckpt.get("hidden", HIDDEN),
            out_dim=ckpt.get("out_dim", OUT_DIM), heads=ckpt.get("heads", GAT_HEADS),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        log(f"  Generating embeddings...")
        with torch.no_grad():
            all_embs = model(X, edge_index).cpu()

        paper_embs = all_embs[paper_nids]
        log(f"  Paper embeddings: {paper_embs.shape}")

        torch.save({
            "embeddings": paper_embs,
            "paper_ids": paper_ids_sorted,
            "quality_scores": qs_arr,
            "model_name": "AR-GATK",
            "embedding_dim": paper_embs.shape[1],
        }, emb_pt)

        # Also save paper_id_to_index mapping
        pid_to_idx = {pid: i for i, pid in enumerate(paper_ids_sorted)}
        with open(emb_dir / "paper_id_to_index.json", "w") as f:
            json.dump(pid_to_idx, f)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()
