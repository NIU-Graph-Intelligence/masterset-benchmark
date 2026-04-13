"""
SciNCL — Citation Graph Embedding Generation (Prerequisite Script)
====================================================================
Generates citation graph embeddings needed by SciNCL's KNN-based
triplet sampling. Uses Node2Vec on the citation graph.

The original SciNCL paper uses PyTorch BigGraph on S2ORC (52M nodes).
For our ~142K paper dataset, Node2Vec is sufficient and practical.

ALL references create edges (undirected) in the citation graph.

Output:
  - DATA_DIR/citation_embeddings.pt
    Format: {"embeddings": tensor [N, dim], "paper_ids": list of str}

Dependencies:
  pip install node2vec networkx --break-system-packages

Usage:
  python 0-scincl-generate-citation-embeddings.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
import polars as pl
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

OUTPUT_PATH = DATA_DIR / "citation_embeddings.pt"

# Node2Vec hyperparameters (matching BigGraph defaults roughly)
DIMENSIONS = 128      # embedding dimension
WALK_LENGTH = 30      # length of each random walk
NUM_WALKS = 20        # number of walks per node
P = 1.0               # return parameter (BFS-like if low)
Q = 1.0               # in-out parameter (DFS-like if low)
WORKERS = 8           # parallel workers
WINDOW = 10           # skip-gram window size
MIN_COUNT = 1         # minimum count for word2vec
EPOCHS_W2V = 5        # word2vec training epochs
SEED = 42


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_citation_graph():
    """Build undirected citation graph from ALL references in train+eval."""
    log("Building citation graph...")
    G = nx.Graph()

    for pq_path in [TRAIN_PARQUET, EVAL_PARQUET]:
        if not pq_path.exists():
            continue
        df = pl.read_parquet(pq_path)
        for row in df.iter_rows(named=True):
            pid = row["paper_id"]
            G.add_node(pid)
            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
            for ref in references:
                matched_id = ref.get("matched_paper_id")
                if matched_id:
                    G.add_edge(pid, matched_id)

    log(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def main():
    log("=" * 80)
    log("SciNCL — Citation Graph Embedding Generation (Node2Vec)")
    log("=" * 80)

    if OUTPUT_PATH.exists():
        log(f"Citation embeddings already exist at {OUTPUT_PATH}")
        log("Delete to regenerate.")
        return

    G = build_citation_graph()

    if G.number_of_nodes() == 0:
        log("ERROR: Empty graph. Check parquet files.")
        return

    # Node2Vec
    log(f"\nRunning Node2Vec (dim={DIMENSIONS}, walks={NUM_WALKS}, "
        f"walk_len={WALK_LENGTH}, p={P}, q={Q})...")

    try:
        from node2vec import Node2Vec
    except ImportError:
        log("ERROR: node2vec not installed. Run: pip install node2vec")
        return

    node2vec = Node2Vec(
        G, dimensions=DIMENSIONS, walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS, p=P, q=Q, workers=WORKERS, seed=SEED,
    )

    log("  Training Word2Vec model on random walks...")
    model = node2vec.fit(
        window=WINDOW, min_count=MIN_COUNT,
        batch_words=4, epochs=EPOCHS_W2V, seed=SEED,
    )

    # Extract embeddings in consistent order
    paper_ids = sorted(G.nodes())
    embeddings = []
    missing = 0

    for pid in paper_ids:
        try:
            emb = model.wv[pid]
            embeddings.append(torch.tensor(emb, dtype=torch.float32))
        except KeyError:
            embeddings.append(torch.zeros(DIMENSIONS, dtype=torch.float32))
            missing += 1

    embeddings_tensor = torch.stack(embeddings)
    log(f"\n  Embedding shape: {embeddings_tensor.shape}")
    if missing > 0:
        log(f"  WARNING: {missing} nodes missing from Word2Vec model")

    # Save
    torch.save({
        "embeddings": embeddings_tensor,
        "paper_ids": paper_ids,
        "method": "node2vec",
        "dimensions": DIMENSIONS,
        "walk_length": WALK_LENGTH,
        "num_walks": NUM_WALKS,
    }, OUTPUT_PATH)

    log(f"\nSaved citation embeddings to {OUTPUT_PATH}")
    log("Now you can run: python 1-scincl-finetune.py")


if __name__ == "__main__":
    main()
