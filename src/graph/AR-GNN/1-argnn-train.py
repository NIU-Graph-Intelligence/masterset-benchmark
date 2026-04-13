"""
AR-GNN (AR-GATK) — Training Script
=====================================
Based on: Kammari & Durga Bhavani S, CODS-COMAD Dec 2024

Trains a 2-layer GAT + FCN on a heterogeneous graph (Paper/Venue/Topic)
with distance-based loss pulling embeddings toward seed papers.

One shared graph is built (ALL references create edges), but seed papers
are selected separately per label type (papers with most positive refs).
Thus 3 separate models are trained.

Resume: checkpoints saved every 20 epochs.

Output (per label type):
  - output/graph/AR-GNN/fine_tuned_models/<type_key>/argnn_final.pt
  - output/graph/AR-GNN/graph_cache.pkl (shared)
"""

import os
import sys
import json
import time
import pickle
import argparse
import torch
from torch.optim import Adam
from pathlib import Path
from dotenv import load_dotenv

from argnn_utils import (
    log, HeterogeneousGraph, ARGATK, distance_loss, select_seeds,
    build_feature_matrix, build_edge_index, build_keyword_communities,
    LABEL_TYPES, IN_DIM, HIDDEN, OUT_DIM, GAT_HEADS, LR, EPOCHS,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

ARGNN_DIR = OUTPUT_DIR / "graph/AR-GNN/"
ARGNN_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_CACHE = ARGNN_DIR / "graph_cache.pkl"
MODELS_DIR = ARGNN_DIR / "fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_or_load_graph(rebuild=False):
    """Build graph once (shared across all label types), cache to disk."""
    if GRAPH_CACHE.exists() and not rebuild:
        log(f"Loading cached graph from {GRAPH_CACHE}...")
        with open(GRAPH_CACHE, "rb") as f:
            return pickle.load(f)

    log("Building heterogeneous graph...")
    graph = HeterogeneousGraph()
    graph.build(TRAIN_PARQUET, EVAL_PARQUET)

    topic_community = build_keyword_communities(graph.paper_topics)
    X = build_feature_matrix(graph, topic_community)
    edge_index = build_edge_index(graph)

    cache = {"graph": graph, "topic_community": topic_community,
             "X": X, "edge_index": edge_index}

    log(f"Saving graph cache to {GRAPH_CACHE}...")
    with open(GRAPH_CACHE, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return cache


def train_for_label(label_key, cache):
    cfg = LABEL_TYPES[label_key]
    model_dir = MODELS_DIR / label_key
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    final_path = model_dir / "argnn_final.pt"
    log_file = model_dir / "training_log.json"

    if final_path.exists():
        log(f"  Final model exists — skipping.")
        return

    graph = cache["graph"]
    X = cache["X"].to(DEVICE)
    edge_index = cache["edge_index"].to(DEVICE)

    # Select seeds for THIS label type
    seed_nids = select_seeds(TRAIN_PARQUET, graph, label_key, n_seeds=10)
    if not seed_nids:
        log(f"  WARNING: No seeds found. Skipping.")
        return

    model = ARGATK(in_dim=IN_DIM, hidden=HIDDEN, out_dim=OUT_DIM, heads=GAT_HEADS).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5)

    # Resume
    start_epoch = 0
    training_log = {"epoch_losses": [], "lr": LR, "epochs": EPOCHS}

    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        latest = checkpoints[-1]
        log(f"  Resuming from {latest}")
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"]
        training_log = ckpt.get("training_log", training_log)

    if start_epoch >= EPOCHS:
        log(f"  Training already complete.")
        return

    log(f"  Training epochs {start_epoch + 1}–{EPOCHS} on {DEVICE}")
    model.train()

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time()
        optimizer.zero_grad()

        embeddings = model(X, edge_index)
        loss = distance_loss(embeddings, seed_nids) / embeddings.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step(loss)

        loss_val = loss.item()
        training_log["epoch_losses"].append(loss_val)

        if epoch % 10 == 0 or epoch == 1:
            log(f"    Epoch {epoch:4d}/{EPOCHS} | loss={loss_val:.6f} | {time.time() - t0:.2f}s")

        if epoch % 20 == 0 or epoch == EPOCHS:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "training_log": training_log,
            }, ckpt_dir / f"checkpoint_epoch_{epoch}.pt")
            # Keep last 2
            all_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                               key=lambda p: int(p.stem.split("_")[-1]))
            for old in all_ckpts[:-2]: old.unlink(missing_ok=True)

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "in_dim": IN_DIM, "hidden": HIDDEN, "out_dim": OUT_DIM, "heads": GAT_HEADS,
        "training_log": training_log,
    }, final_path)

    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=2)

    log(f"  Final model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-graph", action="store_true")
    args = parser.parse_args()

    log("=" * 80)
    log("AR-GNN (AR-GATK) — Training (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    cache = build_or_load_graph(rebuild=args.rebuild_graph)

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Training for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        train_for_label(label_key, cache)

    log("\nAll training complete!")


if __name__ == "__main__":
    main()
