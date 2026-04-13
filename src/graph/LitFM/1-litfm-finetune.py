"""
LitFM Graph Retriever — Fine-Tuning Script
=============================================
Based on: Zhang et al., KDD '25

Trains a neighbor-aware graph retriever with InfoNCE + L1 reconstruction
loss, separately for each of the 3 label types.

Key fix from old code: ALL references create adjacency edges for neighbor
aggregation. Only POSITIVE references (per label threshold) are training edges.

Resume: saves resume checkpoint with full optimizer/scaler state.
Early stopping on validation Precision@5.

Output (per label type):
  - output/graph/LitFM/fine_tuned_models/<type_key>/litfm_best.pt
  - output/graph/LitFM/fine_tuned_models/<type_key>/training_log.json
"""

import os
import sys
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from litfm_utils import (
    log, load_paper_text_map, build_citation_graph, compute_bert_embeddings,
    LitFMRetriever, RetrieverTrainDataset,
    LABEL_TYPES, INPUT_DIM, HIDDEN_DIM, NEIGH_NUM, NEG_SAMPLES,
    BATCH_SIZE_TRAIN, EPOCHS, PATIENCE, LR, LAMBDA_RE, EDGE_SPLIT_SEED, SEP,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

MODELS_DIR = OUTPUT_DIR / "graph/LitFM/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Shared BERT embedding cache (same for all label types)
BERT_CACHE_DIR = OUTPUT_DIR / "graph/LitFM/bert_cache/"
BERT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, val_edges, adjacency, bert_embeddings, all_node_ids, device):
    """Quick validation: Precision@5 on held-out edges."""
    model.eval()
    hits = 0
    total = 0
    k = 5

    # Build a quick lookup: for each source in val_edges, collect positive targets
    val_positives = {}
    for src, dst in val_edges:
        val_positives.setdefault(src, set()).add(dst)

    all_node_list = list(all_node_ids)
    zero_emb = torch.zeros(INPUT_DIM)

    with torch.no_grad():
        for src, positive_set in list(val_positives.items())[:200]:  # limit for speed
            z_q = bert_embeddings.get(src, zero_emb).unsqueeze(0).to(device)
            q_emb = model.get_query_embedding(z_q)

            # Score a sample of candidates
            sample_size = min(500, len(all_node_list))
            candidates = random.sample(all_node_list, sample_size)
            # Ensure positives are included
            candidates = list(set(candidates) | positive_set)

            scores = []
            for cid in candidates:
                z_c = bert_embeddings.get(cid, zero_emb).unsqueeze(0).to(device)
                neighbors = list(adjacency.get(cid, set()))
                if neighbors:
                    sampled = random.choices(neighbors, k=NEIGH_NUM) if len(neighbors) < NEIGH_NUM else random.sample(neighbors, NEIGH_NUM)
                    z_n = torch.stack([bert_embeddings.get(n, zero_emb) for n in sampled]).unsqueeze(0).to(device)
                else:
                    z_n = torch.zeros(1, NEIGH_NUM, INPUT_DIM, device=device)
                c_emb = model.get_candidate_embedding(z_c, z_n)
                score = F.cosine_similarity(q_emb, c_emb, dim=-1).item()
                scores.append((cid, score))

            scores.sort(key=lambda x: -x[1])
            top_k_ids = [cid for cid, _ in scores[:k]]
            hits += sum(1 for cid in top_k_ids if cid in positive_set)
            total += min(k, len(positive_set))

    p_at_5 = hits / max(total, 1)
    model.train()
    return p_at_5


def fine_tune_for_label(label_type_key, paper_text_map, bert_embeddings):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "litfm_best.pt"
    resume_path = model_dir / "litfm_resume.pt"
    log_file = model_dir / "training_log.json"

    if best_model_path.exists() and not resume_path.exists():
        log(f"  Best model already exists — skipping.")
        return

    # Build citation graph
    log(f"  Building citation graph for {cfg['name']}...")
    adjacency, positive_edges, all_nodes = build_citation_graph(TRAIN_PARQUET, label_type_key)

    if len(positive_edges) == 0:
        log(f"  WARNING: No positive edges. Skipping.")
        return

    # Split edges: 70/15/15
    random.seed(EDGE_SPLIT_SEED)
    random.shuffle(positive_edges)
    n = len(positive_edges)
    train_edges = positive_edges[:int(0.7 * n)]
    val_edges = positive_edges[int(0.7 * n):int(0.85 * n)]
    log(f"  Train edges: {len(train_edges)}, Val edges: {len(val_edges)}")

    # Dataset
    dataset = RetrieverTrainDataset(train_edges, adjacency, all_nodes, bert_embeddings)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    model = LitFMRetriever(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_val = 0.0
    patience_counter = 0
    training_log = {"epoch_losses": [], "val_p5": []}

    if resume_path.exists():
        log(f"  Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val = ckpt["best_val_p5"]
        patience_counter = ckpt["patience_counter"]
        training_log = ckpt.get("training_log", training_log)
        log(f"  Resuming from epoch {start_epoch}, best P@5={best_val:.4f}")

    model.train()

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        epoch_loss_nce = 0.0
        epoch_loss_re = 0.0
        num_batches = 0

        for z_q, z_t, z_n, z_nt, z_nn in tqdm(dataloader, desc=f"  Epoch {epoch}", leave=False):
            z_q, z_t, z_n = z_q.to(DEVICE), z_t.to(DEVICE), z_n.to(DEVICE)
            z_nt, z_nn = z_nt.to(DEVICE), z_nn.to(DEVICE)

            with autocast():
                l_nce, l_re = model(z_q, z_t, z_n, z_nt, z_nn)
                loss = l_nce + LAMBDA_RE * l_re

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_nce += l_nce.item()
            epoch_loss_re += l_re.item()
            num_batches += 1

        avg_nce = epoch_loss_nce / max(num_batches, 1)
        avg_re = epoch_loss_re / max(num_batches, 1)

        # Validation
        val_p5 = validate(model, val_edges, adjacency, bert_embeddings, all_nodes, DEVICE)

        training_log["epoch_losses"].append({"nce": avg_nce, "re": avg_re})
        training_log["val_p5"].append(val_p5)

        log(f"  Epoch {epoch}: NCE={avg_nce:.4f} RE={avg_re:.4f} Val P@5={val_p5:.4f}")

        # Early stopping
        if val_p5 > best_val:
            best_val = val_p5
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            log(f"  New best model saved (P@5={best_val:.4f})")
        else:
            patience_counter += 1

        # Save resume checkpoint
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_p5": best_val, "patience_counter": patience_counter,
            "training_log": training_log,
        }, resume_path)

        if patience_counter >= PATIENCE:
            log(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # Cleanup resume checkpoint
    if resume_path.exists():
        resume_path.unlink()

    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=2)

    log(f"  Training complete. Best P@5={best_val:.4f}")


def main():
    log("=" * 80)
    log("LitFM Graph Retriever — Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    # Load paper text for BERT embeddings
    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Combined: {len(paper_text_map)} papers")

    # Compute BERT embeddings (cached, shared across all label types)
    bert_cache = BERT_CACHE_DIR / "bert_embeddings_cache.pkl"
    all_pids = set(paper_text_map.keys())
    bert_embeddings = compute_bert_embeddings(paper_text_map, all_pids, cache_path=bert_cache, device=DEVICE)

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map, bert_embeddings)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()
