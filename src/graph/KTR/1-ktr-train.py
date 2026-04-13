"""
KTR — Knowledge-Aware Topic Reasoning Training Script
========================================================
Based on: Wu et al., IEEE TKDE 2024

Trains KTR model per label type:
  A) KCF: Extract ConceptNet concepts from paper texts
  B) KTC: Build topic clusters via r-hop ConceptNet neighborhoods
  C) CSA: Dual GCN (content + structure) with contrastive alignment
  D) RCR: BCE loss for positive/negative citation pairs

Requires: CONCEPTNET_DB env var pointing to ConceptNet SQLite DB.

ALL references create citation edges (for structure GCN).
Only POSITIVE references (per label threshold) are training signal.

Resume: checkpoint per epoch, early stopping.

Output (per label type):
  - output/graph/KTR/fine_tuned_models/<type_key>/ktr_best/model.pt
"""

import os
import sys
import json
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv

from ktr_utils import (
    log, load_paper_text_map,
    load_conceptnet_concepts, load_conceptnet_graph,
    preprocess_papers, build_topic_clusters, build_citation_network,
    KTRModel, ContrastiveLoss, KTRDataset,
    compute_scibert_embeddings,
    LABEL_TYPES, EMBED_DIM, HIDDEN_DIM, NUM_TOPICS, TOPIC_RADIUS,
    NA_BATCH, NEG_RATIO, CONTRASTIVE_M, LAMBDA_C, EPOCHS, LR, PATIENCE,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
CONCEPTNET_DB = Path(os.getenv("CONCEPTNET_DB", "conceptnet.db"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

KTR_DIR = OUTPUT_DIR / "graph/KTR/"
KTR_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESS_DIR = KTR_DIR / "preprocessed/"
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = KTR_DIR / "fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_paper_data(pid, paper_concepts, paper_topics, cn_graph, scibert_embs):
    """Prepare input data dict for a single paper."""
    concepts = paper_concepts.get(pid, [])
    if not concepts:
        return None

    zero_emb = torch.zeros(EMBED_DIM)

    # Concept embeddings from SciBERT cache (use paper embedding as fallback)
    concept_embs = []
    for c in concepts:
        concept_embs.append(scibert_embs.get(pid, zero_emb))  # Approximate: use paper's embedding
    concept_embs = torch.stack(concept_embs) if concept_embs else zero_emb.unsqueeze(0)

    # Topic indices and adjacency
    topics = paper_topics.get(pid, {"main": [], "side": []})
    topic_indices = []
    topic_adj = []

    for cluster in topics.get("main", [])[:NUM_TOPICS]:
        indices = [i for i, c in enumerate(concepts) if c in cluster]
        adj_list = []
        for i in indices:
            for j in indices:
                if i != j and concepts[i] in cn_graph.get(concepts[j], set()):
                    adj_list.append((indices.index(i) if i in indices else 0,
                                     indices.index(j) if j in indices else 0))
        topic_indices.append(indices)
        topic_adj.append(adj_list)

    # Pad to NUM_TOPICS
    while len(topic_indices) < NUM_TOPICS:
        topic_indices.append([])
        topic_adj.append([])

    return {
        "concept_embs": concept_embs,
        "topic_indices": topic_indices,
        "topic_adj": topic_adj,
    }


def train_for_label(label_key, paper_text_map, paper_concepts, paper_topics,
                    cn_graph, citation_graph, scibert_embs, id_to_idx, num_papers):
    cfg = LABEL_TYPES[label_key]
    model_dir = MODELS_DIR / label_key
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dir = model_dir / "ktr_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / "model.pt"
    log_file = model_dir / "training_log.json"

    if best_path.exists():
        log(f"  Best model exists — skipping.")
        return

    # Dataset
    dataset = KTRDataset(TRAIN_PARQUET, paper_concepts, label_key, paper_text_map)
    if len(dataset) == 0:
        log(f"  WARNING: No samples. Skipping.")
        return

    dataloader = DataLoader(dataset, batch_size=NA_BATCH, shuffle=True,
                            num_workers=0, collate_fn=lambda x: x)

    # Model
    model = KTRModel(num_papers=num_papers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    contrastive_criterion = ContrastiveLoss()

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    training_log = {"epoch_losses": []}

    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        log(f"  Resuming from {latest}")
        ckpt = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        training_log = ckpt.get("training_log", training_log)
        best_loss = ckpt.get("best_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_samples in tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{EPOCHS}", leave=False):
            try:
                batch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                all_content_embs = []
                n_pairs = 0

                for sample in batch_samples:
                    alpha_data = prepare_paper_data(
                        sample["paper_id"], paper_concepts, paper_topics, cn_graph, scibert_embs)
                    pos_data = prepare_paper_data(
                        sample["pos_id"], paper_concepts, paper_topics, cn_graph, scibert_embs)

                    if alpha_data is None or pos_data is None:
                        continue

                    pos_score, content_a, _ = model.forward_pair(alpha_data, pos_data)
                    ce_loss = -torch.log(pos_score.clamp(min=1e-8))

                    # Negative pairs
                    neg_loss = torch.tensor(0.0, device=DEVICE)
                    for neg_id in sample["neg_ids"][:NEG_RATIO]:
                        neg_data = prepare_paper_data(
                            neg_id, paper_concepts, paper_topics, cn_graph, scibert_embs)
                        if neg_data is None: continue
                        neg_score, _, _ = model.forward_pair(alpha_data, neg_data)
                        neg_loss = neg_loss - torch.log((1 - neg_score).clamp(min=1e-8))

                    pair_loss = (ce_loss + neg_loss) / (1 + len(sample["neg_ids"]))
                    batch_loss = batch_loss + pair_loss
                    if content_a is not None:
                        all_content_embs.append(content_a)
                    n_pairs += 1

                if n_pairs == 0: continue
                batch_loss = batch_loss / n_pairs

                # Contrastive loss (CSA)
                cl_loss = torch.tensor(0.0, device=DEVICE)
                if len(all_content_embs) > 1:
                    z_c = torch.stack(all_content_embs)
                    n_b = z_c.size(0)
                    struct_ids = []
                    for s in batch_samples:
                        idx = id_to_idx.get(s["paper_id"], 0)
                        struct_ids.append(idx)
                    struct_ids = torch.tensor(struct_ids[:n_b], dtype=torch.long, device=DEVICE)

                    # Local adjacency
                    adj_i, adj_j, adj_v = [], [], []
                    sample_pids = [s["paper_id"] for s in batch_samples][:n_b]
                    for i, pid in enumerate(sample_pids):
                        cited = citation_graph.get(pid, set())
                        for j, pid2 in enumerate(sample_pids):
                            if pid2 in cited:
                                adj_i.append(i); adj_j.append(j); adj_v.append(1.0)
                        adj_i.append(i); adj_j.append(i); adj_v.append(1.0)

                    adj_sparse = torch.sparse_coo_tensor(
                        torch.tensor([adj_i, adj_j]), torch.tensor(adj_v), (n_b, n_b)).to(DEVICE)
                    z_s = model.structure_gcn(struct_ids, adj_sparse)

                    M = min(CONTRASTIVE_M, n_b - 1)
                    if M > 0:
                        neg_z_c, neg_z_s = [], []
                        for i in range(n_b):
                            indices = list(range(n_b)); indices.remove(i)
                            neg_idx = random.sample(indices, min(M, len(indices)))
                            neg_z_c.append(z_c[neg_idx])
                            neg_z_s.append(z_s[neg_idx])
                        neg_z_c = torch.stack([F.pad(t, (0, 0, 0, M - t.size(0))) for t in neg_z_c])
                        neg_z_s = torch.stack([F.pad(t, (0, 0, 0, M - t.size(0))) for t in neg_z_s])
                        cl_loss = contrastive_criterion(z_c, z_s, neg_z_c, neg_z_s)

                total_loss = batch_loss + LAMBDA_C * cl_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            except Exception as e:
                continue

        avg_loss = epoch_loss / max(n_batches, 1)
        log(f"  Epoch {epoch + 1}/{EPOCHS} — Loss: {avg_loss:.6f}")
        training_log["epoch_losses"].append(avg_loss)

        # Checkpoint
        torch.save({
            "epoch": epoch + 1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_log": training_log, "best_loss": best_loss,
            "patience_counter": patience_counter,
        }, ckpt_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")

        # Keep last 3
        all_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        for old in all_ckpts[:-3]: old.unlink(missing_ok=True)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"num_papers": num_papers, "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM},
                "id_to_idx": id_to_idx,
            }, best_path)
            log(f"  Best model saved (loss: {best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log(f"  Early stopping at epoch {epoch + 1}")
                break

    with open(log_file, "w") as f:
        json.dump(training_log, f, indent=2)


def main():
    log("=" * 80)
    log("KTR — Knowledge-Aware Topic Reasoning Training (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    if not CONCEPTNET_DB.exists():
        log(f"ERROR: ConceptNet DB not found at {CONCEPTNET_DB}")
        log("Set CONCEPTNET_DB env var to the correct path.")
        return

    # Load paper texts
    log("\nLoading paper texts...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Papers: {len(paper_text_map)}")

    # ID mapping
    all_ids = sorted(paper_text_map.keys())
    id_to_idx = {pid: idx for idx, pid in enumerate(all_ids)}
    num_papers = len(all_ids)

    # Preprocessing (shared)
    log("\nLoading ConceptNet...")
    cn_concepts = load_conceptnet_concepts(CONCEPTNET_DB)
    cn_graph = load_conceptnet_graph(CONCEPTNET_DB)

    log("\nPreprocessing papers (KCF)...")
    paper_concepts = preprocess_papers(paper_text_map, cn_concepts,
                                        cache_path=PREPROCESS_DIR / "paper_concepts.pkl")

    log("\nBuilding topic clusters (KTC)...")
    paper_topics = build_topic_clusters(paper_concepts, cn_graph,
                                         cache_path=PREPROCESS_DIR / "paper_topics.pkl")

    log("\nBuilding citation network...")
    citation_graph = build_citation_network(TRAIN_PARQUET, EVAL_PARQUET)

    log("\nComputing SciBERT embeddings...")
    scibert_embs = compute_scibert_embeddings(paper_text_map,
                                              cache_path=PREPROCESS_DIR / "scibert_embeddings.pkl",
                                              device=DEVICE)

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Training for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        train_for_label(label_key, paper_text_map, paper_concepts, paper_topics,
                        cn_graph, citation_graph, scibert_embs, id_to_idx, num_papers)

    log("\nAll training complete!")


if __name__ == "__main__":
    main()
