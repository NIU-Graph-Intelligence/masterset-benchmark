import os
import time
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel

# ======================================================================
# Constants (matching SciNCL paper: Ostendorff et al., EMNLP 2022)
# ======================================================================
MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LEN = 512
BATCH_SIZE_FT = 8           # paper uses 8 with grad accum to 32
GRAD_ACCUM_STEPS = 4        # effective batch size = 32
BATCH_SIZE_EMB = 128
EPOCHS = 2                  # paper trains 2 epochs
LR = 2e-5                   # Adam with weight decay
MARGIN = 1.0                # triplet margin ξ = 1 (matches SPECTER)
SEP = "[SEP]"

# KNN sampling hyperparameters (best settings from paper §5, §6)
K_POSITIVE = 25             # positives from (k+ - 5, k+] = (20, 25] neighbors
C_POSITIVE = 5              # number of positives per query
K_HARD_NEG = 4000           # hard negatives from (k_hard - 2, k_hard] = (3998, 4000] neighbors
C_HARD_NEG = 2              # number of hard negatives per query
C_EASY_NEG = 3              # number of easy negatives (random, filtered)

TYPE_2_THRESHOLD = 4.0
TYPE_3_THRESHOLD = 3.0

LABEL_TYPES = {
    "type_1": {
        "name": "Type 1 (Binary Relevance)",
        "field": "type_1_output",
        "threshold_fn": lambda v: float(v) == 1.0,
        "description": "binary, label == 1",
    },
    "type_2": {
        "name": "Type 2 (Usefulness)",
        "field": "type_2_output",
        "threshold_fn": lambda v: float(v) >= TYPE_2_THRESHOLD,
        "description": f"usefulness >= {TYPE_2_THRESHOLD}",
    },
    "type_3": {
        "name": "Type 3 (Relatedness)",
        "field": "type_3_output",
        "threshold_fn": lambda v: float(v) >= TYPE_3_THRESHOLD,
        "description": f"relatedness >= {TYPE_3_THRESHOLD}",
    },
}

EVAL_K_VALUES = {
    "recall": [10, 50, 100, 500],
    "ndcg": [10, 20, 30, 50],
    "hr": [10, 20],
}


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ======================================================================
# Text Loading
# ======================================================================

def load_paper_text_map(parquet_file):
    df = pl.read_parquet(parquet_file)
    paper_map = {}
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        paper_map[pid] = f"{title} {SEP} {abstract}"
    return paper_map


def load_paper_title_map(parquet_file):
    try:
        df = pl.read_parquet(parquet_file)
        return {row["paper_id"]: row.get("title", "") for row in df.iter_rows(named=True)}
    except Exception as e:
        print(f"Warning: {e}")
        return {}


# ======================================================================
# SciNCL Model: SciBERT + [CLS] token (§3.1 of paper)
# ======================================================================

class SciNCLModel(nn.Module):
    """
    SciNCL uses SciBERT with [CLS] token as document representation.
    No projection head — raw [CLS] embedding is used with triplet margin loss.
    """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_emb


# ======================================================================
# Triplet Margin Loss (Equation 3.1 from paper)
# L = max(||d^Q - d^+||_2 - ||d^Q - d^-||_2 + ξ, 0)
# ======================================================================

class SciNCLTripletLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, query_emb, pos_emb, neg_emb):
        return self.loss_fn(query_emb, pos_emb, neg_emb)


# ======================================================================
# KNN-based Triplet Generation (§3.3, §3.4 of paper)
# ======================================================================

def generate_scincl_triplets(
    train_parquet, paper_text_map, citation_embeddings_path, label_type_key,
    k_pos=K_POSITIVE, c_pos=C_POSITIVE,
    k_hard_neg=K_HARD_NEG, c_hard_neg=C_HARD_NEG,
    c_easy_neg=C_EASY_NEG,
):
    """
    Generate SciNCL-style triplets using KNN on citation graph embeddings.

    For each query paper:
      - Positives: c_pos papers from (k_pos - c_pos, k_pos] nearest neighbors
      - Hard negatives: c_hard_neg papers from (k_hard_neg - c_hard_neg, k_hard_neg] neighbors
      - Easy negatives: c_easy_neg random papers NOT in the neighbor set

    The gap between k_pos and k_hard_neg is the "sample induced margin"
    that prevents collision between positives and negatives.

    Returns list of (query_text, pos_text, neg_text) triplets.
    """
    import faiss

    cfg = LABEL_TYPES[label_type_key]
    log(f"  Generating SciNCL triplets for {cfg['name']}...")

    # Load citation embeddings
    log(f"  Loading citation embeddings from {citation_embeddings_path}...")
    cit_data = torch.load(citation_embeddings_path, map_location="cpu", weights_only=False)
    cit_embeddings = cit_data["embeddings"].numpy().astype("float32")
    cit_paper_ids = cit_data["paper_ids"]  # list of paper_id strings

    # Build paper_id -> citation embedding index mapping
    cit_pid_to_idx = {pid: i for i, pid in enumerate(cit_paper_ids)}

    # Load train papers and filter to those with citation embeddings AND text
    df = pl.read_parquet(train_parquet)
    query_paper_ids = []
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid in cit_pid_to_idx and pid in paper_text_map:
            text = paper_text_map.get(pid, "")
            if text.strip():
                query_paper_ids.append(pid)

    log(f"  Query papers with citation embeddings: {len(query_paper_ids)}")

    # Build FAISS index over citation embeddings
    log(f"  Building FAISS index for KNN search...")
    dim = cit_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)

    # Normalize for cosine similarity
    norms = np.linalg.norm(cit_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cit_embeddings_norm = cit_embeddings / norms
    faiss_index.add(cit_embeddings_norm)

    # KNN search for all query papers
    max_k = max(k_pos, k_hard_neg) + 1  # +1 to exclude self
    log(f"  Running KNN search with max_k={max_k}...")

    query_indices = [cit_pid_to_idx[pid] for pid in query_paper_ids]
    query_vecs = cit_embeddings_norm[query_indices]
    _, knn_indices = faiss_index.search(query_vecs, max_k)

    # All corpus paper_ids (for easy negative sampling)
    all_corpus_ids = list(set(cit_paper_ids) & set(paper_text_map.keys()))

    # Generate triplets
    triplets = []
    rng = np.random.default_rng(42)

    for i, qid in enumerate(query_paper_ids):
        q_text = paper_text_map[qid]
        neighbors = knn_indices[i]

        # Exclude self (first neighbor is usually self)
        neighbor_pids = []
        for idx in neighbors:
            pid = cit_paper_ids[idx]
            if pid != qid:
                neighbor_pids.append(pid)

        if len(neighbor_pids) < max(k_pos, k_hard_neg):
            continue

        # Positives: from (k_pos - c_pos, k_pos] range
        pos_pids = neighbor_pids[k_pos - c_pos: k_pos]

        # Hard negatives: from (k_hard_neg - c_hard_neg, k_hard_neg] range
        hard_neg_pids = neighbor_pids[k_hard_neg - c_hard_neg: k_hard_neg]

        # Easy negatives: random papers NOT in neighbor set
        neighbor_set = set(neighbor_pids[:k_hard_neg])
        neighbor_set.add(qid)

        easy_neg_candidates = [pid for pid in all_corpus_ids if pid not in neighbor_set]
        if len(easy_neg_candidates) < c_easy_neg:
            easy_neg_pids = easy_neg_candidates
        else:
            easy_neg_pids = list(rng.choice(easy_neg_candidates, size=c_easy_neg, replace=False))

        # Create triplets: each positive paired with each negative
        all_neg_pids = hard_neg_pids + easy_neg_pids
        for pos_pid in pos_pids:
            pos_text = paper_text_map.get(pos_pid)
            if not pos_text or not pos_text.strip():
                continue
            for neg_pid in all_neg_pids:
                neg_text = paper_text_map.get(neg_pid)
                if not neg_text or not neg_text.strip():
                    continue
                triplets.append((q_text, pos_text, neg_text))

    log(f"  Generated {len(triplets)} triplets")
    return triplets


# ======================================================================
# Dataset wrapper for pre-generated triplets
# ======================================================================

class SciNCLTripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


# ======================================================================
# Embedding Dataset (per-paper)
# ======================================================================

class PaperTextDataset(Dataset):
    def __init__(self, paper_ids, paper_text_map, tokenizer, max_length=MAX_LEN):
        self.paper_ids = paper_ids
        self.paper_text_map = paper_text_map
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.paper_ids)

    def __getitem__(self, idx):
        pid = self.paper_ids[idx]
        text = self.paper_text_map.get(pid, "")
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "paper_id": pid,
        }


# ======================================================================
# Ground Truth & Metrics
# ======================================================================

def extract_relevant_sets(references):
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id:
            continue
        for key, cfg in LABEL_TYPES.items():
            raw_val = ref.get(cfg["field"])
            if raw_val is None:
                continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError):
                continue
    return relevant


def compute_map(relevant, retrieved):
    if not relevant: return 0.0
    score, num_hits = 0.0, 0
    for i, pid in enumerate(retrieved):
        if pid in relevant:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant)

def compute_mrr(relevant, retrieved):
    for i, pid in enumerate(retrieved):
        if pid in relevant: return 1.0 / (i + 1)
    return 0.0

def compute_recall_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    return sum(1 for pid in retrieved[:k] if pid in relevant) / len(relevant)

def compute_hit_rate_at_k(relevant, retrieved, k):
    return 1.0 if any(item in set(retrieved[:k]) for item in relevant) else 0.0

def compute_dcg_at_k(relevant, retrieved, k):
    dcg = 0.0
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant: dcg += 1.0 / np.log2(i + 2)
    return dcg

def compute_ndcg_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    dcg = compute_dcg_at_k(relevant, retrieved, k)
    idcg = compute_dcg_at_k(relevant, list(relevant) + [0]*k, k)
    return dcg / idcg if idcg > 0 else 0.0

def compute_all_metrics(relevant, retrieved):
    return {
        "map": compute_map(relevant, retrieved),
        "mrr": compute_mrr(relevant, retrieved),
        "recall": {k: compute_recall_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["recall"]},
        "ndcg": {k: compute_ndcg_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["ndcg"]},
        "hr": {k: compute_hit_rate_at_k(relevant, retrieved, k) for k in EVAL_K_VALUES["hr"]},
    }
