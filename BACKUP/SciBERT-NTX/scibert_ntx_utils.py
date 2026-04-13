"""
scibert_ntx_utils.py
Shared utilities for the SciBERT NT-Xent fine-tuning pipeline.
Used by all 3 stages: fine-tuning, embedding generation, and evaluation.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from transformers import AutoModel
from pathlib import Path


# ──────────────────────────────────────────────
# Label configuration
# ──────────────────────────────────────────────
LABEL_TYPES = {
    "type_1": {
        "field": "type_1_output",
        "threshold": 1.0,   # binary: 1.0 → positive
        "comparison": "eq",  # equal to threshold
        "description": "Type I: Baseline/experimental comparison (binary)",
    },
    "type_2": {
        "field": "type_2_output",
        "threshold": 4.0,   # >= 4 → positive
        "comparison": "gte", # greater than or equal
        "description": "Type II: Relevance score (≥4 → must-cite)",
    },
    "type_3": {
        "field": "type_3_output",
        "threshold": 3.0,   # >= 3 → positive
        "comparison": "gte",
        "description": "Type III: Citation frequency (≥3 → must-cite)",
    },
}


def is_positive(ref: dict, label_type: str) -> bool:
    """Check if a reference is positive for a given label type."""
    cfg = LABEL_TYPES[label_type]
    raw_val = ref.get(cfg["field"], None)
    if raw_val is None:
        return False
    try:
        val = float(raw_val)
    except (ValueError, TypeError):
        return False
    if cfg["comparison"] == "eq":
        return val == cfg["threshold"]
    elif cfg["comparison"] == "gte":
        return val >= cfg["threshold"]
    return False


def extract_relevant_sets(references: list, label_type: str) -> list:
    """Extract the list of relevant (positive) matched_paper_ids for a label type."""
    return [
        ref["matched_paper_id"]
        for ref in references
        if ref.get("matched_paper_id") is not None and is_positive(ref, label_type)
    ]


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_paper_text_map(parquet_path: str | Path) -> dict:
    """Build mapping: paper_id (str) → 'title. abstract'."""
    df = pl.read_parquet(str(parquet_path))
    paper_map = {}
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        title = row.get("title", "") or ""
        abstract = row.get("abstract", "") or ""
        paper_map[pid] = f"{title}. {abstract}"
    return paper_map


# ──────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────
class ContrastiveSciBERT(nn.Module):
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", embed_dim: int = 128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        proj_emb = self.projection(cls_emb)
        proj_emb = nn.functional.normalize(proj_emb, dim=-1)
        return proj_emb


# ──────────────────────────────────────────────
# NT-Xent Loss
# ──────────────────────────────────────────────
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb1, emb2):
        B = emb1.size(0)
        Z = torch.cat([emb1, emb2], dim=0)          # [2B, d]
        sim = torch.matmul(Z, Z.T) / self.temperature  # [2B, 2B]
        mask = torch.eye(2 * B, dtype=torch.bool, device=Z.device)
        sim = sim.masked_fill(mask, -1e9)

        labels = torch.arange(B, device=Z.device)
        labels = torch.cat([labels + B, labels], dim=0)

        return self.ce(sim, labels)


# ──────────────────────────────────────────────
# Metric functions
# ──────────────────────────────────────────────
def average_precision(retrieved_ids: list, ground_truth_ids: list) -> float:
    if not ground_truth_ids:
        return 0.0
    gt_set = set(ground_truth_ids)
    score, num_hits = 0.0, 0.0
    for i, item_id in enumerate(retrieved_ids):
        if item_id in gt_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(ground_truth_ids)


def recall_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(ground_truth_ids)) / len(ground_truth_ids)


def precision_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(ground_truth_ids)) / k


def hit_rate_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(item in top_k for item in ground_truth_ids) else 0.0


def reciprocal_rank(retrieved_ids: list, ground_truth_ids: list) -> float:
    gt_set = set(ground_truth_ids)
    for i, item_id in enumerate(retrieved_ids):
        if item_id in gt_set:
            return 1.0 / (i + 1.0)
    return 0.0


def dcg_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    gt_set = set(ground_truth_ids)
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in gt_set:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    dcg = dcg_at_k(retrieved_ids, ground_truth_ids, k)
    ideal = list(ground_truth_ids) + ["_pad_"] * k
    idcg = dcg_at_k(ideal, ground_truth_ids, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(retrieved_ids: list, ground_truth_ids: list) -> dict:
    """Compute all metrics for a single query."""
    return {
        "ap": average_precision(retrieved_ids, ground_truth_ids),
        "mrr": reciprocal_rank(retrieved_ids, ground_truth_ids),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, ground_truth_ids, 10),
        "ndcg_at_20": ndcg_at_k(retrieved_ids, ground_truth_ids, 20),
        "ndcg_at_50": ndcg_at_k(retrieved_ids, ground_truth_ids, 50),
        "recall_at_5": recall_at_k(retrieved_ids, ground_truth_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, ground_truth_ids, 10),
        "recall_at_50": recall_at_k(retrieved_ids, ground_truth_ids, 50),
        "recall_at_100": recall_at_k(retrieved_ids, ground_truth_ids, 100),
        "precision_at_5": precision_at_k(retrieved_ids, ground_truth_ids, 5),
        "precision_at_10": precision_at_k(retrieved_ids, ground_truth_ids, 10),
        "hr_at_10": hit_rate_at_k(retrieved_ids, ground_truth_ids, 10),
        "hr_at_20": hit_rate_at_k(retrieved_ids, ground_truth_ids, 20),
    }


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Average a list of per-query metric dicts."""
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}