import os
import time
import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ======================================================================
# Constants
# ======================================================================
BASE_MODEL = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2"  # proximity adapter for retrieval
MAX_LEN = 512
BATCH_SIZE_FT = 8           # small because triplet = 3 forward passes
GRAD_ACCUM_STEPS = 4        # effective batch size = 8 * 4 = 32
BATCH_SIZE_EMB = 128
EPOCHS = 2
LR = 2e-5
MARGIN = 1.0
SEP = "[SEP]"

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
# SPECTER2 Model Loading
# ======================================================================

def load_specter2_model(load_adapter=True):
    """
    Load SPECTER2 base model with the proximity adapter.
    Uses the `adapters` library (AutoAdapterModel).
    Returns (model, tokenizer).
    """
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL)

    if load_adapter:
        model.load_adapter(ADAPTER_NAME, source="hf", load_as="proximity", set_active=True)
        log(f"  Loaded adapter: {ADAPTER_NAME} (proximity)")

    return model, tokenizer


def get_cls_embedding(model, input_ids, attention_mask):
    """Extract [CLS] token embedding from SPECTER2 (base + adapter)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # outputs[0] is last_hidden_state: [batch, seq_len, hidden_dim]
    cls_emb = outputs[0][:, 0, :]  # [CLS] token
    return cls_emb


# ======================================================================
# Triple Margin Loss (same as original SPECTER)
# ======================================================================

class TripletMarginLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb, pos_emb, neg_emb):
        pos_dist = F.pairwise_distance(query_emb, pos_emb, p=2)
        neg_dist = F.pairwise_distance(query_emb, neg_emb, p=2)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()


# ======================================================================
# Triplet Dataset
# ======================================================================

class SPECTER2TripletDataset(Dataset):
    """(query, positive, negative) triplets for a specific label type."""

    def __init__(self, parquet_file, paper_text_map, label_type_key):
        cfg = LABEL_TYPES[label_type_key]
        field = cfg["field"]
        threshold_fn = cfg["threshold_fn"]

        df = pl.read_parquet(parquet_file)
        self.triplets = []

        for row in df.iter_rows(named=True):
            paper_id = row["paper_id"]
            paper_text = paper_text_map.get(paper_id, "")
            if not paper_text.strip():
                continue

            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
            positives, negatives = [], []

            for ref in references:
                matched_paper_id = ref.get("matched_paper_id")
                if not matched_paper_id:
                    continue
                ref_text = paper_text_map.get(matched_paper_id)
                if not ref_text or not ref_text.strip():
                    continue
                raw_val = ref.get(field)
                if raw_val is None:
                    continue
                try:
                    if threshold_fn(raw_val):
                        positives.append(ref_text)
                    else:
                        negatives.append(ref_text)
                except (ValueError, TypeError):
                    continue

            if positives and negatives:
                for pos_text in positives:
                    for neg_text in negatives:
                        self.triplets.append((paper_text, pos_text, neg_text))

        log(f"    {cfg['name']}: {len(self.triplets)} triplets")

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
            truncation=True, return_tensors="pt", return_token_type_ids=False,
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
    top_k = set(retrieved[:k])
    return 1.0 if any(item in top_k for item in relevant) else 0.0

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
