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
# Constants
# ======================================================================
MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LEN = 512
BATCH_SIZE_FT = 4      # triplet loss uses 3 forward passes, keep small
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
    """Build mapping: paper_id (str) -> 'title [SEP] abstract'"""
    df = pl.read_parquet(parquet_file)
    paper_map = {}
    for row in df.iter_rows(named=True):
        paper_id = row["paper_id"]
        title = (row.get("title", "") or "").strip()
        abstract = (row.get("abstract", "") or "").strip()
        paper_map[paper_id] = f"{title} {SEP} {abstract}"
    return paper_map


def load_paper_title_map(parquet_file):
    try:
        df = pl.read_parquet(parquet_file)
        return {row["paper_id"]: row.get("title", "") for row in df.iter_rows(named=True)}
    except Exception as e:
        print(f"Warning: Could not load title map: {e}")
        return {}


# ======================================================================
# Scalar Mix Layer (original SPECTER approach)
# ======================================================================

class ScalarMix(nn.Module):
    """
    Learnable weighted combination of all BERT layers (including embedding layer).
    As used in the original SPECTER implementation from AllenAI.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in range(num_layers)]
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, tensors):
        if len(tensors) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} tensors, got {len(tensors)}")
        normed_weights = F.softmax(
            torch.stack([p for p in self.scalar_parameters]), dim=0
        )
        mixed = sum(w * t for w, t in zip(normed_weights, tensors))
        return self.gamma * mixed


# ======================================================================
# SPECTER Model: Scalar Mix + Average Pooling
# ======================================================================

class SPECTERScalarMixModel(nn.Module):
    """
    SPECTER model using scalar mixing of all layers and average pooling.
    Matches the original AllenNLP/AllenAI implementation.
    """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = self.encoder.config
        # Scalar mix for all layers (embedding layer + num_hidden_layers)
        num_layers = config.num_hidden_layers + 1
        self.scalar_mix = ScalarMix(num_layers)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # outputs.hidden_states: (embedding_output, layer_1, ..., layer_N)
        all_layers = outputs.hidden_states
        mixed_output = self.scalar_mix(list(all_layers))

        # Average pooling over sequence length (excluding padding)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(mixed_output.size()).float()
        sum_embeddings = torch.sum(mixed_output * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # L2 normalize
        pooled_output = F.normalize(pooled_output, p=2, dim=-1)
        return pooled_output


# ======================================================================
# Triple Margin Loss (SPECTER loss)
# ======================================================================

class SPECTERLoss(nn.Module):
    """
    SPECTER uses triple margin loss:
      L = max(0, margin - sim(q, p) + sim(q, n))
    where sim is cosine similarity (dot product after L2 normalization).
    """

    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb, pos_emb, neg_emb):
        pos_sim = torch.sum(query_emb * pos_emb, dim=-1)
        neg_sim = torch.sum(query_emb * neg_emb, dim=-1)
        loss = torch.clamp(self.margin - pos_sim + neg_sim, min=0.0)
        return loss.mean()


# ======================================================================
# Triplet Dataset: (query, positive, negative) for a specific label type
# ======================================================================

class SPECTERTripletDataset(Dataset):
    """
    Builds (query, positive, negative) triplets for SPECTER training.
    Positive = reference where label is positive for this type.
    Negative = reference where label is NOT positive for this type.
    """

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

            positives = []
            negatives = []

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

            # Create triplets: for each positive, pair with each negative
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
# Embedding Generation Dataset
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
        paper_id = self.paper_ids[idx]
        text = self.paper_text_map.get(paper_id, "")
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "paper_id": paper_id,
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
