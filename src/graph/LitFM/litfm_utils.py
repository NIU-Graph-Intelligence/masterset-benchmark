import os
import time
import json
import pickle
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

# ======================================================================
# Constants (from paper Appendix C)
# ======================================================================
BERT_MODEL = "bert-base-uncased"
BERT_MAX_LEN = 512
INPUT_DIM = 768
HIDDEN_DIM = 768
NEIGH_NUM = 10
NEG_SAMPLES = 10
BATCH_SIZE_TRAIN = 512
EPOCHS = 500
PATIENCE = 5
LR = 1e-5
LAMBDA_RE = 1.0
EDGE_SPLIT_SEED = 42
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
        return {}


# ======================================================================
# Citation Graph Construction
# ======================================================================

def build_citation_graph(train_parquet, label_type_key):
    """
    Build citation graph for a specific label type.
    
    ALL references create edges in the adjacency (for neighbor aggregation).
    But only POSITIVE edges (per label threshold) are used as training signal.
    
    Returns:
      - adjacency: dict[str, set[str]] - ALL neighbors (undirected)
      - positive_edges: list[(src, dst)] - edges where label is positive
      - all_nodes: set of all node IDs
    """
    cfg = LABEL_TYPES[label_type_key]
    field = cfg["field"]
    threshold_fn = cfg["threshold_fn"]

    df = pl.read_parquet(train_parquet)
    adjacency = defaultdict(set)
    positive_edges = []
    all_nodes = set()

    for row in df.iter_rows(named=True):
        paper_id = row["paper_id"]
        all_nodes.add(paper_id)
        references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]

        for ref in references:
            matched_id = ref.get("matched_paper_id")
            if not matched_id:
                continue

            # ALL references create adjacency edges (for message passing)
            adjacency[paper_id].add(matched_id)
            adjacency[matched_id].add(paper_id)
            all_nodes.add(matched_id)

            # Only positive references become training edges
            raw_val = ref.get(field)
            if raw_val is not None:
                try:
                    if threshold_fn(raw_val):
                        positive_edges.append((paper_id, matched_id))
                except (ValueError, TypeError):
                    pass

    log(f"  Nodes: {len(all_nodes)}, All adjacency pairs: {sum(len(v) for v in adjacency.values()) // 2}")
    log(f"  Positive edges ({cfg['name']}): {len(positive_edges)}")
    return adjacency, positive_edges, all_nodes


# ======================================================================
# BERT Embedding Computation (Frozen)
# ======================================================================

def compute_bert_embeddings(paper_text_map, node_ids, cache_path=None, device=None):
    """Compute frozen BERT-base [CLS] embeddings for all papers."""
    if cache_path and cache_path.exists():
        log(f"  Loading cached BERT embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("  Computing BERT embeddings (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    bert = AutoModel.from_pretrained(BERT_MODEL).to(device)
    bert.eval()

    embeddings = {}
    node_list = sorted(node_ids)
    batch_size = 64

    for i in tqdm(range(0, len(node_list), batch_size), desc="  BERT encoding"):
        batch_ids = node_list[i:i + batch_size]
        texts = [paper_text_map.get(nid, "") for nid in batch_ids]
        encoded = tokenizer(texts, padding=True, truncation=True,
                            max_length=BERT_MAX_LEN, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = bert(**encoded, output_hidden_states=True)
            cls_embs = outputs.hidden_states[-1][:, 0, :]
        for j, nid in enumerate(batch_ids):
            embeddings[nid] = cls_embs[j].cpu()

    del bert
    torch.cuda.empty_cache()

    if cache_path:
        log(f"  Caching BERT embeddings to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)

    log(f"  Computed embeddings for {len(embeddings)} nodes")
    return embeddings


# ======================================================================
# LitFM Retriever Model (Paper Section 3.1)
# ======================================================================

class LitFMRetriever(nn.Module):
    """
    QueryEncoder:      q_i = W^q * z_i + b^q
    CandidateEncoder:  c_j = W^c1 * z_j + mean_k(W^c2 * z_k) + b^c
    PseudoQueryMLP:    p_j = MLP(c_j)
    Similarity:        sim(i,j) = cos(q_i, c_j + proj(p_j))
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.cand_self_proj = nn.Linear(input_dim, hidden_dim)
        self.cand_nei_proj = nn.Linear(input_dim, hidden_dim)
        self.pseudo_query_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def encode_query(self, z_query):
        return self.query_proj(z_query)

    def encode_candidate(self, z_target, z_neighbors):
        self_emb = self.cand_self_proj(z_target)
        nei_emb = self.cand_nei_proj(z_neighbors).mean(dim=1) if z_neighbors.dim() == 3 else torch.zeros_like(self_emb)
        return self_emb + nei_emb

    def generate_pseudo_query(self, c_j):
        return self.pseudo_query_mlp(c_j)

    def forward(self, z_query, z_target, z_neighbors, z_neg_targets, z_neg_neighbors):
        B = z_query.size(0)
        neg_K = z_neg_targets.size(1)

        q = self.encode_query(z_query)
        c_pos = self.encode_candidate(z_target, z_neighbors)
        p_pos = self.generate_pseudo_query(c_pos)
        l_re = F.l1_loss(p_pos, z_query, reduction='mean')

        p_pos_h = self.query_proj(p_pos)
        pos_score = F.cosine_similarity(q, c_pos + p_pos_h, dim=-1)

        z_neg_flat = z_neg_targets.reshape(B * neg_K, -1)
        z_neg_nei_flat = z_neg_neighbors.reshape(B * neg_K, NEIGH_NUM, -1)
        c_neg = self.encode_candidate(z_neg_flat, z_neg_nei_flat)
        p_neg = self.generate_pseudo_query(c_neg)
        p_neg_h = self.query_proj(p_neg)

        c_neg = c_neg.reshape(B, neg_K, -1)
        p_neg_h = p_neg_h.reshape(B, neg_K, -1)
        combined_neg = c_neg + p_neg_h
        q_exp = q.unsqueeze(1).expand(-1, neg_K, -1)
        neg_scores = F.cosine_similarity(q_exp, combined_neg, dim=-1)

        logit_scale = self.temperature.exp()
        logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1) * logit_scale
        labels = torch.zeros(B, dtype=torch.long, device=z_query.device)
        l_nce = F.cross_entropy(logits, labels)

        return l_nce, l_re

    def get_candidate_embedding(self, z_target, z_neighbors):
        c = self.encode_candidate(z_target, z_neighbors)
        p = self.generate_pseudo_query(c)
        p_h = self.query_proj(p)
        return c + p_h

    def get_query_embedding(self, z_query):
        return self.encode_query(z_query)


# ======================================================================
# Training Dataset
# ======================================================================

class RetrieverTrainDataset(Dataset):
    def __init__(self, edges, adjacency, all_node_ids, bert_embeddings,
                 neg_samples=NEG_SAMPLES, neigh_num=NEIGH_NUM):
        self.edges = edges
        self.adjacency = adjacency
        self.all_node_ids = list(all_node_ids)
        self.bert_embeddings = bert_embeddings
        self.neg_samples = neg_samples
        self.neigh_num = neigh_num
        self.zero_emb = torch.zeros(INPUT_DIM)

    def __len__(self):
        return len(self.edges)

    def _get_emb(self, nid):
        return self.bert_embeddings.get(nid, self.zero_emb)

    def _sample_neighbors(self, nid):
        neighbors = list(self.adjacency.get(nid, set()))
        if not neighbors:
            return torch.zeros(self.neigh_num, INPUT_DIM)
        if len(neighbors) >= self.neigh_num:
            sampled = random.sample(neighbors, self.neigh_num)
        else:
            sampled = neighbors + [random.choice(neighbors) for _ in range(self.neigh_num - len(neighbors))]
        return torch.stack([self._get_emb(n) for n in sampled])

    def __getitem__(self, idx):
        src, dst = self.edges[idx]
        z_query = self._get_emb(src)
        z_target = self._get_emb(dst)
        z_neighbors = self._sample_neighbors(dst)

        # Negative sampling: random nodes excluding neighbors of src
        src_neighbors = self.adjacency.get(src, set())
        neg_pool = [n for n in self.all_node_ids if n not in src_neighbors and n != src]
        neg_ids = random.choices(neg_pool, k=self.neg_samples) if neg_pool else [self.all_node_ids[0]] * self.neg_samples

        z_neg_targets = torch.stack([self._get_emb(n) for n in neg_ids])
        z_neg_neighbors = torch.stack([self._sample_neighbors(n) for n in neg_ids])

        return z_query, z_target, z_neighbors, z_neg_targets, z_neg_neighbors


# ======================================================================
# Ground Truth & Metrics
# ======================================================================

def extract_relevant_sets(references):
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        matched_paper_id = ref.get("matched_paper_id")
        if not matched_paper_id: continue
        for key, cfg in LABEL_TYPES.items():
            raw_val = ref.get(cfg["field"])
            if raw_val is None: continue
            try:
                if cfg["threshold_fn"](raw_val):
                    relevant[key].add(matched_paper_id)
            except (ValueError, TypeError): continue
    return relevant


def compute_map(relevant, retrieved):
    if not relevant: return 0.0
    score, num_hits = 0.0, 0
    for i, pid in enumerate(retrieved):
        if pid in relevant:
            num_hits += 1; score += num_hits / (i + 1)
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
