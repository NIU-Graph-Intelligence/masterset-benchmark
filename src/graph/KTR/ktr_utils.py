import os
import time
import json
import math
import pickle
import random
import sqlite3
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

# ======================================================================
# Constants (paper Section IV-A)
# ======================================================================
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
EMBED_DIM = 768
HIDDEN_DIM = 128
TOPIC_RADIUS = 2
NUM_TOPICS = 3
GCN_LAYERS = 2
MAX_PATH_LEN = 4
NA_BATCH = 128
NEG_RATIO = 10
CONTRASTIVE_M = 5
LAMBDA_C = 0.2
TAU_TOPIC = 10.0
EPOCHS = 15
LR = 0.001
PATIENCE = 10
ETA = 0.5
MAX_CONCEPTS_PER_PAPER = 100
MAX_PATHS_PER_PAIR = 20
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


# ======================================================================
# MODULE A: Knowledge Concept Filter (KCF) — ConceptNet
# ======================================================================

def load_conceptnet_concepts(db_path):
    """Load all English concept labels from ConceptNet SQLite DB."""
    if not db_path.exists():
        log(f"WARNING: ConceptNet DB not found at {db_path}")
        return set()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT label FROM concepts WHERE language='en'")
        concepts = {row[0].lower().strip() for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        # Try alternate schema
        try:
            cursor.execute("SELECT DISTINCT uri FROM nodes WHERE uri LIKE '/c/en/%'")
            concepts = set()
            for row in cursor.fetchall():
                parts = row[0].split("/")
                if len(parts) >= 4:
                    concepts.add(parts[3].replace("_", " ").lower())
        except:
            concepts = set()
    conn.close()
    return concepts


def load_conceptnet_graph(db_path):
    """Load ConceptNet as adjacency dict: concept -> set of neighbors."""
    if not db_path.exists():
        log(f"WARNING: ConceptNet DB not found at {db_path}")
        return {}
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cn_graph = defaultdict(set)
    try:
        cursor.execute("""
            SELECT start_uri, end_uri FROM edges
            WHERE start_uri LIKE '/c/en/%' AND end_uri LIKE '/c/en/%'
        """)
        for start, end in cursor.fetchall():
            s = start.split("/")[3].replace("_", " ").lower() if len(start.split("/")) >= 4 else None
            e = end.split("/")[3].replace("_", " ").lower() if len(end.split("/")) >= 4 else None
            if s and e:
                cn_graph[s].add(e)
                cn_graph[e].add(s)
    except:
        pass
    conn.close()
    log(f"  ConceptNet graph: {len(cn_graph)} concepts loaded")
    return dict(cn_graph)


def extract_phrases_simple(text, cn_lookup):
    """Simple n-gram extraction matching ConceptNet concepts."""
    if not text: return []
    words = text.lower().split()
    matched = []
    for n in range(3, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if phrase in cn_lookup:
                matched.append(phrase)
    return list(set(matched))[:MAX_CONCEPTS_PER_PAPER]


def preprocess_papers(paper_texts, cn_lookup, cache_path=None):
    """Extract ConceptNet concepts from each paper's text."""
    if cache_path and cache_path.exists():
        log("  Loading cached paper concepts...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log("  Extracting concepts from papers...")
    paper_concepts = {}
    for pid, text in tqdm(paper_texts.items(), desc="  Concept extraction"):
        concepts = extract_phrases_simple(text, cn_lookup)
        if concepts:
            paper_concepts[pid] = concepts

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(paper_concepts, f)

    log(f"  Papers with concepts: {len(paper_concepts)}")
    return paper_concepts


# ======================================================================
# MODULE B: Knowledge-aware Topic Construction (KTC)
# ======================================================================

def build_topic_clusters(paper_concepts, cn_graph, cache_path=None):
    """Group concepts into topics via r-hop clustering in ConceptNet."""
    if cache_path and cache_path.exists():
        log("  Loading cached topic clusters...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    log("  Building topic clusters...")
    paper_topics = {}

    for pid, concepts in tqdm(paper_concepts.items(), desc="  Topic clustering"):
        if not concepts: continue

        # Build local subgraph of r-hop neighbors
        local_nodes = set(concepts)
        for c in concepts:
            neighbors = cn_graph.get(c, set())
            for n in list(neighbors)[:50]:
                local_nodes.add(n)
                for n2 in list(cn_graph.get(n, set()))[:10]:
                    local_nodes.add(n2)

        # Build adjacency within local nodes
        local_adj = defaultdict(set)
        for n in local_nodes:
            for nb in cn_graph.get(n, set()):
                if nb in local_nodes:
                    local_adj[n].add(nb)

        # Connected components → topics
        visited = set()
        clusters = []
        for c in concepts:
            if c in visited: continue
            cluster = set()
            stack = [c]
            while stack:
                node = stack.pop()
                if node in visited: continue
                visited.add(node)
                if node in local_nodes:
                    cluster.add(node)
                    for nb in local_adj.get(node, set()):
                        if nb not in visited:
                            stack.append(nb)
            if cluster:
                # Only keep concepts that are in the paper
                paper_cluster = [x for x in cluster if x in concepts]
                if paper_cluster:
                    clusters.append(paper_cluster)

        # Sort by size, keep top NUM_TOPICS as main + rest as side
        clusters.sort(key=len, reverse=True)
        main_topics = clusters[:NUM_TOPICS]
        side_concepts = []
        for c in clusters[NUM_TOPICS:]:
            side_concepts.extend(c)

        paper_topics[pid] = {"main": main_topics, "side": side_concepts}

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(paper_topics, f)

    log(f"  Papers with topics: {len(paper_topics)}")
    return paper_topics


# ======================================================================
# MODULE C: Models (TopicGCN, StructureGCN, PathGRU, KTRModel)
# ======================================================================

class TopicGCN(nn.Module):
    """GCN for topic-level concept embeddings (Eq. 3-5)."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, eta=0.5):
        super().__init__()
        self.eta = eta
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(nn.Linear(input_dim if l == 0 else hidden_dim, hidden_dim))

    def forward(self, node_feats, adj_indices_list):
        h = node_feats
        N = h.size(0)
        adj = torch.zeros(N, N, device=h.device)
        if adj_indices_list:
            for i, j in adj_indices_list:
                if i < N and j < N:
                    adj[i, j] = 1.0; adj[j, i] = 1.0
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        for W in self.layers:
            self_part = self.eta * W(h)
            neigh_part = (1 - self.eta) * torch.sigmoid(torch.matmul(adj / deg, W(h))) if adj.sum() > 0 else torch.zeros_like(self_part)
            h = self_part + neigh_part
        return h


class StructureGCN(nn.Module):
    """GCN for citation structure (Eq. 6)."""
    def __init__(self, num_nodes, hidden_dim, num_layers=2):
        super().__init__()
        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, node_ids, adj_sparse):
        h = self.node_embed(node_ids)
        for layer in self.layers:
            if adj_sparse is not None and adj_sparse._nnz() > 0:
                h = torch.sigmoid(torch.sparse.mm(adj_sparse, layer(h)))
            else:
                h = torch.sigmoid(layer(h))
        return h


class PathGRU(nn.Module):
    """GRU path encoder for reasoning (Eq. 12)."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, path_embeddings):
        _, h_n = self.gru(path_embeddings)
        return torch.clamp(torch.tanh(self.fc(h_n.squeeze(0))), min=0)


class KTRModel(nn.Module):
    """Full KTR model combining CSA and RCR."""
    def __init__(self, num_papers, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.topic_gcn = TopicGCN(embed_dim, hidden_dim, GCN_LAYERS, ETA)
        self.concept_proj = nn.Linear(embed_dim, hidden_dim)
        self.structure_gcn = StructureGCN(num_papers, hidden_dim, GCN_LAYERS)
        self.path_gru = PathGRU(hidden_dim, hidden_dim)
        self.score_fc = nn.Linear(hidden_dim * 2, 1)
        self.topic_weight_fc = nn.Linear(hidden_dim, 1)

    def get_topic_embeddings(self, concept_embeddings, topic_indices_list, adj_within_topics):
        topic_embs = []
        for t_idx, (indices, adj_list) in enumerate(zip(topic_indices_list, adj_within_topics)):
            if not indices:
                topic_embs.append(torch.zeros(self.hidden_dim, device=concept_embeddings.device))
                continue
            feats = concept_embeddings[indices]
            h = self.topic_gcn(feats, adj_list)
            topic_embs.append(h.mean(dim=0))
        return torch.stack(topic_embs) if topic_embs else torch.zeros(1, self.hidden_dim, device=concept_embeddings.device)

    def compute_topic_score(self, topics_a, topics_b, tau=TAU_TOPIC):
        if topics_a.dim() == 1: topics_a = topics_a.unsqueeze(0)
        if topics_b.dim() == 1: topics_b = topics_b.unsqueeze(0)
        n_a, n_b = topics_a.size(0), topics_b.size(0)
        sim = F.cosine_similarity(topics_a.unsqueeze(1).expand(-1, n_b, -1),
                                   topics_b.unsqueeze(0).expand(n_a, -1, -1), dim=-1)
        w_a = F.softmax(self.topic_weight_fc(topics_a).squeeze(-1) / tau, dim=0)
        w_b = F.softmax(self.topic_weight_fc(topics_b).squeeze(-1) / tau, dim=0)
        score = (w_a.unsqueeze(1) * sim * w_b.unsqueeze(0)).sum()
        return torch.sigmoid(score)

    def forward_pair(self, alpha_data, beta_data):
        """Score a paper pair (alpha, beta)."""
        device = next(self.parameters()).device
        if not alpha_data or not beta_data:
            return torch.tensor(0.5, device=device), None, None

        # Project concept embeddings
        alpha_ce = self.concept_proj(alpha_data["concept_embs"].to(device))
        beta_ce = self.concept_proj(beta_data["concept_embs"].to(device))

        # Topic embeddings
        alpha_topics = self.get_topic_embeddings(alpha_ce, alpha_data.get("topic_indices", [[]]), alpha_data.get("topic_adj", [[]]))
        beta_topics = self.get_topic_embeddings(beta_ce, beta_data.get("topic_indices", [[]]), beta_data.get("topic_adj", [[]]))

        # Topic score
        topic_score = self.compute_topic_score(alpha_topics, beta_topics)

        # Content embedding (mean of topic embeddings)
        content_a = alpha_topics.mean(dim=0)
        content_b = beta_topics.mean(dim=0)

        # Combine topic score with content similarity
        combined = torch.cat([content_a, content_b])
        pair_score = torch.sigmoid(self.score_fc(combined)).squeeze()
        final_score = 0.5 * topic_score + 0.5 * pair_score

        return final_score, content_a, content_b


class ContrastiveLoss(nn.Module):
    """Content-Structure Alignment contrastive loss (Eq. 7-9)."""
    def __init__(self):
        super().__init__()

    def forward(self, z_c, z_s, neg_z_c=None, neg_z_s=None):
        pos_sim = F.cosine_similarity(z_c, z_s, dim=-1)
        pos_loss = -torch.log(torch.sigmoid(pos_sim) + 1e-8).mean()
        if neg_z_c is not None and neg_z_s is not None:
            neg_sim_c = F.cosine_similarity(z_c.unsqueeze(1), neg_z_c, dim=-1)
            neg_sim_s = F.cosine_similarity(z_s.unsqueeze(1), neg_z_s, dim=-1)
            neg_loss = (-torch.log(1 - torch.sigmoid(neg_sim_c) + 1e-8).mean()
                        - torch.log(1 - torch.sigmoid(neg_sim_s) + 1e-8).mean()) * 0.5
            return pos_loss + neg_loss
        return pos_loss


# ======================================================================
# SciBERT Embedding Cache
# ======================================================================

def compute_scibert_embeddings(texts, cache_path=None, device=None):
    """Compute frozen SciBERT [CLS] embeddings."""
    if cache_path and cache_path.exists():
        log("  Loading cached SciBERT embeddings...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from transformers import AutoTokenizer, AutoModel
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("  Computing SciBERT embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL)
    model = AutoModel.from_pretrained(SCIBERT_MODEL).to(device)
    model.eval()

    embeddings = {}
    keys = sorted(texts.keys())
    batch_size = 64

    for i in tqdm(range(0, len(keys), batch_size), desc="  SciBERT"):
        batch_keys = keys[i:i + batch_size]
        batch_texts = [texts[k] for k in batch_keys]
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]
        for j, k in enumerate(batch_keys):
            embeddings[k] = cls[j].cpu()

    del model; torch.cuda.empty_cache()

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings


# ======================================================================
# Dataset for KTR
# ======================================================================

class KTRDataset(Dataset):
    """Returns (anchor_id, pos_id, neg_ids) for a specific label type."""

    def __init__(self, parquet_file, paper_concepts, label_type_key, paper_text_map):
        cfg = LABEL_TYPES[label_type_key]
        field = cfg["field"]
        threshold_fn = cfg["threshold_fn"]

        df = pl.read_parquet(parquet_file)
        self.samples = []

        all_pids = list(paper_concepts.keys())

        for row in df.iter_rows(named=True):
            pid = row["paper_id"]
            if pid not in paper_concepts: continue
            refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]

            pos_ids, neg_ids = [], []
            for ref in refs:
                mid = ref.get("matched_paper_id")
                if not mid or mid not in paper_concepts: continue
                raw_val = ref.get(field)
                if raw_val is None: continue
                try:
                    if threshold_fn(raw_val): pos_ids.append(mid)
                    else: neg_ids.append(mid)
                except: continue

            if pos_ids:
                # Supplement negatives with random papers
                while len(neg_ids) < NEG_RATIO:
                    r = random.choice(all_pids)
                    if r != pid and r not in pos_ids:
                        neg_ids.append(r)
                for pos_id in pos_ids:
                    self.samples.append({"paper_id": pid, "pos_id": pos_id,
                                         "neg_ids": neg_ids[:NEG_RATIO]})

        log(f"    KTR dataset ({cfg['name']}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ======================================================================
# Citation Network (ALL references create edges)
# ======================================================================

def build_citation_network(train_parquet, eval_parquet):
    """Build citation adjacency from ALL references."""
    citation_graph = defaultdict(set)
    for pq in [train_parquet, eval_parquet]:
        if not pq.exists(): continue
        df = pl.read_parquet(pq)
        for row in df.iter_rows(named=True):
            pid = row["paper_id"]
            refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
            for ref in refs:
                mid = ref.get("matched_paper_id")
                if mid:
                    citation_graph[pid].add(mid)
                    citation_graph[mid].add(pid)  # undirected
    return dict(citation_graph)


# ======================================================================
# Ground Truth & Metrics
# ======================================================================

def extract_relevant_sets(references):
    relevant = {key: set() for key in LABEL_TYPES}
    for ref in references:
        mid = ref.get("matched_paper_id")
        if not mid: continue
        for key, cfg in LABEL_TYPES.items():
            raw_val = ref.get(cfg["field"])
            if raw_val is None: continue
            try:
                if cfg["threshold_fn"](raw_val): relevant[key].add(mid)
            except: continue
    return relevant


def compute_map(relevant, retrieved):
    if not relevant: return 0.0
    score, n = 0.0, 0
    for i, p in enumerate(retrieved):
        if p in relevant: n += 1; score += n / (i + 1)
    return score / len(relevant)

def compute_mrr(relevant, retrieved):
    for i, p in enumerate(retrieved):
        if p in relevant: return 1.0 / (i + 1)
    return 0.0

def compute_recall_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    return sum(1 for p in retrieved[:k] if p in relevant) / len(relevant)

def compute_hit_rate_at_k(relevant, retrieved, k):
    return 1.0 if any(item in set(retrieved[:k]) for item in relevant) else 0.0

def compute_dcg_at_k(relevant, retrieved, k):
    return sum(1.0 / np.log2(i + 2) for i, p in enumerate(retrieved[:k]) if p in relevant)

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
