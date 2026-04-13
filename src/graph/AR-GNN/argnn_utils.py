import os
import time
import json
import pickle
import logging
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

try:
    from community import best_partition
except ImportError:
    try:
        from community.community_louvain import best_partition
    except ImportError:
        from networkx.algorithms.community import louvain_communities
        def best_partition(G, random_state=None):
            comms = louvain_communities(G, seed=random_state)
            return {node: i for i, comm in enumerate(comms) for node in comm}

from torch_geometric.nn import GATConv
import networkx as nx

# ======================================================================
# Constants (paper Section 5.5)
# ======================================================================
DELTA = 0.95
CURRENT_YEAR = 2026
LOUVAIN_SEED = 42
IN_DIM = 10
HIDDEN = 10
OUT_DIM = 8
GAT_HEADS = 4
LR = 0.01
EPOCHS = 400
N_COMMUNITIES = 6
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
# Heterogeneous Graph
# ======================================================================

class HeterogeneousGraph:
    """Paper/Venue/Topic nodes with citation/venue/topic edges."""

    def __init__(self):
        self.paper_to_nid = {}
        self.venue_to_nid = {}
        self.topic_to_nid = {}
        self.nid_to_info = {}
        self.citation_edges = []
        self.venue_edges = []
        self.topic_edges = []
        self.paper_year = {}
        self.paper_cite_count = {}
        self.paper_venue = {}
        self.paper_topics = {}
        self._next_nid = 0

    def _add_paper(self, pid):
        if pid not in self.paper_to_nid:
            nid = self._next_nid; self.paper_to_nid[pid] = nid
            self.nid_to_info[nid] = ("paper", pid); self._next_nid += 1
        return self.paper_to_nid[pid]

    def _add_venue(self, v):
        v = v.strip().lower()
        if v not in self.venue_to_nid:
            nid = self._next_nid; self.venue_to_nid[v] = nid
            self.nid_to_info[nid] = ("venue", v); self._next_nid += 1
        return self.venue_to_nid[v]

    def _add_topic(self, t):
        t = t.strip().lower()
        if t not in self.topic_to_nid:
            nid = self._next_nid; self.topic_to_nid[t] = nid
            self.nid_to_info[nid] = ("topic", t); self._next_nid += 1
        return self.topic_to_nid[t]

    def build(self, train_parquet, eval_parquet):
        """Build graph from train+eval parquets. ALL references create citation edges."""
        for pq in [train_parquet, eval_parquet]:
            if not pq.exists(): continue
            df = pl.read_parquet(pq)
            for row in df.iter_rows(named=True):
                pid = row["paper_id"]
                self._add_paper(pid)
                self.paper_year[pid] = int(row.get("year") or CURRENT_YEAR)
                venue = str(row.get("venue") or "unknown").strip().lower()
                self.paper_venue[pid] = venue

                # Venue edges
                if venue:
                    self.venue_edges.append((self.paper_to_nid[pid], self._add_venue(venue)))

                # ALL citation edges (not just positive labels)
                refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
                for ref in refs:
                    matched_id = ref.get("matched_paper_id")
                    if matched_id:
                        if matched_id not in self.paper_to_nid:
                            self._add_paper(matched_id)
                            self.paper_year.setdefault(matched_id, CURRENT_YEAR)
                            self.paper_venue.setdefault(matched_id, "unknown")
                        self.citation_edges.append((self.paper_to_nid[pid], self.paper_to_nid[matched_id]))

        # Citation counts
        cite_count = defaultdict(int)
        for _, dst_nid in self.citation_edges:
            info = self.nid_to_info.get(dst_nid)
            if info and info[0] == "paper":
                cite_count[info[1]] += 1
        self.paper_cite_count = dict(cite_count)

        log(f"  Graph: {len(self.paper_to_nid)} papers, {len(self.venue_to_nid)} venues, "
            f"{len(self.topic_to_nid)} topics | {len(self.citation_edges)} citation edges")


def compute_citation_score(n_cites, age, delta=DELTA):
    return float(np.log1p(n_cites) * (delta ** age))


def build_keyword_communities(paper_topics, n_communities=N_COMMUNITIES):
    G = nx.Graph()
    for pid, topics in paper_topics.items():
        unique = list(set(topics))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                if G.has_edge(unique[i], unique[j]):
                    G[unique[i]][unique[j]]["weight"] += 1
                else:
                    G.add_edge(unique[i], unique[j], weight=1)
    if len(G) == 0: return {}
    gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    partition = best_partition(gcc, random_state=LOUVAIN_SEED)
    community_ids = sorted(set(partition.values()))
    freq = {c: sum(1 for v in partition.values() if v == c) for c in community_ids}
    sorted_comms = sorted(community_ids, key=lambda c: -freq[c])
    remap = {old: min(new, n_communities - 1) for new, old in enumerate(sorted_comms)}
    return {k: remap[v] for k, v in partition.items()}


def build_feature_matrix(graph, topic_community, n_communities=N_COMMUNITIES):
    N = graph._next_nid
    F_DIM = 4 + n_communities
    X = np.zeros((N, F_DIM), dtype=np.float32)

    paper_cs = {}
    venue_papers = defaultdict(list)
    topic_papers_map = defaultdict(list)

    for pid, nid in graph.paper_to_nid.items():
        year = graph.paper_year.get(pid, CURRENT_YEAR)
        age = max(CURRENT_YEAR - year, 0)
        n_c = graph.paper_cite_count.get(pid, 0)
        paper_cs[pid] = compute_citation_score(n_c, age)
        X[nid, 0] = float(age)
        X[nid, 1] = paper_cs[pid]
        venue_papers[graph.paper_venue.get(pid, "unknown")].append(pid)
        for t in graph.paper_topics.get(pid, []):
            topic_papers_map[t].append(pid)

    # Venue quality score
    venue_cs = {v: float(np.mean([paper_cs.get(p, 0) for p in pids])) for v, pids in venue_papers.items()}
    for pid, nid in graph.paper_to_nid.items():
        X[nid, 2] = venue_cs.get(graph.paper_venue.get(pid, "unknown"), 0.0)

    # Topic influence score
    topic_cs = {t: float(np.mean([paper_cs.get(p, 0) for p in pids])) for t, pids in topic_papers_map.items()}
    for pid, nid in graph.paper_to_nid.items():
        topics = graph.paper_topics.get(pid, [])
        X[nid, 3] = float(np.mean([topic_cs.get(t, 0) for t in topics])) if topics else 0.0

    # Keyword community vector
    for pid, nid in graph.paper_to_nid.items():
        kv = np.zeros(n_communities, dtype=np.float32)
        for t in graph.paper_topics.get(pid, []):
            c = topic_community.get(t)
            if c is not None: kv[c] += 1.0
        X[nid, 4:] = kv

    # Venue/topic node features (aggregate from papers)
    for venue, vnid in graph.venue_to_nid.items():
        pids = venue_papers.get(venue, [])
        if pids:
            X[vnid, 1] = float(np.mean([paper_cs.get(p, 0) for p in pids]))
            X[vnid, 2] = venue_cs.get(venue, 0)

    for topic, tnid in graph.topic_to_nid.items():
        pids = topic_papers_map.get(topic, [])
        if pids:
            X[tnid, 1] = float(np.mean([paper_cs.get(p, 0) for p in pids]))
            X[tnid, 3] = topic_cs.get(topic, 0)

    # Normalize columns
    col_max = X.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    X = X / col_max

    return torch.tensor(X, dtype=torch.float32)


def build_edge_index(graph):
    all_edges = graph.citation_edges + graph.venue_edges + graph.topic_edges
    if not all_edges: return torch.zeros((2, 0), dtype=torch.long)
    src, dst = zip(*all_edges)
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


# ======================================================================
# AR-GATK Model (paper Section 4.3.2)
# ======================================================================

class ARGATK(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, out_dim=OUT_DIM, heads=GAT_HEADS):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.1)
        self.fcn = nn.Linear(hidden, out_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, edge_index):
        h = self.act(self.gat1(x, edge_index))
        h = self.act(self.gat2(h, edge_index))
        return F.normalize(self.fcn(h), p=2, dim=-1)


# ======================================================================
# Distance-based Loss (Eq. 18)
# ======================================================================

def distance_loss(embeddings, seed_nids):
    if not seed_nids: return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
    seed_embs = embeddings[seed_nids]
    diffs = embeddings.unsqueeze(1) - seed_embs.unsqueeze(0)
    dists = diffs.norm(dim=-1)
    min_dists, _ = dists.min(dim=-1)
    return min_dists.sum()


def select_seeds(train_parquet, graph, label_type_key, n_seeds=10):
    """Select seed papers that have the most positive references for this label type."""
    cfg = LABEL_TYPES[label_type_key]
    field = cfg["field"]
    threshold_fn = cfg["threshold_fn"]

    df = pl.read_parquet(train_parquet)
    scored = []
    for row in df.iter_rows(named=True):
        pid = row["paper_id"]
        if pid not in graph.paper_to_nid: continue
        refs = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
        n_pos = 0
        for ref in refs:
            raw_val = ref.get(field)
            if raw_val is not None:
                try:
                    if threshold_fn(raw_val): n_pos += 1
                except: pass
        if n_pos > 0:
            scored.append((n_pos, graph.paper_to_nid[pid]))
    scored.sort(reverse=True)
    seeds = [nid for _, nid in scored[:n_seeds]]
    log(f"  Selected {len(seeds)} seeds for {cfg['name']}")
    return seeds


# ======================================================================
# Quality Score (Eq. 1)
# ======================================================================

ALPHA_QS, BETA_QS, GAMMA_QS = 0.30, 0.35, 0.35

def compute_quality_scores(graph, query_topics=None):
    if query_topics is None: query_topics = set()
    paper_ids = sorted(graph.paper_to_nid.keys())
    n = len(paper_ids)
    topic_rel = np.zeros(n, dtype=np.float32)
    cs_arr = np.zeros(n, dtype=np.float32)
    recency = np.zeros(n, dtype=np.float32)

    for i, pid in enumerate(paper_ids):
        topics_p = set(graph.paper_topics.get(pid, []))
        topic_rel[i] = float(len(topics_p & query_topics))
        year = graph.paper_year.get(pid, CURRENT_YEAR)
        age = max(CURRENT_YEAR - year, 0)
        cs_arr[i] = compute_citation_score(graph.paper_cite_count.get(pid, 0), age)
        recency[i] = 1.0 / max(age, 1)

    def safe_norm(a):
        m = a.max(); return a / m if m > 0 else a

    qs = ALPHA_QS * safe_norm(topic_rel) + BETA_QS * safe_norm(cs_arr) + GAMMA_QS * safe_norm(recency)
    return np.array(paper_ids), qs


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
    for i, pid in enumerate(retrieved):
        if pid in relevant: n += 1; score += n / (i + 1)
    return score / len(relevant)

def compute_mrr(relevant, retrieved):
    for i, pid in enumerate(retrieved):
        if pid in relevant: return 1.0 / (i + 1)
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
