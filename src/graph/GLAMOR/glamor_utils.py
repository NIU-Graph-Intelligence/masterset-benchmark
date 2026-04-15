import os
import time
import json
import random
import numpy as np
import polars as pl
from collections import defaultdict

# ======================================================================
# Constants
# ======================================================================
BASE_MODEL = "distilroberta-base"
MAX_SEQ_LENGTH = 512
BATCH_SIZE_MLM = 32
BATCH_SIZE_EMB = 128
MLM_EPOCHS = 3
MASKING_RATIO = 0.15
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
NUM_WALKS_PER_NODE = 5
TERMINATION_PROB = 0.05
MAX_WALK_LENGTH = 10
SEED = 42
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
# Heterogeneous Knowledge Graph
# ======================================================================

class HeterogeneousKG:
    """
    KG with node types: Paper, Year, Venue, Topic
    Edge types: cites, published_in, published_on, belongs_to_topic
    """

    def __init__(self):
        self.nodes = {}
        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        self.paper_node_ids = []
        self.node_text_cache = {}

    def add_node(self, node_id, node_type, attributes=None):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, "attributes": attributes or {}}
            if node_type == "Paper":
                self.paper_node_ids.append(node_id)

    def add_edge(self, src_id, dst_id, relation):
        self.adjacency[src_id].append((dst_id, relation))
        reverse_rel = self._reverse_relation(relation)
        self.reverse_adjacency[dst_id].append((src_id, reverse_rel))

    def _reverse_relation(self, relation):
        return {
            "cites": "cited_by", "published_in": "has_paper",
            "published_on": "has_paper_from", "belongs_to_topic": "topic_of",
        }.get(relation, f"reverse_{relation}")

    def get_neighbors(self, node_id, include_reverse=True):
        neighbors = list(self.adjacency.get(node_id, []))
        if include_reverse:
            neighbors.extend(self.reverse_adjacency.get(node_id, []))
        return neighbors

    def textualize_node(self, node_id):
        if node_id in self.node_text_cache:
            return self.node_text_cache[node_id]
        node = self.nodes.get(node_id)
        if node is None:
            return ""
        ntype = node["type"]
        attrs = node["attributes"]
        if ntype == "Paper":
            title = attrs.get("title", "Unknown")
            venue = attrs.get("venue", "")
            year = attrs.get("year", "")
            text = f"Paper titled {title}"
            if venue: text += f" published in {venue}"
            if year: text += f" in {year}"
        elif ntype == "Venue":
            text = f"Venue {attrs.get('name', node_id)}"
        elif ntype == "Year":
            text = f"Year {attrs.get('name', node_id)}"
        elif ntype == "Topic":
            text = f"Topic {attrs.get('name', node_id)}"
        else:
            text = f"{ntype} {node_id}"
        self.node_text_cache[node_id] = text
        return text

    def textualize_edge(self, relation):
        return {
            "cites": "cites", "cited_by": "is cited by",
            "published_in": "is published in", "has_paper": "contains",
            "published_on": "was published on", "has_paper_from": "has paper from",
            "belongs_to_topic": "belongs to topic", "topic_of": "is a topic of",
        }.get(relation, relation)

    def stats(self):
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node["type"]] += 1
        edge_count = sum(len(v) for v in self.adjacency.values())
        return {"total_nodes": len(self.nodes), "node_types": dict(type_counts),
                "total_edges": edge_count, "paper_nodes": len(self.paper_node_ids)}


def build_knowledge_graph(train_parquet, eval_parquet):
    """Build heterogeneous KG from train/eval parquets. ALL references create edges."""
    log("Building Knowledge Graph...")
    kg = HeterogeneousKG()

    for label, parquet_path in [("train", train_parquet), ("eval", eval_parquet)]:
        if not parquet_path.exists():
            continue
        df = pl.read_parquet(parquet_path)
        log(f"  Processing {label}: {df.height} papers")

        for row in df.iter_rows(named=True):
            paper_id = row["paper_id"]
            title = (row.get("title", "") or "").strip()
            abstract = (row.get("abstract", "") or "").strip()
            venue = row.get("venue", None)
            year = row.get("year", None)

            # Add paper node
            node_id = f"P_{paper_id}"
            kg.add_node(node_id, "Paper", {
                "title": title, "abstract": abstract,
                "venue": venue, "year": year, "paper_id": paper_id,
            })

            # Year node
            if year:
                year_id = f"Y_{year}"
                kg.add_node(year_id, "Year", {"name": str(year)})
                kg.add_edge(node_id, year_id, "published_on")

            # Venue node
            if venue and str(venue).strip():
                venue_id = f"V_{venue}"
                kg.add_node(venue_id, "Venue", {"name": str(venue)})
                kg.add_edge(node_id, venue_id, "published_in")

            # References — ALL references create edges (not just positive labels)
            references = json.loads(row["references"]) if isinstance(row["references"], str) else row["references"]
            if references:
                for ref in references:
                    matched_id = ref.get("matched_paper_id")
                    if matched_id:
                        cited_node_id = f"P_{matched_id}"
                        # Add cited paper as a node if not already present
                        if cited_node_id not in kg.nodes:
                            ref_title = ref.get("title", "") or ref.get("title_from_ref", "") or ""
                            kg.add_node(cited_node_id, "Paper", {
                                "title": ref_title, "abstract": "",
                                "paper_id": matched_id,
                            })
                        kg.add_edge(node_id, cited_node_id, "cites")

            # Topics from authors field isn't available, use venue/year as additional structure
            # (Topics would come from llm_enhanced if available in future)

    stats = kg.stats()
    log(f"  KG: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    log(f"  Types: {stats['node_types']}")
    return kg


def perform_random_walks(kg, num_walks=NUM_WALKS_PER_NODE,
                         termination_prob=TERMINATION_PROB,
                         max_length=MAX_WALK_LENGTH, seed=SEED):
    """Attributed random walks on KG, textualized as sentences."""
    log("Performing random walks...")
    random.seed(seed)
    np.random.seed(seed)

    paper_nodes = kg.paper_node_ids
    log(f"  Paper nodes: {len(paper_nodes)}, Walks/node: {num_walks}")

    walks = []
    for paper_node in paper_nodes:
        for _ in range(num_walks):
            current = paper_node
            parts = [kg.textualize_node(current)]
            for step in range(max_length):
                if random.random() < termination_prob and step > 0:
                    break
                neighbors = kg.get_neighbors(current, include_reverse=True)
                if not neighbors:
                    break
                next_node, relation = random.choice(neighbors)
                parts.append(kg.textualize_edge(relation))
                parts.append(kg.textualize_node(next_node))
                current = next_node
            text = " ".join(parts).strip()
            if len(text) > 10:
                walks.append(text)

    log(f"  Generated {len(walks)} textualized walks")
    return walks


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
