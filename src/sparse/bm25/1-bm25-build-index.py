"""
BM25 Baseline — Index Building Script
=======================================
Builds a BM25 index over all training papers using title + abstract.

BM25 (Best Match 25) is a classic term-frequency based retrieval method.
No training or fine-tuning is needed — it just counts how often query terms
appear in candidate documents, with diminishing returns for repeated terms
and normalization for document length.

Input:
  - data/train_eval_set/v2.0/train_v2.0.csv                              (candidate library)
  - data/train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.csv    (fallback text)

Output:
  - output/sparse/bm25/index/bm25_index.pkl  (serialized index: tokenized corpus + paper IDs + parameters)
"""

import os
import sys
import time
import json
import math
import pickle
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from bm25_utils import load_paper_text_map, tokenize

# -------- Config --------
load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

INDEX_DIR = OUTPUT_DIR / "sparse/bm25/index/"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = INDEX_DIR / "bm25_index.pkl"
METADATA_FILE = INDEX_DIR / "index_metadata.json"

SEP = "[SEP]"

# BM25 hyperparameters (standard defaults)
BM25_K1 = 1.5    # Term frequency saturation parameter
BM25_B = 0.75    # Document length normalization parameter


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ======================================================================
# BM25 Index Building
# ======================================================================

def build_bm25_index(paper_ids, tokenized_docs):
    """
    Build a BM25 index from pre-tokenized documents.

    Returns a dict containing everything needed for scoring:
      - paper_ids: list of paper IDs (aligned with doc index)
      - doc_lengths: list of document lengths
      - avgdl: average document length
      - df: dict mapping term -> number of documents containing it
      - tf: list of Counter objects (one per doc)
      - N: total number of documents
      - k1, b: BM25 parameters
    """
    N = len(tokenized_docs)
    log(f"  Building BM25 index over {N:,} documents...")

    # Document frequencies: how many docs contain each term
    df = Counter()
    doc_lengths = []
    tf_list = []  # term frequencies per document

    for tokens in tqdm(tokenized_docs, desc="Computing TF/DF"):
        tf = Counter(tokens)
        tf_list.append(tf)
        doc_lengths.append(len(tokens))
        # Each unique term in this doc gets +1 to its document frequency
        for term in tf:
            df[term] += 1

    avgdl = np.mean(doc_lengths) if doc_lengths else 1.0

    log(f"  Vocabulary size: {len(df):,}")
    log(f"  Avg document length: {avgdl:.1f} tokens")
    log(f"  Total documents: {N:,}")

    index = {
        "paper_ids": paper_ids,
        "doc_lengths": doc_lengths,
        "avgdl": float(avgdl),
        "df": dict(df),
        "tf": tf_list,
        "N": N,
        "k1": BM25_K1,
        "b": BM25_B,
    }

    return index


def main():
    log("=" * 80)
    log("BM25 Baseline — Index Building")
    log("=" * 80)
    log(f"Python: {sys.version.split()[0]}")

    # ---- Load paper texts ----
    # Train papers are the candidate library
    train_map = load_paper_text_map(TRAIN_PARQUET)
    # All papers are the corpus
    # all_map = load_paper_text_map(ALL_PAPERS_PARQUET) # If needed, not sure right now
    # log(f"  Total papers in text map: {len(train_map):,}")

    log("\nLoading train paper IDs...")
    train_df = pd.read_parquet(TRAIN_PARQUET, columns=["paper_id"])
    train_paper_ids = sorted(train_df["paper_id"].tolist())
    log(f"  Train papers (candidates): {len(train_paper_ids):,}")

    # ---- Tokenize all candidate documents ----
    log("\nTokenizing candidate documents...")
    paper_ids = []
    tokenized_docs = []
    missing_text = 0

    for pid in tqdm(train_paper_ids, desc="Tokenizing"):
        text = train_map.get(pid, "")
        if not text.strip():
            missing_text += 1
            tokens = []
        else:
            tokens = tokenize(text)
        paper_ids.append(pid)
        tokenized_docs.append(tokens)

    if missing_text > 0:
        log(f"  Missing text for {missing_text} train papers.")

    # ---- Build BM25 index ----
    log("\nBuilding BM25 index...")
    index = build_bm25_index(paper_ids, tokenized_docs)

    # ---- Save index ----
    log(f"\nSaving index to {INDEX_FILE}...")
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = INDEX_FILE.stat().st_size / (1024 * 1024)
    log(f"  Index size: {file_size_mb:.1f} MB")

    metadata = {
        "method": "BM25",
        "k1": BM25_K1,
        "b": BM25_B,
        "num_candidates": len(paper_ids),
        "vocab_size": len(index["df"]),
        "avg_doc_length": index["avgdl"],
        "text_source": "title + abstract",
        "tokenization": "lowercase + punctuation removal + stopword filtering",
        "index_file": str(INDEX_FILE),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    log("\n" + "=" * 80)
    log("BM25 index built successfully!")
    log(f"  Candidates: {len(paper_ids):,}")
    log(f"  Vocab: {len(index['df']):,} terms")
    log(f"  Saved to: {INDEX_FILE}")
    log("=" * 80)

if __name__ == "__main__":
    main()