import re
import time
from collections import Counter
import pandas as pd
from tqdm import tqdm

# ======================================================================
# Tokenization
# ======================================================================
SEP = "[SEP]"

# Simple stopwords (common English words that don't carry meaning for retrieval)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "but", "and", "or", "if", "while", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "we", "our", "you", "your", "i", "me", "my", "which", "who", "whom",
    "what", "sep",  # Also remove [SEP] token remnants
}

def tokenize(text):
    """
    Simple whitespace + punctuation tokenizer with lowercasing and stopword removal.
    Good enough for BM25 — we want to match on content words.
    """
    # Lowercase and remove non-alphanumeric characters (keep spaces)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split and filter
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ======================================================================
# Text Loading
# ======================================================================
def load_paper_text_map(parquet_file):
    """Build mapping with title + abstract"""
    df = pd.read_parquet(parquet_file)
    paper_map = {}
    for _, row in df.iterrows():
        paper_id = row["paper_id"]
        title = row.get("title", "") or ""
        abstract = row.get("abstract", "") or ""

        # also strip those fields
        title = title.strip()
        abstract = abstract.strip()

        combined_text = f"{title} {SEP} {abstract}"
        paper_map[paper_id] = combined_text

    return paper_map


# ======================================================================
# Build inverted index for fast scoring in evaluation
# ======================================================================

def build_inverted_index(index):
    """
    Build an inverted index: term -> list of doc indices that contain it.
    This makes BM25 scoring O(matching_docs) instead of O(all_docs) per query term.
    """
    log("  Building inverted index for fast scoring...")
    inv_idx = {}
    for doc_idx, tf in enumerate(tqdm(index["tf"], desc="Inverted index")):
        for term in tf:
            if term not in inv_idx:
                inv_idx[term] = []
            inv_idx[term].append(doc_idx)
    log(f"  Inverted index: {len(inv_idx):,} terms")
    return inv_idx