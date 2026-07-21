"""
KTR — ConceptNet Database Setup (Prerequisite Script)
========================================================
Downloads and converts ConceptNet data into a SQLite database
that the KTR training script needs.

ConceptNet is a large multilingual knowledge graph. We only need
the English subset for concept matching and topic reasoning.

This script:
  1. Downloads ConceptNet assertions CSV (if not cached)
  2. Filters to English-only edges
  3. Builds a SQLite DB with nodes and edges tables

Usage:
  python 0-ktr-setup-conceptnet.py

Then set in your .env:
  CONCEPTNET_DB=/path/to/output/graph/KTR/conceptnet/conceptnet.db

Or pass directly:
  CONCEPTNET_DB=output/graph/KTR/conceptnet/conceptnet.db python 1-ktr-train.py
"""

import os
import sys
import time
import gzip
import sqlite3
import csv
from pathlib import Path
from dotenv import load_dotenv

# -------- Config --------
load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR", "/home/ratul/mustcite/"))
OUTPUT_DIR = ROOT_DIR / "masterset-benchmark/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONCEPTNET_DIR = OUTPUT_DIR / "graph/KTR/conceptnet/"
CONCEPTNET_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = CONCEPTNET_DIR / "conceptnet.db"
ASSERTIONS_GZ = CONCEPTNET_DIR / "conceptnet-assertions-5.7.0.csv.gz"
ASSERTIONS_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def download_conceptnet():
    """Download ConceptNet assertions file."""
    if ASSERTIONS_GZ.exists():
        log(f"  ConceptNet assertions already downloaded: {ASSERTIONS_GZ}")
        return

    log(f"  Downloading ConceptNet assertions (~550MB)...")
    log(f"  URL: {ASSERTIONS_URL}")

    import urllib.request
    urllib.request.urlretrieve(ASSERTIONS_URL, str(ASSERTIONS_GZ))
    log(f"  Downloaded to {ASSERTIONS_GZ}")


def build_database():
    """Parse ConceptNet assertions and build SQLite DB."""
    if DB_PATH.exists():
        log(f"  Database already exists: {DB_PATH}")
        return

    log(f"  Building SQLite database from ConceptNet assertions...")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uri TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            language TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation TEXT NOT NULL,
            start_uri TEXT NOT NULL,
            end_uri TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_uri ON nodes(uri)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_start ON edges(start_uri)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_end ON edges(end_uri)")

    # Parse assertions file
    # Format: /a/[assertion_uri] /r/[Relation] /c/[lang]/[concept] /c/[lang]/[concept] JSON
    node_set = set()
    edge_batch = []
    node_batch = []
    count = 0

    log(f"  Parsing {ASSERTIONS_GZ}...")
    with gzip.open(str(ASSERTIONS_GZ), "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue

            relation = row[1]
            start_uri = row[2]
            end_uri = row[3]

            # Filter to English only
            if not start_uri.startswith("/c/en/") or not end_uri.startswith("/c/en/"):
                continue

            # Extract labels
            start_label = start_uri.split("/")[3].replace("_", " ").lower()
            end_label = end_uri.split("/")[3].replace("_", " ").lower()
            rel_name = relation.split("/")[-1] if "/" in relation else relation

            # Parse weight from JSON metadata
            try:
                import json
                meta = json.loads(row[4]) if len(row) > 4 else {}
                weight = meta.get("weight", 1.0)
            except:
                weight = 1.0

            # Add nodes
            if start_uri not in node_set:
                node_batch.append((start_uri, start_label, "en"))
                node_set.add(start_uri)
            if end_uri not in node_set:
                node_batch.append((end_uri, end_label, "en"))
                node_set.add(end_uri)

            edge_batch.append((rel_name, start_uri, end_uri, weight))
            count += 1

            # Batch insert
            if len(edge_batch) >= 100000:
                if node_batch:
                    cursor.executemany("INSERT OR IGNORE INTO nodes (uri, label, language) VALUES (?, ?, ?)", node_batch)
                    node_batch = []
                cursor.executemany("INSERT INTO edges (relation, start_uri, end_uri, weight) VALUES (?, ?, ?, ?)", edge_batch)
                edge_batch = []
                conn.commit()

                if count % 1000000 == 0:
                    log(f"    Processed {count:,} English edges, {len(node_set):,} nodes...")

    # Final batch
    if node_batch:
        cursor.executemany("INSERT OR IGNORE INTO nodes (uri, label, language) VALUES (?, ?, ?)", node_batch)
    if edge_batch:
        cursor.executemany("INSERT INTO edges (relation, start_uri, end_uri, weight) VALUES (?, ?, ?, ?)", edge_batch)

    conn.commit()

    # Also add a concepts table with just labels for fast lookup
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            label TEXT PRIMARY KEY,
            language TEXT DEFAULT 'en'
        )
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO concepts (label, language)
        SELECT DISTINCT label, language FROM nodes WHERE language='en'
    """)
    conn.commit()

    # Stats
    cursor.execute("SELECT COUNT(*) FROM nodes")
    n_nodes = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM edges")
    n_edges = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM concepts")
    n_concepts = cursor.fetchone()[0]

    conn.close()

    log(f"\n  Database built: {DB_PATH}")
    log(f"  Nodes: {n_nodes:,}")
    log(f"  Edges: {n_edges:,}")
    log(f"  Unique concepts: {n_concepts:,}")


def main():
    log("=" * 80)
    log("KTR — ConceptNet Database Setup")
    log("=" * 80)

    if DB_PATH.exists():
        log(f"ConceptNet DB already exists at {DB_PATH}")
        log(f"\nAdd to your .env file:")
        log(f"  CONCEPTNET_DB={DB_PATH}")
        return

    download_conceptnet()
    build_database()

    log(f"\nSetup complete!")
    log(f"\nAdd to your .env file:")
    log(f"  CONCEPTNET_DB={DB_PATH}")
    log(f"\nThen run: python 1-ktr-train.py")


if __name__ == "__main__":
    main()
