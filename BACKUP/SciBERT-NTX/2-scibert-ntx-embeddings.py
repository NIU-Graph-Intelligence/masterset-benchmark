"""
2-scibert-ntx-embeddings.py
Stage 2: Generate embeddings using the fine-tuned SciBERT-NTX models.

For each label type whose fine-tuned model exists:
  - Embed ALL papers from papers.parquet using that model.
  - Save embeddings + paper_id mapping as .pt files.
  - Skip if embeddings already exist (resume-safe).

This script does NOT require all 3 models to be ready.
It processes whichever models are available.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import polars as pl
from dotenv import load_dotenv

from scibert_ntx_utils import (
    LABEL_TYPES,
    load_paper_text_map,
    ContrastiveSciBERT,
)

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/home/ratul/masterset-benchmark/output"))

PAPERS_PARQUET = DATA_DIR / "papers.parquet"

MODEL_NAME = "allenai/scibert_scivocab_uncased"
EMBED_DIM = 128
MAX_LEN = 512
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
FT_MODEL_DIR = OUTPUT_DIR / "fine_tuned_models" / "SciBERT-NTX"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings" / "SciBERT-NTX"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Dataset — all papers for embedding
# ──────────────────────────────────────────────
class AllPapersDataset(Dataset):
    """Simple dataset of (paper_id, text) for all papers."""

    def __init__(self, paper_text_map: dict):
        self.paper_ids = list(paper_text_map.keys())
        self.texts = [paper_text_map[pid] for pid in self.paper_ids]

    def __len__(self):
        return len(self.paper_ids)

    def __getitem__(self, idx):
        return self.paper_ids[idx], self.texts[idx]


# ──────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────
def final_model_path(label_type: str) -> Path:
    return FT_MODEL_DIR / label_type / "model_final.pt"


def embeddings_path(label_type: str) -> Path:
    return EMBEDDINGS_DIR / f"{label_type}_embeddings.pt"


def is_model_ready(label_type: str) -> bool:
    return final_model_path(label_type).exists()


def is_embeddings_done(label_type: str) -> bool:
    return embeddings_path(label_type).exists()


# ──────────────────────────────────────────────
# Embedding generation
# ──────────────────────────────────────────────
def generate_embeddings(label_type: str, paper_text_map: dict, tokenizer):
    print(f"\n{'='*60}")
    print(f"Generating embeddings for: {label_type}")
    print(f"{'='*60}")

    if is_embeddings_done(label_type):
        print(f"  ✓ Embeddings already exist — skipping.")
        return

    if not is_model_ready(label_type):
        print(f"  ⏳ Fine-tuned model not ready yet — skipping.")
        return

    # Load model
    model = ContrastiveSciBERT(MODEL_NAME, embed_dim=EMBED_DIM).to(DEVICE)
    model.load_state_dict(
        torch.load(final_model_path(label_type), map_location=DEVICE)
    )
    model.eval()
    print(f"  Loaded model from: {final_model_path(label_type)}")

    # Dataset
    dataset = AllPapersDataset(paper_text_map)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Papers to embed: {len(dataset)}")

    all_embeddings = []
    all_paper_ids = []

    with torch.no_grad():
        for paper_ids_batch, texts_batch in tqdm(loader, desc=f"  Embedding [{label_type}]"):
            enc = tokenizer(
                list(texts_batch),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            emb = model(enc.input_ids, enc.attention_mask)
            all_embeddings.append(emb.cpu())
            all_paper_ids.extend(paper_ids_batch)

    embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [N, embed_dim]
    print(f"  Embeddings shape: {embeddings_tensor.shape}")

    # Save
    save_data = {
        "embeddings": embeddings_tensor,
        "paper_ids": all_paper_ids,  # list of str UUIDs
    }
    torch.save(save_data, embeddings_path(label_type))
    print(f"  ✓ Saved: {embeddings_path(label_type)}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SciBERT NT-Xent — Embedding Generation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Papers: {PAPERS_PARQUET}")
    print(f"Output: {EMBEDDINGS_DIR}")

    # Status check
    print("\nModel availability:")
    for lt in LABEL_TYPES:
        ready = "✓ Ready" if is_model_ready(lt) else "✗ Not ready"
        done = " (embeddings exist)" if is_embeddings_done(lt) else ""
        print(f"  {lt}: {ready}{done}")

    # Load data
    print("\nLoading paper text map...")
    paper_text_map = load_paper_text_map(PAPERS_PARQUET)
    print(f"  Loaded {len(paper_text_map)} papers")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Generate embeddings for each available model
    for label_type in LABEL_TYPES:
        generate_embeddings(label_type, paper_text_map, tokenizer)

    print("\n" + "=" * 60)
    print("Embedding generation complete!")
    print("=" * 60)
    for lt in LABEL_TYPES:
        status = "✓ Done" if is_embeddings_done(lt) else "✗ Pending"
        print(f"  {lt}: {status}")


if __name__ == "__main__":
    main()