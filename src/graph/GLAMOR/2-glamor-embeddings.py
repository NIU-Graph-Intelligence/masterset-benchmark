"""
GLAMOR — Embedding Generation Script
=======================================
Passes each paper's title + abstract through the fine-tuned DistilRoBERTa
and extracts the [CLS] token from the last hidden state.

GLAMOR is unsupervised — one set of embeddings for all label types.
Generates separate train and eval embedding files.

Resume: skips already-generated files.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

from glamor_utils import log, load_paper_text_map, BATCH_SIZE_EMB, MAX_SEQ_LENGTH, SEP

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

GLAMOR_DIR = OUTPUT_DIR / "graph/GLAMOR/"
FINAL_MODEL_DIR = GLAMOR_DIR / "fine_tuned_model/"

EMBED_DIR = GLAMOR_DIR / "embeddings/"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings_for_dataset(parquet_path, dataset_name, model, tokenizer, output_dir):
    emb_path = output_dir / f"{dataset_name}_embeddings.pt"
    map_path = output_dir / f"{dataset_name}_paper_id_to_index.json"

    if emb_path.exists() and map_path.exists():
        log(f"  {dataset_name} embeddings already exist — skipping.")
        return

    df = pl.read_parquet(parquet_path)
    paper_ids = df["paper_id"].to_list()
    paper_text_map = load_paper_text_map(parquet_path)

    all_embeddings = []
    all_paper_ids = []

    model.eval()
    for i in tqdm(range(0, len(paper_ids), BATCH_SIZE_EMB), desc=f"  Embedding {dataset_name}"):
        batch_ids = paper_ids[i:i + BATCH_SIZE_EMB]
        batch_texts = [paper_text_map.get(pid, "") for pid in batch_ids]

        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] = position 0 (<s> token for RoBERTa)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)

        all_embeddings.append(cls_emb.cpu())
        all_paper_ids.extend(batch_ids)

    final_emb = torch.cat(all_embeddings, dim=0)
    log(f"  {dataset_name} shape: {final_emb.shape}")

    paper_id_to_idx = {pid: idx for idx, pid in enumerate(all_paper_ids)}

    torch.save({
        "embeddings": final_emb, "paper_ids": all_paper_ids,
        "model_name": "GLAMOR (DistilRoBERTa)", "embedding_dim": final_emb.shape[1],
    }, emb_path)

    with open(map_path, "w") as f:
        json.dump(paper_id_to_idx, f)


def main():
    log("=" * 80)
    log("GLAMOR — Embedding Generation")
    log("=" * 80)

    if not FINAL_MODEL_DIR.exists() or not (FINAL_MODEL_DIR / "config.json").exists():
        log(f"ERROR: Fine-tuned model not found at {FINAL_MODEL_DIR}")
        return

    log(f"Loading fine-tuned model from {FINAL_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModel.from_pretrained(str(FINAL_MODEL_DIR)).to(DEVICE)
    model.eval()
    log(f"  Hidden size: {model.config.hidden_size}")

    generate_embeddings_for_dataset(TRAIN_PARQUET, "train", model, tokenizer, EMBED_DIR)
    generate_embeddings_for_dataset(EVAL_PARQUET, "eval", model, tokenizer, EMBED_DIR)

    log("\nEmbedding generation complete!")


if __name__ == "__main__":
    main()
