"""
SPECTER2 — Embedding Generation Script
========================================
Generates per-paper embeddings using each fine-tuned SPECTER2 model.
Loads the saved model (base + fine-tuned adapter) and embeds train/eval.

Resume: skips models that don't exist, skips already-generated embeddings.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import polars as pl

from specter2_utils import (
    log, load_paper_text_map, get_cls_embedding, PaperTextDataset,
    LABEL_TYPES, BASE_MODEL, MAX_LEN, BATCH_SIZE_EMB,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/SPECTER2/fine_tuned_models/"
EMBEDDINGS_BASE_DIR = OUTPUT_DIR / "dense/SPECTER2/embeddings/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_embeddings_for_dataset(parquet_path, dataset_name, model, tokenizer, output_dir):
    emb_path = output_dir / f"{dataset_name}_embeddings.pt"
    map_path = output_dir / f"{dataset_name}_paper_id_to_index.json"

    if emb_path.exists() and map_path.exists():
        log(f"    {dataset_name} embeddings already exist — skipping.")
        return True

    df = pl.read_parquet(parquet_path)
    paper_ids = df["paper_id"].to_list()
    paper_text_map = load_paper_text_map(parquet_path)

    dataset = PaperTextDataset(paper_ids, paper_text_map, tokenizer, max_length=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_EMB, shuffle=False, num_workers=4, pin_memory=True)

    all_embeddings = []
    all_paper_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    Embedding {dataset_name}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            emb = get_cls_embedding(model, input_ids, attention_mask)
            all_embeddings.append(emb.cpu())
            all_paper_ids.extend(list(batch["paper_id"]))

    final_embeddings = torch.cat(all_embeddings, dim=0)
    log(f"    {dataset_name} shape: {final_embeddings.shape}")

    paper_id_to_idx = {pid: idx for idx, pid in enumerate(all_paper_ids)}

    torch.save({
        "embeddings": final_embeddings,
        "paper_ids": all_paper_ids,
        "model_name": "SPECTER2",
        "embedding_dim": final_embeddings.shape[1],
        "max_length": MAX_LEN,
    }, emb_path)

    with open(map_path, "w") as f:
        json.dump(paper_id_to_idx, f)

    return True


def main():
    log("=" * 80)
    log("SPECTER2 — Embedding Generation")
    log("=" * 80)

    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    generated_count = 0

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Processing: {label_cfg['name']}")
        log(f"{'=' * 80}")

        model_dir = MODELS_DIR / label_key / "final_model"
        if not model_dir.exists():
            log(f"  Fine-tuned model not found — skipping.")
            continue

        log(f"  Loading fine-tuned SPECTER2 model...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoAdapterModel.from_pretrained(str(model_dir))
        model = model.to(DEVICE)
        model.eval()

        emb_dir = EMBEDDINGS_BASE_DIR / label_key
        emb_dir.mkdir(parents=True, exist_ok=True)

        generate_embeddings_for_dataset(TRAIN_PARQUET, "train", model, tokenizer, emb_dir)
        generate_embeddings_for_dataset(EVAL_PARQUET, "eval", model, tokenizer, emb_dir)

        generated_count += 1
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log(f"\nDone! Processed {generated_count}/{len(LABEL_TYPES)} label types.")


if __name__ == "__main__":
    main()
