"""
SciNCL — Fine-Tuning Script
==============================
Fine-tunes SciBERT using Neighborhood Contrastive Learning with
citation graph embeddings, separately for each of the 3 label types.

Following Ostendorff et al. (EMNLP 2022):
  - [CLS] token as document representation
  - Triplet margin loss with ξ=1.0
  - Positives from (k+-5, k+] nearest neighbors in citation embedding space
  - Hard negatives from (k_hard-2, k_hard] neighbors
  - Easy negatives via filtered random sampling
  - Sample induced margin between positives and hard negatives

Requires: pre-computed citation embeddings file (citation_embeddings.pt)
  containing {"embeddings": tensor, "paper_ids": list}

Resume: skips completed models, resumes from last epoch checkpoint.

Output (per label type):
  - output/dense/SciNCL/fine_tuned_models/<type_key>/final_model.pt
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from scincl_utils import (
    log, load_paper_text_map,
    SciNCLModel, SciNCLTripletLoss,
    generate_scincl_triplets, SciNCLTripletDataset,
    LABEL_TYPES, MODEL_NAME, MAX_LEN,
    BATCH_SIZE_FT, GRAD_ACCUM_STEPS, EPOCHS, LR, MARGIN,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

# Pre-computed citation graph embeddings (user must provide this)
CITATION_EMBEDDINGS_PATH = DATA_DIR / "citation_embeddings.pt"

MODELS_DIR = OUTPUT_DIR / "dense/SciNCL/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_for_label(label_type_key, paper_text_map, tokenizer):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = model_dir / "final_model.pt"
    log_file = model_dir / "training_log.json"
    triplets_cache = model_dir / "triplets_cache.pt"

    if final_model_path.exists():
        log(f"  Final model already exists — skipping.")
        return

    # Resume
    completed_epochs = 0
    if log_file.exists():
        with open(log_file, "r") as f:
            training_log = json.load(f)
        completed_epochs = len(training_log.get("epoch_losses", []))
        log(f"  Found {completed_epochs} completed epoch(s).")
    else:
        training_log = {"label_type": label_type_key, "epoch_losses": []}

    if completed_epochs >= EPOCHS:
        last_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            torch.save(ckpt["model_state_dict"], final_model_path)
        return

    # Generate or load cached triplets
    if triplets_cache.exists():
        log(f"  Loading cached triplets...")
        triplets = torch.load(triplets_cache, weights_only=False)
    else:
        triplets = generate_scincl_triplets(
            TRAIN_PARQUET, paper_text_map, CITATION_EMBEDDINGS_PATH, label_type_key,
        )
        torch.save(triplets, triplets_cache)

    if len(triplets) == 0:
        log(f"  WARNING: No triplets generated. Skipping.")
        return

    dataset = SciNCLTripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    model = SciNCLModel(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = SciNCLTripletLoss(margin=MARGIN)

    # Resume
    if completed_epochs > 0:
        resume_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if resume_ckpt.exists():
            log(f"  Resuming from {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            completed_epochs = 0
            training_log["epoch_losses"] = []

    model.train()

    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (q_texts, p_texts, n_texts) in enumerate(tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}")):
            q_enc = tokenizer(list(q_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            p_enc = tokenizer(list(p_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            n_enc = tokenizer(list(n_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

            q_emb = model(q_enc.input_ids, q_enc.attention_mask)
            p_emb = model(p_enc.input_ids, p_enc.attention_mask)
            n_emb = model(n_enc.input_ids, n_enc.attention_mask)

            loss = criterion(q_emb, p_emb, n_emb) / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, model_dir / f"epoch_{epoch}.pt")

        training_log["epoch_losses"].append(avg_loss)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    torch.save(model.state_dict(), final_model_path)
    log(f"  Final model saved to {final_model_path}")


def main():
    log("=" * 80)
    log("SciNCL — Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    # Check citation embeddings
    if not CITATION_EMBEDDINGS_PATH.exists():
        log(f"ERROR: Citation embeddings not found at {CITATION_EMBEDDINGS_PATH}")
        log("Please provide pre-computed citation graph embeddings.")
        log("Expected format: {{\"embeddings\": tensor [N, dim], \"paper_ids\": list of str}}")
        return

    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Combined: {len(paper_text_map)} papers")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map, tokenizer)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()
