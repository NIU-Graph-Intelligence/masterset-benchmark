"""
SPECTER (Scalar Mix) — Fine-Tuning Script
============================================
Fine-tunes SciBERT with scalar mixing of all layers + average pooling +
triple margin loss, separately for each of the 3 label types.

This follows the original SPECTER methodology from AllenAI.

Resume: skips completed models, resumes from last epoch checkpoint.

Output (per label type):
  - output/dense/SPECTER/fine_tuned_models/<type_key>/final_model.pt
  - output/dense/SPECTER/fine_tuned_models/<type_key>/epoch_<N>.pt
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from specter_utils import (
    log, load_paper_text_map,
    SPECTERScalarMixModel, SPECTERLoss, SPECTERTripletDataset,
    LABEL_TYPES, MODEL_NAME, MAX_LEN, BATCH_SIZE_FT, EPOCHS, LR, MARGIN,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/SPECTER/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_for_label(label_type_key, paper_text_map, tokenizer):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = model_dir / "final_model.pt"
    log_file = model_dir / "training_log.json"

    if final_model_path.exists():
        log(f"  Final model already exists — skipping.")
        return

    # Resume check
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
            log(f"  All epochs done. Extracting final model.")
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            torch.save(ckpt["model_state_dict"], final_model_path)
        return

    # Build dataset
    log(f"  Building triplet dataset...")
    dataset = SPECTERTripletDataset(TRAIN_PARQUET, paper_text_map, label_type_key)
    if len(dataset) == 0:
        log(f"  WARNING: No triplets. Skipping.")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    model = SPECTERScalarMixModel(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = SPECTERLoss(margin=MARGIN)

    # LR scheduler
    total_steps = len(dataloader) * (EPOCHS - completed_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )

    # Resume from checkpoint
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

    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for query_texts, pos_texts, neg_texts in tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}"):
            query_enc = tokenizer(list(query_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            pos_enc = tokenizer(list(pos_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            neg_enc = tokenizer(list(neg_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)

            query_emb = model(query_enc.input_ids, query_enc.attention_mask)
            pos_emb = model(pos_enc.input_ids, pos_enc.attention_mask)
            neg_emb = model(neg_enc.input_ids, neg_enc.attention_mask)

            loss = criterion(query_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, model_dir / f"epoch_{epoch}.pt")

        training_log["epoch_losses"].append(avg_loss)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    log(f"  Final model saved to {final_model_path}")


def main():
    log("=" * 80)
    log("SPECTER (Scalar Mix) — Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")

    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Combined text map: {len(paper_text_map)} papers")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map, tokenizer)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()
