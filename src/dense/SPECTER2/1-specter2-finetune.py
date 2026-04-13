"""
SPECTER2 — Fine-Tuning Script
================================
Fine-tunes SPECTER2 (base + proximity adapter) with Triplet Margin Loss,
separately for each of the 3 label types.

Architecture: allenai/specter2_base with allenai/specter2 proximity adapter.
During training, only the adapter weights are updated (base transformer frozen).
[CLS] token from final layer is the document representation.

Resume: skips completed models, resumes from last epoch checkpoint.

Output (per label type):
  - output/dense/SPECTER2/fine_tuned_models/<type_key>/final_model/
  - output/dense/SPECTER2/fine_tuned_models/<type_key>/epoch_<N>.pt
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from specter2_utils import (
    log, load_paper_text_map, load_specter2_model, get_cls_embedding,
    TripletMarginLoss, SPECTER2TripletDataset,
    LABEL_TYPES, BASE_MODEL, MAX_LEN,
    BATCH_SIZE_FT, GRAD_ACCUM_STEPS, EPOCHS, LR, MARGIN,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/SPECTER2/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_for_label(label_type_key, paper_text_map):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_dir = model_dir / "final_model"
    log_file = model_dir / "training_log.json"

    if final_model_dir.exists() and any(final_model_dir.iterdir()):
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
        # Save final from last checkpoint
        last_ckpt = model_dir / f"epoch_{completed_epochs}.pt"
        if last_ckpt.exists():
            log(f"  All epochs done. Saving final model from last checkpoint.")
            model, tokenizer = load_specter2_model(load_adapter=True)
            ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            final_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
        return

    # Dataset
    log(f"  Building triplet dataset...")
    dataset = SPECTER2TripletDataset(TRAIN_PARQUET, paper_text_map, label_type_key)
    if len(dataset) == 0:
        log(f"  WARNING: No triplets. Skipping.")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True, num_workers=2, pin_memory=True)

    # Model: load fresh SPECTER2 base + adapter
    log(f"  Loading SPECTER2 model with proximity adapter...")
    model, tokenizer = load_specter2_model(load_adapter=True)
    model = model.to(DEVICE)

    # Only train adapter parameters (freeze base transformer)
    # The adapter library handles this: adapter params are trainable, base is frozen
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    criterion = TripletMarginLoss(margin=MARGIN)

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

    model.train()

    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for batch_idx, (query_texts, pos_texts, neg_texts) in enumerate(tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}")):
            # Tokenize (no token_type_ids for SPECTER2)
            q_enc = tokenizer(list(query_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt", return_token_type_ids=False).to(DEVICE)
            p_enc = tokenizer(list(pos_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt", return_token_type_ids=False).to(DEVICE)
            n_enc = tokenizer(list(neg_texts), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt", return_token_type_ids=False).to(DEVICE)

            q_emb = get_cls_embedding(model, q_enc["input_ids"], q_enc["attention_mask"])
            p_emb = get_cls_embedding(model, p_enc["input_ids"], p_enc["attention_mask"])
            n_emb = get_cls_embedding(model, n_enc["input_ids"], n_enc["attention_mask"])

            loss = criterion(q_emb, p_emb, n_emb) / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
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

    # Save final model (HuggingFace format for easy loading later)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    log(f"  Final model saved to {final_model_dir}")


def main():
    log("=" * 80)
    log("SPECTER2 — Fine-Tuning (3 Label Types)")
    log("=" * 80)
    log(f"Device: {DEVICE}")
    log(f"Base: {BASE_MODEL}, Adapter: proximity")

    log("\nLoading paper text maps...")
    train_map = load_paper_text_map(TRAIN_PARQUET)
    if Path(ALL_PAPERS_PARQUET).exists():
        all_map = load_paper_text_map(ALL_PAPERS_PARQUET)
        paper_text_map = {**all_map, **train_map}
    else:
        paper_text_map = train_map
    log(f"  Combined: {len(paper_text_map)} papers")

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()
