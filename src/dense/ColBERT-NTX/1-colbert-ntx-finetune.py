"""
ColBERT + NT-Xent Fine-Tuning Script
======================================
Fine-tunes ColBERTv2 with NT-Xent contrastive loss, separately for each
of the 3 label types. Uses mean-pooled token embeddings for the contrastive
objective while preserving ColBERT's late-interaction architecture.

Resume: skips completed models, resumes from last epoch checkpoint.

Output (per label type):
  - output/dense/ColBERT-NTX/fine_tuned_models/<type_key>/final_model/
  - output/dense/ColBERT-NTX/fine_tuned_models/<type_key>/epoch_<N>/
"""

import os
import sys
import time
import json
import shutil
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging
from dotenv import load_dotenv
from unittest.mock import MagicMock

hf_logging.set_verbosity_error()

try:
    import git
    try: git.Repo(search_parent_directories=True)
    except: sys.modules['git'] = MagicMock()
except ImportError:
    sys.modules['git'] = MagicMock()

from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig

from colbert_ntx_utils import (
    log, load_paper_text_map, ColBERTNTXentLoss, ColBERTPositivePairsDataset,
    LABEL_TYPES, CHECKPOINT, DOC_MAXLEN, BATCH_SIZE_FT, EPOCHS, LR, TEMPERATURE,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet"

MODELS_DIR = OUTPUT_DIR / "dense/ColBERT-NTX/fine_tuned_models/"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)


def fine_tune_for_label(label_type_key, paper_text_map):
    cfg = LABEL_TYPES[label_type_key]
    model_dir = MODELS_DIR / label_type_key
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_dir = model_dir / "final_model"
    log_file = model_dir / "training_log.json"

    if final_model_dir.exists() and any(final_model_dir.iterdir()):
        log(f"  Final model already exists — skipping.")
        return

    # Find last completed epoch
    completed_epochs = 0
    if log_file.exists():
        with open(log_file, "r") as f:
            training_log = json.load(f)
        completed_epochs = len(training_log.get("epoch_losses", []))
        log(f"  Found {completed_epochs} completed epoch(s).")
    else:
        training_log = {"label_type": label_type_key, "epoch_losses": []}

    if completed_epochs >= EPOCHS:
        last_ckpt = model_dir / f"epoch_{completed_epochs}"
        if last_ckpt.exists():
            shutil.copytree(last_ckpt, final_model_dir, dirs_exist_ok=True)
        return

    # Build dataset
    log(f"  Building positive-pairs dataset...")
    dataset = ColBERTPositivePairsDataset(TRAIN_PARQUET, paper_text_map, label_type_key)
    if len(dataset) == 0:
        log(f"  WARNING: No positive pairs. Skipping.")
        return
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FT, shuffle=True)

    # Initialize ColBERT model
    colbert_config = ColBERTConfig()
    colbert_config.doc_maxlen = DOC_MAXLEN
    colbert_config.query_maxlen = DOC_MAXLEN
    colbert_config.checkpoint = CHECKPOINT

    model = ColBERT(name=CHECKPOINT, colbert_config=colbert_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = ColBERTNTXentLoss(temperature=TEMPERATURE)

    # Resume from checkpoint
    if completed_epochs > 0:
        resume_dir = model_dir / f"epoch_{completed_epochs}"
        if resume_dir.exists():
            log(f"  Resuming from {resume_dir}")
            opt_path = resume_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE, weights_only=False))
            # Reload model from checkpoint dir
            try:
                model = ColBERT(name=str(resume_dir), colbert_config=colbert_config).to(DEVICE)
            except Exception:
                log(f"  Could not reload from checkpoint, loading state_dict fallback")
                sd_path = resume_dir / "model.pt"
                if sd_path.exists():
                    model.load_state_dict(torch.load(sd_path, map_location=DEVICE, weights_only=False))
        else:
            completed_epochs = 0
            training_log["epoch_losses"] = []

    model.train()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for epoch in range(completed_epochs + 1, EPOCHS + 1):
        epoch_loss = 0.0
        num_batches = 0

        for paper_texts, ref_texts in tqdm(dataloader, desc=f"  Epoch {epoch}/{EPOCHS}"):
            try:
                paper_inputs = TOKENIZER(list(paper_texts), padding="max_length", truncation=True,
                                         max_length=DOC_MAXLEN, return_tensors="pt")
                ref_inputs = TOKENIZER(list(ref_texts), padding="max_length", truncation=True,
                                       max_length=DOC_MAXLEN, return_tensors="pt")
                paper_inputs = {k: v.to(DEVICE) for k, v in paper_inputs.items() if k != "token_type_ids"}
                ref_inputs = {k: v.to(DEVICE) for k, v in ref_inputs.items() if k != "token_type_ids"}

                with autocast():
                    paper_emb = model.doc(**paper_inputs)
                    ref_emb = model.doc(**ref_inputs)

                loss = criterion(paper_emb, ref_emb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1
            except Exception as e:
                log(f"  Error in batch: {e}")
                continue

        avg_loss = epoch_loss / max(num_batches, 1)
        log(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # Save checkpoint (ColBERT-style dir + optimizer)
        ckpt_dir = model_dir / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save(str(ckpt_dir))
        except Exception:
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        training_log["epoch_losses"].append(avg_loss)
        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)

    # Save final model
    final_model_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(final_model_dir))
    except Exception:
        torch.save(model.state_dict(), final_model_dir / "model.pt")

    # Always ensure a valid ColBERT config JSON exists for the Indexer
    config_path = final_model_dir / "artifact.metadata"
    if not config_path.exists() or config_path.stat().st_size == 0:
        import ujson
        colbert_meta = {
            "query_token_id": "[unused0]", "doc_token_id": "[unused1]",
            "query_token": "[Q]", "doc_token": "[D]",
            "similarity": "cosine", "dim": 128, "doc_maxlen": DOC_MAXLEN,
            "query_maxlen": DOC_MAXLEN, "mask_punctuation": True,
            "checkpoint": CHECKPOINT,  # original base model name
        }
        with open(config_path, "w") as f:
            ujson.dump(colbert_meta, f)
        log(f"  Wrote ColBERT config to {config_path}")

    log(f"  Final model saved to {final_model_dir}")


def main():
    log("=" * 80)
    log("ColBERT + NT-Xent Fine-Tuning (3 Label Types)")
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

    for label_key, label_cfg in LABEL_TYPES.items():
        log(f"\n{'=' * 80}")
        log(f"Fine-tuning for: {label_cfg['name']}")
        log(f"{'=' * 80}")
        fine_tune_for_label(label_key, paper_text_map)

    log("\nAll fine-tuning complete!")


if __name__ == "__main__":
    main()