"""
1-scibert-ntx-finetune.py
Stage 1: Fine-tune SciBERT with NT-Xent loss — one model per label type.

Resume logic:
  - If a label's final model file exists → skip entirely.
  - If a label's training was interrupted mid-epoch → resume from last checkpoint.
  - Checkpoints saved every epoch; training log tracks progress.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import polars as pl
from dotenv import load_dotenv

from scibert_ntx_utils import (
    LABEL_TYPES,
    is_positive,
    load_paper_text_map,
    ContrastiveSciBERT,
    NTXentLoss,
)

load_dotenv()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/home/ratul/masterset-benchmark/output"))

TRAIN_PARQUET = DATA_DIR / "train.parquet"
PAPERS_PARQUET = DATA_DIR / "papers.parquet"

MODEL_NAME = "allenai/scibert_scivocab_uncased"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
EMBED_DIM = 128
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
SAVE_DIR = OUTPUT_DIR / "fine_tuned_models" / "SciBERT-NTX"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class PaperReferenceDataset(Dataset):
    """
    Builds (paper_text, ref_text) pairs from positive references
    for a given label type.  Only positive pairs are used for
    contrastive learning — negatives come from in-batch samples.
    """

    def __init__(self, parquet_file: Path, paper_text_map: dict, label_type: str):
        self.pairs = []
        df = pl.read_parquet(str(parquet_file))

        for row in df.iter_rows(named=True):
            paper_id = row["paper_id"]
            paper_text = paper_text_map.get(paper_id, "")
            if not paper_text:
                continue
            for ref in row["references"]:
                if not is_positive(ref, label_type):
                    continue
                ref_id = ref.get("matched_paper_id")
                ref_text = paper_text_map.get(ref_id)
                if ref_text:
                    self.pairs.append((paper_text, ref_text))

        print(f"  [{label_type}] Loaded {len(self.pairs)} positive pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]


# ──────────────────────────────────────────────
# Resume helpers
# ──────────────────────────────────────────────
def get_label_dir(label_type: str) -> Path:
    d = SAVE_DIR / label_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def final_model_path(label_type: str) -> Path:
    return get_label_dir(label_type) / "model_final.pt"


def checkpoint_path(label_type: str, epoch: int) -> Path:
    return get_label_dir(label_type) / f"checkpoint_epoch{epoch}.pt"


def log_path(label_type: str) -> Path:
    return get_label_dir(label_type) / "training_log.json"


def is_training_complete(label_type: str) -> bool:
    return final_model_path(label_type).exists()


def load_training_log(label_type: str) -> dict:
    lp = log_path(label_type)
    if lp.exists():
        with open(lp) as f:
            return json.load(f)
    return {"epoch_losses": [], "completed_epochs": 0, "label_type": label_type}


def save_training_log(label_type: str, log: dict):
    with open(log_path(label_type), "w") as f:
        json.dump(log, f, indent=2)


def find_latest_checkpoint(label_type: str) -> tuple[int, Path | None]:
    """Return (last_completed_epoch, checkpoint_path) or (0, None)."""
    label_dir = get_label_dir(label_type)
    best_epoch = 0
    best_path = None
    for f in label_dir.glob("checkpoint_epoch*.pt"):
        try:
            epoch = int(f.stem.split("epoch")[1])
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = f
        except (ValueError, IndexError):
            continue
    return best_epoch, best_path


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_one_label(label_type: str, paper_text_map: dict, tokenizer):
    cfg = LABEL_TYPES[label_type]
    print(f"\n{'='*60}")
    print(f"Training for {label_type}: {cfg['description']}")
    print(f"{'='*60}")

    if is_training_complete(label_type):
        print(f"  ✓ Final model already exists — skipping.")
        return

    # Check for resume
    resume_epoch, resume_ckpt = find_latest_checkpoint(label_type)
    if resume_epoch > 0:
        print(f"  ↻ Resuming from epoch {resume_epoch} checkpoint")

    # Dataset
    dataset = PaperReferenceDataset(TRAIN_PARQUET, paper_text_map, label_type)
    if len(dataset) == 0:
        print(f"  ⚠ No positive pairs found for {label_type} — skipping.")
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    model = ContrastiveSciBERT(MODEL_NAME, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * (EPOCHS - resume_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = NTXentLoss()

    # Load checkpoint if resuming
    training_log = load_training_log(label_type)
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  Loaded checkpoint: {resume_ckpt.name}")

    # Train remaining epochs
    for epoch in range(resume_epoch + 1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"  Epoch {epoch}/{EPOCHS}")
        for paper_texts, ref_texts in pbar:
            enc1 = tokenizer(
                list(paper_texts),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            enc2 = tokenizer(
                list(ref_texts),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            emb1 = model(enc1.input_ids, enc1.attention_mask)
            emb2 = model(enc2.input_ids, enc2.attention_mask)

            loss = criterion(emb1, emb2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # Update log
        training_log["epoch_losses"].append({"epoch": epoch, "avg_loss": avg_loss})
        training_log["completed_epochs"] = epoch
        save_training_log(label_type, training_log)

        # Save checkpoint
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "avg_loss": avg_loss,
        }
        torch.save(ckpt_data, checkpoint_path(label_type, epoch))
        print(f"  Checkpoint saved: checkpoint_epoch{epoch}.pt")

    # Save final model (just the model weights for downstream use)
    torch.save(model.state_dict(), final_model_path(label_type))
    training_log["status"] = "complete"
    save_training_log(label_type, training_log)
    print(f"  ✓ Final model saved: {final_model_path(label_type)}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SciBERT NT-Xent Fine-Tuning — 3 Label Types")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Train data: {TRAIN_PARQUET}")
    print(f"Output dir: {SAVE_DIR}")

    # Load paper texts
    print("\nLoading paper text map...")
    paper_text_map = load_paper_text_map(PAPERS_PARQUET)
    print(f"  Loaded {len(paper_text_map)} papers")

    # Load tokenizer once (shared across all label types)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Train one model per label type
    for label_type in LABEL_TYPES:
        train_one_label(label_type, paper_text_map, tokenizer)

    print("\n" + "=" * 60)
    print("All fine-tuning complete!")
    print("=" * 60)
    for lt in LABEL_TYPES:
        status = "✓ Done" if is_training_complete(lt) else "✗ Incomplete"
        print(f"  {lt}: {status}")


if __name__ == "__main__":
    main()