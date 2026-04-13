"""
GLAMOR — Fine-Tuning Script (KG + Random Walks + MLM)
=======================================================
Based on: Ali et al., RecSys 2024

GLAMOR is UNSUPERVISED (MLM objective) — no label-specific training.
One model is trained, then evaluated on all 3 label types.

Pipeline:
  Phase 1: Build heterogeneous KG from paper dataset
  Phase 2: Generate attributed random walks and textualize them
  Phase 3: Fine-tune DistilRoBERTa with Masked Language Modeling

Resume: HuggingFace Trainer handles checkpoint resume automatically.

Output:
  - output/graph/GLAMOR/fine_tuned_model/      (HF model dir)
  - output/graph/GLAMOR/corpus/walks.txt
  - output/graph/GLAMOR/training_log.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import random
import gc
import torch
from pathlib import Path
from dotenv import load_dotenv

from glamor_utils import (
    log, build_knowledge_graph, perform_random_walks,
    BASE_MODEL, MAX_SEQ_LENGTH, BATCH_SIZE_MLM, MLM_EPOCHS,
    MASKING_RATIO, LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, SEED,
)

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"

GLAMOR_DIR = OUTPUT_DIR / "graph/GLAMOR/"
GLAMOR_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_DIR = GLAMOR_DIR / "corpus/"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = GLAMOR_DIR / "checkpoints/"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_DIR = GLAMOR_DIR / "fine_tuned_model/"
WALKS_FILE = CORPUS_DIR / "walks.txt"
GRAPH_STATS_FILE = CORPUS_DIR / "graph_stats.json"
LOG_FILE = GLAMOR_DIR / "training_log.json"

TOKENIZED_TRAIN_DIR = CORPUS_DIR / "tokenized_train"
TOKENIZED_VAL_DIR = CORPUS_DIR / "tokenized_val"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_and_cache(walks, tokenizer, cache_dir, split_name):
    from datasets import Dataset as HFDataset
    cache_path = Path(cache_dir)
    if cache_path.exists() and any(cache_path.iterdir()):
        log(f"  Loading cached {split_name} dataset...")
        return HFDataset.load_from_disk(str(cache_path))

    log(f"  Tokenizing {len(walks)} {split_name} walks...")
    raw = HFDataset.from_dict({"text": walks})

    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

    tokenized = raw.map(tok_fn, batched=True, batch_size=5000, num_proc=4,
                        remove_columns=["text"], desc=f"Tokenizing {split_name}")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(cache_path))
    return tokenized


def fine_tune_model(walks):
    from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    log("Phase 3: MLM Fine-Tuning...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    random.shuffle(walks)
    split_idx = int(len(walks) * 0.95)
    train_walks, val_walks = walks[:split_idx], walks[split_idx:]
    log(f"  Train: {len(train_walks)}, Val: {len(val_walks)}")

    train_ds = tokenize_and_cache(train_walks, tokenizer, TOKENIZED_TRAIN_DIR, "train")
    val_ds = tokenize_and_cache(val_walks, tokenizer, TOKENIZED_VAL_DIR, "val")
    del train_walks, val_walks, walks; gc.collect()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MASKING_RATIO)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR), overwrite_output_dir=False,
        num_train_epochs=MLM_EPOCHS, per_device_train_batch_size=BATCH_SIZE_MLM,
        per_device_eval_batch_size=BATCH_SIZE_MLM, learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS,
        logging_steps=500, save_steps=2000, save_total_limit=3,
        eval_strategy="steps", eval_steps=2000,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, fp16=torch.cuda.is_available(),
        dataloader_num_workers=4, dataloader_pin_memory=True,
        report_to="none", seed=SEED,
    )

    existing_ckpts = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))
    resume_ckpt = str(existing_ckpts[-1]) if existing_ckpts else None
    if resume_ckpt:
        log(f"  Resuming from: {resume_ckpt}")

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)

    log("  Starting MLM training...")
    result = trainer.train(resume_from_checkpoint=resume_ckpt)

    training_log = {
        "train_loss": result.training_loss,
        "epochs": MLM_EPOCHS, "base_model": BASE_MODEL,
        "masking_ratio": MASKING_RATIO,
    }
    eval_results = trainer.evaluate()
    training_log["eval_loss"] = eval_results.get("eval_loss", 0)

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    with open(LOG_FILE, "w") as f:
        json.dump(training_log, f, indent=2)

    log(f"  Model saved to {FINAL_MODEL_DIR}")


def main():
    log("=" * 80)
    log("GLAMOR — KG + Random Walks + MLM Fine-Tuning")
    log("=" * 80)

    if FINAL_MODEL_DIR.exists() and (FINAL_MODEL_DIR / "config.json").exists():
        log(f"Final model already exists at {FINAL_MODEL_DIR} — skipping.")
        return

    # Phase 1: Build KG
    kg = build_knowledge_graph(TRAIN_PARQUET, EVAL_PARQUET)
    with open(GRAPH_STATS_FILE, "w") as f:
        json.dump(kg.stats(), f, indent=2)

    # Phase 2: Random walks
    if WALKS_FILE.exists():
        log(f"Loading existing walks from {WALKS_FILE}...")
        with open(WALKS_FILE, "r") as f:
            walks = [line.strip() for line in f if line.strip()]
    else:
        walks = perform_random_walks(kg)
        with open(WALKS_FILE, "w") as f:
            for w in walks: f.write(w + "\n")

    # Phase 3: MLM fine-tuning
    fine_tune_model(walks)

    log("\nGLAMOR fine-tuning complete!")


if __name__ == "__main__":
    main()
