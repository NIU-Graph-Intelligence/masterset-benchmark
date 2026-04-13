import os
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import polars as pl
import json
from tqdm import tqdm
from dotenv import load_dotenv

from scibert_utils import generate_embeddings, SEP, MODEL_NAME, MAX_LENGTH, BATCH_SIZE

# -------- Config --------
load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQUET = DATA_DIR / "train_eval_set/v2.0/train_v2.0.parquet"
EVAL_PARQUET = DATA_DIR / "train_eval_set/v2.0/eval_v2.0.parquet"
ALL_PAPERS_PARQUET = DATA_DIR / "train_eval_set/v2.0/all_papers_with_refs_and_labels_v2.0.parquet" # not sure if needed or not

OUTPUT_EMBEDDINGS_DIR = OUTPUT_DIR / "dense/SciBERT/embeddings/"
OUTPUT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)



def main():
    """Main execution function"""
    print(f"\n{'='*80}")
    print("SciBERT Embedding Generation with title [SEP] abstract")
    print(f"{'='*80}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("Model loaded successfully!")

    # Generate embeddings for train set
    print("\n" + "=" * 80)
    print("TRAIN SET")
    print("=" * 80)
    train_embeddings, train_ids, train_mapping, train_output_data = generate_embeddings(
        TRAIN_PARQUET, "train", tokenizer, model, device
    )

    # Saving Train Embeddings
    train_embeddings_path = OUTPUT_EMBEDDINGS_DIR / f"train_embeddings.pt"
    torch.save(train_output_data, train_embeddings_path)
    print(f"Saved embeddings to: {train_embeddings_path}")

    # Save Train ID mapping
    train_mapping_path = OUTPUT_EMBEDDINGS_DIR / f"train_paper_id_to_index.json"
    with open(train_mapping_path, "w") as f:
        json.dump(train_mapping, f)
    print(f"Saved ID mapping to: {train_mapping_path}")


    # Generate embeddings for eval set
    print("\n" + "=" * 80)
    print("EVAL SET")
    print("=" * 80)
    eval_embeddings, eval_ids, eval_mapping, eval_output_data = generate_embeddings(
        EVAL_PARQUET, "eval", tokenizer, model, device
    )

    # Saving Eval Embeddings
    eval_embeddings_path = OUTPUT_EMBEDDINGS_DIR / f"eval_embeddings.pt"
    torch.save(eval_output_data, eval_embeddings_path)
    print(f"Saved embeddings to: {eval_embeddings_path}")

    # Save Eval ID mapping
    eval_mapping_path = OUTPUT_EMBEDDINGS_DIR / f"eval_paper_id_to_index.json"
    with open(eval_mapping_path, "w") as f:
        json.dump(eval_mapping, f, indent=2)
    print(f"Saved ID mapping to: {eval_mapping_path}")

    # Save combined metadata
    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dim": train_embeddings.shape[1],
        "max_length": MAX_LENGTH,
        "train_papers": len(train_ids),
        "eval_papers": len(eval_ids),
        "train_parquet": str(TRAIN_PARQUET),
        "eval_parquet": str(EVAL_PARQUET),
    }

    metadata_path = OUTPUT_EMBEDDINGS_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Eval embeddings: {eval_embeddings.shape}")
    print(f"All files saved to: {OUTPUT_EMBEDDINGS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()