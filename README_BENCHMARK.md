# MasterSet Benchmark

## Overview

This directory contains baseline experiments and analysis for the MasterSet dataset. It includes dense retrieval models, sparse retrieval, graph-based methods, and comprehensive dataset analysis.

---

## Structure

### 1. Dense Retrieval Models (`src/dense/`)

Baseline dense retrieval models for paper recommendation:

- **ColBERT / ColBERT-NTX**: Dense passage retrieval with late interaction
- **SPECTER / SPECTER2**: Citation-aware scientific paper embeddings
- **SciBERT / SciBERT-NTX**: Scientific BERT-based embeddings
- **SciNCL**: Scientific paper representation learning
- **HAtten-RR**: Hierarchical attention with relevance ranking

Each model includes:
- Finetuning script (Except Pretrained ones: SciBERT, SPECTER, SPECTER2, SciNCL, ColBERT)
- Embedding generation
- Evaluation script

### 2. Sparse Retrieval (`src/sparse/`)

- **BM25**: Traditional sparse retrieval baseline using BM25 ranking

### 3. Graph-Based Methods (`src/graph/`)

Knowledge graph and citation network based approaches:

- **AR-GNN**: Attributed Relational Graph Neural Network
- **KTR**: Knowledge Triples with ConceptNet reasoning
- **LitFM**: Literature Knowledge Fact Mining

### 4. Analysis (`src/analysis/`)

Dataset and model analysis scripts:

- **Annotation Scores**: Inter-annotator agreement (IAA) computation and LLM-human agreement analysis
- **Label Distributions**: Citation context type distributions across the dataset

---

## Output

Results and intermediate outputs are saved in the `output/` directory including:

- Model embeddings
- Evaluation metrics
- Analysis reports
