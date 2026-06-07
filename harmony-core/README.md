# HARMONY-Core v1

HARMONY-Core is a real, runnable codebase implementing the simplified HARMONY architecture. 
It emphasizes adaptive computation over uniform attention by utilizing semantic chunking, local context mixing, a selective state backbone, and external retrieval memory.

---

## Codebase Architecture Overview

HARMONY-Core is structured into modular components:

```
src/harmony/
├── backbone/          # Gated recurrence, state-spaces, and sparse Mixture of Experts (MoE)
├── chunking/          # Semantic chunker converting raw inputs into structured blocks
├── config/            # System configuration settings built using Pydantic
├── data/              # Dataset utilities for downloading and preprocessing corpora (e.g., WikiText/OpenWebText)
├── embeddings/        # Sentence-transformer density vector embeddings loading and caching
├── evaluation/        # Evaluation framework computing metrics (PPL, Recall@k, MRR, Action Acc, Latency)
├── ingestion/         # Parsing formats (TXT, MD, PDF, DOCX) and writing to memory indexes
├── memory/            # Episodic (graph-based) and semantic (vector-based) memory stacks
├── models/            # Core HarmonyModel routing and overall cognitive loop
├── planner/           # High-level strategic, tactical, and execution planner heads
├── prediction/        # Future state prediction using GRU-based world model simulation
├── reasoning/         # Causal, deductive, inductive, and abductive reasoning engine
├── retrieval/         # VectorMemory wrapping FAISS CRUD and dense passage fusion
├── tools/             # Native tool registries (e.g., python executor, web search)
├── training/          # Multi-stage orchestrator managing the 5 training stages
├── utils/             # GPU DeviceManager, half-precision (AMP) wrappers, and utilities
└── verifier/          # Multi-head verifier heads evaluating factual and code correctness
```

---

## Setup and Installation

Install the package in developer mode along with its dependencies:

```bash
pip install -e .[dev]
```

### Main Dependencies:
- PyTorch (with GPU / AMP Support)
- transformers & tokenizers
- sentence-transformers (dense embedding generation)
- faiss-cpu / faiss-gpu (approximate nearest neighbors vector search)
- networkx (knowledge graph structures)
- datasets (Hugging Face corpus loading)
- networkx & pydantic (graph storage & validation)

---

## Training Curriculum (Multi-Stage Orchestrator)

HARMONY-Core utilizes a 5-stage curriculum training orchestrator to achieve stable training profiles:

1. **Stage 1: Backbone Pretraining** — Next-Token Prediction language modeling training.
2. **Stage 2: Retrieval Memory Setup** — Indexing the knowledge base vector store.
3. **Stage 3: Planner Training** — Training strategic and execution planner action routes.
4. **Stage 4: Verifier Training** — Calibrating confidence metrics for logical checks.
5. **Stage 5: Joint Fine-Tuning** — Multi-task balancing across LM, planning, and verification.

---

## Running Verification and Tests

We write unit tests covering all system modules in the `tests/` directory.

To run all unit tests:

```bash
python -m pytest
```

To run tests with code coverage metrics:

```bash
python -m pytest --cov=harmony
```
