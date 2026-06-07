# HARMONY-Core Inference Guide

This guide details how to perform inference with the HARMONY-Core model, covering standard next-token predictions and confidence-based retry planning.

## 1. Standard Inference

To perform basic end-to-end text processing (tokenization, chunking, convolutional mixing, backbone propagation, and prediction):

```python
from harmony.models.harmony_model import HarmonyModel

# Load model (eval mode is active by default in process_text)
model = HarmonyModel()
model.eval()

text = "HARMONY processes sequences in semantic blocks rather than full self-attention grids."
results = model.process_text(text)

# Output contains logits, final backbone states, and processed token IDs
logits = results["logits"]         # (batch=1, num_chunks, vocab_size)
states = results["final_state"]     # (num_layers, batch=1, hidden_size)
chunked_ids = results["chunked_ids"] # (batch=1, num_chunks, chunk_size)

print("Inference completed successfully!")
```

---

## 2. Confidence-Based Retry Inference

When processing complex queries (e.g., mathematics or factual QA), you can trigger the **Adaptive Retry Loop**. If the VerifierHead detects logical inconsistencies or low confidence, it executes recovery actions (like document retrieval or reasoning steps) up to `max_retries` times:

```python
import torch
from harmony.models.harmony_model import HarmonyModel

model = HarmonyModel()
model.eval()

# 1. Setup sample document knowledge base
query = "What is the key algorithm in HARMONY?"
context_doc = "The key algorithm in HARMONY is local convolutional context mixing combined with recurrent selective state backbones."

# Ingest knowledge into memory
q_emb = model.embedding_manager.get_query_embedding(context_doc)
model.memory_manager.semantic.add_document("doc_01", context_doc, q_emb, {"source": "manual"})

# 2. Run query with retry loop
chunked_ids, mask = model.chunker.tokenize_and_chunk(query)

retry_results = model.generate_with_retry(
    chunked_ids, 
    max_retries=3, 
    threshold=0.6,
    original_query_text=query
)

print(f"Action chosen: {retry_results['action']}") # 0=generate, 1=retrieve, etc.
print(f"Confidence score: {retry_results['confidence']:.4f}")
print(f"Number of retries: {retry_results['retries']}")
```
