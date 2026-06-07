# HARMONY-Core Architecture Guide

HARMONY-Core implements an adaptive sequence model designed for long-context, retrieval-augmented reasoning. Rather than using standard quadratic attention over all historical tokens, HARMONY splits sequences into semantic chunks and maintains a hierarchical state representation.

## Core Pipeline

```
Text Input
   │
   ▼
[COMPRESS] (Tokenizer -> Chunker -> LocalMixer)
   │
   ▼
[STATE TRACK] (SelectiveStateBackbone GRU)
   │
   ▼
[RETRIEVE] (VectorMemory / FAISS Ingestion)
   │
   ▼
[PLAN] (PlannerHead Action Selection)
   │
   ▼
[VERIFY] (VerifierHead Confidence Calibration)
   │
   ▼
[GENERATE] (GeneratorHead Vocabulary Projection)
```

---

## 1. COMPRESS (Compression Layer)
*   **Semantic Chunker**: Tokenizes raw text using a subword tokenizer (e.g. GPT-2) and groups subwords into fixed-size chunks (default size: 16 tokens).
*   **LocalMixer**: Combines embeddings inside a chunk into a single vector using a 1D convolution over sequence length and an attention-pooling layer. This condenses the input dimension from `(batch, num_chunks, chunk_size, hidden_size)` to `(batch, num_chunks, hidden_size)`.

## 2. STATE TRACK (Recurrent Backbone)
*   **SelectiveStateBackbone**: Processes chunk representations sequentially using a multi-layer Gated Recurrent Unit (GRU). This allows context tracking in $O(N)$ sequence length complexity, enabling processing of 100k+ tokens without VRAM spikes.
*   **Hierarchical State**: Supports caching and chunkwise recurrent propagation.

## 3. RETRIEVE (External Memory)
*   **VectorMemory**: Uses a FAISS IndexFlatL2 to search document embedding spaces.
*   **Reranker**: Performs cosine similarity matching to re-order the retrieved candidates.
*   **Retriever**: Coordinates semantic search over indexed documents.

## 4. PLAN & VERIFY (Adaptive Control Heads)
*   **PlannerHead**: Classifies the current cognitive state into one of five actions: `0=generate`, `1=retrieve`, `2=reason` (pause and think), `3=verify`, `4=stop`.
*   **VerifierHead**: Yields a confidence score in range `[0, 1]`. If the verifier output falls below a specified threshold, the model executes a retry loop, invoking the planner to perform retrieval or pause/thinking steps.

## 5. GENERATE (Language Generation)
*   **GeneratorHead**: Projects recurrent backbone outputs back into vocabulary dimension space. Ties weights with the input embedding table to reduce model footprints.
