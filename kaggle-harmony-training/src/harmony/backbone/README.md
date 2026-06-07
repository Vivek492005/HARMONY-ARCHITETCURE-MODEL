# HARMONY-Core Backbone Module

The `backbone` module provides the core sequential processing, local token mixing, and routing components of the HARMONY architecture.

## Component Overview

### 1. `HierarchicalCognitiveState`
Maintains multi-timescale memory slots (short, medium, long, and global session states).
- **Short-term state**: Processed via a `nn.GRUCell` sequence transition.
- **Medium-term state**: Compressed representation generated using tanh activation.
- **Global session state**: Updated slowly using Exponential Moving Average (EMA) momentum.

### 2. `LocalMixer`
Fuses adjacent token representations into a single compressed chunk vector using:
- Dilated 1D convolutions (`nn.Conv1d`) over the sequence dimension.
- Sigmoid-gated residual projection.
- Attention pooling (`nn.Linear` to logits) to compress chunk lengths.

### 3. `SelectiveStateBackbone`
The primary sequence model acting as a recurrent state backbone.
- Uses a multi-layer Gated Recurrent Unit (`nn.GRU`) for linear-time complexity $O(N)$ memory processing.
- Supports **chunkwise recurrence** (`process_chunkwise`) to loop over chunk representations sequentially, avoiding peak VRAM spikes on context sizes up to 100k+ tokens.

### 4. `SparseMoE`
A sparse Mixture of Experts (MoE) block:
- routes inputs to top-$k$ experts per token.
- computes gated weights to combine output activations efficiently.
