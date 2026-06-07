# HARMONY-Core Verifier Module

The `verifier` module acts as an internal critic in the HARMONY architecture, assessing generated outputs for logical, math, factual, and structural consistency.

## Component Overview

### 1. `VerifierHead`
A lightweight single-head verifier computing overall confidence metrics.
- Uses Sigmoid activation to output a score in the $[0, 1]$ range.

### 2. `VerifierV2`
An advanced multi-head checker:
- **Factual Confidence**: Evaluates factual grounding against reference documents.
- **Logical Confidence**: Evaluates path consistency and reasoning logical steps.
- **Code Confidence**: Assesses syntax correctness.
- **Overall Confidence**: Aggregated mean output of checker indices.

### 3. `VerifierDataset` & `VerifierTrainer`
BCE loss training pipeline that trains verifiers using correct vs corrupted answer formulations.
- Supports dataset loading of GSM8K with corrupted answers generation.
