# HARMONY-Core Evaluation Module

The `evaluation` module provides verification tools to evaluate model performance across multiple criteria.

## Component Overview

### 1. `HarmonyEvaluator`
An automated suite reporting metrics across five key dimensions:

- **Language Modeling**: Computes cross-entropy loss and Perplexity (PPL) on evaluation datasets.
- **Retrieval Quality**: Evaluates dense passage recall using **Recall@k** and **Mean Reciprocal Rank (MRR)**.
- **Planner Accuracy**: Reports strategic next-action classification accuracy.
- **Verifier Accuracy**: Reports classification metrics including Precision, Recall, Accuracy, and F1 Score for checking correctness.
- **System Performance Benchmarking**: Measures inference latency in milliseconds and GPU/CPU peak memory usage footprint.
