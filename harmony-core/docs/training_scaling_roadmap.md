# HARMONY Phase I & L: Training and Scaling Roadmap

## Phase I: Frontier Training Pipeline
1. **Language Pretraining**: Train the Hierarchical State Backbone on C4/Pile for basic chunk-level sequence prediction.
2. **Long-context Pretraining**: Increase sequence length aggressively. The linear scaling of the `LocalMixer` supports 1M+ tokens.
3. **Retrieval Training**: Use contrastive loss to align `VectorMemory` and the backbone state.
4. **Reasoning Training**: Supervise `ReasoningEngine` outputs on synthetic math/logic reasoning paths.
5. **Planning Training**: Train `HierarchicalTaskPlanner` on agent trajectories.
6. **Verification Training**: Calibrate `VerifierV2` on true/false claims and contradictory inputs.
7. **Tool-use Training**: Fine-tune specific execution actions to match `ToolRegistry` API signatures.
8. **Joint Alignment**: RLHF or DPO over the entire cognitive graph pipeline.

## Phase L: Model Scaling
- **1B to 7B**: Single node multi-GPU using FSDP (Fully Sharded Data Parallel).
- **30B to 70B**: Multi-node using DeepSpeed ZeRO-3 and Tensor Parallelism.
- **200B+**: Pipeline Parallelism + ZeRO-3. SparseMoE (from Phase C/H) will keep active parameters per token around 10% of total parameters, allowing efficient scaling without massive compute overhead.
