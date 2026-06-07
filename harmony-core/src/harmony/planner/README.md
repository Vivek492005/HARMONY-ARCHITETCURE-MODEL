# HARMONY-Core Planner Module

The `planner` module enables structured action routing and hierarchical decision-making within the HARMONY architecture.

## Component Overview

### 1. `HierarchicalTaskPlanner`
Decomposes complex problems into multiple stages:
- **Strategic Planner**: High-level goal planning (`nn.Linear` mapping).
- **Tactical Planner**: Concrete strategy selection (`nn.Linear` mapping).
- **Execution Planner**: Map strategies to discrete action commands (logits over 10 output actions).

### 2. `PlannerHead`
A lightweight classification layer that evaluates state representations and outputs next-action probabilities.
- Actions: `0=generate`, `1=retrieve`, `2=reason`, `3=verify`, `4=stop`.

### 3. `PlannerDataset` & `PlannerTrainer`
Supervised dataset loader and trainer supporting cross-entropy classification updates.
- Supports training bootstrapping on fallback dummy reasoning datasets.
- Includes integration wrappers to load the GSM8K dataset from Hugging Face for task planning training.
