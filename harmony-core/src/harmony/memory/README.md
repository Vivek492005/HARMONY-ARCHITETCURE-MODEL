# HARMONY-Core Memory Module

The `memory` module manages the cognitive memory systems of the architecture, spanning working memory, episodic graph memory, and semantic database consolidations.

## Component Overview

### 1. `CognitiveMemoryStack`
Routes representations across Working, Episodic, Semantic, and Procedural memory interfaces.
- Integrates a neural `importance_scorer` to assess state significance for consolidation.

### 2. `MemoryConsolidator`
Maintains long-term consolidation rules.
- Scores the relevance of individual `MemoryEntry` elements based on timestamp recency decay and similarity to the current state.
- Promotes highly relevant states to long-term memory indexes.

### 3. `GraphMemory`
Episodic memory store built using the NetworkX graph library.
- Triplet-based facts insertion (`add_fact(subject, relation, object)`).
- Subgraph retrieval via multi-hop neighbors extraction.

### 4. `GraphReasoner`
Executes shortest path traversals over `GraphMemory` to find reasoning linkages between entity nodes.

### 5. `MemoryManager`
Orchestrates reads, writes, consolidation, and pruning operations across the entire memory stack.

### 6. `SessionMemory`
Maintains a sliding capacity context window of active events.
