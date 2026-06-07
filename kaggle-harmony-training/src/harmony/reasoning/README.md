# HARMONY-Core Reasoning Module

The `reasoning` module contains structures for managing logical inferences, thought step transitions, and native tool execution.

## Component Overview

### 1. `ReasoningEngine`
Processes internal state vectors and routes reasoning thoughts:
- **Modes**: Support for `Deduction`, `Induction`, `Abduction`, `Causal`, and `Analogical` styles.
- **Thought Transformer**: An MLP mapping states to updated thoughts with residual connection.

### 2. `ToolRegistry`
Native tool integration interface:
- Standard registration: `register_tool(name, func)`.
- Invocation: `execute_tool(name, **kwargs)`.
- Default tools:
  - `python`: Executes code blocks (mocked).
  - `search`: Queries external sources (mocked).
