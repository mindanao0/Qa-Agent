"""
BFT orchestrator configuration constants.

Constraints (Byzantine Fault Tolerance):
    N >= 3*f + 1   (total nodes)
    Q  = 2*f + 1   (quorum threshold)

With f=1 we get N=4 and Q=3 — three matching AST fingerprints out of four
generators are required to certify consensus.

Diversification strategy (breaks correlated LLM hallucinations):
  - Nodes 0,2  → codellama:13b at temperature=0.0 (greedy decoding)
  - Nodes 1,3  → deepseek-coder:6.7b with top_p=0.85 (nucleus sampling)
"""

from __future__ import annotations

from typing import Any

# ── BFT Parameters ───────────────────────────────────────────────────────────
N: int = 4         # Total generator nodes (must satisfy N >= 3*F + 1)
F: int = 1         # Maximum Byzantine faults tolerated
Q: int = 2 * F + 1  # Quorum threshold (= 3 when F=1)

assert N >= 3 * F + 1, "BFT invariant violated: need N >= 3*f + 1"
assert Q == 2 * F + 1, "BFT invariant violated: Q must equal 2*f + 1"

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"

# ── Per-node diversification configs ─────────────────────────────────────────
# Each dict is a kwargs payload for ChatOllama plus the node_id.
NODE_CONFIGS: list[dict[str, Any]] = [
    {"node_id": 0, "model": "codellama:7b",         "temperature": 0.0, "top_p": 1.0},
    {"node_id": 1, "model": "deepseek-coder:6.7b",  "temperature": 0.7, "top_p": 0.85},
    {"node_id": 2, "model": "codellama:7b",         "temperature": 0.0, "top_p": 1.0},
    {"node_id": 3, "model": "deepseek-coder:6.7b",  "temperature": 0.7, "top_p": 0.85},
]

assert len(NODE_CONFIGS) == N, "NODE_CONFIGS length must equal N"

# ── Generator retry policy ───────────────────────────────────────────────────
GENERATOR_MAX_ATTEMPTS: int = 3
GENERATOR_BACKOFF_BASE_SEC: float = 1.0
