"""Compare base vs fine-tuned model quality on val.jsonl.

Run:
    uv run python scripts/eval_finetune.py

Prerequisites:
    - data/training/val.jsonl  (51 examples)
    - Ollama running with qwen2.5-coder:7b-instruct-q4_K_M (base)
    - Ollama running with qa-agent-finetuned (from create_modelfile.py)
    - models/finetune_output/training_metrics.json (from finetune.py)

Output: models/eval_results.json
"""
from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VAL_JSONL = Path("data/training/val.jsonl")
METRICS_PATH = Path("models/finetune_output/training_metrics.json")
EVAL_OUTPUT = Path("models/eval_results.json")

BASE_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
FT_MODEL = "qa-agent-finetuned"

# ---------------------------------------------------------------------------
# Acceptance thresholds (from spec)
# ---------------------------------------------------------------------------
_THRESHOLD_TRAIN_LOSS = 1.50
_THRESHOLD_VAL_LOSS = 2.00
_THRESHOLD_QUALITY_GAIN = 0.05
_THRESHOLD_VRAM_MB = 5800.0

# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------

ScoringType = Literal["pytest", "schema", "vitest", "hypothesis"]


class EvalResult(BaseModel):
    """Output of the evaluation pipeline (matches spec eval_results.json schema)."""

    model_config = ConfigDict(extra="forbid")

    base_pass_rate: float
    ft_pass_rate: float
    quality_gain: float
    training_loss_final: float | None
    val_loss_final: float | None
    vram_peak_mb: float | None
    oom_errors: int
    finetune_status: Literal["PASS", "FAIL"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def get_scoring_type(source: str) -> ScoringType:
    """Map a dataset source string to a scoring strategy."""
    if source == "sprint6_pytest":
        return "pytest"
    if source == "sprint13_schema":
        return "schema"
    if source == "sprint9_vitest":
        return "vitest"
    return "hypothesis"


def score_completion(scoring_type: ScoringType, completion: str) -> bool:
    """
    Score a model completion. Returns True if it passes the scoring heuristic.

    pytest:     valid Python + contains "def test_"
    schema:     valid JSON + has "type" key
    vitest:     contains both "describe" and "expect"
    hypothesis: valid Python syntax (ast.parse succeeds)
    """
    if not completion or not completion.strip():
        return False

    if scoring_type == "pytest":
        if "def test_" not in completion:
            return False
        try:
            ast.parse(completion)
            return True
        except SyntaxError:
            return False

    if scoring_type == "schema":
        try:
            obj = json.loads(completion)
            return isinstance(obj, dict) and "type" in obj
        except (json.JSONDecodeError, ValueError):
            return False

    if scoring_type == "vitest":
        return "describe" in completion and "expect" in completion

    # hypothesis (and any unknown type)
    try:
        ast.parse(completion)
        return True
    except SyntaxError:
        return False
