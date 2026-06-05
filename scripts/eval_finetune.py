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


# ---------------------------------------------------------------------------
# Async Ollama query
# ---------------------------------------------------------------------------


async def query_model(
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 512,
) -> str:
    """
    Send a prompt to an Ollama model via OllamaAdapter.
    Semaphore(1) ensures no concurrent model calls (6 GB VRAM constraint).
    """
    from src.llm.adapter import OllamaAdapter  # noqa: PLC0415

    async with semaphore:
        adapter = OllamaAdapter(model=model, temperature=0.1, max_tokens=max_tokens)
        try:
            return await adapter.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            print(f"[warn] query_model({model!r}) failed: {exc}", file=sys.stderr)
            return ""
        finally:
            await adapter.close()


# ---------------------------------------------------------------------------
# Load training metrics
# ---------------------------------------------------------------------------


def _load_training_metrics() -> dict:
    """Load training_metrics.json produced by finetune.py, or return defaults."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print(
        f"[warn] {METRICS_PATH} not found — training metrics will be None",
        file=sys.stderr,
    )
    return {
        "training_loss_final": None,
        "val_loss_final": None,
        "vram_peak_mb": None,
        "oom_errors": 0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate(
    val_jsonl: Path = VAL_JSONL,
    base_model: str = BASE_MODEL,
    ft_model: str = FT_MODEL,
) -> EvalResult:
    """
    Evaluate base vs fine-tuned model on val.jsonl.

    For each of the 51 examples, sends the prompt to both models (serialized via
    asyncio.Semaphore(1)) and scores the completions. Returns an EvalResult.
    """
    if not val_jsonl.exists():
        raise FileNotFoundError(f"Validation set not found: {val_jsonl}")

    records: list[dict] = []
    with val_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Evaluating {len(records)} examples | base={base_model!r} ft={ft_model!r}")

    # INVARIANT: Semaphore(1) — only 1 model call at a time (6 GB VRAM)
    semaphore = asyncio.Semaphore(1)

    base_passed = 0
    ft_passed = 0

    for i, rec in enumerate(records, 1):
        source = rec.get("source", "unknown")
        prompt = rec.get("prompt", "")
        scoring_type = get_scoring_type(source)

        # Base model
        base_completion = await query_model(base_model, prompt, semaphore)
        base_ok = score_completion(scoring_type, base_completion)
        if base_ok:
            base_passed += 1

        # Fine-tuned model
        ft_completion = await query_model(ft_model, prompt, semaphore)
        ft_ok = score_completion(scoring_type, ft_completion)
        if ft_ok:
            ft_passed += 1

        print(
            f"[{i:02d}/{len(records)}] source={source!r:25s} "
            f"scoring={scoring_type!r:12s} base={'PASS' if base_ok else 'fail'} "
            f"ft={'PASS' if ft_ok else 'fail'}"
        )

    n = len(records)
    base_pass_rate = base_passed / n
    ft_pass_rate = ft_passed / n
    quality_gain = ft_pass_rate - base_pass_rate

    metrics = _load_training_metrics()

    train_loss = metrics.get("training_loss_final")
    val_loss = metrics.get("val_loss_final")
    vram_peak = metrics.get("vram_peak_mb")
    oom_errors = metrics.get("oom_errors", 0)

    train_ok = (train_loss is None) or (train_loss <= _THRESHOLD_TRAIN_LOSS)
    val_ok = (val_loss is None) or (val_loss <= _THRESHOLD_VAL_LOSS)
    vram_ok = (vram_peak is None) or (vram_peak <= _THRESHOLD_VRAM_MB)
    gain_ok = quality_gain >= _THRESHOLD_QUALITY_GAIN
    oom_ok = oom_errors == 0

    status: Literal["PASS", "FAIL"] = "PASS" if all(
        [train_ok, val_ok, vram_ok, gain_ok, oom_ok]
    ) else "FAIL"

    result = EvalResult(
        base_pass_rate=round(base_pass_rate, 4),
        ft_pass_rate=round(ft_pass_rate, 4),
        quality_gain=round(quality_gain, 4),
        training_loss_final=train_loss,
        val_loss_final=val_loss,
        vram_peak_mb=vram_peak,
        oom_errors=oom_errors,
        finetune_status=status,
    )

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nEval results written: {EVAL_OUTPUT}")
    print(result.model_dump_json(indent=2))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        result = asyncio.run(evaluate())
        return 0 if result.finetune_status == "PASS" else 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
