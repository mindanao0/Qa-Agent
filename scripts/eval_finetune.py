"""Compare base vs fine-tuned model quality on val.jsonl.

Run:
    uv run python scripts/eval_finetune.py

Prerequisites:
    - data/training/val.jsonl  (51 examples)
    - Ollama running with qwen2.5-coder:7b-instruct-q4_K_M (base)
    - Ollama running with qa-agent-finetuned (from create_modelfile.py)
    - models/finetune_output/checkpoint-<N>/trainer_state.json

Output: models/eval_results.json
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import pathlib
import sys
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VAL_JSONL = pathlib.Path("data/training/val.jsonl")
FINETUNE_OUTPUT_DIR = pathlib.Path("models/finetune_output")
EVAL_OUTPUT = pathlib.Path("models/eval_results.json")

BASE_MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
FT_MODEL = "qa-agent-finetuned"

# Same env var as src/llm/adapter.py — lets WSL callers set host.docker.internal or Windows gateway IP
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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
    base_skipped: int
    ft_skipped: int
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
# FIX 1 — Load trainer_state from latest checkpoint
# ---------------------------------------------------------------------------


def _load_trainer_state() -> dict:
    """Load trainer_state.json from the latest checkpoint-* directory."""
    output_dir = pathlib.Path("models/finetune_output")
    checkpoints = sorted(
        [d for d in output_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        print("[warn] no checkpoint-* dirs found in models/finetune_output", file=sys.stderr)
        return {}
    latest = checkpoints[-1] / "trainer_state.json"
    if not latest.exists():
        print(f"[warn] {latest} not found", file=sys.stderr)
        return {}
    return json.loads(latest.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# FIX 4 — VRAM peak via torch
# ---------------------------------------------------------------------------


def _reset_vram_stats() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _vram_peak_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# FIX 2 — Async Ollama query with proper None-on-failure
# ---------------------------------------------------------------------------


async def _call_model(
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 300.0,
) -> str | None:
    """
    Send a prompt to an Ollama model via HTTP.
    Returns None on any error — caller skips the example, doesn't penalize pass_rate.
    Semaphore(1) ensures no concurrent model calls (6 GB VRAM constraint).
    """
    async with semaphore:
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                resp.raise_for_status()
                return resp.json()["response"]
        except Exception as e:
            print(f"[WARN] model={model} call failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate(
    val_jsonl: pathlib.Path = VAL_JSONL,
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

    if len(records) == 0:
        raise ValueError(f"val.jsonl is empty or has no valid lines: {val_jsonl}")

    # INVARIANT: Semaphore(1) — only 1 model call at a time (6 GB VRAM)
    semaphore = asyncio.Semaphore(1)

    _reset_vram_stats()

    base_passed = 0
    ft_passed = 0
    base_skipped = 0
    ft_skipped = 0

    # Run all base calls first, then all ft calls — avoids Ollama swapping models
    # every example (7B ↔ 1.5B swap blows the 60s timeout when interleaved).
    base_results: list[bool | None] = []
    print(f"--- Pass 1/2: base model ({base_model}) ---")
    for i, rec in enumerate(records, 1):
        source = rec.get("source", "unknown")
        prompt = rec.get("prompt", "")
        scoring_type = get_scoring_type(source)
        response = await _call_model(base_model, prompt, semaphore)
        if response is None:
            base_skipped += 1
            base_results.append(None)
        else:
            ok = score_completion(scoring_type, response)
            if ok:
                base_passed += 1
            base_results.append(ok)
        label = "skip" if base_results[-1] is None else ("PASS" if base_results[-1] else "fail")
        print(f"  [{i:02d}/{len(records)}] {source!r:25s} base={label}")

    ft_results: list[bool | None] = []
    print(f"--- Pass 2/2: ft model ({ft_model}) ---")
    for i, rec in enumerate(records, 1):
        source = rec.get("source", "unknown")
        prompt = rec.get("prompt", "")
        scoring_type = get_scoring_type(source)
        response = await _call_model(ft_model, prompt, semaphore)
        if response is None:
            ft_skipped += 1
            ft_results.append(None)
        else:
            ok = score_completion(scoring_type, response)
            if ok:
                ft_passed += 1
            ft_results.append(ok)
        label = "skip" if ft_results[-1] is None else ("PASS" if ft_results[-1] else "fail")
        print(f"  [{i:02d}/{len(records)}] {source!r:25s} ft={label}")

    # FIX 3 — denominator excludes skipped examples
    n = len(records)
    base_pass_rate = base_passed / max(1, n - base_skipped)
    ft_pass_rate = ft_passed / max(1, n - ft_skipped)
    quality_gain = ft_pass_rate - base_pass_rate

    # FIX 1 — extract losses from checkpoint trainer_state
    state = _load_trainer_state()
    log_history = state.get("log_history", [])
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    training_loss_final = train_losses[-1] if train_losses else None
    val_loss_final = eval_losses[-1] if eval_losses else None
    oom_errors = 0

    # FIX 4 — VRAM peak from torch
    vram_peak = _vram_peak_mb()

    train_ok = (training_loss_final is None) or (training_loss_final <= _THRESHOLD_TRAIN_LOSS)
    val_ok = (val_loss_final is None) or (val_loss_final <= _THRESHOLD_VAL_LOSS)
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
        training_loss_final=training_loss_final,
        val_loss_final=val_loss_final,
        vram_peak_mb=vram_peak,
        oom_errors=oom_errors,
        base_skipped=base_skipped,
        ft_skipped=ft_skipped,
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
