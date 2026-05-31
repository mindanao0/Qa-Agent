# src/llm/structured.py
"""
Backward-compatible re-export of LLM output schemas + 3-stage JSON repair pipeline.

Schema classes are now defined in src/llm/schemas.py (canonical home).
This module re-exports them for backward compatibility and provides the
legacy enforce_json_output() repair pipeline used as a fallback when
structured_output_engine == "legacy_repair".
"""
import json
import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

# ── Re-export canonical schemas (TD-2 fix) ──────────────────────────────────
from src.llm.schemas import (  # noqa: F401  (re-exported for callers)
    HealedLocator,
    PlaywrightScript,
    SyntheticQAExample,
    TestPlan,
    TestStep,
)

T = TypeVar("T", bound=BaseModel)

# ──────────────────────────────────────────────────────────────────────────────
# JSON repair and schema enforcement (legacy fallback pipeline)
# ──────────────────────────────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Remove ```json and ``` markdown fences, then strip whitespace."""
    text = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _repair_json_string(text: str) -> str:
    """Remove trailing commas before } or ] — the most common LLM JSON error."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_playwright_script_fallback(raw: str) -> dict:
    """
    Stage 3 fallback for PlaywrightScript: extract reasoning + code with regex
    instead of relying on valid JSON encoding of Python source.
    """
    reasoning = ""
    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        reasoning = m.group(1).replace("\\n", "\n").replace('\\"', '"')

    code = ""
    m = re.search(r'"code"\s*:\s*"([\s\S]*?)(?<!\\)"\s*[,}]', raw)
    if m:
        code = (
            m.group(1)
            .replace("\\n", "\n")
            .replace('\\"', '"')
            .replace("\\'", "'")
        )

    if not code:
        m = re.search(r"(import\s+\w+[\s\S]+)", raw)
        if m:
            code = m.group(1)
            code = re.sub(r'"\s*\}?\s*$', "", code)

    if not code:
        raise ValueError("_extract_playwright_script_fallback: could not locate code block")

    locators: list[str] = []
    m = re.search(r'"locators_used"\s*:\s*\[([^\]]*)\]', raw)
    if m:
        locators = [s.strip().strip('"') for s in m.group(1).split(",") if s.strip().strip('"')]

    return {"reasoning": reasoning, "code": code, "locators_used": locators}


def _repair_json(raw: str) -> str:
    """Apply heuristic repairs to common LLM JSON malformation patterns."""
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = re.sub(r"/\*[\s\S]*?\*/", "", raw)
    raw = re.sub(r"(?<![\\])'", '"', raw)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    raw = re.sub(r"\bTrue\b", "true", raw)
    raw = re.sub(r"\bFalse\b", "false", raw)
    raw = re.sub(r"\bNone\b", "null", raw)
    return raw


def _extract_json_from_text(raw: str) -> str:
    """
    Try to isolate a JSON object/array from raw LLM output.
    Attempts, in order:
      1. Strip leading/trailing whitespace and parse directly.
      2. Extract content from a ```json ... ``` code fence.
      3. Extract content from a ``` ... ``` code fence.
      4. Find the first { ... } or [ ... ] balanced block.
    """
    stripped = raw.strip()

    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        match = re.search(pattern, stripped)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate

    start = stripped.find("{")
    if start == -1:
        start = stripped.find("[")
    if start != -1:
        open_char = stripped[start]
        close_char = "}" if open_char == "{" else "]"
        depth = 0
        for i, ch in enumerate(stripped[start:], start=start):
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return stripped[start : i + 1]

    return stripped


def enforce_json_output(schema: type[T], raw_text: str) -> T:
    """
    Parse and validate `raw_text` against `schema`.

    Repair pipeline (legacy fallback — used when structured_output_engine == "legacy_repair"):
      Stage 1: strip fences → json.loads → model_validate
      Stage 2: _repair_json_string → json.loads → model_validate
      Stage 3 (PlaywrightScript only): regex extraction fallback
    """
    candidate = _strip_fences(raw_text)
    if not (candidate.startswith("{") or candidate.startswith("[")):
        candidate = _extract_json_from_text(raw_text)

    try:
        data = json.loads(candidate)
        return schema.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.debug(f"enforce_json_output stage 1 failed ({exc}); trying repair")

    repaired = _repair_json_string(candidate)
    try:
        data = json.loads(repaired)
        return schema.model_validate(data)
    except (json.JSONDecodeError, Exception) as exc:
        logger.debug(f"enforce_json_output stage 2 failed ({exc}); trying stage 3")

    if schema.__name__ == "PlaywrightScript":
        try:
            data = _extract_playwright_script_fallback(raw_text)
            return schema.model_validate(data)
        except Exception as exc:
            logger.debug(f"enforce_json_output stage 3 (PlaywrightScript fallback) failed: {exc}")

    logger.error(
        f"enforce_json_output: all stages exhausted for schema={schema.__name__} | "
        f"raw_text_preview={raw_text[:200]!r}"
    )
    raise ValueError(
        f"Could not parse LLM output as {schema.__name__}: "
        f"raw output (first 200 chars): {raw_text[:200]}"
    )
