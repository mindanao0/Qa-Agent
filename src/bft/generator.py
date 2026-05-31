"""
Async generator node — N copies fan out via ``Send`` and run independently.

Each generator hits a local Ollama server with a single chat-completion call
and converts the produced source code into an AST fingerprint. Any failure
(LLM error, malformed response, normalization crash) is *contained* — the
node always emits a structured :class:`BFTGeneratorOutput`, marking
``status="FAILED"`` when the work could not be completed. This guarantees
that LangGraph never tries to cancel sibling generators because of a missing
node update.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_ollama import ChatOllama
from loguru import logger

from src.config import (
    GENERATOR_BACKOFF_BASE_SEC,
    GENERATOR_MAX_ATTEMPTS,
    OLLAMA_BASE_URL,
)

from .ast_normalizer import ASTNormalizer
from .models import (
    BFTGeneratorInput,
    BFTGeneratorOutput,
    BFTSecurityViolation,
)


_SYSTEM_PROMPT: str = (
    "You are a deterministic Python code generator. "
    "Return ONLY raw Python code. "
    "No markdown. No explanations. No triple backticks. "
    "Do not include language tags or commentary. "
    "Output must be directly executable by python -c."
)

# Matches a leading ```python / ``` fence and a trailing ``` if the model
# disregards the instruction. We strip rather than fail — diversification is
# the whole point.
_FENCE_OPEN = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\n```\s*$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    """Best-effort cleanup of accidental markdown fences."""
    text = _FENCE_OPEN.sub("", text, count=1)
    text = _FENCE_CLOSE.sub("", text, count=1)
    return text.strip()


def _build_llm(model: str, temperature: float, top_p: float) -> ChatOllama:
    """Construct a ChatOllama client pinned to the local Ollama endpoint."""
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        top_p=top_p,
    )


async def _invoke_llm_once(
    llm: ChatOllama, task_prompt: str
) -> str:
    """Single LLM round-trip. Returns the raw assistant content as a string."""
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", task_prompt),
    ]
    result = await llm.ainvoke(messages)
    content: Any = getattr(result, "content", result)
    if isinstance(content, list):
        # langchain may return a list of content blocks for tool-aware models.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    if not isinstance(content, str):
        content = str(content)
    return content


async def run_generator_node(state: BFTGeneratorInput) -> dict[str, list[BFTGeneratorOutput]]:
    """LangGraph node body for one BFT generator.

    Accepts a :class:`BFTGeneratorInput` (dispatched via :class:`Send`) and
    returns a partial state update of the form
    ``{"generator_outputs": [BFTGeneratorOutput]}``. The reducer on
    :class:`BFTGraphState.generator_outputs` will append the single-element
    list onto the running collection.
    """

    # Pydantic models passed through Send may arrive as dicts in some
    # LangGraph runtimes — normalize defensively.
    if isinstance(state, dict):
        state = BFTGeneratorInput.model_validate(state)

    node_id = state.node_id
    logger.info(
        "BFT generator node_id={nid} model={m} temp={t} top_p={p}",
        nid=node_id,
        m=state.model,
        t=state.temperature,
        p=state.top_p,
    )

    llm = _build_llm(state.model, state.temperature, state.top_p)

    last_error: str | None = None

    for attempt in range(1, GENERATOR_MAX_ATTEMPTS + 1):
        try:
            raw = await _invoke_llm_once(llm, state.task_prompt)
            code = _strip_code_fences(raw)
            if not code:
                raise RuntimeError("empty LLM response after fence stripping")

            try:
                fingerprint = ASTNormalizer().normalize(code)
            except BFTSecurityViolation:
                # KS-2 is a global abort signal — re-raise so the orchestrator
                # can surface it. Do not mask as a per-node failure.
                raise

            output = BFTGeneratorOutput(
                node_id=node_id,
                raw_code=code,
                ast_fingerprint=fingerprint,
                status="OK",
                error_msg=None,
            )
            logger.info(
                "BFT generator node_id={nid} OK (attempt {a}, fp_len={fl})",
                nid=node_id,
                a=attempt,
                fl=len(fingerprint),
            )
            return {"generator_outputs": [output]}

        except BFTSecurityViolation:
            raise

        except Exception as exc:  # noqa: BLE001 — we record any error
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            logger.warning(
                "BFT generator node_id={nid} attempt {a} failed: {e}",
                nid=node_id,
                a=attempt,
                e=last_error,
            )
            if attempt < GENERATOR_MAX_ATTEMPTS:
                backoff = GENERATOR_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

    # ── All retries exhausted ───────────────────────────────────────────────
    logger.error(
        "BFT generator node_id={nid} FAILED after {n} attempts: {e}",
        nid=node_id,
        n=GENERATOR_MAX_ATTEMPTS,
        e=last_error,
    )
    failed = BFTGeneratorOutput(
        node_id=node_id,
        raw_code="",
        ast_fingerprint=None,
        status="FAILED",
        error_msg=last_error or "unknown error",
    )
    return {"generator_outputs": [failed]}


__all__: list[str] = ["run_generator_node"]
