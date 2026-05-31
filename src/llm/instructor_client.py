# src/llm/instructor_client.py
"""
Instructor-based structured LLM output client.

This is the preferred entry point for structured generation in Sprint 1+.
It wraps Ollama's OpenAI-compatible /v1 endpoint with the `instructor`
library, which handles Pydantic validation + retry-with-error-feedback
natively.

Usage:
    from src.llm.instructor_client import InstructorClient
    from src.llm.schemas import TestPlan

    client = InstructorClient()
    plan = await client.create_structured(messages, TestPlan)
"""
from __future__ import annotations

import hashlib
import time
from typing import TypeVar

import instructor
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

# Reuse the module-level semaphore from adapter.py so InstructorClient and
# OllamaAdapter never issue more than SEMAPHORE_LIMIT concurrent VRAM calls.
from src.llm.adapter import (
    DEFAULT_MODEL,
    OLLAMA_BASE_URL,
    _inference_semaphore,
)

T = TypeVar("T", bound=BaseModel)


# ──────────────────────────────────────────────────────────────────────────────
# Config + error types
# ──────────────────────────────────────────────────────────────────────────────


class GenerationConfig(BaseModel):
    """Immutable generation parameters passed to Ollama."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = 0.0
    top_p: float = 1.0
    num_predict: int = 2048
    num_ctx: int = 8192


class StructuredGenerationError(Exception):
    """Raised when all instructor retries are exhausted for a structured call."""

    def __init__(self, prompt_hash: str, last_error: Exception) -> None:
        self.prompt_hash = prompt_hash
        self.last_error = last_error
        super().__init__(
            f"Structured generation failed [hash={prompt_hash}]: {last_error}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────


class InstructorClient:
    """
    Async structured LLM client backed by instructor + Ollama JSON mode.

    Behaviour contract
    ------------------
    * On ValidationError: instructor retries up to `max_retries`, feeding the
      validation error back to the LLM as a correction prompt each time.
    * On exhausted retries: raises StructuredGenerationError — the caller
      decides whether to fall back to the legacy repair pipeline.
    * The shared _inference_semaphore from adapter.py is held for the entire
      instructor call (including retries) so VRAM is never over-committed.
    * Returns a validated Pydantic model instance — NEVER a dict, NEVER a str.
    * Logs model / token counts / latency / validation_passed at INFO level.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        semaphore=None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._semaphore = semaphore if semaphore is not None else _inference_semaphore

        v1_url = base_url.rstrip("/") + "/v1"
        self._openai_client = AsyncOpenAI(
            base_url=v1_url,
            api_key="ollama",  # required field; Ollama ignores the value
        )
        self._client = instructor.from_openai(
            self._openai_client,
            mode=instructor.Mode.JSON,
        )

    async def create_structured(
        self,
        prompt: str | list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
    ) -> T:
        """
        Generate a structured response and validate it against `response_model`.

        Args:
            prompt: Either a plain string (converted to a single user message)
                    or a full messages list in OpenAI chat format.
            response_model: The Pydantic V2 model class to validate against.
            temperature: Generation temperature. Default 0.0 (deterministic).

        Returns:
            A validated instance of `response_model`.

        Raises:
            StructuredGenerationError: if instructor exhausts all retries.
        """
        messages: list[dict[str, str]] = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )

        # Stable hash of the last user message for log correlation
        last_content = messages[-1].get("content", "") if messages else ""
        prompt_hash = hashlib.sha256(
            (last_content[:200] if isinstance(last_content, str) else str(last_content)[:200]
             ).encode()
        ).hexdigest()[:8]

        start_ms = time.monotonic() * 1000

        async with self._semaphore:
            try:
                result, completion = (
                    await self._client.chat.completions.create_with_completion(
                        model=self.model,
                        messages=messages,
                        response_model=response_model,
                        max_retries=self.max_retries,
                        temperature=temperature,
                    )
                )

                latency_ms = time.monotonic() * 1000 - start_ms
                usage = getattr(completion, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = (
                    getattr(usage, "completion_tokens", 0) if usage else 0
                )

                logger.info(
                    f"InstructorClient | model={self.model} "
                    f"response_model={response_model.__name__} "
                    f"prompt_tokens={prompt_tokens} "
                    f"output_tokens={completion_tokens} "
                    f"latency_ms={latency_ms:.1f} "
                    f"validation_passed=True"
                )
                return result

            except StructuredGenerationError:
                raise  # don't double-wrap
            except Exception as exc:
                latency_ms = time.monotonic() * 1000 - start_ms
                logger.error(
                    f"InstructorClient | model={self.model} "
                    f"response_model={response_model.__name__} "
                    f"latency_ms={latency_ms:.1f} "
                    f"validation_passed=False "
                    f"error={exc!r}"
                )
                raise StructuredGenerationError(prompt_hash, exc) from exc

    async def close(self) -> None:
        await self._openai_client.close()

    async def __aenter__(self) -> "InstructorClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
