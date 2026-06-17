# src/llm/grammar_client.py
"""Grammar-constrained structured output via Ollama's native `format` parameter.

Phase-1 of the planner-quality track. Instead of generating free JSON and then
validating / repairing it (the instructor path), this client passes the target
Pydantic model's JSON Schema to Ollama's `/api/chat` `format` field. Ollama
compiles that schema into a GBNF grammar inside llama.cpp and constrains token
sampling so the output is schema-valid *by construction* — there is no
post-parse retry. This is the in-stack equivalent of llama.cpp GBNF / outlines
on a project that is pinned to "Ollama localhost:11434 only".

Interface matches InstructorClient.create_structured() so it is a drop-in for
the planner (and the eval harness):

    client = GrammarConstrainedClient()
    obj = await client.create_structured(prompt, MyModel, temperature=0.1)

Returns a validated Pydantic instance, or raises StructuredGenerationError.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

from src.llm.adapter import DEFAULT_MODEL, OLLAMA_BASE_URL, _inference_semaphore
from src.llm.instructor_client import StructuredGenerationError

T = TypeVar("T", bound=BaseModel)

# Constrained decoding guarantees STRUCTURE but not TERMINATION: a model can
# ramble inside a free string field until num_predict cuts it off mid-token,
# producing an unterminated/truncated JSON. Bounding string/array sizes in the
# schema makes the GBNF grammar close the string itself, eliminating that
# failure mode. Limits are generous (a test-case field is normally < ~200 chars).
_MAX_STR_LEN = 240
_MAX_ARRAY_ITEMS = 15


def _bound_schema(node: object) -> None:
    """Recursively inject maxLength on strings + maxItems on arrays (in place)."""
    if isinstance(node, dict):
        if node.get("type") == "string" and "maxLength" not in node:
            node["maxLength"] = _MAX_STR_LEN
        if node.get("type") == "array" and "maxItems" not in node:
            node["maxItems"] = _MAX_ARRAY_ITEMS
        for key in ("properties", "$defs", "items", "definitions"):
            if key in node:
                _bound_schema(node[key])
        # also walk any remaining dict values (e.g. nested property dicts)
        for v in node.values():
            if isinstance(v, (dict, list)):
                _bound_schema(v)
    elif isinstance(node, list):
        for item in node:
            _bound_schema(item)


class GrammarConstrainedClient:
    """Schema-constrained structured client backed by Ollama `format=<schema>`."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        semaphore=None,
        num_ctx: int = 8192,
        num_predict: int = 2048,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._semaphore = semaphore if semaphore is not None else _inference_semaphore
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=5.0),
        )

    async def create_structured(
        self,
        prompt: str | list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
    ) -> T:
        messages: list[dict[str, str]] = (
            prompt if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        last = messages[-1].get("content", "") if messages else ""
        prompt_hash = hashlib.sha256(str(last)[:200].encode()).hexdigest()[:8]

        # The JSON Schema Ollama will compile to a GBNF grammar. extra="forbid"
        # on the models => additionalProperties:false => no stray keys / markdown.
        schema = response_model.model_json_schema()
        _bound_schema(schema)  # bound string/array sizes -> no runaway truncation

        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

        start_ms = time.monotonic() * 1000
        async with self._semaphore:
            try:
                resp = await self._client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                obj = response_model.model_validate(json.loads(content))

                latency_ms = time.monotonic() * 1000 - start_ms
                logger.info(
                    f"GrammarConstrainedClient | model={self.model} "
                    f"response_model={response_model.__name__} "
                    f"prompt_eval={data.get('prompt_eval_count', 0)} "
                    f"eval={data.get('eval_count', 0)} "
                    f"latency_ms={latency_ms:.1f} validation_passed=True"
                )
                return obj
            except (ValidationError, json.JSONDecodeError, httpx.HTTPError, KeyError) as exc:
                latency_ms = time.monotonic() * 1000 - start_ms
                logger.error(
                    f"GrammarConstrainedClient | model={self.model} "
                    f"response_model={response_model.__name__} "
                    f"latency_ms={latency_ms:.1f} validation_passed=False error={exc!r}"
                )
                raise StructuredGenerationError(prompt_hash, exc) from exc

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GrammarConstrainedClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
