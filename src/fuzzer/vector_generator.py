"""AdvancedVectorGenerator — Sprint 13.

Deterministic typed base vectors + best-effort LLM augmentation from the
inferred constraints. Replaces the flat Sprint 10 ``FuzzVectorLibrary``.

``BLOCKED_ACTION_PATTERNS`` are enforced: no delete/remove/transfer/payment/
password vector is ever emitted, and a ``password`` field is never fuzzed.

See docs/superpowers/specs/2026-06-03-sprint13-advanced-api-fuzzing-design.md.
"""
from __future__ import annotations

import asyncio
import re

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.fuzzer.schema_inferrer import InferredSchema
from src.llm.instructor_client import InstructorClient, StructuredGenerationError

BLOCKED_ACTION_PATTERNS = re.compile(
    r"delete|remove|transfer|payment|password", re.IGNORECASE
)

BASE_VECTORS_BY_TYPE: dict[str, list[str]] = {
    "string": [
        "", " " * 500, "null", "<script>alert(1)</script>",
        "' OR '1'='1", "../../../etc/passwd", "𝕳𝖊𝖑𝖑𝖔",
        "a" * 256,   # max length probe
        "a" * 1,     # min length probe
    ],
    "integer": ["0", "-1", "99999999", "1.5", "null", ""],
    "email": ["notanemail", "@nodomain", "a@b", "test+tag@example.com"],
    "slug": ["", "-", "a" * 100, "UPPERCASE", "has space", "../traverse"],
}


class _AugVectors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vectors: list[str]


def infer_field_type(name: str, json_type: str | None) -> str:
    """Map a request field to a BASE_VECTORS_BY_TYPE category."""
    low = name.lower()
    if "email" in low:
        return "email"
    if low in {"slug"}:
        return "slug"
    if low in {"limit", "offset", "page", "count"} or json_type in {"integer", "number"}:
        return "integer"
    return "string"


def fuzzable_fields(schema: InferredSchema) -> list[tuple[str, str]]:
    """(field_name, type_category) pairs from the request schema, BLOCKED removed."""
    props = (schema.request_schema or {}).get("properties", {})
    fields: list[tuple[str, str]] = []
    for name, spec in props.items():
        if BLOCKED_ACTION_PATTERNS.search(name):
            continue  # never fuzz password / delete-style fields
        jtype = spec.get("type") if isinstance(spec, dict) else None
        fields.append((name, infer_field_type(name, jtype)))
    return fields


def base_vectors_for(ftype: str, max_vectors: int = 5) -> list[str]:
    """Deterministic base vectors for a type category (no LLM) — used for path/body
    params that are not top-level request-schema properties (e.g. a slug path param)."""
    base = BASE_VECTORS_BY_TYPE.get(ftype, BASE_VECTORS_BY_TYPE["string"])
    return _clean(list(base), max_vectors)


def _clean(vectors: list[str], max_vectors: int) -> list[str]:
    """Dedupe, drop BLOCKED, cap at max_vectors — order-preserving."""
    out: list[str] = []
    seen: set[str] = set()
    for v in vectors:
        if v in seen or BLOCKED_ACTION_PATTERNS.search(v):
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= max_vectors:
            break
    return out


class AdvancedVectorGenerator:
    """No required constructor args."""

    async def generate(
        self,
        schema: InferredSchema,
        ollama_semaphore: asyncio.Semaphore,
        max_vectors: int = 5,
    ) -> list[str]:
        """Flat, deduped union of fuzz vectors for all fuzzable fields (spec API)."""
        by_field = await self.generate_fields(schema, ollama_semaphore, max_vectors)
        merged: list[str] = []
        seen: set[str] = set()
        for vectors in by_field.values():
            for v in vectors:
                if v not in seen:
                    seen.add(v)
                    merged.append(v)
        return merged

    async def generate_fields(
        self,
        schema: InferredSchema,
        ollama_semaphore: asyncio.Semaphore,
        max_vectors: int = 5,
    ) -> dict[str, list[str]]:
        """Per-field vectors used for targeted injection by the fuzzer."""
        result: dict[str, list[str]] = {}
        fields = fuzzable_fields(schema)
        if not fields:
            return result

        client: InstructorClient | None = None
        try:
            for name, cat in fields:
                base = list(BASE_VECTORS_BY_TYPE.get(cat, BASE_VECTORS_BY_TYPE["string"]))
                aug: list[str] = []
                if schema.constraints:
                    if client is None:
                        client = InstructorClient()
                    aug = await self._augment(
                        client, name, cat, schema.constraints, ollama_semaphore
                    )
                result[name] = _clean([*base, *aug], max_vectors)
        finally:
            if client is not None:
                await client.close()
        return result

    async def _augment(
        self,
        client: InstructorClient,
        field: str,
        category: str,
        constraints: list[str],
        ollama_semaphore: asyncio.Semaphore,
    ) -> list[str]:
        prompt = (
            f"Field '{field}' (type {category}) with constraints: {constraints}\n"
            "Return up to 3 short strings that VIOLATE these constraints to fuzz-test the "
            "field (e.g. for 'min 3 chars' return a 2-char string). "
            "Do NOT include the words delete, remove, transfer, payment or password. "
            "Return JSON with field vectors as a list of short strings."
        )
        try:
            async with ollama_semaphore:
                result = await client.create_structured(
                    prompt=prompt, response_model=_AugVectors, temperature=0.1
                )
            return [v for v in result.vectors if not BLOCKED_ACTION_PATTERNS.search(v)][:3]
        except (StructuredGenerationError, Exception) as exc:
            logger.warning(f"_augment: Ollama failed for field {field!r}: {exc!r}")
            return []


__all__ = [
    "AdvancedVectorGenerator",
    "BASE_VECTORS_BY_TYPE",
    "BLOCKED_ACTION_PATTERNS",
    "base_vectors_for",
    "fuzzable_fields",
    "infer_field_type",
]
