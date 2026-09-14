"""Generic caching wrapper around InstructorClient.create_structured(),
backed by SemanticCache (Priority 3).

Caches by the prompt/query text via embedding similarity — not exact string
match — so semantically similar-but-not-identical requests (e.g. two pages
with near-identical form-page PAM content across different sites in a
multi-site eval) can still route to GUIDED instead of a cold LLM call.
"""
from __future__ import annotations

from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

from src.cache.semantic_cache import SemanticCache
from src.llm.instructor_client import InstructorClient

T = TypeVar("T", bound=BaseModel)


async def cached_create_structured(
    client: InstructorClient,
    cache: SemanticCache,
    query: str,
    response_model: type[T],
    temperature: float = 0.0,
    cache_metadata: dict | None = None,
) -> T:
    """create_structured(), routed through a SemanticCache lookup first.

    CACHE_HIT  (score >= hit_threshold)    — replay the stored JSON response
                                              verbatim; no LLM call.
    GUIDED     (guided_threshold <= score) — prepend the stored JSON as a
                                              one-shot formatting reference,
                                              still call the LLM fresh.
    CACHE_MISS (score < guided_threshold)  — call the LLM as normal.

    Every non-hit call is persisted as its own new cache entry afterward
    (never merged into the GUIDED match's entry — that would silently
    overwrite a different query's stored answer).
    """
    lookup = await cache.lookup(query)

    if lookup.route == "CACHE_HIT" and lookup.cached_entry is not None:
        logger.debug(f"cached_create_structured: CACHE_HIT score={lookup.score:.4f}")
        return response_model.model_validate_json(lookup.cached_entry.playwright_code)

    messages: list[dict[str, str]] = [{"role": "user", "content": query}]
    if lookup.route == "GUIDED" and lookup.cached_entry is not None:
        messages.insert(0, {
            "role": "system",
            "content": (
                "A similar prior request produced this JSON response — use it "
                "only as a formatting/style reference, adapt the content to "
                "this new request:\n" + lookup.cached_entry.playwright_code
            ),
        })

    result = await client.create_structured(messages, response_model, temperature=temperature)

    try:
        await cache.upsert(
            query=query,
            playwright_code=result.model_dump_json(),
            metadata=cache_metadata or {},
        )
    except Exception as exc:
        # Caching is a performance optimization, never a correctness gate.
        logger.warning(f"cached_create_structured: upsert failed (non-fatal): {exc!r}")

    return result


__all__ = ["cached_create_structured"]
