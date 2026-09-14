"""Unit tests for src.cache.cached_structured.cached_create_structured —
the generic SemanticCache-routed wrapper around InstructorClient.create_structured.
Pure mocks (no live LanceDB/embedding model): SemanticCache and InstructorClient
are both AsyncMock, so no network/model download is exercised here."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, ConfigDict

from src.cache.cached_structured import cached_create_structured
from src.cache.semantic_cache import CacheEntry, CacheLookupResult


class _Dummy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def _entry(json_payload: str) -> CacheEntry:
    return CacheEntry(
        cache_id="c1", query_text="q", query_vector=[0.0] * 384,
        playwright_code=json_payload, created_at="t", updated_at="t",
    )


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_value_without_calling_llm():
    cache = AsyncMock()
    cache.lookup.return_value = CacheLookupResult(
        route="CACHE_HIT",
        cached_entry=_entry(_Dummy(value="cached").model_dump_json()),
        score=0.99, query_vector=[0.0] * 384,
    )
    client = AsyncMock()

    result = await cached_create_structured(client, cache, "q", _Dummy, temperature=0.0)

    assert result.value == "cached"
    client.create_structured.assert_not_called()
    cache.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_guided_injects_prior_answer_as_hint_and_upserts_fresh_result():
    cache = AsyncMock()
    cache.lookup.return_value = CacheLookupResult(
        route="GUIDED",
        cached_entry=_entry(_Dummy(value="prior").model_dump_json()),
        score=0.7, query_vector=[0.0] * 384,
    )
    client = AsyncMock()
    client.create_structured.return_value = _Dummy(value="fresh")

    result = await cached_create_structured(client, cache, "q2", _Dummy, temperature=0.0)

    assert result.value == "fresh"
    client.create_structured.assert_called_once()
    messages = client.create_structured.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "prior" in messages[0]["content"]
    cache.upsert.assert_called_once()
    assert cache.upsert.call_args.kwargs["query"] == "q2"
    assert "fresh" in cache.upsert.call_args.kwargs["playwright_code"]


@pytest.mark.asyncio
async def test_cache_miss_calls_llm_with_plain_query_and_upserts():
    cache = AsyncMock()
    cache.lookup.return_value = CacheLookupResult(
        route="CACHE_MISS", cached_entry=None, score=0.1, query_vector=[0.0] * 384,
    )
    client = AsyncMock()
    client.create_structured.return_value = _Dummy(value="new")

    result = await cached_create_structured(client, cache, "q3", _Dummy, temperature=0.0)

    assert result.value == "new"
    messages = client.create_structured.call_args[0][0]
    assert len(messages) == 1 and messages[0]["role"] == "user" and messages[0]["content"] == "q3"
    cache.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_failure_is_swallowed_result_still_returned():
    cache = AsyncMock()
    cache.lookup.return_value = CacheLookupResult(
        route="CACHE_MISS", cached_entry=None, score=0.1, query_vector=[0.0] * 384,
    )
    cache.upsert.side_effect = RuntimeError("db locked")
    client = AsyncMock()
    client.create_structured.return_value = _Dummy(value="still-works")

    result = await cached_create_structured(client, cache, "q4", _Dummy, temperature=0.0)

    assert result.value == "still-works"
