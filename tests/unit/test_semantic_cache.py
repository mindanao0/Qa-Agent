"""Unit tests for SemanticCache (Priority 3 spec)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.cache import CacheLookupResult, SemanticCache


@pytest.fixture
async def cache(tmp_path: Path) -> SemanticCache:
    """Fresh SemanticCache backed by a per-test tmp LanceDB directory."""
    db_path = tmp_path / "semantic_cache.lance"
    c = SemanticCache(db_path=db_path)
    await c.initialize()
    return c


async def test_cache_miss_on_empty_db(cache: SemanticCache) -> None:
    """Lookup against an empty table returns CACHE_MISS with no entry."""
    result: CacheLookupResult = await cache.lookup("some unseen query")

    assert result.route == "CACHE_MISS"
    assert result.cached_entry is None
    assert result.score < cache.guided_threshold
    assert len(result.query_vector) == 384


async def test_cache_hit_above_threshold(cache: SemanticCache) -> None:
    """Upserting then looking up the identical query yields CACHE_HIT (s >= 0.95)."""
    query = "click the login button"
    code = "await page.get_by_role('button', name='Login').click()"
    await cache.upsert(query=query, playwright_code=code, metadata={"source": "unit"})

    result = await cache.lookup(query)

    assert result.route == "CACHE_HIT"
    assert result.cached_entry is not None
    assert result.cached_entry.playwright_code == code
    assert result.score >= cache.hit_threshold


async def test_guided_mode_similar_query(cache: SemanticCache) -> None:
    """A semantically similar but non-identical query routes to GUIDED."""
    await cache.upsert(
        query="login test",
        playwright_code="await page.get_by_role('button', name='Login').click()",
        metadata={"source": "unit"},
    )

    result = await cache.lookup("test the login flow")

    assert result.route == "GUIDED"
    assert result.cached_entry is not None
    assert cache.guided_threshold <= result.score < cache.hit_threshold


async def test_upsert_increments_version(cache: SemanticCache) -> None:
    """Upserting the same normalized query twice bumps version to 2."""
    query = "fill in the email field"

    first = await cache.upsert(
        query=query,
        playwright_code="await page.get_by_label('Email').fill('a@b.co')",
        metadata={"source": "unit"},
    )
    assert first.version == 1

    lookup = await cache.lookup(query)
    assert lookup.cached_entry is not None

    second = await cache.upsert(
        query=query,
        playwright_code="await page.get_by_label('Email').fill('updated@b.co')",
        metadata={"source": "unit"},
        existing_entry=lookup.cached_entry,
    )

    assert second.cache_id == first.cache_id
    assert second.version == 2


async def test_invalidate_removes_entry(cache: SemanticCache) -> None:
    """Invalidating a cached entry makes the same query miss the cache."""
    query = "submit the form"
    entry = await cache.upsert(
        query=query,
        playwright_code="await page.get_by_role('button', name='Submit').click()",
        metadata={"source": "unit"},
    )

    deleted = await cache.invalidate(entry.cache_id)
    assert deleted is True

    result = await cache.lookup(query)
    assert result.route == "CACHE_MISS"
    assert result.cached_entry is None
