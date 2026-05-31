"""Tests for src/memory/episodic_store.py — uses a REAL LanceDB in tmp_path."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.memory import (
    EpisodicStore,
    HealedExperience,
    MemoryHit,
    PostMortem,
)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fake embedding (no Ollama required)
# ─────────────────────────────────────────────────────────────────────────────


async def fake_embed(text: str) -> list[float]:
    """Deterministic 768-dim vector derived from sha256 of the input text."""
    h = hashlib.sha256(text.encode()).digest()
    vec: list[float] = []
    while len(vec) < 768:
        for b in h:
            vec.append((b / 255.0) * 2 - 1)
            if len(vec) == 768:
                break
        h = hashlib.sha256(h).digest()
    return vec


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_healed(
    *,
    domain: str = "example.com",
    failure_signature: str = "sigX",
    winning_strategy: str = "get_by_role:button:Submit",
    bad_strategy: str = "get_by_text:Submit",
    confidence: float = 0.7,
    impact: float = 0.7,
    created_at: datetime | None = None,
    root_cause: str = "Stale locator after re-render",
) -> HealedExperience:
    return HealedExperience(
        memory_id=uuid.uuid4().hex,
        session_id="s-1",
        domain=domain,
        page_url=f"https://{domain}/x",
        page_type="form",
        failure_signature=failure_signature,
        bad_strategy=bad_strategy,
        winning_strategy=winning_strategy,
        root_cause=root_cause,
        confidence=confidence,
        impact_score=impact,
        created_at_iso=_iso(created_at or _now()),
        vector=[],
    )


def _make_pm(
    *,
    domain: str = "example.com",
    failure_signature: str = "sigX",
    locator_candidates: list[str] | None = None,
    impact: float = 0.5,
    confidence: float = 0.0,
    attempt_id: int = 1,
    created_at: datetime | None = None,
) -> PostMortem:
    return PostMortem(
        memory_id=uuid.uuid4().hex,
        session_id="s-1",
        attempt_id=attempt_id,
        domain=domain,
        page_url=f"https://{domain}/y",
        failure_signature=failure_signature,
        current_plan="click Submit",
        locator_candidates=locator_candidates
        or ["get_by_text:Submit", "get_by_role:button:Save"],
        error_message="TimeoutError waiting for locator",
        dom_snapshot_ref="/tmp/snap.html",
        confidence=confidence,
        impact_score=impact,
        created_at_iso=_iso(created_at or _now()),
        vector=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path: Path):
    db_path = tmp_path / "vector_db"
    s = EpisodicStore(db_path=db_path, embedding_fn=fake_embed)
    await s.connect()
    try:
        yield s
    finally:
        await s.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_retrieve_healed_experience(store: EpisodicStore) -> None:
    exp = _make_healed(
        domain="acme.com",
        failure_signature="sig-login",
        winning_strategy="get_by_role:button:LogIn",
        confidence=0.8,
        impact=0.9,
    )
    await store.save_healed_experience(exp)

    hits = await store.retrieve_for_planning(
        query_text="login button broken",
        domain="acme.com",
        failure_signature="sig-login",
        limit=5,
    )

    assert hits, "expected at least one MemoryHit"
    assert any(
        h.strategy == exp.winning_strategy
        and h.memory_type == "healed_experience"
        and h.failure_signature == "sig-login"
        for h in hits
    )


@pytest.mark.asyncio
async def test_post_mortem_is_append_only(store: EpisodicStore) -> None:
    pm1 = _make_pm(domain="d.com", failure_signature="s1")
    pm2 = _make_pm(domain="d.com", failure_signature="s1")
    await store.save_post_mortem(pm1)
    await store.save_post_mortem(pm2)

    # Inspect the raw table — both rows must exist, none mutated.
    tbl = store._post_mortems  # noqa: SLF001 — intentional white-box check
    assert tbl is not None
    rows = tbl.to_arrow().to_pylist()
    assert len(rows) == 2, f"expected exactly 2 rows, got {len(rows)}"
    ids = {r["memory_id"] for r in rows}
    assert ids == {pm1.memory_id, pm2.memory_id}


@pytest.mark.asyncio
async def test_retrieve_filtered_by_domain_and_signature(
    store: EpisodicStore,
) -> None:
    a_x = _make_healed(
        domain="A.com",
        failure_signature="sigX",
        winning_strategy="WIN_A_X",
    )
    a_y = _make_healed(
        domain="A.com",
        failure_signature="sigY",
        winning_strategy="WIN_A_Y",
    )
    b_x = _make_healed(
        domain="B.com",
        failure_signature="sigX",
        winning_strategy="WIN_B_X",
    )
    await store.save_healed_experience(a_x)
    await store.save_healed_experience(a_y)
    await store.save_healed_experience(b_x)

    hits = await store.retrieve_for_planning(
        query_text="anything",
        domain="A.com",
        failure_signature="sigX",
        limit=10,
    )
    strategies = {h.strategy for h in hits}
    assert strategies == {"WIN_A_X"}, (
        f"expected only WIN_A_X, got {strategies}"
    )


@pytest.mark.asyncio
async def test_evolve_policy_prefers_high_confidence_strategies(
    store: EpisodicStore,
) -> None:
    # All same (domain, signature); winner is the one with highest conf*impact.
    low = _make_healed(
        domain="P.com",
        failure_signature="sigP",
        winning_strategy="LOW",
        confidence=0.3,
        impact=0.3,
    )
    mid = _make_healed(
        domain="P.com",
        failure_signature="sigP",
        winning_strategy="MID",
        confidence=0.6,
        impact=0.5,
    )
    high = _make_healed(
        domain="P.com",
        failure_signature="sigP",
        winning_strategy="HIGH",
        confidence=0.9,
        impact=0.9,
    )
    for e in (low, mid, high):
        await store.save_healed_experience(e)

    # A post-mortem so avoid_strategies has content.
    pm = _make_pm(
        domain="P.com",
        failure_signature="sigP",
        locator_candidates=["BAD_CAND_1", "BAD_CAND_2"],
        impact=0.8,
    )
    await store.save_post_mortem(pm)

    policy = await store.evolve_locator_policy("P.com", "sigP")
    assert policy.domain == "P.com"
    assert policy.failure_signature == "sigP"
    assert policy.preferred_strategies, "expected non-empty preferred_strategies"
    assert policy.preferred_strategies[0] == "HIGH"
    # No more than 3
    assert len(policy.preferred_strategies) <= 3
    assert len(policy.avoid_strategies) <= 3
    assert "BAD_CAND_1" in policy.avoid_strategies


@pytest.mark.asyncio
async def test_retrieve_returns_ranked_by_composite_score(
    store: EpisodicStore,
) -> None:
    # All three match the (domain, sig) filter. Scores chosen so the ranking
    # is unambiguous regardless of vector distance:
    #   composite = 0.4*conf + 0.3*impact + 0.3*recency
    now = _now()

    # Entry A — fresh, high confidence, high impact -> high composite
    a = _make_healed(
        domain="R.com",
        failure_signature="sigR",
        winning_strategy="STRAT_A",
        confidence=0.9,
        impact=0.9,
        created_at=now,
    )
    # Entry B — fresh, medium everywhere
    b = _make_healed(
        domain="R.com",
        failure_signature="sigR",
        winning_strategy="STRAT_B",
        confidence=0.5,
        impact=0.5,
        created_at=now,
    )
    # Entry C — old (way past 30-day window) and low scores -> lowest composite
    c = _make_healed(
        domain="R.com",
        failure_signature="sigR",
        winning_strategy="STRAT_C",
        confidence=0.1,
        impact=0.1,
        created_at=now - timedelta(days=120),
    )
    await store.save_healed_experience(a)
    await store.save_healed_experience(b)
    await store.save_healed_experience(c)

    hits = await store.retrieve_for_planning(
        query_text="something",
        domain="R.com",
        failure_signature="sigR",
        limit=3,
    )
    assert len(hits) == 3
    ordered_strategies = [h.strategy for h in hits]
    assert ordered_strategies == ["STRAT_A", "STRAT_B", "STRAT_C"], (
        f"expected ranked order [A,B,C], got {ordered_strategies}"
    )

    # Sanity: returned objects are MemoryHits with correct memory_type
    assert all(isinstance(h, MemoryHit) for h in hits)
    assert all(h.memory_type == "healed_experience" for h in hits)
