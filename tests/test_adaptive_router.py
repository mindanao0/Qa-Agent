"""
Tests for AdaptiveRouter — Sprint 3 / Cluster S3-D / GATE S3-D.

Each test uses a temporary SQLite file via the pytest `tmp_path` fixture
so tests are fully isolated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.routing.adaptive_router import AdaptiveRouter, ConfidenceTier


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


def test_unseen_url_classifies_as_low(db_path: Path) -> None:
    """Empty DB → any (url, requirement, domain) classifies as LOW."""
    router = AdaptiveRouter(db_path=db_path)
    tier = router.classify(
        url="https://example.com/page",
        requirement="click the submit button",
        domain="example.com",
    )
    assert tier == ConfidenceTier.LOW


def test_high_tier_after_one_clean_pass(db_path: Path) -> None:
    """One passing record for an exact (url, requirement) → HIGH tier."""
    router = AdaptiveRouter(db_path=db_path)
    url = "https://example.com/login"
    req = "verify login button is visible"
    dom = "example.com"

    router.record_result(
        url=url,
        requirement=req,
        domain=dom,
        passed=True,
        generated_code="def test_generated(page): pass",
    )

    tier = router.classify(url=url, requirement=req, domain=dom)
    assert tier == ConfidenceTier.HIGH


def test_med_tier_after_domain_history(db_path: Path) -> None:
    """3 passing records on SAME domain but DIFFERENT (url, req) pairs →
    a brand-new (url, req) in the same domain classifies as MED."""
    router = AdaptiveRouter(db_path=db_path)
    dom = "shop.example.com"

    for i in range(3):
        router.record_result(
            url=f"https://shop.example.com/page-{i}",
            requirement=f"requirement number {i}",
            domain=dom,
            passed=True,
            generated_code=f"def test_{i}(page): pass",
        )

    # New (url, requirement) — not in cache; but same domain has 3/3 passing.
    tier = router.classify(
        url="https://shop.example.com/brand-new-page",
        requirement="completely fresh requirement",
        domain=dom,
    )
    assert tier == ConfidenceTier.MED


def test_low_tier_persists_when_failures_dominate(db_path: Path) -> None:
    """3 failing records on the same domain → still LOW (pass_rate too low)."""
    router = AdaptiveRouter(db_path=db_path)
    dom = "flaky.example.com"

    for i in range(3):
        router.record_result(
            url=f"https://flaky.example.com/page-{i}",
            requirement=f"requirement number {i}",
            domain=dom,
            passed=False,
            generated_code=f"def test_{i}(page): pass",
        )

    tier = router.classify(
        url="https://flaky.example.com/new",
        requirement="brand new requirement",
        domain=dom,
    )
    assert tier == ConfidenceTier.LOW


def test_lookup_cached_returns_row_after_record(db_path: Path) -> None:
    """record_result → lookup_cached returns a dict with sha256 hex hash."""
    router = AdaptiveRouter(db_path=db_path)
    url = "https://example.com/checkout"
    req = "verify checkout flow completes"
    dom = "example.com"
    code = "def test_checkout(page): page.get_by_role('button').click()"

    router.record_result(
        url=url,
        requirement=req,
        domain=dom,
        passed=True,
        generated_code=code,
    )

    row = router.lookup_cached(url=url, requirement=req)
    assert row is not None
    assert isinstance(row, dict)
    assert row.get("last_generated_code_hash") is not None
    code_hash = row["last_generated_code_hash"]
    assert isinstance(code_hash, str)
    assert len(code_hash) == 64  # sha256 hex = 64 chars
    # All hex chars
    int(code_hash, 16)
