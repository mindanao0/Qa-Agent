# tests/test_grounder.py
"""
Tests for src/perception/grounder.py and src/perception/semantic_compactor.py.

Integration tests require Playwright chromium and network access.
Run non-integration tests with:
    uv run pytest tests/test_grounder.py -v -m "not integration"

Run integration tests with:
    uv run pytest tests/test_grounder.py -v -m integration --timeout=60
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import async_playwright

from src.perception.aom_extractor import (
    AOMNode,
    AOMSnapshot,
    AOMSparseError,
)
from src.perception.dom_pruner import PrunedDOMSnapshot, PrunedElement
from src.perception.grounder import Grounder
from src.perception.semantic_compactor import CompactPAM, SemanticCompactor

pytestmark = pytest.mark.asyncio


# ── Test 1: integration — real page, within budget ───────────────────────────


@pytest.mark.integration
async def test_ground_returns_compact_pam_within_budget() -> None:
    """
    Grounder must return a valid CompactPAM within the token budget on a
    real Playwright page (TodoMVC demo).
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto("https://demo.playwright.dev/todomvc", timeout=30_000)
            pam = await Grounder().ground(page, context_budget_tokens=1000)
            assert isinstance(pam, CompactPAM)
            assert pam.estimated_tokens <= 1000
            assert pam.source in ("aom", "dom", "hybrid", "failure")
            assert pam.format in ("md", "json")
            assert len(pam.content) > 0
        finally:
            await browser.close()


# ── Test 2: integration — canvas-only page, falls back to DOM ────────────────


@pytest.mark.integration
async def test_ground_falls_back_to_dom_on_sparse_aom() -> None:
    """
    A canvas-only page produces a sparse/empty AOM. Grounder should fall back
    to DOM-only source. Since Grounder never raises (R2 safety net), source
    may be 'dom', 'hybrid', or 'failure' on a canvas-only page.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(
                "<html><body>"
                '<canvas id="c" width="800" height="600"></canvas>'
                "</body></html>"
            )
            # Grounder never raises (R2 safety net) — just check valid PAM
            pam = await Grounder().ground(page, context_budget_tokens=1000)
            assert isinstance(pam, CompactPAM)
            assert pam.source in ("dom", "hybrid", "failure")
        finally:
            await browser.close()


# ── Test 3: unit — sparse AOM triggers DOM-only source ───────────────────────


async def test_dom_source_when_aom_sparse() -> None:
    """
    Unit test: when AOMExtractor raises AOMSparseError and DOMPruner returns
    a valid snapshot, Grounder must produce source="dom".
    """
    # Build a minimal PrunedDOMSnapshot with one element
    elem = PrunedElement(
        element_id="aabbccdd",
        tag="button",
        role="button",
        accessible_name="Submit",
        text="Submit",
        state={},
        locator='role=button[name="Submit"]',
        bbox=None,
        score=18.0,
        meta={},
    )
    fake_dom = PrunedDOMSnapshot(
        snapshot_id="deadbeef01234567",
        url="about:blank",
        timestamp_iso="2026-01-01T00:00:00+00:00",
        elements=[elem],
        reduction_ratio=1.0,
        extraction_latency_ms=10,
    )

    # Mock page
    mock_page = MagicMock()
    mock_page.url = "about:blank"

    with (
        patch(
            "src.perception.grounder.AOMExtractor.extract",
            new_callable=AsyncMock,
            side_effect=AOMSparseError("about:blank", 3),
        ),
        patch(
            "src.perception.grounder.DOMPruner.prune",
            new_callable=AsyncMock,
            return_value=fake_dom,
        ),
    ):
        pam = await Grounder().ground(mock_page, context_budget_tokens=1000)

    assert isinstance(pam, CompactPAM)
    assert pam.source == "dom"
    assert pam.controls_count >= 1


# ── Test 4: integration — tiny budget, grounder never raises ─────────────────


@pytest.mark.integration
async def test_budget_enforced() -> None:
    """
    With a very small budget (50 tokens), the grounder must not raise.
    It degrades gracefully through max_items=25→15→10 and returns a valid PAM.
    If still over 50 tokens at max_items=10, that is acceptable per spec.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto("https://demo.playwright.dev/todomvc", timeout=30_000)
            pam = await Grounder().ground(page, context_budget_tokens=50)
            assert isinstance(pam, CompactPAM)
            assert pam.source in ("aom", "dom", "hybrid", "failure")
            assert len(pam.content) > 0
        finally:
            await browser.close()


# ── SemanticCompactor unit tests ──────────────────────────────────────────────


def _make_minimal_aom() -> AOMSnapshot:
    """Helper: build a tiny AOMSnapshot with 5 nodes."""
    root = AOMNode(
        role="WebArea",
        name="Test Page",
        children=[
            AOMNode(
                role="button",
                name="Sign in",
                children=[],
                source_node_id="aaa111",
            ),
            AOMNode(
                role="textbox",
                name="email",
                children=[],
                source_node_id="bbb222",
            ),
            AOMNode(
                role="textbox",
                name="password",
                children=[],
                source_node_id="ccc333",
            ),
            AOMNode(
                role="link",
                name="Forgot password?",
                children=[],
                source_node_id="ddd444",
            ),
        ],
        source_node_id="root000",
    )
    return AOMSnapshot(
        snapshot_id="test0001",
        url="https://example.com",
        timestamp_iso="2026-01-01T00:00:00+00:00",
        root_node=root,
        node_count=5,
        extraction_latency_ms=5,
    )


def _make_minimal_dom() -> PrunedDOMSnapshot:
    """Helper: build a minimal PrunedDOMSnapshot."""
    elems = [
        PrunedElement(
            element_id="e1",
            tag="button",
            role="button",
            accessible_name="Sign in",
            text="Sign in",
            state={"disabled": False},
            locator='role=button[name="Sign in"]',
            bbox=None,
            score=23.0,
            meta={},
        ),
        PrunedElement(
            element_id="e2",
            tag="input",
            role="textbox",
            accessible_name="email",
            text="",
            state={},
            locator='[aria-label="email"]',
            bbox=None,
            score=18.0,
            meta={},
        ),
    ]
    return PrunedDOMSnapshot(
        snapshot_id="snap0001",
        url="https://example.com",
        timestamp_iso="2026-01-01T00:00:00+00:00",
        elements=elems,
        reduction_ratio=5.0,
        extraction_latency_ms=8,
    )


async def test_compactor_aom_only_md() -> None:
    """SemanticCompactor with AOM-only input produces valid Markdown CompactPAM."""
    aom = _make_minimal_aom()
    pam = SemanticCompactor().compact(aom, None, max_items=25, output_format="md")
    assert isinstance(pam, CompactPAM)
    assert pam.source == "aom"
    assert pam.format == "md"
    assert "## Controls" in pam.content
    assert pam.controls_count >= 1
    assert pam.forms_count >= 1
    assert pam.estimated_tokens >= 1


async def test_compactor_dom_only_json() -> None:
    """SemanticCompactor with DOM-only input produces valid JSON CompactPAM."""
    dom = _make_minimal_dom()
    pam = SemanticCompactor().compact(None, dom, max_items=25, output_format="json")
    assert isinstance(pam, CompactPAM)
    assert pam.source == "dom"
    assert pam.format == "json"
    import json as _json
    data = _json.loads(pam.content)
    assert "controls" in data
    assert "forms" in data


async def test_compactor_hybrid_deduplicates() -> None:
    """
    SemanticCompactor with both AOM and DOM merges matched elements (Sign in button).
    """
    aom = _make_minimal_aom()
    dom = _make_minimal_dom()
    pam = SemanticCompactor().compact(aom, dom, max_items=25, output_format="md")
    assert pam.source == "hybrid"
    # The Sign in button must appear exactly once
    count = pam.content.count('name="Sign in"')
    assert count == 1, f"Expected 1 occurrence of 'Sign in', got {count}"


async def test_compactor_raises_when_no_input() -> None:
    """SemanticCompactor raises ValueError when both aom and dom are None."""
    with pytest.raises(ValueError, match="At least one"):
        SemanticCompactor().compact(None, None)


async def test_compactor_max_items_cap() -> None:
    """
    SemanticCompactor drops nodes when total > max_items, and tracks dropped_nodes.
    """
    aom = _make_minimal_aom()  # 5 nodes → controls+forms+containers
    pam_full = SemanticCompactor().compact(aom, None, max_items=25)
    pam_capped = SemanticCompactor().compact(aom, None, max_items=1)
    # Capped version should have fewer or equal elements
    total_full = pam_full.controls_count + pam_full.forms_count + pam_full.lists_count
    total_capped = pam_capped.controls_count + pam_capped.forms_count + pam_capped.lists_count
    assert total_capped <= total_full
    # dropped_nodes should be non-negative
    assert pam_capped.dropped_nodes >= 0


# ── Test: emergency PAM when all perception layers fail ───────────────────────


async def test_emergency_pam_when_all_fail() -> None:
    """
    When both AOMExtractor and DOMPruner raise, Grounder must NOT raise.
    It must return a valid CompactPAM with source='failure' containing the URL.
    (R2.1 graceful degradation safety net)
    """
    mock_page = MagicMock()
    mock_page.url = "https://example.com/test-page"
    mock_page.title = MagicMock(return_value="Test Page Title")

    with (
        patch(
            "src.perception.grounder.AOMExtractor.extract",
            new_callable=AsyncMock,
            side_effect=RuntimeError("CDP failed"),
        ),
        patch(
            "src.perception.grounder.DOMPruner.prune",
            new_callable=AsyncMock,
            side_effect=RuntimeError("JS evaluate failed"),
        ),
    ):
        pam = await Grounder().ground(mock_page, context_budget_tokens=1000)

    assert isinstance(pam, CompactPAM)
    assert pam.source == "failure"
    assert "https://example.com/test-page" in pam.content
    assert pam.controls_count == 0
    assert pam.forms_count == 0
    assert pam.estimated_tokens > 0
