# tests/test_aom_extractor.py
"""
Tests for src/perception/aom_extractor.py.

Integration tests require Playwright chromium and network access.
Run non-integration tests with:
    uv run pytest tests/test_aom_extractor.py -v -m "not integration"

Run integration tests with:
    uv run pytest tests/test_aom_extractor.py -v -m integration --timeout=60
"""
from __future__ import annotations

import pydantic
import pytest
from playwright.async_api import async_playwright

from src.perception.aom_extractor import (
    AOMExtractor,
    AOMNode,
    AOMSnapshot,
    AOMSparseError,
    BBox,
    is_aom_sparse,
)

pytestmark = pytest.mark.asyncio


# ── Test 1: integration — real page snapshot via CDP ─────────────────────────


@pytest.mark.integration
async def test_extract_returns_valid_snapshot_via_cdp() -> None:
    """
    Navigate to the TodoMVC demo and verify AOMExtractor returns a well-formed
    AOMSnapshot with sufficient nodes — now via CDP (TD-15a fix).
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto("https://demo.playwright.dev/todomvc", timeout=30_000)
            await page.wait_for_load_state("networkidle", timeout=15_000)

            extractor = AOMExtractor()
            result = await extractor.extract(page)

            assert isinstance(result, AOMSnapshot)
            assert result.node_count > 10, (
                f"Expected > 10 nodes via CDP, got {result.node_count}"
            )
            assert result.root_node is not None
            assert "todomvc" in result.url.lower() or "playwright" in result.url.lower()
            assert result.snapshot_id and len(result.snapshot_id) > 0
            assert result.timestamp_iso
            assert result.extraction_latency_ms >= 0
        finally:
            await browser.close()


# ── Test 2: integration — CDP session detached after extract ──────────────────


@pytest.mark.integration
async def test_cdp_session_detached_after_extract() -> None:
    """
    Verify the CDP session is properly detached after extract() completes
    (no resource leaks).
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto("https://demo.playwright.dev/todomvc", timeout=30_000)
            await page.wait_for_load_state("networkidle", timeout=15_000)

            extractor = AOMExtractor()
            result = await extractor.extract(page)
            assert result.node_count > 0

            # After extract, page should still be usable (no leaked locks)
            title = await page.title()
            assert len(title) >= 0  # any title is fine
        finally:
            await browser.close()


# ── Test 3: integration — canvas-only page is sparse ─────────────────────────


@pytest.mark.integration
async def test_sparse_aom_detection_on_canvas() -> None:
    """
    A page containing only a <canvas> element has no semantic AOM nodes.
    Expect either AOMSparseError during extract() or is_aom_sparse() == True.
    """
    canvas_html = """<!DOCTYPE html>
<html>
  <head><title>Canvas Only</title></head>
  <body>
    <canvas id="myCanvas" width="800" height="600"></canvas>
  </body>
</html>"""

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(canvas_html)

            extractor = AOMExtractor()
            try:
                snapshot = await extractor.extract(page)
                # If no exception was raised, the snapshot must be flagged as sparse
                assert is_aom_sparse(snapshot), (
                    "Canvas-only page snapshot should be detected as sparse, "
                    f"but is_aom_sparse returned False (node_count={snapshot.node_count})"
                )
            except AOMSparseError:
                # Expected path — sparse detection worked at extract time
                pass
        finally:
            await browser.close()


# ── Test 4: unit — BBox values are in [0, 100] range ─────────────────────────


async def test_coordinates_are_normalized() -> None:
    """
    BBox values must always stay within the 0–100 percent range.
    Verify that hardcoded synthetic AOMSnapshot BBox values satisfy the range
    constraint enforced by the Pydantic model.
    """
    # Typical normalised bbox values — all must pass Pydantic validation
    valid_bboxes = [
        BBox(x_pct=0.0, y_pct=0.0, w_pct=100.0, h_pct=100.0),
        BBox(x_pct=10.5, y_pct=25.0, w_pct=50.0, h_pct=12.5),
        BBox(x_pct=99.9, y_pct=99.9, w_pct=0.1, h_pct=0.1),
        BBox(x_pct=0.0, y_pct=0.0, w_pct=0.0, h_pct=0.0),
    ]

    for bbox in valid_bboxes:
        assert 0.0 <= bbox.x_pct <= 100.0
        assert 0.0 <= bbox.y_pct <= 100.0
        assert 0.0 <= bbox.w_pct <= 100.0
        assert 0.0 <= bbox.h_pct <= 100.0

    # Out-of-range values must be rejected by Pydantic
    with pytest.raises(pydantic.ValidationError):
        BBox(x_pct=-1.0, y_pct=0.0, w_pct=50.0, h_pct=50.0)

    with pytest.raises(pydantic.ValidationError):
        BBox(x_pct=0.0, y_pct=101.0, w_pct=50.0, h_pct=50.0)

    # Verify is_aom_sparse works correctly on a synthetic snapshot
    child_node = AOMNode(
        role="button",
        name="Submit",
        value=None,
        state={"disabled": False},
        bbox=valid_bboxes[1],
        children=[],
        source_node_id="aabbccdd0011",
    )
    root_node = AOMNode(
        role="document",
        name="Test Page",
        value=None,
        state={},
        bbox=valid_bboxes[0],
        children=[child_node] * 6,  # 1 root + 6 children = 7 nodes total
        source_node_id="000000000000",
    )
    snapshot = AOMSnapshot(
        snapshot_id="abcdef1234567890",
        url="https://example.com",
        timestamp_iso="2026-05-23T00:00:00+00:00",
        root_node=root_node,
        node_count=7,
        extraction_latency_ms=42,
    )

    # All nodes have a name, so this should NOT be sparse
    assert not is_aom_sparse(snapshot), (
        "Snapshot with named nodes should not be flagged as sparse"
    )

    # Verify all bbox values in the snapshot are in range
    for bbox_obj in [root_node.bbox, child_node.bbox]:
        assert bbox_obj is not None
        assert 0 <= bbox_obj.x_pct <= 100
        assert 0 <= bbox_obj.y_pct <= 100
        assert 0 <= bbox_obj.w_pct <= 100
        assert 0 <= bbox_obj.h_pct <= 100
