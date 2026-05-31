"""
Tests for src/contractskill/crawler.py — Sprint 4 / S4-B.

4 tests:
  test_crawl_discovers_nodes_from_seed
  test_deduplication_prevents_revisit
  test_hard_limit_stops_crawl
  test_blocked_actions_skipped

All tests use mock Playwright Page objects and a mock Grounder so no real
browser or LLM calls are made.  SFGStore uses a real SQLite file via tmp_path.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.contractskill.crawler import CrawlerConfig, SFGCrawler, _sha256
from src.contractskill.sfg import SFGStore
from src.perception.semantic_compactor import CompactPAM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compact_pam(content: str = "## Page State\n- URL: http://example.com\n") -> CompactPAM:
    """Build a minimal fake CompactPAM."""
    return CompactPAM(
        format="md",
        content=content,
        controls_count=1,
        forms_count=0,
        lists_count=0,
        estimated_tokens=len(content) // 4 + 1,
        source="aom",
        dropped_nodes=0,
    )


def _make_mock_page(
    url: str = "http://example.com/",
    title: str = "Example Page",
    content: str = "<html><body><a href='http://example.com/about'>About</a></body></html>",
    evaluate_result: list | None = None,
) -> MagicMock:
    """Build a mock Playwright Page."""
    page = MagicMock()
    page.url = url
    page.title = AsyncMock(return_value=title)
    page.goto = AsyncMock(return_value=None)
    page.content = AsyncMock(return_value=content)
    page.evaluate = AsyncMock(return_value=evaluate_result or [])
    page.wait_for_load_state = AsyncMock(return_value=None)
    return page


def _make_mock_grounder(pam: CompactPAM | None = None) -> MagicMock:
    """Build a mock Grounder that returns a fixed CompactPAM."""
    grounder = MagicMock()
    grounder.ground = AsyncMock(return_value=pam or _make_compact_pam())
    return grounder


# ---------------------------------------------------------------------------
# Mock helpers: patch the Playwright context manager used by crawl()
# ---------------------------------------------------------------------------

class _FakeContext:
    """Simulates playwright.async_api.async_playwright() context manager."""

    def __init__(self, page: MagicMock) -> None:
        self._page = page
        self._browser = MagicMock()
        self._ctx = MagicMock()

        self._ctx.new_page = AsyncMock(return_value=self._page)
        self._browser.new_context = AsyncMock(return_value=self._ctx)
        self._browser.close = AsyncMock(return_value=None)

        self.chromium = MagicMock()
        self.chromium.launch = AsyncMock(return_value=self._browser)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio


async def test_crawl_discovers_nodes_from_seed(tmp_path: Path) -> None:
    """
    The crawler must visit the seed URL, ground it, create an SFGNode,
    and persist it in the SFGStore.
    """
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    pam = _make_compact_pam("## Page State\n- URL: http://example.com/\n")
    grounder = _make_mock_grounder(pam)

    page = _make_mock_page(url="http://example.com/", title="Home")
    fake_ctx = _FakeContext(page)

    config = CrawlerConfig(max_pages=1, max_depth=0, max_time_minutes=5)
    crawler = SFGCrawler(store, grounder, config)

    with patch("src.contractskill.crawler.async_playwright", return_value=fake_ctx):
        root = await crawler.crawl("http://example.com/")

    # Root node must be created
    assert root is not None
    assert root.url == "http://example.com/"
    assert root.page_title == "Home"
    assert root.pam_content == pam.content

    # Must be persisted in the store
    retrieved = store.get_node(root.node_id)
    assert retrieved is not None
    assert retrieved.node_id == root.node_id
    assert store.node_count() == 1

    # Grounder must have been called at least once
    grounder.ground.assert_called_once()


async def test_deduplication_prevents_revisit(tmp_path: Path) -> None:
    """
    If the same page state is encountered twice (same url_path + PAM hash),
    the crawler must NOT create a duplicate node.
    """
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    # Both calls to ground() return identical PAM content → same aom_hash
    shared_pam_content = "## Page State\n- URL: http://example.com/\n- title: Home\n"
    pam = _make_compact_pam(shared_pam_content)
    grounder = _make_mock_grounder(pam)

    # The page always reports the same URL → same node_id every time
    page = _make_mock_page(url="http://example.com/", title="Home")
    fake_ctx = _FakeContext(page)

    # Allow up to 5 pages but with no outgoing edges, BFS terminates after 1 visit
    config = CrawlerConfig(max_pages=5, max_depth=2, max_time_minutes=5)
    crawler = SFGCrawler(store, grounder, config)

    with patch("src.contractskill.crawler.async_playwright", return_value=fake_ctx):
        root = await crawler.crawl("http://example.com/")

    # Despite potentially being queued multiple times, only 1 node should exist
    assert store.node_count() == 1
    assert root is not None


async def test_hard_limit_stops_crawl(tmp_path: Path) -> None:
    """
    When max_pages=2, the crawler must stop after visiting 2 distinct nodes.
    The store must contain exactly max_pages nodes (or fewer if fewer discovered).
    """
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    # PAMs with unique content per call → different aom_hash → different node_id
    call_count = 0
    urls = ["http://example.com/", "http://example.com/page2", "http://example.com/page3"]

    async def _dynamic_ground(page, *args, **kwargs):
        nonlocal call_count
        idx = min(call_count, len(urls) - 1)
        content = f"## Page State\n- URL: {urls[idx]}\n- call: {call_count}\n"
        call_count += 1
        return _make_compact_pam(content)

    grounder = MagicMock()
    grounder.ground = _dynamic_ground

    # Simulate page URL cycling through pages
    page_call = 0

    async def _dynamic_goto(url, *args, **kwargs):
        nonlocal page_call
        page_call += 1

    # Create a page whose URL changes on each goto call
    page = MagicMock()
    url_sequence = iter(urls + urls)  # enough for our test

    def _get_url():
        return next(url_sequence, urls[-1])

    # Use a property-like side_effect pattern via __get__ trick — simpler:
    # just return a navigating page that has navigate edges
    page.url = urls[0]
    page.title = AsyncMock(return_value="Page")
    page.goto = AsyncMock(side_effect=_dynamic_goto)
    page.content = AsyncMock(return_value="<html></html>")
    page.wait_for_load_state = AsyncMock(return_value=None)

    navigate_elements = [
        {
            "tag": "a",
            "role": "link",
            "name": "Page 2",
            "type": "",
            "href": "http://example.com/page2",
        },
        {
            "tag": "a",
            "role": "link",
            "name": "Page 3",
            "type": "",
            "href": "http://example.com/page3",
        },
    ]
    page.evaluate = AsyncMock(return_value=navigate_elements)

    fake_ctx = _FakeContext(page)

    # Limit to 2 pages max
    config = CrawlerConfig(max_pages=2, max_depth=5, max_time_minutes=5)
    crawler = SFGCrawler(store, grounder, config)

    with patch("src.contractskill.crawler.async_playwright", return_value=fake_ctx):
        root = await crawler.crawl(urls[0])

    assert root is not None
    # With max_pages=2 the crawler must stop after exactly 2 distinct nodes.
    assert store.node_count() == 2


async def test_blocked_actions_skipped(tmp_path: Path) -> None:
    """
    Edges whose locator matches BLOCKED_ACTION_PATTERNS must NOT be traversed.
    They should be stored with safety_flag='BLOCKED' but not added to BFS queue.
    """
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    pam = _make_compact_pam("## Page State\n- URL: http://example.com/\n")
    grounder = _make_mock_grounder(pam)

    page = _make_mock_page(url="http://example.com/", title="Dashboard")

    # One safe link and one blocked link
    mixed_elements = [
        {
            "tag": "a",
            "role": "link",
            "name": "About",
            "type": "",
            "href": "http://example.com/about",
        },
        {
            "tag": "a",
            "role": "link",
            "name": "Delete Account",
            "type": "",
            "href": "http://example.com/delete",
        },
        {
            "tag": "button",
            "role": "button",
            "name": "Logout",
            "type": "",
            "href": "",
        },
    ]
    page.evaluate = AsyncMock(return_value=mixed_elements)

    fake_ctx = _FakeContext(page)

    # Allow up to 1 page so we process seed but don't follow any links
    config = CrawlerConfig(max_pages=1, max_depth=0, max_time_minutes=5)
    crawler = SFGCrawler(store, grounder, config)

    with patch("src.contractskill.crawler.async_playwright", return_value=fake_ctx):
        root = await crawler.crawl("http://example.com/")

    assert root is not None

    # Root node exists
    assert store.node_count() == 1

    # Check edges stored: blocked ones should be in DB with BLOCKED flag
    stored_edges = store.get_edges_from(root.node_id)

    # The delete and logout links should be stored as BLOCKED (not traversed)
    blocked_edges = [e for e in stored_edges if e.safety_flag == "BLOCKED"]
    safe_edges = [e for e in stored_edges if e.safety_flag == "SAFE"]

    # "Delete Account" link → BLOCKED (href contains "delete")
    # "Logout" button → BLOCKED (name contains "logout")
    # "About" link → SAFE
    blocked_locators = {e.locator for e in blocked_edges}
    safe_locators = {e.locator for e in safe_edges}

    assert any("delete" in loc.lower() for loc in blocked_locators), (
        f"Expected a BLOCKED edge with 'delete' in locator, got: {blocked_locators}"
    )
    assert any("about" in loc.lower() for loc in safe_locators), (
        f"Expected a SAFE edge with 'about' in locator, got: {safe_locators}"
    )
