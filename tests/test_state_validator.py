# tests/test_state_validator.py
"""
Integration tests for src/healing/state_validator.py (Sprint 2 cluster S2-B).

All tests require a live Chromium via Playwright. Run with::

    uv run pytest tests/test_state_validator.py -v -m integration
"""
from __future__ import annotations

import time

import pytest
from playwright.async_api import async_playwright

from src.healing.state_validator import StateValidator

pytestmark = pytest.mark.asyncio


@pytest.mark.integration
async def test_captures_pre_state_as_aom_snapshot() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(
                "<button>Hello</button><h1>World</h1>"
                "<a href='#x'>link</a><label>Name<input/></label>"
                "<ul><li>one</li><li>two</li></ul>"
            )
            v = StateValidator()
            pre = await v.capture_pre_state(page)
            assert pre.url is not None
            # On a meaningful page the hash should be the canonical 16-hex form.
            # (If AOM came back sparse the hash would be "" — that's a regression here.)
            assert len(pre.pre_state_hash) == 16
            assert pre.pre_snapshot.node_count > 0
        finally:
            await browser.close()


@pytest.mark.integration
async def test_classifies_successful_navigation() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            # Start on about:blank
            await page.goto("about:blank")
            v = StateValidator()
            pre = await v.capture_pre_state(page)

            # "Navigate" to a new URL (data: URL changes page.url)
            await page.goto(
                "data:text/html,<html><body>"
                "<h1>Dashboard</h1><button>One</button><button>Two</button>"
                "<a href='#a'>a</a><a href='#b'>b</a><input/>"
                "</body></html>"
            )

            cls = await v.classify_post_action(page, pre, action_id="nav-1")
            assert cls.outcome_label == "Success"
            assert cls.url_changed is True
            assert cls.confidence_score >= 0.8
        finally:
            await browser.close()


@pytest.mark.integration
async def test_classifies_error_state_from_alert_role() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(
                "<h1>Login</h1><label>User<input/></label>"
                "<label>Pass<input/></label><button>Submit</button>"
            )
            v = StateValidator()
            pre = await v.capture_pre_state(page)

            await page.set_content(
                "<h1>Login</h1>"
                "<div role='alert'>Login failed</div>"
                "<label>User<input/></label>"
                "<label>Pass<input/></label><button>Submit</button>"
            )

            cls = await v.classify_post_action(page, pre, action_id="submit-1")
            assert cls.outcome_label == "Error_State"
            assert cls.failure_signature is not None
            assert len(cls.failure_signature) == 16
            assert cls.a11y_delta.error_text_found is True
        finally:
            await browser.close()


@pytest.mark.integration
async def test_classifies_loading_state_from_aria_busy() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content(
                "<h1>Search</h1><label>Query<input/></label><button>Go</button>"
            )
            v = StateValidator()
            pre = await v.capture_pre_state(page)

            # role=progressbar triggers loading_indicators.
            # An empty progressbar element keeps total_delta small (< 3).
            await page.set_content(
                "<h1>Search</h1><label>Query<input/></label><button>Go</button>"
                "<div role='progressbar' aria-label='busy'></div>"
            )

            cls = await v.classify_post_action(page, pre, action_id="go-1")
            assert cls.outcome_label == "Loading_State", (
                f"expected Loading_State, got {cls.outcome_label}; delta={cls.a11y_delta}"
            )
            assert cls.a11y_delta.loading_indicators is True
            assert cls.failure_signature is None
        finally:
            await browser.close()


@pytest.mark.integration
async def test_stability_waits_without_sleep() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content("<h1>Static</h1><p>nothing changing here</p>")
            v = StateValidator()
            t0 = time.monotonic()
            ok = await v.wait_for_stability(page, timeout_ms=2000)
            elapsed_ms = (time.monotonic() - t0) * 1000
            assert ok is True
            assert elapsed_ms < 1800, (
                f"wait_for_stability took {elapsed_ms:.0f}ms — expected < 1800ms"
            )
        finally:
            await browser.close()


@pytest.mark.integration
async def test_classify_post_action_never_raises_on_closed_page() -> None:
    """Optional 6th gate: a closed page must not crash classify_post_action."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.set_content("<button>Hi</button><h1>Title</h1>")
            v = StateValidator()
            pre = await v.capture_pre_state(page)
            await page.close()
            cls = await v.classify_post_action(page, pre, action_id="closed-1")
            assert cls.outcome_label == "Unknown"
            assert cls.confidence_score == 0.0
            assert cls.failure_signature is None
        finally:
            await browser.close()
