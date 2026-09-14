"""Coverage cross-check: merge independent "how much of the site did we see"
signals into one report.

Two entry points:
  run_crosscheck()       the original "necessary first" 3-dimension subset
                          (static_declared, sfg_crawl, js_css_coverage) —
                          kept unchanged for backward compatibility with
                          existing eval/coverage_*.json reports.
  run_full_crosscheck()  all 9 dimensions (the 3 above plus viewport A/B,
                          session-state A/B, keyboard-walk, monkey-walk, and
                          declared API surface) — the production entry point
                          used by UniversalQAAgent(enable_coverage_crosscheck=True).

No single dimension here — human-curated golden dataset included — is
treated as ground truth; agreement across dimensions is the signal, and a
path seen by only one method is a lead to chase, not proof of a gap. See
eval/FINDINGS_explore.md.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Page, async_playwright

from src.contractskill.sfg import SFGStore
from src.universal_qa.coverage.dim_api_spec import collect as collect_api_spec
from src.universal_qa.coverage.dim_js_coverage import JSCoverageSession
from src.universal_qa.coverage.dim_keyboard_walk import collect as collect_keyboard_walk
from src.universal_qa.coverage.dim_monkey_walk import collect as collect_monkey_walk
from src.universal_qa.coverage.dim_session_state import collect as collect_session_state
from src.universal_qa.coverage.dim_static import collect as collect_static
from src.universal_qa.coverage.dim_viewport import collect as collect_viewport
from src.universal_qa.coverage.dimensions import DimensionResult, merge_dimensions
from src.universal_qa.explorer.sfg_traversal import SFGTraversalExplorer

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


async def _collect_shared_session(
    page: Page, ctx: BrowserContext, target_id: str, seed_url: str, safe_mode: bool = True,
) -> list[DimensionResult]:
    """Dimensions that share one browser page: sfg_crawl + js_css_coverage.

    Caller owns the browser/context lifecycle; this only navigates the page.
    """
    js_cov = await JSCoverageSession.start(ctx, page)
    await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)

    results: list[DimensionResult] = []
    tmp = Path(tempfile.mkdtemp(prefix="uqa_covcheck_"))
    try:
        store = SFGStore(db_path=tmp / "sfg.db")
        crawler = SFGTraversalExplorer(store, max_states=150, time_budget_s=180,
                                        safe_mode=safe_mode)
        await crawler.explore(page, page.url, {})

        base = f"{urlsplit(page.url).scheme}://{urlsplit(page.url).netloc}"
        crawl_paths = {_path(n.url) for n in store.get_nodes_by_url_prefix(base)}
        results.append(DimensionResult(
            method="sfg_crawl", target_id=target_id, paths=crawl_paths,
            meta={"state_count": store.node_count(), "blocked": crawler.blocked,
                  "blocked_submits": crawler.blocked_submits, "safe_mode": safe_mode},
        ))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    results.append(await js_cov.stop(target_id))
    return results


async def run_crosscheck(target_id: str, seed_url: str, headless: bool = True,
                          safe_mode: bool = True) -> dict:
    """The original 3-dimension cross-check: static + sfg_crawl + js_coverage."""
    results: list[DimensionResult] = [await collect_static(target_id, seed_url)]

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(user_agent=_UA)
        page = await ctx.new_page()
        try:
            results.extend(
                await _collect_shared_session(page, ctx, target_id, seed_url, safe_mode=safe_mode)
            )
        finally:
            await browser.close()

    merged = merge_dimensions(results)
    merged["target_id"] = target_id
    merged["seed_url"] = seed_url
    return merged


async def run_full_crosscheck(target_id: str, seed_url: str, headless: bool = True,
                               safe_mode: bool = True) -> dict:
    """All 9 coverage dimensions, merged into one report.

    Runs the shared-browser-session dimensions (static, sfg_crawl,
    js_coverage) plus the self-contained ones (viewport A/B, session-state
    A/B, keyboard-walk, monkey-walk, declared API surface) — each of the
    latter manages its own browser/HTTP session. Sequential by design: this
    project targets constrained hardware, and dimension results aren't
    time-sensitive relative to one another.
    """
    results: list[DimensionResult] = [
        await collect_static(target_id, seed_url),
        await collect_api_spec(target_id, seed_url),
    ]

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(user_agent=_UA)
        page = await ctx.new_page()
        try:
            results.extend(
                await _collect_shared_session(page, ctx, target_id, seed_url, safe_mode=safe_mode)
            )
        finally:
            await browser.close()

    results.extend(await collect_viewport(target_id, seed_url, headless=headless, safe_mode=safe_mode))
    results.extend(await collect_session_state(target_id, seed_url, headless=headless, safe_mode=safe_mode))
    results.append(await collect_keyboard_walk(target_id, seed_url, headless=headless))
    results.append(await collect_monkey_walk(target_id, seed_url, headless=headless))

    merged = merge_dimensions(results)
    merged["target_id"] = target_id
    merged["seed_url"] = seed_url
    return merged


__all__ = ["run_crosscheck", "run_full_crosscheck"]
