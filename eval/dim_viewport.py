"""Coverage dimension: viewport A/B.

Runs the *same* SFG crawl twice on the same target — once at a mobile
viewport, once at a desktop viewport — and reports each as its own
paths-bearing DimensionResult. Responsive sites often hide/show entirely
different nav structures (hamburger menu vs. full nav bar) depending on
viewport width; a single-viewport crawl structurally cannot see both, so
this dimension exists to catch what the others can't.

Zero LLM calls. Each viewport gets its own fresh SFGStore (separate temp
dir) and its own fresh browser context — no state is shared between the
two runs.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import async_playwright

from src.contractskill.sfg import SFGStore
from src.universal_qa.explorer.sfg_traversal import SFGTraversalExplorer
from eval.coverage_dimensions import DimensionResult

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# iPhone-ish portrait vs. a common desktop resolution.
_MOBILE_VIEWPORT = {"width": 390, "height": 844}
_DESKTOP_VIEWPORT = {"width": 1920, "height": 1080}


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


async def _crawl_at_viewport(pw, seed_url: str, viewport: dict, method: str,
                              target_id: str, headless: bool,
                              safe_mode: bool) -> DimensionResult:
    tmp = Path(tempfile.mkdtemp(prefix=f"uqa_covcheck_{method}_"))
    browser = await pw.chromium.launch(headless=headless)
    try:
        ctx = await browser.new_context(user_agent=_UA)
        page = await ctx.new_page()
        # Viewport MUST be set before goto()/explore() so the crawler sees
        # the responsive layout (media-query driven nav) from the very
        # first observation, not after a resize.
        await page.set_viewport_size(viewport)
        await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)

        store = SFGStore(db_path=tmp / "sfg.db")
        crawler = SFGTraversalExplorer(store, max_states=150, time_budget_s=180,
                                       safe_mode=safe_mode)
        await crawler.explore(page, page.url, {})

        base = f"{urlsplit(page.url).scheme}://{urlsplit(page.url).netloc}"
        crawl_paths = {_path(n.url) for n in store.get_nodes_by_url_prefix(base)}

        return DimensionResult(
            method=method,
            target_id=target_id,
            paths=crawl_paths,
            meta={
                "viewport": f"{viewport['width']}x{viewport['height']}",
                "state_count": store.node_count(),
                "blocked": crawler.blocked,
                "blocked_submits": crawler.blocked_submits,
                "safe_mode": safe_mode,
            },
        )
    finally:
        await browser.close()
        shutil.rmtree(tmp, ignore_errors=True)


async def collect(target_id: str, seed_url: str, headless: bool = True,
                   safe_mode: bool = True) -> list[DimensionResult]:
    """Run the SFG crawl twice — mobile viewport then desktop viewport —
    each with its own fresh SFGStore and browser context, sequentially
    inside one async_playwright() session. Returns both DimensionResults.
    """
    results: list[DimensionResult] = []
    async with async_playwright() as pw:
        results.append(await _crawl_at_viewport(
            pw, seed_url, _MOBILE_VIEWPORT, "sfg_crawl_mobile",
            target_id, headless, safe_mode,
        ))
        results.append(await _crawl_at_viewport(
            pw, seed_url, _DESKTOP_VIEWPORT, "sfg_crawl_desktop",
            target_id, headless, safe_mode,
        ))
    return results


if __name__ == "__main__":
    import argparse
    import asyncio
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--seed-url", required=True)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--allow-full-flow", action="store_true")
    args = ap.parse_args()

    res = asyncio.run(collect(args.target_id, args.seed_url,
                              headless=not args.headed,
                              safe_mode=not args.allow_full_flow))
    print(json.dumps([
        {"method": r.method, "target_id": r.target_id, "paths": sorted(r.paths), "meta": r.meta}
        for r in res
    ], indent=2, ensure_ascii=False))
