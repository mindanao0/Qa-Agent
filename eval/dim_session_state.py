"""Coverage dimension: session-state A/B.

Runs the *same* SFG crawl twice on the same target — once as a clean
"guest, first visit" session (nothing pre-set), once as a "returning user"
session with localStorage/cookie state seeded BEFORE the crawler's first
navigation — and reports each as its own paths-bearing DimensionResult.
Sites frequently render entirely different UI depending on this
(skip a welcome tour, show a cart badge/mini-cart, show a "welcome back"
banner) that a fresh-session crawl structurally cannot reach.

Zero LLM calls. Each session gets its own fresh SFGStore (separate temp
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

# Generic placeholder seed state. A real caller should pass site-specific
# keys (inspect the target's actual localStorage schema via devtools first)
# — these generic guesses (a cart array, a "seen the onboarding" flag) are
# plausible examples, not a schema any given site is guaranteed to use.
_DEFAULT_SEED_STATE = {
    "cart": '[{"id":"1","qty":2}]',
    "visited_before": "true",
}


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


async def _seed_local_storage(page, seed_url: str, seed_state: dict[str, str]) -> None:
    """Navigate to the origin (required — localStorage is origin-scoped and
    can't be set before a visit), inject each key via window.localStorage,
    then reload so the crawler's very first recorded state already reflects
    the seeded values.
    """
    await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)
    for key, value in seed_state.items():
        await page.evaluate(
            "([k, v]) => window.localStorage.setItem(k, v)", [key, value]
        )
    await page.reload(wait_until="domcontentloaded", timeout=30_000)


async def _crawl(pw, seed_url: str, method: str, target_id: str, headless: bool,
                  safe_mode: bool, seed_state: dict[str, str] | None) -> DimensionResult:
    tmp = Path(tempfile.mkdtemp(prefix=f"uqa_covcheck_{method}_"))
    browser = await pw.chromium.launch(headless=headless)
    try:
        ctx = await browser.new_context(user_agent=_UA)
        page = await ctx.new_page()

        if seed_state:
            await _seed_local_storage(page, seed_url, seed_state)
        else:
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
                "seeded": bool(seed_state),
                "seed_state": dict(seed_state) if seed_state else {},
                "state_count": store.node_count(),
                "blocked": crawler.blocked,
                "blocked_submits": crawler.blocked_submits,
                "safe_mode": safe_mode,
            },
        )
    finally:
        await browser.close()
        shutil.rmtree(tmp, ignore_errors=True)


async def collect(target_id: str, seed_url: str, seed_state: dict[str, str] | None = None,
                   headless: bool = True, safe_mode: bool = True) -> list[DimensionResult]:
    """Run the SFG crawl twice — clean guest session, then a session with
    localStorage pre-seeded — each with its own fresh SFGStore and browser
    context. Returns both DimensionResults ("sfg_crawl_guest" and
    "sfg_crawl_seeded_state").

    `seed_state` is a plain dict of localStorage key -> value pairs injected
    before the seeded run. If omitted, a generic example (a small cart +
    a "visited before" flag) is used — a REAL caller should pass
    site-specific keys inspected from the actual target, since generic
    guesses may not match any given site's real storage schema and the
    seeded run would then be indistinguishable from the guest run.
    """
    effective_seed = dict(seed_state) if seed_state is not None else dict(_DEFAULT_SEED_STATE)

    results: list[DimensionResult] = []
    async with async_playwright() as pw:
        results.append(await _crawl(
            pw, seed_url, "sfg_crawl_guest", target_id, headless, safe_mode,
            seed_state=None,
        ))
        results.append(await _crawl(
            pw, seed_url, "sfg_crawl_seeded_state", target_id, headless, safe_mode,
            seed_state=effective_seed,
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
