#!/usr/bin/env python
"""Coverage cross-check driver.

Runs three independent dimensions on the same target and merges them via
eval/coverage_dimensions.merge_dimensions():
  1. sfg_crawl        — DOM-crawl state/path coverage (existing SFGTraversalExplorer)
  2. static_declared  — sitemap.xml / robots.txt / JS-bundle route grep (eval/dim_static.py)
  3. js_coverage      — real JS function-execution coverage via raw CDP (eval/dim_js_coverage.py)

This is the "necessary first" subset picked out of a longer menu of possible
cross-check dimensions (viewport A/B, keyboard-nav walk, monkey-walk,
role/session-state variation, error-injection, temporal re-runs — not built
yet). See eval/FINDINGS_explore.md and the coverage discussion in this
branch's history for why a single ground truth (human-curated or otherwise)
understates blind spots that agreement across independent methods catches.

Nothing here calls Ollama or an external LLM API.

Run:
  .venv/bin/python eval/run_coverage_crosscheck.py --target-id saucedemo_flow \\
      --seed-url https://www.saucedemo.com/ --out eval/coverage_saucedemo.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright  # noqa: E402

from src.contractskill.sfg import SFGStore  # noqa: E402
from src.universal_qa.explorer.sfg_traversal import SFGTraversalExplorer  # noqa: E402
from eval.coverage_dimensions import DimensionResult, merge_dimensions  # noqa: E402
from eval.dim_static import collect as collect_static  # noqa: E402
from eval.dim_js_coverage import JSCoverageSession  # noqa: E402

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


async def run_crosscheck(target_id: str, seed_url: str, headless: bool = True,
                         safe_mode: bool = True) -> dict:
    results: list[DimensionResult] = [await collect_static(target_id, seed_url)]

    tmp = Path(tempfile.mkdtemp(prefix="uqa_covcheck_"))
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(user_agent=_UA)
        page = await ctx.new_page()
        try:
            js_cov = await JSCoverageSession.start(ctx, page)
            await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)

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

            results.append(await js_cov.stop(target_id))
        finally:
            await browser.close()
            shutil.rmtree(tmp, ignore_errors=True)

    merged = merge_dimensions(results)
    merged["target_id"] = target_id
    merged["seed_url"] = seed_url
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--seed-url", required=True)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--allow-full-flow", action="store_true",
                    help="see eval/run_explore_eval.py --allow-full-flow; sandbox/test targets only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = asyncio.run(run_crosscheck(args.target_id, args.seed_url,
                                        headless=not args.headed,
                                        safe_mode=not args.allow_full_flow))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
