#!/usr/bin/env python
"""Exploration eval harness — measures the explorer on eval/explore_golden.jsonl.

Drives the REAL pipeline per target (discover -> explore, exactly as
UniversalQAAgent does) against live sites, then reports 3 metrics:

  states_discovered   distinct SFG nodes the pipeline builds (node_id =
                      sha256(url_path, grounded-AOM-hash)). Baseline: only the
                      discovery crawl populates the persistent SFG; the current
                      explorer does NOT add post-click/modal/SPA states — that's
                      the bug E1 fixes. Reported alongside diagnostic explorer
                      page / state-change counts.
  reachable_coverage  expected_reachable_pages reached (by path-suffix match)
                      over {discovery URLs} ∪ {explored page URLs}.
  dedup_precision     revisit probe: ground a sample page twice via the crawler;
                      precision = fraction whose node_id is identical both times
                      (i.e. a re-seen state is NOT counted as new). Guards E1's
                      fuzzy dedup from exploding false states.

Live + real Ollama (grounding/form-fill). Nothing mocked.

Run:
  OLLAMA_BASE_URL=http://127.0.0.1:11434 .venv/bin/python eval/run_explore_eval.py \
    --label baseline --out eval/explore_baseline.json [--limit 1] [--only saucedemo_flow]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402

from src.contractskill.crawler import CrawlerConfig, SFGCrawler  # noqa: E402
from src.contractskill.sfg import SFGStore  # noqa: E402
from src.perception.grounder import Grounder  # noqa: E402
from src.universal_qa.auth_manager import AuthManager  # noqa: E402
from src.universal_qa.explorer.nav_map import ExplorerConfig  # noqa: E402
from src.universal_qa.explorer.site_explorer import SiteExplorer  # noqa: E402
from src.universal_qa.site_discovery import SiteDiscovery  # noqa: E402
from src.universal_qa.explorer.sfg_traversal import SFGTraversalExplorer  # noqa: E402

_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "Chrome/120.0.0.0 Safari/537.36")


def _path(u: str) -> str:
    sp = urlsplit(u)
    p = (sp.path or "/").rstrip("/") or "/"
    return p + (("#" + sp.fragment) if sp.fragment else "")


def _reached(expected: str, paths: set[str]) -> bool:
    e = expected.rstrip("/") or "/"
    return any(e == p or p.endswith(e) or e in p for p in paths)


async def _dedup_probe(page, urls: list[str]) -> tuple[float | None, int]:
    """Ground each url twice; precision = fraction with identical node_id."""
    if not urls:
        return None, 0
    tmp = Path(tempfile.mkdtemp(prefix="uqa_dedup_"))
    store = SFGStore(db_path=tmp / "probe.db")
    crawler = SFGCrawler(store, Grounder(), CrawlerConfig())
    stable = total = 0
    try:
        for u in urls:
            try:
                await page.goto(u, wait_until="domcontentloaded", timeout=20_000)
                await page.wait_for_timeout(600)
                n1, _ = await crawler._visit_node(page, None)
                await page.goto(u, wait_until="domcontentloaded", timeout=20_000)
                await page.wait_for_timeout(600)
                n2, _ = await crawler._visit_node(page, None)
                total += 1
                stable += int(n1.node_id == n2.node_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"dedup probe {u} skipped: {exc!r}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return (stable / total if total else None), total


async def eval_target(t: dict, mode: str = "baseline", headless: bool = True) -> dict:
    res: dict = {"target_id": t["target_id"], "seed_url": t["seed_url"],
                 "expected_states_min": t["expected_states_min"]}
    t0 = time.monotonic()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
        ctx = await browser.new_context(user_agent=_UA)
        page = await ctx.new_page()
        try:
            await page.goto(t["seed_url"], wait_until="domcontentloaded", timeout=30_000)
            auth = AuthManager(username=t.get("username"), password=t.get("password"))
            try:
                await auth.setup(page)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{t['target_id']}: auth failed {exc!r}")
            await page.wait_for_timeout(1_500)

            base = f"{urlsplit(page.url).scheme}://{urlsplit(page.url).netloc}"
            seed = page.url

            if mode == "sfg":
                store = SFGStore(db_path=Path(tempfile.mkdtemp(prefix="uqa_sfg_")) / "sfg.db")
                crawler = SFGTraversalExplorer(store, max_states=150, time_budget_s=180)
                await crawler.explore(page, seed)
                states_discovered = store.node_count()
                node_urls = [n.url for n in store.get_nodes_by_url_prefix(base)]
                all_paths = {_path(u) for u in node_urls}
                # dedup probe with the crawler's OWN signature (re-seen state -> same id)
                stable = total = 0
                for u in (node_urls[:3] or [seed]):
                    try:
                        await page.goto(u, wait_until="domcontentloaded", timeout=20_000)
                        await page.wait_for_timeout(500)
                        h1 = await crawler.signature_hash(page)
                        await page.goto(u, wait_until="domcontentloaded", timeout=20_000)
                        await page.wait_for_timeout(500)
                        h2 = await crawler.signature_hash(page)
                        total += 1
                        stable += int(h1 == h2)
                    except Exception:  # noqa: BLE001
                        pass
                dedup_precision = round(stable / total, 4) if total else None
                n_probe = total
                diag_extra = {
                    "edges": store.edge_count(), "clicks": crawler.click_attempts,
                    "click_fail": crawler.click_fail, "healed_l1": crawler.healed_l1,
                    "blocked": crawler.blocked, "dedup_hits": crawler.merges,
                    "observations": crawler.observations,
                }
            else:
                disc = SiteDiscovery(max_pages=25)
                sfg = await disc.discover(page, page.url)
                disc_urls = [n.url for n in sfg.get_nodes_by_url_prefix(base)]
                states_discovered = sfg.node_count()
                cfg = ExplorerConfig(max_pages=25, explore_timeout_min=3, max_depth=4)
                from src.llm.instructor_client import InstructorClient
                explorer = SiteExplorer(auth=auth, config=cfg, client=InstructorClient())
                nav = await explorer.explore(page, disc_urls or [page.url])
                explored_urls = [p.url for p in nav.pages]
                all_paths = {_path(u) for u in (disc_urls + explored_urls)}
                probe_urls = (disc_urls or [page.url])[:3]
                dedup_precision, n_probe = await _dedup_probe(page, probe_urls)
                diag_extra = {
                    "discovery_urls": len(disc_urls), "explored_pages": len(nav.pages),
                    "explorer_state_changes": sum(1 for p in nav.pages for a in p.actions if a.state_change),
                    "flows": len(nav.flows),
                }

            expected = t["expected_reachable_pages"]
            hits = [e for e in expected if _reached(e, all_paths)]
            reachable_coverage = round(len(hits) / len(expected), 4) if expected else None
            res.update({
                "states_discovered": states_discovered,
                "reachable_coverage": reachable_coverage,
                "dedup_precision": dedup_precision,
                "diagnostics": {**diag_extra, "expected_reachable": len(expected),
                                "reached": hits, "dedup_probe_n": n_probe,
                                "elapsed_s": round(time.monotonic() - t0, 1)},
            })
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{t['target_id']} FAILED: {exc!r}")
            res["error"] = repr(exc)[:300]
        finally:
            await browser.close()
    return res


async def run(golden: Path, label: str, mode: str, limit: int | None, only: str | None,
              headless: bool) -> dict:
    targets = [json.loads(l) for l in golden.read_text(encoding="utf-8").splitlines() if l.strip()]
    if only:
        targets = [t for t in targets if t["target_id"] == only]
    if limit:
        targets = targets[:limit]

    per = []
    for i, t in enumerate(targets, 1):
        logger.info(f"[{i}/{len(targets)}] {t['target_id']} — {t['seed_url']} (mode={mode})")
        per.append(await eval_target(t, mode=mode, headless=headless))

    ok = [p for p in per if "states_discovered" in p]
    def _avg(key):
        vals = [p[key] for p in ok if p.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    summary = {
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(per),
        "metrics": {
            "states_discovered_total": sum(p.get("states_discovered", 0) for p in ok),
            "states_discovered_avg": _avg("states_discovered"),
            "reachable_coverage": _avg("reachable_coverage"),
            "dedup_precision": _avg("dedup_precision"),
        },
        "per_target": per,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "eval" / "explore_golden.jsonl"))
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--mode", default="baseline", choices=["baseline", "sfg"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    summary = asyncio.run(run(Path(args.golden), args.label, args.mode, args.limit,
                              args.only, headless=not args.headed))
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    for p in summary["per_target"]:
        sd, rc, dp = p.get("states_discovered"), p.get("reachable_coverage"), p.get("dedup_precision")
        print(f"  {p['target_id']:24s} states={sd} reach={rc} dedup={dp} {p.get('error','')}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
