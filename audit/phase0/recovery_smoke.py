#!/usr/bin/env python3
"""
recovery_smoke.py — Sprint 1 Recovery smoke test.

Runs Grounder against 10 golden dataset URLs and records source distribution.
Gate: at least 7/10 must NOT be source='failure' or 'exception'.

Usage:
    uv run python audit/phase0/recovery_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_GOLDEN_DIR = Path(__file__).parent / "golden_dataset"
_SMOKE_OUT = Path(__file__).parent / "recovery_smoke.json"


async def smoke_test() -> int:
    """Returns failure_count. Non-zero means gate failed."""
    from playwright.async_api import async_playwright
    from src.perception.grounder import Grounder

    # Collect up to 10 URLs from golden dataset (5 passing + 5 failing)
    urls: list[str] = []
    for fname in ("passing.jsonl", "failing.jsonl"):
        fpath = _GOLDEN_DIR / fname
        if fpath.exists():
            for line in fpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    url = rec.get("target_url", "") or rec.get("url", "")
                    if url and url not in urls:
                        urls.append(url)
                except Exception:
                    pass
            if len(urls) >= 10:
                break
    urls = urls[:10]

    print(f"Smoke test: {len(urls)} URLs")
    results = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        for url in urls:
            page = await browser.new_page()
            try:
                print(f"  → {url[:70]}", end=" ", flush=True)
                await page.goto(url, timeout=25_000)
                grounder = Grounder()
                pam = await grounder.ground(page, context_budget_tokens=1000)
                rec = {
                    "url": url,
                    "grounder_source": pam.source,
                    "estimated_tokens": pam.estimated_tokens,
                    "errors": [],
                }
                print(f"source={pam.source} tokens={pam.estimated_tokens}")
            except Exception as exc:
                rec = {
                    "url": url,
                    "grounder_source": "exception",
                    "estimated_tokens": 0,
                    "errors": [str(exc)[:200]],
                }
                print(f"EXCEPTION: {str(exc)[:80]}")
            finally:
                await page.close()
            results.append(rec)
        await browser.close()

    # Write results
    smoke_data = {
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(results),
        "results": results,
    }
    _SMOKE_OUT.write_text(json.dumps(smoke_data, indent=2), encoding="utf-8")
    print(f"\nSmoke results written to {_SMOKE_OUT}")

    # Gate check
    non_failure = [r for r in results if r["grounder_source"] not in ("failure", "exception")]
    failure_count = len(results) - len(non_failure)
    failure_pct = failure_count / len(results) * 100 if results else 100

    print(f"\n{'='*60}")
    print(f"Gate check: {len(non_failure)}/{len(results)} non-failure")
    print(f"  Sources: {[r['grounder_source'] for r in results]}")

    if failure_count >= 3:
        print(f"\n⚠️  GATE FAILED: {failure_count}/10 still source='failure' ({failure_pct:.0f}%)")
        print("Investigate grounder_metrics.jsonl for error patterns.")
        print("Do NOT proceed to full measurement.")
        return failure_count
    else:
        print(f"\n✅ GATE PASSED: only {failure_count}/10 failures. Proceed to full measurement.")
        return 0


if __name__ == "__main__":
    failure_count = asyncio.run(smoke_test())
    sys.exit(1 if failure_count >= 3 else 0)
