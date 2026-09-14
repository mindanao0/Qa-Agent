#!/usr/bin/env python
"""Coverage cross-check CLI.

Core logic now lives in src/universal_qa/coverage/crosscheck.py — this
script is a thin wrapper so the same cross-check the production agent can
run internally (UniversalQAAgent(enable_coverage_crosscheck=True)) is also
reachable standalone for ad-hoc auditing. See eval/FINDINGS_explore.md for
why agreement across independent dimensions is the signal, not any single
dimension (human-curated golden dataset included).

Two report sizes:
  --full   all 9 dimensions (static, sfg_crawl, js_coverage, viewport A/B,
           session-state A/B, keyboard-walk, monkey-walk, api_spec)
  (default) the original "necessary first" 3-dimension subset (static,
           sfg_crawl, js_coverage) — kept for backward compatibility with
           existing eval/coverage_*.json reports.

Run:
  .venv/bin/python eval/run_coverage_crosscheck.py --target-id saucedemo_flow \\
      --seed-url https://www.saucedemo.com/ --out eval/coverage_saucedemo.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.universal_qa.coverage.crosscheck import run_crosscheck, run_full_crosscheck  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--seed-url", required=True)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="run all 9 coverage dimensions instead of the original 3")
    ap.add_argument("--allow-full-flow", action="store_true",
                    help="see eval/run_explore_eval.py --allow-full-flow; sandbox/test targets only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runner = run_full_crosscheck if args.full else run_crosscheck
    report = asyncio.run(runner(args.target_id, args.seed_url,
                                headless=not args.headed,
                                safe_mode=not args.allow_full_flow))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
