"""Continuous Mode — CLI entry point.

    python -m src.continuous --url <URL> [--max-cycles N] [--profile todomvc|generic]

``ContinuousLoopController`` (``src/continuous/loop_controller.py``, Sprint 11) was
built, measured, and gated by ``audit/sprint11/measure_sprint11.py`` — but until now
the only way to actually run it was that hardcoded audit script (fixed demo URL,
writes into the frozen ``audit/sprint11/`` gate artifacts) or importing the class by
hand. This module is the real, user-invocable entry point: same controller,
any target URL, per-run artifacts kept out of the audit trail.

Profiles
--------
todomvc   (default) Drives ``ContinuousLoopController`` for real: N cycles of
          persistent-BrowserContext exploration -> web + code test generation ->
          execute -> self-heal -> LangGraph ``AsyncSqliteSaver`` checkpoint, stopping
          on ``StopConditionEvaluator`` (max-cycles, then coverage-plateau, then
          memory-limit — see ``src/continuous/stop_conditions.py``).
          CAVEAT: the loop's per-cycle action plan (``_CYCLE_PLANS`` in
          loop_controller.py) is written against the TodoMVC demo's DOM
          (``get_by_placeholder("What needs to be done?")``, a "Clear completed"
          button, etc.). Point ``--url`` at a TodoMVC-shaped app, or use
          ``--profile generic`` for an arbitrary site.
generic   Delegates to ``UniversalQAAgent`` (``src/universal_qa/agent.py``) for one
          real exploration + test-generation + execution pass against ANY site.
          This is honestly NOT the LangGraph cycle loop — no persistent-context
          deepening across cycles, no checkpointing, no self-heal accounting — it
          is the fallback for real target URLs until the per-cycle action plan is
          generalized past TodoMVC. ``--max-cycles`` does not apply to it.

Per-run artifacts (LangGraph checkpoint db, SFG store, episodic memory, crypto
audit trail) are written under ``reports/continuous/<run-id>/`` by default — never
into ``audit/sprint11/``, which stays the frozen Sprint 11 gate proof and must not
be mutated by real user runs.

Examples
--------
    python -m src.continuous --url https://demo.playwright.dev/todomvc/#/
    python -m src.continuous --url https://demo.playwright.dev/todomvc/#/ --max-cycles 5
    python -m src.continuous --url https://example.com --profile generic --max-pages 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import uuid

from loguru import logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.continuous",
        description="Continuous Mode — multi-cycle explore/generate/execute/heal loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url", required=True, help="Target application URL")
    parser.add_argument(
        "--max-cycles", type=int, default=10,
        help="[todomvc profile] Stop after this many cycles, 0 = infinite. "
             "StopConditionEvaluator's own default (see ContinuousLoopController); "
             "coverage-plateau and memory-limit can still stop the loop earlier "
             "(default: 10)",
    )
    parser.add_argument(
        "--profile", default="todomvc", choices=["todomvc", "generic"],
        help="'todomvc': the real ContinuousLoopController cycle loop (action plan "
             "is TodoMVC-shaped). 'generic': single-pass UniversalQAAgent run "
             "against any site (default: todomvc)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run id / LangGraph checkpoint thread_id (default: random). Re-running "
             "with the same --run-id and --output-dir resumes from its checkpoint.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for this run's checkpoint/SFG/audit/episodic artifacts "
             "(default: reports/continuous/<run-id>/)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="[generic profile] Max pages to crawl (default: 50)",
    )
    return parser


async def _run_todomvc(args: argparse.Namespace, run_id: str, out_dir: pathlib.Path) -> dict:
    from src.continuous.loop_controller import ContinuousLoopController

    controller = ContinuousLoopController(
        args.url,
        max_cycles=args.max_cycles,
        run_id=run_id,
        sfg_db_path=out_dir / "sfg.db",
        audit_path=out_dir / "audit.jsonl",
        checkpoint_path=out_dir / "checkpoints.db",
        episodic_db_path=out_dir / "episodic",
    )
    state = await controller.run()
    return {
        "profile": "todomvc",
        "run_id": state.get("run_id", run_id),
        "cycles_completed": state.get("cycle", 0),
        "tests_generated": len(state.get("tests_generated", [])),
        "tests_passed": len(state.get("tests_passed", [])),
        "tests_failed": len(state.get("tests_failed", [])),
        "stop_reason": state.get("stop_reason"),
        "otel_spans_emitted": controller.otel_spans_emitted,
    }


async def _run_generic(args: argparse.Namespace) -> dict:
    from src.universal_qa.agent import UniversalQAAgent

    agent = UniversalQAAgent(args.url, max_pages=args.max_pages)
    results = await agent.run()
    passed = sum(1 for r in results if r.passed)
    return {
        "profile": "generic",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
    }


async def _main() -> int:
    args = _build_parser().parse_args()
    run_id = args.run_id or uuid.uuid4().hex
    out_dir = (
        pathlib.Path(args.output_dir) if args.output_dir
        else pathlib.Path("reports/continuous") / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.profile == "generic":
        summary = await _run_generic(args)
    else:
        summary = await _run_todomvc(args, run_id, out_dir)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"continuous mode: artifacts written to {out_dir}")

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {out_dir}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
