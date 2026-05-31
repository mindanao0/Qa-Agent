# audit/phase0/measure_sprint1_day2.py
"""
Sprint 1 Day 1-2 A/B measurement script.

Runs the pilot golden dataset (5 passing + 5 failing) through both engines
and writes audit/phase0/sprint1_day2_results.json.

Usage:
    uv run python audit/phase0/measure_sprint1_day2.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger

from src.llm.adapter import OllamaAdapter
from src.llm.schemas import TestPlan

PASSING_JSONL = ROOT / "audit" / "phase0" / "golden_dataset" / "passing.jsonl"
FAILING_JSONL = ROOT / "audit" / "phase0" / "golden_dataset" / "failing.jsonl"
OUTPUT_PATH = ROOT / "audit" / "phase0" / "sprint1_day2_results.json"
BASELINE_PATH = ROOT / "audit" / "phase0" / "baseline_metrics.json"


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


async def run_generation(
    requirement: str,
    url: str,
    engine: str,
    adapter: OllamaAdapter,
) -> dict:
    """Run a single generation for one dataset entry."""
    os.environ["STRUCTURED_OUTPUT_ENGINE"] = engine

    from src import config_loader
    config_loader._load_yaml.cache_clear()

    from src.agents.planner import PlannerAgent
    from src.agents.generator import GeneratorAgent

    start_ms = time.monotonic() * 1000
    result: dict = {
        "engine": engine,
        "requirement": requirement[:80],
        "url": url,
        "plan_success": False,
        "code_success": False,
        "success": False,
        "latency_ms": 0.0,
        "json_parse_failed": False,
        "error": None,
    }

    try:
        planner = PlannerAgent(adapter=adapter)
        plan = await planner.plan(requirement=requirement, url=url)
        result["plan_success"] = isinstance(plan, TestPlan)

        if result["plan_success"]:
            generator = GeneratorAgent(adapter=adapter)
            script = await generator.generate(test_plan=plan, url=url)
            import ast
            ast.parse(script.code)
            result["code_success"] = True
            result["success"] = True

    except Exception as exc:
        result["success"] = False
        result["json_parse_failed"] = True
        result["error"] = str(exc)[:200]
        logger.warning(f"[{engine}] Failed for '{requirement[:40]}': {exc}")

    result["latency_ms"] = time.monotonic() * 1000 - start_ms
    return result


async def measure_engine(engine: str, cases: list[dict], adapter: OllamaAdapter) -> dict:
    """Run all cases through one engine and aggregate metrics."""
    results = []
    for case in cases:
        r = await run_generation(
            requirement=case["requirement_text"],
            url=case["target_url"],
            engine=engine,
            adapter=adapter,
        )
        results.append(r)
        logger.info(
            f"[{engine}] {case['case_id']} | "
            f"success={r['success']} latency={r['latency_ms']:.0f}ms"
        )

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed_parse = sum(1 for r in results if r["json_parse_failed"])
    latencies = [r["latency_ms"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    metrics: dict = {
        "json_parse_failure_rate": failed_parse / total if total else 0.0,
        "first_run_pass_rate": passed / total if total else 0.0,
        "avg_generation_latency_ms": int(avg_latency),
        "avg_retry_count": 0.0,
        "validation_errors_total": failed_parse,
        "_raw_results": results,
    }

    if engine == "legacy_repair":
        metrics["repair_stage_hits"] = {"stage1": 0, "stage2": 0, "stage3": 0}

    return metrics


def compute_verdict(
    instructor_metrics: dict, legacy_metrics: dict, baseline: dict
) -> str:
    baseline_pass_rate = baseline.get("execution_metrics", {}).get(
        "first_run_pass_rate", 0.286
    )
    inst = instructor_metrics

    if inst["first_run_pass_rate"] < baseline_pass_rate:
        return "REGRESSION"

    if (
        inst["json_parse_failure_rate"] <= 0.05
        and inst["first_run_pass_rate"] >= 0.40
        and inst["avg_generation_latency_ms"]
        <= 1.20 * legacy_metrics["avg_generation_latency_ms"]
    ):
        return "PASS"

    return "FAIL"


async def main() -> None:
    logger.info("Sprint 1 Day 1-2 measurement starting...")

    passing = load_jsonl(PASSING_JSONL)
    failing = load_jsonl(FAILING_JSONL)
    all_cases = passing + failing
    logger.info(f"Dataset: {len(all_cases)} cases ({len(passing)} passing, {len(failing)} failing)")

    baseline: dict = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    adapter = OllamaAdapter()
    try:
        logger.info("=== Measuring instructor engine ===")
        instructor_metrics = await measure_engine("instructor", all_cases, adapter)

        logger.info("=== Measuring legacy_repair engine ===")
        legacy_metrics = await measure_engine("legacy_repair", all_cases, adapter)
    finally:
        await adapter.close()

    inst_pass = instructor_metrics["first_run_pass_rate"]
    leg_pass = legacy_metrics["first_run_pass_rate"]
    inst_lat = instructor_metrics["avg_generation_latency_ms"]
    leg_lat = legacy_metrics["avg_generation_latency_ms"]
    inst_fail_rate = instructor_metrics["json_parse_failure_rate"]
    leg_fail_rate = legacy_metrics["json_parse_failure_rate"]

    delta_pass_pct = (
        ((inst_pass - leg_pass) / leg_pass * 100) if leg_pass > 0 else 0.0
    )
    delta_lat_pct = (
        ((inst_lat - leg_lat) / leg_lat * 100) if leg_lat > 0 else 0.0
    )
    delta_fail_rate_reduction_pct = (
        ((leg_fail_rate - inst_fail_rate) / leg_fail_rate * 100)
        if leg_fail_rate > 0
        else 0.0
    )

    verdict = compute_verdict(instructor_metrics, legacy_metrics, baseline)

    instructor_out = {k: v for k, v in instructor_metrics.items() if k != "_raw_results"}
    legacy_out = {k: v for k, v in legacy_metrics.items() if k != "_raw_results"}

    output = {
        "measured_at_iso": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(all_cases),
        "instructor": instructor_out,
        "legacy_repair": legacy_out,
        "delta": {
            "json_parse_failure_rate_reduction_pct": round(delta_fail_rate_reduction_pct, 2),
            "first_run_pass_rate_improvement_pct": round(delta_pass_pct, 2),
            "latency_change_pct": round(delta_lat_pct, 2),
        },
        "verdict": verdict,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results written to {OUTPUT_PATH}")
    logger.info(f"VERDICT: {verdict}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
