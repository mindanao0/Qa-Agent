"""
Sprint 9 gate measurement.

Gates:
  js_functions_parsed   >= 10
  tests_generated       >= 10
  test_pass_rate        >= 0.70
  metamorphic_pairs     >= 3
  otel_spans_emitted    >= 5
  REGRESSION if pass_rate < 0.75
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import sys

from loguru import logger

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint9_results.json"
_JS_TARGETS_DIR = _PROJECT_ROOT / "src" / "js_targets"
_WORK_DIR = pathlib.Path(__file__).parent / "js_tests"

_GATE_FUNCS = 10
_GATE_TESTS = 10
_GATE_PASS_RATE = 0.70
_GATE_META = 3
_GATE_OTEL = 5
_REGRESSION_THRESHOLD = 0.75


async def _run_pipeline(tracer):
    from src.codetest.js_ast_parser import JSASTParser
    from src.codetest.js_generator import JSTestGenerator
    from src.codetest.js_judge import JSCodeJudge
    from src.codetest.js_executor import JSTestExecutor
    from src.observability.audit_chain import CryptoAuditTrail

    audit_trail = CryptoAuditTrail(
        path=pathlib.Path.home() / ".qa-agent" / "sprint9_audit.jsonl"
    )

    if _WORK_DIR.exists():
        shutil.rmtree(_WORK_DIR)
    _WORK_DIR.mkdir(parents=True)

    parser = JSASTParser()
    generator = JSTestGenerator()
    judge = JSCodeJudge()
    executor = JSTestExecutor(project_root=_PROJECT_ROOT)

    # Stage 1: parse
    async with tracer.span("js.parse", dir=str(_JS_TARGETS_DIR)):
        specs = await parser.parse_dir(_JS_TARGETS_DIR, glob="*.ts")
    logger.info(f"measure_sprint9: parsed {len(specs)} JS function specs")
    audit_trail.append("js_parse_complete", {"count": len(specs)})

    # Stage 2: generate
    async with tracer.span("js.generate", spec_count=len(specs)):
        tests = await generator.generate(specs)
    logger.info(f"measure_sprint9: generated {len(tests)} tests")
    audit_trail.append("js_generate_complete", {"count": len(tests)})

    # Stage 3: judge
    accepted = []
    async with tracer.span("js.judge", test_count=len(tests)):
        for test in tests:
            result = await judge.judge(test)
            if result.grade != "reject":
                accepted.append(test)
    await judge.close()
    logger.info(f"measure_sprint9: accepted {len(accepted)}/{len(tests)} tests")
    audit_trail.append("js_judge_complete", {"accepted": len(accepted)})

    # Stage 4: execute
    async with tracer.span("js.execute", accepted_count=len(accepted)):
        exec_results = await executor.run(accepted, _WORK_DIR)
    passed = sum(1 for r in exec_results if r.passed)
    audit_trail.append("js_execute_complete", {"passed": passed, "total": len(exec_results)})

    # Stage 5: report span
    async with tracer.span("js.report"):
        metamorphic_pairs = sum(1 for t in accepted if t.test_type == "metamorphic")
        pass_rate = passed / len(exec_results) if exec_results else 0.0

    return {
        "js_functions_parsed": len(specs),
        "tests_generated": len(tests),
        "test_pass_rate": round(pass_rate, 4),
        "metamorphic_pairs": metamorphic_pairs,
        "accepted_tests": len(accepted),
        "passed_tests": passed,
    }


async def main() -> None:
    from src.observability.tracer import OTelTracer

    tracer = OTelTracer()

    try:
        metrics = await _run_pipeline(tracer)
    except Exception as exc:
        logger.error(f"measure_sprint9: pipeline error: {exc!r}")
        metrics = {
            "js_functions_parsed": 0,
            "tests_generated": 0,
            "test_pass_rate": 0.0,
            "metamorphic_pairs": 0,
            "accepted_tests": 0,
            "passed_tests": 0,
        }

    otel_spans = tracer.flush()
    regression = metrics["test_pass_rate"] < _REGRESSION_THRESHOLD

    sprint9_pass = (
        metrics["js_functions_parsed"] >= _GATE_FUNCS
        and metrics["tests_generated"] >= _GATE_TESTS
        and metrics["test_pass_rate"] >= _GATE_PASS_RATE
        and metrics["metamorphic_pairs"] >= _GATE_META
        and otel_spans >= _GATE_OTEL
        and not regression
    )

    results = {
        "js_functions_parsed": metrics["js_functions_parsed"],
        "tests_generated": metrics["tests_generated"],
        "test_pass_rate": metrics["test_pass_rate"],
        "metamorphic_pairs": metrics["metamorphic_pairs"],
        "otel_spans_emitted": otel_spans,
        "regression": regression,
        "sprint9_status": "PASS" if sprint9_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Sprint 9 Results ===")
    gates = {
        "js_functions_parsed": _GATE_FUNCS,
        "tests_generated": _GATE_TESTS,
        "test_pass_rate": _GATE_PASS_RATE,
        "metamorphic_pairs": _GATE_META,
        "otel_spans_emitted": _GATE_OTEL,
    }
    for k, v in results.items():
        gate_str = f" (gate >= {gates[k]})" if k in gates else ""
        print(f"  {k}: {v}{gate_str}")
    print(f"\n  -> sprint9_results.json written to {_OUTPUT_PATH}")
    sys.exit(0 if sprint9_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
