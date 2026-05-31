"""
Sprint 6 gate measurement script.

Flow:
  1. ASTParser.parse_module() on each target → collect FunctionSpecs
  2. PytestGenerator.generate(specs) → GeneratedTest list
  3. CodeJudge.judge() on each → revision loop (max 2 attempts per test)
  4. Filter grade != "reject" → accepted tests
  5. TestExecutor.run() on accepted tests
  6. Write audit/sprint6/sprint6_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

from loguru import logger

from src.codetest.ast_parser import FunctionSpec, parse_module
from src.codetest.executor import ExecutionResult, TestExecutor
from src.codetest.generator import GeneratedTest, PytestGenerator
from src.codetest.judge import CodeJudge

_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint6_results.json"

TARGET_MODULES = [
    pathlib.Path("src/contractskill/sfg.py"),
    pathlib.Path("src/contractskill/crawler.py"),
    pathlib.Path("src/contractskill/compiler.py"),
    pathlib.Path("src/contractskill/repair.py"),
    pathlib.Path("src/explorer/hypothesis.py"),
]

_GATE_FUNCS_PARSED = 10
_GATE_TESTS_GENERATED = 10
_GATE_PASS_RATE = 0.70
_GATE_METAMORPHIC = 3
_REGRESSION_THRESHOLD = 0.75


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 6 gate measurement")
    parser.add_argument("--live", action="store_true", help="Run full live pipeline (default behavior)")
    parser.add_argument("--verbose", action="store_true", help="Print per-test PASS/FAIL results")
    parser.add_argument("--rerun-failed", action="store_true", help="Regenerate only failing tests from last run")
    return parser.parse_args()


async def _collect_specs() -> list[FunctionSpec]:
    # parse_module is synchronous; blocking is acceptable for the 5-module audit scope
    all_specs: list[FunctionSpec] = []
    for mod_path in TARGET_MODULES:
        if not mod_path.exists():
            logger.warning(f"measure_sprint6: module not found — {mod_path}")
            continue
        try:
            specs = parse_module(mod_path)
            logger.info(f"measure_sprint6: {mod_path} → {len(specs)} public functions")
            all_specs.extend(specs)
        except Exception as exc:
            logger.warning(f"measure_sprint6: parse failed for {mod_path}: {exc!r}")
    return all_specs


async def _collect_specs_for_modules(module_paths: set[str]) -> list[FunctionSpec]:
    """Parse only the specified module paths (by normalised path string)."""
    all_specs: list[FunctionSpec] = []
    for mod_path in TARGET_MODULES:
        norm = str(mod_path).replace("\\", "/")
        if not any(norm in fp.replace("\\", "/") or fp.replace("\\", "/") in norm for fp in module_paths):
            continue
        if not mod_path.exists():
            logger.warning(f"measure_sprint6: module not found — {mod_path}")
            continue
        try:
            specs = parse_module(mod_path)
            logger.info(f"measure_sprint6: {mod_path} → {len(specs)} public functions")
            all_specs.extend(specs)
        except Exception as exc:
            logger.warning(f"measure_sprint6: parse failed for {mod_path}: {exc!r}")
    return all_specs


async def _judge_with_revision(
    judge: CodeJudge,
    generator: PytestGenerator,
    specs_by_func_id: dict[str, FunctionSpec],
    tests: list[GeneratedTest],
) -> list[GeneratedTest]:
    """Judge each test. If needs_revision, regenerate once and re-judge."""
    accepted: list[GeneratedTest] = []
    for test in tests:
        result = await judge.judge(test)
        if result.grade == "acceptable":
            accepted.append(test)
            continue
        if result.grade == "reject":
            logger.debug(f"measure_sprint6: rejected test_id={test.test_id!r}: {result.feedback}")
            continue
        # needs_revision — one retry
        spec = specs_by_func_id.get(test.func_id)
        if spec is None:
            logger.warning(f"measure_sprint6: no spec for func_id={test.func_id!r}; dropping test")
            continue
        revised = await generator.regenerate_with_feedback(spec, test, result.feedback)
        if revised is None:
            continue
        result2 = await judge.judge(revised)
        if result2.grade == "acceptable":
            accepted.append(revised)
        else:
            logger.debug(
                f"measure_sprint6: revision still failed test_id={test.test_id!r}: {result2.feedback}"
            )
    return accepted


async def _run_tests(
    executor: TestExecutor,
    tests: list[GeneratedTest],
    verbose: bool = False,
) -> list[ExecutionResult]:
    results: list[ExecutionResult] = []
    for test in tests:
        result = await executor.run(test)
        status = "PASS" if result.passed else "FAIL"
        if verbose:
            error_snippet = (result.error_output or "").split("\n")[0][:120]
            print(f"{status} {test.test_id} | {error_snippet}")
        logger.info(f"measure_sprint6: test_id={test.test_id!r} passed={result.passed}")
        results.append(result)
    return results


def _write_results(
    ast_functions_parsed: int,
    tests_generated: int,
    test_pass_rate: float,
    metamorphic_pairs: int,
    regression: bool,
    sprint6_status: str,
    test_details: list[dict] | None = None,
) -> dict:
    gate_results: dict = {
        "ast_functions_parsed": ast_functions_parsed,
        "tests_generated": tests_generated,
        "test_pass_rate": round(test_pass_rate, 6),
        "metamorphic_pairs": metamorphic_pairs,
        "regression": regression,
        "sprint6_status": sprint6_status,
    }
    if test_details is not None:
        gate_results["test_details"] = test_details
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(gate_results, indent=2), encoding="utf-8")
    logger.info(f"measure_sprint6: results written to {_OUTPUT_PATH}")
    print(json.dumps({k: v for k, v in gate_results.items() if k != "test_details"}, indent=2))
    return gate_results


async def main(verbose: bool = False) -> dict:
    ast_functions_parsed = 0
    tests_generated = 0
    test_pass_rate = 0.0
    metamorphic_pairs = 0
    regression = True
    sprint6_status = "FAIL"
    test_details: list[dict] = []

    judge = CodeJudge()
    generator = PytestGenerator()
    executor = TestExecutor()

    try:
        # Step 1: Parse modules
        specs = await _collect_specs()
        ast_functions_parsed = len(specs)
        logger.info(f"measure_sprint6: total functions parsed = {ast_functions_parsed}")

        if not specs:
            logger.error("measure_sprint6: no functions parsed — aborting")
            return _write_results(
                ast_functions_parsed, tests_generated, test_pass_rate,
                metamorphic_pairs, regression, sprint6_status, test_details,
            )

        # Step 2: Generate tests
        raw_tests = await generator.generate(specs)
        logger.info(f"measure_sprint6: raw tests generated = {len(raw_tests)}")

        # Step 3: Judge with revision loop
        specs_by_func_id = {s.func_id: s for s in specs}
        accepted_tests = await _judge_with_revision(judge, generator, specs_by_func_id, raw_tests)
        tests_generated = len(accepted_tests)
        logger.info(f"measure_sprint6: accepted after judging = {tests_generated}")

        # Step 4: Execute accepted tests
        exec_results = await _run_tests(executor, accepted_tests, verbose=verbose)

        # Step 5: Compute gate metrics
        passed_count = sum(1 for r in exec_results if r.passed)
        total = len(exec_results)
        test_pass_rate = passed_count / total if total > 0 else 0.0

        metamorphic_pairs = sum(1 for t in accepted_tests if t.test_type == "metamorphic")

        # regression is only meaningful when tests were actually executed
        regression = total > 0 and test_pass_rate < _REGRESSION_THRESHOLD

        gates_pass = (
            ast_functions_parsed >= _GATE_FUNCS_PARSED
            and tests_generated >= _GATE_TESTS_GENERATED
            and test_pass_rate >= _GATE_PASS_RATE
            and metamorphic_pairs >= _GATE_METAMORPHIC
        )
        sprint6_status = "PASS" if gates_pass else "FAIL"

        # Build per-test details for JSON (used by --rerun-failed)
        test_details = [
            {
                "test_id": test.test_id,
                "func_id": test.func_id,
                "module_path": specs_by_func_id[test.func_id].module_path
                if test.func_id in specs_by_func_id else None,
                "passed": result.passed,
                "error_output": result.error_output,
            }
            for test, result in zip(accepted_tests, exec_results)
        ]

    except Exception as exc:
        logger.error(f"measure_sprint6: unhandled error — {exc!r}")

    finally:
        try:
            await judge.close()
        except Exception:
            pass

    return _write_results(
        ast_functions_parsed, tests_generated, test_pass_rate,
        metamorphic_pairs, regression, sprint6_status, test_details,
    )


async def rerun_failed(verbose: bool = False) -> dict:
    """Regenerate and re-execute only the tests that failed in the last run."""
    if not _OUTPUT_PATH.exists():
        print("sprint6_results.json not found — run without --rerun-failed first")
        return {}

    data = json.loads(_OUTPUT_PATH.read_text(encoding="utf-8"))
    test_details: list[dict] = data.get("test_details", [])

    if not test_details:
        print("No test_details in sprint6_results.json — run without --rerun-failed first to populate it")
        return data

    failed_details = [d for d in test_details if not d["passed"]]
    passed_count_before = sum(1 for d in test_details if d["passed"])
    total_count = len(test_details)

    if not failed_details:
        print("No failed tests — nothing to rerun")
        return data

    failed_func_ids: set[str] = {d["func_id"] for d in failed_details if d.get("func_id")}
    failed_module_paths: set[str] = {
        d["module_path"] for d in failed_details if d.get("module_path")
    }

    print(f"Rerunning {len(failed_details)} failed tests from modules: {sorted(failed_module_paths)}")

    judge = CodeJudge()
    generator = PytestGenerator()
    executor = TestExecutor()

    try:
        # Parse only modules that had failures
        all_specs = await _collect_specs_for_modules(failed_module_paths)

        # Filter to only the failing func_ids
        target_specs = [s for s in all_specs if s.func_id in failed_func_ids]

        if not target_specs:
            print(
                f"No specs matched for failed func_ids {failed_func_ids} — "
                "module paths may have changed; run full pipeline instead"
            )
            return data

        logger.info(f"measure_sprint6 --rerun-failed: regenerating {len(target_specs)} specs")

        raw_tests = await generator.generate(target_specs)
        specs_by_func_id = {s.func_id: s for s in target_specs}
        accepted_tests = await _judge_with_revision(judge, generator, specs_by_func_id, raw_tests)
        new_exec_results = await _run_tests(executor, accepted_tests, verbose=verbose)

        # Build new test_details for retried tests
        new_details_by_func_id: dict[str, dict] = {}
        for test, result in zip(accepted_tests, new_exec_results):
            spec = specs_by_func_id.get(test.func_id)
            new_details_by_func_id[test.func_id] = {
                "test_id": test.test_id,
                "func_id": test.func_id,
                "module_path": spec.module_path if spec else None,
                "passed": result.passed,
                "error_output": result.error_output,
            }

        # Merge: replace failed entries with new results, keep passing entries unchanged
        merged_details: list[dict] = []
        for d in test_details:
            func_id = d.get("func_id", "")
            if func_id in new_details_by_func_id:
                merged_details.append(new_details_by_func_id[func_id])
            else:
                merged_details.append(d)

        merged_passed = sum(1 for d in merged_details if d["passed"])
        test_pass_rate = merged_passed / total_count if total_count > 0 else 0.0

        metamorphic_pairs = data.get("metamorphic_pairs", 0)
        regression = total_count > 0 and test_pass_rate < _REGRESSION_THRESHOLD
        gates_pass = (
            data.get("ast_functions_parsed", 0) >= _GATE_FUNCS_PARSED
            and total_count >= _GATE_TESTS_GENERATED
            and test_pass_rate >= _GATE_PASS_RATE
            and metamorphic_pairs >= _GATE_METAMORPHIC
        )
        sprint6_status = "PASS" if gates_pass else "FAIL"

        return _write_results(
            data.get("ast_functions_parsed", 0),
            total_count,
            test_pass_rate,
            metamorphic_pairs,
            regression,
            sprint6_status,
            test_details=merged_details,
        )

    finally:
        try:
            await judge.close()
        except Exception:
            pass


if __name__ == "__main__":
    args = _parse_args()
    if args.rerun_failed:
        asyncio.run(rerun_failed(verbose=args.verbose))
    else:
        asyncio.run(main(verbose=args.verbose))
