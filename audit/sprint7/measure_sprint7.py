# audit/sprint7/measure_sprint7.py
"""
Sprint 7 gate measurement script.

Measures:
  shadow_dom_elements_found  — ShadowDOMExtractor on the-internet.herokuapp.com/shadowdom
  spa_transitions_handled    — SPARouteTracker on demo.playwright.dev/todomvc/#/
  test_pass_rate             — PytestGenerator + TestExecutor on Sprint 7 source modules
  self_heal_triggered        — RepairEngine.repair() invoked at least once

Gates (PASS requires all four):
  shadow_dom_elements_found  ≥ 5
  spa_transitions_handled    ≥ 3
  test_pass_rate             ≥ 0.75
  self_heal_triggered        ≥ 1
  REGRESSION if test_pass_rate < 0.75
"""
from __future__ import annotations

import asyncio
import datetime
import json
import pathlib
import sys

from loguru import logger
from playwright.async_api import async_playwright

from src.codetest.ast_parser import parse_module
from src.codetest.executor import ExecutionResult, TestExecutor
from src.codetest.generator import GeneratedTest, PytestGenerator
from src.codetest.judge import CodeJudge
from src.contractskill.compiler import ContractSkill, ContractStep
from src.contractskill.repair import RepairEngine
from src.contractskill.sfg import SFGStore
from src.llm.instructor_client import InstructorClient
from src.shadow.extractor import ShadowDOMExtractor
from src.spa.hydration_guard import HydrationGuard
from src.spa.route_tracker import SPARouteTracker

_SPA_URL = "https://demo.playwright.dev/todomvc/#/"
_SHADOW_URL = "https://the-internet.herokuapp.com/shadowdom"
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint7_results.json"
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

_TARGET_MODULES = [
    _PROJECT_ROOT / "src/shadow/locator_builder.py",
    _PROJECT_ROOT / "src/shadow/extractor.py",
    _PROJECT_ROOT / "src/spa/route_tracker.py",
    _PROJECT_ROOT / "src/spa/hydration_guard.py",
]

_GATE_SHADOW_ELEMENTS = 5
_GATE_SPA_TRANSITIONS = 3
_GATE_PASS_RATE = 0.75
_GATE_SELF_HEAL = 1
_REGRESSION_THRESHOLD = 0.75


async def _measure_shadow_dom(page) -> int:
    extractor = ShadowDOMExtractor()
    guard = HydrationGuard()
    try:
        await page.goto(_SHADOW_URL, timeout=30000)
        await guard.wait_stable(page, timeout_ms=8000)
        nodes = await extractor.extract(page)
        logger.info(f"measure_sprint7: shadow DOM nodes found = {len(nodes)}")
        return len(nodes)
    except Exception as exc:
        logger.error(f"measure_sprint7._measure_shadow_dom: {exc!r}")
        return 0


async def _measure_spa_transitions(page) -> int:
    tracker = SPARouteTracker()
    guard = HydrationGuard()
    try:
        await page.goto(_SPA_URL, timeout=30000)
        await tracker.attach(page)
        await guard.wait_stable(page, timeout_ms=8000)

        # Add a todo item to ensure the app is interactive
        todo_input = page.get_by_placeholder("What needs to be done?")
        await todo_input.fill("sprint7 test todo")
        await page.keyboard.press("Enter")
        await guard.wait_stable(page, timeout_ms=3000)

        # Navigate to /active route
        await page.get_by_role("link", name="Active").click()
        await guard.wait_stable(page, timeout_ms=3000)

        # Navigate to /completed route
        await page.get_by_role("link", name="Completed").click()
        await guard.wait_stable(page, timeout_ms=3000)

        # Navigate back to All
        await page.get_by_role("link", name="All").click()
        await guard.wait_stable(page, timeout_ms=3000)

        events = await tracker.flush(page)
        logger.info(f"measure_sprint7: SPA route transitions = {len(events)}")
        for e in events:
            logger.debug(f"  {e.trigger}: {e.from_url} → {e.to_url}")
        return len(events)
    except Exception as exc:
        logger.error(f"measure_sprint7._measure_spa_transitions: {exc!r}")
        return 0


async def _measure_test_pass_rate() -> float:
    all_specs = []
    for mod_path in _TARGET_MODULES:
        if not mod_path.exists():
            logger.warning(f"measure_sprint7: module not found — {mod_path}")
            continue
        try:
            specs = parse_module(mod_path)
            logger.info(f"measure_sprint7: {mod_path} → {len(specs)} public functions")
            all_specs.extend(specs)
        except Exception as exc:
            logger.warning(f"measure_sprint7: parse failed for {mod_path}: {exc!r}")

    if not all_specs:
        logger.error("measure_sprint7: no specs parsed — cannot measure pass rate")
        return 0.0

    generator = PytestGenerator()
    judge = CodeJudge()
    executor = TestExecutor()

    try:
        generated: list[GeneratedTest] = await generator.generate(all_specs)
    except Exception as exc:
        logger.error(f"measure_sprint7: generation failed: {exc!r}")
        return 0.0

    # Judge + revision loop (max 2 attempts per test)
    accepted: list[GeneratedTest] = []
    for test in generated:
        current = test
        for attempt in range(3):
            result = await judge.judge(current)
            if result.grade == "acceptable":
                accepted.append(current)
                break
            if result.grade == "reject":
                break
            # needs_revision — re-run generate with all specs and pick matching test_id
            if attempt < 2:
                try:
                    revised_all = await generator.generate(all_specs)
                    match = next((t for t in revised_all if t.func_id == current.func_id), None)
                    if match:
                        current = match
                    else:
                        break
                except Exception:
                    break

    await judge.close()
    logger.info(f"measure_sprint7: accepted tests = {len(accepted)}/{len(generated)}")

    if not accepted:
        return 0.0

    passed = 0
    for test in accepted:
        exec_result: ExecutionResult = await executor.run(test)
        if exec_result.passed:
            passed += 1
        else:
            logger.debug(f"  FAIL [{test.test_id}]: {(exec_result.error_output or '')[:80]}")

    rate = passed / len(accepted) if accepted else 0.0
    logger.info(f"measure_sprint7: pass_rate = {passed}/{len(accepted)} = {rate:.3f}")
    return rate


async def _measure_self_heal(page) -> int:
    """Invoke RepairEngine on a broken locator and return 1 if call succeeded."""
    sfg_store = SFGStore(_PROJECT_ROOT / "audit/sprint7/sprint7_sfg.db")
    instructor = InstructorClient()
    engine = RepairEngine(instructor_client=instructor, sfg_store=sfg_store)

    broken_step = ContractStep(
        step_number=1,
        action_type="click",
        locator='get_by_role("button", name="__sprint7_nonexistent__")',
        input_value="",
        expected_state_hash="",
    )
    skill = ContractSkill(
        skill_id="sprint7_selfheal_probe",
        goal="sprint7 self-heal probe",
        target_url=page.url,
        domain="ecommerce",
        preconditions=[],
        steps=[broken_step],
        postconditions=[],
        repair_operators=[],
        created_at_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        success_count=0,
        failure_count=1,
    )

    triggered = 0
    try:
        result = await engine.repair(skill, broken_step, "element_not_found", page)
        # RepairEngine.repair() returns a patched ContractSkill when an operator
        # (SelReplace → ArgCorrect → PreInsert) actually produced a fix, or None
        # when every operator is exhausted. Self-heal is only "triggered" when a
        # real patch is produced — NOT merely because repair() was invoked.
        if result is not None:
            triggered = 1
            logger.info("measure_sprint7: RepairEngine.repair() produced a patch — self-heal triggered")
        else:
            triggered = 0
            logger.info("measure_sprint7: RepairEngine.repair() returned None — no self-heal")
    except Exception as exc:
        # repair() was attempted but raised — this is a failed repair, NOT a
        # successful self-heal.
        triggered = 0
        logger.info(f"measure_sprint7: RepairEngine.repair() raised — no self-heal: {exc!r}")
    finally:
        try:
            await instructor.close()
        except Exception:
            pass

    return triggered


async def main() -> None:
    shadow_found = 0
    spa_transitions = 0
    pass_rate = 0.0
    self_heal = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            shadow_found = await _measure_shadow_dom(page)
            spa_transitions = await _measure_spa_transitions(page)
            self_heal = await _measure_self_heal(page)
        finally:
            await browser.close()

    pass_rate = await _measure_test_pass_rate()

    sprint7_pass = (
        shadow_found >= _GATE_SHADOW_ELEMENTS
        and spa_transitions >= _GATE_SPA_TRANSITIONS
        and pass_rate >= _GATE_PASS_RATE
        and self_heal >= _GATE_SELF_HEAL
    )

    results = {
        "shadow_dom_elements_found": shadow_found,
        "spa_transitions_handled": spa_transitions,
        "test_pass_rate": round(pass_rate, 4),
        "self_heal_triggered": self_heal,
        "regression": pass_rate < _REGRESSION_THRESHOLD,
        "sprint7_status": "PASS" if sprint7_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Sprint 7 Results ===")
    for k, v in results.items():
        gate = ""
        if k == "shadow_dom_elements_found":
            gate = f" (gate ≥ {_GATE_SHADOW_ELEMENTS})"
        elif k == "spa_transitions_handled":
            gate = f" (gate ≥ {_GATE_SPA_TRANSITIONS})"
        elif k == "test_pass_rate":
            gate = f" (gate ≥ {_GATE_PASS_RATE})"
        elif k == "self_heal_triggered":
            gate = f" (gate ≥ {_GATE_SELF_HEAL})"
        print(f"  {k}: {v}{gate}")

    print(f"\n  → sprint7_results.json written to {_OUTPUT_PATH}")
    sys.exit(0 if sprint7_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
