from __future__ import annotations

import pathlib
import time

from loguru import logger
from playwright.async_api import Page

from src.agents.observer_driver.observers.accessibility_observer import (
    AccessibilityObserver,
)
from src.agents.observer_driver.observers.security_observer import SecurityObserver
from src.contractskill.sfg import SFGStore
from src.explorer.executor import HypothesisExecutor
from src.explorer.hypothesis import TestHypothesis
from src.llm.instructor_client import InstructorClient
from src.universal_qa.models import StepTrace, TestCase, TestResult
from src.universal_qa.reporters.terminal import TerminalReporter

_XSS_PAYLOAD = "<script>window.__xss_fired=true;</script>"
_SQLI_PAYLOAD = "' OR '1'='1"


def _map_exception(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, TimeoutError) or "timeout" in msg.lower():
        return f"Element not found within 30s — {msg[:120]}"
    if isinstance(exc, AssertionError):
        return f"Expected outcome not met — {msg[:120]}"
    return msg[:200]


class UniversalTestRunner:
    """Executes a list of TestCase objects using Playwright.

    Routes by type:
    - functional    → HypothesisExecutor (existing Sprint 5)
    - accessibility → AccessibilityObserver._scan static method
    - security      → SecurityObserver static methods + XSS/SQLi injection
    """

    def __init__(
        self,
        sfg_store: SFGStore | None = None,
        screenshot_dir: pathlib.Path | None = None,
        terminal_reporter: TerminalReporter | None = None,
    ) -> None:
        instructor = InstructorClient()
        self._hyp_executor = HypothesisExecutor(instructor, sfg_store)
        self._screenshot_dir = screenshot_dir
        self._terminal = terminal_reporter or TerminalReporter()

    async def run(
        self, test_cases: list[TestCase], page: Page
    ) -> list[TestResult]:
        results: list[TestResult] = []
        for tc in test_cases:
            result = await self._dispatch(tc, page)
            self._terminal.report_one(result)
            results.append(result)
        return results

    async def _dispatch(self, tc: TestCase, page: Page) -> TestResult:
        try:
            if tc.type == "accessibility":
                return await self._run_accessibility(tc, page)
            if tc.type == "security":
                return await self._run_security(tc, page)
            return await self._run_functional(tc, page)
        except Exception as exc:
            return TestResult(
                test_case=tc,
                passed=False,
                failure_reason=_map_exception(exc),
                duration_ms=0,
            )

    async def _run_functional(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        hyp = TestHypothesis(
            goal=tc.title,
            start_url=tc.source_url,
            preconditions=tc.preconditions,
            steps=tc.steps,
            expected_outcome=tc.expected_outcome,
            confidence=0.7,
        )
        hyp_result = await self._hyp_executor.execute(hyp, page)
        traces = [
            StepTrace(
                step=step,
                status="passed" if hyp_result.passed else "failed",
                detail=f"executed {hyp_result.steps_executed} step(s)",
                error=hyp_result.failure_reason if not hyp_result.passed else None,
            )
            for step in tc.steps
        ]
        screenshot = await self._maybe_screenshot(page, tc.id, hyp_result.passed)
        return TestResult(
            test_case=tc,
            passed=hyp_result.passed,
            steps_trace=traces,
            failure_reason=hyp_result.failure_reason,
            screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _run_accessibility(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        traces: list[StepTrace] = []

        try:
            await page.goto(tc.source_url, wait_until="domcontentloaded", timeout=30_000)
            traces.append(StepTrace(step=f"Navigate to {tc.source_url}",
                                    status="passed", detail="page loaded"))
        except Exception as exc:
            return TestResult(
                test_case=tc, passed=False,
                steps_trace=traces,
                failure_reason=_map_exception(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        try:
            snapshot = await page.aria_snapshot()
        except Exception:
            snapshot = ""

        aria_findings = AccessibilityObserver._scan(snapshot)
        missing_alt: int = await page.evaluate(
            "() => document.querySelectorAll('img:not([alt])').length"
        )

        passed = not aria_findings and missing_alt == 0
        detail = (
            f"ARIA: {aria_findings if aria_findings else ['ok']}; "
            f"missing alt: {missing_alt}"
        )
        traces.append(StepTrace(
            step="Check accessibility rules",
            status="passed" if passed else "failed",
            detail=detail,
            error="; ".join(aria_findings) if aria_findings else None,
        ))
        if missing_alt:
            traces.append(StepTrace(
                step="Check image alt text",
                status="failed",
                detail=f"{missing_alt} image(s) missing alt attribute",
                error=f"{missing_alt} images missing alt",
            ))

        failure_reason = None
        if not passed:
            parts = []
            if aria_findings:
                parts.extend(aria_findings)
            if missing_alt:
                parts.append(f"{missing_alt} image(s) missing alt text")
            failure_reason = "; ".join(parts)

        screenshot = await self._maybe_screenshot(page, tc.id, passed)
        return TestResult(
            test_case=tc, passed=passed, steps_trace=traces,
            failure_reason=failure_reason, screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _run_security(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        traces: list[StepTrace] = []
        is_xss = "xss" in tc.title.lower()
        payload = _XSS_PAYLOAD if is_xss else _SQLI_PAYLOAD

        try:
            await page.goto(tc.source_url, wait_until="domcontentloaded", timeout=30_000)
            traces.append(StepTrace(step=f"Navigate to {tc.source_url}",
                                    status="passed", detail="page loaded"))
        except Exception as exc:
            return TestResult(
                test_case=tc, passed=False, steps_trace=traces,
                failure_reason=_map_exception(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        inputs = page.get_by_role("textbox")
        count = await inputs.count()
        for i in range(count):
            try:
                await inputs.nth(i).fill(payload, timeout=5_000)
            except Exception:
                pass
        traces.append(StepTrace(
            step=f"Fill {count} input(s) with payload",
            status="passed", detail=f"payload: {payload[:60]}",
        ))

        try:
            await page.get_by_role("button").first.click(timeout=5_000)
            await page.wait_for_timeout(1_000)
        except Exception:
            pass

        passed = True
        failure_reason = None
        if is_xss:
            xss_fired: bool = await page.evaluate("() => !!window.__xss_fired")
            if xss_fired:
                passed = False
                failure_reason = "XSS payload was executed (window.__xss_fired=true)"
        else:
            content = (await page.content()).lower()
            sql_keywords = ["sql syntax", "mysql_fetch", "ora-0", "sqlite", "syntax error near"]
            hit = next((kw for kw in sql_keywords if kw in content), None)
            if hit:
                passed = False
                failure_reason = f"SQL error keyword '{hit}' found in page response"

        missing_headers = await SecurityObserver._missing_headers(page, tc.source_url)
        if missing_headers:
            traces.append(StepTrace(
                step="Check security headers",
                status="failed",
                detail=f"missing: {missing_headers}",
                error=f"Missing headers: {', '.join(missing_headers)}",
            ))

        traces.append(StepTrace(
            step="Verify payload not executed",
            status="passed" if passed else "failed",
            detail="checked page response",
            error=failure_reason,
        ))

        screenshot = await self._maybe_screenshot(page, tc.id, passed)
        return TestResult(
            test_case=tc, passed=passed, steps_trace=traces,
            failure_reason=failure_reason, screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _maybe_screenshot(
        self, page: Page, test_id: str, passed: bool
    ) -> str | None:
        if passed or self._screenshot_dir is None:
            return None
        try:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self._screenshot_dir / f"{test_id}.png"
            await page.screenshot(path=str(path), full_page=False)
            return str(path)
        except Exception:
            return None


__all__ = ["UniversalTestRunner", "_map_exception"]
