"""Performance Observer — captures Chrome DevTools metrics after each action.

Uses an on-demand CDP session per audit (CDP sessions are cheap and one-shot
is safer than holding a session across observer restarts). For navigation
events the observer additionally records ``window.performance.timing`` so a
crude LCP/FCP/DCL trio can be derived without subscribing to PerformanceObserver
APIs from inside the page.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger
from playwright.async_api import Page

from src.agents.observer_driver.observers.base_observer import BaseObserver
from src.agents.observer_driver.state import ObserverReport, Severity
from src.agents.observer_driver.trace_bus import TraceEvent

# Severity thresholds (raw page-load duration). Tuned so reasonable real-world
# load times on slow networks (≤ 8s) stay at info; only pathological loads
# (≥ 15s) escalate to warn. The Observer-Driver baseline must never be
# "critical" purely on synthetic timing — that's reserved for hard failures
# such as CDP returning nothing at all.
PERF_WARN_LOAD_MS = 15_000


class PerformanceObserver(BaseObserver):
    """CDP Performance.getMetrics + window.performance.timing snapshots."""

    def __init__(self, queue) -> None:  # type: ignore[no-untyped-def]
        super().__init__(name="performance", queue=queue)

    async def _audit(
        self, event: TraceEvent, page: Page
    ) -> ObserverReport | None:
        if event.event_type != "driver_action":
            return None

        findings: list[str] = []
        severity: Severity = "info"

        # 1. CDP Performance.getMetrics
        cdp_metrics = await self._cdp_metrics(page)
        for name, value in cdp_metrics.items():
            findings.append(f"cdp.{name}={value}")

        # 2. Approximate LCP / load timing on navigation actions
        if event.payload.get("action_type") == "navigate":
            timing = await self._performance_timing(page)
            for key, value in timing.items():
                findings.append(f"timing.{key}={value}")

            load_ms = timing.get("loadEventEnd_minus_navigationStart_ms")
            if isinstance(load_ms, (int, float)) and load_ms >= PERF_WARN_LOAD_MS:
                severity = "warn"

        if not cdp_metrics:
            severity = "warn"
            findings.append("CDP Performance.getMetrics returned no data")
        if not findings:
            findings = ["no performance metrics captured"]

        return ObserverReport(
            observer_name=self.name,
            findings=findings,
            severity=severity,
            timestamp_ms=int(time.time() * 1000),
        )

    @staticmethod
    async def _cdp_metrics(page: Page) -> dict[str, float]:
        try:
            client = await page.context.new_cdp_session(page)
        except Exception as exc:
            logger.debug(f"performance: new_cdp_session failed: {exc}")
            return {}
        try:
            await client.send("Performance.enable")
            result: dict[str, Any] = await client.send("Performance.getMetrics")
        except Exception as exc:
            logger.debug(f"performance: Performance.getMetrics failed: {exc}")
            return {}
        finally:
            try:
                await client.detach()
            except Exception:
                pass

        out: dict[str, float] = {}
        for entry in result.get("metrics", []):
            name = entry.get("name")
            value = entry.get("value")
            if isinstance(name, str) and isinstance(value, (int, float)):
                out[name] = float(value)
        return out

    @staticmethod
    async def _performance_timing(page: Page) -> dict[str, float]:
        try:
            timing = await page.evaluate(
                """
                () => {
                    const t = window.performance && window.performance.timing;
                    if (!t) return {};
                    return {
                        navigationStart: t.navigationStart,
                        domContentLoadedEventEnd: t.domContentLoadedEventEnd,
                        loadEventEnd: t.loadEventEnd,
                    };
                }
                """
            )
        except Exception:
            return {}

        if not isinstance(timing, dict):
            return {}

        nav_start = timing.get("navigationStart") or 0
        dcl = timing.get("domContentLoadedEventEnd") or 0
        load_end = timing.get("loadEventEnd") or 0

        return {
            "dcl_minus_navigationStart_ms": max(0.0, float(dcl - nav_start)),
            "loadEventEnd_minus_navigationStart_ms": max(0.0, float(load_end - nav_start)),
        }


__all__ = ["PERF_WARN_LOAD_MS", "PerformanceObserver"]
