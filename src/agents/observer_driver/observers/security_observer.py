"""Security Observer — header and mixed-content auditing per navigation.

Per ``driver_action`` event of type ``navigate`` or ``click`` the observer:
  1. Re-fetches the current URL via ``page.request.get`` and inspects headers
     for CSP, X-Frame-Options, and Strict-Transport-Security.
  2. Scans the live DOM for password inputs lacking explicit autocomplete.
  3. Flags mixed HTTP content on HTTPS pages.
"""
from __future__ import annotations

import time

from playwright.async_api import Page

from src.agents.observer_driver.observers.base_observer import BaseObserver
from src.agents.observer_driver.state import ObserverReport, Severity
from src.agents.observer_driver.trace_bus import TraceEvent

_AUDITED_ACTIONS = frozenset({"navigate", "click"})
_REQUIRED_HEADERS = (
    "content-security-policy",
    "x-frame-options",
    "strict-transport-security",
)


class SecurityObserver(BaseObserver):
    """Header hygiene + mixed-content + form-exposure auditor."""

    def __init__(self, queue) -> None:  # type: ignore[no-untyped-def]
        super().__init__(name="security", queue=queue)

    async def _audit(
        self, event: TraceEvent, page: Page
    ) -> ObserverReport | None:
        if event.event_type != "driver_action":
            return None
        if event.payload.get("action_type") not in _AUDITED_ACTIONS:
            return None

        url = event.payload.get("url") or page.url
        findings: list[str] = []
        severity: Severity = "info"

        # 1. Header scan (best-effort — non-HTTP URLs are skipped)
        if url.startswith(("http://", "https://")):
            missing = await self._missing_headers(page, url)
            for header in missing:
                findings.append(f"missing security header: {header}")
            if missing:
                severity = "warn"

        # 2. Mixed-content on HTTPS pages
        if url.startswith("https://"):
            mixed_count = await self._count_mixed_content(page)
            if mixed_count > 0:
                findings.append(
                    f"mixed-content resources on HTTPS page: {mixed_count}"
                )
                severity = "critical"

        # 3. Password inputs without explicit autocomplete attribute
        password_exposure = await self._password_inputs_without_autocomplete(page)
        if password_exposure > 0:
            findings.append(
                f"password input(s) without autocomplete attribute: {password_exposure}"
            )
            if severity != "critical":
                severity = "warn"

        if not findings:
            findings = ["no security issues detected"]

        return ObserverReport(
            observer_name=self.name,
            findings=findings,
            severity=severity,
            timestamp_ms=int(time.time() * 1000),
        )

    @staticmethod
    async def _missing_headers(page: Page, url: str) -> list[str]:
        try:
            response = await page.request.get(url, timeout=10_000)
        except Exception:
            return []
        try:
            headers = {k.lower(): v for k, v in (await response.all_headers()).items()}
        except Exception:
            headers = {}
        return [h for h in _REQUIRED_HEADERS if h not in headers]

    @staticmethod
    async def _count_mixed_content(page: Page) -> int:
        try:
            return int(
                await page.evaluate(
                    """
                    () => {
                        const tags = ['img','script','iframe','link','audio','video','source'];
                        let n = 0;
                        for (const tag of tags) {
                            for (const el of document.querySelectorAll(tag)) {
                                const src = el.src || el.href || '';
                                if (typeof src === 'string' && src.startsWith('http://')) n += 1;
                            }
                        }
                        return n;
                    }
                    """
                )
            )
        except Exception:
            return 0

    @staticmethod
    async def _password_inputs_without_autocomplete(page: Page) -> int:
        try:
            return int(
                await page.evaluate(
                    """
                    () => {
                        let n = 0;
                        for (const el of document.querySelectorAll('input[type=password]')) {
                            if (!el.hasAttribute('autocomplete')) n += 1;
                        }
                        return n;
                    }
                    """
                )
            )
        except Exception:
            return 0


__all__ = ["SecurityObserver"]
