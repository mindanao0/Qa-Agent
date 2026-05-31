"""
Allure reporter integration.

Attaches AxTree snapshots, screenshots, AI reasoning, and healing events
to the current Allure test report as structured artefacts.

Works inside any pytest-asyncio test; the allure context is thread-local
and does not require special async wrapping.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator

from loguru import logger

try:
    import allure
    from allure_commons.types import AttachmentType as _AT

    _ALLURE_AVAILABLE = True
except ImportError:
    _ALLURE_AVAILABLE = False
    allure = None  # type: ignore[assignment]
    _AT = None  # type: ignore[assignment]

# Playwright import is optional here (avoid hard dep at module level)
try:
    from playwright.async_api import Page
except ImportError:
    Page = Any  # type: ignore[assignment,misc]

from src.llm.structured import HealedLocator


# ──────────────────────────────────────────────────────────────────────────────
# Reporter
# ──────────────────────────────────────────────────────────────────────────────


class AllureReporter:
    """
    Facade over the allure-python SDK.

    Every method is a no-op when allure is not installed, so tests continue
    to run even without the reporting dependency.
    """

    # ── Attachments ──────────────────────────────────────────────────────────

    @staticmethod
    def attach_axtree(axtree: str, name: str = "AxTree Snapshot") -> None:
        """Attach the pruned Accessibility Tree snapshot as plain text."""
        if not _ALLURE_AVAILABLE or not axtree.strip():
            return
        try:
            allure.attach(
                body=axtree,
                name=name,
                attachment_type=allure.attachment_type.TEXT,
            )
        except Exception as exc:
            logger.debug(f"AllureReporter.attach_axtree: {exc}")

    @staticmethod
    async def attach_screenshot(
        page: Page,
        name: str = "Screenshot",
        full_page: bool = True,
    ) -> None:
        """Capture and attach a full-page screenshot from *page*."""
        if not _ALLURE_AVAILABLE:
            return
        try:
            screenshot: bytes = await page.screenshot(full_page=full_page)
            allure.attach(
                body=screenshot,
                name=name,
                attachment_type=allure.attachment_type.PNG,
            )
        except Exception as exc:
            logger.debug(f"AllureReporter.attach_screenshot: {exc}")

    @staticmethod
    def attach_ai_reasoning(reasoning: str, name: str = "AI Reasoning") -> None:
        """Attach a free-text AI reasoning trace."""
        if not _ALLURE_AVAILABLE or not reasoning.strip():
            return
        try:
            allure.attach(
                body=reasoning,
                name=name,
                attachment_type=allure.attachment_type.TEXT,
            )
        except Exception as exc:
            logger.debug(f"AllureReporter.attach_ai_reasoning: {exc}")

    @staticmethod
    def attach_healing_event(
        healed: HealedLocator,
        name: str = "Healing Event",
    ) -> None:
        """Attach a structured JSON record of a locator healing event."""
        if not _ALLURE_AVAILABLE:
            return
        try:
            body = json.dumps(
                {
                    "original": healed.original,
                    "healed": healed.healed,
                    "confidence": healed.confidence,
                    "method": healed.method,
                    "reasoning": healed.reasoning,
                },
                indent=2,
                ensure_ascii=False,
            )
            allure.attach(
                body=body,
                name=name,
                attachment_type=allure.attachment_type.JSON,
            )
        except Exception as exc:
            logger.debug(f"AllureReporter.attach_healing_event: {exc}")

    @staticmethod
    def attach_json(data: Any, name: str = "Data") -> None:
        """Attach any JSON-serialisable object."""
        if not _ALLURE_AVAILABLE:
            return
        try:
            allure.attach(
                body=json.dumps(data, indent=2, default=str, ensure_ascii=False),
                name=name,
                attachment_type=allure.attachment_type.JSON,
            )
        except Exception as exc:
            logger.debug(f"AllureReporter.attach_json: {exc}")

    # ── Metadata ─────────────────────────────────────────────────────────────

    @staticmethod
    def set_test_metadata(
        domain: str = "",
        role: str = "",
        url: str = "",
        mode: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Add Allure labels to the current test item."""
        if not _ALLURE_AVAILABLE:
            return
        try:
            if domain:
                allure.label("domain", domain)
            if role:
                allure.label("role", role)
            if url:
                allure.label("url", url)
            if mode:
                allure.label("mode", mode)
            for tag in tags or []:
                allure.tag(tag)
        except Exception as exc:
            logger.debug(f"AllureReporter.set_test_metadata: {exc}")

    # ── Step context managers ─────────────────────────────────────────────────

    @staticmethod
    @contextmanager
    def step(title: str) -> Generator[None, None, None]:
        """Synchronous Allure step (usable in both sync and async tests)."""
        if not _ALLURE_AVAILABLE:
            yield
            return
        with allure.step(title):
            yield

    @staticmethod
    @asynccontextmanager
    async def async_step(
        title: str,
        page: Page | None = None,
        screenshot_on_failure: bool = True,
    ) -> AsyncGenerator[None, None]:
        """
        Async Allure step that captures a failure screenshot when *page* is
        provided and the step body raises an exception.
        """
        if not _ALLURE_AVAILABLE:
            yield
            return
        with allure.step(title):
            try:
                yield
            except Exception:
                if screenshot_on_failure and page is not None:
                    try:
                        screenshot = await page.screenshot(full_page=True)
                        allure.attach(
                            body=screenshot,
                            name=f"Failure: {title}",
                            attachment_type=allure.attachment_type.PNG,
                        )
                    except Exception:
                        pass
                raise

    # ── Test lifecycle helpers ────────────────────────────────────────────────

    @staticmethod
    def title(text: str) -> None:
        if _ALLURE_AVAILABLE:
            try:
                allure.title(text)
            except Exception:
                pass

    @staticmethod
    def description(text: str) -> None:
        if _ALLURE_AVAILABLE:
            try:
                allure.description(text)
            except Exception:
                pass

    @staticmethod
    def severity(level: str = "normal") -> None:
        """level: blocker | critical | normal | minor | trivial"""
        if not _ALLURE_AVAILABLE:
            return
        _map = {
            "blocker": allure.severity_level.BLOCKER,
            "critical": allure.severity_level.CRITICAL,
            "normal": allure.severity_level.NORMAL,
            "minor": allure.severity_level.MINOR,
            "trivial": allure.severity_level.TRIVIAL,
        }
        try:
            allure.severity(_map.get(level.lower(), allure.severity_level.NORMAL))
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Module-level default instance
# ──────────────────────────────────────────────────────────────────────────────

reporter = AllureReporter()
