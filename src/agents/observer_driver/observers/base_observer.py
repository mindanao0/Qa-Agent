"""Abstract base for read-only Observer agents.

An Observer owns NO browser state — it merely reads the page through helper
APIs (request, evaluate, CDP) and emits :class:`ObserverReport` items. Each
concrete subclass implements :meth:`_audit`; the base loop handles queue
draining, stop coordination, and exception isolation.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from loguru import logger
from playwright.async_api import Page

from src.agents.observer_driver.state import ObserverReport
from src.agents.observer_driver.trace_bus import TraceEvent

QUEUE_GET_TIMEOUT_SEC = 2.0


class BaseObserver(ABC):
    """Async observer consuming :class:`TraceEvent` items from its private queue."""

    def __init__(self, name: str, queue: asyncio.Queue[TraceEvent]) -> None:
        self.name = name
        self.queue = queue
        self.reports: list[ObserverReport] = []

    async def run(self, page: Page, stop_event: asyncio.Event) -> list[ObserverReport]:
        """Drain events until *stop_event* is set, then return collected reports.

        The ``asyncio.wait_for`` timeout guarantees we re-check *stop_event*
        every ``QUEUE_GET_TIMEOUT_SEC`` seconds even when no events arrive,
        so the coordinator's ``gather`` never hangs.
        """
        while not stop_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self.queue.get(), timeout=QUEUE_GET_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                continue
            await self._process_event(event, page)

        # After stop, drain anything the Driver published just before signalling.
        while True:
            try:
                event = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._process_event(event, page)

        return self.reports

    async def _process_event(self, event: TraceEvent, page: Page) -> None:
        try:
            report = await self._audit(event, page)
        except Exception as exc:  # noqa: BLE001 — observer failure must not crash run
            logger.warning(
                f"{self.name}: _audit raised {type(exc).__name__}: {exc}"
            )
            return
        if report is not None:
            self.reports.append(report)

    @abstractmethod
    async def _audit(
        self, event: TraceEvent, page: Page
    ) -> ObserverReport | None:
        """Inspect *event* and *page* and optionally produce a report."""


__all__ = ["BaseObserver", "QUEUE_GET_TIMEOUT_SEC"]
