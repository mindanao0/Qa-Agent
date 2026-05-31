import asyncio
from dataclasses import dataclass, field
from typing import FrozenSet

from loguru import logger
from playwright.async_api import Page, Route

BLOCKED_RESOURCE_TYPES: FrozenSet[str] = frozenset({"image", "media", "font", "stylesheet"})
ALLOWED_RESOURCE_TYPES: FrozenSet[str] = frozenset(
    {"document", "script", "xhr", "fetch", "websocket", "other", "eventsource", "manifest"}
)

# Rough average byte savings per blocked resource type (for monitoring)
_AVERAGE_BYTES: dict[str, int] = {
    "image": 80_000,
    "media": 500_000,
    "font": 30_000,
    "stylesheet": 15_000,
}


@dataclass
class FilterStats:
    requests_blocked: int = 0
    requests_allowed: int = 0
    bytes_saved_estimate: int = 0
    blocked_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def bytes_saved_kb(self) -> float:
        return self.bytes_saved_estimate / 1024


class ResourceFilter:
    """
    Installs a Playwright route interceptor that blocks non-essential resource
    types (image, media, font, stylesheet) to reduce RAM and bandwidth consumption.
    Tracks estimated bytes saved per session.
    """

    def __init__(
        self,
        blocked_types: FrozenSet[str] = BLOCKED_RESOURCE_TYPES,
        enabled: bool = True,
    ) -> None:
        self._blocked = blocked_types
        self._enabled = enabled
        self.stats = FilterStats()
        self._lock = asyncio.Lock()

    async def install(self, page: Page) -> None:
        """Attach the route handler to *page*. Call once per page."""
        if not self._enabled:
            logger.debug("ResourceFilter disabled — skipping route installation")
            return
        await page.route("**/*", self._handle_route)
        logger.debug(f"ResourceFilter installed | blocking={sorted(self._blocked)}")

    async def _handle_route(self, route: Route) -> None:
        resource_type: str = route.request.resource_type

        if resource_type in self._blocked:
            estimate = _AVERAGE_BYTES.get(resource_type, 10_000)
            async with self._lock:
                self.stats.requests_blocked += 1
                self.stats.bytes_saved_estimate += estimate
                self.stats.blocked_by_type[resource_type] = (
                    self.stats.blocked_by_type.get(resource_type, 0) + 1
                )
            try:
                await route.abort()
            except Exception:
                pass  # route may already be handled
        else:
            async with self._lock:
                self.stats.requests_allowed += 1
            try:
                await route.continue_()
            except Exception:
                pass

    def log_summary(self) -> None:
        logger.info(
            "ResourceFilter session summary | "
            f"blocked={self.stats.requests_blocked} "
            f"allowed={self.stats.requests_allowed} "
            f"saved≈{self.stats.bytes_saved_kb:.1f}KB "
            f"by_type={self.stats.blocked_by_type}"
        )
