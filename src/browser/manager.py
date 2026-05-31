import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import yaml
from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from .auth_manager import AuthManager
from .resource_filter import ResourceFilter

_CONFIG_PATH = Path("config/agent.yaml")

_DEFAULT_LAUNCH_ARGS: list[str] = [
    "--disable-dev-shm-usage",  # required for WSL2 / Docker
    "--no-sandbox",
    "--disable-gpu",
    "--disable-background-networking",
    "--disable-default-apps",
    "--disable-extensions",
]

_DEFAULT_TIMEOUT_MS = 15_000
_DEFAULT_MAX_CONTEXTS = 3


def _load_browser_config() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("browser", {})
    return {}


class BrowserManager:
    """
    Async lifecycle manager for a shared Chromium browser instance with a
    bounded context pool (default max 3 concurrent, respecting 16GB RAM limit).

    Usage:
        async with BrowserManager() as bm:
            async with bm.new_page(role="admin") as page:
                await page.goto("http://localhost:3000")

    The context pool semaphore prevents OOM by ensuring at most *max_contexts*
    browser contexts exist simultaneously.  ResourceFilter is installed on every
    page by default.
    """

    def __init__(
        self,
        headless: bool | None = None,
        timeout_ms: int | None = None,
        max_contexts: int | None = None,
        launch_args: list[str] | None = None,
        auth_manager: AuthManager | None = None,
        block_resources: bool = True,
    ) -> None:
        cfg = _load_browser_config()
        self._headless: bool = headless if headless is not None else cfg.get("headless", True)
        self._timeout_ms: int = timeout_ms or cfg.get("timeout_ms", _DEFAULT_TIMEOUT_MS)
        self._max_contexts: int = max_contexts or cfg.get("max_contexts", _DEFAULT_MAX_CONTEXTS)
        self._launch_args: list[str] = launch_args or cfg.get("launch_args", _DEFAULT_LAUNCH_ARGS)
        self._block_resources: bool = block_resources
        self._auth_manager: AuthManager | None = auth_manager

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context_semaphore: asyncio.Semaphore = asyncio.Semaphore(self._max_contexts)

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def start(self) -> "BrowserManager":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=self._launch_args,
        )
        logger.info(
            f"BrowserManager started | headless={self._headless} "
            f"max_contexts={self._max_contexts}"
        )
        return self

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("BrowserManager stopped")

    async def __aenter__(self) -> "BrowserManager":
        return await self.start()

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ──────────────────────────────────────────────────────────────────────────
    # Context / Page factories
    # ──────────────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def new_context(
        self,
        role: str | None = None,
        storage_state: dict[str, Any] | None = None,
        extra_context_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BrowserContext, None]:
        """
        Yield a BrowserContext, honouring the pool semaphore.

        If *role* is given and an AuthManager is registered, the storageState
        is fetched (cached or freshly logged in) for that role.
        """
        assert self._browser, "BrowserManager.start() has not been called"

        if role and self._auth_manager and storage_state is None:
            storage_state = await self._auth_manager.get_storage_state(role)

        kwargs: dict[str, Any] = extra_context_kwargs or {}
        if storage_state:
            kwargs["storage_state"] = storage_state

        async with self._context_semaphore:
            context = await self._browser.new_context(**kwargs)
            context.set_default_timeout(self._timeout_ms)
            context.set_default_navigation_timeout(self._timeout_ms)
            logger.debug(
                f"BrowserContext opened | role={role!r} "
                f"has_storage_state={storage_state is not None}"
            )
            try:
                yield context
            finally:
                try:
                    await context.close()
                except Exception as exc:
                    logger.warning(f"Error closing BrowserContext: {exc}")

    @asynccontextmanager
    async def new_page(
        self,
        role: str | None = None,
        storage_state: dict[str, Any] | None = None,
        install_resource_filter: bool | None = None,
        extra_context_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Page, None]:
        """
        Yield a Page ready for test interaction.

        Convenience wrapper around new_context() that opens a single page
        and optionally installs the ResourceFilter.
        """
        should_filter = (
            install_resource_filter
            if install_resource_filter is not None
            else self._block_resources
        )
        async with self.new_context(
            role=role,
            storage_state=storage_state,
            extra_context_kwargs=extra_context_kwargs,
        ) as ctx:
            page = await ctx.new_page()
            resource_filter = ResourceFilter(enabled=should_filter)
            if should_filter:
                await resource_filter.install(page)
            logger.debug(f"Page opened | url={page.url!r} resource_filter={should_filter}")
            try:
                yield page
            finally:
                resource_filter.log_summary()

    @asynccontextmanager
    async def multi_page_context(
        self,
        role: str | None = None,
        storage_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BrowserContext, None]:
        """
        Yield a BrowserContext where callers manage their own pages.
        Useful for multi-tab / multi-window test scenarios.
        """
        async with self.new_context(role=role, storage_state=storage_state) as ctx:
            yield ctx

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    async def capture_storage_state(
        self, context: BrowserContext, role: str | None = None
    ) -> dict[str, Any]:
        """Snapshot the current storageState (useful after UI-driven auth)."""
        state = await context.storage_state()
        if role and self._auth_manager:
            self._auth_manager._save_storage_state(role, state, expiry_timestamp=0)
        return state
