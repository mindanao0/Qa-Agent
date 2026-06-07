# src/universal_qa/agent.py
from __future__ import annotations

import pathlib

from loguru import logger
from playwright.async_api import async_playwright

from src.universal_qa.auth_manager import AuthManager
from src.universal_qa.models import TestResult
from src.universal_qa.reporters.html import HTMLReporter
from src.universal_qa.reporters.terminal import TerminalReporter
from src.universal_qa.site_discovery import SiteDiscovery
from src.universal_qa.test_planner import UniversalTestPlanner
from src.universal_qa.test_runner import UniversalTestRunner

_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]


class UniversalQAAgent:
    """Orchestrates all 4 phases: Discover → Plan → Execute → Report."""

    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        max_pages: int = 50,
        headless: bool = True,
        output_dir: pathlib.Path | None = None,
    ) -> None:
        self._url = url
        self._max_pages = max_pages
        self._headless = headless
        self._output_dir = output_dir or pathlib.Path("reports")
        self._auth = AuthManager(username=username, password=password)
        self._discovery = SiteDiscovery(max_pages=max_pages)
        self._planner = UniversalTestPlanner()
        self._terminal = TerminalReporter()

    async def run(self) -> list[TestResult]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self._headless, args=_LAUNCH_ARGS
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            try:
                logger.info(f"UniversalQAAgent: starting run on {self._url}")

                # Phase 1: Navigate + Auth
                await page.goto(self._url, wait_until="domcontentloaded", timeout=30_000)
                await self._auth.setup(page)

                # Phase 2: Discover
                logger.info("Phase 2: site discovery")
                sfg_store = await self._discovery.discover(page, self._url)

                # Phase 3: Plan
                logger.info("Phase 3: generating test cases")
                test_cases = await self._planner.plan(sfg_store, self._url)
                logger.info(f"  {len(test_cases)} test cases generated")

                # Phase 4: Execute + Report
                logger.info("Phase 4: executing test cases")
                screenshot_dir = self._output_dir / "screenshots"
                runner = UniversalTestRunner(
                    sfg_store=sfg_store,
                    screenshot_dir=screenshot_dir,
                    terminal_reporter=self._terminal,
                )
                results = await runner.run(test_cases, page)

                # Final reports
                self._terminal.report_summary(results)
                html_reporter = HTMLReporter(output_dir=self._output_dir)
                report_path = html_reporter.generate(results)
                logger.info(f"HTML report: {report_path}")

                return results

            finally:
                await browser.close()


__all__ = ["UniversalQAAgent"]
