# src/universal_qa/agent.py
from __future__ import annotations

import pathlib
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import async_playwright

from src.universal_qa.auth_manager import AuthManager
from src.universal_qa.explorer.nav_map import ExploredPage, NavigationMap
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
        explore_timeout: int = 5,
        max_depth: int = 4,
        allow_destructive: bool = False,
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
        from src.universal_qa.explorer.nav_map import ExplorerConfig
        self._explorer_cfg = ExplorerConfig(
            max_pages=max_pages,
            explore_timeout_min=explore_timeout,
            max_depth=max_depth,
            allow_destructive=allow_destructive,
        )

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

                # Phase 2: Discover URLs (fast)
                discover_url = page.url if page.url != self._url else self._url
                logger.info(f"Phase 2: site discovery from {discover_url}")
                sfg_store = await self._discovery.discover(page, discover_url)
                discovered_urls = [
                    n.url for n in sfg_store.get_nodes_by_url_prefix(
                        f"{urlparse(discover_url).scheme}://{urlparse(discover_url).netloc}"
                    )
                ]
                if discover_url not in discovered_urls:
                    discovered_urls.insert(0, discover_url)

                # Phase 3: Explore (thorough) → NavigationMap
                logger.info("Phase 3: interaction-based exploration")
                from src.universal_qa.explorer.site_explorer import SiteExplorer
                explorer = SiteExplorer(
                    auth=self._auth, config=self._explorer_cfg,
                    client=self._planner._client,
                )
                nav_map = await explorer.explore(page, discovered_urls)
                logger.info(
                    f"  Explored {len(nav_map.pages)} pages, {len(nav_map.flows)} flows"
                )

                # Supplement nav_map with Phase 2 pages not reached by Phase 3
                _explored_urls = {p.url.split("?")[0].split("#")[0] for p in nav_map.pages}
                _extra_pages = []
                for _node in sfg_store.get_nodes_by_url_prefix(
                    f"{urlparse(discover_url).scheme}://{urlparse(discover_url).netloc}"
                ):
                    _nurl = _node.url.split("?")[0].split("#")[0]
                    if _nurl not in _explored_urls:
                        _extra_pages.append(ExploredPage(
                            url=_node.url,
                            title=_node.page_title,
                            pam_content=_node.pam_content,
                            actions=[],
                        ))
                if _extra_pages:
                    nav_map = NavigationMap(
                        base_url=nav_map.base_url,
                        pages=nav_map.pages + _extra_pages,
                        flows=nav_map.flows,
                        explored_at_iso=nav_map.explored_at_iso,
                    )
                    logger.info(
                        f"  +{len(_extra_pages)} Phase 2 pages → total {len(nav_map.pages)}"
                    )

                # Phase 4: Plan from NavigationMap
                logger.info("Phase 4: generating test cases")
                test_cases = await self._planner.plan_from_map(nav_map)
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
