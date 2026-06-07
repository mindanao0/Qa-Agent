"""
SiteDiscovery — BFS site crawler using an already-authenticated shared Page.

Uses SFGCrawler._visit_node() directly (NOT crawl()) so the caller's
authenticated BrowserContext is preserved throughout the crawl.
"""
from __future__ import annotations

import pathlib
import tempfile
from collections import deque
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page

from src.contractskill.crawler import CrawlerConfig, SFGCrawler
from src.contractskill.sfg import SFGStore
from src.perception.grounder import Grounder


class SiteDiscovery:
    """BFS site crawler using an already-authenticated shared Page.

    Uses SFGCrawler._visit_node directly (not crawl()) so the caller's
    authenticated BrowserContext is preserved throughout.
    """

    def __init__(self, max_pages: int = 50, max_depth: int = 6) -> None:
        self.max_pages = max_pages
        self.max_depth = max_depth

    async def discover(
        self,
        page: Page,
        start_url: str,
        db_path: pathlib.Path | None = None,
    ) -> SFGStore:
        path = db_path or pathlib.Path(tempfile.mkdtemp(prefix="uqa_sfg_")) / "sfg.db"
        store = SFGStore(db_path=path)
        grounder = Grounder()
        config = CrawlerConfig(max_pages=self.max_pages, max_depth=self.max_depth)
        crawler = SFGCrawler(store, grounder, config)

        base_domain = urlparse(start_url).netloc
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        pages_visited = 0

        while queue and pages_visited < self.max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > self.max_depth:
                continue

            try:
                # ถ้าอยู่หน้าเดิมอยู่แล้ว ไม่ต้อง goto (ป้องกัน SPA reload ก่อน render)
                current = page.url.split("?")[0].split("#")[0]
                target = url.split("?")[0].split("#")[0]
                if current != target:
                    await page.goto(url, wait_until="networkidle", timeout=30_000)
                else:
                    await page.wait_for_load_state("networkidle", timeout=10_000)
                visited.add(url)
                await self._visit_and_record(crawler, page)
                pages_visited += 1
                logger.info(
                    f"SiteDiscovery: visited {url} ({pages_visited}/{self.max_pages})"
                )

                if depth < self.max_depth:
                    links: list[str] = await page.evaluate("""
                        () => {
                            const hrefs = new Set();
                            document.querySelectorAll('a[href]').forEach(a => {
                                const h = a.href;
                                if (h && h.startsWith('http') && !h.endsWith('#') && !/#$/.test(h)) {
                                    hrefs.add(h);
                                }
                            });
                            return Array.from(hrefs);
                        }
                    """)
                    logger.debug(f"SiteDiscovery: found {len(links)} valid links on {url}: {links[:5]}")
                    for link in links:
                        clean = link.split("?")[0].split("#")[0]
                        if urlparse(clean).netloc == base_domain and clean not in visited:
                            queue.append((clean, depth + 1))
            except Exception as exc:
                logger.warning(f"SiteDiscovery: {url} skipped — {exc!r}")

        logger.info(
            f"SiteDiscovery: done — {pages_visited} pages, {store.node_count()} nodes"
        )
        return store

    @staticmethod
    async def _visit_and_record(crawler: SFGCrawler, page: Page):
        # _visit_node returns (SFGNode, pam_tokens) — unpack and return just the node
        node, _ = await crawler._visit_node(page, None)
        return node


__all__ = ["SiteDiscovery"]
