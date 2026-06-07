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
                    # 1. ดึง <a href> links ปกติ
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
                    # 2. Interaction-based discovery สำหรับ SPA (href="#" หรือไม่มี href)
                    interaction_links = await self._discover_by_interaction(
                        page, url, base_domain
                    )
                    all_links = links + interaction_links
                    logger.debug(
                        f"SiteDiscovery: {len(links)} href + {len(interaction_links)} interaction links on {url}"
                    )
                    for link in all_links:
                        clean = link.split("?")[0].split("#")[0]
                        if urlparse(clean).netloc == base_domain and clean not in visited:
                            queue.append((clean, depth + 1))
            except Exception as exc:
                logger.warning(f"SiteDiscovery: {url} skipped — {exc!r}")

        logger.info(
            f"SiteDiscovery: done — {pages_visited} pages, {store.node_count()} nodes"
        )
        return store

    async def _discover_by_interaction(
        self, page: Page, original_url: str, base_domain: str, max_clicks: int = 15
    ) -> list[str]:
        """คลิก SPA navigation elements แล้วจับ URL ที่เปลี่ยน.

        ใช้สำหรับ SPA ที่ใช้ href='#' หรือ JavaScript navigation แทน <a href>.
        """
        _BLOCKED_TEXT = frozenset({
            "add to cart", "remove", "delete", "checkout", "login", "logout",
            "register", "submit", "send", "buy", "purchase", "sign up", "sign in",
            "เพิ่มลงตะกร้า", "ลบ", "ชำระเงิน", "เข้าสู่ระบบ", "ออกจากระบบ",
        })

        candidates: list[dict] = await page.evaluate("""() => {
            const BLOCKED = ['add to cart','remove','delete','checkout','login',
                             'logout','register','submit','send','buy','purchase'];
            const seen = new Set();
            const results = [];
            document.querySelectorAll('a, [role="link"], [role="button"]').forEach(el => {
                const href = el.getAttribute('href');
                // ข้ามถ้า href เป็น URL จริงหรือ path จริง (จัดการโดย href extraction แล้ว)
                if (href && href.startsWith('http')) return;
                if (href && href.startsWith('/') && href.length > 1 && !href.startsWith('/#')) return;
                const text = (el.textContent || '').trim().toLowerCase();
                if (!text || seen.has(text)) return;
                if (BLOCKED.some(b => text.includes(b))) return;
                seen.add(text);
                // หา selector ที่ stable
                const sel = el.id ? '#' + el.id
                    : (el.className && typeof el.className === 'string' && el.className.trim()
                        ? el.tagName.toLowerCase() + '.' + el.className.trim().split(/\s+/)[0]
                        : el.tagName.toLowerCase() + ':nth-of-type(' + (Array.from(el.parentNode?.children || []).indexOf(el) + 1) + ')');
                results.push({selector: sel, text: el.textContent.trim().slice(0, 40)});
            });
            return results.slice(0, 15);
        }""")

        discovered: list[str] = []
        for c in candidates[:max_clicks]:
            try:
                pre_url = page.url.split("?")[0].split("#")[0]
                el = await page.query_selector(c["selector"])
                if not el:
                    continue
                await el.click(timeout=3_000)
                await page.wait_for_timeout(400)
                post_base = page.url.split("?")[0].split("#")[0]
                if (post_base != pre_url
                        and urlparse(post_base).netloc == base_domain
                        and page.url not in discovered):
                    discovered.append(page.url)
                    logger.info(
                        f"SiteDiscovery: interaction found {page.url!r} via '{c['text']}'"
                    )
                # navigate กลับหน้าเดิมถ้า URL เปลี่ยน
                if page.url.split("?")[0].split("#")[0] != original_url.split("?")[0].split("#")[0]:
                    await page.goto(original_url, wait_until="networkidle", timeout=15_000)
            except Exception as exc:
                logger.debug(f"SiteDiscovery: interaction click skipped ({c.get('text','?')!r}): {exc!r}")
                try:
                    if page.url.split("?")[0].split("#")[0] != original_url.split("?")[0].split("#")[0]:
                        await page.goto(original_url, wait_until="networkidle", timeout=15_000)
                except Exception:
                    pass
        return discovered

    @staticmethod
    async def _visit_and_record(crawler: SFGCrawler, page: Page):
        # _visit_node returns (SFGNode, pam_tokens) — unpack and return just the node
        node, _ = await crawler._visit_node(page, None)
        return node


__all__ = ["SiteDiscovery"]
