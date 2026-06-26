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
from src.universal_qa.explorer.form_filler import FormFiller


class SiteDiscovery:
    """BFS site crawler using an already-authenticated shared Page.

    Uses SFGCrawler._visit_node directly (not crawl()) so the caller's
    authenticated BrowserContext is preserved throughout.
    """

    def __init__(self, max_pages: int = 50, max_depth: int = 6) -> None:
        self.max_pages = max_pages
        self.max_depth = max_depth
        self._form_filler = FormFiller()  # dummy-fill only (no LLM client → no GPU)

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
                    # domcontentloaded เร็วกว่า networkidle และไม่ hang บน heavy sites
                    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    # รอ JS render navigation (Magento/SPA mega-menu โหลดหลัง DOMContentLoaded)
                    await page.wait_for_timeout(1_500)
                else:
                    await page.wait_for_load_state("domcontentloaded", timeout=30_000)

                # Detect Cloudflare challenge — รอให้ผ่านก่อน extract links
                _cf_title = (await page.title()).lower()
                if any(kw in _cf_title for kw in ("cloudflare", "just a moment", "attention required", "checking your")):
                    logger.info(f"SiteDiscovery: Cloudflare challenge at {url} — waiting 6s")
                    await page.wait_for_timeout(6_000)
                    await page.wait_for_load_state("domcontentloaded", timeout=15_000)

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
                    # 3. Form-progression: กรอกฟอร์มด้วย dummy แล้วกดปุ่ม advance
                    #    (continue/next/proceed) เพื่อทะลุ multi-step flow เช่น checkout
                    form_links = await self._progress_forms(page, url, base_domain)
                    all_links = links + interaction_links + form_links
                    logger.debug(
                        f"SiteDiscovery: {len(links)} href + {len(interaction_links)} interaction links on {url}"
                    )
                    for link in all_links:
                        clean = link.split("?")[0].split("#")[0]
                        link_netloc = urlparse(clean).netloc
                        # normalize www prefix: www.example.com == example.com
                        if (link_netloc.removeprefix("www.") == base_domain.removeprefix("www.")
                                and clean not in visited):
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
            "add to cart", "remove", "delete", "login", "logout",
            "register", "submit", "send", "buy", "purchase", "sign up", "sign in",
            "finish", "place order", "pay now", "confirm order",
            "เพิ่มลงตะกร้า", "ลบ", "เข้าสู่ระบบ", "ออกจากระบบ",
        })

        candidates: list[dict] = await page.evaluate("""() => {
            // บล็อกเฉพาะ action ที่ "เปลี่ยนข้อมูลจริง" — checkout เป็นแค่ navigation ไปหน้า form
            // (ไม่ใช่การยืนยันคำสั่งซื้อ) จึงค้นพบได้; ส่วนที่ commit จริง (finish/pay/place order) ยังบล็อก
            const BLOCKED = ['add to cart','remove','delete','login',
                             'logout','register','submit','send','buy','purchase',
                             'finish','place order','pay now','confirm order'];
            const seen = new Set();
            const results = [];
            document.querySelectorAll('a, button, input[type=submit], [role="link"], [role="button"]').forEach(el => {
                const href = el.getAttribute('href');
                // ข้ามถ้า href เป็น URL จริงหรือ path จริง (จัดการโดย href extraction แล้ว)
                if (href && href.startsWith('http')) return;
                if (href && href.startsWith('/') && href.length > 1 && !href.startsWith('/#')) return;
                // label: visible text ก่อน, ถ้าว่าง (เช่น icon cart/account ล้วน) ใช้
                // aria-label/title/data-test/id/class แทน เพื่อให้ icon-only links ถูกค้นพบ
                let text = (el.textContent || '').trim().toLowerCase();
                if (!text) {
                    text = (el.getAttribute('aria-label') || el.getAttribute('title')
                            || el.getAttribute('data-test') || el.id
                            || (typeof el.className === 'string' ? el.className : '')
                           ).trim().toLowerCase().replace(/[-_]+/g, ' ');
                }
                if (!text || seen.has(text)) return;
                if (BLOCKED.some(b => text.includes(b))) return;
                seen.add(text);
                // หา selector ที่ stable
                const sel = el.id ? '#' + el.id
                    : (el.className && typeof el.className === 'string' && el.className.trim()
                        ? el.tagName.toLowerCase() + '.' + el.className.trim().split(/\s+/)[0]
                        : el.tagName.toLowerCase() + ':nth-of-type(' + (Array.from(el.parentNode?.children || []).indexOf(el) + 1) + ')');
                results.push({selector: sel, text: (el.textContent.trim() || text).slice(0, 40)});
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

    async def _progress_forms(
        self, page: Page, original_url: str, base_domain: str
    ) -> list[str]:
        """กรอกฟอร์มด้วย dummy data แล้วกดปุ่ม advance เพื่อทะลุหน้าถัดไปของ multi-step flow.

        ปลอดภัย: คลิกเฉพาะปุ่มที่เป็น "ไปต่อ" (continue/next/proceed) เท่านั้น —
        ไม่แตะ finish/pay/place order/register/login (กันการ commit ข้อมูลจริง).
        BFS จะ enqueue หน้าใหม่ที่เจอ แล้ว _progress_forms ทำงานซ้ำ → chain ทั้ง flow.
        """
        # หาปุ่ม advance ตัวแรก (visible) ที่ข้อความตรงกับ allowlist
        advance = await page.evaluate("""() => {
            const ADVANCE = ['continue','next','proceed','ถัดไป','ต่อไป','ดำเนินการต่อ'];
            const BLOCKED = ['finish','pay','place order','confirm order','submit order',
                             'register','sign up','login','sign in'];
            const els = document.querySelectorAll(
                'button, input[type=submit], a, [role="button"]');
            for (const el of els) {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;  // ต้อง visible
                const t = ((el.textContent || el.value || '')).trim().toLowerCase();
                if (!t) continue;
                if (BLOCKED.some(b => t.includes(b))) continue;
                if (!ADVANCE.some(a => t.includes(a))) continue;
                const sel = el.id ? '#' + el.id
                    : (el.getAttribute('data-test') ? '[data-test="' + el.getAttribute('data-test') + '"]'
                       : (el.getAttribute('name') ? el.tagName.toLowerCase() + '[name="' + el.getAttribute('name') + '"]'
                          : null));
                if (!sel) continue;
                return {selector: sel, text: (el.textContent || el.value || '').trim().slice(0, 40)};
            }
            return null;
        }""")
        if not advance:
            return []

        discovered: list[str] = []
        try:
            pre_url = page.url.split("?")[0].split("#")[0]
            filled = await self._form_filler.fill_with_dummy(page)
            el = await page.query_selector(advance["selector"])
            if not el:
                return []
            await el.click(timeout=3_000)
            await page.wait_for_timeout(500)
            post_url = page.url.split("?")[0].split("#")[0]
            if (post_url != pre_url
                    and urlparse(post_url).netloc == base_domain
                    and page.url not in discovered):
                discovered.append(page.url)
                logger.info(
                    f"SiteDiscovery: form-progress found {page.url!r} via "
                    f"'{advance['text']}' (filled {filled} fields)"
                )
            # กลับหน้าเดิม
            if page.url.split("?")[0].split("#")[0] != original_url.split("?")[0].split("#")[0]:
                await page.goto(original_url, wait_until="networkidle", timeout=15_000)
        except Exception as exc:
            logger.debug(f"SiteDiscovery: form-progress skipped — {exc!r}")
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
