from __future__ import annotations

import hashlib
import time
from collections import deque
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page

from src.contractskill.crawler import CrawlerConfig, SFGCrawler
from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS, SFGStore
from src.perception.grounder import Grounder
from src.universal_qa.auth_manager import AuthManager
from src.universal_qa.explorer.element_scanner import ElementScanner
from src.universal_qa.explorer.form_filler import FormFiller
from src.universal_qa.explorer.nav_map import (
    ElementCandidate, ExploredAction, ExploredPage, ExplorerConfig,
    NavigationFlow, NavigationMap,
)
from src.universal_qa.explorer.session_guard import SessionGuard


def _element_key(page_url: str, role: str | None, name: str | None) -> str:
    raw = f"{page_url}|{role or ''}|{name or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_destructive(label: str) -> bool:
    low = (label or "").lower()
    return any(p in low for p in BLOCKED_ACTION_PATTERNS)


def _clean_url(url: str) -> str:
    return url.split("?")[0].split("#")[0]


# JS that locates "menu opener" toggles (hamburger/nav toggles) that hide nav
# behind a click. Reading className/id/aria here is internal DOM scanning, NOT a
# Playwright locator — it only returns accessible names/roles for get_by_* use.
_MENU_OPENER_JS = """
() => {
    const out = [];
    const seen = new Set();
    const NAME_RE = /\\b(menu|open menu|navigation|nav)\\b|เมนู/i;
    const CLS_RE = /(burger|hamburger|menu-btn|menu_btn|nav-toggle|navbar-toggle)/i;
    const SEL = 'button, a, [role=button], [aria-haspopup], [aria-expanded]';
    function isVisible(el) {
        const s = window.getComputedStyle(el);
        if (s.display === 'none' || s.visibility === 'hidden' || +s.opacity === 0)
            return false;
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
    }
    function accName(el) {
        return (el.getAttribute('aria-label')
            || el.getAttribute('title')
            || el.textContent || el.value || '').trim().slice(0, 60);
    }
    function inferRole(el) {
        const r = el.getAttribute('role');
        if (r) return r;
        const t = el.tagName;
        if (t === 'BUTTON') return 'button';
        if (t === 'A') return 'link';
        return null;
    }
    document.querySelectorAll(SEL).forEach(el => {
        if (!isVisible(el)) return;
        const name = accName(el);
        const cls = (el.className && typeof el.className === 'string') ? el.className : '';
        const idv = el.id || '';
        const haspopup = el.getAttribute('aria-haspopup');
        const expanded = el.getAttribute('aria-expanded');
        const isOpener =
            NAME_RE.test(name)
            || CLS_RE.test(cls) || CLS_RE.test(idv)
            || (haspopup && haspopup !== 'false')
            || expanded === 'false';
        if (!isOpener) return;
        const role = inferRole(el);
        const key = (role || '') + '|' + name + '|' + idv;
        if (seen.has(key)) return;
        seen.add(key);
        out.push({name, role});
    });
    return out.slice(0, 3);
}
"""


class SiteExplorer:
    """Phase 3: interaction-based exploration that builds a NavigationMap.

    For each discovered URL, clicks every interactive element sequentially,
    records action->page transitions, fills forms, recovers from session loss,
    and auto-detects multi-step flows.
    """

    def __init__(
        self,
        auth: AuthManager,
        config: ExplorerConfig | None = None,
        client=None,
    ) -> None:
        self._auth = auth
        self._cfg = config or ExplorerConfig()
        self._scanner = ElementScanner()
        self._filler = FormFiller(client=client)
        self._guard = SessionGuard(auth=auth)
        self._sfg_store: SFGStore | None = None
        self._crawler: SFGCrawler | None = None

    async def explore(self, page: Page, discovered_urls: list[str]) -> NavigationMap:
        import tempfile, pathlib, shutil
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="uqa_explore_"))
        sfg_path = tmp_dir / "sfg.db"
        self._sfg_store = SFGStore(db_path=sfg_path)
        self._crawler = SFGCrawler(self._sfg_store, Grounder(), CrawlerConfig())
        try:
            base_domain = urlparse(page.url).netloc
            base_url = f"{urlparse(page.url).scheme}://{base_domain}"
            start = time.monotonic()

            pages: list[ExploredPage] = []
            flows: list[NavigationFlow] = []
            visited_actions: set[str] = set()
            visit_count: dict[str, int] = {}
            queued_urls: set[str] = set(discovered_urls)
            queue: deque[tuple[str, int]] = deque((u, 0) for u in discovered_urls)

            while queue and len(pages) < self._cfg.max_pages:
                url, depth = queue.popleft()
                curl = _clean_url(url)
                if depth > self._cfg.max_depth:
                    continue
                if visit_count.get(curl, 0) >= self._cfg.max_visits_per_url:
                    continue
                if (time.monotonic() - start) / 60.0 >= self._cfg.explore_timeout_min:
                    logger.info("SiteExplorer: timeout reached")
                    break
                visit_count[curl] = visit_count.get(curl, 0) + 1

                try:
                    explored, new_urls, page_flows = await self._explore_page(
                        page, url, depth, base_domain, visited_actions
                    )
                except Exception as exc:
                    logger.warning(f"SiteExplorer: page {url} failed — {exc!r}")
                    continue

                pages.append(explored)
                flows.extend(page_flows)
                logger.info(
                    f"  Pages found: {len(pages)} | Actions: {len(explored.actions)} | Flows: {len(flows)}"
                )
                for nu in new_urls:
                    ncu = _clean_url(nu)
                    if urlparse(ncu).netloc == base_domain and nu not in queued_urls:
                        queued_urls.add(nu)
                        queue.append((nu, depth + 1))

            return NavigationMap(
                base_url=base_url, pages=pages, flows=flows,
                explored_at_iso="",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _explore_page(
        self, page: Page, url: str, depth: int, base_domain: str,
        visited_actions: set[str],
    ) -> tuple[ExploredPage, list[str], list[NavigationFlow]]:
        if _clean_url(page.url) != _clean_url(url):
            await page.goto(url, wait_until="networkidle", timeout=30_000)
        else:
            await page.wait_for_load_state("domcontentloaded", timeout=30_000)

        if await self._guard.is_session_lost(page):
            await self._guard.recover(page)
            await page.goto(url, wait_until="networkidle", timeout=30_000)

        logger.info(f"[EXPLORE] {url}")

        title, pam = await self._record_page_node(page)
        snapshot = await self._snapshot_state(page)

        # Reveal hidden navigation (hamburger/menu toggles) before scanning so the
        # scanner sees logout/about/settings links that are display:none until opened.
        await self._expand_menus(page, url)

        candidates = await self._scanner.scan(page)
        seen_keys: set[str] = {
            _element_key(url, c.role, c.name or c.label) for c in candidates
        }
        rescanned = False  # re-scan at most once per page to avoid loops
        actions: list[ExploredAction] = []
        new_urls: list[str] = []

        _page_start = time.monotonic()
        for cand in candidates:
            if time.monotonic() - _page_start > 45.0:
                logger.info(f"  Per-page time cap reached — stopping element scan for {url}")
                break
            key = _element_key(url, cand.role, cand.name or cand.label)
            if key in visited_actions:
                continue
            destructive = _is_destructive(cand.label)
            if destructive and not self._cfg.allow_destructive:
                actions.append(ExploredAction(
                    page_url=url, action_label=cand.label,
                    element_role=cand.role, element_name=cand.name,
                    element_selector=cand.selector, is_destructive=True,
                ))
                visited_actions.add(key)
                continue

            action = await self._try_click(page, url, cand, base_domain)
            if action is not None:
                actions.append(action)
                visited_actions.add(key)
                if action.leads_to_url:
                    new_urls.append(action.leads_to_url)
                # If a non-navigating click on a menu/expander revealed new
                # elements (state change, no URL change), re-scan once and append
                # any NEW candidates (dedup by _element_key).
                if (
                    not rescanned
                    and action.leads_to_url is None
                    and action.state_change is not None
                    and self._looks_like_expander(cand)
                ):
                    rescanned = True
                    try:
                        fresh = await self._scanner.scan(page)
                    except Exception:
                        fresh = []
                    added = 0
                    for fc in fresh:
                        fkey = _element_key(url, fc.role, fc.name or fc.label)
                        if fkey in seen_keys:
                            continue
                        seen_keys.add(fkey)
                        candidates.append(fc)
                        added += 1
                    if added:
                        logger.info(f"  ↻ re-scan after expander: +{added} candidate(s)")
            if _clean_url(page.url) != _clean_url(url):
                try:
                    await page.goto(url, wait_until="networkidle", timeout=15_000)
                except Exception:
                    pass

        explored = ExploredPage(
            url=url, title=title, pam_content=pam,
            actions=actions, state_snapshot=snapshot,
        )
        flows = self._detect_flows(actions)
        return explored, new_urls, flows

    async def _try_click(
        self, page: Page, url: str, cand: ElementCandidate, base_domain: str,
    ) -> ExploredAction | None:
        pre_url = _clean_url(page.url)
        pre_badge = await self._snapshot_state(page)
        try:
            loc = self._locate(page, cand)
            await loc.click(timeout=1_500)
            await page.wait_for_load_state("networkidle", timeout=1_000)
        except Exception as exc:
            logger.debug(f"  click '{cand.label}' skipped: {exc!r}")
            return None

        post_url = _clean_url(page.url)
        if post_url != pre_url and urlparse(post_url).netloc == base_domain:
            logger.info(f"  → คลิก \"{cand.label}\" ──► navigate: {post_url} ✓")
            return ExploredAction(
                page_url=url, action_label=cand.label,
                element_role=cand.role, element_name=cand.name,
                element_selector=cand.selector, leads_to_url=page.url,
            )

        post_badge = await self._snapshot_state(page)
        if post_badge != pre_badge:
            logger.info(f"  → คลิก \"{cand.label}\" ──► state เปลี่ยน")
            return ExploredAction(
                page_url=url, action_label=cand.label,
                element_role=cand.role, element_name=cand.name,
                element_selector=cand.selector,
                state_change={"snapshot": "changed"},
            )
        return None

    @staticmethod
    def _looks_like_expander(cand: ElementCandidate) -> bool:
        text = f"{cand.label or ''} {cand.name or ''}".lower()
        return any(k in text for k in ("menu", "navigation", "nav", "เมนู"))

    async def _expand_menus(self, page: Page, url: str) -> None:
        """Click up to ~3 visible 'menu opener' toggles to reveal hidden nav.

        These toggles only flip visibility (display:none → visible); they should
        not navigate. If a click changes the URL, we navigate back to `url`.
        Locators stay get_by_role/get_by_text only (project rule). Errors swallowed.
        """
        try:
            openers = await page.evaluate(_MENU_OPENER_JS)
        except Exception as exc:
            logger.debug(f"  _expand_menus: scan skipped — {exc!r}")
            return
        for op in openers[:3]:
            name = (op.get("name") or "").strip()
            role = op.get("role")
            if not name and not role:
                continue
            try:
                if role and name:
                    loc = page.get_by_role(role, name=name).first
                elif name:
                    loc = page.get_by_text(name).first
                else:
                    continue
                await loc.click(timeout=1_500)
                await page.wait_for_timeout(150)
                logger.info(f"  ☰ เปิดเมนู \"{name or role}\" เพื่อเผยลิงก์ที่ซ่อนอยู่")
            except Exception as exc:
                logger.debug(f"  _expand_menus: click '{name or role}' skipped: {exc!r}")
                continue
            # Menu toggles shouldn't navigate; if one did, go back to the page.
            if _clean_url(page.url) != _clean_url(url):
                try:
                    await page.goto(url, wait_until="networkidle", timeout=15_000)
                except Exception:
                    pass

    @staticmethod
    def _locate(page: Page, cand: ElementCandidate):
        # ARIA role+name primary, then text. No CSS selectors (project rule).
        if cand.role and cand.name:
            return page.get_by_role(cand.role, name=cand.name).first
        if cand.name:
            return page.get_by_text(cand.name).first
        return page.get_by_text(cand.label).first

    async def _snapshot_state(self, page: Page) -> str:
        try:
            return await page.evaluate(
                "() => JSON.stringify({ls: {...localStorage}, ss: {...sessionStorage}})"
            )
        except Exception:
            return "{}"

    async def _record_page_node(self, page: Page) -> tuple[str, str]:
        """Ground the page via SFGCrawler._visit_node; return (title, pam_content)."""
        try:
            node, _ = await self._crawler._visit_node(page, None)
            return node.page_title, node.pam_content
        except Exception as exc:
            logger.warning(f"SiteExplorer: grounding failed — {exc!r}")
            try:
                return (await page.title()) or "", ""
            except Exception:
                return "", ""

    def _detect_flows(self, actions: list[ExploredAction]) -> list[NavigationFlow]:
        nav_actions = [a for a in actions if a.leads_to_url]
        flows: list[NavigationFlow] = []
        seen_dests: set[str] = set()
        for i, action in enumerate(nav_actions):
            dest = action.leads_to_url or ""
            if dest and dest not in seen_dests:
                seen_dests.add(dest)
                flows.append(self._build_flow([action], idx=i))
        if len(nav_actions) >= 2:
            first_dest = nav_actions[0].leads_to_url or ""
            last_dest = nav_actions[-1].leads_to_url or ""
            if first_dest != last_dest:
                flows.append(self._build_flow(nav_actions, idx=len(nav_actions)))
        return flows

    def _build_flow(self, steps: list[ExploredAction], idx: int) -> NavigationFlow:
        start = steps[0].page_url
        end = steps[-1].leads_to_url or steps[-1].page_url
        flow_id = "flow_" + hashlib.sha256(f"{start}{end}{idx}".encode()).hexdigest()[:8]
        start_path = urlparse(start).path or "/"
        end_path = urlparse(end).path or "/"
        name = f"{start_path} → {end_path}"
        return NavigationFlow(
            flow_id=flow_id, name=name, steps=steps,
            start_url=start, end_url=end,
        )


__all__ = ["SiteExplorer"]
