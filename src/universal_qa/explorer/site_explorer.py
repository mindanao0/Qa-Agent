from __future__ import annotations

import hashlib
import re as _re
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

        candidates = await self._scanner.scan(page)
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

        if cand.is_in_iframe:
            return await self._try_click_iframe(page, url, cand, base_domain, pre_url, pre_badge)

        loc = await self._resolve_locator(page, cand)
        if loc is None:
            logger.debug(f"  no locator for '{cand.label}' ({cand.role})")
            return None
        try:
            await loc.wait_for(state="visible", timeout=2_000)
            await loc.click(timeout=3_000)
            await page.wait_for_load_state("networkidle", timeout=2_000)
        except Exception as exc:
            logger.debug(f"  click '{cand.label}' failed: {exc!r}")
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

    async def _resolve_locator(self, page: Page, cand: ElementCandidate):
        """Multi-strategy locator waterfall: role+name → label → text. No CSS."""
        async def _count(loc_fn):
            try:
                loc = loc_fn()
                if await loc.count() > 0:
                    return loc
            except Exception:
                pass
            return None

        # 1. role + exact name
        if cand.role and cand.name:
            loc = await _count(lambda: page.get_by_role(cand.role, name=cand.name).first)
            if loc:
                return loc

        # 2. role + partial name (case-insensitive, first 40 chars)
        if cand.role and cand.name and len(cand.name) > 2:
            s = cand.name[:40]
            loc = await _count(
                lambda: page.get_by_role(
                    cand.role, name=_re.compile(_re.escape(s), _re.IGNORECASE)
                ).first
            )
            if loc:
                return loc

        # 3. get_by_label (inputs/selects)
        if cand.name:
            loc = await _count(lambda: page.get_by_label(cand.name).first)
            if loc:
                return loc

        # 4. get_by_text exact
        if cand.name:
            loc = await _count(lambda: page.get_by_text(cand.name, exact=True).first)
            if loc:
                return loc

        # 5. get_by_text partial (trimmed label)
        label = (cand.name or cand.label or "")[:40]
        if label:
            loc = await _count(lambda: page.get_by_text(label).first)
            if loc:
                return loc

        return None

    async def _try_click_iframe(
        self, page: Page, url: str, cand: ElementCandidate, base_domain: str,
        pre_url: str, pre_badge: str,
    ) -> ExploredAction | None:
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            try:
                if cand.role and cand.name:
                    loc = frame.get_by_role(cand.role, name=cand.name).first
                elif cand.name:
                    loc = frame.get_by_text(cand.name, exact=True).first
                else:
                    continue
                if await loc.count() == 0:
                    continue
                await loc.wait_for(state="visible", timeout=2_000)
                await loc.click(timeout=3_000)
                await page.wait_for_load_state("networkidle", timeout=2_000)
            except Exception:
                continue

            post_url = _clean_url(page.url)
            if post_url != pre_url and urlparse(post_url).netloc == base_domain:
                logger.info(f"  → [iframe] คลิก \"{cand.label}\" ──► navigate: {post_url} ✓")
                return ExploredAction(
                    page_url=url, action_label=cand.label,
                    element_role=cand.role, element_name=cand.name,
                    element_selector=cand.selector, leads_to_url=page.url,
                )
            post_badge = await self._snapshot_state(page)
            if post_badge != pre_badge:
                logger.info(f"  → [iframe] คลิก \"{cand.label}\" ──► state เปลี่ยน")
                return ExploredAction(
                    page_url=url, action_label=cand.label,
                    element_role=cand.role, element_name=cand.name,
                    element_selector=cand.selector,
                    state_change={"snapshot": "changed"},
                )
        return None

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
