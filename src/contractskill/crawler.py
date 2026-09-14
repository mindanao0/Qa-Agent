"""
SFG Crawler — Sprint 4 / Cluster S4-B.

BFS-based web crawler that discovers application state flow graphs (SFG) by
navigating pages, grounding them with the perception pipeline, and recording
nodes/edges in the SFGStore.

Crawl limits (all hard stops — ANY one triggers termination):
  - max_time_minutes: wall-clock time
  - max_tokens: cumulative PAM tokens from grounder
  - max_pages: number of distinct nodes visited
  - max_depth: BFS depth from seed
  - max_outgoing_per_node: edges emitted per node

Safety:
  - Edges with is_safe_action() == False get safety_flag='BLOCKED' and are
    skipped during crawl execution (but are stored so callers can inspect them).
  - BLOCKED_ACTION_PATTERNS keywords are never executed.
"""
from __future__ import annotations

import hashlib
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page, async_playwright
from pydantic import BaseModel, ConfigDict

from src.contractskill.sfg import (
    SFGEdge,
    SFGNode,
    SFGStore,
    is_safe_action,
)
from src.perception.grounder import Grounder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# M1: Named constant for navigation timeout (milliseconds)
_NAV_TIMEOUT_MS = 30_000


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class CrawlerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_time_minutes: int = 30
    max_tokens: int = 500_000
    max_pages: int = 100
    max_depth: int = 40
    max_outgoing_per_node: int = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(*parts: str) -> str:
    """Return hex SHA-256 of the concatenation of *parts*."""
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _url_path(url: str) -> str:
    """Return scheme+host+path from a URL (strips query/fragment)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _content_hash(content: str) -> str:
    """SHA-256 hex of page content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Queued item for BFS
# ---------------------------------------------------------------------------


class _QueueItem:
    """Represents a BFS frontier entry."""

    __slots__ = ("url", "depth", "parent_edge")

    def __init__(self, url: str, depth: int, parent_edge: SFGEdge | None) -> None:
        self.url = url
        self.depth = depth
        self.parent_edge: SFGEdge | None = parent_edge


# ---------------------------------------------------------------------------
# SFGCrawler
# ---------------------------------------------------------------------------


class SFGCrawler:
    """
    BFS crawler that maps an application's state flow graph.

    Usage::

        store = SFGStore()
        grounder = Grounder()
        config = CrawlerConfig(max_pages=20)
        crawler = SFGCrawler(store, grounder, config)
        root_node = await crawler.crawl("https://example.com/")
    """

    def __init__(
        self,
        sfg_store: SFGStore,
        grounder: Grounder,
        config: CrawlerConfig,
    ) -> None:
        self._store = sfg_store
        self._grounder = grounder
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    async def crawl(
        self,
        seed_url: str,
        credentials: dict | None = None,
    ) -> SFGNode:
        """
        BFS exploration from seed_url.

        Launches its own Playwright browser context.  Stops when ANY hard limit
        is reached (time / tokens / pages / depth).

        Returns the root SFGNode (seed page), which is always created regardless
        of other limits.
        """
        cfg = self._config
        start_time = time.monotonic()
        token_total: int = 0
        pages_visited: int = 0

        # node_id → SFGNode for all visited nodes
        known_nodes: dict[str, SFGNode] = {}

        # BFS queue
        queue: deque[_QueueItem] = deque()
        queue.append(_QueueItem(url=seed_url, depth=0, parent_edge=None))

        # Track which URLs are already enqueued / visited so we don't re-enqueue
        enqueued_urls: set[str] = {seed_url}

        root_node: SFGNode | None = None

        # M3: Compute token threshold once before the loop (not every iteration).
        # C1: Stop at the actual max_tokens limit (not 90% of it).
        token_threshold = cfg.max_tokens

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            # Optional login credentials
            if credentials:
                await self._handle_login(page, credentials)

            while queue:
                item = queue.popleft()

                # ── Hard limit checks ────────────────────────────────────────
                elapsed_minutes = (time.monotonic() - start_time) / 60.0
                if elapsed_minutes >= cfg.max_time_minutes:
                    logger.info(
                        f"SFGCrawler: stopping — time limit hit "
                        f"({elapsed_minutes:.1f} >= {cfg.max_time_minutes} min)"
                    )
                    break

                if pages_visited >= cfg.max_pages:
                    logger.info(
                        f"SFGCrawler: stopping — page limit hit "
                        f"({pages_visited} >= {cfg.max_pages})"
                    )
                    break

                # Stop when the cumulative PAM token budget is exhausted.
                if token_total >= token_threshold:
                    logger.info(
                        f"SFGCrawler: stopping — token limit reached "
                        f"({token_total} >= {token_threshold})"
                    )
                    break

                if item.depth > cfg.max_depth:
                    logger.debug(f"SFGCrawler: skipping depth {item.depth} > {cfg.max_depth}")
                    continue

                # ── Navigate ────────────────────────────────────────────────
                try:
                    await page.goto(item.url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
                except Exception as exc:
                    logger.warning(f"SFGCrawler: goto failed for {item.url}: {exc}")
                    continue

                # ── Visit node ───────────────────────────────────────────────
                node, pam_tokens = await self._visit_node(page, item.parent_edge)

                if node.node_id in known_nodes:
                    # We've already fully processed this state — skip.
                    # C2: Do NOT accumulate tokens for duplicate nodes.
                    logger.debug(
                        f"SFGCrawler: dedup — node {node.node_id[:12]} already known, skipping"
                    )
                    continue

                # C2: Only count tokens for genuinely new nodes.
                token_total += pam_tokens

                known_nodes[node.node_id] = node
                pages_visited += 1
                logger.info(
                    f"SFGCrawler: visited node {node.node_id[:12]} "
                    f"url={node.url} depth={item.depth} "
                    f"pages={pages_visited}/{cfg.max_pages} "
                    f"tokens={token_total}/{cfg.max_tokens}"
                )

                # Track root
                if root_node is None:
                    root_node = node

                # ── Finalize parent edge now that we have the target node_id ─
                if item.parent_edge is not None:
                    # Re-create the edge with the correct target_node_id
                    finalized_edge = SFGEdge(
                        edge_id=item.parent_edge.edge_id,
                        source_node_id=item.parent_edge.source_node_id,
                        target_node_id=node.node_id,
                        action_type=item.parent_edge.action_type,
                        locator=item.parent_edge.locator,
                        input_value=item.parent_edge.input_value,
                        safety_flag=item.parent_edge.safety_flag,
                        replay_script=item.parent_edge.replay_script,
                    )
                    self._store.upsert_edge(finalized_edge)

                # ── Discover outgoing edges from the current page ──────────
                # Re-navigate to ensure we're on the node's actual page
                edges = await self._discover_edges(page, node)

                for edge in edges:
                    # Store every edge immediately (PENDING target) for auditability
                    self._store.upsert_edge(edge)

                    # Skip blocked edges — don't traverse them
                    if edge.safety_flag == "BLOCKED":
                        logger.debug(
                            f"SFGCrawler: skipping BLOCKED edge {edge.edge_id[:12]} "
                            f"locator={edge.locator!r}"
                        )
                        continue

                    # Only enqueue navigation edges that lead to a new URL
                    if edge.action_type == "navigate" and edge.locator not in enqueued_urls:
                        enqueued_urls.add(edge.locator)
                        if item.depth + 1 <= cfg.max_depth:
                            queue.append(
                                _QueueItem(
                                    url=edge.locator,
                                    depth=item.depth + 1,
                                    parent_edge=edge,
                                )
                            )

            await browser.close()

        # Ensure we always return a root node (create minimal one if something
        # catastrophic prevented even the seed page from being visited)
        if root_node is None:
            logger.warning(
                f"SFGCrawler: root_node is None after crawl — creating stub for {seed_url}"
            )
            root_node = SFGNode(
                node_id=_sha256(seed_url, "stub"),
                url=seed_url,
                page_title="(stub — crawl failed)",
                aom_hash="stub",
                pam_content="",
                coverage_tags=[],
                outgoing_edges=[],
                discovered_at_iso=_now_iso(),
                visit_count=0,
            )
            self._store.upsert_node(root_node)

        return root_node

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _visit_node(
        self,
        page: Page,
        parent_edge: SFGEdge | None,
    ) -> tuple[SFGNode, int]:
        """
        Ground the current page and create/upsert an SFGNode.

        Returns (node, pam_tokens) so the caller can accumulate the token total
        without needing a second estimate_tokens() call.
        """
        compact_pam = await self._grounder.ground(page)

        url = page.url
        url_path = _url_path(url)

        try:
            page_title = await page.title()
        except Exception:
            page_title = "(unknown)"

        aom_hash = _content_hash(compact_pam.content)
        node_id = _sha256(url_path, aom_hash)

        # Infer coverage tags from PAM content
        coverage_tags = _infer_coverage_tags(compact_pam.content, page_title)

        node = SFGNode(
            node_id=node_id,
            url=url,
            page_title=page_title,
            aom_hash=aom_hash,
            pam_content=compact_pam.content,
            coverage_tags=coverage_tags,
            outgoing_edges=[],
            discovered_at_iso=_now_iso(),
            visit_count=1,
        )

        self._store.upsert_node(node)

        return node, compact_pam.estimated_tokens

    async def _discover_edges(
        self,
        page: Page,
        node: SFGNode,
    ) -> list[SFGEdge]:
        """
        Discover interactive elements on the current page and build SFGEdge objects.

        Finds buttons, links, inputs, selects via page.evaluate().
        Returns up to config.max_outgoing_per_node edges.
        """
        try:
            elements = await page.evaluate(_JS_FIND_INTERACTIVE)
        except Exception as exc:
            logger.warning(f"SFGCrawler._discover_edges: evaluate failed: {exc}")
            elements = []

        edges: list[SFGEdge] = []
        seen_edge_ids: set[str] = set()

        for el in elements:
            if len(edges) >= self._config.max_outgoing_per_node:
                break

            tag = (el.get("tag") or "").lower()
            role = (el.get("role") or "").lower()
            name = (el.get("name") or el.get("text") or "").strip()
            href = (el.get("href") or "").strip()
            el_type = (el.get("type") or "").lower()

            # Determine action type
            if tag == "a" and href:
                action_type = "navigate"
            elif tag in ("input", "textarea"):
                if el_type in ("checkbox", "radio"):
                    action_type = "click"
                else:
                    action_type = "fill"
            elif tag == "select":
                action_type = "select"
            else:
                action_type = "click"

            # Build a stable locator string
            locator = _build_locator(tag, role, name, href, action_type)
            if not locator:
                continue

            input_value: str | None = None
            if action_type == "fill":
                input_value = ""  # placeholder; real value set by ContractSkill planner

            # Safety check
            safe = is_safe_action(locator, input_value)
            safety_flag = "SAFE" if safe else "BLOCKED"

            # Also check href for navigate actions
            if action_type == "navigate" and not is_safe_action(href, None):
                safety_flag = "BLOCKED"

            # Build replay script
            replay_script = _build_replay_script(action_type, tag, role, name, href)

            # For navigate edges, store the href as the locator so the BFS
            # can use it as the next URL
            edge_locator = href if action_type == "navigate" else locator

            edge_id = _sha256(node.node_id, action_type, edge_locator)

            if edge_id in seen_edge_ids:
                continue
            seen_edge_ids.add(edge_id)

            edge = SFGEdge(
                edge_id=edge_id,
                source_node_id=node.node_id,
                target_node_id="PENDING",  # resolved when target is visited
                action_type=action_type,
                locator=edge_locator,
                input_value=input_value,
                safety_flag=safety_flag,
                replay_script=replay_script,
            )
            edges.append(edge)

        return edges

    # ── Login helper ─────────────────────────────────────────────────────────

    async def _handle_login(self, page: Page, credentials: dict) -> None:
        """Best-effort login using credentials dict (username, password, login_url)."""
        login_url = credentials.get("login_url")
        if not login_url:
            return
        try:
            await page.goto(login_url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            if username:
                # I3: Locator objects are always truthy; use .count() to check presence.
                username_field = page.get_by_label("Username")
                count = await username_field.count()
                if not count:
                    username_field = page.get_by_label("Email")
                await username_field.fill(username)
            if password:
                password_field = page.get_by_label("Password")
                await password_field.fill(password)
                await password_field.press("Enter")
            await page.wait_for_load_state("domcontentloaded", timeout=10_000)
        except Exception as exc:
            logger.warning(f"SFGCrawler._handle_login: login failed: {exc}")


# ---------------------------------------------------------------------------
# JavaScript helper (injected into page)
# ---------------------------------------------------------------------------

_JS_FIND_INTERACTIVE = """
() => {
    const results = [];
    const seen = new Set();

    function getRole(el) {
        return el.getAttribute('role') || el.tagName.toLowerCase();
    }

    function getName(el) {
        return (
            el.getAttribute('aria-label') ||
            el.getAttribute('title') ||
            el.getAttribute('placeholder') ||
            el.innerText?.trim().slice(0, 80) ||
            ''
        );
    }

    // Buttons and clickable elements
    document.querySelectorAll('button, [role="button"], input[type="submit"], input[type="button"], input[type="reset"]').forEach(el => {
        const key = el.tagName + '|' + getName(el);
        if (seen.has(key)) return;
        seen.add(key);
        results.push({
            tag: el.tagName.toLowerCase(),
            role: getRole(el),
            name: getName(el),
            type: el.type || '',
            href: '',
        });
    });

    // Links
    document.querySelectorAll('a[href]').forEach(el => {
        let href = el.href || '';
        // Skip anchors, javascript:, mailto:, etc.
        if (!href.startsWith('http') && !href.startsWith('/')) return;
        const key = 'a|' + href;
        if (seen.has(key)) return;
        seen.add(key);
        results.push({
            tag: 'a',
            role: 'link',
            name: getName(el) || href,
            type: '',
            href: href,
        });
    });

    // Inputs (text, email, search, number, url, tel, date, time)
    const TEXT_TYPES = new Set(['text','email','search','number','url','tel','date','time','month','week','datetime-local','']);
    document.querySelectorAll('input').forEach(el => {
        if (!TEXT_TYPES.has((el.type || '').toLowerCase())) return;
        const key = 'input|' + getName(el);
        if (seen.has(key)) return;
        seen.add(key);
        results.push({
            tag: 'input',
            role: 'textbox',
            name: getName(el),
            type: el.type || 'text',
            href: '',
        });
    });

    // Textareas
    document.querySelectorAll('textarea').forEach(el => {
        const key = 'textarea|' + getName(el);
        if (seen.has(key)) return;
        seen.add(key);
        results.push({
            tag: 'textarea',
            role: 'textbox',
            name: getName(el),
            type: '',
            href: '',
        });
    });

    // Selects
    document.querySelectorAll('select').forEach(el => {
        const key = 'select|' + getName(el);
        if (seen.has(key)) return;
        seen.add(key);
        results.push({
            tag: 'select',
            role: 'combobox',
            name: getName(el),
            type: '',
            href: '',
        });
    });

    // Checkboxes and radios
    document.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach(el => {
        const key = el.type + '|' + getName(el);
        if (seen.has(key)) return;
        seen.add(key);
        results.push({
            tag: 'input',
            role: el.type,
            name: getName(el),
            type: el.type,
            href: '',
        });
    });

    return results.slice(0, 120);
}
"""


# ---------------------------------------------------------------------------
# Locator + replay script builders
# ---------------------------------------------------------------------------


def _build_locator(
    tag: str,
    role: str,
    name: str,
    href: str,
    action_type: str,
) -> str:
    """Build a stable locator string for an element."""
    if action_type == "navigate":
        return href  # for navigate edges, locator IS the href

    if name:
        if role in ("button", "link", "checkbox", "radio", "tab", "menuitem"):
            return f'role={role}[name="{name}"]'
        if role in ("textbox", "combobox", "searchbox"):
            return f'label="{name}"'
        if tag in ("input", "textarea"):
            return f'label="{name}"'
        if tag == "select":
            return f'label="{name}"'
        return f'role={role}[name="{name}"]'

    # Fallback: use tag + role
    if tag and role:
        return f"role={role}"
    return ""


def _build_replay_script(
    action_type: str,
    tag: str,
    role: str,
    name: str,
    href: str,
) -> str:
    """Build a minimal Playwright Python replay snippet."""
    if action_type == "navigate":
        return f'page.goto("{href}")'

    if action_type == "fill":
        if name:
            return f'page.get_by_label("{name}").fill("value")'
        return 'page.get_by_role("textbox").fill("value")'

    if action_type == "select":
        if name:
            return f'page.get_by_label("{name}").select_option("option")'
        return 'page.get_by_role("combobox").select_option("option")'

    # click (default)
    if name:
        safe_role = role if role else tag
        if safe_role in ("button", "link", "checkbox", "radio", "tab", "menuitem"):
            return f'page.get_by_role("{safe_role}", name="{name}").click()'
        return f'page.get_by_text("{name}").click()'

    return f'page.get_by_role("{role or tag}").click()'


# ---------------------------------------------------------------------------
# Coverage tag inference
# ---------------------------------------------------------------------------

_FORM_KEYWORDS = frozenset({"form", "input", "login", "register", "signup", "checkout", "submit"})
_AUTH_KEYWORDS = frozenset({"login", "sign in", "signin", "auth", "password", "credentials"})
_MODAL_KEYWORDS = frozenset({"modal", "dialog", "popup", "overlay"})
_LIST_KEYWORDS = frozenset({"list", "table", "grid", "results", "items", "records"})


def _infer_coverage_tags(pam_content: str, page_title: str) -> list[str]:
    """Infer coverage tags from PAM content and page title."""
    combined = (pam_content + " " + page_title).lower()
    tags: list[str] = []

    if any(k in combined for k in _AUTH_KEYWORDS):
        tags.append("auth_required")
    if any(k in combined for k in _FORM_KEYWORDS):
        tags.append("form")
    if any(k in combined for k in _MODAL_KEYWORDS):
        tags.append("modal")
    if any(k in combined for k in _LIST_KEYWORDS):
        tags.append("list")

    return tags


__all__ = ["CrawlerConfig", "SFGCrawler"]
