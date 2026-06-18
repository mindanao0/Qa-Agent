"""SFG state-traversal explorer — Phase E1.

Fixes the "explorer only does one page" bug by replacing URL-BFS with a real
STATE-flow-graph crawl: from each state, extract interactive elements, click the
unvisited ones, snapshot the resulting state, dedup by a multi-signal signature
(NOT url alone), and recurse — persisting nodes + edges (with replay scripts)
into the SFGStore.

Design choices (within the project invariants):
  * State restoration = replay-from-seed. To expand a state we re-navigate to the
    seed and replay its recorded click path. Deterministic and bounded (no
    fragile browser back/forward, no shared state across tasks).
  * Multi-signal dedup. node identity = (url_path, normalized-DOM structural hash).
    A new observation that differs in DOM hash but whose interactive a11y
    (role,name) multiset is within `a11y_threshold` Jaccard distance of a known
    same-url node is MERGED (not counted as new) — keeps dedup_precision high
    instead of exploding on volatile DOM. (Reconciles the spec's DOM-hash +
    a11y-distance into a high-precision rule.)
  * Safety: BLOCKED_ACTION_PATTERNS are never clicked; recorded as blocked.
  * Circuit breakers: max_depth, max_actions_per_node, max_states, time budget.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlsplit

from loguru import logger
from playwright.async_api import Page

from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS, SFGEdge, SFGNode, SFGStore
from src.universal_qa.explorer.element_scanner import ElementScanner
from src.universal_qa.explorer.nav_map import ElementCandidate

# Structural DOM signature: tag(+role) skeleton of the body, ignoring volatile
# text / ids / attributes. Stable across content churn, sensitive to structure.
_DOM_SIG_JS = """() => {
  const parts = [];
  const els = document.querySelectorAll('body *');
  for (let i = 0; i < els.length && i < 4000; i++) {
    const el = els[i];
    const role = el.getAttribute && el.getAttribute('role');
    parts.push(el.tagName.toLowerCase() + (role ? (':' + role) : ''));
  }
  return parts.join(',');
}"""


def _is_blocked(label: str) -> bool:
    low = (label or "").lower()
    return any(p in low for p in BLOCKED_ACTION_PATTERNS)


def _url_path(url: str) -> str:
    sp = urlsplit(url)
    return (sp.path or "/").rstrip("/") or "/"


class SFGTraversalExplorer:
    def __init__(
        self,
        store: SFGStore,
        *,
        max_depth: int = 40,
        max_actions_per_node: int = 60,
        max_states: int = 150,
        time_budget_s: float = 180.0,
        a11y_threshold: float = 0.05,
    ) -> None:
        self._store = store
        self._scanner = ElementScanner()
        self.max_depth = max_depth
        self.max_actions = max_actions_per_node
        self.max_states = max_states
        self.time_budget_s = time_budget_s
        self.a11y_threshold = a11y_threshold
        # node_id -> (url_path, frozenset[(role,name)]) for fuzzy merge
        self._sigs: dict[str, tuple[str, frozenset]] = {}
        # telemetry
        self.blocked = 0
        self.click_attempts = 0
        self.click_fail = 0
        self.observations = 0
        self.merges = 0  # fuzzy/exact re-recognitions (dedup hits)

    # ── signature + dedup ────────────────────────────────────────────────
    async def _signature(self, page: Page):
        try:
            dom = await page.evaluate(_DOM_SIG_JS)
        except Exception:
            dom = ""
        dom_hash = hashlib.sha256(dom.encode()).hexdigest()[:16]
        try:
            cands = await self._scanner.scan(page)
        except Exception:
            cands = []
        a11y = frozenset((c.role or "", (c.name or c.label or "")[:40]) for c in cands)
        return _url_path(page.url), dom_hash, a11y, cands

    def _resolve(self, url_path: str, dom_hash: str, a11y: frozenset):
        # Content-aware identity: url path + structural DOM + interactive a11y
        # (role,name) multiset — so in-page state changes (button toggles, cart
        # badge, enable/disable) register as distinct states.
        a11y_key = "|".join(sorted(f"{r}:{n}" for r, n in a11y))
        primary = hashlib.sha256(
            f"{url_path}|{dom_hash}|{a11y_key}".encode()
        ).hexdigest()[:16]
        if primary in self._sigs:
            return primary, False
        # Fuzzy merge only near-identical a11y on the same path (volatile churn),
        # keeping dedup_precision high without collapsing genuine states.
        for nid, (npath, sig) in self._sigs.items():
            if npath != url_path:
                continue
            union = len(a11y | sig) or 1
            dist = 1.0 - (len(a11y & sig) / union)
            if dist <= self.a11y_threshold:
                return nid, False  # merge — re-seen state, not new
        return primary, True

    async def _record(self, page: Page, title: str = ""):
        self.observations += 1
        url_path, dom_hash, a11y, cands = await self._signature(page)
        nid, is_new = self._resolve(url_path, dom_hash, a11y)
        if is_new:
            self._sigs[nid] = (url_path, a11y)
            self._store.upsert_node(SFGNode(
                node_id=nid, url=page.url, page_title=title,
                aom_hash=dom_hash, pam_content="", coverage_tags=[],
                outgoing_edges=[], discovered_at_iso=datetime.now(timezone.utc).isoformat(),
                visit_count=1,
            ))
        else:
            self.merges += 1
        return nid, is_new, cands

    async def signature_hash(self, page: Page) -> str:
        """Deterministic primary node-id for the current page (no side effects).

        Used by the eval harness's dedup probe: re-observing the same state must
        yield the same hash (dedup_precision).
        """
        url_path, dom_hash, a11y, _ = await self._signature(page)
        a11y_key = "|".join(sorted(f"{r}:{n}" for r, n in a11y))
        return hashlib.sha256(f"{url_path}|{dom_hash}|{a11y_key}".encode()).hexdigest()[:16]

    # ── click + replay ───────────────────────────────────────────────────
    @staticmethod
    def _locate(page: Page, c: ElementCandidate):
        if c.role and c.name:
            return page.get_by_role(c.role, name=c.name).first
        if c.name:
            return page.get_by_text(c.name).first
        return page.get_by_text(c.label).first

    async def _click(self, page: Page, c: ElementCandidate) -> bool:
        self.click_attempts += 1
        try:
            await self._locate(page, c).click(timeout=2_000)
            await page.wait_for_timeout(350)
            return True
        except Exception as exc:
            self.click_fail += 1
            logger.debug(f"click '{c.label}' failed: {exc!r}")
            return False

    async def _replay(self, page: Page, seed: str, path: list[dict]) -> bool:
        try:
            await page.goto(seed, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(300)
        except Exception:
            return False
        for step in path:
            cand = ElementCandidate(label=step["label"], role=step.get("role"),
                                    name=step.get("name"), selector=step.get("selector"))
            if not await self._click(page, cand):
                return False
        return True

    @staticmethod
    def _script(seed: str, path: list[dict]) -> str:
        lines = [f"page.goto({seed!r})"]
        for s in path:
            if s.get("role") and s.get("name"):
                lines.append(f"page.get_by_role({s['role']!r}, name={s['name']!r}).first.click()")
            else:
                lines.append(f"page.get_by_text({(s.get('name') or s['label'])!r}).first.click()")
        return json.dumps(lines)

    # ── main loop ────────────────────────────────────────────────────────
    async def explore(self, page: Page, seed_url: str) -> SFGStore:
        start = time.monotonic()
        try:
            await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(500)
        except Exception as exc:
            logger.warning(f"SFGTraversal: seed goto failed {exc!r}")
            return self._store

        title = ""
        try:
            title = await page.title()
        except Exception:
            pass
        root_id, _, _ = await self._record(page, title)
        queue: deque[tuple[str, list[dict], int]] = deque([(root_id, [], 0)])
        expanded: set[str] = set()

        while queue:
            if self._store.node_count() >= self.max_states:
                logger.info("SFGTraversal: max_states reached"); break
            if (time.monotonic() - start) >= self.time_budget_s:
                logger.info("SFGTraversal: time budget reached"); break
            node_id, path, depth = queue.popleft()
            if node_id in expanded or depth > self.max_depth:
                continue
            expanded.add(node_id)

            if not await self._replay(page, seed_url, path):
                continue
            cur_id, _, cands = await self._record(page)

            for cand in cands[: self.max_actions]:
                if _is_blocked(cand.label):
                    self.blocked += 1
                    continue
                # restore to this state before each action attempt
                if not await self._replay(page, seed_url, path):
                    break
                src_id, _, _ = await self._record(page)
                if not await self._click(page, cand):
                    continue
                tgt_id, is_new, _ = await self._record(page)
                if tgt_id == src_id:
                    continue  # no state change
                step = {"label": cand.label, "role": cand.role,
                        "name": cand.name, "selector": cand.selector}
                edge_id = hashlib.sha256(
                    f"{src_id}|{cand.role}|{cand.name or cand.label}".encode()
                ).hexdigest()[:16]
                new_path = path + [step]
                self._store.upsert_edge(SFGEdge(
                    edge_id=edge_id, source_node_id=src_id, target_node_id=tgt_id,
                    action_type="click", locator=f"{cand.role or 'element'}:{cand.name or cand.label}",
                    input_value=None, safety_flag="SAFE",
                    replay_script=self._script(seed_url, new_path),
                ))
                if is_new and depth + 1 <= self.max_depth:
                    queue.append((tgt_id, new_path, depth + 1))

        logger.info(
            f"SFGTraversal: states={self._store.node_count()} edges={self._store.edge_count()} "
            f"blocked={self.blocked} clicks={self.click_attempts}/fail={self.click_fail} "
            f"obs={self.observations} dedup_hits={self.merges}"
        )
        return self._store


__all__ = ["SFGTraversalExplorer"]
