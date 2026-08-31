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
import re
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlsplit

from loguru import logger
from playwright.async_api import Page

from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS, SFGEdge, SFGNode, SFGStore
from src.universal_qa.explorer.element_scanner import ElementScanner
from src.universal_qa.explorer.form_filler import _dummy_for_type
from src.universal_qa.explorer.nav_map import ElementCandidate

# Fill-vs-click classification (Phase F1)
_FILL_ROLES = frozenset({"textbox", "searchbox", "combobox"})
_SUBMIT_KW = ("submit", "login", "log in", "sign in", "signin", "sign up",
              "signup", "search", "send", "save", "continue", "next", "register")
# F1 auto-submits ONLY clearly side-effect-free forms (login / search / filter).
# Registration / contact / generic submit are NOT auto-submitted (they create
# accounts / send messages). F2 hardens this further with an endpoint heuristic.
_SAFE_SUBMIT_KW = ("login", "log in", "sign in", "signin", "search", "filter", "apply")
# Fields that must NEVER receive synthetic data (PII / financial) — Phase F1 invariant
_PII_FINANCIAL_KW = ("card", "cvv", "cvc", "ssn", "social security", "account number",
                     "routing", "iban", "credit", "debit", "amount", "salary", "income")

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


def _is_blocked(label: str, safe_mode: bool = True) -> bool:
    if not safe_mode:
        return False
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
        safe_mode: bool = True,
    ) -> None:
        self._store = store
        self._scanner = ElementScanner()
        # safe_mode=True (default): never click/submit BLOCKED_ACTION_PATTERNS
        # labels or endpoints, and only auto-fill+submit login/search/filter/apply
        # forms. safe_mode=False: lets the crawler carry a form all the way to
        # completion (checkout, registration, ...) — only use against known
        # sandbox/test targets, never against a production or unknown site.
        self.safe_mode = safe_mode
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
        self.healed_l1 = 0  # clicks rescued by Layer-1 (semantic/fuzzy, no LLM)
        self.compound_fills = 0  # forms filled+submitted (F1)
        self.blocked_submits = 0  # form submits refused for safety (F2)
        self.observations = 0
        self.merges = 0  # fuzzy/exact re-recognitions (dedup hits)

    # ── signature + dedup ────────────────────────────────────────────────
    @staticmethod
    def _norm_name(s: str) -> str:
        # strip volatile numeric / price / date / time tokens so re-seeing a
        # state (with rotating ads/counts) stays stable
        s = re.sub(r"\d[\d,.:/$%+-]*", "", (s or "").lower())
        return re.sub(r"\s+", " ", s).strip()[:40]

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
        # STABLE MULTISET: nav + form/button controls (priority<=1), volatile
        # tokens stripped, DUPLICATES PRESERVED (sorted tuple). Counts matter —
        # "1 item in cart" (5 add + 1 remove) differs from "2 in cart" (4 add +
        # 2 remove) — so in-page state changes register; rotating content/ads
        # (priority 2) and numeric churn are excluded → re-seen state is stable.
        a11y = tuple(sorted(
            (c.role or "", self._norm_name(c.name or c.label or ""))
            for c in cands if c.priority <= 1
        ))
        return _url_path(page.url), dom_hash, a11y, cands

    def _primary(self, url_path: str, a11y: tuple) -> str:
        a11y_key = "|".join(f"{r}:{n}" for r, n in a11y)
        return hashlib.sha256(f"{url_path}|{a11y_key}".encode()).hexdigest()[:16]

    def _resolve(self, url_path: str, dom_hash: str, a11y: tuple):
        # Identity = url path + stable interactive a11y multiset (counts included;
        # NOT the volatile structural DOM hash). Exact match — normalization +
        # priority filter already remove the churn that caused false-new states.
        primary = self._primary(url_path, a11y)
        return primary, (primary not in self._sigs)

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
        url_path, _dom_hash, a11y, _ = await self._signature(page)
        return self._primary(url_path, a11y)

    # ── click + replay ───────────────────────────────────────────────────
    @staticmethod
    def _fill_value(cand: ElementCandidate, creds: dict) -> str | None:
        """Value to fill a field (Phase F1). PII/financial-safe; user/pass only from config creds."""
        name = (cand.name or cand.label or "").lower()
        if any(k in name for k in _PII_FINANCIAL_KW):
            return None  # NEVER inject synthetic data into PII/financial fields
        if any(k in name for k in ("password", "passwd", "pwd")):
            return creds.get("password")  # only a real test cred (None -> skip)
        if any(k in name for k in ("username", "user name", "userid", "user id", "login")):
            return creds.get("username")
        if "email" in name or "e-mail" in name:
            cu = creds.get("username") or ""
            return cu if "@" in cu else "test@example.com"
        if cand.role == "searchbox" or "search" in name:
            return "test"
        return _dummy_for_type("text")

    async def _action_blocked(self, page: Page, cand: ElementCandidate) -> bool:
        """True if the submit button's <form> action targets a BLOCKED endpoint
        (transfer/payment/delete/logout) — defense beyond button text (F2)."""
        if not self.safe_mode:
            return False
        name = (cand.name or cand.label or "").strip().lower()
        if not name:
            return False
        try:
            action = await page.evaluate(
                """(txt) => {
                    const els = [...document.querySelectorAll('button,input[type=submit],[role=button]')];
                    const b = els.find(e => ((e.innerText||e.value||'').trim().toLowerCase()).includes(txt));
                    const f = b && b.closest('form');
                    return f ? String(f.getAttribute('action') || f.action || '') : '';
                }""", name)
        except Exception:
            return False
        return any(p in (action or "").lower() for p in BLOCKED_ACTION_PATTERNS)

    @staticmethod
    def _locate(page: Page, c: ElementCandidate):
        if c.role and c.name:
            return page.get_by_role(c.role, name=c.name).first
        if c.name:
            return page.get_by_text(c.name).first
        return page.get_by_text(c.label).first

    @staticmethod
    def _similar(a: str, b: str) -> float:
        a, b = (a or "").strip().lower(), (b or "").strip().lower()
        if not a or not b:
            return 0.0
        try:
            import jellyfish
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            import difflib
            return difflib.SequenceMatcher(None, a, b).ratio()

    def _candidate_locators(self, page: Page, c: ElementCandidate):
        """Semantic locators in priority order: role > label > placeholder > text > test_id."""
        name = c.name or c.label
        if c.role and name:
            yield page.get_by_role(c.role, name=name).first
        if name:
            yield page.get_by_label(name).first
            yield page.get_by_placeholder(name).first
            yield page.get_by_text(name).first
            yield page.get_by_test_id(name).first

    async def _fuzzy_locate(self, page: Page, c: ElementCandidate):
        """Layer-1 heal (no LLM): Jaro-Winkler match the target name against the
        page's real interactive-element names; click the best >= 0.85."""
        target = (c.name or c.label or "").strip()
        if not target:
            return None
        try:
            cands = await self._scanner.scan(page)
        except Exception:
            return None
        best, best_score = None, 0.0
        for cc in cands:
            nm = cc.name or cc.label or ""
            s = self._similar(target, nm)
            if s > best_score:
                best, best_score = cc, s
        if best is not None and best_score >= 0.85:
            return self._locate(page, best)
        return None

    async def _click(self, page: Page, c: ElementCandidate) -> bool:
        self.click_attempts += 1
        # 1) semantic locators in order (getByRole > Label > Placeholder > Text > TestId)
        for loc in self._candidate_locators(page, c):
            try:
                await loc.click(timeout=1_500)
                await page.wait_for_timeout(350)
                return True
            except Exception:
                continue
        # 2) Layer-1 fuzzy heal (no LLM)
        healed = await self._fuzzy_locate(page, c)
        if healed is not None:
            try:
                await healed.click(timeout=1_500)
                await page.wait_for_timeout(350)
                self.healed_l1 += 1
                return True
            except Exception:
                pass
        self.click_fail += 1
        logger.debug(f"click '{c.label}' failed (all layers)")
        return False

    async def _fill(self, page: Page, c: ElementCandidate, value: str) -> bool:
        """Fill a field via the semantic-locator order. NEVER logs the value (creds)."""
        for loc in self._candidate_locators(page, c):
            try:
                await loc.fill(value, timeout=1_500)
                return True
            except Exception:
                continue
        return False

    async def _wait_actionable(self, page: Page, tries: int = 5, step_ms: int = 700) -> None:
        """Poll until interactive elements render — surfaces late SPA forms (F2)."""
        for _ in range(tries):
            try:
                n = await page.evaluate(
                    "() => document.querySelectorAll("
                    "'a,button,input,select,textarea,[role=button],[role=link]').length")
                if n > 1:
                    return
            except Exception:
                pass
            await page.wait_for_timeout(step_ms)

    async def _settle(self, page: Page) -> None:
        """Best-effort wait for SPA hydration before scanning (bounded)."""
        try:
            await page.wait_for_load_state("networkidle", timeout=3_000)
        except Exception:
            pass
        await page.wait_for_timeout(400)
        await self._wait_actionable(page)

    async def _replay(self, page: Page, seed: str, path: list[dict],
                      creds: dict | None = None) -> bool:
        creds = creds or {}
        try:
            await page.goto(seed, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(300)
            await self._wait_actionable(page, tries=4, step_ms=600)
        except Exception:
            return False
        for step in path:
            cand = ElementCandidate(label=step["label"], role=step.get("role"),
                                    name=step.get("name"), selector=step.get("selector"))
            if step.get("fill"):
                val = self._fill_value(cand, creds)  # RE-DERIVED, never stored in path
                if val is None or not await self._fill(page, cand, val):
                    return False
            elif not await self._click(page, cand):
                return False
        return True

    @staticmethod
    def _script(seed: str, path: list[dict]) -> str:
        lines = [f"page.goto({seed!r})"]
        for s in path:
            tgt = s.get("name") or s["label"]
            if s.get("fill"):
                # value intentionally masked — credentials are never stored/logged
                lines.append(f"page.get_by_label({tgt!r}).fill(<test_credential>)")
            elif s.get("role") and s.get("name"):
                lines.append(f"page.get_by_role({s['role']!r}, name={s['name']!r}).first.click()")
            else:
                lines.append(f"page.get_by_text({tgt!r}).first.click()")
        return json.dumps(lines)

    # ── main loop ────────────────────────────────────────────────────────
    async def explore(self, page: Page, seed_url: str, creds: dict | None = None) -> SFGStore:
        creds = creds or {}
        start = time.monotonic()
        try:
            await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)
            await self._settle(page)
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

            if not await self._replay(page, seed_url, path, creds):
                continue
            cur_id, _, cands = await self._record(page)
            has_inputs = any((c.role or "") in _FILL_ROLES for c in cands)

            for cand in cands[: self.max_actions]:
                if (cand.role or "") in _FILL_ROLES:
                    continue  # inputs are filled within a submit compound, not clicked alone
                if _is_blocked(cand.label, self.safe_mode):
                    self.blocked += 1
                    if has_inputs and any(k in (cand.label or "").lower() for k in _SUBMIT_KW):
                        self.blocked_submits += 1  # refused to submit a blocked form
                    continue
                # restore to this state before each action attempt
                if not await self._replay(page, seed_url, path, creds):
                    break
                src_id, _, cur_cands = await self._record(page)
                # Compound fill->submit (F1): before clicking a submit-like button on a
                # form, fill the form's inputs (creds-aware, PII-safe). Recorded as fill
                # steps so the edge's replay reproduces it (values re-derived, never stored).
                fill_steps: list[dict] = []
                _submit_kw = _SAFE_SUBMIT_KW if self.safe_mode else _SUBMIT_KW
                is_safe_submit = any(k in (cand.label or "").lower() for k in _submit_kw)
                if is_safe_submit and has_inputs and await self._action_blocked(page, cand):
                    self.blocked_submits += 1  # safe-looking button, but form posts to a blocked endpoint
                    continue
                if is_safe_submit and has_inputs:
                    for inp in cur_cands:
                        if (inp.role or "") not in _FILL_ROLES:
                            continue
                        val = self._fill_value(inp, creds)
                        if val is None:
                            continue
                        if await self._fill(page, inp, val):
                            fill_steps.append({"label": inp.label, "role": inp.role,
                                               "name": inp.name, "selector": inp.selector,
                                               "fill": True})
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
                new_path = path + fill_steps + [step]
                self._store.upsert_edge(SFGEdge(
                    edge_id=edge_id, source_node_id=src_id, target_node_id=tgt_id,
                    action_type=("fill_submit" if fill_steps else "click"),
                    locator=f"{cand.role or 'element'}:{cand.name or cand.label}",
                    input_value=None, safety_flag="SAFE",
                    replay_script=self._script(seed_url, new_path),
                ))
                if fill_steps:
                    self.compound_fills += 1
                if is_new and depth + 1 <= self.max_depth:
                    queue.append((tgt_id, new_path, depth + 1))

        logger.info(
            f"SFGTraversal: states={self._store.node_count()} edges={self._store.edge_count()} "
            f"blocked={self.blocked} clicks={self.click_attempts}/fail={self.click_fail} "
            f"healed_l1={self.healed_l1} compound_fills={self.compound_fills} "
            f"blocked_submits={self.blocked_submits} obs={self.observations} dedup_hits={self.merges}"
        )
        return self._store


__all__ = ["SFGTraversalExplorer"]
