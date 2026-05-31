# src/healing/state_validator.py
"""
StateValidator — Sprint 2 Cluster S2-B.

Runs BEFORE and AFTER every Playwright action:
  1. capture_pre_state(page) -> PreState
  2. <execute action>
  3. classify_post_action(page, pre, action_id) -> StateClassification

It also exposes wait_for_stability(page, timeout_ms) — a sleep-free DOM
quiescence wait powered by a MutationObserver.

Implementation rules (carry-forward from Sprint 1):
- NEVER call page.wait_for_timeout()
- NEVER call page.accessibility (gone — use AOMExtractor / CDP)
- Every public method is non-raising. On any internal failure we return
  a graceful default object.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

from src.perception.aom_extractor import (
    AOMExtractor,
    AOMNode,
    AOMSnapshot,
    AOMSparseError,
)


# ── Models ────────────────────────────────────────────────────────────────────


class A11yDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes_added: int
    nodes_removed: int
    nodes_changed: int
    error_text_found: bool
    loading_indicators: bool


class PreState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    pre_state_hash: str
    pre_snapshot: AOMSnapshot
    captured_at_iso: str
    # Monotonic seconds at capture; used for time-based heuristics in
    # classify_post_action. Hidden from JSON via private storage.
    captured_monotonic: float = Field(default=0.0)


class StateClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    outcome_label: Literal["Success", "Error_State", "Loading_State", "Unknown"]
    pre_state_hash: str
    post_state_hash: str
    a11y_delta: A11yDelta
    url_changed: bool
    confidence_score: float
    failure_signature: str | None = None


# ── Constants ─────────────────────────────────────────────────────────────────


_ERROR_NAME_PATTERNS = (
    "error",
    "failed",
    "invalid",
    "unable to",
    "required",
    "not found",
)
_LOADING_NAME_PATTERNS = ("loading", "please wait")
_ERROR_ROLES = {"alert", "alertdialog"}
_LOADING_ROLES = {"status", "progressbar"}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _flatten(node: AOMNode) -> list[AOMNode]:
    out = [node]
    for c in node.children:
        out.extend(_flatten(c))
    return out


def _hash_nodes(nodes: list[AOMNode]) -> str:
    """
    Hash a list of AOMNodes deterministically.

    Per-node tuple: (role, name, sorted(state.items()))  (NO bbox, NO source_node_id,
    NO children — children are flattened into the same list).
    """
    parts: list[str] = []
    for n in nodes:
        state_repr = sorted((k, bool(v)) for k, v in (n.state or {}).items())
        parts.append(f"{n.role}|{n.name}|{state_repr}")
    joined = "\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _empty_snapshot(url: str) -> AOMSnapshot:
    """Build a minimal synthetic AOMSnapshot for graceful-failure paths."""
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    root = AOMNode(
        role="WebArea",
        name="",
        value=None,
        state={},
        bbox=None,
        children=[],
        source_node_id="0" * 12,
    )
    return AOMSnapshot(
        snapshot_id="0" * 16,
        url=url or "",
        timestamp_iso=now_iso,
        root_node=root,
        node_count=0,
        extraction_latency_ms=0,
    )


def _empty_delta() -> A11yDelta:
    return A11yDelta(
        nodes_added=0,
        nodes_removed=0,
        nodes_changed=0,
        error_text_found=False,
        loading_indicators=False,
    )


def _detect_error(nodes: list[AOMNode]) -> tuple[bool, str | None, str | None]:
    """
    Return (error_text_found, error_marker, first_role).

    - error_marker: the first matching error name (lowercased) OR "alert" if a
      role-based detection won (used for failure_signature).
    - first_role: role of the first node in the flattened list — used in the
      failure_signature template.
    """
    first_role = nodes[0].role if nodes else ""
    for n in nodes:
        if n.role in _ERROR_ROLES:
            return True, "alert", first_role
    for n in nodes:
        nm = (n.name or "").lower()
        for pat in _ERROR_NAME_PATTERNS:
            if pat in nm:
                return True, pat, first_role
    return False, None, first_role


def _detect_loading(nodes: list[AOMNode]) -> bool:
    for n in nodes:
        if n.role in _LOADING_ROLES:
            return True
        if n.state and n.state.get("aria-busy"):
            return True
        nm = (n.name or "").lower()
        for pat in _LOADING_NAME_PATTERNS:
            if pat in nm:
                return True
    return False


def _safe_url(page: Page) -> str:
    try:
        return page.url or ""
    except Exception:
        return ""


# ── Public class ──────────────────────────────────────────────────────────────


class StateValidator:
    """Captures pre/post AOM and classifies the action outcome."""

    def __init__(self, aom_extractor: AOMExtractor | None = None) -> None:
        self._extractor = aom_extractor or AOMExtractor()

    # ── pre ───────────────────────────────────────────────────────────────────

    async def capture_pre_state(self, page: Page) -> PreState:
        """
        Capture an AOMSnapshot of the current page.

        NEVER raises. On any failure (including AOMSparseError) returns a
        PreState with empty hash and a synthetic empty snapshot, so the
        caller can still proceed.
        """
        url = _safe_url(page)
        captured_iso = datetime.now(tz=timezone.utc).isoformat()
        captured_mono = time.monotonic()

        try:
            snapshot = await self._extractor.extract(page)
            nodes = _flatten(snapshot.root_node)
            pre_hash = _hash_nodes(nodes)
            return PreState(
                url=url,
                pre_state_hash=pre_hash,
                pre_snapshot=snapshot,
                captured_at_iso=captured_iso,
                captured_monotonic=captured_mono,
            )
        except AOMSparseError as exc:
            logger.debug(f"StateValidator.capture_pre_state: sparse AOM at {url}: {exc}")
            return PreState(
                url=url,
                pre_state_hash="",
                pre_snapshot=_empty_snapshot(url),
                captured_at_iso=captured_iso,
                captured_monotonic=captured_mono,
            )
        except Exception as exc:
            logger.warning(
                f"StateValidator.capture_pre_state: unexpected error at {url}: {exc}"
            )
            return PreState(
                url=url,
                pre_state_hash="",
                pre_snapshot=_empty_snapshot(url),
                captured_at_iso=captured_iso,
                captured_monotonic=captured_mono,
            )

    # ── post / classify ───────────────────────────────────────────────────────

    async def classify_post_action(
        self, page: Page, pre: PreState, action_id: str = ""
    ) -> StateClassification:
        """
        Capture a post snapshot and classify the outcome.

        NEVER raises. On any error returns Unknown / 0.0 confidence.
        """
        try:
            return await self._classify_inner(page, pre, action_id)
        except Exception as exc:
            logger.warning(
                f"StateValidator.classify_post_action: failure ({exc!r}) "
                f"action_id={action_id!r} — returning Unknown"
            )
            return StateClassification(
                action_id=action_id,
                outcome_label="Unknown",
                pre_state_hash=pre.pre_state_hash if pre else "",
                post_state_hash="",
                a11y_delta=_empty_delta(),
                url_changed=False,
                confidence_score=0.0,
                failure_signature=None,
            )

    async def _classify_inner(
        self, page: Page, pre: PreState, action_id: str
    ) -> StateClassification:
        post_url = _safe_url(page)
        url_changed = bool(post_url) and bool(pre.url) and post_url != pre.url

        # Capture post snapshot (gracefully). Track whether extraction failed
        # with an unexpected error (vs. sparse) — that's a strong signal we
        # cannot classify meaningfully and should return Unknown.
        post_extract_failed = False
        try:
            post_snapshot = await self._extractor.extract(page)
        except AOMSparseError:
            post_snapshot = _empty_snapshot(post_url)
        except Exception as exc:
            logger.debug(
                f"StateValidator: post extract failed at {post_url}: {exc} — returning Unknown"
            )
            post_snapshot = _empty_snapshot(post_url)
            post_extract_failed = True

        if post_extract_failed:
            return StateClassification(
                action_id=action_id,
                outcome_label="Unknown",
                pre_state_hash=pre.pre_state_hash,
                post_state_hash="",
                a11y_delta=_empty_delta(),
                url_changed=False,
                confidence_score=0.0,
                failure_signature=None,
            )

        pre_nodes = _flatten(pre.pre_snapshot.root_node) if pre.pre_snapshot else []
        post_nodes = _flatten(post_snapshot.root_node) if post_snapshot else []

        post_hash = _hash_nodes(post_nodes) if post_nodes else ""

        # Delta computation: keyed on (role, name)
        pre_keys = {(n.role, n.name) for n in pre_nodes}
        post_keys = {(n.role, n.name) for n in post_nodes}
        nodes_added = len(post_keys - pre_keys)
        nodes_removed = len(pre_keys - post_keys)

        # nodes_changed: same (role, name) on both sides but state dict differs
        pre_by_key: dict[tuple[str, str], dict[str, bool]] = {}
        for n in pre_nodes:
            pre_by_key.setdefault((n.role, n.name), n.state or {})
        post_by_key: dict[tuple[str, str], dict[str, bool]] = {}
        for n in post_nodes:
            post_by_key.setdefault((n.role, n.name), n.state or {})
        common = pre_keys & post_keys
        nodes_changed = 0
        for k in common:
            if pre_by_key.get(k, {}) != post_by_key.get(k, {}):
                nodes_changed += 1

        error_text_found, error_marker, first_role = _detect_error(post_nodes)
        loading_indicators = _detect_loading(post_nodes)

        delta = A11yDelta(
            nodes_added=nodes_added,
            nodes_removed=nodes_removed,
            nodes_changed=nodes_changed,
            error_text_found=error_text_found,
            loading_indicators=loading_indicators,
        )
        total_delta = nodes_added + nodes_removed + nodes_changed

        # ── Classification rules ──
        outcome: Literal["Success", "Error_State", "Loading_State", "Unknown"]
        confidence: float
        failure_signature: str | None = None
        # Rule 1: Error_State
        elapsed_s = max(0.0, time.monotonic() - (pre.captured_monotonic or 0.0))
        no_visible_change = (
            total_delta < 2 and elapsed_s > 2.0 and not url_changed
        )
        # Detect explicit alert role specifically (for confidence tier)
        explicit_alert = any(n.role in _ERROR_ROLES for n in post_nodes)

        if error_text_found:
            outcome = "Error_State"
            confidence = 0.75 if explicit_alert else 0.6
        elif no_visible_change:
            outcome = "Error_State"
            confidence = 0.6
            # error_marker stays None here; mark something for the signature
            error_marker = error_marker or "no_visible_change"
        # Rule 2: Loading_State
        elif loading_indicators and total_delta < 3:
            outcome = "Loading_State"
            confidence = 0.7
        # Rule 3: Success
        elif url_changed:
            outcome = "Success"
            confidence = 0.9
        elif total_delta > 5:
            outcome = "Success"
            confidence = 0.8
        else:
            outcome = "Unknown"
            confidence = 0.4

        # failure_signature: only when Error_State
        if outcome == "Error_State":
            marker = error_marker or "unknown"
            try:
                path = urlparse(post_url).path or ""
            except Exception:
                path = ""
            sig_input = f"{marker}|{path}|{first_role or ''}|action"
            failure_signature = hashlib.sha256(
                sig_input.encode("utf-8")
            ).hexdigest()[:16]

        return StateClassification(
            action_id=action_id,
            outcome_label=outcome,
            pre_state_hash=pre.pre_state_hash,
            post_state_hash=post_hash,
            a11y_delta=delta,
            url_changed=url_changed,
            confidence_score=confidence,
            failure_signature=failure_signature,
        )

    # ── stability wait ────────────────────────────────────────────────────────

    async def wait_for_stability(
        self, page: Page, timeout_ms: int = 5000
    ) -> bool:
        """
        Wait until the DOM stops mutating (no childList/subtree/attribute changes
        for 500ms), or *timeout_ms* elapses.

        Does NOT use page.wait_for_timeout. Uses a single page.evaluate that
        installs a MutationObserver and resolves itself.

        Returns True on quiescence, False on safety timeout or error.
        """
        js = """
        (timeoutMs) => new Promise((resolve) => {
            let timer;
            const observer = new MutationObserver(() => {
                clearTimeout(timer);
                timer = setTimeout(() => { observer.disconnect(); resolve(true); }, 500);
            });
            observer.observe(document.documentElement, {
                childList: true, subtree: true, attributes: true
            });
            timer = setTimeout(() => { observer.disconnect(); resolve(true); }, 500);
            setTimeout(() => { observer.disconnect(); resolve(false); }, timeoutMs);
        })
        """
        try:
            result = await page.evaluate(js, timeout_ms)
            return bool(result)
        except Exception as exc:
            logger.debug(f"StateValidator.wait_for_stability: error {exc!r}")
            return False
