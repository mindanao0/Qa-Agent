# src/perception/dom_pruner.py
"""
DOMPruner — Layer 2 of the Universal DOM Compression Pipeline.

Implements the Prune4Web scoring algorithm entirely in-browser via
page.evaluate(), then maps results to typed Pydantic models in Python.

Core principle: ALL heavy work runs inside the browser via page.evaluate().
Python only processes the already-pruned results.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

from src.perception.aom_extractor import BBox
from src.perception.locator_synthesizer import best_locator


# ── Exception ─────────────────────────────────────────────────────────────────


class DOMPrunerError(Exception):
    """Raised when the browser-side pruning script fails (e.g. CSP violation)."""


# ── Models ────────────────────────────────────────────────────────────────────


class PrunedElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str                          # deterministic fingerprint
    tag: str
    role: str | None = None
    accessible_name: str | None = None       # priority: aria-label > alt > text > placeholder
    text: str = Field(default="", max_length=200)   # trimmed with ellipsis
    state: dict[str, bool] = Field(default_factory=dict)
    locator: str                             # best stable selector per priority
    bbox: BBox | None = None
    score: float                             # Prune4Web total score
    meta: dict = Field(default_factory=dict) # e.g. row_index, column_index for grids


class PrunedDOMSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    url: str
    timestamp_iso: str
    elements: list[PrunedElement]            # FLAT list, not tree
    reduction_ratio: float                   # original_count / pruned_count
    extraction_latency_ms: int


# ── Browser-side pruning script ───────────────────────────────────────────────

JS_PRUNE_SCRIPT = """
() => {
  const SEMANTIC_TAGS = new Set(['button','input','a','select','textarea','label',
    'nav','header','footer','main','section','article','form',
    'h1','h2','h3','h4','h5','h6']);
  const SEMANTIC_ROLES = new Set(['button','link','textbox','checkbox','radio',
    'combobox','listbox','menuitem','tab','dialog','navigation']);
  const INTERACTIVE_ROLES = new Set(['button','link','textbox','checkbox','radio',
    'combobox','listbox','menuitem','tab']);
  const STRIP_TAGS = new Set(['script','style','link','meta','noscript','template']);

  const vw = window.innerWidth || document.documentElement.clientWidth || 1;
  const vh = window.innerHeight || document.documentElement.clientHeight || 1;

  // TD-15b fix: SVG elements return SVGAnimatedString from el.className, not a plain string.
  // Use .baseVal for SVGAnimatedString, getAttribute('class') as final fallback.
  function getClassString(el) {
    const c = el.className;
    if (typeof c === 'string') return c;
    if (c && typeof c.baseVal === 'string') return c.baseVal;
    return el.getAttribute('class') || '';
  }

  function getRole(el) {
    return el.getAttribute('role') || el.tagName.toLowerCase();
  }

  // NOTE: This function is unused; accessible_name is derived inline in the output loop.
  function getAccessibleName(el) {
    return el.getAttribute('aria-label') ||
           el.getAttribute('alt') ||
           (el.innerText || '').trim().slice(0, 200) ||
           el.getAttribute('placeholder') ||
           '';
  }

  function scoreElement(el) {
    const tag = el.tagName.toLowerCase();
    const role = getRole(el);
    if (STRIP_TAGS.has(tag)) return -1;

    let semantic = SEMANTIC_TAGS.has(tag) || SEMANTIC_ROLES.has(role) ? 10 : 0;

    const cs = window.getComputedStyle(el);
    let visibility = (cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0') ? 5 : 0;

    let interaction = (el.tabIndex >= 0 || el.onclick || INTERACTIVE_ROLES.has(role)) ? 8 : 0;

    const textLen = (el.innerText || el.textContent || '').trim().length;
    let textuality = textLen >= 1 && textLen <= 200 ? 2 : (textLen > 200 ? 1 : 0);

    return semantic + visibility + interaction + textuality;
  }

  function getBbox(el) {
    try {
      const r = el.getBoundingClientRect();
      // Skip elements with zero dimensions (display:none, visibility:hidden)
      if (r.width === 0 && r.height === 0) return null;
      return {
        x_pct: Math.round(r.left / vw * 10000) / 100,
        y_pct: Math.round(r.top / vh * 10000) / 100,
        w_pct: Math.round(r.width / vw * 10000) / 100,
        h_pct: Math.round(r.height / vh * 10000) / 100
      };
    } catch(e) { return null; }
  }

  function makeId(el, idx) {
    const id = el.id || '';
    const testid = el.getAttribute('data-testid') || '';
    const tag = el.tagName.toLowerCase();
    const role = getRole(el);
    const key = `${tag}:${role}:${id}:${testid}:${idx}`;
    // simple hash
    let h = 0;
    for (let i = 0; i < key.length; i++) { h = (h * 31 + key.charCodeAt(i)) >>> 0; }
    return h.toString(16).padStart(8, '0');
  }

  // Collect all elements and score them
  // NOTE: Shadow DOM not pierced — querySelectorAll('*') only returns light-DOM elements.
  // Elements inside shadow roots are invisible here. See TD-12.
  const allEls = Array.from(document.querySelectorAll('*'));
  const scored = [];
  for (let i = 0; i < allEls.length; i++) {
    const el = allEls[i];
    const s = scoreElement(el);
    if (s >= 5) scored.push({el, score: s, idx: i});
  }

  // Detect repeating list patterns and collapse them
  const collapsed = new Set();
  for (let i = 0; i < scored.length; i++) {
    const {el} = scored[i];
    const parent = el.parentElement;
    if (!parent) continue;
    const siblings = Array.from(parent.children).filter(c =>
      c.tagName === el.tagName && c.getAttribute('role') === el.getAttribute('role')
    );
    if (siblings.length >= 10) {
      // mark all but first as collapsed
      for (let j = 1; j < siblings.length; j++) {
        collapsed.add(siblings[j]);
      }
    }
  }

  // Build output
  const results = [];
  for (const {el, score, idx} of scored) {
    const isCollapsed = collapsed.has(el);

    if (isCollapsed) continue; // skip non-representative items

    const tag = el.tagName.toLowerCase();
    const role = getRole(el);
    const ariaLabel = el.getAttribute('aria-label') || null;
    const altText = el.getAttribute('alt') || null;
    const innerText = (el.innerText || el.textContent || '').trim();
    const placeholder = el.getAttribute('placeholder') || null;

    const accessible_name = ariaLabel || altText || innerText.slice(0,200) || placeholder || null;
    const text = innerText.length > 200 ? innerText.slice(0,200) + '...' : innerText;

    // Count collapsed siblings for this element
    let itemCount = 1;
    if (el.parentElement) {
      const sibs = Array.from(el.parentElement.children).filter(c =>
        c.tagName === el.tagName && c.getAttribute('role') === el.getAttribute('role')
      );
      if (sibs.length >= 10) itemCount = sibs.length;
    }

    const attrs = {
      id: el.id || null,
      'data-testid': el.getAttribute('data-testid') || null,
      'data-pw': el.getAttribute('data-pw') || null,
      'aria-label': ariaLabel,
      placeholder: placeholder,
      name: el.getAttribute('name') || null,
      type: el.getAttribute('type') || null,
      href: el.getAttribute('href') || null,
      value: (() => { try { return el.value !== undefined ? String(el.value || '') : null; } catch(e) { return null; } })(),
      class: getClassString(el).split(' ').filter(Boolean).slice(0,2).join(' ') || null,
      tag: tag,
      role: role !== tag ? role : null
    };

    const state = {
      checked: !!el.checked,
      disabled: !!el.disabled,
      selected: !!el.selected
    };

    const bbox = getBbox(el);

    const meta = itemCount > 1 ? {item_count: itemCount, is_collapsed_list: true, representative_index: 0} : {};

    results.push({
      element_id: makeId(el, idx),
      tag,
      role: role !== tag ? role : null,
      accessible_name: accessible_name || null,
      text: text.slice(0, 200),
      state,
      attrs,
      bbox,
      score,
      meta
    });
  }

  return {elements: results, original_count: allEls.length};
}
"""


# ── DOMPruner ─────────────────────────────────────────────────────────────────


class DOMPruner:
    """
    Layer 2: Prunes the full DOM to a flat list of scored, semantically relevant
    elements. All scoring runs browser-side; Python only maps results to models.

    Usage::

        pruner = DOMPruner()
        snapshot = await pruner.prune(page)
    """

    async def prune(
        self, page: Page, root_selector: str | None = None
    ) -> PrunedDOMSnapshot:
        """
        Prune the DOM of *page* and return a typed PrunedDOMSnapshot.

        Known limitations (TD-12):
            - Shadow DOM: document.querySelectorAll('*') does not pierce shadow roots.
              Elements inside shadow DOMs are not visible to this pruner. If the page
              uses Web Components, the AOM extractor (Layer 1) is more reliable.
              Full shadow DOM traversal is tracked as TD-12 for Sprint 2.

        Args:
            page: An active Playwright Page.
            root_selector: Reserved for future scoped extraction (currently unused).

        Raises:
            DOMPrunerError: When page.evaluate() fails (e.g. CSP violation).
        """
        t0 = monotonic()
        url: str = page.url

        if root_selector is not None:
            logger.warning(
                f"DOMPruner.prune: root_selector={root_selector!r} is not yet "
                "implemented and will be ignored. Full-page pruning will run."
            )

        # Run the entire scoring/pruning logic in-browser
        try:
            raw: dict[str, Any] = await page.evaluate(JS_PRUNE_SCRIPT)
        except Exception as exc:
            logger.warning(f"DOMPruner: page.evaluate() failed for {url}: {exc}")
            raise DOMPrunerError(
                f"Browser-side pruning script failed for {url}: {exc}"
            ) from exc

        raw_elements: list[dict] = raw.get("elements", [])
        original_count: int = raw.get("original_count", len(raw_elements) or 1)

        # Map raw dicts → PrunedElement objects
        elements: list[PrunedElement] = []
        for elem_dict in raw_elements:
            attrs: dict = elem_dict.get("attrs", {})

            # Enrich attrs with top-level fields that best_locator needs.
            # The JS sets attrs.role = null when role == tag (native elements), so
            # we must pull role and tag from the top-level keys to avoid the CSS
            # fallback for plain <button>/<a> elements that only have visible text.
            attrs_enriched = dict(attrs)
            attrs_enriched["tag"] = elem_dict.get("tag", "")
            # Top-level role is already null-when-tag from JS; use it as-is.
            attrs_enriched["role"] = elem_dict.get("role")
            # Carry accessible_name as a private key so best_locator can build a
            # role+name locator for native elements that lack aria-label.
            if not attrs_enriched.get("aria-label") and elem_dict.get("accessible_name"):
                attrs_enriched["_accessible_name"] = elem_dict.get("accessible_name")

            # Synthesize the best stable locator from enriched attrs
            locator_str = best_locator(attrs_enriched)

            # Clamp bbox values to [0, 100]
            bbox: BBox | None = None
            raw_bbox = elem_dict.get("bbox")
            if raw_bbox:
                try:
                    clamped = {
                        k: max(0.0, min(100.0, float(raw_bbox.get(k, 0))))
                        for k in ("x_pct", "y_pct", "w_pct", "h_pct")
                    }
                    bbox = BBox(**clamped)
                except Exception as exc:
                    logger.debug(f"DOMPruner: bbox clamping failed: {exc}")
                    bbox = None

            try:
                element = PrunedElement(
                    element_id=elem_dict.get("element_id", "unknown"),
                    tag=elem_dict.get("tag", "unknown"),
                    role=elem_dict.get("role"),
                    accessible_name=elem_dict.get("accessible_name"),
                    text=elem_dict.get("text", ""),
                    state=elem_dict.get("state", {}),
                    locator=locator_str,
                    bbox=bbox,
                    score=float(elem_dict.get("score", 0)),
                    meta=elem_dict.get("meta", {}),
                )
            except Exception as exc:
                logger.debug(f"DOMPruner: skipping malformed element: {exc}")
                continue

            elements.append(element)

        # Calculate reduction ratio
        pruned_count = len(elements)
        if pruned_count > 0:
            reduction_ratio = original_count / pruned_count
        else:
            reduction_ratio = 1.0

        # Generate snapshot metadata
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        snapshot_id = hashlib.sha256(
            f"{now_iso}{url}".encode()
        ).hexdigest()[:16]

        latency_ms = int((monotonic() - t0) * 1000)
        logger.info(
            f"DOMPruner: snapshot_id={snapshot_id} original={original_count} "
            f"pruned={pruned_count} ratio={reduction_ratio:.1f}x "
            f"latency={latency_ms}ms url={url}"
        )

        return PrunedDOMSnapshot(
            snapshot_id=snapshot_id,
            url=url,
            timestamp_iso=now_iso,
            elements=elements,
            reduction_ratio=reduction_ratio,
            extraction_latency_ms=latency_ms,
        )
