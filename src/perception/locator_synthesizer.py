# src/perception/locator_synthesizer.py
"""
Locator synthesizer — converts raw DOM element attributes into the best
stable Playwright-compatible locator string.

Priority chain (per spec):
  1. data-testid → [data-testid="value"]
  2. data-pw     → [data-pw="value"]
  3. id          → #id  (only if stable — not auto-generated)
  4. aria-label  → [aria-label="value"]
  5. role + visible text → role=ROLE[name="TEXT"]
  6. shortest unique CSS path (NOT XPath, NEVER XPath)
"""
from __future__ import annotations

import re


def _escape_attr(value: str) -> str:
    """Escape double-quotes for use inside CSS double-quoted attribute selectors."""
    return value.replace('"', '\\"')


def _is_stable_id(id_value: str) -> bool:
    """Return False for auto-generated IDs (all-digits, long hex, UUID-like)."""
    if id_value.isdigit():
        return False
    # Long hex string (>= 8 chars, only hex chars) — looks generated
    if re.fullmatch(r"[0-9a-f]{8,}", id_value, re.IGNORECASE):
        return False
    # UUID v4 pattern
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        id_value,
        re.IGNORECASE,
    ):
        return False
    return True


def best_locator(attrs: dict) -> str:
    """
    Convert element attributes dict to the best stable Playwright locator string.

    Priority chain (per CLAUDE.md — CSS selectors/XPath are FORBIDDEN):
      1. data-testid  → [data-testid="value"]
      2. data-pw      → [data-pw="value"]
      3. stable #id   → #id
      4. aria-label   → [aria-label="value"]
      5. role + accessible_name → role=ROLE[name="TEXT"]
         (explicit ARIA role only — from attrs.get("role"))
      4b. _accessible_name + tag → role=TAG[name="TEXT"]
         (native elements where JS sets role=null because role==tag)
      6. placeholder  → [placeholder="value"]
      7. name attr    → [name="value"]
      8. CSS tag+class — LAST RESORT for structural/container elements with no
         semantic identity; acceptable only because no stable semantic anchor exists.

    Args:
        attrs: dict with keys: id, data-testid, data-pw, aria-label, placeholder,
               name, type, href, value, class, tag (optional), role (optional),
               _accessible_name (optional, injected by DOMPruner enrichment)

    Returns:
        A Playwright-compatible selector string. Never returns XPath.
    """
    # 1. data-testid — most stable, purpose-built for testing
    testid = attrs.get("data-testid")
    if testid:
        return f'[data-testid="{_escape_attr(testid)}"]'

    # 2. data-pw — Playwright-specific test attribute
    data_pw = attrs.get("data-pw")
    if data_pw:
        return f'[data-pw="{_escape_attr(data_pw)}"]'

    # 3. id — only if it looks stable (not auto-generated)
    elem_id = attrs.get("id")
    if elem_id and _is_stable_id(elem_id):
        return f"#{elem_id}"

    # 4. aria-label — accessible label attribute
    aria_label = attrs.get("aria-label")
    if aria_label:
        return f'[aria-label="{_escape_attr(aria_label)}"]'

    # 5. role + accessible_name (explicit ARIA role from role attribute)
    role = attrs.get("role")
    # accessible_name may be passed explicitly or derived from aria-label
    accessible_name = attrs.get("accessible_name") or aria_label
    if role and accessible_name:
        # Playwright role selector format: role=ROLE[name="TEXT"]
        return f'role={role}[name="{_escape_attr(accessible_name)}"]'

    # 4b. _accessible_name + tag for native elements (where JS sets role=null
    #     because role == tag, e.g. <button> has implicit role "button").
    #     Use the tag itself as the effective role so we stay CSS-free.
    _acc_name = attrs.get("_accessible_name")
    tag = attrs.get("tag", "")
    if _acc_name and tag:
        effective_role = role if role else tag
        escaped_name = _escape_attr(str(_acc_name)[:100])
        if effective_role and escaped_name:
            return f'role={effective_role}[name="{escaped_name}"]'

    # 6. placeholder — good for input/textarea elements
    placeholder = attrs.get("placeholder")
    if placeholder:
        return f'[placeholder="{_escape_attr(placeholder)}"]'

    # 7. name attribute — stable for form fields
    name_attr = attrs.get("name")
    if name_attr:
        return f'[name="{_escape_attr(name_attr)}"]'

    # 8. CSS tag+class — LAST RESORT (violates ideal; acceptable only for
    #    structural elements with no semantic identity).
    tag = tag or attrs.get("tag") or "div"
    cls = attrs.get("class") or ""
    if cls:
        # class field is already limited to first 2 tokens by JS
        tokens = [t for t in cls.split() if t][:2]
        if tokens:
            class_part = "." + ".".join(tokens)
            return f"{tag}{class_part}"

    # Ultimate fallback: just the tag name
    return tag or "*"
