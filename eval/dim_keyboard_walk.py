"""Coverage dimension: keyboard-only traversal.

Independent from ``SFGTraversalExplorer`` (which discovers/activates elements
by *clicking* via ``page.get_by_role(...).click()``). This module never
clicks anything — it drives the page purely through ``Tab``/``Enter`` key
presses, the way a keyboard-only or screen-reader user would, and records
which paths are reachable that way.

Why this is a genuinely different measurement, not a redundant one:
  * Click-driven crawling can reach elements that are mouse-only (drag
    handles, hover-reveal menus, ``onclick`` divs with no ``tabindex`` —
    those are literally impossible to focus with Tab, so this dimension
    will NEVER see them. That is an honest, expected blind spot of this
    dimension, not a bug — see the fixture note below).
  * This dimension, in turn, can reach things a click-crawler's element
    scanner might skip (skip-links positioned off-screen, custom widgets
    with a keydown handler but no visible/clickable target, elements a
    viewport-bound click scanner never scrolls to).
  * The whole point of the coverage cross-check is to diff these blind
    spots against each other, not to pick one as "the" crawler.

Algorithm (deliberately simple — NOT a state-graph BFS, just "what paths
are keyboard-reachable from the entry point"):
  1. Navigate to ``seed_url``.
  2. Loop: press ``Tab`` to move focus to the next focusable element, read
     ``document.activeElement``'s tag/role/accessible-name/href.
  3. Safety gate: if the accessible name/label matches
     ``BLOCKED_ACTION_PATTERNS`` (delete/remove/transfer/payment/password/
     logout/deactivate), NEVER press Enter on it — just keep tabbing.
  4. Otherwise, if the focused element looks like something that would
     navigate (an anchor with an href, or a button), press ``Enter`` and
     wait briefly for navigation. If the URL changed, record the new
     path, then return to the seed URL (via ``page.go_back()``, falling
     back to a fresh ``page.goto(seed_url)``) so the walk keeps
     re-sampling the SAME entry point's tab order rather than wandering
     deeper and deeper into whatever page it lands on.
  5. Stop when ``max_tabs`` is reached, ``time_budget_s`` elapses, or the
     focused-element identity we started with reappears (a full loop of
     the tab order) — whichever comes first.

No LLM calls. No CSS/XPath selectors are constructed here (we read
``document.activeElement`` via ``page.evaluate`` rather than locating by
CSS, so this stays inside the project's role/label/text-only locator
convention where it interacts with the page at all).
"""
from __future__ import annotations

import hashlib
import time
from urllib.parse import urlsplit

from playwright.async_api import Page

from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS
from eval.coverage_dimensions import DimensionResult

_ACTIVE_ELEMENT_JS = """() => {
  const el = document.activeElement;
  if (!el || el === document.body) return null;
  const tag = el.tagName ? el.tagName.toLowerCase() : '';
  const role = el.getAttribute ? (el.getAttribute('role') || '') : '';
  const href = (tag === 'a' && el.href) ? el.href : '';
  const name = (el.innerText || el.textContent || el.getAttribute('aria-label') ||
                el.getAttribute('title') || el.value || '').trim().slice(0, 200);
  return {tag, role, href, name};
}"""


def _identity(el: dict) -> str:
    raw = f"{el.get('tag','')}|{el.get('role','')}|{el.get('name','')}|{el.get('href','')}"
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


def _is_blocked(name: str) -> bool:
    low = (name or "").lower()
    return any(p in low for p in BLOCKED_ACTION_PATTERNS)


def _looks_activatable(el: dict) -> bool:
    tag = el.get("tag", "")
    role = el.get("role", "")
    if tag in ("a", "button"):
        return True
    if role in ("link", "button"):
        return True
    return False


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


async def collect(
    target_id: str,
    seed_url: str,
    headless: bool = True,
    max_tabs: int = 200,
    time_budget_s: float = 90.0,
) -> DimensionResult:
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            result = await _walk(page, target_id, seed_url, max_tabs, time_budget_s)
        finally:
            await browser.close()
    return result


async def _walk(
    page: Page,
    target_id: str,
    seed_url: str,
    max_tabs: int,
    time_budget_s: float,
) -> DimensionResult:
    await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)

    seed_path = _path(page.url)
    paths: set[str] = {seed_path}

    tabs_pressed = 0
    activations_attempted = 0
    activations_blocked = 0
    cycle_detected = False
    first_identity: str | None = None
    # Number of Tab presses since the page was last freshly (re)loaded at the
    # seed URL. Navigating away and back (or re-goto'ing the seed) resets
    # document.activeElement to <body>, which would otherwise make every
    # walk rediscover only the FIRST tabbable element forever. We replay
    # this many silent Tab presses after each return-to-seed to restore
    # focus to where we left off, so the walk actually advances through the
    # whole tab order (mirrors this codebase's existing "replay-from-seed"
    # state-restoration pattern used by SFGTraversalExplorer).
    tabs_since_seed = 0

    start = time.monotonic()
    while tabs_pressed < max_tabs and (time.monotonic() - start) < time_budget_s:
        await page.keyboard.press("Tab")
        tabs_pressed += 1
        tabs_since_seed += 1

        try:
            el = await page.evaluate(_ACTIVE_ELEMENT_JS)
        except Exception:  # noqa: BLE001 - page may be mid-navigation
            el = None
        if not el:
            continue

        ident = _identity(el)
        if first_identity is None:
            first_identity = ident
        elif ident == first_identity and tabs_pressed > 1:
            cycle_detected = True
            break

        name = el.get("name", "")
        if not _looks_activatable(el):
            continue

        if _is_blocked(name):
            activations_blocked += 1
            continue

        activations_attempted += 1
        # Pressing Enter and then separately calling wait_for_load_state()
        # races the navigation: if the current page had already reached
        # "domcontentloaded" (the common case, since we're idle waiting on
        # focus), wait_for_load_state() resolves immediately -- BEFORE the
        # Enter-triggered navigation has even started -- so a plain
        # try/except around it silently reads the stale pre-navigation URL.
        # expect_navigation() brackets the key press itself, so it actually
        # waits for a navigation that starts as a result of it (and times
        # out quickly, harmlessly, on the common case of no navigation).
        try:
            async with page.expect_navigation(timeout=5000):
                await page.keyboard.press("Enter")
        except Exception:  # noqa: BLE001 - most Enters don't navigate at all
            pass

        new_path = _path(page.url)
        if new_path != seed_path:
            paths.add(new_path)
            # Return to the seed's tab order rather than wandering deeper.
            try:
                async with page.expect_navigation(timeout=5000):
                    await page.go_back()
            except Exception:  # noqa: BLE001
                pass
            if _path(page.url) != seed_path:
                try:
                    await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)
                except Exception:  # noqa: BLE001
                    break

            # Restore focus to the element we just activated (a fresh
            # load/back-nav resets focus to <body>) so the *next* Tab press
            # advances to the NEXT unseen element instead of re-discovering
            # this same one. tabs_since_seed is left as-is (it already
            # represents "how many tabs to reach the current position").
            for _ in range(tabs_since_seed):
                if tabs_pressed >= max_tabs or (time.monotonic() - start) >= time_budget_s:
                    break
                try:
                    await page.keyboard.press("Tab")
                except Exception:  # noqa: BLE001
                    break
                tabs_pressed += 1

    return DimensionResult(
        method="keyboard_walk",
        target_id=target_id,
        paths=paths,
        meta={
            "tabs_pressed": tabs_pressed,
            "activations_attempted": activations_attempted,
            "activations_blocked": activations_blocked,
            "cycle_detected": cycle_detected,
        },
    )
