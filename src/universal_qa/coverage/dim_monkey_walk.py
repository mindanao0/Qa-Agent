"""Coverage dimension: monkey-walk (random click traversal).

Independent of SFGTraversalExplorer by design. The planner-driven crawler
enumerates and clicks discovered elements according to a plan; it never
"wanders" — it skips paths that don't fit whatever goal/state-flow logic
it's using. A dumb, uniformly-random clicker has the opposite failure mode:
inefficient and occasionally destructive if not gated, but liable to
stumble into states no sane plan would ever chain into.

This module implements exactly that: a seeded random walk (real wandering,
not a star-pattern that returns to seed_url after every click) that clicks
one uniformly-random visible+enabled candidate per step, filters obviously
unsafe actions out of the candidate pool BEFORE selection (never "pick then
skip"), and guards against wandering off-domain by snapping back to the
seed URL when a click lands on a different origin.

Zero LLM calls. Zero reuse of SFGTraversalExplorer internals.
"""
from __future__ import annotations

import random
import time
from urllib.parse import urlsplit

from playwright.async_api import Page

from src.universal_qa.coverage.dimensions import DimensionResult
from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS

# Collects accessible-name-ish label + a locator recipe for each currently
# visible+enabled candidate in one JS round-trip (fast — no per-element
# is_visible()/is_enabled() round trips from Python).
_COLLECT_JS = """
() => {
  const sel = 'a, button, [role=button], [role=link]';
  const els = Array.from(document.querySelectorAll(sel));
  const out = [];
  els.forEach((el, i) => {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') return;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    if (el.hasAttribute('disabled')) return;
    if (el.getAttribute('aria-disabled') === 'true') return;
    const label = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
    el.setAttribute('data-monkey-idx', String(i));
    out.push({idx: i, label});
  });
  return out;
}
"""


def _is_blocked(label: str) -> bool:
    lowered = label.lower()
    return any(p in lowered for p in BLOCKED_ACTION_PATTERNS)


def _path(url: str) -> str:
    return urlsplit(url).path or "/"


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


async def collect(
    target_id: str,
    seed_url: str,
    headless: bool = True,
    max_actions: int = 150,
    time_budget_s: float = 90.0,
    seed: int | None = 42,
) -> DimensionResult:
    """Random-click traversal seeded with ``random.Random(seed)`` so the same
    seed reproduces the same path set on the same target.

    Launches and tears down its own browser/page (mirrors dim_static.collect's
    self-contained shape) so callers can use it the same way as the other
    dimensions; the actual walk loop lives in ``_walk`` below.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            result = await _walk(
                page, target_id, seed_url,
                max_actions=max_actions, time_budget_s=time_budget_s, seed=seed,
            )
        finally:
            await browser.close()
    return result


async def _walk(
    page: Page,
    target_id: str,
    seed_url: str,
    max_actions: int = 150,
    time_budget_s: float = 90.0,
    seed: int | None = 42,
) -> DimensionResult:
    rng = random.Random(seed)
    seed_origin = _origin(seed_url)

    paths: set[str] = set()
    off_domain_hits: list[str] = []
    candidates_filtered_blocked = 0
    actions_taken = 0

    await page.goto(seed_url, wait_until="domcontentloaded", timeout=30_000)
    paths.add(_path(page.url))

    deadline = time.monotonic() + time_budget_s
    while actions_taken < max_actions and time.monotonic() < deadline:
        try:
            raw_candidates = await page.evaluate(_COLLECT_JS)
        except Exception:  # noqa: BLE001 — navigation mid-eval, page torn down, etc.
            raw_candidates = []

        pool = []
        for c in raw_candidates:
            label = c.get("label", "") or ""
            if _is_blocked(label):
                candidates_filtered_blocked += 1
                continue
            pool.append(c)

        if not pool:
            # Nothing safe/clickable on this page — try navigating back to
            # seed to keep the walk alive rather than stalling out.
            if _path(page.url) != _path(seed_url) or not raw_candidates:
                try:
                    await page.goto(seed_url, wait_until="domcontentloaded", timeout=10_000)
                except Exception:  # noqa: BLE001
                    pass
                paths.add(_path(page.url))
                continue
            break

        choice = rng.choice(pool)
        locator = page.locator(f'[data-monkey-idx="{choice["idx"]}"]')

        try:
            await locator.first.click(timeout=5_000)
        except Exception:  # noqa: BLE001 — detached/covered/etc.; skip this step
            actions_taken += 1
            continue

        actions_taken += 1
        await page.wait_for_timeout(500)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3_000)
        except Exception:  # noqa: BLE001
            pass

        current_url = page.url
        if _origin(current_url) != seed_origin:
            off_domain_hits.append(f"{_origin(current_url)}|{_path(current_url)}")
            try:
                await page.goto(seed_url, wait_until="domcontentloaded", timeout=10_000)
            except Exception:  # noqa: BLE001
                pass
        else:
            paths.add(_path(current_url))

    return DimensionResult(
        method="monkey_walk",
        target_id=target_id,
        paths=paths,
        meta={
            "actions_taken": actions_taken,
            "candidates_filtered_blocked": candidates_filtered_blocked,
            "off_domain_hits": off_domain_hits,
            "seed": seed,
            "distinct_paths_found": len(paths),
        },
    )
