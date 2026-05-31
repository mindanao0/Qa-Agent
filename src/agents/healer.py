from __future__ import annotations

import re

from loguru import logger
from playwright.async_api import Page

from src.browser.ax_extractor import extract_axtree, prune_axtree
from src.browser.manager import BrowserManager
from src.healing.engine import HealingEngine
from src.llm.adapter import OllamaAdapter
from src.llm.structured import HealedLocator, PlaywrightScript

# Patterns that extract a Playwright locator string from an error message
_LOCATOR_ERROR_PATTERNS: list[re.Pattern[str]] = [
    # TimeoutError: Timeout 15000ms exceeded waiting for locator('get_by_role...')
    re.compile(r"waiting for\s+(?:locator\(['\"])?(?P<loc>page\.get_by_\w+\([^)]+\))", re.IGNORECASE),
    # Error: locator.click: ... locator=get_by_role('button', name='Login')
    re.compile(r"locator=(?P<loc>(?:page\.)?get_by_\w+\([^)]+\))", re.IGNORECASE),
    # Fallback: any get_by_xxx() pattern in the traceback
    re.compile(r"(?P<loc>page\.get_by_\w+\([^)]+\))"),
]

_MAX_AXTREE_NODES = 200


class CodeHealerAgent:
    """
    Code-level healer: given a failing PlaywrightScript and its execution error,
    attempts to:

      1. Extract the failing Playwright locator from the error traceback.
      2. Delegate locator healing to HealingEngine (fuzzy → AI → VLM signal).
      3. Patch the script source code by replacing the old locator string with
         the healed one.

    If no locator can be extracted the original script is returned unchanged
    (the executor will either succeed on retry or exhaust its retry budget).
    """

    def __init__(
        self,
        adapter: OllamaAdapter,
        browser_manager: BrowserManager,
        healing_engine: HealingEngine | None = None,
        locator_file: str = "locators/locators.json",
    ) -> None:
        self.adapter = adapter
        self.browser_manager = browser_manager
        self.healing_engine = healing_engine or HealingEngine(
            adapter=adapter,
            locator_file=locator_file,
        )

    async def heal_script(
        self,
        script: PlaywrightScript,
        error_output: str,
        url: str,
        role: str = "admin",
    ) -> PlaywrightScript:
        """
        Attempt to heal *script* given the *error_output* from a failed test run.

        Returns a new PlaywrightScript with the patched code, or the original
        if no locator could be extracted or healing produced low confidence.
        """
        if "--timeout" in error_output or "unrecognized arguments" in error_output:
            logger.warning(
                "CodeHealerAgent: pytest argument error detected — "
                "not a locator issue, skipping heal"
            )
            return script

        failing_locator = _extract_locator_from_error(error_output)
        if not failing_locator:
            logger.warning(
                "CodeHealerAgent: no locator found in error output — "
                "returning original script unchanged"
            )
            return script

        logger.info(
            f"CodeHealerAgent: healing locator={failing_locator!r} url={url!r}"
        )

        healed_locator = await self._heal_via_browser(
            failing_locator=failing_locator,
            url=url,
            role=role,
            error_output=error_output,
        )

        if healed_locator is None or healed_locator.healed == failing_locator:
            logger.info("CodeHealerAgent: no improvement — returning original script")
            return script

        patched_code = _patch_locator_in_code(
            code=script.code,
            original=failing_locator,
            healed=healed_locator.healed,
        )

        healed_locators_used = list(
            set(script.locators_used) | {healed_locator.healed}
        )

        logger.info(
            f"CodeHealerAgent: patched script | "
            f"original={failing_locator!r} → healed={healed_locator.healed!r} "
            f"confidence={healed_locator.confidence:.4f} method={healed_locator.method!r}"
        )

        return PlaywrightScript(
            reasoning=(
                f"[Healed by CodeHealerAgent]\n"
                f"Original locator: {failing_locator}\n"
                f"Healed locator:   {healed_locator.healed}\n"
                f"Method: {healed_locator.method} | Confidence: {healed_locator.confidence:.4f}\n"
                f"Reasoning: {healed_locator.reasoning}\n\n"
                f"Original reasoning:\n{script.reasoning}"
            ),
            code=patched_code,
            locators_used=healed_locators_used,
            test_function_name=script.test_function_name,
        )

    async def _heal_via_browser(
        self,
        failing_locator: str,
        url: str,
        role: str,
        error_output: str,
    ) -> HealedLocator | None:
        """Open a fresh page at *url* to capture live AxTree, then heal."""
        try:
            async with self.browser_manager.new_page(role=role) as page:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                except Exception as nav_exc:
                    logger.warning(
                        f"CodeHealerAgent: navigation to {url!r} failed: {nav_exc}"
                    )

                raw = await extract_axtree(page)
                axtree = await prune_axtree(raw, max_nodes=_MAX_AXTREE_NODES)

                return await self.healing_engine.heal(
                    page=page,
                    failed_locator=failing_locator,
                    action="locate",
                    error_message=error_output[:500],
                )
        except Exception as exc:
            logger.error(f"CodeHealerAgent._heal_via_browser: {exc}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_locator_from_error(error_output: str) -> str:
    """
    Scan an error traceback for the first recognisable Playwright locator string.
    Returns empty string if none found.
    """
    for pattern in _LOCATOR_ERROR_PATTERNS:
        m = pattern.search(error_output)
        if m:
            raw = m.group("loc").strip()
            # Normalise: ensure it starts with "page."
            if not raw.startswith("page."):
                raw = "page." + raw
            return raw
    return ""


def _patch_locator_in_code(code: str, original: str, healed: str) -> str:
    """
    Replace every occurrence of *original* locator in *code* with *healed*.

    Falls back to simple string replacement; this is safe because Playwright
    locator strings are distinctive enough to avoid accidental collisions.
    """
    # Strip the leading "page." for matching inside method chains if needed
    stripped_orig = original.removeprefix("page.")
    stripped_healed = healed.removeprefix("page.")

    patched = code
    if original in patched:
        patched = patched.replace(original, healed)
    elif stripped_orig in patched:
        patched = patched.replace(stripped_orig, stripped_healed)
    else:
        logger.warning(
            f"_patch_locator_in_code: could not find {original!r} in code — "
            "returning original code unmodified"
        )
    return patched
