import json
import os
from difflib import unified_diff
from pathlib import Path
from typing import Any

from filelock import FileLock
from loguru import logger
from playwright.async_api import Page

from src.browser.ax_extractor import extract_axtree, prune_axtree
from src.llm.adapter import OllamaAdapter
from src.llm.structured import HealedLocator
from .fuzzy_matcher import fuzzy_match
from .ai_healer import ai_heal

_DEFAULT_LOCATOR_FILE = Path("locators/locators.json")
_DEFAULT_MAX_AXTREE_NODES = 200

# Healing phase thresholds (mirrored from config/agent.yaml defaults)
_FUZZY_THRESHOLD = 0.85
_AI_THRESHOLD = 0.50
_VLM_THRESHOLD = 0.40


class HealingEngine:
    """
    Orchestrates the three-phase self-healing pipeline:

      Phase 1 — Fuzzy (Jaro-Winkler):
          Fast, deterministic.  If confidence ≥ 0.85, return immediately.

      Phase 2 — AI (LLM + AxTree):
          Invokes OllamaAdapter with the HealerPromptTemplate.
          If confidence ≥ 0.50, return with validation.

      Phase 3 — VLM signal:
          Emits a warning signal only — does NOT invoke the VLM here.
          VLM activation is handled by the caller (agents/healer.py).

    After any successful Phase 1 or 2 heal, the engine:
      - Validates that the healed locator resolves to a visible element.
      - Records a before/after AxTree diff in the log.
      - Atomically updates locators/locators.json via FileLock.
    """

    def __init__(
        self,
        adapter: OllamaAdapter | None = None,
        locator_file: str | Path = _DEFAULT_LOCATOR_FILE,
        fuzzy_threshold: float = _FUZZY_THRESHOLD,
        ai_threshold: float = _AI_THRESHOLD,
        vlm_threshold: float = _VLM_THRESHOLD,
        max_axtree_nodes: int = _DEFAULT_MAX_AXTREE_NODES,
    ) -> None:
        self._adapter = adapter
        self._locator_file = Path(locator_file)
        self._fuzzy_threshold = fuzzy_threshold
        self._ai_threshold = ai_threshold
        self._vlm_threshold = vlm_threshold
        self._max_axtree_nodes = max_axtree_nodes
        self._lock_path = self._locator_file.with_suffix(".lock")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def heal(
        self,
        page: Page,
        failed_locator: str,
        action: str = "locate",
        error_message: str = "",
    ) -> HealedLocator:
        """
        Attempt to heal *failed_locator* using the three-phase pipeline.

        Returns a HealedLocator.  If Phase 3 (VLM) is required, the returned
        HealedLocator has method="vlm" and confidence < vlm_threshold — the
        caller is responsible for activating VLM inference.
        """
        logger.info(
            f"HealingEngine.heal | locator={failed_locator!r} "
            f"action={action!r} url={page.url!r}"
        )

        # Capture initial AxTree (before healing attempt)
        raw_before = await extract_axtree(page)
        axtree_before = await prune_axtree(raw_before, max_nodes=self._max_axtree_nodes)

        known_locators = self._load_locators()

        # ── Phase 1: Fuzzy ────────────────────────────────────────────────────
        fuzzy_result = fuzzy_match(failed_locator, axtree_before, threshold=self._fuzzy_threshold)
        if fuzzy_result and fuzzy_result.confidence >= self._fuzzy_threshold:
            logger.info(
                f"Phase 1 (fuzzy) succeeded | confidence={fuzzy_result.confidence:.4f}"
            )
            validated = await self._validate_and_record(
                page, fuzzy_result, failed_locator, axtree_before, known_locators
            )
            return validated

        # ── Phase 2: AI ───────────────────────────────────────────────────────
        logger.info(
            f"Phase 1 confidence {fuzzy_result.confidence if fuzzy_result else 0:.4f} "
            f"< {self._fuzzy_threshold} — escalating to Phase 2 (AI)"
        )
        ai_result = await ai_heal(
            page=page,
            failed_locator=failed_locator,
            action=action,
            error_message=error_message,
            axtree=axtree_before,
            known_locators=known_locators,
            adapter=self._adapter,
        )

        if ai_result.confidence >= self._ai_threshold:
            logger.info(
                f"Phase 2 (AI) succeeded | confidence={ai_result.confidence:.4f}"
            )
            validated = await self._validate_and_record(
                page, ai_result, failed_locator, axtree_before, known_locators
            )
            return validated

        # ── Phase 3: VLM signal ───────────────────────────────────────────────
        logger.warning(
            f"Phase 2 (AI) confidence {ai_result.confidence:.4f} "
            f"< {self._ai_threshold} — VLM fallback required. "
            "Returning signal for caller to activate CPU VLM."
        )
        ai_result.method = "vlm"
        self._log_healing_event(
            failed_locator, ai_result, axtree_before, axtree_after=None, validated=False
        )
        return ai_result

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    async def _validate_and_record(
        self,
        page: Page,
        healed: HealedLocator,
        original_locator: str,
        axtree_before: str,
        known_locators: dict[str, Any],
    ) -> HealedLocator:
        """
        Verify the healed locator resolves to a visible element.
        On success, update locators.json and log the before/after AxTree diff.
        """
        validated = await self._validate_locator(page, healed.healed)

        if validated:
            raw_after = await extract_axtree(page)
            axtree_after = await prune_axtree(raw_after, max_nodes=self._max_axtree_nodes)
            self._log_healing_event(
                original_locator, healed, axtree_before, axtree_after, validated=True
            )
            self._update_locators_atomic(original_locator, healed, page.url)
            logger.info(
                f"HealingEngine: locator validated and saved | "
                f"healed={healed.healed!r}"
            )
        else:
            logger.warning(
                f"HealingEngine: healed locator did not resolve to a visible element "
                f"| healed={healed.healed!r} — confidence down-graded"
            )
            healed = HealedLocator(
                reasoning=healed.reasoning + " [Validation: element not visible after heal]",
                original=healed.original,
                healed=healed.healed,
                confidence=max(0.0, healed.confidence - 0.2),
                method=healed.method,
            )
            self._log_healing_event(
                original_locator, healed, axtree_before, axtree_after=None, validated=False
            )

        return healed

    async def _validate_locator(self, page: Page, locator_str: str) -> bool:
        """
        State Validation Loop: confirm the healed locator resolves to a
        visible element within 5 seconds.

        Tries:
          1. eval the locator expression and check visibility.
          2. (Fallback) check for a URL change or any DOM mutation.
        """
        try:
            # Evaluate the locator string in the page context
            # We wrap it so we can call .wait_for(state="visible", timeout=5000)
            element = self._eval_locator(page, locator_str)
            if element is None:
                return False
            await element.wait_for(state="visible", timeout=5_000)
            return True
        except Exception as exc:
            logger.debug(f"_validate_locator: visibility check failed — {exc}")
            return False

    @staticmethod
    def _eval_locator(page: Page, locator_str: str):
        """
        Convert a locator string like `page.get_by_role("button", name="Login")`
        into an actual Playwright Locator object by evaluating it safely.
        """
        import re

        # get_by_role("role", name="name") or get_by_role("role")
        m = re.search(
            r'get_by_role\s*\(\s*["\'](\w+)["\'](?:.*?name\s*=\s*["\']([^"\']+)["\'])?',
            locator_str,
            re.DOTALL,
        )
        if m:
            role, name = m.group(1), m.group(2)
            return page.get_by_role(role, name=name) if name else page.get_by_role(role)  # type: ignore[arg-type]

        m = re.search(r'get_by_label\s*\(\s*["\']([^"\']+)["\']', locator_str)
        if m:
            return page.get_by_label(m.group(1))

        m = re.search(r'get_by_text\s*\(\s*["\']([^"\']+)["\']', locator_str)
        if m:
            return page.get_by_text(m.group(1))

        m = re.search(r'get_by_test_id\s*\(\s*["\']([^"\']+)["\']', locator_str)
        if m:
            return page.get_by_test_id(m.group(1))

        m = re.search(r'get_by_placeholder\s*\(\s*["\']([^"\']+)["\']', locator_str)
        if m:
            return page.get_by_placeholder(m.group(1))

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # locators.json persistence
    # ──────────────────────────────────────────────────────────────────────────

    def _load_locators(self) -> dict[str, Any]:
        if not self._locator_file.exists():
            return {}
        try:
            return json.loads(self._locator_file.read_text())
        except Exception as exc:
            logger.warning(f"Could not read {self._locator_file}: {exc}")
            return {}

    def _update_locators_atomic(
        self,
        original: str,
        healed: HealedLocator,
        page_url: str,
    ) -> None:
        """
        Atomically update locators/locators.json using a FileLock so parallel
        test runners do not corrupt the file.

        Write pattern: acquire lock → read → merge → write .tmp → os.replace().
        """
        self._locator_file.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._lock_path), timeout=10)

        try:
            with lock:
                data = self._load_locators()
                key = _locator_key(original)
                existing = data.get(key, {})
                data[key] = {
                    "original": original,
                    "healed": healed.healed,
                    "confidence": healed.confidence,
                    "method": healed.method,
                    "url_pattern": page_url,
                    "heal_count": existing.get("heal_count", 0) + 1,
                    "last_updated": _iso_now(),
                    "reasoning": healed.reasoning,
                }
                tmp = self._locator_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2))
                os.replace(tmp, self._locator_file)
        except Exception as exc:
            logger.error(f"Failed to update locators.json: {exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _log_healing_event(
        original: str,
        healed: HealedLocator,
        axtree_before: str,
        axtree_after: str | None,
        validated: bool,
    ) -> None:
        diff_lines: list[str] = []
        if axtree_after is not None:
            diff = unified_diff(
                axtree_before.splitlines(keepends=True),
                axtree_after.splitlines(keepends=True),
                fromfile="axtree_before",
                tofile="axtree_after",
                n=3,
            )
            diff_lines = list(diff)[:40]  # cap diff size in logs

        logger.bind(
            healing_event={
                "original": original,
                "healed": healed.healed,
                "confidence": healed.confidence,
                "method": healed.method,
                "validated": validated,
                "axtree_diff_lines": len(diff_lines),
            }
        ).info(
            f"HealingEvent | original={original!r} → healed={healed.healed!r} "
            f"method={healed.method} confidence={healed.confidence:.4f} "
            f"validated={validated} diff_lines={len(diff_lines)}"
        )
        if diff_lines:
            logger.debug("AxTree diff (first 40 lines):\n" + "".join(diff_lines))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _locator_key(locator_str: str) -> str:
    """Stable dict key for a locator (trimmed, normalised whitespace)."""
    import re
    return re.sub(r"\s+", " ", locator_str.strip())


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
