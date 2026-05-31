"""
HypothesisExecutor — Sprint 5.

Converts a TestHypothesis → ContractSkill → executes each step via Playwright
pattern-matching → applies RepairEngine on first failure.

Execution model
---------------
1. Build a synthetic ContractSkill from hypothesis.steps (semantic strings).
2. For every ContractStep:
   a. Enforce BLOCKED_ACTION_PATTERNS safety gate.
   b. Determine Playwright action from step text keywords (no eval, no LLM).
   c. On TimeoutError / Exception → RepairEngine cascade once (max 1 per hypothesis).
3. Return HypothesisResult (never raises).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.contractskill.compiler import ContractSkill, ContractStep
from src.contractskill.repair import RepairEngine
from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS, SFGStore, is_safe_action
from src.explorer.hypothesis import TestHypothesis
from src.llm.instructor_client import InstructorClient

# ─────────────────────────────────────────────────────────────────────────────
# HypothesisResult schema
# ─────────────────────────────────────────────────────────────────────────────


class HypothesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    passed: bool
    failure_reason: str | None = None
    repair_attempted: bool = False
    steps_executed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_FILL_KEYWORDS = ("fill", "type", "enter", "input")
_CLICK_KEYWORDS = ("click", "press", "tap", "select", "check", "submit")
_NAVIGATE_KEYWORDS = ("navigate", "goto", "go to", "open", "visit")


def _classify_step(step_text: str) -> str:
    """Return 'fill', 'navigate', or 'click' based on keywords in step_text."""
    lower = step_text.lower()
    for kw in _NAVIGATE_KEYWORDS:
        if kw in lower:
            return "navigate"
    for kw in _FILL_KEYWORDS:
        if kw in lower:
            return "fill"
    return "click"


_QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),  # Unicode left/right double quotes
    ("‘", "’"),  # Unicode left/right single quotes
]


def _extract_quoted(text: str) -> str | None:
    """Extract the first quoted substring from text (ASCII or Unicode quotes), or None."""
    for open_q, close_q in _QUOTE_PAIRS:
        start = text.find(open_q)
        if start != -1:
            end = text.find(close_q, start + 1)
            if end != -1:
                return text[start + 1 : end]
    return None


async def _execute_step_on_page(step: ContractStep, page: Page) -> None:
    """
    Execute a single ContractStep on a live Playwright Page.

    Strategy (no eval, no LLM):
    - Classify the step's locator text by keyword.
    - Use page.get_by_* APIs in a fixed precedence order.
    - Raises on Playwright timeout or any other exception.
    """
    step_text = step.locator
    action = _classify_step(step_text)
    quoted_text = _extract_quoted(step_text)

    if action == "navigate":
        # Try to find a URL in the step text; fall back to quoted text.
        url_match = re.search(r"https?://\S+", step_text)
        if url_match:
            target_url = url_match.group(0).rstrip(".,;)")
        elif quoted_text:
            target_url = quoted_text
        else:
            # Cannot determine URL — treat as a click on the text
            action = "click"
            target_url = None

        if action == "navigate" and target_url:
            await page.goto(target_url, timeout=15_000)
            return
        if action == "navigate" and not target_url:
            # Descriptive navigate step with no URL (e.g. "Navigate to the app")
            # — page is already at start_url, skip this step.
            return

    if action == "fill":
        value = step.input_value or quoted_text or "test"
        press_enter = "enter" in step_text.lower() or "press" in step_text.lower()
        # Try label first, then textbox role
        if quoted_text:
            try:
                await page.get_by_label(quoted_text).fill(value, timeout=10_000)
                if press_enter:
                    await page.get_by_label(quoted_text).press("Enter")
                return
            except Exception:
                pass
        await page.get_by_role("textbox").first.fill(value, timeout=10_000)
        if press_enter:
            await page.get_by_role("textbox").first.press("Enter")
        return

    # Default: click
    # Checkbox/complete steps: try checkbox role before generic text matching
    if any(kw in step_text.lower() for kw in ("checkbox", "complet", "mark", "toggle", "check the")):
        try:
            await page.get_by_role("checkbox").first.click(timeout=10_000)
            return
        except Exception:
            pass

    # Priority: exact quoted text → role=button → role=link → generic text
    if quoted_text:
        try:
            await page.get_by_role("button", name=quoted_text).click(timeout=10_000)
            return
        except Exception:
            pass
        try:
            await page.get_by_role("link", name=quoted_text).click(timeout=10_000)
            return
        except Exception:
            pass
        try:
            await page.get_by_text(quoted_text).first.click(timeout=10_000)
            return
        except Exception:
            pass

    # No quoted text — extract the most meaningful word from step_text
    stopwords = {"click", "the", "a", "an", "on", "button", "link", "input", "field", "fill", "type", "into"}
    words = [w.strip(".,") for w in step_text.split() if w.lower().strip(".,") not in stopwords]
    fallback_text = words[-1] if words else step_text
    await page.get_by_text(fallback_text, exact=False).first.click(timeout=10_000)


def _build_synthetic_skill(hypothesis: TestHypothesis) -> ContractSkill:
    """
    Build a ContractSkill from a TestHypothesis's semantic steps list.

    Each step string becomes a ContractStep where:
      - action_type is inferred by keyword
      - locator stores the semantic step text (RepairEngine resolves via AOM)
      - input_value is None (filled from step text at execution time)
      - expected_state_hash is empty string (no pre-verified target)
    """
    steps: list[ContractStep] = [
        ContractStep(
            step_number=i + 1,
            action_type=_classify_step(step_text),
            locator=step_text,
            input_value=None,
            expected_state_hash="",
        )
        for i, step_text in enumerate(hypothesis.steps)
    ]

    skill_id = hashlib.sha256(
        (hypothesis.hypothesis_id + hypothesis.goal).encode()
    ).hexdigest()

    return ContractSkill(
        skill_id=skill_id,
        goal=hypothesis.goal,
        target_url=hypothesis.start_url,
        domain="",
        preconditions=hypothesis.preconditions,
        steps=steps,
        postconditions=[hypothesis.expected_outcome],
        repair_operators=["SelReplace", "PreInsert", "ArgCorrect"],
        created_at_iso=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisExecutor
# ─────────────────────────────────────────────────────────────────────────────


class HypothesisExecutor:
    """
    Converts a TestHypothesis into a ContractSkill and executes it step-by-step
    against a live Playwright Page.

    Safety:
    - BLOCKED_ACTION_PATTERNS checked before every action.
    - At most 1 repair attempt per hypothesis (prevents OOM loops).
    - CDP/AOMExtractor only — page.accessibility is never touched.
    - Never raises: all errors captured in HypothesisResult.
    """

    def __init__(
        self,
        instructor_client: InstructorClient,
        sfg_store: SFGStore,
    ) -> None:
        self._client = instructor_client
        self._sfg_store = sfg_store

    async def execute(
        self,
        hypothesis: TestHypothesis,
        page: Page,
    ) -> HypothesisResult:
        """
        Execute all steps of a TestHypothesis and return HypothesisResult.

        Never raises — all exceptions produce passed=False with failure_reason.
        """
        try:
            return await self._run(hypothesis, page)
        except Exception as exc:  # final safety net
            logger.error(
                f"HypothesisExecutor.execute | unexpected top-level error "
                f"hypothesis_id={hypothesis.hypothesis_id!r}: {exc!r}"
            )
            return HypothesisResult(
                hypothesis_id=hypothesis.hypothesis_id,
                passed=False,
                failure_reason=f"Unexpected error: {exc!r}",
            )

    async def _run(
        self,
        hypothesis: TestHypothesis,
        page: Page,
    ) -> HypothesisResult:
        """Inner execution loop — may be wrapped by execute()."""
        await page.goto(hypothesis.start_url, wait_until="domcontentloaded", timeout=30_000)

        # Set up preconditions: if hypothesis requires an existing todo, add one first.
        needs_existing = any(
            "existing" in p.lower() or ("task" in p.lower() and "no" not in p.lower())
            for p in hypothesis.preconditions
        )
        if needs_existing:
            # Extract expected todo name from hypothesis steps (use first quoted text found)
            todo_name = "Buy milk"
            for step_text in hypothesis.steps:
                q = _extract_quoted(step_text)
                if q and len(q) < 40 and not any(
                    kw in q.lower() for kw in ("navigate", "click", "enter", "press", "button")
                ):
                    todo_name = q
                    break
            try:
                await page.get_by_role("textbox").first.fill(todo_name, timeout=5_000)
                await page.get_by_role("textbox").first.press("Enter")
            except Exception:
                pass  # precondition setup is best-effort

        skill = _build_synthetic_skill(hypothesis)
        repair_attempted = False
        steps_executed = 0

        for step in skill.steps:
            # ── Safety gate ────────────────────────────────────────────────
            if not is_safe_action(step.locator, step.input_value):
                blocked_kws = [
                    p for p in BLOCKED_ACTION_PATTERNS
                    if p in (step.locator + (step.input_value or "")).lower()
                ]
                reason = (
                    f"Step {step.step_number} blocked by BLOCKED_ACTION_PATTERNS "
                    f"(matched: {blocked_kws!r}): locator={step.locator!r}"
                )
                logger.warning(
                    f"HypothesisExecutor | {reason} "
                    f"hypothesis_id={hypothesis.hypothesis_id!r}"
                )
                return HypothesisResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    passed=False,
                    failure_reason=reason,
                    repair_attempted=repair_attempted,
                    steps_executed=steps_executed,
                )

            # ── Execute step ───────────────────────────────────────────────
            step_error: Exception | None = None
            try:
                await _execute_step_on_page(step, page)
                steps_executed += 1
                logger.debug(
                    f"HypothesisExecutor | step {step.step_number} OK "
                    f"hypothesis_id={hypothesis.hypothesis_id!r}"
                )
                continue  # success → next step
            except Exception as exc:
                step_error = exc
                logger.warning(
                    f"HypothesisExecutor | step {step.step_number} failed: {exc!r} "
                    f"hypothesis_id={hypothesis.hypothesis_id!r}"
                )

            # ── Repair (max 1 attempt per hypothesis) ──────────────────────
            if repair_attempted:
                # Already used our one repair budget — fail immediately
                return HypothesisResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    passed=False,
                    failure_reason=(
                        f"Step {step.step_number} failed after repair already attempted: "
                        f"{step_error!r}"
                    ),
                    repair_attempted=True,
                    steps_executed=steps_executed,
                )

            repair_attempted = True
            engine = RepairEngine(
                instructor_client=self._client,
                sfg_store=self._sfg_store,
            )

            patched_skill: ContractSkill | None = None
            try:
                patched_skill = await engine.repair(
                    skill,
                    step,
                    repr(step_error),
                    page,
                )
            except Exception as repair_exc:
                logger.warning(
                    f"HypothesisExecutor | RepairEngine raised: {repair_exc!r} "
                    f"hypothesis_id={hypothesis.hypothesis_id!r}"
                )

            if patched_skill is None:
                # Repair could not patch — fail
                return HypothesisResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    passed=False,
                    failure_reason=(
                        f"Step {step.step_number} failed and RepairEngine returned None: "
                        f"{step_error!r}"
                    ),
                    repair_attempted=True,
                    steps_executed=steps_executed,
                )

            # Find the patched version of the current step (same step_number)
            patched_step = next(
                (s for s in patched_skill.steps if s.step_number == step.step_number),
                None,
            )
            if patched_step is None:
                return HypothesisResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    passed=False,
                    failure_reason=(
                        f"Step {step.step_number} not found in patched skill steps "
                        f"after repair."
                    ),
                    repair_attempted=True,
                    steps_executed=steps_executed,
                )

            # Retry the repaired step
            try:
                await _execute_step_on_page(patched_step, page)
                steps_executed += 1
                # Update skill reference so subsequent steps use the patched skill
                skill = patched_skill
                logger.info(
                    f"HypothesisExecutor | step {step.step_number} succeeded after repair "
                    f"hypothesis_id={hypothesis.hypothesis_id!r}"
                )
            except Exception as retry_exc:
                return HypothesisResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    passed=False,
                    failure_reason=(
                        f"Step {step.step_number} failed even after repair: "
                        f"{retry_exc!r}"
                    ),
                    repair_attempted=True,
                    steps_executed=steps_executed,
                )

        # All steps passed
        logger.info(
            f"HypothesisExecutor | all {steps_executed} steps passed "
            f"hypothesis_id={hypothesis.hypothesis_id!r}"
        )
        return HypothesisResult(
            hypothesis_id=hypothesis.hypothesis_id,
            passed=True,
            repair_attempted=repair_attempted,
            steps_executed=steps_executed,
        )


__all__ = ["HypothesisResult", "HypothesisExecutor"]
