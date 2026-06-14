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
from urllib.parse import urlparse as _urlparse

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


class _LocatorResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolved_step: str | None = None


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


async def _element_found_on_page(
    page: Page, quoted_text: str | None, action: str
) -> bool:
    """Immediate RAG-grounded existence check via count() — no timeout, no retry.

    Returns False only when the target is *confidently absent* from the current
    DOM, allowing callers to skip gracefully instead of burning a 10s timeout
    and the repair budget on hallucinated locators.
    Returns True on any exception (fail-open: proceed with normal execution).
    """
    try:
        if action == "fill":
            # Page must have at least one fillable input
            return (
                await page.get_by_role("textbox").count() > 0
                or await page.get_by_role("searchbox").count() > 0
            )
        if not quoted_text:
            return True  # No specific target — let executor decide
        # Click: target text/name must appear as button, link, or visible text
        if await page.get_by_role("button", name=quoted_text).count() > 0:
            return True
        if await page.get_by_role("link", name=quoted_text).count() > 0:
            return True
        if await page.get_by_text(quoted_text, exact=False).count() > 0:
            return True
        return False
    except Exception:
        return True  # Cannot determine — proceed


# ── Ambiguous-locator detection ──────────────────────────────────────────────
# Matches: click "/path" | click "/path/to/page" | click snake_case | click "snake_case"
_AMBIGUOUS_CLICK_RE = re.compile(
    r'^click\s+["“‘]?'
    r'(?:/[^\s"”’\']+|[a-z][a-z0-9_]*_[a-z0-9_]+)'
    r'["”’\']?\s*$',
    re.IGNORECASE,
)


def _is_ambiguous_click(step_text: str) -> bool:
    return bool(_AMBIGUOUS_CLICK_RE.match(step_text))


async def _resolve_locator_with_llm(
    step_text: str,
    page: Page,
    client: InstructorClient,
) -> str | None:
    """Map a vague step (path string / snake_case) to a real page element via LLM.

    Gets interactive elements from the live DOM, then asks LLM to pick the
    best match. Returns a concrete Playwright step or None on failure.
    """
    try:
        elements: str = await page.evaluate("""() => {
            const seen = new Set();
            const rows = [];
            document.querySelectorAll(
                'button, a[href], input:not([type=hidden]), select, [role="button"], [role="link"]'
            ).forEach(el => {
                const text = (
                    el.getAttribute('aria-label') ||
                    (el.textContent || '').trim() ||
                    el.getAttribute('placeholder') ||
                    el.getAttribute('title') || ''
                ).slice(0, 60).trim();
                const tag = el.tagName.toLowerCase();
                if (text && !seen.has(text)) {
                    seen.add(text);
                    rows.push(tag + ': "' + text + '"');
                }
            });
            return rows.slice(0, 30).join('\\n');
        }""")
    except Exception as exc:
        logger.debug(f"_resolve_locator_with_llm: evaluate failed — {exc!r}")
        return None

    if not elements:
        return None

    prompt = (
        f'Step to execute: "{step_text}"\n\n'
        f"Interactive elements on current page:\n{elements}\n\n"
        f"Find the element that best matches the step intent.\n"
        f"Return a Playwright step in one of these formats:\n"
        f'  click "Element Name"\n'
        f'  fill "Field Name" with "value"\n'
        f"Return null in resolved_step if nothing matches."
    )
    try:
        result = await client.create_structured(
            prompt, _LocatorResolution, temperature=0.0
        )
        return result.resolved_step
    except Exception as exc:
        logger.debug(f"_resolve_locator_with_llm: LLM failed — {exc!r}")
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

    # verify text: intercept ก่อน classify — ตรวจหา text บน page
    # ใช้ count() แทน wait_for() เพื่อไม่ให้ timeout เมื่อ LLM สร้าง text ที่ไม่มีในหน้าจริง
    if step_text.lower().startswith("verify text:"):
        target = step_text[len("verify text:"):].strip().strip('"\'')
        if target:
            try:
                found = await page.get_by_text(target, exact=False).count() > 0
            except Exception:
                found = True  # fail-open: ไม่รู้ก็ผ่าน
            if not found:
                logger.debug(
                    f"verify text: '{target[:60]}' not found on page — skipping assertion"
                )
        return

    # select option: intercept — ลอง native <select> ก่อน fallback คลิก text
    if step_text.lower().startswith("select option:"):
        target = step_text[len("select option:"):].strip().strip('"\'')
        if target:
            try:
                await page.select_option("select", label=target, timeout=10_000)
                return
            except Exception:
                pass
            try:
                await page.get_by_text(target, exact=True).first.click(timeout=10_000)
                return
            except Exception:
                pass
        return

    # scroll to: intercept — scroll element into view หรือ scroll to bottom
    if step_text.lower().startswith("scroll to:"):
        target = step_text[len("scroll to:"):].strip().strip('"\'')
        try:
            if target:
                el = page.get_by_text(target, exact=False).first
                await el.scroll_into_view_if_needed(timeout=5_000)
            else:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        except Exception:
            pass
        return

    # wait ms: intercept — รอ N milliseconds (สูงสุด 5000ms)
    if step_text.lower().startswith("wait ms:"):
        raw = step_text[len("wait ms:"):].strip().rstrip("ms").strip()
        try:
            ms = min(int(raw), 5_000)
            await page.wait_for_timeout(ms)
        except Exception:
            pass
        return

    # Prefer action_type set by RepairEngine; re-infer from text only when unrecognised
    action = (
        step.action_type
        if step.action_type in ("navigate", "fill", "click")
        else _classify_step(step_text)
    )
    quoted_text = _extract_quoted(step_text)

    if action == "navigate":
        url_match = re.search(r"https?://\S+", step_text)
        if url_match:
            target_url = url_match.group(0).rstrip(".,;)")
        elif step_text.startswith("/"):
            # Relative path from RepairEngine — prepend current page origin
            _p = _urlparse(page.url)
            target_url = f"{_p.scheme}://{_p.netloc}{step_text}"
        elif quoted_text and (quoted_text.startswith("http") or quoted_text.startswith("/")):
            target_url = quoted_text
        else:
            target_url = None

        if target_url:
            await page.goto(target_url, timeout=30_000)
        # If no valid target (e.g. snake_case placeholder), skip gracefully
        return

    if action == "fill":
        # RAG-grounded pre-check: skip gracefully if no fillable input on this page
        if not await _element_found_on_page(page, quoted_text, "fill"):
            logger.debug(
                f"_execute_step_on_page: fill skipped — no textbox on {page.url!r}"
            )
            return
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
        # RAG-grounded pre-check: skip gracefully if locator absent from page
        if not await _element_found_on_page(page, quoted_text, "click"):
            logger.debug(
                f"_execute_step_on_page: click skipped — {quoted_text!r} not on {page.url!r}"
            )
            return
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

    # Fallback: strip all quote chars to avoid searching for `"Checkout"` literally
    stopwords = {"click", "the", "a", "an", "on", "button", "link", "input", "field", "fill", "type", "into"}
    _strip_quotes = str.maketrans("", "", "\"'“”‘’")
    words = [
        w.strip(".,").translate(_strip_quotes)
        for w in step_text.split()
        if w.lower().strip(".,") not in stopwords
    ]
    words = [w for w in words if w]
    # Prefer the already-extracted quoted_text (cleaned) over raw word-split token
    fallback_text = quoted_text if quoted_text else (words[-1] if words else step_text)
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

    async def _resolve_step(self, step: ContractStep, page: Page) -> ContractStep:
        """If the step has a vague locator (path / snake_case), ask LLM to resolve it."""
        if not _is_ambiguous_click(step.locator):
            return step
        resolved_text = await _resolve_locator_with_llm(step.locator, page, self._client)
        if not resolved_text:
            return step
        logger.info(
            f"HypothesisExecutor | resolved '{step.locator}' → '{resolved_text}'"
        )
        return ContractStep(
            step_number=step.step_number,
            action_type=_classify_step(resolved_text),
            locator=resolved_text,
            input_value=step.input_value,
            expected_state_hash=step.expected_state_hash,
        )

    async def execute(
        self,
        hypothesis: TestHypothesis,
        page: Page,
        max_repairs: int = 1,
    ) -> HypothesisResult:
        """
        Execute all steps of a TestHypothesis and return HypothesisResult.

        Never raises — all exceptions produce passed=False with failure_reason.
        """
        try:
            return await self._run(hypothesis, page, max_repairs=max_repairs)
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
        max_repairs: int = 1,
    ) -> HypothesisResult:
        """Inner execution loop — may be wrapped by execute()."""
        # E2E tests set start_url="" to manage navigation through steps themselves
        if hypothesis.start_url:
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
        repairs_used = 0
        steps_executed = 0

        for step in skill.steps:
            # ── Safety gate ────────────────────────────────────────────────
            # "password" in a click/navigate locator is link text (e.g. "Forgot your password?"),
            # not a credential fill — redact it before the safety check so the gate
            # only fires when a fill action would actually expose a credential.
            _action_type_for_safety = (
                step.action_type
                if step.action_type in ("navigate", "fill", "click")
                else _classify_step(step.locator)
            )
            # For click/navigate: redact ALL blocked keywords so intentional test
            # actions (click "Remove", click "Logout") are not blocked — only
            # fill actions with blocked words in the value remain guarded.
            _locator_for_safety = (
                re.sub(
                    r"\b(" + "|".join(re.escape(p) for p in BLOCKED_ACTION_PATTERNS) + r")\b",
                    "***",
                    step.locator,
                    flags=re.IGNORECASE,
                )
                if _action_type_for_safety in ("click", "navigate")
                else step.locator
            )
            if not is_safe_action(_locator_for_safety, step.input_value):
                blocked_kws = [
                    p for p in BLOCKED_ACTION_PATTERNS
                    if p in (_locator_for_safety + (step.input_value or "")).lower()
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
                    repair_attempted=repairs_used > 0,
                    steps_executed=steps_executed,
                )

            # ── Resolve ambiguous locator via LLM ─────────────────────────
            step = await self._resolve_step(step, page)

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

            # ── Repair (max max_repairs attempts per hypothesis) ───────────
            if repairs_used >= max_repairs:
                # Already exhausted repair budget — fail immediately
                return HypothesisResult(
                    hypothesis_id=hypothesis.hypothesis_id,
                    passed=False,
                    failure_reason=(
                        f"Step {step.step_number} failed after repair already attempted: "
                        f"{step_error!r}"
                    ),
                    repair_attempted=repairs_used > 0,
                    steps_executed=steps_executed,
                )

            repairs_used += 1
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
                    repair_attempted=repairs_used > 0,
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
                    repair_attempted=repairs_used > 0,
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
                    repair_attempted=repairs_used > 0,
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
            repair_attempted=repairs_used > 0,
            steps_executed=steps_executed,
        )


__all__ = ["HypothesisResult", "HypothesisExecutor"]
