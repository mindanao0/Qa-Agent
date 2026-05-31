"""
RepairEngine — Sprint 4 / Cluster S4-D.

Applies minimal patch operators to a failed ContractSkill step before
falling back to a full SFG re-crawl.

Repair cascade (must follow this exact order):
  1. SelReplace  — re-discovers locator via AOM  (no LLM)
  2. ArgCorrect  — pattern-matches failure_sig    (no LLM)
  3. PreInsert   — infers missing navigation step (1 LLM call, T=0.0)
  4. None        — caller triggers full SFG re-crawl

Usage::

    engine = RepairEngine(instructor_client=client, sfg_store=sfg_store)
    patched = await engine.repair(skill, failed_step, failure_sig, page)
    if patched is None:
        # fall back to full SFG crawl
        ...
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.contractskill.compiler import ContractSkill, ContractStep
from src.contractskill.sfg import SFGStore
from src.llm.instructor_client import InstructorClient
from src.perception.aom_extractor import AOMExtractor, AOMNode, AOMSnapshot
from src.perception.locator_synthesizer import best_locator


# ─────────────────────────────────────────────────────────────────────────────
# LLM response schema for _pre_insert
# ─────────────────────────────────────────────────────────────────────────────


class MissingStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: str
    locator: str


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _flatten_nodes(node: AOMNode) -> list[AOMNode]:
    """Pre-order flat list of all AOMNodes in the tree."""
    result: list[AOMNode] = [node]
    for child in node.children:
        result.extend(_flatten_nodes(child))
    return result


def _node_to_attrs(node: AOMNode) -> dict[str, Any]:
    """Convert an AOMNode to an attrs dict accepted by best_locator()."""
    return {
        "role": node.role,
        "accessible_name": node.name,
        "_accessible_name": node.name,
        "tag": node.role,  # role doubles as tag for AOM nodes
    }


def _locator_intent(locator: str) -> tuple[str, str]:
    """
    Extract (role_hint, name_hint) from a locator string.

    Handles patterns like:
      role=button[name="Submit"]  → ("button", "submit")
      [aria-label="Search"]       → ("", "search")
      #my-id                      → ("", "my-id")
      anything else               → ("", locator.lower())
    """
    # role=ROLE[name="TEXT"]
    m = re.match(r'role=([^\[]+)\[name="([^"]+)"\]', locator)
    if m:
        return m.group(1).lower(), m.group(2).lower()

    # [aria-label="TEXT"]
    m = re.match(r'\[aria-label="([^"]+)"\]', locator)
    if m:
        return "", m.group(1).lower()

    # #id
    m = re.match(r"#(\S+)", locator)
    if m:
        return "", m.group(1).lower()

    return "", locator.lower()


# ─────────────────────────────────────────────────────────────────────────────
# RepairEngine
# ─────────────────────────────────────────────────────────────────────────────


class RepairEngine:
    """
    Applies minimal patch operators to a failed ContractStep.

    Cascade order:
      1. SelReplace  (no LLM, fast)
      2. ArgCorrect  (pattern match, no LLM)
      3. PreInsert   (1 LLM call, T=0.0)
      4. Return None → caller falls back to full SFG re-crawl
    """

    def __init__(
        self,
        instructor_client: InstructorClient,
        sfg_store: SFGStore,
    ) -> None:
        self._client = instructor_client
        self._sfg_store = sfg_store
        self._aom_extractor = AOMExtractor()

    # ── Public API ───────────────────────────────────────────────────────────

    async def repair(
        self,
        skill: ContractSkill,
        failed_step: ContractStep,
        failure_sig: str,
        page: Any,  # playwright.async_api.Page — typed as Any to avoid hard import
    ) -> ContractSkill | None:
        """
        Apply repair operators in cascade order.

        Returns a patched ContractSkill, or None if all operators fail
        (caller should trigger a full SFG re-crawl in that case).
        """
        logger.info(
            f"RepairEngine.repair | skill_id={skill.skill_id[:12]} "
            f"failed_step={failed_step.step_number} failure_sig={failure_sig!r}"
        )

        # 1. SelReplace — no LLM
        new_step = await self._sel_replace(failed_step, page)
        if new_step is not None:
            logger.info("RepairEngine: SelReplace succeeded")
            return self._replace_step(skill, failed_step, new_step)

        # 2. ArgCorrect — pattern match, no LLM
        new_step = await self._arg_correct(failed_step, failure_sig)
        if new_step is not None:
            logger.info("RepairEngine: ArgCorrect succeeded")
            return self._replace_step(skill, failed_step, new_step)

        # 3. PreInsert — 1 LLM call
        try:
            updated_skill = await self._pre_insert(skill, failed_step)
            logger.info("RepairEngine: PreInsert succeeded")
            return updated_skill
        except Exception as exc:
            logger.warning(f"RepairEngine: PreInsert failed: {exc!r}")

        # All operators exhausted
        logger.info(
            "RepairEngine: all operators exhausted — returning None (caller re-crawls)"
        )
        return None

    # ── Operator 1: SelReplace ───────────────────────────────────────────────

    async def _sel_replace(
        self,
        step: ContractStep,
        page: Any,
    ) -> ContractStep | None:
        """
        Re-discover the element via AOM and synthesise a fresh locator.

        No LLM calls.  Returns an updated ContractStep or None if the element
        cannot be found in the current AOM.
        """
        try:
            snapshot: AOMSnapshot = await self._aom_extractor.extract(page)
        except Exception as exc:
            logger.warning(f"RepairEngine._sel_replace: AOM extraction failed: {exc!r}")
            return None

        all_nodes = _flatten_nodes(snapshot.root_node)

        role_hint, name_hint = _locator_intent(step.locator)

        best_node: AOMNode | None = None

        if role_hint and name_hint:
            # Pass 1: exact role + exact name match (preferred)
            for node in all_nodes:
                node_role = node.role.lower()
                node_name = node.name.lower()
                if role_hint == node_role and name_hint == node_name:
                    best_node = node
                    break

            # Pass 2: exact role + substring name match
            if best_node is None:
                for node in all_nodes:
                    node_role = node.role.lower()
                    node_name = node.name.lower()
                    if role_hint == node_role and name_hint in node_name:
                        best_node = node
                        break

        elif name_hint:
            # Pass 1: exact name match (no role constraint)
            for node in all_nodes:
                node_name = node.name.lower()
                if name_hint == node_name:
                    best_node = node
                    break

            # Pass 2: substring name match
            if best_node is None:
                for node in all_nodes:
                    node_name = node.name.lower()
                    if name_hint in node_name:
                        best_node = node
                        break

        if best_node is None:
            logger.debug(
                f"RepairEngine._sel_replace: no AOM node matched "
                f"role_hint={role_hint!r} name_hint={name_hint!r}"
            )
            return None

        attrs = _node_to_attrs(best_node)

        DEGENERATE_LOCATORS = {"*", "div", "span", ""}
        new_locator = best_locator(attrs)
        if new_locator in DEGENERATE_LOCATORS:
            logger.debug(
                f"RepairEngine._sel_replace: best_locator returned degenerate "
                f"locator {new_locator!r} — skipping"
            )
            return None  # not a useful repair

        logger.debug(
            f"RepairEngine._sel_replace: updated locator "
            f"{step.locator!r} → {new_locator!r}"
        )

        # ContractStep is frozen=True — use model_copy to create updated version
        return step.model_copy(update={"locator": new_locator})

    # ── Operator 2: ArgCorrect ───────────────────────────────────────────────

    async def _arg_correct(
        self,
        step: ContractStep,
        failure_sig: str,
    ) -> ContractStep | None:
        """
        Pattern-match failure_sig against known correction rules.

        No LLM calls.  Returns an updated ContractStep or None.
        """
        sig = failure_sig.upper()

        if sig == "WRONG_URL":
            # Try to correct the URL embedded in the locator
            # Look for http(s) URL in locator and strip query string
            m = re.search(r"https?://[^\s\"'>]+", step.locator)
            if m:
                corrected_url = m.group(0).split("?")[0]
                new_locator = step.locator.replace(m.group(0), corrected_url)
                logger.debug(
                    f"RepairEngine._arg_correct WRONG_URL: "
                    f"{step.locator!r} → {new_locator!r}"
                )
                return step.model_copy(update={"locator": new_locator})
            return None

        elif sig == "WRONG_ROLE":
            # The role is a test persona (admin/user), not the ARIA role in the locator.
            # Try toggling input_value if it contains a role token.
            current_val = (step.input_value or "").lower()
            if "admin" in current_val:
                new_val = current_val.replace("admin", "user")
                logger.debug(
                    f"RepairEngine._arg_correct WRONG_ROLE: "
                    f"input_value {step.input_value!r} → {new_val!r}"
                )
                return step.model_copy(update={"input_value": new_val})
            elif "user" in current_val:
                new_val = current_val.replace("user", "admin")
                logger.debug(
                    f"RepairEngine._arg_correct WRONG_ROLE: "
                    f"input_value {step.input_value!r} → {new_val!r}"
                )
                return step.model_copy(update={"input_value": new_val})
            else:
                return None  # cannot pattern-match

        elif sig == "WRONG_VALUE":
            # Replace input_value with empty string
            logger.debug(
                f"RepairEngine._arg_correct WRONG_VALUE: "
                f"clearing input_value={step.input_value!r}"
            )
            return step.model_copy(update={"input_value": ""})

        else:
            # Cannot pattern-match
            logger.debug(
                f"RepairEngine._arg_correct: unrecognised failure_sig={failure_sig!r} — skipping"
            )
            return None

    # ── Operator 3: PreInsert ────────────────────────────────────────────────

    async def _pre_insert(
        self,
        skill: ContractSkill,
        failed_step: ContractStep,
    ) -> ContractSkill:
        """
        Infer a missing navigation step and insert it before failed_step.

        Makes exactly 1 LLM call (T=0.0).  Falls back to a generic navigate
        step if the LLM call fails.
        """
        prompt = (
            f"The step '{failed_step.action_type} {failed_step.locator}' failed. "
            "What navigation step is likely missing before it? "
            "Respond with action_type (click/fill/navigate) and locator only."
        )

        new_action_type = "navigate"
        new_locator = skill.target_url or "/"

        try:
            result: MissingStep = await self._client.create_structured(
                prompt=prompt,
                response_model=MissingStep,
                temperature=0.0,
            )
            new_action_type = result.action_type
            new_locator = result.locator
            logger.debug(
                f"RepairEngine._pre_insert: LLM inferred "
                f"action_type={new_action_type!r} locator={new_locator!r}"
            )
        except Exception as exc:
            logger.warning(
                f"RepairEngine._pre_insert: LLM call failed ({exc!r}); "
                f"inserting generic navigate step to {new_locator!r}"
            )

        # Build the new step with a temporary step_number; re-number below
        insert_step = ContractStep(
            step_number=0,  # placeholder
            action_type=new_action_type,
            locator=new_locator,
            input_value=None,
            expected_state_hash="",
        )

        # Insert before failed_step and re-number
        failed_idx = next(
            (i for i, s in enumerate(skill.steps) if s.step_number == failed_step.step_number),
            len(skill.steps),
        )
        new_steps: list[ContractStep] = []
        for i, s in enumerate(skill.steps[:failed_idx]):
            new_steps.append(s.model_copy(update={"step_number": i + 1}))
        # Insert the new step
        new_steps.append(insert_step.model_copy(update={"step_number": len(new_steps) + 1}))
        # Continue with remaining steps (failed_step onward)
        for s in skill.steps[failed_idx:]:
            new_steps.append(s.model_copy(update={"step_number": len(new_steps) + 1}))

        return skill.model_copy(update={"steps": new_steps})

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _replace_step(
        skill: ContractSkill,
        failed_step: ContractStep,
        new_step: ContractStep,
    ) -> ContractSkill:
        """Return a new ContractSkill with failed_step replaced by new_step."""
        new_steps = [
            new_step if s.step_number == failed_step.step_number else s
            for s in skill.steps
        ]
        return skill.model_copy(update={"steps": new_steps})


__all__ = ["RepairEngine", "MissingStep"]
