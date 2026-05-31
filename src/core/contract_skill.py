"""ContractSkill repository: CRUD over artifacts plus the four patch operators.

The repository deliberately keeps business logic out of healer territory. Patch
operators may only mutate structural locator values, arg payloads, or insert
new pre/post steps. Anything that smells like a logic change is rejected
upstream in the healer agent.
"""
from __future__ import annotations

import hashlib
import uuid

from rich.console import Console

from src.core.state_schema import (
    ActionStep,
    ContractSkillArtifact,
    RecoveryRule,
)

console = Console()


class ContractSkillRepository:
    """In-memory store for ContractSkill artifacts and URL-keyed step caches."""

    def __init__(self) -> None:
        self.skills: dict[str, ContractSkillArtifact] = {}
        self.cache: dict[str, list[ActionStep]] = {}

    def compile_from_trajectory(
        self,
        goal: str,
        steps: list[ActionStep],
        pre: list[str],
        post: list[str],
    ) -> ContractSkillArtifact:
        """Wrap a known-good trajectory into a versioned contract artifact."""
        skill_id = str(uuid.uuid4())
        recovery_rules = [
            RecoveryRule(
                trigger_condition="NOT_FOUND",
                patch_operator="SelReplace",
                patch_payload="",
            ),
            RecoveryRule(
                trigger_condition="TIMEOUT",
                patch_operator="PreInsert",
                patch_payload="",
            ),
            RecoveryRule(
                trigger_condition="INPUT_INVALID",
                patch_operator="ArgCorrect",
                patch_payload="",
            ),
        ]
        artifact = ContractSkillArtifact(
            skill_id=skill_id,
            goal=goal,
            preconditions=list(pre),
            action_steps=list(steps),
            postconditions=list(post),
            recovery_rules=recovery_rules,
            version=1,
            status="active",
        )
        self.skills[skill_id] = artifact
        return artifact

    def apply_patch(
        self,
        artifact: ContractSkillArtifact,
        failed_step_idx: int,
        operator: str,
        patch_payload: str,
    ) -> ContractSkillArtifact:
        """Apply a single structural patch operator and bump the version."""
        steps = list(artifact.action_steps)
        if operator == "SelReplace":
            if not 0 <= failed_step_idx < len(steps):
                raise IndexError(
                    f"SelReplace target {failed_step_idx} out of bounds for {len(steps)} steps"
                )
            target = steps[failed_step_idx]
            steps[failed_step_idx] = target.model_copy(
                update={"locator_value": patch_payload}
            )
        elif operator == "ArgCorrect":
            if not 0 <= failed_step_idx < len(steps):
                raise IndexError(
                    f"ArgCorrect target {failed_step_idx} out of bounds for {len(steps)} steps"
                )
            target = steps[failed_step_idx]
            steps[failed_step_idx] = target.model_copy(update={"payload": patch_payload})
        elif operator == "PreInsert":
            new_step = ActionStep(
                step_index=failed_step_idx,
                intent=f"pre-inserted recovery step before {failed_step_idx}",
                locator_strategy="getByRole",
                locator_value=patch_payload,
                action_type="click",
                payload=None,
            )
            steps.insert(failed_step_idx, new_step)
        elif operator == "PostInsert":
            insertion_idx = failed_step_idx + 1
            new_step = ActionStep(
                step_index=insertion_idx,
                intent=f"post-inserted follow-up step after {failed_step_idx}",
                locator_strategy="getByRole",
                locator_value=patch_payload,
                action_type="click",
                payload=None,
            )
            steps.insert(insertion_idx, new_step)
        else:
            raise ValueError(f"Unknown patch operator: {operator}")

        renumbered: list[ActionStep] = []
        for new_idx, step in enumerate(steps):
            renumbered.append(step.model_copy(update={"step_index": new_idx}))

        patched = artifact.model_copy(
            update={
                "action_steps": renumbered,
                "version": artifact.version + 1,
                "status": "patched",
            }
        )
        self.skills[patched.skill_id] = patched
        return patched

    def get_cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def check_cache(self, url: str) -> list[ActionStep] | None:
        key = self.get_cache_key(url)
        cached = self.cache.get(key)
        if cached is None:
            return None
        return [step.model_copy() for step in cached]

    def write_cache(self, url: str, steps: list[ActionStep]) -> None:
        key = self.get_cache_key(url)
        self.cache[key] = [step.model_copy() for step in steps]


__all__ = ["ContractSkillRepository"]
