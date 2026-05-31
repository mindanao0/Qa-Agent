"""Generator agent — turns the Planner's Strategy into concrete ActionSteps.

The agent uses a hard constrained JSON prompt and validates every emitted step
against the Pydantic schema. URL-keyed caches short-circuit the LLM whenever
the repository already has a trajectory for the current page.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError
from rich.console import Console

from src.core.contract_skill import ContractSkillRepository
from src.core.state_schema import ActionStep, AgentPipelineState

console = Console()

GENERATOR_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_INSTRUCTION = """You are a Playwright Action Generator.
Output ONLY a valid JSON array. NO markdown. NO explanation.
Schema per element:
{
  "step_index": int,
  "intent": str,
  "locator_strategy": "getByRole"|"getByLabel"|"getByTestId",
  "locator_value": str,
  "action_type": "click"|"fill"|"select"|"navigate"|"press",
  "payload": str|null
}
Rules:
- NEVER use CSS classes or XPath
- NEVER use page.locator() with structural selectors
- Max 8 steps per batch
- Group independent fill actions into sequential steps (Bulk Action pattern)
"""


def _strip_fences(text: str) -> str:
    """Remove ```json fences before json.loads."""
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


async def generator_node(state: AgentPipelineState, config: dict) -> dict:
    """LangGraph node: produce a validated ContractSkill artifact."""
    cfg = (config or {}).get("configurable", {}) or {}
    ollama_url = (cfg.get("ollama_url") or OLLAMA_URL).rstrip("/")
    repo: ContractSkillRepository = cfg.get("contract_repo") or ContractSkillRepository()

    current_url = state.get("current_url", "")
    task_goal = state.get("task_goal", "")
    sfg_nodes = state.get("sfg_nodes", []) or []
    snapshot_blurb = sfg_nodes[-1].accessibility_snapshot[:2000] if sfg_nodes else ""

    cached_steps = repo.check_cache(current_url) if current_url else None
    if cached_steps:
        contract = repo.compile_from_trajectory(
            goal=task_goal,
            steps=cached_steps,
            pre=[f"At {current_url}"] if current_url else [],
            post=[f"Goal satisfied: {task_goal}"] if task_goal else [],
        )
        return {
            "active_contract": contract,
            "execution_log": [
                "[GENERATOR] Cache HIT - bypassing inference",
                f"[GENERATOR] Contract compiled: {contract.skill_id}",
            ],
            "mode": "execute",
        }

    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\nTask:\n"
        + f"Goal: {task_goal}\n"
        + f"Current URL: {current_url}\n"
        + f"Snapshot (trimmed): {snapshot_blurb}\n"
        + "Output the JSON array now."
    )
    payload = {
        "model": GENERATOR_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    endpoint = ollama_url + "/api/generate"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            raw_text: str = resp.json().get("response", "")
    except Exception as exc:  # noqa: BLE001
        console.log(f"[GENERATOR] Ollama call failed: {exc}")
        return {
            "execution_log": [f"[GENERATOR] LLM call failed: {exc}"],
            "mode": "heal",
            "failed_step_index": 0,
        }

    text = _strip_fences(raw_text)
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        console.log(f"[GENERATOR] JSON decode failed: {exc}")
        return {
            "execution_log": ["[GENERATOR] JSON decode failed → heal"],
            "mode": "heal",
            "failed_step_index": 0,
        }

    if not isinstance(parsed, list):
        return {
            "execution_log": ["[GENERATOR] Output not a JSON array → heal"],
            "mode": "heal",
            "failed_step_index": 0,
        }

    steps: list[ActionStep] = []
    for raw in parsed[:8]:
        if not isinstance(raw, dict):
            return {
                "execution_log": ["[GENERATOR] Non-object element in array → heal"],
                "mode": "heal",
                "failed_step_index": 0,
            }
        raw.setdefault("payload", None)
        try:
            step = ActionStep.model_validate(raw)
        except ValidationError as exc:
            console.log(f"[GENERATOR] step validation failed: {exc}")
            return {
                "execution_log": [f"[GENERATOR] Validation failed: {exc.errors()[:1]}"],
                "mode": "heal",
                "failed_step_index": 0,
            }
        steps.append(step)

    if not steps:
        return {
            "execution_log": ["[GENERATOR] No steps emitted → heal"],
            "mode": "heal",
            "failed_step_index": 0,
        }

    renumbered = [
        step.model_copy(update={"step_index": idx}) for idx, step in enumerate(steps)
    ]

    contract = repo.compile_from_trajectory(
        goal=task_goal,
        steps=renumbered,
        pre=[f"At {current_url}"] if current_url else [],
        post=[f"Goal satisfied: {task_goal}"] if task_goal else [],
    )

    return {
        "active_contract": contract,
        "execution_log": [f"[GENERATOR] Contract compiled: {contract.skill_id}"],
        "mode": "execute",
    }


__all__ = ["generator_node"]
