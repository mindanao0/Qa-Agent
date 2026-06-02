"""Healer agent — emits the smallest possible structural patch for a failed step.

The healer is intentionally restricted:
  * Only the four allowed patch operators may be applied.
  * Any indication that the LLM wants to change business logic, requirements,
    or the overall flow flips the mode to ``human_review`` instead of patching.
  * Three failed repair attempts also escalate to ``human_review``.
"""
from __future__ import annotations

import json
from typing import Any

from rich.console import Console

from src.core.contract_skill import ContractSkillRepository
from src.core.state_schema import AgentPipelineState
from src.llm.adapter import OllamaAdapter

console = Console()

HEALER_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
ALLOWED_OPERATORS = {"SelReplace", "PreInsert", "ArgCorrect", "PostInsert"}
LOGIC_KEYWORDS = ("logic", "requirement", "business", "flow")


def _strip_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


async def healer_node(state: AgentPipelineState, config: dict) -> dict:
    """LangGraph node: diagnose the failed step and produce a minimal patch."""
    cfg = (config or {}).get("configurable", {}) or {}
    ollama_url = (cfg.get("ollama_url") or OLLAMA_URL).rstrip("/")
    repo: ContractSkillRepository = cfg.get("contract_repo") or ContractSkillRepository()

    contract = state.get("active_contract")
    failed_idx = state.get("failed_step_index")
    repair_attempts = state.get("repair_attempts", 0) or 0

    if contract is None or failed_idx is None:
        console.log("[HEALER] No contract or failed_step_index in state.")
        return {
            "mode": "human_review",
            "execution_log": ["[HEALER] Missing contract/failed_step_index → human_review"],
            "repair_attempts": repair_attempts + 1,
        }

    if failed_idx < 0 or failed_idx >= len(contract.action_steps):
        console.log(f"[HEALER] failed_step_index {failed_idx} out of bounds.")
        return {
            "mode": "human_review",
            "execution_log": [f"[HEALER] Index {failed_idx} OOB → human_review"],
            "repair_attempts": repair_attempts + 1,
        }

    failed_step = contract.action_steps[failed_idx]
    sfg_nodes = state.get("sfg_nodes", []) or []
    snapshot_blurb = sfg_nodes[-1].accessibility_snapshot[:2000] if sfg_nodes else ""
    recovery_dump = [rr.model_dump() for rr in contract.recovery_rules]

    prompt = (
        "A Playwright step failed. Diagnose and output ONLY JSON repair instruction.\n"
        "Schema: {\"operator\": \"SelReplace\"|\"PreInsert\"|\"ArgCorrect\"|\"PostInsert\",\n"
        "         \"patch_payload\": str,\n"
        "         \"reason\": str}\n"
        f"Failed step: {json.dumps(failed_step.model_dump(), ensure_ascii=False)}\n"
        f"Current snapshot: {snapshot_blurb}\n"
        f"Failure condition: {json.dumps(recovery_dump, ensure_ascii=False)}\n"
    )

    # Route through OllamaAdapter (holds _inference_semaphore + VRAM guard) instead
    # of a direct httpx call to :11434.
    adapter = OllamaAdapter(model=HEALER_MODEL, base_url=ollama_url)
    try:
        raw_text: str = await adapter.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        console.log(f"[HEALER] Ollama call failed: {exc}")
        return {
            "mode": "human_review",
            "execution_log": [f"[HEALER] Ollama failure → human_review: {exc}"],
            "repair_attempts": repair_attempts + 1,
        }
    finally:
        await adapter.close()

    text = _strip_fences(raw_text)
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        console.log(f"[HEALER] JSON parse failed: {exc}")
        return {
            "mode": "human_review",
            "execution_log": ["[HEALER] JSON parse failed → human_review"],
            "repair_attempts": repair_attempts + 1,
        }

    if not isinstance(parsed, dict):
        return {
            "mode": "human_review",
            "execution_log": ["[HEALER] Non-object repair → human_review"],
            "repair_attempts": repair_attempts + 1,
        }

    operator = parsed.get("operator", "")
    patch_payload = parsed.get("patch_payload", "")
    reason = (parsed.get("reason") or "").lower()

    if operator not in ALLOWED_OPERATORS:
        return {
            "mode": "human_review",
            "execution_log": [
                f"[HEALER] WARNING illegal operator '{operator}' → human_review"
            ],
            "repair_attempts": repair_attempts + 1,
        }

    if any(word in reason for word in LOGIC_KEYWORDS):
        console.log("[HEALER] WARNING: business-logic change detected; halting.")
        return {
            "mode": "human_review",
            "execution_log": [
                f"[HEALER] WARNING business-logic keyword in reason '{reason[:80]}' → human_review"
            ],
            "repair_attempts": repair_attempts + 1,
        }

    try:
        patched = repo.apply_patch(
            artifact=contract,
            failed_step_idx=failed_idx,
            operator=operator,
            patch_payload=patch_payload,
        )
    except (IndexError, ValueError) as exc:
        console.log(f"[HEALER] apply_patch failed: {exc}")
        return {
            "mode": "human_review",
            "execution_log": [f"[HEALER] Patch apply failed → human_review: {exc}"],
            "repair_attempts": repair_attempts + 1,
        }

    next_attempts = repair_attempts + 1
    if next_attempts >= 3:
        new_mode = "human_review"
        log_line = "[HEALER] Max retries exceeded. Human intervention required."
    else:
        new_mode = "execute"
        log_line = f"[HEALER] Patch applied: {operator} on step {failed_idx}"

    return {
        "active_contract": patched,
        "repair_attempts": next_attempts,
        "execution_log": [log_line],
        "mode": new_mode,
        "failed_step_index": None,
    }


__all__ = ["healer_node"]
