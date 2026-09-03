"""
BFT generator node — Sprint 3 / Cluster S3-B (two-pass fix).

Sprint 3 regression root cause: voting on Python source string diverges by
temperature, so quorum was rare. Fix: BFT now votes on TestPlan JSON
(structured, low format-failure rate), then a single deterministic code
generator (T=0.0) converts the winning plan to Python.

Pipeline (LOW tier):
  PASS 1 — 3 plan generators @ T=(0.0, 0.3, 0.7) → PlanVoter on canonical JSON
  PASS 2 — single code generator @ T=0.0 on the winning plan
  Security — ASTNormalizer detects protected-name remapping
  Judge   — LLM-as-Judge validates the final code

SERIAL execution only (6 GB VRAM constraint).
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from loguru import logger

from src.config_loader import get_bft_enabled
from src.core.bft_consensus import (
    BFT_CONFIGS,
    ASTNormalizer,
    BFTGeneratorConfig,
    PlanVoter,
)
from src.llm.adapter import _inference_semaphore
from src.llm.instructor_client import InstructorClient
from src.llm.judge_client import JudgeClient
from src.llm.schemas import PlaywrightScript, TestPlan
from src.routing.adaptive_router import AdaptiveRouter, ConfidenceTier


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Return the shared module-level inference semaphore."""
    return _inference_semaphore


# ── Prompt builders ──────────────────────────────────────────────────────────


def _build_plan_prompt(state: dict) -> str:
    """Pass 1 prompt: requirement → TestPlan (structured JSON)."""
    requirement = state.get("requirement", "")
    url = state.get("url", "")
    domain = state.get("domain", "general")
    role = state.get("role", "admin")
    page_state = state.get("page_state") or ""

    system_prompt = (
        "You are a senior QA test architect. Convert the requirement into a "
        "structured TestPlan covering happy path, negative cases, edge cases, "
        "and RBAC boundaries. Be deterministic, concise, and consistent."
    )

    body = (
        f"{system_prompt}\n\n"
        + f"Requirement:\n{requirement}\n\n"
        + f"Target URL: {url}\n"
        + f"Role: {role}\n"
        + f"Detected domain: {domain}\n"
        + (f"\nPage state:\n{page_state}\n" if page_state else "")
        + "\nRespond with a JSON object matching TestPlan with fields:\n"
        + "  title, requirement_summary, estimated_complexity (low|medium|high),\n"
        + "  domain, domain_specific_notes (list[str]),\n"
        + "  steps (list of TestStep: step_number, description, action, "
        + "expected_result, role, preconditions),\n"
        + "  rbac_scenarios (list[str]), edge_cases (list[str]).\n\n"
        + "Rules:\n"
        + "- Keep steps minimal and concrete (2-5 steps).\n"
        + "- Use semantic Playwright actions only (visit/click/fill/expect).\n"
        + "- Be deterministic: identical requirements must produce identical plans."
    )
    return body


def _build_code_prompt(plan: TestPlan, state: dict) -> str:
    """Pass 2 prompt: winning TestPlan + state → PlaywrightScript (T=0.0)."""
    url = state.get("url", "")
    page_state = state.get("page_state") or ""
    plan_str = plan.model_dump_json(indent=2)

    system_prompt = (
        "You are an expert Playwright Python QA engineer. "
        "Write sync pytest-playwright tests using the Page fixture. "
        "Use ONLY page.get_by_role(), page.get_by_label(), page.get_by_text(), "
        "page.get_by_test_id(). NEVER CSS selectors or XPath. "
        "NEVER use async/await. Use expect() from playwright.sync_api."
    )

    body = (
        f"{system_prompt}\n\n"
        + (f"Page state:\n{page_state}\n\n" if page_state else "")
        + f"Test plan (consensus):\n{plan_str}\n\n"
        + f"Target URL: {url}\n\n"
        + "Respond with a JSON object matching PlaywrightScript:\n"
        + "  reasoning: 2-3 sentence explanation of locator strategy\n"
        + "  code: complete runnable pytest-playwright test function\n"
        + "  locators_used: list of locator helpers used\n"
        + "  test_function_name: name of the test function\n\n"
        + "RULES for code:\n"
        + "- Function must start with def test_ (sync)\n"
        + "- Import: import pytest; from playwright.sync_api import Page, expect\n"
        + "- Use ONLY get_by_role/label/text/test_id\n"
        + "- Use expect() for assertions\n"
        + "- NO async/await, NO browser.launch(), NO asyncio.sleep()"
    )
    return body


# Back-compat: legacy single-pass prompt — still used by MED-tier path.
def _build_generation_prompt(state: dict) -> str:
    """Legacy single-pass prompt used by MED tier (state has test_plan)."""
    plan_dict = state.get("test_plan") or {}
    if isinstance(plan_dict, dict) and plan_dict:
        try:
            plan = TestPlan.model_validate(plan_dict)
            return _build_code_prompt(plan, state)
        except Exception:
            pass
    # Fallback when no usable plan exists in state — build a code prompt from
    # the bare requirement so the single-pass MED path keeps working.
    stub = TestPlan(
        title=state.get("requirement", "")[:60] or "ad-hoc test",
        requirement_summary=state.get("requirement", ""),
        estimated_complexity="low",
        domain=state.get("domain", "general"),
        steps=[
            {
                "step_number": 1,
                "description": state.get("requirement", ""),
                "action": "visit",
                "expected_result": "Page loads",
                "role": state.get("role", "admin"),
                "preconditions": [],
            }
        ],
    )
    return _build_code_prompt(stub, state)


# ── ContractSkill → Playwright script ────────────────────────────────────────


def _skill_to_script(skill: Any) -> str:
    """Convert ContractSkill steps to a minimal Playwright test string."""
    fn_name = re.sub(r'[^a-z0-9_]', '_', skill.goal.lower())[:40].rstrip('_') or 'unnamed'
    lines = [
        "import pytest",
        "from playwright.sync_api import Page, expect",
        "",
        f"def test_{fn_name}(page: Page):",
        f'    page.goto("{skill.target_url}")',
    ]
    for step in skill.steps:
        if step.action_type == "click":
            lines.append(f'    page.{step.locator}.click()')
        elif step.action_type == "fill":
            safe_value = (step.input_value or "").replace('\\', '\\\\').replace('"', '\\"')
            lines.append(f'    page.{step.locator}.fill("{safe_value}")')
        elif step.action_type == "navigate":
            lines.append(f'    page.goto("{step.locator}")')
        else:
            lines.append(f'    # {step.action_type}: {step.locator}')
    for pc in skill.postconditions[:2]:
        lines.append(f'    # assert: {pc}')
    return "\n".join(lines)


# ── Single-generator helpers ─────────────────────────────────────────────────


async def run_single_generator(
    config: BFTGeneratorConfig,
    prompt: str,
    semaphore: asyncio.Semaphore,
    instructor_client: InstructorClient,
) -> str | None:
    """Pass 2 helper — generate Playwright code from prompt. Returns code or None."""
    start = time.monotonic()
    try:
        async with semaphore:
            script = await instructor_client.create_structured(
                prompt,
                PlaywrightScript,
                temperature=config.temperature,
            )
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            f"BFT code-gen node_id={config.node_id} "
            f"temperature={config.temperature} top_p={config.top_p} "
            f"latency_ms={latency_ms:.1f} success=True "
            f"output_chars={len(script.code)}"
        )
        return script.code
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.warning(
            f"BFT code-gen node_id={config.node_id} "
            f"temperature={config.temperature} "
            f"latency_ms={latency_ms:.1f} success=False error={exc!r}"
        )
        return None


async def run_single_plan_generator(
    config: BFTGeneratorConfig,
    prompt: str,
    semaphore: asyncio.Semaphore,
    instructor_client: InstructorClient,
) -> TestPlan | None:
    """Pass 1 helper — generate one TestPlan candidate. Returns plan or None."""
    start = time.monotonic()
    try:
        async with semaphore:
            plan = await instructor_client.create_structured(
                prompt,
                TestPlan,
                temperature=config.temperature,
            )
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            f"BFT plan-gen node_id={config.node_id} "
            f"temperature={config.temperature} top_p={config.top_p} "
            f"latency_ms={latency_ms:.1f} success=True "
            f"steps={len(plan.steps)}"
        )
        return plan
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.warning(
            f"BFT plan-gen node_id={config.node_id} "
            f"temperature={config.temperature} "
            f"latency_ms={latency_ms:.1f} success=False error={exc!r}"
        )
        return None


def _format_plan_node_results(plans: list[TestPlan | None]) -> list[str]:
    return [
        f"node_{i}:{'PASS' if p is not None else 'FAIL'}"
        for i, p in enumerate(plans)
    ]


# ── Single-generator + judge fallback (shared by MED tier and disabled BFT) ──


async def _single_generator_with_judge(
    state: dict,
    client: InstructorClient,
    judge_client: JudgeClient | None,
    bft_status_on_success: str = "CONSENSUS_REACHED",
    tier_value: str = ConfidenceTier.MED.value,
    reasoning: str = "MED tier: single greedy generator + judge",
) -> dict[str, Any]:
    """One greedy (T=0.0) code gen + judge. Shared by MED tier and BFT-disabled fallback."""
    requirement = state.get("requirement", "")
    domain = state.get("domain", "general")
    semaphore = _get_llm_semaphore()

    prompt = _build_generation_prompt(state)
    result = await run_single_generator(BFT_CONFIGS[0], prompt, semaphore, client)
    node_results = [f"node_0:{'PASS' if result else 'FAIL'}"]
    if result is None:
        return {
            "bft_status": "CONSENSUS_FAILED",
            "bft_node_results": node_results,
            "bft_confidence_tier": tier_value,
            "script": None,
        }
    judge = judge_client or JudgeClient(client)
    verdict = await judge.evaluate(
        generated_code=result,
        requirement=requirement,
        domain=domain,
    )
    if not verdict.approved:
        return {
            "bft_status": "JUDGE_REJECTED",
            "bft_node_results": node_results,
            "bft_confidence_tier": tier_value,
            "script": None,
            "judge_rejection_reason": verdict.rejection_reason or "rejected",
        }
    return {
        "bft_status": bft_status_on_success,
        "bft_node_results": node_results,
        "bft_confidence_tier": tier_value,
        "script": {
            "reasoning": reasoning,
            "code": result,
            "locators_used": [],
            "test_function_name": "test_generated",
        },
    }


# ── Main node ────────────────────────────────────────────────────────────────


async def bft_generator_node(
    state: dict,
    instructor_client: InstructorClient | None = None,
    judge_client: JudgeClient | None = None,
    router: AdaptiveRouter | None = None,
) -> dict[str, Any]:
    """LangGraph BFT node — two-pass: vote on plans, then single code gen.

    Confidence-tier routing (when ``router`` provided):
      HIGH → cached metadata, no LLM call
      MED  → single greedy code gen + judge (uses state's existing test_plan)
      LOW  → full two-pass BFT (3 plans → vote → 1 code → security → judge)

    When ``llm.bft.enabled`` is False in config, this node short-circuits to a
    single greedy generator + judge regardless of tier — hardware constraint
    fallback (Sprint 3 close).

    A contract cache check (S4-E) precedes BFT routing: if a matching
    ContractSkill with success_count >= 2 is found, generation is skipped.
    """
    logger.info("▶ bft_generator_node (two-pass)")

    client = instructor_client or InstructorClient()

    # Contract cache check (S4-E)
    contract_store = state.get("contract_skill_store")
    if contract_store is not None:
        skill = await contract_store.find_matching_skill(
            goal=state.get("requirement", ""),
            url=state.get("url", ""),
            domain=state.get("domain", "general"),
        )
        if skill is not None and skill.success_count >= 2:
            # Use cached skill directly — skip BFT generation
            logger.info(f"bft_generator_node: CONTRACT_CACHE_HIT skill_id={skill.skill_id}")
            return {
                **state,
                "bft_status": "CONTRACT_CACHE_HIT",
                "bft_confidence_tier": "HIGH",
                "bft_node_results": [],
                "script": {"code": _skill_to_script(skill), "language": "python"},
                "contract_skill_id": skill.skill_id,
            }

    # Sprint 3 close: BFT deactivated as a hardware constraint. Single gen + judge.
    if not get_bft_enabled():
        logger.info("BFT: disabled in config — falling back to single generator + judge")
        return await _single_generator_with_judge(
            state,
            client,
            judge_client,
            bft_status_on_success="CONSENSUS_REACHED",
            tier_value=ConfidenceTier.MED.value,
            reasoning="BFT disabled (hardware constraint) — single greedy gen + judge",
        )

    url = state.get("url", "")
    requirement = state.get("requirement", "")
    domain = state.get("domain", "general")

    if router is not None:
        tier = router.classify(url, requirement, domain)
    else:
        tier = ConfidenceTier.LOW

    # ── HIGH tier: cached path ─────────────────────────────────────────
    if tier == ConfidenceTier.HIGH and router is not None:
        cached = router.lookup_cached(url, requirement)
        if cached and cached.get("last_generated_code_hash"):
            logger.info("BFT: HIGH tier — using cached code hash (no LLM call)")
            return {
                "bft_status": "CONSENSUS_REACHED",
                "bft_node_results": ["cache:HIT"],
                "bft_confidence_tier": ConfidenceTier.HIGH.value,
                "script": None,
                "cache_hit": True,
                "cached_code_hash": cached["last_generated_code_hash"],
            }
        tier = ConfidenceTier.LOW

    semaphore = _get_llm_semaphore()

    # ── MED tier: single greedy code gen + judge ──────────────────────
    if tier == ConfidenceTier.MED:
        logger.info("BFT: MED tier — single greedy generator + judge")
        return await _single_generator_with_judge(
            state,
            client,
            judge_client,
            bft_status_on_success="CONSENSUS_REACHED",
            tier_value=ConfidenceTier.MED.value,
            reasoning="MED tier: single greedy generator + judge",
        )

    # ── LOW tier: PASS 1 — BFT vote on TestPlan ───────────────────────
    prompt_for_plan = _build_plan_prompt(state)
    plans: list[TestPlan | None] = []
    for config in BFT_CONFIGS:
        plan = await run_single_plan_generator(config, prompt_for_plan, semaphore, client)
        plans.append(plan)  # SERIAL — NEVER use asyncio.gather here

    winning_plan = PlanVoter().vote(plans)
    plan_node_results = _format_plan_node_results(plans)

    if winning_plan is None:
        logger.error("BFT: CONSENSUS_FAILED — no quorum among 3 plan generators")
        return {
            "bft_status": "CONSENSUS_FAILED",
            "bft_node_results": plan_node_results,
            "bft_confidence_tier": ConfidenceTier.LOW.value,
            "script": None,
        }

    # ── PASS 2 — single deterministic code generator (T=0.0) ──────────
    prompt_for_code = _build_code_prompt(winning_plan, state)
    code_result = await run_single_generator(
        BFT_CONFIGS[0], prompt_for_code, semaphore, client
    )

    if code_result is None:
        logger.error("BFT: CODE_GEN_FAILED — pass 2 code generator returned None")
        return {
            "bft_status": "CODE_GEN_FAILED",
            "bft_node_results": plan_node_results + ["code:FAIL"],
            "bft_confidence_tier": ConfidenceTier.LOW.value,
            "script": None,
        }

    # ── Security check: ASTNormalizer detects protected-name remapping ──
    if ASTNormalizer().normalize(code_result) is None:
        logger.error("BFT: SECURITY_HALT — protected name remap or invalid AST")
        return {
            "bft_status": "SECURITY_HALT",
            "bft_node_results": plan_node_results + ["code:HALT"],
            "bft_confidence_tier": ConfidenceTier.LOW.value,
            "script": None,
        }

    # ── Judge ─────────────────────────────────────────────────────────
    judge = judge_client or JudgeClient(client)
    verdict = await judge.evaluate(
        generated_code=code_result,
        requirement=requirement,
        domain=domain,
    )
    if not verdict.approved:
        logger.error(
            f"BFT: JUDGE_REJECTED — {verdict.rejection_reason or 'no reason'}"
        )
        return {
            "bft_status": "JUDGE_REJECTED",
            "bft_node_results": plan_node_results + ["code:PASS"],
            "bft_confidence_tier": ConfidenceTier.LOW.value,
            "script": None,
            "judge_rejection_reason": verdict.rejection_reason or "rejected",
        }

    logger.info("BFT: CONSENSUS_REACHED (two-pass)")
    return {
        "bft_status": "CONSENSUS_REACHED",
        "bft_node_results": plan_node_results + ["code:PASS"],
        "bft_confidence_tier": ConfidenceTier.LOW.value,
        "test_plan": winning_plan.model_dump(),
        "script": {
            "reasoning": "BFT two-pass: TestPlan consensus → deterministic code",
            "code": code_result,
            "locators_used": [],
            "test_function_name": "test_generated",
        },
    }


__all__ = [
    "run_single_generator",
    "run_single_plan_generator",
    "bft_generator_node",
    "_single_generator_with_judge",
    "_build_generation_prompt",
    "_build_plan_prompt",
    "_build_code_prompt",
    "_skill_to_script",
]
