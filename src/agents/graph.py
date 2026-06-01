"""
LangGraph state machine for the QA Agent.

Topology:

  Generative mode:   planner ──► generator ──► executor ──┐
                                                           │
  Execution mode:                              executor ──┤
                                                           │
                       ┌──────────────── healer ◄──────────┘ (if fail & retries left)
                       │
                       └──────────────► reporter ◄───────── executor (on success / exhausted)

All nodes are async and return partial state updates (dicts).
State is persisted to ~/.qa-agent/sessions/<session_id>.json after each node.
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TYPE_CHECKING, TypedDict

from loguru import logger
from langgraph.graph import END, StateGraph

from src.llm.adapter import OllamaAdapter
from src.llm.structured import PlaywrightScript, TestPlan
from src.browser.manager import BrowserManager
from src.rag.retriever import HybridRetriever
from src.config_loader import (
    get_use_grounder,
    get_context_budget_tokens,
    get_config,
    get_bft_enabled,
)
from src.agents.bft_generator import bft_generator_node as _bft_generator_node
from src.routing.adaptive_router import AdaptiveRouter
from src.llm.judge_client import JudgeClient
from src.llm.instructor_client import InstructorClient
from .planner import PlannerAgent
from .generator import GeneratorAgent
from .healer import CodeHealerAgent
from src.observability.tracer import OTelTracer
from src.observability.structured_logger import StructuredLogger
from src.observability.audit_chain import CryptoAuditTrail
from src.observability.metrics import AgentMetrics

if TYPE_CHECKING:
    from src.healing.state_validator import StateValidator
    from src.memory.episodic_store import EpisodicStore
    from src.contractskill.compiler import ContractSkillStore

_SESSIONS_DIR = Path(os.path.expanduser("~/.qa-agent/sessions"))
_DEFAULT_MAX_RETRIES = 3
_EXECUTOR_TIMEOUT_SEC = 300
_tracer = OTelTracer()
_structured_logger = StructuredLogger()
_audit_trail = CryptoAuditTrail(Path(os.path.expanduser("~/.qa-agent/audit_chain.jsonl")))
_metrics = AgentMetrics.instance()


# ──────────────────────────────────────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────────────────────────────────────


class QAAgentState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    requirement: str
    url: str
    role: str
    domain: str
    mode: Literal["generate", "execute"]

    # ── Intermediate ──────────────────────────────────────────────────────────
    test_plan: dict[str, Any] | None          # serialised TestPlan
    script: dict[str, Any] | None            # serialised PlaywrightScript
    script_path: str                         # path of temp .py file
    page_state: str                          # CompactPAM markdown from Grounder (may be "")
    page: Any | None                         # live Playwright Page for Grounder (optional)

    # ── Execution ─────────────────────────────────────────────────────────────
    execution_result: dict[str, Any] | None  # {"success": bool, "output": str}
    error: str | None
    retry_count: int
    max_retries: int
    state_classification: dict[str, Any] | None  # serialised StateClassification

    # ── Session ───────────────────────────────────────────────────────────────
    session_id: str
    messages: list[dict[str, Any]]           # conversation history (for reporting)

    # ── BFT (Sprint 3) ───────────────────────────────────────────────────────
    # "PENDING" | "CONSENSUS_REACHED" | "CONSENSUS_FAILED" | "JUDGE_REJECTED"
    # | "CODE_GEN_FAILED" | "SECURITY_HALT"
    bft_status: str
    bft_node_results: Annotated[list[str], operator.add]
    bft_confidence_tier: str  # "HIGH" | "MED" | "LOW"

    # ── ContractSkill (Sprint 4 / S4-D) ─────────────────────────────────────
    contract_skill_id: str | None  # skill_id if a ContractSkill was used this run


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder (dependency-injected)
# ──────────────────────────────────────────────────────────────────────────────


def build_graph(
    adapter: OllamaAdapter,
    browser_manager: BrowserManager | None = None,
    retriever: HybridRetriever | None = None,
    locator_file: str = "locators/locators.json",
    max_retries: int = _DEFAULT_MAX_RETRIES,
    state_validator: "StateValidator | None" = None,
    episodic_store: "EpisodicStore | None" = None,
    adaptive_router: "AdaptiveRouter | None" = None,
    contract_skill_store: "ContractSkillStore | None" = None,
) -> Any:  # returns CompiledGraph
    """
    Construct and compile the QA agent state machine.

    All agents are created as closures over the injected dependencies.
    ``state_validator`` and ``episodic_store`` are Sprint 2 additions that
    remain optional for backwards compatibility.
    ``adaptive_router`` is a Sprint 3 addition for BFT confidence-tier routing
    and post-run result recording. Pass ``None`` to disable adaptive routing
    (BFT will then always run the LOW-tier full 3-generator path).
    """
    planner_agent = PlannerAgent(
        adapter=adapter,
        retriever=retriever,
        episodic_store=episodic_store,
        contract_skill_store=contract_skill_store,
    )
    generator_agent = GeneratorAgent(adapter=adapter, retriever=retriever)

    _bm = browser_manager  # may be None for pure generative workflows

    # ── Sprint 3: BFT + adaptive routing closures ─────────────────────────
    _bft_instructor = InstructorClient(
        base_url=adapter.base_url, model=adapter.model, max_retries=1
    )
    _judge_instructor = InstructorClient(
        base_url=adapter.base_url, model=adapter.model, max_retries=1
    )
    _judge_client = JudgeClient(_judge_instructor)
    _router = adaptive_router  # may be None — bft_generator_node handles None

    if _bm:
        healer_agent: CodeHealerAgent | None = CodeHealerAgent(
            adapter=adapter,
            browser_manager=_bm,
            locator_file=locator_file,
        )
    else:
        healer_agent = None

    # ── Nodes ────────────────────────────────────────────────────────────────

    async def planner_node(state: QAAgentState) -> dict[str, Any]:
        logger.info("▶ planner_node")

        async with _tracer.span("node.planner", url=state.get("url", "")):
            _metrics.increment("node.planner.calls")
            # Ground the page to produce a CompactPAM when the feature is enabled
            # and a live Playwright Page is available in state.
            page_state: str = state.get("page_state") or ""
            if get_use_grounder() and state.get("page"):
                try:
                    from src.perception.grounder import Grounder
                    grounder = Grounder()
                    pam = await grounder.ground(
                        state["page"],
                        context_budget_tokens=get_context_budget_tokens(),
                    )
                    page_state = pam.content
                    if pam.source == "failure":
                        page_state = page_state + (
                            "\n\n**DEGRADED MODE**: Perception layer failed. "
                            "Use only the URL and page title above to make a best-effort plan. "
                            "Prefer generic actions (navigate, wait) over selector-specific ones. "
                            "If you cannot proceed safely, return an empty plan."
                        )
                        logger.warning(
                            f"planner_node: Grounder returned failure PAM for "
                            f"{state.get('url', '')} — operating in DEGRADED MODE"
                        )
                    logger.info(
                        f"planner_node: Grounder produced PAM "
                        f"({len(page_state)} chars, source={pam.source})"
                    )
                except Exception as exc:
                    logger.warning(
                        f"planner_node: Grounder failed, proceeding without "
                        f"page state: {exc}"
                    )

            plan = await planner_agent.plan(
                requirement=state.get("requirement", ""),
                url=state.get("url", ""),
                role=state.get("role", "admin"),
                domain=state.get("domain", "general"),
                page_state=page_state,
            )
            # Surface the detected domain to the state so generator/healer can use it.
            update = {
                "test_plan": plan.model_dump(),
                "domain": plan.domain,
                "messages": (state.get("messages") or [])
                + [{
                    "role": "system",
                    "content": (
                        f"[PlannerAgent] Plan created: {plan.title} "
                        f"(domain={plan.domain})"
                    ),
                }],
            }
            _structured_logger.info("PlannerNode", "plan_complete",
                                    payload={"domain": plan.domain})
            _audit_trail.append("planner_complete", {"domain": plan.domain,
                                                      "url": state.get("url", "")})
        _save_session(state, update)
        return update

    async def generator_node(state: QAAgentState) -> dict[str, Any]:
        logger.info("▶ generator_node")
        plan_dict = state.get("test_plan")
        if not plan_dict:
            raise ValueError("generator_node: test_plan is missing from state")

        async with _tracer.span("node.generator", url=state.get("url", "")):
            _metrics.increment("node.generator.calls")
            # Ground the page for the generator pass (re-use planner's page_state
            # if already computed; otherwise ground again for freshness).
            page_state: str = state.get("page_state") or ""
            if get_use_grounder() and state.get("page") and not page_state:
                try:
                    from src.perception.grounder import Grounder
                    grounder = Grounder()
                    pam = await grounder.ground(
                        state["page"],
                        context_budget_tokens=get_context_budget_tokens(),
                    )
                    page_state = pam.content
                    if pam.source == "failure":
                        page_state = page_state + (
                            "\n\n**DEGRADED MODE**: Perception layer failed. "
                            "Use only the URL and page title above to make a best-effort plan. "
                            "Prefer generic actions (navigate, wait) over selector-specific ones. "
                            "If you cannot proceed safely, return an empty plan."
                        )
                        logger.warning(
                            f"generator_node: Grounder returned failure PAM for "
                            f"{state.get('url', '')} — operating in DEGRADED MODE"
                        )
                    logger.info(
                        f"generator_node: Grounder produced PAM "
                        f"({len(page_state)} chars, source={pam.source})"
                    )
                except Exception as exc:
                    logger.warning(
                        f"generator_node: Grounder failed, proceeding without "
                        f"page state: {exc}"
                    )

            plan = TestPlan.model_validate(plan_dict)
            script = await generator_agent.generate(
                test_plan=plan,
                url=state.get("url", ""),
                role=state.get("role", "admin"),
                domain=state.get("domain", "general"),
                page_state=page_state,
            )
            update = {
                "script": script.model_dump(),
                "error": None,
                "messages": (state.get("messages") or [])
                + [{"role": "system", "content": f"[GeneratorAgent] Script generated: {script.test_function_name}"}],
            }
            _structured_logger.info("GeneratorNode", "script_generated",
                                    payload={"func": script.test_function_name})
            _audit_trail.append("generator_complete", {"func": script.test_function_name})
        _save_session(state, update)
        return update

    async def bft_node_wrapper(state: QAAgentState) -> dict[str, Any]:
        """Sprint 3 BFT generator wrapper — uses the bft_generator_node pipeline.

        Returned dict includes ``bft_status``, ``bft_node_results``,
        ``bft_confidence_tier``, ``script`` (None on failure/cache-only-HIGH),
        plus optional ``cache_hit``, ``cached_code_hash``,
        ``judge_rejection_reason``.
        """
        logger.info("▶ bft_generator_node (BFT pipeline)")
        async with _tracer.span("node.bft"):
            _metrics.increment("node.bft.calls")
            update = await _bft_generator_node(
                state,
                instructor_client=_bft_instructor,
                judge_client=_judge_client,
                router=_router,
            )
            update["messages"] = (state.get("messages") or []) + [{
                "role": "system",
                "content": (
                    f"[BFTGenerator] status={update.get('bft_status')} "
                    f"tier={update.get('bft_confidence_tier', 'LOW')}"
                ),
            }]
            _structured_logger.info("BFTNode", "bft_complete",
                                    payload={"status": update.get("bft_status", "")})
            _audit_trail.append("bft_complete", {"status": update.get("bft_status", "")})
        _save_session(state, update)
        return update

    async def executor_node(state: QAAgentState) -> dict[str, Any]:
        logger.info("▶ executor_node")
        script_dict = state.get("script")
        if not script_dict:
            raise ValueError("executor_node: script is missing from state")

        async with _tracer.span("node.executor"):
            _metrics.increment("node.executor.calls")
            code: str = script_dict.get("code", "")
            result = await _run_script(code, timeout=_EXECUTOR_TIMEOUT_SEC)

            update: dict[str, Any] = {
                "execution_result": result,
                "retry_count": state.get("retry_count", 0),
            }

            # ── Sprint 2: StateValidator scaffolding ───────────────────────────
            # _run_script is a subprocess, so there is no in-process live Page
            # to validate post-action.  We still wire the integration here as
            # scaffolding for a future in-process executor.
            live_page = state.get("page")
            if state_validator is not None:
                if live_page is None:
                    logger.info(
                        "executor_node: state_validator configured but no live page "
                        "in state — skipping post-action classification (subprocess executor)"
                    )
                else:
                    try:
                        pre = await state_validator.capture_pre_state(live_page)
                        classification = await state_validator.classify_post_action(
                            live_page, pre, action_id="executor_post"
                        )
                        update["state_classification"] = classification.model_dump()
                        if (
                            classification.outcome_label == "Error_State"
                            and classification.failure_signature
                        ):
                            update["failure_signature"] = classification.failure_signature
                    except Exception as exc:
                        logger.warning(
                            f"executor_node: state validation failed: {exc}"
                        )

            if result["success"]:
                logger.info("executor_node: test PASSED ✓")
                update["error"] = None
                _structured_logger.info("ExecutorNode", "test_passed")
                _audit_trail.append("executor_pass", {"code_len": len(code)})
            elif result.get("returncode") == 5:
                logger.error(
                    "executor_node: exit code 5 = no test functions collected. "
                    "Test functions must start with 'test_' and use sync pytest-playwright format."
                )
                update["error"] = (
                    "No tests collected (pytest exit 5) — "
                    "generated code has no test_ functions or used wrong Playwright API. "
                    "Regenerate using sync pytest-playwright: "
                    "def test_<name>(page: Page) -> None, import from playwright.sync_api."
                )
                _structured_logger.error("ExecutorNode", "test_failed",
                                         payload={"returncode": 5})
                _audit_trail.append("executor_fail", {"returncode": 5})
            else:
                logger.warning(f"executor_node: test FAILED — {result['output'][:200]}")
                update["error"] = result["output"]
                _structured_logger.error("ExecutorNode", "test_failed",
                                         payload={"output": result["output"][:200]})
                _audit_trail.append("executor_fail", {"error": result["output"][:200]})
        _save_session(state, update)
        return update

    async def healer_node(state: QAAgentState) -> dict[str, Any]:
        logger.info("▶ healer_node")
        if not healer_agent:
            logger.warning("healer_node: no browser_manager provided — skipping heal")
            return {"retry_count": state.get("retry_count", 0) + 1}

        script_dict = state.get("script")
        if not script_dict:
            return {"retry_count": state.get("retry_count", 0) + 1}

        async with _tracer.span("node.healer"):
            _metrics.increment("node.healer.calls")
            script = PlaywrightScript.model_validate(script_dict)
            error_output = state.get("error") or ""

            healed_script = await healer_agent.heal_script(
                script=script,
                error_output=error_output,
                url=state.get("url", ""),
                role=state.get("role", "admin"),
            )

            update: dict[str, Any] = {
                "script": healed_script.model_dump(),
                "retry_count": state.get("retry_count", 0) + 1,
                "messages": (state.get("messages") or [])
                + [{"role": "system", "content": "[HealerAgent] Script healed and ready for retry"}],
            }
            _structured_logger.info("HealerNode", "heal_attempt",
                                    payload={"error": error_output[:100]})
            _audit_trail.append("healer_triggered", {"error": error_output[:100]})
        _save_session(state, update)
        return update

    async def reporter_node(state: QAAgentState) -> dict[str, Any]:
        logger.info("▶ reporter_node")
        result = state.get("execution_result") or {}
        success = result.get("success", False)
        retries = state.get("retry_count", 0)

        report: dict[str, Any] = {
            "session_id": state.get("session_id", ""),
            "requirement": state.get("requirement", ""),
            "url": state.get("url", ""),
            "role": state.get("role", ""),
            "domain": state.get("domain", ""),
            "success": success,
            "retry_count": retries,
            "output_snippet": (result.get("output") or "")[:500],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"reporter_node: session={report['session_id']!r} "
            f"success={success} retries={retries}"
        )

        # ── Sprint 3: record outcome into adaptive router cache ──────────
        if _router is not None:
            try:
                script_dict = state.get("script") or {}
                code = (
                    script_dict.get("code")
                    if isinstance(script_dict, dict)
                    else None
                )
                _router.record_result(
                    url=state.get("url", ""),
                    requirement=state.get("requirement", ""),
                    domain=state.get("domain", "general"),
                    passed=success,
                    generated_code=code,
                )
            except Exception as exc:
                logger.warning(
                    f"reporter_node: router.record_result failed: {exc}"
                )

        # ── Sprint 4 S4-D: update ContractSkill counts ───────────────────
        if contract_skill_store is not None:
            skill_id = state.get("contract_skill_id")
            if skill_id:
                try:
                    if success:
                        await contract_skill_store.update_counts(skill_id, success_delta=1)
                    else:
                        await contract_skill_store.update_counts(skill_id, failure_delta=1)
                    logger.debug(
                        f"reporter_node: updated ContractSkill counts | "
                        f"skill_id={skill_id[:12]} success={success}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"reporter_node: contract_skill_store.update_counts failed: {exc}"
                    )

        _save_session(state, {"execution_result": {**result, "report": report}})
        return {}

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_executor(state: QAAgentState) -> str:
        error = state.get("error")
        retry_count = state.get("retry_count", 0)
        _max = state.get("max_retries", max_retries)
        if error and retry_count < _max:
            return "healer"
        return "reporter"

    def _route_entry(state: QAAgentState) -> str:
        mode = state.get("mode", "generate")
        if mode == "execute":
            return "executor"
        return "planner"

    def _route_after_bft(state: QAAgentState) -> str:
        """Post-generator routing for the BFT pipeline.

        Terminal statuses → reporter:
          CONSENSUS_FAILED, JUDGE_REJECTED, CODE_GEN_FAILED, SECURITY_HALT
        script missing (e.g. HIGH-tier cache with no body) → reporter
        else → executor
        """
        status = state.get("bft_status")
        if status in (
            "CONSENSUS_FAILED",
            "JUDGE_REJECTED",
            "CODE_GEN_FAILED",
            "SECURITY_HALT",
        ):
            return "reporter"
        if not state.get("script"):
            return "reporter"
        return "executor"

    # ── Graph assembly ────────────────────────────────────────────────────────

    builder: StateGraph = StateGraph(QAAgentState)

    builder.add_node("planner", planner_node)
    # Feature-flag swap: BFT pipeline vs legacy single-pass generator
    if get_bft_enabled():
        logger.info("build_graph: BFT pipeline ENABLED (llm.bft.enabled=true)")
        builder.add_node("generator", bft_node_wrapper)
    else:
        logger.info("build_graph: BFT pipeline DISABLED — using legacy generator_node")
        builder.add_node("generator", generator_node)
    builder.add_node("executor", executor_node)
    builder.add_node("healer", healer_node)
    builder.add_node("reporter", reporter_node)

    # Entry: conditional on mode
    builder.set_conditional_entry_point(
        _route_entry,
        {"planner": "planner", "executor": "executor"},
    )

    # Generative path
    builder.add_edge("planner", "generator")
    # Sprint 3: generator → executor OR reporter (BFT failure / cache-only)
    if get_bft_enabled():
        builder.add_conditional_edges(
            "generator",
            _route_after_bft,
            {"executor": "executor", "reporter": "reporter"},
        )
    else:
        builder.add_edge("generator", "executor")

    # Executor → conditional
    builder.add_conditional_edges(
        "executor",
        _route_executor,
        {"healer": "healer", "reporter": "reporter"},
    )

    # Healing retry loop
    builder.add_edge("healer", "executor")

    # Terminal
    builder.add_edge("reporter", END)

    return builder.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Default initial state factory
# ──────────────────────────────────────────────────────────────────────────────


def initial_state(
    requirement: str = "",
    url: str = "",
    role: str = "admin",
    domain: str = "general",
    mode: Literal["generate", "execute"] = "generate",
    script_dict: dict[str, Any] | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> QAAgentState:
    """Build the initial state dict for a new agent run."""
    state = QAAgentState(
        requirement=requirement,
        url=url,
        role=role,
        domain=domain,
        mode=mode,
        test_plan=None,
        script=script_dict,
        script_path="",
        execution_result=None,
        error=None,
        retry_count=0,
        max_retries=max_retries,
        session_id=str(uuid.uuid4())[:8],
        messages=[],
    )
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Script execution
# ──────────────────────────────────────────────────────────────────────────────


async def _run_script(code: str, timeout: int = _EXECUTOR_TIMEOUT_SEC) -> dict[str, Any]:
    """
    Write *code* to a temp file and run it with pytest as a subprocess.
    Returns {"success": bool, "output": str, "returncode": int}.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_qa_agent_test.py",
        prefix="qa_",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(code)
        tmp_path = fh.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python", "-m", "pytest", tmp_path,
            "-v", "--tb=short", "-p", "no:warnings",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + 10
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "success": False,
                "output": f"Test runner timed out after {timeout + 10}s",
                "returncode": -1,
            }

        output = stdout_bytes.decode(errors="replace")
        success = proc.returncode == 0
        return {"success": success, "output": output, "returncode": proc.returncode}
    except FileNotFoundError:
        return {
            "success": False,
            "output": "pytest not found — install it with: pip install pytest pytest-asyncio",
            "returncode": -1,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Session persistence
# ──────────────────────────────────────────────────────────────────────────────


def _save_session(state: QAAgentState, update: dict[str, Any]) -> None:
    """Merge *update* into *state* and persist to disk as JSON."""
    try:
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        merged: dict[str, Any] = {**state, **update}
        session_id = merged.get("session_id", "unknown")
        path = _SESSIONS_DIR / f"{session_id}.json"
        tmp = path.with_suffix(".tmp")
        # PlaywrightScript / TestPlan objects are already serialised to dicts
        tmp.write_text(json.dumps(merged, indent=2, default=str))
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning(f"Session persistence failed: {exc}")


def load_session(session_id: str) -> QAAgentState | None:
    """Load a previously persisted session by ID. Returns None if not found."""
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return QAAgentState(**{k: v for k, v in data.items() if k in QAAgentState.__annotations__})
    except Exception as exc:
        logger.error(f"load_session: {exc}")
        return None
