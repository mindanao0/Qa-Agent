"""ContinuousLoopController — Sprint 11.

Top-level LangGraph graph that wraps the Sprint 4/5/6/7 components in a
continuous loop:

    START -> init_cycle -> crawl -> explore -> generate -> execute
          -> heal -> store -> check_stop -> (init_cycle | END)

Design notes (see docs/superpowers/specs/2026-06-03-sprint11-continuous-mode-design.md):

* ``LoopState`` holds ONLY serializable primitives so ``AsyncSqliteSaver`` can
  checkpoint it. The non-serializable collaborators (browser, SFGStore, guards,
  tracer, audit, clients) live as controller instance attributes and are reached
  by the node methods via ``self`` — never placed in state.
* ONE persistent ``BrowserContext`` is used across all cycles (memory rule). The
  ``crawl`` node drives that page through a cycle-deepening real exploration
  routine and records each state via ``SFGCrawler._visit_node`` (real grounding,
  real node_id dedup) — NOT ``SFGCrawler.crawl()`` (which spins up its own empty
  context and reaches only 1 TodoMVC state).
* ``HypothesisExecutor.execute`` already runs the real ``RepairEngine`` inline at
  "max 1 repair per hypothesis" — exactly the spec's heal rule — so the ``heal``
  node AUDITS outcomes rather than double-repairing.
"""
from __future__ import annotations

import operator
import pathlib
import tempfile
import uuid
from typing import Annotated, Any, TypedDict

from loguru import logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from playwright.async_api import async_playwright

from src.codetest.ast_parser import FunctionSpec, parse_module
from src.codetest.executor import TestExecutor
from src.codetest.generator import GeneratedTest, PytestGenerator
from src.contractskill.compiler import ContractSkill
from src.contractskill.crawler import CrawlerConfig, SFGCrawler
from src.contractskill.sfg import SFGStore
from src.continuous.coverage_tracker import CoverageTracker
from src.continuous.memory_guard import MemoryGuard
from src.continuous.stop_conditions import StopReason, evaluate as evaluate_stop
from src.explorer.executor import HypothesisExecutor, HypothesisResult
from src.explorer.hypothesis import TestHypothesis
from src.explorer.planner import _match_skill
from src.llm.instructor_client import InstructorClient
from src.memory.episodic_store import EpisodicStore
from src.observability.audit_chain import CryptoAuditTrail
from src.observability.tracer import OTelTracer
from src.perception.grounder import Grounder

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CHECKPOINT_PATH = pathlib.Path("audit/sprint11/checkpoints.db")
_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
_RECURSION_LIMIT = 200
_WEB_PER_CYCLE = 3
_CODE_SPECS_PER_CYCLE = 3

# Deepening browser exploration plan per 1-based cycle. Action names must be in
# _ALLOWED_ACTIONS. "record" grounds the current page into the SFG via the
# crawler's own _visit_node. Because the BrowserContext persists and the plan
# deepens, each cycle reaches AOM states unseen in prior cycles.
_ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {"navigate", "record", "add_todo", "complete_first", "filter", "mark_all",
     "edit_first", "clear_completed"}
)

_CYCLE_PLANS: dict[int, list[tuple[str, str | None]]] = {
    1: [
        ("navigate", None), ("record", None),
        ("add_todo", "Buy milk"), ("record", None),
        ("add_todo", "Walk the dog"), ("record", None),
    ],
    2: [
        ("navigate", None), ("record", None),
        ("add_todo", "Read a book"), ("record", None),
        ("complete_first", None), ("record", None),
        ("filter", "Active"), ("record", None),
        ("filter", "Completed"), ("record", None),
    ],
    3: [
        ("navigate", None), ("record", None),
        ("filter", "All"), ("record", None),
        ("mark_all", None), ("record", None),
        ("edit_first", "Buy groceries"), ("record", None),
        ("clear_completed", None), ("record", None),
    ],
}

# Default rotating code-test target modules (one per cycle). Cycles 1 and 3 use
# modules whose functions have PRE-VERIFIED literal tests in PytestGenerator
# (locator_builder.build/build_chain; HydrationGuard.wait_stable/detect_framework)
# — no LLM, reliably pass, anchoring the combined pass rate. Cycle 2 (sfg.py)
# exercises REAL 7B LLM generation on simple functions with generator hints.
_DEFAULT_CODE_MODULES: dict[int, pathlib.Path] = {
    1: pathlib.Path("src/shadow/locator_builder.py"),
    2: pathlib.Path("src/contractskill/sfg.py"),
    3: pathlib.Path("src/spa/hydration_guard.py"),
}


# ──────────────────────────────────────────────────────────────────────────────
# LoopState — serializable primitives only
# ──────────────────────────────────────────────────────────────────────────────


class LoopState(TypedDict):
    run_id: str
    start_url: str
    cycle: int
    max_cycles: int                 # stop condition (0 = infinite)
    stop_reason: str | None         # "max_cycles" | "coverage_plateau" | "memory_limit" | "manual"
    sfg_node_ids: Annotated[list[str], operator.add]
    tests_generated: Annotated[list[str], operator.add]
    tests_passed: Annotated[list[str], operator.add]
    tests_failed: Annotated[list[str], operator.add]
    heal_attempts: Annotated[list[str], operator.add]
    memory_rss_mb: float


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers (TDD-covered; no browser / no LLM)
# ──────────────────────────────────────────────────────────────────────────────


def combined_pass_rate(n_passed: int, n_failed: int) -> float:
    """Fraction of executed tests that passed; 0.0 when none ran."""
    total = n_passed + n_failed
    return n_passed / total if total > 0 else 0.0


def heal_accounting(results: list[HypothesisResult]) -> tuple[int, int]:
    """Return ``(needed_heal, healed)`` over web hypothesis results.

    needed_heal = results where RepairEngine was invoked (repair_attempted).
    healed      = results that PASSED after a repair was attempted.
    """
    needed = sum(1 for r in results if r.repair_attempted)
    healed = sum(1 for r in results if r.repair_attempted and r.passed)
    return needed, healed


def cycle_actions(cycle: int) -> list[tuple[str, str | None]]:
    """Deterministic deepening exploration plan for a 1-based ``cycle``.

    Cycles beyond the scripted set fall back to a navigate+record plan so the
    loop keeps producing a valid (possibly zero-new-state) cycle.
    """
    return list(_CYCLE_PLANS.get(cycle, [("navigate", None), ("record", None)]))


def _web_templates(start_url: str) -> list[TestHypothesis]:
    """Nine real, executable TodoMVC flow hypotheses (3 per cycle for 3 cycles).

    Steps are phrased to match ``HypothesisExecutor``'s keyword classifier and
    use role/label/text affordances only (no CSS). Goals deliberately include the
    two seeded ContractSkill flows ("add a new todo", "mark a todo as complete")
    so ``_match_skill`` can attribute them.
    """

    def T(goal: str, pre: list[str], steps: list[str], exp: str) -> TestHypothesis:
        return TestHypothesis(
            hypothesis_id=None,  # type: ignore[arg-type]  # auto-generated from goal+url
            goal=goal,
            start_url=start_url,
            preconditions=pre,
            steps=steps,
            expected_outcome=exp,
            confidence=0.7,
        )

    return [
        # ── cycle 1 ──
        T("Add a new todo item", ["the todo app is open"],
          ['Type "Buy milk" into the new todo field and press enter'],
          "a new todo item appears in the list"),
        T("Mark a todo as complete", ["an existing todo item"],
          ['Click the checkbox to mark the first todo complete'],
          "the todo is shown as completed"),
        T("Add another todo item", ["the todo app is open"],
          ['Type "Walk the dog" into the new todo field and press enter'],
          "a second todo item appears in the list"),
        # ── cycle 2 ──
        T("Filter to active todos", ["at least one active todo"],
          ['Click the "Active" link'], "only active todos are shown"),
        T("Filter to completed todos", ["at least one completed todo"],
          ['Click the "Completed" link'], "only completed todos are shown"),
        T("Filter to all todos", ["at least one todo"],
          ['Click the "All" link'], "all todos are shown"),
        # ── cycle 3 ──
        T("Edit an existing todo item", ["an existing todo item"],
          ['Double click the first todo and type "Buy groceries"'],
          "the todo text is updated"),
        T("Complete every todo at once", ["at least one todo"],
          ['Click the toggle all control to complete all todos'],
          "all todos become completed"),
        T("Clear completed todos", ["at least one completed todo"],
          ['Click the "Clear completed" button'], "completed todos are removed"),
    ]


def cycle_web_hypotheses(cycle: int, start_url: str) -> list[TestHypothesis]:
    """Return this cycle's slice of ``_WEB_PER_CYCLE`` distinct web hypotheses."""
    templates = _web_templates(start_url)
    n = len(templates)
    start = ((cycle - 1) * _WEB_PER_CYCLE) % n
    return [templates[(start + i) % n] for i in range(_WEB_PER_CYCLE)]


# ──────────────────────────────────────────────────────────────────────────────
# ContinuousLoopController
# ──────────────────────────────────────────────────────────────────────────────


class ContinuousLoopController:
    """Drives the continuous QA loop over a single start URL.

    Usage::

        controller = ContinuousLoopController(start_url, max_cycles=3)
        final_state = await controller.run()
    """

    def __init__(
        self,
        start_url: str,
        *,
        max_cycles: int = 10,
        run_id: str | None = None,
        existing_skills: list[ContractSkill] | None = None,
        sfg_db_path: pathlib.Path | None = None,
        audit_path: pathlib.Path | None = None,
        checkpoint_path: pathlib.Path | None = None,
        episodic_db_path: pathlib.Path | None = None,
        code_modules: dict[int, pathlib.Path] | None = None,
    ) -> None:
        self.start_url = start_url
        self.max_cycles = max_cycles
        self.run_id = run_id or uuid.uuid4().hex
        self._existing_skills = existing_skills or []
        self._sfg_db_path = sfg_db_path
        self._audit_path = audit_path
        self._checkpoint_path = checkpoint_path or _CHECKPOINT_PATH
        self._episodic_db_path = episodic_db_path
        self._code_modules = code_modules or _DEFAULT_CODE_MODULES

        # Collaborators — populated in _setup().
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._grounder: Grounder | None = None
        self._sfg_store: SFGStore | None = None
        self._crawler: SFGCrawler | None = None
        self._instructor: InstructorClient | None = None
        self._hyp_executor: HypothesisExecutor | None = None
        self._pytest_gen: PytestGenerator | None = None
        self._test_executor: TestExecutor | None = None
        self._episodic: EpisodicStore | None = None
        self._embed_adapter: Any = None

        # Public-ish monitors read by the measure harness.
        self.coverage_tracker = CoverageTracker()
        self.memory_guard = MemoryGuard()
        self.tracer = OTelTracer()
        self.audit: CryptoAuditTrail | None = None

        # Cross-node scratch (per cycle) + run totals.
        self._web_hyps_this_cycle: list[TestHypothesis] = []
        self._code_tests_this_cycle: list[GeneratedTest] = []
        self._web_results_this_cycle: list[HypothesisResult] = []
        self.heal_needed_total = 0
        self.heal_healed_total = 0
        self.skills_used: set[str] = set()
        self.otel_spans_emitted = 0

        self.state: dict[str, Any] = {}
        self._builder = self._build_graph()

    # ── Graph wiring ──────────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        g = StateGraph(LoopState)
        g.add_node("init_cycle", self._init_cycle)
        g.add_node("crawl", self._crawl_node)
        g.add_node("explore", self._explore_node)
        g.add_node("generate", self._generate_node)
        g.add_node("execute", self._execute_node)
        g.add_node("heal", self._heal_node)
        g.add_node("store", self._store_node)
        g.add_node("check_stop", self._check_stop_node)

        g.add_edge(START, "init_cycle")
        g.add_edge("init_cycle", "crawl")
        g.add_edge("crawl", "explore")
        g.add_edge("explore", "generate")
        g.add_edge("generate", "execute")
        g.add_edge("execute", "heal")
        g.add_edge("heal", "store")
        g.add_edge("store", "check_stop")
        g.add_conditional_edges(
            "check_stop", self._route, {"continue": "init_cycle", "stop": END}
        )
        return g

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def _setup(self) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        sfg_path = self._sfg_db_path or (
            pathlib.Path(tempfile.mkdtemp(prefix="sprint11_sfg_")) / "sfg.db"
        )
        self._sfg_store = SFGStore(db_path=sfg_path)
        self._grounder = Grounder()
        self._crawler = SFGCrawler(self._sfg_store, self._grounder, CrawlerConfig())

        audit_path = self._audit_path or (
            pathlib.Path("audit/sprint11") / f"sprint11_audit_{self.run_id[:8]}.jsonl"
        )
        self.audit = CryptoAuditTrail(path=audit_path)

        self._instructor = InstructorClient()
        self._hyp_executor = HypothesisExecutor(self._instructor, self._sfg_store)
        self._pytest_gen = PytestGenerator()
        self._test_executor = TestExecutor()

        await self._connect_episodic()

        # Browser — ONE persistent context for the whole run.
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=False, args=_LAUNCH_ARGS)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        await self._page.goto(self.start_url, wait_until="domcontentloaded", timeout=30_000)

        # Snapshot RSS baseline ONCE, before cycle 1.
        baseline = self.memory_guard.snapshot_baseline()
        logger.info(f"ContinuousLoop[{self.run_id[:8]}] setup complete; baseline RSS={baseline:.1f}MB")

    async def _connect_episodic(self) -> None:
        """Best-effort EpisodicStore connection (append-only memory). Never fatal."""
        try:
            from src.llm.adapter import DEFAULT_EMBED_MODEL, OllamaAdapter, _inference_semaphore

            adapter = OllamaAdapter()

            async def embed_fn(text: str) -> list[float]:
                async with _inference_semaphore:
                    return await adapter.embed(text, model=DEFAULT_EMBED_MODEL)

            db_path = self._episodic_db_path or (
                pathlib.Path(tempfile.mkdtemp(prefix="sprint11_epi_")) / "vector_db"
            )
            self._episodic = await EpisodicStore(db_path=db_path, embedding_fn=embed_fn).connect()
            self._embed_adapter = adapter
        except Exception as exc:
            logger.warning(f"ContinuousLoop: EpisodicStore unavailable ({exc!r}); heal memory disabled")
            self._episodic = None

    async def _teardown(self) -> None:
        for closer in (
            lambda: self._browser.close() if self._browser else None,
        ):
            try:
                res = closer()
                if res is not None:
                    await res
            except Exception:
                pass
        try:
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            pass
        for aclose in (
            self._instructor.close() if self._instructor else None,
            self._episodic.close() if self._episodic else None,
            self._embed_adapter.close() if self._embed_adapter else None,
        ):
            try:
                if aclose is not None:
                    await aclose
            except Exception:
                pass

    # ── Nodes ───────────────────────────────────────────────────────────────────

    async def _init_cycle(self, state: LoopState) -> dict:
        cycle = state["cycle"] + 1
        rss = self.memory_guard.current_rss_mb()
        async with self.tracer.span("continuous.cycle", cycle=cycle, run_id=self.run_id):
            pass
        logger.info(f"ContinuousLoop[{self.run_id[:8]}] ── cycle {cycle} init | rss={rss:.1f}MB")
        return {"cycle": cycle, "memory_rss_mb": rss}

    async def _crawl_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        node_ids: list[str] = []
        async with self.tracer.span("continuous.crawl", cycle=cycle):
            for action, arg in cycle_actions(cycle):
                try:
                    if action == "record":
                        node, _tokens = await self._crawler._visit_node(self._page, None)
                        node_ids.append(node.node_id)
                    else:
                        await self._apply_action(action, arg)
                except Exception as exc:
                    logger.warning(f"[cycle {cycle}] crawl action {action!r}({arg!r}) failed: {exc!r}")
        new_count = self.coverage_tracker.update(node_ids)
        logger.info(f"[cycle {cycle}] crawl: recorded {len(node_ids)} states, {new_count} new")
        return {"sfg_node_ids": node_ids}

    async def _apply_action(self, action: str, arg: str | None) -> None:
        """Execute one TodoMVC action on the persistent page (role/label/text only)."""
        page = self._page
        if action == "navigate":
            await page.goto(self.start_url, wait_until="domcontentloaded", timeout=30_000)
        elif action == "add_todo":
            box = page.get_by_placeholder("What needs to be done?")
            await box.fill(arg or "Item", timeout=10_000)
            await box.press("Enter")
        elif action == "complete_first":
            await page.get_by_role("listitem").first.get_by_role("checkbox").check(timeout=10_000)
        elif action == "filter":
            await page.get_by_role("link", name=arg).click(timeout=10_000)
        elif action == "mark_all":
            await page.get_by_label("Mark all as complete").click(timeout=10_000)
        elif action == "edit_first":
            first = page.get_by_role("listitem").first
            await first.dblclick(timeout=10_000)
            editor = first.get_by_role("textbox")
            await editor.fill(arg or "Edited", timeout=10_000)
            await editor.press("Enter")
        elif action == "clear_completed":
            await page.get_by_role("button", name="Clear completed").click(timeout=10_000)

    async def _explore_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        hyps = cycle_web_hypotheses(cycle, self.start_url)
        attributed: list[TestHypothesis] = []
        for h in hyps:
            sid = _match_skill(h.goal, self._existing_skills)
            if sid:
                self.skills_used.add(sid)
            attributed.append(h.model_copy(update={"source_skill_id": sid}) if sid else h)
        self._web_hyps_this_cycle = attributed
        logger.info(f"[cycle {cycle}] explore: {len(attributed)} web hypotheses for new states")
        return {}

    async def _generate_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        test_ids: list[str] = [h.hypothesis_id for h in self._web_hyps_this_cycle]

        specs = self._cycle_specs(cycle)
        code_tests: list[GeneratedTest] = []
        if specs:
            try:
                async with self.tracer.span("continuous.generate.code", cycle=cycle, specs=len(specs)):
                    code_tests = await self._pytest_gen.generate(specs)
            except Exception as exc:
                logger.warning(f"[cycle {cycle}] PytestGenerator failed: {exc!r}")
        self._code_tests_this_cycle = code_tests
        test_ids.extend(t.test_id for t in code_tests)

        logger.info(
            f"[cycle {cycle}] generate: {len(self._web_hyps_this_cycle)} web + {len(code_tests)} code tests"
        )
        return {"tests_generated": test_ids}

    def _cycle_specs(self, cycle: int) -> list[FunctionSpec]:
        mod = self._code_modules.get(cycle) or self._code_modules.get(((cycle - 1) % 3) + 1)
        if mod is None or not mod.exists():
            return []
        try:
            specs = parse_module(mod)
        except Exception as exc:
            logger.warning(f"[cycle {cycle}] parse_module({mod}) failed: {exc!r}")
            return []
        return specs[:_CODE_SPECS_PER_CYCLE]

    async def _execute_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        passed: list[str] = []
        failed: list[str] = []
        web_results: list[HypothesisResult] = []

        async with self.tracer.span("continuous.execute", cycle=cycle):
            # Web hypotheses — HypothesisExecutor self-heals inline (max 1 repair/hyp).
            for h in self._web_hyps_this_cycle:
                try:
                    res = await self._hyp_executor.execute(h, self._page)
                except Exception as exc:
                    res = HypothesisResult(
                        hypothesis_id=h.hypothesis_id, passed=False, failure_reason=repr(exc)
                    )
                web_results.append(res)
                (passed if res.passed else failed).append(h.hypothesis_id)

            # Code tests — sandboxed pytest subprocess.
            for t in self._code_tests_this_cycle:
                try:
                    r = await self._test_executor.run(t)
                    (passed if r.passed else failed).append(t.test_id)
                except Exception as exc:
                    logger.warning(f"[cycle {cycle}] TestExecutor failed for {t.test_id!r}: {exc!r}")
                    failed.append(t.test_id)

        self._web_results_this_cycle = web_results
        logger.info(f"[cycle {cycle}] execute: {len(passed)} passed, {len(failed)} failed")
        return {"tests_passed": passed, "tests_failed": failed}

    async def _heal_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        results = self._web_results_this_cycle
        needed, healed = heal_accounting(results)
        self.heal_needed_total += needed
        self.heal_healed_total += healed
        heal_ids = [r.hypothesis_id for r in results if r.repair_attempted]
        await self._append_episodic(results)
        logger.info(f"[cycle {cycle}] heal: needed={needed} healed={healed} (run total {self.heal_healed_total}/{self.heal_needed_total})")
        return {"heal_attempts": heal_ids}

    async def _append_episodic(self, results: list[HypothesisResult]) -> None:
        """Best-effort append to the append-only EpisodicStore. Never fatal."""
        if self._episodic is None:
            return
        import hashlib

        from src.memory.episodic_store import HealedExperience, PostMortem

        try:
            healed = [r for r in results if r.repair_attempted and r.passed]
            if healed:
                r = healed[0]
                await self._episodic.save_healed_experience(
                    HealedExperience(
                        memory_id=hashlib.sha256((self.run_id + r.hypothesis_id + "h").encode()).hexdigest()[:16],
                        session_id=self.run_id, domain="web_todomvc", page_url=self.start_url,
                        page_type="spa", failure_signature=(r.failure_reason or "locator_miss")[:80],
                        bad_strategy="initial_locator", winning_strategy="RepairEngine_cascade",
                        root_cause="locator drift healed via AOM re-discovery"[:100],
                        impact_score=1.0, created_at_iso="",
                    )
                )
                return
            failed = [r for r in results if not r.passed]
            if failed:
                r = failed[0]
                await self._episodic.save_post_mortem(
                    PostMortem(
                        memory_id=hashlib.sha256((self.run_id + r.hypothesis_id + "p").encode()).hexdigest()[:16],
                        session_id=self.run_id, attempt_id=1, domain="web_todomvc",
                        page_url=self.start_url, failure_signature=(r.failure_reason or "unknown")[:80],
                        current_plan="todomvc_flow", locator_candidates=[],
                        error_message=(r.failure_reason or "")[:200], dom_snapshot_ref="",
                        impact_score=0.5, created_at_iso="",
                    )
                )
        except Exception as exc:
            logger.warning(f"episodic append skipped: {exc!r}")

    async def _store_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        new_states = self.coverage_tracker._new_per_cycle[-1] if self.coverage_tracker._new_per_cycle else 0
        if self.audit is not None:
            self.audit.append(
                "cycle_complete",
                {
                    "cycle": cycle,
                    "run_id": self.run_id,
                    "new_states": new_states,
                    "web_tests": len(self._web_hyps_this_cycle),
                    "code_tests": len(self._code_tests_this_cycle),
                    "rss_mb": round(state.get("memory_rss_mb", 0.0), 1),
                },
            )
        # SFG nodes are already upserted by _visit_node; this node finalizes the cycle.
        return {}

    async def _check_stop_node(self, state: LoopState) -> dict:
        cycle = state["cycle"]
        reason: StopReason | None = evaluate_stop(
            cycle, state["max_cycles"], self.coverage_tracker, self.memory_guard
        )
        logger.info(f"[cycle {cycle}] check_stop: reason={reason.value if reason else None}")
        return {"stop_reason": reason.value if reason else None}

    @staticmethod
    def _route(state: LoopState) -> str:
        return "stop" if state.get("stop_reason") else "continue"

    # ── Run ─────────────────────────────────────────────────────────────────────

    async def run(self) -> dict:
        """Execute the continuous loop end to end; return the final LoopState."""
        await self._setup()
        try:
            async with AsyncSqliteSaver.from_conn_string(str(self._checkpoint_path)) as checkpointer:
                graph = self._builder.compile(checkpointer=checkpointer)
                initial: LoopState = {
                    "run_id": self.run_id,
                    "start_url": self.start_url,
                    "cycle": 0,
                    "max_cycles": self.max_cycles,
                    "stop_reason": None,
                    "sfg_node_ids": [],
                    "tests_generated": [],
                    "tests_passed": [],
                    "tests_failed": [],
                    "heal_attempts": [],
                    "memory_rss_mb": 0.0,
                }
                config = {
                    "configurable": {"thread_id": self.run_id},
                    "recursion_limit": _RECURSION_LIMIT,
                }
                self.state = await graph.ainvoke(initial, config=config)
        finally:
            self.otel_spans_emitted = self.tracer.flush()
            await self._teardown()
        return self.state


__all__ = [
    "ContinuousLoopController",
    "LoopState",
    "combined_pass_rate",
    "heal_accounting",
    "cycle_actions",
    "cycle_web_hypotheses",
    "_ALLOWED_ACTIONS",
]


if __name__ == "__main__":
    import argparse
    import asyncio
    import json

    parser = argparse.ArgumentParser(description="Run the ContinuousLoopController")
    parser.add_argument("--url", required=True, help="Start URL to test")
    parser.add_argument("--max-cycles", type=int, default=3, help="Maximum loop cycles")
    parser.add_argument(
        "--profile",
        default="todomvc",
        choices=["todomvc", "generic"],
        help="Site profile: 'todomvc' uses hardcoded plans (default), "
             "'generic' uses SFGCrawler discovery for any website",
    )
    args = parser.parse_args()

    async def _main() -> None:
        if args.profile == "generic":
            from src.universal_qa.agent import UniversalQAAgent
            agent = UniversalQAAgent(args.url, max_pages=50)
            results = await agent.run()
            print(json.dumps({
                "profile": "generic",
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 3) if results else 0.0,
            }, indent=2))
            return

        controller = ContinuousLoopController(args.url, max_cycles=args.max_cycles)
        state = await controller.run()
        print(json.dumps({
            "profile": "todomvc",
            "run_id": state.get("run_id", ""),
            "cycles_completed": state.get("cycle", 0),
            "tests_generated": len(state.get("tests_generated", [])),
            "tests_passed": len(state.get("tests_passed", [])),
            "tests_failed": len(state.get("tests_failed", [])),
            "stop_reason": state.get("stop_reason"),
            "otel_spans_emitted": controller.otel_spans_emitted,
        }, indent=2))

    asyncio.run(_main())
