"""
Tests for src/contractskill/repair.py — Sprint 4 / Cluster S4-D.

4 tests:
  test_sel_replace_repairs_locator_no_llm_call
  test_pre_insert_makes_exactly_one_llm_call
  test_planner_injects_skill_when_confidence_high
  test_reporter_updates_success_count_after_pass
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.contractskill.compiler import ContractSkill, ContractStep
from src.contractskill.repair import MissingStep, RepairEngine
from src.contractskill.sfg import SFGStore
from src.perception.aom_extractor import AOMNode, AOMSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_skill(steps: list[ContractStep] | None = None) -> ContractSkill:
    if steps is None:
        steps = [
            ContractStep(
                step_number=1,
                action_type="click",
                locator='role=button[name="Login"]',
                input_value=None,
                expected_state_hash="abc123",
            )
        ]
    return ContractSkill(
        skill_id=hashlib.sha256(b"test").hexdigest(),
        goal="Login as admin",
        target_url="https://example.com/login",
        domain="example.com",
        preconditions=[],
        steps=steps,
        postconditions=["Dashboard visible"],
        repair_operators=["SelReplace", "PreInsert", "ArgCorrect"],
        created_at_iso=_now_iso(),
        success_count=0,
        failure_count=0,
    )


def _make_aom_snapshot(role: str = "button", name: str = "Login") -> AOMSnapshot:
    """Build a minimal AOMSnapshot with a single interactive node."""
    leaf = AOMNode(
        role=role,
        name=name,
        value=None,
        state={},
        bbox=None,
        children=[],
        source_node_id="leaf001",
    )
    root = AOMNode(
        role="WebArea",
        name="Test Page",
        value=None,
        state={},
        bbox=None,
        children=[leaf],
        source_node_id="root001",
    )
    return AOMSnapshot(
        snapshot_id="snap001",
        url="https://example.com/login",
        timestamp_iso=_now_iso(),
        root_node=root,
        node_count=2,
        extraction_latency_ms=10,
    )


def _make_sfg_store(tmp_path: Path) -> SFGStore:
    return SFGStore(db_path=tmp_path / "state.db")


def _make_mock_instructor() -> MagicMock:
    client = MagicMock()
    client.create_structured = AsyncMock(
        return_value=MissingStep(action_type="navigate", locator="https://example.com/login")
    )
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: SelReplace repairs locator without any LLM call
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sel_replace_repairs_locator_no_llm_call(tmp_path: Path) -> None:
    """
    _sel_replace finds the element in the AOM and returns an updated step.
    No LLM call must be made during the entire operation.
    """
    sfg_store = _make_sfg_store(tmp_path)
    instructor = _make_mock_instructor()

    engine = RepairEngine(instructor_client=instructor, sfg_store=sfg_store)

    # The step has a locator for a "Login" button
    step = ContractStep(
        step_number=1,
        action_type="click",
        locator='role=button[name="Login"]',
        input_value=None,
        expected_state_hash="abc123",
    )

    # Mock AOMExtractor to return a known snapshot containing a "Login" button
    mock_snapshot = _make_aom_snapshot(role="button", name="Login")

    mock_page = MagicMock()  # Page not actually used — we mock the extractor

    with patch.object(engine._aom_extractor, "extract", new=AsyncMock(return_value=mock_snapshot)):
        result = await engine._sel_replace(step, mock_page)

    # SelReplace must produce an updated step
    assert result is not None, "_sel_replace should return a ContractStep, not None"
    assert isinstance(result, ContractStep)

    # Locator should be synthesized from the AOM node
    # The node has role="button" and name="Login" → best_locator produces
    # role=button[name="Login"] (accessible_name route)
    assert result.locator and result.locator not in ("*", "div", "")
    # The step_number and action_type must be preserved
    assert result.step_number == step.step_number
    assert result.action_type == step.action_type

    # Verify NO LLM call was made
    instructor.create_structured.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: PreInsert makes exactly one LLM call
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_insert_makes_exactly_one_llm_call(tmp_path: Path) -> None:
    """
    When SelReplace and ArgCorrect both return None, repair() falls through to
    PreInsert, which must make exactly 1 LLM call and add one step to the skill.
    """
    sfg_store = _make_sfg_store(tmp_path)

    instructor = MagicMock()
    instructor.create_structured = AsyncMock(
        return_value=MissingStep(action_type="navigate", locator="https://example.com/login")
    )

    engine = RepairEngine(instructor_client=instructor, sfg_store=sfg_store)

    original_skill = _make_skill()
    original_step_count = len(original_skill.steps)
    failed_step = original_skill.steps[0]

    # Force SelReplace to return None (AOM extraction fails)
    # Force ArgCorrect to return None (unrecognised failure_sig)
    mock_page = MagicMock()

    with patch.object(
        engine._aom_extractor,
        "extract",
        new=AsyncMock(side_effect=Exception("AOM unavailable")),
    ):
        result = await engine.repair(
            skill=original_skill,
            failed_step=failed_step,
            failure_sig="UNKNOWN_SIG",  # ArgCorrect won't match this
            page=mock_page,
        )

    # repair() should return a patched skill (not None) via PreInsert
    assert result is not None, "repair() should succeed via PreInsert, not return None"
    assert isinstance(result, ContractSkill)

    # One step must have been inserted
    assert len(result.steps) == original_step_count + 1, (
        f"Expected {original_step_count + 1} steps after PreInsert, "
        f"got {len(result.steps)}"
    )

    # Steps must be re-numbered consecutively
    for i, s in enumerate(result.steps):
        assert s.step_number == i + 1, (
            f"Step {i} has step_number={s.step_number}, expected {i + 1}"
        )

    # Exactly one LLM call
    assert instructor.create_structured.call_count == 1, (
        f"Expected exactly 1 LLM call, got {instructor.create_structured.call_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Planner injects ContractSkill when confidence is high
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planner_injects_skill_when_confidence_high(monkeypatch) -> None:
    """
    PlannerAgent.plan() must embed 'ContractSkill Available' text in the
    messages when a matching ContractSkill with success_count >= 3 is found.
    """
    from src.agents.planner import PlannerAgent
    from src.llm.schemas import TestPlan, TestStep

    # Stub domain detection so we don't need Ollama
    async def _stub_detect_domain(**kwargs):
        return "authentication"

    monkeypatch.setattr("src.agents.planner.detect_domain", _stub_detect_domain)

    # Build a fake high-confidence ContractSkill
    high_confidence_skill = _make_skill(
        steps=[
            ContractStep(
                step_number=1,
                action_type="fill",
                locator='role=textbox[name="Username"]',
                input_value="admin",
                expected_state_hash="node_a",
            ),
            ContractStep(
                step_number=2,
                action_type="click",
                locator='role=button[name="Login"]',
                input_value=None,
                expected_state_hash="node_b",
            ),
        ]
    )
    # Manually bump success_count to 3
    high_confidence_skill = high_confidence_skill.model_copy(update={"success_count": 3})

    # Mock ContractSkillStore
    mock_contract_store = MagicMock()
    mock_contract_store.find_matching_skill = AsyncMock(return_value=high_confidence_skill)

    # Stub plan return value
    def _stub_plan() -> TestPlan:
        return TestPlan(
            title="Login test",
            requirement_summary="Verify login works",
            estimated_complexity="low",
            domain="authentication",
            steps=[
                TestStep(
                    step_number=1,
                    description="Navigate to login page",
                    action="navigate",
                    expected_result="Login form visible",
                    role="user",
                )
            ],
        )

    # Capture messages passed to the instructor
    captured_messages: list[list] = []

    async def _capture_create_structured(messages, response_model, temperature=0.0):
        captured_messages.append(messages)
        return _stub_plan()

    instructor = MagicMock()
    instructor.create_structured = AsyncMock(side_effect=_capture_create_structured)

    adapter = MagicMock()
    adapter.base_url = "http://localhost:11434"
    adapter.model = "stub"

    planner = PlannerAgent(
        adapter=adapter,
        retriever=None,
        instructor_client=instructor,
        episodic_store=None,
        contract_skill_store=mock_contract_store,
    )

    plan = await planner.plan(
        requirement="Login as admin",
        url="https://example.com/login",
        role="admin",
        domain="authentication",
    )

    assert plan is not None
    assert plan.title == "Login test"

    # find_matching_skill must have been called
    mock_contract_store.find_matching_skill.assert_awaited_once()

    # "ContractSkill Available" must appear in the prompt sent to the LLM
    assert captured_messages, "Instructor create_structured was never called"
    all_text = "\n".join(m.get("content", "") for m in captured_messages[0])
    assert "ContractSkill Available" in all_text, (
        "'ContractSkill Available' not found in planner prompt. "
        f"Actual prompt excerpt: {all_text[:400]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Reporter updates success count after pass
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reporter_updates_success_count_after_pass() -> None:
    """
    The reporter_node must call contract_skill_store.update_counts() with
    success_delta=1 when the test passes and a contract_skill_id is in state.
    """
    from src.agents.graph import build_graph, initial_state, QAAgentState
    from src.llm.adapter import OllamaAdapter

    SKILL_ID = hashlib.sha256(b"test-skill").hexdigest()

    # Mock ContractSkillStore
    mock_contract_store = MagicMock()
    mock_contract_store.update_counts = AsyncMock()

    # Build graph with the mock contract_skill_store
    mock_adapter = MagicMock(spec=OllamaAdapter)
    mock_adapter.base_url = "http://localhost:11434"
    mock_adapter.model = "stub"

    graph = build_graph(
        adapter=mock_adapter,
        browser_manager=None,
        retriever=None,
        contract_skill_store=mock_contract_store,
    )

    # Construct state that simulates a successful execution with a contract_skill_id
    # We invoke only the reporter_node directly by building a minimal state
    # that satisfies its expectations.
    state: QAAgentState = QAAgentState(
        requirement="Login",
        url="https://example.com/login",
        role="admin",
        domain="authentication",
        mode="generate",
        test_plan=None,
        script={"code": "def test_stub(): pass"},
        script_path="",
        page_state="",
        page=None,
        execution_result={"success": True, "output": "1 passed"},
        error=None,
        retry_count=0,
        max_retries=3,
        session_id="test-session",
        messages=[],
        bft_status="PENDING",
        bft_node_results=[],
        bft_confidence_tier="LOW",
        contract_skill_id=SKILL_ID,
    )

    # Extract the reporter node function directly from the graph node map
    # (reporter is a closure; we need to invoke it via the compiled graph's
    # node registry, or just call the underlying function through the graph).
    # Simpler approach: patch the graph internals isn't clean.
    # Instead, test the reporter logic by running a minimal graph invocation
    # with mode="execute" so it goes executor→reporter without planner/generator.
    #
    # The cleanest approach is to test the reporter_node closure's behaviour
    # by verifying update_counts is called after a successful run.
    # We achieve this by running the full graph in execute mode with a no-op script.

    # Patch _run_script to return immediate success
    with patch("src.agents.graph._run_script", new=AsyncMock(return_value={
        "success": True,
        "output": "1 passed",
        "returncode": 0,
    })):
        state_for_run = initial_state(
            requirement="Login",
            url="https://example.com/login",
            role="admin",
            domain="authentication",
            mode="execute",
            script_dict={"code": "def test_stub(): pass"},
        )
        # Inject the contract_skill_id (TypedDict accepts extra keys via update)
        state_for_run["contract_skill_id"] = SKILL_ID  # type: ignore[typeddict-unknown-key]

        await graph.ainvoke(state_for_run)

    # update_counts must have been called with success_delta=1
    mock_contract_store.update_counts.assert_awaited_once_with(
        SKILL_ID, success_delta=1
    )
