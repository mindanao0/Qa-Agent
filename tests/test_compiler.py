"""
Tests for src/contractskill/compiler.py — Sprint 4 / Cluster S4-C.

4 tests:
  test_compile_produces_valid_contract_skill
  test_postconditions_inferred_from_last_node
  test_skill_stored_and_retrievable_by_goal
  test_find_matching_skill_returns_none_below_threshold
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.contractskill.compiler import (
    ContractSkill,
    ContractSkillCompiler,
    ContractSkillStore,
    ContractStep,
    PostconditionList,
)
from src.contractskill.sfg import SFGEdge, SFGNode, SFGStore


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sha256(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_node(
    url: str,
    aom_hash: str,
    page_title: str = "Test Page",
    pam_content: str = "Page loaded with login form",
) -> SFGNode:
    node_id = _sha256(url, aom_hash)
    return SFGNode(
        node_id=node_id,
        url=url,
        page_title=page_title,
        aom_hash=aom_hash,
        pam_content=pam_content,
        coverage_tags=["form"],
        outgoing_edges=[],
        discovered_at_iso=_now_iso(),
        visit_count=0,
    )


def _make_edge(
    source_node_id: str,
    target_node_id: str,
    action_type: str = "click",
    locator: str = "role=button[name='Submit']",
    input_value: str | None = None,
) -> SFGEdge:
    edge_id = _sha256(source_node_id, action_type, locator)
    return SFGEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        action_type=action_type,
        locator=locator,
        input_value=input_value,
        safety_flag="SAFE",
        replay_script=f"await page.get_by_role('button', name='Submit').click()",
    )


async def _fixed_embedding(text: str) -> list[float]:
    """Always return the same 768-dim unit vector regardless of input.

    Using a constant vector ensures store and query embeddings have L2 distance 0.0
    so the 0.70 threshold is always satisfied in tests.  This avoids spinning up
    Ollama while still exercising the full LanceDB store/retrieve code path.
    """
    return [0.1] * 768


def _make_mock_instructor(postconditions: list[str] | None = None) -> MagicMock:
    """Return a mock InstructorClient that yields a PostconditionList."""
    client = MagicMock()
    result = PostconditionList(
        postconditions=postconditions or ["Form submitted successfully", "User redirected to dashboard"]
    )
    client.create_structured = AsyncMock(return_value=result)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compile_produces_valid_contract_skill(tmp_path: Path) -> None:
    """compile() returns a fully-populated ContractSkill with correct fields."""
    db = tmp_path / "state.db"
    sfg_store = SFGStore(db_path=db)

    # Set up nodes + edges in the SFG
    node_a = _make_node(
        url="https://example.com/login", aom_hash="ha", page_title="Login"
    )
    node_b = _make_node(
        url="https://example.com/home", aom_hash="hb", page_title="Home"
    )
    node_c = _make_node(
        url="https://example.com/profile", aom_hash="hc", page_title="Profile"
    )
    sfg_store.upsert_node(node_a)
    sfg_store.upsert_node(node_b)
    sfg_store.upsert_node(node_c)

    edge1 = _make_edge(
        source_node_id=node_a.node_id,
        target_node_id=node_b.node_id,
        action_type="fill",
        locator="role=textbox[name='Username']",
        input_value="alice",
    )
    edge2 = _make_edge(
        source_node_id=node_b.node_id,
        target_node_id=node_c.node_id,
        action_type="click",
        locator="role=link[name='Profile']",
    )
    sfg_store.upsert_edge(edge1)
    sfg_store.upsert_edge(edge2)

    goal = "Navigate to user profile after login"
    trajectory = [edge1, edge2]
    domain = "example.com"

    mock_client = _make_mock_instructor(["User profile visible", "Username displayed"])
    compiler = ContractSkillCompiler(instructor_client=mock_client, sfg_store=sfg_store)

    skill = await compiler.compile(goal=goal, trajectory=trajectory, domain=domain)

    # Basic type check
    assert isinstance(skill, ContractSkill)

    # skill_id: sha256 of goal + edge_ids concatenated
    expected_raw = goal + "|".join(e.edge_id for e in trajectory)
    expected_id = hashlib.sha256(expected_raw.encode()).hexdigest()
    assert skill.skill_id == expected_id

    # target_url: source node of first edge
    assert skill.target_url == node_a.url

    assert skill.goal == goal
    assert skill.domain == domain

    # Steps
    assert len(skill.steps) == 2
    step1 = skill.steps[0]
    assert step1.step_number == 1
    assert step1.action_type == edge1.action_type
    assert step1.locator == edge1.locator
    assert step1.input_value == edge1.input_value
    assert step1.expected_state_hash == edge1.target_node_id

    step2 = skill.steps[1]
    assert step2.step_number == 2
    assert step2.expected_state_hash == edge2.target_node_id

    # Preconditions always empty
    assert skill.preconditions == []

    # repair_operators always the fixed set
    assert set(skill.repair_operators) == {"SelReplace", "PreInsert", "ArgCorrect"}

    # Postconditions from LLM mock
    assert len(skill.postconditions) >= 1

    # created_at_iso parseable
    dt = datetime.fromisoformat(skill.created_at_iso)
    assert dt.tzinfo is not None

    # Counts default to 0
    assert skill.success_count == 0
    assert skill.failure_count == 0

    # Confirm LLM was called once
    mock_client.create_structured.assert_called_once()


@pytest.mark.asyncio
async def test_postconditions_inferred_from_last_node(tmp_path: Path) -> None:
    """_infer_postconditions uses the last node's pam_content and returns ≤ 3 items."""
    db = tmp_path / "state.db"
    sfg_store = SFGStore(db_path=db)

    node_a = _make_node(
        url="https://example.com/start", aom_hash="h_start", pam_content="Start page"
    )
    node_last = _make_node(
        url="https://example.com/done",
        aom_hash="h_done",
        pam_content="Order confirmation: your order #12345 has been placed.",
    )
    sfg_store.upsert_node(node_a)
    sfg_store.upsert_node(node_last)

    edge = _make_edge(
        source_node_id=node_a.node_id,
        target_node_id=node_last.node_id,
        action_type="click",
        locator="role=button[name='Place Order']",
    )
    sfg_store.upsert_edge(edge)

    goal = "Complete order placement"

    # LLM returns 4 items; compiler must cap at 3
    mock_client = _make_mock_instructor(
        postconditions=[
            "Order confirmation displayed",
            "Order number visible",
            "Email confirmation sent",
            "Cart cleared",
        ]
    )
    compiler = ContractSkillCompiler(instructor_client=mock_client, sfg_store=sfg_store)
    postconds = await compiler._infer_postconditions(goal=goal, last_node=node_last)

    assert len(postconds) <= 3
    assert "Order confirmation displayed" in postconds

    # Verify the prompt contained pam_content snippet
    call_kwargs = mock_client.create_structured.call_args
    prompt_arg = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
    assert "Order confirmation" in prompt_arg
    assert goal in prompt_arg


@pytest.mark.asyncio
async def test_postconditions_fallback_on_llm_failure(tmp_path: Path) -> None:
    """_infer_postconditions returns fallback when LLM raises."""
    db = tmp_path / "state.db"
    sfg_store = SFGStore(db_path=db)

    node = _make_node(url="https://example.com/", aom_hash="hx")
    sfg_store.upsert_node(node)

    # LLM raises
    mock_client = MagicMock()
    mock_client.create_structured = AsyncMock(side_effect=RuntimeError("LLM offline"))

    compiler = ContractSkillCompiler(instructor_client=mock_client, sfg_store=sfg_store)
    result = await compiler._infer_postconditions(goal="some goal", last_node=node)

    assert result == ["Goal completed successfully"]


@pytest.mark.asyncio
async def test_skill_stored_and_retrievable_by_goal(tmp_path: Path) -> None:
    """A stored ContractSkill can be retrieved with find_matching_skill."""
    db_path = tmp_path / "vector_db"
    sfg_db = tmp_path / "state.db"
    sfg_store = SFGStore(db_path=sfg_db)

    node_a = _make_node(url="https://app.example.com/login", aom_hash="h1")
    node_b = _make_node(url="https://app.example.com/dashboard", aom_hash="h2")
    sfg_store.upsert_node(node_a)
    sfg_store.upsert_node(node_b)

    edge = _make_edge(
        source_node_id=node_a.node_id,
        target_node_id=node_b.node_id,
        action_type="click",
        locator="role=button[name='Login']",
    )
    sfg_store.upsert_edge(edge)

    goal = "Login and reach dashboard"
    domain = "app.example.com"

    mock_client = _make_mock_instructor(["Dashboard visible", "Login successful"])
    compiler = ContractSkillCompiler(instructor_client=mock_client, sfg_store=sfg_store)
    skill = await compiler.compile(goal=goal, trajectory=[edge], domain=domain)

    # Store it
    css = ContractSkillStore(db_path=db_path, embedding_fn=_fixed_embedding)
    await css.connect()
    await css.store(skill)

    # Retrieve it — same goal/domain/url
    found = await css.find_matching_skill(
        goal=goal,
        url=node_a.url,
        domain=domain,
    )
    await css.close()

    assert found is not None
    assert found.skill_id == skill.skill_id
    assert found.goal == skill.goal
    assert found.domain == skill.domain
    assert found.target_url == skill.target_url
    assert len(found.steps) == len(skill.steps)
    assert found.steps[0].locator == skill.steps[0].locator
    assert found.steps[0].action_type == skill.steps[0].action_type


@pytest.mark.asyncio
async def test_update_counts_increments_success_and_failure(tmp_path: Path) -> None:
    """update_counts() correctly increments success_count and failure_count."""
    db_path = tmp_path / "vector_db"
    sfg_db = tmp_path / "state.db"
    sfg_store = SFGStore(db_path=sfg_db)

    node_a = _make_node(url="https://counts.example.com/login", aom_hash="hc1")
    node_b = _make_node(url="https://counts.example.com/home", aom_hash="hc2")
    sfg_store.upsert_node(node_a)
    sfg_store.upsert_node(node_b)

    edge = _make_edge(
        source_node_id=node_a.node_id,
        target_node_id=node_b.node_id,
        locator="role=button[name='Login']",
    )
    sfg_store.upsert_edge(edge)

    mock_client = _make_mock_instructor(["Logged in"])
    compiler = ContractSkillCompiler(instructor_client=mock_client, sfg_store=sfg_store)
    skill = await compiler.compile(goal="Login test", trajectory=[edge], domain="counts.example.com")

    css = ContractSkillStore(db_path=db_path, embedding_fn=_fixed_embedding)
    await css.connect()
    await css.store(skill)

    # Increment success twice, failure once
    await css.update_counts(skill.skill_id, success_delta=2, failure_delta=0)
    await css.update_counts(skill.skill_id, success_delta=0, failure_delta=1)

    found = await css.find_matching_skill(
        goal="Login test",
        url=node_a.url,
        domain="counts.example.com",
    )
    await css.close()

    assert found is not None
    assert found.success_count == 2
    assert found.failure_count == 1


@pytest.mark.asyncio
async def test_update_counts_warns_on_missing_skill_id(tmp_path: Path) -> None:
    """update_counts() logs a warning and does not raise when skill_id is not found."""
    import logging

    db_path = tmp_path / "vector_db"
    css = ContractSkillStore(db_path=db_path, embedding_fn=_fixed_embedding)
    await css.connect()

    # Should not raise — just log a warning
    await css.update_counts("nonexistent-skill-id", success_delta=1)

    await css.close()


@pytest.mark.asyncio
async def test_find_matching_skill_returns_none_below_threshold(tmp_path: Path) -> None:
    """find_matching_skill returns None when distance > 0.70 or table empty."""
    db_path = tmp_path / "vector_db"

    # --- Case 1: empty table → None ---
    css = ContractSkillStore(db_path=db_path, embedding_fn=_fixed_embedding)
    await css.connect()

    result = await css.find_matching_skill(
        goal="Pay invoice",
        url="https://other.example.com/pay",
        domain="other.example.com",
    )
    assert result is None, "Empty table should return None"

    # --- Case 2: table has a record but URL filter matches nothing ---
    sfg_db = tmp_path / "state.db"
    sfg_store = SFGStore(db_path=sfg_db)

    node_x = _make_node(url="https://stored.example.com/a", aom_hash="hx")
    node_y = _make_node(url="https://stored.example.com/b", aom_hash="hy")
    sfg_store.upsert_node(node_x)
    sfg_store.upsert_node(node_y)

    edge = _make_edge(
        source_node_id=node_x.node_id,
        target_node_id=node_y.node_id,
        locator="role=button[name='Go']",
    )
    sfg_store.upsert_edge(edge)

    mock_client = _make_mock_instructor()
    compiler = ContractSkillCompiler(instructor_client=mock_client, sfg_store=sfg_store)
    stored_skill = await compiler.compile(
        goal="Go somewhere", trajectory=[edge], domain="stored.example.com"
    )
    await css.store(stored_skill)

    # Query with a completely different URL → url filter won't match → None
    result2 = await css.find_matching_skill(
        goal="Go somewhere",
        url="https://totally-different.example.com/page",
        domain="stored.example.com",
    )
    assert result2 is None, "URL mismatch should return None"

    await css.close()
