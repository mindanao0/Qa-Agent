"""
Tests for src/contractskill/sfg.py — Sprint 4 / S4-A.

5 tests:
  test_node_upsert_and_retrieve
  test_edge_creates_connection_between_nodes
  test_find_path_returns_edge_sequence
  test_safety_filter_blocks_delete_actions
  test_same_url_different_aom_creates_different_nodes
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.contractskill.sfg import (
    BLOCKED_ACTION_PATTERNS,
    SFGEdge,
    SFGNode,
    SFGStore,
    is_safe_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_node(
    url: str,
    aom_hash: str,
    page_title: str = "Test Page",
    coverage_tags: list[str] | None = None,
    outgoing_edges: list[str] | None = None,
    visit_count: int = 0,
) -> SFGNode:
    node_id = _sha256(url, aom_hash)
    return SFGNode(
        node_id=node_id,
        url=url,
        page_title=page_title,
        aom_hash=aom_hash,
        pam_content="<compact pam>",
        coverage_tags=coverage_tags or [],
        outgoing_edges=outgoing_edges or [],
        discovered_at_iso=_now_iso(),
        visit_count=visit_count,
    )


def _make_edge(
    source_node_id: str,
    target_node_id: str,
    action_type: str = "click",
    locator: str = "role=button[name='Submit']",
    input_value: str | None = None,
    safety_flag: str = "SAFE",
    replay_script: str = "await page.get_by_role('button', name='Submit').click()",
) -> SFGEdge:
    edge_id = _sha256(source_node_id, action_type, locator)
    return SFGEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        action_type=action_type,
        locator=locator,
        input_value=input_value,
        safety_flag=safety_flag,
        replay_script=replay_script,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_node_upsert_and_retrieve(tmp_path: Path) -> None:
    """Upserted node can be retrieved with all fields intact."""
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    node = _make_node(
        url="https://example.com/dashboard",
        aom_hash="abc123",
        page_title="Dashboard",
        coverage_tags=["auth_required", "list"],
        visit_count=3,
    )
    store.upsert_node(node)

    retrieved = store.get_node(node.node_id)
    assert retrieved is not None
    assert retrieved.node_id == node.node_id
    assert retrieved.url == "https://example.com/dashboard"
    assert retrieved.page_title == "Dashboard"
    assert retrieved.aom_hash == "abc123"
    assert retrieved.coverage_tags == ["auth_required", "list"]
    assert retrieved.visit_count == 3
    assert store.node_count() == 1


def test_edge_creates_connection_between_nodes(tmp_path: Path) -> None:
    """Edge stored between two nodes is retrievable via get_edges_from."""
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    node_a = _make_node(url="https://example.com/login", aom_hash="hash_a")
    node_b = _make_node(url="https://example.com/home", aom_hash="hash_b")
    store.upsert_node(node_a)
    store.upsert_node(node_b)

    edge = _make_edge(
        source_node_id=node_a.node_id,
        target_node_id=node_b.node_id,
        action_type="click",
        locator="role=button[name='Login']",
        replay_script="await page.get_by_role('button', name='Login').click()",
    )
    store.upsert_edge(edge)

    edges = store.get_edges_from(node_a.node_id)
    assert len(edges) == 1
    e = edges[0]
    assert e.edge_id == edge.edge_id
    assert e.source_node_id == node_a.node_id
    assert e.target_node_id == node_b.node_id
    assert e.action_type == "click"
    assert e.safety_flag == "SAFE"
    assert store.edge_count() == 1

    # No edges from node_b
    assert store.get_edges_from(node_b.node_id) == []


def test_find_path_returns_edge_sequence(tmp_path: Path) -> None:
    """BFS finds a multi-hop path and returns the edge sequence."""
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    # Build: start → middle → goal
    node_start = _make_node(
        url="https://example.com/", aom_hash="h0", page_title="Home"
    )
    node_mid = _make_node(
        url="https://example.com/catalog", aom_hash="h1", page_title="Catalog"
    )
    node_goal = _make_node(
        url="https://example.com/checkout", aom_hash="h2", page_title="Checkout Page"
    )

    store.upsert_node(node_start)
    store.upsert_node(node_mid)
    store.upsert_node(node_goal)

    edge1 = _make_edge(
        source_node_id=node_start.node_id,
        target_node_id=node_mid.node_id,
        locator="role=link[name='Catalog']",
    )
    edge2 = _make_edge(
        source_node_id=node_mid.node_id,
        target_node_id=node_goal.node_id,
        locator="role=link[name='Checkout']",
    )
    store.upsert_edge(edge1)
    store.upsert_edge(edge2)

    path = store.find_path(
        start_url="https://example.com/",
        goal_description="checkout",
    )

    assert path is not None
    assert len(path) == 2
    assert path[0].edge_id == edge1.edge_id
    assert path[1].edge_id == edge2.edge_id


def test_safety_filter_blocks_delete_actions(tmp_path: Path) -> None:
    """is_safe_action returns False for any blocked keyword in locator or input_value."""
    # Safe actions
    assert is_safe_action("role=button[name='Submit']", None) is True
    assert is_safe_action("role=textbox[name='Username']", "alice") is True
    assert is_safe_action("role=link[name='Home']", None) is True

    # Blocked via locator
    assert is_safe_action("role=button[name='Delete Account']", None) is False
    assert is_safe_action("role=button[name='Remove Item']", None) is False
    assert is_safe_action("role=button[name='Logout']", None) is False
    assert is_safe_action("role=button[name='Deactivate']", None) is False

    # Blocked via input_value
    assert is_safe_action("role=textbox[name='Action']", "transfer funds") is False
    assert is_safe_action("role=textbox[name='Field']", "payment method") is False

    # Blocked via input_value containing 'password'
    assert is_safe_action("role=textbox[name='New Value']", "new password here") is False

    # Verify frozenset is complete
    assert "delete" in BLOCKED_ACTION_PATTERNS
    assert "remove" in BLOCKED_ACTION_PATTERNS
    assert "transfer" in BLOCKED_ACTION_PATTERNS
    assert "payment" in BLOCKED_ACTION_PATTERNS
    assert "password" in BLOCKED_ACTION_PATTERNS
    assert "logout" in BLOCKED_ACTION_PATTERNS
    assert "deactivate" in BLOCKED_ACTION_PATTERNS


def test_find_path_returns_none_when_unreachable(tmp_path: Path) -> None:
    """find_path returns None when start_url has no matching nodes or goal is unreachable."""
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    # Case 1: start_url has no matching nodes in the store at all
    result = store.find_path(
        start_url="https://example.com/nonexistent",
        goal_description="checkout",
    )
    assert result is None, "Expected None when start_url matches no stored node"

    # Case 2: graph exists but goal_description matches nothing reachable
    node_a = _make_node(url="https://example.com/home", aom_hash="ha", page_title="Home")
    node_b = _make_node(url="https://example.com/about", aom_hash="hb", page_title="About Us")
    store.upsert_node(node_a)
    store.upsert_node(node_b)
    edge = _make_edge(
        source_node_id=node_a.node_id,
        target_node_id=node_b.node_id,
        locator="role=link[name='About']",
    )
    store.upsert_edge(edge)

    result = store.find_path(
        start_url="https://example.com/home",
        goal_description="checkout",  # no node contains "checkout"
    )
    assert result is None, "Expected None when goal_description matches nothing reachable"


def test_same_url_different_aom_creates_different_nodes(tmp_path: Path) -> None:
    """Two nodes with the same URL but different aom_hash get different node_ids."""
    db = tmp_path / "state.db"
    store = SFGStore(db_path=db)

    url = "https://example.com/form"

    node_v1 = _make_node(url=url, aom_hash="aom_hash_v1", page_title="Form (empty)")
    node_v2 = _make_node(url=url, aom_hash="aom_hash_v2", page_title="Form (filled)")

    # node_ids must differ
    assert node_v1.node_id != node_v2.node_id

    store.upsert_node(node_v1)
    store.upsert_node(node_v2)

    # Both nodes are stored independently
    assert store.node_count() == 2

    retrieved_v1 = store.get_node(node_v1.node_id)
    retrieved_v2 = store.get_node(node_v2.node_id)

    assert retrieved_v1 is not None
    assert retrieved_v2 is not None
    assert retrieved_v1.page_title == "Form (empty)"
    assert retrieved_v2.page_title == "Form (filled)"
    assert retrieved_v1.aom_hash == "aom_hash_v1"
    assert retrieved_v2.aom_hash == "aom_hash_v2"
