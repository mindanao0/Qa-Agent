"""
State Flow Graph (SFG) data model — Sprint 4 / Cluster S4-A.

Provides:
  - SFGNode: Pydantic V2 model representing a page state node
  - SFGEdge: Pydantic V2 model representing a UI action edge between nodes
  - SFGStore: SQLite-backed store for nodes and edges, BFS path-finding
  - BLOCKED_ACTION_PATTERNS: frozenset of unsafe action keywords
  - is_safe_action(): module-level safety filter

Uses the SAME database as AdaptiveRouter (default ~/.qa-agent/state.db),
adding sfg_nodes and sfg_edges tables only. Follows the exact same SQLite
pattern as src/routing/adaptive_router.py.
"""
from __future__ import annotations

import pathlib
import sqlite3
import threading
from collections import deque
from typing import Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Safety filter
# ---------------------------------------------------------------------------

BLOCKED_ACTION_PATTERNS: frozenset[str] = frozenset({
    "delete",
    "remove",
    "transfer",
    "payment",
    "password",
    "logout",
    "deactivate",
})


def is_safe_action(locator: str, input_value: Optional[str]) -> bool:
    """Return True if the action is safe (no blocked keyword found)."""
    combined = (locator + (input_value or "")).lower()
    return not any(p in combined for p in BLOCKED_ACTION_PATTERNS)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SFGNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str          # sha256 of url_path + aom_hash
    url: str
    page_title: str
    aom_hash: str         # hash of AOMSnapshot content
    pam_content: str      # CompactPAM from grounder, ≤1000 tokens
    coverage_tags: list[str]  # auth_required, form, modal, list, etc.
    # Caller-managed: callers must update this list manually after calling
    # upsert_edge(). SFGStore.upsert_edge() does NOT sync this field.
    # Use get_edges_from(node_id) as the authoritative source of outgoing edges.
    outgoing_edges: list[str]  # edge_ids
    discovered_at_iso: str
    visit_count: int = 0


class SFGEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str           # sha256 of source_id + action_type + locator
    source_node_id: str
    target_node_id: str
    action_type: str       # click, fill, navigate, select
    locator: str           # best stable locator
    input_value: Optional[str]
    safety_flag: str       # SAFE | BLOCKED | PENDING
    replay_script: str     # minimal Playwright snippet to reproduce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = pathlib.Path.home() / ".qa-agent" / "state.db"


def _list_to_str(lst: list[str]) -> str:
    """Serialize list[str] as pipe-separated string for SQLite storage.

    Contract: no element may contain '|' — that character is the delimiter
    and its presence would corrupt round-trip deserialization via _str_to_list.
    """
    for elem in lst:
        if "|" in elem:
            raise ValueError(f"List element contains pipe character: {elem!r}")
    return "|".join(lst)


def _str_to_list(s: str) -> list[str]:
    """Deserialize pipe-separated string back to list[str]."""
    if not s:
        return []
    return s.split("|")


def _row_to_node(row: sqlite3.Row) -> SFGNode:
    return SFGNode(
        node_id=row["node_id"],
        url=row["url"],
        page_title=row["page_title"],
        aom_hash=row["aom_hash"],
        pam_content=row["pam_content"],
        coverage_tags=_str_to_list(row["coverage_tags"]),
        outgoing_edges=_str_to_list(row["outgoing_edges"]),
        discovered_at_iso=row["discovered_at_iso"],
        visit_count=row["visit_count"],
    )


def _row_to_edge(row: sqlite3.Row) -> SFGEdge:
    return SFGEdge(
        edge_id=row["edge_id"],
        source_node_id=row["source_node_id"],
        target_node_id=row["target_node_id"],
        action_type=row["action_type"],
        locator=row["locator"],
        input_value=row["input_value"],
        safety_flag=row["safety_flag"],
        replay_script=row["replay_script"],
    )


# ---------------------------------------------------------------------------
# SFGStore
# ---------------------------------------------------------------------------

class SFGStore:
    """SQLite-backed store for SFGNode and SFGEdge records.

    Follows the same threading + connection pattern as AdaptiveRouter:
      - threading.Lock() guards all DB access
      - sqlite3.connect(db_path, timeout=10.0) per operation
      - conn.row_factory = sqlite3.Row
      - _connect() method creates parent dirs and returns connection
    """

    def __init__(self, db_path: pathlib.Path | None = None) -> None:
        self._db_path: pathlib.Path = (db_path or _DEFAULT_DB_PATH).expanduser()
        self._lock = threading.Lock()
        self._ensure_tables()

    # ── Schema ──────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sfg_nodes (
                    node_id           TEXT PRIMARY KEY,
                    url               TEXT NOT NULL,
                    page_title        TEXT NOT NULL,
                    aom_hash          TEXT NOT NULL,
                    pam_content       TEXT NOT NULL,
                    coverage_tags     TEXT NOT NULL DEFAULT '',
                    outgoing_edges    TEXT NOT NULL DEFAULT '',
                    discovered_at_iso TEXT NOT NULL,
                    visit_count       INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sfg_edges (
                    edge_id        TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    action_type    TEXT NOT NULL,
                    locator        TEXT NOT NULL,
                    input_value    TEXT,
                    safety_flag    TEXT NOT NULL,
                    replay_script  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sfg_edges_source "
                "ON sfg_edges(source_node_id)"
            )
            conn.commit()

    # ── Write operations ─────────────────────────────────────────────────

    def upsert_node(self, node: SFGNode) -> None:
        """Insert or replace an SFGNode record."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sfg_nodes
                    (node_id, url, page_title, aom_hash, pam_content,
                     coverage_tags, outgoing_edges, discovered_at_iso, visit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    url               = excluded.url,
                    page_title        = excluded.page_title,
                    aom_hash          = excluded.aom_hash,
                    pam_content       = excluded.pam_content,
                    coverage_tags     = excluded.coverage_tags,
                    outgoing_edges    = excluded.outgoing_edges,
                    discovered_at_iso = excluded.discovered_at_iso,
                    visit_count       = excluded.visit_count
                """,
                (
                    node.node_id,
                    node.url,
                    node.page_title,
                    node.aom_hash,
                    node.pam_content,
                    _list_to_str(node.coverage_tags),
                    _list_to_str(node.outgoing_edges),
                    node.discovered_at_iso,
                    node.visit_count,
                ),
            )
            conn.commit()
        logger.debug(f"SFGStore.upsert_node | node_id={node.node_id[:12]} url={node.url}")

    def upsert_edge(self, edge: SFGEdge) -> None:
        """Insert or replace an SFGEdge record."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sfg_edges
                    (edge_id, source_node_id, target_node_id, action_type,
                     locator, input_value, safety_flag, replay_script)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    source_node_id = excluded.source_node_id,
                    target_node_id = excluded.target_node_id,
                    action_type    = excluded.action_type,
                    locator        = excluded.locator,
                    input_value    = excluded.input_value,
                    safety_flag    = excluded.safety_flag,
                    replay_script  = excluded.replay_script
                """,
                (
                    edge.edge_id,
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.action_type,
                    edge.locator,
                    edge.input_value,
                    edge.safety_flag,
                    edge.replay_script,
                ),
            )
            conn.commit()
        logger.debug(
            f"SFGStore.upsert_edge | edge_id={edge.edge_id[:12]} "
            f"{edge.source_node_id[:8]}→{edge.target_node_id[:8]}"
        )

    # ── Read operations ──────────────────────────────────────────────────

    def get_node(self, node_id: str) -> SFGNode | None:
        """Retrieve a node by node_id, or None if not found."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sfg_nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            return _row_to_node(row) if row else None

    def get_edges_from(self, node_id: str) -> list[SFGEdge]:
        """Return all edges whose source_node_id matches node_id."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sfg_edges WHERE source_node_id = ?",
                (node_id,),
            ).fetchall()
            return [_row_to_edge(r) for r in rows]

    def find_path(
        self,
        start_url: str,
        goal_description: str,
        max_depth: int = 10,
    ) -> list[SFGEdge] | None:
        """BFS over stored edges to find a path from start_url.

        - start_url is matched exactly against sfg_nodes.url
        - goal_description is matched as a substring of url or page_title
        - Returns the first path found within max_depth hops as list[SFGEdge]
        - Returns None if no path found or no starting nodes exist
        - No LLM calls — pure DB graph traversal
        - Returns ``None`` if no path of 1+ edges is found. Note: if
          ``start_url`` already matches ``goal_description``, returns ``None``
          (not a zero-edge path).
        """
        # Fetch all starting node IDs
        with self._lock, self._connect() as conn:
            start_rows = conn.execute(
                "SELECT node_id FROM sfg_nodes WHERE url = ?",
                (start_url,),
            ).fetchall()

        start_ids = [r["node_id"] for r in start_rows]
        if not start_ids:
            logger.debug(f"SFGStore.find_path | no nodes for url={start_url!r}")
            return None

        goal_lower = goal_description.lower()

        def _node_matches_goal(nid: str) -> bool:
            node = self.get_node(nid)
            if node is None:
                return False
            return (
                goal_lower in node.url.lower()
                or goal_lower in node.page_title.lower()
            )

        # BFS: queue holds (current_node_id, path_so_far: list[SFGEdge])
        queue: deque[tuple[str, list[SFGEdge]]] = deque()
        visited: set[str] = set()

        for sid in start_ids:
            queue.append((sid, []))
            visited.add(sid)

        while queue:
            current_id, path = queue.popleft()

            if len(path) > 0 and _node_matches_goal(current_id):
                logger.debug(
                    f"SFGStore.find_path | found path of {len(path)} edges"
                )
                return path

            if len(path) >= max_depth:
                continue

            edges = self.get_edges_from(current_id)
            for edge in edges:
                next_id = edge.target_node_id
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [edge]))

        logger.debug(
            f"SFGStore.find_path | no path found from url={start_url!r} "
            f"to goal={goal_description!r}"
        )
        return None

    # ── Counts ───────────────────────────────────────────────────────────

    def node_count(self) -> int:
        """Return total number of nodes in the store."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM sfg_nodes").fetchone()
            return int(row["cnt"])

    def edge_count(self) -> int:
        """Return total number of edges in the store."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM sfg_edges").fetchone()
            return int(row["cnt"])

    def get_nodes_by_url_prefix(self, url: str) -> list[SFGNode]:
        """Return all nodes whose URL starts with the given prefix."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sfg_nodes WHERE url = ? OR url LIKE ?",
                (url, url.rstrip("/") + "%"),
            ).fetchall()
        return [_row_to_node(row) for row in rows]


__all__ = [
    "BLOCKED_ACTION_PATTERNS",
    "is_safe_action",
    "SFGNode",
    "SFGEdge",
    "SFGStore",
]
