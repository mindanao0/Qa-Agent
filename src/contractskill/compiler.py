"""
ContractSkill Compiler — Sprint 4 / Cluster S4-C.

Provides:
  - ContractStep: Pydantic V2 model for a single compiled step
  - ContractSkill: Pydantic V2 model for a complete skill artifact
  - ContractSkillCompiler: compiles SFG trajectory → ContractSkill
  - ContractSkillStore: LanceDB-backed store for ContractSkill records

LanceDB pattern follows src/memory/episodic_store.py exactly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import lancedb
import pyarrow as pa
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.contractskill.sfg import SFGEdge, SFGNode, SFGStore
from src.llm.instructor_client import InstructorClient
from src.memory.episodic_store import EMBEDDING_DIM

_CONTRACT_SKILLS_TABLE = "contract_skills"
_DEFAULT_DB_PATH = Path.home() / ".qa-agent" / "vector_db"
_DISTANCE_THRESHOLD = 0.70

EmbeddingFn = Callable[[str], Awaitable[list[float]]]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic V2 schemas
# ─────────────────────────────────────────────────────────────────────────────


class ContractStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_number: int
    action_type: str
    locator: str              # from SFGEdge, verified against AOM
    input_value: str | None
    expected_state_hash: str  # target SFGNode.node_id after action


class ContractSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str              # sha256 of goal + trajectory hash
    goal: str
    target_url: str
    domain: str
    preconditions: list[str]
    steps: list[ContractStep]
    postconditions: list[str]
    repair_operators: list[Literal["SelReplace", "PreInsert", "ArgCorrect"]]
    created_at_iso: str
    success_count: int = 0
    failure_count: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# LLM helper schema (not exported)
# ─────────────────────────────────────────────────────────────────────────────


class PostconditionList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    postconditions: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Compiler
# ─────────────────────────────────────────────────────────────────────────────


class ContractSkillCompiler:
    """Compiles a list of SFGEdges (a trajectory) into a ContractSkill."""

    def __init__(
        self,
        instructor_client: InstructorClient,
        sfg_store: SFGStore,
    ) -> None:
        self._client = instructor_client
        self._store = sfg_store

    async def compile(
        self,
        goal: str,
        trajectory: list[SFGEdge],
        domain: str,
    ) -> ContractSkill:
        """
        Given a path through the SFG (list of edges), compile into
        a ContractSkill artifact with LLM-inferred postconditions.
        """
        if not trajectory:
            raise ValueError("trajectory must contain at least one SFGEdge")

        # Determine target_url from the source node of the first edge
        first_edge = trajectory[0]
        source_node = self._store.get_node(first_edge.source_node_id)
        if source_node is None:
            logger.warning(
                f"ContractSkillCompiler: source node {first_edge.source_node_id!r} not found in SFGStore — target_url will be empty"
            )
        target_url = source_node.url if source_node else ""

        # skill_id = sha256(goal + "|".join(edge.edge_id for edge in trajectory))
        edge_ids_str = "|".join(edge.edge_id for edge in trajectory)
        raw = goal + edge_ids_str
        skill_id = hashlib.sha256(raw.encode()).hexdigest()

        # Build steps (1-indexed)
        steps = [
            ContractStep(
                step_number=i + 1,
                action_type=edge.action_type,
                locator=edge.locator,
                input_value=edge.input_value,
                expected_state_hash=edge.target_node_id,
            )
            for i, edge in enumerate(trajectory)
        ]

        # Infer postconditions via LLM
        last_edge = trajectory[-1]
        last_node = self._store.get_node(last_edge.target_node_id)
        postconditions = await self._infer_postconditions(goal, last_node)

        return ContractSkill(
            skill_id=skill_id,
            goal=goal,
            target_url=target_url,
            domain=domain,
            preconditions=[],
            steps=steps,
            postconditions=postconditions,
            repair_operators=["SelReplace", "PreInsert", "ArgCorrect"],
            created_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    async def _infer_postconditions(
        self,
        goal: str,
        last_node: SFGNode | None,
    ) -> list[str]:
        """Single LLM call (T=0.0) to infer what should be true
        after executing the trajectory. Keep under 3 postconditions."""
        if last_node is None:
            logger.warning(
                "ContractSkillCompiler: last node not found — postconditions will be generic"
            )
        pam_snippet = ""
        if last_node is not None:
            pam_snippet = last_node.pam_content[:500]

        prompt = (
            f"Given goal: {goal}\n"
            f"Page after completion: {pam_snippet}\n"
            "List up to 3 postconditions (assertions that should be true after "
            "completing this goal). Be specific and brief."
        )

        try:
            result = await self._client.create_structured(
                prompt=prompt,
                response_model=PostconditionList,
                temperature=0.0,
            )
            # Cap at 3 items
            return result.postconditions[:3]
        except Exception as exc:
            logger.warning(
                f"ContractSkillCompiler._infer_postconditions failed: {exc!r}; "
                "returning default postcondition"
            )
            return ["Goal completed successfully"]


# ─────────────────────────────────────────────────────────────────────────────
# ContractSkillStore (LanceDB)
# ─────────────────────────────────────────────────────────────────────────────


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


class ContractSkillStore:
    """LanceDB-backed store for ContractSkill records.

    Follows the exact same pattern as src/memory/episodic_store.EpisodicStore.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self._db_path: Path = (db_path or _DEFAULT_DB_PATH).expanduser()
        self._embedding_fn: EmbeddingFn | None = embedding_fn
        self._db: Any = None
        self._table: Any = None

    # ── Schema ─────────────────────────────────────────────────────────────

    def _schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("skill_id", pa.string()),
            pa.field("goal", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("target_url", pa.string()),
            pa.field("steps_json", pa.string()),
            pa.field("postconditions_json", pa.string()),
            pa.field("preconditions_json", pa.string()),
            pa.field("created_at_iso", pa.string()),
            pa.field("success_count", pa.int32()),
            pa.field("failure_count", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def connect(self) -> "ContractSkillStore":
        """Open the LanceDB database and create the table if missing."""
        self._db_path.mkdir(parents=True, exist_ok=True)

        def _open() -> tuple[Any, Any]:
            db = lancedb.connect(str(self._db_path))
            names = set(db.table_names())
            if _CONTRACT_SKILLS_TABLE in names:
                table = db.open_table(_CONTRACT_SKILLS_TABLE)
            else:
                table = db.create_table(
                    _CONTRACT_SKILLS_TABLE, schema=self._schema()
                )
            return db, table

        self._db, self._table = await asyncio.to_thread(_open)
        logger.info(
            f"ContractSkillStore connected | path={str(self._db_path)!r} "
            f"table={_CONTRACT_SKILLS_TABLE!r}"
        )
        return self

    async def close(self) -> None:
        self._table = None
        self._db = None

    async def __aenter__(self) -> "ContractSkillStore":
        return await self.connect()

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Embedding helper ────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        if self._embedding_fn is None:
            raise RuntimeError(
                "ContractSkillStore: no embedding_fn configured; "
                "pass one to the constructor."
            )
        vec = await self._embedding_fn(text)
        if len(vec) != EMBEDDING_DIM:
            raise ValueError(
                f"ContractSkillStore: embedding has dim={len(vec)}, "
                f"expected {EMBEDDING_DIM}"
            )
        return [float(v) for v in vec]

    # ── Write ───────────────────────────────────────────────────────────────

    async def store(self, skill: ContractSkill) -> None:
        """Embed and store a ContractSkill in LanceDB."""
        if self._table is None:
            raise RuntimeError("ContractSkillStore not connected — call connect() first")

        # Embed: goal + domain + first-3 locators
        embed_text = (
            skill.goal
            + " "
            + skill.domain
            + " "
            + " ".join(s.locator for s in skill.steps[:3])
        )
        embedding = await self._embed(embed_text)

        steps_json = json.dumps([s.model_dump() for s in skill.steps])
        postconditions_json = json.dumps(skill.postconditions)
        preconditions_json = json.dumps(skill.preconditions)

        record = {
            "skill_id": skill.skill_id,
            "goal": skill.goal,
            "domain": skill.domain,
            "target_url": skill.target_url,
            "steps_json": steps_json,
            "postconditions_json": postconditions_json,
            "preconditions_json": preconditions_json,
            "created_at_iso": skill.created_at_iso,
            "success_count": int(skill.success_count),
            "failure_count": int(skill.failure_count),
            "embedding": embedding,
        }

        await asyncio.to_thread(self._table.add, [record])
        logger.debug(
            f"ContractSkillStore.store | skill_id={skill.skill_id[:12]} "
            f"goal={skill.goal!r} domain={skill.domain}"
        )

    async def update_counts(
        self,
        skill_id: str,
        success_delta: int = 0,
        failure_delta: int = 0,
    ) -> None:
        """Increment success/failure counts for a stored skill by re-reading and re-writing the row."""
        if self._table is None:
            raise RuntimeError("ContractSkillStore not connected — call connect() first")

        def _update() -> bool:
            rows: list[dict[str, Any]] = self._table.to_arrow().to_pylist()
            matching = [r for r in rows if r.get("skill_id") == skill_id]
            if not matching:
                return False
            row = matching[0]
            # Increment counts in-place
            row["success_count"] = int(row.get("success_count", 0)) + success_delta
            row["failure_count"] = int(row.get("failure_count", 0)) + failure_delta
            # Delete all rows with this skill_id then re-add the updated row
            self._table.delete(f"skill_id = '{_sql_escape(skill_id)}'")
            self._table.add([row])
            return True

        found = await asyncio.to_thread(_update)
        if not found:
            logger.warning(
                f"ContractSkillStore.update_counts | skill_id={skill_id!r} not found — no update performed"
            )

    # ── Retrieval ───────────────────────────────────────────────────────────

    async def find_matching_skill(
        self,
        goal: str,
        url: str,
        domain: str,
    ) -> ContractSkill | None:
        """Vector search for the best matching skill for the given goal+url+domain.

        Returns None if the table is empty or the best match vector distance
        exceeds _DISTANCE_THRESHOLD (0.70).
        """
        if self._table is None:
            raise RuntimeError("ContractSkillStore not connected — call connect() first")

        query_text = goal + " " + domain
        query_embedding = await self._embed(query_text)

        escaped_url = _sql_escape(url)
        where_clause = f"target_url = '{escaped_url}'"

        def _search() -> list[dict[str, Any]]:
            try:
                return (
                    self._table.search(query_embedding)
                    .where(where_clause, prefilter=True)
                    .limit(1)
                    .to_list()
                )
            except Exception as exc:
                logger.warning(
                    f"ContractSkillStore vector+filter search failed: {exc}; "
                    "falling back to full scan"
                )
                try:
                    rows: list[dict[str, Any]] = self._table.to_arrow().to_pylist()
                    return [r for r in rows if r.get("target_url") == url][:1]
                except Exception as exc2:
                    logger.error(f"ContractSkillStore fallback scan failed: {exc2}")
                    return []

        rows = await asyncio.to_thread(_search)

        if not rows:
            logger.debug(
                "ContractSkillStore.find_matching_skill | no results for "
                f"url={url!r} goal={goal!r}"
            )
            return None

        row = rows[0]
        # LanceDB returns _distance key when doing vector search;
        # fallback plain-scan rows have no _distance → treat as worst case (1.0)
        distance = row.get("_distance", 1.0)
        if distance > _DISTANCE_THRESHOLD:
            logger.debug(
                f"ContractSkillStore.find_matching_skill | distance={distance:.4f} "
                f"> threshold={_DISTANCE_THRESHOLD}; returning None"
            )
            return None

        # Deserialize steps, postconditions, preconditions from JSON
        steps_raw: list[dict[str, Any]] = json.loads(row.get("steps_json", "[]"))
        steps = [ContractStep(**s) for s in steps_raw]
        postconditions: list[str] = json.loads(row.get("postconditions_json", "[]"))
        preconditions: list[str] = json.loads(row.get("preconditions_json", "[]"))

        skill = ContractSkill(
            skill_id=str(row["skill_id"]),
            goal=str(row["goal"]),
            target_url=str(row["target_url"]),
            domain=str(row["domain"]),
            preconditions=preconditions,
            steps=steps,
            postconditions=postconditions,
            repair_operators=["SelReplace", "PreInsert", "ArgCorrect"],
            created_at_iso=str(row["created_at_iso"]),
            success_count=int(row.get("success_count", 0)),
            failure_count=int(row.get("failure_count", 0)),
        )
        logger.debug(
            f"ContractSkillStore.find_matching_skill | found skill_id={skill.skill_id[:12]} "
            f"distance={distance:.4f}"
        )
        return skill


__all__ = [
    "ContractStep",
    "ContractSkill",
    "PostconditionList",
    "ContractSkillCompiler",
    "ContractSkillStore",
]
