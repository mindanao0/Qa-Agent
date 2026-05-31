"""
Episodic memory store (Sprint 2 / Cluster S2-C).

Append-only LanceDB-backed store for two new tables:
  * healed_experiences — durable record of healing wins
  * post_mortems       — durable record of healing failures

Both tables live in the SAME LanceDB database as the existing `qa_docs`
table managed by `src/rag/store.py` (default `~/.qa-agent/vector_db`).
Embeddings are 768-dim and are produced by an injected async callable
(typically `OllamaAdapter().embed`) so this module never imports the
LLM adapter at module-load time.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import lancedb
import pyarrow as pa
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DIM = 768
_HEALED_TABLE = "healed_experiences"
_POSTMORTEM_TABLE = "post_mortems"
_DEFAULT_DB_PATH = Path.home() / ".qa-agent" / "vector_db"

EmbeddingFn = Callable[[str], Awaitable[list[float]]]


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic V2 schemas (extra="forbid" — no silent field drift)
# ─────────────────────────────────────────────────────────────────────────────


class HealedExperience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    session_id: str
    domain: str
    page_url: str
    page_type: str
    failure_signature: str
    bad_strategy: str
    winning_strategy: str
    root_cause: str  # <=100 chars
    confidence: float = 0.7
    impact_score: float
    created_at_iso: str
    vector: list[float] = Field(default_factory=list)


class PostMortem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    session_id: str
    attempt_id: int
    domain: str
    page_url: str
    failure_signature: str
    current_plan: str
    locator_candidates: list[str]
    error_message: str
    dom_snapshot_ref: str  # FILE PATH, not full DOM
    confidence: float = 0.0
    impact_score: float
    created_at_iso: str
    vector: list[float] = Field(default_factory=list)


class MemoryHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_type: Literal["healed_experience", "post_mortem"]
    failure_signature: str
    strategy: str
    root_cause: str
    confidence: float
    impact_score: float


class LocatorPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    failure_signature: str
    preferred_strategies: list[str]
    avoid_strategies: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.now(timezone.utc)


def _recency_factor(created_at_iso: str, *, half_life_days: float = 30.0) -> float:
    """Linear decay over ``half_life_days``; clamped to [0, 1]."""
    created = _parse_iso(created_at_iso)
    now = datetime.now(timezone.utc)
    days = max(0.0, (now - created).total_seconds() / 86400.0)
    return max(0.0, 1.0 - days / half_life_days)


def _composite_score(confidence: float, impact: float, created_at_iso: str) -> float:
    return (
        0.4 * float(confidence)
        + 0.3 * float(impact)
        + 0.3 * _recency_factor(created_at_iso)
    )


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


class EpisodicStore:
    """Append-only episodic memory backed by LanceDB."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_fn: EmbeddingFn | None = None,
    ) -> None:
        self._db_path: Path = (db_path or _DEFAULT_DB_PATH).expanduser()
        self._embedding_fn: EmbeddingFn | None = embedding_fn
        self._db: Any = None
        self._healed: Any = None
        self._post_mortems: Any = None

    # ── Schemas ────────────────────────────────────────────────────────────

    def _healed_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("domain", pa.string()),
            pa.field("page_url", pa.string()),
            pa.field("page_type", pa.string()),
            pa.field("failure_signature", pa.string()),
            pa.field("bad_strategy", pa.string()),
            pa.field("winning_strategy", pa.string()),
            pa.field("root_cause", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("impact_score", pa.float32()),
            pa.field("created_at_iso", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])

    def _post_mortem_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("memory_id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("attempt_id", pa.int32()),
            pa.field("domain", pa.string()),
            pa.field("page_url", pa.string()),
            pa.field("failure_signature", pa.string()),
            pa.field("current_plan", pa.string()),
            pa.field("locator_candidates", pa.list_(pa.string())),
            pa.field("error_message", pa.string()),
            pa.field("dom_snapshot_ref", pa.string()),
            pa.field("confidence", pa.float32()),
            pa.field("impact_score", pa.float32()),
            pa.field("created_at_iso", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> "EpisodicStore":
        """Open the LanceDB database and create the two tables if missing."""
        self._db_path.mkdir(parents=True, exist_ok=True)

        def _open() -> tuple[Any, Any, Any]:
            db = lancedb.connect(str(self._db_path))
            names = set(db.table_names())
            if _HEALED_TABLE in names:
                healed = db.open_table(_HEALED_TABLE)
            else:
                healed = db.create_table(_HEALED_TABLE, schema=self._healed_schema())
            if _POSTMORTEM_TABLE in names:
                post_mortems = db.open_table(_POSTMORTEM_TABLE)
            else:
                post_mortems = db.create_table(
                    _POSTMORTEM_TABLE, schema=self._post_mortem_schema()
                )
            return db, healed, post_mortems

        self._db, self._healed, self._post_mortems = await asyncio.to_thread(_open)
        logger.info(
            f"EpisodicStore connected | path={str(self._db_path)!r} "
            f"tables=({_HEALED_TABLE},{_POSTMORTEM_TABLE})"
        )
        return self

    async def close(self) -> None:
        self._healed = None
        self._post_mortems = None
        self._db = None

    async def __aenter__(self) -> "EpisodicStore":
        return await self.connect()

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Embedding helper ───────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        if self._embedding_fn is None:
            raise RuntimeError(
                "EpisodicStore: no embedding_fn configured; pass one to the constructor."
            )
        vec = await self._embedding_fn(text)
        if len(vec) != EMBEDDING_DIM:
            raise ValueError(
                f"EpisodicStore: embedding has dim={len(vec)}, expected {EMBEDDING_DIM}"
            )
        return [float(v) for v in vec]

    # ── Save (APPEND-ONLY) ─────────────────────────────────────────────────

    async def save_healed_experience(self, exp: HealedExperience) -> str:
        assert self._healed is not None, "EpisodicStore not connected"
        if not exp.created_at_iso:
            exp = exp.model_copy(update={"created_at_iso": _now_iso()})
        if not exp.vector:
            text = (
                f"{exp.failure_signature}\n{exp.winning_strategy}\n"
                f"{exp.root_cause}\n{exp.page_url}"
            )
            vec = await self._embed(text)
            exp = exp.model_copy(update={"vector": vec})

        record = {
            "memory_id": exp.memory_id,
            "session_id": exp.session_id,
            "domain": exp.domain,
            "page_url": exp.page_url,
            "page_type": exp.page_type,
            "failure_signature": exp.failure_signature,
            "bad_strategy": exp.bad_strategy,
            "winning_strategy": exp.winning_strategy,
            "root_cause": exp.root_cause[:100],
            "confidence": float(exp.confidence),
            "impact_score": float(exp.impact_score),
            "created_at_iso": exp.created_at_iso,
            "embedding": list(exp.vector),
        }

        await asyncio.to_thread(self._healed.add, [record])
        logger.debug(
            f"EpisodicStore.save_healed_experience | id={exp.memory_id} "
            f"domain={exp.domain} sig={exp.failure_signature}"
        )
        return exp.memory_id

    async def save_post_mortem(self, pm: PostMortem) -> str:
        assert self._post_mortems is not None, "EpisodicStore not connected"
        if not pm.created_at_iso:
            pm = pm.model_copy(update={"created_at_iso": _now_iso()})
        if not pm.vector:
            text = (
                f"{pm.failure_signature}\n{pm.current_plan}\n"
                f"{pm.error_message}\n{pm.page_url}"
            )
            vec = await self._embed(text)
            pm = pm.model_copy(update={"vector": vec})

        record = {
            "memory_id": pm.memory_id,
            "session_id": pm.session_id,
            "attempt_id": int(pm.attempt_id),
            "domain": pm.domain,
            "page_url": pm.page_url,
            "failure_signature": pm.failure_signature,
            "current_plan": pm.current_plan,
            "locator_candidates": list(pm.locator_candidates),
            "error_message": pm.error_message,
            "dom_snapshot_ref": pm.dom_snapshot_ref,
            "confidence": float(pm.confidence),
            "impact_score": float(pm.impact_score),
            "created_at_iso": pm.created_at_iso,
            "embedding": list(pm.vector),
        }

        await asyncio.to_thread(self._post_mortems.add, [record])
        logger.debug(
            f"EpisodicStore.save_post_mortem | id={pm.memory_id} "
            f"domain={pm.domain} sig={pm.failure_signature}"
        )
        return pm.memory_id

    # ── Retrieval ──────────────────────────────────────────────────────────

    @staticmethod
    def _where_clause(domain: str, failure_signature: str) -> str:
        d = _sql_escape(domain)
        if failure_signature:
            s = _sql_escape(failure_signature)
            return f"domain = '{d}' AND failure_signature = '{s}'"
        return f"domain = '{d}'"

    def _search_table(
        self,
        table: Any,
        query_embedding: list[float],
        where: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            return (
                table.search(query_embedding)
                .where(where, prefilter=True)
                .limit(limit)
                .to_list()
            )
        except Exception as exc:
            logger.warning(f"EpisodicStore vector+filter search failed: {exc}")
            # Fall back to a plain filtered scan so we still return hits.
            try:
                rows = table.to_arrow().to_pylist()
                # Crude predicate evaluation since LanceDB couldn't apply the filter.
                out: list[dict[str, Any]] = []
                for r in rows:
                    if where == "":
                        out.append(r)
                        continue
                    if self._row_matches(r, where):
                        out.append(r)
                    if len(out) >= limit:
                        break
                return out
            except Exception as exc2:  # pragma: no cover — defensive
                logger.error(f"EpisodicStore fallback scan failed: {exc2}")
                return []

    @staticmethod
    def _row_matches(row: dict[str, Any], where: str) -> bool:
        """Tiny matcher for our very constrained where clauses."""
        parts = [p.strip() for p in where.split(" AND ")]
        for p in parts:
            # form: <col> = '<val>'
            try:
                col, _, val = p.partition("=")
                col = col.strip()
                val = val.strip().strip("'").replace("''", "'")
                if str(row.get(col, "")) != val:
                    return False
            except Exception:
                return False
        return True

    async def retrieve_for_planning(
        self,
        query_text: str,
        domain: str,
        failure_signature: str,
        limit: int = 5,
    ) -> list[MemoryHit]:
        """Hybrid (vector + structured) retrieval across BOTH episodic tables."""
        assert self._healed is not None and self._post_mortems is not None, (
            "EpisodicStore not connected"
        )

        query_embedding = await self._embed(query_text)
        where = self._where_clause(domain, failure_signature)
        candidate_limit = max(limit * 4, limit)

        def _both_searches() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            healed_rows = self._search_table(
                self._healed, query_embedding, where, candidate_limit
            )
            pm_rows = self._search_table(
                self._post_mortems, query_embedding, where, candidate_limit
            )
            return healed_rows, pm_rows

        healed_rows, pm_rows = await asyncio.to_thread(_both_searches)

        scored: list[tuple[float, MemoryHit]] = []

        for r in healed_rows:
            score = _composite_score(
                r.get("confidence", 0.0),
                r.get("impact_score", 0.0),
                str(r.get("created_at_iso", "")),
            )
            scored.append((
                score,
                MemoryHit(
                    memory_type="healed_experience",
                    failure_signature=str(r.get("failure_signature", "")),
                    strategy=str(r.get("winning_strategy", "")),
                    root_cause=str(r.get("root_cause", "")),
                    confidence=float(r.get("confidence", 0.0)),
                    impact_score=float(r.get("impact_score", 0.0)),
                ),
            ))

        for r in pm_rows:
            score = _composite_score(
                r.get("confidence", 0.0),
                r.get("impact_score", 0.0),
                str(r.get("created_at_iso", "")),
            )
            cands = list(r.get("locator_candidates") or [])
            strategy = ";".join(cands[:3])
            scored.append((
                score,
                MemoryHit(
                    memory_type="post_mortem",
                    failure_signature=str(r.get("failure_signature", "")),
                    strategy=strategy,
                    root_cause=str(r.get("error_message", ""))[:100],
                    confidence=float(r.get("confidence", 0.0)),
                    impact_score=float(r.get("impact_score", 0.0)),
                ),
            ))

        scored.sort(key=lambda t: t[0], reverse=True)
        hits = [h for _, h in scored[:limit]]
        logger.debug(
            f"EpisodicStore.retrieve_for_planning | domain={domain} "
            f"sig={failure_signature} healed={len(healed_rows)} "
            f"pm={len(pm_rows)} returned={len(hits)}"
        )
        return hits

    # ── Policy evolution ───────────────────────────────────────────────────

    async def evolve_locator_policy(
        self,
        domain: str,
        failure_signature: str,
    ) -> LocatorPolicy:
        """Roll up healed wins and post-mortem failures into a policy."""
        assert self._healed is not None and self._post_mortems is not None, (
            "EpisodicStore not connected"
        )

        where = self._where_clause(domain, failure_signature)

        def _fetch() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            def _safe(table: Any) -> list[dict[str, Any]]:
                try:
                    return (
                        table.search()
                        .where(where, prefilter=True)
                        .limit(10_000)
                        .to_list()
                    )
                except Exception:
                    try:
                        rows = table.to_arrow().to_pylist()
                        return [r for r in rows if self._row_matches(r, where)]
                    except Exception:
                        return []

            return _safe(self._healed), _safe(self._post_mortems)

        try:
            healed_rows, pm_rows = await asyncio.to_thread(_fetch)
        except Exception as exc:
            logger.warning(f"EpisodicStore.evolve_locator_policy fetch failed: {exc}")
            healed_rows, pm_rows = [], []

        # preferred: top-3 winning_strategy by confidence*impact (dedup, keep best)
        best_for_strategy: dict[str, float] = {}
        for r in healed_rows:
            strat = str(r.get("winning_strategy", "")).strip()
            if not strat:
                continue
            score = float(r.get("confidence", 0.0)) * float(r.get("impact_score", 0.0))
            if score > best_for_strategy.get(strat, float("-inf")):
                best_for_strategy[strat] = score

        preferred = [
            strat
            for strat, _ in sorted(
                best_for_strategy.items(), key=lambda kv: kv[1], reverse=True
            )
        ][:3]

        # avoid: top-3 locator_candidates (flattened, deduped) by impact
        best_for_candidate: dict[str, float] = {}
        for r in pm_rows:
            impact = float(r.get("impact_score", 0.0))
            for cand in r.get("locator_candidates") or []:
                cand_s = str(cand).strip()
                if not cand_s:
                    continue
                if impact > best_for_candidate.get(cand_s, float("-inf")):
                    best_for_candidate[cand_s] = impact

        avoid = [
            cand
            for cand, _ in sorted(
                best_for_candidate.items(), key=lambda kv: kv[1], reverse=True
            )
        ][:3]

        return LocatorPolicy(
            domain=domain,
            failure_signature=failure_signature,
            preferred_strategies=preferred,
            avoid_strategies=avoid,
        )
