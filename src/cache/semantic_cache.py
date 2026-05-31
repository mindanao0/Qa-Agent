"""Semantic Cache (Priority 3) — dual-threshold LanceDB-backed cache.

Implements the routing contract:

    s >= 0.95           → CACHE_HIT   (return stored playwright code; no LLM call)
    0.50 <= s < 0.95    → GUIDED      (inject cached entry as few-shot context)
    s < 0.50            → CACHE_MISS  (cold LLM inference)

Embeddings: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, CPU, normalized).
Vector store: LanceDB embedded with an IVF_HNSW_SQ cosine index, rebuilt every
``index_rebuild_interval`` upserts with ``num_partitions = sqrt(N)``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TABLE_NAME = "semantic_cache"
DEFAULT_DB_PATH = Path("data/semantic_cache.lance")

CacheRoute = Literal["CACHE_HIT", "GUIDED", "CACHE_MISS"]
IndexHealth = Literal["INDEXED", "UNINDEXED", "NEEDS_REBUILD"]


# ─── Embedding singleton ──────────────────────────────────────────────────────

_model_lock = threading.Lock()
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Return the process-wide SentenceTransformer instance (lazy, thread-safe)."""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                logger.info("Loading embedding model %s", EMBEDDING_MODEL_NAME)
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# ─── Schemas ──────────────────────────────────────────────────────────────────


class _CacheRow(LanceModel):
    """LanceDB-typed row schema (mirrors CacheEntry)."""

    cache_id: str
    query_text: str
    query_vector: Vector(EMBEDDING_DIM)  # type: ignore[valid-type]
    playwright_code: str
    hit_count: int
    confidence_score: float
    created_at: str
    updated_at: str
    version: int
    metadata: str


class CacheEntry(BaseModel):
    """A single semantic-cache row."""

    cache_id: str
    query_text: str
    query_vector: list[float]
    playwright_code: str
    hit_count: int = 0
    confidence_score: float = 0.0
    created_at: str
    updated_at: str
    version: int = 1
    metadata: str = "{}"


class CacheLookupResult(BaseModel):
    """Result of a single ``SemanticCache.lookup`` call."""

    route: CacheRoute
    cached_entry: Optional[CacheEntry] = None
    score: float
    query_vector: list[float]
    latency_ms: float = 0.0


class CacheStats(BaseModel):
    """Aggregate statistics about the cache."""

    total_entries: int
    total_hits: int
    avg_confidence: float
    index_health: IndexHealth


# ─── Helpers ──────────────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_query(query: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    return _WHITESPACE_RE.sub(" ", query.strip().lower())


def _hash_query(normalized_query: str) -> str:
    return hashlib.md5(normalized_query.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_entry(row: dict[str, Any]) -> CacheEntry:
    vec = row.get("query_vector", [])
    return CacheEntry(
        cache_id=str(row["cache_id"]),
        query_text=str(row["query_text"]),
        query_vector=[float(x) for x in vec],
        playwright_code=str(row["playwright_code"]),
        hit_count=int(row.get("hit_count", 0)),
        confidence_score=float(row.get("confidence_score", 0.0)),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
        version=int(row.get("version", 1)),
        metadata=str(row.get("metadata", "{}")),
    )


def _entry_to_record(entry: CacheEntry) -> dict[str, Any]:
    return {
        "cache_id": entry.cache_id,
        "query_text": entry.query_text,
        "query_vector": entry.query_vector,
        "playwright_code": entry.playwright_code,
        "hit_count": entry.hit_count,
        "confidence_score": entry.confidence_score,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "version": entry.version,
        "metadata": entry.metadata,
    }


# ─── Main class ───────────────────────────────────────────────────────────────


class SemanticCache:
    """Dual-threshold semantic cache backed by LanceDB.

    All public methods are ``async``. LanceDB and sentence-transformers are
    synchronous; their calls are dispatched via ``asyncio.to_thread`` so the
    event loop stays responsive.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        hit_threshold: float = 0.95,
        guided_threshold: float = 0.50,
        index_rebuild_interval: int = 50,
    ) -> None:
        self.db_path = Path(db_path)
        self.hit_threshold = hit_threshold
        self.guided_threshold = guided_threshold
        self.index_rebuild_interval = index_rebuild_interval

        self._db: Any = None
        self._table: Any = None
        self._upsert_counter: int = 0
        self._write_lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open the LanceDB database and prepare the cache table.

        Creates the parent directory and the typed table if missing, builds the
        IVF_HNSW_SQ index when there are enough rows to make indexing useful,
        and seeds the upsert counter from the current row count.

        Returns:
            None.
        """

        def _open() -> tuple[Any, Any, int]:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(str(self.db_path))
            if TABLE_NAME in db.list_tables():
                tbl = db.open_table(TABLE_NAME)
            else:
                tbl = db.create_table(TABLE_NAME, schema=_CacheRow)
            return db, tbl, tbl.count_rows()

        self._db, self._table, row_count = await asyncio.to_thread(_open)
        self._upsert_counter = row_count % self.index_rebuild_interval

        # Warm up the embedding model in a worker thread so the first lookup
        # isn't slowed by model loading.
        await asyncio.to_thread(_get_embedding_model)

        if row_count > 10:
            await self._rebuild_index()

        logger.info(
            "SemanticCache initialized | path=%s rows=%d", self.db_path, row_count
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def lookup(self, query: str) -> CacheLookupResult:
        """Look up the best-matching cache row for ``query``.

        Args:
            query: Natural-language test intent.

        Returns:
            A :class:`CacheLookupResult` describing the routing decision, the
            matched entry (or ``None`` on miss), the cosine similarity score,
            the query embedding, and the wall-clock latency in milliseconds.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if self._table is None:
            raise RuntimeError("SemanticCache.initialize() must be called first")

        start = time.perf_counter()
        normalized = _normalize_query(query)
        query_vector = await self._encode(normalized)

        def _search() -> list[dict[str, Any]]:
            return self._table.search(query_vector).metric("cosine").limit(1).to_list()

        try:
            rows = await asyncio.to_thread(_search)
        except Exception as exc:  # pragma: no cover — empty table / unindexed corner
            logger.debug("SemanticCache.lookup search raised %s", exc)
            rows = []

        latency_ms = (time.perf_counter() - start) * 1000.0

        if not rows:
            return CacheLookupResult(
                route="CACHE_MISS",
                cached_entry=None,
                score=0.0,
                query_vector=query_vector,
                latency_ms=latency_ms,
            )

        row = rows[0]
        distance = float(row.get("_distance", 1.0))
        score = 1.0 - distance
        entry = _row_to_entry(row)

        if score >= self.hit_threshold:
            route: CacheRoute = "CACHE_HIT"
        elif score >= self.guided_threshold:
            route = "GUIDED"
        else:
            route = "CACHE_MISS"

        if route == "CACHE_HIT":
            await self._record_hit(entry, score)
            entry.hit_count += 1
            entry.confidence_score = score

        logger.debug(
            "SemanticCache.lookup | route=%s score=%.4f latency_ms=%.2f",
            route,
            score,
            latency_ms,
        )

        return CacheLookupResult(
            route=route,
            cached_entry=entry if route != "CACHE_MISS" else None,
            score=score,
            query_vector=query_vector,
            latency_ms=latency_ms,
        )

    async def upsert(
        self,
        query: str,
        playwright_code: str,
        metadata: dict[str, Any],
        existing_entry: Optional[CacheEntry] = None,
    ) -> CacheEntry:
        """Insert or update a cache entry for ``query`` / ``playwright_code``.

        Args:
            query: The natural-language intent associated with this code.
            playwright_code: Deterministic Playwright code produced by the LLM.
            metadata: Arbitrary structured context (source URL, test_type, agent_mode).
            existing_entry: When provided, increments version + hit_count instead
                of starting a fresh row.

        Returns:
            The :class:`CacheEntry` that was written to LanceDB.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if self._table is None:
            raise RuntimeError("SemanticCache.initialize() must be called first")

        normalized = _normalize_query(query)
        cache_id = (
            existing_entry.cache_id if existing_entry else _hash_query(normalized)
        )
        query_vector = (
            existing_entry.query_vector
            if existing_entry
            else await self._encode(normalized)
        )
        now = _now_iso()

        if existing_entry is not None:
            entry = CacheEntry(
                cache_id=cache_id,
                query_text=existing_entry.query_text,
                query_vector=query_vector,
                playwright_code=playwright_code,
                hit_count=existing_entry.hit_count + 1,
                confidence_score=existing_entry.confidence_score,
                created_at=existing_entry.created_at,
                updated_at=now,
                version=existing_entry.version + 1,
                metadata=json.dumps(metadata),
            )
        else:
            entry = CacheEntry(
                cache_id=cache_id,
                query_text=normalized,
                query_vector=query_vector,
                playwright_code=playwright_code,
                hit_count=0,
                confidence_score=0.0,
                created_at=now,
                updated_at=now,
                version=1,
                metadata=json.dumps(metadata),
            )

        record = _entry_to_record(entry)

        async with self._write_lock:
            def _write() -> None:
                self._table.delete(f"cache_id = '{cache_id}'")
                self._table.add([record])

            await asyncio.to_thread(_write)
            self._upsert_counter += 1
            should_rebuild = self._upsert_counter >= self.index_rebuild_interval

        if should_rebuild:
            await self._rebuild_index()

        logger.info(
            "SemanticCache.upsert | cache_id=%s version=%d", cache_id, entry.version
        )
        return entry

    async def invalidate(self, cache_id: str) -> bool:
        """Hard-delete the row identified by ``cache_id``.

        Args:
            cache_id: MD5 identifier of the row to remove.

        Returns:
            ``True`` if a row was deleted, ``False`` if no such row existed.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called yet.
        """
        if self._table is None:
            raise RuntimeError("SemanticCache.initialize() must be called first")

        def _exists_and_delete() -> bool:
            existing = (
                self._table.search()
                .where(f"cache_id = '{cache_id}'", prefilter=True)
                .select(["cache_id"])
                .limit(1)
                .to_list()
            )
            if not existing:
                return False
            self._table.delete(f"cache_id = '{cache_id}'")
            return True

        deleted = await asyncio.to_thread(_exists_and_delete)
        if deleted:
            logger.info("SemanticCache.invalidate | cache_id=%s removed", cache_id)
        else:
            logger.debug("SemanticCache.invalidate | cache_id=%s not found", cache_id)
        return deleted

    async def get_stats(self) -> CacheStats:
        """Return aggregate cache statistics.

        Returns:
            :class:`CacheStats` with total rows, summed hit count, mean confidence
            score, and current index-health label.
        """
        if self._table is None:
            raise RuntimeError("SemanticCache.initialize() must be called first")

        def _collect() -> tuple[int, int, float, IndexHealth]:
            rows = self._table.to_arrow().to_pylist()
            total = len(rows)
            total_hits = sum(int(r.get("hit_count", 0)) for r in rows)
            if total:
                avg_conf = sum(float(r.get("confidence_score", 0.0)) for r in rows) / total
            else:
                avg_conf = 0.0

            health: IndexHealth
            if total <= 10:
                health = "UNINDEXED"
            elif self._upsert_counter >= self.index_rebuild_interval:
                health = "NEEDS_REBUILD"
            else:
                try:
                    indices = self._table.list_indices()
                    health = "INDEXED" if indices else "UNINDEXED"
                except Exception:
                    health = "UNINDEXED"
            return total, total_hits, avg_conf, health

        total, total_hits, avg_conf, health = await asyncio.to_thread(_collect)
        return CacheStats(
            total_entries=total,
            total_hits=total_hits,
            avg_confidence=avg_conf,
            index_health=health,
        )

    # ── Internals ────────────────────────────────────────────────────────────

    async def _encode(self, normalized_query: str) -> list[float]:
        """Encode a normalized query into a 384-dim unit vector."""

        def _run() -> list[float]:
            model = _get_embedding_model()
            vec = model.encode(
                normalized_query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return [float(x) for x in vec.tolist()]

        return await asyncio.to_thread(_run)

    async def _record_hit(self, entry: CacheEntry, score: float) -> None:
        """Persist hit_count + confidence_score updates after a CACHE_HIT."""
        updated = CacheEntry(
            cache_id=entry.cache_id,
            query_text=entry.query_text,
            query_vector=entry.query_vector,
            playwright_code=entry.playwright_code,
            hit_count=entry.hit_count + 1,
            confidence_score=score,
            created_at=entry.created_at,
            updated_at=_now_iso(),
            version=entry.version,
            metadata=entry.metadata,
        )
        record = _entry_to_record(updated)

        async with self._write_lock:
            def _write() -> None:
                self._table.delete(f"cache_id = '{entry.cache_id}'")
                self._table.add([record])

            try:
                await asyncio.to_thread(_write)
            except Exception as exc:  # pragma: no cover — write race
                logger.warning("SemanticCache._record_hit failed: %s", exc)

    async def _rebuild_index(self) -> None:
        """Rebuild the IVF_HNSW_SQ cosine index with sqrt(N) partitions."""

        def _build() -> int:
            row_count = self._table.count_rows()
            num_partitions = max(1, int(math.sqrt(row_count))) if row_count else 1
            try:
                self._table.create_index(
                    metric="cosine",
                    vector_column_name="query_vector",
                    index_type="IVF_HNSW_SQ",
                    num_partitions=num_partitions,
                    replace=True,
                )
            except TypeError:
                # Older LanceDB builds expose a different create_index signature.
                self._table.create_index(
                    metric="cosine",
                    vector_column_name="query_vector",
                    num_partitions=num_partitions,
                    replace=True,
                )
            return row_count

        try:
            row_count = await asyncio.to_thread(_build)
        except Exception as exc:
            logger.warning("SemanticCache._rebuild_index failed: %s", exc)
            return

        self._upsert_counter = 0
        logger.info("SemanticCache index rebuilt | rows=%d", row_count)
