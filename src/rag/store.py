import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from loguru import logger

# nomic-embed-text produces 768-dim vectors
EMBEDDING_DIM = 768
_TABLE_NAME = "qa_docs"
_DEFAULT_DB_PATH = "~/.qa-agent/vector_db"


@dataclass
class RAGChunk:
    """Single chunk of retrieved context with its embedding."""

    id: str
    content: str
    embedding: list[float]
    source: str
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ── Serialisation helpers ──────────────────────────────────────────────────

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "source": self.source,
            "doc_type": self.doc_type,
            "metadata_json": json.dumps(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RAGChunk":
        raw_meta = record.get("metadata_json", "{}")
        try:
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else {}
        except Exception:
            meta = {}
        return cls(
            id=str(record["id"]),
            content=str(record["content"]),
            embedding=list(record.get("embedding", [])),
            source=str(record.get("source", "")),
            doc_type=str(record.get("doc_type", "text")),
            metadata=meta,
            created_at=str(record.get("created_at", "")),
        )

    @staticmethod
    def make_id(content: str, source: str) -> str:
        """Deterministic SHA-256 content-address ID (first 16 hex chars)."""
        digest = hashlib.sha256(f"{source}::{content}".encode()).hexdigest()
        return digest[:16]


class VectorStore:
    """
    LanceDB-backed vector store (embedded mode — no server required).

    All public methods are async; LanceDB sync operations are dispatched
    to a thread pool via asyncio.to_thread() to keep the event loop free.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        embedding_dim: int = EMBEDDING_DIM,
        table_name: str = _TABLE_NAME,
    ) -> None:
        self._db_path = os.path.expanduser(db_path)
        self._embedding_dim = embedding_dim
        self._table_name = table_name
        self._db: Any = None
        self._table: Any = None

    # ── Schema ────────────────────────────────────────────────────────────────

    def _schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), self._embedding_dim)),
            pa.field("source", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("created_at", pa.string()),
        ])

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> "VectorStore":
        """Open (or create) the LanceDB database and the qa_docs table."""
        Path(self._db_path).mkdir(parents=True, exist_ok=True)

        def _open():
            db = lancedb.connect(self._db_path)
            if self._table_name in db.table_names():
                return db, db.open_table(self._table_name)
            tbl = db.create_table(self._table_name, schema=self._schema())
            return db, tbl

        self._db, self._table = await asyncio.to_thread(_open)
        logger.info(
            f"VectorStore connected | path={self._db_path!r} "
            f"table={self._table_name!r} dim={self._embedding_dim}"
        )
        return self

    async def close(self) -> None:
        self._table = None
        self._db = None

    async def __aenter__(self) -> "VectorStore":
        return await self.connect()

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Write ──────────────────────────────────────────────────────────────────

    async def upsert(self, chunks: list[RAGChunk]) -> None:
        """Batch-insert chunks, replacing any existing rows with matching IDs."""
        if not chunks:
            return

        assert self._table is not None, "VectorStore not connected"

        ids = [c.id for c in chunks]
        existing = await self._existing_ids(ids)

        new_chunks = [c for c in chunks if c.id not in existing]
        update_chunks = [c for c in chunks if c.id in existing]

        def _write():
            if new_chunks:
                self._table.add([c.to_record() for c in new_chunks])
            for chunk in update_chunks:
                self._table.delete(f"id = '{chunk.id}'")
                self._table.add([chunk.to_record()])

        await asyncio.to_thread(_write)
        logger.debug(
            f"VectorStore.upsert | new={len(new_chunks)} updated={len(update_chunks)}"
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RAGChunk]:
        """Cosine similarity vector search — returns top_k chunks."""
        assert self._table is not None, "VectorStore not connected"

        def _search() -> list[dict[str, Any]]:
            return (
                self._table.search(query_embedding)
                .limit(top_k)
                .to_list()
            )

        try:
            records = await asyncio.to_thread(_search)
            return [RAGChunk.from_record(r) for r in records]
        except Exception as exc:
            logger.error(f"VectorStore.search failed: {exc}")
            return []

    async def get_all_chunks(self) -> list[RAGChunk]:
        """Fetch every chunk — used to build the BM25 corpus."""
        assert self._table is not None, "VectorStore not connected"

        def _fetch() -> list[dict[str, Any]]:
            # to_arrow() avoids the pandas dependency that to_pandas() carries.
            return self._table.to_arrow().to_pylist()

        try:
            records = await asyncio.to_thread(_fetch)
            return [RAGChunk.from_record(r) for r in records]
        except Exception as exc:
            logger.error(f"VectorStore.get_all_chunks failed: {exc}")
            return []

    async def count(self) -> int:
        assert self._table is not None, "VectorStore not connected"
        def _cnt() -> int:
            return self._table.count_rows()
        return await asyncio.to_thread(_cnt)

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        quoted = ", ".join(f"'{i}'" for i in ids)
        def _query() -> list[dict]:
            return (
                self._table.search()
                .where(f"id IN ({quoted})", prefilter=True)
                .select(["id"])
                .limit(len(ids))
                .to_list()
            )
        try:
            rows = await asyncio.to_thread(_query)
            return {r["id"] for r in rows}
        except Exception:
            # If filter fails (e.g. empty table), treat as no existing ids
            return set()
