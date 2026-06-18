"""Planner example retriever (Phase-3 of the eval track).

Queries the LanceDB index built by eval/build_rag_index.py: given the current
page URL + element, embed "<url> <role> <name>" with Ollama all-minilm (384-dim,
all-MiniLM-L6-v2) and return the top-k most similar PAST quality=1.0 test cases
to inject into the planner prompt as references.

Fail-safe: any error (index missing, Ollama down) returns [] / "" so the planner
silently falls back to no-RAG behaviour — retrieval must never break planning.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

_EMBED_MODEL = "all-minilm"
_DB_PATH = str(Path.home() / ".qa-agent" / "planner_rag")
_TABLE = "planner_examples"


class PlannerExampleRetriever:
    def __init__(
        self,
        db_path: str = _DB_PATH,
        table: str = _TABLE,
        ollama_url: str | None = None,
        top_k: int = 3,
    ) -> None:
        self.db_path = db_path
        self.table_name = table
        self.ollama = (ollama_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.top_k = top_k
        self._table: Any = None
        self._client = httpx.AsyncClient(timeout=60.0)

    def _open(self) -> Any:
        if self._table is None:
            import lancedb
            db = lancedb.connect(self.db_path)
            if self.table_name not in db.table_names():
                raise FileNotFoundError(f"RAG table {self.table_name!r} not found at {self.db_path}")
            self._table = db.open_table(self.table_name)
        return self._table

    async def _embed(self, text: str) -> list[float]:
        r = await self._client.post(
            f"{self.ollama}/api/embed", json={"model": _EMBED_MODEL, "input": [text]}
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]

    async def retrieve(self, url: str, role: str, name: str, top_k: int | None = None) -> list[dict]:
        import asyncio
        k = top_k or self.top_k
        try:
            query = f"{url} {role} {name}"
            emb = await self._embed(query)
            tbl = self._open()
            rows = await asyncio.to_thread(lambda: tbl.search(emb).limit(k).to_list())
            out = []
            for row in rows:
                try:
                    meta = json.loads(row.get("metadata_json", "{}"))
                except Exception:
                    meta = {}
                out.append({"content": row.get("content", ""), "metadata": meta})
            return out
        except Exception as exc:  # noqa: BLE001 — retrieval must never break planning
            logger.warning(f"PlannerExampleRetriever.retrieve failed: {exc!r}")
            return []

    @staticmethod
    def format_references(hits: list[dict]) -> str:
        """Render retrieved test cases as a compact reference block for the prompt."""
        if not hits:
            return ""
        parts = ["ตัวอย่าง test case ที่เกี่ยวข้องกับ element/หน้านี้ (ใช้เป็นแนวทาง):"]
        for h in hits:
            content = h.get("content", "").strip()
            if content:
                parts.append(content)
        return "\n".join(parts)

    async def close(self) -> None:
        await self._client.aclose()


__all__ = ["PlannerExampleRetriever"]
