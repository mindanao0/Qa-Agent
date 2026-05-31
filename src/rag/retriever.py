"""
HybridRetriever — UPDATE 3 universal variant.

Search modes:
    * Hybrid: 70% semantic (cosine) + 30% BM25 keyword, merged via wRRF.
    * Source-filtered: restrict to one or more `source` values
      (e.g. "playwright_docs", "url_crawl", "document").
    * Domain-filtered: restrict to chunks whose metadata json mentions
      a particular domain key (best-effort substring match).

Two helper methods are also exposed for the planner / generator:
    * get_playwright_context(locator_type) — pull docs for a specific locator
      API (e.g. "get_by_role") to ground generated code.
    * get_page_context(url) — pull crawled pages matching *url* so the planner
      knows the actual page structure before writing a test plan.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterable

from loguru import logger
from rank_bm25 import BM25Okapi

from src.llm.adapter import OllamaAdapter
from .store import RAGChunk, VectorStore

_CHARS_PER_TOKEN = 4
_RRF_K = 60           # standard RRF constant
_BM25_TTL_SEC = 300   # rebuild BM25 index if corpus is older than 5 minutes


class HybridRetriever:
    """
    Hybrid semantic + BM25 retriever with optional source / domain filters.

    Backward-compatible: `retrieve(query)` continues to behave exactly as
    before.  `search()` is the new filterable entry point used by UPDATE 3
    callers (planner, generator, healer).
    """

    def __init__(
        self,
        store: VectorStore,
        adapter: OllamaAdapter,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        max_context_tokens: int = 2000,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.max_context_tokens = max_context_tokens

        self._bm25_corpus: list[RAGChunk] = []
        self._bm25_index: BM25Okapi | None = None
        self._bm25_built_at: float = 0.0
        self._bm25_lock: asyncio.Lock = asyncio.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Primary public API
    # ──────────────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        sources: list[str] | None = None,
        domain: str | None = None,
    ) -> list[RAGChunk]:
        """
        Hybrid search with optional filtering.

          * sources – restrict to RAGChunk.source matching any of the given
            doc_type values (e.g. ["playwright_docs", "url_crawl"]).
          * domain  – restrict to chunks whose metadata mentions the domain
            string (case-insensitive substring against metadata_json).

        Returns the top *top_k* chunks merged by wRRF.
        """
        if not query.strip():
            return []
        k = top_k or self.top_k

        query_embedding, _ = await asyncio.gather(
            self.adapter.embed(query),
            self._ensure_bm25_index(),
        )

        # Pull a wider candidate pool from each search, then filter & merge.
        semantic_pool = await self.store.search(query_embedding, top_k=k * 4)
        bm25_pool = self._bm25_search(query, top_k=k * 4)

        if sources:
            sources_set = set(sources)
            semantic_pool = [c for c in semantic_pool if c.doc_type in sources_set]
            bm25_pool = [c for c in bm25_pool if c.doc_type in sources_set]

        if domain:
            needle = domain.lower()
            def _has_domain(chunk: RAGChunk) -> bool:
                # Compare structured metadata first, then fall back to a
                # substring scan of the metadata blob for resilience.
                if chunk.metadata.get("domain", "").lower() == needle:
                    return True
                # serialised metadata may include 'domain' key with the value
                # embedded in url_crawl/document chunks
                values = " ".join(str(v) for v in chunk.metadata.values()).lower()
                return needle in values

            semantic_pool = [c for c in semantic_pool if _has_domain(c)]
            bm25_pool = [c for c in bm25_pool if _has_domain(c)]

        merged = self._wrrf(semantic_pool, bm25_pool)
        selected = merged[:k]

        logger.debug(
            f"HybridRetriever.search | q_len={len(query)} "
            f"sources={sources} domain={domain} "
            f"semantic={len(semantic_pool)} bm25={len(bm25_pool)} "
            f"merged={len(merged)} returned={len(selected)}"
        )
        return selected

    async def retrieve(self, query: str) -> list[RAGChunk]:
        """Back-compat wrapper around search() with no filters."""
        return await self.search(query)

    async def refresh_bm25_index(self) -> None:
        async with self._bm25_lock:
            await self._rebuild_bm25()

    def format_context(
        self,
        chunks: list[RAGChunk],
        max_tokens: int | None = None,
    ) -> str:
        budget_chars = (max_tokens or self.max_context_tokens) * _CHARS_PER_TOKEN
        parts: list[str] = []
        used = 0
        for chunk in chunks:
            entry = f"[Source: {chunk.source} | type: {chunk.doc_type}]\n{chunk.content}"
            if used + len(entry) > budget_chars:
                remaining = budget_chars - used
                if remaining > 100:
                    parts.append(entry[:remaining] + "…")
                break
            parts.append(entry)
            used += len(entry)
        return "\n\n---\n\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # UPDATE 3 helpers — context shortcuts for the planner / generator
    # ──────────────────────────────────────────────────────────────────────────

    async def get_playwright_context(
        self,
        locator_type: str,
        top_k: int = 3,
    ) -> str:
        """
        Return playwright_docs chunks relevant to *locator_type* (e.g.
        "get_by_role", "expect", "auth").  Used by the generator to ground
        its code in accurate API usage.
        """
        if not locator_type:
            return ""
        chunks = await self.search(
            query=f"Playwright {locator_type} usage example",
            top_k=top_k,
            sources=["playwright_docs"],
        )
        return self.format_context(chunks)

    async def get_page_context(
        self,
        url: str,
        top_k: int = 3,
    ) -> str:
        """
        Return previously-crawled page chunks matching *url* — gives the
        planner the AxTree + interactive-element summary for the actual app
        before it drafts the test plan.
        """
        if not url:
            return ""
        chunks = await self.search(
            query=f"page structure for {url}",
            top_k=top_k,
            sources=["url_crawl"],
        )
        # Boost exact-URL matches if any survived the merge — the planner
        # cares about *this* page, not just topical similarity.
        url_lower = url.rstrip("/").lower()
        prioritised: list[RAGChunk] = []
        rest: list[RAGChunk] = []
        for chunk in chunks:
            chunk_url = chunk.metadata.get("url", chunk.source).rstrip("/").lower()
            if chunk_url == url_lower:
                prioritised.append(chunk)
            else:
                rest.append(chunk)
        return self.format_context(prioritised + rest)

    # ──────────────────────────────────────────────────────────────────────────
    # BM25 internals
    # ──────────────────────────────────────────────────────────────────────────

    async def _ensure_bm25_index(self) -> None:
        if (
            self._bm25_index is not None
            and time.monotonic() - self._bm25_built_at < _BM25_TTL_SEC
        ):
            return
        async with self._bm25_lock:
            if (
                self._bm25_index is not None
                and time.monotonic() - self._bm25_built_at < _BM25_TTL_SEC
            ):
                return
            await self._rebuild_bm25()

    async def _rebuild_bm25(self) -> None:
        all_chunks = await self.store.get_all_chunks()
        if not all_chunks:
            logger.debug("BM25 rebuild skipped — store is empty")
            return
        self._bm25_corpus = all_chunks
        tokenized = [_tokenize(c.content) for c in all_chunks]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_built_at = time.monotonic()
        logger.debug(f"BM25 index rebuilt | corpus_size={len(all_chunks)}")

    def _bm25_search(self, query: str, top_k: int) -> list[RAGChunk]:
        if not self._bm25_index or not self._bm25_corpus:
            return []
        query_tokens = _tokenize(query)
        scores = self._bm25_index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            self._bm25_corpus[i]
            for i, score in ranked[:top_k]
            if score > 0.0
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Weighted Reciprocal Rank Fusion
    # ──────────────────────────────────────────────────────────────────────────

    def _wrrf(
        self,
        semantic: Iterable[RAGChunk],
        bm25: Iterable[RAGChunk],
    ) -> list[RAGChunk]:
        rrf_scores: dict[str, float] = {}
        id_to_chunk: dict[str, RAGChunk] = {}

        for rank, chunk in enumerate(semantic):
            rrf_scores[chunk.id] = (
                rrf_scores.get(chunk.id, 0.0)
                + self.semantic_weight / (_RRF_K + rank + 1)
            )
            id_to_chunk[chunk.id] = chunk

        for rank, chunk in enumerate(bm25):
            rrf_scores[chunk.id] = (
                rrf_scores.get(chunk.id, 0.0)
                + self.bm25_weight / (_RRF_K + rank + 1)
            )
            id_to_chunk[chunk.id] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [id_to_chunk[doc_id] for doc_id in sorted_ids]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokeniser suitable for BM25."""
    import re
    return re.findall(r"\w+", text.lower())
