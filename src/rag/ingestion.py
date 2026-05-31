"""
Universal RAG ingestion pipeline (UPDATE 3 specification).

Three ingestion sources, exposed under a single class and a unified
`ingest_all()` entry point:

  1. Playwright documentation ingester — fetches a curated list of public
     Playwright Python docs, converts to clean markdown, chunks by h2/h3
     headers, embeds, and upserts to LanceDB.

  2. URL crawler — drives a headless Playwright browser to discover internal
     links of a target app (depth-limited, same-domain only, capped at 50
     pages), extracts the AxTree + interactive elements + forms per page,
     and ingests them as structured markdown chunks.

  3. Document parser — accepts PDF, DOCX, MD, TXT, JSON, YAML. Each document
     is text-extracted, semantically chunked (~500 tokens per chunk, split on
     headers), embedded, and ingested.

After running, `ingest_all()` prints a rich summary table:

    | Source            | Type     | Chunks Added |
    | playwright_docs   | web      |          245 |
    | https://app.com   | crawl    |           38 |
    | requirements.pdf  | document |           22 |
    | Total             |          |          305 |

The DocumentIngester used by the original generic pipeline is preserved
below as a low-level helper used by all three sources.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.llm.adapter import OllamaAdapter
from src.browser.ax_extractor import extract_axtree, prune_axtree
from src.browser.manager import BrowserManager
from .store import RAGChunk, VectorStore

_CHARS_PER_TOKEN = 4  # rough approximation for English/code mixed content


# ──────────────────────────────────────────────────────────────────────────────
# Low-level DocumentIngester (preserved — used by every source below)
# ──────────────────────────────────────────────────────────────────────────────


class DocumentIngester:
    """
    Generic text → chunks → embeddings → VectorStore pipeline.

      1. Split on paragraph / sentence boundaries respecting chunk_size tokens.
      2. Apply chunk_overlap token prefix from the previous chunk for continuity.
      3. Embed each chunk via OllamaAdapter (nomic-embed-text).
      4. Upsert into VectorStore with deduplication.
    """

    def __init__(
        self,
        store: VectorStore,
        adapter: OllamaAdapter,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        batch_size: int = 8,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

    async def ingest(
        self,
        content: str,
        source: str,
        doc_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if not content.strip():
            logger.warning(f"ingest: empty content for source={source!r} — skipping")
            return 0

        raw_chunks = self._split(content)
        if not raw_chunks:
            return 0

        logger.debug(
            f"Ingesting source={source!r} | chunks={len(raw_chunks)} "
            f"doc_type={doc_type!r}"
        )

        rag_chunks: list[RAGChunk] = []
        for chunk_text in raw_chunks:
            try:
                embedding = await self.adapter.embed(chunk_text)
            except Exception as exc:
                logger.warning(f"Embed failed for chunk from {source!r}: {exc}")
                continue
            rag_chunks.append(
                RAGChunk(
                    id=RAGChunk.make_id(chunk_text, source),
                    content=chunk_text,
                    embedding=embedding,
                    source=source,
                    doc_type=doc_type,
                    metadata=metadata or {},
                )
            )

        if rag_chunks:
            await self.store.upsert(rag_chunks)
        return len(rag_chunks)

    async def ingest_file(
        self,
        path: str | Path,
        doc_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"ingest_file: {file_path} does not exist")
        content = file_path.read_text(encoding="utf-8", errors="replace")
        inferred_type = doc_type or _infer_doc_type(file_path.suffix)
        extra_meta = {"file_path": str(file_path), "file_name": file_path.name}
        merged_meta = {**(metadata or {}), **extra_meta}
        return await self.ingest(
            content=content,
            source=str(file_path),
            doc_type=inferred_type,
            metadata=merged_meta,
        )

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _split(self, text: str) -> list[str]:
        max_chars = self.chunk_size * _CHARS_PER_TOKEN
        overlap_chars = self.chunk_overlap * _CHARS_PER_TOKEN

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]

        flat_segments: list[str] = []
        for para in paragraphs:
            if len(para) > max_chars:
                flat_segments.extend(self._split_by_sentences(para, max_chars))
            else:
                flat_segments.append(para)

        chunks: list[str] = []
        current: str = ""
        for seg in flat_segments:
            if not current:
                current = seg
                continue
            candidate = current + "\n\n" + seg
            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)
                current = seg
        if current:
            chunks.append(current)

        return self._apply_overlap(chunks, overlap_chars)

    @staticmethod
    def _split_by_sentences(text: str, max_chars: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sent) > max_chars:
                    words = sent.split()
                    current = ""
                    for word in words:
                        test = (current + " " + word).strip() if current else word
                        if len(test) <= max_chars:
                            current = test
                        else:
                            if current:
                                chunks.append(current)
                            current = word
                else:
                    current = sent
        if current:
            chunks.append(current)
        return chunks or [text[:max_chars]]

    @staticmethod
    def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
        if len(chunks) <= 1 or overlap_chars == 0:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            prefix = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            result.append(prefix + "\n\n" + chunks[i])
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Universal RAG Ingestion Pipeline (UPDATE 3)
# ──────────────────────────────────────────────────────────────────────────────


_PLAYWRIGHT_DOC_URLS: list[str] = [
    "https://playwright.dev/python/docs/api/class-page",
    "https://playwright.dev/python/docs/api/class-locator",
    "https://playwright.dev/python/docs/api/class-expect",
    "https://playwright.dev/python/docs/locators",
    "https://playwright.dev/python/docs/best-practices",
    "https://playwright.dev/python/docs/assertions",
    "https://playwright.dev/python/docs/auth",
    "https://playwright.dev/python/docs/pages",
]

_CRAWL_MAX_PAGES_DEFAULT = 50
_CRAWL_DELAY_SEC = 1.0
_CRAWL_TIMEOUT_MS = 15_000

# File extensions excluded from URL discovery
_SKIPPABLE_LINK_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".webm",
    ".css", ".woff", ".woff2", ".ttf",
}


class RAGIngestionPipeline:
    """
    Universal pipeline that pulls knowledge from three different source types
    into a single LanceDB-backed vector store.

      pipeline = RAGIngestionPipeline(store, adapter)
      await pipeline.ingest_playwright_docs()
      await pipeline.ingest_url("https://your-app.com")
      await pipeline.ingest_document("path/to/spec.pdf")
      # or all at once:
      await pipeline.ingest_all(sources={
          "playwright_docs": True,
          "urls":            ["https://your-app.com"],
          "documents":       ["spec.pdf", "requirements.docx"],
      })
    """

    def __init__(
        self,
        store: VectorStore,
        adapter: OllamaAdapter,
        browser_manager: BrowserManager | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.browser_manager = browser_manager
        self.doc_ingester = DocumentIngester(
            store=store,
            adapter=adapter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # Sumary table data — populated during ingest_all() calls.
        self.summary: list[tuple[str, str, int]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # SOURCE 1 — Playwright documentation
    # ──────────────────────────────────────────────────────────────────────────

    async def ingest_playwright_docs(
        self,
        urls: list[str] | None = None,
    ) -> int:
        """
        Fetch Playwright documentation pages, parse them, chunk by h2/h3
        headings, and ingest into the vector store.

        Returns the total number of chunks added across all pages.
        """
        urls = urls or list(_PLAYWRIGHT_DOC_URLS)
        total_chunks = 0
        pages_processed = 0

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "qa-agent/0.1 RAG-ingester"},
        ) as client:
            for url in urls:
                try:
                    html = await self._fetch_with_retry(client, url)
                except Exception as exc:
                    logger.warning(f"playwright_docs: skip {url} — {exc}")
                    continue

                soup = BeautifulSoup(html, "html.parser")
                # Drop noise before extraction
                for tag in soup(["nav", "footer", "script", "style", "header"]):
                    tag.decompose()

                main = soup.select_one("article") or soup.select_one("main") or soup
                sections = _chunk_by_headers(main)
                if not sections:
                    logger.debug(f"playwright_docs: no sections extracted from {url}")
                    continue

                page_chunks = 0
                for section_heading, section_markdown in sections:
                    if not section_markdown.strip():
                        continue
                    page_chunks += await self.doc_ingester.ingest(
                        content=section_markdown,
                        source=url,
                        doc_type="playwright_docs",
                        metadata={
                            "url": url,
                            "section": section_heading,
                        },
                    )
                total_chunks += page_chunks
                pages_processed += 1
                logger.info(
                    f"playwright_docs: {url} → {page_chunks} chunks"
                )

        logger.info(
            f"Playwright docs ingested: {total_chunks} chunks "
            f"from {pages_processed} pages"
        )
        self.summary.append(("playwright_docs", "web", total_chunks))
        return total_chunks

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        retries: int = 3,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_exc = exc
                wait = min(8.0, 1.5 ** attempt)
                logger.debug(
                    f"_fetch_with_retry attempt {attempt}/{retries} for {url}: {exc} "
                    f"(retrying in {wait:.1f}s)"
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url}: {last_exc}")

    # ──────────────────────────────────────────────────────────────────────────
    # SOURCE 2 — URL crawler
    # ──────────────────────────────────────────────────────────────────────────

    async def ingest_url(
        self,
        url: str,
        depth: int = 2,
        max_pages: int = _CRAWL_MAX_PAGES_DEFAULT,
        domain_hint: str | None = None,
    ) -> int:
        """
        Crawl *url* (same-domain only, breadth-first, depth-limited) and
        ingest a structured AxTree-derived chunk for each visited page.

        Requires a BrowserManager — if one was not injected at construction,
        a temporary one is created for the duration of the crawl.
        """
        owns_bm = False
        if self.browser_manager is None:
            self.browser_manager = BrowserManager()
            await self.browser_manager.start()
            owns_bm = True

        total_chunks = 0
        visited: set[str] = set()
        try:
            origin = _normalise_origin(url)
            queue: list[tuple[str, int]] = [(url, 0)]
            async with self.browser_manager.new_page() as page:
                while queue and len(visited) < max_pages:
                    current_url, current_depth = queue.pop(0)
                    canon_url, _ = urldefrag(current_url)
                    if canon_url in visited:
                        continue
                    visited.add(canon_url)

                    try:
                        response = await page.goto(
                            canon_url,
                            wait_until="domcontentloaded",
                            timeout=_CRAWL_TIMEOUT_MS,
                        )
                    except Exception as exc:
                        logger.debug(f"crawl: skip {canon_url} ({exc})")
                        continue

                    status = response.status if response else 0
                    if status in (403, 404, 410):
                        logger.debug(f"crawl: skip {canon_url} (HTTP {status})")
                        continue

                    page_markdown = await self._extract_page_markdown(
                        page, canon_url, domain_hint
                    )
                    if page_markdown:
                        added = await self.doc_ingester.ingest(
                            content=page_markdown,
                            source=canon_url,
                            doc_type="url_crawl",
                            metadata={
                                "url": canon_url,
                                "domain": domain_hint or "",
                                "origin": origin,
                                "crawl_depth": current_depth,
                            },
                        )
                        total_chunks += added

                    logger.info(
                        f"Crawled {len(visited)}/{max_pages}: {canon_url}"
                    )

                    # Discover links for next depth (only if budget allows)
                    if current_depth < depth and len(visited) < max_pages:
                        new_links = await _discover_internal_links(page, origin)
                        for link in new_links:
                            link_canon, _ = urldefrag(link)
                            if link_canon and link_canon not in visited:
                                queue.append((link_canon, current_depth + 1))

                    # Polite delay between page loads
                    await asyncio.sleep(_CRAWL_DELAY_SEC)
        finally:
            if owns_bm and self.browser_manager:
                await self.browser_manager.stop()
                self.browser_manager = None

        logger.info(
            f"URL crawl complete | visited={len(visited)} chunks={total_chunks} root={url}"
        )
        self.summary.append((url, "crawl", total_chunks))
        return total_chunks

    async def _extract_page_markdown(
        self,
        page: Any,
        url: str,
        domain_hint: str | None,
    ) -> str:
        """
        Build a markdown chunk from a crawled page summarising AxTree,
        interactive elements, and forms.
        """
        try:
            title = await page.title()
        except Exception:
            title = ""

        meta_description = ""
        with suppress(Exception):
            meta_description = await page.evaluate(
                "() => {"
                "  const m = document.querySelector('meta[name=\"description\"]');"
                "  return m ? m.getAttribute('content') || '' : '';"
                "}"
            )

        interactive: list[str] = []
        with suppress(Exception):
            interactive = await page.evaluate(
                "() => Array.from(document.querySelectorAll("
                "  'button, a[href], [role=button], [role=link], [role=tab]"
                ", [role=menuitem]'"
                ")).slice(0, 40).map(el => {"
                "  const role = el.getAttribute('role') || el.tagName.toLowerCase();"
                "  const name = (el.innerText || el.getAttribute('aria-label') || "
                "    el.getAttribute('title') || '').trim().slice(0, 60);"
                "  return name ? `${role}: ${name}` : null;"
                "}).filter(Boolean)"
            )

        forms: list[str] = []
        with suppress(Exception):
            forms = await page.evaluate(
                "() => Array.from(document.querySelectorAll('input,select,textarea'))"
                ".slice(0, 30).map(el => {"
                "  const label = (el.closest('label') && el.closest('label').innerText) ||"
                "    document.querySelector(`label[for='${el.id}']`)?.innerText ||"
                "    el.getAttribute('placeholder') ||"
                "    el.getAttribute('aria-label') || el.name || '';"
                "  return label ? `${el.tagName.toLowerCase()}[${el.type||''}]: ${label.trim().slice(0,60)}` : null;"
                "}).filter(Boolean)"
            )

        raw_ax = await extract_axtree(page)
        pruned_ax = await prune_axtree(raw_ax, max_nodes=120) if raw_ax else ""

        parts: list[str] = []
        parts.append(f"# Page: {title or url}")
        parts.append(f"\n**URL**: {url}")
        if meta_description:
            parts.append(f"\n**Description**: {meta_description}")
        if domain_hint:
            parts.append(f"\n**Domain**: {domain_hint}")

        if interactive:
            parts.append("\n## Interactive elements\n")
            parts.extend(f"- {item}" for item in interactive)

        if forms:
            parts.append("\n## Form fields\n")
            parts.extend(f"- {item}" for item in forms)

        if pruned_ax:
            parts.append("\n## Accessibility tree (pruned)\n```\n" + pruned_ax + "\n```")

        return "\n".join(parts).strip()

    # ──────────────────────────────────────────────────────────────────────────
    # SOURCE 3 — document parser
    # ──────────────────────────────────────────────────────────────────────────

    async def ingest_document(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest a PDF / DOCX / MD / TXT / JSON / YAML file.

        Each format-specific extractor returns plain text or markdown; the
        result is then chunked, embedded, and upserted.
        """
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ingest_document: {path} does not exist")

        suffix = path.suffix.lower()
        extra_meta = {
            "file": str(path),
            "file_name": path.name,
            "type": suffix.lstrip("."),
        }
        merged_meta = {**(metadata or {}), **extra_meta}

        if suffix == ".pdf":
            chunks_added = await self._ingest_pdf(path, merged_meta)
        elif suffix == ".docx":
            chunks_added = await self._ingest_docx(path, merged_meta)
        elif suffix in (".md", ".txt"):
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks_added = await self.doc_ingester.ingest(
                content=_clean_text(text),
                source=str(path),
                doc_type="document",
                metadata=merged_meta,
            )
        elif suffix in (".json", ".yaml", ".yml"):
            chunks_added = await self._ingest_structured(path, merged_meta, suffix)
        else:
            # Best-effort: treat as text
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks_added = await self.doc_ingester.ingest(
                content=_clean_text(text),
                source=str(path),
                doc_type="document",
                metadata=merged_meta,
            )

        logger.info(f"Document ingested: {path.name} → {chunks_added} chunks")
        self.summary.append((path.name, "document", chunks_added))
        return chunks_added

    async def _ingest_pdf(self, path: Path, meta: dict[str, Any]) -> int:
        try:
            from pypdf import PdfReader  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "pypdf is not installed. Run: uv add pypdf  (or pip install pypdf)"
            )

        reader = PdfReader(str(path))
        total = 0
        any_text = False
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.debug(f"pypdf extract page {page_index} failed: {exc}")
                text = ""
            if not text.strip():
                continue
            any_text = True
            chunk_meta = {**meta, "page": page_index}
            total += await self.doc_ingester.ingest(
                content=_clean_text(text),
                source=f"{path}#page={page_index}",
                doc_type="document",
                metadata=chunk_meta,
            )

        if not any_text:
            logger.warning(
                f"Scanned PDF detected, text extraction limited: {path.name}"
            )
        return total

    async def _ingest_docx(self, path: Path, meta: dict[str, Any]) -> int:
        try:
            from docx import Document  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "python-docx is not installed. Run: uv add python-docx"
            )

        doc = Document(str(path))
        parts: list[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                style = (para.style.name or "").lower() if para.style else ""
                if "heading 1" in style:
                    parts.append(f"\n# {para.text.strip()}")
                elif "heading 2" in style:
                    parts.append(f"\n## {para.text.strip()}")
                elif "heading" in style:
                    parts.append(f"\n### {para.text.strip()}")
                else:
                    parts.append(para.text.strip())

        for table in doc.tables:
            parts.append(_docx_table_to_markdown(table))

        content = _clean_text("\n\n".join(parts))
        return await self.doc_ingester.ingest(
            content=content,
            source=str(path),
            doc_type="document",
            metadata=meta,
        )

    async def _ingest_structured(
        self,
        path: Path,
        meta: dict[str, Any],
        suffix: str,
    ) -> int:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            if suffix == ".json":
                data = json.loads(raw)
            else:
                import yaml  # local import; pyyaml is already a dep
                data = yaml.safe_load(raw)
        except Exception as exc:
            logger.warning(f"Structured parse failed for {path.name}: {exc}")
            data = raw

        rendered = _structured_to_markdown(data, header=path.name)
        return await self.doc_ingester.ingest(
            content=rendered,
            source=str(path),
            doc_type="document",
            metadata=meta,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # UNIFIED ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    async def ingest_all(
        self,
        sources: dict[str, Any],
        console: Console | None = None,
    ) -> dict[str, int]:
        """
        Run every requested source sequentially.

          sources = {
              "playwright_docs": True,
              "urls":            ["https://app1.com", "https://app2.com"],
              "documents":       ["spec.pdf", "requirements.docx"],
          }

        Returns a dict {label: chunks_added} and prints a summary table.
        """
        self.summary = []
        results: dict[str, int] = {}

        if sources.get("playwright_docs"):
            results["playwright_docs"] = await self.ingest_playwright_docs()

        for url in sources.get("urls", []) or []:
            results[url] = await self.ingest_url(url)

        for doc in sources.get("documents", []) or []:
            results[str(doc)] = await self.ingest_document(doc)

        self._print_summary(console)
        return results

    def _print_summary(self, console: Console | None) -> None:
        if console is None:
            console = Console()
        table = Table(title="RAG Ingestion Summary")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Chunks Added", justify="right", style="green")
        total = 0
        for source, kind, chunks in self.summary:
            label = source if len(source) <= 60 else source[:57] + "…"
            table.add_row(label, kind, str(chunks))
            total += chunks
        table.add_row("[bold]Total[/bold]", "", f"[bold]{total}[/bold]")
        console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# Markdown / Header helpers
# ──────────────────────────────────────────────────────────────────────────────


def _chunk_by_headers(soup_root: Any) -> list[tuple[str, str]]:
    """
    Walk *soup_root* and group its content into (heading_text, markdown_chunk)
    pairs, splitting whenever an <h2> or <h3> is encountered.

    Uses markdownify per-chunk so converting one section's worth of HTML keeps
    code-blocks intact.
    """
    try:
        from markdownify import markdownify as _md   # type: ignore[import]
    except ImportError:
        raise ImportError(
            "markdownify is not installed. Run: uv add markdownify"
        )

    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    buffer: list[str] = []

    for child in soup_root.descendants:
        # We only look at direct heading tags & paragraph-like blocks at the
        # top level of `soup_root`.  Anything inside <code> or nested
        # containers is naturally swept up when we re-render with markdownify.
        if getattr(child, "name", None) in ("h2", "h3"):
            if buffer:
                section_md = _md("\n".join(buffer), heading_style="ATX").strip()
                if section_md:
                    sections.append((current_heading, section_md))
                buffer = []
            current_heading = child.get_text(strip=True) or current_heading

    # Whole-document fallback: if no headings were found, emit one section.
    if not sections:
        whole = _md(str(soup_root), heading_style="ATX").strip()
        if whole:
            sections.append((current_heading, whole))
        return sections

    # If a trailing buffer remained, flush it as the final section.
    if buffer:
        section_md = _md("\n".join(buffer), heading_style="ATX").strip()
        if section_md:
            sections.append((current_heading, section_md))

    # Heuristic: the per-tag iteration above can produce thin sections when
    # the structure is JS-heavy. To guarantee coverage we always fall back
    # to a whole-page conversion split on `## ` boundaries.
    whole = _md(str(soup_root), heading_style="ATX").strip()
    split = re.split(r"\n(?=##\s+)", whole)
    fallback_sections: list[tuple[str, str]] = []
    for block in split:
        block = block.strip()
        if not block:
            continue
        heading_match = re.match(r"##\s+(.+)", block)
        heading = heading_match.group(1).strip() if heading_match else "Introduction"
        fallback_sections.append((heading, block))
    if len(fallback_sections) > len(sections):
        return fallback_sections
    return sections


def _normalise_origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


async def _discover_internal_links(page: Any, origin: str) -> list[str]:
    """Return same-origin <a href> URLs found on *page* (excluding asset URLs)."""
    try:
        hrefs: list[str] = await page.eval_on_selector_all(
            "a[href]",
            "elements => elements.map(el => el.href)",
        )
    except Exception:
        hrefs = []

    result: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        if not href:
            continue
        absolute = urljoin(origin + "/", href)
        absolute, _ = urldefrag(absolute)
        parsed = urlparse(absolute)
        if not parsed.scheme.startswith("http"):
            continue
        if f"{parsed.scheme}://{parsed.netloc}" != origin:
            continue
        if any(parsed.path.lower().endswith(ext) for ext in _SKIPPABLE_LINK_EXTENSIONS):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        result.append(absolute)
    return result


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _docx_table_to_markdown(table: Any) -> str:
    """Convert a python-docx table into a GitHub-flavoured markdown table."""
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for row in body:
        md += "| " + " | ".join(row) + " |\n"
    return md


def _structured_to_markdown(data: Any, header: str = "", level: int = 1) -> str:
    """
    Render JSON/YAML data as a readable markdown description.  Keeps the
    structure flat enough that chunkers can find natural break points.
    """
    lines: list[str] = []
    if header:
        lines.append(f"{'#' * min(level, 6)} {header}")

    def _emit(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}- **{k}**:")
                    _emit(v, prefix + "  ")
                else:
                    lines.append(f"{prefix}- **{k}**: {v}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _emit(item, prefix + "  ")
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{obj}")

    _emit(data)
    return "\n".join(lines)


def _infer_doc_type(suffix: str) -> str:
    mapping = {
        ".py": "python",
        ".md": "markdown",
        ".txt": "text",
        ".rst": "rst",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf",
        ".docx": "docx",
    }
    return mapping.get(suffix.lower(), "text")
