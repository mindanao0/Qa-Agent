"""State Flow Graph (SFG) builder and Structured Action Knowledge Graph helpers.

The engine speaks to a live Playwright Page via the accessibility tree first, with
no DOM-screenshot fallback in this module. All long-running calls flow through
``httpx.AsyncClient`` so that the LangGraph pipeline can compose them with the
rest of the async stack.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import httpx
import networkx as nx
from playwright.async_api import Page
from rich.console import Console

from src.core.state_schema import GUIState, SFGEdge

console = Console()

MAX_SNAPSHOT_CHARS = 12_000  # ≈ 3000 tokens at 4 chars / token
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


def _strip_html_noise(text: str) -> str:
    """Remove inline <script>/<style> blocks from any HTML-bearing fragments."""
    return SCRIPT_STYLE_RE.sub("", text)


def _compress_snapshot(raw: str) -> str:
    """Truncate a stringified AxTree to fit the snapshot budget."""
    cleaned = _strip_html_noise(raw)
    if len(cleaned) <= MAX_SNAPSHOT_CHARS:
        return cleaned
    head_budget = MAX_SNAPSHOT_CHARS - 64
    return cleaned[:head_budget] + "\n…[truncated]…"


class SFGEngine:
    """Constructs and queries the SFG / SA-KG used by the pipeline."""

    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self.ollama_url = ollama_url.rstrip("/")

    async def capture_gui_state(self, page: Page) -> GUIState:
        """Snapshot the accessibility tree into a hashed ``GUIState``."""
        try:
            ax_tree: Any = await page.accessibility.snapshot()
        except Exception as exc:  # noqa: BLE001
            console.log(f"[SFG] accessibility snapshot failed: {exc}")
            ax_tree = None

        serialized = json.dumps(ax_tree, ensure_ascii=False) if ax_tree else ""
        compressed = _compress_snapshot(serialized)
        url = page.url
        state_hash = hashlib.sha256((url + compressed).encode("utf-8")).hexdigest()
        return GUIState(
            state_hash=state_hash,
            url=url,
            accessibility_snapshot=compressed,
            timestamp=time.time(),
        )

    async def extract_interactive_edges(
        self, page: Page, source_hash: str
    ) -> list[SFGEdge]:
        """Walk all interactive DOM nodes and emit candidate SFG edges."""
        js = """
        () => {
          const sel = 'button,a,input,select,textarea,[role]';
          return Array.from(document.querySelectorAll(sel)).map(el => ({
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute('role') || '',
            aria_label: el.getAttribute('aria-label') || '',
            name: el.getAttribute('name') || el.textContent?.trim().slice(0, 80) || '',
            type: el.getAttribute('type') || '',
            has_label: !!(el.getAttribute('aria-label') || el.getAttribute('aria-labelledby')),
            test_id: el.getAttribute('data-testid') || ''
          }));
        }
        """
        try:
            raw_elements: list[dict[str, Any]] = await page.evaluate(js)
        except Exception as exc:  # noqa: BLE001
            console.log(f"[SFG] interactive scan failed: {exc}")
            return []

        edges: list[SFGEdge] = []
        for el in raw_elements:
            role = el.get("role") or self._tag_to_role(el.get("tag", ""))
            if not role:
                continue
            if role:
                strategy = "getByRole"
            elif el.get("has_label") or el.get("aria_label"):
                strategy = "getByLabel"
            else:
                strategy = "getByTestId"

            action_type = self._infer_action_type(el)
            accessible_name = el.get("aria_label") or el.get("name") or el.get("test_id") or ""
            edges.append(
                SFGEdge(
                    source_hash=source_hash,
                    target_hash="pending",
                    aria_role=role,
                    accessible_name=accessible_name,
                    action_type=action_type,
                    locator_strategy=strategy,
                )
            )
        return edges

    def build_networkx_graph(
        self, nodes: list[GUIState], edges: list[SFGEdge]
    ) -> nx.DiGraph:
        """Compile nodes/edges into a directed graph; duplicate hashes are merged."""
        graph: nx.DiGraph = nx.DiGraph()
        for node in nodes:
            if graph.has_node(node.state_hash):
                continue
            graph.add_node(node.state_hash, state=node.model_dump())
        for edge in edges:
            if not graph.has_node(edge.source_hash):
                graph.add_node(edge.source_hash, state=None)
            if not graph.has_node(edge.target_hash):
                graph.add_node(edge.target_hash, state=None)
            graph.add_edge(
                edge.source_hash,
                edge.target_hash,
                **edge.model_dump(),
            )
        return graph

    def detect_stagnation(
        self, graph: nx.DiGraph, recent_hashes: list[str]
    ) -> bool:
        """Return True when the agent appears stuck in a UI tarpit."""
        if len(recent_hashes) >= 3 and recent_hashes[-1] == recent_hashes[-2] == recent_hashes[-3]:
            return True
        if recent_hashes:
            current = recent_hashes[-1]
            if graph.has_node(current):
                if graph.out_degree(current) == 0:
                    return True
        return False

    async def compress_memory(
        self, execution_log: list[str], ollama_url: str | None = None
    ) -> str:
        """Ask Ollama to compress recent execution log into a short memory summary."""
        endpoint = (ollama_url or self.ollama_url).rstrip("/") + "/api/generate"
        tail = execution_log[-10:] if execution_log else []
        prompt = (
            "Summarize completed sub-goals in ≤3 sentences: "
            + json.dumps(tail, ensure_ascii=False)
        )
        payload = {
            "model": DEFAULT_OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 2048},
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                text = resp.json().get("response", "")
        except Exception as exc:  # noqa: BLE001
            console.log(f"[SFG] memory compression failed: {exc}")
            return ""
        return text.strip()

    @staticmethod
    def _tag_to_role(tag: str) -> str:
        mapping = {
            "button": "button",
            "a": "link",
            "input": "textbox",
            "select": "combobox",
            "textarea": "textbox",
        }
        return mapping.get(tag, "")

    @staticmethod
    def _infer_action_type(el: dict[str, Any]) -> str:
        tag = el.get("tag", "")
        el_type = (el.get("type") or "").lower()
        if tag in {"input", "textarea"}:
            if el_type in {"button", "submit", "reset", "checkbox", "radio"}:
                return "click"
            return "fill"
        if tag == "select":
            return "select"
        if tag == "a":
            return "click"
        return "click"


__all__ = ["SFGEngine"]
