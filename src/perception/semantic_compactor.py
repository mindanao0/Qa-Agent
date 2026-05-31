# src/perception/semantic_compactor.py
"""
SemanticCompactor — Layer 3 of the Universal DOM Compression Pipeline.

Groups pruned AOM/DOM elements into a structured CompactPAM (Page Action Map)
representation with deterministic refs, proportional capping, and both Markdown
and JSON output formats.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.perception.aom_extractor import AOMNode, AOMSnapshot, _collect_all_nodes
from src.perception.dom_pruner import PrunedDOMSnapshot
from src.perception.token_counter import estimate_tokens as _estimate_tokens


# ── CompactPAM model ──────────────────────────────────────────────────────────


class CompactPAM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["md", "json"]
    content: str                        # the compact representation
    controls_count: int
    forms_count: int
    lists_count: int
    estimated_tokens: int
    source: Literal["aom", "dom", "hybrid", "failure"]
    dropped_nodes: int                  # clipped due to max_items


# ── Internal helpers ──────────────────────────────────────────────────────────

# Role/tag sets for grouping
_CONTROLS_ROLES = {"button", "link", "menuitem", "tab", "option"}
_CONTROLS_TAGS = {"button", "a"}
_FORMS_ROLES = {"textbox", "checkbox", "radio", "combobox", "select", "listbox"}
_FORMS_TAGS = {"input", "select", "textarea"}
_LISTS_ROLES = {"list", "listitem", "grid", "row", "cell", "columnheader", "rowheader"}
_LISTS_TAGS = {"ul", "ol", "li", "table", "tr", "td", "th"}
_IMAGES_ROLES = {"img", "image", "figure"}
_IMAGES_TAGS = {"img", "figure", "picture"}


def _classify(role: str, tag: str) -> str:
    """Return section name for an element based on role and tag."""
    r = (role or "").lower()
    t = (tag or "").lower()
    if r in _CONTROLS_ROLES or t in _CONTROLS_TAGS:
        return "controls"
    if r in _FORMS_ROLES or t in _FORMS_TAGS:
        return "forms"
    if r in _LISTS_ROLES or t in _LISTS_TAGS:
        return "lists"
    if r in _IMAGES_ROLES or t in _IMAGES_TAGS:
        return "images"
    return "containers"


def _flatten_aom(node: AOMNode) -> list[dict[str, Any]]:
    """Flatten AOM tree into a list of dicts."""
    result: list[dict[str, Any]] = []
    all_nodes = _collect_all_nodes(node)
    for i, n in enumerate(all_nodes):
        result.append({
            "role": n.role,
            "name": n.name or "",
            "value": n.value,
            "state": n.state,
            "bbox": n.bbox,
            "source_node_id": n.source_node_id,
            "origin": "aom",
            # Score AOM nodes by depth-inverted order — root-level = higher score
            "score": max(0.0, 20.0 - i * 0.1),
            "tag": "",
            "locator": "",
            "text": n.value or n.name or "",
        })
    return result


def _flatten_dom(dom: PrunedDOMSnapshot) -> list[dict[str, Any]]:
    """Convert DOM pruned elements to unified dicts."""
    result: list[dict[str, Any]] = []
    for el in dom.elements:
        result.append({
            "role": el.role or el.tag or "",
            "name": el.accessible_name or el.text or "",
            "value": None,
            "state": el.state,
            "bbox": el.bbox,
            "source_node_id": el.element_id,
            "origin": "dom",
            "score": el.score,
            "tag": el.tag or "",
            "locator": el.locator or "",
            "text": el.text or "",
        })
    return result


def _names_match(name_a: str, name_b: str) -> bool:
    """Check if two element names match (exact or first-80-chars)."""
    a = (name_a or "")[:80].strip().lower()
    b = (name_b or "")[:80].strip().lower()
    return bool(a and b and a == b)


def _deduplicate_hybrid(
    aom_elems: list[dict[str, Any]],
    dom_elems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge AOM and DOM elements. Where role+name match, keep DOM's locator/score
    but preserve AOM's role/name. Unmatched elements from both sources are included.
    """
    merged: list[dict[str, Any]] = []
    dom_used = set()

    for a in aom_elems:
        matched = False
        for j, d in enumerate(dom_elems):
            if j in dom_used:
                continue
            if a["role"].lower() == (d["role"] or "").lower() and _names_match(a["name"], d["name"]):
                # Merged element: AOM role/name, DOM locator/score/bbox
                merged_elem = dict(a)
                merged_elem["locator"] = d["locator"]
                merged_elem["score"] = d["score"]
                if d["bbox"] is not None:
                    merged_elem["bbox"] = d["bbox"]
                merged_elem["origin"] = "hybrid"
                merged_elem["tag"] = d["tag"]
                merged.append(merged_elem)
                dom_used.add(j)
                matched = True
                break
        if not matched:
            merged.append(a)

    # Add unmatched DOM elements
    for j, d in enumerate(dom_elems):
        if j not in dom_used:
            merged.append(d)

    return merged


def _make_ref_id(prefix: str, role: str, name: str) -> str:
    """Generate a short deterministic ref from role+name."""
    key = (role + name).encode()
    return f"{prefix}{hashlib.sha256(key).hexdigest()[:8]}"


def _format_bbox(bbox: Any) -> str:
    """Format bbox as percentage position string."""
    if bbox is None:
        return ""
    try:
        x = getattr(bbox, "x_pct", None)
        y = getattr(bbox, "y_pct", None)
        if x is not None and y is not None:
            return f" @[{x:.0f}%,{y:.0f}%]"
    except Exception:
        pass
    return ""


def _format_state(state: dict[str, bool]) -> str:
    """Format state dict as a compact string."""
    if not state:
        return ""
    active = [k for k, v in state.items() if v]
    if not active:
        return ""
    return " state=" + ",".join(active)


# ── SemanticCompactor ─────────────────────────────────────────────────────────


class SemanticCompactor:
    """
    Layer 3: Groups AOM/DOM elements into a CompactPAM structured representation.

    Usage::

        compactor = SemanticCompactor()
        pam = compactor.compact(aom_snapshot, dom_snapshot, max_items=25)
    """

    def compact(
        self,
        aom: AOMSnapshot | None,
        dom: PrunedDOMSnapshot | None,
        max_items: int = 25,
        output_format: Literal["md", "json"] = "md",
    ) -> CompactPAM:
        """
        Produce a CompactPAM from AOM and/or DOM snapshots.

        Args:
            aom: AOMSnapshot from AOMExtractor, or None.
            dom: PrunedDOMSnapshot from DOMPruner, or None.
            max_items: Maximum total elements in the output (proportionally capped).
            output_format: "md" for Markdown, "json" for JSON.

        Returns:
            CompactPAM with format, content, counts, token estimate, source.

        Raises:
            ValueError: If both aom and dom are None.
        """
        # 1. Determine source
        if aom is not None and dom is not None:
            source: Literal["aom", "dom", "hybrid", "failure"] = "hybrid"
        elif aom is not None:
            source = "aom"
        elif dom is not None:
            source = "dom"
        else:
            raise ValueError("At least one of aom or dom must be provided")

        # 2. Collect elements
        aom_elems: list[dict[str, Any]] = []
        dom_elems: list[dict[str, Any]] = []

        if aom is not None:
            aom_elems = _flatten_aom(aom.root_node)

        if dom is not None:
            dom_elems = _flatten_dom(dom)

        # 3. Deduplicate hybrid
        if source == "hybrid":
            all_elems = _deduplicate_hybrid(aom_elems, dom_elems)
        elif source == "aom":
            all_elems = aom_elems
        else:
            all_elems = dom_elems

        # 4. Group into sections
        sections: dict[str, list[dict[str, Any]]] = {
            "controls": [],
            "forms": [],
            "lists": [],
            "images": [],
            "containers": [],
        }

        for elem in all_elems:
            section = _classify(elem["role"], elem["tag"])
            sections[section].append(elem)

        # 5. Sort each group by score descending
        for key in sections:
            sections[key].sort(key=lambda e: e["score"], reverse=True)

        # 6. Apply proportional max_items cap
        total_before = sum(len(v) for v in sections.values())
        dropped_nodes = 0

        if total_before > max_items:
            # Scale down each group proportionally
            remaining = max_items
            section_keys = list(sections.keys())
            new_sections: dict[str, list[dict[str, Any]]] = {}

            for i, key in enumerate(section_keys):
                group = sections[key]
                if not group:
                    new_sections[key] = []
                    continue
                if i == len(section_keys) - 1:
                    # Last group gets whatever is left
                    cap = remaining
                else:
                    proportion = len(group) / total_before
                    cap = max(1, math.floor(proportion * max_items))
                    cap = min(cap, remaining)

                new_sections[key] = group[:cap]
                dropped_nodes += len(group) - len(new_sections[key])
                remaining -= len(new_sections[key])
                if remaining <= 0:
                    # Zero out remaining sections
                    for j in range(i + 1, len(section_keys)):
                        k = section_keys[j]
                        dropped_nodes += len(sections[k])
                        new_sections[k] = []
                    break

            sections = new_sections

        # 7. Sort elements within each section by (role, name) for deterministic refs
        section_prefix = {
            "controls": "c",
            "forms": "f",
            "lists": "l",
            "images": "i",
            "containers": "n",
        }

        # 8. Render output
        if output_format == "json":
            content = self._render_json(sections, section_prefix)
        else:
            content = self._render_markdown(sections, section_prefix)

        # Count stats
        controls_count = len(sections["controls"])
        forms_count = len(sections["forms"])
        lists_count = len(sections["lists"])
        estimated_tokens = _estimate_tokens(content)

        logger.debug(
            f"SemanticCompactor: source={source} total={total_before - dropped_nodes} "
            f"dropped={dropped_nodes} tokens={estimated_tokens} format={output_format}"
        )

        return CompactPAM(
            format=output_format,
            content=content,
            controls_count=controls_count,
            forms_count=forms_count,
            lists_count=lists_count,
            estimated_tokens=estimated_tokens,
            source=source,
            dropped_nodes=dropped_nodes,
        )

    def _make_refs(
        self,
        elems: list[dict[str, Any]],
        prefix: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Sort elements by (role, name) and assign deterministic sequential refs.
        Returns list of (ref_id, elem) pairs.
        """
        sorted_elems = sorted(elems, key=lambda e: (e["role"].lower(), e["name"].lower()))
        seen_refs: dict[str, int] = {}
        result: list[tuple[str, dict[str, Any]]] = []

        for elem in sorted_elems:
            base_ref = _make_ref_id(prefix, elem["role"], elem["name"])
            count = seen_refs.get(base_ref, 0)
            seen_refs[base_ref] = count + 1
            # If collision, append count suffix
            ref = base_ref if count == 0 else f"{base_ref}{count}"
            result.append((ref, elem))

        return result

    def _render_markdown(
        self,
        sections: dict[str, list[dict[str, Any]]],
        section_prefix: dict[str, str],
    ) -> str:
        """Render sections as Markdown, skipping empty sections."""
        section_titles = {
            "controls": "Controls",
            "forms": "Forms",
            "lists": "Lists",
            "images": "Images",
            "containers": "Containers",
        }
        lines: list[str] = []

        for section_key, title in section_titles.items():
            elems = sections[section_key]

            # Skip empty sections entirely
            if not elems:
                continue

            lines.append(f"## {title}")
            prefix = section_prefix[section_key]
            ref_pairs = self._make_refs(elems, prefix)

            for ref, elem in ref_pairs:
                role = elem["role"] or ""
                name = elem["name"] or ""
                state_str = _format_state(elem.get("state") or {})
                bbox_str = _format_bbox(elem.get("bbox"))

                # Build line
                parts = [f"- [ref={ref}] role={role}"]
                if name:
                    parts.append(f'name="{name}"')

                # Extra attrs for forms
                if section_key == "forms":
                    value = elem.get("value")
                    if value:
                        parts.append(f'value="{value}"')
                    text = elem.get("text", "")
                    if text and text != name:
                        parts.append(f'placeholder="{text[:50]}"')

                # State
                if state_str:
                    parts.append(state_str.strip())

                # Item count for lists
                if section_key == "lists":
                    meta = elem.get("meta") or {}
                    if isinstance(meta, dict):
                        item_count = meta.get("item_count")
                        if item_count and item_count > 1:
                            parts.append(f"(×{item_count} items)")

                # Bbox
                if bbox_str:
                    parts.append(bbox_str.strip())

                lines.append(" ".join(parts))

            lines.append("")  # blank line between sections

        if not lines:
            return "(no actionable elements found)"
        return "\n".join(lines).strip()

    def _render_json(
        self,
        sections: dict[str, list[dict[str, Any]]],
        section_prefix: dict[str, str],
    ) -> str:
        """Render sections as JSON."""
        output: dict[str, list[dict[str, Any]]] = {}

        for section_key, elems in sections.items():
            prefix = section_prefix[section_key]
            ref_pairs = self._make_refs(elems, prefix)
            section_list: list[dict[str, Any]] = []

            for ref, elem in ref_pairs:
                bbox = elem.get("bbox")
                bbox_val = None
                if bbox is not None:
                    try:
                        bbox_val = [
                            round(getattr(bbox, "x_pct", 0)),
                            round(getattr(bbox, "y_pct", 0)),
                        ]
                    except Exception:
                        bbox_val = None

                item: dict[str, Any] = {
                    "ref": ref,
                    "role": elem["role"],
                    "name": elem["name"],
                }
                state = elem.get("state") or {}
                active_states = {k: v for k, v in state.items() if v}
                if active_states:
                    item["state"] = active_states
                if bbox_val is not None:
                    item["bbox"] = bbox_val
                if elem.get("locator"):
                    item["locator"] = elem["locator"]

                section_list.append(item)

            output[section_key] = section_list

        return json.dumps(output, indent=2)
