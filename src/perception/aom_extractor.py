# src/perception/aom_extractor.py
"""
AOMExtractor — Layer 1 of the Universal DOM Compression Pipeline.

Captures a structured Accessibility Object Model (AOM) snapshot from a
Playwright page using the Chrome DevTools Protocol (CDP) directly.

TD-15a fix: page.accessibility was removed in Playwright Python ≥1.34.
CDP (Accessibility.getFullAXTree) is the universal replacement — works on
all Chromium versions Playwright supports.

Notes:
- Per-node bbox is intentionally skipped here (bbox=None for all nodes).
  Bounding-box population is deferred to the SemanticCompactor layer which
  retrieves geometry from the already-pruned DOM, keeping this extractor fast.
- source_node_id is deterministic: same page structure → same IDs.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field


# ── Models ────────────────────────────────────────────────────────────────────


class BBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_pct: float = Field(ge=0, le=100)
    y_pct: float = Field(ge=0, le=100)
    w_pct: float = Field(ge=0, le=100)
    h_pct: float = Field(ge=0, le=100)


class AOMNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    name: str
    value: str | None = None
    # checked, disabled, expanded, focused, selected
    state: dict[str, bool] = Field(default_factory=dict)
    # bbox=None: populated by SemanticCompactor, not here (performance)
    bbox: BBox | None = None
    children: list["AOMNode"] = Field(default_factory=list)
    # Deterministic: same page state → same ID (sha256 of path/role/name)
    source_node_id: str


# Required for self-referential Pydantic V2 models
AOMNode.model_rebuild()


class AOMSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # sha256 of (timestamp + url), first 16 hex chars
    snapshot_id: str
    url: str
    timestamp_iso: str
    root_node: AOMNode
    node_count: int
    extraction_latency_ms: int


# ── Exception ─────────────────────────────────────────────────────────────────


class AOMSparseError(Exception):
    """Raised when the AOM snapshot is too sparse to be useful."""

    def __init__(self, url: str, node_count: int) -> None:
        self.url = url
        self.node_count = node_count
        super().__init__(f"AOM sparse at {url}: only {node_count} nodes")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _make_node_id(path: str, role: str, name: str) -> str:
    """Deterministic 12-char hex ID from ancestor path + role + name."""
    key = f"{path}/{role}/{name}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _get_cdp_value(field: dict[str, Any] | None) -> str:
    """Extract .value from a CDP typed-value dict, or return empty string."""
    if not field:
        return ""
    return str(field.get("value", "") or "")


def _should_skip_cdp_node(node: dict[str, Any]) -> bool:
    """
    Return True for CDP nodes that are noise and should not appear in AOMSnapshot.

    Skips:
    - ignored=True nodes (CDP marks these explicitly)
    - role="none" with no name (pure structural containers)
    - role="generic" with no name (generic divs/spans with no semantics)
    """
    if node.get("ignored", False):
        return True
    role_val = _get_cdp_value(node.get("role"))
    name_val = _get_cdp_value(node.get("name"))
    if role_val in ("none", "generic") and not name_val:
        return True
    return False


def _build_tree_from_flat(
    nodes: list[dict[str, Any]],
) -> AOMNode | None:
    """
    Build an AOMNode tree from CDP's flat list of AXNode dicts.

    CDP Accessibility.getFullAXTree returns a flat list where each node has:
      nodeId: str
      childIds: list[str]
      parentId: str | None  (may be absent)
      role: {type: str, value: str}
      name: {type: str, value: str}
      properties: list[{name: str, value: {type: str, value: Any}}]
      ignored: bool

    Skipped nodes (ignored=True, or role=none/generic with no name) are treated as
    transparent: their children are promoted to the parent. This handles React apps
    that nest real semantic nodes inside multiple layers of ignored/generic wrappers.

    Returns the root AOMNode, or None if no non-ignored nodes exist.
    Cross-frame references (childIds not in the list) are silently skipped.
    """
    # Index by nodeId for O(1) lookup
    node_map: dict[str, dict[str, Any]] = {}
    for n in nodes:
        nid = n.get("nodeId")
        if nid is not None:
            node_map[nid] = n

    # Find root: node whose parentId is absent or not in the map
    # The root may itself be ignored (it's the RootWebArea wrapper), so
    # check the first node whose parentId is None/missing regardless of skip status.
    root_cdp: dict[str, Any] | None = None
    for n in nodes:
        parent_id = n.get("parentId")
        if parent_id is None or parent_id not in node_map:
            root_cdp = n
            break

    if root_cdp is None:
        return None

    def _collect_visible_children(
        cdp_node: dict[str, Any], path: str, depth: int
    ) -> list[AOMNode]:
        """
        Collect non-skipped child AOMNodes, recursively promoting through
        skipped (ignored/generic/none) nodes. This handles arbitrary nesting
        depth of wrapper divs and ignored containers.
        """
        result: list[AOMNode] = []
        for child_id in cdp_node.get("childIds") or []:
            child_cdp = node_map.get(child_id)
            if child_cdp is None:
                # Cross-frame boundary — skip gracefully
                logger.debug(
                    f"AOMExtractor: child_id={child_id} not in map (cross-frame), skipping"
                )
                continue
            if _should_skip_cdp_node(child_cdp):
                # Transparent node: promote its children up to the current level
                # (recurse with same path + depth, not incrementing depth)
                result.extend(_collect_visible_children(child_cdp, path, depth))
            else:
                child_node = _convert(child_cdp, path, depth + 1)
                if child_node is not None:
                    result.append(child_node)
        return result

    def _convert(cdp_node: dict[str, Any], path: str, depth: int = 0) -> AOMNode | None:
        if depth > 60:
            # Guard against pathological trees
            logger.debug("AOMExtractor: max depth 60 reached, truncating subtree")
            return None

        role = _get_cdp_value(cdp_node.get("role")) or "unknown"
        name = _get_cdp_value(cdp_node.get("name"))

        # Extract state from CDP properties list
        state: dict[str, bool] = {}
        for prop in cdp_node.get("properties") or []:
            prop_name = prop.get("name", "")
            prop_val = prop.get("value", {})
            if prop_name in ("checked", "disabled", "expanded", "focused", "selected"):
                raw = prop_val.get("value")
                if isinstance(raw, bool):
                    state[prop_name] = raw

        node_id = _make_node_id(path, role, name)
        child_path = f"{path}/{role}:{name}"

        children = _collect_visible_children(cdp_node, child_path, depth)

        return AOMNode(
            role=role,
            name=name,
            value=None,  # CDP value field is complex; DOM layer handles values
            state=state,
            bbox=None,  # deferred to SemanticCompactor
            children=children,
            source_node_id=node_id,
        )

    # If the root itself is skipped (e.g. an ignored wrapper), promote its children
    if _should_skip_cdp_node(root_cdp):
        promoted = _collect_visible_children(root_cdp, "/", 0)
        if not promoted:
            return None
        # Wrap promoted children under a synthetic root so we have a single root
        return promoted[0] if len(promoted) == 1 else AOMNode(
            role="WebArea",
            name="",
            value=None,
            state={},
            bbox=None,
            children=promoted,
            source_node_id=_make_node_id("/", "WebArea", ""),
        )

    return _convert(root_cdp, "/")


def _collect_all_nodes(node: AOMNode) -> list[AOMNode]:
    """Flatten the tree into a list (pre-order)."""
    result = [node]
    for child in node.children:
        result.extend(_collect_all_nodes(child))
    return result


# ── Extractor ─────────────────────────────────────────────────────────────────


class AOMExtractor:
    """
    Layer 1: Captures the Playwright Accessibility Object Model as a typed tree.

    Uses CDP (Chrome DevTools Protocol) Accessibility.getFullAXTree directly —
    compatible with all Playwright versions including ≥1.34 where
    page.accessibility was removed.

    Usage::

        extractor = AOMExtractor()
        snapshot = await extractor.extract(page)
    """

    async def extract(
        self, page: Page, root_selector: str | None = None
    ) -> AOMSnapshot:
        """
        Extract the accessibility tree of *page* via CDP and return a typed AOMSnapshot.

        Args:
            page: An active Playwright Page.
            root_selector: Reserved for future scoped extraction (currently unused).

        Raises:
            AOMSparseError: When node_count < 5 (page has no useful semantics).
        """
        t0 = monotonic()
        url: str = page.url

        if root_selector is not None:
            logger.warning(
                f"AOMExtractor.extract: root_selector={root_selector!r} is not yet implemented "
                "and will be ignored. Full-page AOM will be extracted."
            )

        # Open CDP session — always detach in finally to prevent resource leaks
        cdp = await page.context.new_cdp_session(page)
        try:
            await cdp.send("Accessibility.enable")
            result = await cdp.send("Accessibility.getFullAXTree")
        except Exception as exc:
            logger.error(f"AOMExtractor: CDP call failed for {url}: {exc}")
            raise AOMSparseError(url, 0) from exc
        finally:
            try:
                await cdp.detach()
            except Exception:
                pass  # Already detached or page closed — ignore

        raw_nodes: list[dict[str, Any]] = result.get("nodes", [])

        if not raw_nodes:
            logger.warning(f"AOMExtractor: empty CDP result for {url}")
            raise AOMSparseError(url, 0)

        # Build typed tree from flat CDP node list
        root_node = _build_tree_from_flat(raw_nodes)

        if root_node is None:
            logger.warning(f"AOMExtractor: no non-ignored nodes found for {url}")
            raise AOMSparseError(url, 0)

        # Count nodes; raise if too sparse
        all_nodes = _collect_all_nodes(root_node)
        node_count = len(all_nodes)
        if node_count < 5:
            logger.warning(
                f"AOMExtractor: sparse snapshot ({node_count} nodes) for {url}"
            )
            raise AOMSparseError(url, node_count)

        # Generate snapshot metadata
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        snapshot_id = hashlib.sha256(
            f"{now_iso}{url}".encode()
        ).hexdigest()[:16]

        latency_ms = int((monotonic() - t0) * 1000)
        logger.info(
            f"AOMExtractor: snapshot_id={snapshot_id} nodes={node_count} "
            f"latency={latency_ms}ms url={url}"
        )

        return AOMSnapshot(
            snapshot_id=snapshot_id,
            url=url,
            timestamp_iso=now_iso,
            root_node=root_node,
            node_count=node_count,
            extraction_latency_ms=latency_ms,
        )


# ── Utility ───────────────────────────────────────────────────────────────────


def is_aom_sparse(snapshot: AOMSnapshot) -> bool:
    """
    Return True if the AOM snapshot is too sparse to be useful for grounding.

    Conditions (either is sufficient):
    - node_count < 5, OR
    - >50% of nodes have empty name AND empty value AND no state flags
    """
    if snapshot.node_count < 5:
        return True

    all_nodes = _collect_all_nodes(snapshot.root_node)
    if not all_nodes:
        return True

    empty_count = sum(
        1
        for n in all_nodes
        if not n.name and not n.value and not n.state
    )
    return (empty_count / len(all_nodes)) > 0.5
