# src/shadow/extractor.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ShadowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    host_role: str
    shadow_mode: Literal["open", "closed"]
    children: list[str]   # child nodeIds as strings
    ax_label: str | None


class ShadowDOMExtractor:
    """CDP + JS-fallback Shadow DOM extractor. No required constructor args."""

    async def extract(self, page: Any) -> list[ShadowNode]:
        # Primary: CDP DOM.getFlattenedDocument with pierce
        cdp_nodes = await self._extract_cdp(page)
        if cdp_nodes:
            return cdp_nodes
        # Fallback: JS-based traversal for custom elements whose shadow roots
        # are not reported by CDP DOM.getFlattenedDocument (Chromium quirk with
        # declarative / autonomous custom elements on some sites).
        return await self._extract_js(page)

    async def _extract_cdp(self, page: Any) -> list[ShadowNode]:
        client = await page.context.new_cdp_session(page)
        try:
            await client.send("DOM.enable")
            result = await client.send(
                "DOM.getFlattenedDocument", {"depth": -1, "pierce": True}
            )
        finally:
            await client.detach()

        nodes: list[dict] = result.get("nodes", [])
        node_map: dict[int, dict] = {n["nodeId"]: n for n in nodes}
        shadow_nodes: list[ShadowNode] = []

        for node in nodes:
            shadow_root_type = node.get("shadowRootType")
            if shadow_root_type not in ("open", "closed", "user-agent"):
                continue

            parent_id = node.get("parentId")
            host_role = "unknown"
            if parent_id and parent_id in node_map:
                parent = node_map[parent_id]
                host_role = (
                    parent.get("localName")
                    or parent.get("nodeName", "unknown").lower()
                )

            children = [
                str(n["nodeId"])
                for n in nodes
                if n.get("parentId") == node["nodeId"]
            ]

            shadow_nodes.append(
                ShadowNode(
                    node_id=str(node["nodeId"]),
                    host_role=host_role,
                    shadow_mode="open" if shadow_root_type in ("open", "user-agent") else "closed",  # user-agent shadows are treated as open for downstream consumers
                    children=children,
                    ax_label=None,  # populated by external callers; extract() returns structural CDP data only
                )
            )

        return shadow_nodes

    async def _extract_js(self, page: Any) -> list[ShadowNode]:
        """JS-based fallback that walks open shadow roots via document.querySelectorAll."""
        raw: list[dict] = await page.evaluate(
            """
            () => {
                const results = [];
                let nodeCounter = 1000000;  // offset to avoid CDP nodeId collision

                function walkShadow(root, depth) {
                    if (depth > 10) return;
                    const all = root.querySelectorAll('*');
                    for (const el of all) {
                        const sr = el.shadowRoot;
                        if (!sr) continue;
                        const hostTag = el.tagName.toLowerCase();
                        const shadowChildren = Array.from(sr.childNodes).map((c, i) => {
                            return String(nodeCounter + i + 1);
                        });
                        results.push({
                            node_id: String(nodeCounter++),
                            host_role: hostTag,
                            shadow_mode: 'open',
                            children: shadowChildren,
                            ax_label: el.getAttribute('aria-label') || null
                        });
                        // Also emit each direct shadow child as a node
                        const directChildren = Array.from(sr.children);
                        for (const child of directChildren) {
                            results.push({
                                node_id: String(nodeCounter++),
                                host_role: child.tagName.toLowerCase(),
                                shadow_mode: 'open',
                                children: [],
                                ax_label: child.getAttribute('aria-label') || null
                            });
                            // recurse into nested shadow roots
                            walkShadow(sr, depth + 1);
                        }
                    }
                }

                walkShadow(document, 0);
                return results;
            }
            """
        )
        return [ShadowNode(**item) for item in raw]

    def merge_into_ax(
        self,
        ax_nodes: list[dict],
        shadow_nodes: list[ShadowNode],
    ) -> list[dict]:
        """Inject shadow ax_labels into AX tree nodes by matching host_role to node role.

        No-op when ax_label is None on all shadow nodes (the normal path from extract()).
        # last ShadowNode with a given host_role wins if duplicates exist
        """
        role_to_label: dict[str, str] = {
            sn.host_role: sn.ax_label
            for sn in shadow_nodes
            if sn.ax_label is not None
        }
        merged: list[dict] = []
        for node in ax_nodes:
            node_role = (node.get("role") or {}).get("value", "")
            if node_role in role_to_label:
                node = {**node, "shadow_label": role_to_label[node_role]}
            merged.append(node)
        return merged


__all__ = ["ShadowDOMExtractor", "ShadowNode"]
