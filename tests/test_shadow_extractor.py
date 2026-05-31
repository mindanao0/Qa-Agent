# tests/test_shadow_extractor.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.shadow.extractor import ShadowDOMExtractor, ShadowNode


# ──────────────── extract() ────────────────


@pytest.mark.asyncio
async def test_extract_single_open_shadow_root():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {
        "nodes": [
            {"nodeId": 1, "nodeName": "DIV", "localName": "div", "parentId": 0},
            {
                "nodeId": 2,
                "nodeName": "#shadow-root",
                "shadowRootType": "open",
                "parentId": 1,
            },
            {"nodeId": 3, "nodeName": "BUTTON", "localName": "button", "parentId": 2},
        ]
    }

    extractor = ShadowDOMExtractor()
    result = await extractor.extract(page)

    assert len(result) == 1
    node = result[0]
    assert node.node_id == "2"
    assert node.host_role == "div"
    assert node.shadow_mode == "open"
    assert "3" in node.children
    assert node.ax_label is None


@pytest.mark.asyncio
async def test_extract_closed_shadow_root():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {
        "nodes": [
            {"nodeId": 10, "nodeName": "MY-ELEM", "localName": "my-elem", "parentId": 0},
            {
                "nodeId": 11,
                "nodeName": "#shadow-root",
                "shadowRootType": "closed",
                "parentId": 10,
            },
        ]
    }

    extractor = ShadowDOMExtractor()
    result = await extractor.extract(page)

    assert len(result) == 1
    assert result[0].shadow_mode == "closed"
    assert result[0].host_role == "my-elem"


@pytest.mark.asyncio
async def test_extract_user_agent_shadow_mapped_to_open():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {
        "nodes": [
            {"nodeId": 20, "nodeName": "INPUT", "localName": "input", "parentId": 0},
            {
                "nodeId": 21,
                "nodeName": "#shadow-root",
                "shadowRootType": "user-agent",
                "parentId": 20,
            },
        ]
    }

    extractor = ShadowDOMExtractor()
    result = await extractor.extract(page)

    assert len(result) == 1
    assert result[0].shadow_mode == "open"


@pytest.mark.asyncio
async def test_extract_no_shadow_roots_returns_empty():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {
        "nodes": [
            {"nodeId": 1, "nodeName": "DIV", "localName": "div", "parentId": 0},
            {"nodeId": 2, "nodeName": "SPAN", "localName": "span", "parentId": 1},
        ]
    }

    extractor = ShadowDOMExtractor()
    result = await extractor.extract(page)

    assert result == []


@pytest.mark.asyncio
async def test_extract_multiple_shadow_roots():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {
        "nodes": [
            {"nodeId": 1, "nodeName": "DIV", "localName": "div", "parentId": 0},
            {"nodeId": 2, "nodeName": "#shadow-root", "shadowRootType": "open", "parentId": 1},
            {"nodeId": 3, "nodeName": "SPAN", "localName": "span", "parentId": 0},
            {"nodeId": 4, "nodeName": "#shadow-root", "shadowRootType": "closed", "parentId": 3},
        ]
    }

    extractor = ShadowDOMExtractor()
    result = await extractor.extract(page)

    assert len(result) == 2


@pytest.mark.asyncio
async def test_extract_calls_cdp_with_pierce():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {"nodes": []}

    extractor = ShadowDOMExtractor()
    await extractor.extract(page)

    client.send.assert_called_once_with(
        "DOM.getFlattenedDocument", {"depth": -1, "pierce": True}
    )


@pytest.mark.asyncio
async def test_extract_detaches_cdp_session():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.return_value = {"nodes": []}

    extractor = ShadowDOMExtractor()
    await extractor.extract(page)

    client.detach.assert_called_once()


@pytest.mark.asyncio
async def test_extract_detaches_cdp_session_on_error():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    client.send.side_effect = RuntimeError("CDP error")

    extractor = ShadowDOMExtractor()
    with pytest.raises(RuntimeError):
        await extractor.extract(page)

    client.detach.assert_called_once()


# ──────────────── merge_into_ax() ────────────────


def test_merge_injects_shadow_label_by_host_role():
    extractor = ShadowDOMExtractor()
    ax_nodes = [
        {"role": {"value": "button"}, "name": {"value": "Submit"}},
        {"role": {"value": "textbox"}, "name": {"value": "Email"}},
    ]
    shadow_nodes = [
        ShadowNode(
            node_id="5",
            host_role="button",
            shadow_mode="open",
            children=[],
            ax_label="Shadow Submit",
        )
    ]
    merged = extractor.merge_into_ax(ax_nodes, shadow_nodes)

    assert len(merged) == 2
    assert merged[0]["shadow_label"] == "Shadow Submit"
    assert "shadow_label" not in merged[1]


def test_merge_skips_shadow_nodes_with_no_label():
    extractor = ShadowDOMExtractor()
    ax_nodes = [{"role": {"value": "button"}, "name": {"value": "OK"}}]
    shadow_nodes = [
        ShadowNode(
            node_id="9",
            host_role="button",
            shadow_mode="open",
            children=[],
            ax_label=None,
        )
    ]
    merged = extractor.merge_into_ax(ax_nodes, shadow_nodes)
    assert "shadow_label" not in merged[0]


def test_merge_returns_same_count_as_input():
    extractor = ShadowDOMExtractor()
    ax_nodes = [{"role": {"value": "link"}}, {"role": {"value": "heading"}}]
    shadow_nodes: list[ShadowNode] = []
    merged = extractor.merge_into_ax(ax_nodes, shadow_nodes)
    assert len(merged) == 2
