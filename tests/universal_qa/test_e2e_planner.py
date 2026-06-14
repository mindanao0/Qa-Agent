from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.universal_qa.e2e_planner import E2EFlowPlanner, _E2EResponse, _E2EFlowItem
from src.universal_qa.explorer.nav_map import ExploredAction, ExploredPage, NavigationMap
from src.universal_qa.models import TestCase


def _make_nav_map(urls: list[str] | None = None) -> NavigationMap:
    pages = []
    for url in (urls or ["https://x.com/login", "https://x.com/cart", "https://x.com/checkout"]):
        title = url.rstrip("/").rsplit("/", 1)[-1].capitalize() or "Home"
        pages.append(ExploredPage(
            url=url, title=title, pam_content="",
            actions=[ExploredAction(page_url=url, action_label="Buy", element_role="button")],
        ))
    return NavigationMap(base_url="https://x.com", pages=pages, flows=[], explored_at_iso="")


def _mock_item(title: str = "E2E flow", priority: str = "high") -> _E2EFlowItem:
    item = MagicMock(spec=_E2EFlowItem)
    item.title = title
    item.priority = priority
    item.preconditions = ["เข้าสู่ระบบแล้ว"]
    item.steps = [
        "เปิดหน้า https://x.com/login",
        "กรอก \"Username\" ด้วย user",
        "คลิกปุ่ม \"Login\"",
        "เปิดหน้า https://x.com/cart",
        "คลิกปุ่ม \"Checkout\"",
    ]
    item.expected_outcome = "สำเร็จ"
    return item


# ── no credentials → return [] ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_returns_empty_without_credentials():
    client = AsyncMock()
    planner = E2EFlowPlanner(client)
    nav_map = _make_nav_map()

    result = await planner.plan(nav_map, username=None, password=None)
    assert result == []
    client.create_structured.assert_not_called()


@pytest.mark.asyncio
async def test_plan_returns_empty_with_only_username():
    client = AsyncMock()
    planner = E2EFlowPlanner(client)
    result = await planner.plan(_make_nav_map(), username="user", password=None)
    assert result == []


# ── happy path ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_returns_test_cases_with_credentials():
    client = AsyncMock()
    mock_response = MagicMock(spec=_E2EResponse)
    mock_response.test_cases = [_mock_item("login → checkout"), _mock_item("login → profile", "medium")]
    client.create_structured = AsyncMock(return_value=mock_response)

    planner = E2EFlowPlanner(client)
    result = await planner.plan(_make_nav_map(), username="user", password="pass")

    assert len(result) == 2
    assert all(isinstance(tc, TestCase) for tc in result)
    assert all(tc.type == "e2e" for tc in result)
    assert result[0].title == "login → checkout"
    assert result[1].priority == "medium"


@pytest.mark.asyncio
async def test_plan_uses_login_url_as_source_url():
    """source_url ต้องชี้ไปยัง login page ที่พบ."""
    client = AsyncMock()
    mock_response = MagicMock(spec=_E2EResponse)
    mock_response.test_cases = [_mock_item()]
    client.create_structured = AsyncMock(return_value=mock_response)

    planner = E2EFlowPlanner(client)
    nav_map = _make_nav_map(["https://x.com/login", "https://x.com/cart"])
    result = await planner.plan(nav_map, username="u", password="p")

    assert result[0].source_url == "https://x.com/login"


@pytest.mark.asyncio
async def test_plan_falls_back_to_base_url_when_no_login_page():
    """ถ้าไม่มีหน้า login ต้อง fallback ไปใช้ base_url."""
    client = AsyncMock()
    mock_response = MagicMock(spec=_E2EResponse)
    mock_response.test_cases = [_mock_item()]
    client.create_structured = AsyncMock(return_value=mock_response)

    planner = E2EFlowPlanner(client)
    nav_map = _make_nav_map(["https://x.com/cart", "https://x.com/checkout"])
    result = await planner.plan(nav_map, username="u", password="p")

    assert result[0].source_url == "https://x.com"


# ── LLM failure → graceful empty ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_returns_empty_on_llm_failure():
    client = AsyncMock()
    client.create_structured = AsyncMock(side_effect=RuntimeError("Ollama down"))

    planner = E2EFlowPlanner(client)
    result = await planner.plan(_make_nav_map(), username="u", password="p")

    assert result == []


# ── priority normalisation ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_normalises_unknown_priority_to_medium():
    """LLM อาจส่ง priority ที่ไม่ถูกต้อง ต้อง fallback เป็น medium."""
    client = AsyncMock()
    bad_item = _mock_item(priority="critical")  # ไม่ใช่ high/medium/low
    mock_response = MagicMock(spec=_E2EResponse)
    mock_response.test_cases = [bad_item]
    client.create_structured = AsyncMock(return_value=mock_response)

    planner = E2EFlowPlanner(client)
    result = await planner.plan(_make_nav_map(), username="u", password="p")

    assert result[0].priority == "medium"


# ── prompt content ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_includes_credentials_in_prompt():
    """Prompt ที่ส่งไปยัง LLM ต้องมี username และ password."""
    client = AsyncMock()
    mock_response = MagicMock(spec=_E2EResponse)
    mock_response.test_cases = []
    client.create_structured = AsyncMock(return_value=mock_response)

    planner = E2EFlowPlanner(client)
    await planner.plan(_make_nav_map(), username="standard_user", password="secret_sauce")

    call_args = client.create_structured.call_args
    prompt_text = call_args[0][0]  # first positional arg
    assert "standard_user" in prompt_text
    assert "secret_sauce" in prompt_text
