# tests/test_spa_hydration_guard.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from src.spa.hydration_guard import HydrationGuard


@pytest.mark.asyncio
async def test_detect_framework_react():
    page = AsyncMock()
    page.evaluate.return_value = "react"
    guard = HydrationGuard()
    result = await guard.detect_framework(page)
    assert result == "react"


@pytest.mark.asyncio
async def test_detect_framework_vue():
    page = AsyncMock()
    page.evaluate.return_value = "vue"
    guard = HydrationGuard()
    result = await guard.detect_framework(page)
    assert result == "vue"


@pytest.mark.asyncio
async def test_detect_framework_angular():
    page = AsyncMock()
    page.evaluate.return_value = "angular"
    guard = HydrationGuard()
    result = await guard.detect_framework(page)
    assert result == "angular"


@pytest.mark.asyncio
async def test_detect_framework_unknown():
    page = AsyncMock()
    page.evaluate.return_value = "unknown"
    guard = HydrationGuard()
    result = await guard.detect_framework(page)
    assert result == "unknown"


@pytest.mark.asyncio
async def test_detect_framework_calls_evaluate_with_js():
    page = AsyncMock()
    page.evaluate.return_value = "unknown"
    guard = HydrationGuard()
    await guard.detect_framework(page)
    page.evaluate.assert_called_once()
    js: str = page.evaluate.call_args[0][0]
    assert "getAllAngularTestabilities" in js
    assert "__vue_app__" in js


@pytest.mark.asyncio
async def test_wait_stable_completes_when_page_ready():
    """wait_stable should complete without raising when page is already complete."""
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)

    # evaluate: first call → readyState "complete"; second call → framework "unknown"
    page.evaluate = AsyncMock(side_effect=["complete", "unknown"])

    guard = HydrationGuard()
    await guard.wait_stable(page, timeout_ms=2000)
    # No assertion needed — success means no exception raised


@pytest.mark.asyncio
async def test_wait_stable_enables_cdp_network():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    page.evaluate = AsyncMock(side_effect=["complete", "unknown"])

    guard = HydrationGuard()
    await guard.wait_stable(page, timeout_ms=2000)

    client.send.assert_any_call("Network.enable")


@pytest.mark.asyncio
async def test_wait_stable_detaches_cdp_session():
    page = MagicMock()
    client = AsyncMock()
    page.context.new_cdp_session = AsyncMock(return_value=client)
    page.evaluate = AsyncMock(side_effect=["complete", "unknown"])

    guard = HydrationGuard()
    await guard.wait_stable(page, timeout_ms=2000)

    client.detach.assert_called_once()
