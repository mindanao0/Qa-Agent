import pytest
from unittest.mock import AsyncMock, MagicMock
from src.universal_qa.explorer.session_guard import SessionGuard


@pytest.mark.asyncio
async def test_is_session_lost_true_when_password_visible():
    guard = SessionGuard(auth=MagicMock())
    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=object())  # password input exists
    page.title = AsyncMock(return_value="Some Page")
    assert await guard.is_session_lost(page) is True


@pytest.mark.asyncio
async def test_is_session_lost_false_when_no_login_signals():
    guard = SessionGuard(auth=MagicMock())
    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    page.title = AsyncMock(return_value="Dashboard")
    assert await guard.is_session_lost(page) is False


@pytest.mark.asyncio
async def test_recover_calls_auth_setup_and_counts():
    auth = MagicMock()
    auth.setup = AsyncMock(return_value=("u", "p"))
    guard = SessionGuard(auth=auth, max_attempts=3)
    page = AsyncMock()
    ok = await guard.recover(page)
    assert ok is True
    auth.setup.assert_awaited_once()


@pytest.mark.asyncio
async def test_recover_stops_after_max_attempts():
    auth = MagicMock()
    auth.setup = AsyncMock(return_value=None)
    guard = SessionGuard(auth=auth, max_attempts=2)
    page = AsyncMock()
    await guard.recover(page)
    await guard.recover(page)
    third = await guard.recover(page)
    assert third is False  # exceeded max_attempts
