import pytest
from unittest.mock import AsyncMock, MagicMock
from src.universal_qa.auth_manager import AuthManager


def test_auth_manager_stores_credentials():
    am = AuthManager(username="user@test.com", password="pass123")
    assert am._username == "user@test.com"
    assert am._password == "pass123"


def test_auth_manager_no_credentials():
    am = AuthManager()
    assert am._username is None


def test_choose_strategy_with_credentials():
    am = AuthManager(username="u@x.com", password="p")
    assert am._strategy() == "login"


def test_choose_strategy_without_credentials():
    am = AuthManager()
    assert am._strategy() == "auto_register"


@pytest.mark.asyncio
async def test_setup_skips_when_no_form_found():
    am = AuthManager()
    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    page.goto = AsyncMock()

    result = await am.setup(page)
    assert result is None  # no auth form found → unauthenticated
