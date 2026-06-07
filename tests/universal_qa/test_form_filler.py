import pytest
from unittest.mock import AsyncMock, MagicMock
from src.universal_qa.explorer.form_filler import FormFiller, _dummy_for_type


def test_dummy_for_email():
    assert "@" in _dummy_for_type("email")


def test_dummy_for_password():
    assert _dummy_for_type("password") == "QaTest123!"


def test_dummy_for_number():
    assert _dummy_for_type("number").isdigit()


def test_dummy_for_unknown_returns_text():
    assert _dummy_for_type("color") == "Test Input"


@pytest.mark.asyncio
async def test_fill_with_dummy_fills_each_input():
    filler = FormFiller()
    page = AsyncMock()
    # two inputs: email + text
    email_input = AsyncMock()
    text_input = AsyncMock()
    email_input.get_attribute = AsyncMock(return_value="email")
    text_input.get_attribute = AsyncMock(return_value="text")
    page.query_selector_all = AsyncMock(return_value=[email_input, text_input])

    filled = await filler.fill_with_dummy(page)
    assert filled == 2
    email_input.fill.assert_awaited_once()
    text_input.fill.assert_awaited_once()
