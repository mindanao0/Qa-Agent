"""Tests for executor._is_ambiguous_click and _resolve_locator_with_llm."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.explorer.executor import (
    _is_ambiguous_click,
    _resolve_locator_with_llm,
    HypothesisExecutor,
)
from src.contractskill.compiler import ContractStep


# ─── _is_ambiguous_click ─────────────────────────────────────────────────────

@pytest.mark.parametrize("step", [
    'click "/login"',
    'click "/products/list"',
    'click /dashboard',
    'click submit_button',
    'click "submit_button"',
    'click select_product_page',
    'click "add_to_cart"',
])
def test_is_ambiguous_click_true(step):
    assert _is_ambiguous_click(step) is True


@pytest.mark.parametrize("step", [
    'click "Login"',
    'click "Add to Cart"',
    'click "Submit"',
    'fill "Email" with "test@test.com"',
    'navigate to https://example.com',
    'verify text: "Welcome"',
])
def test_is_ambiguous_click_false(step):
    assert _is_ambiguous_click(step) is False


# ─── _resolve_locator_with_llm ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_returns_llm_result():
    """LLM resolution maps path click → real element name."""
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value='a: "Login"\nbutton: "Register"')

    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.resolved_step = 'click "Login"'
    mock_client.create_structured = AsyncMock(return_value=mock_result)

    result = await _resolve_locator_with_llm('click "/login"', page, mock_client)
    assert result == 'click "Login"'


@pytest.mark.asyncio
async def test_resolve_returns_none_when_no_elements():
    """ถ้าหน้า page ไม่มี element เลย ต้อง return None."""
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value="")

    mock_client = MagicMock()
    result = await _resolve_locator_with_llm('click "/login"', page, mock_client)
    assert result is None
    mock_client.create_structured.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_returns_none_on_llm_failure():
    """LLM error ต้อง return None (ไม่ raise)."""
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value='button: "Login"')

    mock_client = MagicMock()
    mock_client.create_structured = AsyncMock(side_effect=Exception("LLM timeout"))

    result = await _resolve_locator_with_llm('click "submit_button"', page, mock_client)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_returns_none_when_evaluate_fails():
    """page.evaluate error ต้อง return None (ไม่ raise)."""
    page = AsyncMock()
    page.evaluate = AsyncMock(side_effect=Exception("page closed"))

    mock_client = MagicMock()
    result = await _resolve_locator_with_llm('click "/login"', page, mock_client)
    assert result is None


# ─── HypothesisExecutor._resolve_step ────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_step_replaces_ambiguous_locator(tmp_path):
    """_resolve_step ต้อง return ContractStep ใหม่ที่มี locator ถูกต้อง."""
    from src.contractskill.sfg import SFGStore

    store = SFGStore(db_path=tmp_path / "sfg.db")
    mock_client = MagicMock()
    executor = HypothesisExecutor(mock_client, store)

    page = AsyncMock()
    page.evaluate = AsyncMock(return_value='a: "Login"')
    mock_result = MagicMock()
    mock_result.resolved_step = 'click "Login"'
    mock_client.create_structured = AsyncMock(return_value=mock_result)

    step = ContractStep(
        step_number=1,
        action_type="click",
        locator='click "/login"',
        input_value=None,
        expected_state_hash="",
    )
    resolved = await executor._resolve_step(step, page)
    assert resolved.locator == 'click "Login"'
    assert resolved.step_number == 1


@pytest.mark.asyncio
async def test_resolve_step_passthrough_for_clear_locator(tmp_path):
    """step ที่ locator ชัดเจนอยู่แล้ว ต้อง return ตัวเดิมโดยไม่เรียก LLM."""
    from src.contractskill.sfg import SFGStore

    store = SFGStore(db_path=tmp_path / "sfg.db")
    mock_client = MagicMock()
    executor = HypothesisExecutor(mock_client, store)

    page = AsyncMock()
    step = ContractStep(
        step_number=2,
        action_type="click",
        locator='click "Login"',
        input_value=None,
        expected_state_hash="",
    )
    resolved = await executor._resolve_step(step, page)
    assert resolved is step
    mock_client.create_structured.assert_not_called()
