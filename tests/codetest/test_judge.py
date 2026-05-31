import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.codetest.generator import GeneratedTest
from src.codetest.judge import CodeJudge, JudgeResult, MetamorphicCheckResult


def _make_test(
    code: str = "def test_foo():\n    assert 1 + 1 == 2",
    test_type: str = "happy_path",
    metamorphic_relation: str | None = None,
) -> GeneratedTest:
    return GeneratedTest(
        test_id="t001",
        func_id="abc1234567",
        test_code=code,
        test_type=test_type,
        metamorphic_relation=metamorphic_relation,
    )


@pytest.mark.asyncio
async def test_acceptable_test():
    judge = CodeJudge()
    result = await judge.judge(_make_test())
    assert result.grade == "acceptable"
    assert result.feedback == ""
    await judge.close()


@pytest.mark.asyncio
async def test_reject_no_assert():
    judge = CodeJudge()
    result = await judge.judge(_make_test(code="def test_no_assert():\n    pass"))
    assert result.grade == "reject"
    assert "assert" in result.feedback.lower()
    await judge.close()


@pytest.mark.asyncio
async def test_reject_sleep():
    judge = CodeJudge()
    result = await judge.judge(
        _make_test(code="import time\ndef test_sleep():\n    time.sleep(1)\n    assert True")
    )
    assert result.grade == "reject"
    assert "sleep" in result.feedback.lower()
    await judge.close()


@pytest.mark.asyncio
async def test_reject_bad_func_name():
    judge = CodeJudge()
    result = await judge.judge(_make_test(code="def check_foo():\n    assert True"))
    assert result.grade == "reject"
    assert "test_" in result.feedback.lower()
    await judge.close()


@pytest.mark.asyncio
async def test_non_metamorphic_skips_llm_check():
    judge = CodeJudge()
    with patch.object(judge, "_check_metamorphic", new_callable=AsyncMock) as mock_check:
        result = await judge.judge(_make_test(test_type="happy_path", metamorphic_relation=None))
    mock_check.assert_not_called()
    assert result.grade == "acceptable"
    await judge.close()


@pytest.mark.asyncio
async def test_metamorphic_valid_calls_llm():
    judge = CodeJudge()
    mock_result = MetamorphicCheckResult(valid=True, reason="logically sound")
    with patch.object(judge._client, "create_structured", AsyncMock(return_value=mock_result)) as mock_cs:
        result = await judge.judge(
            _make_test(
                test_type="metamorphic",
                metamorphic_relation="f(x+1) > f(x) for positive x",
            )
        )
    assert result.grade == "acceptable"
    mock_cs.assert_called_once()
    await judge.close()


@pytest.mark.asyncio
async def test_metamorphic_invalid_is_needs_revision():
    judge = CodeJudge()
    mock_result = MetamorphicCheckResult(valid=False, reason="relation is incorrect")
    with patch.object(judge._client, "create_structured", AsyncMock(return_value=mock_result)):
        result = await judge.judge(
            _make_test(
                test_type="metamorphic",
                metamorphic_relation="f(x+1) < f(x) always",
            )
        )
    assert result.grade == "needs_revision"
    assert "relation" in result.feedback.lower()
    await judge.close()


@pytest.mark.asyncio
async def test_metamorphic_llm_failure_passes_check():
    from src.llm.instructor_client import StructuredGenerationError
    judge = CodeJudge()
    with patch.object(
        judge._client,
        "create_structured",
        AsyncMock(side_effect=StructuredGenerationError("h", ValueError("fail")))
    ):
        result = await judge.judge(
            _make_test(
                test_type="metamorphic",
                metamorphic_relation="f(x) > 0 for all x",
            )
        )
    # LLM failure should fail open → acceptable
    assert result.grade == "acceptable"
    await judge.close()


@pytest.mark.asyncio
async def test_reject_takes_priority_over_metamorphic():
    """Checks 1-3 rejection skips LLM check 4 even for metamorphic tests."""
    judge = CodeJudge()
    with patch.object(judge, "_check_metamorphic", new_callable=AsyncMock) as mock_check:
        result = await judge.judge(
            _make_test(
                code="def check_bad():\n    pass",  # fails check 1 (no assert) and check 3 (no test_)
                test_type="metamorphic",
                metamorphic_relation="f(x+1) > f(x)",
            )
        )
    mock_check.assert_not_called()
    assert result.grade == "reject"
    await judge.close()


@pytest.mark.asyncio
async def test_helper_function_before_test_is_acceptable():
    """Check 3 should pass if ANY def starts with test_, not just the first."""
    code = "def _build_payload(x):\n    return x\n\ndef test_foo():\n    assert _build_payload(1) == 1\n"
    judge = CodeJudge()
    result = await judge.judge(_make_test(code=code))
    assert result.grade == "acceptable"
    await judge.close()


@pytest.mark.asyncio
async def test_empty_metamorphic_relation_skips_llm():
    """Empty string metamorphic_relation should not call LLM."""
    judge = CodeJudge()
    with patch.object(judge, "_check_metamorphic", new_callable=AsyncMock) as mock_check:
        result = await judge.judge(
            _make_test(
                test_type="metamorphic",
                metamorphic_relation="",  # empty string — falsy
            )
        )
    mock_check.assert_not_called()
    assert result.grade == "acceptable"
    await judge.close()
