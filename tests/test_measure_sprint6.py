"""Unit tests for measure_sprint6 helper functions (no Ollama required)."""
import json
import pathlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import audit.sprint6.measure_sprint6 as m


def test_write_results_creates_json(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_OUTPUT_PATH", tmp_path / "sprint6_results.json")
    result = m._write_results(
        ast_functions_parsed=12,
        tests_generated=11,
        test_pass_rate=0.8,
        metamorphic_pairs=4,
        regression=False,
        sprint6_status="PASS",
    )
    assert result["sprint6_status"] == "PASS"
    data = json.loads((tmp_path / "sprint6_results.json").read_text())
    assert data["ast_functions_parsed"] == 12
    assert data["test_pass_rate"] == 0.8


def test_write_results_rounds_pass_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_OUTPUT_PATH", tmp_path / "sprint6_results.json")
    m._write_results(0, 0, 0.123456789, 0, True, "FAIL")
    data = json.loads((tmp_path / "sprint6_results.json").read_text())
    assert data["test_pass_rate"] == 0.123457  # rounded to 6 decimals


@pytest.mark.asyncio
async def test_collect_specs_skips_missing_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "TARGET_MODULES", [
        tmp_path / "nonexistent.py",
    ])
    specs = await m._collect_specs()
    assert specs == []


@pytest.mark.asyncio
async def test_collect_specs_parses_real_module(monkeypatch):
    """Parse a real existing module — sfg.py should yield >= 5 functions."""
    monkeypatch.setattr(m, "TARGET_MODULES", [
        pathlib.Path("src/contractskill/sfg.py"),
    ])
    specs = await m._collect_specs()
    # NOTE: asserts against live sfg.py — update threshold if sfg public API shrinks below 5
    assert len(specs) >= 5


@pytest.mark.asyncio
async def test_judge_with_revision_accepts_good_test():
    from src.codetest.generator import GeneratedTest
    from src.codetest.judge import JudgeResult

    good_test = GeneratedTest(
        test_id="t1", func_id="f1",
        test_code="def test_foo():\n    assert True",
        test_type="happy_path", metamorphic_relation=None,
    )

    mock_judge = AsyncMock()
    mock_judge.judge = AsyncMock(return_value=JudgeResult(grade="acceptable", feedback=""))
    mock_generator = AsyncMock()
    specs_by_id = {}

    result = await m._judge_with_revision(mock_judge, mock_generator, specs_by_id, [good_test])
    assert len(result) == 1
    assert result[0].test_id == "t1"


@pytest.mark.asyncio
async def test_judge_with_revision_rejects_bad_test():
    from src.codetest.generator import GeneratedTest
    from src.codetest.judge import JudgeResult

    bad_test = GeneratedTest(
        test_id="t2", func_id="f2",
        test_code="def check_foo():\n    pass",
        test_type="happy_path", metamorphic_relation=None,
    )

    mock_judge = AsyncMock()
    mock_judge.judge = AsyncMock(return_value=JudgeResult(grade="reject", feedback="no assert"))
    mock_generator = AsyncMock()
    specs_by_id = {}

    result = await m._judge_with_revision(mock_judge, mock_generator, specs_by_id, [bad_test])
    assert result == []


@pytest.mark.asyncio
async def test_judge_with_revision_retries_needs_revision():
    from src.codetest.ast_parser import FunctionSpec, ArgSpec
    from src.codetest.generator import GeneratedTest
    from src.codetest.judge import JudgeResult

    spec = FunctionSpec(
        func_id="f3", module_path="x.py", func_name="foo",
        args=[], return_type=None, docstring=None, decorators=[], complexity=1,
    )
    test = GeneratedTest(
        test_id="t3", func_id="f3",
        test_code="def test_foo():\n    assert True",
        test_type="metamorphic", metamorphic_relation="bad relation",
    )
    revised = GeneratedTest(
        test_id="t3r", func_id="f3",
        test_code="def test_foo():\n    assert True",
        test_type="metamorphic", metamorphic_relation="f(x+1) > f(x)",
    )

    mock_judge = AsyncMock()
    mock_judge.judge = AsyncMock(side_effect=[
        JudgeResult(grade="needs_revision", feedback="bad relation"),
        JudgeResult(grade="acceptable", feedback=""),
    ])
    mock_generator = AsyncMock()
    mock_generator.regenerate_with_feedback = AsyncMock(return_value=revised)

    result = await m._judge_with_revision(mock_judge, mock_generator, {"f3": spec}, [test])
    assert len(result) == 1
    assert result[0].test_id == "t3r"


def test_write_results_fail_case(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_OUTPUT_PATH", tmp_path / "sprint6_results.json")
    result = m._write_results(0, 0, 0.0, 0, True, "FAIL")
    assert result["sprint6_status"] == "FAIL"
    assert result["regression"] is True
    data = json.loads((tmp_path / "sprint6_results.json").read_text())
    assert data["sprint6_status"] == "FAIL"


@pytest.mark.asyncio
async def test_main_no_regression_when_no_tests(tmp_path, monkeypatch):
    """When accepted_tests is empty, regression should be False (not a real regression)."""
    monkeypatch.setattr(m, "_OUTPUT_PATH", tmp_path / "sprint6_results.json")
    # _collect_specs returns specs, but all tests get rejected
    from src.codetest.ast_parser import FunctionSpec
    fake_spec = FunctionSpec(
        func_id="f1", module_path="x.py", func_name="foo",
        args=[], return_type=None, docstring=None, decorators=[], complexity=1,
    )
    from src.codetest.generator import GeneratedTest
    fake_test = GeneratedTest(
        test_id="t1", func_id="f1",
        test_code="def check_foo():\n    pass",
        test_type="happy_path", metamorphic_relation=None,
    )
    from src.codetest.judge import JudgeResult

    with (
        patch("audit.sprint6.measure_sprint6._collect_specs", AsyncMock(return_value=[fake_spec])),
        patch("audit.sprint6.measure_sprint6.PytestGenerator") as MockGen,
        patch("audit.sprint6.measure_sprint6.CodeJudge") as MockJudge,
        patch("audit.sprint6.measure_sprint6.TestExecutor"),
    ):
        mock_gen_instance = AsyncMock()
        mock_gen_instance.generate = AsyncMock(return_value=[fake_test])
        mock_gen_instance.regenerate_with_feedback = AsyncMock(return_value=None)
        MockGen.return_value = mock_gen_instance

        mock_judge_instance = AsyncMock()
        mock_judge_instance.judge = AsyncMock(return_value=JudgeResult(grade="reject", feedback="bad"))
        mock_judge_instance.close = AsyncMock()
        MockJudge.return_value = mock_judge_instance

        result = await m.main()

    assert result["regression"] is False  # no tests executed → not a regression
    assert result["tests_generated"] == 0
