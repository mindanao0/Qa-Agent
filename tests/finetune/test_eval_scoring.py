"""Tests for eval scoring functions — no Ollama or GPU required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.eval_finetune import (
    EvalResult,
    get_scoring_type,
    score_completion,
)


class TestGetScoringType:
    def test_sprint6_pytest(self):
        assert get_scoring_type("sprint6_pytest") == "pytest"

    def test_sprint13_schema(self):
        assert get_scoring_type("sprint13_schema") == "schema"

    def test_sprint9_vitest(self):
        assert get_scoring_type("sprint9_vitest") == "vitest"

    def test_sprint5_hypothesis(self):
        assert get_scoring_type("sprint5_hypothesis") == "hypothesis"

    def test_sprint14_pbt(self):
        assert get_scoring_type("sprint14_pbt") == "hypothesis"

    def test_unknown_source_defaults_to_hypothesis(self):
        assert get_scoring_type("unknown_source") == "hypothesis"


class TestScoreCompletion:
    # -- pytest scoring --
    def test_pytest_valid_code(self):
        code = "import pytest\n\ndef test_add():\n    assert 1 + 1 == 2\n"
        result = score_completion("pytest", code)
        assert isinstance(result, bool)

    def test_pytest_must_have_def_test(self):
        code = "def helper(): pass"
        assert score_completion("pytest", code) is False

    def test_pytest_syntax_error_is_false(self):
        assert score_completion("pytest", "def test(: pass") is False

    def test_pytest_empty_is_false(self):
        assert score_completion("pytest", "") is False

    def test_pytest_with_test_func(self):
        code = "def test_something():\n    assert True\n"
        assert score_completion("pytest", code) is True

    # -- schema scoring --
    def test_schema_valid_json_object(self):
        schema = json.dumps({
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        })
        assert score_completion("schema", schema) is True

    def test_schema_invalid_json(self):
        assert score_completion("schema", "not json at all") is False

    def test_schema_empty_string(self):
        assert score_completion("schema", "") is False

    def test_schema_valid_json_array_passes(self):
        schema = json.dumps({"type": "array", "items": {"type": "string"}})
        assert score_completion("schema", schema) is True

    def test_schema_json_without_type_key(self):
        schema = json.dumps({"foo": "bar"})
        assert score_completion("schema", schema) is False

    # -- vitest scoring --
    def test_vitest_valid(self):
        code = (
            "describe('add', () => {\n"
            "  it('works', () => {\n"
            "    expect(1 + 1).toBe(2);\n"
            "  });\n"
            "});\n"
        )
        assert score_completion("vitest", code) is True

    def test_vitest_missing_describe_is_false(self):
        assert score_completion("vitest", "expect(1).toBe(1);") is False

    def test_vitest_missing_expect_is_false(self):
        assert score_completion("vitest", "describe('x', () => {});") is False

    def test_vitest_empty_is_false(self):
        assert score_completion("vitest", "") is False

    # -- hypothesis scoring --
    def test_hypothesis_valid_python(self):
        code = (
            "from hypothesis import given\nimport hypothesis.strategies as st\n\n"
            "@given(st.integers())\ndef test_prop(n):\n    assert n + 1 > n\n"
        )
        assert score_completion("hypothesis", code) is True

    def test_hypothesis_invalid_python(self):
        assert score_completion("hypothesis", "def test(: pass") is False

    def test_hypothesis_empty(self):
        assert score_completion("hypothesis", "") is False

    def test_hypothesis_valid_python_no_hypothesis_import(self):
        code = "def test_x():\n    x = 1\n    assert x == 1\n"
        assert score_completion("hypothesis", code) is True


class TestEvalResult:
    def test_pass_status_when_all_criteria_met(self):
        r = EvalResult(
            base_pass_rate=0.5,
            ft_pass_rate=0.6,
            quality_gain=0.1,
            training_loss_final=1.2,
            val_loss_final=1.8,
            vram_peak_mb=4800.0,
            oom_errors=0,
            finetune_status="PASS",
        )
        assert r.finetune_status == "PASS"

    def test_fail_status(self):
        r = EvalResult(
            base_pass_rate=0.5,
            ft_pass_rate=0.5,
            quality_gain=0.0,
            training_loss_final=2.0,
            val_loss_final=2.5,
            vram_peak_mb=4800.0,
            oom_errors=0,
            finetune_status="FAIL",
        )
        assert r.finetune_status == "FAIL"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            EvalResult(
                base_pass_rate=0.5,
                ft_pass_rate=0.6,
                quality_gain=0.1,
                training_loss_final=1.2,
                val_loss_final=1.8,
                vram_peak_mb=4800.0,
                oom_errors=0,
                finetune_status="PASS",
                unknown_field="x",
            )

    def test_none_fields_allowed(self):
        r = EvalResult(
            base_pass_rate=0.5,
            ft_pass_rate=0.6,
            quality_gain=0.1,
            training_loss_final=1.2,
            val_loss_final=None,
            vram_peak_mb=None,
            oom_errors=0,
            finetune_status="PASS",
        )
        assert r.val_loss_final is None
        assert r.vram_peak_mb is None

    def test_model_dump_json_round_trip(self):
        r = EvalResult(
            base_pass_rate=0.6,
            ft_pass_rate=0.7,
            quality_gain=0.1,
            training_loss_final=1.1,
            val_loss_final=1.5,
            vram_peak_mb=5000.0,
            oom_errors=0,
            finetune_status="PASS",
        )
        data = json.loads(r.model_dump_json())
        assert data["quality_gain"] == pytest.approx(0.1)
        assert data["finetune_status"] == "PASS"
