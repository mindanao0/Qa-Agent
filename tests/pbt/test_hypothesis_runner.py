"""Sprint 14 — HypothesisRunner unit tests.

Pure build/parse logic is network-free and fast. One hermetic end-to-end test
(no Ollama; spawns ``uv run pytest``) proves real counterexample extraction.
"""
import ast

import pytest

from src.codetest.executor import SecurityASTChecker
from src.pbt.hypothesis_runner import FUNCTION_TARGETS, HypothesisRunner
from src.pbt.invariant_extractor import Invariant, fallback_invariant


def _inv(**kw):
    base = dict(
        invariant_id="abc1234567",
        source="function",
        source_id="_words_relate",
        description="symmetric",
        property_type="commutative",
        hypothesis_strategy="st.tuples(st.integers(), st.integers())",
    )
    base.update(kw)
    return Invariant(**base)


# ── build ────────────────────────────────────────────────────────────────────
def test_build_function_test_is_valid_python_and_security_clean():
    code = HypothesisRunner()._build_test(_inv())
    ast.parse(code)  # raises if invalid python
    assert SecurityASTChecker.check(code) is None  # not blocked
    assert "@given" in code and "@settings(max_examples=50" in code
    assert "_words_relate" in code


def test_build_endpoint_test_uses_base_url_and_is_security_clean():
    inv = _inv(
        source="endpoint",
        source_id="/api/articles?limit",
        property_type="invariant_output",
        hypothesis_strategy="st.integers(min_value=-3, max_value=1000)",
    )
    code = HypothesisRunner(base_url="https://example.test")._build_test(inv)
    ast.parse(code)
    assert SecurityASTChecker.check(code) is None
    assert "https://example.test" in code
    assert "/api/articles" in code and "limit" in code
    assert "status < 500" in code
    # must send a browser User-Agent — habsida 403-blocks the default urllib UA,
    # which would make the test vacuously pass against a bot-block page.
    assert "User-Agent" in code


def test_function_targets_registry_yields_at_least_ten_invariants():
    pairs = [(n, pt) for n, t in FUNCTION_TARGETS.items() for pt in t["props"]]
    assert len(pairs) >= 10


# ── parse (against the REAL Hypothesis 6.155 output captured empirically) ──────
def test_parse_falsifying_strips_pytest_gutter_multiline():
    out = (
        "E   Falsifying example: test_fail(\n"
        "E       value=0,\n"
        "E   )\n"
    )
    assert HypothesisRunner()._parse_falsifying(out) == "0"


def test_parse_falsifying_single_line():
    out = "Falsifying example: test_abc(t='0')"
    assert HypothesisRunner()._parse_falsifying(out) == "'0'"


def test_parse_falsifying_returns_none_when_absent():
    assert HypothesisRunner()._parse_falsifying("1 passed in 0.2s") is None


def test_parse_examples_run_sums_passing_and_failing_over_phases():
    stats = (
        "    - 9 passing examples, 1 failing examples, 0 invalid examples\n"
        "    - 0 passing examples, 1 failing examples, 0 invalid examples\n"
    )
    assert HypothesisRunner()._parse_examples_run(stats) == 11


def test_parse_examples_run_passing_run():
    stats = "    - 50 passing examples, 0 failing examples, 0 invalid examples\n"
    assert HypothesisRunner()._parse_examples_run(stats) == 50


def test_parse_reproduce_blob():
    out = (
        "You can reproduce this example by temporarily adding "
        "@reproduce_failure('6.155.1', b'AEEA') as a decorator on your test case"
    )
    assert HypothesisRunner()._parse_reproduce(out) == "@reproduce_failure('6.155.1', b'AEEA')"


# ── hermetic end-to-end (no Ollama; real subprocess) ──────────────────────────
async def test_run_finds_real_counterexample_and_passes_holding():
    holding = _inv()  # _words_relate / commutative — genuinely holds
    failing = fallback_invariant()  # _meaningful_words non-empty — genuinely fails
    results = await HypothesisRunner().run([holding, failing])
    by = {r.invariant_id: r for r in results}

    h = by[holding.invariant_id]
    assert h.passed is True and h.counterexample_found is False
    assert h.examples_run >= 1

    f = by[failing.invariant_id]
    assert f.counterexample_found is True
    assert f.passed is False
    assert f.counterexample is not None
    assert 0 < f.counterexample_size <= 10
