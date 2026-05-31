from __future__ import annotations

import ast

import pytest

from src.core.bft_consensus import (
    ASTNormalizer,
    BFTSecurityHalt,
    BFTVoter,
    PlanVoter,
    _NameCanonicalizer,
)
from src.llm.schemas import TestPlan, TestStep


def test_normalizer_strips_docstrings() -> None:
    with_docs = '''"""Module docstring."""

def foo():
    """Function docstring."""
    return 1
'''
    without_docs = '''def foo():
    return 1
'''
    n = ASTNormalizer()
    assert n.normalize(with_docs) == n.normalize(without_docs)


def test_normalizer_canonicalizes_local_names() -> None:
    src = "x = 1\ny = x + 2\n"
    out = ASTNormalizer().normalize(src)
    assert out is not None
    assert "var_1" in out
    assert "var_2" in out
    assert "x" not in out
    assert "y" not in out


def test_normalizer_preserves_protected_names() -> None:
    src = 'page.goto("/")\nexpect(page).to_have_url("/")\n'
    out = ASTNormalizer().normalize(src)
    assert out is not None
    assert "page" in out
    assert "expect" in out


def test_normalizer_returns_none_on_syntax_error() -> None:
    assert ASTNormalizer().normalize("def broken(:") is None


def test_normalizer_raises_security_halt_on_remap() -> None:
    # ASTNormalizer.normalize catches the halt internally and returns None
    assert ASTNormalizer().normalize("page = 1") is None
    # Direct transformer call must raise BFTSecurityHalt
    with pytest.raises(BFTSecurityHalt):
        _NameCanonicalizer().visit(ast.parse("page = 1"))


def test_voter_quorum_reached_two_identical() -> None:
    # Two semantically equivalent, textually different candidates
    c1 = '"""doc one."""\nx = 1\ny = x + 2\n'
    c2 = '"""different doc."""\na = 1\nb = a + 2\n'
    c3 = 'z = 99\nw = z * z\n'  # different shape
    winner = BFTVoter().vote([c1, c2, c3])
    assert winner in (c1, c2)


def test_voter_no_quorum_all_different() -> None:
    c1 = 'x = 1\n'
    c2 = 'page.goto("/a")\n'
    c3 = 'def foo():\n    return 42\n'
    assert BFTVoter().vote([c1, c2, c3]) is None


def test_voter_returns_none_when_all_none() -> None:
    assert BFTVoter().vote([None, None, None]) is None


# ── PlanVoter tests (Sprint 3 two-pass fix) ──────────────────────────────────


def _make_plan(title: str = "Verify login", domain: str = "authentication") -> TestPlan:
    return TestPlan(
        title=title,
        requirement_summary="Login page renders with email and password fields",
        estimated_complexity="low",
        domain=domain,
        steps=[
            TestStep(
                step_number=1,
                description="Visit login page",
                action="visit",
                expected_result="Login form renders",
                role="admin",
                preconditions=[],
            ),
            TestStep(
                step_number=2,
                description="See email field",
                action="expect",
                expected_result="Email field is visible",
                role="admin",
                preconditions=[],
            ),
        ],
    )


def test_plan_voter_quorum_two_identical_plans() -> None:
    p1 = _make_plan()
    p2 = _make_plan()  # canonically identical to p1
    p3 = _make_plan(title="Different title")  # different shape
    winner = PlanVoter().vote([p1, p2, p3])
    assert winner is not None
    assert winner.title == p1.title


def test_plan_voter_no_quorum_all_different() -> None:
    p1 = _make_plan(title="Verify A")
    p2 = _make_plan(title="Verify B")
    p3 = _make_plan(title="Verify C")
    assert PlanVoter().vote([p1, p2, p3]) is None


def test_plan_canonical_is_deterministic() -> None:
    voter = PlanVoter()
    p1 = _make_plan()
    p2 = _make_plan()
    assert voter._canonical(p1) == voter._canonical(p2)
