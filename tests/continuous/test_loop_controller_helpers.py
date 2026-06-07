"""Tests for the pure helpers in loop_controller (no browser / no LLM)."""
from __future__ import annotations

from src.continuous.loop_controller import (
    _ALLOWED_ACTIONS,
    combined_pass_rate,
    cycle_actions,
    cycle_web_hypotheses,
    heal_accounting,
)
from src.explorer.executor import HypothesisResult


# ── combined_pass_rate ──────────────────────────────────────────────────────


def test_pass_rate_zero_when_no_tests():
    assert combined_pass_rate(0, 0) == 0.0


def test_pass_rate_all_passed():
    assert combined_pass_rate(4, 0) == 1.0


def test_pass_rate_mixed():
    assert combined_pass_rate(3, 1) == 0.75


# ── heal_accounting ─────────────────────────────────────────────────────────


def _result(passed: bool, repair: bool) -> HypothesisResult:
    return HypothesisResult(
        hypothesis_id="h", passed=passed, repair_attempted=repair, steps_executed=1
    )


def test_heal_accounting_no_repairs():
    needed, healed = heal_accounting([_result(True, False), _result(False, False)])
    assert needed == 0
    assert healed == 0


def test_heal_accounting_counts_needed_and_healed():
    results = [
        _result(True, False),   # passed, no repair  -> neither
        _result(True, True),    # passed after repair -> needed + healed
        _result(False, True),   # repaired but still failed -> needed only
    ]
    needed, healed = heal_accounting(results)
    assert needed == 2
    assert healed == 1


# ── cycle_actions (deepening exploration plan) ──────────────────────────────


def test_cycle_actions_nonempty_and_valid():
    actions = cycle_actions(1)
    assert actions, "cycle 1 must have actions"
    for name, _arg in actions:
        assert name in _ALLOWED_ACTIONS


def test_cycle_actions_deepen_per_cycle():
    # Different cycles drive a different action plan -> chance of new states.
    assert cycle_actions(1) != cycle_actions(2)
    assert cycle_actions(2) != cycle_actions(3)


def test_cycle_actions_unknown_cycle_falls_back_to_record():
    # Beyond the scripted cycles, still produce a valid (record-only) plan.
    actions = cycle_actions(99)
    assert actions
    for name, _arg in actions:
        assert name in _ALLOWED_ACTIONS


# ── cycle_web_hypotheses ────────────────────────────────────────────────────


def test_cycle_web_hypotheses_have_start_url_and_steps():
    hyps = cycle_web_hypotheses(1, "https://example.com/")
    assert hyps
    for h in hyps:
        assert h.start_url == "https://example.com/"
        assert h.steps  # at least one step
        assert h.goal


def test_cycle_web_hypotheses_differ_across_cycles():
    c1 = {h.goal for h in cycle_web_hypotheses(1, "https://example.com/")}
    c2 = {h.goal for h in cycle_web_hypotheses(2, "https://example.com/")}
    assert c1 != c2


def test_cycle_web_hypotheses_include_skill_aligned_goals():
    """Across the run, goals include the two seeded ContractSkill flows so
    planner._match_skill can attribute them (drives skills_reused)."""
    all_goals = {
        h.goal.lower()
        for c in (1, 2, 3)
        for h in cycle_web_hypotheses(c, "https://example.com/")
    }
    joined = " | ".join(all_goals)
    assert "add" in joined and "todo" in joined
    assert "complete" in joined or "mark" in joined
