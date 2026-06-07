"""Tests for StopConditionEvaluator (Sprint 11 pure stop logic)."""
from __future__ import annotations

from src.continuous.stop_conditions import StopReason, evaluate


class _FakeCoverage:
    def __init__(self, plateau: bool) -> None:
        self._plateau = plateau

    def is_plateau(self) -> bool:
        return self._plateau


class _FakeMemory:
    def __init__(self, stable: bool) -> None:
        self._stable = stable

    def is_stable(self) -> bool:
        return self._stable


def test_max_cycles_reached_returns_max_cycles():
    reason = evaluate(3, 3, _FakeCoverage(False), _FakeMemory(True))
    assert reason is StopReason.MAX_CYCLES


def test_max_cycles_zero_means_infinite():
    # cycle far above zero, but max_cycles=0 disables the cap.
    reason = evaluate(99, 0, _FakeCoverage(False), _FakeMemory(True))
    assert reason is None


def test_coverage_plateau_returns_plateau():
    reason = evaluate(1, 3, _FakeCoverage(True), _FakeMemory(True))
    assert reason is StopReason.COVERAGE_PLATEAU


def test_memory_unstable_returns_memory_limit():
    reason = evaluate(1, 3, _FakeCoverage(False), _FakeMemory(False))
    assert reason is StopReason.MEMORY_LIMIT


def test_no_condition_returns_none():
    reason = evaluate(1, 3, _FakeCoverage(False), _FakeMemory(True))
    assert reason is None


def test_max_cycles_takes_precedence_over_other_conditions():
    reason = evaluate(3, 3, _FakeCoverage(True), _FakeMemory(False))
    assert reason is StopReason.MAX_CYCLES


def test_stop_reason_values_are_strings():
    assert StopReason.MAX_CYCLES.value == "max_cycles"
    assert StopReason.COVERAGE_PLATEAU.value == "coverage_plateau"
    assert StopReason.MEMORY_LIMIT.value == "memory_limit"
    assert StopReason.MANUAL.value == "manual"
