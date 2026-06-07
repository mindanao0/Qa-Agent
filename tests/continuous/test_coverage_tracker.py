"""Tests for CoverageTracker (Sprint 11 plateau detector)."""
from __future__ import annotations

from src.continuous.coverage_tracker import CoverageTracker


def test_update_returns_count_of_new_nodes():
    tracker = CoverageTracker()
    assert tracker.update(["a", "b", "c"]) == 3


def test_update_dedups_against_previously_seen():
    tracker = CoverageTracker()
    tracker.update(["a", "b"])
    # Only "c" is new this cycle.
    assert tracker.update(["b", "c"]) == 1


def test_update_counts_distinct_within_call():
    """Duplicates inside one cycle's node_ids count once (honest distinct states)."""
    tracker = CoverageTracker()
    assert tracker.update(["a", "a", "b"]) == 2


def test_new_per_cycle_records_each_cycle():
    tracker = CoverageTracker()
    tracker.update(["a", "b"])
    tracker.update(["b", "c"])
    assert tracker._new_per_cycle == [2, 1]


def test_is_plateau_false_before_threshold_cycles():
    tracker = CoverageTracker(plateau_threshold=2)
    tracker.update([])  # only one cycle recorded
    assert tracker.is_plateau() is False


def test_is_plateau_true_when_last_n_cycles_zero_new():
    tracker = CoverageTracker(plateau_threshold=2)
    tracker.update([])
    tracker.update([])
    assert tracker.is_plateau() is True


def test_is_plateau_false_when_recent_cycle_had_new():
    tracker = CoverageTracker(plateau_threshold=2)
    tracker.update(["a"])  # new=1
    tracker.update([])     # new=0
    # last two cycles = [1, 0] -> not all zero
    assert tracker.is_plateau() is False


def test_is_plateau_considers_only_last_n_cycles():
    tracker = CoverageTracker(plateau_threshold=2)
    tracker.update(["a"])  # 1
    tracker.update([])     # 0
    tracker.update([])     # 0  -> last two [0,0]
    assert tracker.is_plateau() is True
