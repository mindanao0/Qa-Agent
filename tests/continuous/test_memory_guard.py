"""Tests for MemoryGuard (Sprint 11 RSS monitor)."""
from __future__ import annotations

import pytest

from src.continuous.memory_guard import MemoryGuard


def test_is_stable_true_before_baseline_snapshot():
    guard = MemoryGuard()
    assert guard.is_stable() is True


def test_growth_is_zero_before_baseline_snapshot():
    guard = MemoryGuard()
    assert guard.growth_mb() == 0.0


def test_snapshot_baseline_returns_positive_rss():
    guard = MemoryGuard()
    baseline = guard.snapshot_baseline()
    assert baseline > 0.0


def test_current_rss_is_positive():
    guard = MemoryGuard()
    assert guard.current_rss_mb() > 0.0


def test_is_stable_false_when_growth_exceeds_limit(monkeypatch):
    guard = MemoryGuard(max_growth_mb=100.0)
    guard.snapshot_baseline()
    baseline = guard._baseline_rss
    assert baseline is not None
    monkeypatch.setattr(guard, "current_rss_mb", lambda: baseline + 150.0)
    assert guard.is_stable() is False
    assert guard.growth_mb() == pytest.approx(150.0)


def test_is_stable_true_when_growth_within_limit(monkeypatch):
    guard = MemoryGuard(max_growth_mb=200.0)
    guard.snapshot_baseline()
    baseline = guard._baseline_rss
    assert baseline is not None
    monkeypatch.setattr(guard, "current_rss_mb", lambda: baseline + 50.0)
    assert guard.is_stable() is True
    assert guard.growth_mb() == pytest.approx(50.0)
