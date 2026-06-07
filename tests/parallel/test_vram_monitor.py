"""Tests for VRAMMonitor (Sprint 15).

These run on any machine — when NVML is unavailable the monitor degrades
gracefully and the tests assert that contract. When NVML *is* available the
real-query branches are exercised opportunistically.
"""
from __future__ import annotations

import asyncio

import pytest

from src.parallel.vram_monitor import VRAMExceededError, VRAMMonitor


def test_constructs_without_args():
    monitor = VRAMMonitor()
    assert isinstance(monitor.available, bool)


def test_used_mb_none_when_unavailable():
    monitor = VRAMMonitor()
    if not monitor.available:
        assert monitor.used_mb() is None
    else:
        used = monitor.used_mb()
        assert used is None or used >= 0.0


def test_is_stable_true_when_unavailable():
    monitor = VRAMMonitor()
    if not monitor.available:
        assert monitor.is_stable() is True


async def test_monitor_during_returns_result_and_peak():
    monitor = VRAMMonitor()

    async def work() -> str:
        await asyncio.sleep(0.01)
        return "done"

    result, peak = await monitor.monitor_during(work(), poll_interval_s=0.01)
    assert result == "done"
    # peak is None when NVML unavailable, else a real reading.
    assert peak is None or peak >= 0.0


async def test_monitor_during_raises_on_exceed(monkeypatch):
    monitor = VRAMMonitor()
    # Force the "available + over limit" branch regardless of real hardware.
    monkeypatch.setattr(monitor, "used_mb", lambda: 5900.0)

    async def work() -> str:
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(VRAMExceededError):
        await monitor.monitor_during(work(), poll_interval_s=0.01)


async def test_monitor_during_propagates_coro_result_under_threshold(monkeypatch):
    monitor = VRAMMonitor()
    monkeypatch.setattr(monitor, "used_mb", lambda: 1234.0)

    async def work() -> int:
        await asyncio.sleep(0.01)
        return 42

    result, peak = await monitor.monitor_during(work(), poll_interval_s=0.01)
    assert result == 42
    assert peak == pytest.approx(1234.0)


def test_is_stable_threshold(monkeypatch):
    monitor = VRAMMonitor()
    monkeypatch.setattr(monitor, "used_mb", lambda: 5000.0)
    assert monitor.is_stable() is True
    monkeypatch.setattr(monitor, "used_mb", lambda: 5900.0)
    assert monitor.is_stable() is False
