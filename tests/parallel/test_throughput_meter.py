"""Tests for ThroughputMeter (Sprint 15 — pure timing math, no browser/LLM)."""
from __future__ import annotations

import pytest

from src.parallel.throughput_meter import ThroughputMeter


def test_record_sequential_returns_tps():
    meter = ThroughputMeter()
    assert meter.record_sequential(12, 6.0) == pytest.approx(2.0)
    assert meter.sequential_tps == pytest.approx(2.0)


def test_record_parallel_returns_tps_and_workers():
    meter = ThroughputMeter()
    assert meter.record_parallel(12, 3.0, n_workers=3) == pytest.approx(4.0)
    assert meter.parallel_tps == pytest.approx(4.0)
    assert meter.n_workers == 3


def test_throughput_gain_is_ratio():
    meter = ThroughputMeter()
    meter.record_sequential(12, 6.0)   # 2.0 tps
    meter.record_parallel(12, 2.0, n_workers=3)  # 6.0 tps
    assert meter.throughput_gain() == pytest.approx(3.0)


def test_throughput_gain_requires_both_records():
    meter = ThroughputMeter()
    meter.record_sequential(12, 6.0)
    with pytest.raises(ValueError):
        meter.throughput_gain()


def test_throughput_gain_rejects_zero_baseline():
    meter = ThroughputMeter()
    meter.record_sequential(0, 6.0)   # 0 tasks -> 0 tps
    meter.record_parallel(12, 2.0)
    with pytest.raises(ValueError):
        meter.throughput_gain()


def test_tps_zero_on_nonpositive_duration():
    meter = ThroughputMeter()
    assert meter.record_sequential(12, 0.0) == 0.0
    assert meter.record_parallel(12, -1.0) == 0.0


def test_to_dict_has_expected_keys_and_gain_none_when_incomplete():
    meter = ThroughputMeter()
    meter.record_sequential(12, 6.0)
    d = meter.to_dict()
    assert set(d) == {"sequential_tps", "parallel_tps", "throughput_gain", "n_workers"}
    assert d["throughput_gain"] is None  # parallel not recorded yet


def test_to_dict_gain_present_when_complete():
    meter = ThroughputMeter()
    meter.record_sequential(12, 6.0)
    meter.record_parallel(12, 2.0, n_workers=3)
    d = meter.to_dict()
    assert d["throughput_gain"] == pytest.approx(3.0)
    assert d["n_workers"] == 3
