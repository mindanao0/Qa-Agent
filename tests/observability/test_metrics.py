"""Tests for AgentMetrics."""
from __future__ import annotations
import pytest

from src.observability.metrics import AgentMetrics


@pytest.fixture(autouse=True)
def reset_metrics():
    AgentMetrics.instance().reset()
    yield
    AgentMetrics.instance().reset()


def test_singleton_returns_same_instance():
    a = AgentMetrics.instance()
    b = AgentMetrics.instance()
    assert a is b


def test_increment_counter():
    m = AgentMetrics.instance()
    m.increment("spans_emitted")
    m.increment("spans_emitted")
    snapshot = m.export()
    assert snapshot["spans_emitted"]["count"] == 2


def test_record_histogram():
    m = AgentMetrics.instance()
    m.record("latency_ms", 100.0)
    m.record("latency_ms", 200.0)
    m.record("latency_ms", 300.0)
    snap = m.export()
    assert "latency_ms" in snap
    assert "p50" in snap["latency_ms"]
    assert "p95" in snap["latency_ms"]
    assert "p99" in snap["latency_ms"]


def test_reset_clears_all():
    m = AgentMetrics.instance()
    m.increment("x", 5)
    m.record("y", 1.0)
    m.reset()
    assert m.export() == {}


def test_increment_by_value():
    m = AgentMetrics.instance()
    m.increment("nodes", 3)
    snap = m.export()
    assert snap["nodes"]["count"] == 3
