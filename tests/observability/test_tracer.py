"""Tests for OTelTracer."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

import pytest

from src.observability.tracer import OTelTracer


@pytest.fixture
def tmp_tracer(tmp_path: Path) -> OTelTracer:
    tracer = OTelTracer(traces_dir=tmp_path)
    return tracer


def test_span_writes_jsonl(tmp_tracer: OTelTracer, tmp_path: Path):
    asyncio.run(_span_writes_jsonl(tmp_tracer, tmp_path))


async def _span_writes_jsonl(tracer: OTelTracer, tmp_path: Path):
    async with tracer.span("test.operation", model="gpt-fake", custom="val"):
        pass
    count = tracer.flush()
    assert count >= 1
    files = list(tmp_path.glob("otel_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[0])
    assert entry["name"] == "test.operation"
    assert "trace_id" in entry
    assert "span_id" in entry
    assert "start_ms" in entry
    assert "end_ms" in entry
    assert entry["attrs"]["model"] == "gpt-fake"


def test_span_nesting_produces_parent_span_id(tmp_tracer: OTelTracer, tmp_path: Path):
    asyncio.run(_span_nesting(tmp_tracer, tmp_path))


async def _span_nesting(tracer: OTelTracer, tmp_path: Path):
    async with tracer.span("outer"):
        async with tracer.span("inner"):
            pass
    tracer.flush()
    files = list(tmp_path.glob("otel_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    # Two spans: outer and inner (inner is written first since it finishes first)
    assert len(lines) == 2
    entries = [json.loads(l) for l in lines]
    inner = next(e for e in entries if e["name"] == "inner")
    outer = next(e for e in entries if e["name"] == "outer")
    assert inner["parent_span_id"] == outer["span_id"]


def test_flush_returns_count(tmp_tracer: OTelTracer):
    asyncio.run(_flush_count(tmp_tracer))


async def _flush_count(tracer: OTelTracer):
    async with tracer.span("a"):
        pass
    async with tracer.span("b"):
        pass
    count = tracer.flush()
    assert count == 2


def test_span_records_error_attr(tmp_tracer: OTelTracer, tmp_path: Path):
    asyncio.run(_span_error(tmp_tracer, tmp_path))


async def _span_error(tracer: OTelTracer, tmp_path: Path):
    try:
        async with tracer.span("err.op"):
            raise ValueError("boom")
    except ValueError:
        pass
    tracer.flush()
    files = list(tmp_path.glob("otel_*.jsonl"))
    lines = files[0].read_text().strip().splitlines()
    entry = json.loads(lines[0])
    assert entry.get("error") is True
