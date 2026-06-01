"""Tests for StructuredLogger."""
from __future__ import annotations
import json
import uuid
from pathlib import Path

import pytest

from src.observability.structured_logger import LogEntry, StructuredLogger


@pytest.fixture
def tmp_logger(tmp_path: Path) -> StructuredLogger:
    run_id = uuid.uuid4().hex
    return StructuredLogger(logs_dir=tmp_path, run_id=run_id)


def test_log_entry_has_required_fields(tmp_logger: StructuredLogger, tmp_path: Path):
    tmp_logger.info("SFGCrawler", "crawl_start", duration_ms=12.5, payload={"url": "http://x"})
    files = list(tmp_path.glob("run_*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text().strip().splitlines()[0])
    required = {"ts", "level", "component", "event", "trace_id", "span_id", "run_id", "duration_ms", "payload"}
    assert required <= entry.keys()


def test_log_entry_fields_count_gte_8(tmp_logger: StructuredLogger, tmp_path: Path):
    tmp_logger.warn("Judge", "low_confidence")
    files = list(tmp_path.glob("run_*.jsonl"))
    entry = json.loads(files[0].read_text().strip().splitlines()[0])
    assert len(entry) >= 8


def test_log_entry_forbids_extra_fields():
    with pytest.raises(Exception):
        LogEntry(
            ts=1.0, level="INFO", component="X", event="e",
            trace_id="t", span_id="s", run_id="r",
            duration_ms=None, payload={}, unexpected_field="bad"
        )


def test_multiple_entries_all_flushed(tmp_logger: StructuredLogger, tmp_path: Path):
    for i in range(3):
        tmp_logger.info("Comp", f"event_{i}")
    files = list(tmp_path.glob("run_*.jsonl"))
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 3


def test_error_level_logged(tmp_logger: StructuredLogger, tmp_path: Path):
    tmp_logger.error("Executor", "script_failed", payload={"exit_code": 1})
    files = list(tmp_path.glob("run_*.jsonl"))
    entry = json.loads(files[0].read_text().strip().splitlines()[0])
    assert entry["level"] == "ERROR"
