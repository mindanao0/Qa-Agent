"""StructuredLogger — JSON log entries with ≥8 fields per entry."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.observability.tracer import _active_span_id

_LOGS_DEFAULT = Path("logs/structured")


class LogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ts: float
    level: Literal["DEBUG", "INFO", "WARN", "ERROR"]
    component: str
    event: str
    trace_id: str
    span_id: str
    run_id: str
    duration_ms: float | None
    payload: dict


class StructuredLogger:
    """Flush-on-every-write structured logger. No buffering."""

    def __init__(
        self,
        logs_dir: Path | None = None,
        run_id: str | None = None,
        trace_id: str = "",
    ) -> None:
        self._dir = logs_dir or _LOGS_DEFAULT
        self._run_id = run_id or uuid.uuid4().hex
        self._trace_id = trace_id

    def _write(
        self,
        level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
        component: str,
        event: str,
        duration_ms: float | None = None,
        payload: dict | None = None,
    ) -> LogEntry:
        self._dir.mkdir(parents=True, exist_ok=True)
        entry = LogEntry(
            ts=time.time(),
            level=level,
            component=component,
            event=event,
            trace_id=self._trace_id,
            span_id=_active_span_id.get() or "",
            run_id=self._run_id,
            duration_ms=duration_ms,
            payload=payload or {},
        )
        out_path = self._dir / f"run_{self._run_id}.jsonl"
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def debug(self, component: str, event: str, **kwargs) -> LogEntry:
        return self._write("DEBUG", component, event, **kwargs)

    def info(self, component: str, event: str, **kwargs) -> LogEntry:
        return self._write("INFO", component, event, **kwargs)

    def warn(self, component: str, event: str, **kwargs) -> LogEntry:
        return self._write("WARN", component, event, **kwargs)

    def error(self, component: str, event: str, **kwargs) -> LogEntry:
        return self._write("ERROR", component, event, **kwargs)

    def bind_trace(self, trace_id: str) -> None:
        self._trace_id = trace_id

    @property
    def run_id(self) -> str:
        return self._run_id
