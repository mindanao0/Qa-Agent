"""OTelTracer — local JSONL span emitter, no external collector needed."""
from __future__ import annotations

import contextlib
import contextvars
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TRACES_DEFAULT = Path("logs/traces")

# Context var so nested spans can pick up parent_span_id
_active_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_span_id", default=None
)


class OTelTracer:
    """No required constructor args. Lazy-init on first span."""

    def __init__(self, traces_dir: Path | None = None) -> None:
        self._dir = traces_dir or _TRACES_DEFAULT
        self._pending: list[dict[str, Any]] = []
        self._trace_id: str = uuid.uuid4().hex

    @contextlib.asynccontextmanager
    async def span(self, name: str, **attrs: Any):
        self._dir.mkdir(parents=True, exist_ok=True)
        span_id = uuid.uuid4().hex
        parent_span_id = _active_span_id.get()
        token = _active_span_id.set(span_id)
        start_ms = time.time() * 1000
        error = False
        try:
            yield span_id
        except Exception:
            error = True
            raise
        finally:
            end_ms = time.time() * 1000
            _active_span_id.reset(token)
            entry: dict[str, Any] = {
                "trace_id": self._trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "name": name,
                "start_ms": round(start_ms, 3),
                "end_ms": round(end_ms, 3),
                "attrs": attrs,
            }
            if error:
                entry["error"] = True
            self._pending.append(entry)
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            out_path = self._dir / f"otel_{date_str}.jsonl"
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    def flush(self) -> int:
        count = len(self._pending)
        self._pending.clear()
        return count

    @property
    def active_span_id(self) -> str | None:
        return _active_span_id.get()

    @property
    def trace_id(self) -> str:
        return self._trace_id
