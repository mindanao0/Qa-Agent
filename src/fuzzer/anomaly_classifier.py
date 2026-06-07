"""AnomalyClassifier — Sprint 13.

Distinguishes REAL anomalies from expected backend behavior. Grounded in live
probing of ``realworld.habsida.net`` (2026-06-03), which disproved the spec's
naive "200-on-SQLi → injection" rule: that backend handles SQLi/XSS safely
(200 + empty result), so flagging it would manufacture false positives. The
genuine anomaly class here is **5xx on malformed input** (e.g. ``limit=-1`` → 500,
slug ``../../../etc/passwd`` → 500).

See docs/superpowers/specs/2026-06-03-sprint13-advanced-api-fuzzing-design.md.
"""
from __future__ import annotations

import json
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict

# Statuses that are a CORRECT client-side rejection of bad input → NOT an anomaly.
_EXPECTED_4XX: frozenset[int] = frozenset({400, 401, 403, 404, 422})
_OK_STATUSES: frozenset[int] = frozenset({200, 201, 204})
_TIMEOUT_MS = 10_000.0

# A vector is an injection probe if it looks like path-traversal / SQLi / XSS.
_INJECTION_RE = re.compile(
    r"\.\./|\.\.\\|%2e%2e|/etc/|/passwd"          # traversal
    r"|'\s*or\s|--|union\s+select|;\s*drop|'1'='1"  # SQLi
    r"|<script|onerror\s*=|onload\s*=|javascript:|alert\(",  # XSS
    re.IGNORECASE,
)

# Markers that a SQL/templating error actually LEAKED into a 2xx body (real effect).
_SQL_ERROR_RE = re.compile(
    r"sql syntax|sqlite_|sqlite3|syntax error|ORA-\d|PG::|near \"|unterminated",
    re.IGNORECASE,
)


class AnomalyType(str, Enum):
    UNEXPECTED_STATUS = "unexpected_status"
    SCHEMA_DRIFT = "schema_drift"
    INJECTION_SIGNAL = "injection_signal"  # 5xx on injection vector, or 2xx with effect
    TIMEOUT = "timeout"
    EXPECTED_4XX = "expected_4xx"  # NOT an anomaly


class FuzzResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    vector: str
    status_code: int
    anomaly: bool
    anomaly_type: AnomalyType | None
    false_positive: bool
    duration_ms: float


def is_injection_vector(vector: str) -> bool:
    """True if the fuzz vector looks like a traversal / SQLi / XSS probe."""
    return bool(_INJECTION_RE.search(vector or ""))


def _missing_required(body: object, response_schema: dict | None) -> bool:
    """True if a 2xx dict body is missing a field genson marked as always-required."""
    if not response_schema or not isinstance(body, dict):
        return False
    required = response_schema.get("required")
    if not isinstance(required, list) or not required:
        return False
    return any(field not in body for field in required)


def _injection_has_effect(body: object) -> bool:
    """True if a 2xx response to an injection vector shows a real effect.

    On a secure backend an injection vector is reflected/ignored and the body is
    safe (e.g. an empty result set). It is only a real INJECTION_SIGNAL if a SQL
    error string leaked into the body — which indicates the payload reached the
    query layer.
    """
    if isinstance(body, dict):
        text = json.dumps(body)
    elif isinstance(body, str):
        text = body
    else:
        return False
    return bool(_SQL_ERROR_RE.search(text))


class AnomalyClassifier:
    """No required constructor args."""

    def classify(
        self,
        endpoint: str,
        vector: str,
        status_code: int,
        duration_ms: float,
        body: object = None,
        response_schema: dict | None = None,
        timed_out: bool = False,
    ) -> FuzzResult:
        injection = is_injection_vector(vector)
        anomaly = False
        atype: AnomalyType | None = None
        false_positive = False

        if timed_out or duration_ms > _TIMEOUT_MS:
            anomaly, atype = True, AnomalyType.TIMEOUT

        elif status_code >= 500:
            anomaly = True
            atype = AnomalyType.INJECTION_SIGNAL if injection else AnomalyType.UNEXPECTED_STATUS

        elif status_code in _EXPECTED_4XX:
            # Correct rejection of bad input — the backend behaving as designed.
            anomaly, atype = False, AnomalyType.EXPECTED_4XX

        elif status_code in _OK_STATUSES:
            if _missing_required(body, response_schema):
                anomaly, atype = True, AnomalyType.SCHEMA_DRIFT
            elif injection and _injection_has_effect(body):
                anomaly, atype = True, AnomalyType.INJECTION_SIGNAL
            # else: safe 2xx (incl. safely-handled injection) → not an anomaly.

        # else: 3xx / other → not classified as anomaly.

        return FuzzResult(
            endpoint=endpoint,
            vector=(vector or "")[:100],
            status_code=status_code,
            anomaly=anomaly,
            anomaly_type=atype,
            false_positive=false_positive,
            duration_ms=round(duration_ms, 2),
        )


__all__ = [
    "AnomalyClassifier",
    "AnomalyType",
    "FuzzResult",
    "is_injection_vector",
]
