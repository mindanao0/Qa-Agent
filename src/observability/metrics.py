"""AgentMetrics — lightweight in-process counter/histogram, no Prometheus needed."""
from __future__ import annotations

import math
from typing import ClassVar


class AgentMetrics:
    """Singleton. Thread-safe for single-threaded async usage."""

    _instance: ClassVar[AgentMetrics | None] = None

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}

    @classmethod
    def instance(cls) -> AgentMetrics:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def increment(self, key: str, value: int = 1) -> None:
        self._counters[key] = self._counters.get(key, 0) + value

    def record(self, key: str, value: float) -> None:
        self._histograms.setdefault(key, []).append(value)

    def export(self) -> dict:
        result: dict = {}
        for key, count in self._counters.items():
            result[key] = {"count": count}
        for key, values in self._histograms.items():
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            result[key] = {
                "p50": self._percentile(sorted_vals, 50),
                "p95": self._percentile(sorted_vals, 95),
                "p99": self._percentile(sorted_vals, 99),
                "count": n,
            }
        return result

    def reset(self) -> None:
        self._counters.clear()
        self._histograms.clear()

    @staticmethod
    def _percentile(sorted_vals: list[float], p: int) -> float:
        if not sorted_vals:
            return 0.0
        idx = max(0, math.ceil(len(sorted_vals) * p / 100) - 1)
        return sorted_vals[idx]
