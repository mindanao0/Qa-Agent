"""ThroughputMeter — Sprint 15.

Measures tasks-per-second (TPS) for a sequential baseline vs a parallel
(fan-out) run and exposes the throughput gain. All numbers come from real
wall-clock durations recorded by the caller — nothing is estimated or
hardcoded (see RULES in the Sprint 15 spec).
"""
from __future__ import annotations


class ThroughputMeter:
    """No required constructor args.

    Usage::

        meter = ThroughputMeter()
        meter.record_sequential(12, seq_seconds)
        meter.record_parallel(12, par_seconds, n_workers=3)
        gain = meter.throughput_gain()   # parallel_tps / sequential_tps
    """

    def __init__(self) -> None:
        self._sequential_tps: float | None = None
        self._parallel_tps: float | None = None
        self._n_workers: int = 0

    @staticmethod
    def _tps(n_tasks: int, duration_s: float) -> float:
        """Tasks-per-second; 0.0 when no tasks or non-positive duration."""
        if n_tasks <= 0 or duration_s <= 0.0:
            return 0.0
        return n_tasks / duration_s

    def record_sequential(self, n_tasks: int, duration_s: float) -> float:
        """Record + return the sequential TPS = ``n_tasks / duration_s``."""
        self._sequential_tps = self._tps(n_tasks, duration_s)
        return self._sequential_tps

    def record_parallel(
        self, n_tasks: int, duration_s: float, n_workers: int = 0
    ) -> float:
        """Record + return the parallel TPS = ``n_tasks / duration_s``."""
        self._parallel_tps = self._tps(n_tasks, duration_s)
        if n_workers:
            self._n_workers = n_workers
        return self._parallel_tps

    def throughput_gain(self) -> float:
        """Return ``parallel_tps / sequential_tps``.

        Requires both records present and a non-zero sequential baseline.
        """
        if self._sequential_tps is None or self._parallel_tps is None:
            raise ValueError(
                "throughput_gain requires both record_sequential and "
                "record_parallel to have been called"
            )
        if self._sequential_tps <= 0.0:
            raise ValueError("sequential TPS is zero; cannot compute gain")
        return self._parallel_tps / self._sequential_tps

    @property
    def sequential_tps(self) -> float | None:
        return self._sequential_tps

    @property
    def parallel_tps(self) -> float | None:
        return self._parallel_tps

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def to_dict(self) -> dict:
        gain: float | None
        try:
            gain = self.throughput_gain()
        except ValueError:
            gain = None
        return {
            "sequential_tps": self._sequential_tps,
            "parallel_tps": self._parallel_tps,
            "throughput_gain": gain,
            "n_workers": self._n_workers,
        }


__all__ = ["ThroughputMeter"]
