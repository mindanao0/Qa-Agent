"""MemoryGuard — Sprint 11.

RSS monitor that guards against a memory leak across continuous-loop cycles.
``snapshot_baseline()`` is called ONCE before cycle 1; ``is_stable()`` then
reports whether RSS growth has stayed under ``max_growth_mb``.
"""
from __future__ import annotations

import psutil


class MemoryGuard:
    """No required constructor args."""

    def __init__(self, max_growth_mb: float = 200.0) -> None:
        self._baseline_rss: float | None = None
        self._max_growth = max_growth_mb
        self._proc = psutil.Process()

    def snapshot_baseline(self) -> float:
        """Call once before cycle 1 starts. Returns baseline RSS in MB."""
        self._baseline_rss = self._proc.memory_info().rss / 1024 / 1024
        return self._baseline_rss

    def current_rss_mb(self) -> float:
        return self._proc.memory_info().rss / 1024 / 1024

    def is_stable(self) -> bool:
        """True if current RSS - baseline < max_growth_mb."""
        if self._baseline_rss is None:
            return True
        return (self.current_rss_mb() - self._baseline_rss) < self._max_growth

    def growth_mb(self) -> float:
        if self._baseline_rss is None:
            return 0.0
        return self.current_rss_mb() - self._baseline_rss


__all__ = ["MemoryGuard"]
