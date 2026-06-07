"""StopConditionEvaluator — Sprint 11.

Pure function (no LLM, no I/O) that decides whether the continuous loop should
terminate, and why.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycles / keep this module dependency-light
    from src.continuous.coverage_tracker import CoverageTracker
    from src.continuous.memory_guard import MemoryGuard


class StopReason(str, Enum):
    MAX_CYCLES = "max_cycles"
    COVERAGE_PLATEAU = "coverage_plateau"
    MEMORY_LIMIT = "memory_limit"
    MANUAL = "manual"


def evaluate(
    cycle: int,
    max_cycles: int,
    coverage: "CoverageTracker",
    memory: "MemoryGuard",
) -> StopReason | None:
    """Return the first matching stop reason, or None to keep looping.

    Precedence: max_cycles -> coverage_plateau -> memory_limit. ``max_cycles``
    of 0 means "infinite" (the cap is disabled).
    """
    if max_cycles > 0 and cycle >= max_cycles:
        return StopReason.MAX_CYCLES
    if coverage.is_plateau():
        return StopReason.COVERAGE_PLATEAU
    if not memory.is_stable():
        return StopReason.MEMORY_LIMIT
    return None


__all__ = ["StopReason", "evaluate"]
