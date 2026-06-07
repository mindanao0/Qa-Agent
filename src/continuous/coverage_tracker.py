"""CoveragePlateau detector — Sprint 11.

Tracks which SFG ``node_id``s have been seen across cycles and detects a
coverage plateau (N consecutive cycles with zero new states), one of the
loop's stop conditions.
"""
from __future__ import annotations


class CoverageTracker:
    """No required constructor args."""

    def __init__(self, plateau_threshold: int = 2) -> None:
        self._seen_nodes: set[str] = set()
        self._new_per_cycle: list[int] = []
        self._plateau_thresh = plateau_threshold

    def update(self, node_ids: list[str]) -> int:
        """Record this cycle's node_ids; return count of NEW (unseen) node_ids.

        Duplicates within a single cycle's ``node_ids`` are counted once — the
        metric is *distinct* new states, so a state grounded twice in one cycle
        is one new state, not two.
        """
        new_ids = {n for n in node_ids if n not in self._seen_nodes}
        self._seen_nodes.update(node_ids)
        self._new_per_cycle.append(len(new_ids))
        return len(new_ids)

    def is_plateau(self) -> bool:
        """True if the last ``plateau_threshold`` cycles all had 0 new states."""
        if len(self._new_per_cycle) < self._plateau_thresh:
            return False
        return all(n == 0 for n in self._new_per_cycle[-self._plateau_thresh:])


__all__ = ["CoverageTracker"]
