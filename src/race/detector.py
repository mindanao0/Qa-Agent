"""ConflictDetector — post-run analysis of RaceResult list."""
from __future__ import annotations

from src.race.swarm import RaceResult


class ConflictDetector:
    """No required constructor args."""

    def analyze(self, results: list[RaceResult]) -> dict:
        if not results:
            return {
                "total_scenarios": 0,
                "conflicts_found": 0,
                "conflict_rate": 0.0,
                "worst_scenario_id": None,
                "interleaving_patterns": [],
            }

        conflicts = [r for r in results if r.conflict_found]
        worst = max(conflicts, key=lambda r: r.duration_ms) if conflicts else None
        all_timestamps = [ts for r in results for ts in r.interleaving]

        return {
            "total_scenarios": len(results),
            "conflicts_found": len(conflicts),
            "conflict_rate": round(len(conflicts) / len(results), 4),
            "worst_scenario_id": worst.scenario_id if worst else None,
            "interleaving_patterns": all_timestamps,
        }


__all__ = ["ConflictDetector"]
