"""InterleavingRecorder — Sprint 12.

Records per-agent action start/end timestamps + status so a race scenario's
interleaving can be reconstructed as a human-readable pattern, e.g.
``"agent0-start, agent1-start, agent1-end(422), agent0-end(200)"``.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    action: str
    started_at: float        # time.monotonic() ms
    ended_at: float
    status_code: int | None
    response_hash: str        # sha256[:8] of response body (or "")


class InterleavingRecorder:
    """No required constructor args. Records EVERY agent event (start + end)."""

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []

    def record(self, event: AgentEvent) -> None:
        self._events.append(event)

    def pattern(self) -> str:
        """Interleaved timeline of all agent start/end points, ordered by time.

        Each event contributes two points: ``<agent>-start`` at ``started_at`` and
        ``<agent>-end(<status>)`` at ``ended_at`` (the ``(<status>)`` suffix is
        omitted when ``status_code`` is None). Equal timestamps keep insertion order.
        """
        points: list[tuple[float, str]] = []
        for e in self._events:
            points.append((e.started_at, f"{e.agent_id}-start"))
            end_label = f"{e.agent_id}-end"
            if e.status_code is not None:
                end_label += f"({e.status_code})"
            points.append((e.ended_at, end_label))
        points.sort(key=lambda p: p[0])
        return ", ".join(label for _, label in points)

    def to_dict(self) -> dict:
        return {
            "events": [e.model_dump() for e in self._events],
            "pattern": self.pattern(),
        }


__all__ = ["AgentEvent", "InterleavingRecorder"]
