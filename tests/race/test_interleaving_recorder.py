"""Tests for InterleavingRecorder (Sprint 12)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.race.interleaving_recorder import AgentEvent, InterleavingRecorder


def _ev(agent_id, started, ended, status):
    return AgentEvent(
        agent_id=agent_id, action="conduit_register",
        started_at=started, ended_at=ended, status_code=status, response_hash="abcd1234",
    )


def test_agent_event_extra_forbidden():
    with pytest.raises(ValidationError):
        AgentEvent(
            agent_id="a", action="x", started_at=0.0, ended_at=1.0,
            status_code=200, response_hash="h", bad="nope",
        )


def test_empty_recorder_pattern_is_empty():
    assert InterleavingRecorder().pattern() == ""


def test_record_then_to_dict_has_events_and_pattern():
    rec = InterleavingRecorder()
    rec.record(_ev("agent0", 0.0, 10.0, 200))
    d = rec.to_dict()
    assert len(d["events"]) == 1
    assert "pattern" in d
    assert d["events"][0]["status_code"] == 200


def test_pattern_orders_starts_and_ends_by_timestamp():
    rec = InterleavingRecorder()
    rec.record(_ev("agent0", 0.0, 10.0, 200))
    rec.record(_ev("agent1", 1.0, 5.0, 422))
    # timeline: a0-start(0), a1-start(1), a1-end(5,422), a0-end(10,200)
    assert rec.pattern() == "agent0-start, agent1-start, agent1-end(422), agent0-end(200)"


def test_pattern_omits_parens_when_status_none():
    rec = InterleavingRecorder()
    rec.record(_ev("agent0", 0.0, 2.0, None))
    assert rec.pattern() == "agent0-start, agent0-end"


def test_record_every_event_counts_both_agents():
    rec = InterleavingRecorder()
    rec.record(_ev("agent0", 0.0, 3.0, 201))
    rec.record(_ev("agent1", 0.5, 4.0, 422))
    assert len(rec._events) == 2
    # 4 timeline points (2 starts + 2 ends)
    assert rec.pattern().count(",") == 3
