import pytest
from pydantic import ValidationError

from src.race.swarm import (
    BLOCKED_ACTION_PATTERNS,
    RaceResult,
    RaceScenario,
    SynchronizationDriftError,
)


def test_race_scenario_valid():
    s = RaceScenario(
        scenario_id="s1",
        description="test",
        agents=3,
        action="add_todo",
        target_url="https://example.com",
        overlap_ms=50,
        expected_safe=True,
    )
    assert s.scenario_id == "s1"
    assert s.agents == 3
    assert s.overlap_ms == 50


def test_race_scenario_default_overlap():
    s = RaceScenario(
        scenario_id="s1",
        description="d",
        agents=2,
        action="add_todo",
        target_url="https://example.com",
        expected_safe=True,
    )
    assert s.overlap_ms == 50


def test_race_scenario_extra_field_forbidden():
    with pytest.raises(ValidationError):
        RaceScenario(
            scenario_id="s1",
            description="d",
            agents=2,
            action="add_todo",
            target_url="https://example.com",
            expected_safe=True,
            extra_field="bad",
        )


def test_race_result_valid():
    r = RaceResult(
        scenario_id="s1",
        conflict_found=True,
        interleaving=["2026-06-02T00:00:00+00:00"],
        error_summary="timeout",
        duration_ms=1234.5,
    )
    assert r.conflict_found is True
    assert r.error_summary == "timeout"


def test_race_result_none_error():
    r = RaceResult(
        scenario_id="s1",
        conflict_found=False,
        interleaving=[],
        error_summary=None,
        duration_ms=0.0,
    )
    assert r.error_summary is None


def test_race_result_extra_field_forbidden():
    with pytest.raises(ValidationError):
        RaceResult(
            scenario_id="s1",
            conflict_found=False,
            interleaving=[],
            error_summary=None,
            duration_ms=0.0,
            bad="x",
        )


def test_blocked_patterns_match():
    assert BLOCKED_ACTION_PATTERNS.search("delete_item")
    assert BLOCKED_ACTION_PATTERNS.search("remove_user")
    assert BLOCKED_ACTION_PATTERNS.search("transfer_funds")
    assert BLOCKED_ACTION_PATTERNS.search("payment_method")
    assert BLOCKED_ACTION_PATTERNS.search("change_password")


def test_blocked_patterns_no_match():
    assert not BLOCKED_ACTION_PATTERNS.search("add_todo")
    assert not BLOCKED_ACTION_PATTERNS.search("toggle_all")
    assert not BLOCKED_ACTION_PATTERNS.search("clear_completed")


def test_synchronization_drift_error_is_exception():
    err = SynchronizationDriftError("drift 120ms > 100ms")
    assert isinstance(err, Exception)
    assert "120ms" in str(err)
