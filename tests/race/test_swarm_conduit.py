"""TDD for Sprint 12 swarm additions: conflict logic + payload field + goto-skip."""
from __future__ import annotations

from src.race.swarm import RaceScenario, _compute_conflict, _skips_navigation


# ── _compute_conflict (spec logic: status OR hash-divergence OR errors) ──────


def test_conflict_on_bad_status():
    # register race: one 200 winner, two 422 losers
    assert _compute_conflict([200, 422, 422], ["h1", "h2", "h2"], []) is True


def test_no_conflict_when_all_ok_and_identical():
    assert _compute_conflict([200, 200], ["h", "h"], []) is False


def test_conflict_on_hash_divergence_only():
    # all OK statuses but divergent semantic state
    assert _compute_conflict([200, 200], ["h1", "h2"], []) is True


def test_conflict_on_errors_only():
    assert _compute_conflict([200, 200], ["h", "h"], ["boom"]) is True


def test_all_ok_statuses_201_204_safe():
    assert _compute_conflict([200, 201, 204], ["h"], []) is False


def test_no_statuses_no_hashes_no_conflict():
    assert _compute_conflict([], [], []) is False


# ── RaceScenario.payload (new optional field, backward compatible) ───────────


def test_scenario_payload_defaults_none():
    s = RaceScenario(
        scenario_id="s", description="d", agents=2, action="conduit_read_tags",
        target_url="https://x", expected_safe=True,
    )
    assert s.payload is None


def test_scenario_payload_accepts_dict():
    s = RaceScenario(
        scenario_id="s", description="d", agents=2, action="conduit_register",
        target_url="https://x", expected_safe=False, payload={"username": "race_abc"},
    )
    assert s.payload == {"username": "race_abc"}


# ── _skips_navigation (HTTP/conduit actions need no page.goto) ───────────────


def test_http_and_conduit_actions_skip_navigation():
    assert _skips_navigation("http_get") is True
    assert _skips_navigation("http_post") is True
    assert _skips_navigation("conduit_register") is True
    assert _skips_navigation("conduit_read_tags") is True


def test_ui_actions_do_navigate():
    assert _skips_navigation("add_todo") is False
    assert _skips_navigation("toggle_all") is False
