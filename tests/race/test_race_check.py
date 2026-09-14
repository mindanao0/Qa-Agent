"""Unit tests for src/universal_qa/race_check.py — the safe, generic
production wiring of RaceConditionSwarm. All mock-based: no live browser or
network.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.race.swarm import RaceResult
from src.universal_qa.race_check import build_safe_scenarios, run_race_check, run_scenarios


# ── build_safe_scenarios (pure) ───────────────────────────────────────────


def test_build_safe_scenarios_basic():
    urls = ["https://x.com/a", "https://x.com/b"]
    scenarios = build_safe_scenarios(urls)
    assert len(scenarios) == 2
    assert all(s.action == "http_get" for s in scenarios)
    assert all(s.expected_safe is True for s in scenarios)
    assert {s.target_url for s in scenarios} == set(urls)


def test_build_safe_scenarios_dedupes_preserving_order():
    urls = ["https://x.com/a", "https://x.com/b", "https://x.com/a"]
    scenarios = build_safe_scenarios(urls)
    assert len(scenarios) == 2
    assert scenarios[0].target_url == "https://x.com/a"
    assert scenarios[1].target_url == "https://x.com/b"


def test_build_safe_scenarios_skips_falsy_urls():
    scenarios = build_safe_scenarios(["https://x.com/a", "", None, "https://x.com/b"])
    assert len(scenarios) == 2


def test_build_safe_scenarios_caps_at_max_scenarios():
    urls = [f"https://x.com/{i}" for i in range(10)]
    scenarios = build_safe_scenarios(urls, max_scenarios=3)
    assert len(scenarios) == 3


def test_build_safe_scenarios_clamps_agents_to_minimum_two():
    scenarios = build_safe_scenarios(["https://x.com/a"], agents_per_scenario=1)
    assert scenarios[0].agents == 2


def test_build_safe_scenarios_respects_agents_and_overlap():
    scenarios = build_safe_scenarios(
        ["https://x.com/a"], agents_per_scenario=4, overlap_ms=99
    )
    assert scenarios[0].agents == 4
    assert scenarios[0].overlap_ms == 99


def test_build_safe_scenarios_empty_input():
    assert build_safe_scenarios([]) == []


# ── run_scenarios (mocks RaceConditionSwarm.run) ──────────────────────────


@pytest.mark.asyncio
async def test_run_scenarios_no_conflict():
    scenarios = build_safe_scenarios(["https://x.com/a", "https://x.com/b"])
    fake_result = RaceResult(
        scenario_id="race_get_1", conflict_found=False,
        interleaving=["t1", "t2"], error_summary=None, duration_ms=10.0,
    )
    with patch(
        "src.universal_qa.race_check.RaceConditionSwarm.run",
        new=AsyncMock(return_value=fake_result),
    ):
        report = await run_scenarios(scenarios, browser=object())

    assert report["scenarios_tested"] == 2
    assert report["conflicts_found"] == 0
    assert report["scenario_set"] == "read_only_safe"
    assert len(report["results"]) == 2


@pytest.mark.asyncio
async def test_run_scenarios_captures_exception_as_conflict():
    scenarios = build_safe_scenarios(["https://x.com/a"])

    async def _boom(*args, **kwargs):
        raise RuntimeError("navigation exploded")

    with patch("src.universal_qa.race_check.RaceConditionSwarm.run", new=_boom):
        report = await run_scenarios(scenarios, browser=object())

    assert report["scenarios_tested"] == 1
    assert report["conflicts_found"] == 1
    assert "navigation exploded" in report["results"][0]["error_summary"]


@pytest.mark.asyncio
async def test_run_scenarios_conflict_found_true():
    scenarios = build_safe_scenarios(["https://x.com/a"])
    fake_result = RaceResult(
        scenario_id="race_get_1", conflict_found=True,
        interleaving=[], error_summary="500", duration_ms=5.0,
    )
    with patch(
        "src.universal_qa.race_check.RaceConditionSwarm.run",
        new=AsyncMock(return_value=fake_result),
    ):
        report = await run_scenarios(scenarios, browser=object())

    assert report["conflicts_found"] == 1
    assert report["conflict_rate"] == 1.0


# ── run_race_check (mocks async_playwright + browser launch) ─────────────


@pytest.mark.asyncio
async def test_run_race_check_launches_isolated_browser_and_closes_it():
    fake_result = RaceResult(
        scenario_id="race_get_1", conflict_found=False,
        interleaving=[], error_summary=None, duration_ms=1.0,
    )
    fake_browser = AsyncMock()
    fake_pw_ctx = AsyncMock()
    fake_pw_ctx.chromium.launch = AsyncMock(return_value=fake_browser)

    fake_playwright_cm = AsyncMock()
    fake_playwright_cm.__aenter__ = AsyncMock(return_value=fake_pw_ctx)
    fake_playwright_cm.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "src.universal_qa.race_check.async_playwright", return_value=fake_playwright_cm
    ), patch(
        "src.universal_qa.race_check.RaceConditionSwarm.run",
        new=AsyncMock(return_value=fake_result),
    ):
        report = await run_race_check(["https://x.com/a"], headless=True)

    fake_pw_ctx.chromium.launch.assert_awaited_once()
    fake_browser.close.assert_awaited_once()
    assert report["scenarios_tested"] == 1
    assert report["allow_destructive_requested"] is False


@pytest.mark.asyncio
async def test_run_race_check_no_urls_short_circuits_without_launching_browser():
    with patch("src.universal_qa.race_check.async_playwright") as mock_pw:
        report = await run_race_check([], headless=True)

    mock_pw.assert_not_called()
    assert report["scenarios_tested"] == 0
    assert report["conflicts_found"] == 0


@pytest.mark.asyncio
async def test_run_race_check_allow_destructive_logs_but_still_safe_only(caplog):
    fake_result = RaceResult(
        scenario_id="race_get_1", conflict_found=False,
        interleaving=[], error_summary=None, duration_ms=1.0,
    )
    fake_browser = AsyncMock()
    fake_pw_ctx = AsyncMock()
    fake_pw_ctx.chromium.launch = AsyncMock(return_value=fake_browser)
    fake_playwright_cm = AsyncMock()
    fake_playwright_cm.__aenter__ = AsyncMock(return_value=fake_pw_ctx)
    fake_playwright_cm.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "src.universal_qa.race_check.async_playwright", return_value=fake_playwright_cm
    ), patch(
        "src.universal_qa.race_check.RaceConditionSwarm.run",
        new=AsyncMock(return_value=fake_result),
    ):
        report = await run_race_check(
            ["https://x.com/a"], headless=True, allow_destructive=True
        )

    # Still only the safe http_get scenario shape, never anything destructive.
    assert report["scenario_set"] == "read_only_safe"
    assert all(r["scenario_id"].startswith("race_get_") for r in report["results"])
    assert report["allow_destructive_requested"] is True
