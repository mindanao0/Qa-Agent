"""Unit tests for UniversalQAAgent's opt-in race-testing wiring (mirrors
enable_coverage_crosscheck's PR #5 pattern). Mock-based: no live browser.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.universal_qa.agent import UniversalQAAgent
from src.universal_qa.explorer.nav_map import ExploredPage, NavigationMap


def _make_nav_map(urls: list[str]) -> NavigationMap:
    return NavigationMap(
        base_url=urls[0] if urls else "https://x.com",
        pages=[
            ExploredPage(url=u, title="t", pam_content="", actions=[]) for u in urls
        ],
        flows=[],
        explored_at_iso="2026-09-14T00:00:00+00:00",
    )


# ── constructor flag resolution ───────────────────────────────────────────


def test_race_testing_defaults_off(monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: False
    )
    monkeypatch.setattr("src.universal_qa.agent.get_race_config", lambda: {})
    agent = UniversalQAAgent(url="https://x.com")
    assert agent._enable_race_testing is False


def test_race_testing_reads_config_when_none(monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: True
    )
    monkeypatch.setattr(
        "src.universal_qa.agent.get_race_config",
        lambda: {"allow_destructive_scenarios": True},
    )
    agent = UniversalQAAgent(url="https://x.com")
    assert agent._enable_race_testing is True
    assert agent._allow_destructive_race_scenarios is True


def test_race_testing_explicit_constructor_arg_overrides_config(monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: True
    )
    monkeypatch.setattr("src.universal_qa.agent.get_race_config", lambda: {})
    agent = UniversalQAAgent(url="https://x.com", enable_race_testing=False)
    assert agent._enable_race_testing is False


def test_allow_destructive_explicit_constructor_arg_overrides_config(monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: False
    )
    monkeypatch.setattr(
        "src.universal_qa.agent.get_race_config",
        lambda: {"allow_destructive_scenarios": True},
    )
    agent = UniversalQAAgent(
        url="https://x.com", allow_destructive_race_scenarios=False
    )
    assert agent._allow_destructive_race_scenarios is False


# ── _run_race_testing (the private phase method) ──────────────────────────


@pytest.mark.asyncio
async def test_run_race_testing_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: False
    )
    monkeypatch.setattr("src.universal_qa.agent.get_race_config", lambda: {})
    agent = UniversalQAAgent(url="https://x.com", output_dir=tmp_path)
    nav_map = _make_nav_map(["https://x.com/a", "https://x.com/b"])

    fake_report = {"scenarios_tested": 2, "conflicts_found": 0, "results": []}
    with patch(
        "src.universal_qa.race_check.run_race_check",
        new=AsyncMock(return_value=fake_report),
    ) as mock_run:
        await agent._run_race_testing(nav_map)

    mock_run.assert_awaited_once()
    called_urls = mock_run.call_args.args[0]
    assert called_urls == ["https://x.com/a", "https://x.com/b"]

    out_path = tmp_path / "race_report.json"
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == fake_report


@pytest.mark.asyncio
async def test_run_race_testing_falls_back_to_base_url_when_no_pages(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: False
    )
    monkeypatch.setattr("src.universal_qa.agent.get_race_config", lambda: {})
    agent = UniversalQAAgent(url="https://x.com", output_dir=tmp_path)
    nav_map = _make_nav_map([])

    with patch(
        "src.universal_qa.race_check.run_race_check",
        new=AsyncMock(return_value={"scenarios_tested": 0}),
    ) as mock_run:
        await agent._run_race_testing(nav_map)

    called_urls = mock_run.call_args.args[0]
    assert called_urls == ["https://x.com"]


@pytest.mark.asyncio
async def test_run_race_testing_failure_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: False
    )
    monkeypatch.setattr("src.universal_qa.agent.get_race_config", lambda: {})
    agent = UniversalQAAgent(url="https://x.com", output_dir=tmp_path)
    nav_map = _make_nav_map(["https://x.com/a"])

    async def _boom(*args, **kwargs):
        raise RuntimeError("browser launch failed")

    with patch("src.universal_qa.race_check.run_race_check", new=_boom):
        await agent._run_race_testing(nav_map)  # must not raise

    assert not (tmp_path / "race_report.json").exists()


@pytest.mark.asyncio
async def test_run_race_testing_passes_config_values_through(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.universal_qa.agent.get_use_race_testing", lambda: False
    )
    monkeypatch.setattr(
        "src.universal_qa.agent.get_race_config",
        lambda: {"agents_per_scenario": 3, "overlap_ms": 150, "max_scenarios": 2},
    )
    agent = UniversalQAAgent(
        url="https://x.com", output_dir=tmp_path,
        allow_destructive_race_scenarios=True,
    )
    nav_map = _make_nav_map(["https://x.com/a"])

    with patch(
        "src.universal_qa.race_check.run_race_check",
        new=AsyncMock(return_value={"scenarios_tested": 1}),
    ) as mock_run:
        await agent._run_race_testing(nav_map)

    kwargs = mock_run.call_args.kwargs
    assert kwargs["agents_per_scenario"] == 3
    assert kwargs["overlap_ms"] == 150
    assert kwargs["max_scenarios"] == 2
    assert kwargs["allow_destructive"] is True
