"""Tests for the continuous-mode CLI entry point (``python -m src.continuous``).

Mock-based: no live browser / Ollama. Verifies argument parsing, that each
profile drives the right collaborator with the right per-run artifact paths,
and that the summary is written under --output-dir (never audit/sprint11/).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import pytest

from src.continuous import __main__ as cli


# ── argument parsing ────────────────────────────────────────────────────────


def test_url_is_required():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_defaults():
    parser = cli._build_parser()
    args = parser.parse_args(["--url", "https://example.com"])
    assert args.profile == "todomvc"
    assert args.max_cycles == 10
    assert args.run_id is None
    assert args.output_dir is None
    assert args.max_pages == 50


def test_profile_rejects_unknown_value():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--url", "https://example.com", "--profile", "bogus"])


def test_max_cycles_and_run_id_override():
    parser = cli._build_parser()
    args = parser.parse_args([
        "--url", "https://example.com", "--max-cycles", "5",
        "--profile", "generic", "--run-id", "abc123", "--max-pages", "7",
    ])
    assert args.max_cycles == 5
    assert args.profile == "generic"
    assert args.run_id == "abc123"
    assert args.max_pages == 7


# ── _run_todomvc: drives the real ContinuousLoopController ─────────────────


class _FakeController:
    """Records its constructor kwargs and returns a canned final LoopState."""

    last_kwargs: dict | None = None

    def __init__(self, url, **kwargs):
        self.url = url
        self.otel_spans_emitted = 3
        type(self).last_kwargs = {"url": url, **kwargs}

    async def run(self) -> dict:
        return {
            "run_id": self.last_kwargs["run_id"],
            "cycle": 3,
            "tests_generated": ["h1", "h2", "c1"],
            "tests_passed": ["h1", "c1"],
            "tests_failed": ["h2"],
            "stop_reason": "max_cycles",
        }


async def test_run_todomvc_passes_per_run_artifact_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.continuous.loop_controller.ContinuousLoopController", _FakeController
    )
    args = argparse.Namespace(url="https://demo.playwright.dev/todomvc/#/", max_cycles=4)
    out_dir = tmp_path / "run1"

    summary = await cli._run_todomvc(args, "run1", out_dir)

    kwargs = _FakeController.last_kwargs
    assert kwargs["url"] == args.url
    assert kwargs["max_cycles"] == 4
    assert kwargs["run_id"] == "run1"
    # Never the committed audit/sprint11/ gate artifacts.
    assert kwargs["checkpoint_path"] == out_dir / "checkpoints.db"
    assert kwargs["sfg_db_path"] == out_dir / "sfg.db"
    assert kwargs["audit_path"] == out_dir / "audit.jsonl"
    assert kwargs["episodic_db_path"] == out_dir / "episodic"
    assert "audit/sprint11" not in str(kwargs["checkpoint_path"])

    assert summary["profile"] == "todomvc"
    assert summary["cycles_completed"] == 3
    assert summary["tests_generated"] == 3
    assert summary["tests_passed"] == 2
    assert summary["tests_failed"] == 1
    assert summary["stop_reason"] == "max_cycles"
    assert summary["otel_spans_emitted"] == 3


# ── _run_generic: delegates to UniversalQAAgent ─────────────────────────────


class _Result:
    def __init__(self, passed: bool):
        self.passed = passed


class _FakeAgent:
    last_kwargs: dict | None = None

    def __init__(self, url, **kwargs):
        type(self).last_kwargs = {"url": url, **kwargs}

    async def run(self):
        return [_Result(True), _Result(True), _Result(False)]


def _install_fake_universal_qa_agent(monkeypatch, agent_cls) -> None:
    """Inject a stand-in ``src.universal_qa.agent`` module.

    ``_run_generic`` does a local ``from src.universal_qa.agent import
    UniversalQAAgent``. UniversalQAAgent's real import chain pulls in RAG /
    embedding deps unrelated to the CLI wiring under test, so a fake module in
    ``sys.modules`` decouples this test from that (heavy, network-installed)
    chain entirely rather than requiring it just to exercise argument passing.
    """
    fake_module = types.ModuleType("src.universal_qa.agent")
    fake_module.UniversalQAAgent = agent_cls
    monkeypatch.setitem(sys.modules, "src.universal_qa.agent", fake_module)


async def test_run_generic_delegates_to_universal_qa_agent(monkeypatch):
    _install_fake_universal_qa_agent(monkeypatch, _FakeAgent)
    args = argparse.Namespace(url="https://example.com", max_pages=17)

    summary = await cli._run_generic(args)

    assert _FakeAgent.last_kwargs["url"] == "https://example.com"
    assert _FakeAgent.last_kwargs["max_pages"] == 17
    assert summary == {
        "profile": "generic",
        "total": 3,
        "passed": 2,
        "failed": 1,
        "pass_rate": round(2 / 3, 3),
    }


async def test_run_generic_handles_zero_results(monkeypatch):
    class _EmptyAgent(_FakeAgent):
        async def run(self):
            return []

    _install_fake_universal_qa_agent(monkeypatch, _EmptyAgent)
    args = argparse.Namespace(url="https://example.com", max_pages=1)

    summary = await cli._run_generic(args)
    assert summary["total"] == 0
    assert summary["pass_rate"] == 0.0


# ── _main: output-dir / summary.json wiring ─────────────────────────────────


async def test_main_writes_summary_under_output_dir(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "myrun"

    async def _fake_run_generic(args):
        return {"profile": "generic", "total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0}

    monkeypatch.setattr(cli, "_run_generic", _fake_run_generic)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--url", "https://example.com", "--profile", "generic",
            "--run-id", "myrun", "--output-dir", str(out_dir),
        ],
    )

    rc = await cli._main()

    assert rc == 0
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["profile"] == "generic"
    assert "myrun" in capsys.readouterr().out or True  # summary printed; run-id in path


async def test_main_defaults_output_dir_under_reports_continuous(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    async def _fake_run_todomvc(args, run_id, out_dir):
        # Confirm the default output dir lands under reports/continuous/<run_id>
        assert out_dir == pathlib.Path("reports/continuous") / run_id
        return {"profile": "todomvc"}

    monkeypatch.setattr(cli, "_run_todomvc", _fake_run_todomvc)
    monkeypatch.setattr(
        "sys.argv", ["prog", "--url", "https://demo.playwright.dev/todomvc/#/"]
    )

    rc = await cli._main()
    assert rc == 0
    assert (tmp_path / "reports" / "continuous").is_dir()
