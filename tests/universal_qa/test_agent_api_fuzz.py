"""Unit tests for the opt-in API fuzz phase wired into UniversalQAAgent
(mirrors enable_coverage_crosscheck's pattern — see src/universal_qa/agent.py).

All tests are mock-based: no live browser, no network, no Ollama. Heavy
optional ML deps (sentence-transformers/torch, pulled in transitively by
src.cache.semantic_cache) are stubbed at import time so these tests can run
in environments where that optional stack isn't installed — see the
`_stub_sentence_transformers()` call below.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _stub_sentence_transformers() -> None:
    """Insert a lightweight fake `sentence_transformers` module if the real
    one (or its torch dependency) isn't importable in this environment.
    UniversalQAAgent only imports the SemanticCache *type*, and never
    constructs one unless the semantic cache is explicitly enabled (it isn't
    in these tests) — so a non-functional stub is sufficient.
    """
    if "sentence_transformers" in sys.modules:
        return
    try:
        import sentence_transformers  # noqa: F401
        return
    except Exception:
        pass
    fake = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("stub SentenceTransformer — not usable in unit tests")

    fake.SentenceTransformer = _StubSentenceTransformer
    sys.modules["sentence_transformers"] = fake


_stub_sentence_transformers()

from src import config_loader  # noqa: E402
from src.universal_qa.agent import UniversalQAAgent  # noqa: E402
from src.universal_qa.explorer.nav_map import NavigationMap  # noqa: E402


# ── constructor / flag-resolution tests ──────────────────────────────────


def test_api_fuzz_disabled_by_default(monkeypatch) -> None:
    """Zero-config construction: both the constructor default and the config
    default are False, so the phase must be off (zero behavior change)."""
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    agent = UniversalQAAgent(url="https://example.com")
    assert agent._enable_api_fuzz is False


def test_api_fuzz_enabled_by_explicit_constructor_flag(monkeypatch) -> None:
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    agent = UniversalQAAgent(url="https://example.com", enable_api_fuzz=True)
    assert agent._enable_api_fuzz is True


def test_api_fuzz_enabled_by_config_without_constructor_flag(monkeypatch) -> None:
    """fuzzer.enable_api_fuzz: true in config/agent.yaml turns it on even
    when the caller passes no constructor arg (mirrors cache's config-only gate)."""
    monkeypatch.setattr(
        config_loader, "_load_yaml", lambda: {"fuzzer": {"enable_api_fuzz": True}}
    )
    agent = UniversalQAAgent(url="https://example.com")
    assert agent._enable_api_fuzz is True


def test_api_fuzz_stays_off_when_config_disabled_and_flag_not_passed(monkeypatch) -> None:
    monkeypatch.setattr(
        config_loader, "_load_yaml", lambda: {"fuzzer": {"enable_api_fuzz": False}}
    )
    agent = UniversalQAAgent(url="https://example.com")
    assert agent._enable_api_fuzz is False


# ── run() call-site wiring: flag off is a no-op, flag on calls through ──


def _fake_async_playwright():
    """A fake async_playwright() context manager wired to no-op browser mocks."""
    mock_page = MagicMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.url = "https://example.com/"

    mock_context = MagicMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)

    class _FakeAPW:
        async def __aenter__(self_inner):
            return mock_pw

        async def __aexit__(self_inner, *exc):
            return False

    return MagicMock(return_value=_FakeAPW())


@pytest.fixture
def wired_agent(monkeypatch):
    """A UniversalQAAgent with every collaborator mocked out so run() completes
    without touching a real browser/LLM, for testing the call-site wiring only."""
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})

    def _make(**kwargs):
        agent = UniversalQAAgent(url="https://example.com", **kwargs)

        mock_sfg_store = MagicMock()
        mock_sfg_store.get_nodes_by_url_prefix = MagicMock(return_value=[])
        agent._discovery.discover = AsyncMock(return_value=mock_sfg_store)
        agent._auth.setup = AsyncMock(return_value=None)
        agent._planner.plan_from_map = AsyncMock(return_value=[])
        agent._terminal.report_summary = MagicMock()
        agent._run_coverage_crosscheck = AsyncMock()
        agent._run_api_fuzz = AsyncMock()

        empty_nav_map = NavigationMap(
            base_url="https://example.com", pages=[], flows=[], explored_at_iso="2026-01-01T00:00:00"
        )

        patches = [
            patch("src.universal_qa.agent.async_playwright", _fake_async_playwright()),
            patch(
                "src.universal_qa.explorer.site_explorer.SiteExplorer.explore",
                AsyncMock(return_value=empty_nav_map),
            ),
            patch("src.universal_qa.test_runner.UniversalTestRunner.run", AsyncMock(return_value=[])),
            patch("src.universal_qa.reporters.html.HTMLReporter.generate", MagicMock(return_value="report.html")),
        ]
        return agent, patches

    return _make


async def test_run_skips_api_fuzz_phase_when_disabled(wired_agent) -> None:
    agent, patches = wired_agent(enable_api_fuzz=False)
    with patches[0], patches[1], patches[2], patches[3]:
        await agent.run()

    agent._run_api_fuzz.assert_not_awaited()


async def test_run_calls_api_fuzz_phase_when_enabled(wired_agent) -> None:
    agent, patches = wired_agent(enable_api_fuzz=True)
    with patches[0], patches[1], patches[2], patches[3]:
        await agent.run()

    agent._run_api_fuzz.assert_awaited_once()
    # Called with the post-auth discover URL (positional arg), same as coverage crosscheck.
    assert agent._run_api_fuzz.await_args.args == ("https://example.com/",)


# ── _run_api_fuzz phase body: report shape, GET-only scope, failure isolation ──


def _make_schema(endpoint: str, method: str = "GET", props: dict | None = None):
    from src.fuzzer.schema_inferrer import InferredSchema

    return InferredSchema(
        endpoint=endpoint,
        method=method,
        request_schema={"type": "object", "properties": props or {}},
        response_schema={},
        constraints=[],
        coverage_score=1,
        is_candidate=False,
    )


async def test_run_api_fuzz_writes_report_and_skips_non_get(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    agent = UniversalQAAgent(url="https://example.com", output_dir=tmp_path)

    get_schema = _make_schema("/api/articles", "GET", {"limit": {"type": "integer"}})
    post_schema = _make_schema("/api/users", "POST", {"email": {"type": "string"}})

    mock_inferrer = MagicMock()
    mock_inferrer.capture = AsyncMock(return_value=["trace"])
    mock_inferrer.infer = AsyncMock(return_value=[get_schema, post_schema])

    mock_generator = MagicMock()
    mock_generator.generate_fields = AsyncMock(return_value={"limit": ["-1", "abc"]})

    from src.fuzzer.anomaly_classifier import AnomalyType, FuzzResult as ClassifiedResult

    fake_results = [
        ClassifiedResult(
            endpoint="GET /api/articles", vector="-1", status_code=500,
            anomaly=True, anomaly_type=AnomalyType.UNEXPECTED_STATUS,
            false_positive=False, duration_ms=12.0,
        ),
        ClassifiedResult(
            endpoint="GET /api/articles", vector="abc", status_code=200,
            anomaly=False, anomaly_type=None, false_positive=False, duration_ms=8.0,
        ),
    ]
    mock_fuzzer = MagicMock()
    mock_fuzzer.fuzz_endpoint = AsyncMock(return_value=fake_results)

    with (
        patch("src.fuzzer.schema_inferrer.SchemaInferrer", return_value=mock_inferrer),
        patch("src.fuzzer.vector_generator.AdvancedVectorGenerator", return_value=mock_generator),
        patch("src.fuzzer.api_fuzzer.AutonomousAPIFuzzer", return_value=mock_fuzzer),
        patch("src.universal_qa.agent.async_playwright", _fake_async_playwright()),
    ):
        await agent._run_api_fuzz("https://example.com")

    # POST endpoint was discovered but must NEVER be fuzzed (read-only scope).
    mock_generator.generate_fields.assert_called_once()
    called_schema = mock_generator.generate_fields.call_args.args[0]
    assert called_schema.method == "GET"
    assert called_schema.endpoint == "/api/articles"

    mock_fuzzer.fuzz_endpoint.assert_called_once()
    sent_requests = mock_fuzzer.fuzz_endpoint.call_args.args[1]
    assert all(r.method == "GET" for r in sent_requests)

    report_path = tmp_path / "api_fuzz_report.json"
    assert report_path.exists()
    import json

    report = json.loads(report_path.read_text())
    assert report["endpoints_discovered"] == 2
    assert report["endpoints_fuzzed"] == 1
    assert report["vectors_sent"] == 2
    assert report["anomalies_found"] == 1
    assert report["anomaly_breakdown"] == {"unexpected_status": 1}


async def test_run_api_fuzz_never_weakens_blocked_action_patterns(tmp_path, monkeypatch) -> None:
    """A discovered GET endpoint whose path itself matches BLOCKED_ACTION_PATTERNS
    (e.g. a delete-style GET route) must be skipped, never fuzzed."""
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    agent = UniversalQAAgent(url="https://example.com", output_dir=tmp_path)

    blocked_schema = _make_schema("/api/account/delete", "GET", {"id": {"type": "integer"}})

    mock_inferrer = MagicMock()
    mock_inferrer.capture = AsyncMock(return_value=["trace"])
    mock_inferrer.infer = AsyncMock(return_value=[blocked_schema])

    mock_generator = MagicMock()
    mock_generator.generate_fields = AsyncMock(return_value={"id": ["-1"]})

    mock_fuzzer = MagicMock()
    mock_fuzzer.fuzz_endpoint = AsyncMock(return_value=[])

    with (
        patch("src.fuzzer.schema_inferrer.SchemaInferrer", return_value=mock_inferrer),
        patch("src.fuzzer.vector_generator.AdvancedVectorGenerator", return_value=mock_generator),
        patch("src.fuzzer.api_fuzzer.AutonomousAPIFuzzer", return_value=mock_fuzzer),
        patch("src.universal_qa.agent.async_playwright", _fake_async_playwright()),
    ):
        await agent._run_api_fuzz("https://example.com")

    mock_generator.generate_fields.assert_not_called()
    mock_fuzzer.fuzz_endpoint.assert_not_called()

    import json

    report = json.loads((tmp_path / "api_fuzz_report.json").read_text())
    assert report["endpoints_fuzzed"] == 0
    assert report["vectors_sent"] == 0


async def test_run_api_fuzz_failure_is_isolated_and_reported(tmp_path, monkeypatch) -> None:
    """A hard failure inside the phase (e.g. SchemaInferrer raising) must never
    propagate — it's caught, logged, and still leaves a report file behind."""
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    agent = UniversalQAAgent(url="https://example.com", output_dir=tmp_path)

    with patch(
        "src.universal_qa.agent.async_playwright",
        MagicMock(side_effect=RuntimeError("boom")),
    ):
        await agent._run_api_fuzz("https://example.com")  # must not raise

    import json

    report = json.loads((tmp_path / "api_fuzz_report.json").read_text())
    assert "error" in report
    assert report["endpoints_discovered"] == 0
