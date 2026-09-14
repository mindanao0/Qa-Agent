"""Unit tests for the Observer→Driver feedback channel.

PURE Python, fully offline: NO browser, NO Ollama, NO network. Covers three layers:

* the deterministic distiller (:func:`distill_feedback`),
* the Driver prompt-assembly hint block (:func:`_build_prompt`), and
* the coordinator wiring (``step_node``) with the browser/LLM boundaries monkeypatched
  — proving a finding at step N reaches the Driver prompt at step N+1 without any live
  dependency.

The live end-to-end path is exercised by ``tests/test_observer_driver.py`` (needs a
headed Chromium + real Ollama) and is intentionally NOT run here.
"""
from __future__ import annotations

from pathlib import Path

# Ensure project root is importable when run directly via ``pytest tests/...``.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.observer_driver.driver_agent import _build_prompt
from src.agents.observer_driver.feedback import (
    MAX_DIRECTIVES,
    distill_feedback,
    finding_hash,
)
from src.agents.observer_driver.state import (
    AgentState,
    DriverAction,
    DriverFeedback,
    ObserverReport,
)

# A stable marker that begins the JSON action contract in the Driver prompt.
_CONTRACT_MARKER = "Reply with a SINGLE JSON object"


def _report(name: str, findings: list[str], severity: str) -> ObserverReport:
    return ObserverReport(
        observer_name=name,
        findings=findings,
        severity=severity,  # type: ignore[arg-type]
        timestamp_ms=0,  # deterministic; never used by the distiller
    )


# --------------------------------------------------------------------------- #
# Distiller
# --------------------------------------------------------------------------- #
def test_info_dropped_warn_and_critical_kept() -> None:
    reports = [
        _report("security", ["no security issues detected"], "info"),
        _report("performance", ["cdp.JSHeapUsedSize=1234.0"], "info"),
        _report("accessibility", ["missing accessible name on 2 node(s)"], "warn"),
        _report("security", ["mixed-content resources on HTTPS page: 3"], "critical"),
    ]
    hint = distill_feedback(reports, step=0, already_acted=frozenset())
    assert hint is not None
    joined = "\n".join(hint.directives)
    # info-level filler must be gone
    assert "no security issues detected" not in joined
    assert "cdp.JSHeapUsedSize" not in joined
    # warn + critical survive
    assert "missing accessible name on 2 node(s)" in hint.directives
    assert "mixed-content resources on HTTPS page: 3" in hint.directives


def test_top_severity_lifted_from_report_level() -> None:
    """A finding inherits its CONTAINING report's (report-level) severity."""
    reports = [
        _report("security", ["mixed-content resources on HTTPS page: 1"], "critical"),
    ]
    hint = distill_feedback(reports, step=7, already_acted=frozenset())
    assert hint is not None
    assert hint.top_severity == "critical"
    assert hint.from_step == 7
    assert hint.directives == ["mixed-content resources on HTTPS page: 1"]

    # A report-level "warn" lifts every one of its findings to warn (not info).
    warn_reports = [_report("accessibility", ["a", "b"], "warn")]
    warn_hint = distill_feedback(warn_reports, step=1, already_acted=frozenset())
    assert warn_hint is not None
    assert warn_hint.top_severity == "warn"
    assert warn_hint.directives == ["a", "b"]


def test_deterministic_high_to_low_ordering() -> None:
    # Reports encountered warn-first, then critical; output must be critical-first,
    # stable within an equal-severity tier, and identical across repeated calls.
    reports = [
        _report("performance", ["w1", "w2"], "warn"),
        _report("security", ["c1", "c2"], "critical"),
    ]
    hint = distill_feedback(reports, step=3, already_acted=frozenset())
    assert hint is not None
    # critical (c1, c2) ahead of warn (w1); capped at 3
    assert hint.directives == ["c1", "c2", "w1"]
    assert hint.top_severity == "critical"

    # Determinism: same inputs → byte-identical output.
    again = distill_feedback(reports, step=3, already_acted=frozenset())
    assert again is not None
    assert again.model_dump() == hint.model_dump()


def test_cap_at_three_directives() -> None:
    reports = [_report("security", ["f1", "f2", "f3", "f4", "f5"], "warn")]
    hint = distill_feedback(reports, step=0, already_acted=frozenset())
    assert hint is not None
    assert len(hint.directives) == MAX_DIRECTIVES == 3
    assert len(hint.finding_hashes) == 3
    # Stable, so the FIRST three encountered survive.
    assert hint.directives == ["f1", "f2", "f3"]


def test_dedup_within_call_by_hash() -> None:
    # Same observer + same finding across two reports → one candidate.
    dupe = "missing security header: content-security-policy"
    reports = [
        _report("security", [dupe], "warn"),
        _report("security", [dupe], "warn"),
    ]
    hint = distill_feedback(reports, step=0, already_acted=frozenset())
    assert hint is not None
    assert hint.directives == [dupe]
    assert len(hint.finding_hashes) == 1


def test_cross_call_dedup_via_already_acted() -> None:
    dupe = "missing security header: content-security-policy"
    reports = [
        _report("security", [dupe], "warn"),
        _report("accessibility", ["missing accessible name on 1 node(s)"], "warn"),
    ]
    already = frozenset({finding_hash("security", dupe)})
    hint = distill_feedback(reports, step=2, already_acted=already)
    assert hint is not None
    # The already-surfaced security finding is suppressed; only the new one remains.
    assert hint.directives == ["missing accessible name on 1 node(s)"]
    assert finding_hash("security", dupe) not in hint.finding_hashes


def test_none_when_no_new_finding() -> None:
    # All info → nothing reaches "warn".
    only_info = [_report("security", ["no security issues detected"], "info")]
    assert distill_feedback(only_info, step=0, already_acted=frozenset()) is None

    # Empty reports → None.
    assert distill_feedback([], step=0, already_acted=frozenset()) is None

    # Every warn finding already acted on → None (no re-nagging).
    f = "missing accessible name on 1 node(s)"
    reports = [_report("accessibility", [f], "warn")]
    already = frozenset({finding_hash("accessibility", f)})
    assert distill_feedback(reports, step=9, already_acted=already) is None


def test_finding_hashes_match_selected() -> None:
    reports = [
        _report("security", ["c1"], "critical"),
        _report("accessibility", ["w1"], "warn"),
    ]
    hint = distill_feedback(reports, step=0, already_acted=frozenset())
    assert hint is not None
    # Hashes correspond exactly, positionally, to the surfaced directives.
    expected = [finding_hash("security", "c1"), finding_hash("accessibility", "w1")]
    assert hint.finding_hashes == expected
    # 12 hex chars each (SHA-1 truncation contract).
    assert all(len(h) == 12 for h in hint.finding_hashes)


def test_surplus_findings_not_recorded_so_they_can_surface_later() -> None:
    """Only the surfaced top-3 are hashed → the 4th/5th can surface on a later step."""
    reports = [_report("security", ["f1", "f2", "f3", "f4", "f5"], "warn")]
    first = distill_feedback(reports, step=0, already_acted=frozenset())
    assert first is not None
    assert first.directives == ["f1", "f2", "f3"]
    # Next step: the three surfaced hashes are now "already acted".
    already = frozenset(first.finding_hashes)
    second = distill_feedback(reports, step=1, already_acted=already)
    assert second is not None
    assert second.directives == ["f4", "f5"]


# --------------------------------------------------------------------------- #
# Driver prompt hint block
# --------------------------------------------------------------------------- #
def test_build_prompt_hint_absent_when_none() -> None:
    prompt = _build_prompt("- button \"OK\"", 0, "https://x.test")
    assert "PRIOR AUDIT SIGNALS" not in prompt
    assert _CONTRACT_MARKER in prompt


def test_build_prompt_hint_present_when_given() -> None:
    hint = DriverFeedback(
        from_step=2,
        directives=[
            "missing security header: content-security-policy",
            "missing accessible name on 2 node(s)",
        ],
        top_severity="warn",
        finding_hashes=["aaaaaaaaaaaa", "bbbbbbbbbbbb"],
    )
    prompt = _build_prompt("- button \"OK\"", 3, "https://x.test", hint)
    assert "PRIOR AUDIT SIGNALS" in prompt
    assert "from step 2" in prompt
    assert "top severity warn" in prompt
    for directive in hint.directives:
        assert directive in prompt
    # The advisory guard line must be present.
    assert "Never fabricate a target" in prompt


def test_build_prompt_hint_does_not_alter_json_contract() -> None:
    """A hint must never perturb the JSON action-contract portion of the prompt."""
    hint = DriverFeedback(
        from_step=1,
        directives=["mixed-content resources on HTTPS page: 3"],
        top_severity="critical",
        finding_hashes=["cccccccccccc"],
    )
    ax, step, url = "- link \"Home\"", 4, "https://x.test"
    with_hint = _build_prompt(ax, step, url, hint)
    without_hint = _build_prompt(ax, step, url)

    # The entire contract tail (from the marker onward) is byte-identical.
    assert _CONTRACT_MARKER in with_hint and _CONTRACT_MARKER in without_hint
    tail_with = with_hint[with_hint.index(_CONTRACT_MARKER):]
    tail_without = without_hint[without_hint.index(_CONTRACT_MARKER):]
    assert tail_with == tail_without
    # And the specific action-schema lines survive verbatim.
    assert '"action_type": one of "click", "fill", "navigate", "assert", "noop"' in with_hint


# --------------------------------------------------------------------------- #
# Integration-lite: distill → prompt (no live deps)
# --------------------------------------------------------------------------- #
def test_critical_finding_reaches_driver_prompt_next_step() -> None:
    """A critical finding at step N is distilled and appears in the step-N+1 prompt."""
    reports = [
        _report("security", ["mixed-content resources on HTTPS page: 3"], "critical"),
    ]
    hint = distill_feedback(reports, step=4, already_acted=frozenset())
    assert hint is not None
    assert hint.top_severity == "critical"

    # Feed the hint into the NEXT step's prompt (step 5).
    prompt = _build_prompt("- link \"Home\"", 5, "https://x.test", hint)
    assert "mixed-content resources on HTTPS page: 3" in prompt
    assert "from step 4" in prompt


# --------------------------------------------------------------------------- #
# Coordinator wiring (browser/LLM boundaries monkeypatched — no live deps)
# --------------------------------------------------------------------------- #
def _make_fake_observer(reports: list[ObserverReport]):
    """Return a factory usable as ``ObserverClass(queue)`` whose run() yields reports."""

    class _Inst:
        async def run(self, page, stop_event):  # noqa: ANN001 — test double
            return list(reports)

    def _factory(queue):  # noqa: ANN001 — matches ObserverClass(queue) call shape
        return _Inst()

    return _factory


async def test_step_node_wires_hint_roundtrip(monkeypatch) -> None:  # noqa: ANN001
    """step_node: forwards pending_hint into plan_action AND distils this step's
    reports into the next pending_hint + acted_finding_hashes — the pure wiring,
    proven without a live browser or Ollama by patching the three boundary calls."""
    import src.agents.observer_driver.coordinator as coord

    captured: dict[str, object] = {}

    async def fake_capture_axtree(page, step):  # noqa: ANN001
        return Path(f"agent_state/ax_snapshot_{step}.yml")

    async def fake_plan_action(
        ax_path, step, url, ollama_url="", model="", hint=None  # noqa: ANN001
    ):
        captured["hint_in"] = hint
        captured["step"] = step
        return DriverAction(action_type="noop", ax_snapshot_path=ax_path)

    async def fake_execute_action(page, action, bus):  # noqa: ANN001
        return {"action_type": "noop", "outcome": "noop", "url": "https://x.test"}

    monkeypatch.setattr(coord, "capture_axtree", fake_capture_axtree)
    monkeypatch.setattr(coord, "plan_action", fake_plan_action)
    monkeypatch.setattr(coord, "execute_action", fake_execute_action)

    critical = _report(
        "security", ["mixed-content resources on HTTPS page: 3"], "critical"
    )
    monkeypatch.setattr(coord, "AccessibilityObserver", _make_fake_observer([]))
    monkeypatch.setattr(coord, "SecurityObserver", _make_fake_observer([critical]))
    monkeypatch.setattr(coord, "PerformanceObserver", _make_fake_observer([]))

    incoming = DriverFeedback(
        from_step=0,
        directives=["a prior directive"],
        top_severity="warn",
        finding_hashes=["deadbeef0000"],
    )
    state = AgentState(
        url="https://x.test",
        current_step=1,
        max_steps=5,
        pending_hint=incoming,
    )

    update = await coord.step_node(state, {"configurable": {"page": object()}})

    # 1. The pending_hint from state was forwarded VERBATIM into plan_action.
    assert captured["hint_in"] is incoming
    assert captured["step"] == 1

    # 2. This step's critical report was distilled into a FRESH hint for the next step.
    new_hint = update["pending_hint"]
    assert isinstance(new_hint, DriverFeedback)
    assert new_hint.from_step == 1
    assert new_hint.top_severity == "critical"
    assert new_hint.directives == ["mixed-content resources on HTTPS page: 3"]

    # 3. Surfaced hashes are returned so the operator.add reducer accumulates them.
    assert update["acted_finding_hashes"] == new_hint.finding_hashes
    assert update["current_step"] == 2


async def test_step_node_clears_hint_when_nothing_new(monkeypatch) -> None:  # noqa: ANN001
    """When a step surfaces nothing new, pending_hint is written as None (cleared)."""
    import src.agents.observer_driver.coordinator as coord

    async def fake_capture_axtree(page, step):  # noqa: ANN001
        return Path("agent_state/ax_snapshot_0.yml")

    async def fake_plan_action(
        ax_path, step, url, ollama_url="", model="", hint=None  # noqa: ANN001
    ):
        return DriverAction(action_type="noop", ax_snapshot_path=ax_path)

    async def fake_execute_action(page, action, bus):  # noqa: ANN001
        return {"action_type": "noop", "outcome": "noop", "url": "https://x.test"}

    monkeypatch.setattr(coord, "capture_axtree", fake_capture_axtree)
    monkeypatch.setattr(coord, "plan_action", fake_plan_action)
    monkeypatch.setattr(coord, "execute_action", fake_execute_action)

    # Only info-level filler → distiller returns None.
    info_only = _report("security", ["no security issues detected"], "info")
    monkeypatch.setattr(coord, "AccessibilityObserver", _make_fake_observer([]))
    monkeypatch.setattr(coord, "SecurityObserver", _make_fake_observer([info_only]))
    monkeypatch.setattr(coord, "PerformanceObserver", _make_fake_observer([]))

    state = AgentState(url="https://x.test", current_step=0, max_steps=5)
    update = await coord.step_node(state, {"configurable": {"page": object()}})

    assert update["pending_hint"] is None
    # No new hint → acted_finding_hashes is NOT written (reducer left untouched).
    assert "acted_finding_hashes" not in update
