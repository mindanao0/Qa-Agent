"""Unit tests for ci.runners.privileged_reporter."""
from __future__ import annotations

from pathlib import Path

import pytest

from ci.runners.privileged_reporter import (
    PrivilegedReporter,
    QualityGateError,
    TestSuiteReport,
)


def _write_xml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


PASSING_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="suite" tests="2" failures="0" errors="0" skipped="0">
    <testcase classname="t.A" name="ok_one" time="0.1">
      <properties>
        <property name="ai_semantic_consistency_score" value="0.95"/>
        <property name="e2e_execution_latency_sec" value="0.21"/>
        <property name="attachment1" value="data:image/png;base64,IIII"/>
      </properties>
    </testcase>
    <testcase classname="t.A" name="ok_two" time="0.2">
      <properties>
        <property name="ai_semantic_consistency_score" value="0.88"/>
        <property name="e2e_execution_latency_sec" value="0.33"/>
        <property name="attachment1" value="https://example.com/traces/run/a.zip"/>
      </properties>
    </testcase>
  </testsuite>
</testsuites>
"""

FAILING_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="suite" tests="2" failures="1" errors="0" skipped="0">
    <testcase classname="t.A" name="ok_one" time="0.1">
      <properties>
        <property name="ai_semantic_consistency_score" value="0.95"/>
      </properties>
    </testcase>
    <testcase classname="t.A" name="bad_one" time="0.2">
      <failure message="boom" type="AssertionError">trace</failure>
      <properties>
        <property name="failure_trace" value="line 12"/>
      </properties>
    </testcase>
  </testsuite>
</testsuites>
"""

LOW_AI_SCORE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="suite" tests="2" failures="0" errors="0" skipped="0">
    <testcase classname="t.A" name="ok_one" time="0.1">
      <properties>
        <property name="ai_semantic_consistency_score" value="0.30"/>
      </properties>
    </testcase>
    <testcase classname="t.A" name="ok_two" time="0.1">
      <properties>
        <property name="ai_semantic_consistency_score" value="0.40"/>
      </properties>
    </testcase>
  </testsuite>
</testsuites>
"""

ERROR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="suite" tests="1" failures="0" errors="1" skipped="0">
    <testcase classname="t.A" name="ok_one" time="0.1">
      <error message="setup failed">Traceback...</error>
    </testcase>
  </testsuite>
</testsuites>
"""


async def test_parse_counts_pass_fail_skip_error(tmp_path: Path) -> None:
    """parse_junit_xml returns a TestSuiteReport with correct totals."""
    p = _write_xml(tmp_path, "good.xml", PASSING_XML)
    report = await PrivilegedReporter().parse_junit_xml(p)

    assert isinstance(report, TestSuiteReport)
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.errors == 0
    assert report.skipped == 0


async def test_parse_extracts_ai_scores_and_latencies(tmp_path: Path) -> None:
    """ai_semantic_scores and execution_latencies are extracted from properties."""
    p = _write_xml(tmp_path, "good.xml", PASSING_XML)
    report = await PrivilegedReporter().parse_junit_xml(p)

    assert report.ai_semantic_scores == [0.95, 0.88]
    assert report.execution_latencies == [0.21, 0.33]


async def test_parse_extracts_attachment_urls(tmp_path: Path) -> None:
    """attachmentN properties that look like trace URLs are collected."""
    p = _write_xml(tmp_path, "good.xml", PASSING_XML)
    report = await PrivilegedReporter().parse_junit_xml(p)

    assert "https://example.com/traces/run/a.zip" in report.playwright_trace_urls


async def test_quality_gate_passes_for_healthy_run(tmp_path: Path) -> None:
    """A passing run with high AI scores satisfies the gate."""
    p = _write_xml(tmp_path, "good.xml", PASSING_XML)
    reporter = PrivilegedReporter()
    report = await reporter.parse_junit_xml(p)

    # No exception.
    await reporter.enforce_quality_gate(report)


async def test_quality_gate_fails_on_failures(tmp_path: Path) -> None:
    """Any failure trips the gate."""
    p = _write_xml(tmp_path, "fail.xml", FAILING_XML)
    reporter = PrivilegedReporter()
    report = await reporter.parse_junit_xml(p)

    with pytest.raises(QualityGateError):
        await reporter.enforce_quality_gate(report)


async def test_quality_gate_fails_on_low_ai_score(tmp_path: Path) -> None:
    """Mean AI semantic score below 0.85 trips the gate even on pass runs."""
    p = _write_xml(tmp_path, "low.xml", LOW_AI_SCORE_XML)
    reporter = PrivilegedReporter()
    report = await reporter.parse_junit_xml(p)

    with pytest.raises(QualityGateError):
        await reporter.enforce_quality_gate(report)


async def test_quality_gate_fails_on_errors(tmp_path: Path) -> None:
    """Any error count > 0 trips the gate."""
    p = _write_xml(tmp_path, "err.xml", ERROR_XML)
    reporter = PrivilegedReporter()
    report = await reporter.parse_junit_xml(p)

    with pytest.raises(QualityGateError):
        await reporter.enforce_quality_gate(report)


async def test_to_summary_json_returns_serializable_summary(tmp_path: Path) -> None:
    """to_summary_json returns a JSON-string summary suitable for PR comments."""
    p = _write_xml(tmp_path, "good.xml", PASSING_XML)
    reporter = PrivilegedReporter()
    report = await reporter.parse_junit_xml(p)

    summary = reporter.to_summary_json(report)

    import json

    parsed = json.loads(summary)
    assert parsed["total"] == 2
    assert parsed["failed"] == 0
    assert "mean_ai_semantic_score" in parsed


async def test_parse_rejects_xxe_payload(tmp_path: Path) -> None:
    """Files with a DOCTYPE / external entity are refused by defusedxml."""
    xxe = """<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<testsuites><testsuite name="x"><testcase classname="a" name="y"/></testsuite></testsuites>
"""
    p = _write_xml(tmp_path, "xxe.xml", xxe)
    with pytest.raises(ValueError):
        await PrivilegedReporter().parse_junit_xml(p)
