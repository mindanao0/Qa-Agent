"""Privileged JUnit-XML reporter.

Runs ONLY in the trusted (workflow_run) zone, after the untrusted runner has
produced a results.xml artifact. Inputs are still treated as untrusted: parsing
uses ``defusedxml`` to block XXE / billion-laughs attacks.

Pipeline:
    1. ``parse_junit_xml`` → :class:`TestSuiteReport`
    2. ``enforce_quality_gate`` raises :class:`QualityGateError` on regressions
    3. ``to_summary_json`` produces a sanitized JSON payload for PR annotation
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from statistics import mean
from typing import Final, Optional
from urllib.parse import urlparse

import defusedxml.ElementTree as defused_etree
from defusedxml.common import DefusedXmlException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MIN_MEAN_AI_SCORE: Final[float] = 0.85

# Property keys we know about. Anything else in <properties> is preserved as
# free-form metadata but is not surfaced via dedicated fields.
_AI_SCORE_KEY: Final[str] = "ai_semantic_consistency_score"
_LATENCY_KEY: Final[str] = "e2e_execution_latency_sec"
_FAILURE_TRACE_KEY: Final[str] = "failure_trace"
_ATTACHMENT_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^attachment\d+$")

# Strip control chars + angle brackets out of any text we may surface in a PR
# comment so a malicious test name cannot inject markdown / HTML.
_PR_SAFE_RE: Final[re.Pattern[str]] = re.compile(r"[<>`\x00-\x1f\x7f]")


class QualityGateError(RuntimeError):
    """Raised when a parsed report violates the quality gate."""


class TestSuiteReport(BaseModel):
    """Aggregated, sanitized view of a JUnit XML file."""

    # Stop pytest from trying to collect this Pydantic model as a test class.
    __test__ = False

    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    ai_semantic_scores: list[float] = Field(default_factory=list)
    execution_latencies: list[float] = Field(default_factory=list)
    playwright_trace_urls: list[str] = Field(default_factory=list)
    failure_traces: list[str] = Field(default_factory=list)


def _safe_text(value: str, *, limit: int = 240) -> str:
    """Strip control chars / angle brackets and truncate for PR display."""
    cleaned = _PR_SAFE_RE.sub("", value)
    return cleaned[:limit]


def _looks_like_trace_url(value: str) -> bool:
    """Return True for http(s) URLs only (data: URIs are not trace links)."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class PrivilegedReporter:
    """Trusted artifact processor.

    All public methods are ``async``; XML parsing is dispatched via
    ``asyncio.to_thread`` so the event loop is not blocked on large reports.
    """

    async def parse_junit_xml(self, xml_path: Path) -> TestSuiteReport:
        """Parse ``xml_path`` into a :class:`TestSuiteReport`.

        Args:
            xml_path: Path to a JUnit XML file produced by pytest.

        Returns:
            A :class:`TestSuiteReport` summarising counts and AI telemetry.

        Raises:
            ValueError: If the file is malformed or contains forbidden XML
                features (DOCTYPE, external entities, billion-laughs).
            FileNotFoundError: If ``xml_path`` does not exist.
        """
        path = Path(xml_path)
        if not path.is_file():
            raise FileNotFoundError(f"junit xml not found: {path}")

        def _parse() -> TestSuiteReport:
            try:
                tree = defused_etree.parse(
                    str(path),
                    forbid_dtd=True,
                    forbid_entities=True,
                    forbid_external=True,
                )
            except DefusedXmlException as exc:
                raise ValueError(
                    f"junit xml rejected by defusedxml: {type(exc).__name__}"
                ) from exc
            return _build_report(tree.getroot())

        return await asyncio.to_thread(_parse)

    async def enforce_quality_gate(self, report: TestSuiteReport) -> None:
        """Raise :class:`QualityGateError` if the report fails the gate.

        Gate rules (any one trips):
            * ``failed > 0``
            * ``errors > 0``
            * mean of ``ai_semantic_scores`` < 0.85 (only when scores exist)

        Args:
            report: Parsed report from :meth:`parse_junit_xml`.

        Raises:
            QualityGateError: When any rule is violated.
        """
        violations: list[str] = []
        if report.failed > 0:
            violations.append(f"failed={report.failed}")
        if report.errors > 0:
            violations.append(f"errors={report.errors}")
        if report.ai_semantic_scores:
            mean_score = mean(report.ai_semantic_scores)
            if mean_score < MIN_MEAN_AI_SCORE:
                violations.append(
                    f"mean_ai_semantic_score={mean_score:.3f}<{MIN_MEAN_AI_SCORE}"
                )

        if violations:
            joined = ",".join(violations)
            logger.error("quality_gate_failed reasons=%s", joined)
            raise QualityGateError(f"Quality gate failed: {joined}")

    def to_summary_json(self, report: TestSuiteReport) -> str:
        """Return a JSON-string summary safe to drop into a PR comment.

        Args:
            report: Parsed report.

        Returns:
            A JSON string with totals, mean AI score (or ``null``) and a
            *sanitized* list of failure trace previews suitable for posting.
        """
        mean_ai: Optional[float] = (
            round(mean(report.ai_semantic_scores), 4)
            if report.ai_semantic_scores
            else None
        )
        mean_latency: Optional[float] = (
            round(mean(report.execution_latencies), 4)
            if report.execution_latencies
            else None
        )
        return json.dumps(
            {
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "errors": report.errors,
                "skipped": report.skipped,
                "mean_ai_semantic_score": mean_ai,
                "mean_execution_latency_sec": mean_latency,
                "playwright_trace_urls": [
                    _safe_text(u) for u in report.playwright_trace_urls[:25]
                ],
                "failure_trace_previews": [
                    _safe_text(f) for f in report.failure_traces[:25]
                ],
            },
            ensure_ascii=True,
            indent=2,
        )


def _local(tag: str) -> str:
    """Return the local-name of an XML tag (strip any namespace prefix)."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _collect_testcases(root) -> list:
    """Find every <testcase> regardless of <testsuites>/<testsuite> nesting."""
    cases = []
    if _local(root.tag) == "testsuites":
        for suite in root:
            if _local(suite.tag) == "testsuite":
                cases.extend(c for c in suite if _local(c.tag) == "testcase")
    elif _local(root.tag) == "testsuite":
        cases.extend(c for c in root if _local(c.tag) == "testcase")
    return cases


def _build_report(root) -> TestSuiteReport:
    cases = _collect_testcases(root)

    total = len(cases)
    failed = 0
    errors = 0
    skipped = 0
    ai_scores: list[float] = []
    latencies: list[float] = []
    trace_urls: list[str] = []
    failure_traces: list[str] = []

    for case in cases:
        child_tags = {_local(c.tag) for c in case}
        if "failure" in child_tags:
            failed += 1
        if "error" in child_tags:
            errors += 1
        if "skipped" in child_tags:
            skipped += 1

        for child in case:
            local = _local(child.tag)
            if local == "properties":
                for prop in child:
                    if _local(prop.tag) != "property":
                        continue
                    name = prop.attrib.get("name", "")
                    value = prop.attrib.get("value", "")
                    if name == _AI_SCORE_KEY:
                        try:
                            ai_scores.append(float(value))
                        except ValueError:
                            logger.warning("invalid ai_score value ignored")
                    elif name == _LATENCY_KEY:
                        try:
                            latencies.append(float(value))
                        except ValueError:
                            logger.warning("invalid latency value ignored")
                    elif _ATTACHMENT_KEY_RE.match(name) and _looks_like_trace_url(value):
                        trace_urls.append(value)
                    elif name == _FAILURE_TRACE_KEY:
                        failure_traces.append(value)
            elif local in {"failure", "error"}:
                msg = child.attrib.get("message", "")
                if msg:
                    failure_traces.append(msg)

    passed = max(0, total - failed - errors - skipped)

    return TestSuiteReport(
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        ai_semantic_scores=ai_scores,
        execution_latencies=latencies,
        playwright_trace_urls=trace_urls,
        failure_traces=failure_traces,
    )


__all__ = [
    "PrivilegedReporter",
    "TestSuiteReport",
    "QualityGateError",
    "MIN_MEAN_AI_SCORE",
]
