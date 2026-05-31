"""Unit tests for ci.validators.xml_schema_validator."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ci.validators import xml_schema_validator as v

VALID_JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="suite1" tests="2" failures="0" errors="0" skipped="0" time="0.5">
    <testcase classname="t.A" name="test_passes" time="0.1">
      <properties>
        <property name="ai_semantic_consistency_score" value="0.92" />
        <property name="e2e_execution_latency_sec" value="0.124" />
      </properties>
    </testcase>
    <testcase classname="t.A" name="test_skip" time="0.0">
      <skipped message="reason" />
    </testcase>
  </testsuite>
</testsuites>
"""

DOCTYPE_ATTACK = """<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<testsuites><testsuite name="x"><testcase name="y" classname="z"/></testsuite></testsuites>
"""

EXTERNAL_ENTITY_ATTACK = """<?xml version="1.0"?>
<!DOCTYPE root [<!ENTITY a "boom">]>
<testsuites>&a;</testsuites>
"""

NON_STANDARD_ROOT = """<?xml version="1.0"?>
<malicious_payload><exec>rm -rf /</exec></malicious_payload>
"""

PROPERTIES_OUTSIDE_TESTCASE = """<?xml version="1.0"?>
<testsuites>
  <testsuite name="s" tests="1" failures="0" errors="0">
    <properties>
      <property name="ai_semantic_consistency_score" value="0.99" />
    </properties>
    <testcase name="x" classname="y" time="0" />
  </testsuite>
  <evil_node />
</testsuites>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_valid_junit_xml_passes(tmp_path: Path) -> None:
    """A well-formed JUnit XML with permitted properties validates cleanly."""
    p = _write(tmp_path, "ok.xml", VALID_JUNIT)
    assert v.validate_file(p) is True


def test_rejects_doctype(tmp_path: Path) -> None:
    """XXE-style DOCTYPE declarations are refused outright."""
    p = _write(tmp_path, "doctype.xml", DOCTYPE_ATTACK)
    assert v.validate_file(p) is False


def test_rejects_external_entity(tmp_path: Path) -> None:
    """External entity references are refused."""
    p = _write(tmp_path, "entity.xml", EXTERNAL_ENTITY_ATTACK)
    assert v.validate_file(p) is False


def test_rejects_non_standard_root(tmp_path: Path) -> None:
    """Anything whose root element isn't testsuite(s) is refused."""
    p = _write(tmp_path, "alien.xml", NON_STANDARD_ROOT)
    assert v.validate_file(p) is False


def test_rejects_unexpected_root_siblings(tmp_path: Path) -> None:
    """Unexpected elements anywhere in the tree are refused."""
    p = _write(tmp_path, "evil.xml", PROPERTIES_OUTSIDE_TESTCASE)
    assert v.validate_file(p) is False


def test_logs_omit_raw_xml(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Validator must NEVER log raw XML payloads on failure (log injection vector)."""
    p = _write(tmp_path, "doctype.xml", DOCTYPE_ATTACK)
    with caplog.at_level("ERROR"):
        v.validate_file(p)
    # Raw XML body must not appear verbatim:
    assert "<!DOCTYPE" not in caplog.text
    assert "file:///etc/passwd" not in caplog.text


def test_cli_returns_zero_for_valid_xml(tmp_path: Path) -> None:
    """CLI exits 0 for a valid file."""
    p = _write(tmp_path, "ok.xml", VALID_JUNIT)
    result = subprocess.run(
        [sys.executable, "-m", "ci.validators.xml_schema_validator", str(p)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_returns_nonzero_for_invalid_xml(tmp_path: Path) -> None:
    """CLI exits 1 for an invalid file."""
    p = _write(tmp_path, "bad.xml", DOCTYPE_ATTACK)
    result = subprocess.run(
        [sys.executable, "-m", "ci.validators.xml_schema_validator", str(p)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
