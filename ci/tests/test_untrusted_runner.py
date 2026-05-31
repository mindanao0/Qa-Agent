"""Unit tests for ci.runners.untrusted_runner — focused on the pure-logic
exit-code mapping (the async subprocess path is exercised end-to-end in CI)."""
from __future__ import annotations

import pytest

from ci.runners.untrusted_runner import (
    PytestExitCode,
    UntrustedTestRunner,
    map_returncode,
)


@pytest.mark.parametrize(
    "returncode,expected",
    [
        (0, PytestExitCode.OK),
        (1, PytestExitCode.TESTS_FAILED),
        (2, PytestExitCode.INTERRUPTED),
        (3, PytestExitCode.INTERNAL_ERROR),
        (4, PytestExitCode.USAGE_ERROR),
        (5, PytestExitCode.NO_TESTS_COLLECTED),
        (137, PytestExitCode.UNKNOWN),
        (-1, PytestExitCode.UNKNOWN),
    ],
)
def test_map_returncode(returncode: int, expected: PytestExitCode) -> None:
    """All standard pytest exit codes map to their enum; everything else → UNKNOWN."""
    assert map_returncode(returncode) is expected


def test_runner_default_args_include_required_pytest_flags() -> None:
    """The runner's pytest command must include the spec-mandated flags."""
    runner = UntrustedTestRunner()
    args = runner.pytest_args()

    assert args[0] == "pytest"
    assert "tests/e2e/" in args
    assert "-v" in args
    assert "--numprocesses=auto" in args
    assert any(a.startswith("--junitxml=") for a in args)
    assert "--tb=short" in args
    assert "--tracing=retain-on-failure" in args
    assert "--screenshot=only-on-failure" in args


def test_runner_uses_workspace_reports_path() -> None:
    """The junit XML path must land under workspace/untrusted_reports/."""
    args = UntrustedTestRunner().pytest_args()
    junit_arg = next(a for a in args if a.startswith("--junitxml="))
    assert junit_arg.endswith("workspace/untrusted_reports/results.xml")
