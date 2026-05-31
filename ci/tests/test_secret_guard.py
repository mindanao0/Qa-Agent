"""Unit tests for ci.security.secret_guard."""
from __future__ import annotations

import pytest

from ci.security import secret_guard


def test_scan_environment_detects_token_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """scan_environment surfaces any var whose name matches a forbidden pattern."""
    monkeypatch.setenv("PROD_API_KEY", "xxx")
    monkeypatch.setenv("HARMLESS_VAR", "ok")

    violations = secret_guard.scan_environment(env={"PROD_API_KEY": "xxx", "HARMLESS_VAR": "ok"})

    assert "PROD_API_KEY" in violations
    assert "HARMLESS_VAR" not in violations


def test_scan_environment_detects_password_secret_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Common secret-like names (PASSWORD, SECRET, TOKEN, CREDENTIAL) are all caught."""
    env = {
        "DB_PASSWORD": "x",
        "MY_SECRET": "x",
        "GITHUB_TOKEN": "x",
        "AWS_CREDENTIAL_FILE": "x",
        "AWS_SECRET_ACCESS_KEY": "x",
        "BENIGN": "x",
    }
    violations = set(secret_guard.scan_environment(env=env))

    assert {"DB_PASSWORD", "MY_SECRET", "GITHUB_TOKEN",
            "AWS_CREDENTIAL_FILE", "AWS_SECRET_ACCESS_KEY"} <= violations
    assert "BENIGN" not in violations


def test_scan_environment_empty_when_clean() -> None:
    """A clean environment produces no violations."""
    assert secret_guard.scan_environment(env={"FOO": "bar", "BAZ": "qux"}) == []


def test_enforce_clean_environment_exits_126_on_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enforce_clean_environment hard-exits with code 126 when secrets are present."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leak")

    with pytest.raises(SystemExit) as exc_info:
        secret_guard.enforce_clean_environment()

    assert exc_info.value.code == 126


def test_enforce_clean_environment_logs_only_names(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When refusing to run, the guard logs variable NAMES but never values."""
    monkeypatch.setenv("PROD_API_KEY", "super-secret-value-xyz")

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            secret_guard.enforce_clean_environment()

    full_log = caplog.text
    assert "PROD_API_KEY" in full_log
    assert "super-secret-value-xyz" not in full_log


def test_enforce_clean_environment_passes_when_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env is clean, enforce_clean_environment is a no-op (no exit)."""
    for key in list(monkeypatch._setenv if hasattr(monkeypatch, "_setenv") else []):
        monkeypatch.delenv(key, raising=False)
    # Provide a sanitized env explicitly:
    secret_guard.enforce_clean_environment(env={"PATH": "/usr/bin", "LANG": "C"})


def test_sanitize_shell_input_passes_safe_string() -> None:
    """A string without shell metacharacters is returned unchanged."""
    assert secret_guard.sanitize_shell_input("feature-branch-42") == "feature-branch-42"


@pytest.mark.parametrize(
    "dangerous",
    [
        "rm -rf /; echo pwned",
        "foo | bar",
        "$(whoami)",
        "`id`",
        "a && b",
        "x > out.txt",
        "x < in.txt",
        "name with (parens)",
        "name with {braces}",
        "name with [brackets]",
        "name with *glob?",
        "name with ~tilde",
        'name with "quotes"',
        "name with 'single'",
        "backslash\\here",
    ],
)
def test_sanitize_shell_input_rejects_metacharacters(dangerous: str) -> None:
    """Any unsafe shell metacharacter triggers ValueError; nothing is executed."""
    with pytest.raises(ValueError):
        secret_guard.sanitize_shell_input(dangerous)
