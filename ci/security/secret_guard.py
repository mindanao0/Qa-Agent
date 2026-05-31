"""Secret guard — refuse to run when secret-like env vars are present.

Imported by the untrusted CI runner so that any accidental leakage of
production credentials into a pull-request workflow halts the job with
exit code ``126`` *before* any test code is executed.

Security invariants:
    * Variable **names** are logged on violation — values are NEVER logged.
    * Shell input is sanitized with a strict allow-list approach: presence of
      any common metacharacter raises ``ValueError`` and the caller must
      refuse to execute.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Final, Mapping, Optional

logger = logging.getLogger(__name__)

FORBIDDEN_PATTERNS: Final[tuple[str, ...]] = (
    r".*KEY.*",
    r".*SECRET.*",
    r".*TOKEN.*",
    r".*PASSWORD.*",
    r".*CREDENTIAL.*",
    r".*API_KEY.*",
    r"PROD_.*",
    r"AWS_SECRET.*",
)

_COMPILED_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p) for p in FORBIDDEN_PATTERNS
)

EXIT_FORBIDDEN_ENV: Final[int] = 126

# Shell metacharacters we refuse to forward to any subprocess.
_FORBIDDEN_SHELL_CHARS: Final[frozenset[str]] = frozenset(
    ";|&$`\\\"'<>(){}[]*?~"
)


def scan_environment(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Return the sorted list of env-var names that match a forbidden pattern.

    Args:
        env: Optional mapping to scan. Defaults to ``os.environ``.

    Returns:
        Sorted list of offending variable **names** (values are never returned).
    """
    source: Mapping[str, str] = env if env is not None else os.environ
    violations: set[str] = set()
    for name in source:
        for pattern in _COMPILED_PATTERNS:
            if pattern.fullmatch(name):
                violations.add(name)
                break
    return sorted(violations)


def enforce_clean_environment(env: Optional[Mapping[str, str]] = None) -> None:
    """Abort the process with exit 126 if any forbidden env var is present.

    Args:
        env: Optional mapping to scan. Defaults to ``os.environ``.

    Raises:
        SystemExit: With code ``126`` when violations are found.
    """
    violations = scan_environment(env=env)
    if not violations:
        return

    logger.error(
        "Untrusted CI guard refusing to run: %d forbidden env var(s) detected",
        len(violations),
    )
    for name in violations:
        logger.error("forbidden_env_var=%s", name)

    sys.exit(EXIT_FORBIDDEN_ENV)


def sanitize_shell_input(value: str) -> str:
    """Return ``value`` if free of shell metacharacters; raise otherwise.

    Args:
        value: A candidate command-line argument (typically PR title or branch).

    Returns:
        The input string, unchanged, when it contains no forbidden character.

    Raises:
        ValueError: When any shell metacharacter is present. The caller MUST
            refuse to execute the underlying command.
    """
    if any(ch in _FORBIDDEN_SHELL_CHARS for ch in value):
        logger.warning("sanitize_shell_input rejected input (length=%d)", len(value))
        raise ValueError(
            "Input contains shell metacharacters; refusing to forward to subprocess."
        )
    return value


__all__ = [
    "FORBIDDEN_PATTERNS",
    "EXIT_FORBIDDEN_ENV",
    "scan_environment",
    "enforce_clean_environment",
    "sanitize_shell_input",
]
