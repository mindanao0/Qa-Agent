"""Untrusted CI runner — executes pytest in the zero-secret PR context.

Trust posture:
    * Refuses to start if any forbidden env var is present (exit 126).
    * Spawns pytest via ``asyncio.create_subprocess_exec`` — never ``shell=True``.
    * Streams stdout/stderr in real time; never buffers via ``communicate()``.
    * Propagates pytest's structured exit code unchanged via ``sys.exit``.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from enum import IntEnum
from pathlib import Path
from typing import Final, Sequence

from ci.security import secret_guard

logger = logging.getLogger(__name__)

REPORTS_DIR: Final[Path] = Path("workspace/untrusted_reports")
RESULTS_XML: Final[Path] = REPORTS_DIR / "results.xml"


class PytestExitCode(IntEnum):
    """Standard pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html).

    A sentinel ``UNKNOWN = 99`` covers signals and anything outside 0..5.
    """

    OK = 0
    TESTS_FAILED = 1
    INTERRUPTED = 2
    INTERNAL_ERROR = 3
    USAGE_ERROR = 4
    NO_TESTS_COLLECTED = 5
    UNKNOWN = 99


def map_returncode(returncode: int) -> PytestExitCode:
    """Map a raw subprocess return code to a :class:`PytestExitCode` member.

    Args:
        returncode: Exit code returned by the pytest subprocess.

    Returns:
        The matching ``PytestExitCode`` enum member, or ``UNKNOWN`` for any
        out-of-range value (including signals and negative codes).
    """
    match returncode:
        case 0:
            return PytestExitCode.OK
        case 1:
            return PytestExitCode.TESTS_FAILED
        case 2:
            return PytestExitCode.INTERRUPTED
        case 3:
            return PytestExitCode.INTERNAL_ERROR
        case 4:
            return PytestExitCode.USAGE_ERROR
        case 5:
            return PytestExitCode.NO_TESTS_COLLECTED
        case _:
            return PytestExitCode.UNKNOWN


class UntrustedTestRunner:
    """Async pytest orchestrator for the untrusted CI zone."""

    def __init__(
        self,
        test_path: Path = Path("tests/e2e/"),
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        self.test_path = test_path
        self.reports_dir = reports_dir
        self.results_xml = reports_dir / "results.xml"

    def pytest_args(self) -> list[str]:
        """Return the exact argv list passed to ``create_subprocess_exec``.

        Returns:
            The argv list, starting with ``"pytest"``. Kept pure so callers
            (and tests) can introspect the command without spawning anything.
        """
        test_path_posix = self.test_path.as_posix().rstrip("/") + "/"
        return [
            "pytest",
            test_path_posix,
            "-v",
            "--numprocesses=auto",
            f"--junitxml={self.results_xml.as_posix()}",
            "--tb=short",
            "--tracing=retain-on-failure",
            "--screenshot=only-on-failure",
        ]

    async def run(self) -> PytestExitCode:
        """Run pytest as an async subprocess and stream its output.

        Returns:
            The mapped :class:`PytestExitCode`.

        Raises:
            SystemExit: From the guard via ``secret_guard.enforce_clean_environment``
                if forbidden env vars are detected (exit code 126).
        """
        secret_guard.enforce_clean_environment()

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        args = self.pytest_args()
        logger.info("untrusted_runner spawning pytest argv=%s", args)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await asyncio.gather(
            _stream(proc.stdout, sys.stdout),
            _stream(proc.stderr, sys.stderr),
        )

        returncode = await proc.wait()
        mapped = map_returncode(returncode)
        logger.info(
            "untrusted_runner pytest finished returncode=%d mapped=%s",
            returncode,
            mapped.name,
        )
        return mapped


async def _stream(reader: asyncio.StreamReader | None, sink) -> None:
    """Forward bytes line-by-line so logs appear live, not at process exit."""
    if reader is None:
        return
    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            sink.write(line.decode("utf-8", errors="replace"))
            sink.flush()
        except Exception:  # pragma: no cover — write failures are not fatal
            pass


async def _amain(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    runner = UntrustedTestRunner()
    mapped = await runner.run()
    return int(mapped.value)


def main() -> None:
    """CLI entrypoint — propagates pytest's exit code to the OS, no fall-through."""
    code = asyncio.run(_amain())
    sys.exit(code)


if __name__ == "__main__":  # pragma: no cover — covered by CI workflow
    main()


__all__ = [
    "UntrustedTestRunner",
    "PytestExitCode",
    "map_returncode",
    "main",
]
