"""JSTestExecutor — Sprint 9. Runs Vitest tests via subprocess, parses JSON results."""
from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
import uuid

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_generator import GeneratedJSTest
from src.observability.tracer import OTelTracer

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_VITEST_TIMEOUT = 60

_tracer = OTelTracer()


def _normalize_escapes(code: str) -> str:
    """Fix LLM double-escaped newlines: literal backslash-n → actual newline."""
    # Safe unconditional: '\\n' (2 chars) → '\n' (1 char); actual newlines (1 char) are unaffected
    return code.replace("\\n", "\n")


class JSTestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    func_id: str
    passed: bool
    error: str | None
    duration_ms: float


class JSTestExecutor:
    """Runs generated Vitest tests via subprocess. Returns JSTestResult per test."""

    def __init__(self, project_root: pathlib.Path | None = None) -> None:
        self._root = project_root or _PROJECT_ROOT

    async def run(
        self,
        tests: list[GeneratedJSTest],
        work_dir: pathlib.Path,
    ) -> list[JSTestResult]:
        if not tests:
            return []
        work_dir.mkdir(parents=True, exist_ok=True)

        test_files: list[pathlib.Path] = []
        for t in tests:
            p = work_dir / f"test_{t.test_id}.test.ts"
            code = _normalize_escapes(t.test_code)
            p.write_text(code, encoding="utf-8")
            test_files.append(p)

        results_file = work_dir / f"vitest_results_{uuid.uuid4().hex[:8]}.json"

        async with _tracer.span("js_test.execute", test_count=len(tests)):
            result_map = await asyncio.to_thread(
                self._run_vitest, test_files, results_file
            )

        out: list[JSTestResult] = []
        for t in tests:
            info = result_map.get(
                t.test_id,
                {"passed": False, "error": "no vitest result", "duration_ms": 0.0},
            )
            out.append(JSTestResult(
                test_id=t.test_id,
                func_id=t.func_id,
                passed=info["passed"],
                error=info.get("error"),
                duration_ms=float(info.get("duration_ms", 0.0)),
            ))

        results_file.unlink(missing_ok=True)
        return out

    def _run_vitest(
        self,
        test_files: list[pathlib.Path],
        results_file: pathlib.Path,
    ) -> dict[str, dict]:
        # On Windows, npx is a .cmd script and needs shell=True (or explicit npx.cmd)
        npx_bin = "npx.cmd" if sys.platform == "win32" else "npx"
        cmd = [
            npx_bin, "vitest", "run",
            "--reporter=json",
            f"--outputFile={results_file.absolute()}",
        ] + [str(f.absolute()) for f in test_files]

        try:
            subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                timeout=_VITEST_TIMEOUT,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.error(f"JSTestExecutor: vitest failed: {exc!r}")
            return {}

        if not results_file.exists():
            logger.warning("JSTestExecutor: results file not written")
            return {
                f.stem.removeprefix("test_"): {
                    "passed": False,
                    "error": "vitest did not produce output",
                    "duration_ms": 0.0,
                }
                for f in test_files
            }

        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"JSTestExecutor: bad results JSON: {exc}")
            return {}

        result_map: dict[str, dict] = {}
        for suite in data.get("testResults", []):
            # vitest JSON reporter uses "name" (not "testFilePath") for the file path
            path_str = suite.get("name", "") or suite.get("testFilePath", "")
            fname = pathlib.Path(path_str).name
            test_id = fname.removesuffix(".test.ts").removeprefix("test_")
            status = suite.get("status", "failed")
            passed = status == "passed"
            duration = sum(
                float(r.get("duration") or 0)
                for r in suite.get("assertionResults", [])
            )
            error: str | None = None
            if not passed:
                msgs = [
                    m for r in suite.get("assertionResults", [])
                    for m in (r.get("failureMessages") or [])
                ]
                error = "; ".join(msgs)[:500] if msgs else "test failed"
            result_map[test_id] = {
                "passed": passed,
                "error": error,
                "duration_ms": duration,
            }

        return result_map


__all__ = ["JSTestExecutor", "JSTestResult"]
