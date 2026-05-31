"""
TestExecutor — Sprint 6.

SecurityASTChecker blocks dangerous imports/calls before execution.
TestExecutor sandboxes pytest in a subprocess with 30s timeout.
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import re
import tempfile
import uuid

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.generator import GeneratedTest

_BLOCKED_IMPORTS: frozenset[str] = frozenset({"os", "sys", "subprocess", "ctypes"})
_TIMEOUT_SECONDS = 30


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str
    passed: bool
    error_output: str | None


class SecurityASTChecker:
    """AST-level static checker for generated test code."""

    @staticmethod
    def check(code: str) -> str | None:
        """Return None if safe, error string if blocked."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in _BLOCKED_IMPORTS:
                        return f"blocked import: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                top = module.split(".")[0]
                if top in _BLOCKED_IMPORTS:
                    return f"blocked import from: {module}"

            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in ("eval", "exec", "getattr"):
                    return f"blocked call: {func.id}()"
                if isinstance(func, ast.Attribute) and func.attr in ("eval", "exec"):
                    return f"blocked attribute call: .{func.attr}()"

        return None


def _parse_pytest_outcome(stdout: str) -> bool:
    """True if pytest reports passing with no failures/errors."""
    lower = stdout.lower()
    if re.search(r"\b(failed|error)\b", lower):
        return False
    if re.search(r"\bpassed\b", lower):
        return True
    return False


class TestExecutor:
    """Runs generated pytest code in a sandboxed subprocess."""

    def __init__(self, work_dir: pathlib.Path | None = None) -> None:
        self._work_dir = work_dir or pathlib.Path(tempfile.gettempdir())

    async def run(self, test: GeneratedTest) -> ExecutionResult:
        code = test.test_code

        security_error = SecurityASTChecker.check(code)
        if security_error is not None:
            logger.warning(f"TestExecutor: SECURITY block test_id={test.test_id!r}: {security_error}")
            return ExecutionResult(
                test_id=test.test_id,
                passed=False,
                error_output=f"SECURITY: {security_error}",
            )

        tmp_file = self._work_dir / f"test_sprint6_{uuid.uuid4().hex[:8]}.py"
        try:
            tmp_file.write_text(code, encoding="utf-8")
            return await self._execute(test.test_id, tmp_file)
        finally:
            tmp_file.unlink(missing_ok=True)

    async def _execute(self, test_id: str, tmp_file: pathlib.Path) -> ExecutionResult:
        import os
        import signal
        import subprocess as _sp
        import sys

        env = dict(os.environ)
        project_root = str(pathlib.Path(__file__).parent.parent.parent)
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        for cmd in (
            ["uv", "run", "pytest", str(tmp_file), "-v", "--tb=short"],
            ["python", "-m", "pytest", str(tmp_file), "-v", "--tb=short"],
        ):
            try:
                spawn_kwargs: dict = {}
                if sys.platform == "win32":
                    spawn_kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP
                else:
                    spawn_kwargs["start_new_session"] = True

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    **spawn_kwargs,
                )
                try:
                    stdout_bytes, _ = await asyncio.wait_for(
                        proc.communicate(), timeout=_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"TestExecutor: timeout test_id={test_id!r} — killing process tree")
                    if sys.platform == "win32":
                        _sp.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            check=False,
                            capture_output=True,
                        )
                    else:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            proc.kill()
                    try:
                        await asyncio.wait_for(proc.communicate(), timeout=5)
                    except (asyncio.TimeoutError, Exception):
                        pass
                    return ExecutionResult(
                        test_id=test_id, passed=False, error_output="TIMEOUT: exceeded 30s"
                    )

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                passed = _parse_pytest_outcome(stdout)
                return ExecutionResult(
                    test_id=test_id,
                    passed=passed,
                    error_output=None if passed else (stdout[:1000] + "\n...<truncated>" if len(stdout) > 1000 else stdout),
                )

            except (FileNotFoundError, OSError) as exc:
                if cmd[0] == "python":
                    # Both uv and python failed — give up
                    return ExecutionResult(
                        test_id=test_id,
                        passed=False,
                        error_output=f"EXECUTOR: runner failed — {exc}",
                    )
                # uv not found — fall through to python fallback
                continue

        # Should not reach here
        return ExecutionResult(test_id=test_id, passed=False, error_output="EXECUTOR: no runner found")


__all__ = ["SecurityASTChecker", "TestExecutor", "ExecutionResult"]
