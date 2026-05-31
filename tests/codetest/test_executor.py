import asyncio
import pathlib
import pytest
from src.codetest.executor import SecurityASTChecker, TestExecutor, ExecutionResult
from src.codetest.generator import GeneratedTest


def _make_test(code: str, test_type: str = "happy_path") -> GeneratedTest:
    return GeneratedTest(
        test_id="t001",
        func_id="abc1234567",
        test_code=code,
        test_type=test_type,
        metamorphic_relation=None,
    )


# ── SecurityASTChecker tests (all sync, no async needed) ──────────────────────

def test_security_allows_safe_code():
    code = "def test_add():\n    assert 1 + 1 == 2\n"
    assert SecurityASTChecker.check(code) is None


def test_security_blocks_os_import():
    code = "import os\ndef test_x():\n    assert os.getcwd()\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "os" in result


def test_security_blocks_os_path_import():
    code = "import os.path\ndef test_x():\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None


def test_security_blocks_subprocess():
    code = "import subprocess\ndef test_x():\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "subprocess" in result


def test_security_blocks_sys():
    code = "import sys\ndef test_x():\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None


def test_security_blocks_ctypes():
    code = "import ctypes\ndef test_x():\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None


def test_security_blocks_eval():
    code = "def test_x():\n    eval('1+1')\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "eval" in result


def test_security_blocks_exec():
    code = "def test_x():\n    exec('x=1')\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "exec" in result


def test_security_allows_pytest():
    code = "import pytest\ndef test_x():\n    assert True\n"
    assert SecurityASTChecker.check(code) is None


def test_security_allows_from_pytest_import():
    code = "from pytest import raises\ndef test_x():\n    assert True\n"
    assert SecurityASTChecker.check(code) is None


def test_security_returns_syntax_error_string():
    code = "def broken(:\n    pass\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "SyntaxError" in result


# ── TestExecutor tests (async, need subprocess) ───────────────────────────────

@pytest.mark.asyncio
async def test_executor_passes_simple_test(tmp_path):
    code = "def test_always_passes():\n    assert 1 + 1 == 2\n"
    executor = TestExecutor(work_dir=tmp_path)
    result = await executor.run(_make_test(code))
    assert isinstance(result, ExecutionResult)
    assert result.passed is True
    assert result.test_id == "t001"
    assert result.error_output is None


@pytest.mark.asyncio
async def test_executor_detects_failure(tmp_path):
    code = "def test_always_fails():\n    assert 1 == 2\n"
    executor = TestExecutor(work_dir=tmp_path)
    result = await executor.run(_make_test(code))
    assert result.passed is False
    assert result.error_output is not None


@pytest.mark.asyncio
async def test_executor_blocks_unsafe_code(tmp_path):
    code = "import os\ndef test_x():\n    assert os.getcwd()\n"
    executor = TestExecutor(work_dir=tmp_path)
    result = await executor.run(_make_test(code))
    assert result.passed is False
    assert "SECURITY" in (result.error_output or "")


@pytest.mark.asyncio
async def test_executor_cleans_up_temp_file(tmp_path):
    """Temp file should be deleted after run, pass or fail."""
    code = "def test_foo():\n    assert True\n"
    executor = TestExecutor(work_dir=tmp_path)
    before = set(tmp_path.glob("*.py"))
    await executor.run(_make_test(code))
    after = set(tmp_path.glob("*.py"))
    assert after == before  # no leftover files


def test_security_blocks_from_os_import():
    """from-import of a blocked module must be blocked."""
    code = "from os import getcwd\ndef test_x():\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "os" in result


def test_security_blocks_ctypes_util():
    """Dotted import of blocked module top-level is blocked."""
    code = "import ctypes.util\ndef test_x():\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None


def test_security_blocks_getattr():
    """getattr() calls are blocked to prevent eval/exec bypass."""
    code = "def test_x():\n    getattr(__builtins__, 'eval')('1+1')\n    assert True\n"
    result = SecurityASTChecker.check(code)
    assert result is not None
    assert "getattr" in result


@pytest.mark.asyncio
async def test_executor_timeout(tmp_path, monkeypatch):
    """Test that a hanging test is killed after timeout."""
    import src.codetest.executor as executor_mod
    monkeypatch.setattr(executor_mod, "_TIMEOUT_SECONDS", 2)  # 2s for fast test
    # Use a busy loop (no imports needed, not blocked by SecurityASTChecker):
    code = "def test_hang():\n    i = 0\n    while True:\n        i += 1\n    assert i > 0\n"
    executor = TestExecutor(work_dir=tmp_path)
    result = await executor.run(_make_test(code))
    assert result.passed is False
    assert result.error_output is not None
    assert "TIMEOUT" in (result.error_output or "")
