"""Unit tests for JSTestExecutor (mocked subprocess)."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.codetest.js_generator import GeneratedJSTest
from src.codetest.js_executor import JSTestExecutor, JSTestResult


def _test(test_id: str = "abc123") -> GeneratedJSTest:
    return GeneratedJSTest(
        test_id=test_id, func_id="f1",
        test_code="import { it, expect } from 'vitest'\nit('x',()=>{expect(1).toBe(1)})",
        test_type="happy_path", metamorphic_relation=None,
    )


def _make_proc(returncode: int = 0):
    p = MagicMock()
    p.returncode = returncode
    p.stdout = b""
    p.stderr = b""
    return p


def _vitest_json(file_path: str, status: str = "passed", duration: float = 10.0) -> dict:
    return {
        "testResults": [{
            "testFilePath": file_path,
            "status": status,
            "assertionResults": [{
                "status": status, "title": "works",
                "duration": duration, "failureMessages": [],
            }],
        }]
    }


def test_run_returns_passed_result(tmp_path: Path):
    executor = JSTestExecutor(project_root=tmp_path)
    test = _test("abc123")

    def fake_run(cmd, **kwargs):
        # Find --outputFile arg and write fake results
        for arg in cmd:
            arg_str = str(arg)
            if arg_str.startswith("--outputFile="):
                results_path = Path(arg_str.replace("--outputFile=", ""))
                # Find test file path from cmd args
                test_file = next((a for a in cmd if "test_abc123" in str(a)), str(tmp_path / "tests" / "test_abc123.test.ts"))
                results_path.parent.mkdir(parents=True, exist_ok=True)
                results_path.write_text(json.dumps(_vitest_json(str(test_file), "passed", 12.0)))
                break
        return _make_proc(0)

    with patch("subprocess.run", side_effect=fake_run):
        results = asyncio.run(executor.run([test], tmp_path / "tests"))

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].test_id == "abc123"
    assert results[0].duration_ms == 12.0


def test_run_returns_failed_when_results_file_missing(tmp_path: Path):
    executor = JSTestExecutor(project_root=tmp_path)
    test = _test("fail999")
    with patch("subprocess.run", return_value=_make_proc(1)):
        results = asyncio.run(executor.run([test], tmp_path / "tests"))
    assert len(results) == 1
    assert results[0].passed is False


def test_js_test_result_extra_forbid():
    with pytest.raises(Exception):
        JSTestResult(test_id="x", func_id="y", passed=True, error=None, duration_ms=1.0, bad="oops")


def test_run_empty_tests_returns_empty(tmp_path: Path):
    executor = JSTestExecutor(project_root=tmp_path)
    results = asyncio.run(executor.run([], tmp_path / "tests"))
    assert results == []
