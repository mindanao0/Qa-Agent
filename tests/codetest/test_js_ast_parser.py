"""Unit tests for JSASTParser (mocked subprocess)."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.codetest.js_ast_parser import JSASTParser, JSFunctionSpec


MOCK_WALKER_OUTPUT = json.dumps([
    {
        "func_name": "clamp",
        "params": ["value", "min", "max"],
        "return_type": "number",
        "is_async": False,
        "is_exported": True,
        "jsdoc": "* Clamp a number",
        "complexity": 3,
    },
    {
        "func_name": "sum",
        "params": ["arr"],
        "return_type": "number",
        "is_async": False,
        "is_exported": True,
        "jsdoc": None,
        "complexity": 2,
    },
])


def _make_proc(stdout: str, returncode: int = 0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


def test_parse_file_returns_specs(tmp_path: Path):
    fake_ts = tmp_path / "utils.ts"
    fake_ts.write_text("export function clamp() {}")
    with patch("subprocess.run", return_value=_make_proc(MOCK_WALKER_OUTPUT)):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_file(fake_ts))
    assert len(specs) == 2
    assert specs[0].func_name == "clamp"
    assert specs[0].is_exported is True
    assert specs[0].complexity == 3
    assert isinstance(specs[0].func_id, str) and len(specs[0].func_id) == 10


def test_parse_file_returns_empty_on_walker_error(tmp_path: Path):
    fake_ts = tmp_path / "broken.ts"
    fake_ts.write_text("")
    with patch("subprocess.run", return_value=_make_proc("", returncode=1)):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_file(fake_ts))
    assert specs == []


def test_parse_file_returns_empty_on_bad_json(tmp_path: Path):
    fake_ts = tmp_path / "bad.ts"
    fake_ts.write_text("")
    with patch("subprocess.run", return_value=_make_proc("NOT JSON")):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_file(fake_ts))
    assert specs == []


def test_jsfunctionspec_extra_forbid():
    with pytest.raises(Exception):
        JSFunctionSpec(
            func_id="abc", module_path="x.ts", func_name="f",
            params=[], return_type=None, is_async=False,
            is_exported=True, jsdoc=None, complexity=1,
            unexpected="bad",
        )


def test_parse_dir_aggregates_files(tmp_path: Path):
    (tmp_path / "a.ts").write_text("export function a() {}")
    (tmp_path / "b.ts").write_text("export function b() {}")
    single = json.dumps([{
        "func_name": "x", "params": [], "return_type": None,
        "is_async": False, "is_exported": True, "jsdoc": None, "complexity": 1,
    }])
    with patch("subprocess.run", return_value=_make_proc(single)):
        import asyncio
        specs = asyncio.run(JSASTParser().parse_dir(tmp_path, glob="*.ts"))
    assert len(specs) == 2  # one per file
