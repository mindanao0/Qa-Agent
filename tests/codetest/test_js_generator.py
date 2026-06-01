"""Unit tests for JSTestGenerator (mocked InstructorClient)."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codetest.js_ast_parser import JSFunctionSpec
from src.codetest.js_generator import GeneratedJSTest, JSTestGenerator


def _spec(name: str = "clamp", complexity: int = 3) -> JSFunctionSpec:
    return JSFunctionSpec(
        func_id="abc1234567",
        module_path="src/js_targets/utils.ts",
        func_name=name,
        params=["value", "min", "max"],
        return_type="number",
        is_async=False,
        is_exported=True,
        jsdoc="* Clamp a number",
        complexity=complexity,
    )


def _mock_test(func_id: str = "abc1234567") -> GeneratedJSTest:
    return GeneratedJSTest(
        test_id="test_clamp",
        func_id=func_id,
        test_code=(
            "function clamp(v,mn,mx){return v<mn?mn:v>mx?mx:v}\n"
            "import { describe, it, expect } from 'vitest'\n"
            "describe('clamp',()=>{it('works',()=>{expect(clamp(5,0,10)).toBe(5)})})"
        ),
        test_type="metamorphic",
        metamorphic_relation="clamp(x+1,0,10) >= clamp(x,0,10) for all x in [0,9]",
    )


def test_generate_returns_list():
    import asyncio
    mock_client = MagicMock()
    mock_client.create_structured = AsyncMock(return_value=_mock_test())
    mock_client.close = AsyncMock()
    with patch("src.codetest.js_generator.InstructorClient", return_value=mock_client):
        result = asyncio.run(JSTestGenerator().generate([_spec()]))
    assert len(result) == 1
    assert result[0].func_id == "abc1234567"


def test_generate_empty_specs_returns_empty():
    import asyncio
    result = asyncio.run(JSTestGenerator().generate([]))
    assert result == []


def test_generated_js_test_extra_forbid():
    with pytest.raises(Exception):
        GeneratedJSTest(
            test_id="x", func_id="y", test_code="code",
            test_type="happy_path", metamorphic_relation=None,
            bad_field="oops",
        )


def test_high_complexity_prompt_requests_metamorphic():
    from src.codetest.js_generator import _build_prompt
    spec = _spec(complexity=3)
    prompt = _build_prompt(spec)
    assert "metamorphic" in prompt.lower()


def test_low_complexity_prompt_does_not_force_metamorphic():
    from src.codetest.js_generator import _build_prompt
    spec = _spec(complexity=1)
    prompt = _build_prompt(spec)
    assert "happy_path" in prompt or "edge_case" in prompt
