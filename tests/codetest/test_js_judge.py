"""Unit tests for JSCodeJudge (deterministic checks — no LLM needed for checks 1-3)."""
from __future__ import annotations
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.codetest.js_generator import GeneratedJSTest
from src.codetest.js_judge import JSCodeJudge


def _test(code: str, ttype: str = "happy_path", relation: str | None = None) -> GeneratedJSTest:
    return GeneratedJSTest(
        test_id="t1", func_id="f1",
        test_code=code, test_type=ttype,  # type: ignore[arg-type]
        metamorphic_relation=relation,
    )


GOOD_CODE = (
    "function clamp(v,mn,mx){return v<mn?mn:v>mx?mx:v}\n"
    "import { it, expect } from 'vitest'\n"
    "it('works', () => { expect(clamp(5,0,10)).toBe(5) })"
)


def test_acceptable_on_good_code():
    result = asyncio.run(JSCodeJudge().judge(_test(GOOD_CODE)))
    assert result.grade == "acceptable"


def test_reject_missing_expect():
    code = "function f(){return 1}\nimport {it} from 'vitest'\nit('x',()=>{const x=f()})"
    result = asyncio.run(JSCodeJudge().judge(_test(code)))
    assert result.grade == "reject"
    assert "expect" in result.feedback


def test_reject_settimeout():
    code = GOOD_CODE + "\nsetTimeout(() => {}, 1000)"
    result = asyncio.run(JSCodeJudge().judge(_test(code)))
    assert result.grade == "reject"
    assert "setTimeout" in result.feedback


def test_reject_no_it_or_test_block():
    code = "function f(){return 1}\nimport {expect} from 'vitest'\nexpect(f()).toBe(1)"
    result = asyncio.run(JSCodeJudge().judge(_test(code)))
    assert result.grade == "reject"
    assert "it(" in result.feedback or "test(" in result.feedback


def test_metamorphic_vague_needs_revision():
    result = asyncio.run(JSCodeJudge().judge(_test(GOOD_CODE, "metamorphic", "always returns a number")))
    assert result.grade == "needs_revision"


def test_metamorphic_valid_passes_static_check():
    with patch.object(JSCodeJudge, "_check_metamorphic", new=AsyncMock(
        return_value=MagicMock(valid=True, reason="ok")
    )):
        result = asyncio.run(JSCodeJudge().judge(
            _test(GOOD_CODE, "metamorphic", "clamp(x+1,0,10) >= clamp(x,0,10) for all x in [0,9]")
        ))
    assert result.grade == "acceptable"
