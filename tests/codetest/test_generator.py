import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.codetest.ast_parser import FunctionSpec, ArgSpec
from src.codetest.generator import PytestGenerator, GeneratedTest


def _make_spec(func_name: str = "add", return_type: str = "int") -> FunctionSpec:
    return FunctionSpec(
        func_id="abc1234567",
        module_path="src/contractskill/sfg.py",
        func_name=func_name,
        args=[
            ArgSpec(name="x", annotation="int", default=None),
            ArgSpec(name="y", annotation="int", default="0"),
        ],
        return_type=return_type,
        docstring="Add two numbers.",
        decorators=[],
        complexity=1,
    )


@pytest.mark.asyncio
async def test_generate_returns_list_of_generated_tests():
    specs = [_make_spec()]
    mock_test = GeneratedTest(
        test_id="t001",
        func_id="abc1234567",
        test_code="def test_add():\n    assert add(1, 2) == 3",
        test_type="happy_path",
        metamorphic_relation=None,
    )

    with patch("src.codetest.generator.InstructorClient") as MockClient:
        instance = AsyncMock()
        instance.create_structured = AsyncMock(return_value=mock_test)
        instance.close = AsyncMock()
        MockClient.return_value = instance

        gen = PytestGenerator()
        results = await gen.generate(specs)

    assert len(results) == 1
    assert isinstance(results[0], GeneratedTest)
    assert results[0].test_type == "happy_path"


@pytest.mark.asyncio
async def test_generate_empty_specs():
    gen = PytestGenerator()
    results = await gen.generate([])
    assert results == []


def test_generated_test_schema_forbids_extra():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GeneratedTest(
            test_id="t1",
            func_id="abc",
            test_code="def test_x(): pass",
            test_type="happy_path",
            metamorphic_relation=None,
            extra_field="forbidden",
        )


def test_generated_test_metamorphic_relation_optional():
    t = GeneratedTest(
        test_id="t2",
        func_id="abc",
        test_code="def test_x(): assert True",
        test_type="metamorphic",
        metamorphic_relation="f(x+1) >= f(x) for positive x",
    )
    assert t.metamorphic_relation is not None


@pytest.mark.asyncio
async def test_generate_skips_on_structured_generation_error():
    from src.llm.instructor_client import StructuredGenerationError
    specs = [_make_spec("add"), _make_spec("sub")]
    good_test = GeneratedTest(
        test_id="t_good",
        func_id="abc1234567",
        test_code="def test_sub():\n    assert 2 - 1 == 1",
        test_type="happy_path",
        metamorphic_relation=None,
    )
    with patch("src.codetest.generator.InstructorClient") as MockClient:
        instance = AsyncMock()
        instance.create_structured = AsyncMock(
            side_effect=[StructuredGenerationError("h", ValueError("bad")), good_test]
        )
        instance.close = AsyncMock()
        MockClient.return_value = instance
        gen = PytestGenerator()
        results = await gen.generate(specs)
    assert len(results) == 1
    assert results[0].test_id == "t_good"


@pytest.mark.asyncio
async def test_regenerate_returns_none_on_failure():
    from src.llm.instructor_client import StructuredGenerationError
    spec = _make_spec()
    original = GeneratedTest(
        test_id="t_orig",
        func_id="abc1234567",
        test_code="def test_x():\n    pass",
        test_type="happy_path",
        metamorphic_relation=None,
    )
    with patch("src.codetest.generator.InstructorClient") as MockClient:
        instance = AsyncMock()
        instance.create_structured = AsyncMock(
            side_effect=StructuredGenerationError("h", ValueError("bad"))
        )
        instance.close = AsyncMock()
        MockClient.return_value = instance
        gen = PytestGenerator()
        result = await gen.regenerate_with_feedback(spec, original, "missing assert")
    assert result is None


@pytest.mark.asyncio
async def test_generate_overrides_func_id_from_spec():
    specs = [_make_spec()]
    # LLM returns wrong func_id
    mock_test = GeneratedTest(
        test_id="t_override",
        func_id="WRONG_LLM_ID",  # LLM hallucinated a bad func_id
        test_code="def test_add():\n    assert add(1, 2) == 3",
        test_type="happy_path",
        metamorphic_relation=None,
    )
    with patch("src.codetest.generator.InstructorClient") as MockClient:
        instance = AsyncMock()
        instance.create_structured = AsyncMock(return_value=mock_test)
        instance.close = AsyncMock()
        MockClient.return_value = instance
        gen = PytestGenerator()
        results = await gen.generate(specs)
    assert len(results) == 1
    assert results[0].func_id == "abc1234567"  # spec's func_id wins
