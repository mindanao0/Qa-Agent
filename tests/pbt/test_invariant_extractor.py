"""Sprint 14 — InvariantExtractor unit tests (network-free; LLM boundary faked)."""
import asyncio

import pytest

from src.codetest.ast_parser import ArgSpec, FunctionSpec
from src.fuzzer.schema_inferrer import InferredSchema
from src.pbt.hypothesis_runner import FUNCTION_TARGETS, HypothesisRunner
from src.pbt.invariant_extractor import (
    Invariant,
    InvariantExtractor,
    endpoint_robustness_invariant,
    fallback_invariant,
    make_invariant_id,
)


def test_make_invariant_id_is_deterministic_10_hex():
    a = make_invariant_id("function", "_words_relate", "commutative", "symmetric")
    b = make_invariant_id("function", "_words_relate", "commutative", "symmetric")
    assert a == b and len(a) == 10
    assert all(c in "0123456789abcdef" for c in a)
    assert a != make_invariant_id("function", "_words_relate", "idempotent", "symmetric")


def test_invariant_model_forbids_extra_fields():
    inv = Invariant(
        invariant_id="abc1234567", source="function", source_id="x",
        description="d", property_type="bounded", hypothesis_strategy="st.integers()",
    )
    assert inv.property_type == "bounded"
    with pytest.raises(Exception):
        Invariant(
            invariant_id="x", source="function", source_id="x", description="d",
            property_type="bounded", hypothesis_strategy="st.integers()", oops="no",
        )


def test_endpoint_robustness_invariant_shape():
    inv = endpoint_robustness_invariant("/api/articles", "GET", "limit")
    assert inv.source == "endpoint"
    assert inv.source_id == "/api/articles?limit"
    assert inv.property_type == "invariant_output"
    assert "integers" in inv.hypothesis_strategy


def test_fallback_invariant_is_buildable_by_runner():
    inv = fallback_invariant()
    assert inv.source == "function"
    code = HypothesisRunner()._build_test(inv)  # must not raise
    assert "_meaningful_words" in code


class _BoomClient:
    """Stand-in for InstructorClient whose every call fails (Ollama down)."""

    async def create_structured(self, **_):
        raise RuntimeError("ollama unreachable")

    async def close(self):
        pass


async def test_extract_from_functions_falls_back_to_canonical_on_llm_failure():
    spec = FunctionSpec(
        func_id="x", module_path="src/explorer/planner.py", func_name="_words_relate",
        args=[ArgSpec(name="a", annotation="str", default=None),
              ArgSpec(name="b", annotation="str", default=None)],
        return_type="bool", docstring="related?", decorators=[], complexity=1,
    )
    ext = InvariantExtractor(client=_BoomClient())
    invs = await ext.extract_from_functions([spec], asyncio.Semaphore(1))
    # canonical fallback still yields the verified property types for the target
    types = {i.property_type for i in invs if i.source_id == "_words_relate"}
    assert "commutative" in types
    # every emitted invariant is buildable
    runner = HypothesisRunner()
    for i in invs:
        runner._build_test(i)


async def test_extract_from_functions_ignores_unknown_targets():
    spec = FunctionSpec(
        func_id="y", module_path="src/whatever.py", func_name="not_a_target",
        args=[], return_type="None", docstring=None, decorators=[], complexity=1,
    )
    ext = InvariantExtractor(client=_BoomClient())
    invs = await ext.extract_from_functions([spec], asyncio.Semaphore(1))
    assert invs == []


async def test_extract_from_schemas_builds_robustness_invariants_from_int_params():
    schema = InferredSchema(
        endpoint="/api/articles", method="GET",
        request_schema={"type": "object", "properties": {
            "limit": {"type": "integer"}, "offset": {"type": "integer"},
            "tag": {"type": "string"},
        }},
        response_schema={"type": "object", "properties": {"articles": {"type": "array"}}},
        constraints=[], coverage_score=3,
    )
    ext = InvariantExtractor(client=_BoomClient())
    invs = await ext.extract_from_schemas([schema], asyncio.Semaphore(1))
    params = {i.source_id for i in invs}
    assert params == {"/api/articles?limit", "/api/articles?offset"}
    assert all(i.source == "endpoint" for i in invs)
