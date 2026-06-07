"""Sprint 13 — AdvancedVectorGenerator tests (network-free; constraints=[] path)."""
import asyncio

from src.fuzzer.schema_inferrer import InferredSchema
from src.fuzzer.vector_generator import (
    BASE_VECTORS_BY_TYPE,
    BLOCKED_ACTION_PATTERNS,
    AdvancedVectorGenerator,
    fuzzable_fields,
    infer_field_type,
)


def _schema(props, constraints=None):
    return InferredSchema(
        endpoint="/api/articles", method="GET",
        request_schema={"type": "object", "properties": props},
        response_schema={}, constraints=constraints or [], coverage_score=3,
    )


def test_base_vectors_present():
    assert set(BASE_VECTORS_BY_TYPE) == {"string", "integer", "email", "slug"}


def test_infer_field_type():
    assert infer_field_type("email", None) == "email"
    assert infer_field_type("limit", None) == "integer"
    assert infer_field_type("offset", None) == "integer"
    assert infer_field_type("slug", None) == "slug"
    assert infer_field_type("title", None) == "string"
    assert infer_field_type("anything", "integer") == "integer"


def test_fuzzable_fields_excludes_password():
    fields = dict(fuzzable_fields(_schema({
        "limit": {"type": "integer"},
        "tag": {"type": "string"},
        "password": {"type": "string"},
    })))
    assert fields == {"limit": "integer", "tag": "string"}  # password BLOCKED


def test_generate_fields_base_only_no_network():
    gen = AdvancedVectorGenerator()
    schema = _schema({"limit": {"type": "integer"}, "tag": {"type": "string"}})
    out = asyncio.run(gen.generate_fields(schema, asyncio.Semaphore(1), max_vectors=5))
    assert set(out) == {"limit", "tag"}
    for field, vectors in out.items():
        assert len(vectors) <= 5
        assert len(vectors) == len(set(vectors))                 # unique
        assert all(not BLOCKED_ACTION_PATTERNS.search(v) for v in vectors)  # no BLOCKED
    # integer field draws from the integer base set
    assert out["limit"][0] == "0"


def test_generate_flat_union_deduped():
    gen = AdvancedVectorGenerator()
    schema = _schema({"limit": {"type": "integer"}, "offset": {"type": "integer"}})
    flat = asyncio.run(gen.generate(schema, asyncio.Semaphore(1), max_vectors=6))
    assert len(flat) == len(set(flat))   # deduped union (both integer → overlap collapses)


def test_no_fuzzable_fields_returns_empty():
    gen = AdvancedVectorGenerator()
    schema = _schema({"password": {"type": "string"}})  # only BLOCKED field
    out = asyncio.run(gen.generate_fields(schema, asyncio.Semaphore(1)))
    assert out == {}
