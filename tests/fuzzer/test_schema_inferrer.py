"""Sprint 13 — SchemaInferrer tests (network-free; synthetic traces)."""
import asyncio

import pytest
from pydantic import ValidationError

from src.fuzzer.schema_inferrer import (
    InferredSchema,
    SchemaInferrer,
    TraceRecord,
    normalize_endpoint,
)
from src.fuzzer.schema_inferrer import _redact  # noqa: PLC2701 (test of internal helper)


def _trace(method, endpoint, req=None, status=200, resp=None, auth=False):
    return TraceRecord(
        trace_id="t" + endpoint[-4:], ui_action="probe", endpoint_hint=endpoint,
        method=method, request_body=req, response_status=status, response_body=resp,
        auth_present=auth, captured_at=1.0,
    )


def test_normalize_endpoint_slug():
    assert normalize_endpoint("https://x/api/articles/water-nn") == "/api/articles/{slug}"
    assert normalize_endpoint("/api/articles/some-slug?comments=1") == "/api/articles/{slug}"


def test_normalize_endpoint_profiles():
    assert normalize_endpoint("/api/profiles/bob") == "/api/profiles/{username}"


def test_normalize_endpoint_collection_and_query():
    assert normalize_endpoint("/api/articles?limit=5") == "/api/articles"
    assert normalize_endpoint("/api/tags") == "/api/tags"


def test_redact_sensitive_keys():
    out = _redact({"password": "p", "user": {"token": "t", "name": "n"}, "xs": [{"Authorization": "z"}]})
    assert out["password"] == "__REDACTED__"
    assert out["user"]["token"] == "__REDACTED__"
    assert out["user"]["name"] == "n"
    assert out["xs"][0]["Authorization"] == "__REDACTED__"


def test_tracerecord_extra_forbidden():
    with pytest.raises(ValidationError):
        TraceRecord(trace_id="a", ui_action="b", endpoint_hint="/api/x", method="GET",
                    request_body=None, response_status=200, response_body=None,
                    auth_present=False, captured_at=1.0, bad="no")


def test_inferredschema_defaults_is_candidate():
    s = InferredSchema(endpoint="/api/x", method="GET", request_schema={}, response_schema={},
                       constraints=[], coverage_score=0)
    assert s.is_candidate is True


def test_infer_groups_and_coverage(monkeypatch):
    async def _no_llm(self, *a, **k):
        return []
    monkeypatch.setattr(SchemaInferrer, "_infer_constraints", _no_llm)

    body200 = {"articles": [{"slug": "a"}], "articlesCount": 1}
    traces = [
        _trace("GET", "/api/articles", req={"limit": "1"}, resp=body200),
        _trace("GET", "/api/articles", req={"limit": "5"}, resp=body200),
        _trace("GET", "/api/articles", req={"tag": "Tech"}, resp=body200),
        _trace("GET", "/api/tags", resp={"tags": ["a", "b"]}),
        _trace("POST", "/api/users", req={"user": {"username": "u"}}, resp={"user": {"username": "u"}}),
    ]
    schemas = asyncio.run(SchemaInferrer().infer(traces, asyncio.Semaphore(1)))
    by = {(s.endpoint, s.method): s for s in schemas}

    arts = by[("/api/articles", "GET")]
    assert arts.coverage_score == 3
    assert arts.is_candidate is False          # >= 3 distinct shapes → stabilized
    assert arts.response_schema.get("properties")  # genson built a real response schema
    assert arts.request_schema.get("properties")   # query params became a request schema

    tags = by[("/api/tags", "GET")]
    assert tags.is_candidate is True           # only one shape


def test_infer_no_200_means_empty_response_schema(monkeypatch):
    async def _no_llm(self, *a, **k):
        return []
    monkeypatch.setattr(SchemaInferrer, "_infer_constraints", _no_llm)
    traces = [_trace("GET", "/api/profiles/{username}", status=401, resp={"errors": {}})]
    schemas = asyncio.run(SchemaInferrer().infer(traces, asyncio.Semaphore(1)))
    assert schemas[0].response_schema == {}    # 401 only → no inferred response schema
