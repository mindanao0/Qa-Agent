"""TDD for HTTP-response normalization used by Sprint 10 race scenarios.

When race agents hit a real REST backend (jsonplaceholder), the semantic-hash
detector must compare the *logical* response, not volatile server-assigned
fields. `_normalize_http_response` strips the auto-generated `id` so that N
agents performing the SAME request produce an IDENTICAL signature (=> no false
conflict), while genuinely different status/body still diverge.
"""
from __future__ import annotations

from src.race.swarm import _normalize_http_response


def test_identical_logical_response_normalizes_equal_despite_id():
    a = _normalize_http_response(201, '{"id": 201, "title": "x", "userId": 1, "completed": false}')
    b = _normalize_http_response(201, '{"id": 999, "title": "x", "userId": 1, "completed": false}')
    assert a == b


def test_different_status_diverges():
    ok = _normalize_http_response(200, '{"title": "x"}')
    created = _normalize_http_response(201, '{"title": "x"}')
    assert ok != created


def test_different_body_diverges():
    a = _normalize_http_response(200, '{"title": "milk"}')
    b = _normalize_http_response(200, '{"title": "bread"}')
    assert a != b


def test_non_json_body_passthrough():
    assert _normalize_http_response(200, "not-json") == "200|not-json"


def test_key_order_does_not_matter():
    a = _normalize_http_response(200, '{"a": 1, "b": 2}')
    b = _normalize_http_response(200, '{"b": 2, "a": 1}')
    assert a == b
