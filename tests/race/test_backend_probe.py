"""Tests for BackendProbe pure helpers (Sprint 12)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.race.backend_probe import BackendProbeResult, _has_real_data


def test_backend_probe_result_extra_forbidden():
    with pytest.raises(ValidationError):
        BackendProbeResult(
            url="https://x", has_real_backend=True, api_endpoints=[],
            probe_note=None, bad="nope",
        )


def test_real_data_non_empty_tags():
    assert _has_real_data({"tags": ["python", "react"]}) is True


def test_empty_tags_is_not_real():
    assert _has_real_data({"tags": []}) is False


def test_real_data_articles_count():
    assert _has_real_data({"articles": [{"slug": "a"}], "articlesCount": 1}) is True


def test_zero_articles_is_not_real():
    assert _has_real_data({"articles": [], "articlesCount": 0}) is False


def test_jsonplaceholder_shape_is_not_real():
    # jsonplaceholder /api/tags would 404; a generic object is not conduit data.
    assert _has_real_data({"id": 1, "title": "x"}) is False


def test_non_dict_is_not_real():
    assert _has_real_data([]) is False
    assert _has_real_data("nope") is False
    assert _has_real_data({}) is False
