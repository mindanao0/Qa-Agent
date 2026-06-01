import pytest
from pydantic import ValidationError

from src.fuzzer.api_fuzzer import FuzzResult, FuzzTarget


def test_fuzz_target_valid():
    t = FuzzTarget(
        endpoint="https://jsonplaceholder.typicode.com/todos",
        method="GET",
        schema={"type": "array"},
        fuzz_vectors=["null", "-1"],
    )
    assert t.method == "GET"
    assert len(t.fuzz_vectors) == 2


def test_fuzz_target_extra_forbidden():
    with pytest.raises(ValidationError):
        FuzzTarget(
            endpoint="https://example.com/api",
            method="GET",
            schema={},
            fuzz_vectors=[],
            extra="bad",
        )


def test_fuzz_result_valid():
    r = FuzzResult(
        endpoint="https://jsonplaceholder.typicode.com/todos",
        vector="-1",
        status_code=404,
        anomaly=True,
        anomaly_type="unexpected_status",
        duration_ms=52.3,
    )
    assert r.anomaly is True
    assert r.anomaly_type == "unexpected_status"


def test_fuzz_result_none_anomaly_type():
    r = FuzzResult(
        endpoint="https://example.com",
        vector="",
        status_code=200,
        anomaly=False,
        anomaly_type=None,
        duration_ms=10.0,
    )
    assert r.anomaly_type is None


def test_fuzz_result_extra_forbidden():
    with pytest.raises(ValidationError):
        FuzzResult(
            endpoint="https://example.com",
            vector="x",
            status_code=200,
            anomaly=False,
            anomaly_type=None,
            duration_ms=1.0,
            bad_field="no",
        )
