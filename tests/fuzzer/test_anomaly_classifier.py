"""Sprint 13 — AnomalyClassifier grounded-rule tests.

Encodes the live-probed truth of realworld.habsida.net: 5xx on malformed input
is the real anomaly; SQLi/XSS that returns a safe 200 is NOT an anomaly (flagging
it would be a false positive); 4xx are correct rejections.
"""
import pytest
from pydantic import ValidationError

from src.fuzzer.anomaly_classifier import (
    AnomalyClassifier,
    AnomalyType,
    FuzzResult,
    is_injection_vector,
)

CL = AnomalyClassifier()
EP = "/api/articles"


def _c(vector, status, **kw):
    return CL.classify(endpoint=EP, vector=vector, status_code=status,
                       duration_ms=kw.pop("duration_ms", 50.0), **kw)


def test_500_non_injection_is_unexpected_status():
    r = _c("-1", 500)  # limit=-1 → real 500
    assert r.anomaly is True
    assert r.anomaly_type == AnomalyType.UNEXPECTED_STATUS
    assert r.false_positive is False


def test_500_on_injection_is_injection_signal():
    r = _c("../../../etc/passwd", 500)  # traversal slug → real 500
    assert r.anomaly is True
    assert r.anomaly_type == AnomalyType.INJECTION_SIGNAL
    assert r.false_positive is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_4xx_is_expected_not_anomaly(status):
    r = _c("abc", status)
    assert r.anomaly is False
    assert r.anomaly_type == AnomalyType.EXPECTED_4XX
    assert r.false_positive is False


def test_safe_200_on_sqli_is_not_anomaly():
    # The key honesty rule: SQLi handled safely (empty result) is NOT an anomaly.
    r = _c("' OR '1'='1", 200, body={"articles": [], "articlesCount": 0})
    assert r.anomaly is False
    assert r.anomaly_type is None


def test_safe_200_on_xss_is_not_anomaly():
    r = _c("<script>alert(1)</script>", 200, body={"articles": [], "articlesCount": 0})
    assert r.anomaly is False


def test_200_injection_with_sql_error_leak_is_injection_signal():
    r = _c("' OR '1'='1", 200, body={"errors": {"body": ["SQLITE_ERROR: near \"OR\""]}})
    assert r.anomaly is True
    assert r.anomaly_type == AnomalyType.INJECTION_SIGNAL


def test_200_missing_required_field_is_schema_drift():
    schema = {"type": "object", "properties": {"articles": {}}, "required": ["articles"]}
    r = _c("x", 200, body={"unexpected": 1}, response_schema=schema)
    assert r.anomaly is True
    assert r.anomaly_type == AnomalyType.SCHEMA_DRIFT


def test_200_with_required_field_present_is_clean():
    schema = {"type": "object", "properties": {"articles": {}}, "required": ["articles"]}
    r = _c("5", 200, body={"articles": [], "articlesCount": 0}, response_schema=schema)
    assert r.anomaly is False
    assert r.anomaly_type is None


def test_timeout_by_duration():
    r = _c("0", 200, duration_ms=11_000.0)
    assert r.anomaly is True
    assert r.anomaly_type == AnomalyType.TIMEOUT


def test_timeout_by_flag():
    r = _c("0", 0, timed_out=True)
    assert r.anomaly is True
    assert r.anomaly_type == AnomalyType.TIMEOUT


def test_is_injection_vector():
    assert is_injection_vector("../../../etc/passwd")
    assert is_injection_vector("' OR '1'='1")
    assert is_injection_vector("<script>alert(1)</script>")
    assert is_injection_vector("UNION SELECT * FROM users")
    assert not is_injection_vector("5")
    assert not is_injection_vector("notanemail")


def test_fuzzresult_extra_forbidden():
    with pytest.raises(ValidationError):
        FuzzResult(endpoint=EP, vector="x", status_code=200, anomaly=False,
                   anomaly_type=None, false_positive=False, duration_ms=1.0, bad="no")


def test_vector_truncated_to_100():
    r = _c("a" * 500, 500)
    assert len(r.vector) == 100
