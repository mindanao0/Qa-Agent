from src.race.detector import ConflictDetector
from src.race.swarm import RaceResult


def _make_result(scenario_id: str, conflict_found: bool, duration_ms: float = 100.0) -> RaceResult:
    return RaceResult(
        scenario_id=scenario_id,
        conflict_found=conflict_found,
        interleaving=["2026-06-02T00:00:00+00:00"],
        error_summary="err" if conflict_found else None,
        duration_ms=duration_ms,
    )


def test_analyze_empty():
    detector = ConflictDetector()
    result = detector.analyze([])
    assert result["total_scenarios"] == 0
    assert result["conflicts_found"] == 0
    assert result["conflict_rate"] == 0.0
    assert result["worst_scenario_id"] is None
    assert result["interleaving_patterns"] == []


def test_analyze_no_conflicts():
    results = [_make_result("s1", False), _make_result("s2", False)]
    detector = ConflictDetector()
    out = detector.analyze(results)
    assert out["total_scenarios"] == 2
    assert out["conflicts_found"] == 0
    assert out["conflict_rate"] == 0.0
    assert out["worst_scenario_id"] is None


def test_analyze_with_conflicts():
    results = [
        _make_result("s1", False, 50.0),
        _make_result("s2", True, 200.0),
        _make_result("s3", True, 150.0),
    ]
    detector = ConflictDetector()
    out = detector.analyze(results)
    assert out["total_scenarios"] == 3
    assert out["conflicts_found"] == 2
    assert abs(out["conflict_rate"] - 2 / 3) < 0.001
    assert out["worst_scenario_id"] == "s2"  # highest duration among conflicts


def test_analyze_interleaving_patterns():
    r1 = RaceResult(
        scenario_id="s1",
        conflict_found=False,
        interleaving=["ts1", "ts2"],
        error_summary=None,
        duration_ms=10.0,
    )
    r2 = RaceResult(
        scenario_id="s2",
        conflict_found=True,
        interleaving=["ts3"],
        error_summary="err",
        duration_ms=20.0,
    )
    detector = ConflictDetector()
    out = detector.analyze([r1, r2])
    assert "ts1" in out["interleaving_patterns"]
    assert "ts2" in out["interleaving_patterns"]
    assert "ts3" in out["interleaving_patterns"]


def test_analyze_returns_dict_keys():
    detector = ConflictDetector()
    out = detector.analyze([])
    assert set(out.keys()) == {
        "total_scenarios",
        "conflicts_found",
        "conflict_rate",
        "worst_scenario_id",
        "interleaving_patterns",
    }
