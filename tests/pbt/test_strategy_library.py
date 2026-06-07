"""Sprint 14 — strategy_library tests (network-free)."""
import hypothesis.strategies as st

from src.pbt.strategy_library import PROPERTY_TYPES, STRATEGY_MAP, strategy_for


def test_strategy_map_keys_are_the_six_property_types():
    assert set(STRATEGY_MAP) == set(PROPERTY_TYPES)
    assert set(PROPERTY_TYPES) == {
        "idempotent",
        "monotonic",
        "bounded",
        "roundtrip",
        "commutative",
        "invariant_output",
    }


def test_strategy_for_returns_the_mapped_string():
    for key in PROPERTY_TYPES:
        assert strategy_for(key) == STRATEGY_MAP[key]


def test_strategy_for_unknown_key_falls_back_without_raising():
    # a stray LLM classification must never crash a measurement run
    fallback = strategy_for("not_a_real_property_type")
    assert fallback.startswith("st.")


def test_every_strategy_string_builds_a_real_hypothesis_strategy():
    namespace = {"st": st}
    for key, code in STRATEGY_MAP.items():
        strat = eval(code, namespace)  # noqa: S307 - evaluating our own constants
        assert hasattr(strat, "example"), f"{key} did not build a strategy"
