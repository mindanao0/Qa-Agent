"""Sprint 14 — pre-built Hypothesis strategies mapped to ``property_type``.

The LLM only ever receives the *keys* of ``STRATEGY_MAP`` (the six property types);
the :class:`HypothesisRunner` substitutes the real strategy string. This keeps the
7B model from needing to know the Hypothesis API.

``roundtrip`` uses the modern ``exclude_categories`` kwarg (the spec's
``blacklist_categories`` alias was removed in Hypothesis 6.x).
"""
from __future__ import annotations

from typing import Final

PROPERTY_TYPES: Final[tuple[str, ...]] = (
    "idempotent",
    "monotonic",
    "bounded",
    "roundtrip",
    "commutative",
    "invariant_output",
)

STRATEGY_MAP: Final[dict[str, str]] = {
    "idempotent": "st.text(min_size=0, max_size=50)",
    "monotonic": "st.integers(min_value=0, max_value=1000)",
    "bounded": "st.lists(st.integers(), max_size=100)",
    "roundtrip": "st.text(alphabet=st.characters(exclude_categories=('Cs',)))",
    "commutative": "st.tuples(st.integers(), st.integers())",
    "invariant_output": "st.fixed_dictionaries({'key': st.text(min_size=1)})",
}

_DEFAULT: Final[str] = "st.text(min_size=0, max_size=50)"


def strategy_for(property_type: str) -> str:
    """Resolve a ``property_type`` key to its Hypothesis strategy string.

    Unknown keys fall back to a safe text strategy (never raises) so a stray LLM
    classification can't crash a measurement run.
    """
    return STRATEGY_MAP.get(property_type, _DEFAULT)


__all__ = ["PROPERTY_TYPES", "STRATEGY_MAP", "strategy_for"]
