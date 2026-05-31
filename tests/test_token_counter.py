# tests/test_token_counter.py
"""
Unit tests for src/perception/token_counter.py (TD-13 fix).

Tests:
  - English text: exact token count
  - Thai/CJK text: proves new estimator counts MORE than the old len//4 hack
  - Caching: encoder is instantiated at most once across repeated calls
"""
from __future__ import annotations

import pytest

from src.perception.token_counter import _get_encoder, estimate_tokens


# ── test_english_text_estimate ────────────────────────────────────────────────


def test_english_text_estimate() -> None:
    """'hello world' is exactly 2 BPE tokens with cl100k_base."""
    result = estimate_tokens("hello world")
    assert result == 2, f"Expected 2 tokens for 'hello world', got {result}"


# ── test_thai_text_estimate ───────────────────────────────────────────────────


def test_thai_text_estimate() -> None:
    """
    Thai text 'สวัสดี' (6 Unicode chars) must yield MORE tokens than the old
    len(text) // 4 approximation would.

    The old formula: 6 // 4 = 1  (wildly wrong — each Thai grapheme cluster
    can be 3-4 bytes and multiple BPE tokens).

    The tiktoken cl100k_base tokenizer encodes each Thai byte-sequence
    individually, so the real count is always > 1.

    This test proves the CJK/Thai fix (TD-13) is working.
    """
    text = "สวัสดี"
    old_estimate = len(text) // 4  # = 1 for "สวัสดี" (6 chars)
    new_estimate = estimate_tokens(text)
    assert new_estimate > old_estimate, (
        f"New estimate ({new_estimate}) must exceed the broken len//4 estimate "
        f"({old_estimate}) for Thai text '{text}'"
    )
    # Also sanity-check: must be at least 1 token
    assert new_estimate >= 1


# ── test_caching ─────────────────────────────────────────────────────────────


def test_caching() -> None:
    """
    _get_encoder() is lru_cached — the encoder object must be constructed only
    once per encoding name even across multiple estimate_tokens() calls.
    """
    # Reset cache so we start from a known state
    _get_encoder.cache_clear()

    # First call — must be a cache miss (currsize 0 → 1)
    _ = estimate_tokens("first call", encoding="cl100k_base")
    info_after_first = _get_encoder.cache_info()
    assert info_after_first.misses == 1, (
        f"Expected 1 miss after first call; got {info_after_first}"
    )
    assert info_after_first.currsize == 1

    # Second call with same encoding — must be a cache hit (hits 0 → 1)
    _ = estimate_tokens("second call", encoding="cl100k_base")
    info_after_second = _get_encoder.cache_info()
    assert info_after_second.hits == 1, (
        f"Expected 1 cache hit after second call; got {info_after_second}"
    )
    # misses should NOT have increased (still 1)
    assert info_after_second.misses == 1, (
        f"Miss count should remain 1; got {info_after_second}"
    )
