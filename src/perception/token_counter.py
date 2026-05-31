# src/perception/token_counter.py
"""
Token counting utility for the Universal DOM Compression Pipeline.

Uses tiktoken cl100k_base (OpenAI's BPE tokenizer) as a safe conservative
estimator for Qwen2.5-coder token counts.

Safety margin note:
    cl100k_base over-estimates Qwen2.5-coder tokens by approximately 10-15%.
    This is intentional and SAFE — it makes the context budget enforcement
    conservative (we stop slightly before the real limit), preventing stealth
    context overflow on non-English content (CJK, Thai, Arabic, etc.) where
    the old `len(text) // 4` approximation under-counted by 4× or more.

    DO NOT switch to the Qwen tokenizer — it is not in tiktoken's stdlib
    and adding it is out of scope. The 10-15% over-estimate is the desired
    production behaviour.

Usage::

    from src.perception.token_counter import estimate_tokens

    n = estimate_tokens("สวัสดี")   # Thai text — correctly > 1 token
    n = estimate_tokens("hello world")  # → 2 tokens
"""
from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def _get_encoder(encoding: str = "cl100k_base") -> tiktoken.Encoding:
    """Return a cached tiktoken Encoding object.

    Cached by encoding name — the encoder is expensive to initialise
    (loads ~1 MB vocab file) so we keep at most *maxsize* live instances.
    """
    return tiktoken.get_encoding(encoding)


def estimate_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Return the estimated token count of *text* using the specified BPE encoding.

    Args:
        text:     The string to count tokens for.
        encoding: tiktoken encoding name (default: ``"cl100k_base"``).
                  cl100k_base is used by GPT-4 / text-embedding-ada-002 and is
                  a good conservative proxy for Qwen2.5-coder (over-estimates by
                  ~10-15%, which is safe for budget enforcement).

    Returns:
        Token count as an integer (≥ 1 for any non-empty string).
    """
    if not text:
        return 0
    return len(_get_encoder(encoding).encode(text))
