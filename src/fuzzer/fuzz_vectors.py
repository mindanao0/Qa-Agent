"""FuzzVectorLibrary — deterministic base vectors, no LLM needed."""
from __future__ import annotations

BASE_VECTORS: list[str] = [
    "",                           # empty string
    " " * 1000,                   # whitespace flood
    "null",                       # null literal
    "undefined",
    "<script>alert(1)</script>",  # XSS probe
    "' OR '1'='1",                # SQL injection probe
    "0",
    "-1",
    "9" * 20,                     # integer overflow
    "𝕳𝖊𝖑𝖑𝖔",                    # unicode stress
]

__all__ = ["BASE_VECTORS"]
