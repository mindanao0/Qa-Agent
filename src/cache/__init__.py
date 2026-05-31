"""Semantic Cache package — dual-threshold LanceDB cache for QA agent."""
from src.cache.semantic_cache import (
    CacheEntry,
    CacheLookupResult,
    CacheStats,
    SemanticCache,
)

__all__ = [
    "SemanticCache",
    "CacheLookupResult",
    "CacheEntry",
    "CacheStats",
]
