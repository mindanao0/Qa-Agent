# src/config_loader.py
"""
Lightweight loader for config/agent.yaml with env-var overrides.

Usage:
    from src.config_loader import get_structured_output_engine
    engine = get_structured_output_engine()  # "instructor" or "legacy_repair"
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "agent.yaml"


# NOTE: _load_yaml is cached with lru_cache(maxsize=1). In tests that mutate agent.yaml,
# call _load_yaml.cache_clear() to force a fresh read.
@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    """Load and cache agent.yaml. Returns {} if the file is missing."""
    try:
        return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def get_config() -> dict:
    """Return the full parsed config/agent.yaml dict (cached)."""
    return _load_yaml()


def get_structured_output_engine() -> str:
    """
    Return the active structured-output engine name.

    Priority:
      1. STRUCTURED_OUTPUT_ENGINE env var (for CI overrides)
      2. config/agent.yaml llm.structured_output_engine
      3. Hard default: "instructor"
    """
    env_override = os.getenv("STRUCTURED_OUTPUT_ENGINE")
    if env_override:
        return env_override
    cfg = _load_yaml()
    return cfg.get("llm", {}).get("structured_output_engine", "instructor")


def get_use_grounder() -> bool:
    """Return True if the grounder perception layer is enabled in config.

    Defaults to True when the 'perception.use_grounder' key is absent from config.
    Set to False in agent.yaml to disable the perception layer and use legacy extraction.
    """
    config = get_config()
    perception = config.get("perception", {})
    return bool(perception.get("use_grounder", True))


def get_context_budget_tokens() -> int:
    """Return the context budget token limit from config."""
    config = get_config()
    perception = config.get("perception", {})
    return int(perception.get("context_budget_tokens", 1000))


def get_bft_enabled() -> bool:
    """Return True if Sprint 3 BFT generator pipeline is enabled.

    Reads ``llm.bft.enabled`` from config/agent.yaml. Defaults to False.
    """
    config = get_config()
    return bool(config.get("llm", {}).get("bft", {}).get("enabled", False))


def get_exploration_config() -> dict:
    """Return the exploration section from config/agent.yaml as a dict."""
    config = get_config()
    return config.get("exploration", {})


def get_use_semantic_cache() -> bool:
    """Return True if the semantic cache (Priority 3) is enabled in config.

    Defaults to False (opt-in) — see src/cache/semantic_cache.py.
    """
    config = get_config()
    return bool(config.get("cache", {}).get("use_semantic_cache", False))


def get_semantic_cache_config() -> dict:
    """Return the cache section from config/agent.yaml as a dict."""
    config = get_config()
    return config.get("cache", {})


def get_use_worker_pool() -> bool:
    """Return True if the BrowserWorkerPool (Sprint 15) parallel executor is
    enabled in config.

    Defaults to False (opt-in) — see src/parallel/worker_pool.py.
    """
    config = get_config()
    return bool(config.get("parallel", {}).get("use_worker_pool", False))


def get_parallel_config() -> dict:
    """Return the parallel section from config/agent.yaml as a dict."""
    config = get_config()
    return config.get("parallel", {})


def get_use_race_testing() -> bool:
    """Return True if opt-in race-condition swarm testing (Sprint 10/12) is
    enabled in config.

    Defaults to False (opt-in) — see src/race/swarm.py and
    src/universal_qa/race_check.py.
    """
    config = get_config()
    return bool(config.get("race", {}).get("enable_race_testing", False))


def get_race_config() -> dict:
    """Return the race section from config/agent.yaml as a dict."""
    config = get_config()
    return config.get("race", {})
