"""Unit tests for the fuzzer config gate (src/config_loader.get_use_api_fuzz /
get_api_fuzz_config) — mirrors the semantic-cache config gate tests' style.
"""
from __future__ import annotations

from src import config_loader
from src.config_loader import get_api_fuzz_config, get_use_api_fuzz


def _reload_config() -> None:
    """Force config_loader._load_yaml to re-read config/agent.yaml."""
    config_loader._load_yaml.cache_clear()


def test_get_use_api_fuzz_defaults_false() -> None:
    """The real config/agent.yaml ships with fuzzer.enable_api_fuzz: false."""
    _reload_config()
    assert get_use_api_fuzz() is False


def test_get_use_api_fuzz_reads_config_true(monkeypatch) -> None:
    monkeypatch.setattr(
        config_loader,
        "_load_yaml",
        lambda: {"fuzzer": {"enable_api_fuzz": True}},
    )
    assert get_use_api_fuzz() is True


def test_get_use_api_fuzz_missing_section_defaults_false(monkeypatch) -> None:
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    assert get_use_api_fuzz() is False


def test_get_api_fuzz_config_returns_section(monkeypatch) -> None:
    section = {"enable_api_fuzz": True, "max_endpoints": 7, "max_vectors_per_field": 3}
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {"fuzzer": section})
    assert get_api_fuzz_config() == section


def test_get_api_fuzz_config_defaults_empty_dict(monkeypatch) -> None:
    monkeypatch.setattr(config_loader, "_load_yaml", lambda: {})
    assert get_api_fuzz_config() == {}
