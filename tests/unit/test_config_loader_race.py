"""Unit tests for the race-testing config plumbing (mirrors the semantic
cache section's get_use_semantic_cache/get_semantic_cache_config style).
"""
from __future__ import annotations

from src import config_loader


def test_get_use_race_testing_defaults_false(monkeypatch):
    monkeypatch.setattr(config_loader, "get_config", lambda: {})
    assert config_loader.get_use_race_testing() is False


def test_get_use_race_testing_reads_true(monkeypatch):
    monkeypatch.setattr(
        config_loader, "get_config", lambda: {"race": {"enable_race_testing": True}}
    )
    assert config_loader.get_use_race_testing() is True


def test_get_race_config_defaults_empty_dict(monkeypatch):
    monkeypatch.setattr(config_loader, "get_config", lambda: {})
    assert config_loader.get_race_config() == {}


def test_get_race_config_returns_section(monkeypatch):
    section = {"enable_race_testing": True, "allow_destructive_scenarios": False,
               "agents_per_scenario": 3, "overlap_ms": 150, "max_scenarios": 4}
    monkeypatch.setattr(config_loader, "get_config", lambda: {"race": section})
    assert config_loader.get_race_config() == section


def test_agent_yaml_has_race_section_disabled_by_default():
    """The real shipped config/agent.yaml must default race testing off."""
    config_loader._load_yaml.cache_clear()
    try:
        assert config_loader.get_use_race_testing() is False
        race_cfg = config_loader.get_race_config()
        assert race_cfg.get("allow_destructive_scenarios") is False
    finally:
        config_loader._load_yaml.cache_clear()
