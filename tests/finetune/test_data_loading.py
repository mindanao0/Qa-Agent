"""Tests for JSONL loading and ChatML conversion."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.finetune import load_jsonl, to_chatml, TRAIN_JSONL, VAL_JSONL


class TestLoadJsonl:
    def test_train_jsonl_exists(self):
        assert TRAIN_JSONL.exists(), f"{TRAIN_JSONL} not found"

    def test_train_jsonl_count(self):
        records = load_jsonl(TRAIN_JSONL)
        assert len(records) >= 449

    def test_val_jsonl_count(self):
        records = load_jsonl(VAL_JSONL)
        assert len(records) == 51

    def test_each_record_has_prompt_and_completion(self):
        records = load_jsonl(TRAIN_JSONL)
        for r in records[:20]:
            assert "prompt" in r, f"Missing 'prompt' in {r}"
            assert "completion" in r, f"Missing 'completion' in {r}"
            assert r["prompt"], "prompt is empty"
            assert r["completion"], "completion is empty"

    def test_load_from_tmp_file(self, tmp_path: Path):
        p = tmp_path / "test.jsonl"
        p.write_text(
            '{"prompt": "hello", "completion": "world"}\n'
            '{"prompt": "foo", "completion": "bar"}\n',
            encoding="utf-8",
        )
        records = load_jsonl(p)
        assert len(records) == 2
        assert records[0]["prompt"] == "hello"

    def test_skips_blank_lines(self, tmp_path: Path):
        p = tmp_path / "blank.jsonl"
        p.write_text(
            '{"prompt": "a", "completion": "b"}\n\n\n',
            encoding="utf-8",
        )
        records = load_jsonl(p)
        assert len(records) == 1

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl(Path("/nonexistent/path.jsonl"))


class TestToChatml:
    def test_produces_two_messages(self):
        msgs = to_chatml({"prompt": "Write a test", "completion": "def test_it(): pass"})
        assert len(msgs) == 2

    def test_user_role(self):
        msgs = to_chatml({"prompt": "Q", "completion": "A"})
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Q"

    def test_assistant_role(self):
        msgs = to_chatml({"prompt": "Q", "completion": "A"})
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "A"

    def test_real_example_round_trip(self):
        records = load_jsonl(TRAIN_JSONL)
        r = records[0]
        msgs = to_chatml(r)
        assert msgs[0]["content"] == r["prompt"]
        assert msgs[1]["content"] == r["completion"]
