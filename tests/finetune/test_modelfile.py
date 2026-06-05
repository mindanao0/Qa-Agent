"""Tests for Modelfile generation — no GPU or Ollama required."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.create_modelfile import write_modelfile, MODELFILE_TEMPLATE, MODEL_TAG


class TestWriteModelfile:
    def test_creates_modelfile(self, tmp_path: Path):
        gguf = tmp_path / "qwen2.5-coder-finetuned.Q4_K_M.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert mf.exists()

    def test_modelfile_named_correctly(self, tmp_path: Path):
        gguf = tmp_path / "model.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert mf.name == "Modelfile"

    def test_from_line_uses_relative_path(self, tmp_path: Path):
        gguf = tmp_path / "qwen2.5-coder-finetuned.Q4_K_M.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        content = mf.read_text()
        assert f"FROM ./{gguf.name}" in content

    def test_temperature_parameter(self, tmp_path: Path):
        gguf = tmp_path / "x.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert "PARAMETER temperature 0.1" in mf.read_text()

    def test_num_ctx_parameter(self, tmp_path: Path):
        gguf = tmp_path / "x.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        assert "PARAMETER num_ctx 2048" in mf.read_text()

    def test_system_prompt_present(self, tmp_path: Path):
        gguf = tmp_path / "x.gguf"
        gguf.touch()
        mf = write_modelfile(gguf_path=gguf, output_dir=tmp_path)
        content = mf.read_text()
        assert "SYSTEM" in content
        assert "QA" in content or "qa" in content.lower()

    def test_template_has_from_placeholder(self):
        assert "{gguf_name}" in MODELFILE_TEMPLATE

    def test_model_tag_correct(self):
        assert MODEL_TAG == "qa-agent-finetuned"
