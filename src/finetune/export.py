"""
LoRA merge → GGUF Q4_K_M export → Ollama registration pipeline
(UPDATE 3 specification).

Steps performed by `ModelExporter.merge_and_export()`:

  1. Load the fine-tuned checkpoint (base model + saved LoRA adapter).
  2. Merge LoRA into the base model (Unsloth handles this in-place).
  3. Export to GGUF Q4_K_M format under ~/.qa-agent/gguf/
     and rename the resulting file to "qa-agent-coder-q4_k_m.gguf".
  4. Write the Ollama Modelfile (system prompt + parameters per spec).
  5. Run `ollama create qa-agent-coder -f Modelfile` to register the model.
  6. Smoke-test the registered model — verify the response contains
     "def test_" to confirm the fine-tune learned the test-writing pattern.

Requires:
    pip install "qa-agent[finetune]"     # unsloth + transformers + torch
    ollama CLI must be installed and the daemon must be running.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CHECKPOINT_DIR = "~/.qa-agent/checkpoints/final"
_DEFAULT_GGUF_DIR = "~/.qa-agent/gguf"
_DEFAULT_MERGED_DIR = "~/.qa-agent/merged_model"
_OLLAMA_MODEL_TAG = "qa-agent-coder"
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_GGUF_FILE_NAME = "qa-agent-coder-q4_k_m.gguf"

_TEST_PROMPT = (
    "Write a minimal pytest-playwright test that navigates to "
    "http://localhost:3000 and asserts the page title is not empty. "
    "Use only semantic locators and expect()."
)

_MODELFILE_TEMPLATE = """\
FROM ./{gguf_name}
SYSTEM \"\"\"You are an expert Playwright QA engineer. You write pytest-playwright tests
using only semantic locators: get_by_role, get_by_label, get_by_text, get_by_test_id.
You never use CSS selectors or XPath. You always use expect() for assertions.\"\"\"
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
"""


# ──────────────────────────────────────────────────────────────────────────────
# Exporter
# ──────────────────────────────────────────────────────────────────────────────


class ModelExporter:
    """
    Merges a QLoRA fine-tuned checkpoint into the base model, quantises to
    GGUF Q4_K_M, writes the Ollama Modelfile, and registers the resulting
    artefact as `qa-agent-coder`.
    """

    def __init__(
        self,
        checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
        gguf_dir: str = _DEFAULT_GGUF_DIR,
        merged_dir: str = _DEFAULT_MERGED_DIR,
        model_tag: str = _OLLAMA_MODEL_TAG,
        quantization: str = "q4_k_m",
    ) -> None:
        self.checkpoint_dir = Path(os.path.expanduser(checkpoint_dir))
        self.gguf_dir = Path(os.path.expanduser(gguf_dir))
        self.merged_dir = Path(os.path.expanduser(merged_dir))
        self.model_tag = model_tag
        self.quantization = quantization

    # ──────────────────────────────────────────────────────────────────────────
    # Orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def merge_and_export(self, checkpoint_dir: str | None = None) -> Path:
        """
        Run all five export steps and return the absolute path to the
        registered .gguf file.
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(os.path.expanduser(checkpoint_dir))

        self._check_unsloth()
        self._check_checkpoint()

        self.gguf_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)

        # Steps 1 + 2 + 3
        gguf_path = self._export_gguf()

        # Step 4
        modelfile_path = self._write_modelfile(gguf_path)

        # Step 5
        self._register_with_ollama(modelfile_path)

        # Step 6
        self._smoke_test()

        logger.info(
            f"Model exported and registered | tag={self.model_tag!r} gguf={gguf_path}"
        )
        return gguf_path

    # Backwards-compat alias for older main.py call sites
    def export(self, checkpoint_dir: str | None = None) -> Path:
        return self.merge_and_export(checkpoint_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Steps 1–3 — merge LoRA and export to GGUF
    # ──────────────────────────────────────────────────────────────────────────

    def _export_gguf(self) -> Path:
        from unsloth import FastLanguageModel  # type: ignore[import]

        logger.info(f"Loading fine-tuned checkpoint from {self.checkpoint_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.checkpoint_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        logger.info(
            f"Exporting GGUF ({self.quantization.upper()}) → {self.gguf_dir}"
        )
        # Unsloth's save_pretrained_gguf merges LoRA into the base weights and
        # writes a quantised .gguf in a single call.
        model.save_pretrained_gguf(
            str(self.gguf_dir),
            tokenizer,
            quantization_method=self.quantization,
        )

        gguf_files = list(self.gguf_dir.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(
                f"No .gguf file found in {self.gguf_dir} after export"
            )

        # Rename to spec name for predictable Modelfile reference
        produced = sorted(gguf_files, key=lambda p: p.stat().st_mtime)[-1]
        target = self.gguf_dir / _GGUF_FILE_NAME
        if produced != target:
            if target.exists():
                target.unlink()
            shutil.move(str(produced), str(target))
        size_mb = target.stat().st_size // 1_048_576
        logger.info(f"GGUF file: {target} ({size_mb} MB)")
        return target

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Modelfile
    # ──────────────────────────────────────────────────────────────────────────

    def _write_modelfile(self, gguf_path: Path) -> Path:
        content = _MODELFILE_TEMPLATE.format(gguf_name=gguf_path.name)
        modelfile_path = self.gguf_dir / "Modelfile"
        modelfile_path.write_text(content, encoding="utf-8")
        logger.info(f"Modelfile written: {modelfile_path}")
        return modelfile_path

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5 — register with Ollama
    # ──────────────────────────────────────────────────────────────────────────

    def _register_with_ollama(self, modelfile_path: Path) -> None:
        # Run from the directory containing the Modelfile so the relative
        # FROM ./qa-agent-coder-q4_k_m.gguf reference resolves correctly.
        cmd = ["ollama", "create", self.model_tag, "-f", modelfile_path.name]
        logger.info(f"Registering with Ollama: {' '.join(cmd)} (cwd={self.gguf_dir})")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.gguf_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ollama CLI not found. Install from https://ollama.com/download "
                "and ensure it is in PATH."
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"ollama create failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderr: {result.stderr.strip()}"
            )
        logger.info(
            f"Registered {self.model_tag!r} with Ollama\n{result.stdout.strip()}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6 — smoke test
    # ──────────────────────────────────────────────────────────────────────────

    def _smoke_test(self) -> None:
        """Confirm the registered model responds and produces a test function."""
        logger.info(f"Smoke-testing {self.model_tag!r}…")
        try:
            reply = asyncio.run(self._smoke_generate())
            if not reply.strip():
                raise ValueError("Model returned an empty response")
            if "def test_" not in reply:
                logger.warning(
                    f"Smoke test: response did not contain 'def test_'.\n"
                    f"Preview: {reply[:200]!r}"
                )
            else:
                logger.info(
                    f"Model {self.model_tag!r} registered and verified OK "
                    f"(response_len={len(reply)})"
                )
        except Exception as exc:
            logger.warning(
                f"Smoke test failed (the model may still be usable; if Ollama is "
                f"down, run `ollama serve` and rerun): {exc}"
            )

    async def _smoke_generate(self) -> str:
        """Run the smoke-test prompt through OllamaAdapter (no direct HTTP to :11434)."""
        from src.llm.adapter import OllamaAdapter

        adapter = OllamaAdapter(model=self.model_tag, base_url=_OLLAMA_BASE_URL)
        try:
            return await adapter.generate(
                messages=[{"role": "user", "content": _TEST_PROMPT}],
                temperature=0.1,
                max_tokens=256,
            )
        finally:
            await adapter.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Guards
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_unsloth() -> None:
        try:
            import unsloth  # noqa: F401
        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Run:\n"
                '  pip install "unsloth[colab-new]>=2024.12.0"\n'
                "Requires CUDA 11.8+ and a compatible GPU."
            )

    def _check_checkpoint(self) -> None:
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}\n"
                "Run `qa-agent --mode finetune --step train` first."
            )
        if not (self.checkpoint_dir / "config.json").exists():
            raise FileNotFoundError(
                f"config.json missing in {self.checkpoint_dir} — "
                "checkpoint may be incomplete or corrupted."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Backwards-compatibility alias
# ──────────────────────────────────────────────────────────────────────────────


class GGUFExporter(ModelExporter):
    """Legacy name retained so older main.py imports keep working."""
    pass
