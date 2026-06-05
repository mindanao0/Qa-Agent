"""Create Ollama Modelfile for the fine-tuned model and register it.

Run:
    uv run python scripts/create_modelfile.py

Requires:
    - models/qwen2.5-coder-finetuned/qwen2.5-coder-finetuned.Q4_K_M.gguf (from export_gguf.py)
    - ollama CLI installed and daemon running

After registration, verify with:
    ollama run qa-agent-finetuned "Generate a pytest test for..."
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

GGUF_DIR = Path("models/qwen2.5-coder-finetuned")
GGUF_NAME = "qwen2.5-coder-finetuned.Q4_K_M.gguf"
MODEL_TAG = "qa-agent-finetuned"

MODELFILE_TEMPLATE = """\
FROM ./{gguf_name}
SYSTEM \"\"\"You are an expert QA automation engineer. You write pytest-playwright \
tests using only semantic locators: get_by_role, get_by_label, get_by_text, \
get_by_test_id. You never use CSS selectors or XPath. You always use expect() \
for assertions. Function names must start with test_.\"\"\"
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
"""


def write_modelfile(
    gguf_path: Path,
    output_dir: Path,
) -> Path:
    """Write Modelfile to output_dir. Return the Modelfile path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    content = MODELFILE_TEMPLATE.format(gguf_name=gguf_path.name)
    modelfile_path = output_dir / "Modelfile"
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"Modelfile written: {modelfile_path}")
    return modelfile_path


def register_with_ollama(modelfile_path: Path, model_tag: str = MODEL_TAG) -> None:
    """Run `ollama create <tag> -f Modelfile` from the Modelfile's directory."""
    cmd = ["ollama", "create", model_tag, "-f", modelfile_path.name]
    print(f"Running: {' '.join(cmd)}  (cwd={modelfile_path.parent})")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(modelfile_path.parent),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
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
    print(f"Registered {model_tag!r} with Ollama.")
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> int:
    gguf_path = GGUF_DIR / GGUF_NAME
    if not gguf_path.exists():
        print(f"ERROR: GGUF file not found: {gguf_path}", file=sys.stderr)
        print("Run export_gguf.py first.", file=sys.stderr)
        return 1
    try:
        modelfile_path = write_modelfile(gguf_path=gguf_path, output_dir=GGUF_DIR)
        register_with_ollama(modelfile_path)
        print(f"\nDone! Test with:")
        print(f'  ollama run {MODEL_TAG} "Generate a pytest test for login page"')
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
