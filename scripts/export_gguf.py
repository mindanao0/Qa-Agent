"""Convert fine-tuned LoRA adapter → GGUF Q4_K_M for Ollama.

Run:
    uv run python scripts/export_gguf.py

Input:  models/finetune_output/final   (trained checkpoint from finetune.py)
Output: models/qwen2.5-coder-finetuned/qwen2.5-coder-finetuned.Q4_K_M.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

CHECKPOINT_DIR = Path("models/finetune_output/final")
OUTPUT_DIR = Path("models/qwen2.5-coder-finetuned")
GGUF_NAME = "qwen2.5-coder-finetuned.Q4_K_M.gguf"
MAX_SEQ_LENGTH = 1024


def export(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    output_dir: Path = OUTPUT_DIR,
    gguf_name: str = GGUF_NAME,
) -> Path:
    """
    Load the fine-tuned checkpoint, merge LoRA, and save as GGUF Q4_K_M.

    Returns the Path to the produced .gguf file.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            f"Unsloth not installed: {exc}\n"
            "Run: uv run python scripts/install_finetune_deps.py"
        ) from exc

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}\n"
            "Run finetune.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Exporting GGUF Q4_K_M → {output_dir}")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Rename whatever Unsloth produced to the stable name.
    candidates = sorted(output_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"No .gguf file found in {output_dir} after export")

    target = output_dir / gguf_name
    if candidates[-1] != target:
        candidates[-1].replace(target)
        print(f"Renamed {candidates[-1].name} → {target.name}")

    print(f"GGUF export complete: {target}")
    return target


def main() -> int:
    try:
        export()
        return 0
    except (ImportError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
