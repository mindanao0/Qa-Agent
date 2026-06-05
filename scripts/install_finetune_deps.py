"""Install Unsloth + QLoRA fine-tuning dependencies.

Run:
    uv run python scripts/install_finetune_deps.py

On Windows, if bitsandbytes fails use:
    pip install bitsandbytes \
        --index-url https://jllllll.github.io/bitsandbytes-windows-webui
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys

DEPS = [
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "accelerate>=0.29.0",
    "datasets>=2.18.0",
]

_VERIFY_IMPORTS = [
    "unsloth", "transformers", "trl", "peft",
    "bitsandbytes", "accelerate", "datasets",
]


def install() -> bool:
    """Install all deps. Return True on success."""
    for dep in DEPS:
        print(f"Installing: {dep}")
        # Use 'uv pip install' — uv venvs don't ship pip by default.
        result = subprocess.run(
            ["uv", "pip", "install", dep],
            check=False,
        )
        if result.returncode != 0:
            print(f"FAILED: {dep}")
            return False
    return True


def verify() -> bool:
    """Verify all packages are findable. Return True if all OK.

    Uses importlib.util.find_spec (no-import check) because torch/unsloth/triton
    can segfault or raise at import time on Windows before a GPU context is active.
    """
    ok = True
    for pkg in _VERIFY_IMPORTS:
        spec = importlib.util.find_spec(pkg)
        if spec is not None:
            print(f"OK:      {pkg}")
        else:
            print(f"MISSING: {pkg}")
            ok = False
    return ok


def main() -> int:
    if not install():
        return 1
    print("\nVerifying imports...")
    if not verify():
        return 1
    print("\nAll dependencies installed and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
