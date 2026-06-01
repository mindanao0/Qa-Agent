"""One-time setup: install Node.js devDependencies for Sprint 9."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def main() -> None:
    pkg = _ROOT / "package.json"
    if not pkg.exists():
        print(f"ERROR: {pkg} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[setup_vitest] Running npm install in {_ROOT}")
    result = subprocess.run(
        ["npm", "install"],
        cwd=_ROOT,
        timeout=180,
    )
    if result.returncode != 0:
        print("[setup_vitest] npm install FAILED", file=sys.stderr)
        sys.exit(1)
    print("[setup_vitest] npm install complete")
    node_modules = _ROOT / "node_modules"
    vitest_bin = node_modules / ".bin" / "vitest"
    if not vitest_bin.exists() and not (node_modules / ".bin" / "vitest.cmd").exists():
        print(f"[setup_vitest] WARNING: vitest binary not found at {vitest_bin}", file=sys.stderr)
    else:
        print("[setup_vitest] vitest binary found — ready")


if __name__ == "__main__":
    main()
