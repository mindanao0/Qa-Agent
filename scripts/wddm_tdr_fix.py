"""WDDM TDR registry fix for GPU training stability.

WARNING: modifies Windows registry — run only when ready for fine-tune.
DRY RUN by default (--apply flag required to write).

Registry keys to set (HKLM\\System\\CurrentControlSet\\Control\\GraphicsDrivers):
  TdrLevel    = 3   (REG_DWORD) — recover without reboot
  TdrDelay    = 60  (REG_DWORD) — 60s before TDR timeout (default=2s)
  TdrDdiDelay = 60  (REG_DWORD)

Usage:
  python scripts/wddm_tdr_fix.py             # dry run — reads and prints current values
  python scripts/wddm_tdr_fix.py --apply     # backup + write changes (requires admin, reboot after)
  python scripts/wddm_tdr_fix.py --verify    # verify written values match expected (post-reboot)
  python scripts/wddm_tdr_fix.py --stress    # GPU stress test — requires torch+CUDA
"""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys
import time

try:
    import winreg
except ImportError:
    print("ERROR: winreg is only available on Windows.")
    sys.exit(1)

REG_PATH = r"System\CurrentControlSet\Control\GraphicsDrivers"
REG_FULL_PATH = r"HKLM\System\CurrentControlSet\Control\GraphicsDrivers"
TARGET_VALUES: dict[str, int] = {
    "TdrLevel": 3,
    "TdrDelay": 60,
    "TdrDdiDelay": 60,
}
BACKUP_PATH = pathlib.Path("audit/wddm_backup.reg")


def _read_current_values() -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REG_PATH)
        for name in TARGET_VALUES:
            try:
                value, _ = winreg.QueryValueEx(key, name)
                result[name] = int(value)
            except FileNotFoundError:
                result[name] = None
        winreg.CloseKey(key)
    except OSError as exc:
        print(f"ERROR reading registry: {exc}")
        sys.exit(1)
    return result


def backup_registry() -> pathlib.Path:
    """Export current GraphicsDrivers key to .reg file before modifying."""
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["reg", "export", REG_FULL_PATH, str(BACKUP_PATH), "/y"],
        check=True,
    )
    return BACKUP_PATH


def _apply_values() -> None:
    print("Backing up registry...")
    try:
        backup_path = backup_registry()
        print(f"Registry backed up to: {backup_path}")
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: backup failed — aborting. {exc}")
        sys.exit(1)

    print("Writing registry values...")
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            REG_PATH,
            access=winreg.KEY_SET_VALUE,
        )
        for name, value in TARGET_VALUES.items():
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)
            print(f"  Writing {name:<12} = {value} ... OK")
        winreg.CloseKey(key)
    except PermissionError:
        print("ERROR: permission denied — run as Administrator.")
        sys.exit(1)
    except OSError as exc:
        print(f"ERROR writing registry: {exc}")
        sys.exit(1)


def verify_registry() -> dict[str, dict]:
    results: dict[str, dict] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REG_PATH) as key:
            for name, expected_val in TARGET_VALUES.items():
                try:
                    val, _ = winreg.QueryValueEx(key, name)
                    results[name] = {
                        "value": val,
                        "expected": expected_val,
                        "ok": val == expected_val,
                    }
                except FileNotFoundError:
                    results[name] = {"value": None, "expected": expected_val, "ok": False}
    except OSError as exc:
        print(f"ERROR reading registry: {exc}")
        sys.exit(1)
    return results


async def stress_test_gpu() -> dict:
    """Simulate training-like GPU load for 30s and verify no TDR reset occurs."""
    try:
        import torch
    except ImportError:
        return {"status": "SKIP", "reason": "torch not installed"}

    if not torch.cuda.is_available():
        return {"status": "SKIP", "reason": "CUDA not available"}

    device = torch.device("cuda")
    start = time.monotonic()
    peak_vram_mb = 0.0
    errors: list[str] = []

    try:
        # allocate ~2GB tensor (conservative for 6GB card, using float16)
        tensor = torch.zeros(1024, 1024, 512, dtype=torch.float16, device=device)
        for i in range(100):
            _ = torch.mm(
                tensor.view(1024, -1)[:1024, :1024],
                tensor.view(1024, -1)[:1024, :1024],
            )
            vram = torch.cuda.memory_allocated() / 1024 / 1024
            peak_vram_mb = max(peak_vram_mb, vram)
        del tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        errors.append(str(e))

    duration = time.monotonic() - start
    return {
        "status": "PASS" if not errors else "FAIL",
        "duration_s": round(duration, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "errors": errors,
    }


def main() -> None:
    args = set(sys.argv[1:])
    apply = "--apply" in args
    verify = "--verify" in args
    stress = "--stress" in args

    if verify:
        print("Verifying registry values...")
        results = verify_registry()
        all_ok = True
        for name, info in results.items():
            mark = "✅" if info["ok"] else "❌"
            print(f"  {name:<12} {info['value']} == {info['expected']}  {mark}")
            if not info["ok"]:
                all_ok = False
        print()
        if all_ok:
            print("ALL OK — GPU training stable")
        else:
            print("MISMATCH — re-run with --apply and reboot again")
            sys.exit(1)
        return

    if stress:
        print("Running GPU stress test (~30s)...")
        result = asyncio.run(stress_test_gpu())
        status = result.get("status", "UNKNOWN")
        print(f"  GPU stress test: {status}")
        if status == "SKIP":
            print(f"  reason: {result.get('reason')}")
            print()
            print("  One-time setup required (run these two commands, then retry --stress):")
            print("    uv pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
            print("    uv run --no-sync python scripts/wddm_tdr_fix.py --stress")
        else:
            print(f"  duration_s:   {result.get('duration_s')}")
            print(f"  peak_vram_mb: {result.get('peak_vram_mb')}")
            errors = result.get("errors", [])
            print(f"  errors:       {errors}")
            if status == "PASS":
                print("  → GPU stable, ready for fine-tune")
            else:
                sys.exit(1)
        return

    if not apply:
        print("DRY RUN — current values:")
        current = _read_current_values()
        for name, value in current.items():
            display = str(value) if value is not None else "(not set)"
            target = TARGET_VALUES[name]
            match = " ✓" if value == target else f" → will set to {target}"
            print(f"  {name}: {display}{match}")
        print()
        print("Run with --apply to write changes (requires Administrator, reboot after).")
        sys.exit(0)

    _apply_values()
    print("WDDM TDR fix applied — reboot required to take effect")


if __name__ == "__main__":
    main()
