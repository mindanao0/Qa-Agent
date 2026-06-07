## CONTEXT
PROJECT: D:\Code\qa-agent
SCOPE: Apply WDDM TDR registry fix — prerequisites สำหรับ GPU training
OS: Windows 11 bare-metal (no Docker, no WSL)
GPU: GTX 1660 Ti 6GB VRAM
SCRIPT: scripts/wddm_tdr_fix.py (already created in cleanup)

## OBJECTIVE
GOAL: Apply TDR registry values + verify GPU stability
      ก่อนรัน fine-tune pipeline

## STEP 1 — Dry run ก่อน (อ่านค่าปัจจุบัน)

uv run python scripts/wddm_tdr_fix.py
# Expected output:
# DRY RUN — current values:
#   TdrLevel:    (not set) → will set to 3
#   TdrDelay:    (not set) → will set to 60
#   TdrDdiDelay: (not set) → will set to 60
# Run with --apply to write changes

## STEP 2 — Backup current registry state

# Add backup capability to scripts/wddm_tdr_fix.py:
#
# def backup_registry() -> pathlib.Path:
#     """Export current GraphicsDrivers key to .reg file before modifying."""
#     import subprocess
#     backup_path = pathlib.Path("audit/wddm_backup.reg")
#     subprocess.run([
#         "reg", "export",
#         r"HKLM\System\CurrentControlSet\Control\GraphicsDrivers",
#         str(backup_path), "/y"
#     ], check=True)
#     return backup_path
#
# Call backup_registry() BEFORE any registry write in --apply mode
# Print: "Registry backed up to: audit/wddm_backup.reg"

## STEP 3 — Apply fix

uv run python scripts/wddm_tdr_fix.py --apply
# Expected:
# Registry backed up to: audit/wddm_backup.reg
# Writing TdrLevel    = 3  ... OK
# Writing TdrDelay    = 60 ... OK
# Writing TdrDdiDelay = 60 ... OK
# WDDM TDR fix applied — reboot required to take effect

## STEP 4 — Verify after reboot

# Add --verify flag to scripts/wddm_tdr_fix.py:
#
# def verify_registry() -> dict:
#     import winreg
#     key_path = r"System\CurrentControlSet\Control\GraphicsDrivers"
#     expected = {"TdrLevel": 3, "TdrDelay": 60, "TdrDdiDelay": 60}
#     results = {}
#     with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
#         for name, expected_val in expected.items():
#             try:
#                 val, _ = winreg.QueryValueEx(key, name)
#                 results[name] = {
#                     "value":    val,
#                     "expected": expected_val,
#                     "ok":       val == expected_val,
#                 }
#             except FileNotFoundError:
#                 results[name] = {"value": None, "expected": expected_val, "ok": False}
#     return results
#
# Run after reboot:
# uv run python scripts/wddm_tdr_fix.py --verify
# Expected:
#   TdrLevel:    3  == 3  ✅
#   TdrDelay:    60 == 60 ✅
#   TdrDdiDelay: 60 == 60 ✅
#   ALL OK — GPU training stable

## STEP 5 — GPU stress test (after reboot + verify)

# Add to scripts/wddm_tdr_fix.py --stress flag:
#
# async def stress_test_gpu() -> dict:
#     """
#     Simulate training-like GPU load for 30s
#     Verify no TDR reset occurs
#     """
#     import torch
#     if not torch.cuda.is_available():
#         return {"status": "SKIP", "reason": "CUDA not available"}
#
#     device = torch.device("cuda")
#     start = time.monotonic()
#     peak_vram_mb = 0.0
#     errors = []
#
#     try:
#         # allocate ~4GB tensor (conservative for 6GB card)
#         tensor = torch.zeros(1024, 1024, 1024, dtype=torch.float16, device=device)
#         for i in range(100):
#             # matrix multiply — GPU compute load
#             _ = torch.mm(
#                 tensor.view(1024, -1)[:1024, :1024],
#                 tensor.view(1024, -1)[:1024, :1024],
#             )
#             vram = torch.cuda.memory_allocated() / 1024 / 1024
#             peak_vram_mb = max(peak_vram_mb, vram)
#         del tensor
#         torch.cuda.empty_cache()
#     except RuntimeError as e:
#         errors.append(str(e))
#
#     duration = time.monotonic() - start
#     return {
#         "status":       "PASS" if not errors else "FAIL",
#         "duration_s":   round(duration, 2),
#         "peak_vram_mb": round(peak_vram_mb, 1),
#         "errors":       errors,
#     }
#
# uv run python scripts/wddm_tdr_fix.py --stress
# Expected:
#   GPU stress test: PASS
#   duration: ~30s
#   peak_vram_mb: ~4000
#   errors: []
#   → GPU stable, ready for fine-tune

## RULES:
# - backup_registry() ต้องสำเร็จก่อน --apply จึงจะเขียน registry
# - ถ้า backup fail → abort (ห้ามเขียน registry โดยไม่มี backup)
# - winreg module เท่านั้น (stdlib) — ห้าม third-party registry libs
# - pathlib.Path สำหรับ file operations
# - --stress ต้องการ torch (uv add torch --index-url
#   https://download.pytorch.org/whl/cu121 ถ้ายังไม่มี)
# - ถ้า CUDA ไม่พร้อม → --stress returns SKIP (ไม่ใช่ FAIL)

## OUTPUT CONTRACT
รายงานหลัง reboot + verify + stress:

WDDM TDR REPORT
===============
backup:       OK — audit/wddm_backup.reg
apply:        OK — 3 keys written
reboot:       DONE
verify:
  TdrLevel:    3  ✅
  TdrDelay:    60 ✅
  TdrDdiDelay: 60 ✅
stress_test:
  status:       PASS | SKIP
  peak_vram_mb: <float>
  duration_s:   <float>
ready_for_finetune: true