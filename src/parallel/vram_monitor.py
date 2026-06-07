"""VRAMMonitor — Sprint 15.

Polls GPU memory during the parallel run via the Windows NVIDIA Management
Library (``nvml.dll``) through ctypes. On any machine where NVML is unavailable
(no NVIDIA driver, headless CI, Linux without the DLL) the monitor degrades
gracefully: ``used_mb()`` returns ``None`` and ``is_stable()`` returns ``True``
so a run on non-NVIDIA hardware is never penalised (see Sprint 15 RULES).

The 6 GB GTX 1660 Ti budget leaves a 200 MB headroom, so the hard ceiling is
5800 MB — crossing it during ``monitor_during`` raises ``VRAMExceededError``.
"""
from __future__ import annotations

import asyncio
import ctypes
from typing import Any, Coroutine

from loguru import logger

# Leave 200 MB headroom on a 6 GB card.
_VRAM_HARD_LIMIT_MB = 5800.0

# Candidate locations for nvml.dll on Windows (in load order).
_NVML_DLL_CANDIDATES = (
    "nvml.dll",
    r"C:\Windows\System32\nvml.dll",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll",
)

_NVML_SUCCESS = 0


class VRAMExceededError(RuntimeError):
    """Raised when polled VRAM crosses the hard limit during a monitored run."""


class _NvmlMemory(ctypes.Structure):
    """Mirror of nvmlMemory_t (three unsigned long long: total, free, used)."""

    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class VRAMMonitor:
    """No required constructor args. Graceful fallback if NVML unavailable."""

    def __init__(self) -> None:
        self._nvml: Any = None
        self._handle: ctypes.c_void_p | None = None
        self._available: bool = False
        try:
            self._nvml = self._load_nvml()
            self._init_device()
            self._available = True
            logger.debug("VRAMMonitor: NVML initialised, GPU 0 handle acquired")
        except Exception as exc:  # OSError, AttributeError, any NVML failure
            logger.info(f"VRAMMonitor: NVML unavailable ({exc!r}); VRAM checks degrade to True")
            self._available = False

    # ── NVML bootstrap ──────────────────────────────────────────────────────

    @staticmethod
    def _load_nvml() -> Any:
        last_exc: Exception | None = None
        for candidate in _NVML_DLL_CANDIDATES:
            try:
                return ctypes.CDLL(candidate)
            except OSError as exc:
                last_exc = exc
        raise OSError(f"could not load nvml.dll from {_NVML_DLL_CANDIDATES}: {last_exc!r}")

    def _init_device(self) -> None:
        # nvmlInit_v2 is preferred; fall back to the legacy nvmlInit symbol.
        init = getattr(self._nvml, "nvmlInit_v2", None) or getattr(self._nvml, "nvmlInit")
        rc = init()
        if rc != _NVML_SUCCESS:
            raise OSError(f"nvmlInit returned {rc}")
        handle = ctypes.c_void_p()
        get_handle = (
            getattr(self._nvml, "nvmlDeviceGetHandleByIndex_v2", None)
            or getattr(self._nvml, "nvmlDeviceGetHandleByIndex")
        )
        rc = get_handle(0, ctypes.byref(handle))
        if rc != _NVML_SUCCESS:
            raise OSError(f"nvmlDeviceGetHandleByIndex returned {rc}")
        self._handle = handle

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def used_mb(self) -> float | None:
        """Return VRAM used in MB, or ``None`` if NVML unavailable."""
        if not self._available or self._handle is None:
            return None
        try:
            mem = _NvmlMemory()
            rc = self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(mem))
            if rc != _NVML_SUCCESS:
                return None
            return mem.used / (1024.0 * 1024.0)
        except Exception as exc:
            logger.warning(f"VRAMMonitor.used_mb failed: {exc!r}")
            return None

    def is_stable(self, threshold_mb: float = _VRAM_HARD_LIMIT_MB) -> bool:
        """True if current used VRAM < ``threshold_mb`` (or NVML unavailable)."""
        used = self.used_mb()
        if used is None:
            return True
        return used < threshold_mb

    # ── Monitoring ──────────────────────────────────────────────────────────────

    async def monitor_during(
        self,
        coro: Coroutine,
        poll_interval_s: float = 0.5,
    ) -> tuple[Any, float | None]:
        """Run ``coro`` while polling VRAM every ``poll_interval_s``.

        Returns ``(coro_result, peak_vram_mb)`` where ``peak_vram_mb`` is the
        maximum observed used VRAM, or ``None`` if NVML was unavailable for the
        whole run. Raises ``VRAMExceededError`` if any sample crosses the hard
        limit (5800 MB) — the running coroutine is cancelled first.
        """
        task = asyncio.ensure_future(coro)
        peak: float | None = None
        try:
            while not task.done():
                used = self.used_mb()
                if used is not None:
                    peak = used if peak is None else max(peak, used)
                    if used > _VRAM_HARD_LIMIT_MB:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                        raise VRAMExceededError(
                            f"VRAM {used:.0f}MB exceeded hard limit {_VRAM_HARD_LIMIT_MB:.0f}MB"
                        )
                await asyncio.sleep(poll_interval_s)
            result = await task
        except VRAMExceededError:
            raise
        except Exception:
            # Surface coroutine errors after ensuring the task is finalised.
            raise
        # Final sample after completion.
        used = self.used_mb()
        if used is not None:
            peak = used if peak is None else max(peak, used)
        return result, peak


__all__ = ["VRAMMonitor", "VRAMExceededError"]
