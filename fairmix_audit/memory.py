from __future__ import annotations

import ctypes
import ctypes.util
import gc
import resource
import sys
from collections.abc import Callable


_MALLOC_TRIM: Callable[[int], int] | None | bool = None


def rss_mb() -> float:
    """Return current RSS on Linux, falling back to peak RSS elsewhere."""
    if sys.platform.startswith("linux"):
        proc_rss = _rss_mb_from_proc_status()
        if proc_rss is not None:
            return proc_rss

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _rss_mb_from_proc_status() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    return _rss_mb_from_proc_status_text(text)


def _rss_mb_from_proc_status_text(text: str) -> float | None:
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            value = float(parts[1])
        except ValueError:
            return None
        unit = parts[2].lower() if len(parts) > 2 else "kb"
        if unit == "kb":
            return value / 1024
        if unit == "mb":
            return value
        if unit == "gb":
            return value * 1024
        return None
    return None


def trim_process_memory() -> bool:
    """Collect Python garbage and ask glibc to release free heap pages."""
    gc.collect()
    trim = _load_malloc_trim()
    if trim is None:
        return False
    try:
        return bool(trim(0))
    except (OSError, TypeError, ValueError):
        return False


def _load_malloc_trim() -> Callable[[int], int] | None:
    global _MALLOC_TRIM
    if _MALLOC_TRIM is False:
        return None
    if _MALLOC_TRIM is not None:
        return _MALLOC_TRIM
    if not sys.platform.startswith("linux"):
        _MALLOC_TRIM = False
        return None
    try:
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_name)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except (AttributeError, OSError):
        _MALLOC_TRIM = False
        return None
    _MALLOC_TRIM = trim
    return trim
