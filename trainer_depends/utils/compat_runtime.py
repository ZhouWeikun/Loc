import ctypes
import os
import sys
from pathlib import Path


def preload_conda_libstdcpp():
    """Prefer the active conda env's libstdc++ before importing C++ extensions."""
    if os.name != "posix":
        return None

    candidates = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libstdc++.so.6")

    try:
        exec_prefix = Path(sys.executable).resolve().parent.parent
    except OSError:
        exec_prefix = None
    if exec_prefix is not None:
        candidates.append(exec_prefix / "lib" / "libstdc++.so.6")

    seen = set()
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            ctypes.CDLL(str(candidate), mode=mode)
            return str(candidate)
        except OSError:
            continue

    return None
