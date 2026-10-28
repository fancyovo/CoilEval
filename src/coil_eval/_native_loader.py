"""Locate packaged libraries and prepare their CUDA runtime dependencies."""
from __future__ import annotations

import ctypes
from importlib import metadata
from pathlib import Path
import threading

_DIRECTORY = Path(__file__).resolve().parent / "_native"
_LOCK = threading.Lock()
_HANDLES: list[ctypes.CDLL] = []
_READY = False
# Load leaves first: a dependency's RUNPATH is not inherited by its children.
_RUNTIME_LIBRARIES = (
    ("nvidia-cuda-runtime-cu12", "cuda_runtime", "libcudart.so.12"),
    ("nvidia-nvjitlink-cu12", "nvjitlink", "libnvJitLink.so.12"),
    ("nvidia-cublas-cu12", "cublas", "libcublasLt.so.12"),
    ("nvidia-cublas-cu12", "cublas", "libcublas.so.12"),
    ("nvidia-cusparse-cu12", "cusparse", "libcusparse.so.12"),
    ("nvidia-cusolver-cu12", "cusolver", "libcusolver.so.11"),
)


def _prepare_runtime() -> None:
    global _READY
    with _LOCK:
        if _READY:
            return
        for distribution, component, soname in _RUNTIME_LIBRARIES:
            try:
                dist = metadata.distribution(distribution)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Missing CUDA runtime dependency {distribution}. "
                    "Install the CoilEval gpu extra, for example pip install 'coil-eval[gpu]'."
                ) from exc
            path = Path(dist.locate_file(f"nvidia/{component}/lib/{soname}"))
            try:
                _HANDLES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
            except OSError as exc:
                raise RuntimeError(f"Could not load CUDA runtime {path}: {exc}") from exc
        _READY = True


def resolve_library(library=None, *, name="libcoil_eval_native.so") -> Path:
    """Explicit paths retain their own runtime; packaged paths use wheel deps."""
    if library is not None:
        path = Path(library).expanduser().resolve()
        if path.parent == _DIRECTORY.resolve():
            _prepare_runtime()
        return path
    path = _DIRECTORY / name
    if not path.is_file():
        raise FileNotFoundError(
            f"The installed CoilEval package does not contain {name}. "
            "Install a Linux binary wheel, or build the native backend from source "
            "and pass library='/path/to/libcoil_eval_native.so'."
        )
    _prepare_runtime()
    return path
