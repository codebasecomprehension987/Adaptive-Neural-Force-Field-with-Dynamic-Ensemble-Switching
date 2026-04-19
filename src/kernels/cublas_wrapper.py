"""
anff/kernels/cublas_wrapper.py
==============================
Host-side cuBLAS batched GEMM wrapper, invoked via `jax.pure_callback`
from `MessagePassingLayer._batched_gemm`.

Why a separate module?
----------------------
Keeping the ctypes / shared-library loading isolated here means that:
  a) The rest of the codebase imports cleanly on hosts without CUDA.
  b) The cuBLAS library path is configurable via environment variable.
  c) Unit tests can monkey-patch `_cublas_sgemm` without touching JAX internals.

Dispatch priority
-----------------
1. `libcublas_anff.so` — project-compiled wrapper around cuBLAS
   `cublasSgemmBatched`.
2. `cupy.cublas` — if CuPy is installed, use its managed cuBLAS handle.
3. NumPy `matmul` — CPU fallback; correct but bypasses all GPU acceleration.

Environment variables
---------------------
ANFF_CUBLAS_LIB
    Path to the compiled `libcublas_anff.so`.  Defaults to the package's
    ``src/kernels/build/`` directory.
ANFF_FORCE_NUMPY_GEMM
    Set to ``"1"`` to force the NumPy fallback (useful for CI and debugging).
"""

from __future__ import annotations

import os
import ctypes
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None
_cupy_cublas = None
_backend: str = "numpy"


def _load_backends() -> None:
    global _lib, _cupy_cublas, _backend

    if os.environ.get("ANFF_FORCE_NUMPY_GEMM", "0") == "1":
        log.info("ANFF_FORCE_NUMPY_GEMM=1: using NumPy GEMM fallback.")
        return

    # --- Attempt 1: project-compiled shared library ---
    default_lib_path = (
        Path(__file__).parent / "build" / "libcublas_anff.so"
    )
    lib_path = Path(os.environ.get("ANFF_CUBLAS_LIB", str(default_lib_path)))

    if lib_path.exists():
        try:
            _lib = ctypes.CDLL(str(lib_path))
            _lib.anff_sgemm_batched.restype  = ctypes.c_int
            _lib.anff_sgemm_batched.argtypes = [
                ctypes.c_void_p,  # A  (float*)
                ctypes.c_void_p,  # B  (float*)
                ctypes.c_void_p,  # C  (float*)
                ctypes.c_int,     # M
                ctypes.c_int,     # N
                ctypes.c_int,     # K
                ctypes.c_int,     # batch_count
            ]
            _backend = "libcublas_anff"
            log.info(f"Loaded cuBLAS wrapper from {lib_path}")
            return
        except OSError as exc:
            log.warning(f"Failed to load {lib_path}: {exc}")

    # --- Attempt 2: CuPy ---
    try:
        import cupy
        import cupy.cublas as _cp_cublas
        _cupy_cublas = _cp_cublas
        _backend = "cupy"
        log.info("Using CuPy cuBLAS backend.")
        return
    except ImportError:
        pass

    log.info("No CUDA backend available; using NumPy GEMM fallback.")


_load_backends()


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def cublas_sgemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute ``C = A @ B`` using the best available GEMM backend.

    Parameters
    ----------
    A : np.ndarray, shape [M, K], dtype float32
    B : np.ndarray, shape [K, N], dtype float32

    Returns
    -------
    C : np.ndarray, shape [M, N], dtype float32
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    if _backend == "libcublas_anff":
        return _gemm_via_clib(A, B)
    elif _backend == "cupy":
        return _gemm_via_cupy(A, B)
    else:
        return np.matmul(A, B)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _gemm_via_clib(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M, K = A.shape
    _, N = B.shape
    C = np.empty((M, N), dtype=np.float32)

    ret = _lib.anff_sgemm_batched(  # type: ignore[union-attr]
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p),
        C.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_int(1),   # batch_count = 1 for 2-D inputs
    )
    if ret != 0:
        raise RuntimeError(f"anff_sgemm_batched returned error code {ret}")
    return C


def _gemm_via_cupy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    import cupy as cp
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    C_gpu = cp.matmul(A_gpu, B_gpu)
    return cp.asnumpy(C_gpu)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def backend_info() -> dict:
    """Return the active backend name and library path for logging."""
    return {
        "backend": _backend,
        "lib_path": str(
            Path(os.environ.get(
                "ANFF_CUBLAS_LIB",
                str(Path(__file__).parent / "build" / "libcublas_anff.so")
            ))
        ),
    }
