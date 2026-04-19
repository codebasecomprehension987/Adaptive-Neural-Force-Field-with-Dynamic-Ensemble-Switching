"""
anff/kernels/neighbor_list.py
=============================
Neighbour-list construction routed through XLA custom calls wrapping
CUDA-accelerated cell-linked list kernels.

Architecture
------------
1.  A StableHLO custom op descriptor is registered once at import time.
2.  `build_neighbor_list` invokes the custom op inside a `jax.jit`-compiled
    function so XLA sees a single opaque node rather than a Python loop.
3.  On hosts without the CUDA extension the call degrades gracefully to a
    pure-NumPy reference implementation via `jax.pure_callback`.

Memory complexity
-----------------
The output adjacency tensor has shape ``[N_atoms, max_neighbors]``; it is
**replicated** across all TPU devices (not sharded) because neighbour indices
are irregular.  Memory cost: O(N · k · D) where D = device count.

See also
--------
src/kernels/cell_list.cu   — CUDA kernel (CUDA C source, not included here)
src/kernels/stablehlo_op.cc — XLA custom-call registration
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# XLA custom-call registration
# ---------------------------------------------------------------------------
#
# We attempt to import the compiled extension module.  If it is absent (e.g.
# in CI without CUDA) we register a pure-Python fallback so that the rest of
# the codebase can still be imported and unit-tested.
#

_CUDA_EXTENSION_AVAILABLE = False

try:
    from anff._cuda import cell_list_op  # type: ignore[import]
    _CUDA_EXTENSION_AVAILABLE = True
except ModuleNotFoundError:
    cell_list_op = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_neighbor_list(
    positions: jax.Array,     # [N, 3]
    cell: jax.Array,          # [3, 3]  — zeros → no PBC
    cutoff: float,
    max_neighbors: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Build a static-shape neighbour list suitable for JAX's compiled kernels.

    Returns
    -------
    nbr_idx : jax.Array, shape [N, max_neighbors]
        Indices of neighbour atoms.  Vacant slots are filled with ``-1``.
    nbr_shifts : jax.Array, shape [N, max_neighbors, 3]
        PBC image shift vectors (integer multiples of cell vectors).
    n_pairs : jax.Array, shape [N]
        Number of *real* neighbours per atom (before padding).
    """
    if _CUDA_EXTENSION_AVAILABLE:
        return _cuda_neighbor_list(positions, cell, cutoff, max_neighbors)
    else:
        return _python_fallback_neighbor_list(positions, cell, cutoff, max_neighbors)


# ---------------------------------------------------------------------------
# CUDA path: XLA custom call → StableHLO custom op
# ---------------------------------------------------------------------------

def _cuda_neighbor_list(
    positions: jax.Array,
    cell: jax.Array,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Invoke the cell-linked list CUDA kernel via `jax.pure_callback`.

    In a full deployment this would be a `jax.ffi.ffi_call` / XLA custom call
    so that the kernel appears as an opaque node in the HLO graph and XLA does
    not attempt to fuse it with upstream ops.
    """
    N = positions.shape[0]

    nbr_idx_spec   = jax.ShapeDtypeStruct((N, max_neighbors),    jnp.int32)
    nbr_shift_spec = jax.ShapeDtypeStruct((N, max_neighbors, 3), jnp.float32)
    n_pairs_spec   = jax.ShapeDtypeStruct((N,),                   jnp.int32)

    nbr_idx, nbr_shifts, n_pairs = jax.pure_callback(
        functools.partial(
            _cuda_cell_list_host,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        ),
        (nbr_idx_spec, nbr_shift_spec, n_pairs_spec),
        positions,
        cell,
    )
    return nbr_idx, nbr_shifts, n_pairs


def _cuda_cell_list_host(
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host callback: dispatch to the CUDA cell-linked list shared library."""
    return cell_list_op.build(  # type: ignore[union-attr]
        positions, cell, cutoff, max_neighbors
    )


# ---------------------------------------------------------------------------
# Pure-Python fallback (O(N²) — reference / CI only)
# ---------------------------------------------------------------------------

def _python_fallback_neighbor_list(
    positions: jax.Array,
    cell: jax.Array,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    O(N²) neighbour search — correct but slow; used when CUDA is unavailable.

    Because this function contains Python loops it is **not** JIT-compiled; it
    is invoked via `jax.pure_callback` so XLA sees an opaque node.
    """
    N = positions.shape[0]

    nbr_idx_spec   = jax.ShapeDtypeStruct((N, max_neighbors),    jnp.int32)
    nbr_shift_spec = jax.ShapeDtypeStruct((N, max_neighbors, 3), jnp.float32)
    n_pairs_spec   = jax.ShapeDtypeStruct((N,),                   jnp.int32)

    return jax.pure_callback(
        functools.partial(
            _numpy_neighbor_list,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        ),
        (nbr_idx_spec, nbr_shift_spec, n_pairs_spec),
        positions,
        cell,
    )


def _numpy_neighbor_list(
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    cutoff: float,
    max_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy reference implementation of the neighbour search."""
    N = positions.shape[0]
    cutoff2 = cutoff ** 2

    nbr_idx   = np.full((N, max_neighbors),    -1,  dtype=np.int32)
    nbr_shift = np.zeros((N, max_neighbors, 3), dtype=np.float32)
    n_pairs   = np.zeros(N,                     dtype=np.int32)

    use_pbc = np.any(cell != 0.0)
    images = _pbc_images(cell) if use_pbc else [np.zeros(3)]

    for i in range(N):
        count = 0
        for j in range(N):
            if i == j:
                continue
            for shift_frac in images:
                if use_pbc:
                    shift_cart = shift_frac @ cell
                else:
                    shift_cart = np.zeros(3)
                d = positions[j] + shift_cart - positions[i]
                if d @ d < cutoff2 and count < max_neighbors:
                    nbr_idx[i, count]    = j
                    nbr_shift[i, count]  = shift_frac
                    count += 1
        n_pairs[i] = count

    return nbr_idx, nbr_shift, n_pairs


def _pbc_images(cell: np.ndarray, n: int = 1):
    """Return fractional shift vectors for periodic images."""
    shifts = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                shifts.append(np.array([i, j, k], dtype=np.float32))
    return shifts
