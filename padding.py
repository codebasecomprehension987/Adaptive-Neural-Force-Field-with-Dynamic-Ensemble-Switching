"""
anff/utils/padding.py
=====================
Array padding / unpadding helpers.

Padding strategy
----------------
JAX requires static shapes for JIT compilation.  We pad every system to the
**same** ``max_atoms`` value within a batch, then mask out padded entries with
a boolean array.

Memory trade-off (from spec)
-----------------------------
A 500-atom system padded for a 2 000-atom neighbour list blows working memory
by 16×.  To mitigate this we bucket systems by size, compiling one XLA module
per bucket rather than one global maximum.

Bucketing is handled at the `ForceField` level (see `_get_or_compile`);
these utilities only handle the padding arithmetic for a single system.
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# Default bucket boundaries (atom counts).  Adjust for your workload.
ATOM_COUNT_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)


def bucket_size(n_atoms: int) -> int:
    """Return the smallest bucket size >= n_atoms."""
    for b in ATOM_COUNT_BUCKETS:
        if n_atoms <= b:
            return b
    # Larger than any bucket — round up to next power of two.
    return int(2 ** np.ceil(np.log2(n_atoms)))


def pad_to_max(
    positions: jax.Array,       # [N, 3]
    atomic_numbers: jax.Array,  # [N]
    n_atoms: int,
    max_atoms: int | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Pad *positions* and *atomic_numbers* to *max_atoms* rows.

    Returns
    -------
    padded_pos : [max_atoms, 3]
    padded_Z   : [max_atoms]
    pad_mask   : [max_atoms]  — True for real atoms, False for padding
    """
    if max_atoms is None:
        max_atoms = bucket_size(n_atoms)

    pad_len = max_atoms - n_atoms
    assert pad_len >= 0, (
        f"n_atoms={n_atoms} exceeds max_atoms={max_atoms}"
    )

    padded_pos = jnp.concatenate([
        positions,
        jnp.zeros((pad_len, 3), dtype=positions.dtype),
    ], axis=0)

    padded_Z = jnp.concatenate([
        atomic_numbers,
        jnp.zeros(pad_len, dtype=atomic_numbers.dtype),
    ], axis=0)

    pad_mask = jnp.concatenate([
        jnp.ones(n_atoms, dtype=bool),
        jnp.zeros(pad_len, dtype=bool),
    ], axis=0)

    return padded_pos, padded_Z, pad_mask


def unpad_results(padded_array: jax.Array, n_atoms: int) -> jax.Array:
    """
    Strip padding rows from an output array.

    Works for any shape ``[max_atoms, *rest]``.
    """
    return padded_array[:n_atoms]
