"""
anff/utils/sharding.py
======================
Device mesh construction and parameter-sharding helpers for multi-TPU pods.

Replication vs. sharding (from spec)
--------------------------------------
The sparse adjacency tensor (shape [N_atoms, max_neighbors]) must be
*replicated* rather than sharded because neighbour indices are irregular and
cannot be evenly partitioned along the atom axis without costly communication.

Memory cost: O(N · k · D) where
  N = atom count
  k = max_neighbors
  D = device count

Alternative — rematerialise during backward pass (O(N · k) per gradient step)
— is controlled by the `rematerialise_graph` flag on `ForceField`.

64-TPU pod example
------------------
With N=2 000, k=64, D=64, fp32:
  Replicated adjacency:  2000 × 64 × 64 × 4 B ≈ 32 MB per device
  Rematerialise:         2000 × 64 × 4 B ≈ 0.5 MB per gradient step
"""

from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def make_mesh(device_count: int, axis_name: str = "devices") -> Mesh | None:
    """
    Construct a 1-D device mesh for data-parallel training.

    Returns ``None`` when ``device_count == 1`` (no sharding needed).
    """
    if device_count <= 1:
        return None

    devices = jax.devices()[:device_count]
    if len(devices) < device_count:
        raise RuntimeError(
            f"Requested {device_count} devices but only {len(devices)} available."
        )
    return Mesh(devices, axis_names=(axis_name,))


def shard_params(params: dict, mesh: Mesh | None) -> dict:
    """
    Shard model parameters across the device mesh using `NamedSharding`.

    Parameters are replicated on all devices (PartitionSpec with no axis
    names), which is correct for the weight tensors.  The adjacency tensor
    is also replicated; see module docstring.
    """
    if mesh is None:
        return params

    replicated = NamedSharding(mesh, P())  # () = replicate on all devices

    def _replicate(leaf):
        return jax.device_put(leaf, replicated)

    return jax.tree_util.tree_map(_replicate, params)


def replicate_adjacency(
    nbr_idx: jax.Array,
    nbr_shifts: jax.Array,
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array]:
    """
    Explicitly replicate the neighbour-index tensor across all devices.

    This is the design choice described in the spec: we accept O(N·k·D)
    memory overhead rather than paying the XLA communication cost of
    re-gathering irregular neighbour indices during the backward pass.
    """
    if mesh is None:
        return nbr_idx, nbr_shifts

    replicated = NamedSharding(mesh, P())
    nbr_idx    = jax.device_put(nbr_idx,    replicated)
    nbr_shifts = jax.device_put(nbr_shifts, replicated)
    return nbr_idx, nbr_shifts


def gradient_checkpoint_tradeoff(
    n_atoms: int,
    max_neighbors: int,
    device_count: int,
    dtype_bytes: int = 4,
) -> dict:
    """
    Compute memory/FLOP trade-off statistics for replicate vs. rematerialise.

    Returns a plain dict for logging / monitoring.
    """
    replicated_mb = (n_atoms * max_neighbors * device_count * dtype_bytes) / 1e6
    remat_mb_per_step = (n_atoms * max_neighbors * dtype_bytes) / 1e6

    return {
        "replicated_adjacency_mb":      replicated_mb,
        "rematerialise_per_grad_step_mb": remat_mb_per_step,
        "replication_overhead_factor":  replicated_mb / max(remat_mb_per_step, 1e-9),
        "n_atoms":                      n_atoms,
        "max_neighbors":                max_neighbors,
        "device_count":                 device_count,
    }
