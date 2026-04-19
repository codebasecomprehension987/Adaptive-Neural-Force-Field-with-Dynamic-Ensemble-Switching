"""
anff/core/forcefield.py
=======================
Adaptive Neural Force Field — top-level orchestrator.

Design choices
--------------
* Atom counts vary per batch.  We pad ALL arrays to `max_atoms` once here so
  downstream JAX kernels see static shapes and can be compiled with
  `jax.jit(static_argnums=...)`.
* Gradient checkpointing is applied at the message-passing boundary via
  `jax.checkpoint`; see `_mp_layer_checkpoint_policy`.
* The "ensemble switcher" decides which level-of-theory to run; because
  `jax.lax.cond` executes both branches on TPUs, we gate ensemble selection
  **before** `jit` dispatch so only the chosen branch is compiled/run.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .message_passing import MessagePassingLayer
from ..ensemble.switcher import EnsembleSwitcher, LevelOfTheory
from ..kernels.neighbor_list import build_neighbor_list
from ..utils.padding import pad_to_max, unpad_results
from ..utils.sharding import make_mesh, shard_params


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class AtomicSystem(NamedTuple):
    """Minimal representation of a molecular system passed to the force field."""
    positions: jax.Array          # [N, 3]  Cartesian coords, Ångström
    atomic_numbers: jax.Array     # [N]     integer element indices
    cell: jax.Array               # [3, 3]  unit-cell matrix (Å); zeros → no PBC
    n_atoms: int                  # actual atom count before padding


class ForceFieldOutput(NamedTuple):
    energy: jax.Array             # []  scalar, eV
    forces: jax.Array             # [N, 3]  eV / Å
    stress: Optional[jax.Array]   # [3, 3] or None


# ---------------------------------------------------------------------------
# Checkpoint policy
# ---------------------------------------------------------------------------

def _mp_checkpoint_policy(prim, *_, **__):
    """
    Rematerialise activations at the MP-layer boundary to avoid storing all
    intermediate feature tensors across the backward pass.

    Returning True means: do NOT save this value in the forward pass; recompute
    it during the backward pass instead.
    """
    # Rematerialise everything except cheap element-wise ops.
    saveable = {
        "integer_pow", "mul", "add", "sub", "neg", "convert_element_type",
    }
    return prim.name not in saveable


# ---------------------------------------------------------------------------
# Core force field class
# ---------------------------------------------------------------------------

class AdaptiveNeuralForceField:
    """
    Adaptive Neural Force Field with dynamic ensemble switching.

    Parameters
    ----------
    cutoff_radius:
        Neighbour-list cutoff in Ångström.
    max_neighbors:
        Maximum neighbours per atom.  Determines the static shape of the
        sparse adjacency tensor ``[N_atoms, max_neighbors]``.
    n_mp_layers:
        Number of message-passing iterations.
    feature_dim:
        Width of the node-feature vectors.
    device_count:
        Number of TPU/GPU devices for data-parallel sharding.  If > 1 the
        sparse adjacency tensor is *replicated* across all devices; see the
        complexity note in the project spec.
    """

    def __init__(
        self,
        cutoff_radius: float = 5.0,
        max_neighbors: int = 64,
        n_mp_layers: int = 3,
        feature_dim: int = 128,
        device_count: int = 1,
    ) -> None:
        self.cutoff_radius = cutoff_radius
        self.max_neighbors = max_neighbors
        self.n_mp_layers = n_mp_layers
        self.feature_dim = feature_dim
        self.device_count = device_count

        # Message-passing stack
        self._mp_layers = [
            MessagePassingLayer(feature_dim=feature_dim, name=f"mp_{i}")
            for i in range(n_mp_layers)
        ]

        # Ensemble switcher — picks GFN2-xTB vs. NNP at runtime
        self._switcher = EnsembleSwitcher()

        # Sharding mesh (no-op when device_count == 1)
        self._mesh = make_mesh(device_count)

        # JIT-compiled energy+force computation, keyed by (max_atoms, theory)
        self._jit_cache: dict[tuple, any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        system: AtomicSystem,
        *,
        level_of_theory: Optional[LevelOfTheory] = None,
    ) -> ForceFieldOutput:
        """
        Compute energy, forces (and optionally stress) for *system*.

        Parameters
        ----------
        system:
            Molecular system to evaluate.
        level_of_theory:
            Override the automatic ensemble selector.  Useful for benchmarking.
        """
        lot = level_of_theory or self._switcher.select(system)

        # 1. Build neighbour list on the device (XLA custom call → CUDA kernel)
        nbr_idx, nbr_shifts, n_pairs = build_neighbor_list(
            positions=system.positions,
            cell=system.cell,
            cutoff=self.cutoff_radius,
            max_neighbors=self.max_neighbors,
        )

        # 2. Pad positions to a static max_atoms shape so XLA can compile once.
        #    Memory cost: O(max_atoms × feature_dim) — acceptable.
        padded_pos, padded_Z, pad_mask = pad_to_max(
            system.positions, system.atomic_numbers, system.n_atoms
        )

        # 3. Dispatch to the correct compiled kernel.
        #    We avoid jax.lax.cond here because on TPU both branches execute.
        compute_fn = self._get_or_compile(lot, padded_pos.shape[0])
        energy, forces_padded = compute_fn(
            padded_pos, padded_Z, nbr_idx, nbr_shifts, pad_mask
        )

        # 4. Strip padding from forces.
        forces = unpad_results(forces_padded, system.n_atoms)

        return ForceFieldOutput(energy=energy, forces=forces, stress=None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_compile(self, lot: LevelOfTheory, max_atoms: int):
        """Return a JIT-compiled compute function, compiling on first call."""
        key = (lot, max_atoms)
        if key not in self._jit_cache:
            self._jit_cache[key] = jax.jit(
                functools.partial(self._energy_and_forces, lot=lot)
            )
        return self._jit_cache[key]

    def _energy_and_forces(
        self,
        positions: jax.Array,
        atomic_numbers: jax.Array,
        nbr_idx: jax.Array,
        nbr_shifts: jax.Array,
        pad_mask: jax.Array,
        *,
        lot: LevelOfTheory,
    ):
        """
        Forward pass: embed → message-pass → readout.

        Gradient checkpointing is applied at the MP boundary; the rematerialised
        segment is compiled as a separate XLA module (second compilation artefact
        noted in the project spec).
        """
        # --- Embedding ---
        node_features = self._embed(atomic_numbers)  # [max_atoms, feature_dim]

        # --- Message-passing with checkpointing ---
        def mp_forward(node_feats):
            for layer in self._mp_layers:
                node_feats = layer(node_feats, nbr_idx, nbr_shifts, pad_mask)
            return node_feats

        node_features = jax.checkpoint(mp_forward, policy=_mp_checkpoint_policy)(
            node_features
        )

        # --- Energy readout ---
        per_atom_energy = self._readout(node_features, pad_mask, lot=lot)
        energy = jnp.sum(per_atom_energy)

        # --- Forces via reverse-mode AD ---
        forces = -jax.grad(lambda p: jnp.sum(
            self._readout(
                jax.checkpoint(mp_forward, policy=_mp_checkpoint_policy)(
                    self._embed(atomic_numbers)
                ),
                pad_mask,
                lot=lot,
            )
        ))(positions)

        return energy, forces

    # ------------------------------------------------------------------
    # Sub-network stubs (replaced by trained weights in production)
    # ------------------------------------------------------------------

    def _embed(self, atomic_numbers: jax.Array) -> jax.Array:
        """One-hot embed atomic numbers into feature vectors."""
        one_hot = jax.nn.one_hot(atomic_numbers, num_classes=self.feature_dim)
        return one_hot  # [max_atoms, feature_dim]

    def _readout(
        self,
        node_features: jax.Array,
        pad_mask: jax.Array,
        *,
        lot: LevelOfTheory,
    ) -> jax.Array:
        """
        Per-atom energy readout MLP stub.

        In production this is replaced by a trained MLP whose weights depend on
        `lot` (different heads for GFN2-xTB vs. NNP).
        """
        scale = 1.0 if lot == LevelOfTheory.NNP else 0.5
        raw = jnp.sum(node_features, axis=-1) * scale
        return jnp.where(pad_mask, raw, 0.0)
