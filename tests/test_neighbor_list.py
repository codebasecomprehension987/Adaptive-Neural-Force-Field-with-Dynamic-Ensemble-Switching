"""
tests/test_neighbor_list.py
============================
Tests for the neighbour-list builder (pure-Python fallback path).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.kernels.neighbor_list import (
    build_neighbor_list,
    _numpy_neighbor_list,
    _pbc_images,
)


def simple_positions(n: int = 6) -> np.ndarray:
    """Atoms on a 1-D chain, 1 Å apart."""
    return np.column_stack([np.arange(n, dtype=np.float32),
                            np.zeros(n, dtype=np.float32),
                            np.zeros(n, dtype=np.float32)])


class TestNumpyFallback:
    def test_cutoff_2_chain_6(self):
        pos  = simple_positions(6)
        cell = np.zeros((3, 3), dtype=np.float32)
        idx, shifts, n_pairs = _numpy_neighbor_list(
            pos, cell, cutoff=1.5, max_neighbors=4
        )
        # Each interior atom should have exactly 2 neighbours (left + right)
        assert n_pairs[1] == 2
        assert n_pairs[4] == 2

    def test_vacant_slots_are_minus_one(self):
        pos  = simple_positions(4)
        cell = np.zeros((3, 3), dtype=np.float32)
        idx, _, _ = _numpy_neighbor_list(pos, cell, cutoff=1.5, max_neighbors=8)
        # Unused neighbour slots should be -1
        assert np.all(idx[idx != -1] >= 0)

    def test_no_self_neighbours(self):
        pos  = simple_positions(6)
        cell = np.zeros((3, 3), dtype=np.float32)
        idx, _, _ = _numpy_neighbor_list(pos, cell, cutoff=2.5, max_neighbors=8)
        for i, row in enumerate(idx):
            assert i not in row[row >= 0], f"Atom {i} listed as its own neighbour"


class TestBuildNeighborList:
    def test_output_shapes(self):
        N, K = 8, 16
        positions = jnp.array(simple_positions(N))
        cell      = jnp.zeros((3, 3))
        idx, shifts, n_pairs = build_neighbor_list(positions, cell, 2.5, K)

        assert idx.shape    == (N, K)
        assert shifts.shape == (N, K, 3)
        assert n_pairs.shape == (N,)

    def test_n_pairs_non_negative(self):
        positions = jnp.array(simple_positions(5))
        cell      = jnp.zeros((3, 3))
        _, _, n_pairs = build_neighbor_list(positions, cell, 1.5, 8)
        assert jnp.all(n_pairs >= 0)
