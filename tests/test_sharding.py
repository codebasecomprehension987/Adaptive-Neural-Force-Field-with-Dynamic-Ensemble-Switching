"""
tests/test_sharding.py
======================
Tests for sharding helpers and the memory trade-off diagnostic.

Note: `make_mesh` and `replicate_adjacency` call `jax.devices()`.
In CI with a single CPU device, `device_count > 1` raises RuntimeError;
those tests are marked `skip` unless multiple devices are present.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.utils.sharding import (
    make_mesh,
    shard_params,
    replicate_adjacency,
    gradient_checkpoint_tradeoff,
)


N_DEVICES = len(jax.devices())


# ---------------------------------------------------------------------------
# make_mesh
# ---------------------------------------------------------------------------

class TestMakeMesh:
    def test_single_device_returns_none(self):
        mesh = make_mesh(device_count=1)
        assert mesh is None

    @pytest.mark.skipif(N_DEVICES < 2, reason="Requires ≥ 2 devices")
    def test_multi_device_returns_mesh(self):
        mesh = make_mesh(device_count=2)
        assert mesh is not None

    def test_more_devices_than_available_raises(self):
        with pytest.raises(RuntimeError, match="available"):
            make_mesh(device_count=N_DEVICES + 1)


# ---------------------------------------------------------------------------
# shard_params
# ---------------------------------------------------------------------------

class TestShardParams:
    def test_none_mesh_returns_params_unchanged(self):
        params = {"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}
        out    = shard_params(params, mesh=None)
        np.testing.assert_array_equal(out["w"], params["w"])
        np.testing.assert_array_equal(out["b"], params["b"])

    def test_nested_params_are_traversed(self):
        params = {"layer_0": {"w": jnp.ones((8, 8))}, "layer_1": {"b": jnp.zeros(8)}}
        out    = shard_params(params, mesh=None)
        assert out["layer_0"]["w"].shape == (8, 8)


# ---------------------------------------------------------------------------
# replicate_adjacency
# ---------------------------------------------------------------------------

class TestReplicateAdjacency:
    def test_none_mesh_returns_arrays_unchanged(self):
        idx    = jnp.full((10, 16), -1, dtype=jnp.int32)
        shifts = jnp.zeros((10, 16, 3))
        r_idx, r_shifts = replicate_adjacency(idx, shifts, mesh=None)
        np.testing.assert_array_equal(r_idx,    idx)
        np.testing.assert_array_equal(r_shifts, shifts)


# ---------------------------------------------------------------------------
# gradient_checkpoint_tradeoff
# ---------------------------------------------------------------------------

class TestGradientCheckpointTradeoff:
    def test_returns_expected_keys(self):
        result = gradient_checkpoint_tradeoff(
            n_atoms=500, max_neighbors=64, device_count=64
        )
        required_keys = {
            "replicated_adjacency_mb",
            "rematerialise_per_grad_step_mb",
            "replication_overhead_factor",
            "n_atoms",
            "max_neighbors",
            "device_count",
        }
        assert required_keys.issubset(result.keys())

    def test_replicated_is_device_count_times_remat(self):
        D = 8
        result = gradient_checkpoint_tradeoff(
            n_atoms=200, max_neighbors=32, device_count=D
        )
        ratio = result["replicated_adjacency_mb"] / result["rematerialise_per_grad_step_mb"]
        assert abs(ratio - D) < 1e-5, f"Expected ratio={D}, got {ratio}"

    def test_larger_system_costs_more(self):
        small = gradient_checkpoint_tradeoff(100,  32, 1)
        large = gradient_checkpoint_tradeoff(1000, 32, 1)
        assert large["replicated_adjacency_mb"] > small["replicated_adjacency_mb"]

    def test_more_devices_cost_more(self):
        d1  = gradient_checkpoint_tradeoff(200, 64, device_count=1)
        d64 = gradient_checkpoint_tradeoff(200, 64, device_count=64)
        assert d64["replicated_adjacency_mb"] > d1["replicated_adjacency_mb"]
