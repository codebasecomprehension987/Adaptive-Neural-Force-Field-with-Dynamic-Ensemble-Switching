"""
tests/test_padding.py
=====================
Tests for padding / unpadding helpers and bucket logic.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.utils.padding import pad_to_max, unpad_results, bucket_size, ATOM_COUNT_BUCKETS


class TestBucketSize:
    def test_exact_bucket(self):
        for b in ATOM_COUNT_BUCKETS:
            assert bucket_size(b) == b

    def test_one_above_bucket(self):
        b0 = ATOM_COUNT_BUCKETS[0]
        b1 = ATOM_COUNT_BUCKETS[1]
        assert bucket_size(b0 + 1) == b1

    def test_power_of_two_beyond_table(self):
        result = bucket_size(5000)
        assert result >= 5000
        # Must be a power of two
        assert (result & (result - 1)) == 0


class TestPadToMax:
    def test_shapes(self):
        pos = jnp.ones((10, 3))
        Z   = jnp.ones(10, dtype=jnp.int32)
        p_pos, p_Z, mask = pad_to_max(pos, Z, n_atoms=10, max_atoms=16)
        assert p_pos.shape == (16, 3)
        assert p_Z.shape   == (16,)
        assert mask.shape  == (16,)

    def test_mask_values(self):
        pos = jnp.ones((5, 3))
        Z   = jnp.ones(5, dtype=jnp.int32)
        _, _, mask = pad_to_max(pos, Z, n_atoms=5, max_atoms=8)
        assert jnp.all(mask[:5])
        assert not jnp.any(mask[5:])

    def test_padding_rows_are_zero(self):
        pos = jnp.ones((4, 3))
        Z   = jnp.ones(4, dtype=jnp.int32)
        p_pos, p_Z, _ = pad_to_max(pos, Z, n_atoms=4, max_atoms=8)
        np.testing.assert_array_equal(p_pos[4:], 0.0)
        np.testing.assert_array_equal(p_Z[4:],   0)

    def test_no_padding_needed(self):
        pos = jnp.ones((8, 3))
        Z   = jnp.ones(8, dtype=jnp.int32)
        p_pos, p_Z, mask = pad_to_max(pos, Z, n_atoms=8, max_atoms=8)
        assert jnp.all(mask)

    def test_raises_if_n_atoms_exceeds_max(self):
        pos = jnp.ones((10, 3))
        Z   = jnp.ones(10, dtype=jnp.int32)
        with pytest.raises(AssertionError):
            pad_to_max(pos, Z, n_atoms=10, max_atoms=8)


class TestUnpadResults:
    def test_strips_padding(self):
        arr = jnp.arange(20).reshape(10, 2)
        out = unpad_results(arr, n_atoms=4)
        assert out.shape == (4, 2)
        np.testing.assert_array_equal(out, arr[:4])
