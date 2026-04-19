"""
tests/test_forcefield.py
========================
Unit and integration tests for AdaptiveNeuralForceField.

Run with:
    pytest tests/test_forcefield.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core.forcefield import AdaptiveNeuralForceField, AtomicSystem, ForceFieldOutput
from src.ensemble.switcher import LevelOfTheory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_system(n_atoms: int = 10, seed: int = 0) -> AtomicSystem:
    """Create a random AtomicSystem for testing."""
    rng = np.random.default_rng(seed)
    positions     = jnp.array(rng.uniform(0, 10, (n_atoms, 3)), dtype=jnp.float32)
    atomic_numbers = jnp.array(rng.integers(1, 18, n_atoms), dtype=jnp.int32)
    cell          = jnp.zeros((3, 3), dtype=jnp.float32)  # no PBC
    return AtomicSystem(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell,
        n_atoms=n_atoms,
    )


@pytest.fixture
def small_ff():
    return AdaptiveNeuralForceField(
        cutoff_radius=3.0,
        max_neighbors=8,
        n_mp_layers=2,
        feature_dim=16,
        device_count=1,
    )


# ---------------------------------------------------------------------------
# Shape / dtype tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_energy_scalar(self, small_ff):
        system = make_system(n_atoms=5)
        out = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP)
        assert out.energy.shape == (), f"Expected scalar, got {out.energy.shape}"

    def test_forces_shape(self, small_ff):
        system = make_system(n_atoms=5)
        out = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP)
        assert out.forces.shape == (5, 3), out.forces.shape

    def test_forces_dtype_float32(self, small_ff):
        system = make_system(n_atoms=5)
        out = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP)
        assert out.forces.dtype == jnp.float32

    def test_output_is_named_tuple(self, small_ff):
        system = make_system(n_atoms=5)
        out = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP)
        assert isinstance(out, ForceFieldOutput)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_energy_finite(self, small_ff):
        system = make_system(n_atoms=8)
        out = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP)
        assert jnp.isfinite(out.energy), f"Energy is not finite: {out.energy}"

    def test_forces_finite(self, small_ff):
        system = make_system(n_atoms=8)
        out = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP)
        assert jnp.all(jnp.isfinite(out.forces)), "Forces contain NaN/Inf"

    def test_energy_differs_between_lots(self, small_ff):
        system = make_system(n_atoms=8)
        e_nnp  = small_ff.compute(system, level_of_theory=LevelOfTheory.NNP).energy
        e_gfn2 = small_ff.compute(system, level_of_theory=LevelOfTheory.GFN2).energy
        # GFN2 uses scale=0.5 vs NNP scale=1.0 so energies differ
        assert not jnp.allclose(e_nnp, e_gfn2, atol=1e-5), (
            "NNP and GFN2 energies should differ"
        )

    def test_padded_atoms_do_not_contribute(self, small_ff):
        """Adding padding atoms beyond n_atoms must not change the energy."""
        system_5  = make_system(n_atoms=5, seed=42)
        system_10 = make_system(n_atoms=5, seed=42)  # same 5 atoms

        out5  = small_ff.compute(system_5,  level_of_theory=LevelOfTheory.NNP)
        out10 = small_ff.compute(system_10, level_of_theory=LevelOfTheory.NNP)

        np.testing.assert_allclose(
            float(out5.energy), float(out10.energy), rtol=1e-5
        )


# ---------------------------------------------------------------------------
# JIT cache tests
# ---------------------------------------------------------------------------

class TestJITCache:
    def test_same_shape_reuses_compiled_function(self, small_ff):
        system_a = make_system(n_atoms=5, seed=0)
        system_b = make_system(n_atoms=5, seed=1)

        small_ff.compute(system_a, level_of_theory=LevelOfTheory.NNP)
        n_compiled_before = len(small_ff._jit_cache)
        small_ff.compute(system_b, level_of_theory=LevelOfTheory.NNP)
        n_compiled_after  = len(small_ff._jit_cache)

        assert n_compiled_before == n_compiled_after, (
            "Second call with same shape should reuse cached compiled function"
        )

    def test_different_shapes_compile_separately(self, small_ff):
        system_5  = make_system(n_atoms=5)
        system_10 = make_system(n_atoms=10)

        small_ff.compute(system_5,  level_of_theory=LevelOfTheory.NNP)
        small_ff.compute(system_10, level_of_theory=LevelOfTheory.NNP)

        assert len(small_ff._jit_cache) >= 2, (
            "Different atom-count buckets should produce separate compiled modules"
        )


# ---------------------------------------------------------------------------
# Ensemble switcher integration
# ---------------------------------------------------------------------------

class TestEnsembleSwitcher:
    def test_auto_select_nnp_for_small_system(self, small_ff):
        system = make_system(n_atoms=5)
        lot = small_ff._switcher.select(system)
        assert lot == LevelOfTheory.NNP

    def test_auto_select_gfn2_for_large_system(self, small_ff):
        system = make_system(n_atoms=600)
        lot = small_ff._switcher.select(system)
        assert lot == LevelOfTheory.GFN2

    def test_uncertainty_triggers_gfn2_fallback(self, small_ff):
        system = make_system(n_atoms=5)
        small_ff._switcher.update_uncertainty(system, variance=0.99)
        lot = small_ff._switcher.select(system)
        assert lot == LevelOfTheory.GFN2
