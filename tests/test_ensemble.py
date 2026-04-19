"""
tests/test_ensemble.py
======================
Tests for EnsembleSwitcher and LevelOfTheory selection logic.
"""

import os
import pytest
import jax.numpy as jnp

from src.ensemble.switcher import EnsembleSwitcher, LevelOfTheory, SwitcherConfig
from src.core.forcefield import AtomicSystem


def make_system(n_atoms: int) -> AtomicSystem:
    return AtomicSystem(
        positions      = jnp.zeros((n_atoms, 3)),
        atomic_numbers = jnp.ones(n_atoms, dtype=jnp.int32),
        cell           = jnp.zeros((3, 3)),
        n_atoms        = n_atoms,
    )


# ---------------------------------------------------------------------------
# Size heuristic
# ---------------------------------------------------------------------------

class TestSizeHeuristic:
    def test_small_system_selects_nnp(self):
        sw  = EnsembleSwitcher()
        lot = sw.select(make_system(100))
        assert lot == LevelOfTheory.NNP

    def test_exactly_at_boundary_selects_nnp(self):
        sw  = EnsembleSwitcher(SwitcherConfig(nnp_max_atoms=200))
        lot = sw.select(make_system(200))
        assert lot == LevelOfTheory.NNP

    def test_one_above_boundary_selects_gfn2(self):
        sw  = EnsembleSwitcher(SwitcherConfig(nnp_max_atoms=200))
        lot = sw.select(make_system(201))
        assert lot == LevelOfTheory.GFN2

    def test_large_system_selects_gfn2(self):
        sw  = EnsembleSwitcher()
        lot = sw.select(make_system(1000))
        assert lot == LevelOfTheory.GFN2


# ---------------------------------------------------------------------------
# Uncertainty-triggered fallback
# ---------------------------------------------------------------------------

class TestUncertaintyFallback:
    def test_low_variance_keeps_nnp(self):
        sw  = EnsembleSwitcher(SwitcherConfig(uncertainty_threshold=0.2))
        sys = make_system(10)
        sw.update_uncertainty(sys, variance=0.05)
        lot = sw.select(sys)
        assert lot == LevelOfTheory.NNP

    def test_high_variance_triggers_gfn2(self):
        sw  = EnsembleSwitcher(SwitcherConfig(uncertainty_threshold=0.2))
        sys = make_system(10)
        sw.update_uncertainty(sys, variance=0.99)
        lot = sw.select(sys)
        assert lot == LevelOfTheory.GFN2

    def test_variance_exactly_at_threshold_keeps_nnp(self):
        sw  = EnsembleSwitcher(SwitcherConfig(uncertainty_threshold=0.2))
        sys = make_system(10)
        sw.update_uncertainty(sys, variance=0.2)
        # variance == threshold → NOT strictly greater → NNP
        lot = sw.select(sys)
        assert lot == LevelOfTheory.NNP


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------

class TestEnvOverride:
    def test_env_forces_gfn2(self, monkeypatch):
        monkeypatch.setenv("ANFF_LEVEL_OF_THEORY", "gfn2_xtb")
        sw  = EnsembleSwitcher(SwitcherConfig())   # re-reads env
        lot = sw.select(make_system(5))
        assert lot == LevelOfTheory.GFN2

    def test_env_forces_nnp(self, monkeypatch):
        monkeypatch.setenv("ANFF_LEVEL_OF_THEORY", "nnp")
        sw  = EnsembleSwitcher(SwitcherConfig())
        lot = sw.select(make_system(1000))
        assert lot == LevelOfTheory.NNP

    def test_invalid_env_raises(self, monkeypatch):
        monkeypatch.setenv("ANFF_LEVEL_OF_THEORY", "unicorn")
        with pytest.raises(ValueError, match="Unknown"):
            SwitcherConfig()


# ---------------------------------------------------------------------------
# History logging
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_is_recorded(self):
        sw  = EnsembleSwitcher()
        sys = make_system(10)
        for _ in range(5):
            sw.select(sys)
        assert len(sw.cfg.history) == 5

    def test_history_capped_at_1000(self):
        sw  = EnsembleSwitcher()
        sys = make_system(10)
        for _ in range(1200):
            sw.select(sys)
        assert len(sw.cfg.history) == 1000

    def test_report_sums_to_one(self):
        sw  = EnsembleSwitcher()
        sw.select(make_system(10))
        sw.select(make_system(1000))
        report = sw.report()
        total = sum(report.values())
        assert abs(total - 1.0) < 1e-6, f"Report fractions sum to {total}, expected 1.0"
