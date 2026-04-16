"""
anff/ensemble/switcher.py
=========================
Dynamic ensemble switching between levels of theory (LoT).

Problem
-------
`jax.lax.cond` does NOT short-circuit on TPUs: both branches execute,
paying full FLOPs for the discarded branch.  We therefore gate the LoT
selection **outside** JAX's tracing boundary so that only the selected
branch is ever traced and compiled.

Strategy
--------
1.  `EnsembleSwitcher.select()` is a plain Python method (no JAX involvement).
2.  The caller uses the returned `LevelOfTheory` token to look up a
    pre-compiled `jit`-ted function from the JIT cache in `ForceField`.
3.  This means each LoT has its own compiled XLA module; switching LoT is a
    Python-level branch, not a JAX-level branch.

Switching criteria
------------------
* Default: NNP for systems ≤ 500 atoms; GFN2-xTB for larger systems where
  the NNP message-passing cost dominates.
* Override via ``ANFF_LEVEL_OF_THEORY`` environment variable.
* Confidence score from the NNP uncertainty head can trigger a GFN2-xTB
  fallback at runtime (see `_uncertainty_triggered_fallback`).
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.forcefield import AtomicSystem


class LevelOfTheory(enum.Enum):
    """Supported levels of theory."""
    NNP     = "nnp"       # Neural network potential (fast, approximate)
    GFN2    = "gfn2_xtb"  # GFN2-xTB semi-empirical (slower, more accurate)
    HYBRID  = "hybrid"    # Energy-weighted mixture of NNP and GFN2


@dataclass
class SwitcherConfig:
    """Hyper-parameters controlling automatic LoT selection."""
    nnp_max_atoms: int = 500
    """Use NNP for systems with at most this many atoms."""

    uncertainty_threshold: float = 0.15
    """NNP ensemble variance (eV) above which GFN2 is invoked as a fallback."""

    force_lot: str | None = None
    """Hard override; set via ``ANFF_LEVEL_OF_THEORY`` env var."""

    history: list[LevelOfTheory] = field(default_factory=list)
    """Rolling log of selections (last 1 000 entries)."""

    def __post_init__(self):
        env = os.environ.get("ANFF_LEVEL_OF_THEORY", "").strip().lower()
        if env:
            try:
                self.force_lot = LevelOfTheory(env)
            except ValueError:
                valid = [e.value for e in LevelOfTheory]
                raise ValueError(
                    f"Unknown ANFF_LEVEL_OF_THEORY='{env}'.  Valid values: {valid}"
                )


class EnsembleSwitcher:
    """
    Selects a `LevelOfTheory` for each `AtomicSystem` without involving JAX.

    This is a plain Python object — it deliberately lives outside the JAX
    transformation stack so that LoT dispatch is a Python ``if``, not a
    ``jax.lax.cond``.
    """

    def __init__(self, config: SwitcherConfig | None = None) -> None:
        self.cfg = config or SwitcherConfig()
        self._nnp_variance_cache: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def select(self, system: "AtomicSystem") -> LevelOfTheory:
        """
        Choose a level of theory for *system*.

        Priority
        --------
        1. Hard override (env var / config).
        2. Uncertainty-triggered GFN2 fallback.
        3. Size-based heuristic.
        """
        # 1. Hard override
        if self.cfg.force_lot is not None:
            lot = self.cfg.force_lot  # type: ignore[assignment]
            self._log(lot)
            return lot

        # 2. Uncertainty fallback
        lot = self._uncertainty_triggered_fallback(system) or \
              self._size_heuristic(system)

        self._log(lot)
        return lot

    def report(self) -> dict:
        """Return a summary of recent LoT selections for monitoring."""
        from collections import Counter
        counts = Counter(self.cfg.history)
        total  = len(self.cfg.history) or 1
        return {lot.value: counts[lot] / total for lot in LevelOfTheory}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _size_heuristic(self, system: "AtomicSystem") -> LevelOfTheory:
        if system.n_atoms <= self.cfg.nnp_max_atoms:
            return LevelOfTheory.NNP
        return LevelOfTheory.GFN2

    def _uncertainty_triggered_fallback(
        self, system: "AtomicSystem"
    ) -> LevelOfTheory | None:
        """
        If a cached NNP uncertainty estimate for this system exceeds the
        threshold, fall back to GFN2-xTB.

        The variance is cached by a hash of the atomic numbers vector;
        it is populated by `ForceField._update_uncertainty_cache()` after
        each NNP evaluation.
        """
        key = id(system.atomic_numbers)
        variance = self._nnp_variance_cache.get(key, 0.0)
        if variance > self.cfg.uncertainty_threshold:
            return LevelOfTheory.GFN2
        return None

    def update_uncertainty(self, system: "AtomicSystem", variance: float) -> None:
        """Called by ForceField after each NNP run to update the cache."""
        key = id(system.atomic_numbers)
        self._nnp_variance_cache[key] = variance

    def _log(self, lot: LevelOfTheory) -> None:
        self.cfg.history.append(lot)
        if len(self.cfg.history) > 1_000:
            self.cfg.history = self.cfg.history[-1_000:]
