"""
scripts/evaluate.py
===================
Evaluation script: run the force field over a dataset and report
energy MAE, force MAE, and ensemble selection statistics.

Usage:
    python scripts/evaluate.py --config configs/default.yaml [--checkpoint path/to/ckpt]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.forcefield import AdaptiveNeuralForceField, AtomicSystem
from src.ensemble.switcher import LevelOfTheory


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dummy_eval_dataset(n_systems: int = 32, seed: int = 99):
    """Synthetic dataset with known 'ground truth' energies for evaluation."""
    rng = np.random.default_rng(seed)
    systems, gt_energies, gt_forces = [], [], []
    for _ in range(n_systems):
        n = rng.integers(5, 50)
        systems.append(AtomicSystem(
            positions      = jnp.array(rng.uniform(0, 10, (n, 3)), dtype=jnp.float32),
            atomic_numbers = jnp.array(rng.integers(1, 10, n),     dtype=jnp.int32),
            cell           = jnp.zeros((3, 3),                      dtype=jnp.float32),
            n_atoms        = int(n),
        ))
        gt_energies.append(rng.normal())
        gt_forces.append(rng.normal(size=(n, 3)).astype(np.float32))
    return systems, gt_energies, gt_forces


def evaluate(config: dict, checkpoint: str | None = None) -> dict:
    ff = AdaptiveNeuralForceField(
        cutoff_radius = config.get("cutoff_radius", 5.0),
        max_neighbors = config.get("max_neighbors", 32),
        n_mp_layers   = config.get("n_mp_layers",   3),
        feature_dim   = config.get("feature_dim",   64),
        device_count  = config.get("device_count",  1),
    )

    systems, gt_energies, gt_forces = dummy_eval_dataset(
        n_systems=config.get("n_eval", 32)
    )

    energy_errors, force_errors, timings = [], [], []
    lot_counts = {lot: 0 for lot in LevelOfTheory}

    for sys_, e_gt, f_gt in zip(systems, gt_energies, gt_forces):
        t0  = time.perf_counter()
        out = ff.compute(sys_)
        dt  = time.perf_counter() - t0

        energy_errors.append(abs(float(out.energy) - e_gt))
        force_mae = float(jnp.mean(jnp.abs(out.forces - jnp.array(f_gt))))
        force_errors.append(force_mae)
        timings.append(dt)

        lot = ff._switcher.select(sys_)
        lot_counts[lot] += 1

    results = {
        "n_systems":       len(systems),
        "energy_mae_ev":   float(np.mean(energy_errors)),
        "force_mae_ev_ang":float(np.mean(force_errors)),
        "mean_time_s":     float(np.mean(timings)),
        "lot_report":      ff._switcher.report(),
        "lot_counts":      {k.value: v for k, v in lot_counts.items()},
    }

    _print_results(results)
    _save_results(results)
    return results


def _print_results(r: dict) -> None:
    print("\n" + "=" * 52)
    print("  ANFF Evaluation Results")
    print("=" * 52)
    print(f"  Systems evaluated   : {r['n_systems']}")
    print(f"  Energy MAE          : {r['energy_mae_ev']:.4f} eV")
    print(f"  Force MAE           : {r['force_mae_ev_ang']:.4f} eV/Å")
    print(f"  Mean wall time/sys  : {r['mean_time_s']*1000:.2f} ms")
    print(f"  LoT fractions       : {r['lot_report']}")
    print("=" * 52 + "\n")


def _save_results(r: dict, path: str = "eval_results.csv") -> None:
    flat = {
        "n_systems":        r["n_systems"],
        "energy_mae_ev":    r["energy_mae_ev"],
        "force_mae_ev_ang": r["force_mae_ev_ang"],
        "mean_time_s":      r["mean_time_s"],
    }
    flat.update({f"lot_{k}": v for k, v in r["lot_counts"].items()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        writer.writeheader()
        writer.writerow(flat)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ANFF")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (optional)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config   = load_config(cfg_path) if cfg_path.exists() else {}
    evaluate(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
