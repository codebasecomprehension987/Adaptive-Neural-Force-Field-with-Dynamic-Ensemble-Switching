"""
benchmarks/benchmark_padding_memory.py
=======================================
Empirical measurement of the memory overhead introduced by static padding.

Usage:
    python benchmarks/benchmark_padding_memory.py

Outputs a CSV table: system_size, padded_size, padding_overhead_factor,
peak_memory_mb (estimated).
"""

import csv
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.padding import bucket_size, pad_to_max
from src.utils.sharding import gradient_checkpoint_tradeoff

SYSTEM_SIZES   = [50, 100, 200, 500, 750, 1000, 1500, 2000]
MAX_NEIGHBORS  = 64
DEVICE_COUNTS  = [1, 8, 64]
FEATURE_DIM    = 128
DTYPE_BYTES    = 4  # float32


def estimate_peak_memory_mb(n_atoms: int, max_atoms: int, max_neighbors: int,
                             feature_dim: int, device_count: int) -> float:
    """Rough peak-memory model (one device)."""
    # Node features
    node_feat_mb = max_atoms * feature_dim * DTYPE_BYTES / 1e6
    # Adjacency tensor replicated
    adj_mb       = max_atoms * max_neighbors * DTYPE_BYTES / 1e6
    # Pair messages (N, K, D)
    msg_mb       = max_atoms * max_neighbors * feature_dim * DTYPE_BYTES / 1e6
    return node_feat_mb + adj_mb + msg_mb


def main():
    rows = []
    print(f"{'N_real':>8} {'N_padded':>10} {'Overhead':>10} "
          f"{'Peak_MB (D=1)':>16} {'Peak_MB (D=64)':>16} {'Adj repl D=64 MB':>18}")
    print("-" * 84)

    for n in SYSTEM_SIZES:
        padded  = bucket_size(n)
        overhead = padded / n

        peak_1   = estimate_peak_memory_mb(n, padded, MAX_NEIGHBORS, FEATURE_DIM, 1)
        peak_64  = estimate_peak_memory_mb(n, padded, MAX_NEIGHBORS, FEATURE_DIM, 64)

        tradeoff = gradient_checkpoint_tradeoff(padded, MAX_NEIGHBORS, 64)
        adj_repl = tradeoff["replicated_adjacency_mb"]

        print(f"{n:>8} {padded:>10} {overhead:>9.2f}x "
              f"{peak_1:>16.1f} {peak_64:>16.1f} {adj_repl:>18.1f}")

        rows.append({
            "n_real": n,
            "n_padded": padded,
            "padding_overhead": overhead,
            "peak_mb_d1": peak_1,
            "peak_mb_d64": peak_64,
            "adj_replicated_mb_d64": adj_repl,
        })

    out = Path(__file__).parent / "padding_memory_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
