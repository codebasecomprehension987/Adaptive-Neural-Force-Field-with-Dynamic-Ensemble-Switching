# Adaptive Neural Force Field (ANFF)

A JAX-based molecular force-field framework with dynamic ensemble switching between neural-network potentials (NNP) and semi-empirical quantum methods (GFN2-xTB), accelerated by CUDA custom kernels and designed for distributed TPU training.

---

## Architecture Overview

```
AtomicSystem
     │
     ▼
EnsembleSwitcher          ← Python-level branch (avoids jax.lax.cond TPU overhead)
  ├── LevelOfTheory.NNP
  └── LevelOfTheory.GFN2
     │
     ▼
build_neighbor_list       ← XLA custom call → CUDA cell-linked list kernel
     │
     ▼
pad_to_max                ← Static shapes for XLA (bucketed to minimise waste)
     │
     ▼
MessagePassingLayer ×N    ← jax.checkpoint at boundary; cuBLAS GEMM via pure_callback
     │
     ▼
Energy readout + forces   ← jax.grad, replicated adjacency across TPU pod
```

---

## Key Design Decisions

### 1. Static-Shape Padding with Bucketing
JAX's XLA backend requires static tensor shapes at compile time. Naively padding every batch to the global maximum atom count causes quadratic memory waste (a 500-atom system padded for 2 000 atoms blows working memory by 16×). ANFF instead buckets systems by size (`[64, 128, 256, 512, 1024, 2048, 4096]` atoms), compiling one XLA module per bucket.

### 2. Python-Level Ensemble Switching
`jax.lax.cond` does **not** short-circuit on TPUs — both branches execute, paying full FLOPs for the discarded branch. ANFF selects the level of theory (NNP vs. GFN2-xTB) in plain Python *before* entering the JAX tracing boundary, so only the chosen branch is compiled.

### 3. Neighbour List via XLA Custom Calls
Neighbour-list construction is routed through `jax.pure_callback` (upgradeable to `jax.ffi.ffi_call`) wrapping CUDA cell-linked list kernels. This prevents XLA from fusing the sparse index gather with downstream pair-potential accumulation in ways that fragment L2 cache.

### 4. cuBLAS GEMM for Message Aggregation
The many-body message-passing aggregation step delegates to cuBLAS batched GEMM via `jax.pure_callback`, bypassing XLA's fusion heuristic which produces poor L2 locality under ragged neighbour sparsity.

### 5. Gradient Checkpointing + TPU Sharding
On a 64-TPU pod the sparse adjacency tensor (`[N, max_neighbors]`) is **replicated** rather than sharded (O(N·k·D) memory) because irregular neighbour indices cannot be evenly partitioned. Gradient checkpointing via `jax.checkpoint` at the message-passing boundary trades recomputation (O(N·k) per gradient step) for activation memory.

---

## Repository Structure

```
anff/
├── src/
│   ├── core/
│   │   ├── forcefield.py        # Top-level orchestrator
│   │   └── message_passing.py   # GNN layer with cuBLAS delegation
│   ├── ensemble/
│   │   └── switcher.py          # Dynamic LoT selection (Python-level)
│   ├── kernels/
│   │   ├── neighbor_list.py     # XLA custom call wrapper + NumPy fallback
│   │   └── cublas_wrapper.py    # cuBLAS batched GEMM host callback
│   └── utils/
│       ├── padding.py            # Bucketed padding / unpadding
│       └── sharding.py           # TPU mesh + replication helpers
├── tests/
│   ├── test_forcefield.py
│   ├── test_neighbor_list.py
│   ├── test_padding.py
│   ├── test_ensemble.py
│   └── test_sharding.py
├── benchmarks/
│   └── benchmark_padding_memory.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── docs/
│   ├── architecture.md
│   └── complexity_analysis.md
├── configs/
│   ├── default.yaml
│   └── tpu_pod.yaml
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Core (CPU / GPU)
pip install -e ".[dev]"

# With CUDA kernel support (requires NVCC ≥ 11.8)
pip install -e ".[cuda]"
cd src/kernels && make
```

### Requirements
- Python ≥ 3.10
- JAX ≥ 0.4.25 (with `jaxlib` matching your accelerator)
- optax ≥ 0.2.0
- PyYAML

---

## Quick Start

```python
import jax.numpy as jnp
from src.core.forcefield import AdaptiveNeuralForceField, AtomicSystem

ff = AdaptiveNeuralForceField(
    cutoff_radius=5.0,
    max_neighbors=64,
    n_mp_layers=3,
    feature_dim=128,
)

system = AtomicSystem(
    positions      = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
    atomic_numbers = jnp.array([1, 8]),
    cell           = jnp.zeros((3, 3)),
    n_atoms        = 2,
)

out = ff.compute(system)
print(f"Energy: {out.energy:.4f} eV")
print(f"Forces:\n{out.forces}")
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Benchmarks

```bash
python benchmarks/benchmark_padding_memory.py
```

Outputs a CSV with padding overhead factors and estimated peak memory across system sizes and device counts.

---

## Training

```bash
python scripts/train.py --config configs/default.yaml

# 64-TPU pod
python scripts/train.py --config configs/tpu_pod.yaml
```

---

## Complexity Summary

| Quantity | Cost |
|---|---|
| Adjacency replication (64 TPUs) | O(N · k · 64) memory |
| Rematerialisation alternative | O(N · k) per gradient step |
| Padding overhead (worst case) | ≤ 2× per bucket boundary |
| MP-layer recomputation (checkpointing) | 1 additional XLA compilation |

---

## License

MIT License. See `LICENSE` for details.
