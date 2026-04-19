# ANFF Architecture

## Overview

The Adaptive Neural Force Field (ANFF) is structured in four layers:

```
User code
  └── AdaptiveNeuralForceField.compute()       [src/core/forcefield.py]
        ├── EnsembleSwitcher.select()           [src/ensemble/switcher.py]
        ├── build_neighbor_list()               [src/kernels/neighbor_list.py]
        ├── pad_to_max()                        [src/utils/padding.py]
        └── MessagePassingLayer × N             [src/core/message_passing.py]
              └── _batched_gemm()  →  cublas_wrapper.py
```

---

## Static-Shape Constraint

JAX's XLA backend traces a computation graph from Python code once per unique set of argument shapes, then compiles that graph to machine code. This means **all tensor shapes must be known at trace time** — you cannot have a loop over "however many atoms this molecule has."

ANFF resolves this with **bucketed padding**:

| Real atom count | Padded to |
|---|---|
| 1 – 64 | 64 |
| 65 – 128 | 128 |
| 129 – 256 | 256 |
| 257 – 512 | 512 |
| 513 – 1 024 | 1 024 |
| 1 025 – 2 048 | 2 048 |
| > 2 048 | next power of two |

A boolean mask `pad_mask[i]` is `True` for real atoms and `False` for padding; all energy contributions from padded rows are zeroed before summation.

---

## Ensemble Switching

`jax.lax.cond(pred, true_fn, false_fn, *args)` evaluates **both** branches on TPUs (XLA compiles a single kernel containing both). For two force-field levels of theory with vastly different FLOPs this is unacceptable.

ANFF's solution: selection is a Python `if` **before** JAX tracing. `EnsembleSwitcher.select()` returns a `LevelOfTheory` enum; `ForceField._get_or_compile()` uses it as a dict key into a JIT-compiled function cache. Each LoT therefore has its own compiled XLA module, and only the selected module runs.

---

## Neighbour-List Construction

The cell-linked list algorithm partitions space into cubic cells of side ≥ cutoff radius, then only checks atom pairs in the same or adjacent cells (27 cells in 3-D). This reduces pairwise search from O(N²) to O(N·k) where k is the average neighbour count.

The CUDA implementation (`src/kernels/cell_list.cu`, not shown — compiled separately) is exposed to JAX via `jax.pure_callback`. In a production deployment this is replaced by a `jax.ffi.ffi_call` custom op so that the HLO graph contains a single opaque node rather than a Python callback.

---

## Message Passing and L2 Cache

Standard XLA GEMM fusion is beneficial for dense, regular computations. Under ragged sparsity (interaction cutoffs produce different neighbour counts per atom) XLA's fusion can cause L2 cache thrashing because the fused kernel touches multiple large sparse tensors simultaneously.

ANFF routes the aggregation GEMM through `jax.pure_callback` → `cublas_wrapper.py` → cuBLAS `cublasSgemmBatched`. cuBLAS uses a separately tuned GEMM kernel that respects L2 cache blocking parameters, resulting in better throughput for the [N·K, 2D] × [2D, D] matmul at the heart of the message MLP.

---

## Gradient Flow

```
Loss
  │
  └── jax.grad
        │
        ├── readout MLP       (no checkpoint — cheap)
        │
        ├── [CHECKPOINT BOUNDARY — jax.checkpoint]
        │
        ├── MessagePassingLayer × N
        │     └── rematerialised during backward pass
        │
        └── embedding         (no checkpoint — cheap)
```

The checkpoint boundary introduces a **second XLA compilation** for the rematerialised segment (as noted in the spec). This is unavoidable: XLA cannot share compiled code between the forward and backward segments when `jax.checkpoint` changes the gradient accumulation strategy.

---

## TPU Sharding

On a 64-device pod, `jax.sharding.NamedSharding` with `PartitionSpec()` (empty spec = replicate) places a full copy of both model parameters and the adjacency tensor on every device. Gradients are reduced via `jax.lax.pmean` across the mesh before the optimiser update.

The alternative — sharding the adjacency tensor along the atom axis — would require `jax.lax.all_gather` during the backward pass, incurring O(N·k·D) communication each gradient step. Replication trades memory for communication, which is the right trade-off when D=64 and the adjacency tensor fits comfortably in HBM.
