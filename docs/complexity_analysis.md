# Complexity Analysis

## Notation

| Symbol | Meaning |
|---|---|
| N | Atom count (real, before padding) |
| N' | Padded atom count (next bucket boundary) |
| k | Max neighbours per atom (`max_neighbors`) |
| D | Device count |
| F | Feature dimension |
| L | Number of message-passing layers |

---

## Time Complexity

### Neighbour-list construction
- Cell-linked list: **O(N)** cell assignment + **O(N·k)** pair search
- Padding does not increase the pair-search cost because vacant slots are masked.

### Single message-passing layer
| Step | Cost |
|---|---|
| Gather h_j | O(N'·k·F) |
| Message MLP (GEMM) | O(N'·k·2F·F) = O(N'·k·F²) |
| Aggregation (sum) | O(N'·k·F) |
| Update MLP (GEMM) | O(N'·F²) |

Total per layer: **O(N'·k·F²)**. Over L layers: **O(L·N'·k·F²)**.

### Gradient computation
Without checkpointing: O(L·N'·k·F²) forward + O(L·N'·k·F²) backward = **2× forward cost**.

With `jax.checkpoint` at the MP boundary: the rematerialised segment is recomputed once during the backward pass, giving the same 2× asymptotic cost but halving peak activation memory.

---

## Space Complexity

### Activation memory (single device, without checkpointing)

| Tensor | Shape | Size |
|---|---|---|
| Node features (all layers) | [L+1, N', F] | O(L·N'·F) |
| Pair messages | [N', k, F] | O(N'·k·F) |
| Adjacency (idx + shifts) | [N', k, 4] | O(N'·k) |

Peak: **O(L·N'·F + N'·k·F)**.

### With gradient checkpointing

Only one layer's activations are stored at a time; the MP stack is rematerialised:

Peak: **O(N'·F + N'·k·F)** — independent of L.

---

## Padding Overhead

For a system of N real atoms padded to N' = bucket(N):

- Memory overhead factor: N'/N
- Worst case: N = bucket\_boundary + 1 → N' = 2N → 2× overhead
- Spec example: N=500, N'=2000 (global max, no bucketing) → 16× overhead
- With bucketing: N=500 → N'=512 → 1.024× overhead ✓

---

## Replication vs. Rematerialisation (64-TPU Pod)

The sparse adjacency tensor cannot be sharded across the atom axis (irregular neighbour counts prevent even partitioning). Two options:

### Option A — Replicate

Replicate `[N', k]` int32 index tensor and `[N', k, 3]` float32 shift tensor on all D devices.

```
Memory per device = N' × k × (4 + 12) bytes  [idx + shifts]
```

For N'=2048, k=64, D=64:

```
2048 × 64 × 16 B = 2.1 MB per device   (adjacency only)
```

Total across pod: 2.1 MB × 64 = **134 MB** (negligible vs. HBM capacity).

### Option B — Rematerialise

Do not store the adjacency tensor in the backward pass; recompute it from positions.

```
Cost per gradient step = O(N'·k) recomputation = neighbour-list rebuild
```

For N'=2048, k=64: one additional O(N) cell-linked list pass per gradient step.

### Decision

ANFF defaults to **replication** (Option A). The 2.1 MB per device is negligible and the cell-linked list rebuild cost (though O(N)) has a large constant due to atomic operations in the CUDA kernel.

---

## XLA Compilation Budget

| Event | XLA modules compiled |
|---|---|
| First call per (bucket_size, LoT) | 1 |
| `jax.checkpoint` rematerialised segment | +1 |
| Different bucket size | +1 per new bucket |
| Different LoT | +1 per LoT |

For 7 bucket sizes × 2 LoTs = **up to 14 XLA modules** + 14 rematerialised segments = 28 total. Each compilation takes ~10–60 s on a TPU pod; the cache is persistent across Python sessions via `jax.config.jax_compilation_cache_dir`.
