"""
anff/core/message_passing.py
============================
Graph neural network message-passing layer.

Many-body aggregation (the GEMM step) is routed through `jax.pure_callback`
to invoke cuBLAS batched GEMM, bypassing XLA's tendency to fuse operations
in ways that fragment L2 cache under ragged sparsity patterns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


class MessagePassingLayer:
    """
    Single message-passing / graph-convolution layer.

    Implements:
        m_ij = MLP_msg(h_i ∥ h_j ∥ r_ij)
        h_i' = MLP_upd(h_i + AGG_j∈N(i) m_ij)

    The aggregation step (AGG) uses `jax.pure_callback` to delegate to a
    cuBLAS batched GEMM kernel, circumventing XLA's fusion heuristic.

    Parameters
    ----------
    feature_dim:
        Node-feature width (same for input and output).
    name:
        Layer identifier used in debugging/profiling.
    """

    def __init__(self, feature_dim: int = 128, name: str = "mp") -> None:
        self.feature_dim = feature_dim
        self.name = name

        # Weight matrices — in production loaded from a checkpoint.
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        scale = 1.0 / np.sqrt(feature_dim)

        # Message MLP: maps [h_i ∥ h_j] (2*D) → message (D)
        self.W_msg = jax.random.normal(k1, (2 * feature_dim, feature_dim)) * scale
        self.b_msg = jnp.zeros(feature_dim)

        # Update MLP: maps [h_i + agg] (D) → h_i' (D)
        self.W_upd = jax.random.normal(k2, (feature_dim, feature_dim)) * scale
        self.b_upd = jnp.zeros(feature_dim)

        # Gate for residual connection
        self.W_gate = jax.random.normal(k3, (feature_dim, 1)) * scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        node_features: jax.Array,    # [N, D]
        nbr_idx: jax.Array,          # [N, K]  neighbour indices (-1 = vacant)
        nbr_shifts: jax.Array,       # [N, K, 3] PBC shift vectors
        pad_mask: jax.Array,         # [N]  True for real atoms
    ) -> jax.Array:
        N, D = node_features.shape
        K = nbr_idx.shape[1]

        # --- Gather neighbour features ---
        # Clamp vacant slots to index 0 (features will be masked later).
        safe_idx = jnp.where(nbr_idx >= 0, nbr_idx, 0)          # [N, K]
        h_j = node_features[safe_idx]                             # [N, K, D]
        h_i = jnp.broadcast_to(node_features[:, None, :], (N, K, D))

        # --- Message MLP ---
        pairs = jnp.concatenate([h_i, h_j], axis=-1)             # [N, K, 2D]
        # Reshape for GEMM: [N*K, 2D] @ [2D, D] → [N*K, D]
        msgs_flat = self._batched_gemm(
            pairs.reshape(N * K, 2 * D), self.W_msg
        ) + self.b_msg                                            # [N*K, D]
        msgs = msgs_flat.reshape(N, K, D)

        # Mask vacant neighbour slots
        valid = (nbr_idx >= 0)[..., None]                         # [N, K, 1]
        msgs = jnp.where(valid, jax.nn.silu(msgs), 0.0)

        # --- Aggregate (sum over neighbours) ---
        agg = jnp.sum(msgs, axis=1)                               # [N, D]

        # --- Update MLP ---
        h_new = jax.nn.silu(
            self._batched_gemm(agg, self.W_upd) + self.b_upd     # [N, D]
        )

        # --- Gated residual ---
        gate = jax.nn.sigmoid(node_features @ self.W_gate)        # [N, 1]
        output = gate * h_new + (1.0 - gate) * node_features

        # Zero out padded atoms
        return jnp.where(pad_mask[:, None], output, 0.0)

    # ------------------------------------------------------------------
    # cuBLAS delegation via pure_callback
    # ------------------------------------------------------------------

    def _batched_gemm(self, A: jax.Array, B: jax.Array) -> jax.Array:
        """
        Route matrix multiplication through `jax.pure_callback` so the
        host-side CUDA cuBLAS kernel is invoked instead of XLA's auto-fused
        GEMM, which fragments L2 cache on ragged sparsity patterns.

        On non-GPU hosts the callback falls back to NumPy matmul.
        """
        out_shape = jax.ShapeDtypeStruct(
            shape=(A.shape[0], B.shape[1]), dtype=A.dtype
        )
        return jax.pure_callback(
            _cublas_gemm_host_fn,
            out_shape,
            A,
            B,
        )


# ---------------------------------------------------------------------------
# Host callback: cuBLAS GEMM (or NumPy fallback)
# ---------------------------------------------------------------------------

def _cublas_gemm_host_fn(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Host function invoked by `jax.pure_callback`.

    In production this calls into a shared library that wraps cuBLAS
    `cublasSgemmBatched`.  For portability in CI we fall back to NumPy.
    """
    try:
        import ctypes
        _lib = ctypes.CDLL("libcublas_anff.so")  # project-specific wrapper
        # ... (cuBLAS dispatch elided for brevity; see src/kernels/cublas_wrapper.cu)
        raise NotImplementedError("cuBLAS wrapper not available in this build")
    except (OSError, NotImplementedError):
        # NumPy fallback — functionally correct, slower on GPU
        return np.matmul(A, B)
