"""Stage 2 of geometry estimation: a distance matrix -> a relative layout.

We only know inter-device *distances*, so the recovered constellation is fixed
only up to a rigid motion plus reflection (distances cannot see chirality). That
gauge freedom is expected and is handled by callers that align the layout to
truth/GPS before scoring.

Pipeline:

1. **Classical MDS init.** Double-centre the squared-distance matrix
   ``B = -0.5 * J @ D2 @ J`` (``J = I - 11ᵀ/K``); its top three eigenpairs give a
   3-D embedding ``X0``. Missing/invalid distances are first bootstrapped with a
   shortest-path estimate over the valid-edge graph so the eigen-decomposition is
   well-posed.
2. **Weighted nonlinear refine.** ``scipy.optimize.least_squares`` minimises, per
   *valid* edge, ``(||p_i - p_j|| - d_ij) * sqrt(W_ij)`` with a robust
   ``soft_l1`` loss and the trust-region (``trf``) method.
3. **Gauge fix.** Subtract the centroid to pin the translation.
4. **Covariance.** Per-device 3x3 blocks from :func:`transforms.gn_covariance`
   on the refined Jacobian, scaled by the residual variance.

Imports nothing from :mod:`dronetracking.sim` (ground-truth firewall).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.csgraph import shortest_path

from ..datatypes import DistanceMatrix
from ..estimation.interfaces import RelativeLayout
from ..transforms import gn_covariance

# Floor for the unit-weight residual variance (reduced chi-square) when a fit is
# numerically perfect (cost == 0, e.g. noise-free data). Matches the ranging
# variance floor scale so a noise-free layout reports a ~mm-scale, finite
# covariance rather than an exactly-zero (degenerate) one.
_S2_FLOOR = 1e-9


def _bootstrap_distances(dm: DistanceMatrix) -> np.ndarray:
    """Fill missing/invalid distances with a shortest-path estimate.

    Builds an undirected graph from the valid edges and runs Dijkstra; any
    unmeasured pair is approximated by its shortest valid-edge path. This is only
    to seed the MDS embedding — the refinement step ignores these synthetic
    edges (their weight is zero) and trusts the measured ones.
    """
    K = dm.n_devices
    D = dm.D.copy()

    graph = np.where(dm.valid, dm.D, 0.0)
    graph = np.nan_to_num(graph, nan=0.0)  # csgraph treats 0 as "no edge"
    sp = shortest_path(graph, method="D", directed=False)

    missing = ~dm.valid & ~np.eye(K, dtype=bool)
    fill = sp.copy()
    # Where even the graph is disconnected (inf), fall back to a large but finite
    # span so the eigen-decomposition stays numerically sane.
    finite_max = np.max(sp[np.isfinite(sp)]) if np.any(np.isfinite(sp)) else 1.0
    fill[~np.isfinite(fill)] = finite_max if finite_max > 0 else 1.0
    D[missing] = fill[missing]
    np.fill_diagonal(D, 0.0)
    return D


def _classical_mds(D: np.ndarray) -> np.ndarray:
    """Classical (Torgerson) MDS to a 3-D embedding ``X0`` (K, 3)."""
    K = D.shape[0]
    D2 = D**2
    J = np.eye(K) - np.ones((K, K)) / K
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)  # symmetrise against round-off

    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    order = np.argsort(eigvals)[::-1][:3]  # top 3
    top_vals = np.clip(eigvals[order], 0.0, None)  # clamp negatives to 0
    top_vecs = eigvecs[:, order]
    X0 = top_vecs * np.sqrt(top_vals)[np.newaxis, :]
    return X0


def estimate_layout(dm: DistanceMatrix) -> RelativeLayout:
    """Recover a 3-D relative device layout from a :class:`DistanceMatrix`."""
    device_ids = tuple(dm.device_ids)
    K = len(device_ids)

    # --- init -------------------------------------------------------------- #
    D_seed = _bootstrap_distances(dm)
    X0 = _classical_mds(D_seed)

    # Valid measured edges drive the refinement (upper triangle, symmetric).
    iu, ju = np.triu_indices(K, 1)
    edge_mask = dm.valid[iu, ju]
    ei, ej = iu[edge_mask], ju[edge_mask]
    d_obs = dm.D[ei, ej]
    sqrt_w = np.sqrt(dm.W[ei, ej])
    n_edges = ei.size

    def residuals(x: np.ndarray) -> np.ndarray:
        P = x.reshape(K, 3)
        diff = P[ei] - P[ej]
        dist = np.linalg.norm(diff, axis=1)
        return (dist - d_obs) * sqrt_w

    # --- refine ------------------------------------------------------------ #
    if n_edges >= 3:
        res = least_squares(
            residuals,
            X0.ravel(),
            loss="soft_l1",
            method="trf",
        )
        P = res.x.reshape(K, 3)
        jac = res.jac
        cost = res.cost
    else:
        # Too few edges to refine; keep the MDS init and synthesise a trivial
        # (residual-free) problem so covariance falls out gauge-safe.
        P = X0
        jac = np.zeros((max(n_edges, 1), 3 * K))
        cost = 0.0

    # --- gauge fix: centre the cloud -------------------------------------- #
    P = P - P.mean(axis=0, keepdims=True)

    # --- covariance -------------------------------------------------------- #
    # Reduced chi-square as the unit-weight residual variance. dof = (#valid
    # edges) - (#free params), where the 3K coordinates carry a 6-DOF rigid gauge
    # (3 translation + 3 rotation) that distance data cannot observe. On a
    # perfect (noise-free) fit cost -> 0, so we floor s2 at a tiny value so the
    # reported covariance is appropriately small rather than zero/degenerate.
    dof = max(n_edges - (3 * K - 6), 1)
    s2 = 2.0 * cost / max(dof, 1)
    if not np.isfinite(s2) or s2 <= 0.0:
        s2 = _S2_FLOOR

    # The weighted Jacobian carries the absolute weight scale (W = 1/var can be
    # ~1e9 when variances hit the floor), which makes Jᵀ J badly conditioned for
    # the pseudoinverse. gn_covariance is exactly invariant to a uniform rescale
    # (cov = s2 * pinv(Jᵀ J), and (J/c)ᵀ(J/c) scales by 1/c²), so we factor out an
    # RMS reference scale and undo it via s2 — identical answer, well-conditioned.
    jac = np.asarray(jac, dtype=float)
    j_scale = float(np.sqrt(np.mean(jac**2)))
    if not (j_scale > 0.0 and np.isfinite(j_scale)):
        j_scale = 1.0

    # A distance-only layout is inherently rank-deficient (the 6-DOF gauge),
    # which is exactly what gn_covariance's pseudoinverse is built to absorb
    # (gauge directions zeroed, not blown up). numpy's pinv still emits a benign
    # divide/overflow RuntimeWarning while inverting the zero singular values
    # internally; the returned covariance is finite and correct, so we silence
    # that one expected warning rather than leak noise to the caller.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cov_full = gn_covariance(jac / j_scale, s2 / j_scale**2)  # (3K, 3K)

    covariances = np.empty((K, 3, 3), dtype=float)
    for k in range(K):
        sl = slice(3 * k, 3 * k + 3)
        covariances[k] = cov_full[sl, sl]

    return RelativeLayout(
        device_ids=device_ids,
        positions_local=P,
        covariances=covariances,
    )
