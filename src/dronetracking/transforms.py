"""Geometry helpers shared across estimation and evaluation.

- :func:`umeyama` — least-squares similarity (rotation + translation + optional scale)
  aligning one point cloud onto another. Reflection is forbidden by default (the
  determinant correction keeps the rotation proper), which is correct for
  georeferencing onto real GPS anchors. ``allow_reflection=True`` is used when
  scoring a relative layout against truth, because pure distance geometry cannot
  observe chirality.
- :func:`gdop` — geometric dilution of precision for a sensor constellation.
- :func:`gn_covariance` — Gauss-Newton parameter covariance from a residual
  Jacobian, using the pseudoinverse so gauge/null directions stay finite.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Similarity(NamedTuple):
    """A similarity transform ``y = scale * (R @ x) + t``."""

    R: np.ndarray  # (D, D) orthogonal
    t: np.ndarray  # (D,)
    scale: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        P = np.atleast_2d(pts)
        out = self.scale * (P @ self.R.T) + self.t
        return out[0] if single else out

    @property
    def is_reflection(self) -> bool:
        return bool(np.linalg.det(self.R) < 0)


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scaling: bool = True,
    allow_reflection: bool = False,
) -> Similarity:
    """Least-squares similarity transform mapping ``src`` onto ``dst`` (Umeyama 1991).

    ``src`` and ``dst`` are ``(N, D)`` arrays of corresponding points.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n, d = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n  # (D, D) cross-covariance
    U, S, Vt = np.linalg.svd(cov)

    # Correction matrix: identity, unless we must flip the last axis to keep R proper.
    D_corr = np.eye(d)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        D_corr[-1, -1] = -1.0

    R = U @ D_corr @ Vt

    if with_scaling:
        var_src = (src_c**2).sum() / n
        scale = float(np.trace(np.diag(S) @ D_corr) / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return Similarity(R=R, t=t, scale=scale)


def gdop(target: np.ndarray, sensors: np.ndarray) -> float:
    """Position geometric dilution of precision for ``sensors`` observing ``target``.

    Built from unit line-of-sight vectors; large when sensors are poorly spread
    (e.g. collinear/coplanar relative to the target).
    """
    target = np.asarray(target, dtype=float)
    sensors = np.asarray(sensors, dtype=float)
    los = sensors - target
    norms = np.linalg.norm(los, axis=1, keepdims=True)
    G = los / norms  # (M, D) unit vectors
    try:
        cov = np.linalg.inv(G.T @ G)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(np.trace(cov)))


def gn_covariance(jac: np.ndarray, residual_variance: float = 1.0) -> np.ndarray:
    """Gauss-Newton parameter covariance ``var * pinv(JᵀJ)``.

    The pseudoinverse keeps rank-deficient (gauge) directions finite — they are
    zeroed rather than blowing up to infinity.
    """
    jac = np.asarray(jac, dtype=float)
    return residual_variance * np.linalg.pinv(jac.T @ jac)
