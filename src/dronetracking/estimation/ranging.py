"""Stage 1 of geometry estimation: raw two-way-ranging timestamps -> distances.

Each :class:`RangingRecord` carries the four event times of one symmetric
two-way ranging exchange. The classic estimator cancels the (unknown, identical)
clock *offset* of each device::

    ToF = 0.5 * ((t4 - t1) - (t3 - t2))
    distance = ToF * speed_of_sound

Offset cancels exactly in the two differences; a residual clock *skew* (drift)
leaves a small scale error, and timestamp jitter adds zero-mean noise. We collect
every round for an unordered device pair and collapse them robustly (median /
MAD outlier rejection, then mean of the survivors) into a single distance with a
variance, then assemble the symmetric :class:`DistanceMatrix`.

A cheap-but-real triangle-inequality screen flags edges that are geometrically
impossible given the rest of the array, so a single blown range cannot poison
the downstream multidimensional-scaling layout.

This module is part of the estimation package and therefore imports **nothing**
from :mod:`dronetracking.sim` (the ground-truth firewall). It consumes the
observation bundle structurally (``device_ids``, ``ranging``,
``speed_of_sound_mps``).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import median_abs_deviation

from ..datatypes import DistanceMatrix

# Variance floor (m^2). Caps the weight ``W = 1/var`` so a noise-free pair (whose
# empirical variance is ~0) yields a large-but-finite weight instead of dividing
# by zero. 1e-9 m^2 == a 1-sigma of ~30 micrometres, far below any real range error.
_VARIANCE_FLOOR_M2 = 1e-9

# Robust-sigma multiplier for outlier rejection within a pair's rounds.
_ROBUST_SIGMA_K = 3.0

# 1.4826 makes 1.4826*MAD a consistent estimator of sigma for Gaussian data.
_MAD_TO_SIGMA = 1.4826

# Absolute slack (m) added to the robust-sigma rejection window. Keeps the screen
# from discarding samples that differ only by floating-point round-off when the
# real spread is ~0 (noise-free data): a micron is far below any genuine range
# outlier but well above ULP-level jitter.
_REJECT_ABS_FLOOR_M = 1e-6


def _collapse_pair(distances: np.ndarray) -> Tuple[float, float, int]:
    """Robustly collapse one pair's per-round distance samples.

    Returns ``(distance, variance_of_mean, n_survivors)``. Uses a median/MAD
    screen (drop |x - median| > 3 * 1.4826*MAD), then the plain mean of the
    survivors and the variance *of that mean* (sample variance / n), floored so
    the weight stays finite.
    """
    x = np.asarray(distances, dtype=float)
    n = x.size
    if n == 0:
        return float("nan"), float("inf"), 0
    if n == 1:
        return float(x[0]), _VARIANCE_FLOOR_M2, 1

    med = float(np.median(x))
    mad = float(median_abs_deviation(x, scale=1.0))
    robust_sigma = _MAD_TO_SIGMA * mad

    # Window = a few robust sigma, plus an absolute floor so round-off-only spread
    # (MAD ~ 0 on noise-free data) never trips the screen and discards good rounds.
    window = _ROBUST_SIGMA_K * robust_sigma + _REJECT_ABS_FLOOR_M
    keep = np.abs(x - med) <= window
    survivors = x[keep]
    if survivors.size == 0:  # degenerate; fall back to all samples
        survivors = x

    m = survivors.size
    distance = float(np.mean(survivors))
    if m > 1:
        sample_var = float(np.var(survivors, ddof=1))
        var_of_mean = sample_var / m
    else:
        var_of_mean = _VARIANCE_FLOOR_M2
    variance = max(var_of_mean, _VARIANCE_FLOOR_M2)
    return distance, variance, m


def flag_triangle_outliers(
    D: np.ndarray, valid: np.ndarray, slack_m: float
) -> np.ndarray:
    """Triangle-inequality screen: drop edges that are geometrically impossible.

    For every triple ``(i, j, k)`` whose three edges are all present and valid,
    the triangle inequality requires ``D[i,j] <= D[i,k] + D[k,j]`` (and cyclic).
    A violation beyond ``slack_m`` (noise headroom) implicates the *longest* edge
    of the triple — the most likely over-estimate. We accumulate, per edge, the
    total violation magnitude it is implicated in, then invalidate edges whose
    score is a gross outlier relative to the rest.

    ``D`` is the (K, K) symmetric distance matrix; ``valid`` is the current
    boolean validity mask (diagonal False). Returns a *new* validity mask with
    flagged edges set False (symmetrically).
    """
    D = np.asarray(D, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    K = D.shape[0]
    out = valid.copy()

    score = np.zeros((K, K), dtype=float)
    n_violations = 0
    for i in range(K):
        for j in range(i + 1, K):
            if not valid[i, j]:
                continue
            for k in range(K):
                if k == i or k == j:
                    continue
                if not (valid[i, k] and valid[j, k]):
                    continue
                a, b, c = D[i, j], D[i, k], D[j, k]
                # Worst overshoot of any one side beyond the sum of the other two.
                overshoot = max(a - (b + c), b - (a + c), c - (a + b))
                if overshoot > slack_m:
                    n_violations += 1
                    # Blame the longest edge of this triple.
                    edges = [(a, i, j), (b, i, k), (c, j, k)]
                    _, p, q = max(edges, key=lambda e: e[0])
                    score[p, q] += overshoot
                    score[q, p] += overshoot

    if n_violations == 0:
        return out

    # An edge is an outlier if its accumulated violation stands out from the bulk
    # of edges. We take the cut over *all* upper-triangle scores (most of which are
    # zero in a healthy array), so a lone gross offender — whose score swamps the
    # rest — is caught even when it is the only violator (median/MAD of just the
    # offenders would put the threshold at the offender itself). The slack floor
    # keeps borderline, noise-sized violations from being flagged.
    upper = score[np.triu_indices(K, 1)]
    med = float(np.median(upper))
    mad = float(median_abs_deviation(upper, scale=1.0))
    thresh = max(med + _ROBUST_SIGMA_K * _MAD_TO_SIGMA * mad, slack_m)

    for i in range(K):
        for j in range(i + 1, K):
            if score[i, j] > thresh:
                out[i, j] = False
                out[j, i] = False
    return out


def build_distance_matrix(observations) -> DistanceMatrix:
    """Estimate the pairwise device :class:`DistanceMatrix` from ranging records.

    ``observations`` must expose ``device_ids`` (ordered), ``ranging`` (an
    iterable of two-way-ranging records with ``initiator``/``responder`` and
    ``t1_local_i``/``t2_local_j``/``t3_local_j``/``t4_local_i``), and
    ``speed_of_sound_mps``.
    """
    device_ids: Tuple[str, ...] = tuple(observations.device_ids)
    K = len(device_ids)
    index = {dev: i for i, dev in enumerate(device_ids)}
    c = float(observations.speed_of_sound_mps)

    # Gather per-round distance samples per unordered pair (i < j by matrix index).
    samples: Dict[Tuple[int, int], List[float]] = {}
    for rec in observations.ranging:
        i = index[rec.initiator]
        j = index[rec.responder]
        tof = 0.5 * (
            (rec.t4_local_i - rec.t1_local_i) - (rec.t3_local_j - rec.t2_local_j)
        )
        dist = tof * c
        key = (i, j) if i < j else (j, i)
        samples.setdefault(key, []).append(dist)

    D = np.full((K, K), np.nan, dtype=float)
    np.fill_diagonal(D, 0.0)
    W = np.zeros((K, K), dtype=float)
    counts = np.zeros((K, K), dtype=int)
    valid = np.zeros((K, K), dtype=bool)

    var_for_slack: List[float] = []
    for (i, j), vals in samples.items():
        distance, variance, n = _collapse_pair(np.asarray(vals))
        if n == 0 or not np.isfinite(distance):
            continue
        D[i, j] = D[j, i] = distance
        w = 1.0 / variance if variance > 0 else 0.0
        W[i, j] = W[j, i] = w
        counts[i, j] = counts[j, i] = n
        valid[i, j] = valid[j, i] = True
        var_for_slack.append(variance)

    # Triangle-inequality slack: a few sigma of the typical per-edge range error
    # (sqrt of the median variance-of-mean, scaled up to a single-measurement
    # spread), with a small absolute floor so the noise-free case isn't brittle.
    if var_for_slack:
        typical_sigma = float(np.sqrt(np.median(var_for_slack)))
    else:
        typical_sigma = 0.0
    slack_m = max(10.0 * typical_sigma, 0.5)
    valid = flag_triangle_outliers(D, valid, slack_m=slack_m)

    # Zero the weight of anything the triangle screen rejected.
    W = np.where(valid, W, 0.0)

    return DistanceMatrix(
        device_ids=device_ids,
        D=D,
        W=W,
        counts=counts,
        valid=valid,
    )
