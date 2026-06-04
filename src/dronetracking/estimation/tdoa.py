"""TDOA target localization: acoustic arrival times -> per-emission position fix.

One acoustic emission is heard at several devices, each stamping the arrival into its
own local clock. :class:`~dronetracking.estimation.interfaces.ClockEstimates` maps every
arrival onto a common reference timebase, and the *unknown emission time cancels* when we
form arrival differences — so the geometry is a classic hyperbolic (range-difference)
multilateration.

Per emission we:

1. Lift each arrival to the reference timebase ``tau_i = clocks.to_reference(id, toa)``.
2. Pick the earliest-arriving device as the TDOA reference, giving range differences
   ``d_i = c * (tau_i - tau_ref) = ||x - p_i|| - ||x - p_ref||``.
3. Solve a closed-form linear seed (Chan / spherical-interpolation: square the
   range-difference identity and subtract the reference equation to get a system that is
   linear in ``(x, R_ref)``), via :func:`numpy.linalg.lstsq`.
4. Refine ``x`` with :func:`scipy.optimize.least_squares` (``loss="soft_l1"``) on the
   per-device residual ``(modeled_range_diff - measured) / sigma``.
5. Attach covariance (:func:`~dronetracking.transforms.gn_covariance`, inverse Fisher),
   GDOP, and the residual RMS.

This module is part of the estimation package and therefore MUST NOT import
``dronetracking.sim`` (the ground-truth firewall).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.optimize import least_squares

from .. import transforms
from ..datatypes import TargetFix
from .interfaces import ClockEstimates, RelativeLayout

# A 3D position fix needs four independent arrivals (three range differences + the
# reference range / depth). Fewer cannot pin x, y and z.
MIN_DEVICES_3D = 4


def _gather(arrivals, clocks: ClockEstimates, layout: RelativeLayout):
    """Map arrivals to (reference-timebase tau, sensor position) arrays."""
    taus = []
    positions = []
    for arr in arrivals:
        taus.append(float(clocks.to_reference(arr.device_id, arr.toa_local_s)))
        positions.append(np.asarray(layout.position_of(arr.device_id), dtype=float))
    return np.asarray(taus, dtype=float), np.asarray(positions, dtype=float)


def _linear_seed(positions: np.ndarray, d: np.ndarray, ref: int) -> np.ndarray:
    """Closed-form Chan-style seed for the source position.

    With reference sensor ``ref`` (range ``R_ref = ||x - p_ref||``) and range differences
    ``d_i = ||x - p_i|| - R_ref``, squaring ``||x - p_i|| = R_ref + d_i`` and subtracting
    the reference's ``||x - p_ref||^2`` gives, for every non-reference sensor ``i``::

        2 (p_i - p_ref)^T x + 2 d_i R_ref = ||p_i||^2 - ||p_ref||^2 - d_i^2

    a linear system in the 4-vector ``[x; R_ref]`` solved by least squares.
    """
    p_ref = positions[ref]
    rows = []
    rhs = []
    n = positions.shape[0]
    for i in range(n):
        if i == ref:
            continue
        p_i = positions[i]
        rows.append(np.concatenate([2.0 * (p_i - p_ref), [2.0 * d[i]]]))
        rhs.append(float(p_i @ p_i - p_ref @ p_ref - d[i] ** 2))
    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3]


def localize_emission(
    arrivals: Sequence,
    clocks: ClockEstimates,
    layout: RelativeLayout,
    speed_of_sound_mps: float,
    toa_var_s2: float = 1e-8,
) -> TargetFix:
    """Localize the source of ONE acoustic emission from its multi-device arrivals.

    Parameters
    ----------
    arrivals
        ``AcousticArrival``-like objects (``.device_id``, ``.toa_local_s``) for a single
        emission — i.e. all sharing one ``emission_idx``.
    clocks
        Clock estimates used to lift each local arrival onto the reference timebase.
    layout
        Relative device layout providing each device's position.
    speed_of_sound_mps
        Propagation speed ``c``.
    toa_var_s2
        Per-arrival time-of-arrival variance; the range-difference measurement variance is
        ``2 * c^2 * toa_var_s2`` (two independent ToAs differenced), used to weight the
        residuals and scale the reported covariance.

    Returns
    -------
    TargetFix
        Position, covariance, GDOP, residual RMS, device count and emission timestamp.
    """
    arrivals = list(arrivals)
    if len(arrivals) < MIN_DEVICES_3D:
        raise ValueError(
            f"need >= {MIN_DEVICES_3D} devices for a 3D TDOA fix, got {len(arrivals)}"
        )

    c = float(speed_of_sound_mps)
    taus, positions = _gather(arrivals, clocks, layout)

    # Reference = earliest arrival (shortest path -> best-conditioned differences).
    ref = int(np.argmin(taus))

    # Measured range differences relative to the reference sensor.
    d = c * (taus - taus[ref])  # d[ref] == 0

    # Range-difference measurement sigma: differencing two ToAs of variance toa_var_s2.
    sigma = float(np.sqrt(max(2.0 * c * c * toa_var_s2, np.finfo(float).tiny)))
    meas_var = sigma * sigma

    # --- closed-form linear seed -----------------------------------------------------
    x0 = _linear_seed(positions, d, ref)

    # --- nonlinear refine -------------------------------------------------------------
    non_ref = [i for i in range(len(arrivals)) if i != ref]
    p_ref = positions[ref]
    d_non_ref = d[non_ref]
    p_non_ref = positions[non_ref]

    def residuals(x):
        r_ref = np.linalg.norm(x - p_ref)
        r_i = np.linalg.norm(p_non_ref - x, axis=1)
        return (r_i - r_ref - d_non_ref) / sigma

    res = least_squares(residuals, x0, loss="soft_l1", method="trf")
    x = res.x

    # --- diagnostics ------------------------------------------------------------------
    # Covariance: inverse Fisher from the (sigma-normalized) Jacobian. Because the
    # residuals are already sigma-normalized, pinv(JᵀJ) is m² for unit-variance
    # residuals; we scale by the EMPIRICAL variance of the normalized residuals
    # (s2_hat), which self-calibrates for an under/over-estimated sigma and for clock/
    # layout error the assumed sigma never saw. A floor keeps it from collapsing to an
    # over-confident zero on a (near) perfect fit. This is what makes NEES honest.
    dof = max(res.fun.size - 3, 1)
    s2_hat = max(float(np.sum(res.fun ** 2) / dof), 1.0)
    cov = transforms.gn_covariance(res.jac, s2_hat)
    gdop = transforms.gdop(x, positions)
    final = residuals(x) * sigma  # back to meters
    residual_rms = float(np.sqrt(np.mean(final ** 2))) if final.size else 0.0
    t_mean = float(np.mean(taus))

    return TargetFix(
        position=np.asarray(x, dtype=float),
        cov=np.asarray(cov, dtype=float),
        gdop=float(gdop),
        residual_rms=residual_rms,
        n_devices=len(arrivals),
        t=t_mean,
    )


def localize_all(
    observations,
    clocks: ClockEstimates,
    layout: RelativeLayout,
    toa_var_s2: float = 1e-8,
) -> List[TargetFix]:
    """Localize every emission with enough devices, sorted by emission time.

    Groups ``observations.acoustic`` by ``emission_idx``; each group with at least
    :data:`MIN_DEVICES_3D` devices yields one :class:`TargetFix`. Groups with fewer
    devices are skipped. The returned list is sorted by fix timestamp ``t``.
    """
    c = float(observations.speed_of_sound_mps)

    groups = {}
    for arr in observations.acoustic:
        groups.setdefault(arr.emission_idx, []).append(arr)

    fixes: List[TargetFix] = []
    for emission_idx in sorted(groups):
        group = groups[emission_idx]
        if len(group) < MIN_DEVICES_3D:
            continue
        fixes.append(localize_emission(group, clocks, layout, c, toa_var_s2=toa_var_s2))

    fixes.sort(key=lambda f: f.t)
    return fixes
