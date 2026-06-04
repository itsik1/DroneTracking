"""Target tracking: a stream of per-emission :class:`TargetFix` -> a smoothed :class:`Track`.

A single linear Kalman filter over a 9-element, constant-acceleration-capable state::

    [x, y, z, vx, vy, vz, ax, ay, az]

The transition is block-diagonal across the three spatial axes; each axis uses the
constant-acceleration kinematic block. ``model="cv"`` (the default) zeros the acceleration
process — the filter then runs constant-velocity with white-noise acceleration entering
through velocity. ``model="ca"`` lets acceleration evolve as a random walk. Process noise is
scaled by ``sigma_a`` (the continuous white-noise acceleration spectral density, m/s^2).

Each measurement is a position with ``R = fix.cov``. A normalized-innovation-squared (NIS)
chi-square gate rejects updates whose innovation is implausibly large, so a single bad fix
cannot drag the track.

``Tracker`` is the seam where multi-target association will eventually live: today it
forwards a single fused fix per timestep to one filter, asserting there is exactly one.

This module is part of the estimation package and therefore MUST NOT import
``dronetracking.sim`` (the ground-truth firewall).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import chi2

from ..datatypes import TargetFix
from .interfaces import Track

# NIS gate: reject an update whose normalized innovation squared exceeds the 99th
# percentile of a chi-square with 3 dof (position measurement dimension).
_NIS_DOF = 3
_NIS_GATE = float(chi2.ppf(0.99, df=_NIS_DOF))

# State layout: position (0:3), velocity (3:6), acceleration (6:9).
_N_STATE = 9
_POS = slice(0, 3)
_VEL = slice(3, 6)


def _transition(dt: float, model: str) -> np.ndarray:
    """Block-diagonal kinematic state-transition matrix for one step ``dt``.

    Per axis the constant-acceleration block is ``[[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]``.
    For ``model="cv"`` acceleration is dropped (its rows/cols are decoupled), so the state
    propagates at constant velocity.
    """
    F = np.eye(_N_STATE)
    for axis in range(3):
        p, v, a = axis, axis + 3, axis + 6
        F[p, v] = dt
        if model == "ca":
            F[p, a] = 0.5 * dt * dt
            F[v, a] = dt
        # model == "cv": leave accel decoupled (identity on the accel diagonal).
    return F


def _process_noise(dt: float, sigma_a: float, model: str) -> np.ndarray:
    """Discrete process-noise covariance from continuous white-noise acceleration.

    CV uses the piecewise-white-noise-acceleration model on [pos, vel] (accel block zero).
    CA uses the full [pos, vel, acc] white-noise-jerk-equivalent block. Scaled by
    ``sigma_a**2``.
    """
    q = float(sigma_a) ** 2
    Q = np.zeros((_N_STATE, _N_STATE))
    for axis in range(3):
        p, v, a = axis, axis + 3, axis + 6
        if model == "ca":
            # Continuous white-noise acceleration integrated over the CA block.
            Q[p, p] = q * dt ** 5 / 20.0
            Q[p, v] = q * dt ** 4 / 8.0
            Q[p, a] = q * dt ** 3 / 6.0
            Q[v, p] = q * dt ** 4 / 8.0
            Q[v, v] = q * dt ** 3 / 3.0
            Q[v, a] = q * dt ** 2 / 2.0
            Q[a, p] = q * dt ** 3 / 6.0
            Q[a, v] = q * dt ** 2 / 2.0
            Q[a, a] = q * dt
        else:  # "cv": white-noise acceleration drives velocity; accel state stays zero.
            Q[p, p] = q * dt ** 3 / 3.0
            Q[p, v] = q * dt ** 2 / 2.0
            Q[v, p] = q * dt ** 2 / 2.0
            Q[v, v] = q * dt
    return Q


def _measurement_matrix() -> np.ndarray:
    """H selects position from the state."""
    H = np.zeros((3, _N_STATE))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    return H


def _mm(*mats: np.ndarray) -> np.ndarray:
    """Chained matrix product via :func:`numpy.dot`.

    Used instead of the ``@`` operator because numpy 2.0.2's matmul SIMD kernel raises a
    spurious ``divide by zero / overflow encountered in matmul`` floating-point warning on
    small/odd-sized operands even when the result is finite and correct; ``dot`` does not.
    """
    out = np.asarray(mats[0], dtype=float)
    for m in mats[1:]:
        out = out.dot(np.asarray(m, dtype=float))
    return out


class _KalmanFilter:
    """Single-target linear Kalman filter over the 9-state CA-capable layout."""

    def __init__(self, model: str = "cv", sigma_a: float = 2.0):
        if model not in ("cv", "ca"):
            raise ValueError(f"model must be 'cv' or 'ca', got {model!r}")
        self.model = model
        self.sigma_a = float(sigma_a)
        self.H = _measurement_matrix()
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._t_last: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def _initialize(self, fix: TargetFix, t: float) -> None:
        x = np.zeros(_N_STATE)
        x[_POS] = np.asarray(fix.position, dtype=float)
        P = np.eye(_N_STATE)
        P[_POS, _POS] = np.asarray(fix.cov, dtype=float)
        # Generous priors on the unobserved velocity/acceleration states.
        P[_VEL, _VEL] = np.eye(3) * 1.0e3
        P[6:9, 6:9] = np.eye(3) * 1.0e2
        self._x, self._P, self._t_last = x, P, t

    def step(self, fix: TargetFix, t: float) -> None:
        """Predict to ``t`` then (NIS-gated) update with ``fix``."""
        if not self.initialized:
            self._initialize(fix, t)
            return

        dt = float(t - self._t_last)
        self._t_last = t

        # --- predict ---
        if dt > 0:
            F = _transition(dt, self.model)
            Q = _process_noise(dt, self.sigma_a, self.model)
            self._x = F.dot(self._x)
            self._P = _mm(F, self._P, F.T) + Q

        # --- gate + update ---
        z = np.asarray(fix.position, dtype=float)
        R = np.asarray(fix.cov, dtype=float)
        y = z - self.H.dot(self._x)  # innovation
        S = _mm(self.H, self._P, self.H.T) + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > _NIS_GATE:
            return  # reject this measurement, keep the prediction

        K = _mm(self._P, self.H.T, np.linalg.inv(S))
        self._x = self._x + K.dot(y)
        # Joseph form keeps P symmetric positive-definite under gating.
        IKH = np.eye(_N_STATE) - K.dot(self.H)
        self._P = _mm(IKH, self._P, IKH.T) + _mm(K, R, K.T)

    @property
    def position(self) -> np.ndarray:
        return self._x[_POS].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._x[_VEL].copy()

    @property
    def position_cov(self) -> np.ndarray:
        return self._P[_POS, _POS].copy()


def track_target(
    fixes: Sequence[TargetFix],
    model: str = "cv",
    sigma_a: float = 2.0,
) -> Track:
    """Run a single-target Kalman filter over a time-ordered sequence of fixes.

    Parameters
    ----------
    fixes
        Per-emission :class:`TargetFix` measurements (position + ``cov``). Sorted by ``t``
        internally; ``dt`` is taken from consecutive timestamps.
    model
        ``"cv"`` (constant velocity, zero acceleration process — default) or ``"ca"``
        (constant acceleration random walk).
    sigma_a
        White-noise acceleration spectral density (m/s^2) scaling the process noise.

    Returns
    -------
    Track
        Filtered ``times_s`` (T,), ``positions_local`` (T,3), ``covariances`` (T,3,3) and
        ``velocities`` (T,3).
    """
    ordered = sorted(fixes, key=lambda f: f.t)
    if not ordered:
        empty = np.empty((0,))
        return Track(
            times_s=empty,
            positions_local=np.empty((0, 3)),
            covariances=np.empty((0, 3, 3)),
            velocities=np.empty((0, 3)),
        )

    kf = _KalmanFilter(model=model, sigma_a=sigma_a)
    times, positions, covs, vels = [], [], [], []
    for fix in ordered:
        kf.step(fix, fix.t)
        times.append(fix.t)
        positions.append(kf.position)
        covs.append(kf.position_cov)
        vels.append(kf.velocity)

    return Track(
        times_s=np.asarray(times, dtype=float),
        positions_local=np.asarray(positions, dtype=float),
        covariances=np.asarray(covs, dtype=float),
        velocities=np.asarray(vels, dtype=float),
    )


class Tracker:
    """Stateful single-target tracker and the seam for future multi-target tracking.

    Feed fixes one timestep at a time via :meth:`update`, then read the accumulated
    :class:`Track` with :meth:`track`.

    Multi-target note
    -----------------
    :meth:`update` accepts a *list* of fixes per timestep — the shape a multi-target system
    needs. Today exactly one fix per timestep is supported and forwarded to a single filter.
    Data association (gating each fix to a track hypothesis, e.g. global-nearest-neighbour or
    JPDA, and spawning/retiring tracks) would slot in at the marked point in :meth:`update`,
    replacing the single-target assertion with a per-track routing of the fix list.
    """

    def __init__(self, model: str = "cv", sigma_a: float = 2.0):
        self.model = model
        self.sigma_a = float(sigma_a)
        self._kf = _KalmanFilter(model=model, sigma_a=sigma_a)
        self._times: List[float] = []
        self._positions: List[np.ndarray] = []
        self._covs: List[np.ndarray] = []
        self._vels: List[np.ndarray] = []

    def update(self, fixes_list: Sequence[TargetFix], t: float) -> None:
        """Ingest the fixes observed at time ``t`` and advance the filter one step.

        ``fixes_list`` is the per-timestep measurement set. With a single target there must
        be exactly one fix; multiple simultaneous fixes are not yet associated.
        """
        fixes_list = list(fixes_list)

        # >>> Multi-target association would go here: route each fix in `fixes_list` to the
        #     correct track hypothesis (gating + assignment), spawn/retire tracks as needed.
        #     For now we require a single target and fuse to one filter.
        if len(fixes_list) != 1:
            raise NotImplementedError(
                "Tracker currently supports a single target per timestep; "
                f"got {len(fixes_list)} fixes (association not implemented yet)."
            )
        fix = fixes_list[0]

        self._kf.step(fix, t)
        self._times.append(t)
        self._positions.append(self._kf.position)
        self._covs.append(self._kf.position_cov)
        self._vels.append(self._kf.velocity)

    def track(self) -> Track:
        """Return the :class:`Track` accumulated so far."""
        if not self._times:
            return Track(
                times_s=np.empty((0,)),
                positions_local=np.empty((0, 3)),
                covariances=np.empty((0, 3, 3)),
                velocities=np.empty((0, 3)),
            )
        return Track(
            times_s=np.asarray(self._times, dtype=float),
            positions_local=np.asarray(self._positions, dtype=float),
            covariances=np.asarray(self._covs, dtype=float),
            velocities=np.asarray(self._vels, dtype=float),
        )
