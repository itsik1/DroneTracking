"""Multi-target tracking: a stream of per-frame fix SETS -> several labelled tracks.

Where :mod:`estimation.tracking` runs one Kalman filter over one fix per timestep, the
real world hands us, at each emission time, a *set* of position fixes (one per drone in
view) with NO labels telling us which fix belongs to which target. This module adds the
two missing pieces:

1. **Per-frame fixes** — :func:`localize_frames` groups raw acoustic arrivals by
   ``(emission_idx, source)`` and runs the frozen :func:`estimation.tdoa.localize_emission`
   on each group with enough devices, yielding, per emission time, the SET of fixes. The
   ``source`` tag is used ONLY to form clean (un-mixed) fixes; it is dropped before the
   tracker sees them, so association cannot cheat by reading the true label.

2. **Data association + track management** — :func:`track_targets` runs a bank of
   single-target Kalman filters (reusing :class:`estimation.tracking._KalmanFilter`). At
   each frame it predicts every live track forward, scores every (track, fix) pair by the
   squared Mahalanobis distance in the measurement innovation covariance, applies a
   chi-square gate, and solves the global-nearest-neighbour assignment with
   :func:`scipy.optimize.linear_sum_assignment`. Matched fixes update their track;
   unmatched fixes spawn tentative tracks (confirmed after ``birth_min_hits`` hits);
   tracks that miss ``death_max_misses`` frames in a row are terminated.

This module is part of the estimation package and therefore MUST NOT import
``dronetracking.sim`` (the ground-truth firewall, enforced by tests/test_no_truth_leak.py).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ..datatypes import TargetFix
from . import tdoa
from .interfaces import ClockEstimates, RelativeLayout, Track
from .tracking import _KalmanFilter

# Measurement dimension (a 3D position fix) -> chi-square dof for the association gate.
_GATE_DOF = 3
# Default 99th-percentile chi-square gate on the squared Mahalanobis distance between a
# track's prediction and a candidate fix.
_DEFAULT_GATE_CHI2 = float(chi2.ppf(0.99, df=_GATE_DOF))

# A finite stand-in for "infinitely costly / gated out" in the assignment cost matrix
# (linear_sum_assignment cannot ingest np.inf).
_BIG_COST = 1.0e9


def localize_frames(
    arrivals: Sequence,
    clocks: ClockEstimates,
    layout: RelativeLayout,
    speed_of_sound: float,
    toa_var_s2: float = 1e-8,
) -> List[Tuple[float, List[TargetFix]]]:
    """Group multi-source arrivals into per-emission-time SETS of position fixes.

    Parameters
    ----------
    arrivals
        ``AcousticArrival``-like objects carrying ``.device_id``, ``.emission_idx``,
        ``.toa_local_s`` and ``.source`` (the per-target tag from the multi-target sim).
    clocks, layout, speed_of_sound, toa_var_s2
        Passed straight through to :func:`estimation.tdoa.localize_emission`.

    Returns
    -------
    list of ``(t, fixes)``
        One entry per emission time (frame), sorted by ``t``. ``fixes`` is the SET of
        :class:`TargetFix` recovered at that emission — one per ``source`` that had at
        least :data:`estimation.tdoa.MIN_DEVICES_3D` devices. The frame timestamp ``t`` is
        the mean fix time across the frame's fixes (the reference-timebase emission time).

    Notes
    -----
    ``source`` is consumed here purely to *separate* the overlapping arrivals into clean,
    single-target groups (so each TDOA solve sees one drone). The returned fixes carry no
    source label, so :func:`track_targets` must re-derive identity from geometry alone.
    """
    # Group arrivals by (emission_idx, source). Each clean group is one drone heard by
    # several devices at one emission.
    groups: dict = {}
    for arr in arrivals:
        groups.setdefault((arr.emission_idx, arr.source), []).append(arr)

    # Collect fixes per emission_idx (the frame), discarding the source label afterwards.
    per_emission: dict = {}
    for (emission_idx, _source), group in groups.items():
        if len(group) < tdoa.MIN_DEVICES_3D:
            continue
        fix = tdoa.localize_emission(group, clocks, layout, speed_of_sound, toa_var_s2=toa_var_s2)
        per_emission.setdefault(emission_idx, []).append(fix)

    frames: List[Tuple[float, List[TargetFix]]] = []
    for emission_idx in sorted(per_emission):
        fixes = per_emission[emission_idx]
        # Frame time: mean of the fixes' reference-timebase timestamps. Sort the fixes
        # within a frame by time only for determinism; association is order-independent.
        fixes.sort(key=lambda f: f.t)
        t = float(np.mean([f.t for f in fixes]))
        frames.append((t, fixes))

    frames.sort(key=lambda tf: tf[0])
    return frames


class _TrackHypothesis:
    """A live track: a single-target Kalman filter plus association bookkeeping.

    Holds the filter, the accumulated (time, position, covariance, velocity) samples, the
    consecutive-hit/consecutive-miss counters that drive birth and death, and a stable
    integer id assigned at creation (so a confirmed track's identity never changes).
    """

    __slots__ = (
        "uid", "kf", "times", "positions", "covs", "vels",
        "hits", "misses", "confirmed", "t_last",
    )

    def __init__(self, uid: int, fix: TargetFix, t: float, model: str, sigma_a: float):
        self.uid = uid
        self.kf = _KalmanFilter(model=model, sigma_a=sigma_a)
        self.times: List[float] = []
        self.positions: List[np.ndarray] = []
        self.covs: List[np.ndarray] = []
        self.vels: List[np.ndarray] = []
        self.hits = 0
        self.misses = 0
        self.confirmed = False
        self.t_last = t
        # Seed the filter with the spawning fix and record the first sample.
        self._absorb(fix, t)

    def _absorb(self, fix: TargetFix, t: float) -> None:
        """Kalman-step on ``fix`` at time ``t`` and append the resulting sample."""
        self.kf.step(fix, t)
        self.t_last = t
        self.times.append(t)
        self.positions.append(self.kf.position)
        self.covs.append(self.kf.position_cov)
        self.vels.append(self.kf.velocity)

    def predict_to(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted (position, position-covariance) at time ``t`` WITHOUT mutating state.

        Used to score candidate fixes before assignment; the actual predict/update happens
        in :meth:`update` (or is skipped on a miss). Mirrors the filter's predict step.
        """
        # Before the first update the filter is uninitialised; its current sample is the
        # best available prediction.
        if not self.kf.initialized:
            return self.positions[-1].copy(), self.covs[-1].copy()
        x = self.kf._x.copy()
        P = self.kf._P.copy()
        dt = float(t - self.kf._t_last)
        if dt > 0:
            from .tracking import _process_noise, _transition  # local: avoid cycle at import

            F = _transition(dt, self.kf.model)
            Q = _process_noise(dt, self.kf.sigma_a, self.kf.model)
            x = F.dot(x)
            P = F.dot(P).dot(F.T) + Q
        H = self.kf.H
        return H.dot(x), H.dot(P).dot(H.T)

    def update(self, fix: TargetFix, t: float) -> None:
        """Associate ``fix`` to this track: Kalman update + register a hit."""
        self._absorb(fix, t)
        self.hits += 1
        self.misses = 0

    def miss(self) -> None:
        """Register a frame with no associated fix."""
        self.misses += 1

    def to_track(self, target_id: str) -> Track:
        """Materialize the accumulated samples into a labelled :class:`Track`."""
        return Track(
            times_s=np.asarray(self.times, dtype=float),
            positions_local=np.asarray(self.positions, dtype=float),
            covariances=np.asarray(self.covs, dtype=float),
            velocities=np.asarray(self.vels, dtype=float),
            target_id=target_id,
        )


def track_targets(
    frames: Sequence[Tuple[float, Sequence[TargetFix]]],
    gate_chi2: float = _DEFAULT_GATE_CHI2,
    birth_min_hits: int = 2,
    death_max_misses: int = 3,
    model: str = "cv",
    sigma_a: float = 2.0,
) -> List[Track]:
    """Track an unknown, time-varying number of targets across a sequence of fix SETS.

    Parameters
    ----------
    frames
        Time-ordered ``(t, fixes)`` frames, as returned by :func:`localize_frames`.
    gate_chi2
        Chi-square gate (squared-Mahalanobis threshold, 3 dof) for admitting a
        (track, fix) pair into the assignment. Pairs beyond the gate cannot be matched.
    birth_min_hits
        A tentative track is *confirmed* once it has accumulated this many associated
        fixes (hits). Only confirmed tracks are returned.
    death_max_misses
        A track is terminated after this many *consecutive* frames with no associated fix.
    model, sigma_a
        Forwarded to each per-track :class:`estimation.tracking._KalmanFilter`.

    Returns
    -------
    list of :class:`Track`
        The confirmed tracks, each with a distinct ``target_id`` (``"T0"``, ``"T1"``, ...),
        ordered by birth (creation) order.

    Method
    ------
    At each frame the live tracks are predicted to the frame time; the squared Mahalanobis
    distance ``yᵀ S⁻¹ y`` (``y`` = fix − predicted position, ``S = H P Hᵀ + R_fix``) scores
    every (track, fix) pair. Pairs outside ``gate_chi2`` are forbidden. A single global
    assignment (:func:`scipy.optimize.linear_sum_assignment`) picks the lowest-total-cost
    set of non-conflicting matches (global nearest neighbour). Matched fixes update their
    track; unmatched fixes spawn tentative tracks; unmatched tracks register a miss and are
    pruned once they exceed ``death_max_misses``.
    """
    live: List[_TrackHypothesis] = []
    confirmed: List[_TrackHypothesis] = []
    next_uid = 0

    for t, fixes in frames:
        fixes = list(fixes)

        # ---- score + gate every (live track, fix) pair -----------------------------
        if live and fixes:
            n_tracks = len(live)
            n_fixes = len(fixes)
            cost = np.full((n_tracks, n_fixes), _BIG_COST, dtype=float)
            gated = np.zeros((n_tracks, n_fixes), dtype=bool)

            preds = [trk.predict_to(t) for trk in live]
            for ti, (pred_x, pred_P) in enumerate(preds):
                for fi, fix in enumerate(fixes):
                    z = np.asarray(fix.position, dtype=float)
                    R = np.asarray(fix.cov, dtype=float)
                    S = pred_P + R
                    y = z - pred_x
                    try:
                        d2 = float(y @ np.linalg.solve(S, y))
                    except np.linalg.LinAlgError:
                        continue
                    if d2 <= gate_chi2:
                        cost[ti, fi] = d2
                        gated[ti, fi] = True

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_fixes = set()
            for ti, fi in zip(row_idx, col_idx):
                if not gated[ti, fi]:
                    continue  # assignment fell on a gated-out (BIG_COST) cell -> not a match
                live[ti].update(fixes[fi], t)
                matched_tracks.add(ti)
                matched_fixes.add(fi)
        else:
            matched_tracks = set()
            matched_fixes = set()

        # ---- unmatched live tracks register a miss ----------------------------------
        for ti, trk in enumerate(live):
            if ti not in matched_tracks:
                trk.miss()

        # ---- unmatched fixes spawn tentative tracks ---------------------------------
        for fi, fix in enumerate(fixes):
            if fi not in matched_fixes:
                live.append(_TrackHypothesis(next_uid, fix, t, model=model, sigma_a=sigma_a))
                next_uid += 1

        # ---- promote tentative tracks that have enough hits -------------------------
        for trk in live:
            if not trk.confirmed and trk.hits >= birth_min_hits:
                trk.confirmed = True

        # ---- terminate tracks that have missed too many frames in a row ------------
        survivors: List[_TrackHypothesis] = []
        for trk in live:
            if trk.misses > death_max_misses:
                if trk.confirmed:
                    confirmed.append(trk)  # retire a confirmed track to the output set
                # tentative tracks that die are simply discarded
            else:
                survivors.append(trk)
        live = survivors

    # Any still-live confirmed tracks at the end of the run are also output.
    for trk in live:
        if trk.confirmed:
            confirmed.append(trk)

    # Stable, distinct ids in birth order.
    confirmed.sort(key=lambda h: h.uid)
    return [trk.to_track(f"T{i}") for i, trk in enumerate(confirmed)]
