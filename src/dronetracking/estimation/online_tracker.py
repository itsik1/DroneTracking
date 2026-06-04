"""Online (incremental) multi-target tracker: one frame in, updated tracks out.

Where :func:`estimation.multi_target.track_targets` is a *batch* tracker — it consumes
the whole list of ``(t, fixes)`` frames and runs the data-association + track-management
loop over all of them in one call — :class:`OnlineTracker` is the same algorithm turned
*stateful*. You feed it one frame at a time via :meth:`update`; it carries the live track
hypotheses across calls and does only O(tracks x fixes) work per frame. This is the shape
a real-time streaming engine wants: the old engine re-ran ``track_targets`` over the
entire growing prefix every emission (O(frames^2) overall); an :class:`OnlineTracker`
advanced one ``update`` per emission does the same job in O(frames) total.

The per-frame logic is identical to :func:`track_targets`:

1. Predict every live track to the frame time ``t``.
2. Score every (track, fix) pair by the squared Mahalanobis distance
   ``y^T S^-1 y`` (``y`` = fix - predicted position, ``S = H P H^T + R_fix``) and apply a
   chi-square gate.
3. Solve the global-nearest-neighbour assignment with
   :func:`scipy.optimize.linear_sum_assignment`; matched fixes Kalman-update their track
   (reusing :class:`estimation.tracking._KalmanFilter`).
4. Unmatched live tracks register a miss; unmatched fixes spawn tentative tracks.
5. Tentative tracks are confirmed once they reach ``birth_min_hits`` hits; tracks that
   miss ``death_max_misses`` frames in a row are terminated.

The single behavioural difference from the batch function is in how ``target_id`` strings
are minted (see :meth:`tracks`): the batch function relabels its final confirmed set to a
dense ``T0..Tn`` only at the very end, which is impossible online (a track's id would have
to change as later tracks are born). :class:`OnlineTracker` therefore assigns each track a
*stable* id ``"T{birth_uid}"`` at confirmation, so a confirmed track keeps the same id for
the rest of the run and across every :meth:`tracks` call. The set of confirmed tracks, the
track count, and the filtered geometry are identical to the batch tracker on the same
frames.

This module is part of the estimation package and therefore MUST NOT import
``dronetracking.sim`` (the ground-truth firewall, enforced by tests/test_no_truth_leak.py).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from ..datatypes import TargetFix
from .interfaces import Track
from .tracking import _KalmanFilter, _process_noise, _transition

# Measurement dimension (a 3D position fix) -> chi-square dof for the association gate.
_GATE_DOF = 3
# Default 99th-percentile chi-square gate on the squared Mahalanobis distance between a
# track's prediction and a candidate fix (matches estimation.multi_target).
_DEFAULT_GATE_CHI2 = float(chi2.ppf(0.99, df=_GATE_DOF))

# A finite stand-in for "infinitely costly / gated out" in the assignment cost matrix
# (linear_sum_assignment cannot ingest np.inf). Matches estimation.multi_target.
_BIG_COST = 1.0e9


class _TrackHypothesis:
    """A live track: a single-target Kalman filter plus association bookkeeping.

    Holds the filter, the accumulated (time, position, covariance, velocity) samples, the
    consecutive-hit/consecutive-miss counters that drive birth and death, and a stable
    integer id assigned at creation (so a confirmed track's identity never changes). This
    mirrors :class:`estimation.multi_target._TrackHypothesis` exactly so the online tracker
    is bug-for-bug comparable to the batch tracker.
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
        # Seed the filter with the spawning fix and record the first sample. The seed is
        # NOT counted as a hit (matches batch): a track needs `birth_min_hits` subsequent
        # associations to confirm.
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


class OnlineTracker:
    """Stateful global-nearest-neighbour multi-target tracker, advanced one frame at a time.

    Parameters
    ----------
    model, sigma_a
        Forwarded to each per-track :class:`estimation.tracking._KalmanFilter` (``"cv"`` =
        constant velocity, ``"ca"`` = constant acceleration; ``sigma_a`` is the white-noise
        acceleration spectral density, m/s^2).
    gate_chi2
        Chi-square gate (squared-Mahalanobis threshold, 3 dof) for admitting a (track, fix)
        pair into the assignment. Pairs beyond the gate cannot be matched. Defaults to the
        99th percentile of chi-square with 3 dof, matching
        :func:`estimation.multi_target.track_targets`.
    birth_min_hits
        A tentative track is *confirmed* once it accumulates this many associated fixes
        (hits beyond the spawning fix). Only confirmed tracks are returned by
        :meth:`tracks`.
    death_max_misses
        A track is terminated after this many *consecutive* frames with no associated fix.

    Usage
    -----
    Feed time-ordered frames one at a time::

        trk = OnlineTracker()
        for t, fixes in frames:
            trk.update(fixes, t)
        confirmed = trk.tracks()

    State persists across :meth:`update` calls: each call does only O(tracks x fixes) work
    and never reprocesses earlier frames.
    """

    def __init__(
        self,
        *,
        model: str = "cv",
        sigma_a: float = 2.0,
        gate_chi2: float = _DEFAULT_GATE_CHI2,
        birth_min_hits: int = 2,
        death_max_misses: int = 3,
    ):
        self.model = model
        self.sigma_a = float(sigma_a)
        self.gate_chi2 = float(gate_chi2)
        self.birth_min_hits = int(birth_min_hits)
        self.death_max_misses = int(death_max_misses)

        # Live track hypotheses (tentative + confirmed-but-still-tracking).
        self._live: List[_TrackHypothesis] = []
        # Confirmed tracks that have since terminated (retired to the output set).
        self._retired: List[_TrackHypothesis] = []
        # Monotonic birth counter -> stable per-track id.
        self._next_uid = 0

    def update(self, fixes: Sequence[TargetFix], t: float) -> None:
        """Ingest one frame's fixes at time ``t`` and advance the tracker by one step.

        Predicts every live track to ``t``, gates + globally assigns this frame's fixes to
        tracks (squared-Mahalanobis cost, chi-square gate, :func:`linear_sum_assignment`),
        Kalman-updates matched tracks, spawns tentative tracks for unmatched fixes,
        registers a miss on unmatched tracks, confirms tracks that reached
        ``birth_min_hits`` hits, and terminates tracks that missed ``death_max_misses``
        frames in a row. The per-frame logic is identical to one iteration of
        :func:`estimation.multi_target.track_targets`.
        """
        fixes = list(fixes)
        live = self._live

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
                    if d2 <= self.gate_chi2:
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
                live.append(
                    _TrackHypothesis(self._next_uid, fix, t, model=self.model, sigma_a=self.sigma_a)
                )
                self._next_uid += 1

        # ---- promote tentative tracks that have enough hits -------------------------
        for trk in live:
            if not trk.confirmed and trk.hits >= self.birth_min_hits:
                trk.confirmed = True

        # ---- terminate tracks that have missed too many frames in a row ------------
        survivors: List[_TrackHypothesis] = []
        for trk in live:
            if trk.misses > self.death_max_misses:
                if trk.confirmed:
                    self._retired.append(trk)  # retire a confirmed track to the output set
                # tentative tracks that die are simply discarded
            else:
                survivors.append(trk)
        self._live = survivors

    def tracks(self) -> List[Track]:
        """Return the confirmed tracks so far, with stable, distinct ``target_id``s.

        Includes both confirmed tracks that have terminated and confirmed tracks still
        being tracked, ordered by birth (creation) order. Each track's id is ``"T{uid}"``
        for its stable birth uid, so a confirmed track keeps the same id across every call
        to this method even as new tracks are born (the property a streaming consumer
        needs). The *set* of confirmed tracks and their geometry match
        :func:`estimation.multi_target.track_targets` on the same frames; only the id string
        scheme differs (the batch function compresses ids to a dense ``T0..Tn`` once, at the
        end of the whole run, which is not possible incrementally).
        """
        confirmed = [trk for trk in self._retired if trk.confirmed]
        confirmed += [trk for trk in self._live if trk.confirmed]
        confirmed.sort(key=lambda h: h.uid)
        return [trk.to_track(f"T{trk.uid}") for trk in confirmed]
