"""Compute a flat, JSON-serializable metrics dict comparing estimates to truth.

This is the only module besides the harness that sees both the ground-truth
``World`` and the pipeline ``Estimates``. Every metric is keyed with a
``"<group>.<name>"`` string so the result stays flat (trivially JSON-dumpable and
easy to diff across runs). Each block is wrapped so a degraded/failed estimation
stage yields ``NaN`` fields instead of raising — a partial scorecard beats a crash.

Conventions:
  * Distances/positions are in meters.
  * Clock offsets in seconds, drifts in ppm.
  * Layout is scored after a reflection-allowed, no-scale alignment to truth
    (see :mod:`dronetracking.eval.alignment`).
  * Clock offset/drift are observable only up to a global gauge, so both the
    estimated and true series are referenced to the reference device before the
    error is taken.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from dronetracking import geo
from dronetracking.eval.alignment import align_to_truth

NAN = float("nan")


# --------------------------------------------------------------------------- #
# small numeric helpers (all NaN-safe on degraded input)
# --------------------------------------------------------------------------- #
def _rmse(per_point_sq_err: np.ndarray) -> float:
    """Root-mean-square from an array of squared errors; NaN if empty."""
    arr = np.asarray(per_point_sq_err, dtype=float)
    if arr.size == 0:
        return NAN
    return float(np.sqrt(np.mean(arr)))


def _as_2d(a: Any, cols: int) -> Optional[np.ndarray]:
    """Coerce to an ``(n, cols)`` float array, or return ``None`` if not possible."""
    if a is None:
        return None
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != cols:
        return None
    return arr


def _matched_len(*arrays: Optional[np.ndarray]) -> int:
    """Common row count across arrays (the min), or 0 if any is missing/empty."""
    lengths = []
    for a in arrays:
        if a is None or a.shape[0] == 0:
            return 0
        lengths.append(a.shape[0])
    return min(lengths)


# --------------------------------------------------------------------------- #
# metric blocks
# --------------------------------------------------------------------------- #
def _layout_alignment(world: Any, estimates: Any):
    """Similarity mapping the estimated layout frame onto the true (ENU) frame.

    Returned once and reused for both device localization and tracking, so the track
    (which lives in the *arbitrary* layout frame) is scored in the truth frame using
    geometry alignment only — independent of the GPS-anchor georeferencing metric.
    Returns ``(sim, order)`` or ``None`` if not computable.
    """
    try:
        layout = estimates.layout
        order = list(layout.device_ids)
        idx = {d: i for i, d in enumerate(order)}
        est_pos = np.asarray(layout.positions_local, dtype=float)
        est_pos = np.array([est_pos[idx[d]] for d in order], dtype=float)
        truth = np.asarray(world.positions_matrix(), dtype=float)
        if est_pos.shape != truth.shape or est_pos.shape[0] < 3:
            return None
        return align_to_truth(est_pos, truth), order, est_pos, truth
    except Exception:
        return None


def _device_localization(world: Any, estimates: Any, alignment: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "device_localization.rmse_m": NAN,
        "device_localization.max_m": NAN,
        "device_localization.alignment_scale": NAN,
        "device_localization.alignment_was_reflected": False,
    }
    if alignment is None:
        return out
    try:
        sim, order, est_pos, truth = alignment
        aligned = sim.apply(est_pos)
        per_dev = np.linalg.norm(aligned - truth, axis=1)

        out["device_localization.rmse_m"] = _rmse(per_dev**2)
        out["device_localization.max_m"] = float(np.max(per_dev))
        out["device_localization.alignment_scale"] = float(sim.scale)
        out["device_localization.alignment_was_reflected"] = bool(sim.is_reflection)
        # Per-device error, keyed flatly so the dict stays JSON-flat.
        for d, e in zip(order, per_dev):
            out[f"device_localization.error_m.{d}"] = float(e)
    except Exception:  # degraded run: keep the NaN defaults
        pass
    return out


def _gauge_removed(series: Dict[str, float], ids: Sequence[str], ref: Optional[str]) -> np.ndarray:
    """Series values over ``ids`` with the reference (or mean) subtracted."""
    vals = np.array([float(series[d]) for d in ids], dtype=float)
    if ref is not None and ref in series:
        return vals - float(series[ref])
    return vals - float(np.mean(vals))


def _clock_sync(world: Any, estimates: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "clock_sync.offset_rmse_s": NAN,
        "clock_sync.drift_rmse_ppm": NAN,
    }
    try:
        clocks = estimates.clocks
        ref = getattr(clocks, "reference_id", None)
        # Devices common to estimate and truth, in the estimate's declared order.
        ids = [
            d
            for d in clocks.device_ids
            if d in clocks.offsets_s
            and d in world.clock_offsets
            and d in clocks.drifts_ppm
            and d in world.clock_drifts_ppm
        ]
        if len(ids) == 0:
            return out

        est_off = _gauge_removed(clocks.offsets_s, ids, ref)
        true_off = _gauge_removed(world.clock_offsets, ids, ref)
        out["clock_sync.offset_rmse_s"] = _rmse((est_off - true_off) ** 2)

        est_drift = _gauge_removed(clocks.drifts_ppm, ids, ref)
        true_drift = _gauge_removed(world.clock_drifts_ppm, ids, ref)
        out["clock_sync.drift_rmse_ppm"] = _rmse((est_drift - true_drift) ** 2)
    except Exception:
        pass
    return out


def _nees_mean(err: np.ndarray, covs: Any) -> float:
    """Mean normalized estimation error squared ``e^T P^-1 e`` over time.

    Honest covariances give a mean near the position dimension (3). Singular or
    missing covariances at a step are skipped; NaN if none are usable.
    """
    try:
        covs = np.asarray(covs, dtype=float)
        if covs.ndim != 3 or covs.shape[1:] != (3, 3) or covs.shape[0] < err.shape[0]:
            return NAN
        vals = []
        for i in range(err.shape[0]):
            P = covs[i]
            e = err[i]
            try:
                vals.append(float(e @ np.linalg.solve(P, e)))
            except np.linalg.LinAlgError:
                vals.append(float(e @ (np.linalg.pinv(P) @ e)))
        if not vals:
            return NAN
        return float(np.mean(vals))
    except Exception:
        return NAN


def _tracking(world: Any, estimates: Any, alignment: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "tracking.rmse_m": NAN,
        "tracking.rmse_xy_m": NAN,
        "tracking.rmse_z_m": NAN,
        "tracking.final_error_m": NAN,
        "tracking.nees_mean": NAN,
    }
    try:
        track = estimates.track
        est_pos = _as_2d(track.positions_local, 3)
        truth = _as_2d(world.true_track, 3)
        n = _matched_len(est_pos, truth)
        if n == 0:
            return out

        # The track lives in the arbitrary layout frame; map it into the truth frame
        # via the device-geometry alignment before scoring. Covariances rotate too,
        # so NEES stays frame-consistent.
        covs = getattr(track, "covariances", None)
        if alignment is not None:
            sim = alignment[0]
            est_pos = sim.apply(est_pos)
            if covs is not None:
                covs = np.asarray(covs, dtype=float)
                if covs.ndim == 3 and covs.shape[1:] == (3, 3):
                    covs = (sim.scale**2) * np.einsum("ij,njk,lk->nil", sim.R, covs, sim.R)

        # Match by index (estimates align step-for-step with the true track).
        e = est_pos[:n]
        g = truth[:n]
        err = e - g
        sq = np.sum(err**2, axis=1)

        out["tracking.rmse_m"] = _rmse(sq)
        out["tracking.rmse_xy_m"] = _rmse(np.sum(err[:, :2] ** 2, axis=1))
        out["tracking.rmse_z_m"] = _rmse(err[:, 2] ** 2)
        out["tracking.final_error_m"] = float(np.linalg.norm(err[-1]))
        out["tracking.nees_mean"] = _nees_mean(err, covs)
    except Exception:
        pass
    return out


def _georeferencing(world: Any, estimates: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "georeferencing.rmse_m": NAN,
        "georeferencing.altitude_rmse_m": NAN,
    }
    try:
        gt = estimates.geo_track
        est_ll = _as_2d(gt.latlon, 2)
        truth = _as_2d(world.true_track, 3)
        n = _matched_len(est_ll, truth)
        if n > 0:
            true_lat, true_lon = geo.enu_to_latlon(
                truth[:n, 0], truth[:n, 1], world.origin_latlon
            )
            d = geo.haversine_m(
                est_ll[:n, 0], est_ll[:n, 1], true_lat, true_lon
            )
            out["georeferencing.rmse_m"] = _rmse(np.asarray(d, dtype=float) ** 2)

        est_alt = np.asarray(getattr(gt, "altitude_m", []), dtype=float).ravel()
        if est_alt.size and truth is not None and truth.shape[0]:
            na = min(est_alt.shape[0], truth.shape[0])
            alt_err = est_alt[:na] - truth[:na, 2]
            out["georeferencing.altitude_rmse_m"] = _rmse(alt_err**2)
    except Exception:
        pass
    return out


def _scenario(world: Any, observations: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scenario.name": None,
        "scenario.n_devices": NAN,
        "scenario.n_anchors": NAN,
    }
    try:
        name = getattr(world, "name", None)
        if name is None:
            name = getattr(world, "scenario_name", None)
        out["scenario.name"] = name if name is None else str(name)
    except Exception:
        pass
    try:
        out["scenario.n_devices"] = int(len(world.device_ids))
    except Exception:
        pass
    try:
        out["scenario.n_anchors"] = int(len(observations.anchor_gps))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def compute_metrics(world: Any, observations: Any, estimates: Any) -> Dict[str, Any]:
    """Score ``estimates`` against the ground-truth ``world``.

    Returns a flat, JSON-serializable dict keyed ``"<group>.<name>"`` covering
    device localization, clock sync, tracking, georeferencing, and a scenario
    summary. Never raises on a degraded run — failing blocks report ``NaN``.
    """
    metrics: Dict[str, Any] = {}
    alignment = _layout_alignment(world, estimates)  # computed once, shared
    metrics.update(_scenario(world, observations))
    metrics.update(_device_localization(world, estimates, alignment))
    metrics.update(_clock_sync(world, estimates))
    metrics.update(_tracking(world, estimates, alignment))
    metrics.update(_georeferencing(world, estimates))
    return metrics
