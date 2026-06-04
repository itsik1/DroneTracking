"""Iteration-2 phase-specific metrics, scored against ground truth.

Additive to :mod:`dronetracking.eval.metrics` (which stays the single-target baseline).
Each function returns a flat, JSON-serializable dict and never raises on a degraded run.
``world`` is duck-typed (same attributes as the SIM ``World``, plus ``true_tracks`` and
``positions_matrix_at``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from dronetracking import geo
from dronetracking.eval.alignment import align_to_truth

NAN = float("nan")


def _ordered_layout(layout, device_ids) -> np.ndarray:
    return np.array([layout.position_of(d) for d in device_ids], dtype=float)


def _nearest_point_rmse(path: np.ndarray, truth: np.ndarray) -> float:
    """RMSE of each path point's distance to the nearest truth point (time-offset robust)."""
    d2 = np.sum((path[:, None, :] - truth[None, :, :]) ** 2, axis=2)  # (T, N)
    return float(np.sqrt(np.mean(np.min(d2, axis=1))))


def multi_target_metrics(world: Any, tracks: List, layout) -> Dict[str, Any]:
    """Match each true drone to its nearest estimated track and report accuracy."""
    out: Dict[str, Any] = {
        "multi_target.n_true": NAN,
        "multi_target.n_tracks": len(tracks),
        "multi_target.mean_rmse_m": NAN,
        "multi_target.max_rmse_m": NAN,
    }
    try:
        true_tracks = dict(getattr(world, "true_tracks", {}) or {})
        out["multi_target.n_true"] = len(true_tracks)
        if not true_tracks or not tracks:
            return out

        # Align the layout frame to truth, then carry tracks into the truth frame.
        sim = align_to_truth(_ordered_layout(layout, world.device_ids), world.positions_matrix())
        aligned = [sim.apply(np.atleast_2d(np.asarray(t.positions_local, dtype=float))) for t in tracks]

        per_drone = {}
        for src, truth in true_tracks.items():
            truth = np.atleast_2d(np.asarray(truth, dtype=float))
            errs = [_nearest_point_rmse(a, truth) for a in aligned]
            per_drone[src] = float(np.min(errs))  # best-matching track
            out[f"multi_target.rmse_m.src{src}"] = per_drone[src]
        vals = list(per_drone.values())
        out["multi_target.mean_rmse_m"] = float(np.mean(vals))
        out["multi_target.max_rmse_m"] = float(np.max(vals))
    except Exception:
        pass
    return out


def geometry_metrics(world: Any, geometry_series: List[Tuple[float, Any]]) -> Dict[str, Any]:
    """Per-window device-localization RMSE for a moving-device run (truth at each window time)."""
    out: Dict[str, Any] = {
        "geometry.n_windows": len(geometry_series) if geometry_series else 0,
        "geometry.mean_window_rmse_m": NAN,
        "geometry.max_window_rmse_m": NAN,
    }
    try:
        rmses = []
        for t_center, layout in geometry_series:
            truth_t = world.positions_matrix_at(float(t_center))
            est = _ordered_layout(layout, world.device_ids)
            if est.shape != truth_t.shape or est.shape[0] < 3:
                continue
            aligned = align_to_truth(est, truth_t).apply(est)
            rmses.append(float(np.sqrt(np.mean(np.sum((aligned - truth_t) ** 2, axis=1)))))
        if rmses:
            out["geometry.mean_window_rmse_m"] = float(np.mean(rmses))
            out["geometry.max_window_rmse_m"] = float(np.max(rmses))
    except Exception:
        pass
    return out


def gps_denied_metrics(world: Any, geo_track: Any, blackout_windows: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
    """Georef error split by GPS-available vs dead-reckoned frames, plus track continuity."""
    out: Dict[str, Any] = {
        "gps_denied.rmse_available_m": NAN,
        "gps_denied.rmse_blackout_m": NAN,
        "gps_denied.max_step_m": NAN,
    }
    try:
        est_ll = np.atleast_2d(np.asarray(geo_track.latlon, dtype=float))
        truth = np.atleast_2d(np.asarray(world.true_track, dtype=float))
        n = min(est_ll.shape[0], truth.shape[0])
        if n == 0:
            return out
        true_lat, true_lon = geo.enu_to_latlon(truth[:n, 0], truth[:n, 1], world.origin_latlon)
        err = np.asarray(geo.haversine_m(est_ll[:n, 0], est_ll[:n, 1], true_lat, true_lon), dtype=float)

        dr = getattr(geo_track, "dead_reckoned", None)
        if dr is not None:
            dr = np.asarray(dr, dtype=bool)[:n]
        else:
            times = np.asarray(geo_track.times_s, dtype=float)[:n]
            dr = np.array([any(a <= t <= b for (a, b) in blackout_windows) for t in times])

        if np.any(~dr):
            out["gps_denied.rmse_available_m"] = float(np.sqrt(np.mean(err[~dr] ** 2)))
        if np.any(dr):
            out["gps_denied.rmse_blackout_m"] = float(np.sqrt(np.mean(err[dr] ** 2)))

        # Continuity: largest consecutive-frame step in ENU meters (no jump = smooth blend).
        e, nn = geo.latlon_to_enu(est_ll[:n, 0], est_ll[:n, 1], world.origin_latlon)
        enu = np.column_stack([np.atleast_1d(e), np.atleast_1d(nn)])
        if enu.shape[0] >= 2:
            out["gps_denied.max_step_m"] = float(np.max(np.linalg.norm(np.diff(enu, axis=0), axis=1)))
    except Exception:
        pass
    return out
