"""Matplotlib diagnostic plots for a drone-tracking run (headless, ``Agg``).

Writes four PNGs into ``out_dir``:

1. ``local_frame_devices.png`` — true vs. estimated (rigidly aligned) device
   positions in the local ENU frame, with residual segments.
2. ``device_localization_error.png`` — per-device localization error bar chart.
3. ``tracking_error_over_time.png`` — horizontal/3D track error vs. time, RMSE annotated.
4. ``trajectory_topdown.png`` — top-down (east/north) true vs. estimated trajectory.

``world`` is duck-typed (see :mod:`dronetracking.viz.map_view`); it is never
imported. Only frozen :mod:`dronetracking.geo` / :mod:`dronetracking.transforms`
and the estimation interfaces are used.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; set before importing pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

from dronetracking import geo, transforms  # noqa: E402
from dronetracking.estimation.interfaces import Estimates  # noqa: E402

_BLUE = "#1f77b4"
_ORANGE = "#ff7f0e"
_GRAY = "#888888"
_GREEN = "#2ca02c"
_RED = "#d62728"


def save_diagnostics(world, estimates: Estimates, scenario, out_dir, geo_tracks=None) -> List[Path]:
    """Render the diagnostic PNG set into ``out_dir``; return the written paths.

    ``geo_tracks`` (optional): when more than one is given (multi-target), the
    trajectory and tracking-error plots show every true drone and every estimated
    track instead of just the single primary pair.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = getattr(scenario, "name", "scenario")
    multi = geo_tracks is not None and len(geo_tracks) > 1
    paths: List[Path] = []

    paths.append(_plot_local_frame_devices(world, estimates, name, out_dir))
    paths.append(_plot_device_error_bars(world, estimates, name, out_dir))
    if multi:
        paths.append(_plot_tracking_error_multi(world, geo_tracks, name, out_dir))
        paths.append(_plot_trajectory_topdown_multi(world, geo_tracks, name, out_dir))
    else:
        paths.append(_plot_tracking_error(world, estimates, name, out_dir))
        paths.append(_plot_trajectory_topdown(world, estimates, name, out_dir))

    return paths


# --------------------------------------------------------------------------- #
# Individual figures
# --------------------------------------------------------------------------- #


def _plot_local_frame_devices(world, estimates: Estimates, name: str, out_dir: Path) -> Path:
    truth = np.asarray(world.positions_matrix(), dtype=float)
    est = np.asarray(estimates.layout.positions_local, dtype=float)
    aligned = _align(est, truth)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(truth[:, 0], truth[:, 1], c=_BLUE, s=60, label="true", zorder=3)
    ax.scatter(aligned[:, 0], aligned[:, 1], c=_ORANGE, marker="x", s=70,
               label="estimated (aligned)", zorder=4)

    # Residual segments true <-> estimated.
    for i in range(min(len(truth), len(aligned))):
        ax.plot([truth[i, 0], aligned[i, 0]], [truth[i, 1], aligned[i, 1]],
                color=_GRAY, lw=1.0, zorder=2)

    for i, did in enumerate(estimates.layout.device_ids):
        if i < len(truth):
            ax.annotate(str(did), (truth[i, 0], truth[i, 1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Local-frame device layout — {name}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "local_frame_devices.png")


def _plot_device_error_bars(world, estimates: Estimates, name: str, out_dir: Path) -> Path:
    truth = np.asarray(world.positions_matrix(), dtype=float)
    est = np.asarray(estimates.layout.positions_local, dtype=float)
    aligned = _align(est, truth)

    k = min(len(truth), len(aligned))
    errors = np.linalg.norm(aligned[:k] - truth[:k], axis=1)
    labels = [str(d) for d in estimates.layout.device_ids[:k]]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    xs = np.arange(k)
    ax.bar(xs, errors, color=_ORANGE, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("position error (m)")
    ax.set_title(f"Per-device localization error — {name}")
    ax.grid(True, axis="y", alpha=0.3)
    if k:
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ax.axhline(rmse, color=_RED, ls="--", lw=1.2, label=f"RMSE = {rmse:.2f} m")
        ax.legend(loc="best")
    return _save(fig, out_dir / "device_localization_error.png")


def _plot_tracking_error(world, estimates: Estimates, name: str, out_dir: Path) -> Path:
    est_enu, est_times = _estimated_track_enu(world, estimates)
    true_enu = _true_track_at(world, est_times)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    if est_enu.shape[0] and true_enu.shape[0]:
        m = min(est_enu.shape[0], true_enu.shape[0])
        diff = est_enu[:m] - true_enu[:m]
        horiz = np.linalg.norm(diff[:, :2], axis=1)
        full = np.linalg.norm(diff, axis=1)
        t = est_times[:m]

        ax.plot(t, horiz, color=_BLUE, lw=1.8, label="horizontal error")
        ax.plot(t, full, color=_RED, lw=1.4, ls="--", label="3D error")

        rmse_h = float(np.sqrt(np.mean(horiz ** 2)))
        rmse_3d = float(np.sqrt(np.mean(full ** 2)))
        ax.annotate(
            f"horiz RMSE = {rmse_h:.2f} m\n3D RMSE = {rmse_3d:.2f} m",
            xy=(0.02, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=_GRAY, alpha=0.85),
        )
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "no track samples", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("error (m)")
    ax.set_title(f"Tracking error over time — {name}")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "tracking_error_over_time.png")


def _plot_trajectory_topdown(world, estimates: Estimates, name: str, out_dir: Path) -> Path:
    true_track = np.atleast_2d(np.asarray(world.true_track, dtype=float))
    est_enu, _ = _estimated_track_enu(world, estimates)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    if true_track.size:
        ax.plot(true_track[:, 0], true_track[:, 1], color=_GREEN, lw=2.2,
                marker="o", ms=3, label="true")
    if est_enu.size:
        ax.plot(est_enu[:, 0], est_enu[:, 1], color=_RED, lw=1.6, ls="--",
                marker="x", ms=4, label="estimated")

    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Top-down trajectory — {name}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "trajectory_topdown.png")


_MULTI_PALETTE = ["#d62728", "#9467bd", "#17becf", "#bcbd22", "#e377c2", "#8c564b"]


def _plot_trajectory_topdown_multi(world, geo_tracks, name: str, out_dir: Path) -> Path:
    """Top-down view of ALL true drones (green) vs ALL estimated tracks (palette)."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    true_tracks = dict(getattr(world, "true_tracks", {}) or {})
    first = True
    for _src, tr in sorted(true_tracks.items()):
        tr = np.atleast_2d(np.asarray(tr, dtype=float))
        ax.plot(tr[:, 0], tr[:, 1], color=_GREEN, lw=2.2, marker="o", ms=2,
                label="true" if first else None, zorder=2)
        first = False
    for i, gt in enumerate(geo_tracks):
        enu, _ = _geo_track_enu(world, gt)
        if enu.size:
            c = _MULTI_PALETTE[i % len(_MULTI_PALETTE)]
            tid = getattr(gt, "target_id", None) or f"T{i}"
            ax.plot(enu[:, 0], enu[:, 1], color=c, lw=1.6, ls="--", marker="x", ms=3,
                    label=f"est {tid}", zorder=3)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title(f"Top-down trajectories — {name}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "trajectory_topdown.png")


def _plot_tracking_error_multi(world, geo_tracks, name: str, out_dir: Path) -> Path:
    """Per-target 3D error over time, each estimated track matched to its nearest true drone."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    true_tracks = dict(getattr(world, "true_tracks", {}) or {})
    true_times = np.asarray(world.true_track_times, dtype=float)
    plotted = False
    for i, gt in enumerate(geo_tracks):
        enu, et = _geo_track_enu(world, gt)
        if enu.size == 0 or not true_tracks:
            continue
        best_src = min(true_tracks, key=lambda s: _mean_nearest(enu, np.atleast_2d(np.asarray(true_tracks[s], float))))
        truth_rs = _resample(np.atleast_2d(np.asarray(true_tracks[best_src], float)), true_times, et)
        m = min(len(enu), len(truth_rs))
        if m == 0:
            continue
        err = np.linalg.norm(enu[:m] - truth_rs[:m], axis=1)
        c = _MULTI_PALETTE[i % len(_MULTI_PALETTE)]
        tid = getattr(gt, "target_id", None) or f"T{i}"
        ax.plot(et[:m], err, color=c, lw=1.6,
                label=f"{tid}→drone{best_src} (RMSE {np.sqrt(np.mean(err**2)):.2f} m)")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no tracks", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("3D error (m)")
    ax.set_title(f"Per-target tracking error — {name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir / "tracking_error_over_time.png")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _geo_track_enu(world, gt):
    """A GeoTrack -> (ENU (T,3), times) using the world origin."""
    latlon = np.atleast_2d(np.asarray(gt.latlon, dtype=float))
    times = np.atleast_1d(np.asarray(gt.times_s, dtype=float))
    if latlon.size == 0:
        return np.empty((0, 3)), times
    origin = tuple(world.origin_latlon)
    east, north = geo.latlon_to_enu(latlon[:, 0], latlon[:, 1], origin)
    east = np.atleast_1d(np.asarray(east, dtype=float))
    north = np.atleast_1d(np.asarray(north, dtype=float))
    alt = np.atleast_1d(np.asarray(gt.altitude_m, dtype=float))
    if alt.shape[0] != east.shape[0]:
        alt = np.zeros_like(east)
    return np.column_stack([east, north, alt]), times


def _mean_nearest(path: np.ndarray, truth: np.ndarray) -> float:
    d2 = np.sum((path[:, None, :] - truth[None, :, :]) ** 2, axis=2)
    return float(np.sqrt(np.mean(np.min(d2, axis=1))))


def _resample(track: np.ndarray, src_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    src_times = np.atleast_1d(np.asarray(src_times, dtype=float))
    query_times = np.atleast_1d(np.asarray(query_times, dtype=float))
    if track.shape[0] == 0 or src_times.size == 0 or query_times.size == 0:
        return np.empty((0, 3))
    if src_times.shape == query_times.shape and np.allclose(src_times, query_times):
        return track
    order = np.argsort(src_times)
    st = src_times[order]
    return np.column_stack([np.interp(query_times, st, track[order, j]) for j in range(track.shape[1])])


def _align(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rigid (no-scale, reflection-allowed) alignment of layout onto truth."""
    est = np.asarray(est, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if est.shape != truth.shape or est.shape[0] == 0:
        return est
    try:
        sim = transforms.umeyama(est, truth, with_scaling=False, allow_reflection=True)
        aligned = np.atleast_2d(sim.apply(est))
        if not np.all(np.isfinite(aligned)):
            return est
        return aligned
    except (ValueError, np.linalg.LinAlgError):
        return est


def _estimated_track_enu(world, estimates: Estimates):
    """Estimated track in ENU meters (east, north from lat/lon; up from altitude)."""
    gt = estimates.geo_track
    latlon = np.atleast_2d(np.asarray(gt.latlon, dtype=float))
    times = np.atleast_1d(np.asarray(gt.times_s, dtype=float))
    if latlon.size == 0:
        return np.empty((0, 3)), times
    origin = tuple(world.origin_latlon)
    east, north = geo.latlon_to_enu(latlon[:, 0], latlon[:, 1], origin)
    east = np.atleast_1d(np.asarray(east, dtype=float))
    north = np.atleast_1d(np.asarray(north, dtype=float))
    alt = np.atleast_1d(np.asarray(gt.altitude_m, dtype=float))
    if alt.shape[0] != east.shape[0]:
        alt = np.zeros_like(east)
    enu = np.column_stack([east, north, alt])
    return enu, times


def _true_track_at(world, query_times: np.ndarray) -> np.ndarray:
    """True ENU track resampled at ``query_times`` (per-axis linear interpolation).

    Matching truth to the estimated-track timestamps keeps the error curve honest
    even when the two share neither length nor sampling.
    """
    true_track = np.atleast_2d(np.asarray(world.true_track, dtype=float))
    true_times = np.atleast_1d(np.asarray(world.true_track_times, dtype=float))
    query_times = np.atleast_1d(np.asarray(query_times, dtype=float))

    if true_track.size == 0 or true_times.size == 0 or query_times.size == 0:
        return np.empty((0, 3))

    # Identical sampling (the common case) -> use directly, no interpolation.
    if true_times.shape == query_times.shape and np.allclose(true_times, query_times):
        return true_track

    order = np.argsort(true_times)
    tt = true_times[order]
    cols = [np.interp(query_times, tt, true_track[order, j]) for j in range(true_track.shape[1])]
    return np.column_stack(cols)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
