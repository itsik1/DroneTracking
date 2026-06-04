"""Interactive folium map of a drone-tracking run.

Renders an OpenStreetMap-tiled HTML map (no API key) centered on the world
origin, with toggleable layers for true vs. estimated device positions, GPS
anchors, the true vs. estimated drone track, and 95% confidence ellipses along
the estimated track.

The ``world`` argument is **duck-typed** (see the iteration-1 contract): it is
never imported here. We rely only on these attributes/methods::

    device_ids, device_positions (dict id -> (3,)),
    anchor_latlon (dict id -> (lat, lon)), origin_latlon,
    true_track (N, 3), true_track_times (N,), positions_matrix() -> (K, 3)

Geometry uses the frozen :mod:`dronetracking.geo` / :mod:`dronetracking.transforms`.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Tuple

import folium
import numpy as np
from folium.plugins import TimestampedGeoJson
from scipy.stats import chi2

from dronetracking import geo, transforms
from dronetracking.estimation.interfaces import Estimates

# Fixed epoch the (relative) sim times are mapped onto for the time slider. Fixed (not
# `now()`) so animated maps are reproducible.
_ANIM_EPOCH = datetime.datetime(2024, 1, 1, 0, 0, 0)
# Draw an evolving confidence ellipse every Nth track frame (denser would clutter).
_ANIM_ELLIPSE_EVERY = 4

# How often to draw a confidence ellipse along the estimated track (every Nth point).
_ELLIPSE_EVERY = 3
# 95% containment for a 2-DoF (horizontal) Gaussian.
_CHI2_95_DF2 = float(chi2.ppf(0.95, df=2))
# Reject covariance blocks whose semi-axes exceed this (meters) as nonphysical.
_MAX_SEMI_AXIS_M = 1.0e5

_BLUE = "#1f77b4"   # true devices
_ORANGE = "#ff7f0e"  # estimated devices
_GRAY = "#888888"   # residual lines
_GREEN = "#2ca02c"  # true track
_RED = "#d62728"    # estimated track / ellipses
# Per-target palette for multi-target maps (estimated tracks).
_TRACK_PALETTE = ["#d62728", "#9467bd", "#17becf", "#bcbd22", "#e377c2", "#8c564b"]


def render_map(world, estimates: Estimates, scenario, out_path, geo_tracks=None) -> Path:
    """Build and save an interactive folium map; return ``Path(out_path)``.

    Parameters
    ----------
    world : duck-typed
        Ground-truth world (see module docstring for the required attributes).
    estimates : Estimates
        Pipeline output (layout / track / geo_track are consumed here).
    scenario : object
        Scenario-like object; only ``.name`` is read (best-effort) for the title.
    out_path : str | os.PathLike
        Destination HTML file.
    geo_tracks : list[GeoTrack] | None
        If given (multi-target), every track is drawn in a distinct color and all true
        drone tracks (``world.true_tracks``) are shown; otherwise the single estimated
        track from ``estimates`` is drawn.
    """
    out_path = Path(out_path)
    origin = tuple(world.origin_latlon)

    fmap = folium.Map(
        location=[float(origin[0]), float(origin[1])],
        zoom_start=18,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    _add_true_devices(fmap, world)
    _add_estimated_devices(fmap, world, estimates)
    _add_anchors(fmap, world)

    multi = geo_tracks is not None and len(geo_tracks) > 1
    if multi:
        _add_multi_true_tracks(fmap, world)
        _add_multi_estimated_tracks(fmap, geo_tracks, origin)
    else:
        _add_true_track(fmap, world)
        _add_estimated_track(fmap, estimates)
        _add_confidence_ellipses(fmap, estimates, origin)

    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def _add_multi_true_tracks(fmap: folium.Map, world) -> None:
    """Draw every true drone trajectory (multi-target), one color per source."""
    fg = folium.FeatureGroup(name="True tracks", show=True)
    origin = tuple(world.origin_latlon)
    true_tracks = dict(getattr(world, "true_tracks", {}) or {})
    for src, track in sorted(true_tracks.items()):
        track = np.atleast_2d(np.asarray(track, dtype=float))
        if track.shape[0] < 2:
            continue
        lat, lon = geo.enu_to_latlon(track[:, 0], track[:, 1], origin)
        folium.PolyLine(
            locations=_stack_latlon(lat, lon), color=_GREEN, weight=3, opacity=0.85,
            tooltip=f"true drone {src}",
        ).add_to(fg)
    fg.add_to(fmap)


def _add_multi_estimated_tracks(fmap: folium.Map, geo_tracks, origin) -> None:
    """Draw every estimated track in its own color (multi-target)."""
    fg = folium.FeatureGroup(name="Estimated tracks", show=True)
    for i, gt in enumerate(geo_tracks):
        color = _TRACK_PALETTE[i % len(_TRACK_PALETTE)]
        latlon = np.atleast_2d(np.asarray(gt.latlon, dtype=float))
        if latlon.shape[0] < 2:
            continue
        locs = [[float(p[0]), float(p[1])] for p in latlon]
        tid = getattr(gt, "target_id", None) or f"T{i}"
        folium.PolyLine(
            locations=locs, color=color, weight=3, opacity=0.95, dash_array="8,8",
            tooltip=f"estimated {tid}",
        ).add_to(fg)
        folium.CircleMarker(
            location=locs[-1], radius=5, color=color, fill=True, fill_color=color,
            tooltip=f"{tid} latest",
        ).add_to(fg)
    fg.add_to(fmap)


def render_animated_map(world, estimates: Estimates, scenario, out_path) -> Path:
    """Build and save a TIME-ANIMATED folium map; return ``Path(out_path)``.

    A Leaflet TimeDimension slider plays the drone forward in time: the true (green)
    and estimated (red) tracks draw progressively with a marker at the live position,
    and the estimated 95% confidence ellipse is laid down every few frames, building an
    "uncertainty corridor" as the target moves. Stationary devices and GPS anchors are
    drawn as static context layers.

    Relative sim times are mapped onto a fixed epoch, so the output is reproducible.
    """
    out_path = Path(out_path)
    origin = tuple(world.origin_latlon)

    fmap = folium.Map(
        location=[float(origin[0]), float(origin[1])],
        zoom_start=18,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Static context (devices + anchors don't move).
    _add_true_devices(fmap, world)
    _add_anchors(fmap, world)

    features: List[dict] = []

    # True track: animated growing trail with a live marker.
    true_track = np.atleast_2d(np.asarray(world.true_track, dtype=float))
    if true_track.shape[0] >= 2:
        tlat, tlon = geo.enu_to_latlon(true_track[:, 0], true_track[:, 1], origin)
        features.append(
            _anim_line(_lonlat(tlat, tlon), _iso_times(world.true_track_times), _GREEN, "true track")
        )

    # Estimated track + evolving confidence ellipses.
    gt = estimates.geo_track
    est_latlon = np.atleast_2d(np.asarray(gt.latlon, dtype=float))
    est_times = _iso_times(gt.times_s)
    if est_latlon.shape[0] >= 2:
        features.append(
            _anim_line(
                [[float(p[1]), float(p[0])] for p in est_latlon], est_times, _RED, "estimated track"
            )
        )
        features.extend(_anim_ellipses(gt, est_latlon, est_times, origin))

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period=_anim_period(world.true_track_times),
        add_last_point=True,
        auto_play=True,
        loop=True,
        max_speed=10,
        loop_button=True,
        date_options="mm:ss",
        time_slider_drag_update=True,
        duration=None,  # features persist once shown -> trails grow, ellipses form a corridor
    ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


# --------------------------------------------------------------------------- #
# Layer builders
# --------------------------------------------------------------------------- #


def _add_true_devices(fmap: folium.Map, world) -> None:
    fg = folium.FeatureGroup(name="True devices", show=True)
    origin = tuple(world.origin_latlon)
    for did in world.device_ids:
        pos = np.asarray(world.device_positions[did], dtype=float)
        lat, lon = geo.enu_to_latlon(pos[0], pos[1], origin)
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=6,
            color=_BLUE,
            fill=True,
            fill_color=_BLUE,
            fill_opacity=0.9,
            tooltip=f"true {did}",
            popup=f"true {did}: ENU ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m",
        ).add_to(fg)
    fg.add_to(fmap)


def _add_estimated_devices(fmap: folium.Map, world, estimates: Estimates) -> None:
    """Estimated device positions, rigidly aligned to truth, with residual lines.

    The estimated layout lives in an arbitrary gauge-free local frame, so we align
    it onto the true device ENU positions with a rigid (no-scale, reflection-allowed)
    Umeyama fit before projecting to lat/lon.
    """
    fg = folium.FeatureGroup(name="Estimated devices", show=True)
    fg_res = folium.FeatureGroup(name="Device residuals", show=True)
    origin = tuple(world.origin_latlon)

    est_positions = np.asarray(estimates.layout.positions_local, dtype=float)
    truth = np.asarray(world.positions_matrix(), dtype=float)

    aligned = _align_layout_to_truth(est_positions, truth)

    for idx, did in enumerate(estimates.layout.device_ids):
        ep = aligned[idx]
        elat, elon = geo.enu_to_latlon(ep[0], ep[1], origin)
        folium.CircleMarker(
            location=[float(elat), float(elon)],
            radius=6,
            color=_ORANGE,
            fill=True,
            fill_color=_ORANGE,
            fill_opacity=0.9,
            tooltip=f"est {did}",
            popup=f"estimated {did}: ENU ({ep[0]:.1f}, {ep[1]:.1f}, {ep[2]:.1f}) m",
        ).add_to(fg)

        # Residual line connecting estimate to the corresponding true position.
        if did in world.device_positions:
            tp = np.asarray(world.device_positions[did], dtype=float)
            tlat, tlon = geo.enu_to_latlon(tp[0], tp[1], origin)
            folium.PolyLine(
                locations=[[float(tlat), float(tlon)], [float(elat), float(elon)]],
                color=_GRAY,
                weight=1.5,
                opacity=0.8,
            ).add_to(fg_res)

    fg.add_to(fmap)
    fg_res.add_to(fmap)


def _add_anchors(fmap: folium.Map, world) -> None:
    fg = folium.FeatureGroup(name="GPS anchors", show=True)
    for did, latlon in world.anchor_latlon.items():
        lat, lon = float(latlon[0]), float(latlon[1])
        folium.Marker(
            location=[lat, lon],
            tooltip=f"GPS anchor {did}",
            icon=folium.Icon(color="green", icon="tower-broadcast", prefix="fa"),
        ).add_to(fg)
    fg.add_to(fmap)


def _add_true_track(fmap: folium.Map, world) -> None:
    fg = folium.FeatureGroup(name="True track", show=True)
    origin = tuple(world.origin_latlon)
    track = np.asarray(world.true_track, dtype=float)
    if track.size:
        track = np.atleast_2d(track)
        lat, lon = geo.enu_to_latlon(track[:, 0], track[:, 1], origin)
        locs = _stack_latlon(lat, lon)
        if len(locs) >= 2:
            folium.PolyLine(locations=locs, color=_GREEN, weight=3, opacity=0.9,
                            tooltip="true track").add_to(fg)
        elif len(locs) == 1:
            folium.CircleMarker(location=locs[0], radius=4, color=_GREEN, fill=True,
                                fill_color=_GREEN, tooltip="true track").add_to(fg)
    fg.add_to(fmap)


def _add_estimated_track(fmap: folium.Map, estimates: Estimates) -> None:
    fg = folium.FeatureGroup(name="Estimated track", show=True)
    latlon = np.asarray(estimates.geo_track.latlon, dtype=float)
    if latlon.size:
        latlon = np.atleast_2d(latlon)
        locs = [[float(p[0]), float(p[1])] for p in latlon]
        if len(locs) >= 2:
            folium.PolyLine(
                locations=locs,
                color=_RED,
                weight=3,
                opacity=0.9,
                dash_array="8,8",
                tooltip="estimated track",
            ).add_to(fg)
        elif len(locs) == 1:
            folium.CircleMarker(location=locs[0], radius=4, color=_RED, fill=True,
                                fill_color=_RED, tooltip="estimated track").add_to(fg)
    fg.add_to(fmap)


def _add_confidence_ellipses(fmap: folium.Map, estimates: Estimates, origin: Tuple[float, float]) -> None:
    """Draw 95% horizontal confidence ellipses along the estimated track.

    For every Nth track point we take the 2x2 horizontal block of ``cov_enu``,
    eigendecompose it, build a 64-point ellipse ring *in ENU meters* centered on
    the track's ENU position (recovered from its lat/lon), then project each ring
    point back to lat/lon and draw a closed polygon. Degenerate, non-finite, or
    nonphysically huge covariances are skipped.
    """
    fg = folium.FeatureGroup(name="Track 95% ellipses", show=True)

    gt = estimates.geo_track
    latlon = np.atleast_2d(np.asarray(gt.latlon, dtype=float))
    cov_enu = np.asarray(gt.cov_enu, dtype=float)
    n = latlon.shape[0]
    if n == 0 or cov_enu.ndim != 3:
        fg.add_to(fmap)
        return

    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])  # (64, 2)

    for k in range(min(n, cov_enu.shape[0])):
        if k % _ELLIPSE_EVERY != 0:
            continue
        ring = _ellipse_ring_enu(cov_enu[k], unit_circle)
        if ring is None:
            continue  # degenerate / non-finite / huge -> skip

        # Recover the track ENU center from its lat/lon, offset the ring, reproject.
        center_lat, center_lon = float(latlon[k, 0]), float(latlon[k, 1])
        ce, cn = geo.latlon_to_enu(center_lat, center_lon, origin)
        ring_e = ring[:, 0] + float(ce)
        ring_n = ring[:, 1] + float(cn)
        rlat, rlon = geo.enu_to_latlon(ring_e, ring_n, origin)
        poly = _stack_latlon(rlat, rlon)
        if len(poly) < 3:
            continue
        folium.Polygon(
            locations=poly,
            color=_RED,
            weight=1,
            fill=True,
            fill_color=_RED,
            fill_opacity=0.12,
            tooltip=f"95% ellipse @ step {k}",
        ).add_to(fg)

    fg.add_to(fmap)


# --------------------------------------------------------------------------- #
# Numerics helpers
# --------------------------------------------------------------------------- #


def _align_layout_to_truth(est_positions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rigidly align the estimated layout onto the true ENU positions.

    Uses ``umeyama(est, truth, with_scaling=False, allow_reflection=True)`` per the
    contract (distance geometry cannot observe chirality, so reflection is allowed).
    Falls back to the raw estimate if the fit is not computable.
    """
    est_positions = np.asarray(est_positions, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if est_positions.shape != truth.shape or est_positions.shape[0] == 0:
        return est_positions
    try:
        sim = transforms.umeyama(
            est_positions, truth, with_scaling=False, allow_reflection=True
        )
        aligned = sim.apply(est_positions)
        if not np.all(np.isfinite(aligned)):
            return est_positions
        return np.atleast_2d(aligned)
    except (ValueError, np.linalg.LinAlgError):
        return est_positions


def _ellipse_ring_enu(cov3: np.ndarray, unit_circle: np.ndarray):
    """Return a (64, 2) ENU-meter ellipse ring for the 95% region, or ``None``.

    ``cov3`` is the (3, 3) ENU covariance; only its horizontal (east/north) 2x2
    block is used. Returns ``None`` for non-finite, non-positive, or nonphysically
    large covariances so the caller can skip them.
    """
    cov3 = np.asarray(cov3, dtype=float)
    if cov3.shape != (3, 3) or not np.all(np.isfinite(cov3)):
        return None

    block = cov3[:2, :2]
    if not np.all(np.isfinite(block)):
        return None

    # Symmetrize defensively before eigh.
    block = 0.5 * (block + block.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(block)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return None
    # Degenerate (singular/near-zero) or negative-definite -> skip.
    if np.any(eigvals <= 0) or np.min(eigvals) < 1e-9:
        return None

    semi = np.sqrt(_CHI2_95_DF2) * np.sqrt(eigvals)  # (2,) semi-axis lengths (m)
    if not np.all(np.isfinite(semi)) or np.max(semi) > _MAX_SEMI_AXIS_M:
        return None

    # ring = R @ (semi * unit_circle); columns of eigvecs are the axes directions.
    ring = (unit_circle * semi) @ eigvecs.T  # (64, 2)
    if not np.all(np.isfinite(ring)):
        return None
    return ring


def _stack_latlon(lat, lon) -> List[List[float]]:
    """Pack possibly-scalar lat/lon arrays into a list of ``[lat, lon]`` pairs."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    return [[float(a), float(b)] for a, b in zip(lat, lon)]


# --------------------------------------------------------------------------- #
# Animation helpers (TimestampedGeoJson)
# --------------------------------------------------------------------------- #


def _iso_times(times_s) -> List[str]:
    """Map relative sim times (seconds) onto ISO timestamps for the time slider."""
    times = np.atleast_1d(np.asarray(times_s, dtype=float))
    return [(_ANIM_EPOCH + datetime.timedelta(seconds=float(t))).isoformat() for t in times]


def _anim_period(times_s) -> str:
    """ISO-8601 period between frames, derived from the median time step."""
    times = np.atleast_1d(np.asarray(times_s, dtype=float))
    dt = float(np.median(np.diff(times))) if times.size >= 2 else 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    return f"PT{dt:g}S"


def _lonlat(lat, lon) -> List[List[float]]:
    """Pack lat/lon arrays into GeoJSON ``[lon, lat]`` coordinate order."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    return [[float(b), float(a)] for a, b in zip(lat, lon)]


def _anim_line(coords_lonlat: List[List[float]], times: List[str], color: str, label: str) -> dict:
    """A TimestampedGeoJson LineString feature (one timestamp per coordinate)."""
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords_lonlat},
        "properties": {
            "times": list(times),
            "style": {"color": color, "weight": 4, "opacity": 0.9},
            "tooltip": label,
        },
    }


def _anim_ellipses(geo_track, est_latlon: np.ndarray, est_times: List[str], origin) -> List[dict]:
    """Per-frame 95% ellipse polygons (each tagged with its frame's timestamp)."""
    cov_enu = np.asarray(geo_track.cov_enu, dtype=float)
    if cov_enu.ndim != 3:
        return []
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    feats: List[dict] = []
    for k in range(min(est_latlon.shape[0], cov_enu.shape[0])):
        if k % _ANIM_ELLIPSE_EVERY != 0:
            continue
        ring = _ellipse_ring_enu(cov_enu[k], unit_circle)
        if ring is None:
            continue
        clat, clon = float(est_latlon[k, 0]), float(est_latlon[k, 1])
        ce, cn = geo.latlon_to_enu(clat, clon, origin)
        rlat, rlon = geo.enu_to_latlon(ring[:, 0] + float(ce), ring[:, 1] + float(cn), origin)
        ring_lonlat = _lonlat(rlat, rlon)
        if len(ring_lonlat) < 3:
            continue
        ring_lonlat.append(ring_lonlat[0])  # close the polygon ring
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring_lonlat]},
                "properties": {
                    "times": [est_times[k]],
                    "style": {"color": _RED, "weight": 1},
                    "fill": True,
                    "fillColor": _RED,
                    "fillOpacity": 0.12,
                },
            }
        )
    return feats
