"""Acceptance tests for the visualization layer (folium map + matplotlib plots).

The ``world`` is duck-typed here via ``types.SimpleNamespace`` so these tests do
not depend on ``dronetracking.sim.world`` (built concurrently). Only the frozen
estimation interfaces and the foundation modules (geo) are imported for real.
"""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before any pyplot import

import numpy as np

from dronetracking import geo
from dronetracking.estimation.interfaces import (
    ClockEstimates,
    Estimates,
    GeoTrack,
    RelativeLayout,
    Track,
)
from dronetracking.viz import map_view, plots

ORIGIN = (32.0, 34.8)  # somewhere with a non-trivial cos(lat)


def _device_positions():
    """Four non-coplanar-ish ground anchors plus one elevated device (ENU meters)."""
    return {
        "dev0": np.array([0.0, 0.0, 0.0]),
        "dev1": np.array([100.0, 0.0, 0.0]),
        "dev2": np.array([0.0, 100.0, 0.0]),
        "dev3": np.array([100.0, 100.0, 5.0]),
    }


def _true_track():
    t = np.linspace(0.0, 9.0, 10)
    track = np.column_stack([
        20.0 + 5.0 * t,            # east
        30.0 + 2.0 * t,            # north
        50.0 + 0.0 * t,            # up
    ])
    return t, track


def _make_world(cov_enu=None):
    device_positions = _device_positions()
    device_ids = ("dev0", "dev1", "dev2", "dev3")
    # Anchor lat/lon derived from the true ENU positions so geo is self-consistent.
    anchor_latlon = {}
    for did, pos in device_positions.items():
        lat, lon = geo.enu_to_latlon(pos[0], pos[1], ORIGIN)
        anchor_latlon[did] = (float(lat), float(lon))

    times, track = _true_track()

    def positions_matrix():
        return np.array([device_positions[d] for d in device_ids])

    return types.SimpleNamespace(
        device_ids=device_ids,
        device_positions=device_positions,
        anchor_latlon=anchor_latlon,
        origin_latlon=ORIGIN,
        true_track=track,
        true_track_times=times,
        positions_matrix=positions_matrix,
    )


def _make_estimates(world, cov_enu=None):
    """Build a simple Estimates where the estimated layout equals the truth.

    The GeoTrack is built by converting the true ENU track to lat/lon, so the
    estimated track lines up with the true track.
    """
    device_ids = world.device_ids
    positions = world.positions_matrix()
    layout = RelativeLayout(device_ids=device_ids, positions_local=positions, covariances=None)

    clocks = ClockEstimates(
        device_ids=device_ids,
        offsets_s={d: 0.0 for d in device_ids},
        drifts_ppm={d: 0.0 for d in device_ids},
        reference_id=device_ids[0],
    )

    times = world.true_track_times
    track_pos = world.true_track
    T = track_pos.shape[0]
    track = Track(
        times_s=times,
        positions_local=track_pos,
        covariances=np.tile(np.eye(3) * 4.0, (T, 1, 1)),
        velocities=None,
    )

    lat, lon = geo.enu_to_latlon(track_pos[:, 0], track_pos[:, 1], world.origin_latlon)
    latlon = np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)])
    altitude = track_pos[:, 2].copy()
    if cov_enu is None:
        cov_enu = np.tile(np.diag([9.0, 4.0, 25.0]), (T, 1, 1))
    geo_track = GeoTrack(times_s=times, latlon=latlon, altitude_m=altitude, cov_enu=cov_enu)

    return Estimates(layout=layout, clocks=clocks, track=track, geo_track=geo_track)


def _scenario():
    """A minimal scenario-like object; viz only needs a name (duck-typed)."""
    return types.SimpleNamespace(name="acceptance_scenario")


# --------------------------------------------------------------------------- #
# render_map
# --------------------------------------------------------------------------- #


def test_render_map_writes_nonempty_html(tmp_path):
    world = _make_world()
    estimates = _make_estimates(world)
    out_path = tmp_path / "map.html"

    result = map_view.render_map(world, estimates, _scenario(), out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    text = out_path.read_text().lower()
    assert "folium" in text or "leaflet" in text


def test_render_map_returns_path_for_str_input(tmp_path):
    world = _make_world()
    estimates = _make_estimates(world)
    out_path = tmp_path / "map_str.html"

    from pathlib import Path

    result = map_view.render_map(world, estimates, _scenario(), str(out_path))
    assert isinstance(result, Path)
    assert result.exists()


def test_render_map_degenerate_covariance_does_not_crash(tmp_path):
    """An ellipse with a degenerate (and non-finite) covariance must be skipped, not crash."""
    world = _make_world()
    T = world.true_track.shape[0]

    cov = np.tile(np.diag([9.0, 4.0, 25.0]), (T, 1, 1)).astype(float)
    # Degenerate: zero covariance block (singular -> zero semi-axes).
    cov[1] = np.zeros((3, 3))
    # Non-finite: NaN / inf must be guarded.
    cov[2] = np.full((3, 3), np.nan)
    cov[3] = np.diag([np.inf, 1.0, 1.0])
    # Negative-definite horizontal block (eigh could give negatives) must be guarded.
    cov[4] = np.diag([-5.0, -5.0, 1.0])

    estimates = _make_estimates(world, cov_enu=cov)
    out_path = tmp_path / "map_degenerate.html"

    result = map_view.render_map(world, estimates, _scenario(), out_path)
    assert result.exists()
    assert out_path.stat().st_size > 0


def test_render_map_single_point_track(tmp_path):
    """A length-1 track (PolyLine with one vertex) must not crash rendering."""
    world = _make_world()
    # Collapse the track to a single sample.
    world.true_track = world.true_track[:1]
    world.true_track_times = world.true_track_times[:1]
    estimates = _make_estimates(world)
    out_path = tmp_path / "map_single.html"

    result = map_view.render_map(world, estimates, _scenario(), out_path)
    assert result.exists()
    assert out_path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# save_diagnostics
# --------------------------------------------------------------------------- #


def test_save_diagnostics_writes_expected_pngs(tmp_path):
    world = _make_world()
    estimates = _make_estimates(world)
    out_dir = tmp_path / "diag"

    paths = plots.save_diagnostics(world, estimates, _scenario(), out_dir)

    assert isinstance(paths, list)
    assert len(paths) >= 4  # local map, error bars, tracking error, top-down trajectory

    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"

    # Filenames must be unique (no overwrite collisions).
    assert len({p.name for p in paths}) == len(paths)


def test_save_diagnostics_creates_missing_out_dir(tmp_path):
    world = _make_world()
    estimates = _make_estimates(world)
    out_dir = tmp_path / "nested" / "does_not_exist_yet"

    paths = plots.save_diagnostics(world, estimates, _scenario(), out_dir)
    assert out_dir.exists()
    assert all(p.parent == out_dir for p in paths)
