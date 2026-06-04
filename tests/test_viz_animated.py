"""Tests for the animated (time-slider) folium map."""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")

import numpy as np

from dronetracking import geo
from dronetracking.estimation.interfaces import (
    ClockEstimates,
    Estimates,
    GeoTrack,
    RelativeLayout,
    Track,
)
from dronetracking.viz.map_view import render_animated_map

ORIGIN = (32.0853, 34.7818)


def _world_and_estimates(T=12):
    ids = ("dev0", "dev1", "dev2", "dev3")
    pos = {
        "dev0": np.array([0.0, 0.0, 0.0]),
        "dev1": np.array([100.0, 0.0, 5.0]),
        "dev2": np.array([100.0, 100.0, 2.0]),
        "dev3": np.array([0.0, 100.0, 8.0]),
    }
    t = np.linspace(0.0, 6.0, T)
    true_track = np.column_stack([10 + 5 * t, 5 + 3 * t, 50 + 0 * t])
    alat, alon = geo.enu_to_latlon(0.0, 0.0, ORIGIN)
    world = types.SimpleNamespace(
        origin_latlon=ORIGIN,
        device_ids=ids,
        device_positions=pos,
        anchor_latlon={"dev0": (float(alat), float(alon))},
        true_track=true_track,
        true_track_times=t,
        positions_matrix=lambda: np.array([pos[d] for d in ids]),
    )
    lat, lon = geo.enu_to_latlon(true_track[:, 0], true_track[:, 1], ORIGIN)
    geo_track = GeoTrack(
        times_s=t,
        latlon=np.column_stack([lat, lon]),
        altitude_m=true_track[:, 2].copy(),
        cov_enu=np.tile(np.diag([4.0, 4.0, 9.0]), (T, 1, 1)),
    )
    est = Estimates(
        layout=RelativeLayout(ids, np.array([pos[d] for d in ids]), None),
        clocks=ClockEstimates(ids, {d: 0.0 for d in ids}, {d: 0.0 for d in ids}, "dev0"),
        track=Track(times_s=t, positions_local=true_track.copy(), covariances=np.tile(np.eye(3), (T, 1, 1))),
        geo_track=geo_track,
    )
    return world, est


def test_render_animated_map_writes_timestamped_html(tmp_path):
    world, est = _world_and_estimates()
    out = render_animated_map(world, est, types.SimpleNamespace(name="t"), tmp_path / "anim.html")
    assert out.exists() and out.stat().st_size > 0
    html = out.read_text().lower()
    # folium's TimestampedGeoJson injects the Leaflet TimeDimension control.
    assert "timedimension" in html or "timestamp" in html
    assert "leaflet" in html


def test_render_animated_map_survives_degenerate_covariance(tmp_path):
    world, est = _world_and_estimates()
    est.geo_track.cov_enu[2] = np.full((3, 3), np.nan)  # a bad frame must not crash rendering
    out = render_animated_map(world, est, types.SimpleNamespace(name="t"), tmp_path / "anim2.html")
    assert out.exists() and out.stat().st_size > 0
