"""Acceptance tests for the GEOREFERENCING estimation stage.

All fixtures are hand-built — no `dronetracking.sim` import is needed. We import
the ``AnchorGps`` *dataclass* only (a frozen contract type, not sim logic) to
build perfect GPS anchors.

Strategy: pick an origin and TRUE device ENU positions. Perfect anchors are made
by mapping each anchor's true ENU back to lat/lon (the geo projection is an exact
analytic inverse, so ``solve_transform`` reconstructs the ENU targets to machine
precision). The local frame is the true ENU scrambled by a KNOWN rotation +
translation ``Similarity``; ``solve_transform`` must recover its inverse, mapping
the scrambled local positions back onto the true ENU.
"""

from __future__ import annotations

import numpy as np
import pytest

from dronetracking import geo
from dronetracking.estimation import georeference
from dronetracking.estimation.interfaces import RelativeLayout, Track
from dronetracking.sim.observations import AnchorGps
from dronetracking.transforms import Similarity

ORIGIN = (32.0853, 34.7818)  # (lat, lon)

# TRUE device ENU positions (east, north, up) in meters. Devices 0-3 are the
# anchors; they are non-collinear (a clear 3D spread, not all in one line/plane).
TRUE_ENU = {
    "dev0": np.array([0.0, 0.0, 0.0]),
    "dev1": np.array([120.0, 5.0, 2.0]),
    "dev2": np.array([10.0, 90.0, 4.0]),
    "dev3": np.array([60.0, 70.0, 25.0]),
    "dev4": np.array([-30.0, 40.0, 8.0]),  # extra device, not an anchor
}
ANCHOR_IDS = ("dev0", "dev1", "dev2", "dev3")


def _rotation_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Proper rotation (det = +1, no reflection) from intrinsic Z-Y-X angles (rad)."""
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


# KNOWN forward transform: ENU -> LOCAL (a rigid scramble of the world frame).
KNOWN_R = _rotation_zyx(0.7, -0.35, 0.9)
KNOWN_T = np.array([17.0, -42.0, 6.5])
KNOWN_ENU_TO_LOCAL = Similarity(R=KNOWN_R, t=KNOWN_T, scale=1.0)


def _anchor_gps_perfect() -> list[AnchorGps]:
    """Perfect GPS anchors: true ENU -> lat/lon, altitude = true up."""
    out = []
    for did in ANCHOR_IDS:
        e, n, u = TRUE_ENU[did]
        lat, lon = geo.enu_to_latlon(e, n, ORIGIN)
        out.append(AnchorGps(device_id=did, lat=float(lat), lon=float(lon), altitude_m=float(u)))
    return out


def _layout_from_known() -> RelativeLayout:
    """Layout positions = KNOWN ENU->LOCAL transform applied to every true ENU."""
    ids = tuple(TRUE_ENU.keys())
    true_stack = np.stack([TRUE_ENU[d] for d in ids])  # (N, 3)
    local = KNOWN_ENU_TO_LOCAL.apply(true_stack)
    return RelativeLayout(device_ids=ids, positions_local=local, covariances=None)


def test_solve_transform_recovers_inverse_on_anchors():
    layout = _layout_from_known()
    anchor_gps = _anchor_gps_perfect()

    transform = georeference.solve_transform(layout, anchor_gps, ORIGIN, with_scaling=False)
    assert isinstance(transform, Similarity)
    assert not transform.is_reflection

    # Applying LOCAL->ENU to the layout anchor positions must return the TRUE ENU.
    local_anchors = np.stack([layout.position_of(d) for d in ANCHOR_IDS])
    recovered = transform.apply(local_anchors)
    true_anchors = np.stack([TRUE_ENU[d] for d in ANCHOR_IDS])
    err = np.max(np.abs(recovered - true_anchors))
    assert err < 1e-6, f"anchor ENU recovery error {err}"

    # It really is the inverse of the known scramble (rigid, scale 1).
    assert transform.scale == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(transform.R, KNOWN_R.T, atol=1e-9)


def test_solve_transform_with_scaling_flag():
    """A pure rigid scramble is recovered with scale ~ 1 even when scaling is enabled."""
    layout = _layout_from_known()
    anchor_gps = _anchor_gps_perfect()
    transform = georeference.solve_transform(layout, anchor_gps, ORIGIN, with_scaling=True)
    local_anchors = np.stack([layout.position_of(d) for d in ANCHOR_IDS])
    recovered = transform.apply(local_anchors)
    true_anchors = np.stack([TRUE_ENU[d] for d in ANCHOR_IDS])
    assert np.max(np.abs(recovered - true_anchors)) < 1e-6
    assert transform.scale == pytest.approx(1.0, abs=1e-6)


def test_georeference_track_recovers_true_latlon():
    layout = _layout_from_known()
    anchor_gps = _anchor_gps_perfect()
    transform = georeference.solve_transform(layout, anchor_gps, ORIGIN, with_scaling=False)

    # A few TRUE ENU track points (east, north, up).
    true_track_enu = np.array(
        [
            [5.0, 10.0, 30.0],
            [25.0, 35.0, 45.0],
            [70.0, 20.0, 60.0],
            [-15.0, 55.0, 38.0],
        ]
    )
    times = np.array([0.0, 0.5, 1.0, 1.5])

    # Scramble into the LOCAL frame with the SAME known transform.
    local_track = KNOWN_ENU_TO_LOCAL.apply(true_track_enu)
    covs = np.stack([np.diag([1.0, 2.0, 9.0]) for _ in range(len(times))])
    track = Track(times_s=times, positions_local=local_track, covariances=covs)

    geo_track = georeference.georeference_track(track, transform, ORIGIN)

    # Shapes and time passthrough.
    assert geo_track.latlon.shape == (4, 2)
    assert geo_track.altitude_m.shape == (4,)
    assert geo_track.cov_enu.shape == (4, 3, 3)
    assert np.array_equal(geo_track.times_s, times)

    # Expected lat/lon = geo.enu_to_latlon of the TRUE east/north.
    exp_lat, exp_lon = geo.enu_to_latlon(true_track_enu[:, 0], true_track_enu[:, 1], ORIGIN)
    assert np.max(np.abs(geo_track.latlon[:, 0] - exp_lat)) < 1e-6
    assert np.max(np.abs(geo_track.latlon[:, 1] - exp_lon)) < 1e-6
    # Altitude = true up.
    assert np.max(np.abs(geo_track.altitude_m - true_track_enu[:, 2])) < 1e-6

    # Covariance is rotated by R (scale 1 here): cov_enu = R @ cov_local @ R.T.
    R = transform.R
    for k in range(len(times)):
        exp_cov = R @ covs[k] @ R.T
        assert np.allclose(geo_track.cov_enu[k], exp_cov, atol=1e-9)
        # Symmetric and PSD-preserving (trace preserved under rotation).
        assert np.allclose(geo_track.cov_enu[k], geo_track.cov_enu[k].T, atol=1e-12)
        assert np.trace(geo_track.cov_enu[k]) == pytest.approx(np.trace(covs[k]), abs=1e-9)


def test_covariance_scales_with_similarity_scale():
    """When the recovered transform has scale s, cov propagates with s^2."""
    # Build a layout that is the true ENU shrunk by 0.5 then translated (with scaling).
    ids = tuple(TRUE_ENU.keys())
    true_stack = np.stack([TRUE_ENU[d] for d in ids])
    scramble = Similarity(R=KNOWN_R, t=KNOWN_T, scale=0.5)
    layout = RelativeLayout(device_ids=ids, positions_local=scramble.apply(true_stack), covariances=None)
    anchor_gps = _anchor_gps_perfect()

    transform = georeference.solve_transform(layout, anchor_gps, ORIGIN, with_scaling=True)
    # Recovered scale should be the inverse of the scramble scale (~2.0).
    assert transform.scale == pytest.approx(2.0, rel=1e-6)

    cov_local = np.diag([1.0, 1.0, 1.0])
    track = Track(
        times_s=np.array([0.0]),
        positions_local=scramble.apply(np.array([[10.0, 20.0, 30.0]])),
        covariances=cov_local[None, :, :],
    )
    geo_track = georeference.georeference_track(track, transform, ORIGIN)
    exp_cov = transform.scale**2 * (transform.R @ cov_local @ transform.R.T)
    assert np.allclose(geo_track.cov_enu[0], exp_cov, atol=1e-9)


def test_solve_transform_requires_three_anchors():
    layout = _layout_from_known()
    two_anchors = _anchor_gps_perfect()[:2]
    with pytest.raises(ValueError):
        georeference.solve_transform(layout, two_anchors, ORIGIN)


def test_solve_transform_rejects_collinear_anchors():
    # Three anchors lying exactly on a line in ENU.
    origin = ORIGIN
    collinear_enu = {
        "dev0": np.array([0.0, 0.0, 0.0]),
        "dev1": np.array([50.0, 0.0, 0.0]),
        "dev2": np.array([100.0, 0.0, 0.0]),
    }
    ids = tuple(collinear_enu.keys())
    true_stack = np.stack([collinear_enu[d] for d in ids])
    layout = RelativeLayout(
        device_ids=ids,
        positions_local=KNOWN_ENU_TO_LOCAL.apply(true_stack),
        covariances=None,
    )
    anchors = []
    for did in ids:
        e, n, u = collinear_enu[did]
        lat, lon = geo.enu_to_latlon(e, n, origin)
        anchors.append(AnchorGps(device_id=did, lat=float(lat), lon=float(lon), altitude_m=float(u)))

    with pytest.raises(ValueError):
        georeference.solve_transform(layout, anchors, origin)
