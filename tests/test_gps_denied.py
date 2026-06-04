"""Acceptance tests for Ph9 — GPS-DENIED OPERATION.

All fixtures are hand-built — no ``dronetracking.sim`` logic is imported. We use
the ``AnchorGps`` *dataclass* only (a frozen contract type) to build perfect GPS
anchors, exactly as ``tests/test_est_georef.py`` does.

Setup mirrors the georeferencing tests:
  * Pick an ENU origin and TRUE device ENU positions (>=4 non-coplanar anchors).
  * Perfect anchors map each true ENU back to lat/lon via the exact geo inverse.
  * The LOCAL frame is the true ENU scrambled by a KNOWN rigid ``Similarity``.

The twist for Ph9 is DRIFT: the *true* LOCAL->ENU transform slowly changes over
time (e.g. the array creeps). The georeferencer can only re-observe that
transform when GPS is up; during a blackout it must HOLD the last good one
(dead-reckon) and, on GPS return, BLEND held->fresh so the georeferenced track
shows no jump.

We probe the drift by building both the LOCAL track (truth scrambled by the
time-varying transform) AND the true lat/lon, so we can measure georef error
frame by frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from dronetracking import geo
from dronetracking.estimation import georeference, gps_denied
from dronetracking.estimation.interfaces import GeoTrack, RelativeLayout, Track
from dronetracking.sim.observations import AnchorGps
from dronetracking.transforms import Similarity

ORIGIN = (32.0853, 34.7818)  # (lat, lon)

# TRUE device ENU positions (east, north, up), meters. dev0-3 are the anchors and
# are clearly non-coplanar (varied heights), so the 3D LOCAL->ENU fit is unique.
TRUE_ENU = {
    "dev0": np.array([0.0, 0.0, 0.0]),
    "dev1": np.array([180.0, 8.0, 16.0]),
    "dev2": np.array([12.0, 170.0, 4.0]),
    "dev3": np.array([95.0, 110.0, 28.0]),
    "dev4": np.array([60.0, 60.0, 2.0]),  # extra device, not an anchor
}
ANCHOR_IDS = ("dev0", "dev1", "dev2", "dev3")


def _rotation_z(yaw: float) -> np.ndarray:
    """Proper rotation about +Z (det=+1) by ``yaw`` radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# The transform we model is ENU->LOCAL (a scramble of the world frame). The
# estimator solves/holds its inverse, LOCAL->ENU, to georeference.
def _enu_to_local_at(t: float) -> Similarity:
    """TRUE time-varying ENU->LOCAL scramble: a slowly creeping rotation+translation.

    At t=0 it is a fixed baseline; thereafter the yaw rotates a touch and the
    origin creeps, so a transform solved before a blackout grows stale during it.
    """
    yaw = 0.6 + 0.01 * t  # ~0.01 rad/s yaw drift
    R = _rotation_z(yaw)
    t_vec = np.array([20.0 + 0.5 * t, -35.0 - 0.4 * t, 6.5 + 0.05 * t])  # m, creeping
    return Similarity(R=R, t=t_vec, scale=1.0)


def _local_to_enu_at(t: float) -> Similarity:
    """Inverse of :func:`_enu_to_local_at`: the TRUE LOCAL->ENU transform at ``t``.

    This is what ``transform_provider`` hands the georeferencer (only usable by it
    when GPS is up). For a rigid (scale 1) similarity y = R x + t, the inverse is
    x = R.T (y - t) = R.T y + (-R.T t).
    """
    fwd = _enu_to_local_at(t)
    R_inv = fwd.R.T
    t_inv = -R_inv @ fwd.t
    return Similarity(R=R_inv, t=t_inv, scale=1.0)


def _anchor_gps_perfect() -> list[AnchorGps]:
    """Perfect GPS anchors: true ENU -> lat/lon, altitude = true up."""
    out = []
    for did in ANCHOR_IDS:
        e, n, u = TRUE_ENU[did]
        lat, lon = geo.enu_to_latlon(e, n, ORIGIN)
        out.append(
            AnchorGps(device_id=did, lat=float(lat), lon=float(lon), altitude_m=float(u))
        )
    return out


def _layout_at(t: float) -> RelativeLayout:
    """Device layout in the LOCAL frame at time ``t`` (true ENU scrambled by drift).

    The georeferencer is given a SINGLE static layout (the t=0 one) — it does not
    know the array crept. That stale layout, fed to ``solve_transform``, is exactly
    why a re-solve after a blackout differs from the held transform.
    """
    ids = tuple(TRUE_ENU.keys())
    true_stack = np.stack([TRUE_ENU[d] for d in ids])  # (N, 3)
    local = _enu_to_local_at(t).apply(true_stack)
    return RelativeLayout(device_ids=ids, positions_local=local, covariances=None)


def _true_track_enu(times: np.ndarray) -> np.ndarray:
    """A smooth TRUE drone trajectory in ENU (east, north, up), one row per time."""
    # A gentle diagonal climb across the field.
    east = 10.0 + 6.0 * times
    north = 150.0 - 4.0 * times
    up = 40.0 + 0.5 * times
    return np.column_stack([east, north, up])


def _local_track(times: np.ndarray, true_enu: np.ndarray) -> np.ndarray:
    """Carry the TRUE ENU track into the LOCAL frame using the per-time drift.

    Frame k uses the true transform at ``times[k]`` — this is what the tracker
    would actually report in the (creeping) local frame.
    """
    local = np.empty_like(true_enu)
    for k, t in enumerate(times):
        local[k] = _enu_to_local_at(float(t)).apply(true_enu[k])
    return local


def _build_track(times: np.ndarray) -> tuple[Track, np.ndarray]:
    """Return (LOCAL-frame Track, TRUE ENU positions) sharing the same times."""
    true_enu = _true_track_enu(times)
    local = _local_track(times, true_enu)
    covs = np.stack([np.diag([1.0, 1.0, 1.0]) for _ in times])
    track = Track(times_s=times, positions_local=local, covariances=covs)
    return track, true_enu


def _geo_error_m(geo_track: GeoTrack, true_enu: np.ndarray) -> np.ndarray:
    """Per-frame horizontal+vertical georef error (m) vs the true ENU track."""
    true_lat, true_lon = geo.enu_to_latlon(true_enu[:, 0], true_enu[:, 1], ORIGIN)
    horiz = geo.haversine_m(
        geo_track.latlon[:, 0], geo_track.latlon[:, 1], true_lat, true_lon
    )
    vert = np.abs(geo_track.altitude_m - true_enu[:, 2])
    return np.sqrt(np.asarray(horiz) ** 2 + vert**2)


# --------------------------------------------------------------------------- #
# gps_status
# --------------------------------------------------------------------------- #


def test_gps_status_matches_blackout_windows():
    times = np.linspace(0.0, 10.0, 21)  # 0, 0.5, ..., 10.0
    windows = [(3.0, 6.0)]
    status = gps_denied.gps_status(times, windows)

    assert status.dtype == bool
    assert status.shape == times.shape
    # GPS is UNavailable inside the closed window [3, 6] and available elsewhere.
    expected = np.array([not (3.0 <= t <= 6.0) for t in times])
    assert np.array_equal(status, expected)


def test_gps_status_no_windows_all_available():
    times = np.linspace(0.0, 5.0, 11)
    status = gps_denied.gps_status(times, [])
    assert status.dtype == bool
    assert status.all()


def test_gps_status_multiple_windows():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    windows = [(1.0, 2.0), (4.0, 5.0)]
    status = gps_denied.gps_status(times, windows)
    expected = np.array([True, False, False, True, False, False, True])
    assert np.array_equal(status, expected)


# --------------------------------------------------------------------------- #
# georeference_with_blackout — coverage & exposure
# --------------------------------------------------------------------------- #


def test_all_frames_get_a_position_none_dropped():
    times = np.linspace(0.0, 20.0, 41)
    track, _ = _build_track(times)
    layout = _layout_at(0.0)
    anchor_gps = _anchor_gps_perfect()
    windows = [(8.0, 13.0)]

    result = gps_denied.georeference_with_blackout(
        layout, anchor_gps, track, ORIGIN, windows,
        transform_provider=_local_to_enu_at,
    )

    # It is (or behaves as) a GeoTrack covering EVERY track frame.
    assert isinstance(result, GeoTrack)
    assert result.latlon.shape == (times.size, 2)
    assert result.altitude_m.shape == (times.size,)
    assert result.cov_enu.shape == (times.size, 3, 3)
    assert np.array_equal(result.times_s, times)
    # No NaNs sneaked in.
    assert np.isfinite(result.latlon).all()
    assert np.isfinite(result.altitude_m).all()


def test_dead_reckoned_frames_are_exposed_and_match_blackout():
    times = np.linspace(0.0, 20.0, 41)
    track, _ = _build_track(times)
    layout = _layout_at(0.0)
    anchor_gps = _anchor_gps_perfect()
    windows = [(8.0, 13.0)]

    result = gps_denied.georeference_with_blackout(
        layout, anchor_gps, track, ORIGIN, windows,
        transform_provider=_local_to_enu_at,
    )

    dr = result.dead_reckoned
    assert dr.dtype == bool
    assert dr.shape == times.shape
    # Dead-reckoned exactly when GPS is unavailable (inside the closed window).
    expected_dr = np.array([8.0 <= t <= 13.0 for t in times])
    assert np.array_equal(dr, expected_dr)

    # The companion availability flag is the logical inverse where dead-reckoned.
    assert np.array_equal(result.gps_available, ~expected_dr)


# --------------------------------------------------------------------------- #
# georeference_with_blackout — continuity & error behavior
# --------------------------------------------------------------------------- #


def test_track_continuous_across_recovery_boundary():
    """No teleport: the step between consecutive ENU positions stays small even
    right where GPS returns and the transform is being re-solved/blended."""
    times = np.linspace(0.0, 20.0, 81)  # dt = 0.25 s, fine enough to bound steps
    track, _ = _build_track(times)
    layout = _layout_at(0.0)
    anchor_gps = _anchor_gps_perfect()
    windows = [(8.0, 13.0)]
    blend_s = 2.0

    result = gps_denied.georeference_with_blackout(
        layout, anchor_gps, track, ORIGIN, windows,
        recovery_blend_s=blend_s, transform_provider=_local_to_enu_at,
    )

    # Reconstruct ENU from the geo output to measure frame-to-frame steps.
    east, north = geo.latlon_to_enu(
        result.latlon[:, 0], result.latlon[:, 1], ORIGIN
    )
    enu = np.column_stack([np.asarray(east), np.asarray(north), result.altitude_m])
    steps = np.linalg.norm(np.diff(enu, axis=0), axis=1)

    # Baseline: how far the TRUE drone moves per frame (smooth motion).
    true_enu = _true_track_enu(times)
    true_steps = np.linalg.norm(np.diff(true_enu, axis=0), axis=1)
    max_true_step = float(true_steps.max())

    # The recovery boundary is the first frame at/after the window end.
    recov_idx = int(np.argmax(times >= windows[0][1]))

    # If we naively SNAPPED to the fresh transform at recovery (no blend), the
    # step at the boundary would jump by the accumulated drift. Verify there IS a
    # meaningful drift to correct, so this test isn't vacuous.
    snap_jump = _snap_boundary_jump(layout, anchor_gps, track, windows, recov_idx)
    assert snap_jump > 0.5, f"drift too small to be a meaningful test: {snap_jump} m"

    # With blending, every step (including across recovery) is on the order of the
    # true motion plus a small blended correction — nowhere near the snap jump.
    boundary_steps = steps[max(0, recov_idx - 1): recov_idx + 1]
    assert boundary_steps.max() < max(0.5, 3.0 * max_true_step), (
        f"discontinuity at recovery: step {boundary_steps.max():.3f} m "
        f"(true motion ~{max_true_step:.3f} m/frame, snap jump {snap_jump:.3f} m)"
    )
    # And globally there is no single teleport.
    assert steps.max() < max(0.5, 3.0 * max_true_step)


def _snap_boundary_jump(layout, anchor_gps, track, windows, recov_idx) -> float:
    """ENU jump at the recovery frame IF we snapped held->fresh with no blending.

    Held transform = solved at the blackout START (last good GPS). Fresh =
    re-solved at recovery. The jump is the difference in applied ENU for the
    recovery frame's local position. This is the discontinuity blending removes.
    """
    t_start = windows[0][0]
    # Last good frame before blackout (where the held transform was last refreshed).
    pre_idx = max(0, int(np.argmax(track.times_s >= t_start)) - 1)
    # held = transform last observable before the blackout; fresh = re-solved at
    # recovery. The provider supplies the TRUE (drifting) transform at each time.
    held = _local_to_enu_at(float(track.times_s[pre_idx]))
    fresh = _local_to_enu_at(float(track.times_s[recov_idx]))
    p = track.positions_local[recov_idx]
    return float(np.linalg.norm(fresh.apply(p) - held.apply(p)))


def test_error_during_blackout_bounded_and_shrinks_after_recovery():
    times = np.linspace(0.0, 24.0, 97)  # dt = 0.25 s
    track, true_enu = _build_track(times)
    layout = _layout_at(0.0)
    anchor_gps = _anchor_gps_perfect()
    windows = [(8.0, 14.0)]
    blend_s = 2.0

    result = gps_denied.georeference_with_blackout(
        layout, anchor_gps, track, ORIGIN, windows,
        recovery_blend_s=blend_s, transform_provider=_local_to_enu_at,
    )
    err = _geo_error_m(result, true_enu)

    avail = gps_denied.gps_status(times, windows)
    t0, t1 = windows[0]

    # (a) Before the blackout, GPS is live and tracking the true transform: error
    # is ~machine-zero (perfect anchors, exact inverse).
    pre = err[(times < t0)]
    assert pre.max() < 1e-6, f"pre-blackout error should be ~0, got {pre.max()}"

    # (b) During the blackout the held transform goes stale, so error GROWS but
    # stays BOUNDED (proportional to drift * blackout length, not unbounded).
    during = err[(times >= t0) & (times <= t1)]
    assert during.max() > 1e-3, "expected measurable dead-reckoning drift error"
    assert during.max() < 25.0, f"dead-reckoning error unbounded: {during.max()} m"

    # (c) After recovery + blend settles, error shrinks back toward ~0 (drift
    # corrected). Sample a frame a couple blend-windows past recovery.
    settle_t = t1 + 3.0 * blend_s
    settled = err[times >= settle_t]
    assert settled.size > 0
    # Settled error is far smaller than the worst dead-reckoned error...
    assert settled.max() < 0.25 * during.max(), (
        f"error did not shrink after recovery: settled {settled.max():.3f} m "
        f"vs during {during.max():.3f} m"
    )
    # ...and is back near zero now that GPS re-pinned the (live) transform.
    assert settled.max() < 1e-3, f"post-recovery error should re-converge, got {settled.max()}"


def test_no_blackout_matches_plain_georeference_track():
    """With an empty blackout list and a STATIC transform, the result equals the
    plain georeferencer frame for frame (and nothing is dead-reckoned)."""
    times = np.linspace(0.0, 5.0, 11)
    # Static truth so the static georeferencer is the right oracle.
    true_enu = _true_track_enu(times)
    static = _enu_to_local_at(0.0)
    local = static.apply(true_enu)
    covs = np.stack([np.diag([1.0, 2.0, 3.0]) for _ in times])
    track = Track(times_s=times, positions_local=local, covariances=covs)
    layout = _layout_at(0.0)
    anchor_gps = _anchor_gps_perfect()

    result = gps_denied.georeference_with_blackout(
        layout, anchor_gps, track, ORIGIN, []
    )
    transform = georeference.solve_transform(layout, anchor_gps, ORIGIN)
    plain = georeference.georeference_track(track, transform, ORIGIN)

    assert np.allclose(result.latlon, plain.latlon, atol=1e-9)
    assert np.allclose(result.altitude_m, plain.altitude_m, atol=1e-9)
    assert np.allclose(result.cov_enu, plain.cov_enu, atol=1e-9)
    assert not result.dead_reckoned.any()
    assert result.gps_available.all()


def test_default_transform_provider_is_static_solve_transform():
    """Without a transform_provider, the function falls back to a single static
    solve_transform and still dead-reckons through a blackout without dropping
    frames or introducing a jump."""
    times = np.linspace(0.0, 10.0, 41)
    true_enu = _true_track_enu(times)
    static = _enu_to_local_at(0.0)
    local = static.apply(true_enu)
    covs = np.stack([np.eye(3) for _ in times])
    track = Track(times_s=times, positions_local=local, covariances=covs)
    layout = _layout_at(0.0)
    anchor_gps = _anchor_gps_perfect()
    windows = [(4.0, 7.0)]

    result = gps_denied.georeference_with_blackout(
        layout, anchor_gps, track, ORIGIN, windows
    )
    # All frames covered, dead-reckoning flagged in-window.
    assert result.latlon.shape == (times.size, 2)
    expected_dr = np.array([4.0 <= t <= 7.0 for t in times])
    assert np.array_equal(result.dead_reckoned, expected_dr)

    # With a STATIC truth the held transform never goes stale, so georef error is
    # ~0 throughout — including during the blackout (dead-reckoning a static frame
    # is exact) and across recovery (no jump).
    err = _geo_error_m(result, true_enu)
    assert err.max() < 1e-6, f"static-truth dead-reckoning should be exact, got {err.max()}"
