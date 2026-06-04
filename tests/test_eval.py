"""Tests for the evaluation harness (`dronetracking.eval`).

The `world` argument is duck-typed (the SIM `World` is built concurrently), so
every fixture here is a `types.SimpleNamespace` with the documented attributes and
a closure standing in for `positions_matrix()`. Only the FROZEN interface and
transform/geo helpers are imported for real.
"""

from __future__ import annotations

import json
import math
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronetracking import geo
from dronetracking.estimation.interfaces import (
    ClockEstimates,
    Estimates,
    GeoTrack,
    RelativeLayout,
    Track,
)
from dronetracking.eval import alignment, metrics, report

# A fixed proper rotation + translation used to scramble the "estimated" frame.
R0 = Rotation.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()
T0 = np.array([7.0, -3.0, 2.0])
REFLECT = np.diag([1.0, 1.0, -1.0])  # improper: flips chirality


# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #
def make_world(
    *,
    device_positions,
    device_ids,
    clock_offsets,
    clock_drifts_ppm,
    origin_latlon,
    true_track,
    true_track_times,
    anchor_latlon=None,
    name="unit_scenario",
):
    """Build a duck-typed stand-in for the SIM `World`."""
    positions_local = np.asarray(device_positions, dtype=float)
    pos_map = {d: positions_local[i] for i, d in enumerate(device_ids)}

    def positions_matrix():
        return np.array([pos_map[d] for d in device_ids], dtype=float)

    return types.SimpleNamespace(
        name=name,
        device_ids=tuple(device_ids),
        device_positions=pos_map,
        clock_offsets=dict(clock_offsets),
        clock_drifts_ppm=dict(clock_drifts_ppm),
        anchor_latlon=anchor_latlon or {},
        origin_latlon=origin_latlon,
        true_track=np.asarray(true_track, dtype=float),
        true_track_times=np.asarray(true_track_times, dtype=float),
        positions_matrix=positions_matrix,
    )


def make_observations(*, device_ids, anchor_gps=()):  # minimal duck-typed stand-in
    return types.SimpleNamespace(
        device_ids=tuple(device_ids),
        ranging=(),
        acoustic=(),
        anchor_gps=tuple(anchor_gps),
        speed_of_sound_mps=343.0,
        sample_rate_hz=48000.0,
    )


def perfect_estimates(world, *, scramble=R0, translate=T0, reflect=False):
    """Build an `Estimates` that is exactly correct up to a similarity on the layout.

    The relative layout is the truth rotated/translated (and optionally reflected) —
    align_to_truth must undo it. Clocks/track/geo equal truth exactly.
    """
    R = scramble @ REFLECT if reflect else scramble
    truth_pos = world.positions_matrix()
    layout_pos = truth_pos @ R.T + translate

    layout = RelativeLayout(
        device_ids=tuple(world.device_ids),
        positions_local=layout_pos,
        covariances=None,
    )

    ref = world.device_ids[0]
    clocks = ClockEstimates(
        device_ids=tuple(world.device_ids),
        offsets_s=dict(world.clock_offsets),
        drifts_ppm=dict(world.clock_drifts_ppm),
        reference_id=ref,
    )

    T = world.true_track.shape[0]
    # The track lives in the SAME arbitrary layout frame as the devices (as in the real
    # pipeline). compute_metrics maps it back to truth via the geometry alignment.
    track_layout = world.true_track @ R.T + translate
    track = Track(
        times_s=world.true_track_times.copy(),
        positions_local=track_layout,
        covariances=np.tile(np.eye(3), (T, 1, 1)),
    )

    lat, lon = geo.enu_to_latlon(
        world.true_track[:, 0], world.true_track[:, 1], world.origin_latlon
    )
    geo_track = GeoTrack(
        times_s=world.true_track_times.copy(),
        latlon=np.column_stack([lat, lon]),
        altitude_m=world.true_track[:, 2].copy(),
        cov_enu=np.tile(np.eye(3), (T, 1, 1)),
    )

    return Estimates(layout=layout, clocks=clocks, track=track, geo_track=geo_track)


@pytest.fixture
def world():
    device_ids = ("dev0", "dev1", "dev2", "dev3", "dev4")
    device_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [50.0, 0.0, 1.0],
            [0.0, 50.0, -2.0],
            [50.0, 50.0, 0.5],
            [25.0, 25.0, 3.0],
        ]
    )
    # dev0 is the reference: true offset 0, drift 0 (per the locked convention).
    clock_offsets = {"dev0": 0.0, "dev1": 1e-3, "dev2": -2e-3, "dev3": 5e-4, "dev4": 3e-3}
    clock_drifts_ppm = {"dev0": 0.0, "dev1": 12.0, "dev2": -8.0, "dev3": 4.0, "dev4": -15.0}
    t = np.linspace(0.0, 9.0, 10)
    true_track = np.column_stack(
        [10.0 + 2.0 * t, 5.0 + 1.0 * t, 30.0 + 0.5 * np.sin(t)]
    )
    return make_world(
        device_ids=device_ids,
        device_positions=device_positions,
        clock_offsets=clock_offsets,
        clock_drifts_ppm=clock_drifts_ppm,
        origin_latlon=(32.0853, 34.7818),  # Tel Aviv-ish
        true_track=true_track,
        true_track_times=t,
    )


# --------------------------------------------------------------------------- #
# align_to_truth
# --------------------------------------------------------------------------- #
def test_align_to_truth_recovers_rotation_translation():
    truth = np.random.default_rng(1).standard_normal((12, 3)) * 10.0
    estimated = truth @ R0.T + T0  # estimated frame is a rotated/translated truth
    sim = alignment.align_to_truth(estimated, truth)
    aligned = sim.apply(estimated)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - truth) ** 2, axis=1))))
    assert rmse == pytest.approx(0.0, abs=1e-9)
    assert not sim.is_reflection


def test_align_to_truth_handles_reflection():
    truth = np.random.default_rng(2).standard_normal((12, 3)) * 10.0
    estimated = truth @ (R0 @ REFLECT).T + T0  # improper: needs reflection to fit
    sim = alignment.align_to_truth(estimated, truth)
    aligned = sim.apply(estimated)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - truth) ** 2, axis=1))))
    assert rmse == pytest.approx(0.0, abs=1e-9)
    assert sim.is_reflection  # reflection flag correctly set


def test_align_to_truth_no_scaling():
    truth = np.random.default_rng(3).standard_normal((10, 3)) * 5.0
    estimated = 3.0 * (truth @ R0.T) + T0  # scaled — but with_scaling=False
    sim = alignment.align_to_truth(estimated, truth)
    assert sim.scale == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# compute_metrics — perfect case
# --------------------------------------------------------------------------- #
def test_compute_metrics_perfect_no_reflection(world):
    est = perfect_estimates(world, reflect=False)
    obs = make_observations(device_ids=world.device_ids, anchor_gps=(1, 2, 3))
    m = metrics.compute_metrics(world, obs, est)

    # device localization
    assert m["device_localization.rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["device_localization.max_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["device_localization.alignment_scale"] == pytest.approx(1.0, abs=1e-9)
    assert m["device_localization.alignment_was_reflected"] is False

    # clock sync (gauge removed)
    assert m["clock_sync.offset_rmse_s"] == pytest.approx(0.0, abs=1e-12)
    assert m["clock_sync.drift_rmse_ppm"] == pytest.approx(0.0, abs=1e-9)

    # tracking
    assert m["tracking.rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking.rmse_xy_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking.rmse_z_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking.final_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(m["tracking.nees_mean"])  # identity cov, zero err -> 0, finite

    # georeferencing
    assert m["georeferencing.rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["georeferencing.altitude_rmse_m"] == pytest.approx(0.0, abs=1e-9)

    # scenario block
    assert m["scenario.n_devices"] == 5
    assert m["scenario.n_anchors"] == 3
    assert m["scenario.name"] == "unit_scenario"


def test_compute_metrics_perfect_with_reflection(world):
    est = perfect_estimates(world, reflect=True)
    obs = make_observations(device_ids=world.device_ids)
    m = metrics.compute_metrics(world, obs, est)
    # A reflected layout still aligns to truth exactly (chirality unobservable).
    assert m["device_localization.rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["device_localization.alignment_was_reflected"] is True


def test_compute_metrics_is_json_serializable_and_flat(world):
    est = perfect_estimates(world)
    obs = make_observations(device_ids=world.device_ids, anchor_gps=(1, 2, 3))
    m = metrics.compute_metrics(world, obs, est)
    # Flat dict (no nested dict/list of dicts) and JSON round-trips.
    for k, v in m.items():
        assert isinstance(k, str)
        assert not isinstance(v, dict)
    s = json.dumps(m)
    assert json.loads(s)["scenario.n_devices"] == 5


# --------------------------------------------------------------------------- #
# compute_metrics — slightly-off cases (injected error must match report)
# --------------------------------------------------------------------------- #
def test_tracking_error_matches_injection(world):
    # Identity layout frame so the xy/z split is preserved through alignment.
    est = perfect_estimates(world, scramble=np.eye(3), translate=np.zeros(3))
    # Shift every track position by +2 m in x: constant 2 m error everywhere.
    est.track.positions_local[:, 0] += 2.0
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    assert m["tracking.rmse_m"] == pytest.approx(2.0, abs=1e-9)
    assert m["tracking.rmse_xy_m"] == pytest.approx(2.0, abs=1e-9)
    assert m["tracking.rmse_z_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking.final_error_m"] == pytest.approx(2.0, abs=1e-9)


def test_tracking_z_error_matches_injection(world):
    # Identity layout frame so a pure-vertical injection stays pure-vertical.
    est = perfect_estimates(world, scramble=np.eye(3), translate=np.zeros(3))
    est.track.positions_local[:, 2] += 0.5  # pure vertical error
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    assert m["tracking.rmse_z_m"] == pytest.approx(0.5, abs=1e-9)
    assert m["tracking.rmse_xy_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["tracking.rmse_m"] == pytest.approx(0.5, abs=1e-9)


def test_clock_offset_error_matches_injection(world):
    est = perfect_estimates(world)
    # Perturb a single non-reference device's offset by a known amount.
    est.clocks.offsets_s["dev2"] += 1e-3
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    # One device off by 1e-3 over N=5 devices (gauge ref = dev0, which is 0 on both
    # sides so subtracting it does not change the others): RMSE = sqrt((1e-3)^2 / 5).
    expected = math.sqrt((1e-3) ** 2 / 5)
    assert m["clock_sync.offset_rmse_s"] == pytest.approx(expected, abs=1e-12)


def test_clock_drift_error_matches_injection(world):
    est = perfect_estimates(world)
    est.clocks.drifts_ppm["dev3"] += 6.0
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    expected = math.sqrt((6.0) ** 2 / 5)
    assert m["clock_sync.drift_rmse_ppm"] == pytest.approx(expected, abs=1e-9)


def test_georeference_error_matches_injection(world):
    est = perfect_estimates(world)
    # Push altitude up by a known 1.5 m everywhere.
    est.geo_track.altitude_m += 1.5
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    assert m["georeferencing.altitude_rmse_m"] == pytest.approx(1.5, abs=1e-9)


def test_localization_error_matches_injection(world):
    # Build a layout that is truth (identity frame) plus a single offset device, so
    # alignment cannot absorb it and the residual shows up as RMSE.
    truth = world.positions_matrix()
    layout_pos = truth.copy()
    layout_pos[2] += np.array([3.0, 4.0, 0.0])  # 5 m error on one device
    est = perfect_estimates(world)
    est.layout.positions_local = layout_pos
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    # A best-fit rigid alignment absorbs part of a single-point offset into the
    # gauge (rotation + translation), so the residual is spread across devices —
    # but the perturbed device must remain the worst one and the error is real.
    assert m["device_localization.rmse_m"] > 0.5
    assert m["device_localization.max_m"] > m["device_localization.rmse_m"]
    per_dev = {
        d: m[f"device_localization.error_m.{d}"] for d in world.device_ids
    }
    assert max(per_dev, key=per_dev.get) == "dev2"  # the device we offset


# --------------------------------------------------------------------------- #
# Degraded run must not throw
# --------------------------------------------------------------------------- #
def test_compute_metrics_degraded_returns_nan_not_throw(world):
    est = perfect_estimates(world)
    # Empty track / mismatched arrays simulate a failed estimation stage.
    est.track.positions_local = np.empty((0, 3))
    est.track.times_s = np.empty((0,))
    est.track.covariances = np.empty((0, 3, 3))
    est.geo_track.latlon = np.empty((0, 2))
    est.geo_track.altitude_m = np.empty((0,))
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    assert math.isnan(m["tracking.rmse_m"])
    assert math.isnan(m["georeferencing.rmse_m"])
    # Device localization still works, so it stays finite.
    assert math.isfinite(m["device_localization.rmse_m"])


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_print_report_runs(world, capsys):
    est = perfect_estimates(world)
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    report.print_report(m)
    out = capsys.readouterr().out
    assert "device_localization" in out.lower() or "localization" in out.lower()
    assert len(out.strip()) > 0


def test_save_report_round_trips(world, tmp_path):
    est = perfect_estimates(world)
    m = metrics.compute_metrics(world, make_observations(device_ids=world.device_ids), est)
    path = tmp_path / "metrics.json"
    report.save_report(m, path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["scenario.n_devices"] == 5
    assert loaded["tracking.rmse_m"] == pytest.approx(m["tracking.rmse_m"], abs=1e-12)
