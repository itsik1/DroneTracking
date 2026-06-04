"""Acceptance tests for Ph3 — moving devices / continuous geometry.

Covers the moving-device ranging simulator (:mod:`dronetracking.sim.device_motion`)
and the windowed geometry tracker
(:mod:`dronetracking.estimation.geometry_tracking`).

Per the ground-truth firewall, estimation *source* must never import
``dronetracking.sim``; these *tests* may import the frozen sim leaf functions to
build realistic fixtures and to score against truth — which is exactly what we do
here (``generate_moving_ranging`` / ``device_positions_at`` for the truth side,
``track_geometry`` / ``estimate_velocities`` for the estimate side).

Tolerances (drift-free, noise-free moving array):
- Each window's recovered layout, after similarity alignment (reflection allowed)
  to that window's TRUE device positions, has per-device RMSE comfortably below a
  centimetre — the only error is the within-window motion smear.
- Recovered per-device velocity matches truth in both direction (cosine ~1) and
  magnitude (within a small fraction + absolute floor).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from dronetracking.config import load_scenario
from dronetracking.sim.scenario import DeviceSpec, NoiseSpec, Scenario, TrajectorySpec
from dronetracking.transforms import umeyama

from dronetracking.sim.device_motion import (
    device_positions_at,
    generate_moving_ranging,
)
from dronetracking.estimation.geometry_tracking import (
    estimate_velocities,
    track_geometry,
)

SCN = Path(__file__).resolve().parents[1] / "scenarios"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _moving_scenario(velocities, *, duration_s=30.0, rounds=90):
    """A clean (drift-free, noise-free) moving-device scenario.

    Six devices over a ~200 m field. ``dev0..dev3`` are GPS anchors with good
    horizontal *and vertical* spread (heights 0/30/8/20 m) — a well-conditioned
    3-D gauge — and ``dev4..dev5`` are interior nodes. The caller supplies a
    per-device constant ``velocity_mps`` (length 6); typically the four anchors
    stay put so the tracker can pin a stable common frame to them while the
    interior nodes drift. Clocks are offset-free and drift-free so two-way
    ranging recovers the geometric distance exactly and the initiator transmit
    timestamp ``t1_local_i`` equals global time (clean windowing).
    """
    base = [
        (0.0, 0.0, 0.0),
        (200.0, 0.0, 30.0),
        (200.0, 200.0, 8.0),
        (0.0, 200.0, 20.0),
        (100.0, 100.0, 15.0),
        (120.0, 80.0, 5.0),
    ]
    devices = tuple(
        DeviceSpec(
            id=f"dev{i}",
            position_m=base[i],
            clock_offset_s=0.0,
            clock_drift_ppm=0.0,
            proc_delay_s=0.002,
            has_gps=(i < 4),
            velocity_mps=velocities[i],
        )
        for i in range(len(base))
    )
    trajectory = TrajectorySpec(
        kind="linear",
        z_m=60.0,
        params={"start_m": [-50.0, 50.0], "end_m": [250.0, 150.0]},
    )
    return Scenario(
        name="moving_test",
        seed=7,
        speed_of_sound_mps=343.0,
        sample_rate_hz=48000.0,
        duration_s=duration_s,
        dt_s=0.5,
        ranging_rounds=rounds,
        origin_latlon=(32.0853, 34.7818),
        devices=devices,
        trajectory=trajectory,
        noise=NoiseSpec(),
    )


def _true_positions_at(scenario, t):
    """(K, 3) true device positions at time ``t`` in scenario id order."""
    pos = device_positions_at(scenario, t)
    return np.array([pos[d.id] for d in scenario.devices], dtype=float)


def _window_rmse_to_truth(layout, scenario, t):
    """Per-device RMSE of a window's layout vs truth at the window centre.

    A relative layout is gauge-free up to rotation/translation/reflection, so we
    align it to the true positions with umeyama (no scaling, reflection allowed)
    before scoring.
    """
    P = layout.positions_local
    truth = _true_positions_at(scenario, t)
    sim = umeyama(P, truth, with_scaling=False, allow_reflection=True)
    aligned = sim.apply(P)
    per_device = np.linalg.norm(aligned - truth, axis=1)
    return float(np.sqrt(np.mean(per_device**2)))


def _align_series_to_world(series, scenario):
    """Map a tracked series into the true world frame with one rigid transform.

    ``track_geometry`` returns layouts in an arbitrary but *consistent*
    gauge-free frame (rotation/translation/reflection). Velocity is frame
    dependent, so to compare recovered velocities against the true world-frame
    ``velocity_mps`` we first fix that single gauge: solve one umeyama (no
    scaling, reflection allowed) from the FIRST window's layout onto its true
    positions, then apply that same transform to every window. Returns a new
    ``[(t, RelativeLayout-in-world-frame), ...]``.
    """
    from dronetracking.estimation.interfaces import RelativeLayout

    t0, first = series[0]
    truth0 = _true_positions_at(scenario, t0)
    gauge = umeyama(
        first.positions_local, truth0, with_scaling=False, allow_reflection=True
    )
    out = []
    for t, layout in series:
        moved = gauge.apply(np.asarray(layout.positions_local, dtype=float))
        out.append((t, RelativeLayout(device_ids=layout.device_ids, positions_local=moved)))
    return out


# --------------------------------------------------------------------------- #
# sim.device_motion
# --------------------------------------------------------------------------- #
def test_device_positions_at_drifts_linearly():
    vels = [(1.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 5
    sc = _moving_scenario(vels)

    at0 = device_positions_at(sc, 0.0)
    at10 = device_positions_at(sc, 10.0)

    # dev0 moved +10 m in x; a static device did not move at all.
    assert np.allclose(at0["dev0"], [0.0, 0.0, 0.0])
    assert np.allclose(at10["dev0"], [10.0, 0.0, 0.0])
    assert np.allclose(at10["dev1"], at0["dev1"])
    # keys cover every device, values are (3,)
    assert set(at0) == set(sc.device_ids)
    assert at0["dev0"].shape == (3,)


def test_generate_moving_ranging_distance_grows_with_motion():
    """Two devices closing on each other: the recovered range shrinks over time.

    With clean clocks the classic estimator ToF = 0.5*((t4-t1)-(t3-t2)) recovers
    the geometric distance at each round's transmit time, so a round late in the
    run must read a shorter range than an early round.
    """
    # dev0 moves +x toward dev1 (at x=200); everyone else static.
    vels = [(2.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 5
    sc = _moving_scenario(vels, duration_s=20.0, rounds=2)
    rng = np.random.default_rng(0)
    records = generate_moving_ranging(sc, rng)

    c = sc.speed_of_sound_mps

    def recovered_range(rec):
        tof = 0.5 * ((rec.t4_local_i - rec.t1_local_i) - (rec.t3_local_j - rec.t2_local_j))
        return tof * c

    pair = [r for r in records if {r.initiator, r.responder} == {"dev0", "dev1"}]
    pair.sort(key=lambda r: r.round_idx)
    early, late = pair[0], pair[-1]

    # Truth: at t=0 distance is sqrt(200^2+18^2); at t=20 dev0 is at x=40 so it is
    # 160 m closer in x. Range must shrink by ~ (distance(0) - distance(20)).
    d0 = float(np.linalg.norm(_true_positions_at(sc, 0.0)[0] - _true_positions_at(sc, 0.0)[1]))
    d20 = float(np.linalg.norm(_true_positions_at(sc, 20.0)[0] - _true_positions_at(sc, 20.0)[1]))
    assert d20 < d0  # sanity on the fixture
    assert recovered_range(early) == pytest.approx(d0, abs=1e-6)
    assert recovered_range(late) == pytest.approx(d20, abs=1e-6)


def test_generate_moving_ranging_spreads_over_duration():
    """Exchanges are spread across the whole duration so motion is observable."""
    vels = [(1.0, 0.0, 0.0)] + [(0.0, 0.0, 0.0)] * 5
    sc = _moving_scenario(vels, duration_s=30.0, rounds=60)
    records = generate_moving_ranging(sc, np.random.default_rng(0))

    t1s = np.array([r.t1_local_i for r in records])
    assert t1s.min() == pytest.approx(0.0, abs=1e-9)
    assert t1s.max() == pytest.approx(sc.duration_s, abs=1e-6)
    # every unordered pair present at every round
    K = len(sc.device_ids)
    assert len(records) == K * (K - 1) // 2 * sc.ranging_rounds


# --------------------------------------------------------------------------- #
# estimation.geometry_tracking — per-window layout accuracy
# --------------------------------------------------------------------------- #
def test_track_geometry_returns_time_series():
    # Anchors dev0..3 fixed; the two interior nodes drift.
    vels = [(0.0, 0.0, 0.0)] * 4 + [(0.5, 0.0, 0.0), (0.0, -0.5, 0.0)]
    sc = _moving_scenario(vels, duration_s=30.0, rounds=90)
    records = generate_moving_ranging(sc, np.random.default_rng(0))

    series = track_geometry(
        records,
        device_ids=sc.device_ids,
        speed_of_sound=sc.speed_of_sound_mps,
        window_s=6.0,
        step_s=3.0,
    )
    assert len(series) >= 3
    # monotonically increasing window centres
    centers = [t for t, _ in series]
    assert all(centers[k] < centers[k + 1] for k in range(len(centers) - 1))
    for t, layout in series:
        assert layout.device_ids == sc.device_ids
        assert layout.positions_local.shape == (len(sc.device_ids), 3)


def test_each_window_layout_matches_true_positions():
    """Each window's layout, aligned to that window's TRUE positions, has small RMSE."""
    # Anchors dev0..3 fixed; interior nodes drift (incl. a vertical mover).
    vels = [(0.0, 0.0, 0.0)] * 4 + [(0.8, 0.0, 0.0), (0.0, -0.6, 0.4)]
    sc = _moving_scenario(vels, duration_s=30.0, rounds=120)
    records = generate_moving_ranging(sc, np.random.default_rng(0))

    series = track_geometry(
        records,
        device_ids=sc.device_ids,
        speed_of_sound=sc.speed_of_sound_mps,
        window_s=5.0,
        step_s=2.5,
    )
    assert len(series) >= 4
    for t, layout in series:
        rmse = _window_rmse_to_truth(layout, sc, t)
        # The only error is the within-window motion smear (a few cm at <=1 m/s
        # over a 5 s window). No noise, no drift -> sub-decimetre comfortably.
        assert rmse < 0.25, f"window @ t={t:.2f}: RMSE {rmse:.4f} m too high"


# --------------------------------------------------------------------------- #
# estimation.geometry_tracking — recovered velocity
# --------------------------------------------------------------------------- #
def test_recovered_velocities_match_truth_direction_and_magnitude():
    # Four fixed anchors (dev0..3) define the gauge; two interior movers.
    vels = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -0.8, 0.0),
    ]
    sc = _moving_scenario(vels, duration_s=40.0, rounds=160)
    records = generate_moving_ranging(sc, np.random.default_rng(0))

    series = track_geometry(
        records,
        device_ids=sc.device_ids,
        speed_of_sound=sc.speed_of_sound_mps,
        window_s=5.0,
        step_s=2.5,
    )
    # Velocities are frame-dependent: fix the gauge to the world frame first.
    est = estimate_velocities(_align_series_to_world(series, sc))

    truth = {f"dev{i}": np.asarray(vels[i], dtype=float) for i in range(len(vels))}
    for dev_id, v_true in truth.items():
        v_est = np.asarray(est[dev_id], dtype=float)
        speed_true = float(np.linalg.norm(v_true))
        speed_est = float(np.linalg.norm(v_est))

        # Magnitude: within 0.2 m/s absolute (covers the static devices too).
        assert abs(speed_est - speed_true) < 0.2, (
            f"{dev_id}: |v| est {speed_est:.3f} vs true {speed_true:.3f}"
        )
        # Direction: for genuinely moving devices the unit vectors must align.
        if speed_true > 0.2:
            cos = float(np.dot(v_true, v_est) / (speed_true * speed_est + 1e-12))
            assert cos > 0.9, f"{dev_id}: direction cos {cos:.3f} too low"


def test_velocities_zero_for_static_array():
    """A fully static array yields ~zero recovered velocity for every device."""
    vels = [(0.0, 0.0, 0.0)] * 6
    sc = _moving_scenario(vels, duration_s=30.0, rounds=120)
    records = generate_moving_ranging(sc, np.random.default_rng(0))

    series = track_geometry(
        records,
        device_ids=sc.device_ids,
        speed_of_sound=sc.speed_of_sound_mps,
        window_s=5.0,
        step_s=2.5,
    )
    # A static array drifts nowhere; gauge choice is irrelevant for ~zero speed.
    est = estimate_velocities(_align_series_to_world(series, sc))
    for dev_id in sc.device_ids:
        assert float(np.linalg.norm(est[dev_id])) < 0.1


# --------------------------------------------------------------------------- #
# Bundled demo scenario
# --------------------------------------------------------------------------- #
def test_moving_devices_scenario_loads_and_tracks():
    sc = load_scenario(SCN / "moving_devices.yaml")
    assert sc.devices_move  # at least one device has nonzero velocity
    assert len(sc.devices) >= 5
    assert len(sc.anchors) >= 4
    # varied anchor heights
    anchor_z = sorted({round(a.position_m[2], 3) for a in sc.anchors})
    assert len(anchor_z) >= 3
    assert sc.ranging_rounds >= 60

    records = generate_moving_ranging(sc, np.random.default_rng(sc.seed))
    series = track_geometry(
        records,
        device_ids=sc.device_ids,
        speed_of_sound=sc.speed_of_sound_mps,
        window_s=6.0,
        step_s=3.0,
    )
    assert len(series) >= 3
    # The scenario carries moderate noise, so use a looser per-window bound.
    for t, layout in series:
        rmse = _window_rmse_to_truth(layout, sc, t)
        assert rmse < 2.0, f"scenario window @ t={t:.2f}: RMSE {rmse:.3f} m"
