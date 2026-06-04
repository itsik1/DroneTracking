import numpy as np
import pytest

from dronetracking.sim.scenario import Scenario, TrajectorySpec, NoiseSpec, DeviceSpec
from dronetracking.sim.clocks import global_from_local
from dronetracking.sim.trajectory import trajectory_position
from dronetracking.sim.ranging import generate_ranging_records
from dronetracking.sim.acoustic import generate_acoustic_arrivals, emission_times

C = 343.0


def _scn(devices, noise=None, rounds=20, duration=10.0, dt=0.5):
    return Scenario(
        name="t", seed=0, speed_of_sound_mps=C, sample_rate_hz=48000.0,
        duration_s=duration, dt_s=dt, ranging_rounds=rounds, origin_latlon=(32.0, 34.0),
        devices=tuple(devices),
        trajectory=TrajectorySpec("linear", {"start_m": [-50, 50], "end_m": [150, 50]}, z_m=60.0),
        noise=noise or NoiseSpec(),
    )


# ---- ranging ----

def test_ranging_recovers_exact_distance_with_no_noise():
    # SDS-TWR: ToF = 1/2 * ((t4-t1) - (t3-t2)); offset cancels by construction.
    sc = _scn([
        DeviceSpec("a", (0, 0, 0), clock_offset_s=0.1, clock_drift_ppm=0.0),
        DeviceSpec("b", (100, 0, 0), clock_offset_s=0.3, clock_drift_ppm=0.0),
    ])
    recs = generate_ranging_records(sc, np.random.default_rng(0))
    assert len(recs) == sc.ranging_rounds  # one pair, N rounds
    for r in recs:
        tof = 0.5 * ((r.t4_local_i - r.t1_local_i) - (r.t3_local_j - r.t2_local_j))
        assert tof * C == pytest.approx(100.0, abs=1e-9)


def test_ranging_per_exchange_offset_recovers_relative_bias():
    sc = _scn([
        DeviceSpec("a", (0, 0, 0), clock_offset_s=0.1, clock_drift_ppm=0.0),
        DeviceSpec("b", (100, 0, 0), clock_offset_s=0.3, clock_drift_ppm=0.0),
    ])
    recs = generate_ranging_records(sc, np.random.default_rng(0))
    for r in recs:
        offset = 0.5 * ((r.t2_local_j - r.t1_local_i) + (r.t3_local_j - r.t4_local_i))
        assert offset == pytest.approx(0.2, abs=1e-9)  # b_b - b_a


def test_ranging_offset_slope_equals_relative_skew():
    # Under drift, the per-exchange offset estimate is linear in transmit time with
    # slope = relative skew -- this is exactly what clock_sync regresses.
    sc = _scn([
        DeviceSpec("a", (0, 0, 0), clock_offset_s=0.0, clock_drift_ppm=0.0),
        DeviceSpec("b", (100, 0, 0), clock_offset_s=0.0, clock_drift_ppm=50.0),
    ], rounds=30, duration=20.0)
    recs = sorted(generate_ranging_records(sc, np.random.default_rng(0)), key=lambda r: r.round_idx)
    tx = np.linspace(0.0, sc.duration_s, sc.ranging_rounds)
    offsets = [0.5 * ((r.t2_local_j - r.t1_local_i) + (r.t3_local_j - r.t4_local_i)) for r in recs]
    slope = np.polyfit(tx, offsets, 1)[0]
    assert slope == pytest.approx(50e-6, abs=1e-9)  # (s_b - s_a)


def test_ranging_all_pairs_present():
    sc = _scn([DeviceSpec(f"d{i}", (i * 10, 0, 0)) for i in range(4)], rounds=3)
    recs = generate_ranging_records(sc, np.random.default_rng(0))
    pairs = {(r.initiator, r.responder) for r in recs}
    assert len(pairs) == 6  # 4 choose 2
    assert len(recs) == 6 * 3


# ---- acoustic ----

def test_acoustic_noisefree_arrival_recovers_global_time_of_flight():
    sc = _scn([DeviceSpec("a", (0, 0, 0), clock_offset_s=0.2, clock_drift_ppm=30.0)])
    arrivals = generate_acoustic_arrivals(sc, np.random.default_rng(0))
    times = emission_times(sc)
    for arr in arrivals:
        g = global_from_local(0.2, 30.0, arr.toa_local_s)
        t_k = times[arr.emission_idx]
        drone = trajectory_position(sc, t_k)
        L = np.linalg.norm(drone - np.array([0.0, 0.0, 0.0]))
        assert g == pytest.approx(t_k + L / C, abs=1e-9)


def test_acoustic_arrival_difference_is_independent_of_emission_time():
    # The unknown emission time cancels in a TDOA difference; per emission,
    # (global_arrival_a - global_arrival_b) == (L_a - L_b)/c.
    devs = [
        DeviceSpec("a", (0, 0, 0), clock_offset_s=0.1, clock_drift_ppm=10.0),
        DeviceSpec("b", (120, 30, 0), clock_offset_s=-0.2, clock_drift_ppm=-15.0),
    ]
    sc = _scn(devs)
    arrivals = generate_acoustic_arrivals(sc, np.random.default_rng(0))
    times = emission_times(sc)
    by_emission = {}
    for arr in arrivals:
        by_emission.setdefault(arr.emission_idx, {})[arr.device_id] = arr.toa_local_s
    for k, d in by_emission.items():
        ga = global_from_local(0.1, 10.0, d["a"])
        gb = global_from_local(-0.2, -15.0, d["b"])
        drone = trajectory_position(sc, times[k])
        La = np.linalg.norm(drone - np.array([0.0, 0.0, 0.0]))
        Lb = np.linalg.norm(drone - np.array([120.0, 30.0, 0.0]))
        assert (ga - gb) == pytest.approx((La - Lb) / C, abs=1e-9)
