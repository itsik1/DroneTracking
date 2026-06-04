import numpy as np
import pytest

from dronetracking.sim.clocks import device_local_time, global_from_local
from dronetracking.sim.trajectory import trajectory_position
from dronetracking.sim.scenario import Scenario, TrajectorySpec, NoiseSpec, DeviceSpec


# ---- clocks ----

def test_local_clock_matches_locked_convention():
    # local = t*(1 + ppm*1e-6) + offset  (must match ClockEstimates.to_reference inverse)
    assert device_local_time(0.2, 0.0, 3.0) == pytest.approx(3.2)
    assert device_local_time(0.0, 1_000_000.0, 1.0) == pytest.approx(2.0)  # s=1 -> doubles


def test_local_clock_roundtrips_to_global():
    g = 12.5
    loc = device_local_time(0.1, 50.0, g)
    assert global_from_local(0.1, 50.0, loc) == pytest.approx(g, abs=1e-12)


def test_clock_functions_are_vectorized():
    g = np.array([0.0, 1.0, 2.0])
    loc = device_local_time(0.3, 20.0, g)
    assert np.allclose(global_from_local(0.3, 20.0, loc), g, atol=1e-12)


# ---- trajectory ----

def _scn(traj):
    return Scenario(
        name="t", seed=0, speed_of_sound_mps=343.0, sample_rate_hz=48000.0,
        duration_s=10.0, dt_s=1.0, ranging_rounds=5, origin_latlon=(32.0, 34.0),
        devices=(DeviceSpec("d0", (0, 0, 0)),), trajectory=traj, noise=NoiseSpec(),
    )


def test_linear_trajectory_hits_endpoints_and_midpoint():
    sc = _scn(TrajectorySpec("linear", {"start_m": [0, 0], "end_m": [100, 50]}, z_m=60.0))
    assert np.allclose(trajectory_position(sc, 0.0), [0, 0, 60])
    assert np.allclose(trajectory_position(sc, 5.0), [50, 25, 60])
    assert np.allclose(trajectory_position(sc, 10.0), [100, 50, 60])


def test_circular_trajectory_keeps_radius_and_altitude():
    sc = _scn(TrajectorySpec("circular", {"center_m": [10, 20], "radius_m": 30, "angular_rate_rad_s": 0.5}, z_m=40.0))
    assert np.allclose(trajectory_position(sc, 0.0), [40, 20, 40])  # cos0=1, sin0=0
    for t in (1.0, 2.5, 4.0):
        p = trajectory_position(sc, t)
        assert np.hypot(p[0] - 10, p[1] - 20) == pytest.approx(30.0)
        assert p[2] == pytest.approx(40.0)


def test_waypoints_trajectory_interpolates():
    sc = _scn(TrajectorySpec("waypoints", {"points_m": [[0, 0, 0], [100, 0, 10]]}, z_m=25.0))
    assert np.allclose(trajectory_position(sc, 5.0), [50, 0, 25])
