"""Tests for the clock-synchronization estimation stage.

Strategy (TDD): build a Scenario whose ``device_ids[0]`` has true offset 0 and
drift 0 (matching all shipped scenarios) plus several devices with known nonzero
offsets/drifts. Generate realistic ranging/acoustic observations with the FROZEN
sim leaf functions (allowed in estimation *tests* for fixtures only) and assert:

  1. Recovered (offset, drift) relative to dev0 match the truth (noise-free).
  2. TDOA CONSISTENCY: for one emission and any two devices i, j,
        clocks.to_reference(i, toa_i) - clocks.to_reference(j, toa_j)
        == (L_i - L_j) / c
     with L_* the true ranges to the true drone position (the key property the
     whole pipeline relies on).
  3. The TDOA-consistency error stays small under measurement noise.
"""

import numpy as np
import pytest

# FROZEN sim leaf functions -- usable in estimation TESTS for fixtures only.
from dronetracking.sim.scenario import Scenario, TrajectorySpec, NoiseSpec, DeviceSpec
from dronetracking.sim.ranging import generate_ranging_records
from dronetracking.sim.acoustic import generate_acoustic_arrivals, emission_times
from dronetracking.sim.trajectory import trajectory_position
from dronetracking.sim.observations import Observations

from dronetracking.estimation.clock_sync import estimate_clocks

C = 343.0

# True clocks. dev0 is the reference: offset 0, drift 0 (matches shipped scenarios).
TRUE_DEVICES = [
    DeviceSpec("dev0", (0.0, 0.0, 0.0), clock_offset_s=0.0, clock_drift_ppm=0.0),
    DeviceSpec("dev1", (120.0, 0.0, 0.0), clock_offset_s=0.15, clock_drift_ppm=40.0),
    DeviceSpec("dev2", (0.0, 140.0, 2.0), clock_offset_s=-0.30, clock_drift_ppm=-25.0),
    DeviceSpec("dev3", (130.0, 110.0, -3.0), clock_offset_s=0.42, clock_drift_ppm=12.5),
    DeviceSpec("dev4", (60.0, 60.0, 5.0), clock_offset_s=-0.08, clock_drift_ppm=60.0),
]


def _scenario(noise=None, rounds=40, duration=30.0, dt=0.5):
    """A multi-device scenario with a moving drone and a long ranging window.

    A long duration / many rounds makes the clock skew (drift) observable from how
    the per-exchange offset estimate changes with transmit time.
    """
    return Scenario(
        name="clock_sync_fixture",
        seed=0,
        speed_of_sound_mps=C,
        sample_rate_hz=48000.0,
        duration_s=duration,
        dt_s=dt,
        ranging_rounds=rounds,
        origin_latlon=(32.0, 34.0),
        devices=tuple(TRUE_DEVICES),
        trajectory=TrajectorySpec(
            "linear", {"start_m": [-40.0, 30.0], "end_m": [180.0, 90.0]}, z_m=70.0
        ),
        noise=noise or NoiseSpec(),
    )


def _observations(scenario, rng):
    """Assemble an Observations bundle from the frozen sim generators."""
    ranging = generate_ranging_records(scenario, rng)
    acoustic = generate_acoustic_arrivals(scenario, rng)
    return Observations(
        device_ids=scenario.device_ids,
        ranging=ranging,
        acoustic=acoustic,
        anchor_gps=(),
        speed_of_sound_mps=scenario.speed_of_sound_mps,
        sample_rate_hz=scenario.sample_rate_hz,
    )


def _true_clocks():
    return (
        {d.id: d.clock_offset_s for d in TRUE_DEVICES},
        {d.id: d.clock_drift_ppm for d in TRUE_DEVICES},
    )


def _arrivals_by_emission(observations):
    by_emission = {}
    for arr in observations.acoustic:
        by_emission.setdefault(arr.emission_idx, {})[arr.device_id] = arr.toa_local_s
    return by_emission


# --------------------------------------------------------------------------- #
# Basic contract
# --------------------------------------------------------------------------- #

def test_reference_defaults_to_first_device_and_is_zeroed():
    sc = _scenario()
    obs = _observations(sc, np.random.default_rng(0))
    clocks = estimate_clocks(obs)

    assert clocks.reference_id == "dev0"
    assert clocks.device_ids == sc.device_ids
    assert clocks.offsets_s["dev0"] == 0.0
    assert clocks.drifts_ppm["dev0"] == 0.0
    assert clocks.covariances is None
    # Every device must be represented.
    assert set(clocks.offsets_s) == set(sc.device_ids)
    assert set(clocks.drifts_ppm) == set(sc.device_ids)


def test_explicit_reference_is_zeroed():
    sc = _scenario()
    obs = _observations(sc, np.random.default_rng(0))
    clocks = estimate_clocks(obs, reference_id="dev2")

    assert clocks.reference_id == "dev2"
    assert clocks.offsets_s["dev2"] == 0.0
    assert clocks.drifts_ppm["dev2"] == 0.0


def test_unknown_reference_raises():
    sc = _scenario()
    obs = _observations(sc, np.random.default_rng(0))
    with pytest.raises(ValueError):
        estimate_clocks(obs, reference_id="nope")


# --------------------------------------------------------------------------- #
# 1. Recovery of offsets & drifts (noise-free)
# --------------------------------------------------------------------------- #

def test_recovers_offsets_and_drifts_noise_free():
    sc = _scenario()
    obs = _observations(sc, np.random.default_rng(0))
    clocks = estimate_clocks(obs)

    true_off, true_drift = _true_clocks()
    # Everything is relative to dev0, whose true (offset, drift) is (0, 0), so the
    # recovered values should match the absolute truth directly.
    for dev in sc.device_ids:
        assert clocks.offsets_s[dev] == pytest.approx(true_off[dev], abs=1e-6)
        assert clocks.drifts_ppm[dev] == pytest.approx(true_drift[dev], abs=1e-2)


def test_recovery_is_relative_to_reference():
    # With a non-dev0 reference, recovered values equal (truth - reference truth).
    sc = _scenario()
    obs = _observations(sc, np.random.default_rng(0))
    ref = "dev3"
    clocks = estimate_clocks(obs, reference_id=ref)

    true_off, true_drift = _true_clocks()
    for dev in sc.device_ids:
        assert clocks.offsets_s[dev] == pytest.approx(
            true_off[dev] - true_off[ref], abs=1e-6
        )
        assert clocks.drifts_ppm[dev] == pytest.approx(
            true_drift[dev] - true_drift[ref], abs=1e-2
        )


# --------------------------------------------------------------------------- #
# 2. TDOA consistency (the key test), noise-free
# --------------------------------------------------------------------------- #

def _tdoa_errors(scenario, observations, clocks):
    """Max abs error between recovered TDOA and true range-difference TDOA.

    For each emission and each device pair, the reference-timebase arrival
    difference must equal the true geometric range difference / c.
    """
    positions = {d.id: np.asarray(d.position_m, float) for d in scenario.devices}
    times = emission_times(scenario)
    by_emission = _arrivals_by_emission(observations)
    c = scenario.speed_of_sound_mps

    max_err = 0.0
    for k, toas in by_emission.items():
        drone = trajectory_position(scenario, float(times[k]))
        ranges = {dev: float(np.linalg.norm(drone - positions[dev])) for dev in toas}
        devs = sorted(toas)
        for a in range(len(devs)):
            for b in range(a + 1, len(devs)):
                i, j = devs[a], devs[b]
                recovered = clocks.to_reference(i, toas[i]) - clocks.to_reference(
                    j, toas[j]
                )
                truth = (ranges[i] - ranges[j]) / c
                max_err = max(max_err, abs(recovered - truth))
    return max_err


def test_tdoa_consistency_noise_free():
    sc = _scenario()
    obs = _observations(sc, np.random.default_rng(0))
    clocks = estimate_clocks(obs)

    # Spot-check the exact property the contract states for one emission, two devices.
    positions = {d.id: np.asarray(d.position_m, float) for d in sc.devices}
    times = emission_times(sc)
    by_emission = _arrivals_by_emission(obs)
    k = sorted(by_emission)[len(by_emission) // 2]
    toas = by_emission[k]
    drone = trajectory_position(sc, float(times[k]))
    i, j = "dev1", "dev3"
    L_i = float(np.linalg.norm(drone - positions[i]))
    L_j = float(np.linalg.norm(drone - positions[j]))
    recovered = clocks.to_reference(i, toas[i]) - clocks.to_reference(j, toas[j])
    assert recovered == pytest.approx((L_i - L_j) / C, abs=1e-6)

    # And over every emission / every pair.
    assert _tdoa_errors(sc, obs, clocks) < 1e-6


# --------------------------------------------------------------------------- #
# 3. TDOA consistency under noise
# --------------------------------------------------------------------------- #

def test_tdoa_consistency_noisy():
    # Realistic timestamp/ToA jitter. Theil-Sen over many rounds keeps the recovered
    # clocks accurate enough that TDOA consistency holds to well under 1e-4 s.
    noise = NoiseSpec(
        ranging_timestamp_std_s=2e-6,
        toa_std_s=2e-6,
        proc_delay_jitter_s=1e-6,
    )
    sc = _scenario(noise=noise, rounds=120, duration=60.0)
    obs = _observations(sc, np.random.default_rng(7))
    clocks = estimate_clocks(obs)

    # Offsets/drifts still close to truth.
    true_off, true_drift = _true_clocks()
    for dev in sc.device_ids:
        assert clocks.offsets_s[dev] == pytest.approx(true_off[dev], abs=5e-5)
        assert clocks.drifts_ppm[dev] == pytest.approx(true_drift[dev], abs=2.0)

    # The key acceptance criterion: TDOA consistency error stays small.
    # Tolerance 1e-4 s: with the toa jitter above the per-arrival error is a few
    # microseconds, so pairwise differences stay comfortably under 100 us.
    assert _tdoa_errors(sc, obs, clocks) < 1e-4
