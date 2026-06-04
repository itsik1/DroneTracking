import dataclasses
from pathlib import Path

import numpy as np
import pytest

from dronetracking.config import load_scenario
from dronetracking.sim.simulator import simulate
from dronetracking.sim.acoustic import emission_times
from dronetracking.sim.trajectory import trajectory_position
from dronetracking.geo import enu_to_latlon

SCN = Path(__file__).resolve().parents[1] / "scenarios"


def test_simulate_returns_observations_and_world():
    sc = load_scenario(SCN / "field_5dev.yaml")
    obs, world = simulate(sc)
    assert obs.device_ids == sc.device_ids
    assert len(obs.anchor_gps) == len(sc.anchors)
    assert obs.speed_of_sound_mps == sc.speed_of_sound_mps
    assert world.true_track.shape == (len(emission_times(sc)), 3)
    assert world.true_track_times.shape == (len(emission_times(sc)),)


def test_world_holds_true_positions_and_clocks():
    sc = load_scenario(SCN / "field_5dev.yaml")
    _, world = simulate(sc)
    assert np.allclose(world.device_positions["dev1"], [200.0, 0.0, 18.0])
    assert world.clock_offsets["dev1"] == pytest.approx(0.12)
    assert world.clock_drifts_ppm["dev1"] == pytest.approx(30.0)
    # positions_matrix is in device_ids order
    assert world.positions_matrix().shape == (5, 3)
    assert np.allclose(world.positions_matrix()[1], [200.0, 0.0, 18.0])


def test_anchor_gps_derived_from_true_enu_position_noisefree():
    sc = load_scenario(SCN / "noisefree_ideal.yaml")  # gps_pos_std_m = 0
    obs, world = simulate(sc)
    assert len(obs.anchor_gps) == len(sc.anchors)
    for ag in obs.anchor_gps:
        e, n, z = world.device_positions[ag.device_id]
        lat, lon = enu_to_latlon(e, n, sc.origin_latlon)
        assert ag.lat == pytest.approx(float(lat), abs=1e-12)
        assert ag.lon == pytest.approx(float(lon), abs=1e-12)
        # With zero GPS noise the reported altitude is exactly the true z.
        assert ag.altitude_m == pytest.approx(float(z), abs=1e-12)
        # And World stores the noise-free truth lat/lon for the same anchor.
        wlat, wlon = world.anchor_latlon[ag.device_id]
        assert wlat == pytest.approx(float(lat), abs=1e-12)
        assert wlon == pytest.approx(float(lon), abs=1e-12)


def test_world_true_track_matches_trajectory_at_emission_times():
    sc = load_scenario(SCN / "field_5dev.yaml")
    _, world = simulate(sc)
    times = emission_times(sc)
    assert np.array_equal(world.true_track_times, times)
    expected = np.array([trajectory_position(sc, float(t)) for t in times])
    assert np.allclose(world.true_track, expected)
    assert world.origin_latlon == sc.origin_latlon


def test_anchor_latlon_only_for_gps_devices():
    sc = load_scenario(SCN / "field_5dev.yaml")
    obs, world = simulate(sc)
    gps_ids = {d.id for d in sc.devices if d.has_gps}
    assert set(world.anchor_latlon) == gps_ids
    assert {ag.device_id for ag in obs.anchor_gps} == gps_ids


def test_gps_noise_does_not_shift_ranging_or_acoustic_streams():
    # The three child RNGs (ranging, acoustic, GPS) are independent: changing only the
    # GPS noise must leave ranging and acoustic draws bit-for-bit identical.
    base = load_scenario(SCN / "field_5dev.yaml")
    bumped_noise = dataclasses.replace(base.noise, gps_pos_std_m=base.noise.gps_pos_std_m + 5.0)
    bumped = dataclasses.replace(base, noise=bumped_noise)

    o_base, _ = simulate(base)
    o_bumped, _ = simulate(bumped)

    assert o_base.ranging == o_bumped.ranging
    assert o_base.acoustic == o_bumped.acoustic
    # ...but the GPS anchors themselves did change.
    assert o_base.anchor_gps != o_bumped.anchor_gps


def test_simulation_is_reproducible_with_same_seed():
    sc = load_scenario(SCN / "field_5dev.yaml")
    o1, _ = simulate(sc)
    o2, _ = simulate(sc)
    assert o1.ranging[0] == o2.ranging[0]  # frozen dataclass equality
    assert o1.acoustic[5] == o2.acoustic[5]


def test_different_seed_changes_the_noise():
    sc = load_scenario(SCN / "field_5dev.yaml")
    sc_other = load_scenario(SCN / "field_5dev.yaml", seed_override=999)
    o1, _ = simulate(sc)
    o2, _ = simulate(sc_other)
    assert o1.ranging[0] != o2.ranging[0]
