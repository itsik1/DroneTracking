"""Tests for the device-feed / hardware abstraction (``dronetracking.sources``).

The contract: a ``DeviceFeed`` is the boundary the estimation/streaming layer reads
from. The reference ``SimulatedDeviceFeed`` must be a perfect stand-in for a direct
``simulate(scenario)`` call — ``as_observations()`` is field-for-field identical to
``simulate(scenario)[0]`` for the same scenario/seed, and ``.world`` matches
``simulate(scenario)[1]``. The abstract base cannot be instantiated, and the
``LiveDeviceFeed`` skeleton raises ``NotImplementedError`` for every measurement until
a real device network supplies it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dronetracking.config import load_scenario
from dronetracking.sim.observations import (
    AcousticArrival,
    AnchorGps,
    Observations,
    RangingRecord,
)
from dronetracking.sim.simulator import simulate
from dronetracking.sim.world import World
from dronetracking.sources import DeviceFeed, LiveDeviceFeed, SimulatedDeviceFeed

SCN = Path(__file__).resolve().parents[1] / "scenarios"

# A representative spread of scenarios: baseline, multi-target, moving devices,
# GPS-blackout, and sparse anchors. The simulated feed must reproduce every one.
_SCENARIO_FILES = [
    "field_5dev.yaml",
    "multi_drone.yaml",
    "moving_devices.yaml",
    "gps_denied.yaml",
    "sparse_anchors_circular.yaml",
]


# --------------------------------------------------------------------------- #
# DeviceFeed is an abstract boundary, not a usable class.
# --------------------------------------------------------------------------- #
def test_device_feed_is_abstract_and_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DeviceFeed()  # type: ignore[abstract]


def test_device_feed_declares_the_full_measurement_surface():
    # The five measurement accessors + the two timebase properties + the concrete
    # bundler are the documented hardware contract; assert they all exist.
    for name in (
        "device_ids",
        "ranging_records",
        "acoustic_arrivals",
        "anchor_gps",
        "speed_of_sound_mps",
        "sample_rate_hz",
        "as_observations",
    ):
        assert hasattr(DeviceFeed, name)
    # as_observations is concrete (provided by the base), the rest are abstract.
    assert "as_observations" not in DeviceFeed.__abstractmethods__
    for name in (
        "device_ids",
        "ranging_records",
        "acoustic_arrivals",
        "anchor_gps",
        "speed_of_sound_mps",
        "sample_rate_hz",
    ):
        assert name in DeviceFeed.__abstractmethods__


# --------------------------------------------------------------------------- #
# SimulatedDeviceFeed is a drop-in replacement for simulate(scenario).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario_file", _SCENARIO_FILES)
def test_simulated_feed_as_observations_equals_simulate(scenario_file):
    scenario = load_scenario(SCN / scenario_file)
    feed = SimulatedDeviceFeed(scenario)

    obs_feed = feed.as_observations()
    obs_sim, _ = simulate(scenario)

    # Same type, and field-for-field equal (Observations + its members are frozen
    # dataclasses, so == is a deep structural comparison).
    assert isinstance(obs_feed, Observations)
    assert obs_feed == obs_sim


@pytest.mark.parametrize("scenario_file", _SCENARIO_FILES)
def test_simulated_feed_individual_accessors_match_observations(scenario_file):
    scenario = load_scenario(SCN / scenario_file)
    feed = SimulatedDeviceFeed(scenario)
    obs_sim, _ = simulate(scenario)

    assert feed.device_ids() == obs_sim.device_ids
    assert feed.ranging_records() == obs_sim.ranging
    assert feed.acoustic_arrivals() == obs_sim.acoustic
    assert feed.anchor_gps() == obs_sim.anchor_gps
    assert feed.speed_of_sound_mps == obs_sim.speed_of_sound_mps
    assert feed.sample_rate_hz == obs_sim.sample_rate_hz


@pytest.mark.parametrize("scenario_file", _SCENARIO_FILES)
def test_simulated_feed_accessor_element_types(scenario_file):
    # The accessors must hand back exactly the contract types (tuples of the
    # frozen records), so downstream code can rely on them.
    scenario = load_scenario(SCN / scenario_file)
    feed = SimulatedDeviceFeed(scenario)

    assert isinstance(feed.device_ids(), tuple)
    assert all(isinstance(d, str) for d in feed.device_ids())

    ranging = feed.ranging_records()
    assert isinstance(ranging, tuple)
    assert all(isinstance(r, RangingRecord) for r in ranging)

    acoustic = feed.acoustic_arrivals()
    assert isinstance(acoustic, tuple)
    assert all(isinstance(a, AcousticArrival) for a in acoustic)

    anchors = feed.anchor_gps()
    assert isinstance(anchors, tuple)
    assert all(isinstance(g, AnchorGps) for g in anchors)

    assert isinstance(feed.speed_of_sound_mps, float)
    assert isinstance(feed.sample_rate_hz, float)


def test_simulated_feed_is_a_device_feed():
    scenario = load_scenario(SCN / "field_5dev.yaml")
    assert isinstance(SimulatedDeviceFeed(scenario), DeviceFeed)


def test_simulated_feed_simulates_once_and_caches():
    # __init__ runs simulate() exactly once; repeated accessors return the SAME
    # cached objects (identity), so a feed is a cheap, stable snapshot.
    scenario = load_scenario(SCN / "field_5dev.yaml")
    feed = SimulatedDeviceFeed(scenario)

    assert feed.ranging_records() is feed.ranging_records()
    assert feed.acoustic_arrivals() is feed.acoustic_arrivals()
    assert feed.anchor_gps() is feed.anchor_gps()
    assert feed.as_observations() is feed.as_observations()
    assert feed.world is feed.world


# --------------------------------------------------------------------------- #
# .world is the sim-only ground truth, for eval.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario_file", _SCENARIO_FILES)
def test_simulated_feed_world_matches_simulate(scenario_file):
    scenario = load_scenario(SCN / scenario_file)
    feed = SimulatedDeviceFeed(scenario)
    _, world_sim = simulate(scenario)

    world_feed = feed.world
    assert isinstance(world_feed, World)

    # World holds numpy arrays / dicts of arrays, so compare structurally.
    assert world_feed.device_ids == world_sim.device_ids
    assert world_feed.origin_latlon == world_sim.origin_latlon
    assert world_feed.anchor_latlon == world_sim.anchor_latlon
    assert world_feed.clock_offsets == world_sim.clock_offsets
    assert world_feed.clock_drifts_ppm == world_sim.clock_drifts_ppm
    assert np.array_equal(world_feed.true_track, world_sim.true_track)
    assert np.array_equal(world_feed.true_track_times, world_sim.true_track_times)

    assert set(world_feed.device_positions) == set(world_sim.device_positions)
    for dev_id, pos in world_sim.device_positions.items():
        assert np.array_equal(world_feed.device_positions[dev_id], pos)

    assert set(world_feed.true_tracks) == set(world_sim.true_tracks)
    for src, trk in world_sim.true_tracks.items():
        assert np.array_equal(world_feed.true_tracks[src], trk)


# --------------------------------------------------------------------------- #
# LiveDeviceFeed is a documented skeleton: same ABC, everything NotImplemented.
# --------------------------------------------------------------------------- #
def test_live_feed_is_a_device_feed_and_instantiable():
    # Unlike the ABC, the live skeleton CAN be constructed (it overrides every
    # abstract member) — it just refuses to produce data until wired to hardware.
    feed = LiveDeviceFeed()
    assert isinstance(feed, DeviceFeed)


def test_live_feed_methods_raise_not_implemented():
    feed = LiveDeviceFeed()
    with pytest.raises(NotImplementedError):
        feed.device_ids()
    with pytest.raises(NotImplementedError):
        feed.ranging_records()
    with pytest.raises(NotImplementedError):
        feed.acoustic_arrivals()
    with pytest.raises(NotImplementedError):
        feed.anchor_gps()


def test_live_feed_properties_raise_not_implemented():
    feed = LiveDeviceFeed()
    with pytest.raises(NotImplementedError):
        _ = feed.speed_of_sound_mps
    with pytest.raises(NotImplementedError):
        _ = feed.sample_rate_hz


def test_live_feed_as_observations_raises_not_implemented():
    # The concrete bundler calls the abstract accessors, so on the live skeleton it
    # surfaces NotImplementedError too (rather than a confusing AttributeError).
    feed = LiveDeviceFeed()
    with pytest.raises(NotImplementedError):
        feed.as_observations()


def test_live_feed_methods_document_the_hardware_contract():
    # Every NotImplemented stub must carry a docstring describing what real hardware
    # must supply — this file IS the hardware contract.
    for name in (
        "device_ids",
        "ranging_records",
        "acoustic_arrivals",
        "anchor_gps",
        "speed_of_sound_mps",
        "sample_rate_hz",
    ):
        member = getattr(LiveDeviceFeed, name)
        # properties wrap their getter in fget; methods are plain functions.
        func = member.fget if isinstance(member, property) else member
        assert func.__doc__ and func.__doc__.strip(), f"{name} needs a docstring"
