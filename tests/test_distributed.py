"""Tests for the distributed runtime over sockets (Iteration 4 — B).

Two layers, mirroring the contract:

1. **Protocol (pure):** ``encode_batch`` -> ``decode_batch`` round-trips a per-device
   batch field-for-field, including the *full float precision* of every timestamp, and
   handles the empty-slice case (a device that initiated no ranging / has no GPS).
2. **Loopback (integration):** a :class:`SocketDeviceFeed` listens on an OS-assigned port
   in a daemon thread; one :class:`DeviceAgent` per device publishes its own slice from a
   separate thread to ``127.0.0.1:feed.port``; after every thread joins, the feed's
   assembled ``Observations`` must equal the reference ``SimulatedDeviceFeed``'s — the
   ranging / acoustic / anchor *sets* compared order-insensitively.

The integration test is written to be deterministic and not flaky: a single shared seeded
feed (so every agent sees the same world), generous timeouts, and explicit thread joins.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import pytest

from dronetracking.config import load_scenario
from dronetracking.live import protocol
from dronetracking.live.agent import DeviceAgent, device_slice
from dronetracking.sim.observations import (
    AcousticArrival,
    AnchorGps,
    Observations,
    RangingRecord,
)
from dronetracking.sources.simulated import SimulatedDeviceFeed
from dronetracking.sources.socket_feed import SocketDeviceFeed

SCN = Path(__file__).resolve().parents[1] / "scenarios"

# Multi-device scenarios that exercise the union: field_5dev has a mic-only non-anchor
# device (empty ranging+anchor slice), multi_drone exercises multi-source acoustic.
_SCENARIO_FILES = ["field_5dev.yaml", "multi_drone.yaml"]


# --------------------------------------------------------------------------- #
# 1. protocol: exact round-trip
# --------------------------------------------------------------------------- #
def _sample_batch():
    """A hand-built batch with awkward float timestamps + non-default acoustic fields."""
    ranging = (
        RangingRecord(
            initiator="dev0",
            responder="dev1",
            round_idx=0,
            t1_local_i=1.8167285212903988e-05,
            t2_local_j=0.7054655592934146,
            t3_local_j=0.7075405981279447,
            t4_local_i=1.1730465577159028,
        ),
        RangingRecord(
            initiator="dev0",
            responder="dev2",
            round_idx=7,
            t1_local_i=3.141592653589793,
            t2_local_j=2.718281828459045,
            t3_local_j=1.4142135623730951,
            t4_local_i=0.5772156649015329,
        ),
    )
    acoustic = (
        AcousticArrival(device_id="dev0", emission_idx=0, toa_local_s=0.2704306108843925),
        AcousticArrival(
            device_id="dev0", emission_idx=3, toa_local_s=9.999999999999999e-08,
            source=2, confidence=0.4242424242424242,
        ),
    )
    anchor_gps = (
        AnchorGps(
            device_id="dev0",
            lat=32.08532528887028,
            lon=34.7817649849817,
            altitude_m=1.4469423605727023,
        ),
    )
    return ranging, acoustic, anchor_gps


def test_protocol_round_trips_a_batch_exactly():
    ranging, acoustic, anchor_gps = _sample_batch()
    raw = protocol.encode_batch(
        device_id="dev0",
        ranging=ranging,
        acoustic=acoustic,
        anchor_gps=anchor_gps,
        speed_of_sound_mps=343.0,
        sample_rate_hz=48000.0,
    )
    assert isinstance(raw, bytes)
    assert raw.endswith(b"\n")  # newline-framed

    decoded = protocol.decode_batch(raw)
    assert decoded["device_id"] == "dev0"
    assert decoded["speed_of_sound_mps"] == 343.0
    assert decoded["sample_rate_hz"] == 48000.0

    # Frozen dataclasses: == is structural, and float fields must match to the last bit.
    assert decoded["ranging"] == ranging
    assert decoded["acoustic"] == acoustic
    assert decoded["anchor_gps"] == anchor_gps

    # Element types are the contract dataclasses, not dicts.
    assert all(isinstance(r, RangingRecord) for r in decoded["ranging"])
    assert all(isinstance(a, AcousticArrival) for a in decoded["acoustic"])
    assert all(isinstance(g, AnchorGps) for g in decoded["anchor_gps"])


def test_protocol_preserves_full_float_precision_of_timestamps():
    # Explicitly assert bit-exact timestamp survival (repr equality => same double).
    ranging, _, _ = _sample_batch()
    decoded = protocol.decode_batch(
        protocol.encode_batch("dev0", ranging, (), (), 343.0, 48000.0)
    )
    for orig, got in zip(ranging, decoded["ranging"]):
        for field in ("t1_local_i", "t2_local_j", "t3_local_j", "t4_local_i"):
            assert repr(getattr(got, field)) == repr(getattr(orig, field))


def test_protocol_round_trips_empty_slice():
    # A device that initiated no ranging and has no GPS still produces a valid batch.
    raw = protocol.encode_batch(
        device_id="dev4",
        ranging=(),
        acoustic=(AcousticArrival(device_id="dev4", emission_idx=0, toa_local_s=0.5),),
        anchor_gps=(),
        speed_of_sound_mps=343.0,
        sample_rate_hz=48000.0,
    )
    decoded = protocol.decode_batch(raw)
    assert decoded["device_id"] == "dev4"
    assert decoded["ranging"] == ()
    assert decoded["anchor_gps"] == ()
    assert len(decoded["acoustic"]) == 1


def test_protocol_decode_tolerates_missing_trailing_newline():
    # decode_batch should accept the JSON with or without the framing newline.
    raw = protocol.encode_batch("dev0", (), (), (), 343.0, 48000.0)
    assert protocol.decode_batch(raw) == protocol.decode_batch(raw.rstrip(b"\n"))


# --------------------------------------------------------------------------- #
# agent slice: a device only ever sees/sends its own data, and slices partition cleanly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario_file", _SCENARIO_FILES)
def test_device_slices_partition_the_full_observations(scenario_file):
    scenario = load_scenario(SCN / scenario_file)
    feed = SimulatedDeviceFeed(scenario)
    obs = feed.as_observations()

    union_r: Counter = Counter()
    union_a: Counter = Counter()
    union_g: Counter = Counter()
    for did in obs.device_ids:
        ranging, acoustic, anchor_gps = device_slice(feed, did)
        # Each slice only contains this device's own data.
        assert all(r.initiator == did for r in ranging)
        assert all(a.device_id == did for a in acoustic)
        assert all(g.device_id == did for g in anchor_gps)
        union_r.update(ranging)
        union_a.update(acoustic)
        union_g.update(anchor_gps)

    # The disjoint slices reassemble the whole, exactly (multiset equality).
    assert union_r == Counter(obs.ranging)
    assert union_a == Counter(obs.acoustic)
    assert union_g == Counter(obs.anchor_gps)


# --------------------------------------------------------------------------- #
# 2. loopback: agents publish over TCP, coordinator reassembles == reference feed
# --------------------------------------------------------------------------- #
def _run_loopback(scenario, *, timeout_s: float = 20.0) -> Observations:
    """Run the full publish/collect cycle over loopback TCP and return the assembly."""
    # One shared, seeded feed so every agent sees the identical world.
    sim_feed = SimulatedDeviceFeed(scenario)
    device_ids = list(sim_feed.device_ids())

    feed = SocketDeviceFeed(host="127.0.0.1", port=0)
    collected: dict = {}

    def _collect():
        collected.update(feed.collect(device_ids, timeout_s=timeout_s))

    coordinator = threading.Thread(target=_collect, name="coordinator", daemon=True)
    coordinator.start()

    # Publish each device's slice from its own thread, against the bound port.
    errors: list = []

    def _publish(did):
        try:
            DeviceAgent().publish("127.0.0.1", feed.port, scenario, did, feed=sim_feed)
        except Exception as exc:  # pragma: no cover - surfaced via assert below
            errors.append((did, exc))

    agents = [
        threading.Thread(target=_publish, args=(did,), name=f"agent-{did}", daemon=True)
        for did in device_ids
    ]
    for t in agents:
        t.start()
    for t in agents:
        t.join(timeout=timeout_s)
    coordinator.join(timeout=timeout_s)

    feed.close()

    assert not errors, f"agent publish errors: {errors}"
    assert not coordinator.is_alive(), "coordinator did not finish in time"
    assert set(collected) == set(device_ids), "not every device reported"

    return feed.as_observations()


@pytest.mark.parametrize("scenario_file", _SCENARIO_FILES)
def test_loopback_assembles_observations_equal_to_simulated_feed(scenario_file):
    scenario = load_scenario(SCN / scenario_file)
    reference = SimulatedDeviceFeed(scenario).as_observations()

    assembled = _run_loopback(scenario)

    assert isinstance(assembled, Observations)
    # Same device set + order (we drive collect() with the sim feed's device order).
    assert assembled.device_ids == reference.device_ids
    # Timebase constants carried over the wire.
    assert assembled.speed_of_sound_mps == reference.speed_of_sound_mps
    assert assembled.sample_rate_hz == reference.sample_rate_hz

    # The measurement SETS match order-insensitively (frozen dataclasses are hashable).
    assert Counter(assembled.ranging) == Counter(reference.ranging)
    assert Counter(assembled.acoustic) == Counter(reference.acoustic)
    assert Counter(assembled.anchor_gps) == Counter(reference.anchor_gps)

    # And nothing was dropped or duplicated.
    assert len(assembled.ranging) == len(reference.ranging)
    assert len(assembled.acoustic) == len(reference.acoustic)
    assert len(assembled.anchor_gps) == len(reference.anchor_gps)


def test_socket_feed_assigns_a_port_on_bind():
    # port 0 -> OS picks a real, nonzero port, exposed before any connection.
    feed = SocketDeviceFeed(host="127.0.0.1", port=0)
    try:
        assert isinstance(feed.port, int)
        assert feed.port > 0
    finally:
        feed.close()


def test_socket_feed_is_a_device_feed():
    feed = SocketDeviceFeed(host="127.0.0.1", port=0)
    try:
        from dronetracking.sources.base import DeviceFeed

        assert isinstance(feed, DeviceFeed)
    finally:
        feed.close()


def test_collect_times_out_cleanly_when_a_device_never_reports():
    # If an expected device never connects, collect() returns what arrived (here: none)
    # within the timeout, rather than hanging forever.
    feed = SocketDeviceFeed(host="127.0.0.1", port=0)
    try:
        got = feed.collect(["ghost"], timeout_s=0.5)
        assert got == {}
        assert feed.device_ids() == ()
        assert feed.as_observations().ranging == ()
    finally:
        feed.close()
