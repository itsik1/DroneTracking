"""Tests for the acoustic-ranging backend (``dronetracking.webapp.ranging``).

Strategy (no hardware, fully deterministic):

* Place devices at a known 2-D geometry (meters). The true pairwise distance is
  the Euclidean distance between two devices.
* Synthesize each round's four SDS-TWR timestamps from that true distance, a
  per-device clock *offset* (which must cancel), and a fixed responder
  *turnaround*. The real-time timeline is::

      A emits at real T;  B hears at T + tof;  B replies at T + tof + T_reply;
      A hears at T + 2*tof + T_reply           (tof = distance / c)

  with each device stamping events in its *own* clock (real + offset). Then::

      t1 = T + off_A
      t2 = T + tof + off_B
      t3 = T + tof + T_reply + off_B
      t4 = T + 2*tof + T_reply + off_A

  so ``(t4-t1) - (t3-t2) = 2*tof`` and the recovered distance is exact.
* Feed those halves to a :class:`RangingCoordinator` (and, end to end, a
  :class:`Session`) and assert the recovered pairwise distances and the relative
  layout's pairwise distances match truth within tolerance.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from dronetracking.webapp.ranging import (
    DEFAULT_CHIRP,
    SPEED_OF_SOUND_MPS,
    RangingCoordinator,
    sds_twr_distance,
)
from dronetracking.webapp.session import Session

C = SPEED_OF_SOUND_MPS
T_REPLY = 0.020  # fixed responder turnaround (s)


class FakeClock:
    """A hand-cranked monotonic clock for deterministic scheduling tests."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _round_timestamps(
    distance_m: float,
    *,
    t_emit: float = 0.0,
    off_a: float = 0.0,
    off_b: float = 0.0,
    t_reply: float = T_REPLY,
    c: float = C,
) -> tuple[float, float, float, float]:
    """Synthesize (t1, t2, t3, t4) for a round with a known true distance.

    Clock offsets ``off_a``/``off_b`` are arbitrary and must cancel in the
    SDS-TWR formula, exercising the sync-free property.
    """
    tof = distance_m / c
    t1 = t_emit + off_a
    t2 = t_emit + tof + off_b
    t3 = t_emit + tof + t_reply + off_b
    t4 = t_emit + 2.0 * tof + t_reply + off_a
    return t1, t2, t3, t4


# --------------------------------------------------------------------------- #
# (1) sds_twr_distance recovers a known distance
# --------------------------------------------------------------------------- #


def test_sds_twr_recovers_known_distance_with_offsets():
    true_d = 4.2  # meters
    # Deliberately large, unequal clock offsets — they must cancel exactly.
    t1, t2, t3, t4 = _round_timestamps(
        true_d, t_emit=12.5, off_a=100.0, off_b=-37.0
    )
    d = sds_twr_distance(t1, t2, t3, t4)
    assert d == pytest.approx(true_d, abs=1e-9)


def test_sds_twr_matches_closed_form():
    t1, t2, t3, t4 = 1.0, 1.5, 2.0, 3.2
    expected = C * ((t4 - t1) - (t3 - t2)) / 2.0
    assert sds_twr_distance(t1, t2, t3, t4) == pytest.approx(expected, abs=1e-12)


def test_sds_twr_respects_custom_speed_of_sound():
    true_d = 10.0
    c = 300.0
    t1, t2, t3, t4 = _round_timestamps(true_d, c=c)
    assert sds_twr_distance(t1, t2, t3, t4, speed_of_sound_mps=c) == pytest.approx(
        true_d, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Coordinator: scheduling
# --------------------------------------------------------------------------- #


def test_current_command_none_for_fewer_than_two_devices():
    rc = RangingCoordinator()
    assert rc.current_command([], now=0.0) is None
    assert rc.current_command(["d1"], now=0.0) is None


def test_current_command_shape_and_round_robin_advance():
    rc = RangingCoordinator(round_period_s=3.0)
    ids = ["d1", "d2", "d3"]

    cmd = rc.current_command(ids, now=0.0)
    assert cmd is not None
    assert set(cmd.keys()) == {"round", "initiator", "responder", "chirp"}
    assert cmd["initiator"] in ids and cmd["responder"] in ids
    assert cmd["initiator"] != cmd["responder"]
    assert cmd["chirp"] == DEFAULT_CHIRP

    # Idempotent within the window: same round, same pair.
    assert rc.current_command(ids, now=1.0) == cmd

    # After the window elapses, the round advances and the pair changes.
    cmd2 = rc.current_command(ids, now=3.5)
    assert cmd2["round"] == cmd["round"] + 1
    assert (cmd2["initiator"], cmd2["responder"]) != (
        cmd["initiator"],
        cmd["responder"],
    )


def test_round_robin_cycles_all_pairs():
    rc = RangingCoordinator(round_period_s=1.0)
    ids = ["a", "b", "c"]  # 3 unordered pairs
    seen = set()
    now = 0.0
    for _ in range(6):
        cmd = rc.current_command(ids, now=now)
        seen.add(frozenset((cmd["initiator"], cmd["responder"])))
        now += 1.5  # past the window each time -> advance
    assert seen == {
        frozenset(("a", "b")),
        frozenset(("a", "c")),
        frozenset(("b", "c")),
    }


# --------------------------------------------------------------------------- #
# Coordinator: pairing halves -> distances (robust median over repeats)
# --------------------------------------------------------------------------- #


def _feed_round(
    rc: RangingCoordinator,
    rnd: int,
    initiator: str,
    responder: str,
    distance_m: float,
    *,
    off_a: float = 0.0,
    off_b: float = 0.0,
    t_emit: float = 0.0,
) -> None:
    """Submit both halves of one round to the coordinator."""
    t1, t2, t3, t4 = _round_timestamps(
        distance_m, t_emit=t_emit, off_a=off_a, off_b=off_b
    )
    rc.submit(initiator, [{"round": rnd, "role": "init", "t1": t1, "t4": t4}])
    rc.submit(responder, [{"round": rnd, "role": "resp", "t2": t2, "t3": t3}])


def test_single_round_yields_correct_distance():
    rc = RangingCoordinator()
    _feed_round(rc, 0, "d1", "d2", 7.5, off_a=5.0, off_b=-2.0)
    dists = rc.distances()
    assert len(dists) == 1
    rec = dists[0]
    assert (rec["a"], rec["b"]) == ("d1", "d2")
    assert rec["m"] == pytest.approx(7.5, abs=1e-6)


def test_half_exchange_order_independent():
    """Submitting the responder half first must still pair correctly."""
    rc = RangingCoordinator()
    t1, t2, t3, t4 = _round_timestamps(3.0)
    rc.submit("d2", [{"round": 9, "role": "resp", "t2": t2, "t3": t3}])
    assert rc.distances() == []  # only one half so far
    rc.submit("d1", [{"round": 9, "role": "init", "t1": t1, "t4": t4}])
    dists = rc.distances()
    assert len(dists) == 1
    assert dists[0]["m"] == pytest.approx(3.0, abs=1e-6)


def test_robust_median_over_repeats_rejects_outlier():
    rc = RangingCoordinator()
    # Five clean rounds of the same pair at 6.0 m, plus one wild outlier round.
    for rnd in range(5):
        _feed_round(rc, rnd, "d1", "d2", 6.0, t_emit=float(rnd))
    # Outlier: pretend a missed peak produced a 50 m reading (still plausible).
    t1, t2, t3, t4 = _round_timestamps(50.0)
    rc.submit("d1", [{"round": 99, "role": "init", "t1": t1, "t4": t4}])
    rc.submit("d2", [{"round": 99, "role": "resp", "t2": t2, "t3": t3}])

    m = next(r["m"] for r in rc.distances() if {r["a"], r["b"]} == {"d1", "d2"})
    # Median of [6,6,6,6,6,50] = 6 -> outlier ignored.
    assert m == pytest.approx(6.0, abs=1e-6)


def test_implausible_negative_distance_is_dropped():
    rc = RangingCoordinator()
    # A missed/misordered peak makes (t4 - t1) < (t3 - t2) -> negative range.
    # Here (t4-t1)=0.0 but (t3-t2)=0.5, so distance < 0 -> dropped, no sample.
    rc.submit("d1", [{"round": 0, "role": "init", "t1": 0.0, "t4": 0.0}])
    rc.submit("d2", [{"round": 0, "role": "resp", "t2": 0.0, "t3": 0.5}])
    assert sds_twr_distance(0.0, 0.0, 0.5, 0.0) < 0.0  # guard the premise
    assert rc.distances() == []


def test_malformed_entries_are_ignored():
    rc = RangingCoordinator()
    rc.submit("d1", [{"role": "init", "t1": 0.0, "t4": 1.0}])  # no round
    rc.submit("d1", [{"round": 1, "role": "bogus", "t1": 0.0}])  # bad role
    rc.submit("d1", [{"round": 2, "role": "init", "t1": 0.0}])  # missing t4
    rc.submit("d1", [])  # empty
    assert rc.distances() == []


# --------------------------------------------------------------------------- #
# Coordinator: DistanceMatrix
# --------------------------------------------------------------------------- #


def test_distance_matrix_marks_measured_and_unmeasured():
    rc = RangingCoordinator()
    _feed_round(rc, 0, "d1", "d2", 5.0)  # only the d1-d2 edge measured
    dm = rc.distance_matrix(["d1", "d2", "d3"])

    assert dm.device_ids == ("d1", "d2", "d3")
    # Diagonal: zero distance, valid.
    assert np.allclose(np.diag(dm.D), 0.0)
    assert np.all(np.diag(dm.valid))
    # Measured edge.
    assert dm.D[0, 1] == pytest.approx(5.0, abs=1e-6)
    assert dm.D[1, 0] == pytest.approx(5.0, abs=1e-6)
    assert dm.valid[0, 1] and dm.valid[1, 0]
    assert dm.W[0, 1] > 0.0
    assert dm.counts[0, 1] == 1
    # Unmeasured edges: NaN, invalid, zero weight.
    assert math.isnan(dm.D[0, 2])
    assert math.isnan(dm.D[1, 2])
    assert not dm.valid[0, 2] and not dm.valid[1, 2]
    assert dm.W[0, 2] == 0.0
    assert dm.n_valid_edges == 1


# --------------------------------------------------------------------------- #
# (2) End-to-end: known geometry -> distances + relative layout (>=3 devices)
# --------------------------------------------------------------------------- #

# A spread-out 2-D layout (meters). Non-collinear so the 2-D fit is well posed.
LAYOUT_XY = {
    "d1": (0.0, 0.0),
    "d2": (10.0, 0.0),
    "d3": (3.0, 8.0),
    "d4": (12.0, 7.0),
}


def _true_distance(a: str, b: str) -> float:
    ax, ay = LAYOUT_XY[a]
    bx, by = LAYOUT_XY[b]
    return math.hypot(ax - bx, ay - by)


def _feed_all_pairs(
    rc: RangingCoordinator, ids: list[str], *, repeats: int = 1
) -> None:
    """Feed (optionally repeated) clean rounds for every unordered pair."""
    rnd = 0
    for rep in range(repeats):
        for a, b in itertools.combinations(ids, 2):
            # Vary clock offsets per round to prove they keep cancelling.
            _feed_round(
                rc,
                rnd,
                a,
                b,
                _true_distance(a, b),
                off_a=0.1 * rnd,
                off_b=-0.05 * rnd,
                t_emit=float(rnd),
            )
            rnd += 1


def _pairwise_distance_error(ids, xy) -> float:
    """Max abs error between layout pairwise distances and truth (meters)."""
    pos = {did: np.asarray(p, dtype=float) for did, p in zip(ids, xy)}
    worst = 0.0
    for a, b in itertools.combinations(ids, 2):
        recovered = float(np.linalg.norm(pos[a] - pos[b]))
        worst = max(worst, abs(recovered - _true_distance(a, b)))
    return worst


def test_coordinator_recovers_all_pairwise_distances():
    ids = list(LAYOUT_XY)
    rc = RangingCoordinator()
    _feed_all_pairs(rc, ids)

    by_pair = {frozenset((r["a"], r["b"])): r["m"] for r in rc.distances()}
    for a, b in itertools.combinations(ids, 2):
        assert by_pair[frozenset((a, b))] == pytest.approx(
            _true_distance(a, b), abs=1e-6
        )


def test_session_emits_relative_layout_matching_truth():
    """3-4 devices, no GPS: relative.xy_m pairwise distances match truth."""
    clock = FakeClock()
    s = Session(time_fn=clock)
    ids = list(LAYOUT_XY)
    for did in ids:
        s.upsert_device(did, did)
        # Report (no gps, no audio) so the device is online; ranging is fed below.
        s.report(did, {"t_client_ms": 0, "gps": None, "audio": None})

    # Feed every pair's half-exchanges through the public report() path.
    rnd = 0
    for a, b in itertools.combinations(ids, 2):
        d = _true_distance(a, b)
        t1, t2, t3, t4 = _round_timestamps(d, off_a=0.3, off_b=-0.2)
        s.report(a, {"ranging": [{"round": rnd, "role": "init", "t1": t1, "t4": t4}]})
        s.report(b, {"ranging": [{"round": rnd, "role": "resp", "t2": t2, "t3": t3}]})
        rnd += 1

    out = s.state()

    # distances present and correct.
    by_pair = {frozenset((r["a"], r["b"])): r["m"] for r in out["distances"]}
    for a, b in itertools.combinations(ids, 2):
        assert by_pair[frozenset((a, b))] == pytest.approx(
            _true_distance(a, b), abs=1e-3
        )

    # relative layout present, covering all four devices.
    rel = out["relative"]
    assert rel is not None
    assert set(rel["device_ids"]) == set(ids)
    assert len(rel["xy_m"]) == len(ids)
    # Pairwise distances of the recovered layout match truth (rigid-gauge free).
    err = _pairwise_distance_error(rel["device_ids"], rel["xy_m"])
    assert err < 0.5, f"relative layout pairwise error {err:.3f} m"


def test_session_relative_layout_three_devices():
    """Exactly 3 devices still yields a 2-D relative layout matching truth."""
    clock = FakeClock()
    s = Session(time_fn=clock)
    ids = ["d1", "d2", "d3"]
    for did in ids:
        s.upsert_device(did, did)
        s.report(did, {"t_client_ms": 0, "gps": None, "audio": None})

    rnd = 0
    for a, b in itertools.combinations(ids, 2):
        d = _true_distance(a, b)
        t1, t2, t3, t4 = _round_timestamps(d)
        s.report(a, {"ranging": [{"round": rnd, "role": "init", "t1": t1, "t4": t4}]})
        s.report(b, {"ranging": [{"round": rnd, "role": "resp", "t2": t2, "t3": t3}]})
        rnd += 1

    rel = s.state()["relative"]
    assert rel is not None
    assert set(rel["device_ids"]) == set(ids)
    err = _pairwise_distance_error(rel["device_ids"], rel["xy_m"])
    assert err < 0.5


def test_session_relative_layout_aligns_to_gps():
    """With GPS fixes present, the relative layout lands in the GPS/ENU frame.

    The session aligns the relative cloud onto the GPS devices' positions in an
    ENU frame about *their centroid*, so the emitted ``xy_m`` is centroid-ENU.
    We reconstruct each device's lat/lon from that centroid and check it matches
    its GPS truth — i.e. the GPS-denied frame and the GPS frame coincide.
    """
    from dronetracking.geo import enu_to_latlon, haversine_m

    origin = (32.0853, 34.7818)
    clock = FakeClock()
    s = Session(time_fn=clock)
    ids = list(LAYOUT_XY)

    # Treat LAYOUT_XY as ENU (east, north) about the origin and give every device
    # a GPS fix there, so the aligned relative layout should match these latlons.
    truth_ll = {}
    for did in ids:
        e, n = LAYOUT_XY[did]
        lat, lon = enu_to_latlon(float(e), float(n), origin)
        truth_ll[did] = (float(lat), float(lon))
        s.upsert_device(did, did)
        s.report(
            did,
            {
                "t_client_ms": 0,
                "gps": {"lat": lat, "lon": lon, "accuracy_m": 5.0},
                "audio": None,
            },
        )

    rnd = 0
    for a, b in itertools.combinations(ids, 2):
        d = _true_distance(a, b)
        t1, t2, t3, t4 = _round_timestamps(d)
        s.report(a, {"ranging": [{"round": rnd, "role": "init", "t1": t1, "t4": t4}]})
        s.report(b, {"ranging": [{"round": rnd, "role": "resp", "t2": t2, "t3": t3}]})
        rnd += 1

    rel = s.state()["relative"]
    assert rel is not None

    # The session's alignment origin is the centroid of the kept GPS latlons.
    kept = rel["device_ids"]
    centroid = (
        float(np.mean([truth_ll[d][0] for d in kept])),
        float(np.mean([truth_ll[d][1] for d in kept])),
    )
    # Each aligned (x, y) is centroid-ENU; back to lat/lon it must match truth.
    for did, (x, y) in zip(kept, rel["xy_m"]):
        lat, lon = enu_to_latlon(float(x), float(y), centroid)
        tlat, tlon = truth_ll[did]
        err = float(haversine_m(lat, lon, tlat, tlon))
        assert err < 1.0, f"{did} aligned {err:.3f} m from GPS truth"


# --------------------------------------------------------------------------- #
# (3) Two devices -> single distance, no relative layout
# --------------------------------------------------------------------------- #


def test_two_devices_single_distance_no_layout():
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did in ("d1", "d2"):
        s.upsert_device(did, did)
        s.report(did, {"t_client_ms": 0, "gps": None, "audio": None})

    true_d = _true_distance("d1", "d2")  # 10.0 m
    t1, t2, t3, t4 = _round_timestamps(true_d, off_a=7.0, off_b=-3.0)
    s.report("d1", {"ranging": [{"round": 0, "role": "init", "t1": t1, "t4": t4}]})
    s.report("d2", {"ranging": [{"round": 0, "role": "resp", "t2": t2, "t3": t3}]})

    out = s.state()
    assert out["relative"] is None  # 1-D: no 2-D layout
    assert len(out["distances"]) == 1
    rec = out["distances"][0]
    assert {rec["a"], rec["b"]} == {"d1", "d2"}
    assert rec["m"] == pytest.approx(true_d, abs=1e-3)


def test_state_includes_command_for_two_online_devices():
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did in ("d1", "d2"):
        s.upsert_device(did, did)
        s.report(did, {"t_client_ms": 0, "gps": None, "audio": None})

    out = s.state()
    assert "command" in out and "distances" in out and "relative" in out
    cmd = out["command"]["ranging"]
    assert cmd is not None
    assert {cmd["initiator"], cmd["responder"]} == {"d1", "d2"}
    assert cmd["chirp"] == DEFAULT_CHIRP


def test_command_is_null_with_fewer_than_two_online():
    clock = FakeClock()
    s = Session(time_fn=clock)
    s.upsert_device("d1", "d1")
    s.report("d1", {"t_client_ms": 0, "gps": None, "audio": None})
    out = s.state()
    assert out["command"]["ranging"] is None
    assert out["relative"] is None
