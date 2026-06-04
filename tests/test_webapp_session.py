"""Tests for the adaptive web-app :class:`Session` (the brain).

Strategy (no hardware, fully deterministic):

* Lay devices out at a known **ENU** geometry and convert to lat/lon with
  ``geo.enu_to_latlon`` so the fixtures are real coordinates.
* Place a synthetic source at a known ENU point and set every device's reported
  ``level = sqrt(A) / r_i`` (so received power ``= A / r_i**2`` exactly matches the
  model the Session inverts).
* Drive an **injected clock** so pruning is deterministic.

We assert the Session recovers the source lat/lon to within a few meters
(measured with ``geo.haversine_m``) from >= 3 detecting+positioned devices,
falls back to a region for exactly 2, and emits ``None`` for 0 detecting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dronetracking.geo import enu_to_latlon, haversine_m, latlon_to_enu
from dronetracking.webapp.session import Session

# A field-scale reference origin (Tel Aviv-ish), matching the rest of the suite.
ORIGIN = (32.0853, 34.7818)

# True source and amplitude used across the energy-localization tests.
SOURCE_EN = (15.0, 40.0)  # (east, north) meters relative to ORIGIN
TRUE_A = 4.0  # amplitude (power = A / r^2)
EPS = 1e-6  # power-model softening, must match the Session


class FakeClock:
    """A hand-cranked monotonic clock for deterministic pruning tests."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _enu_to_latlon_point(east: float, north: float) -> tuple[float, float]:
    lat, lon = enu_to_latlon(float(east), float(north), ORIGIN)
    return float(lat), float(lon)


def _level_for(device_en: tuple[float, float]) -> float:
    """Reported linear amplitude for a device given the true source/amplitude.

    ``power = A / r^2`` so ``level = sqrt(A) / r``.
    """
    dx = device_en[0] - SOURCE_EN[0]
    dy = device_en[1] - SOURCE_EN[1]
    r = math.hypot(dx, dy)
    return math.sqrt(TRUE_A) / max(r, 1e-9)


def _gps_payload(lat: float, lon: float) -> dict:
    return {"lat": lat, "lon": lon, "accuracy_m": 5.0}


def _audio_payload(level: float, detected: bool, confidence: float = 0.8) -> dict:
    return {
        "level": float(level),
        "detected": bool(detected),
        "confidence": float(confidence),
        "peak_hz": 180.0,
    }


def _report(
    session: Session,
    device_id: str,
    *,
    en: tuple[float, float],
    detected: bool,
    t_client_ms: int = 0,
) -> tuple[float, float]:
    """Place a device at ``en`` (ENU) with a level matching the synthetic source.

    Returns the device's (lat, lon) for downstream assertions.
    """
    lat, lon = _enu_to_latlon_point(*en)
    level = _level_for(en)
    session.report(
        device_id,
        {
            "t_client_ms": t_client_ms,
            "gps": _gps_payload(lat, lon),
            "audio": _audio_payload(level, detected),
        },
    )
    return lat, lon


# A spread-out 4-device layout (meters, ENU). Surrounds the source so the
# energy fit is well-conditioned.
LAYOUT_EN = {
    "d1": (-60.0, -10.0),
    "d2": (70.0, 5.0),
    "d3": (0.0, 90.0),
    "d4": (40.0, -50.0),
}


# --------------------------------------------------------------------------- #
# Core API / shape
# --------------------------------------------------------------------------- #


def test_state_shape_is_contract_compliant():
    s = Session(time_fn=FakeClock())
    s.upsert_device("d1", "Alice")
    out = s.state()

    assert set(out.keys()) == {"devices", "source", "network", "computed", "note"}
    assert isinstance(out["devices"], list)
    assert isinstance(out["note"], str)

    net = out["network"]
    assert set(net.keys()) == {"n_devices", "n_gps", "n_detecting", "connected"}

    comp = out["computed"]
    assert set(comp.keys()) == {"positioning", "source"}
    assert comp["positioning"] in ("gps", "ranging", "none")
    assert comp["source"] in ("energy", "region", "none")

    dev = out["devices"][0]
    assert set(dev.keys()) == {
        "id",
        "name",
        "lat",
        "lon",
        "has_gps",
        "has_mic",
        "level",
        "detected",
        "confidence",
        "online",
    }


def test_upsert_creates_device_and_renames():
    s = Session(time_fn=FakeClock())
    s.upsert_device("d1", "Alice")
    assert s.state()["devices"][0]["name"] == "Alice"
    # Re-upsert with a new name updates in place (no duplicate device).
    s.upsert_device("d1", "Renamed")
    devs = s.state()["devices"]
    assert len(devs) == 1
    assert devs[0]["name"] == "Renamed"


def test_upsert_with_none_name_uses_id_fallback():
    s = Session(time_fn=FakeClock())
    s.upsert_device("d7", None)
    name = s.state()["devices"][0]["name"]
    assert name and "d7" in name


def test_report_auto_registers_unknown_device():
    s = Session(time_fn=FakeClock())
    s.report("ghost", {"t_client_ms": 1, "gps": None, "audio": None})
    ids = {d["id"] for d in s.state()["devices"]}
    assert "ghost" in ids


# --------------------------------------------------------------------------- #
# Capability flags: has_gps / has_mic latch once seen
# --------------------------------------------------------------------------- #


def test_capability_flags_latch_once_seen():
    s = Session(time_fn=FakeClock())
    s.upsert_device("d1", "A")

    # First report: no gps, audio present.
    s.report("d1", {"t_client_ms": 1, "gps": None, "audio": _audio_payload(0.1, False)})
    dev = s.state()["devices"][0]
    assert dev["has_gps"] is False
    assert dev["has_mic"] is True

    # Then a gps fix arrives, audio now null -> both flags should stay/latch true.
    s.report("d1", {"t_client_ms": 2, "gps": _gps_payload(*ORIGIN), "audio": None})
    dev = s.state()["devices"][0]
    assert dev["has_gps"] is True
    assert dev["has_mic"] is True  # latched from before


# --------------------------------------------------------------------------- #
# Positioning
# --------------------------------------------------------------------------- #


def test_gps_positioning_places_devices_and_sets_mode():
    s = Session(time_fn=FakeClock())
    lat, lon = _enu_to_latlon_point(10.0, 20.0)
    s.upsert_device("d1", "A")
    s.report("d1", {"t_client_ms": 1, "gps": _gps_payload(lat, lon), "audio": None})

    out = s.state()
    dev = out["devices"][0]
    assert dev["lat"] == pytest.approx(lat, abs=1e-9)
    assert dev["lon"] == pytest.approx(lon, abs=1e-9)
    assert out["computed"]["positioning"] == "gps"


def test_no_gps_means_null_position_and_no_positioning():
    s = Session(time_fn=FakeClock())
    s.upsert_device("d1", "A")
    s.report("d1", {"t_client_ms": 1, "gps": None, "audio": _audio_payload(0.2, True)})

    out = s.state()
    dev = out["devices"][0]
    assert dev["lat"] is None
    assert dev["lon"] is None
    assert out["computed"]["positioning"] == "none"


# --------------------------------------------------------------------------- #
# Energy source localization (the key algorithm)
# --------------------------------------------------------------------------- #


def test_energy_localization_recovers_source_within_a_few_meters():
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did, en in LAYOUT_EN.items():
        s.upsert_device(did, did)
        _report(s, did, en=en, detected=True)

    out = s.state()
    assert out["computed"]["source"] == "energy"
    src = out["source"]
    assert src is not None

    true_lat, true_lon = _enu_to_latlon_point(*SOURCE_EN)
    err = float(haversine_m(src["lat"], src["lon"], true_lat, true_lon))
    assert err < 5.0, f"recovered source {err:.2f} m from truth"

    # Quality fields are sane.
    assert src["error_m"] >= 0.0
    assert 0.0 <= src["confidence"] <= 1.0


def test_energy_localization_three_devices_is_a_point_fix():
    clock = FakeClock()
    s = Session(time_fn=clock)
    three = dict(list(LAYOUT_EN.items())[:3])
    for did, en in three.items():
        s.upsert_device(did, did)
        _report(s, did, en=en, detected=True)

    out = s.state()
    assert out["computed"]["source"] == "energy"
    true_lat, true_lon = _enu_to_latlon_point(*SOURCE_EN)
    err = float(haversine_m(out["source"]["lat"], out["source"]["lon"], true_lat, true_lon))
    assert err < 5.0


def test_only_detecting_devices_count_toward_the_fix():
    """A 4th non-detecting device must not break a clean 3-device fix."""
    clock = FakeClock()
    s = Session(time_fn=clock)
    items = list(LAYOUT_EN.items())
    for did, en in items[:3]:
        s.upsert_device(did, did)
        _report(s, did, en=en, detected=True)
    # 4th device is positioned but NOT detecting -> excluded from energy fit.
    did4, en4 = items[3]
    s.upsert_device(did4, did4)
    _report(s, did4, en=en4, detected=False)

    out = s.state()
    assert out["computed"]["source"] == "energy"
    assert out["network"]["n_detecting"] == 3
    true_lat, true_lon = _enu_to_latlon_point(*SOURCE_EN)
    err = float(haversine_m(out["source"]["lat"], out["source"]["lon"], true_lat, true_lon))
    assert err < 5.0


def test_two_detecting_devices_give_a_region_not_none():
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did in ("d1", "d2"):
        s.upsert_device(did, did)
        _report(s, did, en=LAYOUT_EN[did], detected=True)

    out = s.state()
    assert out["computed"]["source"] == "region"
    src = out["source"]
    assert src is not None
    # The region point is between the two devices (weighted toward the louder one,
    # i.e. the one closer to the true source), so it sits in their bounding box.
    lat1, lon1 = _enu_to_latlon_point(*LAYOUT_EN["d1"])
    lat2, lon2 = _enu_to_latlon_point(*LAYOUT_EN["d2"])
    assert min(lat1, lat2) - 1e-6 <= src["lat"] <= max(lat1, lat2) + 1e-6
    assert min(lon1, lon2) - 1e-6 <= src["lon"] <= max(lon1, lon2) + 1e-6
    assert src["error_m"] >= 0.0


def test_region_is_weighted_toward_the_louder_device():
    """The louder (closer-to-source) device should pull the region toward it."""
    clock = FakeClock()
    s = Session(time_fn=clock)
    # d_near is much closer to the source than d_far -> higher level.
    near_en = (15.0, 50.0)  # 10 m from source
    far_en = (15.0, -60.0)  # 100 m from source
    s.upsert_device("near", "near")
    s.upsert_device("far", "far")
    _report(s, "near", en=near_en, detected=True)
    _report(s, "far", en=far_en, detected=True)

    out = s.state()
    assert out["computed"]["source"] == "region"
    # Midpoint north is -5; weighted point should be pulled north toward "near".
    near_lat, _ = _enu_to_latlon_point(*near_en)
    far_lat, _ = _enu_to_latlon_point(*far_en)
    mid_lat = 0.5 * (near_lat + far_lat)
    assert out["source"]["lat"] > mid_lat


def test_zero_detecting_devices_yields_no_source():
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did, en in LAYOUT_EN.items():
        s.upsert_device(did, did)
        _report(s, did, en=en, detected=False)  # positioned but silent

    out = s.state()
    assert out["source"] is None
    assert out["computed"]["source"] == "none"
    assert out["network"]["n_detecting"] == 0


def test_detecting_without_position_yields_no_source():
    """Energy localization needs positions; detecting-but-unplaced -> none."""
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did in ("d1", "d2", "d3"):
        s.upsert_device(did, did)
        # detected, but NO gps -> no ENU position available
        s.report(did, {"t_client_ms": 1, "gps": None, "audio": _audio_payload(0.5, True)})

    out = s.state()
    assert out["source"] is None
    assert out["computed"]["source"] == "none"


# --------------------------------------------------------------------------- #
# Network summary
# --------------------------------------------------------------------------- #


def test_network_summary_counts():
    clock = FakeClock()
    s = Session(time_fn=clock)
    # d1: gps + detecting; d2: gps + not detecting; d3: no gps + detecting
    s.upsert_device("d1", "d1")
    s.upsert_device("d2", "d2")
    s.upsert_device("d3", "d3")
    _report(s, "d1", en=LAYOUT_EN["d1"], detected=True)
    _report(s, "d2", en=LAYOUT_EN["d2"], detected=False)
    s.report("d3", {"t_client_ms": 1, "gps": None, "audio": _audio_payload(0.4, True)})

    net = s.state()["network"]
    assert net["n_devices"] == 3
    assert net["n_gps"] == 2
    assert net["n_detecting"] == 2
    assert net["connected"] is True


def test_network_not_connected_when_all_stale():
    clock = FakeClock()
    s = Session(time_fn=clock)
    s.upsert_device("d1", "d1")
    _report(s, "d1", en=LAYOUT_EN["d1"], detected=True)
    assert s.state()["network"]["connected"] is True

    # Advance past the online window; without pruning, the device is offline.
    clock.advance(10.0)
    out = s.state()
    assert out["devices"][0]["online"] is False
    assert out["network"]["connected"] is False


# --------------------------------------------------------------------------- #
# Pruning (deterministic via injected clock)
# --------------------------------------------------------------------------- #


def test_prune_drops_devices_past_max_age():
    clock = FakeClock()
    s = Session(time_fn=clock)
    s.upsert_device("d1", "d1")
    _report(s, "d1", en=LAYOUT_EN["d1"], detected=True)
    assert len(s.state()["devices"]) == 1

    clock.advance(6.0)  # past default max_age_s=5.0
    s.prune(max_age_s=5.0)
    assert len(s.state()["devices"]) == 0


def test_prune_keeps_recent_devices():
    clock = FakeClock()
    s = Session(time_fn=clock)
    s.upsert_device("d1", "d1")
    _report(s, "d1", en=LAYOUT_EN["d1"], detected=True)

    clock.advance(2.0)  # within max_age
    s.prune(max_age_s=5.0)
    assert len(s.state()["devices"]) == 1


def test_fresh_report_resets_last_seen_for_pruning():
    clock = FakeClock()
    s = Session(time_fn=clock)
    s.upsert_device("d1", "d1")
    _report(s, "d1", en=LAYOUT_EN["d1"], detected=True)

    clock.advance(4.0)
    _report(s, "d1", en=LAYOUT_EN["d1"], detected=True)  # refresh last-seen
    clock.advance(2.0)  # 2 s since refresh -> still fresh
    s.prune(max_age_s=5.0)
    assert len(s.state()["devices"]) == 1


def test_stale_device_excluded_from_localization_before_prune():
    """An offline device must not feed the energy fit even before prune()."""
    clock = FakeClock()
    s = Session(time_fn=clock)
    # Three good detecting devices at t0.
    for did in ("d1", "d2", "d3"):
        s.upsert_device(did, did)
        _report(s, did, en=LAYOUT_EN[did], detected=True)
    # A 4th device reports a bogus position at t0, then goes stale.
    s.upsert_device("d4", "d4")
    bogus_lat, bogus_lon = _enu_to_latlon_point(500.0, 500.0)
    s.report(
        "d4",
        {"t_client_ms": 1, "gps": _gps_payload(bogus_lat, bogus_lon), "audio": _audio_payload(0.9, True)},
    )

    # Move time forward so only d4 is stale -> refresh d1..d3.
    clock.advance(6.0)
    for did in ("d1", "d2", "d3"):
        _report(s, did, en=LAYOUT_EN[did], detected=True)

    out = s.state()
    # d4 is offline and excluded; the fit stays clean.
    assert out["network"]["n_detecting"] == 3
    true_lat, true_lon = _enu_to_latlon_point(*SOURCE_EN)
    err = float(haversine_m(out["source"]["lat"], out["source"]["lon"], true_lat, true_lon))
    assert err < 5.0


# --------------------------------------------------------------------------- #
# Robustness / noise
# --------------------------------------------------------------------------- #


def test_energy_localization_is_robust_to_small_level_noise():
    """With small multiplicative noise on levels, the fix degrades gracefully."""
    rng = np.random.default_rng(0)
    clock = FakeClock()
    s = Session(time_fn=clock)
    for did, en in LAYOUT_EN.items():
        s.upsert_device(did, did)
        lat, lon = _enu_to_latlon_point(*en)
        level = _level_for(en) * float(rng.normal(1.0, 0.05))  # 5% noise
        s.report(
            did,
            {"t_client_ms": 0, "gps": _gps_payload(lat, lon), "audio": _audio_payload(level, True)},
        )

    out = s.state()
    assert out["computed"]["source"] == "energy"
    true_lat, true_lon = _enu_to_latlon_point(*SOURCE_EN)
    err = float(haversine_m(out["source"]["lat"], out["source"]["lon"], true_lat, true_lon))
    # Looser tolerance under noise, but still a meaningful localization.
    assert err < 15.0
