import numpy as np
import pytest

from dronetracking.geo import latlon_to_enu, enu_to_latlon, haversine_m

ORIGIN = (32.0853, 34.7818)  # Tel Aviv-ish reference


def test_origin_maps_to_zero():
    e, n = latlon_to_enu(ORIGIN[0], ORIGIN[1], ORIGIN)
    assert e == pytest.approx(0.0, abs=1e-9)
    assert n == pytest.approx(0.0, abs=1e-9)


def test_enu_roundtrip_recovers_latlon():
    lat, lon = 32.0901, 34.7760
    e, n = latlon_to_enu(lat, lon, ORIGIN)
    lat2, lon2 = enu_to_latlon(e, n, ORIGIN)
    assert lat2 == pytest.approx(lat, abs=1e-9)
    assert lon2 == pytest.approx(lon, abs=1e-9)


def test_known_displacement_roundtrips_exactly():
    # 100 m north, 0 m east -> convert to lat/lon and back
    lat, lon = enu_to_latlon(0.0, 100.0, ORIGIN)
    e, n = latlon_to_enu(lat, lon, ORIGIN)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert n == pytest.approx(100.0, abs=1e-6)


def test_haversine_matches_enu_distance_at_field_scale():
    lat, lon = enu_to_latlon(100.0, 100.0, ORIGIN)  # 100 m E, 100 m N
    d = haversine_m(ORIGIN[0], ORIGIN[1], lat, lon)
    assert d == pytest.approx(np.hypot(100.0, 100.0), rel=1e-3)


def test_haversine_zero_for_same_point():
    assert haversine_m(*ORIGIN, *ORIGIN) == pytest.approx(0.0, abs=1e-9)


def test_latlon_to_enu_is_vectorized():
    lats = np.array([32.0901, 32.0853])
    lons = np.array([34.7760, 34.7818])
    e, n = latlon_to_enu(lats, lons, ORIGIN)
    assert np.asarray(e).shape == (2,)
    assert np.asarray(n).shape == (2,)
    # second point IS the origin -> maps to (0, 0)
    assert n[1] == pytest.approx(0.0, abs=1e-9)
    assert e[1] == pytest.approx(0.0, abs=1e-9)
