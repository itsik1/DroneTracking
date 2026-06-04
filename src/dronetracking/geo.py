"""Geodetic conversions at field scale.

A local **equirectangular** (tangent-plane) projection about a fixed origin maps
between geodetic lat/lon (degrees) and local East-North coordinates (meters). At
field scale (hundreds of meters) this is accurate to well under a millimeter and,
crucially, is an *exact* analytic inverse so round-trips lose no precision.

Altitude/Up is not projected here — it maps directly to a height offset and is
handled by callers (georeferencing, viz).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

# IUGG mean Earth radius (R1). The exact value is immaterial at field scale because
# forward/inverse use the same constant, but a standard value keeps haversine honest.
EARTH_RADIUS_M = 6371008.8

Origin = Tuple[float, float]  # (lat_deg, lon_deg)
ArrayLike = Union[float, np.ndarray]


def latlon_to_enu(lat: ArrayLike, lon: ArrayLike, origin: Origin) -> Tuple[ArrayLike, ArrayLike]:
    """Project geodetic lat/lon (deg) to local East/North meters about ``origin``."""
    lat0, lon0 = origin
    cos_phi0 = np.cos(np.radians(lat0))
    east = EARTH_RADIUS_M * np.radians(np.asarray(lon, dtype=float) - lon0) * cos_phi0
    north = EARTH_RADIUS_M * np.radians(np.asarray(lat, dtype=float) - lat0)
    return east, north


def enu_to_latlon(east: ArrayLike, north: ArrayLike, origin: Origin) -> Tuple[ArrayLike, ArrayLike]:
    """Inverse of :func:`latlon_to_enu`: local East/North meters -> lat/lon (deg)."""
    lat0, lon0 = origin
    cos_phi0 = np.cos(np.radians(lat0))
    lat = lat0 + np.degrees(np.asarray(north, dtype=float) / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(np.asarray(east, dtype=float) / (EARTH_RADIUS_M * cos_phi0))
    return lat, lon


def haversine_m(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> ArrayLike:
    """Great-circle distance in meters between two geodetic points (degrees)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2, dtype=float) - lat1)
    dlam = np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
