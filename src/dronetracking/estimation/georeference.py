"""Georeferencing: tie the gauge-free local frame to real-world coordinates.

Relative localization recovers device positions only up to an arbitrary
similarity (rotation + translation + scale + reflection). The GPS-anchored
devices break that gauge: each anchor's reported lat/lon/altitude pins a known
ENU target, so a single least-squares similarity (:func:`transforms.umeyama`)
maps the whole LOCAL frame onto ENU. Tracks are then carried into ENU and
projected to lat/lon via :mod:`dronetracking.geo`.

This module is on the estimation side of the ground-truth firewall: it never
imports ``dronetracking.sim`` logic. (The ``AnchorGps`` *type* is a frozen
contract dataclass — passing instances in is fine; we only read their fields.)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from dronetracking import geo, transforms
from dronetracking.estimation.interfaces import GeoTrack, RelativeLayout, Track
from dronetracking.transforms import Similarity

Origin = Tuple[float, float]  # (lat_deg, lon_deg)

# Minimum spread (meters) in the second principal direction of the anchor cloud.
# Below this the anchors are effectively collinear and the rotation about the
# anchor line is unobservable, so the LOCAL->ENU fit is degenerate.
_COLLINEARITY_EPS_M = 1e-6


def solve_transform(
    layout: RelativeLayout,
    anchor_gps: Sequence,
    origin_latlon: Origin,
    with_scaling: bool = False,
) -> Similarity:
    """Fit the LOCAL-frame -> ENU similarity from GPS-anchored devices.

    For each :class:`~dronetracking.sim.observations.AnchorGps` the ENU target is
    ``(east, north) = geo.latlon_to_enu(lat, lon, origin)`` with ``up = altitude_m``.
    The source points are the anchors' arbitrary-frame positions from ``layout``.
    Returns the :class:`Similarity` mapping LOCAL -> ENU. Reflection is ALLOWED:
    the local frame from distance geometry has an arbitrary chirality (MDS cannot
    observe handedness), so the anchors must be free to resolve it — forcing a proper
    rotation would fail to fit a reflected layout.

    Raises:
        ValueError: if fewer than 3 anchors are given, if an anchor is missing
            from ``layout``, or if the anchors are (near-)collinear.
    """
    anchors = list(anchor_gps)
    if len(anchors) < 3:
        raise ValueError(
            f"georeferencing needs >= 3 GPS anchors, got {len(anchors)}"
        )

    src_list = []
    dst_list = []
    for a in anchors:
        # Source: this anchor's position in the arbitrary local frame.
        try:
            src_list.append(np.asarray(layout.position_of(a.device_id), dtype=float))
        except ValueError as exc:  # device_id not in layout.device_ids
            raise ValueError(
                f"anchor device {a.device_id!r} is not present in the layout"
            ) from exc
        # Target: anchor ENU from its GPS fix.
        east, north = geo.latlon_to_enu(a.lat, a.lon, origin_latlon)
        dst_list.append([float(east), float(north), float(a.altitude_m)])

    src = np.asarray(src_list, dtype=float)  # (M, 3)
    dst = np.asarray(dst_list, dtype=float)  # (M, 3)

    _check_non_collinear(dst)

    return transforms.umeyama(
        src, dst, with_scaling=with_scaling, allow_reflection=True
    )


def _check_non_collinear(points: np.ndarray) -> None:
    """Raise ``ValueError`` if the (M,3) ENU anchors are (near-)collinear.

    The centered anchor cloud's singular values measure its spread along each
    principal axis. The *second* singular value is the extent perpendicular to
    the dominant line; if it collapses to ~0 the anchors lie on a single line and
    rotation about that line is unobservable. A coplanar (but non-collinear)
    ground array — the common field case — has a healthy second singular value
    and passes, even though its third (out-of-plane) value may be tiny.
    """
    centered = points - points.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    # sv is sorted descending; sv[1] is the second-largest. Pad in case of <2
    # singular values (cannot happen for M>=3, 3 columns, but be defensive).
    second = sv[1] if sv.size >= 2 else 0.0
    if second <= _COLLINEARITY_EPS_M:
        raise ValueError(
            "GPS anchors are (near-)collinear "
            f"(2nd singular value {second:.3e} m <= {_COLLINEARITY_EPS_M:.0e} m); "
            "need a non-collinear anchor geometry to fix the local->ENU rotation"
        )


def georeference_track(
    track: Track,
    transform: Similarity,
    origin_latlon: Origin,
) -> GeoTrack:
    """Carry a LOCAL-frame :class:`Track` into geodetic coordinates.

    Each local position maps to ENU via ``transform.apply`` then to lat/lon via
    :func:`geo.enu_to_latlon`; altitude is the ENU up component. Per-step
    covariance is rotated/scaled into ENU as ``s^2 * R @ cov @ R.T`` (``R``, ``s``
    from ``transform``).
    """
    positions_local = np.asarray(track.positions_local, dtype=float)  # (T, 3)
    enu = transform.apply(positions_local)  # (T, 3)
    enu = np.atleast_2d(enu)

    lat, lon = geo.enu_to_latlon(enu[:, 0], enu[:, 1], origin_latlon)
    latlon = np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])
    altitude_m = enu[:, 2].copy()

    R = np.asarray(transform.R, dtype=float)
    s2 = float(transform.scale) ** 2
    covs_local = np.asarray(track.covariances, dtype=float)  # (T, 3, 3)
    # cov_enu[k] = s^2 * R @ cov_local[k] @ R.T  (vectorized over k).
    cov_enu = s2 * np.einsum("ij,tjk,lk->til", R, covs_local, R)

    return GeoTrack(
        times_s=np.asarray(track.times_s),
        latlon=latlon,
        altitude_m=altitude_m,
        cov_enu=cov_enu,
    )
