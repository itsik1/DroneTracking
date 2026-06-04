"""Ph9 — GPS-DENIED OPERATION: georeferencing that survives GPS blackouts.

The georeferencer ties the gauge-free LOCAL frame to ENU with a similarity solved
from GPS anchors (:func:`dronetracking.estimation.georeference.solve_transform`).
That solve is only possible while GPS is available. This module makes the
georeferencer robust to *blackout windows* during which no fresh transform can be
obtained, and to slow *drift* of the true LOCAL->ENU transform (e.g. the sensor
array creeps) between solves:

* **Hold (dead-reckon).** During a blackout we keep applying the last transform
  observed before GPS dropped. The georeferenced track keeps flowing; its error
  grows only as far as the array drifts over the blackout (bounded, not blowing
  up).
* **Blend (smooth recovery).** When GPS returns, the freshly re-solved transform
  generally differs from the stale held one. Snapping to it would teleport the
  track. Instead we BLEND the *applied ENU positions* from held->fresh over
  ``recovery_blend_s`` seconds, so there is no discontinuity — the accumulated
  drift is corrected smoothly.

The transform is treated as time-varying via an optional ``transform_provider(t)``
giving the TRUE LOCAL->ENU similarity at time ``t`` (the value the georeferencer
would re-solve to if GPS were up then). The default provider is a single static
``solve_transform(layout, anchor_gps, origin)`` — the classic fixed-array case,
where holding is exact and there is nothing to correct.

This file is on the estimation side of the ground-truth firewall: it imports NO
``dronetracking.sim`` logic. The ``AnchorGps`` *type* is a frozen contract
dataclass; passing instances in is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from dronetracking import geo
from dronetracking.estimation.georeference import solve_transform
from dronetracking.estimation.interfaces import GeoTrack, RelativeLayout, Track
from dronetracking.transforms import Similarity

Origin = Tuple[float, float]  # (lat_deg, lon_deg)
BlackoutWindows = Sequence[Tuple[float, float]]
TransformProvider = Callable[[float], Similarity]


@dataclass
class BlackoutGeoTrack(GeoTrack):
    """A :class:`GeoTrack` that also records GPS availability per frame.

    It IS a ``GeoTrack`` (subclass), so it drops into anything expecting one. Two
    extra boolean arrays, aligned with ``times_s``, expose the blackout handling:

    * ``dead_reckoned[k]`` — frame ``k`` was georeferenced by HOLDING the last good
      transform (GPS was down), not by a fresh solve.
    * ``gps_available[k]`` — GPS was usable at frame ``k`` (the logical inverse of
      ``dead_reckoned``).
    """

    dead_reckoned: np.ndarray = None  # (T,) bool — frame held/dead-reckoned
    gps_available: np.ndarray = None  # (T,) bool — GPS usable at frame


def gps_status(
    track_times: np.ndarray,
    blackout_windows: BlackoutWindows,
) -> np.ndarray:
    """Per-frame GPS availability for ``track_times``.

    Returns a boolean array the same shape as ``track_times`` where ``True`` means
    GPS is available. A frame is UNavailable iff its time lies inside any closed
    blackout window ``[start, end]`` (inclusive endpoints) — matching
    :meth:`dronetracking.sim.scenario.Scenario.gps_available`.
    """
    times = np.asarray(track_times, dtype=float)
    available = np.ones(times.shape, dtype=bool)
    for start, end in blackout_windows:
        available &= ~((times >= float(start)) & (times <= float(end)))
    return available


def _enu_from_transform(transform: Similarity, positions_local: np.ndarray) -> np.ndarray:
    """Apply a LOCAL->ENU similarity to (T,3) local positions, returning (T,3) ENU."""
    return np.atleast_2d(transform.apply(np.asarray(positions_local, dtype=float)))


def _cov_to_enu(transform: Similarity, cov_local: np.ndarray) -> np.ndarray:
    """Rotate/scale a single 3x3 local covariance into ENU: ``s^2 R cov R.T``."""
    R = np.asarray(transform.R, dtype=float)
    s2 = float(transform.scale) ** 2
    return s2 * (R @ np.asarray(cov_local, dtype=float) @ R.T)


def georeference_with_blackout(
    layout: RelativeLayout,
    anchor_gps: Sequence,
    track: Track,
    origin: Origin,
    blackout_windows: BlackoutWindows,
    recovery_blend_s: float = 2.0,
    transform_provider: Optional[TransformProvider] = None,
) -> BlackoutGeoTrack:
    """Georeference a LOCAL-frame track, holding through GPS blackouts and
    blending smoothly on recovery.

    Behaves like :func:`dronetracking.estimation.georeference.georeference_track`
    OUTSIDE ``blackout_windows`` (GPS up → use the freshly observable LOCAL->ENU
    transform). DURING a blackout the last good transform is held (dead-reckoning).
    On GPS RETURN the transform is re-solved and the applied ENU positions are
    blended held->fresh over ``recovery_blend_s`` so the georeferenced track has no
    discontinuity.

    Args:
        layout: device positions in the LOCAL frame (used by the default provider's
            static ``solve_transform``).
        anchor_gps: GPS-anchored devices (``AnchorGps``), passed to the default
            provider's ``solve_transform``.
        track: the LOCAL-frame target track to georeference (all frames returned).
        origin: ``(lat, lon)`` ENU tangent-plane origin.
        blackout_windows: ``[(start, end), ...]`` GPS-denied closed intervals.
        recovery_blend_s: seconds over which to blend held->fresh after each
            recovery (<=0 snaps immediately). Defaults to 2.0.
        transform_provider: optional ``t -> Similarity`` giving the TRUE/observable
            LOCAL->ENU transform at time ``t`` (to exercise drift). Defaults to a
            single static ``solve_transform(layout, anchor_gps, origin)``.

    Returns:
        A :class:`BlackoutGeoTrack` covering every track frame, exposing which
        frames were dead-reckoned via ``.dead_reckoned`` / ``.gps_available``.
    """
    times = np.asarray(track.times_s, dtype=float)
    positions_local = np.asarray(track.positions_local, dtype=float)
    covs_local = np.asarray(track.covariances, dtype=float)
    n = times.size

    if transform_provider is None:
        static = solve_transform(layout, anchor_gps, origin)
        transform_provider = lambda _t: static  # noqa: E731 (tiny static closure)

    available = gps_status(times, blackout_windows)

    enu = np.empty((n, 3), dtype=float)
    cov_enu = np.empty((n, 3, 3), dtype=float)

    # Transform held from the last GPS-up frame; carried through blackouts.
    held: Optional[Similarity] = None
    # Active recovery blend: frames in [blend_start_t, blend_start_t + blend) ramp
    # from `held_at_recovery` (frozen) to the live fresh transform.
    blending = False
    blend_start_t = 0.0
    held_at_recovery: Optional[Similarity] = None
    blend = max(0.0, float(recovery_blend_s))

    prev_available = True  # so a blackout starting at frame 0 still "enters"

    for k in range(n):
        t = float(times[k])
        p = positions_local[k]

        if available[k]:
            fresh = transform_provider(t)

            just_recovered = (not prev_available) and held is not None and blend > 0.0
            if just_recovered:
                # GPS just came back after a blackout: start blending from the
                # stale held transform to the fresh one over `recovery_blend_s`.
                blending = True
                blend_start_t = t
                held_at_recovery = held

            if blending:
                w = (t - blend_start_t) / blend  # 0 -> 1 across the window
                if w >= 1.0:
                    blending = False
                    enu_k = fresh.apply(p)
                    cov_k = _cov_to_enu(fresh, covs_local[k])
                else:
                    # Blend the APPLIED ENU positions (guarantees C0 continuity
                    # regardless of how the rotations differ). Covariance follows
                    # the converging-to (fresh) transform.
                    enu_held = held_at_recovery.apply(p)
                    enu_fresh = fresh.apply(p)
                    enu_k = (1.0 - w) * enu_held + w * enu_fresh
                    cov_k = _cov_to_enu(fresh, covs_local[k])
            else:
                enu_k = fresh.apply(p)
                cov_k = _cov_to_enu(fresh, covs_local[k])

            # GPS is up: refresh the held transform so the dead-reckoning baseline
            # for the NEXT blackout is current.
            held = fresh
        else:
            # Blackout: hold the last good transform (dead-reckon). If GPS was
            # never up yet, fall back to this frame's provider value so we still
            # produce a position (degenerate edge case).
            blending = False
            if held is None:
                held = transform_provider(t)
            enu_k = held.apply(p)
            cov_k = _cov_to_enu(held, covs_local[k])

        enu[k] = np.asarray(enu_k, dtype=float).reshape(3)
        cov_enu[k] = cov_k
        prev_available = bool(available[k])

    lat, lon = geo.enu_to_latlon(enu[:, 0], enu[:, 1], origin)
    latlon = np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])

    dead_reckoned = ~available

    return BlackoutGeoTrack(
        times_s=times,
        latlon=latlon,
        altitude_m=enu[:, 2].copy(),
        cov_enu=cov_enu,
        dead_reckoned=dead_reckoned,
        gps_available=available,
    )
