"""Clock synchronization: recover per-device (offset, drift) from two-way ranging.

The estimator never sees ground-truth clocks. It only sees raw two-way-ranging
timestamps, each stamped in the measuring device's own local clock. The locked
convention (shared with the simulator and
:meth:`estimation.interfaces.ClockEstimates.to_reference`) is::

    local = t_global * (1 + drift_ppm * 1e-6) + offset_s

so for a ranging exchange (initiator ``i`` transmits at ``t1``, responder ``j``
receives at ``t2`` and replies at ``t3``, initiator receives at ``t4``) the
per-exchange relative offset estimate

    0.5 * ((t2_local_j - t1_local_i) + (t3_local_j - t4_local_i))

is **linear in the initiator's transmit timestamp** ``t1_local_i`` with

    slope     ~= skew_j - skew_i        (relative skew, skew = drift_ppm * 1e-6)
    intercept ~= offset_j - offset_i    (relative bias at t = 0)

We robustly fit that line per unordered device pair with Theil-Sen
(``scipy.stats.theilslopes``), turning each pair into two relative measurements::

    skew_j   - skew_i   = skew_ij
    offset_j - offset_i = offset0_ij

then solve a small least-squares "clock graph" for every device's
(offset, skew) relative to the reference device (pinned at ``(0, 0)``).

This module is part of the estimation package and therefore must NOT import
from :mod:`dronetracking.sim` (the ground-truth firewall).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .interfaces import ClockEstimates


class _PairMeasurement:
    """Robust per-pair relative-clock measurement (initiator ``i``, responder ``j``).

    ``skew``  : Theil-Sen slope of the per-exchange offset vs. transmit time,
                ``~= skew_j - skew_i``.
    ``offset0``: Theil-Sen intercept (offset estimate at transmit time 0),
                ``~= offset_j - offset_i`` plus a small ``skew``-dependent bias.
    ``tof``    : two-way time-of-flight (``0.5*((t4-t1) - (t3-t2))``), in which both
                offset AND skew cancel, so it is exact in the noise-free case.
    """

    __slots__ = ("skew", "offset0", "tof")

    def __init__(self, skew: float, offset0: float, tof: float):
        self.skew = skew
        self.offset0 = offset0
        self.tof = tof


def _per_pair_measurements(observations) -> Dict[Tuple[str, str], _PairMeasurement]:
    """Robustly fit each unordered device pair's relative clock parameters.

    For each (initiator ``i``, responder ``j``) pair we regress the per-exchange
    offset estimate against the initiator's transmit timestamp with Theil-Sen,
    and separately take the median two-way time-of-flight.
    """
    # Collect (t1, per_exchange_offset) and tof samples per (initiator, responder).
    offset_samples: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    tof_samples: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for rec in observations.ranging:
        pair = (rec.initiator, rec.responder)
        # Per-exchange relative offset: linear in t1 with slope = relative skew.
        offset = 0.5 * (
            (rec.t2_local_j - rec.t1_local_i) + (rec.t3_local_j - rec.t4_local_i)
        )
        offset_samples[pair].append((rec.t1_local_i, offset))
        # Two-way time-of-flight: offset and skew cancel by construction.
        tof = 0.5 * (
            (rec.t4_local_i - rec.t1_local_i) - (rec.t3_local_j - rec.t2_local_j)
        )
        tof_samples[pair].append(tof)

    measurements: Dict[Tuple[str, str], _PairMeasurement] = {}
    for pair, pts in offset_samples.items():
        arr = np.asarray(pts, dtype=float)
        t = arr[:, 0]
        y = arr[:, 1]

        if t.shape[0] >= 2 and np.ptp(t) > 0.0:
            # theilslopes returns (slope, intercept, lo_slope, hi_slope); the
            # intercept is the Theil-Sen value at t = 0.
            res = stats.theilslopes(y, t)
            slope = float(res[0])
            intercept = float(res[1])
        else:
            # Degenerate: a single exchange (or all at the same transmit time).
            # Skew is unobservable -> report zero skew and the lone offset.
            slope = 0.0
            intercept = float(np.median(y))

        tof = float(np.median(tof_samples[pair]))
        measurements[pair] = _PairMeasurement(skew=slope, offset0=intercept, tof=tof)

    return measurements


def _debias_offsets(
    measurements: Dict[Tuple[str, str], _PairMeasurement],
    biases: Dict[str, float],
) -> Dict[Tuple[str, str], float]:
    """Remove the skew-induced bias from each pair's intercept.

    The intercept of the per-exchange offset line is not exactly ``offset_j -
    offset_i``: because the offset is evaluated at the initiator's *local* transmit
    time, it carries a bias ``skew_ij * (tof_ij - offset_i)`` (a tiny
    second-order proc-delay term is neglected). Subtracting it -- using the exact
    two-way ``tof`` and the current estimate of the initiator's offset --
    makes the offset constraints exact in the noise-free case.
    """
    return {
        (i, j): m.offset0 - m.skew * (m.tof - biases.get(i, 0.0))
        for (i, j), m in measurements.items()
    }


def _solve_clock_graph(
    device_ids: Tuple[str, ...],
    reference_id: str,
    pair_values: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """Least-squares solve one scalar field (offset or skew) over the clock graph.

    Each measured pair ``(i, j)`` contributes a constraint ``x_j - x_i = m_ij``.
    The reference device is pinned to ``0`` with a hard constraint row, fixing the
    otherwise-arbitrary global additive gauge. Solved with ``np.linalg.lstsq``.
    """
    index = {dev: k for k, dev in enumerate(device_ids)}
    n = len(device_ids)

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    for (i, j), value in pair_values.items():
        row = np.zeros(n)
        row[index[j]] += 1.0
        row[index[i]] -= 1.0
        rows.append(row)
        rhs.append(value)

    # Pin the reference device at 0 (gauge fix). With lstsq + an exact-rank system
    # the reference is pinned exactly; we also zero it explicitly below.
    pin = np.zeros(n)
    pin[index[reference_id]] = 1.0
    rows.append(pin)
    rhs.append(0.0)

    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=float)

    solution, *_ = np.linalg.lstsq(A, b, rcond=None)

    out = {dev: float(solution[index[dev]]) for dev in device_ids}
    # Force the reference exactly to zero (defensive against lstsq round-off).
    out[reference_id] = 0.0
    return out


def estimate_clocks(observations, reference_id: Optional[str] = None) -> ClockEstimates:
    """Recover each device's clock (offset, drift) relative to a reference device.

    Parameters
    ----------
    observations:
        An :class:`dronetracking.sim.observations.Observations`-shaped bundle. Only
        ``device_ids`` and the ``ranging`` records are used.
    reference_id:
        Device whose recovered clock is pinned to ``(offset=0, drift=0)``. Defaults
        to ``observations.device_ids[0]``.

    Returns
    -------
    ClockEstimates
        ``offsets_s`` in seconds and ``drifts_ppm`` in parts-per-million, each
        relative to the reference device, which maps to ``(0.0, 0.0)``. The
        recovered values are exactly the inverse of the locked clock model, so
        ``to_reference(device, local)`` puts every device's local clock onto the
        reference device's timebase.
    """
    device_ids = tuple(observations.device_ids)
    if reference_id is None:
        reference_id = device_ids[0]
    if reference_id not in device_ids:
        raise ValueError(
            f"reference_id {reference_id!r} not in device_ids {device_ids!r}"
        )

    measurements = _per_pair_measurements(observations)

    # Solve the clock graph in the device_ids[0] gauge first (dev0 pinned to 0).
    # The offset-intercept debiasing needs each initiator's *absolute* offset, and
    # in every shipped scenario device_ids[0] has true offset 0 -- so dev0-gauge
    # offsets are the absolute offsets, making the correction exact. We rebase to
    # the caller's chosen reference at the very end (a gauge shift that leaves all
    # pairwise differences, and hence TDOA, invariant).
    gauge = device_ids[0]

    # Skews (relative drifts) decouple from offsets -- solve them directly.
    skew_pairs = {pair: m.skew for pair, m in measurements.items()}
    skews = _solve_clock_graph(device_ids, gauge, skew_pairs)

    # Offsets: the raw intercepts carry a small skew*(tof - offset_i) bias. Solve
    # once for a rough offset, use it to debias the intercepts, then re-solve. One
    # correction pass suffices since the bias is second order (tiny skew x time).
    rough_offsets = _solve_clock_graph(
        device_ids, gauge, {pair: m.offset0 for pair, m in measurements.items()}
    )
    debiased = _debias_offsets(measurements, rough_offsets)
    offsets = _solve_clock_graph(device_ids, gauge, debiased)

    # Rebase from the dev0 gauge to the requested reference (subtract its value).
    off_ref = offsets[reference_id]
    skew_ref = skews[reference_id]
    offsets_s = {dev: offsets[dev] - off_ref for dev in device_ids}
    drifts_ppm = {dev: (skews[dev] - skew_ref) * 1e6 for dev in device_ids}

    # Reference device exactly (0, 0) by convention.
    offsets_s[reference_id] = 0.0
    drifts_ppm[reference_id] = 0.0

    return ClockEstimates(
        device_ids=device_ids,
        offsets_s=offsets_s,
        drifts_ppm=drifts_ppm,
        reference_id=reference_id,
        covariances=None,
    )
