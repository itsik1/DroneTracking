"""Ph3 — windowed geometry tracking for a *moving* device array.

When devices drift, the constellation is no longer a single static layout but a
time series of them. We slide a time window over the two-way ranging records,
solve one relative layout per window (reusing the iteration-1 geometry stages),
stitch the per-window layouts into a single consistent frame, and optionally
smooth each device's track over time.

Pipeline per call:

1. **Bucket** records into overlapping windows of width ``window_s`` stepped by
   ``step_s``, keyed by the initiator transmit timestamp ``t1_local_i`` (the
   moment the exchange started). Each window's centre time labels its layout.
2. **Per-window geometry.** Wrap the window's records in a tiny observations-shaped
   adapter and reuse :func:`estimation.ranging.build_distance_matrix` then
   :func:`estimation.relative_localization.estimate_layout`.
3. **Align to a common frame.** Distance geometry is gauge-free (rotation +
   translation + reflection), so consecutive windows can come out arbitrarily
   posed. We register each window onto the running common frame with
   :func:`transforms.umeyama` (no rescale; reflection allowed) over the devices
   the two windows share, chaining so the whole series lives in one frame.
4. **Smooth (optional).** A per-device constant-velocity Kalman filter over the
   aligned position series suppresses per-window jitter while tracking the drift.

:func:`estimate_velocities` finite-differences the aligned series (robust
per-device least-squares slope) to recover each device's velocity *in the common
frame*.

This module is part of the estimation package and imports **nothing** from
:mod:`dronetracking.sim` (the ground-truth firewall). It consumes ranging records
structurally (``.initiator`` / ``.responder`` / the four ``t*_local_*`` fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..estimation.interfaces import RelativeLayout
from ..estimation.ranging import build_distance_matrix
from ..estimation.relative_localization import estimate_layout
from ..transforms import umeyama


@dataclass
class _WindowObs:
    """Minimal observations-shaped view of one window's records.

    :func:`estimation.ranging.build_distance_matrix` consumes only these three
    attributes, so this is all we need to reuse the iteration-1 geometry stage on
    a windowed slice — no dependency on the full ``Observations`` bundle (or on
    ``sim``).
    """

    device_ids: Tuple[str, ...]
    ranging: Tuple[object, ...]
    speed_of_sound_mps: float


def _window_edges(
    t_min: float, t_max: float, window_s: float, step_s: float
) -> List[Tuple[float, float]]:
    """Sliding ``[start, start+window_s)`` windows covering ``[t_min, t_max]``.

    Windows step by ``step_s`` and are anchored so the last window's centre does
    not run past ``t_max`` (the final window is clamped to end at ``t_max``). A
    degenerate span yields a single window.
    """
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    if step_s <= 0.0:
        raise ValueError("step_s must be positive")

    span = t_max - t_min
    if span <= 0.0:
        return [(t_min, t_min + window_s)]

    edges: List[Tuple[float, float]] = []
    start = t_min
    # Stop once a window's *start* passes the last point; the final partial window
    # is appended explicitly so the tail of the run is always covered.
    while start <= t_max - window_s + 1e-9:
        edges.append((start, start + window_s))
        start += step_s
    if not edges or edges[-1][1] < t_max - 1e-9:
        last_start = max(t_min, t_max - window_s)
        if not edges or abs(last_start - edges[-1][0]) > 1e-9:
            edges.append((last_start, last_start + window_s))
    return edges


def _bucket_by_window(
    records: Sequence[object], edges: Sequence[Tuple[float, float]]
) -> List[Tuple[float, List[object]]]:
    """Group records into each window by their initiator transmit timestamp.

    Returns ``[(t_center, [records...]), ...]`` for windows that actually contain
    records. A record may fall in several overlapping windows.
    """
    t1 = np.array([r.t1_local_i for r in records], dtype=float)
    out: List[Tuple[float, List[object]]] = []
    for lo, hi in edges:
        mask = (t1 >= lo) & (t1 <= hi)
        if not np.any(mask):
            continue
        idx = np.nonzero(mask)[0]
        bucket = [records[k] for k in idx]
        out.append((0.5 * (lo + hi), bucket))
    return out


def _align_to_reference(
    ref: RelativeLayout, cur: RelativeLayout
) -> np.ndarray:
    """Rigidly register ``cur`` onto ``ref``'s frame over their shared devices.

    Returns ``cur``'s positions transformed into ``ref``'s frame (same device
    order as ``cur``). Uses :func:`transforms.umeyama` with no rescale and
    reflection allowed (distance geometry cannot see chirality, so successive
    windows may differ by a reflection that we must absorb here).
    """
    ref_index = {d: i for i, d in enumerate(ref.device_ids)}
    shared = [d for d in cur.device_ids if d in ref_index]
    if len(shared) < 3:
        # Not enough correspondences to fix a 3-D pose; leave ``cur`` as-is.
        return np.asarray(cur.positions_local, dtype=float)

    cur_index = {d: i for i, d in enumerate(cur.device_ids)}
    src = np.array([cur.positions_local[cur_index[d]] for d in shared], dtype=float)
    dst = np.array([ref.positions_local[ref_index[d]] for d in shared], dtype=float)
    sim = umeyama(src, dst, with_scaling=False, allow_reflection=True)
    return sim.apply(np.asarray(cur.positions_local, dtype=float))


def _refine_to_stable_frame(positions: np.ndarray) -> np.ndarray:
    """Re-pose the whole series in a frame anchored to the *stable* devices.

    Chaining each window onto the previous one (``_align_to_reference``) keeps the
    series internally consistent, but a rigid registration of a *deforming* cloud
    is biased: the moving devices drag the best-fit transform, smearing their
    true motion across the static ones (everyone ends up looking like they move a
    little, the real movers look slower). To remove that bias we:

    1. From the chained series, score each device by how far it wanders over time
       (path spread about its own mean). Movers score high; static/slow devices
       score low.
    2. Take the low-score majority as gauge anchors (median/MAD cut), and require
       at least four so the 3-D pose — including the vertical — is well fixed;
       fall back to all devices if too few qualify.
    3. Re-align every window onto the *first* window using **only** those anchors
       (rigid, reflection allowed). The static anchors now define one fixed frame;
       each mover's recovered motion is its true motion in that frame.

    ``positions`` is the chained ``(T, N, 3)`` series; returns the re-anchored
    ``(T, N, 3)``. With a single window or fewer than three anchors this is a
    no-op. Everything here is gauge-free (no ground truth) — it only exploits the
    fact that most of the array is stationary.
    """
    T, N, _ = positions.shape
    if T < 2 or N < 3:
        return positions.copy()

    # Per-device wander: total spread of its track about its own centroid.
    spread = np.array(
        [np.linalg.norm(positions[:, n, :] - positions[:, n, :].mean(axis=0), axis=1).sum()
         for n in range(N)],
        dtype=float,
    )
    med = float(np.median(spread))
    mad = float(np.median(np.abs(spread - med))) + 1e-9
    cut = med + 3.0 * 1.4826 * mad
    anchors = np.nonzero(spread <= cut)[0]

    # Need >=4 well-fixing anchors; if the cut is too aggressive (or the array is
    # mostly moving), widen to the four most-stable devices, else use all.
    if anchors.size < 4:
        if N >= 4:
            anchors = np.argsort(spread)[:4]
        else:
            anchors = np.arange(N)
    if anchors.size < 3:
        return positions.copy()

    ref = positions[0]
    out = np.empty_like(positions)
    for k in range(T):
        sim = umeyama(
            positions[k][anchors], ref[anchors],
            with_scaling=False, allow_reflection=True,
        )
        out[k] = sim.apply(positions[k])
    return out


def _smooth_positions(
    times: np.ndarray, positions: np.ndarray, *, meas_var: float = 0.04
) -> np.ndarray:
    """Per-device constant-velocity Kalman smoother over the aligned series.

    ``positions`` is ``(T, N, 3)``. Each device coordinate is filtered
    independently with a constant-velocity model (state ``[pos, vel]``), forward
    then RTS-smoothed back, which removes per-window jitter while following the
    real drift. ``meas_var`` (m^2) is the per-window position measurement
    variance; the process noise is set modestly so genuine motion is tracked.

    Returns smoothed ``(T, N, 3)``. With a single window it is a no-op.
    """
    T = times.shape[0]
    if T < 2:
        return positions.copy()

    N = positions.shape[1]
    out = np.empty_like(positions)

    q = 1.0  # spectral density of the constant-velocity process noise (m^2/s^3)
    R = float(meas_var)

    for n in range(N):
        for axis in range(3):
            z = positions[:, n, axis]
            # Forward pass storage.
            xf = np.zeros((T, 2))
            Pf = np.zeros((T, 2, 2))
            xp = np.zeros((T, 2))
            Pp = np.zeros((T, 2, 2))

            x = np.array([z[0], 0.0])
            P = np.array([[R, 0.0], [0.0, 1.0]])
            H = np.array([[1.0, 0.0]])
            for k in range(T):
                if k == 0:
                    xp[k] = x
                    Pp[k] = P
                else:
                    dt = float(times[k] - times[k - 1])
                    F = np.array([[1.0, dt], [0.0, 1.0]])
                    Q = q * np.array(
                        [
                            [dt**3 / 3.0, dt**2 / 2.0],
                            [dt**2 / 2.0, dt],
                        ]
                    )
                    x = F @ xf[k - 1]
                    P = F @ Pf[k - 1] @ F.T + Q
                    xp[k] = x
                    Pp[k] = P
                # Update.
                S = H @ P @ H.T + R
                K = (P @ H.T) / S
                x = x + (K * (z[k] - H @ x)).ravel()
                P = (np.eye(2) - K @ H) @ P
                xf[k] = x
                Pf[k] = P

            # RTS backward smoother.
            xs = xf.copy()
            Ps = Pf.copy()
            for k in range(T - 2, -1, -1):
                dt = float(times[k + 1] - times[k])
                F = np.array([[1.0, dt], [0.0, 1.0]])
                C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
                xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
                Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

            out[:, n, axis] = xs[:, 0]

    return out


def track_geometry(
    ranging_records: Sequence[object],
    device_ids: Sequence[str],
    speed_of_sound: float,
    window_s: float,
    step_s: float,
    *,
    smooth: bool = True,
) -> List[Tuple[float, RelativeLayout]]:
    """Track the array geometry over time from windowed ranging.

    Slides a ``window_s`` window (stepped by ``step_s``) over ``ranging_records``,
    bucketing by initiator transmit timestamp; solves one relative layout per
    window (reusing :func:`estimation.ranging.build_distance_matrix` +
    :func:`estimation.relative_localization.estimate_layout`); registers
    consecutive layouts into one common frame (Umeyama); and, when ``smooth``,
    applies a per-device constant-velocity Kalman smoother over the aligned
    positions.

    Returns ``[(t_center, RelativeLayout), ...]`` ordered by time. The layouts
    share one (gauge-free) frame fixed by the first window; ``device_ids`` order
    is preserved in every layout.
    """
    device_ids = tuple(device_ids)
    records = list(ranging_records)
    if not records:
        return []

    t1 = np.array([r.t1_local_i for r in records], dtype=float)
    edges = _window_edges(float(t1.min()), float(t1.max()), window_s, step_s)
    buckets = _bucket_by_window(records, edges)

    # --- per-window geometry ---------------------------------------------- #
    raw: List[Tuple[float, RelativeLayout]] = []
    for t_center, bucket in buckets:
        obs = _WindowObs(
            device_ids=device_ids,
            ranging=tuple(bucket),
            speed_of_sound_mps=float(speed_of_sound),
        )
        dm = build_distance_matrix(obs)
        # Need enough valid edges for a 3-D layout; skip starved windows.
        if dm.n_valid_edges < 3:
            continue
        layout = estimate_layout(dm)
        raw.append((t_center, layout))

    if not raw:
        return []

    # --- chain into a common frame ---------------------------------------- #
    aligned_positions: List[np.ndarray] = [np.asarray(raw[0][1].positions_local, dtype=float)]
    ref = raw[0][1]
    for _, layout in raw[1:]:
        moved = _align_to_reference(ref, layout)
        aligned_positions.append(moved)
        # Re-anchor the reference to the just-aligned layout so the frame is
        # carried forward window-to-window (devices may enter/leave a window).
        ref = RelativeLayout(device_ids=layout.device_ids, positions_local=moved)

    times = np.array([t for t, _ in raw], dtype=float)
    pos_series = np.stack(aligned_positions, axis=0)  # (T, N, 3)

    # --- re-anchor the frame on the stable devices ------------------------ #
    # Chained registration of a deforming cloud biases velocities; pin the gauge
    # to the static majority so each mover's recovered motion is unbiased.
    if pos_series.shape[0] >= 2:
        pos_series = _refine_to_stable_frame(pos_series)

    # --- optional temporal smoothing -------------------------------------- #
    if smooth and pos_series.shape[0] >= 2:
        pos_series = _smooth_positions(times, pos_series)

    out: List[Tuple[float, RelativeLayout]] = []
    for k, (t_center, layout) in enumerate(raw):
        out.append(
            (
                t_center,
                RelativeLayout(
                    device_ids=device_ids,
                    positions_local=pos_series[k],
                    covariances=layout.covariances,
                ),
            )
        )
    return out


def estimate_velocities(
    series: Sequence[Tuple[float, RelativeLayout]]
) -> Dict[str, np.ndarray]:
    """Recover each device's velocity by finite-differencing the aligned series.

    ``series`` is the output of :func:`track_geometry` (layouts in one common
    frame). For each device we fit a least-squares straight line to its (t, x),
    (t, y), (t, z) tracks; the slope is the velocity *in the series' common
    frame*. Returns ``{device_id: (3,) ndarray}`` (m/s).

    With fewer than two windows there is no motion to observe and every velocity
    is zero. Velocities live in the same gauge-free frame as ``series``; compare
    against truth only after aligning that frame to the world (see the tests).
    """
    series = list(series)
    if len(series) < 2:
        ids = series[0][1].device_ids if series else ()
        return {d: np.zeros(3) for d in ids}

    times = np.array([t for t, _ in series], dtype=float)
    device_ids = series[0][1].device_ids
    positions = np.stack(
        [np.asarray(layout.positions_local, dtype=float) for _, layout in series],
        axis=0,
    )  # (T, N, 3)

    # Least-squares slope per device per axis: vel = cov(t, x) / var(t).
    t_c = times - times.mean()
    denom = float(np.dot(t_c, t_c))
    out: Dict[str, np.ndarray] = {}
    for n, dev_id in enumerate(device_ids):
        vel = np.zeros(3)
        if denom > 0.0:
            for axis in range(3):
                x = positions[:, n, axis]
                vel[axis] = float(np.dot(t_c, x - x.mean()) / denom)
        out[dev_id] = vel
    return out
