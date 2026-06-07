"""Acoustic ranging backend for the zero-install browser app.

Devices measure inter-device *distance* acoustically with symmetric double-sided
two-way ranging (SDS-TWR): an initiator A emits a chirp at A-local ``t1``; the
responder B hears it at B-local ``t2``, emits a reply at B-local ``t3``; A hears
that reply at A-local ``t4``. Because the two unknown clock offsets cancel in the
difference, the one-way time of flight is ``((t4 - t1) - (t3 - t2)) / 2`` and the
range is that times the speed of sound::

    distance = speed_of_sound * ((t4 - t1) - (t3 - t2)) / 2

All timestamps are each device's own ``AudioContext.currentTime`` (seconds);
no clock synchronization is required.

The :class:`RangingCoordinator` is the server-side brain of the protocol:

* It schedules **rounds** round-robin over the ordered pairs of currently-online
  devices, advancing to the next pair every :data:`ROUND_PERIOD_S` seconds, and
  exposes the current instruction via :meth:`RangingCoordinator.current_command`.
* Devices report their completed half-exchanges via
  :meth:`RangingCoordinator.submit`; the coordinator pairs the two halves of a
  round by round number and turns each completed round into a pairwise distance
  (a **robust median** over repeated measurements of the same pair).
* :meth:`RangingCoordinator.distances` returns the latest per-pair distances and
  :meth:`RangingCoordinator.distance_matrix` packs them into the estimation
  pipeline's :class:`~dronetracking.datatypes.DistanceMatrix` (``NaN`` where a
  pair was never measured, with the ``valid`` mask set on measured edges).

Honest scope: browser audio timing is jittery, so a single round is rough
(~1 m best case). The robust median over repeats and the downstream MDS refine
(:func:`dronetracking.estimation.relative_localization.estimate_layout`) are what
make the recovered layout usable. Real-device validation and turnaround tuning
are still required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datatypes import DistanceMatrix

# Nominal speed of sound in air (~20 C). The frontend and tests share this value.
SPEED_OF_SOUND_MPS = 343.0

# Default chirp the coordinator instructs devices to emit: a short linear sweep in a
# NEAR-ULTRASONIC band (~18-20 kHz) — inaudible to most people but still within phone
# speaker/mic range and cross-correlatable. (May need per-device tuning; some speakers
# roll off above ~20 kHz and Nyquist is ~22 kHz at a 44.1 kHz sample rate.)
DEFAULT_CHIRP = {"f0": 18000.0, "f1": 20000.0, "dur_s": 0.06}

# How long each round (one ordered pair) stays the active instruction before the
# schedule advances to the next pair. A few seconds gives the browser time to
# emit, hear, reply, hear, and POST a half-exchange.
ROUND_PERIOD_S = 3.0

# A measured pairwise distance must be finite and within this many meters to be
# accepted; clearly-bogus values (negative TOF from a missed peak, or absurd
# ranges) are dropped before the median so they cannot poison a pair.
MAX_PLAUSIBLE_DISTANCE_M = 1000.0


def sds_twr_distance(
    t1: float,
    t2: float,
    t3: float,
    t4: float,
    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
) -> float:
    """Symmetric double-sided two-way ranging distance from one round's timestamps.

    ``t1``/``t4`` are the initiator-local emit/receive times; ``t2``/``t3`` are
    the responder-local receive/emit times. The clock offsets cancel, leaving::

        distance = c * ((t4 - t1) - (t3 - t2)) / 2

    Returns the range in meters (may be negative if the inputs are inconsistent —
    callers reject implausible values).
    """
    tof = ((float(t4) - float(t1)) - (float(t3) - float(t2))) / 2.0
    return float(speed_of_sound_mps) * tof


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Unordered key for a device pair (so A->B and B->A accumulate together)."""
    return (a, b) if a <= b else (b, a)


@dataclass
class _Round:
    """Accumulator for the two half-exchanges of a single round.

    The initiator half carries ``t1``/``t4`` and is submitted by the initiator
    device; the responder half carries ``t2``/``t3`` and is submitted by the
    responder device. We keep each half's submitting ``device_id`` so the pair is
    derived from who actually answered, not just from the schedule.
    """

    init_device: Optional[str] = None
    t1: Optional[float] = None
    t4: Optional[float] = None
    resp_device: Optional[str] = None
    t2: Optional[float] = None
    t3: Optional[float] = None
    consumed: bool = False  # distance already folded into the per-pair samples

    @property
    def complete(self) -> bool:
        return (
            self.init_device is not None
            and self.resp_device is not None
            and self.t1 is not None
            and self.t4 is not None
            and self.t2 is not None
            and self.t3 is not None
        )


class RangingCoordinator:
    """Schedules SDS-TWR rounds and turns reported half-exchanges into distances.

    Parameters
    ----------
    speed_of_sound_mps:
        Speed of sound used to convert time-of-flight to range.
    round_period_s:
        Seconds each ordered pair remains the active instruction before the
        round-robin schedule advances.
    chirp:
        The chirp descriptor handed to devices in every command.
    """

    def __init__(
        self,
        speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
        round_period_s: float = ROUND_PERIOD_S,
        chirp: Optional[dict] = None,
    ) -> None:
        self._c = float(speed_of_sound_mps)
        self._round_period_s = float(round_period_s)
        self._chirp = dict(chirp) if chirp is not None else dict(DEFAULT_CHIRP)

        # round number -> accumulating half-exchanges
        self._rounds: Dict[int, _Round] = {}
        # unordered pair key -> list of accepted distance samples (meters)
        self._samples: Dict[Tuple[str, str], List[float]] = {}

        # Schedule bookkeeping: the wall-clock time the *current* round started,
        # the current round index, and the pair it names.
        self._round_index: int = 0
        self._round_started_at: Optional[float] = None
        self._current_pair: Optional[Tuple[str, str]] = None

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ordered_pairs(online_ids: List[str]) -> List[Tuple[str, str]]:
        """All ordered (initiator, responder) pairs over the online devices.

        Order is deterministic (sorted ids, i<j) so the round-robin is stable and
        every unordered pair is visited exactly once per cycle.
        """
        ids = sorted(set(online_ids))
        pairs: List[Tuple[str, str]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
        return pairs

    def current_command(
        self, online_ids: List[str], now: float
    ) -> Optional[dict]:
        """Return the active ranging instruction, advancing the schedule on ``now``.

        Returns ``None`` when fewer than two devices are online (nothing to
        range). Otherwise returns::

            {"round": int, "initiator": str, "responder": str,
             "chirp": {"f0": ..., "f1": ..., "dur_s": ...}}

        The pair cycles round-robin over all online pairs, advancing once
        ``round_period_s`` has elapsed since the current round began. Calling this
        repeatedly within the same window returns the *same* round (idempotent),
        so two devices reading the SSE stream see one consistent instruction.
        """
        pairs = self._ordered_pairs(online_ids)
        if not pairs:
            self._current_pair = None
            self._round_started_at = None
            return None

        now = float(now)
        # (Re)initialize on the first call or whenever no round is active yet.
        if self._round_started_at is None or self._current_pair is None:
            self._round_started_at = now
            initiator, responder = pairs[self._round_index % len(pairs)]
            self._current_pair = (initiator, responder)
        elif (now - self._round_started_at) >= self._round_period_s:
            # Window elapsed: advance to the next ordered pair and bump the round.
            self._round_index += 1
            self._round_started_at = now
            initiator, responder = pairs[self._round_index % len(pairs)]
            self._current_pair = (initiator, responder)
        else:
            # Still inside the current window: keep the active pair. If the online
            # set shifted and the active pair is no longer valid, re-pick now.
            if self._current_pair not in pairs:
                initiator, responder = pairs[self._round_index % len(pairs)]
                self._current_pair = (initiator, responder)

        initiator, responder = self._current_pair
        return {
            "round": int(self._round_index),
            "initiator": initiator,
            "responder": responder,
            "chirp": dict(self._chirp),
        }

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def submit(self, device_id: str, entries: List[dict]) -> None:
        """Record this device's completed half-exchanges.

        Each entry is one of (per the protocol)::

            {"round": int, "role": "init", "t1": float, "t4": float}
            {"round": int, "role": "resp", "t2": float, "t3": float}

        Halves are paired by ``round``; when both arrive, the round's distance is
        computed and appended to that pair's sample list (subject to a plausibility
        gate). Malformed entries are ignored so a noisy client can't crash the
        snapshot.
        """
        if not entries:
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rnd = entry.get("round")
            role = entry.get("role")
            if rnd is None or role not in ("init", "resp"):
                continue
            try:
                rnd = int(rnd)
            except (TypeError, ValueError):
                continue

            rec = self._rounds.get(rnd)
            if rec is None:
                rec = _Round()
                self._rounds[rnd] = rec

            try:
                if role == "init":
                    rec.init_device = str(device_id)
                    rec.t1 = float(entry["t1"])
                    rec.t4 = float(entry["t4"])
                else:  # "resp"
                    rec.resp_device = str(device_id)
                    rec.t2 = float(entry["t2"])
                    rec.t3 = float(entry["t3"])
            except (KeyError, TypeError, ValueError):
                # A half missing/garbling its timestamps: leave the round pending.
                continue

            self._maybe_finalize(rnd)

    def _maybe_finalize(self, rnd: int) -> None:
        """If round ``rnd`` now has both halves, fold its distance into the pair."""
        rec = self._rounds.get(rnd)
        if rec is None or rec.consumed or not rec.complete:
            return
        if rec.init_device == rec.resp_device:
            # A device cannot range against itself; drop the degenerate round.
            rec.consumed = True
            return

        d = sds_twr_distance(rec.t1, rec.t2, rec.t3, rec.t4, self._c)
        rec.consumed = True
        if not np.isfinite(d) or d < 0.0 or d > MAX_PLAUSIBLE_DISTANCE_M:
            return  # implausible (missed peak / wrong reply): don't poison the pair
        key = _pair_key(rec.init_device, rec.resp_device)
        self._samples.setdefault(key, []).append(float(d))

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def distances(self) -> List[dict]:
        """Latest per-pair distances as ``[{"a": str, "b": str, "m": float}]``.

        Each pair's distance is the **robust median** over its accepted samples,
        so repeated rounds average out browser-audio jitter. Sorted by pair for a
        stable, deterministic order.
        """
        out: List[dict] = []
        for (a, b), samples in self._samples.items():
            if not samples:
                continue
            m = float(np.median(np.asarray(samples, dtype=float)))
            out.append({"a": a, "b": b, "m": m})
        out.sort(key=lambda r: (r["a"], r["b"]))
        return out

    def _median_distance(self, a: str, b: str) -> Optional[float]:
        samples = self._samples.get(_pair_key(a, b))
        if not samples:
            return None
        return float(np.median(np.asarray(samples, dtype=float)))

    def distance_matrix(self, ids: List[str]) -> DistanceMatrix:
        """Pack measured distances over ``ids`` into a :class:`DistanceMatrix`.

        Unmeasured pairs are ``NaN`` in ``D`` with weight ``0`` and ``valid``
        ``False``; the diagonal is ``0``/valid. Measured edges get the robust
        median distance, ``valid=True``, ``counts`` = number of accepted samples,
        and a unit weight (the downstream refine treats all measured edges
        equally — browser jitter is not separable per edge here).
        """
        ids = list(ids)
        K = len(ids)
        D = np.full((K, K), np.nan, dtype=float)
        W = np.zeros((K, K), dtype=float)
        counts = np.zeros((K, K), dtype=float)
        valid = np.zeros((K, K), dtype=bool)

        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(valid, True)

        for i in range(K):
            for j in range(i + 1, K):
                key = _pair_key(ids[i], ids[j])
                samples = self._samples.get(key)
                if not samples:
                    continue
                m = float(np.median(np.asarray(samples, dtype=float)))
                n = float(len(samples))
                D[i, j] = D[j, i] = m
                W[i, j] = W[j, i] = 1.0
                counts[i, j] = counts[j, i] = n
                valid[i, j] = valid[j, i] = True

        return DistanceMatrix(
            device_ids=tuple(ids),
            D=D,
            W=W,
            counts=counts,
            valid=valid,
        )

    def measured_pairs(self) -> List[Tuple[str, str]]:
        """Unordered pairs that currently have at least one accepted sample."""
        return [key for key, samples in self._samples.items() if samples]
