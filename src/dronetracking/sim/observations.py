"""The observation bundle — the ONLY data that crosses from `sim` into `estimation`.

Everything here is something a real device could actually measure: raw two-way
ranging timestamps (each in the measuring device's own clock) and drone-signal
arrival times (likewise). Deliberately ABSENT: true positions, true clock
offsets/drifts, true emission times, the true track. That omission is the
ground-truth firewall, and tests/test_no_truth_leak.py enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RangingRecord:
    """One two-way ranging exchange. t1/t4 are on the initiator's clock; t2/t3 the responder's."""

    initiator: str
    responder: str
    round_idx: int
    t1_local_i: float  # initiator transmits
    t2_local_j: float  # responder receives
    t3_local_j: float  # responder replies
    t4_local_i: float  # initiator receives


@dataclass(frozen=True)
class AcousticArrival:
    """Arrival time of one drone emission at one device, in that device's local clock."""

    device_id: str
    emission_idx: int  # groups arrivals from the same emission across devices
    toa_local_s: float


@dataclass(frozen=True)
class AnchorGps:
    """A GPS-anchored device's reported position (noisy): lat/lon + altitude."""

    device_id: str
    lat: float
    lon: float
    altitude_m: float


@dataclass(frozen=True)
class Observations:
    """The complete set of measurements handed to the estimation pipeline."""

    device_ids: Tuple[str, ...]
    ranging: Tuple[RangingRecord, ...]
    acoustic: Tuple[AcousticArrival, ...]
    anchor_gps: Tuple[AnchorGps, ...]
    speed_of_sound_mps: float
    sample_rate_hz: float
