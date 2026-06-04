"""The estimation output contract.

These are the types the pipeline assembles and that ``eval``/``viz`` consume.
Defining them up front lets the rest of the system be built and tested before the
algorithms exist. Every estimate carries (or can carry) a covariance.

Clock convention (locked, shared with the simulator):
    A device's local clock reads ``local = t*(1 + ppm*1e-6) + offset`` for global
    time ``t``. :meth:`ClockEstimates.to_reference` is the exact inverse, mapping a
    device's local timestamp onto the common reference timebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass
class RelativeLayout:
    """Recovered device positions in an arbitrary (gauge-free) local frame."""

    device_ids: Tuple[str, ...]
    positions_local: np.ndarray  # (N, 3)
    covariances: Optional[np.ndarray] = None  # (N, 3, 3) or None

    @property
    def n_devices(self) -> int:
        return len(self.device_ids)

    def position_of(self, device_id: str) -> np.ndarray:
        return self.positions_local[self.device_ids.index(device_id)]


@dataclass
class ClockEstimates:
    """Per-device clock offset/drift relative to a reference device."""

    device_ids: Tuple[str, ...]
    offsets_s: Dict[str, float]
    drifts_ppm: Dict[str, float]
    reference_id: str
    covariances: Optional[Dict[str, np.ndarray]] = None  # device_id -> 2x2

    def to_reference(self, device_id: str, local_time: ArrayLike) -> ArrayLike:
        """Map a device's local timestamp onto the common reference timebase."""
        b = self.offsets_s[device_id]
        s = self.drifts_ppm[device_id] * 1e-6
        return (np.asarray(local_time, dtype=float) - b) / (1.0 + s)


@dataclass
class Track:
    """A tracked target's smoothed/filtered trajectory with per-step covariance."""

    times_s: np.ndarray  # (T,)
    positions_local: np.ndarray  # (T, 3)
    covariances: np.ndarray  # (T, 3, 3)
    velocities: Optional[np.ndarray] = None  # (T, 3)
    target_id: Optional[str] = None  # set by the multi-target tracker (Ph6)

    @property
    def final_position(self) -> np.ndarray:
        return self.positions_local[-1]


@dataclass
class GeoTrack:
    """A track georeferenced to real-world coordinates."""

    times_s: np.ndarray  # (T,)
    latlon: np.ndarray  # (T, 2)
    altitude_m: np.ndarray  # (T,)
    cov_enu: np.ndarray  # (T, 3, 3)


@dataclass
class Estimates:
    """The complete output of the estimation pipeline."""

    layout: RelativeLayout
    clocks: ClockEstimates
    track: Track
    geo_track: GeoTrack
