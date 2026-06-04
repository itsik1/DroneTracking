"""Ground-truth bundle for one simulated run.

The ``World`` is the counterpart to :class:`~dronetracking.sim.observations.Observations`:
where ``Observations`` carries only what a real device could measure, ``World`` carries
the truth used to *score* an estimate (true device positions, true clock offsets/drifts,
true anchor lat/lon, the true drone track). Nothing in ``dronetracking.estimation`` may
import this (the ground-truth firewall).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

LatLon = Tuple[float, float]


@dataclass(frozen=True)
class World:
    """Ground truth for a single :func:`~dronetracking.sim.simulator.simulate` run."""

    device_ids: Tuple[str, ...]
    device_positions: Dict[str, np.ndarray]  # id -> true (x, y, z) ENU meters at t=0, shape (3,)
    clock_offsets: Dict[str, float]  # id -> true offset_s
    clock_drifts_ppm: Dict[str, float]  # id -> true drift in ppm
    anchor_latlon: Dict[str, LatLon]  # id -> noise-free true (lat, lon) for GPS anchors
    origin_latlon: LatLon  # ENU tangent-plane origin
    true_track: np.ndarray  # (N, 3) true positions of the PRIMARY drone at each emission time
    true_track_times: np.ndarray  # (N,) global emission times
    # --- iteration-2 truth (defaulted for backward compatibility) ---
    true_tracks: Dict[int, np.ndarray] = field(default_factory=dict)  # source -> (N,3) (multi-target)
    device_velocities: Dict[str, np.ndarray] = field(default_factory=dict)  # id -> (3,) m/s (moving devices)

    def positions_matrix(self) -> np.ndarray:
        """True device positions (at t=0) stacked in ``device_ids`` order, shape ``(K, 3)``."""
        return np.array([self.device_positions[d] for d in self.device_ids], dtype=float)

    def positions_matrix_at(self, t: float) -> np.ndarray:
        """True device positions at time ``t`` (accounts for constant-velocity drift)."""
        out = []
        for d in self.device_ids:
            p = np.asarray(self.device_positions[d], dtype=float)
            v = np.asarray(self.device_velocities.get(d, (0.0, 0.0, 0.0)), dtype=float)
            out.append(p + v * t)
        return np.array(out, dtype=float)
