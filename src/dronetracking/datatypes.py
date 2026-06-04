"""Cross-stage data types that flow *between* estimation stages.

These are intermediate plumbing types (not the final estimation API — see
``estimation.interfaces`` for that). They carry numpy arrays only and import
nothing from ``sim`` or the rest of the package, so they are safe to share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# A target fix is judged "vertically weak" when the vertical variance dominates the
# mean horizontal variance by this factor — a near-coplanar ground array barely
# observes altitude, and we surface that rather than hide it in a single radius.
WEAK_VERTICAL_RATIO = 10.0


@dataclass
class DistanceMatrix:
    """Pairwise distance estimates between devices, with weights and validity."""

    device_ids: Tuple[str, ...]
    D: np.ndarray  # (K, K) meters, symmetric, zero diagonal; NaN where unmeasured
    W: np.ndarray  # (K, K) weights = 1/variance; 0 where missing/invalid
    counts: np.ndarray  # (K, K) int measurement counts
    valid: np.ndarray  # (K, K) bool mask after outlier rejection

    @property
    def n_devices(self) -> int:
        return len(self.device_ids)

    @property
    def n_valid_edges(self) -> int:
        return int(np.sum(np.triu(self.valid, 1)))


@dataclass
class TargetFix:
    """A single-emission TDOA position fix with covariance and geometry diagnostics."""

    position: np.ndarray  # (3,) in the relative device frame
    cov: np.ndarray  # (3, 3) position covariance
    gdop: float
    residual_rms: float
    n_devices: int
    t: float  # reference-timebase timestamp of the emission

    @property
    def error_radius(self) -> float:
        """1-sigma spherical-equivalent radius, sqrt(trace(cov))."""
        return float(np.sqrt(np.trace(self.cov)))

    @property
    def vertical_std(self) -> float:
        return float(np.sqrt(self.cov[2, 2]))

    @property
    def horizontal_std(self) -> float:
        return float(np.sqrt(self.cov[0, 0] + self.cov[1, 1]))

    @property
    def weak_vertical(self) -> bool:
        """True when vertical uncertainty dominates the horizontal — report, don't bury."""
        mean_horizontal = 0.5 * (self.cov[0, 0] + self.cov[1, 1])
        if mean_horizontal <= 0:
            return True
        return self.cov[2, 2] > WEAK_VERTICAL_RATIO * mean_horizontal
