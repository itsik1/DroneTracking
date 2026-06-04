"""Per-device clock model (ground truth).

Locked convention, shared with :meth:`estimation.interfaces.ClockEstimates.to_reference`:

    local = t_global * (1 + drift_ppm * 1e-6) + offset_s

so a device with positive offset reads "ahead" and a device with positive drift runs
"fast". The estimator's job is to recover (offset, drift) *relative to a reference
device* well enough to put every arrival on a common timebase.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def device_local_time(offset_s: float, drift_ppm: float, t_global: ArrayLike) -> ArrayLike:
    """Global time -> this device's local clock reading."""
    return np.asarray(t_global, dtype=float) * (1.0 + drift_ppm * 1e-6) + offset_s


def global_from_local(offset_s: float, drift_ppm: float, t_local: ArrayLike) -> ArrayLike:
    """Inverse of :func:`device_local_time`: local clock reading -> global time."""
    return (np.asarray(t_local, dtype=float) - offset_s) / (1.0 + drift_ppm * 1e-6)
