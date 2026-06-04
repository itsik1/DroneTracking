"""Generate synthetic drone-signal arrival times (ground truth -> raw ToA).

The drone "emits" once per ``dt_s``. For each emission the true drone position gives a
global arrival time at each device (emission time + range/c); that is stamped into the
device's local clock and perturbed by arrival jitter. The emission time itself is NOT
emitted — only the per-device local arrivals, which is all TDOA needs (the unknown
emission time cancels in arrival differences).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .clocks import device_local_time
from .observations import AcousticArrival
from .scenario import Scenario
from .trajectory import trajectory_position


def emission_times(scenario: Scenario) -> np.ndarray:
    """Global times at which the drone emits its acoustic signature."""
    return np.arange(0.0, scenario.duration_s, scenario.dt_s)


def generate_acoustic_arrivals(scenario: Scenario, rng: np.random.Generator) -> Tuple[AcousticArrival, ...]:
    c = scenario.speed_of_sound_mps
    toa_std = scenario.noise.toa_std_s
    positions = {d.id: np.asarray(d.position_m, dtype=float) for d in scenario.devices}

    arrivals = []
    for k, t_k in enumerate(emission_times(scenario)):
        drone = trajectory_position(scenario, float(t_k))
        for d in scenario.devices:
            rng_m = float(np.linalg.norm(drone - positions[d.id]))
            g_arrival = float(t_k) + rng_m / c
            toa_local = device_local_time(d.clock_offset_s, d.clock_drift_ppm, g_arrival) + rng.normal(0.0, toa_std)
            arrivals.append(AcousticArrival(device_id=d.id, emission_idx=k, toa_local_s=float(toa_local)))
    return tuple(arrivals)
