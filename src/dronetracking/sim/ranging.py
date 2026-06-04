"""Generate synthetic two-way ranging exchanges (ground truth -> raw timestamps).

For each unordered device pair the initiator (lower index) pings the responder, which
holds for its processing delay and replies. All four event times are computed in global
time, then stamped into the respective device's local clock and perturbed by timestamp
jitter. Exchanges are spread across the whole duration so that clock skew (drift) is
observable from how the recovered offset changes over time.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .clocks import device_local_time
from .observations import RangingRecord
from .scenario import Scenario


def generate_ranging_records(scenario: Scenario, rng: np.random.Generator) -> Tuple[RangingRecord, ...]:
    devices = scenario.devices
    c = scenario.speed_of_sound_mps
    ts_std = scenario.noise.ranging_timestamp_std_s
    pd_jitter = scenario.noise.proc_delay_jitter_s
    tx_times = np.linspace(0.0, scenario.duration_s, scenario.ranging_rounds)

    positions = {d.id: np.asarray(d.position_m, dtype=float) for d in devices}
    records = []

    for a in range(len(devices)):
        for b in range(a + 1, len(devices)):
            i, j = devices[a], devices[b]
            tof = float(np.linalg.norm(positions[i.id] - positions[j.id])) / c
            for r, t_tx in enumerate(tx_times):
                d_turnaround = j.proc_delay_s + rng.normal(0.0, pd_jitter)
                # Global event times: tx -> receive -> reply -> receive.
                g1, g2 = t_tx, t_tx + tof
                g3, g4 = g2 + d_turnaround, g2 + d_turnaround + tof
                records.append(
                    RangingRecord(
                        initiator=i.id,
                        responder=j.id,
                        round_idx=r,
                        t1_local_i=float(device_local_time(i.clock_offset_s, i.clock_drift_ppm, g1) + rng.normal(0.0, ts_std)),
                        t2_local_j=float(device_local_time(j.clock_offset_s, j.clock_drift_ppm, g2) + rng.normal(0.0, ts_std)),
                        t3_local_j=float(device_local_time(j.clock_offset_s, j.clock_drift_ppm, g3) + rng.normal(0.0, ts_std)),
                        t4_local_i=float(device_local_time(i.clock_offset_s, i.clock_drift_ppm, g4) + rng.normal(0.0, ts_std)),
                    )
                )
    return tuple(records)
