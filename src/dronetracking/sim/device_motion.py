"""Generate two-way ranging for *moving* devices (Ph3 — continuous geometry).

Iteration-1 ranging (:mod:`dronetracking.sim.ranging`) freezes each device at its
``t=0`` position. Here every device may drift at a constant ``velocity_mps``, so
the geometry the array measures is time-varying. We reuse the exact same two-way
ranging physics, but recompute the inter-device time-of-flight at **each
exchange's transmit time** ``t_tx`` via :meth:`DeviceSpec.position_at`, so a
window of records taken near time ``t`` reflects the constellation as it was at
``t``.

This module lives in the ``sim`` package and so MAY import ``sim`` freely
(ground truth). The downstream estimator
(:mod:`dronetracking.estimation.geometry_tracking`) consumes only the resulting
:class:`RangingRecord` timestamps.

Modelling choices, kept faithful to :func:`sim.ranging.generate_ranging_records`:

- For each unordered device pair the lower-index device is the initiator.
- Exchanges are spread uniformly across ``[0, duration_s]`` (``ranging_rounds``
  of them) so the drift is observable from how the recovered ranges evolve.
- The transmit time ``t_tx`` is the *global* time of the first ping. The
  initiator and responder positions are both evaluated at ``t_tx``; over one
  exchange (a few ms) the additional drift is negligible, matching the
  quasi-static assumption of the windowed tracker.
- All four event times are stamped into the respective device's local clock and
  perturbed by the scenario's timestamp / processing-delay jitter.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .clocks import device_local_time
from .observations import RangingRecord
from .scenario import Scenario


def device_positions_at(scenario: Scenario, t: float) -> Dict[str, np.ndarray]:
    """True device positions at global time ``t``.

    Returns ``{device_id: (3,) ndarray}`` using each device's constant-velocity
    drift (static devices are unchanged). This is the ground-truth handle tests
    align a recovered window-layout against.
    """
    return {
        d.id: np.asarray(d.position_at(t), dtype=float) for d in scenario.devices
    }


def generate_moving_ranging(
    scenario: Scenario, rng: np.random.Generator
) -> Tuple[RangingRecord, ...]:
    """Two-way ranging records for a (possibly) moving array.

    Like :func:`sim.ranging.generate_ranging_records`, but the pair distance is
    re-evaluated at every exchange's transmit time so device motion shows up in
    the timestamps. Produces one :class:`RangingRecord` per (unordered pair,
    round); ``round_idx`` indexes the transmit-time schedule.
    """
    devices = scenario.devices
    c = scenario.speed_of_sound_mps
    ts_std = scenario.noise.ranging_timestamp_std_s
    pd_jitter = scenario.noise.proc_delay_jitter_s
    tx_times = np.linspace(0.0, scenario.duration_s, scenario.ranging_rounds)

    records = []
    for a in range(len(devices)):
        for b in range(a + 1, len(devices)):
            i, j = devices[a], devices[b]
            for r, t_tx in enumerate(tx_times):
                t_tx = float(t_tx)
                # Re-evaluate both endpoints at THIS exchange's transmit time.
                pi = np.asarray(i.position_at(t_tx), dtype=float)
                pj = np.asarray(j.position_at(t_tx), dtype=float)
                tof = float(np.linalg.norm(pi - pj)) / c

                d_turnaround = j.proc_delay_s + rng.normal(0.0, pd_jitter)
                # Global event times: tx -> receive -> reply -> receive.
                g1, g2 = t_tx, t_tx + tof
                g3, g4 = g2 + d_turnaround, g2 + d_turnaround + tof
                records.append(
                    RangingRecord(
                        initiator=i.id,
                        responder=j.id,
                        round_idx=r,
                        t1_local_i=float(
                            device_local_time(i.clock_offset_s, i.clock_drift_ppm, g1)
                            + rng.normal(0.0, ts_std)
                        ),
                        t2_local_j=float(
                            device_local_time(j.clock_offset_s, j.clock_drift_ppm, g2)
                            + rng.normal(0.0, ts_std)
                        ),
                        t3_local_j=float(
                            device_local_time(j.clock_offset_s, j.clock_drift_ppm, g3)
                            + rng.normal(0.0, ts_std)
                        ),
                        t4_local_i=float(
                            device_local_time(i.clock_offset_s, i.clock_drift_ppm, g4)
                            + rng.normal(0.0, ts_std)
                        ),
                    )
                )
    return tuple(records)
