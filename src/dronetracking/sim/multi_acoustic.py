"""Multi-target acoustic ground truth: several drones -> tagged per-device arrivals.

The single-target :func:`sim.acoustic.generate_acoustic_arrivals` emits one
:class:`AcousticArrival` per (device, emission) for the sole drone ``scenario.trajectory``.
This module is the multi-target generalization: it loops over EVERY target in
``scenario.all_drones`` (the primary plus any ``extra_drones``) and produces one arrival
per (device, emission, drone), tagging each with ``source=k`` (the drone index).

The physics is identical to :mod:`sim.acoustic` — each drone emits once per ``dt_s``; for
every emission the drone's true position gives a global arrival time at each device
(emission time + range / c), stamped into the device's local clock and perturbed by
arrival jitter. The unknown emission time is not exposed; only the per-device local
arrivals are, which is all TDOA needs.

Because the frozen :func:`sim.trajectory.trajectory_position` evaluates
``scenario.trajectory`` only, each drone is evaluated by swapping that one field on a copy
of the (frozen) scenario via :func:`dataclasses.replace` and reusing the same frozen
trajectory math — no trajectory logic is reimplemented here.

This file lives in ``sim`` and MAY use the ``sim`` package freely; the ground-truth
firewall only forbids the *estimation* package from importing ``sim``.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np

from .acoustic import emission_times
from .clocks import device_local_time
from .observations import AcousticArrival
from .scenario import Scenario, TrajectorySpec
from .trajectory import trajectory_position


def _drone_position(scenario: Scenario, drone: TrajectorySpec, t: float) -> np.ndarray:
    """True position (3,) of one target ``drone`` at time ``t``.

    Reuses the frozen :func:`sim.trajectory.trajectory_position`, which reads
    ``scenario.trajectory``; we substitute ``drone`` into a copy of the scenario so the
    same kinematics serve every target.
    """
    view = scenario if drone is scenario.trajectory else dataclasses.replace(scenario, trajectory=drone)
    return trajectory_position(view, t)


def generate_multi_arrivals(scenario: Scenario, rng: np.random.Generator) -> Tuple[AcousticArrival, ...]:
    """Per-device acoustic arrivals for every target in ``scenario.all_drones``.

    For each drone ``k`` and each emission ``j`` (global time ``t_j`` from
    :func:`sim.acoustic.emission_times`), and each device ``d``::

        range          = ||drone_k(t_j) - p_d||
        global arrival = t_j + range / c
        local arrival  = device_local_time(b_d, ppm_d, global arrival) + N(0, toa_std)

    yielding one :class:`AcousticArrival(device_id=d, emission_idx=j, toa_local_s=...,
    source=k)`. Returns ``len(all_drones) * n_devices * n_emissions`` arrivals.
    """
    c = scenario.speed_of_sound_mps
    toa_std = scenario.noise.toa_std_s
    positions = {d.id: np.asarray(d.position_m, dtype=float) for d in scenario.devices}
    times = emission_times(scenario)

    arrivals = []
    for k, drone in enumerate(scenario.all_drones):
        for j, t_j in enumerate(times):
            drone_xyz = _drone_position(scenario, drone, float(t_j))
            for d in scenario.devices:
                rng_m = float(np.linalg.norm(drone_xyz - positions[d.id]))
                g_arrival = float(t_j) + rng_m / c
                toa_local = device_local_time(
                    d.clock_offset_s, d.clock_drift_ppm, g_arrival
                ) + rng.normal(0.0, toa_std)
                arrivals.append(
                    AcousticArrival(
                        device_id=d.id,
                        emission_idx=j,
                        toa_local_s=float(toa_local),
                        source=k,
                    )
                )
    return tuple(arrivals)


def true_tracks(scenario: Scenario) -> Dict[int, np.ndarray]:
    """Ground-truth trajectory of every target, sampled at the emission times.

    Returns ``{drone_index: positions}`` where each value is an ``(N, 3)`` array of the
    drone's true position at the ``N`` emission times (same sampling as
    :func:`generate_multi_arrivals`). This is truth — for tests/eval only; it never
    crosses into the estimation pipeline.
    """
    times = emission_times(scenario)
    tracks: Dict[int, np.ndarray] = {}
    for k, drone in enumerate(scenario.all_drones):
        tracks[k] = np.asarray(
            [_drone_position(scenario, drone, float(t)) for t in times], dtype=float
        )
    return tracks
