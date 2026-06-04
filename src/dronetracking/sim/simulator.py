"""Top-level simulator: a :class:`Scenario` in, ``(Observations, World)`` out.

This is the only place that holds both the truth and the measurements at once. It wires
the leaf generators (ranging, acoustic) together, synthesises the noisy GPS anchors, and
packages the ground truth into a :class:`World`. Randomness flows from a single seeded
root generator, split into three independent children (ranging, acoustic, GPS) via
:meth:`numpy.random.Generator.spawn` so that perturbing one noise source leaves the
others bit-for-bit unchanged.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .. import geo
from .acoustic import emission_times, generate_acoustic_arrivals
from .device_motion import generate_moving_ranging
from .multi_acoustic import generate_multi_arrivals, true_tracks
from .observations import AnchorGps, Observations
from .ranging import generate_ranging_records
from .scenario import Scenario
from .trajectory import trajectory_position
from .world import World


def simulate(scenario: Scenario) -> Tuple[Observations, World]:
    """Run the synthetic world and return the measurements plus the ground truth.

    Honors iteration-2 scenario features: moving devices (``velocity_mps``) switch to
    time-varying ranging; extra drones (``extra_drones``) switch to multi-target
    acoustic generation with per-source tags.
    """
    # One seeded root, split into three independent streams so changing one noise
    # source (e.g. GPS) does not shift the random draws of the others.
    ranging_rng, acoustic_rng, gps_rng = np.random.default_rng(scenario.seed).spawn(3)

    ranging = (
        generate_moving_ranging(scenario, ranging_rng)
        if scenario.devices_move
        else generate_ranging_records(scenario, ranging_rng)
    )
    acoustic = (
        generate_multi_arrivals(scenario, acoustic_rng)
        if scenario.extra_drones
        else generate_acoustic_arrivals(scenario, acoustic_rng)
    )

    origin = scenario.origin_latlon
    gps_std = scenario.noise.gps_pos_std_m

    anchor_latlon = {}
    anchor_gps = []
    for dev in scenario.anchors:
        x, y, z = (float(v) for v in dev.position_m)

        # Noise-free truth: true (x, y) -> lat/lon.
        true_lat, true_lon = geo.enu_to_latlon(x, y, origin)
        anchor_latlon[dev.id] = (float(true_lat), float(true_lon))

        # Emitted (noisy) GPS: perturb (x, y, z) by independent N(0, gps_std) metres,
        # convert the perturbed (x, y) to lat/lon, altitude = perturbed z.
        px = x + gps_rng.normal(0.0, gps_std)
        py = y + gps_rng.normal(0.0, gps_std)
        pz = z + gps_rng.normal(0.0, gps_std)
        noisy_lat, noisy_lon = geo.enu_to_latlon(px, py, origin)
        anchor_gps.append(
            AnchorGps(
                device_id=dev.id,
                lat=float(noisy_lat),
                lon=float(noisy_lon),
                altitude_m=float(pz),
            )
        )

    observations = Observations(
        device_ids=scenario.device_ids,
        ranging=ranging,
        acoustic=acoustic,
        anchor_gps=tuple(anchor_gps),
        speed_of_sound_mps=scenario.speed_of_sound_mps,
        sample_rate_hz=scenario.sample_rate_hz,
    )

    times = emission_times(scenario)
    true_track = np.array(
        [trajectory_position(scenario, float(t)) for t in times], dtype=float
    )
    # Per-source truth: multi-target if extra drones, else just the primary as source 0.
    tracks_truth = true_tracks(scenario) if scenario.extra_drones else {0: true_track}

    world = World(
        device_ids=scenario.device_ids,
        device_positions={
            d.id: np.asarray(d.position_m, dtype=float) for d in scenario.devices
        },
        clock_offsets={d.id: d.clock_offset_s for d in scenario.devices},
        clock_drifts_ppm={d.id: d.clock_drift_ppm for d in scenario.devices},
        anchor_latlon=anchor_latlon,
        origin_latlon=origin,
        true_track=true_track,
        true_track_times=times,
        true_tracks=tracks_truth,
        device_velocities={
            d.id: np.asarray(d.velocity_mps, dtype=float) for d in scenario.devices
        },
    )

    return observations, world
