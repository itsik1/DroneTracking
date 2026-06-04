"""End-to-end orchestration: synthetic world -> estimation -> evaluation.

The conductor. Routes by scenario feature so the iteration-2 phases are exercised
through one entry point, while the baseline (single drone, static devices, idealized
arrivals, full GPS) path is unchanged:

- moving devices (``velocity_mps``)  -> continuous geometry tracking (Ph3)
- extra drones (``extra_drones``)    -> multi-target tracking (Ph6)
- ``detect=True``                    -> acoustic detection from synthesized audio (Ph4)
- ``gps_blackout`` windows           -> GPS-denied georeferencing with blend (Ph9)

Like ``eval``, this module may import both ``sim`` and ``estimation``; the firewall
only forbids the *estimation* package from importing ``sim``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sim.scenario import Scenario
from .sim.acoustic import emission_times
from .sim.audio import synthesize_captures, reference_pulse
from .sources.simulated import SimulatedDeviceFeed
from .estimation.ranging import build_distance_matrix
from .estimation.relative_localization import estimate_layout
from .estimation.clock_sync import estimate_clocks
from .estimation.tdoa import localize_all
from .estimation.joint_clock import localize_all_joint
from .estimation.tracking import track_target
from .estimation.georeference import solve_transform, georeference_track
from .estimation.detection import detect_arrivals
from .estimation.multi_target import localize_frames, track_targets
from .estimation.geometry_tracking import track_geometry
from .estimation.gps_denied import georeference_with_blackout
from .estimation.interfaces import Estimates, Track, GeoTrack, RelativeLayout
from .sim.observations import AcousticArrival
from .eval.metrics import compute_metrics
from .eval import phase_metrics


@dataclass
class PipelineResult:
    scenario: Scenario
    observations: Any
    world: Any
    estimates: Estimates  # primary track (tracks[0]); device/clock/georef are global
    metrics: Dict[str, Any]
    tracks: List[Track] = field(default_factory=list)  # all tracks (single-target -> 1)
    geo_tracks: List[GeoTrack] = field(default_factory=list)
    geometry_series: Optional[List[Tuple[float, RelativeLayout]]] = None  # moving devices

    @property
    def is_multi_target(self) -> bool:
        return len(self.tracks) > 1 or bool(self.scenario.extra_drones)


def run_pipeline(scenario: Scenario, *, model: str = "cv", sigma_a: float = 2.0,
                 detect: bool = False, joint_clock: bool = False, clock_prior_s: float = 1e-4,
                 feed=None) -> PipelineResult:
    # Read measurements through a DeviceFeed (the hardware-abstraction seam): the default
    # is the simulator; a real LiveDeviceFeed drops in here with no downstream changes.
    if feed is None:
        feed = SimulatedDeviceFeed(scenario)
    observations = feed.as_observations()
    world = getattr(feed, "world", None)

    # 1. Geometry: a live series if devices move, else a single static layout.
    geometry_series = None
    if scenario.devices_move:
        win = max(scenario.duration_s / 6.0, 2.0 * scenario.dt_s)
        geometry_series = track_geometry(
            observations.ranging, observations.device_ids,
            observations.speed_of_sound_mps, window_s=win, step_s=win / 2.0,
        )
        layout = _representative_layout(geometry_series, scenario.duration_s / 2.0)
    else:
        layout = estimate_layout(build_distance_matrix(observations))

    # 2. Clocks (no shared clock assumed).
    clocks = estimate_clocks(observations)

    # 3. Optional acoustic detection from synthesized audio (single-target only).
    if detect and not scenario.extra_drones:
        observations = _detect_arrivals_into(observations, scenario)

    # 4. Targets: multi-target association if extra drones, else single-target track.
    if scenario.extra_drones:
        frames = localize_frames(observations.acoustic, clocks, layout, observations.speed_of_sound_mps)
        tracks = track_targets(frames)
    else:
        if joint_clock:
            fixes = localize_all_joint(observations, clocks, layout,
                                       observations.speed_of_sound_mps, clock_prior_s=clock_prior_s)
        else:
            fixes = localize_all(observations, clocks, layout)
        tracks = [track_target(fixes, model=model, sigma_a=sigma_a)]

    # 5. Georeference each track (GPS-denied blend if a blackout is scheduled).
    transform = solve_transform(layout, observations.anchor_gps, scenario.origin_latlon)
    geo_tracks = []
    for tr in tracks:
        if scenario.gps_blackout:
            gt = georeference_with_blackout(
                layout, observations.anchor_gps, tr, scenario.origin_latlon, scenario.gps_blackout
            )
        else:
            gt = georeference_track(tr, transform, scenario.origin_latlon)
        geo_tracks.append(gt)

    estimates = Estimates(layout=layout, clocks=clocks, track=tracks[0], geo_track=geo_tracks[0])

    # 6. Score against ground truth — only available with a simulated feed. A real
    # LiveDeviceFeed has no truth, so metrics are skipped (estimates still produced).
    metrics: Dict[str, Any] = {}
    if world is not None:
        metrics = compute_metrics(world, observations, estimates)
        if scenario.extra_drones:
            metrics.update(phase_metrics.multi_target_metrics(world, tracks, estimates.layout))
        if geometry_series is not None:
            metrics.update(phase_metrics.geometry_metrics(world, geometry_series))
        if scenario.gps_blackout:
            metrics.update(phase_metrics.gps_denied_metrics(world, geo_tracks[0], scenario.gps_blackout))

    return PipelineResult(
        scenario=scenario, observations=observations, world=world,
        estimates=estimates, metrics=metrics, tracks=tracks,
        geo_tracks=geo_tracks, geometry_series=geometry_series,
    )


def _representative_layout(geometry_series, t_ref: float) -> RelativeLayout:
    """The geometry-series layout whose window center is nearest ``t_ref``."""
    return min(geometry_series, key=lambda kv: abs(kv[0] - t_ref))[1]


def _detect_arrivals_into(observations, scenario: Scenario):
    """Synthesize audio, run matched-filter detection, swap in the detected arrivals."""
    audio_rng = np.random.default_rng(scenario.seed + 9973)
    captures = synthesize_captures(scenario, audio_rng)
    detected = detect_arrivals(
        captures, reference_pulse(scenario),
        n_emissions=len(emission_times(scenario)), dt_s=scenario.dt_s,
    )
    acoustic = tuple(
        AcousticArrival(
            device_id=d.device_id, emission_idx=d.emission_idx,
            toa_local_s=d.toa_local_s, confidence=getattr(d, "confidence", 1.0),
        )
        for d in detected
    )
    return replace(observations, acoustic=acoustic)
