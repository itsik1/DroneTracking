"""Streaming estimation engine: calibrate once, then process emissions in time order.

Mirrors live operation. ``setup`` (in ``__init__``) does the one-time network
calibration — relative geometry, clock offsets/drifts, and the GPS georeference — then
:meth:`StreamEngine.snapshots` walks the drone emissions in time order, growing the
track estimate one frame at a time and yielding a georeferenced :class:`Snapshot` per
step. Swapping the simulated source for real device feeds leaves this loop unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List

import numpy as np

from .. import geo
from ..estimation.clock_sync import estimate_clocks
from ..estimation.detection import detect_arrivals
from ..estimation.geometry_tracking import track_geometry
from ..estimation.georeference import solve_transform
from ..estimation.multi_target import localize_frames, track_targets
from ..estimation.ranging import build_distance_matrix
from ..estimation.relative_localization import estimate_layout
from ..sim.acoustic import emission_times
from ..sim.audio import reference_pulse, synthesize_captures
from ..sim.observations import AcousticArrival
from ..sim.scenario import Scenario
from ..sim.simulator import simulate


@dataclass
class Snapshot:
    """One frame of live state, JSON-serializable for the dashboard."""

    t: float
    index: int
    total: int
    devices: List[Dict[str, Any]]  # estimated device positions (georeferenced)
    anchors: List[Dict[str, Any]]  # GPS anchors
    targets: List[Dict[str, Any]]  # tracked drones: id, lat, lon, alt, r_m (1σ horiz radius)
    true_targets: List[Dict[str, Any]]  # ground-truth drone positions (sim overlay)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t, "index": self.index, "total": self.total,
            "devices": self.devices, "anchors": self.anchors,
            "targets": self.targets, "true_targets": self.true_targets,
        }


class StreamEngine:
    """Calibrate a scenario's network, then stream per-emission state snapshots."""

    def __init__(self, scenario: Scenario, *, detect: bool = False, model: str = "cv", sigma_a: float = 2.0):
        self.scenario = scenario
        self.model = model
        self.sigma_a = sigma_a

        observations, self.world = simulate(scenario)

        # --- one-time calibration: geometry, clocks, georeference ---
        if scenario.devices_move:
            win = max(scenario.duration_s / 6.0, 2.0 * scenario.dt_s)
            series = track_geometry(
                observations.ranging, observations.device_ids,
                observations.speed_of_sound_mps, window_s=win, step_s=win / 2.0,
            )
            self.layout = min(series, key=lambda kv: abs(kv[0] - scenario.duration_s / 2.0))[1]
        else:
            self.layout = estimate_layout(build_distance_matrix(observations))

        self.clocks = estimate_clocks(observations)

        if detect and not scenario.extra_drones:
            observations = self._detect(observations)

        self.transform = solve_transform(self.layout, observations.anchor_gps, scenario.origin_latlon)
        self.origin = tuple(scenario.origin_latlon)

        # Per-emission target fixes, in time order — the live "feed".
        self.frames = localize_frames(
            observations.acoustic, self.clocks, self.layout, observations.speed_of_sound_mps
        )

        self._devices = self._device_latlon()
        self._anchors = [
            {"id": k, "lat": float(v[0]), "lon": float(v[1])}
            for k, v in self.world.anchor_latlon.items()
        ]

    # -- snapshot stream -----------------------------------------------------
    def snapshots(self) -> Iterator[Snapshot]:
        n = len(self.frames)
        true_tracks = dict(getattr(self.world, "true_tracks", {}) or {})
        for i in range(n):
            tracks = track_targets(self.frames[: i + 1])  # online: tracker over the prefix
            t = float(self.frames[i][0])

            targets = []
            for k, tr in enumerate(tracks):
                enu = self.transform.apply(np.asarray(tr.positions_local[-1], dtype=float))
                lat, lon = geo.enu_to_latlon(enu[0], enu[1], self.origin)
                targets.append({
                    "id": tr.target_id or f"T{k}",
                    "lat": float(lat), "lon": float(lon), "alt": float(enu[2]),
                    "r_m": self._horizontal_radius(np.asarray(tr.covariances[-1], dtype=float)),
                })

            true_targets = []
            for src, trk in sorted(true_tracks.items()):
                j = min(i, len(trk) - 1)
                la, lo = geo.enu_to_latlon(trk[j][0], trk[j][1], self.origin)
                true_targets.append({"src": int(src), "lat": float(la), "lon": float(lo)})

            yield Snapshot(t, i, n, self._devices, self._anchors, targets, true_targets)

    # -- helpers -------------------------------------------------------------
    def _device_latlon(self) -> List[Dict[str, Any]]:
        out = []
        for d in self.layout.device_ids:
            enu = self.transform.apply(self.layout.position_of(d))
            lat, lon = geo.enu_to_latlon(enu[0], enu[1], self.origin)
            out.append({"id": d, "lat": float(lat), "lon": float(lon)})
        return out

    def _horizontal_radius(self, cov_local: np.ndarray) -> float:
        """1σ horizontal radius (m) of a track covariance, rotated into ENU."""
        R, s = self.transform.R, self.transform.scale
        cov_enu = (s * s) * (R @ cov_local @ R.T)
        return float(np.sqrt(max(cov_enu[0, 0] + cov_enu[1, 1], 0.0)))

    def _detect(self, observations):
        rng = np.random.default_rng(self.scenario.seed + 9973)
        captures = synthesize_captures(self.scenario, rng)
        detected = detect_arrivals(
            captures, reference_pulse(self.scenario),
            n_emissions=len(emission_times(self.scenario)), dt_s=self.scenario.dt_s,
        )
        acoustic = tuple(
            AcousticArrival(
                device_id=d.device_id, emission_idx=d.emission_idx,
                toa_local_s=d.toa_local_s, confidence=getattr(d, "confidence", 1.0),
            )
            for d in detected
        )
        return replace(observations, acoustic=acoustic)
