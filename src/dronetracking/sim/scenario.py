"""Immutable, typed description of a synthetic world.

These dataclasses are ground truth (positions, clocks, trajectory) and live in the
``sim`` package. They are frozen so a scenario cannot mutate mid-run (reproducibility).
GPS anchor lat/lon is *not* stored here — the simulator derives it from each device's
true position and the scenario origin, guaranteeing consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

Vec3 = Tuple[float, float, float]
LatLon = Tuple[float, float]


@dataclass(frozen=True)
class DeviceSpec:
    id: str
    position_m: Vec3  # true (x, y, z) in local ENU meters about the scenario origin
    clock_offset_s: float = 0.0  # b_i: bias at t=0
    clock_drift_ppm: float = 0.0  # s_i in parts-per-million (local = t*(1+s) + b)
    proc_delay_s: float = 0.002  # deliberate two-way-ranging turnaround delay
    has_gps: bool = False  # is this a georeferencing anchor?


@dataclass(frozen=True)
class TrajectorySpec:
    kind: str  # "linear" | "circular" | "waypoints"
    params: Dict  # kind-specific (see sim.trajectory)
    z_m: float = 50.0  # default altitude


@dataclass(frozen=True)
class NoiseSpec:
    ranging_timestamp_std_s: float = 0.0  # jitter on each two-way-ranging timestamp
    toa_std_s: float = 0.0  # jitter on drone-signal arrival times
    proc_delay_jitter_s: float = 0.0  # random part of processing delay
    gps_pos_std_m: float = 0.0  # anchor GPS horizontal noise (meters)


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    speed_of_sound_mps: float
    sample_rate_hz: float
    duration_s: float
    dt_s: float  # seconds between drone emissions / trajectory samples
    ranging_rounds: int  # two-way ranging exchanges per device pair
    origin_latlon: LatLon  # ENU tangent-plane origin (defines anchors' real-world frame)
    devices: Tuple[DeviceSpec, ...]
    trajectory: TrajectorySpec
    noise: NoiseSpec = field(default_factory=NoiseSpec)

    @property
    def device_ids(self) -> Tuple[str, ...]:
        return tuple(d.id for d in self.devices)

    @property
    def anchors(self) -> Tuple[DeviceSpec, ...]:
        return tuple(d for d in self.devices if d.has_gps)
