"""Load and validate YAML scenario files into typed :class:`Scenario` objects.

This is the boundary where untyped YAML becomes a typed, validated, immutable
scenario. The rest of the code never touches raw dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from .sim.scenario import DeviceSpec, NoiseSpec, Scenario, TrajectorySpec

_VALID_TRAJECTORY_KINDS = {"linear", "circular", "waypoints"}


def load_scenario(path: Union[str, Path], seed_override: Optional[int] = None) -> Scenario:
    """Read a YAML scenario file and return a validated :class:`Scenario`."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return scenario_from_dict(raw, seed_override=seed_override)


def scenario_from_dict(raw: Dict, seed_override: Optional[int] = None) -> Scenario:
    """Build and validate a :class:`Scenario` from a plain dict (e.g. parsed YAML)."""
    raw_devices = raw.get("devices") or []
    if not raw_devices:
        raise ValueError("scenario must define at least one device")

    devices = tuple(
        DeviceSpec(
            id=str(d["id"]),
            position_m=tuple(float(x) for x in d["position_m"]),
            clock_offset_s=float(d.get("clock_offset_s", 0.0)),
            clock_drift_ppm=float(d.get("clock_drift_ppm", 0.0)),
            proc_delay_s=float(d.get("proc_delay_s", 0.002)),
            has_gps=bool(d.get("has_gps", False)),
            velocity_mps=tuple(float(x) for x in d.get("velocity_mps", (0.0, 0.0, 0.0))),
        )
        for d in raw_devices
    )

    ids = [d.id for d in devices]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate device ids in scenario: {ids}")

    trajectory = _parse_trajectory(raw["trajectory"])
    extra_drones = tuple(_parse_trajectory(t) for t in raw.get("extra_drones", ()))
    gps_blackout = tuple(
        (float(w[0]), float(w[1])) for w in raw.get("gps_blackout", ())
    )

    noise = NoiseSpec(**(raw.get("noise") or {}))
    seed = seed_override if seed_override is not None else int(raw["seed"])
    origin = tuple(float(x) for x in raw["origin_latlon"])

    return Scenario(
        name=str(raw["name"]),
        seed=seed,
        speed_of_sound_mps=float(raw["speed_of_sound_mps"]),
        sample_rate_hz=float(raw["sample_rate_hz"]),
        duration_s=float(raw["duration_s"]),
        dt_s=float(raw["dt_s"]),
        ranging_rounds=int(raw["ranging_rounds"]),
        origin_latlon=origin,
        devices=devices,
        trajectory=trajectory,
        noise=noise,
        extra_drones=extra_drones,
        gps_blackout=gps_blackout,
        audio=dict(raw.get("audio", {})),
    )


def _parse_trajectory(traj_raw: Dict) -> TrajectorySpec:
    """Build and validate a TrajectorySpec from a raw trajectory dict."""
    trajectory = TrajectorySpec(
        kind=str(traj_raw["kind"]),
        params=dict(traj_raw.get("params", {})),
        z_m=float(traj_raw.get("z_m", 50.0)),
    )
    if trajectory.kind not in _VALID_TRAJECTORY_KINDS:
        raise ValueError(
            f"unknown trajectory kind {trajectory.kind!r}; "
            f"expected one of {sorted(_VALID_TRAJECTORY_KINDS)}"
        )
    return trajectory
