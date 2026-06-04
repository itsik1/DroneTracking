"""Ground-truth drone trajectory: true position as a function of time.

Lives in ``sim`` (it is truth). Supported kinds: ``linear`` (start->end over the full
duration), ``circular`` (constant radius/altitude), ``waypoints`` (piecewise-linear
interpolation of [x, y, t] points). Altitude is taken from ``trajectory.z_m``.
"""

from __future__ import annotations

import numpy as np

from .scenario import Scenario


def trajectory_position(scenario: Scenario, t: float) -> np.ndarray:
    """True drone position (3,) at time ``t`` seconds."""
    traj = scenario.trajectory
    z = traj.z_m
    p = traj.params

    if traj.kind == "linear":
        start = np.asarray(p["start_m"], dtype=float)
        end = np.asarray(p["end_m"], dtype=float)
        frac = t / scenario.duration_s if scenario.duration_s > 0 else 0.0
        xy = start + frac * (end - start)
        return np.array([xy[0], xy[1], z])

    if traj.kind == "circular":
        cx, cy = p["center_m"]
        r = float(p["radius_m"])
        w = float(p["angular_rate_rad_s"])
        return np.array([cx + r * np.cos(w * t), cy + r * np.sin(w * t), z])

    if traj.kind == "waypoints":
        pts = np.asarray(p["points_m"], dtype=float)  # (M, 3): x, y, t
        ts = pts[:, 2]
        x = np.interp(t, ts, pts[:, 0])
        y = np.interp(t, ts, pts[:, 1])
        return np.array([x, y, z])

    raise ValueError(f"unknown trajectory kind {traj.kind!r}")
