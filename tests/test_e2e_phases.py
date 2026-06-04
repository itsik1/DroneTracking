"""End-to-end acceptance tests for the iteration-2 phases (Ph3/4/6/9).

Each runs the full integrated pipeline on that phase's seeded scenario and asserts the
phase-specific metric stays within a documented tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dronetracking.config import load_scenario
from dronetracking.pipeline import run_pipeline

SCN = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.mark.slow
def test_multi_target_recovers_all_drones():
    r = run_pipeline(load_scenario(SCN / "multi_drone.yaml"))
    m = r.metrics
    assert m["multi_target.n_true"] == 3
    assert m["multi_target.n_tracks"] == 3  # right count, no spurious/missed tracks
    assert m["multi_target.mean_rmse_m"] < 1.0
    assert m["multi_target.max_rmse_m"] < 2.0
    assert len(r.tracks) == 3
    # distinct identities
    assert len({t.target_id for t in r.tracks}) == 3


@pytest.mark.slow
def test_moving_devices_geometry_tracked():
    r = run_pipeline(load_scenario(SCN / "moving_devices.yaml"))
    m = r.metrics
    assert m["geometry.n_windows"] >= 3
    assert m["geometry.mean_window_rmse_m"] < 2.0  # layout tracked as devices drift
    assert m["geometry.max_window_rmse_m"] < 5.0
    assert r.geometry_series is not None and len(r.geometry_series) >= 3


@pytest.mark.slow
def test_gps_denied_holds_and_reconverges():
    r = run_pipeline(load_scenario(SCN / "gps_denied.yaml"))
    m = r.metrics
    # Error during the blackout stays bounded (held transform), not catastrophic.
    assert m["gps_denied.rmse_blackout_m"] < 25.0
    assert m["gps_denied.rmse_available_m"] < 10.0
    # No discontinuity at recovery: largest frame-to-frame step is modest.
    assert m["gps_denied.max_step_m"] < 25.0


@pytest.mark.slow
def test_detection_localizes_from_synthesized_audio():
    r = run_pipeline(load_scenario(SCN / "detection_demo.yaml"), detect=True)
    m = r.metrics
    assert m["device_localization.rmse_m"] < 0.10
    assert m["tracking.rmse_m"] < 0.50  # drone tracked from matched-filter detections
