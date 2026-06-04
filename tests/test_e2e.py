"""End-to-end pipeline tests — the real acceptance gate.

The noise-free scenario proves the stages compose to recover truth (up to the
*physical* skew-during-flight ranging bias, which is ~cm, not numerical zero). The
noisy scenario is seeded, so it is deterministic, and asserts each stage stays within
a documented tolerance. Georeferencing accuracy is bounded by GPS-anchor noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dronetracking.config import load_scenario
from dronetracking.pipeline import run_pipeline

SCN = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.mark.slow
def test_noisefree_pipeline_recovers_truth_to_physical_limit():
    metrics = run_pipeline(load_scenario(SCN / "noisefree_ideal.yaml")).metrics
    # No measurement noise; residual is the physical skew-during-flight ranging bias.
    assert metrics["device_localization.rmse_m"] < 0.05
    assert metrics["clock_sync.offset_rmse_s"] < 1e-6
    assert metrics["clock_sync.drift_rmse_ppm"] < 0.01
    assert metrics["tracking.rmse_m"] < 0.10
    assert metrics["georeferencing.rmse_m"] < 0.05
    assert metrics["georeferencing.altitude_rmse_m"] < 0.10
    # A reflected layout would have flipped altitude (>>1 m); confirm it didn't.
    assert metrics["device_localization.rmse_m"] < 0.05


@pytest.mark.slow
def test_noisy_pipeline_stays_within_tolerances():
    # Seed is fixed in the YAML, so this is deterministic.
    metrics = run_pipeline(load_scenario(SCN / "field_5dev.yaml")).metrics
    assert metrics["device_localization.rmse_m"] < 0.10
    assert metrics["tracking.rmse_m"] < 0.50
    # Georeferencing is limited by GPS-anchor noise (~2 m), per the accuracy targets.
    assert metrics["georeferencing.rmse_m"] < 5.0
    assert metrics["georeferencing.altitude_rmse_m"] < 5.0


def test_pipeline_produces_complete_consistent_estimates():
    result = run_pipeline(load_scenario(SCN / "noisefree_ideal.yaml"))
    est = result.estimates
    assert est.layout.n_devices == 5
    assert est.track.positions_local.shape[1] == 3
    assert est.geo_track.latlon.shape[1] == 2
    # Track and its georeferenced counterpart are time-aligned.
    assert len(est.geo_track.latlon) == len(est.track.times_s)
    assert len(est.geo_track.altitude_m) == len(est.track.times_s)
    # Metrics dict is flat and JSON-friendly.
    for key, value in result.metrics.items():
        assert isinstance(key, str)
        assert not isinstance(value, (dict, list))
