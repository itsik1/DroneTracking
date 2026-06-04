"""Tests for the robustness / accuracy study harness (``dronetracking.studies``).

These exercise the *public* sweep API end-to-end against the headline scenario.
The grids are deliberately tiny (2-3 grid points, 2 seeds) so the suite stays fast;
the heavier device sweep is marked ``slow``.

What we pin down:
  * ``sweep_noise`` runs and shows the EXPECTED MONOTONIC TREND (tracking RMSE grows
    with the noise factor: factor 2.0 > factor 0.5);
  * the returned result is JSON-serializable (``json.dumps`` round-trips);
  * ``plot_sweep`` writes non-empty PNG(s) to ``tmp_path``.

Only ``config`` (to load the scenario) and the study package are imported; the
sweeps drive the real ``pipeline`` internally.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dronetracking import config
from dronetracking.studies import sweep

SCENARIO_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "field_5dev.yaml"


@pytest.fixture(scope="module")
def field_5dev():
    return config.load_scenario(SCENARIO_PATH)


@pytest.fixture(scope="module")
def noise_result(field_5dev):
    """One small noise sweep, shared across the (read-only) assertions below."""
    return sweep.sweep_noise(field_5dev, factors=[0.5, 1.0, 2.0], seeds=2)


# --------------------------------------------------------------------------- shape

def test_sweep_noise_result_shape(noise_result):
    assert noise_result["kind"] == "noise"
    assert noise_result["metric_keys"] == [
        "tracking.rmse_m",
        "device_localization.rmse_m",
        "georeferencing.rmse_m",
    ]
    points = noise_result["points"]
    assert [p["factor"] for p in points] == [0.5, 1.0, 2.0]
    for p in points:
        assert p["seeds"] == 2
        for key in noise_result["metric_keys"]:
            assert key in p["mean"] and key in p["median"]
            assert isinstance(p["mean"][key], float)
            assert isinstance(p["median"][key], float)


# ----------------------------------------------------------------- monotonic trend

def test_sweep_noise_monotonic_trend(noise_result):
    """More measurement noise must degrade the tracker: RMSE(2.0) > RMSE(0.5)."""
    points = {p["factor"]: p for p in noise_result["points"]}
    lo = points[0.5]["mean"]["tracking.rmse_m"]
    hi = points[2.0]["mean"]["tracking.rmse_m"]
    assert hi > lo, f"expected tracking RMSE to grow with noise, got {lo!r} -> {hi!r}"


# -------------------------------------------------------------- JSON-serializable

def test_sweep_noise_json_serializable(noise_result):
    blob = json.dumps(noise_result)  # must not raise
    assert json.loads(blob) == noise_result


# --------------------------------------------------------------------- plotting

def test_plot_sweep_writes_nonempty_png(noise_result, tmp_path):
    paths = sweep.plot_sweep(noise_result, tmp_path, title="noise sweep")
    assert paths, "plot_sweep returned no paths"
    for p in paths:
        assert isinstance(p, Path)
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0


# --------------------------------------------------------------- device sweep (slow)

@pytest.mark.slow
def test_sweep_devices_runs_and_keeps_anchors(field_5dev):
    """Device sweep over valid counts; each point keeps >=4 anchors and reports GDOP."""
    result = sweep.sweep_devices(field_5dev, counts=[4, 5], seeds=2)
    assert result["kind"] == "devices"
    points = result["points"]
    # field_5dev has exactly 4 anchors, so count 4 and 5 are both well-posed.
    assert [p["n_devices"] for p in points] == [4, 5]
    for p in points:
        assert p["n_anchors"] >= 4
        assert isinstance(p["mean"]["tracking.rmse_m"], float)
        # representative GDOP is available from the recovered layout.
        assert "gdop" in p and isinstance(p["gdop"], float)
    json.dumps(result)  # JSON-serializable too


@pytest.mark.slow
def test_sweep_devices_skips_underdetermined(field_5dev):
    """Counts that cannot keep >=4 anchors are skipped, not crashed."""
    result = sweep.sweep_devices(field_5dev, counts=[3, 4], seeds=1)
    # count 3 -> only 3 anchors among the first 3 devices -> skipped.
    assert [p["n_devices"] for p in result["points"]] == [4]
    assert result["skipped"] == [3]
