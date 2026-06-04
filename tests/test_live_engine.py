"""Tests for the streaming engine (live mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dronetracking import geo
from dronetracking.config import load_scenario
from dronetracking.live.engine import StreamEngine

SCN = Path(__file__).resolve().parents[1] / "scenarios"


@pytest.mark.slow
def test_engine_streams_snapshots_converging_to_truth():
    eng = StreamEngine(load_scenario(SCN / "field_5dev.yaml"))
    snaps = list(eng.snapshots())
    assert len(snaps) == len(eng.frames) > 5

    last = snaps[-1]
    assert last.devices and last.anchors  # static context present
    assert any(s.targets for s in snaps)  # the target gets tracked at some point
    assert last.index == last.total - 1

    # The streamed (georeferenced) target ends up near the true drone.
    tgt = last.targets[0]
    truth = last.true_targets[0]
    assert geo.haversine_m(tgt["lat"], tgt["lon"], truth["lat"], truth["lon"]) < 10.0

    # Every snapshot is JSON-serializable (it goes over SSE as JSON).
    json.dumps(last.to_dict())


@pytest.mark.slow
def test_engine_multi_target_tracks_all_three():
    eng = StreamEngine(load_scenario(SCN / "multi_drone.yaml"))
    snaps = list(eng.snapshots())
    assert len(snaps[-1].true_targets) == 3
    # By the end of the run all three drones are being tracked.
    assert len(snaps[-1].targets) == 3
