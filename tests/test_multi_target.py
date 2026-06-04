"""Acceptance tests for the multi-target tracking stage (Ph6).

These tests ISOLATE the data-association + multi-target tracking logic from the other
estimators (relative localization, clock sync): the layout is built from the TRUE device
positions and the clocks carry the TRUE offsets/drifts (dev0 reference, per the locked
convention). That makes every per-(emission, source) TDOA fix essentially exact, so any
identity swap or wrong track count is attributable to the *association*, not to upstream
noise.

Acoustic fixtures come from the sim leaf ``generate_multi_arrivals`` (importing sim leaf
functions in *tests* is permitted by the contract — only estimation *source* must not
import sim). ``estimation.multi_target`` itself never imports sim; the firewall test
(tests/test_no_truth_leak.py) enforces that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Frozen estimation contract / shared types (allowed in source and tests).
from dronetracking.estimation.interfaces import ClockEstimates, RelativeLayout, Track

# Module under test.
from dronetracking.estimation import multi_target

# Frozen sim leaf functions / config — TESTS may use these for realistic fixtures.
from dronetracking.config import load_scenario, scenario_from_dict
from dronetracking.sim.multi_acoustic import generate_multi_arrivals, true_tracks

C = 343.0
_SCENARIO_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "multi_drone.yaml"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

def _truth_layout_and_clocks(scenario):
    """TRUE device layout + TRUE clocks (dev0 reference). Isolates association."""
    device_ids = scenario.device_ids
    positions = np.array([d.position_m for d in scenario.devices], dtype=float)
    layout = RelativeLayout(device_ids=device_ids, positions_local=positions, covariances=None)
    offsets = {d.id: d.clock_offset_s for d in scenario.devices}
    drifts = {d.id: d.clock_drift_ppm for d in scenario.devices}
    clocks = ClockEstimates(
        device_ids=device_ids,
        offsets_s=offsets,
        drifts_ppm=drifts,
        reference_id=device_ids[0],
    )
    return layout, clocks


def _two_drone_scenario():
    """Two well-separated drones (one linear, one circular) over a 5-device array.

    Built by hand (not from YAML) so the association test does not depend on the demo
    scenario file. Non-coplanar anchors -> a well-conditioned 3D fix.
    """
    raw = {
        "name": "two_drone_test",
        "seed": 0,
        "speed_of_sound_mps": C,
        "sample_rate_hz": 48000.0,
        "duration_s": 12.0,
        "dt_s": 0.5,
        "ranging_rounds": 10,
        "origin_latlon": [32.0, 34.0],
        "devices": [
            {"id": "dev0", "position_m": [0.0, 0.0, 0.0], "clock_offset_s": 0.0, "clock_drift_ppm": 0.0, "has_gps": True},
            {"id": "dev1", "position_m": [200.0, 0.0, 12.0], "clock_offset_s": 0.10, "clock_drift_ppm": 25.0, "has_gps": True},
            {"id": "dev2", "position_m": [0.0, 200.0, 8.0], "clock_offset_s": -0.07, "clock_drift_ppm": -18.0, "has_gps": True},
            {"id": "dev3", "position_m": [200.0, 200.0, 20.0], "clock_offset_s": 0.04, "clock_drift_ppm": 40.0, "has_gps": True},
            {"id": "dev4", "position_m": [100.0, 100.0, 35.0], "clock_offset_s": -0.12, "clock_drift_ppm": 15.0},
        ],
        # Drone 0: linear sweep high overhead.
        "trajectory": {
            "kind": "linear",
            "z_m": 120.0,
            "params": {"start_m": [-40.0, 60.0], "end_m": [260.0, 140.0]},
        },
        # Drone 1: a circle on the opposite side, lower and well separated from drone 0.
        "extra_drones": [
            {
                "kind": "circular",
                "z_m": 70.0,
                "params": {"center_m": [100.0, 100.0], "radius_m": 40.0, "angular_rate_rad_s": 0.25},
            }
        ],
    }
    return scenario_from_dict(raw)


def _emission_index_of(t, frame_times):
    """Map a track-sample time back to its emission index.

    ``true_tracks`` is indexed by emission, but ``TargetFix.t`` (hence a track's
    ``times_s``) is the *reference-timebase emission time* = emission time + range / c, so
    it is offset from the bare emission time by the (per-frame) propagation delay — for a
    high drone that offset can exceed ``dt_s``, which makes ``round(t / dt)`` land on the
    wrong emission. Frame ``i`` corresponds to emission ``i`` (every emission yields one
    frame), so the robust map is "nearest frame time -> that frame's index".
    """
    return int(np.argmin(np.abs(np.asarray(frame_times) - t)))


def _match_tracks_to_truth(tracks, truth, frame_times):
    """Greedy nearest assignment of each truth drone to the track whose samples best fit.

    Returns dict ``{drone_idx: (track, mean_err)}``. ``frame_times`` maps each track sample
    time back to its emission index (truth is indexed by emission).
    """
    assignment = {}
    used = set()
    for k, truth_xyz in truth.items():
        best_err = np.inf
        best_track = None
        for ti, tr in enumerate(tracks):
            if ti in used:
                continue
            # Compare each track sample to the truth at its emission index.
            errs = []
            for t, pos in zip(tr.times_s, tr.positions_local):
                idx = min(_emission_index_of(t, frame_times), len(truth_xyz) - 1)
                errs.append(np.linalg.norm(pos - truth_xyz[idx]))
            err = float(np.mean(errs)) if errs else np.inf
            if err < best_err:
                best_err = err
                best_track = ti
        assignment[k] = (tracks[best_track], best_err)
        used.add(best_track)
    return assignment


# --------------------------------------------------------------------------------------
# generate_multi_arrivals (sim leaf used as a fixture)
# --------------------------------------------------------------------------------------

def test_generate_multi_arrivals_tags_each_source_and_covers_all_devices():
    sc = _two_drone_scenario()
    rng = np.random.default_rng(0)
    arrivals = generate_multi_arrivals(sc, rng)

    n_drones = len(sc.all_drones)
    n_dev = len(sc.devices)
    n_emit = len(np.arange(0.0, sc.duration_s, sc.dt_s))

    # One arrival per (device, emission, drone).
    assert len(arrivals) == n_drones * n_dev * n_emit
    assert {a.source for a in arrivals} == set(range(n_drones))

    # Every (emission, source) group has all devices present.
    groups = {}
    for a in arrivals:
        groups.setdefault((a.emission_idx, a.source), set()).add(a.device_id)
    for key, ids in groups.items():
        assert ids == set(sc.device_ids), key


def test_true_tracks_shapes_and_distinct_per_drone():
    sc = _two_drone_scenario()
    truth = true_tracks(sc)
    n_emit = len(np.arange(0.0, sc.duration_s, sc.dt_s))

    assert set(truth.keys()) == set(range(len(sc.all_drones)))
    for k, arr in truth.items():
        assert arr.shape == (n_emit, 3)
    # The two drones are well separated at every emission.
    sep = np.linalg.norm(truth[0] - truth[1], axis=1)
    assert sep.min() > 30.0


# --------------------------------------------------------------------------------------
# localize_frames
# --------------------------------------------------------------------------------------

def test_localize_frames_yields_one_fix_per_source_per_frame_at_truth():
    sc = _two_drone_scenario()
    layout, clocks = _truth_layout_and_clocks(sc)
    arrivals = generate_multi_arrivals(sc, np.random.default_rng(1))

    frames = multi_target.localize_frames(arrivals, clocks, layout, sc.speed_of_sound_mps)
    truth = true_tracks(sc)
    n_emit = len(np.arange(0.0, sc.duration_s, sc.dt_s))

    # One frame per emission, each with exactly n_drones fixes (every source has 5 devices).
    assert len(frames) == n_emit
    # Frames are time-ordered.
    times = [t for t, _ in frames]
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    n_drones = len(sc.all_drones)
    for emission_idx, (t, fixes) in enumerate(frames):
        assert len(fixes) == n_drones
        # The set of fixes must match the set of true drone positions at this emission
        # (order-independent: localize_frames must not leak source ordering downstream).
        truth_pts = np.array([truth[k][emission_idx] for k in range(n_drones)])
        for fix in fixes:
            d = np.min(np.linalg.norm(truth_pts - fix.position, axis=1))
            assert d < 1.0, f"frame {emission_idx}: fix {fix.position} not near any truth"


def test_localize_frames_skips_sources_with_too_few_devices():
    sc = _two_drone_scenario()
    layout, clocks = _truth_layout_and_clocks(sc)
    arrivals = list(generate_multi_arrivals(sc, np.random.default_rng(2)))

    # Cripple source 1 at emission 0 (leave only 3 devices) -> that fix must be dropped,
    # but source 0 at emission 0 must still produce a fix.
    kept = [
        a for a in arrivals
        if not (a.emission_idx == 0 and a.source == 1 and a.device_id in ("dev3", "dev4"))
    ]
    frames = multi_target.localize_frames(kept, clocks, layout, sc.speed_of_sound_mps)
    frame0 = frames[0]
    assert len(frame0[1]) == 1  # only source 0 survived at emission 0


# --------------------------------------------------------------------------------------
# track_targets: the core multi-target acceptance test.
# --------------------------------------------------------------------------------------

def test_track_targets_recovers_two_distinct_tracks_no_identity_swap():
    sc = _two_drone_scenario()
    layout, clocks = _truth_layout_and_clocks(sc)
    arrivals = generate_multi_arrivals(sc, np.random.default_rng(3))
    frames = multi_target.localize_frames(arrivals, clocks, layout, sc.speed_of_sound_mps)

    tracks = multi_target.track_targets(frames)
    truth = true_tracks(sc)
    frame_times = [t for t, _ in frames]

    # Exactly two confirmed tracks, with distinct target ids.
    assert len(tracks) == 2, f"expected 2 tracks, got {len(tracks)}"
    ids = [tr.target_id for tr in tracks]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == 2, f"target ids not distinct: {ids}"

    # Each truth drone is matched by one track within a tight tolerance.
    assignment = _match_tracks_to_truth(tracks, truth, frame_times)
    matched_tracks = [tr for tr, _ in assignment.values()]
    assert len({id(t) for t in matched_tracks}) == 2  # a 1:1 matching
    for k, (tr, err) in assignment.items():
        assert err < 5.0, f"drone {k}: mean track error {err:.2f} m"

    # Identity stability: each track, over its life, stays closer to ITS drone than to
    # the other drone at every sample (no mid-run swap).
    for k, (tr, _) in assignment.items():
        other = 1 - k
        for t, pos in zip(tr.times_s, tr.positions_local):
            idx = min(_emission_index_of(t, frame_times), truth[k].shape[0] - 1)
            d_self = np.linalg.norm(pos - truth[k][idx])
            d_other = np.linalg.norm(pos - truth[other][idx])
            assert d_self < d_other, (
                f"track for drone {k} at t={t}: closer to drone {other} "
                f"({d_other:.1f} m) than to {k} ({d_self:.1f} m) -> identity swap"
            )

    # Each confirmed track spans most of the run (covers the bulk of the emissions).
    n_emit = len(np.arange(0.0, sc.duration_s, sc.dt_s))
    for tr in tracks:
        assert len(tr.times_s) >= int(0.7 * n_emit)


def test_track_targets_single_target_scenario_yields_one_track():
    """Degenerate (one drone) case: GNN must still confirm exactly one track."""
    sc = _two_drone_scenario()
    # Drop the extra drone -> a single-target scenario.
    import dataclasses

    sc1 = dataclasses.replace(sc, extra_drones=())
    layout, clocks = _truth_layout_and_clocks(sc1)
    arrivals = generate_multi_arrivals(sc1, np.random.default_rng(4))
    frames = multi_target.localize_frames(arrivals, clocks, layout, sc1.speed_of_sound_mps)

    tracks = multi_target.track_targets(frames)
    assert len(tracks) == 1
    truth = true_tracks(sc1)[0]
    frame_times = [t for t, _ in frames]
    tr = tracks[0]
    errs = [
        np.linalg.norm(pos - truth[min(_emission_index_of(t, frame_times), truth.shape[0] - 1)])
        for t, pos in zip(tr.times_s, tr.positions_local)
    ]
    assert float(np.mean(errs)) < 5.0


def test_track_targets_returns_empty_for_no_frames():
    assert multi_target.track_targets([]) == []


# --------------------------------------------------------------------------------------
# The demo scenario file must load and exercise the full pipeline.
# --------------------------------------------------------------------------------------

def test_multi_drone_scenario_file_loads_and_tracks():
    assert _SCENARIO_PATH.exists(), f"missing demo scenario {_SCENARIO_PATH}"
    sc = load_scenario(_SCENARIO_PATH)

    # >=5 devices, >=4 GPS anchors at varied heights (georef-ready).
    assert len(sc.devices) >= 5
    anchors = sc.anchors
    assert len(anchors) >= 4
    anchor_z = sorted({round(a.position_m[2], 3) for a in anchors})
    assert len(anchor_z) >= 3, f"anchors not at varied heights: {anchor_z}"

    # Primary + 1-2 extra drones.
    assert 2 <= len(sc.all_drones) <= 3

    layout, clocks = _truth_layout_and_clocks(sc)
    arrivals = generate_multi_arrivals(sc, np.random.default_rng(5))
    frames = multi_target.localize_frames(arrivals, clocks, layout, sc.speed_of_sound_mps)
    tracks = multi_target.track_targets(frames)

    # One confirmed track per drone, identities distinct.
    assert len(tracks) == len(sc.all_drones)
    assert len({tr.target_id for tr in tracks}) == len(tracks)

    truth = true_tracks(sc)
    frame_times = [t for t, _ in frames]
    assignment = _match_tracks_to_truth(tracks, truth, frame_times)
    assert len({id(t) for t, _ in assignment.values()}) == len(sc.all_drones)
    for k, (tr, err) in assignment.items():
        assert err < 8.0, f"demo drone {k}: mean track error {err:.2f} m"
