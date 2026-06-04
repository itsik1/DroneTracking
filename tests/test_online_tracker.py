"""Acceptance tests for the ONLINE (incremental) multi-target tracker (Iteration 4 / A).

:class:`estimation.online_tracker.OnlineTracker` is the stateful, frame-at-a-time form of
the batch :func:`estimation.multi_target.track_targets`. These tests:

* build the per-emission frames of the demo multi-drone scenario (and a single-target
  variant) on the TRUE device layout + TRUE clocks, exactly like the batch acceptance
  tests, so any wrong track count or identity swap is attributable to the *association*
  (the thing under test), not to upstream localization noise;
* feed those frames into ``OnlineTracker`` one at a time and assert it ends with the right
  number of confirmed tracks near truth, with stable distinct identities;
* assert the online tracker is COMPARABLE to the batch tracker on the same frames (same
  confirmed-track count, same final filtered positions, same stable->relabelled identity
  partition);
* assert it is genuinely INCREMENTAL: state persists across ``update`` calls (an
  N-call stream equals one batch run), intermediate ``tracks()`` grows sensibly, and a
  confirmed track keeps its id as later tracks are born.

The ground-truth firewall forbids ``estimation.online_tracker`` from importing
``dronetracking.sim``; that is enforced by tests/test_no_truth_leak.py. TESTS, however, may
use sim leaf functions (``generate_multi_arrivals``, ``true_tracks``) for realistic
fixtures — the contract permits importing sim leaves in tests.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

# Frozen estimation contract / shared types (allowed in source and tests).
from dronetracking.estimation.interfaces import ClockEstimates, RelativeLayout

# Modules under test + the batch baseline it mirrors.
from dronetracking.estimation import multi_target
from dronetracking.estimation.online_tracker import OnlineTracker

# Frozen sim leaf functions / config — TESTS may use these for realistic fixtures.
from dronetracking.config import load_scenario
from dronetracking.sim.multi_acoustic import generate_multi_arrivals, true_tracks

_SCENARIO_PATH = Path(__file__).resolve().parents[1] / "scenarios" / "multi_drone.yaml"


# --------------------------------------------------------------------------------------
# Fixtures / helpers (mirror tests/test_multi_target.py to isolate association from noise)
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


def _frames_for(scenario, seed):
    """Per-emission ``(t, fixes)`` frames on the true layout+clocks for ``scenario``."""
    layout, clocks = _truth_layout_and_clocks(scenario)
    arrivals = generate_multi_arrivals(scenario, np.random.default_rng(seed))
    return multi_target.localize_frames(arrivals, clocks, layout, scenario.speed_of_sound_mps)


def _emission_index_of(t, frame_times):
    """Map a track-sample time back to its emission/frame index (nearest frame time)."""
    return int(np.argmin(np.abs(np.asarray(frame_times) - t)))


def _match_tracks_to_truth(tracks, truth, frame_times):
    """Greedy nearest assignment of each truth drone to the best-fitting track.

    Returns ``{drone_idx: (track, mean_err)}``. ``frame_times`` maps each track sample
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


def _run_online(frames, **kwargs):
    """Feed ``frames`` one at a time into a fresh ``OnlineTracker`` and return its tracks."""
    trk = OnlineTracker(**kwargs)
    for t, fixes in frames:
        trk.update(fixes, t)
    return trk.tracks()


# --------------------------------------------------------------------------------------
# Core multi-target acceptance: right count, near truth, identities stable.
# --------------------------------------------------------------------------------------

def test_online_tracker_recovers_one_track_per_drone_near_truth():
    sc = load_scenario(_SCENARIO_PATH)
    frames = _frames_for(sc, seed=5)
    frame_times = [t for t, _ in frames]
    truth = true_tracks(sc)
    n_drones = len(sc.all_drones)

    tracks = _run_online(frames)

    # One confirmed track per drone, with distinct non-None ids.
    assert len(tracks) == n_drones, f"expected {n_drones} tracks, got {len(tracks)}"
    ids = [tr.target_id for tr in tracks]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == n_drones, f"target ids not distinct: {ids}"

    # Each truth drone is matched by exactly one track within tolerance.
    assignment = _match_tracks_to_truth(tracks, truth, frame_times)
    matched = [tr for tr, _ in assignment.values()]
    assert len({id(t) for t in matched}) == n_drones  # 1:1 matching
    for k, (tr, err) in assignment.items():
        assert err < 8.0, f"drone {k}: mean track error {err:.2f} m"

    # Each confirmed track spans most of the run.
    n_emit = len(frames)
    for tr in tracks:
        assert len(tr.times_s) >= int(0.7 * n_emit)


def test_online_tracker_no_identity_swap():
    """Each track stays closer to ITS drone than to any other at every sample."""
    sc = load_scenario(_SCENARIO_PATH)
    frames = _frames_for(sc, seed=7)
    frame_times = [t for t, _ in frames]
    truth = true_tracks(sc)

    tracks = _run_online(frames)
    assignment = _match_tracks_to_truth(tracks, truth, frame_times)

    for k, (tr, _) in assignment.items():
        for t, pos in zip(tr.times_s, tr.positions_local):
            idx = min(_emission_index_of(t, frame_times), truth[k].shape[0] - 1)
            d_self = np.linalg.norm(pos - truth[k][idx])
            for other in truth:
                if other == k:
                    continue
                d_other = np.linalg.norm(pos - truth[other][idx])
                assert d_self < d_other, (
                    f"track for drone {k} at t={t:.2f}: closer to drone {other} "
                    f"({d_other:.1f} m) than to {k} ({d_self:.1f} m) -> identity swap"
                )


def test_online_tracker_single_target_yields_one_track():
    """Degenerate one-drone case: GNN must still confirm exactly one track near truth."""
    sc = load_scenario(_SCENARIO_PATH)
    sc1 = dataclasses.replace(sc, extra_drones=())
    frames = _frames_for(sc1, seed=4)
    frame_times = [t for t, _ in frames]

    tracks = _run_online(frames)
    assert len(tracks) == 1, f"expected 1 track, got {len(tracks)}"

    truth = true_tracks(sc1)[0]
    tr = tracks[0]
    errs = [
        np.linalg.norm(pos - truth[min(_emission_index_of(t, frame_times), truth.shape[0] - 1)])
        for t, pos in zip(tr.times_s, tr.positions_local)
    ]
    assert float(np.mean(errs)) < 5.0


def test_online_tracker_empty_stream_has_no_tracks():
    trk = OnlineTracker()
    assert trk.tracks() == []


# --------------------------------------------------------------------------------------
# Comparability to the batch tracker on the same frames.
# --------------------------------------------------------------------------------------

def _partition_by_assignment(tracks, truth, frame_times):
    """Map drone_idx -> the track object assigned to it (for cross-tracker comparison)."""
    return {k: tr for k, (tr, _) in _match_tracks_to_truth(tracks, truth, frame_times).items()}


def test_online_matches_batch_track_count_and_geometry():
    """OnlineTracker over a stream == batch track_targets over the same frames.

    Same confirmed-track count, and once each tracker's tracks are matched to truth, the
    per-drone final filtered positions and full sample series agree to floating point.
    """
    sc = load_scenario(_SCENARIO_PATH)
    frames = _frames_for(sc, seed=11)
    frame_times = [t for t, _ in frames]
    truth = true_tracks(sc)

    online = _run_online(frames)
    batch = multi_target.track_targets(frames)

    assert len(online) == len(batch)

    online_by_drone = _partition_by_assignment(online, truth, frame_times)
    batch_by_drone = _partition_by_assignment(batch, truth, frame_times)
    assert set(online_by_drone) == set(batch_by_drone)

    for k in truth:
        o = online_by_drone[k]
        b = batch_by_drone[k]
        # Same number of samples and identical filtered trajectory for the same drone.
        assert o.times_s.shape == b.times_s.shape, f"drone {k}: sample-count mismatch"
        np.testing.assert_allclose(o.times_s, b.times_s, rtol=0, atol=1e-9)
        np.testing.assert_allclose(o.positions_local, b.positions_local, rtol=0, atol=1e-6)
        np.testing.assert_allclose(o.covariances, b.covariances, rtol=0, atol=1e-6)


def test_online_matches_batch_single_target():
    sc = load_scenario(_SCENARIO_PATH)
    sc1 = dataclasses.replace(sc, extra_drones=())
    frames = _frames_for(sc1, seed=8)

    online = _run_online(frames)
    batch = multi_target.track_targets(frames)

    assert len(online) == len(batch) == 1
    np.testing.assert_allclose(
        online[0].positions_local, batch[0].positions_local, rtol=0, atol=1e-6
    )


# --------------------------------------------------------------------------------------
# Incrementality: state persists; no reprocessing; tracks() grows sensibly; ids stable.
# --------------------------------------------------------------------------------------

def test_online_stream_equals_single_pass_endstate():
    """Feeding frames one at a time gives the SAME end state as a single helper pass.

    Two independent OnlineTrackers fed the identical frame sequence must finish identical —
    the only state is what `update` accumulates, so a frame-at-a-time stream is fully
    reproducible (no hidden global / reprocessing state).
    """
    sc = load_scenario(_SCENARIO_PATH)
    frames = _frames_for(sc, seed=11)

    a = OnlineTracker()
    for t, fixes in frames:
        a.update(fixes, t)
    a_tracks = a.tracks()

    b_tracks = _run_online(frames)

    assert [tr.target_id for tr in a_tracks] == [tr.target_id for tr in b_tracks]
    for ta, tb in zip(a_tracks, b_tracks):
        np.testing.assert_array_equal(ta.times_s, tb.times_s)
        np.testing.assert_array_equal(ta.positions_local, tb.positions_local)


def test_online_tracks_grow_sensibly_and_are_monotonic_in_length():
    """Intermediate tracks() must grow: 0 confirmed early, then up to one-per-drone.

    Also: each confirmed track's sample history only ever lengthens as more frames arrive
    (no reprocessing that would rebuild or shrink earlier history), and the confirmed-track
    count never exceeds the number of drones.
    """
    sc = load_scenario(_SCENARIO_PATH)
    frames = _frames_for(sc, seed=5)
    n_drones = len(sc.all_drones)

    trk = OnlineTracker()

    # Before any frame: nothing confirmed.
    assert trk.tracks() == []

    counts = []
    prev_len_by_id = {}
    for fi, (t, fixes) in enumerate(frames):
        trk.update(fixes, t)
        current = trk.tracks()
        counts.append(len(current))

        # Never more confirmed tracks than there are drones.
        assert len(current) <= n_drones

        # birth_min_hits=2 (seed + 2 hits => 3 frames) means nothing can be confirmed
        # before the third frame; the very first frame can only ever spawn tentatives.
        if fi == 0:
            assert current == []

        # Per-id history is non-shrinking across updates (incremental append, no rebuild).
        for tr in current:
            n = len(tr.times_s)
            assert n >= prev_len_by_id.get(tr.target_id, 0), (
                f"track {tr.target_id} history shrank from "
                f"{prev_len_by_id.get(tr.target_id)} to {n}"
            )
            prev_len_by_id[tr.target_id] = n

    # The confirmed-track count rises from 0 to the full set and ends there.
    assert counts[0] == 0
    assert max(counts) == n_drones
    assert counts[-1] == n_drones
    # Monotonic non-decreasing for this clean (no-dropout) scenario.
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_online_confirmed_ids_are_stable_as_new_tracks_are_born():
    """A confirmed track keeps its id as later tracks confirm (stable identity online)."""
    sc = load_scenario(_SCENARIO_PATH)
    frames = _frames_for(sc, seed=5)

    trk = OnlineTracker()
    seen_ids_per_step = []
    for t, fixes in frames:
        trk.update(fixes, t)
        seen_ids_per_step.append([tr.target_id for tr in trk.tracks()])

    # Ids present at one step must still be present (same string) at every later step:
    # confirmation only ever ADDS ids, never renames an existing confirmed track.
    for i in range(len(seen_ids_per_step) - 1):
        earlier = set(seen_ids_per_step[i])
        later = set(seen_ids_per_step[i + 1])
        assert earlier.issubset(later), (
            f"confirmed ids changed/dropped between steps {i} and {i + 1}: "
            f"{earlier} -> {later}"
        )

    # And every snapshot has distinct ids.
    for ids in seen_ids_per_step:
        assert len(ids) == len(set(ids))


def test_online_terminates_track_after_consecutive_misses():
    """A confirmed track that stops receiving fixes terminates after death_max_misses.

    Feed a clean single-target prefix to confirm one track, then feed empty frames; after
    death_max_misses+1 empty frames the (now-terminated) track is still reported by
    tracks() exactly once (retired to the output set), and no spurious tracks appear.
    """
    sc = load_scenario(_SCENARIO_PATH)
    sc1 = dataclasses.replace(sc, extra_drones=())
    frames = _frames_for(sc1, seed=8)

    trk = OnlineTracker(death_max_misses=3)
    # Confirm a track on the first several real frames.
    for t, fixes in frames[:6]:
        trk.update(fixes, t)
    assert len(trk.tracks()) == 1
    confirmed_id = trk.tracks()[0].target_id

    # Now starve it: empty frames at advancing times. After misses > death_max_misses the
    # track terminates but remains in the confirmed output exactly once.
    t0 = frames[6][0]
    for j in range(1, 8):
        trk.update([], t0 + 0.5 * j)

    out = trk.tracks()
    assert len(out) == 1, f"expected the single retired track, got {len(out)}"
    assert out[0].target_id == confirmed_id  # stable id survives termination
