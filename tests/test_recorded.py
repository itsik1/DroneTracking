"""Tests for the recorded-audio device feed (Iteration 4 — hardware bringup).

:class:`~dronetracking.sources.recorded.RecordedAudioFeed` is the "ingest real recorded
device data" bridge: it reads per-device WAVs + a ``meta.json`` from a directory, runs the
matched-filter detector on the audio, and serves the standard ``DeviceFeed`` surface.

We synthesize a small but realistic recorded dataset in ``tmp_path`` the same way real
hardware would lay one down:

* per-device waveforms come from :func:`sim.audio.synthesize_captures`, written as WAVs
  with :func:`scipy.io.wavfile.write`;
* ``meta.json`` carries the ranging exchanges + GPS anchors (encoded with the live wire
  :mod:`~dronetracking.live.protocol`, exactly as an on-device agent would publish them),
  the two timebase constants, the emission count / spacing, and the reference-pulse params.

Then the feed must reproduce the ranging / anchor sets exactly (they are passed through
from meta) and recover acoustic arrivals whose times land within a few samples of the
true local arrivals the simulator placed in the audio.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from dronetracking.config import load_scenario
from dronetracking.live import protocol
from dronetracking.sim.audio import reference_pulse, synthesize_captures
from dronetracking.sim.observations import (
    AcousticArrival,
    AnchorGps,
    Observations,
    RangingRecord,
)
from dronetracking.sources.recorded import RecordedAudioFeed, write_recorded_dataset
from dronetracking.sources.simulated import SimulatedDeviceFeed

SCN = Path(__file__).resolve().parents[1] / "scenarios"
# detection_demo is purpose-built for this: dt_s (2 s) >> the ~0.7 s cross-device range
# spread, a clean SNR, and a single drone — so matched-filter ordering maps to emission idx.
SCENARIO_FILE = "detection_demo.yaml"


# --------------------------------------------------------------------------- #
# dataset builder: mirror how real hardware would lay a recording on disk
# --------------------------------------------------------------------------- #
def _build_dataset(directory: Path, scenario, *, dtype=np.float32):
    """Synthesize per-device WAVs + meta.json into ``directory``; return the sim feed.

    Uses the public ``write_recorded_dataset`` helper so the test exercises the same
    on-disk layout the feed reads. Returns the :class:`SimulatedDeviceFeed` whose
    ground truth the assertions compare against.
    """
    sim_feed = SimulatedDeviceFeed(scenario)
    rng = np.random.default_rng(scenario.seed + 9973)  # same stream the pipeline uses
    captures = synthesize_captures(scenario, rng)
    write_recorded_dataset(directory, sim_feed, captures, scenario, wav_dtype=dtype)
    return sim_feed, captures


# --------------------------------------------------------------------------- #
# 1. the feed is a DeviceFeed and round-trips ranging / anchor_gps from meta
# --------------------------------------------------------------------------- #
def test_recorded_feed_is_a_device_feed(tmp_path):
    scenario = load_scenario(SCN / SCENARIO_FILE)
    _build_dataset(tmp_path, scenario)

    feed = RecordedAudioFeed(tmp_path)
    from dronetracking.sources.base import DeviceFeed

    assert isinstance(feed, DeviceFeed)


def test_recorded_feed_passes_through_ranging_anchor_and_timebase(tmp_path):
    scenario = load_scenario(SCN / SCENARIO_FILE)
    sim_feed, _ = _build_dataset(tmp_path, scenario)
    ref = sim_feed.as_observations()

    feed = RecordedAudioFeed(tmp_path)
    obs = feed.as_observations()

    assert isinstance(obs, Observations)
    # Device order preserved from meta (fixes downstream matrix row order).
    assert obs.device_ids == ref.device_ids
    # Timebase constants survive the round-trip through meta.
    assert obs.speed_of_sound_mps == ref.speed_of_sound_mps
    assert obs.sample_rate_hz == ref.sample_rate_hz

    # Ranging + anchor sets are passed through verbatim (order-insensitive, exact).
    assert Counter(obs.ranging) == Counter(ref.ranging)
    assert Counter(obs.anchor_gps) == Counter(ref.anchor_gps)
    assert all(isinstance(r, RangingRecord) for r in obs.ranging)
    assert all(isinstance(g, AnchorGps) for g in obs.anchor_gps)


# --------------------------------------------------------------------------- #
# 2. acoustic arrivals are recovered from the WAVs near the true arrivals
# --------------------------------------------------------------------------- #
def test_recorded_feed_recovers_acoustic_arrivals_near_truth(tmp_path):
    scenario = load_scenario(SCN / SCENARIO_FILE)
    sim_feed, _ = _build_dataset(tmp_path, scenario)
    ref = sim_feed.as_observations()
    fs = ref.sample_rate_hz

    feed = RecordedAudioFeed(tmp_path)
    obs = feed.as_observations()

    assert all(isinstance(a, AcousticArrival) for a in obs.acoustic)
    # One arrival per (device, emission): same count the simulator produced.
    assert len(obs.acoustic) == len(ref.acoustic)

    # Index detections and truth by (device_id, emission_idx) and compare TOAs.
    det = {(a.device_id, a.emission_idx): a.toa_local_s for a in obs.acoustic}
    truth = {(a.device_id, a.emission_idx): a.toa_local_s for a in ref.acoustic}
    assert set(det) == set(truth), "detected (device, emission) keys must match truth"

    # Detected arrival times must land within a few samples of the true local arrival.
    tol_s = 3.0 / fs  # within 3 sample periods
    errs = np.array([abs(det[k] - truth[k]) for k in truth])
    assert errs.max() <= tol_s, (
        f"max arrival error {errs.max()*1e3:.3f} ms exceeds {tol_s*1e3:.3f} ms "
        f"(median {np.median(errs)*1e3:.3f} ms)"
    )


# --------------------------------------------------------------------------- #
# 3. the assembled feed drives the same downstream contract as a sim feed
# --------------------------------------------------------------------------- #
def test_recorded_feed_drives_the_pipeline(tmp_path):
    # End-to-end smoke: a recorded feed is a drop-in for run_pipeline (no ground truth,
    # so metrics are skipped, but estimates + a track must come out).
    from dronetracking.pipeline import run_pipeline

    scenario = load_scenario(SCN / SCENARIO_FILE)
    _build_dataset(tmp_path, scenario)

    feed = RecordedAudioFeed(tmp_path)
    result = run_pipeline(scenario, feed=feed)

    assert result.tracks, "pipeline produced no track from the recorded feed"
    # A real feed carries no ground truth, so the pipeline skips metrics.
    assert result.world is None
    # The georeferenced track has points (anchors came through meta).
    assert len(result.estimates.geo_track.latlon) > 0


# --------------------------------------------------------------------------- #
# 4. robustness: float64 WAVs and a device order taken straight from meta
# --------------------------------------------------------------------------- #
def test_recorded_feed_reads_float64_wavs(tmp_path):
    scenario = load_scenario(SCN / SCENARIO_FILE)
    sim_feed, _ = _build_dataset(tmp_path, scenario, dtype=np.float64)
    ref = sim_feed.as_observations()
    fs = ref.sample_rate_hz

    feed = RecordedAudioFeed(tmp_path)
    obs = feed.as_observations()

    det = {(a.device_id, a.emission_idx): a.toa_local_s for a in obs.acoustic}
    truth = {(a.device_id, a.emission_idx): a.toa_local_s for a in ref.acoustic}
    assert set(det) == set(truth)
    errs = np.array([abs(det[k] - truth[k]) for k in truth])
    assert errs.max() <= 3.0 / fs


def test_recorded_feed_missing_meta_is_a_clear_error(tmp_path):
    # Pointing the feed at an empty directory must fail loudly, not silently.
    with pytest.raises(FileNotFoundError):
        RecordedAudioFeed(tmp_path).as_observations()
