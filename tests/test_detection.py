"""Ph4 — acoustic detection / DSP tests (TDD).

The simulator (``sim.audio``) renders, per device, a full-duration waveform with a
KNOWN pulse placed at each emission's true local-clock arrival time, plus rotor
background + noise at a configurable SNR. The detector (``estimation.detection``)
runs a matched filter against the reference pulse, peak-picks per emission window,
and recovers the per-device local arrival times.

Ground truth for the arrival times comes from the *existing* acoustic generator
(:func:`sim.acoustic.generate_acoustic_arrivals`) on the SAME scenario, so the two
code paths must agree to within a few samples at high SNR.

Note the firewall: ``estimation.detection`` may NOT import ``sim``. It therefore
returns its own lightweight :class:`DetectedArrival` records; the orchestrator maps
those onto :class:`sim.observations.AcousticArrival` for the TDOA flow. Tests MAY
import sim, and do so to build ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from dronetracking.sim.scenario import Scenario, TrajectorySpec, NoiseSpec, DeviceSpec
from dronetracking.sim.acoustic import generate_acoustic_arrivals, emission_times
from dronetracking.sim import audio as sim_audio
from dronetracking.sim.audio import AudioCapture, synthesize_captures, reference_pulse
from dronetracking.estimation.detection import DetectedArrival, detect_arrivals

C = 343.0


def _scn(snr_db=40.0, dt=0.5, duration=2.0, toa_std=0.0, audio_extra=None):
    """A small 4-device scenario; few emissions for speed.

    Devices placed asymmetrically so per-device ranges (hence arrival times) differ.
    A modest sample rate keeps the rendered waveforms small but still resolves the
    pulse to within a couple of samples.
    """
    audio = {"snr_db": snr_db, "pulse": "chirp", "f0": 1500.0, "f1": 4500.0,
             "pulse_dur_s": 0.01}
    if audio_extra:
        audio.update(audio_extra)
    return Scenario(
        name="det", seed=0, speed_of_sound_mps=C, sample_rate_hz=16000.0,
        duration_s=duration, dt_s=dt, ranging_rounds=4, origin_latlon=(32.0, 34.0),
        devices=(
            DeviceSpec("a", (0.0, 0.0, 0.0)),
            DeviceSpec("b", (120.0, 0.0, 0.0)),
            DeviceSpec("c", (0.0, 140.0, 0.0)),
            DeviceSpec("d", (90.0, 90.0, 10.0)),
        ),
        trajectory=TrajectorySpec("linear", {"start_m": [-40, 40], "end_m": [160, 90]}, z_m=55.0),
        noise=NoiseSpec(toa_std_s=toa_std),
        audio=audio,
    )


def _truth_by_device(scenario):
    """{device_id: {emission_idx: true_toa_local_s}} from the existing generator."""
    out = {}
    for arr in generate_acoustic_arrivals(scenario, np.random.default_rng(scenario.seed)):
        out.setdefault(arr.device_id, {})[arr.emission_idx] = arr.toa_local_s
    return out


# ----------------------------------------------------------------------------
# sim.audio: reference pulse + capture synthesis
# ----------------------------------------------------------------------------

def test_reference_pulse_shape_and_normalization():
    sc = _scn()
    ref = reference_pulse(sc)
    assert isinstance(ref, np.ndarray)
    assert ref.ndim == 1
    # ~pulse_dur_s * fs samples
    expected = int(round(sc.audio["pulse_dur_s"] * sc.sample_rate_hz))
    assert abs(ref.size - expected) <= 1
    assert ref.size > 8
    assert np.isfinite(ref).all()
    # Non-trivial, finite-energy, zero-mean-ish AC waveform (a chirp).
    assert np.linalg.norm(ref) > 0.0


def test_reference_pulse_is_deterministic():
    sc = _scn()
    assert np.array_equal(reference_pulse(sc), reference_pulse(sc))


def test_synthesize_captures_structure():
    sc = _scn()
    caps = synthesize_captures(sc, np.random.default_rng(1))
    assert set(caps) == set(sc.device_ids)
    n_expected = int(round(sc.duration_s * sc.sample_rate_hz))
    for dev_id, cap in caps.items():
        assert isinstance(cap, AudioCapture)
        assert cap.device_id == dev_id
        assert cap.sample_rate_hz == sc.sample_rate_hz
        # Full-duration waveform.
        assert abs(cap.samples.size - n_expected) <= 1
        assert cap.samples.ndim == 1
        assert np.isfinite(cap.samples).all()


def test_synthesize_captures_is_reproducible_given_rng():
    sc = _scn()
    a = synthesize_captures(sc, np.random.default_rng(7))
    b = synthesize_captures(sc, np.random.default_rng(7))
    for k in a:
        assert np.array_equal(a[k].samples, b[k].samples)


def test_higher_snr_gives_larger_pulse_to_noise_ratio():
    """A high-SNR capture should have visibly larger peak-to-background contrast.

    Sanity that the SNR knob actually scales the noise floor relative to the pulse.
    """
    sc_hi = _scn(snr_db=40.0)
    sc_lo = _scn(snr_db=-5.0)
    ref = reference_pulse(sc_hi)
    cap_hi = synthesize_captures(sc_hi, np.random.default_rng(3))["a"]
    cap_lo = synthesize_captures(sc_lo, np.random.default_rng(3))["a"]
    from scipy.signal import fftconvolve

    def contrast(samples):
        mf = fftconvolve(samples, ref[::-1], mode="full")
        env = np.abs(mf)
        return env.max() / (np.median(env) + 1e-12)

    assert contrast(cap_hi.samples) > contrast(cap_lo.samples)


# ----------------------------------------------------------------------------
# estimation.detection: matched filter recovers arrivals
# ----------------------------------------------------------------------------

def test_detected_arrival_is_firewall_clean_dataclass():
    # DetectedArrival is detection.py's OWN type (no sim import in detection.py).
    da = DetectedArrival(device_id="a", emission_idx=0, toa_local_s=1.0, confidence=0.9)
    assert da.device_id == "a"
    assert da.emission_idx == 0
    assert da.toa_local_s == 1.0
    assert 0.0 <= da.confidence <= 1.0


def test_high_snr_recovers_true_arrivals_within_a_few_samples():
    sc = _scn(snr_db=40.0)
    fs = sc.sample_rate_hz
    n_emissions = len(emission_times(sc))
    ref = reference_pulse(sc)
    caps = synthesize_captures(sc, np.random.default_rng(11))

    detected = detect_arrivals(caps, ref, n_emissions=n_emissions, dt_s=sc.dt_s)
    assert len(detected) == n_emissions * len(sc.device_ids)

    by_dev = {}
    for d in detected:
        by_dev.setdefault(d.device_id, {})[d.emission_idx] = d
    truth = _truth_by_device(sc)

    tol_s = 3.0 / fs  # within a few samples
    for dev_id in sc.device_ids:
        for k in range(n_emissions):
            assert k in by_dev[dev_id], f"missing detection for {dev_id} emission {k}"
            err = abs(by_dev[dev_id][k].toa_local_s - truth[dev_id][k])
            assert err <= tol_s, f"{dev_id}/{k}: err {err*fs:.2f} samples"
            assert by_dev[dev_id][k].confidence > 0.6  # confident at high SNR


def test_high_snr_tdoa_differences_match_truth():
    """The detector's per-emission inter-device differences (what TDOA consumes)
    must match the truth differences to within a couple of samples."""
    sc = _scn(snr_db=40.0)
    fs = sc.sample_rate_hz
    n_emissions = len(emission_times(sc))
    ref = reference_pulse(sc)
    caps = synthesize_captures(sc, np.random.default_rng(5))
    detected = detect_arrivals(caps, ref, n_emissions=n_emissions, dt_s=sc.dt_s)

    det = {}
    for d in detected:
        det.setdefault(d.emission_idx, {})[d.device_id] = d.toa_local_s
    truth = _truth_by_device(sc)

    for k in range(n_emissions):
        d_ab = det[k]["a"] - det[k]["b"]
        t_ab = truth["a"][k] - truth["b"][k]
        assert abs(d_ab - t_ab) <= 4.0 / fs


def test_low_snr_degrades_gracefully():
    """At low SNR detection still returns one record per (device, emission) and the
    confidence drops relative to high SNR; high-confidence detections stay bounded."""
    sc_hi = _scn(snr_db=40.0)
    sc_lo = _scn(snr_db=-6.0)
    fs = sc_lo.sample_rate_hz
    n_emissions = len(emission_times(sc_lo))
    ref = reference_pulse(sc_lo)

    det_hi = detect_arrivals(
        synthesize_captures(sc_hi, np.random.default_rng(2)),
        ref, n_emissions=n_emissions, dt_s=sc_hi.dt_s)
    det_lo = detect_arrivals(
        synthesize_captures(sc_lo, np.random.default_rng(2)),
        ref, n_emissions=n_emissions, dt_s=sc_lo.dt_s)

    # Same structural completeness regardless of SNR.
    assert len(det_lo) == n_emissions * len(sc_lo.device_ids)
    # Confidence in [0,1].
    assert all(0.0 <= d.confidence <= 1.0 for d in det_lo)

    mean_hi = float(np.mean([d.confidence for d in det_hi]))
    mean_lo = float(np.mean([d.confidence for d in det_lo]))
    assert mean_lo < mean_hi  # graceful degradation: noisier -> less confident

    # The confident low-SNR detections are still roughly right (bounded error):
    # for the detections the detector is most sure of, error stays within a window.
    truth = _truth_by_device(sc_lo)
    good = [d for d in det_lo if d.confidence >= np.median([x.confidence for x in det_lo])]
    assert good  # at least some
    win_s = 3.0 * sc_lo.dt_s
    errs = [abs(d.toa_local_s - truth[d.device_id][d.emission_idx]) for d in good]
    assert np.median(errs) <= win_s


def test_detect_arrivals_assigns_each_emission_window():
    """Emission indices returned are exactly 0..n_emissions-1 for every device."""
    sc = _scn(snr_db=30.0)
    n_emissions = len(emission_times(sc))
    ref = reference_pulse(sc)
    caps = synthesize_captures(sc, np.random.default_rng(9))
    detected = detect_arrivals(caps, ref, n_emissions=n_emissions, dt_s=sc.dt_s)
    for dev_id in sc.device_ids:
        idxs = sorted(d.emission_idx for d in detected if d.device_id == dev_id)
        assert idxs == list(range(n_emissions))
