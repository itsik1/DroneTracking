"""Ph6 — overlapping-source acoustic separation tests (TDD).

When several drones emit at once, each device's microphone records a MIX of their
signatures. If every drone uses a DISTINCT pulse (here: a chirp in a different
frequency band), a matched-filter BANK separates them: correlating a capture against
source ``k``'s reference pulse responds to source ``k`` and largely rejects the others
(near-orthogonal bands give near-zero cross-correlation).

These tests synthesize, per device, a mixed capture = sum of TWO distinct chirp
pulse-trains (source A low band, source B high band) placed at different known local
arrival times, plus mild Gaussian noise. The chirps are built here with
``scipy.signal.chirp`` (the contract permits building them in the TEST). We then run
``separate_arrivals`` with both reference pulses and assert:

  (a) each source's recovered arrival times match THAT source's true arrivals to
      within a few samples, and
  (b) low cross-talk — source A's matched filter does not report source B's pulse
      times (and vice versa).

Index -> time convention (matches :mod:`sim.audio` / :mod:`estimation.detection`):
a length-``m`` pulse copied into the capture starting at sample ``start`` produces a
full-correlation peak at ``start + (m - 1)``; the detector inverts that to
``toa_local = t0_local + start / fs``. With ``t0_local = 0`` here, a pulse placed at
sample ``start`` is recovered at ``toa = start / fs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.signal import chirp

from dronetracking.estimation.separation import (
    SeparatedArrival,
    separate_arrivals,
    to_acoustic_arrivals,
)

FS = 16000.0  # modest sample rate keeps the rendered waveforms small but resolved
C = 343.0


@dataclass
class _Capture:
    """Minimal duck-typed capture (the detector only needs these three attrs).

    Intentionally NOT ``sim.audio.AudioCapture`` so the test exercises the same
    structural contract the firewall-clean detector relies on.
    """

    device_id: str
    samples: np.ndarray
    sample_rate_hz: float
    t0_local_s: float


def _chirp_pulse(f0: float, f1: float, dur_s: float = 0.01, fs: float = FS) -> np.ndarray:
    """A Hann-windowed, unit-energy linear chirp from ``f0`` to ``f1`` Hz.

    Unit-energy normalisation mirrors :func:`sim.audio.reference_pulse` so the
    matched-filter response scale is comparable across the two reference pulses.
    """
    n = max(int(round(dur_s * fs)), 2)
    t = np.arange(n) / fs
    pulse = np.hanning(n) * chirp(t, f0=f0, f1=f1, t1=t[-1] if n > 1 else dur_s, method="linear")
    norm = np.linalg.norm(pulse)
    if norm > 0.0:
        pulse = pulse / norm
    return pulse


def _place(samples: np.ndarray, pulse: np.ndarray, start: int) -> None:
    """Add ``pulse`` into ``samples`` starting at sample ``start`` (clipped to bounds)."""
    m = pulse.size
    if start >= samples.size or start + m <= 0:
        return
    lo = max(start, 0)
    hi = min(start + m, samples.size)
    samples[lo:hi] += pulse[lo - start: hi - start]


# Two well-separated bands -> near-orthogonal templates -> low cross-talk.
REF_A = _chirp_pulse(f0=1500.0, f1=3000.0)
REF_B = _chirp_pulse(f0=5500.0, f1=7500.0)
REFS = {"A": REF_A, "B": REF_B}


def _mixed_scene(seed: int = 0, snr_db: float = 30.0):
    """Build mixed two-source captures over a few devices.

    Returns ``(captures, true_starts, n_emissions, dt_s)`` where
    ``true_starts[source][device_id]`` is the list of true pulse start samples for
    that source at that device, in emission order. Source A and source B occupy
    DIFFERENT, device-dependent arrival times (as if two drones at two locations).
    """
    rng = np.random.default_rng(seed)
    fs = FS
    dt_s = 0.5
    n_emissions = 3
    duration_s = (n_emissions + 0.5) * dt_s
    n_samples = int(round(duration_s * fs))

    device_ids = ["d0", "d1", "d2", "d3"]
    # Per-source, per-device range-induced delays (seconds). Distinct geometry per
    # source so A and B land at different times (and differ across devices for TDOA).
    delay_a = {"d0": 0.012, "d1": 0.027, "d2": 0.041, "d3": 0.018}
    delay_b = {"d0": 0.033, "d1": 0.009, "d2": 0.022, "d3": 0.047}

    pulse_power = float(np.mean(REF_A ** 2))
    snr = 10.0 ** (snr_db / 10.0)
    noise_std = float(np.sqrt(pulse_power / snr))

    captures = {}
    true_starts = {"A": {}, "B": {}}
    for dev in device_ids:
        s = np.zeros(n_samples, dtype=float)
        starts_a, starts_b = [], []
        for k in range(n_emissions):
            t_emit = 0.05 + k * dt_s  # small lead-in so k=0 isn't at sample 0
            sa = int(round((t_emit + delay_a[dev]) * fs))
            sb = int(round((t_emit + delay_b[dev]) * fs))
            _place(s, REF_A, sa)
            _place(s, REF_B, sb)
            starts_a.append(sa)
            starts_b.append(sb)
        s += rng.normal(0.0, noise_std, n_samples)
        captures[dev] = _Capture(dev, s, fs, 0.0)
        true_starts["A"][dev] = starts_a
        true_starts["B"][dev] = starts_b
    return captures, true_starts, n_emissions, dt_s


# ---------------------------------------------------------------------------
# Light data type
# ---------------------------------------------------------------------------

def test_separated_arrival_carries_source_and_arrival_fields():
    """SeparatedArrival is separation.py's OWN firewall-clean type, tagging a
    detection with its source key while exposing the arrival fields TDOA needs."""
    sa = SeparatedArrival(source="A", device_id="d0", emission_idx=0,
                          toa_local_s=0.5, confidence=0.9)
    assert sa.source == "A"
    assert sa.device_id == "d0"
    assert sa.emission_idx == 0
    assert sa.toa_local_s == 0.5
    assert 0.0 <= sa.confidence <= 1.0


# ---------------------------------------------------------------------------
# Separation accuracy
# ---------------------------------------------------------------------------

def test_separate_returns_one_list_per_source_with_full_framing():
    caps, true_starts, n_emissions, dt_s = _mixed_scene()
    sep = separate_arrivals(caps, REFS, n_emissions=n_emissions, dt_s=dt_s)

    assert set(sep) == set(REFS)  # one entry per source key
    for src in REFS:
        # n_emissions per device, all tagged with this source.
        assert len(sep[src]) == n_emissions * len(caps)
        assert all(isinstance(a, SeparatedArrival) for a in sep[src])
        assert all(a.source == src for a in sep[src])
        # Emission indices are exactly 0..n_emissions-1 for every device.
        for dev in caps:
            idxs = sorted(a.emission_idx for a in sep[src] if a.device_id == dev)
            assert idxs == list(range(n_emissions))


def test_each_source_arrivals_match_that_sources_truth_within_a_few_samples():
    caps, true_starts, n_emissions, dt_s = _mixed_scene(seed=1)
    fs = FS
    sep = separate_arrivals(caps, REFS, n_emissions=n_emissions, dt_s=dt_s)

    tol_samples = 3.0
    for src in ("A", "B"):
        recovered = {}
        for a in sep[src]:
            recovered.setdefault(a.device_id, {})[a.emission_idx] = a.toa_local_s
        for dev in caps:
            for k in range(n_emissions):
                true_toa = true_starts[src][dev][k] / fs
                err_samples = abs(recovered[dev][k] - true_toa) * fs
                assert err_samples <= tol_samples, (
                    f"source {src} {dev}/{k}: err {err_samples:.2f} samples")


def test_tdoa_differences_per_source_match_truth():
    """The inter-device arrival differences (what TDOA actually consumes) match
    truth per source to within a couple of samples."""
    caps, true_starts, n_emissions, dt_s = _mixed_scene(seed=2)
    fs = FS
    sep = separate_arrivals(caps, REFS, n_emissions=n_emissions, dt_s=dt_s)

    for src in ("A", "B"):
        rec = {}
        for a in sep[src]:
            rec.setdefault(a.emission_idx, {})[a.device_id] = a.toa_local_s
        for k in range(n_emissions):
            d_meas = rec[k]["d0"] - rec[k]["d1"]
            d_true = (true_starts[src]["d0"][k] - true_starts[src]["d1"][k]) / fs
            assert abs(d_meas - d_true) <= 4.0 / fs


# ---------------------------------------------------------------------------
# Cross-talk rejection
# ---------------------------------------------------------------------------

def test_low_cross_talk_a_filter_does_not_report_b_times_and_vice_versa():
    """Source A's matched filter must lock onto A's pulses, not B's (and vice versa).

    For each device we check every recovered arrival is far closer to one of THAT
    source's true times than to any of the OTHER source's true times.
    """
    caps, true_starts, n_emissions, dt_s = _mixed_scene(seed=3)
    fs = FS
    sep = separate_arrivals(caps, REFS, n_emissions=n_emissions, dt_s=dt_s)

    near_tol = 3.0 / fs       # an own-source pulse is within a few samples
    cross_guard = 0.004       # ~64 samples @16k: a B pulse is many samples off for A's filter

    for src, other in (("A", "B"), ("B", "A")):
        for a in sep[src]:
            own_times = [s / fs for s in true_starts[src][a.device_id]]
            other_times = [s / fs for s in true_starts[other][a.device_id]]
            d_own = min(abs(a.toa_local_s - t) for t in own_times)
            d_other = min(abs(a.toa_local_s - t) for t in other_times)
            assert d_own <= near_tol, (
                f"{src} {a.device_id}/{a.emission_idx}: not near an own pulse "
                f"({d_own*fs:.1f} samples)")
            # And it is NOT sitting on one of the other source's pulse times.
            assert d_other > cross_guard, (
                f"{src} filter reported a {other} pulse time at {a.device_id}/"
                f"{a.emission_idx} ({d_other*fs:.1f} samples from a {other} pulse)")


def test_cross_correlation_of_reference_bands_is_small():
    """Sanity: the two reference templates are near-orthogonal (distinct bands),
    which is WHY the matched-filter bank separates them with low cross-talk."""
    # Normalised peak cross-correlation between the two unit-energy templates.
    xcorr = np.correlate(REF_A, REF_B, mode="full")
    auto_a = np.correlate(REF_A, REF_A, mode="full").max()
    auto_b = np.correlate(REF_B, REF_B, mode="full").max()
    peak_cross = np.abs(xcorr).max() / np.sqrt(auto_a * auto_b)
    assert peak_cross < 0.3  # weak overlap -> the filters reject each other's pulses


# ---------------------------------------------------------------------------
# Mapping into the multi-target TDOA path
# ---------------------------------------------------------------------------

def test_to_acoustic_arrivals_carries_distinct_source_labels():
    """Flatten the per-source result into AcousticArrival-like records whose
    ``source`` field distinguishes the targets, so multi_target.localize_frames
    groups by (emission_idx, source) into clean single-target fixes."""
    caps, true_starts, n_emissions, dt_s = _mixed_scene(seed=4)
    sep = separate_arrivals(caps, REFS, n_emissions=n_emissions, dt_s=dt_s)
    flat = to_acoustic_arrivals(sep)

    assert isinstance(flat, tuple)
    # One per (source, device, emission).
    assert len(flat) == len(REFS) * len(caps) * n_emissions
    # Every record carries device_id, emission_idx, toa_local_s and a source label.
    for r in flat:
        assert hasattr(r, "device_id") and hasattr(r, "emission_idx")
        assert hasattr(r, "toa_local_s") and hasattr(r, "source")

    # The two sources map to two DISTINCT integer source labels.
    src_labels = {r.source for r in flat}
    assert len(src_labels) == 2

    # Grouping by (emission_idx, source) yields exactly one clean group per
    # (source, emission), each with all devices -> a full TDOA frame per drone.
    groups = {}
    for r in flat:
        groups.setdefault((r.emission_idx, r.source), set()).add(r.device_id)
    assert len(groups) == n_emissions * len(REFS)
    for key, devs in groups.items():
        assert devs == set(caps)


def test_to_acoustic_arrivals_preserves_arrival_times():
    """The flattened records keep each detection's toa_local_s unchanged."""
    caps, _truth, n_emissions, dt_s = _mixed_scene(seed=5)
    sep = separate_arrivals(caps, REFS, n_emissions=n_emissions, dt_s=dt_s)
    flat = to_acoustic_arrivals(sep)

    # to_acoustic_arrivals maps the string source keys to distinct integer labels in
    # sorted-key order ("A" -> 0, "B" -> 1); reproduce that mapping for the lookup.
    label = {key: i for i, key in enumerate(sorted(sep))}
    by_key = {(r.source, r.device_id, r.emission_idx): r.toa_local_s for r in flat}
    for src, arrivals in sep.items():
        for a in arrivals:
            key = (label[src], a.device_id, a.emission_idx)
            assert by_key[key] == pytest.approx(a.toa_local_s)
