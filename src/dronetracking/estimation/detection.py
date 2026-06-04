"""Acoustic pulse detection via matched filtering (Ph4 detection stage).

Recovers, from raw per-device waveforms, the local-clock arrival time of each drone
emission. A drone emits a KNOWN pulse once per ``dt_s``; correlating a capture with
that reference pulse (a matched filter — optimal for a known signal in white noise)
produces a sharp peak at each arrival. We peak-pick the strongest, well-separated
peaks, order them in time (a device's own arrivals are monotone and ~``dt_s`` apart),
and read the local arrival time straight off each peak's sample index. Confidence is
the peak's prominence relative to the local noise floor, squashed to ``[0, 1]``.

GROUND-TRUTH FIREWALL: this module is under ``estimation`` and therefore must NOT
import ``dronetracking.sim`` (enforced by ``tests/test_no_truth_leak.py``). It cannot
import :class:`sim.observations.AcousticArrival`. Instead it returns its OWN lightweight
:class:`DetectedArrival` records, which carry exactly the fields TDOA needs
(``device_id``, ``emission_idx``, ``toa_local_s``, ``confidence``). The orchestrator
maps each ``DetectedArrival`` onto an ``AcousticArrival`` before feeding the TDOA flow:

    AcousticArrival(device_id=d.device_id, emission_idx=d.emission_idx,
                    toa_local_s=d.toa_local_s, confidence=d.confidence)

The two dataclasses are field-compatible by design, so the mapping is a plain copy.

Index -> time convention (must match :mod:`sim.audio`): a pulse of length ``m`` copied
into the capture starting at sample ``start`` produces a full-correlation peak at
``start + (m - 1)``. So ``start = peak_full - (m - 1)`` and, with the capture's
``t0_local_s`` as the time of sample 0, ``toa_local = t0_local + start / fs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, fftconvolve, find_peaks, sosfiltfilt

# Confidence scale: a chosen peak's deflection above the off-peak noise floor, measured
# in standard deviations of that floor, is mapped through 1 - exp(-z / _CONF_TAU_SIGMAS).
# A clean (noise-free-ish) peak sits thousands of sigmas up -> confidence ~1; a peak only
# a handful of sigmas above a high noise floor -> a middling confidence that falls with SNR.
_CONF_TAU_SIGMAS = 8.0


@dataclass(frozen=True)
class DetectedArrival:
    """A detected drone-emission arrival at one device (firewall-clean analogue of
    :class:`sim.observations.AcousticArrival`).

    ``toa_local_s`` is in the recording device's own clock; ``confidence`` is in
    ``[0, 1]`` (1 = a clean, unambiguous peak; near 0 = barely above the noise floor).
    """

    device_id: str
    emission_idx: int
    toa_local_s: float
    confidence: float = 1.0


def _bandpass(samples: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase band-pass to suppress out-of-band rotor hum before matched filtering.

    Best-effort: if the band is degenerate for this sample rate, return the input
    unchanged (the matched filter alone already rejects most out-of-band energy).
    """
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.999)
    if not (0.0 < lo_n < hi_n < 1.0):
        return samples
    try:
        sos = butter(4, [lo_n, hi_n], btype="band", output="sos")
        return sosfiltfilt(sos, samples)
    except (ValueError, RuntimeError):
        return samples


def _passband_from_reference(reference_pulse: np.ndarray, fs: float) -> Tuple[float, float]:
    """Estimate the reference pulse's occupied band (for a guard band-pass) via its FFT.

    Returns ``(lo_hz, hi_hz)`` covering the bins above 10% of peak spectral magnitude,
    padded by half an octave. Falls back to a wide band on any degeneracy.
    """
    n = reference_pulse.size
    if n < 4:
        return 0.0, 0.5 * fs
    spec = np.abs(np.fft.rfft(reference_pulse))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if spec.max() <= 0.0:
        return 0.0, 0.5 * fs
    mask = spec >= 0.1 * spec.max()
    band = freqs[mask]
    if band.size == 0:
        return 0.0, 0.5 * fs
    lo = max(band.min() * 0.7, 0.0)
    hi = min(band.max() * 1.4, 0.5 * fs)
    return float(lo), float(hi)


def _matched_filter_envelope(samples: np.ndarray, reference_pulse: np.ndarray) -> np.ndarray:
    """Full-mode matched-filter response magnitude (correlation with the template)."""
    mf = fftconvolve(samples, reference_pulse[::-1], mode="full")
    return np.abs(mf)


def _detect_one(
    samples: np.ndarray,
    reference_pulse: np.ndarray,
    fs: float,
    t0_local_s: float,
    n_emissions: int,
    dt_s: float,
):
    """Detect ``n_emissions`` arrivals in one capture -> list of (toa_local_s, confidence).

    Peaks are picked from the matched-filter envelope, ordered in time, and converted
    to local arrival times. Always returns exactly ``n_emissions`` entries (a per-window
    argmax fallback fills any gap so downstream framing stays complete even at low SNR).
    """
    m = reference_pulse.size
    lo, hi = _passband_from_reference(reference_pulse, fs)
    filtered = _bandpass(samples, fs, lo, hi)
    env = _matched_filter_envelope(filtered, reference_pulse)

    # A device's own consecutive arrivals are >= ~dt_s apart; require at least half
    # that separation so two close peaks aren't both grabbed for the same emission.
    min_dist = max(int(round(0.5 * dt_s * fs)), 1)

    peak_idx, _ = find_peaks(env, distance=min_dist)
    if peak_idx.size:
        order = np.argsort(env[peak_idx])[::-1]
        chosen = peak_idx[order[:n_emissions]]
    else:
        chosen = np.array([], dtype=int)

    # Fallback: if we found fewer distinct peaks than emissions (very low SNR), split
    # the envelope into n_emissions equal windows and take each window's argmax.
    if chosen.size < n_emissions:
        edges = np.linspace(0, env.size, n_emissions + 1).astype(int)
        chosen = np.array([lo_i + int(np.argmax(env[lo_i:hi_i]))
                           for lo_i, hi_i in zip(edges[:-1], edges[1:]) if hi_i > lo_i])

    chosen = np.sort(np.unique(chosen))[:n_emissions]

    # Noise floor from the envelope OUTSIDE the chosen peaks: confidence is each peak's
    # deflection above that floor, in floor-sigmas. (Peak prominence saturates because a
    # matched filter yields a tall peak even in noise; what actually tracks SNR is how
    # far the peak sits above the surrounding noise level.)
    off_peak = np.ones(env.size, dtype=bool)
    for c in chosen:
        off_peak[max(0, int(c) - min_dist): int(c) + min_dist] = False
    bg = env[off_peak]
    if bg.size:
        base = float(np.median(bg))
        scale = float(bg.std()) + 1e-12
    else:  # degenerate (everything masked): fall back to whole-envelope stats
        base = float(np.median(env))
        scale = float(env.std()) + 1e-12

    results = []
    for peak_full in chosen:
        start = int(peak_full) - (m - 1)  # invert the full-correlation lag
        toa_local = t0_local_s + start / fs
        z = (env[int(peak_full)] - base) / scale  # peak deflection, in noise sigmas
        confidence = float(np.clip(1.0 - np.exp(-max(z, 0.0) / _CONF_TAU_SIGMAS), 0.0, 1.0))
        results.append((float(toa_local), confidence))
    return results


def detect_arrivals(
    captures: Dict[str, "object"],
    reference_pulse: np.ndarray,
    n_emissions: int,
    dt_s: float,
) -> Tuple[DetectedArrival, ...]:
    """Detect per-device emission arrivals from raw captures via matched filtering.

    Parameters
    ----------
    captures:
        ``{device_id: capture}`` where each capture has ``samples`` (1-D ``np.ndarray``),
        ``sample_rate_hz`` and ``t0_local_s`` (typically :class:`sim.audio.AudioCapture`,
        but any duck-typed object with those attributes works — keeping this module free
        of any ``sim`` import).
    reference_pulse:
        The known pulse template (e.g. :func:`sim.audio.reference_pulse`).
    n_emissions:
        Number of drone emissions to recover per device.
    dt_s:
        Nominal seconds between emissions (sets the peak-separation guard).

    Returns
    -------
    A tuple of :class:`DetectedArrival`, ``n_emissions`` per device. ``emission_idx``
    is assigned in time order (a device's arrivals are monotone in its local clock),
    so detections for the same emission share an index across devices for TDOA framing.
    """
    reference_pulse = np.asarray(reference_pulse, dtype=float)
    arrivals = []
    for device_id in sorted(captures):
        cap = captures[device_id]
        samples = np.asarray(cap.samples, dtype=float)
        fs = float(cap.sample_rate_hz)
        t0 = float(cap.t0_local_s)
        detections = _detect_one(samples, reference_pulse, fs, t0, n_emissions, dt_s)
        for k, (toa_local, confidence) in enumerate(detections):
            arrivals.append(
                DetectedArrival(
                    device_id=device_id,
                    emission_idx=k,
                    toa_local_s=toa_local,
                    confidence=confidence,
                )
            )
    return tuple(arrivals)
