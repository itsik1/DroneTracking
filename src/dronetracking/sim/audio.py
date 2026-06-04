"""Render full-duration acoustic waveforms per device (Ph4 acoustic synthesis).

This is the *signal-level* counterpart to :mod:`sim.acoustic`. Where ``sim.acoustic``
emits only the per-device local arrival *times* (the abstract measurement TDOA needs),
this module produces the raw audio a microphone would actually record: a KNOWN pulse
(a linear chirp by default) emitted once per ``dt_s``, copied into each device's track
delayed to that device's true LOCAL-clock arrival time, buried in rotor-harmonic
background plus Gaussian noise at a configurable SNR.

The arrival physics is *identical* to :func:`sim.acoustic.generate_acoustic_arrivals`
so the detector (:mod:`estimation.detection`) can be validated against that generator:

    global_arrival = emit_time + range / c
    local_arrival  = device_local_time(offset, drift, global_arrival)
    sample_index   = round((local_arrival - t0_local) * fs)

``scenario.audio`` configures the synthesis (all keys optional, defaults below):

    snr_db              pulse SNR vs. (rotor background + noise), in dB        [10.0]
    pulse               "chirp" (linear FM) or "tone" (windowed sinusoid)     ["chirp"]
    f0                  chirp start / tone frequency, Hz                       [2000.0]
    f1                  chirp end frequency, Hz (chirp only)                   [6000.0]
    pulse_dur_s         pulse length, seconds                                  [0.012]
    rotor_fundamental_hz  rotor blade-pass fundamental for the background, Hz  [90.0]
    rotor_harmonics     number of rotor harmonics in the background           [4]
    rotor_level         background amplitude relative to the pulse (0 -> none) [0.5]

Both ``synthesize_captures`` and ``reference_pulse`` use ``scenario.sample_rate_hz``.
The pulse's reference instant is its FIRST sample, which is also the convention the
matched-filter detector inverts; keep the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .clocks import device_local_time
from .scenario import Scenario
from .trajectory import trajectory_position

# Defaults for every scenario.audio key, so a scenario need not specify any.
_AUDIO_DEFAULTS = {
    "snr_db": 10.0,
    "pulse": "chirp",
    "f0": 2000.0,
    "f1": 6000.0,
    "pulse_dur_s": 0.012,
    "rotor_fundamental_hz": 90.0,
    "rotor_harmonics": 4,
    "rotor_level": 0.5,
}


@dataclass
class AudioCapture:
    """One device's recorded waveform over the whole scenario.

    ``samples[n]`` is the signal at local-clock time ``t0_local_s + n / sample_rate_hz``.
    """

    device_id: str
    samples: np.ndarray
    sample_rate_hz: float
    t0_local_s: float


def _audio_params(scenario: Scenario) -> Dict:
    """scenario.audio merged over the defaults (scenario values win)."""
    p = dict(_AUDIO_DEFAULTS)
    p.update(scenario.audio or {})
    return p


def reference_pulse(scenario: Scenario) -> np.ndarray:
    """The KNOWN drone pulse template, sampled at ``scenario.sample_rate_hz``.

    A Hann-windowed linear chirp (default) or windowed tone, unit-energy-normalised so
    the matched filter's response scale is independent of pulse length. Deterministic
    (no RNG): the detector is handed this exact array as its template.
    """
    p = _audio_params(scenario)
    fs = float(scenario.sample_rate_hz)
    n = int(round(float(p["pulse_dur_s"]) * fs))
    n = max(n, 2)
    t = np.arange(n) / fs
    dur = n / fs

    f0 = float(p["f0"])
    if p["pulse"] == "tone":
        phase = 2.0 * np.pi * f0 * t
    else:  # linear chirp f0 -> f1
        f1 = float(p["f1"])
        k = (f1 - f0) / dur  # sweep rate (Hz/s)
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)

    window = np.hanning(n)
    pulse = window * np.sin(phase)

    norm = np.linalg.norm(pulse)
    if norm > 0.0:
        pulse = pulse / norm
    return pulse


def synthesize_captures(scenario: Scenario, rng: np.random.Generator) -> Dict[str, AudioCapture]:
    """Render a full-duration waveform per device with the pulse train + background.

    For every emission ``k`` at global time ``t_k`` (``sim.acoustic.emission_times``),
    the true drone position gives each device a range, hence a global arrival
    ``t_k + range/c``, stamped into the device's local clock. The known pulse is added
    starting at that local-clock sample index. Rotor-harmonic background and Gaussian
    noise are then mixed in at the configured SNR (pulse power vs. background+noise).

    All device tracks share ``t0_local_s = 0.0`` (local clock at global t=0), so a
    detector can recover the local arrival time directly from a peak's sample index.
    """
    p = _audio_params(scenario)
    fs = float(scenario.sample_rate_hz)
    c = float(scenario.speed_of_sound_mps)
    n_samples = int(round(scenario.duration_s * fs))
    n_samples = max(n_samples, 1)

    pulse = reference_pulse(scenario)
    m = pulse.size
    pulse_power = float(np.mean(pulse ** 2)) if m else 0.0

    snr_db = float(p["snr_db"])
    rotor_level = float(p["rotor_level"])
    f_rotor = float(p["rotor_fundamental_hz"])
    n_harm = int(p["rotor_harmonics"])

    emit_times = np.arange(0.0, scenario.duration_s, scenario.dt_s)
    t = np.arange(n_samples) / fs  # local-clock time axis (t0_local = 0)

    captures: Dict[str, AudioCapture] = {}
    for d in scenario.devices:
        pos = np.asarray(d.position_m, dtype=float)
        clean = np.zeros(n_samples, dtype=float)

        # Place the known pulse at each emission's true LOCAL arrival time.
        for t_k in emit_times:
            drone = trajectory_position(scenario, float(t_k))
            rng_m = float(np.linalg.norm(drone - pos))
            g_arrival = float(t_k) + rng_m / c
            local_arrival = float(device_local_time(d.clock_offset_s, d.clock_drift_ppm, g_arrival))
            start = int(round(local_arrival * fs))  # t0_local = 0
            if start >= n_samples or start + m <= 0:
                continue
            lo = max(start, 0)
            hi = min(start + m, n_samples)
            clean[lo:hi] += pulse[lo - start: hi - start]

        # Rotor-harmonic background (deterministic tonal hum, random phase per device).
        background = np.zeros(n_samples, dtype=float)
        if rotor_level > 0.0 and n_harm > 0 and f_rotor > 0.0:
            for h in range(1, n_harm + 1):
                phi = rng.uniform(0.0, 2.0 * np.pi)
                background += (1.0 / h) * np.sin(2.0 * np.pi * f_rotor * h * t + phi)
            bg_power = float(np.mean(background ** 2))
            if bg_power > 0.0:
                # Scale background so its power is `rotor_level^2` of the pulse power.
                target = (rotor_level ** 2) * pulse_power
                background *= np.sqrt(target / bg_power)

        # Gaussian noise set by SNR: pulse_power / total_noise_power = 10^(snr/10),
        # where total noise = rotor background + white noise.
        signal_to_noise = 10.0 ** (snr_db / 10.0)
        total_noise_power = pulse_power / signal_to_noise if pulse_power > 0.0 else 0.0
        bg_power = float(np.mean(background ** 2))
        white_power = max(total_noise_power - bg_power, 0.0)
        noise = rng.normal(0.0, np.sqrt(white_power), n_samples) if white_power > 0.0 else 0.0

        samples = clean + background + noise
        captures[d.id] = AudioCapture(
            device_id=d.id,
            samples=samples,
            sample_rate_hz=fs,
            t0_local_s=0.0,
        )
    return captures
