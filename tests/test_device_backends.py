"""Iteration 6 (A) — capture-backend tests (TDD).

The device runtime captures through a :class:`~dronetracking.device.backend.CaptureBackend`.
Two implementations live in :mod:`dronetracking.device.backends`:

* :class:`MockBackend` — sim-driven and deterministic. Its :meth:`record` returns this
  device's synthesized waveform with ``t0_local_s = 0.0``, so running the matched-filter
  detector (:func:`dronetracking.estimation.detection.detect_arrivals`) on it must
  recover the SAME per-device local arrival times the existing acoustic generator
  (:func:`dronetracking.sim.acoustic.generate_acoustic_arrivals`) produces for the same
  scenario — to within a few samples. We also check that ``ranging_records()`` only
  carries this device's initiations, ``gps()`` is present for anchors and ``None``
  otherwise, and that it satisfies the :class:`CaptureBackend` interface.
* :class:`SoundDeviceBackend` — a real microphone via the optional ``sounddevice``
  library. ``sounddevice`` is NOT installed in this environment, so we assert that
  constructing the backend raises a clear, actionable error (the guarded import).

Tests MAY import ``sim`` (to build ground truth); the firewall only restricts
``estimation``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dronetracking.config import load_scenario
from dronetracking.device.backend import CaptureBackend
from dronetracking.device.backends import MockBackend, SoundDeviceBackend
from dronetracking.estimation.detection import detect_arrivals
from dronetracking.sim.acoustic import emission_times, generate_acoustic_arrivals
from dronetracking.sim.audio import reference_pulse
from dronetracking.sim.scenario import (
    DeviceSpec,
    NoiseSpec,
    Scenario,
    TrajectorySpec,
)

SCENARIO_PATH = "scenarios/detection_demo.yaml"


# ----------------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    """The detection-tuned demo scenario (dt_s >> range-delay spread, clean SNR)."""
    return load_scenario(SCENARIO_PATH)


def _built_scenario() -> Scenario:
    """A small, very-high-SNR scenario built in-test (the contract allows building one).

    Few emissions and asymmetric device placement keep it fast while still resolving
    each pulse to within a sample or two. Used for the strict few-samples assertion so
    the test does not depend on the on-disk scenario's exact SNR.
    """
    return Scenario(
        name="backend_det",
        seed=3,
        speed_of_sound_mps=343.0,
        sample_rate_hz=16000.0,
        duration_s=2.0,
        dt_s=0.5,
        ranging_rounds=4,
        origin_latlon=(32.0, 34.0),
        devices=(
            DeviceSpec("a", (0.0, 0.0, 0.0), has_gps=True),
            DeviceSpec("b", (120.0, 0.0, 0.0), has_gps=True),
            DeviceSpec("c", (0.0, 140.0, 0.0)),
            DeviceSpec("d", (90.0, 90.0, 10.0)),
        ),
        trajectory=TrajectorySpec(
            "linear", {"start_m": [-40, 40], "end_m": [160, 90]}, z_m=55.0
        ),
        noise=NoiseSpec(toa_std_s=0.0),
        audio={"snr_db": 40.0, "pulse": "chirp", "f0": 1500.0, "f1": 4500.0,
               "pulse_dur_s": 0.01},
    )


def _truth_by_device(scenario: Scenario) -> dict:
    """{device_id: {emission_idx: true_toa_local_s}} from the existing sim generator.

    Seeded from ``scenario.seed`` to match how the simulator (and hence MockBackend)
    drives its acoustic stream's spawned child generator.
    """
    out: dict = {}
    rng = np.random.default_rng(scenario.seed)
    for arr in generate_acoustic_arrivals(scenario, rng):
        out.setdefault(arr.device_id, {})[arr.emission_idx] = arr.toa_local_s
    return out


class _Capture:
    """Minimal duck-typed capture for :func:`detect_arrivals`.

    ``detect_arrivals`` only needs ``samples`` / ``sample_rate_hz`` / ``t0_local_s``;
    we feed it exactly what ``MockBackend.record()`` returns so the test exercises the
    backend's output, not :class:`sim.audio.AudioCapture` directly.
    """

    def __init__(self, samples: np.ndarray, sample_rate_hz: float, t0_local_s: float):
        self.samples = samples
        self.sample_rate_hz = sample_rate_hz
        self.t0_local_s = t0_local_s


def _detect_via_record(backend: MockBackend, scenario: Scenario):
    """Record from the backend and run detection on that audio -> {emission_idx: toa}."""
    fs = backend.sample_rate_hz
    n_emissions = len(emission_times(scenario))
    ref = reference_pulse(scenario)

    samples, t0 = backend.record(scenario.duration_s)
    cap = _Capture(samples, fs, t0)
    detected = detect_arrivals(
        {backend.device_id: cap}, ref, n_emissions=n_emissions, dt_s=scenario.dt_s
    )
    assert len(detected) == n_emissions
    return {d.emission_idx: d.toa_local_s for d in detected}


# ----------------------------------------------------------------------------
# MockBackend: interface + structure
# ----------------------------------------------------------------------------


def test_mock_backend_is_a_capture_backend(scenario):
    backend = MockBackend(scenario, scenario.device_ids[0])
    assert isinstance(backend, CaptureBackend)


def test_mock_backend_basic_properties(scenario):
    dev = scenario.device_ids[0]
    backend = MockBackend(scenario, dev)
    assert backend.device_id == dev
    assert backend.sample_rate_hz == scenario.sample_rate_hz
    # local_time() is a simple, deterministic value.
    assert isinstance(backend.local_time(), float)
    assert backend.local_time() == backend.local_time()


def test_mock_backend_unknown_device_raises(scenario):
    with pytest.raises(ValueError):
        MockBackend(scenario, "not_a_device")


def test_mock_backend_record_shape_and_t0(scenario):
    backend = MockBackend(scenario, scenario.device_ids[0])
    fs = scenario.sample_rate_hz

    samples, t0 = backend.record(scenario.duration_s)
    assert t0 == 0.0  # synthesizer convention: first sample at local t=0
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert np.isfinite(samples).all()
    # Full-duration request -> the whole rendered waveform.
    assert abs(samples.size - int(round(scenario.duration_s * fs))) <= 1

    # A shorter request returns a leading slice.
    half, t0_half = backend.record(scenario.duration_s / 2.0)
    assert t0_half == 0.0
    assert abs(half.size - int(round(scenario.duration_s / 2.0 * fs))) <= 1
    assert half.size < samples.size


def test_mock_backend_is_deterministic(scenario):
    """Two backends for the same scenario (default rng) produce identical audio."""
    a, _ = MockBackend(scenario, scenario.device_ids[0]).record(scenario.duration_s)
    b, _ = MockBackend(scenario, scenario.device_ids[0]).record(scenario.duration_s)
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------
# MockBackend: detection on record() recovers the sim's true arrivals
# ----------------------------------------------------------------------------


def test_mock_backend_record_recovers_true_arrivals_demo_scenario(scenario):
    """Detection on each device's recorded audio matches the sim truth within a few
    samples, on the on-disk detection_demo scenario."""
    fs = scenario.sample_rate_hz
    n_emissions = len(emission_times(scenario))
    truth = _truth_by_device(scenario)
    tol_s = 3.0 / fs  # within a few samples

    for dev in scenario.device_ids:
        backend = MockBackend(scenario, dev)
        recovered = _detect_via_record(backend, scenario)
        for k in range(n_emissions):
            err = abs(recovered[k] - truth[dev][k])
            assert err <= tol_s, f"{dev}/{k}: err {err * fs:.2f} samples"


def test_mock_backend_record_recovers_true_arrivals_built_scenario():
    """Same check on an in-test very-high-SNR scenario (independent of the disk file)."""
    sc = _built_scenario()
    fs = sc.sample_rate_hz
    n_emissions = len(emission_times(sc))
    truth = _truth_by_device(sc)
    tol_s = 3.0 / fs

    for dev in sc.device_ids:
        backend = MockBackend(sc, dev)
        recovered = _detect_via_record(backend, sc)
        for k in range(n_emissions):
            err = abs(recovered[k] - truth[dev][k])
            assert err <= tol_s, f"{dev}/{k}: err {err * fs:.2f} samples"


def test_mock_backend_record_recovers_tdoa_differences(scenario):
    """The inter-device arrival differences (what TDOA actually consumes) recovered
    from two devices' recordings match the truth differences within a couple samples."""
    fs = scenario.sample_rate_hz
    n_emissions = len(emission_times(scenario))
    truth = _truth_by_device(scenario)

    dev_a, dev_b = scenario.device_ids[0], scenario.device_ids[1]
    rec_a = _detect_via_record(MockBackend(scenario, dev_a), scenario)
    rec_b = _detect_via_record(MockBackend(scenario, dev_b), scenario)

    for k in range(n_emissions):
        d_ab = rec_a[k] - rec_b[k]
        t_ab = truth[dev_a][k] - truth[dev_b][k]
        assert abs(d_ab - t_ab) <= 4.0 / fs


# ----------------------------------------------------------------------------
# MockBackend: ranging + gps
# ----------------------------------------------------------------------------


def test_mock_backend_ranging_records_are_this_devices_initiations(scenario):
    """Every ranging record served carries initiator == this device, for ALL devices."""
    for dev in scenario.device_ids:
        backend = MockBackend(scenario, dev)
        records = backend.ranging_records()
        assert isinstance(records, tuple)
        assert all(r.initiator == dev for r in records)


def test_mock_backend_serves_some_ranging_for_an_initiator(scenario):
    """At least one device (an anchor that initiates) has a non-empty ranging set, so the
    initiator==device_id filter is exercised on real records, not just vacuously."""
    backend = MockBackend(scenario, scenario.device_ids[0])
    records = backend.ranging_records()
    assert len(records) > 0
    assert all(r.initiator == scenario.device_ids[0] for r in records)


def test_mock_backend_gps_present_for_anchors_and_none_otherwise(scenario):
    """gps() returns (lat, lon, altitude_m) for GPS anchors, None for non-anchors."""
    anchor_ids = {d.id for d in scenario.anchors}
    assert anchor_ids  # the demo scenario has anchors
    non_anchor_ids = set(scenario.device_ids) - anchor_ids
    assert non_anchor_ids  # ...and at least one non-anchor

    for dev in scenario.device_ids:
        fix = MockBackend(scenario, dev).gps()
        if dev in anchor_ids:
            assert fix is not None
            assert len(fix) == 3
            lat, lon, alt = fix
            assert all(isinstance(v, float) for v in (lat, lon, alt))
        else:
            assert fix is None


# ----------------------------------------------------------------------------
# SoundDeviceBackend: guarded import (sounddevice is NOT installed here)
# ----------------------------------------------------------------------------


def _sounddevice_installed() -> bool:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    _sounddevice_installed(),
    reason="sounddevice IS installed; the guarded-error path cannot be exercised here.",
)
def test_sounddevice_backend_construction_raises_clear_error_when_missing():
    """Constructing the real-mic backend without sounddevice raises a clear, actionable
    error (mentioning 'sounddevice') rather than failing obscurely later."""
    with pytest.raises((RuntimeError, ImportError)) as excinfo:
        SoundDeviceBackend("dev0")
    assert "sounddevice" in str(excinfo.value).lower()


@pytest.mark.skipif(
    _sounddevice_installed(),
    reason="sounddevice IS installed; the guarded-error path cannot be exercised here.",
)
def test_sounddevice_backend_record_raises_clear_error_when_missing():
    """Recording is also unreachable without sounddevice: it must surface the same
    clear, actionable guarded error (here, at construction before record is callable)."""
    with pytest.raises((RuntimeError, ImportError)) as excinfo:
        backend = SoundDeviceBackend("dev0", sample_rate_hz=48000.0)
        backend.record(0.1)
    assert "sounddevice" in str(excinfo.value).lower()
