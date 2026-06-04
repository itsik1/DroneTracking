"""Concrete :class:`~dronetracking.device.backend.CaptureBackend` implementations.

Two backends share the single sensor interface the device agent reads from, so the
agent code is identical whether it runs against the simulator or real hardware:

* :class:`MockBackend` — sim-driven, deterministic. It runs the simulator once and
  renders the per-device acoustic waveforms once, then serves this device's slice of
  that world: its synthesized microphone audio (so matched-filter detection on it
  reproduces the sim's true arrivals), the ranging records where this device is the
  initiator, and its GPS fix if it is a georeferencing anchor. Used by tests and any
  sim-driven run of the device runtime.
* :class:`SoundDeviceBackend` — a real microphone via the optional ``sounddevice``
  library. The import is GUARDED: if ``sounddevice`` is unavailable, construction
  raises a clear, actionable :class:`RuntimeError` rather than failing obscurely deep
  in :meth:`record`. Two-way ranging and GPS are hardware-bringup tasks (documented on
  the respective methods), so this backend captures audio only.

Local clocks: a backend's :meth:`local_time` is its OWN clock and carries an unknown
offset/drift vs. peers — the system never assumes a shared clock. The mock returns a
deterministic value; the real backend reads ``time.monotonic()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..sim.audio import AudioCapture, synthesize_captures
from ..sim.scenario import Scenario
from ..sim.simulator import simulate
from .backend import CaptureBackend


class MockBackend(CaptureBackend):
    """A deterministic, sim-driven :class:`CaptureBackend` for one device.

    Runs :func:`~dronetracking.sim.simulator.simulate` and
    :func:`~dronetracking.sim.audio.synthesize_captures` exactly once at construction,
    then serves this device's slice of that single synthetic world:

    * :meth:`record` returns this device's full synthesized waveform with
      ``t0_local_s = 0.0`` (the synthesizer places each pulse at the device's
      local-clock arrival time with ``t0_local = 0``), so running
      :func:`~dronetracking.estimation.detection.detect_arrivals` on it recovers the
      sim's true per-device arrival times.
    * :meth:`ranging_records` returns the sim's ranging exchanges where this device is
      the initiator.
    * :meth:`gps` returns this device's GPS fix if it is a georeferencing anchor, else
      ``None``.

    The simulation is seeded from ``scenario.seed`` (via :func:`simulate`); the only
    extra randomness is the acoustic-synthesis ``rng`` (rotor phases + white noise),
    which defaults to a fixed seed so a ``MockBackend`` is fully reproducible.
    """

    def __init__(
        self,
        scenario: Scenario,
        device_id: str,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        known = scenario.device_ids
        if device_id not in known:
            raise ValueError(
                f"device_id {device_id!r} is not in scenario {scenario.name!r}; "
                f"known devices: {list(known)}"
            )

        self._scenario = scenario
        self._device_id = device_id
        self._sample_rate_hz = float(scenario.sample_rate_hz)

        # Run the synthetic world ONCE; the Observations carry the measurements (ranging,
        # GPS) and the World is unused here (ground-truth firewall is not our concern, but
        # we serve only measurable quantities anyway).
        self._observations, _ = simulate(scenario)

        # Render every device's full-duration waveform ONCE. Default to a fixed seed so a
        # MockBackend constructed without an explicit rng is deterministic across runs.
        if rng is None:
            rng = np.random.default_rng(scenario.seed)
        captures = synthesize_captures(scenario, rng)
        self._capture: AudioCapture = captures[device_id]

        # This device's ranging exchanges as initiator (four-timestamp SDS-TWR).
        self._ranging = tuple(
            r for r in self._observations.ranging if r.initiator == device_id
        )

        # This device's GPS fix, if it is a georeferencing anchor.
        self._gps: Optional[Tuple[float, float, float]] = None
        for g in self._observations.anchor_gps:
            if g.device_id == device_id:
                self._gps = (float(g.lat), float(g.lon), float(g.altitude_m))
                break

    # -- CaptureBackend interface --------------------------------------------

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    def local_time(self) -> float:
        """A simple deterministic local-clock reading (seconds).

        The mock world is timeless, so this returns ``0.0`` rather than a wall clock —
        enough to satisfy the interface and keep mock-driven runs reproducible.
        """
        return 0.0

    def record(self, duration_s: float) -> Tuple[np.ndarray, float]:
        """Return this device's synthesized waveform and ``t0_local_s = 0.0``.

        ``duration_s`` selects the leading ``round(duration_s * sample_rate_hz)``
        samples of the pre-rendered full-scenario waveform (clamped to its length);
        a non-positive ``duration_s`` returns the whole capture. ``t0_local_s`` is
        ``0.0`` to match the synthesizer convention, so a detected peak at sample ``k``
        is an arrival at ``k / sample_rate_hz`` in this device's local clock and
        detection on it reproduces the sim's true arrivals.
        """
        samples = self._capture.samples
        if duration_s is not None and duration_s > 0.0:
            n = int(round(float(duration_s) * self._sample_rate_hz))
            n = min(max(n, 1), samples.size)
            samples = samples[:n]
        # Return a copy so callers can't mutate the cached capture.
        return np.array(samples, dtype=float, copy=True), 0.0

    def ranging_records(self):
        """The sim's two-way ranging exchanges initiated by this device."""
        return self._ranging

    def gps(self) -> Optional[Tuple[float, float, float]]:
        """This device's GPS fix ``(lat, lon, altitude_m)``, or ``None`` if not an anchor."""
        return self._gps


# Guarded import message reused by construction and (defensively) by record().
_SOUNDDEVICE_HINT = (
    "SoundDeviceBackend requires the optional 'sounddevice' package (PortAudio "
    "bindings), which is not installed. Install it with `pip install sounddevice` "
    "(and ensure the PortAudio system library is present), or use MockBackend for "
    "simulation-driven runs."
)


def _import_sounddevice():
    """Import ``sounddevice`` or raise a clear, actionable :class:`RuntimeError`.

    Isolated so both construction and (defensively) recording surface the same message,
    and so tests can assert on the guarded failure without ``sounddevice`` installed.
    """
    try:
        import sounddevice  # type: ignore
    except Exception as exc:  # ImportError, or OSError if PortAudio is missing
        raise RuntimeError(_SOUNDDEVICE_HINT) from exc
    return sounddevice


class SoundDeviceBackend(CaptureBackend):
    """A real-microphone :class:`CaptureBackend` via the ``sounddevice`` library.

    Captures mono audio with :func:`sounddevice.rec` and timestamps it on
    :func:`time.monotonic`. The ``sounddevice`` import is GUARDED at construction: if
    the package (or its PortAudio backend) is unavailable, the constructor raises a
    clear :class:`RuntimeError` explaining how to install it, rather than failing
    obscurely on the first :meth:`record`.

    This backend captures audio only. Two-way ranging (:meth:`ranging_records`) and a
    GPS fix (:meth:`gps`) are hardware-bringup tasks documented on those methods.
    """

    def __init__(self, device_id: str, *, sample_rate_hz: float = 48000.0) -> None:
        # GUARDED import: fail fast and clearly at construction if sounddevice is absent.
        self._sd = _import_sounddevice()
        self._device_id = str(device_id)
        self._sample_rate_hz = float(sample_rate_hz)

    # -- CaptureBackend interface --------------------------------------------

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    def local_time(self) -> float:
        """This device's local clock, read from :func:`time.monotonic` (seconds).

        Monotonic (never steps backward) and has an unknown offset vs. peers — exactly
        the unsynchronized local clock the rest of the system expects.
        """
        return time.monotonic()

    def record(self, duration_s: float) -> Tuple[np.ndarray, float]:
        """Capture ``duration_s`` of mono microphone audio via :func:`sounddevice.rec`.

        Returns ``(samples, t0_local_s)`` where ``samples`` is a 1-D float array and
        ``t0_local_s`` is :meth:`local_time` sampled immediately before the capture
        starts, so a detected peak at sample ``k`` is an arrival at
        ``t0_local_s + k / sample_rate_hz``.
        """
        n = int(round(float(duration_s) * self._sample_rate_hz))
        n = max(n, 1)
        t0 = self.local_time()
        recording = self._sd.rec(
            n, samplerate=self._sample_rate_hz, channels=1, dtype="float64"
        )
        self._sd.wait()  # block until the capture finishes
        samples = np.asarray(recording, dtype=float).reshape(-1)
        return samples, float(t0)

    def ranging_records(self):
        """Not implemented on real hardware yet — chirp-ranging bringup.

        Real two-way ranging requires emitting a known chirp through this device's
        speaker (:meth:`play`) and recording the peer's timestamped reply to form the
        four-timestamp SDS-TWR record. That speaker/echo bringup is the documented
        hardware task; until it lands, this backend raises rather than fabricating
        ranging records. Sim-driven runs use :class:`MockBackend`, which serves the
        simulator's ranging instead.
        """
        raise NotImplementedError(
            "SoundDeviceBackend has no two-way ranging yet: it requires chirp-ranging "
            "bringup (emit a chirp via SoundDeviceBackend.play and timestamp the peer's "
            "reply for four-timestamp SDS-TWR). Use MockBackend for sim-driven ranging."
        )

    def gps(self) -> Optional[Tuple[float, float, float]]:
        """No GPS fix from this backend.

        Reading an OS GPS fix is platform-specific (CoreLocation on macOS, gpsd/Location
        APIs elsewhere) and is intentionally out of scope here, so this returns ``None``.
        A GPS-equipped device would supply its fix through a separate, platform-specific
        backend.
        """
        return None

    def play(self, signal: np.ndarray) -> float:
        """Emit ``signal`` through the speaker via :func:`sounddevice.play`.

        Returns the local emit time (:meth:`local_time` sampled just before playback
        starts). Provided so chirp-ranging bringup (see :meth:`ranging_records`) has a
        speaker primitive to build on.
        """
        data = np.asarray(signal, dtype=float).reshape(-1, 1)
        t0 = self.local_time()
        self._sd.play(data, samplerate=self._sample_rate_hz)
        self._sd.wait()
        return float(t0)
