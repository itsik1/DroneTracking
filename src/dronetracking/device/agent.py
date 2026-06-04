"""On-device capture agent: capture -> detect -> publish (the real-capture runtime).

Where :class:`dronetracking.live.agent.DeviceAgent` slices an *already-simulated*
``Observations`` and ships it, :class:`DeviceCaptureAgent` is the genuine on-device
application: it reads from a :class:`~dronetracking.device.backend.CaptureBackend`
(a real microphone, or the deterministic mock), runs the matched-filter detector on the
raw audio, and publishes only what *this* device measured to the coordinator over TCP.

The capture->publish cycle, identical across backends:

1. :meth:`record` ``duration_s`` of mono audio via ``backend.record(...)``, yielding
   ``(samples, t0_local_s)`` in the device's own (unsynced) clock,
2. run :func:`dronetracking.estimation.detection.detect_arrivals` to recover, per
   emission, a :class:`~dronetracking.estimation.detection.DetectedArrival` whose
   ``toa_local_s`` is already ``t0_local + peak_index / sr`` in that local clock,
3. map each ``DetectedArrival`` onto the contract
   :class:`~dronetracking.sim.observations.AcousticArrival` (a field-for-field copy —
   the two dataclasses are compatible by design),
4. gather ``backend.ranging_records()`` and ``backend.gps()`` (the GPS fix, if present,
   becomes one :class:`~dronetracking.sim.observations.AnchorGps`),
5. encode the batch with :mod:`dronetracking.live.protocol` and ship it over a
   short-lived TCP connection (connect, send, half-close), exactly as the simulated
   :class:`~dronetracking.live.agent.DeviceAgent` does.

The detector needs a couple of facts the raw waveform doesn't carry: the KNOWN
``reference_pulse`` template, how many emissions to recover (``n_emissions``), and the
nominal spacing between them (``dt_s``, the peak-separation guard). Those are supplied at
construction so a single agent can be pointed at any backend.

This module is device *infrastructure*, not estimation, so it freely imports
``sim``/``live``/``estimation`` (only ``dronetracking/estimation/`` is firewalled from
``sim``). The detector itself stays firewall-clean; this agent is where its
``DetectedArrival`` output is re-expressed as the contract ``AcousticArrival``.
"""

from __future__ import annotations

import socket
from typing import Optional, Tuple

import numpy as np

from ..estimation.detection import detect_arrivals
from ..live import protocol
from ..sim.observations import AcousticArrival, AnchorGps, RangingRecord
from .backend import CaptureBackend

# Generous default so a slow/contended loopback connect doesn't spuriously fail
# (mirrors live.agent's choice — the connect, not the work, is what we guard).
_CONNECT_TIMEOUT_S = 10.0

# Default speed of sound (dry air, ~20 C). A timebase constant the wire protocol carries
# but the CaptureBackend ABC does not expose; override per-deployment (the CLI sets it
# from the scenario). Only used to fill the batch header; no physics is done here.
_DEFAULT_SPEED_OF_SOUND_MPS = 343.0


class _Capture:
    """Minimal duck-typed capture for :func:`detect_arrivals`.

    The detector reads only ``samples`` / ``sample_rate_hz`` / ``t0_local_s`` off each
    capture (it never imports :class:`sim.audio.AudioCapture`), so this tiny stand-in is
    all it needs — and it keeps the agent independent of the ``sim`` audio types.
    """

    __slots__ = ("samples", "sample_rate_hz", "t0_local_s")

    def __init__(self, samples: np.ndarray, sample_rate_hz: float, t0_local_s: float) -> None:
        self.samples = samples
        self.sample_rate_hz = sample_rate_hz
        self.t0_local_s = t0_local_s


class DeviceCaptureAgent:
    """The on-device half of the real-capture runtime: capture, detect, publish.

    Construct one per device with its :class:`CaptureBackend` and the detector params
    (``reference_pulse``, ``n_emissions``, ``dt_s``). :meth:`run` does the whole
    record->detect->connect->send->close cycle and returns the exact bytes sent (handy
    for assertions/logging). Holds no per-call state, so it may be reused.
    """

    def __init__(
        self,
        backend: CaptureBackend,
        reference_pulse: np.ndarray,
        n_emissions: int,
        dt_s: float,
        *,
        speed_of_sound_mps: float = _DEFAULT_SPEED_OF_SOUND_MPS,
        connect_timeout_s: float = _CONNECT_TIMEOUT_S,
    ) -> None:
        """Args:
        backend: the sensor this device captures through (real mic or mock).
        reference_pulse: the KNOWN drone pulse template the matched filter correlates
            against (e.g. :func:`dronetracking.sim.audio.reference_pulse`).
        n_emissions: how many drone emissions to recover per capture.
        dt_s: nominal seconds between emissions (the detector's peak-separation guard).
        speed_of_sound_mps: timebase constant written into the batch header (the backend
            ABC does not carry it). Defaults to ~343 m/s; the CLI sets it from the scenario.
        connect_timeout_s: socket connect/send timeout (seconds).
        """
        self.backend = backend
        self.reference_pulse = np.asarray(reference_pulse, dtype=float)
        self.n_emissions = int(n_emissions)
        self.dt_s = float(dt_s)
        self.speed_of_sound_mps = float(speed_of_sound_mps)
        self.connect_timeout_s = float(connect_timeout_s)

    # -- capture + detect ----------------------------------------------------- #
    def detect(
        self, duration_s: float
    ) -> Tuple[Tuple[AcousticArrival, ...], Tuple[RangingRecord, ...], Tuple[AnchorGps, ...]]:
        """Record ``duration_s`` of audio and build this device's measurement slice.

        Records via the backend, runs the matched-filter detector on the raw samples,
        and assembles the three contract streams this device owns:

        * **acoustic**: each detected emission as an
          :class:`~dronetracking.sim.observations.AcousticArrival` (a field copy of the
          detector's ``DetectedArrival`` — ``toa_local_s`` is already in this device's
          local clock),
        * **ranging**: ``backend.ranging_records()`` (the exchanges it initiated),
        * **anchor_gps**: this device's GPS fix as one
          :class:`~dronetracking.sim.observations.AnchorGps`, or empty if it has no fix.

        Returns ``(acoustic, ranging, anchor_gps)`` — exactly the slice
        :func:`dronetracking.live.protocol.encode_batch` consumes.
        """
        samples, t0_local_s = self.backend.record(duration_s)
        sr = float(self.backend.sample_rate_hz)
        device_id = self.backend.device_id

        capture = _Capture(np.asarray(samples, dtype=float), sr, float(t0_local_s))
        detections = detect_arrivals(
            {device_id: capture},
            self.reference_pulse,
            self.n_emissions,
            self.dt_s,
        )
        # DetectedArrival -> AcousticArrival is a plain field copy (compatible by design).
        # detect_arrivals already stamps device_id from the capture key, so this matches.
        acoustic = tuple(
            AcousticArrival(
                device_id=d.device_id,
                emission_idx=d.emission_idx,
                toa_local_s=d.toa_local_s,
                confidence=d.confidence,
            )
            for d in detections
        )

        ranging = tuple(self.backend.ranging_records())

        fix = self.backend.gps()
        if fix is None:
            anchor_gps: Tuple[AnchorGps, ...] = ()
        else:
            lat, lon, altitude_m = fix
            anchor_gps = (
                AnchorGps(
                    device_id=device_id,
                    lat=float(lat),
                    lon=float(lon),
                    altitude_m=float(altitude_m),
                ),
            )

        return acoustic, ranging, anchor_gps

    # -- publish -------------------------------------------------------------- #
    def run(self, host: str, port: int, duration_s: float = 1.0) -> bytes:
        """Capture, detect, and publish this device's batch to ``host:port``.

        Records ``duration_s`` of audio, detects arrivals, encodes the slice with
        :func:`dronetracking.live.protocol.encode_batch`, and ships it over a short-lived
        TCP connection (connect, ``sendall``, half-close the write side so the coordinator
        sees a clean end-of-batch, close).

        Args:
            host: coordinator host (e.g. ``"127.0.0.1"``).
            port: coordinator TCP port (the :class:`SocketDeviceFeed`'s ``.port``).
            duration_s: how long to record before detecting/publishing.

        Returns:
            The exact bytes sent on the wire (the encoded batch).
        """
        acoustic, ranging, anchor_gps = self.detect(duration_s)
        message = protocol.encode_batch(
            device_id=self.backend.device_id,
            ranging=ranging,
            acoustic=acoustic,
            anchor_gps=anchor_gps,
            speed_of_sound_mps=self.speed_of_sound_mps,
            sample_rate_hz=float(self.backend.sample_rate_hz),
        )

        # Short-lived connection mirroring live.agent.DeviceAgent.publish: connect, send
        # the whole batch, then half-close our write side so the reader sees end-of-batch
        # (it loops until newline-or-EOF) even before the socket is fully torn down.
        with socket.create_connection((host, port), timeout=self.connect_timeout_s) as sock:
            sock.sendall(message)
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass  # peer may have already closed; the batch was sent via sendall.

        return message
