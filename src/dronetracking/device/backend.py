"""The sensor abstraction a device runtime captures through.

A ``CaptureBackend`` is everything the on-device agent needs from its hardware: a local
clock (NOT synchronized with peers), microphone audio, this device's two-way ranging
records, and an optional GPS fix. Implementations live in ``device.backends``:

* ``MockBackend`` — sim-driven, deterministic, for testing.
* ``SoundDeviceBackend`` — a real microphone via the ``sounddevice`` library.

The agent code is identical across backends; only the backend changes between a
simulated run and a real device.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..sim.observations import RangingRecord


class CaptureBackend(ABC):
    """Sensor interface the device agent reads from (see module docstring)."""

    @property
    @abstractmethod
    def device_id(self) -> str:
        ...

    @property
    @abstractmethod
    def sample_rate_hz(self) -> float:
        ...

    @abstractmethod
    def local_time(self) -> float:
        """This device's local clock reading (seconds). Has an unknown offset/drift vs
        peers — the system never assumes a shared clock."""

    @abstractmethod
    def record(self, duration_s: float) -> Tuple[np.ndarray, float]:
        """Capture ``duration_s`` of mono microphone audio.

        Returns ``(samples, t0_local_s)`` where ``t0_local_s`` is this device's local-clock
        timestamp of the first sample, so a detected peak at sample ``k`` is an arrival at
        ``t0_local_s + k / sample_rate_hz``.
        """

    @abstractmethod
    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        """This device's two-way ranging exchanges as initiator (four-timestamp SDS-TWR).

        On real hardware this is produced by emitting a chirp via the speaker and recording
        the peer's reply; in simulation it is provided by the mock backend.
        """

    @abstractmethod
    def gps(self) -> Optional[Tuple[float, float, float]]:
        """Current GPS fix ``(lat, lon, altitude_m)`` or ``None`` if this device has no GPS."""

    def play(self, signal: np.ndarray) -> float:
        """Emit a signal (e.g. a ranging chirp) through the speaker; return the local emit
        time. Optional — receive-only devices may leave this unimplemented."""
        raise NotImplementedError("this backend cannot emit audio")
