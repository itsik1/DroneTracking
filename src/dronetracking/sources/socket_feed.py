"""Coordinator-side device feed: assemble per-device batches arriving over TCP.

:class:`SocketDeviceFeed` is the network counterpart of
:class:`~dronetracking.sources.simulated.SimulatedDeviceFeed`. Instead of running the
simulator in-process, it binds a TCP listener and waits for each device's
:class:`~dronetracking.live.agent.DeviceAgent` to connect and push its own measurement
slice (encoded by :mod:`dronetracking.live.protocol`). Once every expected device has
reported, it unions the slices into a single :class:`~dronetracking.sim.observations.Observations`
and serves the standard :class:`~dronetracking.sources.base.DeviceFeed` surface — so the
entire batch pipeline and streaming engine run against real, distributed input unchanged.

Lifecycle:

* construct with ``SocketDeviceFeed(host="127.0.0.1", port=0)`` — port ``0`` asks the OS
  for a free port, exposed afterwards as :attr:`port` (so a test can connect to it),
* call :meth:`collect` with the device ids to wait for; it accepts connections and reads
  each batch until all expected devices have reported (or the timeout elapses),
* then read it like any feed (``feed.as_observations()`` etc.).

Robustness notes baked in for determinism (the loopback test must never flake):

* the listener is bound and listening *before* the constructor returns, so an agent can
  connect the instant it has the port — no accept/connect race,
* each connection is drained until newline-or-EOF, so a batch split across multiple TCP
  segments is reassembled correctly (no short-read bugs),
* the device order downstream is exactly ``expected_device_ids`` (deterministic matrix
  row order), regardless of the order connections happened to arrive.
"""

from __future__ import annotations

import socket
import threading
from typing import Dict, List, Sequence, Tuple

from ..live import protocol
from ..sim.observations import AcousticArrival, AnchorGps, RangingRecord
from .base import DeviceFeed

# Default backlog and read chunk size; both comfortably oversized for the use case.
_LISTEN_BACKLOG = 64
_RECV_CHUNK = 65536


class SocketDeviceFeed(DeviceFeed):
    """A :class:`DeviceFeed` assembled from per-device batches received over TCP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Bind and start listening immediately (so agents can connect right away).

        Args:
            host: interface to bind (``"127.0.0.1"`` for loopback).
            port: TCP port; ``0`` lets the OS assign a free one, then read :attr:`port`.
        """
        self.host = host
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(_LISTEN_BACKLOG)
        # Resolve the actually-bound port (meaningful when port==0 was requested).
        self.port = self._server.getsockname()[1]

        # Per-device decoded batches, plus the order we want to expose downstream.
        self._batches: Dict[str, dict] = {}
        self._expected_order: Tuple[str, ...] = ()
        self._collected = False
        self._closed = False
        # Default timebase, overwritten from the first batch that carries one.
        self._speed_of_sound_mps: float = 0.0
        self._sample_rate_hz: float = 0.0

    # -- collection ----------------------------------------------------------- #
    def collect(
        self, expected_device_ids: Sequence[str], timeout_s: float = 10.0
    ) -> Dict[str, dict]:
        """Accept connections and read batches until all expected devices report.

        Blocks (intended to run in the coordinator thread) accepting one connection per
        device, decoding each batch, and stopping as soon as every id in
        ``expected_device_ids`` has been seen — or when ``timeout_s`` elapses overall.

        Args:
            expected_device_ids: the device ids to wait for. Also fixes the downstream
                device order (the row order of distance/geometry matrices).
            timeout_s: wall-clock budget for the whole collection.

        Returns:
            The mapping ``{device_id: decoded_batch}`` collected so far (a dict from
            :func:`dronetracking.live.protocol.decode_batch`). On timeout it returns what
            arrived; :meth:`as_observations` then reflects only the devices that reported.
        """
        self._expected_order = tuple(expected_device_ids)
        expected = set(self._expected_order)
        deadline = _monotonic() + float(timeout_s)

        try:
            while expected - set(self._batches):
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    break  # overall timeout: return what we have so far.
                # Bound accept() by the remaining budget so a missing device can't hang.
                self._server.settimeout(remaining)
                try:
                    conn, _addr = self._server.accept()
                except socket.timeout:
                    break
                with conn:
                    # The per-connection read also respects the overall deadline.
                    conn.settimeout(max(deadline - _monotonic(), 0.0))
                    try:
                        raw = _recv_one_message(conn)
                    except (socket.timeout, OSError):
                        continue  # ignore a stalled/broken connection; keep waiting.
                if not raw:
                    continue
                batch = protocol.decode_batch(raw)
                device_id = batch["device_id"]
                # Capture the timebase from the first batch (all devices share it).
                if not self._batches:
                    self._speed_of_sound_mps = batch["speed_of_sound_mps"]
                    self._sample_rate_hz = batch["sample_rate_hz"]
                # Last writer wins if a device somehow reports twice; idempotent in practice.
                self._batches[device_id] = batch
        finally:
            self._server.settimeout(None)

        self._collected = True
        return dict(self._batches)

    # -- DeviceFeed measurement surface (unions of the collected batches) ------ #
    def device_ids(self) -> Tuple[str, ...]:
        """Expected device order, restricted to devices that actually reported.

        Order follows ``expected_device_ids`` from :meth:`collect`; ids that never
        reported are dropped so the matrices stay consistent with the data present.
        """
        reported = set(self._batches)
        ordered = tuple(d for d in self._expected_order if d in reported)
        # Include any unexpected reporters (deterministically) so no data is silently lost.
        extras = tuple(d for d in self._batches if d not in set(self._expected_order))
        return ordered + extras

    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        """All ranging exchanges, unioned across devices in :meth:`device_ids` order."""
        out: List[RangingRecord] = []
        for did in self.device_ids():
            out.extend(self._batches[did]["ranging"])
        return tuple(out)

    def acoustic_arrivals(self) -> Tuple[AcousticArrival, ...]:
        """All acoustic arrivals, unioned across devices in :meth:`device_ids` order."""
        out: List[AcousticArrival] = []
        for did in self.device_ids():
            out.extend(self._batches[did]["acoustic"])
        return tuple(out)

    def anchor_gps(self) -> Tuple[AnchorGps, ...]:
        """All GPS anchor fixes, unioned across devices in :meth:`device_ids` order."""
        out: List[AnchorGps] = []
        for did in self.device_ids():
            out.extend(self._batches[did]["anchor_gps"])
        return tuple(out)

    @property
    def speed_of_sound_mps(self) -> float:
        """Speed of sound (m/s), taken from the reporting devices' batches."""
        return self._speed_of_sound_mps

    @property
    def sample_rate_hz(self) -> float:
        """Acoustic sample rate (Hz), taken from the reporting devices' batches."""
        return self._sample_rate_hz

    # -- lifecycle ------------------------------------------------------------ #
    def close(self) -> None:
        """Close the listening socket. Safe to call more than once."""
        if not self._closed:
            self._closed = True
            try:
                self._server.close()
            except OSError:
                pass

    def __enter__(self) -> "SocketDeviceFeed":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _recv_one_message(conn: socket.socket) -> bytes:
    """Read one newline-terminated batch from ``conn``, reassembling partial reads.

    Accumulates chunks until the framing newline is seen (one complete message) or the
    peer closes its write side (EOF). Returns the bytes up to and including the newline,
    or whatever preceded EOF — exactly what :func:`protocol.decode_batch` tolerates.
    """
    buf = bytearray()
    while protocol.LINE_DELIMITER not in buf:
        chunk = conn.recv(_RECV_CHUNK)
        if not chunk:  # EOF: peer half-closed after sending the whole batch.
            break
        buf.extend(chunk)
    return bytes(buf)


def _monotonic() -> float:
    """Monotonic clock for timeout math (import-local so tests can't see/patch it)."""
    import time

    return time.monotonic()
