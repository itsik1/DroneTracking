"""On-device agent: publish THIS device's measurement slice over TCP.

A :class:`DeviceAgent` models the code that runs *on a single device*. It owns one
device id and, when asked to :meth:`publish`, it:

1. obtains the full simulated measurement set (a :class:`SimulatedDeviceFeed`, which is
   the stand-in for "what this device's own sensors recorded"),
2. extracts only the slice this device is responsible for — ranging exchanges it
   *initiated*, acoustic arrivals *it* heard, and its own GPS fix(es) — so a device only
   ever sees and sends its own data,
3. encodes that slice with :mod:`dronetracking.live.protocol`, and
4. opens a short-lived TCP connection to the coordinator, sends the bytes, and closes.

This is the concrete hardware bridge: swapping the simulated slice for a real sensor read
(same per-device filtering, same protocol) turns this into genuine on-device firmware
talking to the coordinator's :class:`~dronetracking.sources.socket_feed.SocketDeviceFeed`.
"""

from __future__ import annotations

import socket
from typing import Optional, Tuple

from ..sim.observations import AcousticArrival, AnchorGps, RangingRecord
from ..sim.scenario import Scenario
from ..sources.simulated import SimulatedDeviceFeed
from . import protocol

# Generous default so a slow/contended loopback connect doesn't spuriously fail.
_CONNECT_TIMEOUT_S = 10.0


def device_slice(
    feed: SimulatedDeviceFeed, device_id: str
) -> Tuple[Tuple[RangingRecord, ...], Tuple[AcousticArrival, ...], Tuple[AnchorGps, ...]]:
    """Return the (ranging, acoustic, anchor_gps) slice owned by ``device_id``.

    The partition rule (which, unioned over every device, reproduces the full
    ``Observations`` exactly):

    * **ranging**: records this device *initiated* (``initiator == device_id``). Each
      two-way round is owned by its initiator, so initiator-partitioning is a clean cut.
    * **acoustic**: arrivals this device heard (``device_id`` matches).
    * **anchor_gps**: this device's own GPS fix(es) (``device_id`` matches); empty for
      non-anchor devices.
    """
    obs = feed.as_observations()
    ranging = tuple(r for r in obs.ranging if r.initiator == device_id)
    acoustic = tuple(a for a in obs.acoustic if a.device_id == device_id)
    anchor_gps = tuple(g for g in obs.anchor_gps if g.device_id == device_id)
    return ranging, acoustic, anchor_gps


class DeviceAgent:
    """The on-device half of the distributed runtime: builds and publishes one slice.

    Stateless beyond its construction; :meth:`publish` does the whole build->connect->send
    ->close cycle. Construct one per device (or reuse — it holds no per-call state).
    """

    def __init__(self, *, connect_timeout_s: float = _CONNECT_TIMEOUT_S) -> None:
        """Args: ``connect_timeout_s`` — socket connect/send timeout (seconds)."""
        self.connect_timeout_s = float(connect_timeout_s)

    def publish(
        self,
        host: str,
        port: int,
        scenario: Scenario,
        device_id: str,
        feed: Optional[SimulatedDeviceFeed] = None,
    ) -> bytes:
        """Build this device's slice and send it to the coordinator at ``host:port``.

        Args:
            host: coordinator host (e.g. ``"127.0.0.1"``).
            port: coordinator TCP port (the :class:`SocketDeviceFeed`'s ``.port``).
            scenario: the scenario to simulate, if ``feed`` is not supplied.
            device_id: which device this agent speaks for.
            feed: optional pre-built :class:`SimulatedDeviceFeed`. If ``None``, one is
                built from ``scenario``. Passing a shared feed avoids re-running the
                simulator once per device (and guarantees every agent sees the same
                seeded world — important for the loopback test).

        Returns:
            The exact bytes sent on the wire (the encoded batch), for assertions/logging.
        """
        if feed is None:
            feed = SimulatedDeviceFeed(scenario)

        ranging, acoustic, anchor_gps = device_slice(feed, device_id)
        message = protocol.encode_batch(
            device_id=device_id,
            ranging=ranging,
            acoustic=acoustic,
            anchor_gps=anchor_gps,
            speed_of_sound_mps=feed.speed_of_sound_mps,
            sample_rate_hz=feed.sample_rate_hz,
        )

        # Short-lived connection: connect, send the whole batch, then close so the
        # coordinator sees EOF and knows this device has fully reported.
        with socket.create_connection((host, port), timeout=self.connect_timeout_s) as sock:
            sock.sendall(message)
            # Half-close our write side: signals end-of-batch to the reader even before
            # the socket is fully torn down (the reader loops until newline-or-EOF).
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass  # peer may have already closed; the batch was sent via sendall.

        return message
