"""The hardware contract: what a real device network must provide to drive this stack.

:class:`LiveDeviceFeed` is a *skeleton* — it subclasses :class:`DeviceFeed` and overrides
every abstract member, but each one raises :class:`NotImplementedError` with a docstring
spelling out precisely what a real implementation must supply and where that data comes
from on real hardware. It is intentionally constructible (so it documents and type-checks
as a concrete feed) yet refuses to produce data until wired to a live network.

Filling these methods in — backed by the radio transport layer, on-device acoustic
detectors, and GPS receivers — is the entire integration work needed to run the existing
estimation pipeline against real devices instead of the simulator. No estimation code
changes: it already consumes a :class:`DeviceFeed`.

This module imports nothing from ``sim``; it is the production-facing half of the boundary.
"""

from __future__ import annotations

from typing import Tuple

from ..sim.observations import AcousticArrival, AnchorGps, RangingRecord
from .base import DeviceFeed


class LiveDeviceFeed(DeviceFeed):
    """Placeholder feed for a real distributed device network.

    Every accessor raises :class:`NotImplementedError` documenting the real-world data
    source. A concrete subclass (or a filled-in version of this class) would typically
    hold handles to:

    * the **network transport layer** (Wi-Fi / BLE / mesh) that carries two-way ranging
      exchanges and aggregates per-device reports,
    * the **on-device acoustic detectors** that timestamp drone emissions in each
      device's local clock, and
    * the **GPS receivers** on anchor devices.

    Timebase assumption: like the simulator, all timestamps are reported in each device's
    *own local clock* (offsets/drifts are estimated downstream by clock synchronization,
    not assumed away here). A real implementation must preserve that convention so the
    existing clock-sync and TDOA stages apply unchanged.
    """

    def device_ids(self) -> Tuple[str, ...]:
        """Return the live network's device identifiers, in a stable, fixed order.

        A real implementation gets this from the network/discovery layer (the set of
        devices currently enrolled and reachable). The order must be deterministic for a
        given session, because it fixes the row order of the distance and geometry
        matrices downstream.
        """
        raise NotImplementedError(
            "LiveDeviceFeed.device_ids: enumerate the enrolled/reachable devices from "
            "the network discovery layer, in a stable order."
        )

    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        """Return the two-way ranging exchanges collected from the transport layer.

        A real implementation drains the radio transport's two-way-ranging results: for
        each initiator/responder round it reports the four timestamps
        (t1/t4 on the initiator's clock, t2/t3 on the responder's), each in that device's
        *local* clock — never pre-corrected. These drive inter-device distance estimation
        and clock synchronization.
        """
        raise NotImplementedError(
            "LiveDeviceFeed.ranging_records: collect two-way-ranging exchanges from the "
            "radio transport layer as RangingRecord(t1..t4 in each device's local clock)."
        )

    def acoustic_arrivals(self) -> Tuple[AcousticArrival, ...]:
        """Return drone-emission arrival times from on-device acoustic detection.

        A real implementation gathers, from each device's microphone + detector, the
        time of arrival (in that device's local clock) of each detected drone emission,
        tagged with an ``emission_idx`` that groups the same emission across devices, the
        ``source`` target index, and a detector ``confidence`` in [0, 1]. These are the
        TDOA observations.
        """
        raise NotImplementedError(
            "LiveDeviceFeed.acoustic_arrivals: gather per-device times of arrival from "
            "on-device acoustic detection as AcousticArrival(..., source, confidence)."
        )

    def anchor_gps(self) -> Tuple[AnchorGps, ...]:
        """Return GPS fixes from the anchor devices' receivers.

        A real implementation reads each GPS-equipped device's receiver and reports its
        lat/lon/altitude (with whatever real-world noise the receiver has). These anchor
        the relative solution to real-world coordinates; non-GPS devices contribute none.
        """
        raise NotImplementedError(
            "LiveDeviceFeed.anchor_gps: read lat/lon/altitude from each anchor device's "
            "GPS receiver as AnchorGps records."
        )

    @property
    def speed_of_sound_mps(self) -> float:
        """Return the operative speed of sound (m/s) for the deployment.

        A real implementation supplies this from a configured constant or, better, from
        live atmospheric conditions (temperature/humidity), since it scales every
        acoustic time-of-flight to range.
        """
        raise NotImplementedError(
            "LiveDeviceFeed.speed_of_sound_mps: provide the operative speed of sound, "
            "from configuration or live atmospheric conditions."
        )

    @property
    def sample_rate_hz(self) -> float:
        """Return the devices' acoustic sampling rate (Hz).

        A real implementation reports the microphones' sample rate, which bounds the
        timing resolution of arrival detection. All devices are assumed to share it.
        """
        raise NotImplementedError(
            "LiveDeviceFeed.sample_rate_hz: report the devices' acoustic sampling rate."
        )
