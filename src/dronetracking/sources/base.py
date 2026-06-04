"""The device-feed boundary: what the estimation/streaming layer reads measurements from.

A :class:`DeviceFeed` is the seam between *how measurements are produced* (a simulator
today, a real device network tomorrow) and *how they are consumed* (the batch pipeline
and the streaming engine). Everything above this line — geometry, clock sync, TDOA,
georeferencing — works purely against the :class:`~dronetracking.sim.observations.Observations`
bundle. So if a feed can produce that bundle, the entire estimation stack runs unchanged.

The five abstract accessors mirror exactly the fields of ``Observations`` (device ids,
ranging exchanges, acoustic arrivals, GPS anchors) plus the two physical constants the
solvers need (speed of sound, sample rate). The single concrete method
:meth:`DeviceFeed.as_observations` bundles them, so a subclass implements the parts and
gets the whole for free.

This package is a sim *adapter* layer, not estimation, so it MAY import ``sim`` — but the
abstract base itself only depends on the measurable contract types, never on ``World`` or
any ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from ..sim.observations import AcousticArrival, AnchorGps, Observations, RangingRecord


class DeviceFeed(ABC):
    """Abstract source of distributed-sensing measurements.

    Implementations supply the four measurement streams and the two timebase constants;
    :meth:`as_observations` assembles them into the :class:`Observations` bundle that the
    rest of the system consumes. Two concrete feeds exist:

    * :class:`~dronetracking.sources.simulated.SimulatedDeviceFeed` — wraps the simulator
      (the reference / test feed), and additionally exposes ``.world`` ground truth.
    * :class:`~dronetracking.sources.live.LiveDeviceFeed` — the documented skeleton a real
      device network would fill in.
    """

    # -- abstract measurement surface (one accessor per Observations field) ---

    @abstractmethod
    def device_ids(self) -> Tuple[str, ...]:
        """Stable identifiers of every device in the network, in a fixed order.

        This order defines the row order of the relative-geometry and distance
        matrices downstream, so it must be deterministic across a run.
        """

    @abstractmethod
    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        """All two-way ranging exchanges between devices.

        Each :class:`~dronetracking.sim.observations.RangingRecord` carries the four
        timestamps of one initiator/responder round, **each in its own device's local
        clock**. These feed inter-device distance estimation and clock synchronization.
        """

    @abstractmethod
    def acoustic_arrivals(self) -> Tuple[AcousticArrival, ...]:
        """Arrival times of drone emissions at the listening devices.

        Each :class:`~dronetracking.sim.observations.AcousticArrival` is one
        (device, emission) time of arrival in the device's local clock, tagged with the
        source target and a detector confidence. These are the TDOA observations.
        """

    @abstractmethod
    def anchor_gps(self) -> Tuple[AnchorGps, ...]:
        """Reported GPS fixes for the GPS-equipped (anchor) devices.

        Each :class:`~dronetracking.sim.observations.AnchorGps` is a (possibly noisy)
        lat/lon/altitude for one anchor; these georeference the relative solution into
        real-world coordinates. Devices without GPS contribute nothing here.
        """

    @property
    @abstractmethod
    def speed_of_sound_mps(self) -> float:
        """Speed of sound (m/s) used to convert acoustic time-of-flight to range."""

    @property
    @abstractmethod
    def sample_rate_hz(self) -> float:
        """Acoustic sampling rate (Hz) — the timing resolution of arrival detection."""

    # -- concrete bundler: free for every subclass ---------------------------

    def as_observations(self) -> Observations:
        """Bundle the abstract streams into the :class:`Observations` contract type.

        This is what makes a feed a drop-in for ``simulate(scenario)[0]``: the existing
        batch pipeline and streaming engine consume an ``Observations``, and any feed can
        produce one. Defined once here so subclasses only implement the parts.
        """
        return Observations(
            device_ids=self.device_ids(),
            ranging=self.ranging_records(),
            acoustic=self.acoustic_arrivals(),
            anchor_gps=self.anchor_gps(),
            speed_of_sound_mps=self.speed_of_sound_mps,
            sample_rate_hz=self.sample_rate_hz,
        )
