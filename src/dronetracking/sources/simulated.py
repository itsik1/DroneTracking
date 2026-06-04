"""The reference device feed: the simulator behind the :class:`DeviceFeed` boundary.

:class:`SimulatedDeviceFeed` runs :func:`~dronetracking.sim.simulator.simulate` **once**
in ``__init__`` and serves every abstract accessor straight from the resulting
:class:`~dronetracking.sim.observations.Observations`. Because it caches the simulator
output, ``feed.as_observations()`` is field-for-field identical to ``simulate(scenario)[0]``
for the same scenario/seed — so swapping ``simulate(scenario)`` for
``SimulatedDeviceFeed(scenario)`` anywhere in the stack is behavior-preserving.

It also exposes :attr:`world` — the sim-only ground truth — which lives *beside* the
``DeviceFeed`` contract rather than inside it. Estimation never sees ``world`` (the
ground-truth firewall); only evaluation code reaches for it.
"""

from __future__ import annotations

from typing import Tuple

from ..sim.observations import AcousticArrival, AnchorGps, Observations, RangingRecord
from ..sim.scenario import Scenario
from ..sim.simulator import simulate
from ..sim.world import World
from .base import DeviceFeed


class SimulatedDeviceFeed(DeviceFeed):
    """A :class:`DeviceFeed` backed by one :func:`simulate` run of a :class:`Scenario`.

    The simulation is run exactly once at construction; all accessors return cached
    references to that single ``Observations``/``World`` pair (so repeated reads are free
    and identical). This is the feed used by tests and by any sim-driven run of the
    pipeline / streaming engine.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        # Run the synthetic world ONCE; keep both halves. `_observations` satisfies the
        # DeviceFeed contract; `_world` is sim-only ground truth for evaluation.
        self._observations, self._world = simulate(scenario)

    # -- DeviceFeed measurement surface (served from the cached Observations) -

    def device_ids(self) -> Tuple[str, ...]:
        return self._observations.device_ids

    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        return self._observations.ranging

    def acoustic_arrivals(self) -> Tuple[AcousticArrival, ...]:
        return self._observations.acoustic

    def anchor_gps(self) -> Tuple[AnchorGps, ...]:
        return self._observations.anchor_gps

    @property
    def speed_of_sound_mps(self) -> float:
        return self._observations.speed_of_sound_mps

    @property
    def sample_rate_hz(self) -> float:
        return self._observations.sample_rate_hz

    # -- fast path: hand back the cached bundle directly ---------------------

    def as_observations(self) -> Observations:
        """Return the simulator's ``Observations`` directly (the cached instance).

        Equivalent to the base implementation, but returns the very object the simulator
        produced (same identity across calls) instead of rebuilding one — and it is, by
        construction, equal to ``simulate(scenario)[0]``.
        """
        return self._observations

    # -- sim-only ground truth (NOT part of the DeviceFeed contract) ---------

    @property
    def world(self) -> World:
        """The simulator's ground-truth :class:`World`, for evaluation/scoring only.

        Matches ``simulate(scenario)[1]`` for this scenario. Deliberately outside the
        :class:`DeviceFeed` interface: a real feed has no ground truth, and estimation
        must never read it (the ground-truth firewall).
        """
        return self._world
