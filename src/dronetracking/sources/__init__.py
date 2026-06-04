"""Device-feed / hardware abstraction layer.

This package defines the boundary the estimation and streaming layers read measurements
from. A :class:`DeviceFeed` produces the
:class:`~dronetracking.sim.observations.Observations` bundle the rest of the system
consumes, so the source of measurements can be swapped without touching the pipeline:

* :class:`SimulatedDeviceFeed` wraps the simulator (the reference feed; also carries
  ground truth via ``.world`` for evaluation).
* :class:`LiveDeviceFeed` is the documented skeleton a real device network fills in.

Named ``sources`` (not ``io``) to avoid shadowing the stdlib ``io`` module. As a sim
adapter layer (not estimation), it may import ``dronetracking.sim``.
"""

from __future__ import annotations

from .base import DeviceFeed
from .live import LiveDeviceFeed
from .simulated import SimulatedDeviceFeed

__all__ = ["DeviceFeed", "SimulatedDeviceFeed", "LiveDeviceFeed"]
