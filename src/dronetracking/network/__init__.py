"""Network formation (Phase 1): nodes, the radio transport, and discovery.

This package is *infrastructure*, not estimation — it MAY import ``dronetracking.sim``
because :class:`~dronetracking.network.transport.SimulatedTransport` uses true device
positions to decide which radio links physically exist. Discovery itself assumes no known
positions; only the simulated radio underneath does.

Typical use::

    from dronetracking.network import NetworkManager
    mgr = NetworkManager(scenario)
    mgr.form_network()
    mgr.is_connected()      # -> bool
    mgr.neighbors("dev0")   # -> ("dev1", "dev4", ...)
    mgr.health()            # -> {online, mean_battery, mean_link_quality, isolated, ...}
"""

from __future__ import annotations

from .discovery import NetworkGraph, NetworkManager, discover
from .node import Node, NodeRegistry, capabilities
from .transport import (
    RADIO_PRESETS,
    Packet,
    RadioProfile,
    SimulatedTransport,
    Transport,
)

__all__ = [
    # node
    "Node",
    "NodeRegistry",
    "capabilities",
    # transport
    "Transport",
    "SimulatedTransport",
    "RadioProfile",
    "RADIO_PRESETS",
    "Packet",
    # discovery
    "NetworkGraph",
    "discover",
    "NetworkManager",
]
