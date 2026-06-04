"""Neighbor discovery and the network-formation manager.

:func:`discover` runs a HELLO sweep: every online node broadcasts to every other node;
peers that receive the HELLO (link in range and the packet not dropped) become
neighbors, yielding an undirected adjacency :class:`NetworkGraph`. Discovery makes **no**
assumption about known positions — it only observes which HELLOs arrive; the *simulated*
transport underneath is what decides reachability from true positions.

:class:`NetworkManager` ties it together for a :class:`~dronetracking.sim.scenario.Scenario`:
it builds the :class:`~dronetracking.network.node.NodeRegistry` and a transport from the
scenario's ``network`` block, forms the graph, and answers neighbor / connectivity /
health questions for the orchestrator and dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from ..sim.scenario import Scenario
from .node import Node, NodeRegistry
from .transport import SimulatedTransport, Transport


@dataclass
class NetworkGraph:
    """Undirected adjacency graph over node ids.

    A link is recorded only if *both directions* were heard during discovery (a symmetric
    radio link). ``link_quality`` carries the per-edge quality (min of the two directions)
    so the manager can summarise mean link health.
    """

    node_ids: Tuple[str, ...]
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)
    link_quality: Dict[FrozenSet[str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for nid in self.node_ids:
            self.adjacency.setdefault(nid, set())

    # -- mutation ------------------------------------------------------------
    def add_edge(self, a: str, b: str, quality: float = 1.0) -> None:
        """Record a symmetric link ``a <-> b`` with the given quality."""
        if a == b:
            return
        self.adjacency.setdefault(a, set()).add(b)
        self.adjacency.setdefault(b, set()).add(a)
        self.link_quality[frozenset((a, b))] = float(quality)

    # -- queries -------------------------------------------------------------
    def neighbors(self, node_id: str) -> Tuple[str, ...]:
        """Neighbors of ``node_id``, sorted for stable output."""
        return tuple(sorted(self.adjacency.get(node_id, set())))

    def degree(self, node_id: str) -> int:
        return len(self.adjacency.get(node_id, set()))

    def edges(self) -> Tuple[Tuple[str, str], ...]:
        """All undirected edges as sorted ``(a, b)`` pairs, deduplicated."""
        seen: Set[FrozenSet[str]] = set()
        out: List[Tuple[str, str]] = []
        for a, nbrs in self.adjacency.items():
            for b in nbrs:
                key = frozenset((a, b))
                if key not in seen:
                    seen.add(key)
                    out.append(tuple(sorted((a, b))))  # type: ignore[arg-type]
        return tuple(sorted(out))

    def isolated(self) -> Tuple[str, ...]:
        """Node ids with no neighbors at all."""
        return tuple(n for n in self.node_ids if self.degree(n) == 0)

    def quality_of(self, a: str, b: str) -> float:
        """Recorded link quality of edge ``a-b`` (0.0 if no such edge)."""
        return float(self.link_quality.get(frozenset((a, b)), 0.0))

    def components(self) -> Tuple[FrozenSet[str], ...]:
        """Connected components as frozensets of node ids (BFS over the adjacency)."""
        unvisited: Set[str] = set(self.node_ids)
        comps: List[FrozenSet[str]] = []
        while unvisited:
            start = next(iter(unvisited))
            stack = [start]
            seen: Set[str] = set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for nb in self.adjacency.get(cur, set()):
                    if nb not in seen:
                        stack.append(nb)
            comps.append(frozenset(seen))
            unvisited -= seen
        return tuple(comps)

    def is_connected(self) -> bool:
        """True if every node reaches every other (single connected component).

        The empty graph is trivially connected; a single node is connected.
        """
        if len(self.node_ids) <= 1:
            return True
        return len(self.components()) == 1

    def mean_link_quality(self) -> float:
        """Mean quality over all recorded edges (0.0 if there are no edges)."""
        if not self.link_quality:
            return 0.0
        return float(sum(self.link_quality.values()) / len(self.link_quality))


def discover(
    devices: Union[NodeRegistry, Iterable[Node]],
    transport: Transport,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> NetworkGraph:
    """Run a HELLO broadcast sweep and return the resulting adjacency graph.

    Each online node broadcasts a HELLO to every other node via ``transport.send``;
    after the sweep, ``transport.deliver`` is drained to see which HELLOs arrived. An
    undirected edge is added only when HELLOs were heard in *both* directions (a usable,
    symmetric link); its quality is the worse of the two directions' link qualities.

    ``rng`` is accepted for signature symmetry with the rest of the API; the randomness
    that matters (packet loss) lives inside the transport. It is unused here but kept so
    callers can thread a seed uniformly.
    """
    del rng  # randomness is owned by the transport; kept for a uniform call signature.

    registry = devices if isinstance(devices, NodeRegistry) else NodeRegistry(devices)
    node_ids = registry.ids
    online = set(registry.online_ids())

    graph = NetworkGraph(node_ids=node_ids)

    # HELLO sweep: every online node -> every other node.
    for src in node_ids:
        if src not in online:
            continue
        for dst in node_ids:
            if dst == src or dst not in online:
                continue
            transport.send(src, dst, {"type": "HELLO", "from": src})

    # Collect who actually heard whom (directed), with the link quality observed.
    heard: Set[FrozenSet[str]] = set()
    directed: Set[Tuple[str, str]] = set()
    qual: Dict[FrozenSet[str], float] = {}
    for pkt in transport.deliver():
        directed.add((pkt.src, pkt.dst))
        key = frozenset((pkt.src, pkt.dst))
        # Edge quality is the worse direction we observed (conservative).
        prev = qual.get(key)
        qual[key] = pkt.quality if prev is None else min(prev, pkt.quality)

    # A symmetric link needs both directions delivered.
    for (a, b) in directed:
        if (b, a) in directed:
            heard.add(frozenset((a, b)))

    for key in heard:
        a, b = tuple(key)
        graph.add_edge(a, b, quality=qual.get(key, 0.0))

    return graph


class NetworkManager:
    """Form and query the device network for a scenario.

    Construction reads the scenario's ``network`` block (``comm_range_m``, ``latency_s``,
    ``loss_prob``, ``kind``) and builds a :class:`NodeRegistry` plus a
    :class:`SimulatedTransport` from the devices' true positions. Call
    :meth:`form_network` to run discovery (also done lazily on first query).
    """

    def __init__(
        self,
        scenario: Scenario,
        rng: Optional[Union[int, np.random.Generator]] = None,
        transport: Optional[Transport] = None,
    ):
        self.scenario = scenario
        self.registry = NodeRegistry.from_specs(scenario.devices)

        net = dict(scenario.network or {})
        self.kind = str(net.get("kind", "wifi"))
        self._rng = (
            rng if isinstance(rng, np.random.Generator)
            else np.random.default_rng(scenario.seed if rng is None else rng)
        )

        if transport is not None:
            self.transport: Transport = transport
        else:
            # True positions at t=0 drive the (simulated) radio reachability.
            positions = {d.id: tuple(float(v) for v in d.position_m) for d in scenario.devices}
            self.transport = SimulatedTransport(
                positions=positions,
                comm_range_m=net.get("comm_range_m"),
                latency_s=net.get("latency_s"),
                loss_prob=net.get("loss_prob"),
                kind=self.kind,
                rng=self._rng,
            )

        self.graph: Optional[NetworkGraph] = None

    # -- formation -----------------------------------------------------------
    def form_network(self) -> NetworkGraph:
        """Run HELLO discovery and cache the resulting graph."""
        self.graph = discover(self.registry, self.transport, self._rng)
        return self.graph

    def _ensure_graph(self) -> NetworkGraph:
        if self.graph is None:
            self.form_network()
        assert self.graph is not None
        return self.graph

    # -- queries -------------------------------------------------------------
    def neighbors(self, node_id: str) -> Tuple[str, ...]:
        """Discovered neighbors of ``node_id``."""
        return self._ensure_graph().neighbors(node_id)

    def is_connected(self) -> bool:
        """True if the discovered graph is a single connected component."""
        return self._ensure_graph().is_connected()

    def health(self) -> Dict[str, object]:
        """Per-node and network-wide health summary (JSON-friendly).

        Returns a dict with::

            {
              "online":        # number of online nodes
              "total":         # number of member nodes
              "mean_battery":  # mean battery_frac over members
              "mean_link_quality":  # mean quality over discovered edges
              "isolated":      # ids with no neighbors
              "connected":     # graph connectivity (bool)
              "n_components":  # number of connected components
              "nodes": [ {id, online, battery_frac, capabilities, degree,
                          neighbors, confidence}, ... ]
            }
        """
        graph = self._ensure_graph()
        nodes_out: List[Dict[str, object]] = []
        for node in self.registry.nodes():
            nodes_out.append(
                {
                    "id": node.id,
                    "online": bool(node.online),
                    "battery_frac": float(node.battery_frac),
                    "capabilities": list(node.capabilities()),
                    "has_gps": bool(node.has_gps),
                    "degree": graph.degree(node.id),
                    "neighbors": list(graph.neighbors(node.id)),
                    "confidence": float(node.confidence),
                }
            )

        return {
            "online": self.registry.online_count(),
            "total": len(self.registry),
            "mean_battery": self.registry.mean_battery(),
            "mean_link_quality": graph.mean_link_quality(),
            "isolated": list(graph.isolated()),
            "connected": graph.is_connected(),
            "n_components": len(graph.components()),
            "nodes": nodes_out,
        }
