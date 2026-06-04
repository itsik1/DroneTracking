"""Network nodes — one per physical device — and a registry over them.

A :class:`Node` is the network-layer view of a device: its identity, remaining
battery, sensing/emitting capabilities, and a liveness/confidence summary. It is the
counterpart to :class:`~dronetracking.sim.scenario.DeviceSpec` (the ground-truth spec),
projecting only the fields the networking layer cares about. :func:`Node.from_spec`
builds one from a spec so a scenario's devices become a network roster.

The :class:`NodeRegistry` is an ordered ``id -> Node`` map answering membership and
health questions (who is online, mean battery, which nodes are gone).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Iterator, List, Tuple

from ..sim.scenario import DeviceSpec


@dataclass(frozen=True)
class Node:
    """The network-layer view of one device.

    ``online`` is the link/liveness flag (a node may be a member yet unreachable);
    ``confidence`` in [0,1] is a soft health score the discovery layer can fold in
    (e.g. degraded by low battery or poor links).
    """

    id: str
    battery_frac: float = 1.0
    has_mic: bool = True
    has_speaker: bool = True
    has_gps: bool = False
    online: bool = True
    confidence: float = 1.0

    @staticmethod
    def from_spec(spec: DeviceSpec) -> "Node":
        """Project a :class:`DeviceSpec` onto its network :class:`Node`."""
        return Node(
            id=spec.id,
            battery_frac=float(spec.battery_frac),
            has_mic=bool(spec.has_mic),
            has_speaker=bool(spec.has_speaker),
            has_gps=bool(spec.has_gps),
            online=True,
            confidence=1.0,
        )

    def capabilities(self) -> Tuple[str, ...]:
        """The capability tags this node advertises (sorted, stable)."""
        return capabilities(self)

    def can_anchor(self) -> bool:
        """True if this node can act as a GPS georeferencing anchor."""
        return bool(self.has_gps)


def capabilities(node: Node) -> Tuple[str, ...]:
    """Capability tags advertised by ``node`` (e.g. for HELLO payloads).

    Returns a sorted tuple drawn from ``{"mic", "speaker", "gps"}`` so two nodes with
    the same hardware always advertise the same, order-independent set.
    """
    tags: List[str] = []
    if node.has_mic:
        tags.append("mic")
    if node.has_speaker:
        tags.append("speaker")
    if node.has_gps:
        tags.append("gps")
    return tuple(sorted(tags))


class NodeRegistry:
    """An ordered ``id -> Node`` map with membership and health queries.

    Insertion order is preserved so iteration is deterministic and matches the
    scenario's device order. Lookups by id are O(1).
    """

    def __init__(self, nodes: Iterable[Node] = ()):
        self._nodes: "Dict[str, Node]" = {}
        for n in nodes:
            self.add(n)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_specs(cls, specs: Iterable[DeviceSpec]) -> "NodeRegistry":
        """Build a registry from device specs (e.g. ``scenario.devices``)."""
        return cls(Node.from_spec(s) for s in specs)

    def add(self, node: Node) -> None:
        """Insert or replace a node by id."""
        self._nodes[node.id] = node

    # -- membership ----------------------------------------------------------
    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    def __getitem__(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def get(self, node_id: str) -> "Node | None":
        return self._nodes.get(node_id)

    @property
    def ids(self) -> Tuple[str, ...]:
        """All member ids in insertion order."""
        return tuple(self._nodes.keys())

    def nodes(self) -> Tuple[Node, ...]:
        """All member nodes in insertion order."""
        return tuple(self._nodes.values())

    # -- health / state ------------------------------------------------------
    def online_ids(self) -> Tuple[str, ...]:
        """Ids of members currently flagged online, in insertion order."""
        return tuple(n.id for n in self._nodes.values() if n.online)

    def offline_ids(self) -> Tuple[str, ...]:
        """Ids of members currently flagged offline, in insertion order."""
        return tuple(n.id for n in self._nodes.values() if not n.online)

    def online_count(self) -> int:
        return sum(1 for n in self._nodes.values() if n.online)

    def anchors(self) -> Tuple[str, ...]:
        """Ids of GPS-capable members (georeferencing anchors)."""
        return tuple(n.id for n in self._nodes.values() if n.has_gps)

    def mean_battery(self) -> float:
        """Mean ``battery_frac`` over all members (0.0 if empty)."""
        if not self._nodes:
            return 0.0
        return float(sum(n.battery_frac for n in self._nodes.values()) / len(self._nodes))

    def set_online(self, node_id: str, online: bool) -> None:
        """Flip a member's online flag (immutably replaces the stored node)."""
        node = self._nodes[node_id]
        self._nodes[node_id] = replace(node, online=bool(online))

    def set_confidence(self, node_id: str, confidence: float) -> None:
        """Set a member's confidence score (immutably replaces the stored node)."""
        node = self._nodes[node_id]
        self._nodes[node_id] = replace(node, confidence=float(confidence))
