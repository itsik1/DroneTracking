"""The radio layer: an abstract :class:`Transport` and a simulated implementation.

A ``Transport`` is the seam a real radio (BLE / Wi-Fi / mesh) would slot into. The
networking code above it only ever asks "can A reach B, with what latency and link
quality, and did this packet get through?" — never how. :class:`SimulatedTransport`
answers those using the *true* device positions (hence this package may import ``sim``):
a link physically exists iff the two devices are within ``comm_range_m``; quality falls
off with distance; each ``send`` is independently dropped with probability ``loss_prob``.

``kind`` selects a preset radio profile (range / latency / loss) for BLE, Wi-Fi, or a
longer-range mesh; any explicitly passed parameter overrides its preset.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Vec = Sequence[float]


# -- radio presets -----------------------------------------------------------
@dataclass(frozen=True)
class RadioProfile:
    """Default range / latency / loss for a class of radio."""

    comm_range_m: float
    latency_s: float
    loss_prob: float


# Rough, deliberately-distinct stand-ins for real radios:
#  - BLE: short range, low latency, lossy-ish.
#  - Wi-Fi: medium range, low latency, reliable.
#  - mesh: long range (multi-hop-ish reach), higher latency, more loss.
RADIO_PRESETS: Dict[str, RadioProfile] = {
    "ble": RadioProfile(comm_range_m=80.0, latency_s=0.03, loss_prob=0.10),
    "wifi": RadioProfile(comm_range_m=250.0, latency_s=0.005, loss_prob=0.02),
    "mesh": RadioProfile(comm_range_m=600.0, latency_s=0.05, loss_prob=0.15),
}


@dataclass(frozen=True)
class Packet:
    """A delivered packet returned by :meth:`Transport.deliver`."""

    src: str
    dst: str
    payload: object
    latency_s: float
    quality: float


class Transport(abc.ABC):
    """Abstract per-link radio.

    Concrete transports model reachability, latency, and loss between device ids.
    :meth:`send` enqueues a unicast payload (returning whether it will be delivered);
    :meth:`deliver` drains everything sent so far. :meth:`reachable` and
    :meth:`link_quality` describe the *physical* link irrespective of random loss.
    """

    @abc.abstractmethod
    def reachable(self, a: str, b: str) -> bool:
        """True if a packet from ``a`` could physically reach ``b`` (range permitting)."""

    @abc.abstractmethod
    def link_quality(self, a: str, b: str) -> float:
        """Link quality from ``a`` to ``b`` in [0,1] (0 if out of range)."""

    @abc.abstractmethod
    def latency(self, a: str, b: str) -> float:
        """Expected one-way latency (seconds) from ``a`` to ``b``."""

    @abc.abstractmethod
    def send(self, src: str, dst: str, payload: object) -> bool:
        """Enqueue ``payload`` from ``src`` to ``dst``.

        Returns ``True`` if it will be delivered, ``False`` if dropped (out of range
        or a random loss event). Delivered packets are retrievable via :meth:`deliver`.
        """

    @abc.abstractmethod
    def deliver(self) -> Tuple[Packet, ...]:
        """Drain and return every packet delivered since the last call."""


class SimulatedTransport(Transport):
    """A radio simulated from true device positions.

    Parameters
    ----------
    positions:
        ``id -> (x, y, z)`` true device positions (ENU metres). Accepts any mapping of
        id to a length-3 sequence / ndarray.
    comm_range_m, latency_s, loss_prob:
        Link parameters. Each defaults to ``kind``'s preset when left ``None``.
    kind:
        ``"ble" | "wifi" | "mesh"`` — selects the default :class:`RadioProfile`.
    rng:
        Source of randomness for packet loss. An ``int`` seeds a fresh generator.
    """

    def __init__(
        self,
        positions: Mapping[str, Vec],
        comm_range_m: Optional[float] = None,
        latency_s: Optional[float] = None,
        loss_prob: Optional[float] = None,
        kind: str = "wifi",
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        if kind not in RADIO_PRESETS:
            raise ValueError(
                f"unknown transport kind {kind!r}; expected one of {sorted(RADIO_PRESETS)}"
            )
        preset = RADIO_PRESETS[kind]
        self.kind = kind
        self.comm_range_m = float(preset.comm_range_m if comm_range_m is None else comm_range_m)
        self.latency_s = float(preset.latency_s if latency_s is None else latency_s)
        self.loss_prob = float(preset.loss_prob if loss_prob is None else loss_prob)
        if not (0.0 <= self.loss_prob <= 1.0):
            raise ValueError(f"loss_prob must be in [0,1], got {self.loss_prob}")

        self._pos: Dict[str, np.ndarray] = {
            str(k): np.asarray(v, dtype=float) for k, v in positions.items()
        }
        self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self._inbox: list = []

    # -- physical link model -------------------------------------------------
    def distance(self, a: str, b: str) -> float:
        """Euclidean distance (m) between two devices' true positions."""
        return float(np.linalg.norm(self._pos[a] - self._pos[b]))

    def reachable(self, a: str, b: str) -> bool:
        if a == b:
            return True
        if a not in self._pos or b not in self._pos:
            return False
        return self.distance(a, b) <= self.comm_range_m

    def link_quality(self, a: str, b: str) -> float:
        """Quality in [0,1]: 1 at zero distance, linearly to 0 at ``comm_range_m``.

        Returns 0.0 when out of range. A self-link is perfect (1.0).
        """
        if a == b:
            return 1.0
        if not self.reachable(a, b):
            return 0.0
        if self.comm_range_m <= 0.0:
            return 0.0
        q = 1.0 - self.distance(a, b) / self.comm_range_m
        return float(min(1.0, max(0.0, q)))

    def latency(self, a: str, b: str) -> float:
        """One-way latency: base ``latency_s`` plus a small distance-proportional term."""
        if a == b:
            return 0.0
        # Add up to ~50% of base latency at the edge of range (farther = slower).
        edge = 0.5 * self.latency_s * (self.distance(a, b) / self.comm_range_m if self.comm_range_m else 0.0)
        return float(self.latency_s + edge)

    # -- packet flow ---------------------------------------------------------
    def send(self, src: str, dst: str, payload: object) -> bool:
        if not self.reachable(src, dst):
            return False
        # Independent Bernoulli loss per packet.
        if self._rng.random() < self.loss_prob:
            return False
        self._inbox.append(
            Packet(
                src=src,
                dst=dst,
                payload=payload,
                latency_s=self.latency(src, dst),
                quality=self.link_quality(src, dst),
            )
        )
        return True

    def deliver(self) -> Tuple[Packet, ...]:
        out = tuple(self._inbox)
        self._inbox = []
        return out
