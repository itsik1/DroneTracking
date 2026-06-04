"""Wire protocol for one device's measurement batch (line-delimited JSON).

This is the *pure* serialization seam of the distributed runtime: each device encodes
the slice of measurements it alone observed into a single self-describing message, and
the coordinator decodes it back into the contract dataclasses
(:class:`~dronetracking.sim.observations.RangingRecord`,
:class:`~dronetracking.sim.observations.AcousticArrival`,
:class:`~dronetracking.sim.observations.AnchorGps`). Nothing here touches a socket — it
maps batches <-> ``bytes`` so it can be unit-tested in isolation and reused by any
transport.

Framing: one batch == one JSON object terminated by a single newline (``\\n``). That makes
the message self-delimiting on a TCP byte stream — a reader accumulates bytes until it
sees the newline and then has exactly one complete message. The payload is a flat dict
with the device id, the two timebase constants, and three lists (ranging/acoustic/anchor)
of per-record dicts whose keys mirror the dataclass fields exactly.

Round-trip fidelity is the contract: ``decode_batch(encode_batch(...))`` reconstructs the
dataclasses field-for-field, including the full float precision of every timestamp. We
rely on Python's ``json`` using ``repr``-grade ``float`` formatting (round-trip-safe since
Python 3.1), so an IEEE-754 double survives encode->decode unchanged.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

from ..sim.observations import AcousticArrival, AnchorGps, RangingRecord

# Protocol version + message framing. Bump VERSION if the wire shape changes; the
# delimiter is what makes a batch self-contained on a raw byte stream.
PROTOCOL_VERSION = 1
LINE_DELIMITER = b"\n"


# --------------------------------------------------------------------------- #
# per-record <-> dict (keys mirror the dataclass fields exactly)
# --------------------------------------------------------------------------- #
def _ranging_to_dict(r: RangingRecord) -> Dict:
    return {
        "initiator": r.initiator,
        "responder": r.responder,
        "round_idx": r.round_idx,
        "t1_local_i": r.t1_local_i,
        "t2_local_j": r.t2_local_j,
        "t3_local_j": r.t3_local_j,
        "t4_local_i": r.t4_local_i,
    }


def _ranging_from_dict(d: Dict) -> RangingRecord:
    return RangingRecord(
        initiator=str(d["initiator"]),
        responder=str(d["responder"]),
        round_idx=int(d["round_idx"]),
        t1_local_i=float(d["t1_local_i"]),
        t2_local_j=float(d["t2_local_j"]),
        t3_local_j=float(d["t3_local_j"]),
        t4_local_i=float(d["t4_local_i"]),
    )


def _acoustic_to_dict(a: AcousticArrival) -> Dict:
    return {
        "device_id": a.device_id,
        "emission_idx": a.emission_idx,
        "toa_local_s": a.toa_local_s,
        "source": a.source,
        "confidence": a.confidence,
    }


def _acoustic_from_dict(d: Dict) -> AcousticArrival:
    return AcousticArrival(
        device_id=str(d["device_id"]),
        emission_idx=int(d["emission_idx"]),
        toa_local_s=float(d["toa_local_s"]),
        source=int(d["source"]),
        confidence=float(d["confidence"]),
    )


def _anchor_to_dict(g: AnchorGps) -> Dict:
    return {
        "device_id": g.device_id,
        "lat": g.lat,
        "lon": g.lon,
        "altitude_m": g.altitude_m,
    }


def _anchor_from_dict(d: Dict) -> AnchorGps:
    return AnchorGps(
        device_id=str(d["device_id"]),
        lat=float(d["lat"]),
        lon=float(d["lon"]),
        altitude_m=float(d["altitude_m"]),
    )


# --------------------------------------------------------------------------- #
# public API: encode / decode one device's batch
# --------------------------------------------------------------------------- #
def encode_batch(
    device_id: str,
    ranging: Sequence[RangingRecord],
    acoustic: Sequence[AcousticArrival],
    anchor_gps: Sequence[AnchorGps],
    speed_of_sound_mps: float,
    sample_rate_hz: float,
) -> bytes:
    """Serialize one device's measurement batch to a newline-terminated JSON message.

    Args:
        device_id: identifier of the publishing device (the batch's owner).
        ranging: this device's two-way-ranging exchanges (typically those it initiated).
        acoustic: this device's acoustic arrivals.
        anchor_gps: this device's GPS fix(es) (empty for non-anchor devices).
        speed_of_sound_mps: the operative speed of sound (timebase constant).
        sample_rate_hz: the acoustic sampling rate (timebase constant).

    Returns:
        ``bytes`` containing exactly one JSON object followed by a single ``\\n``. The
        message is self-delimiting on a TCP stream and round-trips losslessly through
        :func:`decode_batch` (timestamps preserved to full float precision).
    """
    payload = {
        "version": PROTOCOL_VERSION,
        "device_id": str(device_id),
        "speed_of_sound_mps": float(speed_of_sound_mps),
        "sample_rate_hz": float(sample_rate_hz),
        "ranging": [_ranging_to_dict(r) for r in ranging],
        "acoustic": [_acoustic_to_dict(a) for a in acoustic],
        "anchor_gps": [_anchor_to_dict(g) for g in anchor_gps],
    }
    # No whitespace tweaks that would drop precision: json emits round-trip-safe floats.
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8") + LINE_DELIMITER


def decode_batch(data: bytes) -> Dict:
    """Deserialize a batch produced by :func:`encode_batch` back into contract types.

    Accepts the encoded ``bytes`` (with or without the trailing newline) or an already
    decoded ``str``. Reconstructs each list element into its frozen dataclass so the
    result is a drop-in for the corresponding ``Observations`` fields.

    Returns:
        A dict with keys ``device_id`` (str), ``speed_of_sound_mps`` (float),
        ``sample_rate_hz`` (float), ``ranging`` (tuple[RangingRecord, ...]),
        ``acoustic`` (tuple[AcousticArrival, ...]), ``anchor_gps`` (tuple[AnchorGps, ...]),
        and ``version`` (int).
    """
    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8")
    else:
        text = data
    # Tolerate the framing newline (and any incidental surrounding whitespace).
    obj = json.loads(text)

    ranging: List[RangingRecord] = [_ranging_from_dict(d) for d in obj.get("ranging", [])]
    acoustic: List[AcousticArrival] = [_acoustic_from_dict(d) for d in obj.get("acoustic", [])]
    anchor_gps: List[AnchorGps] = [_anchor_from_dict(d) for d in obj.get("anchor_gps", [])]

    return {
        "version": int(obj.get("version", PROTOCOL_VERSION)),
        "device_id": str(obj["device_id"]),
        "speed_of_sound_mps": float(obj["speed_of_sound_mps"]),
        "sample_rate_hz": float(obj["sample_rate_hz"]),
        "ranging": tuple(ranging),
        "acoustic": tuple(acoustic),
        "anchor_gps": tuple(anchor_gps),
    }
