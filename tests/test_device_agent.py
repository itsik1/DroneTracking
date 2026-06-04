"""Loopback test for the on-device capture agent (capture -> detect -> publish).

We exercise :class:`dronetracking.device.agent.DeviceCaptureAgent` end to end over a real
TCP socket, but with a fully synthetic, deterministic world:

* a small in-test :class:`CaptureBackend` (``FakeBackend``) returns a waveform we built by
  hand — a KNOWN reference pulse copied in at KNOWN sample offsets — plus a couple of
  ranging records and a GPS fix we inject directly. This keeps the test independent of
  agent A's ``device.backends`` (which may not exist yet),
* the coordinator side is the real
  :class:`dronetracking.sources.socket_feed.SocketDeviceFeed`, bound on an OS-assigned
  port and ``collect``-ing in a daemon thread,
* each device's agent ``run(...)`` connects to ``127.0.0.1:feed.port`` and ships its batch.

Then we assert ``feed.as_observations()`` carries, for every device:

* the detected acoustic arrivals at local-clock times within a few samples of where we
  placed the pulses (matched-filter detection recovers the emission times we encoded),
* the exact ranging records and GPS fix we injected (passed through the wire protocol).

Everything is deterministic (no RNG in the synthesized signal), uses generous timeouts,
and joins every thread.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

from dronetracking.device.agent import DeviceCaptureAgent
from dronetracking.device.backend import CaptureBackend
from dronetracking.sim.observations import RangingRecord
from dronetracking.sources.socket_feed import SocketDeviceFeed

# --- synthetic capture parameters (deterministic; chosen to give clean detections) --- #
_SAMPLE_RATE_HZ = 16000.0
_DT_S = 0.5  # nominal emission spacing -> detector peak-separation guard
_N_EMISSIONS = 3
_SPEED_OF_SOUND_MPS = 343.0
_TOTAL_SAMPLES = 16000  # 1.0 s of audio


def _reference_pulse() -> np.ndarray:
    """A Hann-windowed linear chirp, unit-energy-normalised.

    Mirrors :func:`dronetracking.sim.audio.reference_pulse` (same shape the matched
    filter expects) but built locally so the test owns no scenario file.
    """
    n = int(round(0.02 * _SAMPLE_RATE_HZ))  # 20 ms pulse
    t = np.arange(n) / _SAMPLE_RATE_HZ
    dur = n / _SAMPLE_RATE_HZ
    f0, f1 = 1500.0, 5500.0
    k = (f1 - f0) / dur
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    pulse = np.hanning(n) * np.sin(phase)
    norm = np.linalg.norm(pulse)
    return pulse / norm if norm > 0.0 else pulse


_REFERENCE_PULSE = _reference_pulse()


def _synthesize(start_samples: Tuple[int, ...], t0_local_s: float) -> np.ndarray:
    """Place the reference pulse at each ``start`` sample offset in a zero waveform.

    The first sample is local-clock time ``t0_local_s``, so a pulse whose first sample is
    at index ``start`` is an arrival at ``t0_local_s + start / sr`` — the time the detector
    must recover. No noise: detection here is exact up to sub-sample peak rounding.
    """
    samples = np.zeros(_TOTAL_SAMPLES, dtype=float)
    m = _REFERENCE_PULSE.size
    for start in start_samples:
        samples[start : start + m] += _REFERENCE_PULSE
    return samples


class FakeBackend(CaptureBackend):
    """In-test :class:`CaptureBackend`: a hand-built waveform + injected ranging/GPS.

    Independent of agent A's ``device.backends`` so this test stands alone. ``record``
    returns the synthesized samples and a fixed ``t0_local_s``; ``ranging_records`` and
    ``gps`` return exactly what we hand it at construction.
    """

    def __init__(
        self,
        device_id: str,
        start_samples: Tuple[int, ...],
        t0_local_s: float,
        ranging: Tuple[RangingRecord, ...],
        gps_fix: Optional[Tuple[float, float, float]],
    ) -> None:
        self._device_id = device_id
        self._start_samples = start_samples
        self._t0_local_s = float(t0_local_s)
        self._ranging = ranging
        self._gps_fix = gps_fix

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def sample_rate_hz(self) -> float:
        return _SAMPLE_RATE_HZ

    def local_time(self) -> float:
        return self._t0_local_s

    def record(self, duration_s: float) -> Tuple[np.ndarray, float]:
        # Ignore duration_s for the fake: we return the whole pre-built waveform.
        return _synthesize(self._start_samples, self._t0_local_s), self._t0_local_s

    def ranging_records(self) -> Tuple[RangingRecord, ...]:
        return self._ranging

    def gps(self) -> Optional[Tuple[float, float, float]]:
        return self._gps_fix


def _make_device_plan():
    """Per-device synthetic plan: pulse offsets, t0, ranging records, optional GPS.

    Pulse offsets are spaced > ``_DT_S`` apart (and away from the buffer edges) so the
    detector's peak-separation guard cleanly resolves exactly ``_N_EMISSIONS`` arrivals
    in time order. Two anchors carry a GPS fix; one device has none.
    """
    sr = _SAMPLE_RATE_HZ
    plan = {
        "dev0": {
            # arrivals at ~0.1, 0.7, 1.4 emission-spacings apart in samples
            "starts": (1000, 5000, 9000),
            "t0": 0.0,
            "ranging": (
                RangingRecord("dev0", "dev1", 0, 0.001000, 0.587000, 0.589000, 0.001500),
                RangingRecord("dev0", "dev2", 0, 0.002000, 0.301000, 0.303000, 0.002600),
            ),
            "gps": (32.0853, 34.7818, 0.0),
        },
        "dev1": {
            "starts": (1500, 5600, 9700),
            "t0": 0.12,  # a non-zero local-clock offset (devices aren't synced)
            "ranging": (
                RangingRecord("dev1", "dev2", 0, 0.121000, 0.405000, 0.407000, 0.121800),
            ),
            "gps": (32.0871, 34.7835, 18.0),
        },
        "dev2": {
            "starts": (2000, 6100, 10200),
            "t0": -0.08,
            "ranging": (),  # a device that initiated no ranging this batch
            "gps": None,  # no GPS on this device
        },
    }
    # Expected local-clock arrival times each device should report (t0 + start/sr).
    for d in plan.values():
        d["expected_toa"] = tuple(d["t0"] + s / sr for s in d["starts"])
    return plan


def test_capture_agent_loopback_publishes_detected_batch():
    plan = _make_device_plan()
    device_ids = tuple(plan.keys())  # fixes the coordinator's expected order

    feed = SocketDeviceFeed(host="127.0.0.1", port=0)
    collector = threading.Thread(
        target=lambda: feed.collect(device_ids, timeout_s=30.0), daemon=True
    )
    collector.start()

    # Each device runs its own agent against the coordinator's port, in its own thread.
    agent_threads = []
    for did in device_ids:
        spec = plan[did]
        backend = FakeBackend(
            did,
            start_samples=spec["starts"],
            t0_local_s=spec["t0"],
            ranging=spec["ranging"],
            gps_fix=spec["gps"],
        )
        agent = DeviceCaptureAgent(
            backend,
            reference_pulse=_REFERENCE_PULSE,
            n_emissions=_N_EMISSIONS,
            dt_s=_DT_S,
            speed_of_sound_mps=_SPEED_OF_SOUND_MPS,
        )
        th = threading.Thread(
            target=agent.run,
            args=("127.0.0.1", feed.port),
            kwargs={"duration_s": _TOTAL_SAMPLES / _SAMPLE_RATE_HZ},
            daemon=True,
        )
        th.start()
        agent_threads.append(th)

    for th in agent_threads:
        th.join(timeout=30.0)
        assert not th.is_alive(), "an agent thread did not finish in time"
    collector.join(timeout=30.0)
    assert not collector.is_alive(), "collector thread did not finish in time"

    feed.close()

    obs = feed.as_observations()

    # --- timebase carried through the protocol --------------------------------------- #
    assert obs.sample_rate_hz == _SAMPLE_RATE_HZ
    assert obs.speed_of_sound_mps == _SPEED_OF_SOUND_MPS

    # --- every device reported, in the expected order -------------------------------- #
    assert feed.device_ids() == device_ids

    # --- acoustic arrivals: detected times within a few samples of where we placed them #
    tol_s = 3.0 / _SAMPLE_RATE_HZ  # within a few samples
    for did in device_ids:
        det = sorted(
            (a for a in obs.acoustic if a.device_id == did),
            key=lambda a: a.emission_idx,
        )
        assert len(det) == _N_EMISSIONS, f"{did}: wrong arrival count"
        # emission_idx is assigned in time order, 0..n-1
        assert [a.emission_idx for a in det] == list(range(_N_EMISSIONS))
        recovered = [a.toa_local_s for a in det]
        expected = sorted(plan[did]["expected_toa"])
        for got, want in zip(recovered, expected):
            assert abs(got - want) <= tol_s, (
                f"{did}: detected toa {got} not within {tol_s}s of {want}"
            )

    # --- ranging records passed through exactly (per-device, then unioned) ------------ #
    for did in device_ids:
        got = tuple(r for r in obs.ranging if r.initiator == did)
        assert got == plan[did]["ranging"], f"{did}: ranging mismatch"
    total_ranging = sum(len(plan[d]["ranging"]) for d in device_ids)
    assert len(obs.ranging) == total_ranging

    # --- GPS fixes passed through; the non-anchor contributes none -------------------- #
    gps_by_device = {g.device_id: g for g in obs.anchor_gps}
    assert "dev2" not in gps_by_device  # no GPS on dev2
    for did in ("dev0", "dev1"):
        fix = plan[did]["gps"]
        g = gps_by_device[did]
        assert (g.lat, g.lon, g.altitude_m) == fix
    assert len(obs.anchor_gps) == 2


def test_run_returns_encoded_bytes_round_trippable():
    """``run`` returns exactly the bytes put on the wire; they decode to this batch."""
    from dronetracking.live import protocol

    plan = _make_device_plan()
    feed = SocketDeviceFeed(host="127.0.0.1", port=0)
    collector = threading.Thread(
        target=lambda: feed.collect(("dev0",), timeout_s=30.0), daemon=True
    )
    collector.start()

    spec = plan["dev0"]
    backend = FakeBackend(
        "dev0",
        start_samples=spec["starts"],
        t0_local_s=spec["t0"],
        ranging=spec["ranging"],
        gps_fix=spec["gps"],
    )
    agent = DeviceCaptureAgent(
        backend,
        reference_pulse=_REFERENCE_PULSE,
        n_emissions=_N_EMISSIONS,
        dt_s=_DT_S,
        speed_of_sound_mps=_SPEED_OF_SOUND_MPS,
    )
    sent = agent.run("127.0.0.1", feed.port, duration_s=_TOTAL_SAMPLES / _SAMPLE_RATE_HZ)
    collector.join(timeout=30.0)
    assert not collector.is_alive()
    feed.close()

    assert isinstance(sent, bytes) and sent.endswith(protocol.LINE_DELIMITER)
    decoded = protocol.decode_batch(sent)
    assert decoded["device_id"] == "dev0"
    assert decoded["sample_rate_hz"] == _SAMPLE_RATE_HZ
    assert decoded["speed_of_sound_mps"] == _SPEED_OF_SOUND_MPS
    assert len(decoded["acoustic"]) == _N_EMISSIONS
    assert decoded["ranging"] == spec["ranging"]
    assert len(decoded["anchor_gps"]) == 1
