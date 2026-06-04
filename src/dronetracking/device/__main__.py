"""Run ONE device's capture agent against a coordinator.

    python -m dronetracking.device --coordinator HOST:PORT --scenario S --id devN \
        [--real] [--duration D]

This is the on-device entry point: it builds a :class:`CaptureBackend`, wraps it in a
:class:`~dronetracking.device.agent.DeviceCaptureAgent`, captures->detects->publishes its
own measurement slice to the coordinator's
:class:`~dronetracking.sources.socket_feed.SocketDeviceFeed`, and prints what it sent.

The scenario supplies the detector's knobs (the KNOWN reference pulse, the emission
count, and the inter-emission spacing) and the timebase the wire protocol carries:

* ``reference_pulse`` <- :func:`dronetracking.sim.audio.reference_pulse(scenario)`
* ``n_emissions``     <- ``len(dronetracking.sim.acoustic.emission_times(scenario))``
* ``dt_s``            <- ``scenario.dt_s``
* ``speed_of_sound_mps`` <- ``scenario.speed_of_sound_mps``

Backend selection:

* default -> :class:`dronetracking.device.backends.MockBackend(scenario, id)` — the
  deterministic sim-driven backend (its synthesized waveform reproduces the scenario's
  arrivals when detected),
* ``--real`` -> :class:`dronetracking.device.backends.SoundDeviceBackend(id, sample_rate_hz=...)`
  — a real microphone (needs the ``sounddevice`` library; raises a clear, actionable
  error if it's unavailable).

Both backends live in agent A's ``device.backends`` module; they're imported lazily here
so a missing/optional dependency surfaces only when that backend is actually selected.

Pair with the coordinator side:

    python -m dronetracking.live.coordinator --scenario scenarios/detection_demo.yaml --port 9100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_scenario
from ..sim.acoustic import emission_times
from ..sim.audio import reference_pulse
from .agent import DeviceCaptureAgent


def _parse_host_port(value: str) -> tuple[str, int]:
    """Parse ``HOST:PORT`` (the coordinator address) into ``(host, port)``.

    The host may itself contain colons (an IPv6 literal); only the final colon separates
    the port, so we split from the right exactly once.
    """
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"--coordinator must be HOST:PORT (got {value!r})"
        )
    host, _, port_str = value.rpartition(":")
    if not host or not port_str:
        raise argparse.ArgumentTypeError(
            f"--coordinator must be HOST:PORT (got {value!r})"
        )
    try:
        port = int(port_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--coordinator port must be an integer (got {port_str!r})"
        )
    return host, port


def _build_backend(args, scenario):
    """Construct the selected :class:`CaptureBackend` (lazy import of agent A's module).

    Imported here, not at module top, so that selecting the mock backend never drags in
    the optional ``sounddevice`` dependency, and a clear error only surfaces when the
    real backend is actually requested.
    """
    from . import backends  # agent A's module: MockBackend + SoundDeviceBackend

    if args.real:
        return backends.SoundDeviceBackend(
            args.id, sample_rate_hz=scenario.sample_rate_hz
        )
    return backends.MockBackend(scenario, args.id)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dronetracking.device",
        description="Capture on a device, detect drone arrivals, and publish to the coordinator.",
    )
    p.add_argument(
        "--coordinator",
        required=True,
        type=_parse_host_port,
        metavar="HOST:PORT",
        help="coordinator address, e.g. 127.0.0.1:9100",
    )
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--id", required=True, help="this device's id, e.g. dev0")
    p.add_argument(
        "--real",
        action="store_true",
        help="use the real microphone (SoundDeviceBackend) instead of the mock",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="seconds of audio to record (default: the scenario duration)",
    )
    p.add_argument("--seed", type=int, default=None, help="scenario seed override")
    args = p.parse_args(argv)

    host, port = args.coordinator
    scenario = load_scenario(args.scenario, seed_override=args.seed)

    # Detector knobs + timebase straight off the scenario (same template the synthesizer
    # used, so detection on the mock backend reproduces the scenario's arrivals).
    pulse = reference_pulse(scenario)
    n_emissions = int(len(emission_times(scenario)))
    duration_s = float(args.duration) if args.duration is not None else float(scenario.duration_s)

    backend = _build_backend(args, scenario)
    agent = DeviceCaptureAgent(
        backend,
        reference_pulse=pulse,
        n_emissions=n_emissions,
        dt_s=scenario.dt_s,
        speed_of_sound_mps=scenario.speed_of_sound_mps,
    )

    backend_name = type(backend).__name__
    print(
        f"[{args.id}] capturing {duration_s:.3f}s via {backend_name} "
        f"(sr={backend.sample_rate_hz:g} Hz, expecting {n_emissions} emissions)"
    )

    # Build the slice once so we can summarize it, then publish the same data.
    acoustic, ranging, anchor_gps = agent.detect(duration_s)
    gps_desc = (
        f"{anchor_gps[0].lat:.6f},{anchor_gps[0].lon:.6f}@{anchor_gps[0].altitude_m:.1f}m"
        if anchor_gps
        else "none"
    )
    print(
        f"[{args.id}] detected {len(acoustic)} acoustic arrival(s), "
        f"{len(ranging)} ranging record(s), gps={gps_desc}"
    )

    message = agent.run(host, port, duration_s=duration_s)
    print(f"[{args.id}] sent {len(message)} bytes to coordinator at {host}:{port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
