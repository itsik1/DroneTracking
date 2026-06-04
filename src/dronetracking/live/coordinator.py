"""Distributed runtime coordinator: assemble per-device feeds over TCP, then estimate.

    python -m dronetracking.live.coordinator --scenario scenarios/field_5dev.yaml

Each device publishes only its own measurements to this coordinator over a socket; the
coordinator reassembles them into the standard ``Observations`` and runs the pipeline —
exactly the data path a real deployment uses (here the per-device data is simulated and
the agents run as local threads, but the wire protocol and assembly are real). On real
hardware, run ``DeviceAgent.publish(...)`` on each device against this coordinator's host.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from ..config import load_scenario
from ..pipeline import run_pipeline
from ..sources.simulated import SimulatedDeviceFeed
from ..sources.socket_feed import SocketDeviceFeed
from .agent import DeviceAgent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dronetracking.live.coordinator",
        description="Devices publish measurements over TCP; assemble them and run the pipeline.",
    )
    p.add_argument("--scenario", required=True, type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="0 = OS-assigned free port")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario, seed_override=args.seed)
    device_ids = [d.id for d in scenario.devices]
    shared = SimulatedDeviceFeed(scenario)  # identical seeded data behind every agent

    feed = SocketDeviceFeed(host=args.host, port=args.port)
    print(f"coordinator listening on {args.host}:{feed.port} — expecting {len(device_ids)} devices")

    collector = threading.Thread(
        target=lambda: feed.collect(device_ids, timeout_s=args.timeout), daemon=True
    )
    collector.start()

    agents = []
    for did in device_ids:
        th = threading.Thread(
            target=DeviceAgent().publish,
            args=(args.host, feed.port, scenario, did),
            kwargs={"feed": shared},
            daemon=True,
        )
        th.start()
        agents.append(th)
    for th in agents:
        th.join(timeout=args.timeout)
    collector.join(timeout=args.timeout)

    obs, ref = feed.as_observations(), shared.as_observations()
    matches = (
        len(obs.ranging) == len(ref.ranging)
        and len(obs.acoustic) == len(ref.acoustic)
        and len(obs.anchor_gps) == len(ref.anchor_gps)
    )
    print(
        f"assembled over sockets: {len(obs.ranging)} ranging, {len(obs.acoustic)} arrivals, "
        f"{len(obs.anchor_gps)} GPS — matches simulator: {matches}"
    )

    # Estimate purely from the distributed feed (a live feed carries no ground truth).
    result = run_pipeline(scenario, feed=feed)
    geo_pts = len(result.estimates.geo_track.latlon)
    print(f"pipeline over distributed feed -> {len(result.tracks)} track(s), {geo_pts} georeferenced points")

    try:
        feed.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
