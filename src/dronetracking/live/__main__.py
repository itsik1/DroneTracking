"""CLI: launch the live dashboard for a scenario.

    python -m dronetracking.live --scenario scenarios/multi_drone.yaml
    python -m dronetracking.live --scenario scenarios/field_5dev.yaml --speed 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_scenario
from .server import serve


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dronetracking.live",
        description="Stream a scenario to a live browser dashboard (the hardware-facing view).",
    )
    p.add_argument("--scenario", required=True, type=Path, help="scenario YAML to stream")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (>1 = faster)")
    p.add_argument("--seed", type=int, default=None, help="override the scenario RNG seed")
    p.add_argument("--detect", action="store_true", help="localize from synthesized audio (Ph4)")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario, seed_override=args.seed)
    serve(scenario, host=args.host, port=args.port, speed=args.speed, detect=args.detect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
