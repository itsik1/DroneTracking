"""CLI for the robustness / accuracy studies.

    python -m dronetracking.studies --scenario scenarios/field_5dev.yaml \
        --kind noise --out-dir output

Runs the requested sweep over ``scenarios/<...>.yaml``, writes ``<name>_<kind>_sweep.json``
and the matching PNG(s) into ``--out-dir``, and prints a short table to stdout.

Flags:
  --scenario PATH    scenario YAML to sweep (required)
  --kind {noise,devices}   which sweep (default: noise)
  --seeds N          random seeds per grid point (default: 3)
  --factors ...      noise scale factors (noise sweep; default: 0.5 1.0 2.0 4.0)
  --counts ...       device counts (devices sweep; default: every count from 4..all)
  --out-dir DIR      output directory (default: output)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .. import config
from . import sweep


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dronetracking.studies",
        description="Run a robustness/accuracy parameter sweep over the pipeline.",
    )
    p.add_argument("--scenario", required=True, help="path to a scenario YAML file")
    p.add_argument(
        "--kind", choices=["noise", "devices"], default="noise",
        help="which sweep to run (default: noise)",
    )
    p.add_argument(
        "--seeds", type=int, default=3,
        help="random seeds per grid point (default: 3)",
    )
    p.add_argument(
        "--factors", type=float, nargs="+", default=None,
        help="noise scale factors for --kind noise (default: 0.5 1.0 2.0 4.0)",
    )
    p.add_argument(
        "--counts", type=int, nargs="+", default=None,
        help="device counts for --kind devices (default: 4 .. all devices)",
    )
    p.add_argument(
        "--out-dir", default="output",
        help="directory for the JSON + PNG outputs (default: output)",
    )
    return p


def _fmt(value: Any) -> str:
    """Format a metric/GDOP cell for the printed table."""
    if value is None:
        return "    n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "    nan"
    return f"{f:8.4f}"


def _print_table(result: Dict[str, Any]) -> None:
    """Pretty-print the sweep as an aligned text table (mean of each metric)."""
    param = result["param"]
    metric_keys: List[str] = result["metric_keys"]
    kind = result["kind"]

    # Short column headers from the metric keys (drop the trailing "_m").
    short = {
        "tracking.rmse_m": "track_rmse",
        "device_localization.rmse_m": "dev_rmse",
        "georeferencing.rmse_m": "geo_rmse",
    }
    cols = [short.get(k, k) for k in metric_keys]
    has_gdop = kind == "devices"

    header = f"{param:>14}" + "".join(f"{c:>12}" for c in cols)
    if has_gdop:
        header += f"{'gdop':>12}"
    print(f"\n# {result['scenario']} — {kind} sweep (seeds={result['seeds']})")
    print(header)
    print("-" * len(header))

    for pt in result["points"]:
        row = f"{pt[param]:>14}"
        for key in metric_keys:
            row += f"{_fmt(pt['mean'].get(key)):>12}"
        if has_gdop:
            row += f"{_fmt(pt.get('gdop')):>12}"
        print(row)

    skipped = result.get("skipped")
    if skipped:
        print(f"\nskipped (need >={sweep.MIN_ANCHORS} anchors): {skipped}")


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    scenario = config.load_scenario(args.scenario)

    if args.kind == "noise":
        factors = args.factors if args.factors is not None else [0.5, 1.0, 2.0, 4.0]
        result = sweep.sweep_noise(scenario, factors=factors, seeds=args.seeds)
    else:  # devices
        if args.counts is not None:
            counts = args.counts
        else:
            counts = list(range(sweep.MIN_ANCHORS, len(scenario.devices) + 1))
        result = sweep.sweep_devices(scenario, counts=counts, seeds=args.seeds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{scenario.name}_{args.kind}_sweep.json"
    json_path.write_text(json.dumps(result, indent=2))

    png_paths = sweep.plot_sweep(
        result, out_dir, title=f"{scenario.name}: accuracy vs {result['param']}"
    )

    _print_table(result)
    print(f"\nwrote {json_path}")
    for p in png_paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
