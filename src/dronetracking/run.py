"""Command-line entry point: run the full pipeline on a scenario and emit artifacts.

    python -m dronetracking.run --scenario scenarios/field_5dev.yaml

Writes a folium map (HTML), matplotlib diagnostic plots (PNG), and a metrics JSON to
the output directory, and prints the metrics report to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_scenario
from .pipeline import run_pipeline
from .eval.report import print_report, save_report
from .viz.map_view import render_map, render_animated_map
from .viz.plots import save_diagnostics


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dronetracking",
        description="Run the distributed acoustic drone-localization pipeline on a scenario.",
    )
    p.add_argument("--scenario", required=True, type=Path, help="path to a scenario YAML file")
    p.add_argument("--out-dir", default=Path("output"), type=Path, help="directory for artifacts")
    p.add_argument("--seed", type=int, default=None, help="override the scenario's RNG seed")
    p.add_argument("--model", default="cv", choices=["cv", "ca"], help="tracking motion model")
    p.add_argument("--sigma-a", type=float, default=2.0, help="tracking process-noise accel std")
    p.add_argument("--detect", action="store_true", help="localize from synthesized audio via DSP detection (Ph4)")
    p.add_argument("--joint-clock", action="store_true", help="co-estimate residual clock offsets with position (single-target)")
    p.add_argument("--clock-prior-s", type=float, default=1e-4, help="prior std for joint clock offsets (s)")
    p.add_argument("--no-map", action="store_true", help="skip the folium map")
    p.add_argument("--no-plots", action="store_true", help="skip the matplotlib diagnostics")
    args = p.parse_args(argv)

    scenario = load_scenario(args.scenario, seed_override=args.seed)
    result = run_pipeline(scenario, model=args.model, sigma_a=args.sigma_a, detect=args.detect,
                          joint_clock=args.joint_clock, clock_prior_s=args.clock_prior_s)

    print_report(result.metrics)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_report(result.metrics, out_dir / f"{scenario.name}_metrics.json")

    if not args.no_map:
        map_path = render_map(result.world, result.estimates, scenario, out_dir / f"{scenario.name}_map.html", geo_tracks=result.geo_tracks)
        anim_path = render_animated_map(result.world, result.estimates, scenario, out_dir / f"{scenario.name}_animated.html")
        print(f"\nmap:     {map_path}")
        print(f"animated:{anim_path}")
    if not args.no_plots:
        plot_paths = save_diagnostics(result.world, result.estimates, scenario, out_dir, geo_tracks=result.geo_tracks)
        print(f"plots:   {len(plot_paths)} written to {out_dir}/")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
