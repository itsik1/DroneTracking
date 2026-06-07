---
name: dronetracking-eval
description: >-
  Developer reference for measuring and visualizing accuracy in DroneTracking —
  the eval/ package (the ONLY place ground truth meets estimates), the viz/
  package (folium maps + matplotlib diagnostics), and the studies/ package
  (parameter sweeps over noise and device count). Load this whenever you compute
  or interpret accuracy metrics (device-localization RMSE, tracking RMSE,
  georeferencing error, clock-sync error, NEES), align an estimated frame to
  truth, render the output/*.html maps or PNG plots, run or extend a robustness
  sweep, or add a new phase-specific metric. Trigger on file names metrics.py,
  alignment.py, phase_metrics.py, report.py, map_view.py, plots.py, sweep.py, or
  tests test_eval, test_viz, test_viz_animated, test_studies; and on the CLI
  `python -m dronetracking.studies`.
---

# DroneTracking — Evaluation, visualization & studies

`eval/` is the **only** place ground truth (`World`) meets estimates — the
firewall's comparison point. `viz/` renders artifacts; `studies/` quantifies how
accuracy moves as a knob turns. All three live downstream of the firewall (they
may see truth; `estimation/` may not).

## eval/ — metrics (the comparison point)

- `alignment.align_to_truth(estimated(N,3), truth(N,3)) -> Similarity` — rigid fit with `scale=1.0`, `allow_reflection=True` (distance geometry is chirality-blind, so the estimated layout frame must be aligned to truth before scoring).
- `metrics.compute_metrics(world, observations, estimates) -> dict` — a **flat** `"group.name"` dict (trivial JSON; every block wrapped so a degraded run yields `NaN` rather than raising). Groups:
  - `scenario.*` — name, n_devices, n_anchors.
  - `device_localization.*` — `rmse_m`, `max_m`, `alignment_scale`, `alignment_was_reflected`, per-device `error_m.<id>` (after `align_to_truth`).
  - `clock_sync.*` — `offset_rmse_s`, `drift_rmse_ppm` (both gauge-removed to the reference device first).
  - `tracking.*` — `rmse_m`, `rmse_xy_m`, `rmse_z_m`, `final_error_m`, `nees_mean` (mean `eᵀP⁻¹e`; honest covariances → ≈3). The track is mapped from its arbitrary layout frame into truth via the device-geometry alignment before scoring.
  - `georeferencing.*` — `rmse_m` (haversine between estimated/true lat/lon), `altitude_rmse_m`.
- `phase_metrics.py` — additive, scenario-feature-specific:
  - `multi_target_metrics(world, tracks, layout)` → `multi_target.{n_true, n_tracks, mean_rmse_m, max_rmse_m, rmse_m.src<k>}` (matches each true drone to nearest track).
  - `geometry_metrics(world, geometry_series)` → `geometry.{n_windows, mean_window_rmse_m, max_window_rmse_m}` (truth sampled at each window center).
  - `gps_denied_metrics(world, geo_track, blackout_windows)` → `gps_denied.{rmse_available_m, rmse_blackout_m, max_step_m}` (error split by GPS-up vs dead-reckoned; `max_step_m` is the largest frame-to-frame jump → continuity check).
- `report.py` — `print_report(metrics)` (grouped, unit-annotated scorecard) and `save_report(metrics, path)` (JSON with NaN/Inf support, no silent loss).

`pipeline.run_pipeline` calls `compute_metrics` only when `world is not None` (i.e. the feed is simulated) and appends the phase metrics that apply. Real feeds (recorded/socket/live) have no truth, so `metrics` is empty there.

## viz/ — artifacts (folium + matplotlib, headless)

- `map_view.render_map(world, estimates, scenario, out_path, geo_tracks=None) -> Path` — interactive folium/OSM map (no API key): true vs estimated device positions, GPS anchors, drone track(s), 95% confidence ellipses; `geo_tracks` triggers a multi-target palette.
- `map_view.render_animated_map(world, estimates, scenario, out_path)` — Leaflet TimeDimension slider; tracks draw progressively into an uncertainty corridor.
- `plots.save_diagnostics(world, estimates, scenario, out_dir, geo_tracks=None) -> List[Path]` — 4 PNGs (`Agg` backend): `local_frame_devices.png`, `device_localization_error.png`, `tracking_error_over_time.png`, `trajectory_topdown.png`.

`run.py` wires all of these into `output/` (suppress with `--no-map` / `--no-plots`). viz uses a **duck-typed** world (expected attributes only), never importing `sim.World` directly.

## studies/ — parameter sweeps

```bash
.venv/bin/python -m dronetracking.studies --scenario scenarios/field_5dev.yaml --kind noise   --factors 0.5 1.0 2.0 4.0 --seeds 3
.venv/bin/python -m dronetracking.studies --scenario scenarios/field_5dev.yaml --kind devices --counts 4 5 --seeds 3
```

- `sweep.sweep_noise(base_scenario, factors, seeds=3)` — scales every `NoiseSpec` field by a common factor (via `dataclasses.replace`); more noise must degrade accuracy (monotonic sanity check).
- `sweep.sweep_devices(base_scenario, counts, seeds=3)` — varies participating device count; **skips counts that can't keep ≥4 GPS anchors** (georef would be ill-posed); reports GDOP alongside RMSEs.
- `sweep.plot_sweep(result, out_dir, title=None)` — error-vs-parameter curves.
- CLI (`__main__.py`): `--scenario`, `--kind {noise,devices}`, `--seeds`, `--factors`, `--counts`, `--out-dir`. Emits `<name>_<kind>_sweep.json` + `.png` and prints a mean-metrics table.

Headline finding the sweeps confirm: tracking & device localization stay cm-level across a 4× noise range, while georeferencing scales with GPS-anchor noise — the acoustic core is robust and accuracy is GPS-anchor-bound.

## Conventions

- Metrics are flat `"group.name"` keys so JSON diffs across runs are trivial — keep that convention when adding metrics, and wrap new blocks so failures degrade to `NaN`.
- Always align the estimated frame to truth (reflection allowed, no scale) before any position error — raw layout coordinates are gauge-free and not comparable.
- Report vertical separately from horizontal (`rmse_z_m` vs `rmse_xy_m`); a near-ground array has weak vertical observability and a single error radius would hide it.
- `nees_mean ≈ 3` is the honesty check on tracking covariances; far from 3 means over/under-confident covariances, not necessarily worse RMSE.
