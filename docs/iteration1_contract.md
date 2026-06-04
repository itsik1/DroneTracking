# Iteration 1 — Integration Contract (read this first)

This is the **single source of truth** for module boundaries, data types, signatures,
and numerical conventions. Every parallel agent builds to this so the pieces integrate.

## Project facts

- Repo root: `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`
- Package: `dronetracking` (src layout under `src/dronetracking/`), installed editable.
- **Run tests with the venv:** `.venv/bin/python -m pytest tests/<your_file>.py -q`
- Python 3.9, numpy 2.0, scipy 1.13, matplotlib 3.9 (use `Agg` backend in tests), folium 0.20.
- TDD: write the test first, watch it fail, implement, watch it pass. Run **only your own
  test file(s)** so you are not blocked by modules other agents are building concurrently.

## Hard rules

- **Only create/edit the files assigned to you.** Do not touch any other file, especially
  the FROZEN files below. No edits to `pyproject.toml`, other agents' modules, or shared
  contract files.
- The estimation package must **never import from `dronetracking.sim`** in its *source*
  (the ground-truth firewall). Estimation **tests** MAY import the already-built sim leaf
  functions (see below) to build realistic fixtures — that's fine.
- Reuse the foundation/contract modules; do not reimplement geo/transform/dataclass logic.

## FROZEN modules (built, tested green — import and reuse, do NOT modify)

### `dronetracking.geo`
- `latlon_to_enu(lat, lon, origin) -> (east_m, north_m)`  (origin = `(lat, lon)` tuple; vectorized)
- `enu_to_latlon(east_m, north_m, origin) -> (lat, lon)`
- `haversine_m(lat1, lon1, lat2, lon2) -> meters`

### `dronetracking.transforms`
- `umeyama(src, dst, with_scaling=True, allow_reflection=False) -> Similarity`
  - `Similarity(R, t, scale)` with `.apply(points)` (single (3,) or (N,3)) and `.is_reflection`.
- `gdop(target, sensors) -> float`  (sensors (M,3))
- `gn_covariance(jac, residual_variance=1.0) -> cov`  (uses pinv; gauge-safe)

### `dronetracking.datatypes`
- `DistanceMatrix(device_ids, D, W, counts, valid)`; props `n_devices`, `n_valid_edges`.
  - `D` (K,K) meters symmetric, 0 diagonal, NaN where unmeasured; `W` (K,K)=1/variance, 0 if invalid/missing;
    `counts` (K,K) int; `valid` (K,K) bool.
- `TargetFix(position(3,), cov(3,3), gdop, residual_rms, n_devices, t)`;
  props `error_radius=sqrt(trace(cov))`, `vertical_std`, `horizontal_std`, `weak_vertical`.

### `dronetracking.estimation.interfaces`
- `RelativeLayout(device_ids, positions_local (N,3), covariances (N,3,3)|None)`; `n_devices`, `position_of(id)`.
- `ClockEstimates(device_ids, offsets_s: dict, drifts_ppm: dict, reference_id, covariances: dict|None=None)`
  - `to_reference(device_id, local_time) = (local_time - offsets_s[id]) / (1 + drifts_ppm[id]*1e-6)`
- `Track(times_s (T,), positions_local (T,3), covariances (T,3,3), velocities (T,3)|None)`; `final_position`.
- `GeoTrack(times_s (T,), latlon (T,2), altitude_m (T,), cov_enu (T,3,3))`.
- `Estimates(layout, clocks, track, geo_track)`.

### `dronetracking.sim.observations`
- `RangingRecord(initiator, responder, round_idx, t1_local_i, t2_local_j, t3_local_j, t4_local_i)`
- `AcousticArrival(device_id, emission_idx, toa_local_s)`
- `AnchorGps(device_id, lat, lon, altitude_m)`
- `Observations(device_ids, ranging, acoustic, anchor_gps, speed_of_sound_mps, sample_rate_hz)`

### `dronetracking.sim.scenario`, `dronetracking.config`
- `Scenario`, `DeviceSpec`, `TrajectorySpec`, `NoiseSpec` (see file). `Scenario.device_ids`, `Scenario.anchors`.
- `config.load_scenario(path, seed_override=None)`, `config.scenario_from_dict(d, seed_override=None)`.

### sim leaf functions (FROZEN; usable in estimation TESTS for fixtures)
- `dronetracking.sim.clocks.device_local_time(offset_s, drift_ppm, t_global)` and `global_from_local(...)`.
- `dronetracking.sim.trajectory.trajectory_position(scenario, t) -> (3,)`.
- `dronetracking.sim.ranging.generate_ranging_records(scenario, rng) -> tuple[RangingRecord]`.
- `dronetracking.sim.acoustic.generate_acoustic_arrivals(scenario, rng) -> tuple[AcousticArrival]`,
  `emission_times(scenario) -> (N,)`.
- Do NOT import `dronetracking.sim.simulator` (built concurrently by the SIM agent).

## LOCKED clock convention (everyone must obey)

A device's local clock reads `local = t_global * (1 + drift_ppm*1e-6) + offset_s`.
`ClockEstimates.to_reference` is the exact inverse. The estimator recovers each device's
`(offset, drift)` **relative to a reference device** (default `device_ids[0]`), with the
reference's recovered params `(0, 0)`. In all shipped scenarios `dev0`/`device_ids[0]` has
true offset 0 and drift 0, so `to_reference` recovers global time and TDOA differences are
exact in the noise-free case.

Two-way ranging cancels offset: `ToF = 0.5*((t4-t1) - (t3-t2))`, `distance = ToF * c`.
Per-exchange relative offset estimate: `0.5*((t2-t1) + (t3-t4))`, which is **linear in the
transmit time with slope = relative skew** (this is how drift is recovered).

## Public signatures each agent must expose (so `pipeline.py` can wire them)

> Keyword args / extra tuning params are fine; keep the names and core positional shape.

- SIM:
  - `dronetracking.sim.world.World(device_ids, device_positions: dict[str,(3,)], clock_offsets: dict,
    clock_drifts_ppm: dict, anchor_latlon: dict[str,(lat,lon)], origin_latlon, true_track (N,3),
    true_track_times (N,))` with `positions_matrix() -> (K,3)` in `device_ids` order.
  - `dronetracking.sim.simulator.simulate(scenario) -> (Observations, World)`.
- EST geometry:
  - `dronetracking.estimation.ranging.build_distance_matrix(observations) -> DistanceMatrix`
  - `dronetracking.estimation.relative_localization.estimate_layout(dm: DistanceMatrix) -> RelativeLayout`
- EST clocks:
  - `dronetracking.estimation.clock_sync.estimate_clocks(observations, reference_id=None) -> ClockEstimates`
- EST target:
  - `dronetracking.estimation.tdoa.localize_all(observations, clocks, layout) -> list[TargetFix]`
    (groups acoustic arrivals by `emission_idx`; each emission -> one `TargetFix`; skips emissions
    with < 4 devices). Also expose `localize_emission(arrivals, clocks, layout, speed_of_sound_mps) -> TargetFix`.
  - `dronetracking.estimation.tracking.track_target(fixes: list[TargetFix], model="cv", sigma_a=...) -> Track`
    (linear KF, R = fix.cov, NIS gating). Also a `Tracker` class wrapper as the multi-target seam.
- EST georef:
  - `dronetracking.estimation.georeference.solve_transform(layout, anchor_gps, origin_latlon, with_scaling=False) -> Similarity`
    (local frame -> ENU; uses anchor (e,n,alt) from GPS via geo). Requires >=3 non-collinear anchors;
    raise `ValueError` otherwise.
  - `dronetracking.estimation.georeference.georeference_track(track: Track, transform: Similarity, origin_latlon) -> GeoTrack`
- EVAL:
  - `dronetracking.eval.alignment.align_to_truth(estimated (N,3), truth (N,3)) -> Similarity` (umeyama,
    `allow_reflection=True`, `with_scaling=False`).
  - `dronetracking.eval.metrics.compute_metrics(world, observations, estimates) -> dict` (flat, JSON-able).
  - `dronetracking.eval.report.print_report(metrics)` and `save_report(metrics, path)`.
- VIZ:
  - `dronetracking.viz.map_view.render_map(world, estimates, scenario, out_path) -> Path` (folium HTML).
  - `dronetracking.viz.plots.save_diagnostics(world, estimates, scenario, out_dir) -> list[Path]` (matplotlib PNGs).

## Numerical guidance / pitfalls (do these honestly)

- **Relative localization**: classical MDS init (double-center sq-distances, `eigh`, top-3),
  then `scipy.optimize.least_squares` WLS refine, residual `(||pi-pj|| - d_ij)/sigma_ij`,
  `loss="soft_l1"`. Center the result (gauge). Covariance via `gn_covariance(res.jac, s2)`.
  Distances are gauge-free (rotation/translation/reflection arbitrary) — that's expected.
- **Clock sync**: per-pair Theil-Sen (`scipy.stats.theilslopes`) slope = relative skew; build a
  least-squares clock graph (per-device offset & skew rel. to reference) with `np.linalg.lstsq`.
- **TDOA**: closed-form linear seed (spherical-interpolation / Chan-style), then `least_squares`
  (`loss="soft_l1"`) refine. Covariance = `gn_covariance` / inverse Fisher. Compute `gdop`.
  **Report vertical variance separately** (a near-coplanar ground array barely observes altitude).
- **Tracking**: 9-state CA-capable layout `[x,y,z,vx,vy,vz,ax,ay,az]`, CV default (zero accel
  process), measurement is position with `R = fix.cov`. NIS chi-square gating to reject outliers.
- **Georeference**: full 3D `umeyama(with_scaling=False, allow_reflection=False)` from layout anchor
  positions to anchor ENU `(east, north, altitude)`. Then any local point -> ENU via `.apply` ->
  lat/lon via `geo.enu_to_latlon`, altitude = ENU up.
- **Eval alignment**: relative coords are arbitrary up to similarity *including reflection*, so use
  `allow_reflection=True` when scoring layout-vs-truth. Clock offset is observable only up to a global
  constant — compare offsets/drifts after subtracting the reference device's value on both sides.
- **NEES** (tracking consistency): mean of `e^T P^-1 e` over time should be ≈ 3 (state pos dim) if
  covariances are honest.

## Output expected from each agent

Return a short summary: the public functions you implemented (with final signatures), the test
file(s) you added, and the pytest result (counts). Note any deviation from this contract.
