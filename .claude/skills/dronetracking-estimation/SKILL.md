---
name: dronetracking-estimation
description: >-
  Developer reference for the DroneTracking estimation pipeline — the
  GPS-free, clock-sync-free localization algorithms under
  src/dronetracking/estimation/. Load this whenever you read, debug, or modify
  acoustic ranging→distance matrices, MDS relative layout, clock-offset/drift
  recovery, TDOA target localization, Kalman tracking (single & multi-target),
  joint clock+position estimation, georeferencing, GPS-denied blending,
  matched-filter detection, or overlapping-source separation. Also load when a
  bug touches covariances/GDOP/NEES, gauge or reflection ambiguity, weighted
  least squares, or the ground-truth firewall on the estimation side. Trigger on
  file names like tdoa.py, clock_sync.py, relative_localization.py, ranging.py,
  tracking.py, online_tracker.py, multi_target.py, joint_clock.py,
  georeference.py, gps_denied.py, detection.py, separation.py, interfaces.py.
---

# DroneTracking — Estimation pipeline

The `estimation/` package turns raw `Observations` into georeferenced tracks. It
**never imports `sim`** (firewall — `tests/test_no_truth_leak.py` enforces it);
it consumes duck-typed observation objects and returns the types in
`interfaces.py`. Every stage produces honest covariances (empirical residual
variance scaling, NaN/gauge-safe).

## Stage cascade (entry function per stage)

```
Observations
 ├─ ranging.build_distance_matrix(obs) ───────────────→ DistanceMatrix
 ├─ relative_localization.estimate_layout(dm) ────────→ RelativeLayout   (gauge-free 3D)
 ├─ clock_sync.estimate_clocks(obs, reference_id=None)→ ClockEstimates   (relative to ref)
 ├─ tdoa.localize_all(obs, clocks, layout) ───────────→ List[TargetFix]  (≥4 devices/emission)
 ├─ tracking.track_target(fixes, model="cv") ─────────→ Track            (Kalman, NIS-gated)
 └─ georeference.solve_transform(layout, anchors, origin) + georeference_track(track, sim, origin) → GeoTrack
```

Variants (selected by `pipeline.py` from scenario features):
- **Multi-target (Ph6):** `multi_target.localize_frames(...)` → `track_targets(frames)` (GNN association via `scipy.optimize.linear_sum_assignment`, χ² gate, birth/death). Streaming equivalent: `online_tracker.OnlineTracker.update(fixes, t)` / `.tracks()` — identical per-frame logic; only difference is the online tracker assigns a **stable** id at confirmation while batch relabels `T0..Tn` at the end.
- **Joint clock (Ph8):** `joint_clock.localize_all_joint(arrivals, clocks, layout, c, *, clock_prior_s=1e-4, min_devices=5)` — co-estimates one residual offset per device with all emission positions; identifiable only across multiple emissions; Schur-complement marginal covariance.
- **Moving devices (Ph3):** `geometry_tracking.track_geometry(records, ids, c, window_s, step_s, smooth=True)` → `[(t_center, RelativeLayout)]`; chained Umeyama alignment + stable-anchor re-anchoring + optional per-device CV Kalman smoothing.
- **Detection (Ph4):** `detection.detect_arrivals(captures, reference_pulse, n_emissions, dt_s)` → `DetectedArrival(device_id, emission_idx, toa_local_s, confidence)`. Matched filter; confidence = `1 - exp(-z/8)` on peak deflection above noise floor (NOT raw peak height).
- **Separation (Ph6 overlap):** `separation.separate_arrivals(captures, reference_pulses: {source: pulse}, n_emissions, dt_s)` → per-source arrivals; `to_acoustic_arrivals(...)` flattens to source-tagged firewall-clean records.
- **GPS-denied (Ph9):** `gps_denied.georeference_with_blackout(layout, anchors, track, origin, blackout_windows, recovery_blend_s=2.0)` → `BlackoutGeoTrack` (per-frame `dead_reckoned`/`gps_available`); holds last good transform during blackout, blends **applied ENU positions** (not transforms) on return for C⁰ continuity.

## Output types (`interfaces.py`)

- `RelativeLayout(device_ids, positions_local(N,3), covariances(N,3,3)?)` — `.position_of(id)`. Arbitrary gauge-free frame.
- `ClockEstimates(device_ids, offsets_s, drifts_ppm, reference_id, covariances?)` — `.to_reference(id, local_time)` maps a device's local timestamp onto the reference timebase. Convention: `local = t·(1+ppm·1e-6) + offset`.
- `Track(times_s, positions_local(T,3), covariances(T,3,3), velocities?, target_id?)`.
- `GeoTrack(times_s, latlon(T,2), altitude_m(T,), cov_enu(T,3,3))`.
- `Estimates(layout, clocks, track, geo_track)` — the full bundle.

## How each core stage works (so you don't re-derive it)

- **Ranging → distances:** per pair, distance `= c·½((t4−t1)−(t3−t2))`; robust collapse over rounds (median/MAD, k=3, with an absolute floor because noise-free MAD≈0); triangle-inequality screen blames the longest violating edge. Variance floor `1e-9 m²` keeps `W=1/var` finite.
- **Layout (MDS):** shortest-path bootstrap for missing edges (seed only) → classical MDS double-centering → weighted nonlinear refine (`least_squares`, `soft_l1`, `trf`) on `(‖pᵢ−pⱼ‖−dᵢⱼ)·√W` → centroid gauge. Covariance via `gn_covariance` with reduced-χ² scaling; dof = `#edges − (3K−6)` (6 rigid DOF unobservable). Reflection is allowed — distance geometry is chirality-blind.
- **Clock sync:** per-pair Theil-Sen regression of per-exchange offset vs `t1` (slope ≈ relative skew, intercept ≈ relative offset; two-way ToF is offset/skew-immune) → least-squares over the clock graph with the reference pinned to (0,0) → one skew-bias correction pass.
- **TDOA:** lift each arrival to the reference timebase via `clocks.to_reference`; earliest-arriving device is the TDOA reference; Chan-style linear seed → nonlinear refine (`soft_l1`); covariance `s²·pinv(JᵀJ)` with `s²=max(SSR/dof, 1.0)` (floor at 1 for honest NEES); GDOP + residual RMS reported. `MIN_DEVICES_3D = 4`.
- **Tracking:** per-axis constant-velocity (`"cv"`, default) or constant-acceleration (`"ca"`) Kalman filter; position-only measurements; **NIS gate** rejects updates with innovation χ² > 11.35 (99th pct, 3 dof). `track_target` is batch; `Tracker`/`OnlineTracker` are stateful.
- **Georeference:** `solve_transform` fits a similarity (rotation+translation, optional scale, reflection allowed) from ≥3 non-collinear local anchors to their ENU targets; `georeference_track` carries positions to lat/lon and rotates covariances `cov_enu = s²·R·cov_local·Rᵀ`.

## Gotchas

- Detection/separation return **own** types (`DetectedArrival`, `SeparatedArrival`, `AcousticArrivalLike`), field-compatible with `sim.observations.AcousticArrival` but never the sim type — the orchestrator maps them outside the firewall. Don't `isinstance`-check against sim types.
- A `TargetFix` with zero/near-singular covariance can break the multi-target gate's `solve(S, y)`; those pairs are silently skipped.
- Joint-clock with a single emission is prior-dominated (rank-deficient) — needs multiple emissions to identify offsets.
- Moving-device velocities are biased toward the moving center of mass unless ≥3–4 stable anchors qualify for re-anchoring.
- All nonlinear refines use `soft_l1` robust loss — don't swap to linear `least_squares` expecting the same outlier tolerance.

Tests: `test_est_*`, `test_tdoa`-style, `test_tracking`, `test_multi_target`, `test_online_tracker`, `test_joint_clock`, `test_geometry_tracking`, `test_gps_denied`, `test_detection`, `test_separation`, `test_no_truth_leak`.
