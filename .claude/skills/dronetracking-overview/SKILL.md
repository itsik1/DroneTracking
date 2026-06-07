---
name: dronetracking-overview
description: >-
  Read-first orientation for the DroneTracking repo — a distributed acoustic
  drone-localization system (GPS-free self-localization via acoustic ranging,
  clock-sync-free TDOA target tracking, georeferencing to lat/lon). Load this
  WHENEVER you start work in this repo, are asked "how does this work / where
  is X / how do I run it", touch the top-level glue (config.py, datatypes.py,
  geo.py, transforms.py, pipeline.py, run.py), edit scenario YAML, run the
  pipeline/tests, or need to know the ground-truth firewall and which of the
  sibling dronetracking-* skills to open next. Use it even when the request
  doesn't name a specific module — it's the map of the whole project.
---

# DroneTracking — System Overview

A **simulation-first** testbed for a distributed acoustic drone-localization
network. Ordinary devices (phones, laptops) localize **themselves** without GPS
(acoustic ranging → relative layout), then **detect and track a drone**
acoustically via TDOA, and **georeference** the result onto a real map — all
**without synchronized clocks**. It began as a Python sim with known ground
truth (so accuracy is measurable at every stage) and now also has a zero-install
browser app for real phones, which is the **current focus**.

Environment: Python 3.9, venv at `.venv`. **ALWAYS use `.venv/bin/python`** —
the shell's `python` is the system one with no deps installed. Package source is
under `src/dronetracking/`. The full test suite (`311 passed`) is the real
acceptance gate.

## The one architectural rule: the ground-truth firewall

`sim/` **owns truth** (true positions, clock offsets/drifts, emission times,
trajectories) and emits only `Observations` — what a real device could actually
measure. `estimation/` is **structurally forbidden** from importing `sim` (or
truth of any kind); it consumes duck-typed `Observations` and produces
`Estimates`. `eval/` is the **only** place that compares the two. This is
enforced by `tests/test_no_truth_leak.py` (AST check: no `estimation/*.py`
imports `dronetracking.sim`; structural check: `Observations` exposes no
ground-truth fields). When adding code, respect this: anything that needs truth
goes in `sim/`, `eval/`, or the orchestrator (`pipeline.py`) — never in
`estimation/`.

## Directory map → which skill to open

| Area | Path | Open this skill |
|---|---|---|
| Self-localization & tracking algorithms (ranging→layout, clock sync, TDOA, tracking, georef, detection, separation, multi-target) | `src/dronetracking/estimation/` | **dronetracking-estimation** |
| The simulator & ground-truth owner; scenario YAML schema | `src/dronetracking/sim/` | **dronetracking-simulator** |
| Zero-install browser app for real phones (CURRENT FOCUS) | `src/dronetracking/webapp/` | **dronetracking-webapp** |
| Live streaming engine, distributed TCP runtime, DeviceFeed, capture backends, network mesh | `src/dronetracking/{live,device,sources,network}/` | **dronetracking-runtime** |
| Accuracy metrics, map/plot artifacts, parameter sweeps | `src/dronetracking/{eval,viz,studies}/` | **dronetracking-eval** |
| Top-level glue, scenarios, how to run | (this file) | — |

## The lingua-franca data types

These cross module boundaries; know them before changing signatures.

- `sim/observations.py` — `Observations(device_ids, ranging, acoustic, anchor_gps, speed_of_sound_mps, sample_rate_hz)`, the sim→estimation contract. Members:
  - `RangingRecord(initiator, responder, round_idx, t1_local_i, t2_local_j, t3_local_j, t4_local_i)` — one two-way SDS-TWR exchange; t1/t4 on initiator's clock, t2/t3 on responder's.
  - `AcousticArrival(device_id, emission_idx, toa_local_s, source=0, confidence=1.0)` — one drone-emission arrival at one device. The global emission time is **never** emitted (it cancels in TDOA differences).
  - `AnchorGps(device_id, lat, lon, altitude_m)` — a noisy GPS fix.
- `datatypes.py` (numpy-only, no sim imports) — intermediate plumbing:
  - `DistanceMatrix(device_ids, D, W, counts, valid)` — pairwise distances `D` (NaN where unmeasured), weights `W=1/var`, validity mask.
  - `TargetFix(position(3,), cov(3,3), gdop, residual_rms, n_devices, t)` — one TDOA fix; `.error_radius`, `.weak_vertical` helpers.
- `estimation/interfaces.py` — the estimation outputs: `RelativeLayout`, `ClockEstimates`, `Track`, `GeoTrack`, and the bundle `Estimates(layout, clocks, track, geo_track)`. **Clock convention (locked, shared with sim):** `local = t_global * (1 + drift_ppm*1e-6) + offset_s`; `ClockEstimates.to_reference(device_id, local_time)` inverts it.

## Top-level glue

- `config.py` — `load_scenario(path, seed_override=None) -> Scenario`, `scenario_from_dict(raw, ...)`. Validates devices, unique ids, trajectory kinds, blackout windows.
- `geo.py` — local equirectangular tangent-plane projection about a fixed origin: `latlon_to_enu(lat, lon, origin) -> (east, north)`, `enu_to_latlon(east, north, origin)`, `haversine_m(...)`. `EARTH_RADIUS_M = 6371008.8`. ENU: E = R·cos(lat0)·Δlon, N = R·Δlat, Up handled by callers.
- `transforms.py` — `Similarity(R, t, scale)` with `.apply(points)`; `umeyama(src, dst, with_scaling=True, allow_reflection=False)` (least-squares similarity registration); `gdop(target, sensors)`; `gn_covariance(jac, residual_variance=1.0)` (= `var·pinv(JᵀJ)`, pseudoinverse keeps gauge nulls finite).
- `pipeline.py` — `run_pipeline(scenario, *, model="cv", sigma_a=2.0, detect=False, joint_clock=False, clock_prior_s=1e-4, feed=None) -> PipelineResult`. Branches on scenario features (see below). Returns `PipelineResult(scenario, observations, world, estimates, metrics, tracks, geo_tracks, geometry_series)`.
- `run.py` — the CLI (`python -m dronetracking.run`). Emits a folium map, an animated map, `*_metrics.json`, and PNG diagnostics into `output/`.

**Pipeline phase routing** (one entry point, many phases — set by the scenario, not by code paths you pick):

| Trigger | Phase | Effect |
|---|---|---|
| any device has `velocity_mps` | Ph3 moving devices | windowed geometry tracking |
| `extra_drones` non-empty | Ph6 multi-target | per-frame association + per-source tracks |
| `--detect` / `detect=True` (and no extra drones) | Ph4 detection | matched-filter arrivals from synthesized audio |
| `--joint-clock` | Ph8 joint clock | co-estimate residual clock offsets with position |
| `gps_blackout` windows | Ph9 GPS-denied | dead-reckon + blended georeferencing |

## Scenarios (`scenarios/*.yaml`)

`noisefree_ideal` (cm-level truth), `field_5dev` (consumer noise, the headline),
`sparse_anchors_circular` (high noise), `multi_drone` (Ph6), `detection_demo`
(Ph4 audio), `moving_devices` (Ph3), `gps_denied` (Ph9), `network_demo` (Ph1
mesh). The YAML schema is documented in **dronetracking-simulator**.

## How to run (always `.venv/bin/python`)

```bash
.venv/bin/python -m pytest -q                       # the acceptance gate (~311 green); -m "not slow" for the fast loop
.venv/bin/python -m dronetracking.run --scenario scenarios/field_5dev.yaml   # full pipeline → output/*.html + metrics
.venv/bin/python -m dronetracking.live --scenario scenarios/multi_drone.yaml --speed 2   # live SSE dashboard at :8000
.venv/bin/python -m dronetracking.webapp --tunnel   # zero-install browser app for real phones (CURRENT FOCUS)
.venv/bin/python -m dronetracking.studies --scenario scenarios/field_5dev.yaml --kind noise   # accuracy sweep
# distributed runtime: coordinator + one process per device over TCP — see dronetracking-runtime
```

## Conventions & honest limits

- **Frame:** ENU meters about `origin_latlon`. Device layout from ranging is **gauge-free** (rigid motion + reflection ambiguous) until ≥3 non-collinear GPS anchors pin it.
- **Vertical error always exceeds horizontal** — a near-ground array barely observes altitude; report z separately, never hidden in one radius.
- **Georeferencing accuracy is bounded by GPS-anchor noise** (~2 m consumer GPS), not the acoustic core.
- **Unambiguous 3D georef needs ≥4 non-coplanar anchors.** TDOA needs ≥4 devices per emission; joint-clock needs ≥5.
- Reproducibility: scenarios are frozen dataclasses; sweeps use `dataclasses.replace`. The simulator seeds independent RNG streams (ranging/acoustic/gps) so changing one noise source leaves the others bit-identical.
