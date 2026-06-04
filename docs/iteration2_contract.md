# Iteration 2 — Integration Contract (read this first)

Four new phases extend the working iteration-1 testbed. Each agent owns **NEW, disjoint
files** and writes unit tests; the orchestrator wires everything into the shared
simulator/pipeline/eval/viz afterward. Do not edit shared files.

## Project facts (unchanged from iteration 1)
- Repo: `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`, package `dronetracking` (src layout), venv `.venv`.
- Run tests: `.venv/bin/python -m pytest tests/<your_file>.py -q`. Run only YOUR file(s).
- TDD: test first, watch fail, implement, pass.
- **Ground-truth firewall:** files under `dronetracking/estimation/` must NOT import `dronetracking.sim`
  (enforced by `tests/test_no_truth_leak.py`). Your `sim/*` files MAY use `sim` freely. Any TEST may import sim.

## Hard rules
- **Create only your assigned files.** Do NOT edit: `sim/scenario.py`, `config.py`, `sim/simulator.py`,
  `sim/observations.py`, `sim/acoustic.py`, `sim/ranging.py`, `sim/world.py`, `pipeline.py`, `run.py`,
  `estimation/interfaces.py`, `datatypes.py`, `eval/*`, `viz/*`, or other agents' files. The orchestrator
  integrates. (You MAY add a new `scenarios/*.yaml`.)
- Reuse existing modules; don't reinvent geo/transforms/estimation stages.

## Already scaffolded (available now — import & use, do not modify)
- `sim/scenario.py`: `DeviceSpec.velocity_mps` (+ `DeviceSpec.position_at(t)`), and on `Scenario`:
  `extra_drones: tuple[TrajectorySpec,...]`, `gps_blackout: tuple[(start,end),...]`, `audio: dict`,
  plus properties `all_drones`, `devices_move`, and `gps_available(t)`.
- `sim/observations.py`: `AcousticArrival` now has `source: int = 0` and `confidence: float = 1.0`.
- `estimation/interfaces.py`: `Track` now has `target_id: Optional[str] = None`.
- Existing reusable APIs: `sim.acoustic.generate_acoustic_arrivals/emission_times`, `sim.ranging.generate_ranging_records`,
  `sim.trajectory.trajectory_position`, `sim.clocks.device_local_time/global_from_local`, `sim.simulator.simulate`,
  `config.load_scenario`, `estimation.ranging.build_distance_matrix`, `estimation.relative_localization.estimate_layout`,
  `estimation.clock_sync.estimate_clocks`, `estimation.tdoa.localize_emission/localize_all`,
  `estimation.tracking.track_target` (+ internal `_KalmanFilter`), `estimation.georeference.solve_transform/georeference_track`,
  `transforms.umeyama/gdop/gn_covariance`, `geo.*`. Contract types: `DistanceMatrix`, `TargetFix`,
  `RelativeLayout`, `ClockEstimates`, `Track`, `GeoTrack`, `Estimates`, `RangingRecord`, `AcousticArrival`, `AnchorGps`.

---

## Ph4 — Acoustic detection / DSP  (agent owns these files)
Files: `src/dronetracking/sim/audio.py`, `src/dronetracking/estimation/detection.py`, `tests/test_detection.py`.

Model the drone signature as a short **known pulse emitted once per `dt_s`** (e.g. a linear chirp or a
burst of rotor harmonics) embedded in rotor-harmonic background + Gaussian noise at a configurable SNR
(`scenario.audio`, e.g. `{"snr_db": 10, "pulse": "chirp", "f0": 2000, ...}` — you define the keys, with
defaults). Each device records the pulse train delayed to its true local-clock arrival times.

- `sim/audio.py`:
  - `@dataclass AudioCapture(device_id: str, samples: np.ndarray, sample_rate_hz: float, t0_local_s: float)`
  - `synthesize_captures(scenario, rng) -> dict[str, AudioCapture]` — per device, render a waveform of
    length `duration_s` at `scenario.sample_rate_hz`; place the known pulse at each emission's true local
    arrival time (use the same physics as `sim/acoustic`: global arrival = emit_time + range/c, then
    `device_local_time`). Add harmonic background + noise per `scenario.audio` SNR. Also expose the
    reference pulse: `reference_pulse(scenario) -> np.ndarray`.
- `estimation/detection.py` (NO sim import):
  - `detect_arrivals(captures: dict[str, AudioCapture], reference_pulse, n_emissions, dt_s) -> tuple[AcousticArrival,...]`
    — band-pass + **matched filter** (correlate each capture with `reference_pulse`), pick the top peaks,
    convert peak sample index -> local arrival time, emit `AcousticArrival(device_id, emission_idx, toa_local_s, confidence=...)`
    where confidence comes from peak prominence / SNR. Group peaks into emissions by time.
- Tests: synthesize at high SNR -> recovered arrivals match the true arrivals (from `sim.acoustic.generate_acoustic_arrivals`
  on the same scenario) to within a few samples; confidence high. Low SNR -> graceful degradation (lower confidence,
  bounded error). Keep `n_emissions` small for speed.

## Ph6 — Multi-target tracking  (agent owns these files)
Files: `src/dronetracking/sim/multi_acoustic.py`, `src/dronetracking/estimation/multi_target.py`,
`tests/test_multi_target.py`, `scenarios/multi_drone.yaml`.

- `sim/multi_acoustic.py`: `generate_multi_arrivals(scenario, rng) -> tuple[AcousticArrival,...]` — like
  `sim.acoustic.generate_acoustic_arrivals` but loops over `scenario.all_drones`, tagging each arrival with
  `source=k` (drone index). One `AcousticArrival` per (device, emission, drone). Also
  `true_tracks(scenario) -> dict[int, np.ndarray]` (per-drone (N,3) truth, from `trajectory_position`).
- `estimation/multi_target.py` (NO sim import):
  - `localize_frames(arrivals, clocks, layout, speed_of_sound) -> list[(t, list[TargetFix])]` — group by
    (emission_idx, source), call `tdoa.localize_emission` per group (>=4 devices). This yields a SET of
    fixes per frame. (Source is used only to form clean fixes — the tracker must NOT use it for association.)
  - `track_targets(frames, gate_chi2=..., birth_min_hits=2, death_max_misses=3) -> list[Track]` — at each
    frame, **global-nearest-neighbour association** (`scipy.optimize.linear_sum_assignment` on Mahalanobis
    distance between each live track's predicted position and each fix) with a chi-square gate; unmatched fixes
    spawn tentative tracks (confirm after `birth_min_hits`); tracks with `death_max_misses` consecutive misses
    are terminated. Each track is a single-target Kalman filter (reuse `estimation.tracking`). Return confirmed
    `Track`s with distinct `target_id`.
- Tests: 2–3 drones (well separated), perfect layout+clocks (true values), arrivals via `generate_multi_arrivals`;
  assert the right NUMBER of confirmed tracks, each matching a true drone within tolerance, and that association
  doesn't swap identities mid-run. Also a crossing-tracks case is a nice-to-have.
- `scenarios/multi_drone.yaml`: primary `trajectory` + 1–2 `extra_drones`, e.g. one linear + one circular.

## Ph3 — Moving devices / continuous geometry  (agent owns these files)
Files: `src/dronetracking/sim/device_motion.py`, `src/dronetracking/estimation/geometry_tracking.py`,
`tests/test_geometry_tracking.py`, `scenarios/moving_devices.yaml`.

- `sim/device_motion.py`: `generate_moving_ranging(scenario, rng) -> tuple[RangingRecord,...]` — like
  `sim.ranging.generate_ranging_records` but evaluate each device's position at the exchange transmit time via
  `DeviceSpec.position_at(t_tx)` (devices drift at `velocity_mps`). Also `device_positions_at(scenario, t) -> dict[id,(3,)]`.
- `estimation/geometry_tracking.py` (NO sim import):
  - `track_geometry(ranging_records, device_ids, speed_of_sound, window_s, step_s) -> list[(t_center, RelativeLayout)]`
    — slide a time window over the ranging records (bucket by the initiator transmit timestamp), build a
    `DistanceMatrix` per window (reuse `ranging.build_distance_matrix` by wrapping records in a small
    observations-shaped object) and an `estimate_layout` per window; align consecutive windows to a common
    frame (Umeyama) and optionally smooth per-device positions over time (simple per-device Kalman). Return the
    time series of layouts.
- Tests: devices moving at known constant velocity; assert each window's layout (after aligning to that window's
  true device positions) has small RMSE, and that the recovered per-device velocity ≈ truth.
- `scenarios/moving_devices.yaml`: a few devices with nonzero `velocity_mps` (slow drift, e.g. 0.5–2 m/s).

## Ph9 — GPS-denied operation  (agent owns these files)
Files: `src/dronetracking/estimation/gps_denied.py`, `tests/test_gps_denied.py`, `scenarios/gps_denied.yaml`.

- `estimation/gps_denied.py` (NO sim import):
  - `georeference_with_blackout(layout, anchor_gps, track, origin, blackout_windows, recovery_blend_s=...) -> GeoTrack`
    — like `georeference.georeference_track`, but the LOCAL->ENU transform is treated as time-varying/available
    only outside `blackout_windows`. During a blackout, HOLD the last good transform (dead-reckon the georef).
    On GPS return, re-solve the transform and **blend** from the held one to the new one over `recovery_blend_s`
    so the georeferenced track has NO discontinuity (smooth drift correction). Outside blackout, behaves like the
    normal georeferencer. Return a `GeoTrack` covering all track times, plus expose which frames were dead-reckoned.
  - Also `gps_status(track_times, blackout_windows) -> np.ndarray[bool]` (available per frame).
- To exercise drift, accept an optional `transform_provider(t) -> Similarity` (a callable giving the "true"
  time-varying transform, e.g. when devices drift); default to a single static transform from `solve_transform`.
- Tests: a blackout window mid-run; assert (a) all track frames get a georeferenced position (none dropped),
  (b) the held-then-blended track is continuous (no jump > a small threshold at the recovery boundary),
  (c) error during blackout is bounded and shrinks after recovery. Build inputs by hand (layout + AnchorGps +
  a Track) — no sim needed.
- `scenarios/gps_denied.yaml`: a normal field scenario plus a `gps_blackout: [[t0, t1]]` window.

---

## Output expected from each agent
Return: the public functions/classes you implemented (final signatures), the new test file(s) + their pass
counts, the demo scenario (if any), and any interaction you foresee with the other phases. Note any deviation.
