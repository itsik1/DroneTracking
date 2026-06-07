---
name: dronetracking-simulator
description: >-
  Developer reference for the DroneTracking simulator under
  src/dronetracking/sim/ — the exclusive OWNER of ground truth that synthesizes
  a virtual world (device positions, clocks, drone trajectories, acoustic
  propagation) and emits only measurable Observations. Load this whenever you
  add or edit a scenario YAML, change how truth is generated or noise is
  injected, work on acoustic/ranging/audio synthesis, the clock model,
  trajectories, moving devices, multi-target emissions, or the World/Observations
  data structures. Also load when you need the exact scenario YAML schema (device
  specs, noise, trajectory kinds, extra_drones, gps_blackout, audio, network) or
  when a test in test_sim_*, test_simulator, test_config, or test_no_truth_leak
  is involved. Trigger on file names scenario.py, simulator.py, world.py,
  observations.py, acoustic.py, audio.py, clocks.py, trajectory.py, ranging.py,
  device_motion.py, multi_acoustic.py.
---

# DroneTracking — Simulator (ground-truth owner)

`sim/` is the **only** package allowed to hold ground truth. It builds a world
and emits two things from `simulate(scenario) -> (Observations, World)`:
- `Observations` — everything a real device could measure (handed to `estimation`).
- `World` — the truth used only by `eval`/`viz` for scoring. **`estimation` may never import any of this** (enforced by `tests/test_no_truth_leak.py`).

## Entry point & routing

`simulator.simulate(scenario)` (in `simulator.py`):
1. Seeds one root RNG from `scenario.seed`, then `spawn()`s 3 independent children — **ranging / acoustic / gps** — so changing one noise source leaves the others bit-identical.
2. Routes by scenario feature:
   - `scenario.devices_move` → `device_motion.generate_moving_ranging` else `ranging.generate_ranging_records`.
   - `scenario.extra_drones` → `multi_acoustic.generate_multi_arrivals` else `acoustic.generate_acoustic_arrivals`.
3. Synthesizes noisy GPS anchors: true ENU→lat/lon for truth, perturbed ENU→lat/lon for the emitted `AnchorGps`.
4. Returns the measurement bundle + the truth bundle.

## Truth vs. measurement (the boundary)

| Quantity | Truth (in `World`) | Emitted (in `Observations`) |
|---|---|---|
| Device positions | ENU (x,y,z) at t=0 + velocity | noisy `AnchorGps` (lat/lon/alt) only |
| Clocks | true `offset_s`, `drift_ppm` | none — recovered by estimation |
| Drone trajectory | full position(t) for every target | per-device local arrival times only |
| Emission times | global times at `dt_s` spacing | **never emitted** (cancels in TDOA) |
| Two-way ranging | true inter-device distance | four local-clock timestamps t1..t4 + jitter |

Clock model (`clocks.py`, locked & shared with `estimation.interfaces`):
`local = t_global·(1 + drift_ppm·1e-6) + offset_s`. Helpers `device_local_time(...)`, `global_from_local(...)`.

## Key generators

- `acoustic.emission_times(scenario)`, `acoustic.generate_acoustic_arrivals(scenario, rng)` — per-device ToA = emission_time + range/c, stamped into the device clock, jittered by `noise.toa_std_s`.
- `audio.synthesize_captures(scenario, rng) -> {device_id: AudioCapture(samples, sample_rate_hz, t0_local_s)}` + `audio.reference_pulse(scenario)` — full waveforms for matched-filter detection (Ph4). The pulse's reference instant is its **first sample** — the detector must invert with the same convention. Defaults: chirp 2000→6000 Hz, 0.012 s, `snr_db=10`, rotor fundamental 90 Hz × 4 harmonics.
- `ranging.generate_ranging_records(scenario, rng)` — static two-way exchanges; events spread across `[0, duration_s]` so drift is observable; jittered by `noise.ranging_timestamp_std_s` and `noise.proc_delay_jitter_s`.
- `device_motion.{device_positions_at, generate_moving_ranging}` — Ph3; re-evaluates both endpoints at each exchange's transmit time.
- `multi_acoustic.{generate_multi_arrivals, true_tracks}` — Ph6; one arrival per (device, emission, drone) tagged `source=k`.
- `trajectory.trajectory_position(scenario, t)` — `"linear"` (start_m/end_m), `"circular"` (center_m/radius_m/angular_rate_rad_s), `"waypoints"` (points_m=[[x,y,t],...]); altitude from `z_m`.

## Data structures

- `observations.py`: `RangingRecord`, `AcousticArrival(device_id, emission_idx, toa_local_s, source=0, confidence=1.0)`, `AnchorGps`, `Observations(device_ids, ranging, acoustic, anchor_gps, speed_of_sound_mps, sample_rate_hz)`. Adding a ground-truth field here will fail `test_no_truth_leak`.
- `world.py`: `World(device_ids, device_positions, clock_offsets, clock_drifts_ppm, anchor_latlon, origin_latlon, true_track, true_track_times, true_tracks, device_velocities)`; `.positions_matrix()`, `.positions_matrix_at(t)`.

## Scenario YAML schema (loaded via `config.load_scenario`)

```yaml
name: str
seed: int
speed_of_sound_mps: 343.0
sample_rate_hz: 48000.0
duration_s: float
dt_s: float                 # seconds between emissions / trajectory samples
ranging_rounds: int         # two-way exchanges per device pair
origin_latlon: [lat, lon]   # ENU tangent-plane origin
noise:                      # all optional, default 0 (noiseless)
  ranging_timestamp_std_s: 3e-5
  toa_std_s: 5e-5
  proc_delay_jitter_s: 0.0
  gps_pos_std_m: 2.0
devices:
  - id: dev0
    position_m: [x, y, z]
    clock_offset_s: 0.0
    clock_drift_ppm: 0.0
    proc_delay_s: 0.002     # two-way-ranging turnaround
    has_gps: false
    velocity_mps: [0,0,0]   # Ph3 moving devices
    battery_frac: 1.0       # Ph1 networking
    has_mic: true
    has_speaker: true
trajectory:                 # primary drone (target 0)
  kind: linear | circular | waypoints
  z_m: 50.0
  params: { ... }           # kind-specific
extra_drones: [ {kind, z_m, params}, ... ]   # Ph6 multi-target
gps_blackout: [ [start_s, end_s], ... ]      # Ph9 GPS-denied
audio: { snr_db, pulse, f0, f1, pulse_dur_s, rotor_fundamental_hz, rotor_harmonics, rotor_level }  # Ph4
network: { ... }            # Ph1 discovery/transport
```

`Scenario` is a **frozen** dataclass (`scenario.py`); `Vec3 = (float,float,float)`, `LatLon = (float,float)`. Properties: `device_ids`, `anchors` (has_gps), `all_drones`, `devices_move`, `gps_available(t)`. `DeviceSpec.position_at(t)` applies constant-velocity drift.

## Gotchas

- **Don't emit the global emission time** anywhere in `Observations` — TDOA depends on it being unknown. Per-device arrivals share an `emission_idx`.
- **Pulse reference = first sample.** A mismatch with `detection.py` causes a systematic timing bias.
- **Emission spacing aliasing:** if `dt_s` is below the cross-device range-delay spread, per-device pulses alias in detection — keep `dt_s` comfortably larger (e.g. detection_demo uses 2.0 s).
- Iteration-2 fields (`extra_drones`, `gps_blackout`, `audio`, `network`) default to empty so older scenarios still load.
- The noise-free residual (~cm) is the **physical** clock-skew-during-flight ranging bias, not numerical error — averaging can't remove it.
