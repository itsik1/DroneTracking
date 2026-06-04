# DroneTracking — Distributed Acoustic Localization (Iteration 1)

A **simulation-first** testbed for a distributed acoustic drone-localization network.
Ordinary devices (simulated here) localize themselves **without GPS** and **without
synchronized clocks**, then detect and track a drone acoustically and georeference the
result onto a real map.

This iteration validates the *algorithms* against synthetic data with **known ground
truth**, so accuracy can be measured at every stage — before any real microphone, radio,
or GPS exists.

## Pipeline

```
sim (truth) ──> Observations ──> estimation ──> Estimates ──> eval + viz
                                  │
   relative localization (MDS + weighted least-squares)
   → clock offset/drift recovery (two-way timestamps, no shared clock)
   → TDOA target localization (closed-form seed + robust refine)
   → Kalman tracking
   → georeferencing to lat/lon (similarity transform from GPS anchors)
```

**Ground-truth firewall:** the `estimation` package is structurally forbidden from
importing truth. The `sim` package owns truth and emits only what a real device could
measure; `eval` is the only place that compares the two.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip && python -m pip install -e ".[dev]"

# Run the full pipeline on the headline scenario:
python -m dronetracking.run --scenario scenarios/field_5dev.yaml
open output/field_5dev_map.html

# LIVE dashboard — watch a scenario stream in real time in your browser:
python -m dronetracking.live --scenario scenarios/multi_drone.yaml --speed 2
# then open http://127.0.0.1:8000

# Tests (the real acceptance gate):
pytest -m "not slow"          # fast inner loop
pytest                        # everything, incl. end-to-end
```

### Live / streaming mode

`python -m dronetracking.live` runs the **streaming engine**: it calibrates the network
once (relative geometry, clocks, georeference), then processes drone emissions *one at a
time in time order* — updating the tracks incrementally and pushing a state snapshot per
step to a browser Leaflet map over Server-Sent Events. This is the hardware-facing shape
of the system: when real devices replace the simulator, only the engine's data source
changes — the dashboard is unchanged. `--speed N` accelerates playback; `--detect` drives
it from synthesized audio.

## Results (iteration 1)

Measured against ground truth (`pytest -m slow` asserts these stay in tolerance):

| scenario | device localization | drone tracking (xy / z) | georeferencing (alt) |
|---|---|---|---|
| noise-free | 1.6 cm | 2.8 cm (0.8 / 2.7) | 0.8 cm (2.7 cm) |
| `field_5dev` (consumer noise) | 0.6 cm | 6.2 cm (2.8 / 5.5) | **2.0 m** (1.3 m) |
| `sparse_anchors_circular` (high noise) | 3.6 cm | 20 cm (9 / 18) | 0.41 m (1.4 m) |

Honest findings the simulation surfaced:

- **Vertical error always exceeds horizontal.** A roughly ground-level array barely
  observes altitude (weak vertical observability) — reported separately, never hidden
  in a single error radius.
- **Georeferencing accuracy is bounded by GPS-anchor noise** (~2 m for consumer GPS),
  not by the acoustic core — matching the project's stated accuracy targets.
- **Unambiguous 3D georeferencing needs ≥4 non-coplanar anchors.** With exactly 3
  (coplanar) anchors the vertical handedness is mathematically ambiguous and the drone's
  altitude flips; the shipped scenarios place some anchors at height.
- The noise-free residual (~cm) is the *physical* clock-skew-during-flight ranging bias,
  not numerical error — averaging cannot remove it.

## Results (iteration 2 — Phases 3 / 4 / 6 / 9)

Four more phases, each routed through the same pipeline by scenario feature and asserted
by `pytest -m slow`:

| phase | scenario | command | result |
|---|---|---|---|
| **Multi-target (Ph6)** | `multi_drone.yaml` | `run --scenario scenarios/multi_drone.yaml` | 3/3 drones tracked, GNN association, per-drone RMSE 4.5–8.8 cm |
| **Acoustic detection (Ph4)** | `detection_demo.yaml` | `run --scenario scenarios/detection_demo.yaml --detect` | drone localized from synthesized audio (matched filter); tracking 2.6 cm |
| **Moving devices (Ph3)** | `moving_devices.yaml` | `run --scenario scenarios/moving_devices.yaml` | live geometry tracked as devices drift; per-window layout RMSE 0.19 m |
| **GPS-denied (Ph9)** | `gps_denied.yaml` | `run --scenario scenarios/gps_denied.yaml` | holds local frame through a blackout, smooth re-alignment on return; error bounded ~1.6 m |

How they integrate: `run_pipeline` branches on the scenario — `extra_drones` → multi-target
association, `velocity_mps` → continuous geometry tracking, `--detect` → DSP detection
replaces idealized arrivals, `gps_blackout` windows → dead-reckoned-and-blended
georeferencing. The estimators stay behind the ground-truth firewall.

Findings: matched-filter detection needs emission spacing (`dt_s`) above the cross-device
range-delay spread or per-device emissions alias; multi-target association is done from
motion only (sources are not used as identity); moving-device geometry needs ≥4 well-spread
static anchors to fix the gauge.

## Results (iteration 3 — networking, hardware bridge, joint clock)

- **Phase 1 — Network Formation** (`dronetracking.network`): a node registry (battery,
  mic/speaker/GPS capabilities, confidence), simulated peer discovery forming a device
  mesh, a transport model with per-link-type (BLE/Wi-Fi/mesh) range/latency/loss, and a
  `NetworkManager` reporting connectivity and health. The mesh + health now render live in
  the dashboard.
- **Hardware bridge** (`dronetracking.sources`): a `DeviceFeed` interface the pipeline and
  streaming engine read through. `SimulatedDeviceFeed` is the reference; `LiveDeviceFeed` is
  the documented contract for real devices. Swapping the feed is the *only* change needed to
  go from simulation to hardware — nothing downstream moves.
- **Joint clock+position** (`estimation.joint_clock`, `--joint-clock`): co-estimates residual
  per-device clock offsets with the target position (one shared offset per device across
  emissions). Cuts localization error 2.5–7.8× when clock sync leaves residual error; no
  regression on clean clocks.

## Results (iteration 4 — real-time, distributed, quantified)

- **Online tracking** (`estimation/online_tracker.py`): a stateful `OnlineTracker.update(fixes, t)`
  that the streaming engine now drives one frame at a time (incremental association + KF +
  birth/death), replacing the per-frame batch re-run — real-time, O(tracks×fixes) per step.
- **Distributed runtime** (`live/protocol.py`, `live/agent.py`, `sources/socket_feed.py`,
  `live/coordinator.py`): each device publishes only its own measurements over TCP; the
  coordinator reassembles them into `Observations` and runs the pipeline. Verified end-to-end —
  5 device agents over sockets reconstruct the simulator's data exactly, then estimate. This is
  the literal data path real hardware uses (`python -m dronetracking.live.coordinator`).
- **Robustness study** (`studies/`, `python -m dronetracking.studies`): sweeps noise / device
  count and plots accuracy vs parameter. Confirms the headline finding — tracking and device
  localization stay at cm-level across a 4× noise range while georeferencing scales with GPS
  noise, i.e. the acoustic core is robust and accuracy is GPS-anchor-bound.

## Results (iteration 5 — separation + the road to hardware)

- **Overlapping-source separation** (`estimation/separation.py`): a matched-filter bank that
  separates several simultaneously-emitting drones (distinct signatures) from each device's
  mixed audio into per-source arrivals, which feed the multi-target TDOA path. Zero cross-talk
  and sub-sample accuracy in tests.
- **Recorded-data feed** (`sources/recorded.py`): `RecordedAudioFeed` ingests per-device WAVs +
  a `meta.json` from disk and runs detection → arrivals, so a session captured on real devices
  replays through the exact same pipeline (`run_pipeline(scenario, feed=RecordedAudioFeed(dir))`).
- **Hardware bringup guide** ([`docs/hardware_bringup.md`](docs/hardware_bringup.md)): how to
  deploy on real devices — per-device capture, the calibrate-then-track flow, the SDS-TWR clock
  scheme, `sounddevice`/GPS capture, known limits, and a staged bringup checklist.

## Scope

Iterations 1–5 implement **Phases 1–9** of the project vision in simulation — network formation,
GPS-free relative localization, clock-sync-free TDOA, single/multi-target tracking, acoustic
detection and overlapping-source separation, continuous geometry under motion, georeferencing,
GPS-denied operation, live streaming, joint clock+position refinement, and a real socket-based
distributed runtime — all behind a hardware-feed abstraction (simulated, recorded-file, and
socket feeds today; a `LiveDeviceFeed` skeleton + bringup guide for real sensors).

The remaining frontier is **real hardware**: implement the documented `LiveDeviceFeed`/device
agent against physical microphones, radios, and clocks. The interfaces, wire protocol, recorded
replay path, and bringup guide are all in place — what's left needs devices, not more simulation.
