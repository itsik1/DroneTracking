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

# Tests (the real acceptance gate):
pytest -m "not slow"          # fast inner loop
pytest                        # everything, incl. end-to-end
```

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

## Scope

Iterations 1–2 cover Phases 2–9 of the project vision **in simulation**: relative
localization, clock-sync-free TDOA, single- and multi-target tracking, acoustic detection,
continuous geometry under motion, georeferencing, and GPS-denied operation. Still deferred:
Phase 1 networking/device discovery, joint clock+position estimation, overlapping-source
acoustic separation, and **real hardware** (the next major leap — real microphones, radios,
and clocks behind the same interfaces).
