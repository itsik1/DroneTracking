---
name: dronetracking-runtime
description: >-
  Developer reference for the DroneTracking real-time & distributed runtime —
  the live streaming engine and SSE dashboard (src/dronetracking/live/), the
  TCP coordinator + per-device agent processes, the DeviceFeed hardware-bridge
  abstraction (src/dronetracking/sources/), the capture backends
  (src/dronetracking/device/, mock vs real microphone via sounddevice), and the
  simulated network mesh/health layer (src/dronetracking/network/). Load this
  whenever you work on streaming/online tracking playback, the live Leaflet
  dashboard, the distributed coordinator/agent wire protocol, swapping the data
  feed from simulator to recorded WAVs to live sockets to hardware, real-mic
  capture, or peer discovery/transport/link-quality. Trigger on file names
  engine.py, coordinator.py, protocol.py, agent.py, server.py (live), backend.py,
  backends.py, base.py, simulated.py, recorded.py, socket_feed.py, live.py
  (sources), discovery.py, node.py, transport.py; or tests test_live_engine,
  test_distributed, test_device_agent, test_device_backends, test_sources,
  test_recorded, test_network.
---

# DroneTracking — Real-time & distributed runtime

This is the hardware-facing shape of the system. The estimation pipeline reads
through a `DeviceFeed`; swapping the feed (simulator → recorded → sockets →
hardware) is the **only** change needed to go from simulation to real devices —
nothing downstream moves. The ground-truth firewall holds across the socket
boundary: only `SimulatedDeviceFeed` carries `.world`, and it lives **outside**
the `DeviceFeed` interface.

## 1. DeviceFeed abstraction (`sources/`)

`base.DeviceFeed` (ABC): implement `device_ids()`, `ranging_records()`, `acoustic_arrivals()`, `anchor_gps()`, and properties `speed_of_sound_mps`, `sample_rate_hz`. The free method `as_observations()` bundles them into the standard `Observations` contract.

| Feed | Source | Has ground truth? |
|---|---|---|
| `SimulatedDeviceFeed(scenario)` | runs `simulate()` once | yes — `.world` (sim-only, NOT in the interface) |
| `RecordedAudioFeed(directory)` | per-device WAVs + `meta.json`; matched-filter detection on load | no |
| `SocketDeviceFeed(host, port)` | reassembles per-device TCP batches | no |
| `LiveDeviceFeed` | skeleton/contract for real sensors — every accessor raises `NotImplementedError` with a docstring describing the real data source | no |

`RecordedAudioFeed` on-disk layout: `meta.json` (device_ids, speed_of_sound_mps, sample_rate_hz, n_emissions, dt_s, `reference` pulse params, per-device `t0_local_s`, `audio_files`, `batches` with ranging+gps) + one mono `{id}.wav` per device. `write_recorded_dataset(dir, feed, captures, scenario)` is the inverse (capture a real session, replay it through the exact pipeline).

## 2. Streaming engine (`live/engine.py`)

`StreamEngine(scenario, *, detect=False, model="cv", sigma_a=2.0, feed=None)` calibrates **once** in `__init__` — geometry (`track_geometry`/`estimate_layout`), clocks (`estimate_clocks`), optional detection from synthesized audio, georeference (`solve_transform`), and per-emission fixes (`localize_frames`). Then `snapshots() -> Iterator[Snapshot]` walks emissions in time order, driving a stateful `OnlineTracker.update(fixes, t)` (O(tracks×fixes)/step — no batch re-run) and yielding one `Snapshot` per frame.

`Snapshot(t, index, total, devices, anchors, targets, true_targets, links, net)` → `.to_dict()`. `targets` carry `{id, lat, lon, alt, r_m}` where `r_m` is a 1σ horizontal radius; `true_targets` come from `self.world` when the feed is simulated; `links`/`net` come from the network layer (best-effort, exceptions swallowed).

## 3. Live dashboard (`live/server.py`, `live/__main__.py`)

`python -m dronetracking.live --scenario <yaml> [--host 127.0.0.1] [--port 8000] [--speed N] [--detect] [--seed]`. `serve(scenario, host, port, speed, detect)` exposes `GET /` (embedded Leaflet HTML) and `GET /events` (SSE of `Snapshot`s, paced at `dt_s/speed`). It renders devices, GPS anchors, tracked drones + 1σ circles, true-drone overlay, mesh links, and a network-health HUD. **Difference from the webapp:** this replays a scenario through the engine for visualization; the webapp ingests live sensor reports from real phones. Both consume identical `Observations`.

## 4. Distributed runtime (coordinator + per-device agents over TCP)

Wire protocol (`live/protocol.py`, `PROTOCOL_VERSION = 1`): one device's batch = one line-delimited JSON object (`\n`-terminated, self-delimiting on TCP). `encode_batch(device_id, ranging, acoustic, anchor_gps, speed_of_sound_mps, sample_rate_hz) -> bytes` / `decode_batch(data) -> dict`. Every timestamp stays in its **originating device's local clock** (unsynchronized) — clock sync happens downstream in `estimate_clocks`.

Per-device partition (`live/agent.device_slice`): ranging where `initiator == id`, acoustic/GPS where `device_id == id`; union over devices = full `Observations`.

```bash
# coordinator (listen-only, waits for external device processes):
.venv/bin/python -m dronetracking.live.coordinator --scenario scenarios/detection_demo.yaml --port 9123 --external
# one process per device (each captures, detects, publishes its own slice):
.venv/bin/python -m dronetracking.device --coordinator 127.0.0.1:9123 --scenario scenarios/detection_demo.yaml --id dev0 [--real]
```

- `live/coordinator.py main()` flags: `--scenario`, `--host`, `--port` (0 = OS-assigned), `--timeout`, `--seed`, `--external` (don't spawn internal sim agents — wait for real device processes). It binds `SocketDeviceFeed`, then `collect(expected_device_ids, timeout_s)` accepts one connection per device and runs `run_pipeline(scenario, feed=feed)`.
- `live/agent.DeviceAgent.publish(host, port, scenario, device_id, feed=None)` publishes a simulated device's slice (used for internal agents).
- `device/agent.DeviceCaptureAgent(backend, reference_pulse, n_emissions, dt_s, ...)`: `.detect(duration_s)` records → `detect_arrivals` → maps `DetectedArrival`→`AcousticArrival`, gathers ranging + GPS; `.run(host, port, duration_s)` does the full capture→detect→publish cycle. `device/__main__.py` flags: `--coordinator HOST:PORT`, `--scenario`, `--id`, `--real`, `--duration`, `--seed`.

## 5. Capture backends (`device/backend.py`, `device/backends.py`)

`CaptureBackend` (ABC): `device_id`, `sample_rate_hz`, `local_time()`, `record(duration_s) -> (samples, t0_local_s)`, `ranging_records()`, `gps()`, `play(signal)`.
- `MockBackend(scenario, device_id, rng=None)` — runs `simulate()` + `synthesize_captures()` once and serves this device's slice; `local_time()=0.0` (deterministic). Detection on its waveform reproduces the simulator's true arrivals.
- `SoundDeviceBackend(device_id, sample_rate_hz=48000)` — **real microphone** via a guarded `sounddevice` import (clear `RuntimeError` if missing). `local_time()=time.monotonic()`; `record` uses `sounddevice.rec()`. Real-mic chirp ranging and GPS are not yet wired (`ranging_records()` raises `NotImplementedError`, `gps()` returns None) — this is the hardware-bringup frontier (see `docs/hardware_bringup.md`).

`--real` selects `SoundDeviceBackend`; default is `MockBackend`. The agent, protocol, coordinator, detector, and downstream pipeline are identical for both — mock→real mic is the only swap.

## 6. Network layer (`network/`)

- `node.Node(id, battery_frac, has_mic, has_speaker, has_gps, online, confidence)` — `.capabilities()`, `.can_anchor()`; `node.NodeRegistry.from_specs(scenario.devices)` with `online_ids()`, `anchors()`, `mean_battery()`, `set_online/set_confidence`.
- `transport.SimulatedTransport(positions, ..., kind)` with `RADIO_PRESETS` ble(80 m/0.03 s/10%), wifi(250 m/0.005 s/2%), mesh(600 m/0.05 s/15%): `reachable`, `link_quality` (linear falloff), `latency`, `send` (Bernoulli loss), `deliver`. It uses true positions for reachability, but discovery above it only observes which HELLO packets arrive — no position leakage.
- `discovery.discover(devices, transport, rng) -> NetworkGraph` (HELLO sweep; symmetric edges) and `discovery.NetworkManager(scenario, ...)` → `form_network()`, `is_connected()`, `health()` (counts, mean battery, mean link quality, components). The streaming engine pulls this into `Snapshot.net`/`links`.

## Gotchas

- `--external` is required when running real `dronetracking.device` processes — otherwise the coordinator spawns its own internal sim agents and won't wait.
- `SoundDeviceBackend` needs `sounddevice` (not a base dependency) and real hardware; tests use `MockBackend`.
- `SocketDeviceFeed.collect` fixes device order to `expected_device_ids` (deterministic matrices); the listener binds before agents connect (no accept/connect race); each batch is read until newline-or-EOF.
- Don't reach for `.world` on anything but `SimulatedDeviceFeed` — it doesn't exist on the other feeds (that's the firewall).
