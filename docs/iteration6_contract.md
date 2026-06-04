# Iteration 6 — Integration Contract (read this first)

The real-capture device runtime: code that runs *on a device*, capturing through a
`CaptureBackend` and publishing to the coordinator. Two agents, disjoint files.

## Shared rules (unchanged — see docs/iteration4_contract.md for the full statement)
- Repo `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`, package `dronetracking`, venv `.venv`.
- Run tests: `.venv/bin/python -m pytest tests/<file> -q`. Run only YOUR file(s). TDD: test first.
- Create ONLY your assigned files. The `device/` package is infrastructure (NOT estimation) and may import
  `sim`/`sources`/`live`/`estimation`. The firewall only restricts `dronetracking/estimation/`.

## Already scaffolded (use, don't modify)
- `device/backend.py` — the `CaptureBackend` ABC. Abstract: `device_id` (prop), `sample_rate_hz` (prop),
  `local_time() -> float`, `record(duration_s) -> (samples, t0_local_s)`, `ranging_records() -> tuple[RangingRecord]`,
  `gps() -> Optional[(lat,lon,alt)]`; optional `play(signal) -> float`.
- Reusable: `sim.simulator.simulate`, `sources.simulated.SimulatedDeviceFeed`, `sim.audio.{synthesize_captures,reference_pulse,AudioCapture}`,
  `sim.acoustic.emission_times`, `estimation.detection.detect_arrivals`, `sim.observations.{RangingRecord,AcousticArrival,AnchorGps}`,
  `live.protocol.{encode_batch,decode_batch}`, `sources.socket_feed.SocketDeviceFeed`, `geo.*`, `config.load_scenario`.

---

## A — Capture backends  (agent owns `device/backends.py`, `tests/test_device_backends.py`)
Implement `CaptureBackend` (from `device.backend`) twice:

- `MockBackend(scenario, device_id, *, rng=None)` — sim-driven, deterministic, for testing. Run
  `simulate(scenario)` (or a `SimulatedDeviceFeed`) once and `sim.audio.synthesize_captures` once; then:
  - `record(duration_s)` returns this device's synthesized waveform and `t0_local_s = 0.0` (the synthesizer places
    pulses at each device's local-clock arrival time with t0=0, so detection on it reproduces the sim's arrivals).
  - `ranging_records()` returns the sim's ranging records with `initiator == device_id`.
  - `gps()` returns this device's `AnchorGps` as `(lat, lon, altitude_m)` if it is an anchor, else `None`.
  - `sample_rate_hz` from the scenario; `local_time()` may be a simple deterministic value.
- `SoundDeviceBackend(device_id, *, sample_rate_hz=48000.0)` — real microphone via `sounddevice` (import GUARDED:
  raise a clear `RuntimeError`/`ImportError` with an actionable message if `sounddevice` is unavailable).
  `record` uses `sounddevice.rec`; `local_time()` uses `time.monotonic()`; `ranging_records()` raises
  `NotImplementedError` with a docstring pointing to the chirp-ranging bringup step (real two-way ranging is the
  documented hardware task); `gps()` returns `None` (OS GPS is platform-specific — documented).
- Tests (`tests/test_device_backends.py`): `MockBackend` for a scenario — `record()` audio, run
  `detect_arrivals` on it with the scenario's reference pulse, assert recovered arrival times match the sim's true
  arrivals (`sim.acoustic.generate_acoustic_arrivals`) within a few samples; `ranging_records()` all have
  `initiator == device_id`; `gps()` present for anchors / `None` otherwise; it `isinstance(..., CaptureBackend)`.
  `SoundDeviceBackend`: assert constructing/recording raises the clear guarded error when `sounddevice` is not
  installed (it is not installed here).

## B — Device agent + CLI  (agent owns `device/agent.py`, `device/__main__.py`, `tests/test_device_agent.py`)
The on-device application: capture → detect → publish.

- `device/agent.py`: `class DeviceCaptureAgent` (or functions) that, given a `CaptureBackend` plus detection params
  (`reference_pulse`, `n_emissions`, `dt_s`): records `duration` of audio via `backend.record(...)`, runs
  `estimation.detection.detect_arrivals` to get per-emission `DetectedArrival`s, converts them to
  `AcousticArrival`s (their `toa_local_s` is `t0_local + peak_index/sr` — already in the device's local clock),
  gathers `backend.ranging_records()` and `backend.gps()`, and publishes its batch to the coordinator with
  `live.protocol.encode_batch` over a TCP socket (you may reuse `live.agent`'s connect/send pattern). Expose a
  `run(host, port, ...)` that connects, sends, closes.
- `device/__main__.py`: `python -m dronetracking.device --coordinator HOST:PORT --scenario S --id devN [--real] [--duration D]`.
  Default backend `MockBackend(scenario, id)` (uses the scenario's reference pulse + emission count); `--real`
  selects `SoundDeviceBackend`. Prints what it captured/sent.
- Tests (`tests/test_device_agent.py`): use an IN-TEST minimal `CaptureBackend` subclass (or `MockBackend` if you
  prefer — but a small in-test fake keeps you independent of agent A). LOOPBACK: start a
  `sources.socket_feed.SocketDeviceFeed` on port 0 in a thread running `collect(device_ids)`; run the device agent
  for each device against `127.0.0.1:feed.port`; assert `feed.as_observations()` carries each device's acoustic
  arrivals with times close to the truth (and ranging/gps passed through). Deterministic, generous timeouts, joined.

---

## Output expected from each agent
Public API + final signatures, the on-disk/test approach, test results, how the agent wires to the existing
`live.coordinator`, and any deviation. Be explicit about what is validated via mock vs what needs real hardware.
