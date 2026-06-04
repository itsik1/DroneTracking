# Hardware Bringup Guide

How to run the drone-tracking system on **real devices** instead of the simulator. The
estimation stack does not change at all — it consumes a
[`DeviceFeed`](../src/dronetracking/sources/base.py) that produces an
[`Observations`](../src/dronetracking/sim/observations.py) bundle, and that bundle is
made of things a real device can actually measure. This guide explains the architecture,
the two-phase calibrate-then-track flow and *why* it is required, how to implement a real
feed/agent against hardware, the physical limits to plan around, and a step-by-step bringup
checklist that validates each stage against the existing `eval`/`studies` tooling.

If you only want to **replay recordings you already captured** (audio + ranging dumps),
skip to [Ingesting recorded data](#ingesting-recorded-data-the-bridge-you-have-today) — the
[`RecordedAudioFeed`](../src/dronetracking/sources/recorded.py) does that today with no new
code.

---

## 1. Architecture recap

```
   ┌─────────── device A ───────────┐        ┌─────────── device B ───────────┐
   │ mic ─► detect (matched filter) │        │ mic ─► detect (matched filter) │
   │ speaker ─► emit ranging chirp  │        │ speaker ─► emit ranging chirp  │
   │ GPS (if anchor)                │        │ GPS (if anchor)                │
   │                                │        │                                │
   │   DeviceAgent.publish(...)     │        │   DeviceAgent.publish(...)     │
   └──────────────┬─────────────────┘        └───────────────┬────────────────┘
                  │  JSON batch over TCP (live/protocol.py)   │
                  └──────────────────┬────────────────────────┘
                                     ▼
                        ┌──────────────────────────────┐
                        │  coordinator                  │
                        │  SocketDeviceFeed.collect(...) │  unions per-device batches
                        │            │                  │
                        │            ▼                  │
                        │  run_pipeline(scenario, feed) │  geometry → clocks → TDOA →
                        │                               │  tracking → georeference
                        └──────────────────────────────┘
```

* **Each device runs a [`DeviceAgent`](../src/dronetracking/live/agent.py)** that captures
  *its own* measurements and publishes them to the coordinator over the existing
  line-delimited JSON wire protocol ([`live/protocol.py`](../src/dronetracking/live/protocol.py)).
  A device only ever sees and sends its own slice: ranging exchanges it *initiated*,
  acoustic arrivals *it* heard, and its own GPS fix. The union of all slices is exactly the
  full `Observations` (this partition is tested in `tests/test_distributed.py`).
* **The coordinator runs a [`SocketDeviceFeed`](../src/dronetracking/sources/socket_feed.py)**,
  a `DeviceFeed` that binds a TCP listener, calls `collect(expected_device_ids, timeout_s=...)`
  to accept one batch per device, and unions them into a single `Observations`.
* **The pipeline ([`pipeline.run_pipeline`](../src/dronetracking/pipeline.py)) consumes that
  feed and is unchanged**: `run_pipeline(scenario, feed=socket_feed)` runs geometry →
  clock-sync → TDOA → tracking → georeferencing. Because a live feed carries no ground
  truth (`feed.world` is absent), the pipeline simply *skips* the scoring step and still
  produces estimates.

The simulator-backed [`SimulatedDeviceFeed`](../src/dronetracking/sources/simulated.py) and
the live [`SocketDeviceFeed`](../src/dronetracking/sources/socket_feed.py) are
interchangeable behind this seam, so you develop against the simulator and swap in real
hardware with a one-line change.

### The five things every device must report

The feed surface is one accessor per `Observations` field (plus two timebase constants).
On real hardware each maps to a sensor read:

| Accessor | Real-world source |
|---|---|
| `device_ids()` | the enrolled/reachable devices, in a **stable** order (it fixes the matrix row order downstream) |
| `ranging_records()` | two-way-ranging four-timestamp rounds, each timestamp **in the measuring device's own clock** |
| `acoustic_arrivals()` | matched-filter detections of the drone pulse, per device, in that device's local clock |
| `anchor_gps()` | lat/lon/alt from the GPS-equipped (anchor) devices; non-GPS devices contribute none |
| `speed_of_sound_mps` | a configured constant, or live atmospheric (temp/humidity) |
| `sample_rate_hz` | the microphones' sample rate (bounds arrival-timing resolution) |

---

## 2. What each device needs

* **A microphone** (required). This is the primary sensor — it timestamps the drone's
  acoustic emissions. Every tracking device needs one. (`DeviceSpec.has_mic` models this.)
* **A speaker** (recommended). Used during calibration to emit the known ranging chirp so a
  pair of devices can run two-way ranging and recover their separation + relative clock.
  A device with no speaker (`has_speaker=False`) can still *respond* to others' chirps and
  still listen for the drone; it just cannot *initiate* a ranging round.
* **GPS** (on ≥4 devices, the *anchors*). Georeferences the relative solution into real-world
  lat/lon. The more non-coplanar anchors, the better the vertical fix.
* **A network path to the coordinator** (Wi-Fi / Ethernet / mesh). Only needs to carry the
  small JSON batch each device publishes — *not* raw audio. Bandwidth is trivial; what
  matters is that every device can reach `coordinator_host:port`.

Time sync between devices is **not** a hardware requirement — see the next section. That is
the whole point of two-way ranging.

---

## 3. The two-phase flow: calibrate, then track (and why)

> **There is no shared clock.** Each device free-runs on its own oscillator with an unknown
> bias and drift. We never assume a common time base; we *estimate* the relative clocks from
> the ranging timestamps and lock everything to one reference device. This is the single most
> important thing to understand about the system.

### The locked-clock convention

Every device's local clock relates to a notional global time by an affine map (the convention
is shared by the simulator and by `estimation.clock_sync` / `estimation.interfaces`):

```
local = t_global * (1 + drift_ppm * 1e-6) + offset_s
```

`offset_s` (bias at t=0) and `drift_ppm` (oscillator skew) are **unknown per device**. The
reference device is pinned at `(offset=0, drift=0)`; everyone else is solved relative to it.
Absolute global time is unobservable and unnecessary — TDOA only needs *consistent relative*
timing, which the locked convention provides.

### Why two phases

The pipeline needs three things before it can place the drone:

1. **Relative geometry** — where the devices are, relative to each other (an array shape).
2. **Per-device clocks** — each device's `(offset, drift)` relative to the reference.
3. **A georeference** — the rigid transform from the relative frame to real-world lat/lon.

(1) and (2) both come from **two-way ranging**, and (3) comes from **GPS anchors**. None of
these involve the drone. So bringup is naturally two phases:

* **Phase A — Calibration.** Run two-way ranging across device pairs to recover the relative
  geometry (`estimation.relative_localization.estimate_layout` from a distance matrix) *and*
  the clock offsets/drifts (`estimation.clock_sync.estimate_clocks`), and read GPS on the
  anchors to solve the georeferencing transform (`estimation.georeference.solve_transform`).
  Calibration can run while the array is static and before any target is present.
* **Phase B — Live target tracking.** With geometry + clocks + transform locked, the devices
  only need to keep timestamping drone emissions. Each acoustic arrival, corrected to the
  reference clock, becomes a TDOA observation; the pipeline localizes each emission and the
  tracker (`estimation.tracking` / `estimation.multi_target`) produces the track, which is
  then georeferenced through the Phase-A transform.

If the array moves, geometry is re-estimated continuously
(`estimation.geometry_tracking.track_geometry`) instead of once — but the calibrate-then-track
logic is the same.

### The SDS-TWR four-timestamp scheme

Two-way ranging is what lets us range *and* sync without a shared clock. One symmetric round
between initiator `i` and responder `j` records four timestamps, **each in its own device's
local clock**:

```
t1_local_i : i transmits the chirp        (i's clock)
t2_local_j : j receives it                 (j's clock)
t3_local_j : j replies (after proc delay)  (j's clock)
t4_local_i : i receives the reply          (i's clock)
```

Time-of-flight cancels both the offset and the skew:

```
ToF      = 0.5 * ((t4 - t1) - (t3 - t2))      # offset & skew cancel exactly
distance = ToF * speed_of_sound_mps
```

And the *relative clock* falls out of the same four numbers: per round the quantity
`0.5*((t2 - t1) + (t3 - t4))` is **linear in i's transmit time** `t1`, with slope ≈ relative
skew (`skew_j - skew_i`) and intercept ≈ relative offset (`offset_j - offset_i`). `clock_sync`
fits that line robustly (Theil–Sen) per pair, then solves a small least-squares clock graph.

So the *only* on-device requirement is: **stamp those four events in the device's own clock
and never pre-correct them.** Send the raw timestamps; the coordinator does the rest.

---

## 4. Implementing a real feed/agent against hardware

You have two integration points. Pick whichever fits your deployment:

* **Streaming/distributed:** fill in a live `DeviceAgent` (on each device) +
  `SocketDeviceFeed` (on the coordinator) — real-time.
* **Offline/replay:** capture to disk and use the
  [`RecordedAudioFeed`](../src/dronetracking/sources/recorded.py) (already implemented).

Either way, the contract is the same five accessors. Below is what each one takes on real
hardware.

### 4.1 Acoustic capture + detection (the mic path)

Capture a mono waveform with [`sounddevice`](https://python-sounddevice.readthedocs.io/) or
PyAudio at a fixed `sample_rate_hz` (e.g. 16 kHz → ~2 cm timing resolution at 343 m/s), and
record the local-clock time of sample 0 (`t0_local_s`). Then run the **existing matched
filter** — you do not write your own detector:

```python
from dronetracking.estimation.detection import detect_arrivals
# `captures` is {device_id: obj} where obj has .samples, .sample_rate_hz, .t0_local_s
detections = detect_arrivals(captures, reference_pulse, n_emissions=N, dt_s=DT)
# each DetectedArrival -> AcousticArrival(device_id, emission_idx, toa_local_s, source=0, confidence)
```

The detector matched-filters the recording against the **known drone pulse template**
(`reference_pulse`, a Hann-windowed linear chirp by default; see
[`sim/audio.py`](../src/dronetracking/sim/audio.py)), peak-picks the strongest well-separated
peaks, orders them in time, and reads the local arrival time off each peak's sample index.
The index→time convention (`toa_local = t0_local + start/fs`, `start = peak_full − (m−1)`)
is baked into both the synthesizer and the detector — keep them in sync.

> A live `sounddevice` capture helper is sketched (guarded behind an optional import) at the
> bottom of `recorded.py`'s module docstring; the **tested** deliverable is the recorded-file
> path. For a real-time agent, you would capture a rolling buffer, run `detect_arrivals` on
> each emission window, and publish the resulting `AcousticArrival`s in the batch.

### 4.2 Ranging (the speaker path)

To range against a peer, **emit the known chirp and record both ends**, timestamping the four
SDS-TWR events in each device's local clock:

1. initiator plays the chirp via `sounddevice.play` and stamps `t1_local_i` at playout;
2. responder's mic detects it (same matched filter) and stamps `t2_local_j`;
3. responder replies (plays its own chirp) after a known/processing delay, stamping `t3_local_j`;
4. initiator's mic detects the reply and stamps `t4_local_i`.

Assemble a `RangingRecord(initiator, responder, round_idx, t1_local_i, t2_local_j,
t3_local_j, t4_local_i)`. Run several `ranging_rounds` per pair so `estimation.ranging` can
reject outliers (median/MAD) and average. **Do not** offset-correct the timestamps — the raw
local stamps are the input the clock-sync stage needs.

> If your hardware has a dedicated ranging radio (UWB), use its native two-way-ranging
> timestamps instead of acoustic chirps — the `RangingRecord` four-timestamp shape is the same.

### 4.3 GPS (the anchor path)

On each GPS-equipped device, read the OS location service (e.g. `gpsd`, CoreLocation, Android
`LocationManager`) and report `AnchorGps(device_id, lat, lon, altitude_m)`. Non-anchor devices
report nothing here. These fixes anchor the relative solution to the real world via
`solve_transform`, so their accuracy directly bounds georeferencing accuracy.

### 4.4 Wiring the agent

The on-device agent mirrors [`DeviceAgent.publish`](../src/dronetracking/live/agent.py): build
this device's `(ranging, acoustic, anchor_gps)` slice from real sensors, `encode_batch(...)`
it, open a short-lived TCP connection to the coordinator, `sendall`, half-close, done. The
coordinator side is already complete — see
[`live/coordinator.py`](../src/dronetracking/live/coordinator.py) for the assemble-then-run
loop. The only change from the simulated agent is *where the slice comes from* (real sensors
vs. a `SimulatedDeviceFeed` slice); the protocol and the coordinator are identical.

---

## 5. Ingesting recorded data (the bridge you have today)

[`RecordedAudioFeed`](../src/dronetracking/sources/recorded.py) is a fully-implemented
`DeviceFeed` that replays a directory of recordings — the concrete "ingest real recorded
device data" path. It reads per-device WAVs, runs the matched-filter detector to recover
acoustic arrivals, and reads ranging + GPS + timebase from a sidecar `meta.json`.

### On-disk layout

```
{dir}/
  meta.json            # ranging + anchor GPS + timebase + detection params
  {device_id}.wav      # one mono WAV per acoustic device (float32 or int PCM)
```

`meta.json` (version 1) carries:

| key | meaning |
|---|---|
| `device_ids` | the stable downstream device order |
| `speed_of_sound_mps`, `sample_rate_hz` | the two timebase constants |
| `n_emissions` | drone emissions to recover per device (peak count) |
| `dt_s` | nominal emission spacing (matched-filter peak-separation guard) |
| `reference` | reference-pulse params (the `scenario.audio` keys: `pulse`, `f0`, `f1`, `pulse_dur_s`, …) used to rebuild the matched-filter template *bit-identically* |
| `t0_local_s` | per-device local-clock time of WAV sample 0 |
| `audio_files` | optional explicit `{device_id: filename}` (defaults to `{id}.wav`) |
| `batches` | one **`live/protocol` batch object per device** — that device's ranging + anchor GPS + timebase (acoustic empty, since it is recovered from the WAV) |

The `batches` entries are exactly what the live wire protocol emits, so a real coordinator can
persist each agent's published batch verbatim, drop the WAV beside it, and replay the whole
session offline. Use the bundled writer to produce a dataset from any feed (or from captured
audio):

```python
from dronetracking.sources.recorded import write_recorded_dataset
write_recorded_dataset(out_dir, feed, captures, scenario)   # captures: {device_id: AudioCapture}
```

### Pointing the pipeline at recorded data

```python
from dronetracking.sources.recorded import RecordedAudioFeed
from dronetracking.pipeline import run_pipeline

feed   = RecordedAudioFeed("/path/to/recording")
result = run_pipeline(scenario, feed=feed)     # geometry, clocks, TDOA, tracking, georef
print(len(result.tracks), "track(s);", len(result.estimates.geo_track.latlon), "georeferenced points")
```

(Because a recorded feed carries no ground truth, `result.world is None` and metrics are
skipped — exactly like a live feed.)

---

## 6. Known limits to plan around

* **Emission spacing must beat the range-delay spread.** The detector groups peaks into
  emissions by *time order*, assuming a device's consecutive arrivals are ~`dt_s` apart. If
  `dt_s` is smaller than the *cross-device* spread of the same emission's arrival times (the
  array's diameter / speed-of-sound), emissions overlap and the per-device peak↔emission
  mapping breaks. **Rule of thumb:** keep `dt_s` comfortably above `array_diameter / c`
  (e.g. a 200 m array at 343 m/s spreads ~0.6 s, so `dt_s = 2 s` in `detection_demo.yaml`
  leaves a wide margin).
* **Weak vertical observability for a near-coplanar ground array.** If all devices sit at
  nearly the same altitude (a flat field), the array is poorly conditioned in *z*: horizontal
  position and ground track are well constrained, but the drone's altitude estimate is soft.
  Spread devices in height where you can, or accept larger vertical error.
* **Georeferencing accuracy is GPS-anchor bound.** The relative solution is internally
  consistent to centimetre-scale timing, but the absolute lat/lon is only as good as the
  anchor GPS fixes (and their geometry). Consumer GPS (~2–5 m) caps absolute accuracy there;
  use survey-grade/RTK anchors for better.
* **Need ≥4 non-coplanar anchors.** Solving the rigid relative→world transform (rotation +
  translation, including altitude) needs at least four GPS anchors that are *not* all in one
  plane. Fewer (or coplanar) anchors leave the transform under-determined, especially in
  altitude/tilt. Plan anchor placement accordingly.
* **Clock drift is estimated, not assumed away** — but very large/unstable drift between
  ranging and tracking can degrade the lock. Re-run calibration if devices have been running
  long enough for oscillators to wander, or if temperature has shifted substantially.

---

## 7. Step-by-step bringup checklist

Each step has a concrete validation against the existing `eval` / `studies` tooling. Do them
in order — a failure at stage *n* will cascade into *n+1*.

### Stage 0 — Bench check in simulation (no hardware)

* [ ] `.venv/bin/python -m pytest tests/test_recorded.py tests/test_distributed.py -q` passes.
* [ ] Run the distributed loopback end-to-end:
  `.venv/bin/python -m dronetracking.live.coordinator --scenario scenarios/field_5dev.yaml`.
  Confirm it prints `matches simulator: True` and produces a track. This proves the
  protocol + coordinator + pipeline path works before any device is involved.

### Stage 1 — Single-device acoustic capture

* [ ] Record a known chirp on one device; confirm `sample_rate_hz` and that you can recover
  `t0_local_s`.
* [ ] Drop the WAV + a minimal `meta.json` into a directory and run
  `RecordedAudioFeed(dir).acoustic_arrivals()`. Confirm you get `n_emissions` detections at
  plausible times. (This is exactly what `tests/test_recorded.py` checks against truth — the
  detector lands within a few samples of the true arrival.)

### Stage 2 — Calibration: ranging → geometry + clocks

* [ ] Run several two-way-ranging rounds between every reachable pair; assemble
  `RangingRecord`s with raw local timestamps.
* [ ] Build a feed from a recording (or live agents) and check the distance matrix
  (`estimation.ranging.build_distance_matrix`) against a tape-measure of the real layout.
* [ ] Run `estimation.relative_localization.estimate_layout` and confirm the recovered array
  shape matches the physical layout (up to rotation/reflection — it is a *relative* frame).
* [ ] Run `estimation.clock_sync.estimate_clocks`; sanity-check that relative offsets/drifts
  are stable across repeated calibrations.

### Stage 3 — Georeferencing: GPS anchors

* [ ] Confirm ≥4 non-coplanar anchors are reporting `AnchorGps`.
* [ ] Run `estimation.georeference.solve_transform` and check the residuals (how well the
  transform maps each anchor's relative position to its GPS fix). Large residuals ⇒ bad
  anchor geometry or a bad fix.

### Stage 4 — Live tracking

* [ ] Fly (or place) the drone emitting the known pulse every `dt_s` (validate the
  `dt_s > array_diameter/c` margin from §6).
* [ ] Run the full pipeline: `run_pipeline(scenario, feed=<your feed>)`. Confirm
  `result.tracks` is non-empty and `result.estimates.geo_track.latlon` is populated.
* [ ] Eyeball the georeferenced track against the known flight (or a handheld GPS on the
  drone) for a smoke check.

### Stage 5 — Quantify with `studies`

You cannot compute RMSE on real data (no ground truth), so use the **simulator** to predict
how your real configuration *should* perform, then bracket your field results:

* [ ] Build a `Scenario` mirroring your real deployment (device count, layout, noise levels,
  `dt_s`, SNR).
* [ ] Sweep noise:
  `.venv/bin/python -m dronetracking.studies --scenario scenarios/field_5dev.yaml --kind noise --out-dir output`.
  Read off the expected `tracking.rmse_m` / `device_localization.rmse_m` /
  `georeferencing.rmse_m` at your noise level.
* [ ] Sweep device count (`--kind devices`) to see how dropping a device or an anchor (keeping
  ≥4) degrades accuracy + GDOP — useful for planning redundancy.
* [ ] Use these curves as the acceptance bar: if the recorded/live track diverges far more
  than the simulator predicts for your noise budget, suspect a calibration or detection
  problem and go back to the relevant stage.

---

## 8. Quick reference

| You have… | Use… |
|---|---|
| the simulator | `SimulatedDeviceFeed(scenario)` (carries `.world` truth) |
| recorded WAVs + `meta.json` | `RecordedAudioFeed(dir)` |
| live devices over the network | `DeviceAgent.publish(...)` per device + `SocketDeviceFeed` coordinator |
| any feed | `run_pipeline(scenario, feed=feed)` → tracks + georeferenced track |
| a deployment to size | `python -m dronetracking.studies --kind noise|devices` |
```
