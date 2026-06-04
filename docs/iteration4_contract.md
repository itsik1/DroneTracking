# Iteration 4 — Integration Contract (read this first)

Three forward-looking pieces toward a real-time, genuinely distributed, quantified system.
Each agent owns NEW, disjoint files + tests; the orchestrator integrates. Same rules as
prior iterations.

## Project facts
- Repo `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`, package `dronetracking` (src layout), venv `.venv`.
- Run tests: `.venv/bin/python -m pytest tests/<your_file>.py -q`. Run only YOUR file(s). TDD: test first.
- **Ground-truth firewall:** `dronetracking/estimation/` must NOT import `dronetracking.sim` (enforced by
  `tests/test_no_truth_leak.py`). `live/`, `sources/`, `studies/` are NOT estimation and may import freely.
- **Create only your assigned files.** Don't edit shared files; the orchestrator integrates. You may add scenarios.

## Reusable APIs (all working)
- `config.load_scenario`, `sim.simulator.simulate`, `sources.simulated.SimulatedDeviceFeed`,
  `sources.base.DeviceFeed`, `sources.live.LiveDeviceFeed`, `sim.observations.{Observations,RangingRecord,AcousticArrival,AnchorGps}`.
- `estimation.tdoa.localize_emission`, `estimation.multi_target.{localize_frames,track_targets}`,
  `estimation.tracking.{track_target,_KalmanFilter,Tracker}`, `estimation.interfaces.{TargetFix,Track,...}`,
  `transforms.*`, `geo.*`.
- `pipeline.run_pipeline(scenario, *, model, sigma_a, detect, joint_clock, clock_prior_s, feed) -> PipelineResult`
  (`.metrics` is a flat dict with keys like `tracking.rmse_m`, `device_localization.rmse_m`, `georeferencing.rmse_m`).

---

## A — Online incremental tracker  (agent owns `estimation/online_tracker.py`)
Files: `src/dronetracking/estimation/online_tracker.py`, `tests/test_online_tracker.py`.  Firewall: NO `sim` import in source.

A STATEFUL tracker that updates one frame at a time (the streaming engine currently re-runs the batch
`track_targets` over the whole prefix every frame — replace that with O(tracks×fixes)-per-frame online updates).

- `class OnlineTracker` with:
  - `__init__(self, *, model="cv", sigma_a=2.0, gate_chi2=..., birth_min_hits=2, death_max_misses=3)`
  - `update(self, fixes: Sequence[TargetFix], t: float) -> None` — predict live tracks to `t`, associate the
    frame's fixes via global-nearest-neighbour (`scipy.optimize.linear_sum_assignment` on Mahalanobis distance,
    chi-square gate), update matched tracks (reuse `estimation.tracking._KalmanFilter`), spawn tentative tracks
    for unmatched fixes (confirm after `birth_min_hits`), and increment misses (terminate after `death_max_misses`).
  - `tracks(self) -> list[Track]` — confirmed tracks so far, with stable `target_id`s.
  - Keep per-track history so `tracks()` returns full `Track`s (times_s, positions_local, covariances).
- Tests: feed the per-emission frames of a multi-drone scenario (build via `multi_target.localize_frames` on
  true layout+clocks) into the OnlineTracker frame-by-frame; assert it ends with the right number of confirmed
  tracks near truth, identities stable, and results comparable to batch `track_targets` on the same frames.
  Also a single-target case. Confirm it is incremental (state persists; calling `update` N times does not
  reprocess earlier frames — e.g. assert intermediate `tracks()` grows sensibly).

## B — Distributed runtime over sockets  (agent owns `live/protocol.py`, `live/agent.py`, `sources/socket_feed.py`)
Files: those three + `tests/test_distributed.py`. May import `sim`/`sources`.

Make the system genuinely distributed: each device publishes ITS OWN measurements over TCP; a coordinator feed
assembles them. This is the concrete hardware bridge (swap the per-device sim slice for real sensors later).

- `live/protocol.py`: pure encode/decode of a per-device measurement batch to/from a line-delimited JSON message:
  `encode_batch(device_id, ranging, acoustic, anchor_gps, speed_of_sound_mps, sample_rate_hz) -> bytes` and
  `decode_batch(data) -> dict`. Round-trip must preserve the contract dataclasses (reconstruct `RangingRecord`,
  `AcousticArrival`, `AnchorGps`). Pure + fully unit-tested.
- `live/agent.py`: `class DeviceAgent` representing on-device code. `publish(host, port, scenario, device_id, feed=None)`
  computes THIS device's slice from a `SimulatedDeviceFeed` — ranging records where `initiator == device_id`,
  acoustic arrivals where `device_id` matches, anchor_gps where `device_id` matches — connects to host:port, sends
  the encoded batch, closes. (Each device only ever sees/sends its own data.)
- `sources/socket_feed.py`: `class SocketDeviceFeed(DeviceFeed)` — binds a TCP listener (support port 0 ->
  OS-assigned, expose `.port`), `collect(expected_device_ids, timeout_s=...)` accepts connections and reads each
  agent's batch until all expected devices reported (or timeout), then implements every `DeviceFeed` method by
  unioning the collected batches into a single `Observations` (via `as_observations()`).
- Tests (`tests/test_distributed.py`): (1) protocol round-trips a batch exactly; (2) LOOPBACK — start a
  `SocketDeviceFeed` on port 0 in a daemon thread calling `collect(device_ids)`, spawn one `DeviceAgent.publish`
  per device (threads) against `127.0.0.1:feed.port`, then assert `feed.as_observations()` equals
  `SimulatedDeviceFeed(scenario).as_observations()` (same ranging/acoustic/anchor sets, order-insensitive). Use
  generous timeouts + proper join; must be deterministic, not flaky.

## C — Robustness / accuracy study  (agent owns `studies/`)
Files: `src/dronetracking/studies/__init__.py`, `studies/sweep.py`, `studies/__main__.py`, `tests/test_studies.py`.
May import `config`/`pipeline` freely.

Quantify the system: run `run_pipeline` across parameter grids and report accuracy vs the parameter.

- `sweep.py`:
  - `sweep_noise(base_scenario, factors, seeds) -> dict` — scale the scenario's NoiseSpec by each factor (use
    `dataclasses.replace`), run `run_pipeline` over seeds, aggregate (mean/median) `tracking.rmse_m`,
    `device_localization.rmse_m`, `georeferencing.rmse_m`.
  - `sweep_devices(...)` — vary the number of participating devices (subset the scenario's devices; keep >=4
    anchors) and report accuracy + a representative GDOP.
  - Return a structured, JSON-serializable result (param values + aggregated metrics per point).
  - `plot_sweep(result, out_dir) -> list[Path]` — matplotlib error-vs-parameter curves (Agg backend).
- `__main__.py`: `python -m dronetracking.studies --scenario ... --kind noise|devices --out-dir output` runs a
  sweep, writes the JSON + plots, prints a short table.
- Tests: a small `sweep_noise` (2-3 factors, 2 seeds) on `field_5dev` runs and shows the expected monotonic trend
  (more noise -> larger tracking RMSE); results JSON-serializable; `plot_sweep` writes non-empty PNGs to `tmp_path`.

---

## Output expected from each agent
Public API + final signatures, new test file(s) + pass counts, any new scenario, foreseen interaction with the
orchestrator (how to wire it in), and any deviation.
