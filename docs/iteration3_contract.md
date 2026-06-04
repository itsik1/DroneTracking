# Iteration 3 — Integration Contract (read this first)

Three more pieces extend the working iteration-1/2 testbed. Each agent owns **NEW,
disjoint files** + unit tests; the orchestrator integrates afterward. Same rules as before.

## Project facts
- Repo `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`, package `dronetracking` (src layout), venv `.venv`.
- Run tests: `.venv/bin/python -m pytest tests/<your_file>.py -q`. Run only YOUR file(s). TDD: test first.
- **Ground-truth firewall:** files under `dronetracking/estimation/` must NOT import `dronetracking.sim`
  (enforced by `tests/test_no_truth_leak.py`). The `network/` and `sources/` packages are NOT estimation and
  MAY import `sim` freely (they are infrastructure / sim adapters). Any TEST may import sim.

## Hard rules
- **Create only your assigned files.** Do NOT edit shared files (`sim/*`, `config.py`, `pipeline.py`, `run.py`,
  `estimation/*` except your own new module, `eval/*`, `viz/*`, `live/*`). The orchestrator integrates.
  You MAY add a new `scenarios/*.yaml`.
- Reuse existing modules; don't reinvent.

## Already scaffolded (use, don't modify)
- `sim/scenario.py` `DeviceSpec` now has `battery_frac: float=1.0`, `has_mic: bool=True`, `has_speaker: bool=True`
  (plus existing `has_gps`, `velocity_mps`, `position_at(t)`). `Scenario` has `network: dict={}` (and `audio`,
  `extra_drones`, `gps_blackout`, properties `all_drones`/`devices_move`/`gps_available`).
- Contract types: `Observations`, `RangingRecord`, `AcousticArrival(…, source, confidence)`, `AnchorGps(…, altitude_m)`,
  `World` (has `true_tracks`, `positions_matrix_at`), `RelativeLayout`, `ClockEstimates`, `TargetFix`, `Track`, `GeoTrack`.
- Reusable APIs: `sim.simulator.simulate`, `config.load_scenario`, all `estimation.*` stages,
  `transforms.umeyama/gdop/gn_covariance`, `geo.*`.

---

## Ph1 — Network Formation  (agent owns `network/`)
Files: `src/dronetracking/network/__init__.py`, `network/node.py`, `network/transport.py`,
`network/discovery.py`, `tests/test_network.py`, optional `scenarios/network_demo.yaml`.

Realize the spec's Phase 1 in simulation. May import `sim` (uses true positions to decide which radio links
physically exist — discovery doesn't assume known positions, but the *simulator* of the radio layer does).

- `node.py`: `@dataclass Node(id, battery_frac, has_mic, has_speaker, has_gps, online=True, confidence=1.0)` plus
  `from_spec(DeviceSpec) -> Node`. A `capabilities()` helper and a `NodeRegistry` mapping id -> Node with
  membership/health queries.
- `transport.py`: `Transport` ABC (`send(src, dst, payload)`, `deliver()` / link model) and
  `SimulatedTransport(positions, comm_range_m, latency_s, loss_prob, kind="wifi", rng=...)` modeling per-link
  reachability (within `comm_range_m`), latency, and packet loss. `link_quality(a, b) -> float` in [0,1].
  The spec lists BLE/Wi-Fi/mesh — model `kind` presets with different range/latency/loss defaults.
- `discovery.py`: `discover(devices, transport, rng) -> NetworkGraph` — each node broadcasts HELLO, hears peers it
  can reach, forming an adjacency graph; and a `NetworkManager(scenario)` that builds the registry + graph,
  exposes `form_network()`, `neighbors(id)`, `is_connected()` (graph connectivity), and `health()` (per-node +
  network summary: online count, mean battery, mean link quality, isolated nodes).
- Tests: devices within `comm_range_m` are neighbors and others aren't; the graph is connected for a dense layout
  and partitioned for a sparse one; capabilities/battery surface from specs; transport drops ~`loss_prob` of
  packets over many trials (seeded); `is_connected`/`health` correct on hand-built layouts.
- `scenarios/network_demo.yaml` (optional): devices with varied `battery_frac`/`has_mic`/`has_speaker` and a
  `network: {comm_range_m: ..., loss_prob: ..., kind: ...}` block.

## Device-feed / hardware abstraction  (agent owns `sources/`)
Files: `src/dronetracking/sources/__init__.py`, `sources/base.py`, `sources/simulated.py`,
`sources/live.py`, `tests/test_sources.py`.  (Package is `sources`, NOT `io` — avoid shadowing stdlib `io`.)

Define the boundary the estimation/streaming layer reads from, so a real device network can replace the simulator.
May import `sim`.

- `base.py`: `class DeviceFeed(ABC)` with abstract methods/properties: `device_ids() -> tuple[str,...]`,
  `ranging_records() -> tuple[RangingRecord,...]`, `acoustic_arrivals() -> tuple[AcousticArrival,...]`,
  `anchor_gps() -> tuple[AnchorGps,...]`, `speed_of_sound_mps -> float`, `sample_rate_hz -> float`, and a concrete
  `as_observations() -> Observations` that bundles them (so existing batch code consumes a feed unchanged).
- `simulated.py`: `SimulatedDeviceFeed(scenario)` runs `simulate(scenario)` once and implements every method from the
  resulting `Observations`; also exposes `.world` (sim-only ground truth, for eval). This is the reference feed.
- `live.py`: `LiveDeviceFeed` skeleton — same ABC, every method raising `NotImplementedError` with a docstring
  describing exactly what a real implementation must supply (ranging exchanges from the transport layer, arrivals
  from on-device acoustic detection, GPS from device receivers, etc.). This documents the hardware contract.
- Tests: `SimulatedDeviceFeed(scenario).as_observations()` equals `simulate(scenario)[0]` field-for-field (same seed);
  the ABC cannot be instantiated directly; `LiveDeviceFeed` methods raise `NotImplementedError`.

## Joint clock + position estimation  (agent owns `estimation/joint_clock.py`)
Files: `src/dronetracking/estimation/joint_clock.py`, `tests/test_joint_clock.py`.  (Firewall: NO `sim` import in source.)

The seam left in iteration 1: when clock sync leaves a residual per-device timing error, co-estimate small clock
nuisance corrections together with the target position (prior-regularized) in the TDOA solve.

- `localize_emission_joint(arrivals, clocks, layout, speed_of_sound, *, clock_prior_s=..., toa_var_s2=...) -> TargetFix`:
  start from the same range-difference formulation as `estimation.tdoa.localize_emission` (you may import it for the
  closed-form seed / compare), but augment the least-squares unknowns with a residual clock offset `δ_i` per device
  (relative to the reference), each tied to 0 by a Gaussian prior row `δ_i / clock_prior_s`. Solve with
  `scipy.optimize.least_squares` (`loss="soft_l1"`); report position + covariance (`transforms.gn_covariance` on the
  position block) + gdop + residual_rms. Document that with redundancy this absorbs residual clock error the plain
  solver cannot.
- Tests: take a scenario, build the TRUE layout and a ClockEstimates that is deliberately PERTURBED from truth
  (inject e.g. 1e-4–5e-4 s residual offset on a couple of devices). Show `localize_emission_joint` recovers the true
  drone position with materially lower error than `tdoa.localize_emission` given the same wrong clocks; and that on
  clean clocks it matches plain TDOA (no regression). Build inputs via the frozen sim leaf functions in the TEST.

---

## Output expected from each agent
Return: public functions/classes + final signatures, new test file(s) + pass counts, any new scenario, and any
foreseen interaction with other phases / the orchestrator. Note deviations.
