# Web App — Acoustic Ranging (relative location) Contract

Add GPS-free relative localization to the browser app: devices measure inter-device
**distance** acoustically (two-way SDS-TWR via speaker chirps + mic cross-correlation),
the coordinator turns the distances into a **relative layout** (reusing the existing MDS
pipeline), and the page shows it. Two agents; one shared protocol extension below.

Honest scope: browser audio timing is jittery, so ranging is ROUGH (~1 m best case) and
needs real devices to validate. With exactly **2 devices** you get the distance between
them (1-D); **3+** gives a 2-D relative map. Keep manual placement as the fallback.

## Shared rules
- Repo `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`, venv `.venv`.
- Run only your own test file. Create/edit ONLY your assigned files.
- Reuse the FULL existing pipeline where possible: `estimation.ranging.build_distance_matrix` is for raw
  timestamps, but you already have measured DISTANCES — feed them into a `DistanceMatrix` directly and call
  `estimation.relative_localization.estimate_layout`. Also reuse `geo`, `transforms`.

## Protocol extension (both agents MUST match)
Two-way SDS-TWR per round between an ordered pair (initiator A, responder B):
A emits a chirp at A-local `t1`; B hears it at B-local `t2`; B emits a reply at B-local `t3`; A hears the
reply at A-local `t4`.  `distance = speed_of_sound * ((t4 - t1) - (t3 - t2)) / 2`.  (Offsets cancel; all
timestamps are each device's own `AudioContext.currentTime`, in seconds.)

- `GET /api/events` state gains:
  - `command`: `{ "ranging": { "round": int, "initiator": str, "responder": str,
      "chirp": { "f0": 1500, "f1": 4500, "dur_s": 0.05 } } | null }`  — the current instruction.
  - `relative`: `{ "device_ids": [...], "xy_m": [[x,y],...] } | null` — recovered relative layout (centered;
     arbitrary rotation). Present once ≥3 devices have mutual distances.
  - `distances`: `[ { "a": str, "b": str, "m": float } ]` — latest measured pairwise distances.
- `POST /api/report` body gains an optional `ranging` array of completed half-exchanges this device did:
  - initiator: `{ "round": int, "role": "init", "t1": float, "t4": float }`
  - responder: `{ "round": int, "role": "resp", "t2": float, "t3": float }`
  (Send the entry once for the round; the server pairs the two halves.)

---

## A — Ranging backend (testable)  (agent owns `webapp/ranging.py`, edits `webapp/server.py` + `webapp/session.py`, `tests/test_webapp_ranging.py`)
- `webapp/ranging.py`:
  - `sds_twr_distance(t1, t2, t3, t4, speed_of_sound_mps=343.0) -> float`.
  - `class RangingCoordinator`: schedules rounds across all online device pairs (round-robin), exposes
    `current_command(online_ids) -> dict|None` (advance every few seconds), `submit(device_id, entries)` to
    record half-exchanges, pairs halves by round -> distances (robust median per pair over repeats), and
    `distances() -> list[{a,b,m}]` + `distance_matrix(ids) -> DistanceMatrix` (NaN where unmeasured).
- Wire into `webapp/session.py`: hold a `RangingCoordinator`; `report()` passes any `ranging` entries to it;
  `state()` adds `command`, `distances`, and `relative` — build the `DistanceMatrix` from measured distances and,
  if ≥3 devices have a (near-)complete set, call `estimate_layout` and emit centered `xy_m`. Devices WITHOUT GPS
  can now be positioned from `relative` (anchor/orient to any GPS devices via `transforms.umeyama` if present, else
  show the bare relative layout). For exactly 2 devices, emit the single distance (no `relative` layout).
- Wire into `webapp/server.py`: `state()` already flows to SSE — just ensure the new keys pass through; reports
  with `ranging` reach the session. (Minimal edits.)
- Tests (`tests/test_webapp_ranging.py`): `sds_twr_distance` recovers a known distance from synthetic timestamps;
  feeding a `RangingCoordinator`/`Session` synthetic half-exchanges for a known geometry yields the right pairwise
  distances and, for ≥3 devices, a `relative.xy_m` whose pairwise distances match truth (after alignment) within a
  small tolerance; 2 devices -> correct single distance, no layout. Run `.venv/bin/python -m pytest tests/test_webapp_ranging.py -q`.

## B — Ranging frontend (browser)  (agent owns edits to `webapp/static/app.js`, `webapp/static/index.html`, `webapp/static/app.css`)
Not unit-testable here — write clean standard-Web-Audio code matching the protocol; keep it from breaking the
existing detection path (run `node --check app.js`).
- Generate a linear chirp buffer from `command.ranging.chirp` (Web Audio `AudioBuffer`).
- Continuously (or on-command) capture mic audio and **cross-correlate** with the chirp template to detect arrivals,
  timestamping peaks in `AudioContext.currentTime`.
- When `state.command.ranging` names me as initiator: play the chirp (record `t1` = scheduled emit time), then
  detect the reply (`t4`); POST `ranging:[{round,role:"init",t1,t4}]`. As responder: detect the incoming chirp
  (`t2`), play a reply (`t3`); POST `{round,role:"resp",t2,t3}`. Use a short fixed responder turnaround.
- UI: show measured `distances` (e.g. “phone ↔ computer ≈ 4.2 m”) and render `relative.xy_m` as a small
  relative-layout view (dots in meters) when present — so two devices’ relative location appears WITHOUT manual
  pinning. Keep the “Place me” manual option as a labelled fallback. Add a clear "Ranging…" status.
- Be defensive: if mic/AudioContext unavailable, skip ranging silently (detection still works).

---

## Output expected
Public APIs/signatures, the exact protocol fields emitted/consumed, test results (backend), and an honest note on
browser-audio accuracy + what needs real-device tuning.
