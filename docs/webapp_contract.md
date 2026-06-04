# Web App — Integration Contract (read this first)

A zero-install, capability-adaptive browser app. Three agents, disjoint files, one shared
HTTP/JSON protocol (below). Stdlib-only server (no new Python deps); Leaflet from CDN.

## Shared rules
- Repo `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`, package `dronetracking`, venv `.venv`.
- Run tests: `.venv/bin/python -m pytest tests/<file> -q`. Run only YOUR file(s). TDD where testable.
- Create ONLY your assigned files. `webapp/` is infrastructure (may import `sim`/`estimation`/`geo`/`network`).
- Reuse: `geo.{latlon_to_enu,enu_to_latlon,haversine_m}`, `transforms`, `config`, `network` (for health), and the
  estimation stages if/when ranging data is present.

## The wire protocol (browser <-> server) — ALL THREE AGENTS MUST MATCH THIS EXACTLY

JSON over HTTP. Endpoints (served under `/api`):
- `POST /api/join`  body `{ "name": str? }`  ->  `{ "device_id": str, "params": { "report_interval_s": 0.5,
  "target_band_hz": [120, 4000] } }`
- `POST /api/report`  body:
  ```json
  { "device_id": "d3", "t_client_ms": 1719,
    "gps": { "lat": 32.08, "lon": 34.78, "accuracy_m": 8.0 } | null,
    "audio": { "level": 0.42, "detected": true, "confidence": 0.7, "peak_hz": 180.0 } | null }
  ```
  `level` is linear RMS amplitude in [0,1] (NOT dB). Returns `{ "ok": true }`.
- `GET /api/events`  -> Server-Sent Events; each message is one state snapshot:
  ```json
  { "devices": [ { "id","name","lat"|null,"lon"|null,"has_gps":bool,"has_mic":bool,
                   "level":float,"detected":bool,"confidence":float,"online":bool } ],
    "source": { "lat":float,"lon":float,"confidence":float,"error_m":float } | null,
    "network": { "n_devices":int,"n_gps":int,"n_detecting":int,"connected":bool },
    "computed": { "positioning":"gps"|"ranging"|"none", "source":"energy"|"region"|"none" },
    "note": str }
  ```
- Static: `GET /` -> the SPA (`index.html`); `GET /app.js`, `GET /app.css` from `webapp/static/`.

Devices that haven't reported within ~5 s are marked `online:false` and dropped from localization.

---

## A — Web server  (agent owns `webapp/server.py`, `webapp/__main__.py`, `tests/test_webapp_server.py`)
Stdlib `http.server` (`ThreadingHTTPServer`) — no new deps. Implements the protocol above:
- Routes: `POST /api/join` (assign a device id; register), `POST /api/report` (update that device's latest state +
  last-seen time), `GET /api/events` (SSE: every `report_interval_s` push the current state from the Session),
  static file serving from `webapp/static/`.
- Holds a `Session` (from agent B — code against the interface below; in TESTS use a tiny in-test fake Session so
  you don't depend on agent B). Calls `session.upsert_device(id, name)`, `session.report(id, payload)`, and
  `session.state()` for the SSE snapshots; prunes stale devices via `session`.
- `webapp/__main__.py`: `python -m dronetracking.webapp [--host 0.0.0.0] [--port 8000] [--https] [--cert FILE --key FILE]`.
  Default HTTP on localhost. `--https` wraps the socket with `ssl` (generate a self-signed cert with `openssl` if
  `--cert` not given, or document the one-liner). On start, print the LAN URL(s) and a clear note that phones need
  the https/tunnel URL for mic access.
- Tests: a mock HTTP client (`http.client`/`urllib`) JOINs 3 devices, POSTs reports (with/without gps+audio), and
  reads one `/api/events` SSE message; assert device count, that a reported device appears, and JSON shape. Use an
  in-test fake Session implementing `upsert_device/report/state/prune`.

`Session` interface the server codes against (provided by agent B):
```python
class Session:
    def upsert_device(self, device_id: str, name: str | None) -> None
    def report(self, device_id: str, payload: dict) -> None      # the /api/report body
    def prune(self, max_age_s: float = 5.0) -> None
    def state(self) -> dict                                       # the /api/events snapshot above
```

## B — Adaptive session  (agent owns `webapp/session.py`, `tests/test_webapp_session.py`)
The brain: turn whatever devices reported into the richest state possible.
- `class Session` implementing the interface above. Track per device: name, last gps, last audio, last-seen time,
  has_gps/has_mic (inferred from whether gps/audio have ever been non-null).
- `state()` computes, capability-adaptively:
  - **positioning**: devices with a GPS fix are placed at their lat/lon (`positioning:"gps"`). (If a device has no
    GPS but ranging data exists, you MAY use `estimation` to place it — optional; otherwise `lat/lon=null`.)
  - **source localization by ENERGY (sync-free, the key method)**: among devices that are `detected` and have a
    position, model received power `~ A / r^2`. Convert GPS -> local ENU (`geo.latlon_to_enu` about the device
    centroid), solve for source `(x,y)` (+ amplitude `A`) by nonlinear least squares
    (`scipy.optimize.least_squares`) on the level residuals (work in power = level**2). >=3 detecting+positioned
    devices -> a point fix (`source:"energy"`); exactly 2 -> a coarse level-weighted region/midpoint
    (`source:"region"`); else `source:null`/`"none"`. Convert the ENU solution back to lat/lon
    (`geo.enu_to_latlon`); set `error_m` from the fit residual / spread, `confidence` from #devices & detection.
  - **network** summary (counts, `connected` = >=1 online).
- `prune(max_age_s)` marks/drops devices not seen recently.
- Tests: build synthetic devices at known GPS points; place a synthetic source and set each device's `level` to
  `sqrt(A)/r` (so power = A/r^2); after `report`s, assert `state()["source"]` lat/lon is within a tolerance of the
  true source for >=3 detecting devices; 2 devices -> a region (not None); 0 detecting -> source None; pruning works;
  positioning reflects GPS. Keep device counts small.

## C — Browser frontend  (agent owns `webapp/static/index.html`, `webapp/static/app.js`, `webapp/static/app.css`)
The page each device opens. Cannot be unit-tested here (no browser/mic) — write clean, self-contained standard-API
code that MATCHES the protocol exactly; the orchestrator will load-test it via the server.
- On load: `POST /api/join` -> device_id. Then start:
  - **Mic**: `navigator.mediaDevices.getUserMedia({audio})` -> WebAudio `AnalyserNode`; compute RMS `level` (0..1)
    and band energy in `target_band_hz`; `detected` = band energy over an adaptive threshold; `peak_hz` = dominant
    bin; draw a live spectrogram/level meter on a `<canvas>`.
  - **GPS**: `navigator.geolocation.watchPosition` -> lat/lon/accuracy (degrade gracefully if denied -> gps null).
  - Every `report_interval_s`, `POST /api/report` with the current gps + audio.
  - `EventSource('/api/events')` -> render a **Leaflet** map (OpenStreetMap tiles, CDN): device markers (colored by
    level; ring if detected; label name), the estimated **source** marker + an `error_m` circle, and a status panel
    (devices online, my role/capabilities, what's being computed). Show clear prompts to grant mic/location.
- Must work on mobile Safari/Chrome (responsive, large tap targets). No build step — plain HTML/JS/CSS + Leaflet CDN.

---

## Output expected from each agent
Public API/signatures (server routes; Session methods; the JSON shapes you emit/consume), test results,
mock-vs-needs-hardware notes, and how the orchestrator wires it together (server holds the real Session; serves
`webapp/static/`).
