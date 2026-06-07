---
name: dronetracking-webapp
description: >-
  Developer reference for the DroneTracking zero-install browser app under
  src/dronetracking/webapp/ — the CURRENT FOCUS: real phones + a laptop join a
  URL over a Cloudflare tunnel, capture mic + GPS, and the stdlib HTTP server
  fuses them into live source localization and acoustic relative-location. Load
  this whenever you touch the web server, the adaptive Session, the browser
  client (index.html/app.js/app.css), acoustic SDS-TWR ranging in the browser,
  the --tunnel/--https/--debug launcher, the /api wire protocol, or are
  DEBUGGING on-device symptoms ("zero devices", "no ping/reply heard",
  "detection zero", chirp band/volume tuning, iOS mic, SSE vs /api/state). Also
  load for tests test_webapp_server, test_webapp_session, test_webapp_ranging.
  Trigger on file names server.py, session.py (webapp), ranging.py (webapp),
  __main__.py, app.js, index.html, app.css.
---

# DroneTracking — Web app (real phones, current focus)

Zero-install, capability-adaptive browser app. Devices install nothing — they
open a URL, grant mic + location, and become nodes. Only the PC running the
coordinator needs the repo/venv. Stdlib-only Python server (no new deps);
Leaflet + QRCode from CDN; no build step. Run it:

```bash
.venv/bin/python -m dronetracking.webapp --tunnel --debug   # public https link + QR; logs each report
# --https for a LAN self-signed cert instead; --no-qr to suppress the terminal QR
```

Phones need https for mic access → use `--tunnel` (needs `cloudflared`: `brew install cloudflared`) or `--https`.

## Files

- `server.py` — stdlib `ThreadingHTTPServer`. `make_server(session, host, port, debug)` and `serve(...)`. Routes below. `--debug` logs each report (gps/level/detection + ranging halves) via `_log_report`. Prunes devices unseen > `MAX_DEVICE_AGE_S=5.0`. Device ids are `d0, d1, ...`.
- `session.py` — `class Session` (the brain). Methods `upsert_device(id, name)`, `report(id, payload)`, `prune(max_age_s=5.0)`, `state() -> dict`. Holds a `RangingCoordinator`.
- `ranging.py` — `sds_twr_distance(t1,t2,t3,t4, speed_of_sound_mps=343.0)` and `class RangingCoordinator` (scheduling + distance recovery). `DEFAULT_CHIRP = {"f0":18000.0,"f1":20000.0,"dur_s":0.06}`, `ROUND_PERIOD_S=3.0`.
- `__main__.py` — CLI: `--host` (0.0.0.0), `--port` (8000), `--tunnel`, `--https`, `--cert/--key`, `--no-qr`, `--debug`. Tunnel mode spawns `cloudflared tunnel --url http://localhost:PORT`, scrapes the `*.trycloudflare.com` URL, prints a banner + QR.
- `static/{index.html, app.js, app.css}` — the SPA each device opens.

## Wire protocol (browser ↔ server, under `/api`)

- `POST /api/join` `{name?}` → `{device_id, params:{report_interval_s:0.5, target_band_hz:[120,4000]}}`
- `POST /api/report` `{device_id, t_client_ms, gps:{lat,lon,accuracy_m}|null, audio:{level,detected,confidence,peak_hz}|null, ranging?:[...]}` → `{ok:true}`. `level` is linear RMS in [0,1], NOT dB. Ranging halves: initiator `{round, role:"init", t1, t4}`, responder `{round, role:"resp", t2, t3}`.
- `GET /api/events` → SSE; one state snapshot per `report_interval_s`.
- `GET /api/state` → **one-shot** snapshot (the polling fallback — same JSON as SSE).
- `GET /`, `/app.js`, `/app.css` → static.

Snapshot shape (`session.state()`):
```json
{ "devices":[{"id","name","lat","lon","has_gps","has_mic","level","detected","confidence","online"}],
  "source":{"lat","lon","confidence","error_m"}|null,
  "network":{"n_devices","n_gps","n_detecting","connected"},
  "computed":{"positioning":"gps"|"none","source":"energy"|"region"|"none"},
  "note": str,
  "command":{"ranging":{"round","initiator","responder","chirp":{f0,f1,dur_s}}|null},
  "distances":[{"a","b","m"}],
  "relative":{"device_ids":[...],"xy_m":[[x,y],...]}|null }
```
Note the implementation reports `positioning` as `"gps"|"none"` (GPS-less devices are placed via the separate `relative` layout, not by setting `positioning:"ranging"`).

## Capability-adaptive ladder (in `session.py`)

- **1 device** → live level + detection only.
- **2 devices** → their **distance** (acoustic ranging); if both detect + positioned, a coarse `source:"region"` (level-weighted midpoint, deliberately coarse error).
- **3+ devices** → `source:"energy"` (nonlinear least-squares: `level² ~ A/(r²+eps)` in ENU via `scipy.optimize.least_squares`, `trf`) **and** a `relative` MDS layout (`estimate_layout` on the measured `DistanceMatrix`, centered; aligned to GPS via `transforms.umeyama` with `with_scaling=False, allow_reflection=True` if ≥2 kept devices have GPS).

Source localization is **sync-free** — it uses received sound level (∝ 1/distance), so it works on ordinary phones with no clock sync. Positions come from GPS; acoustic ranging is the GPS-denied fallback.

## Acoustic ranging (browser, `ranging.py` + `app.js`)

Two-way SDS-TWR per ordered pair: `distance = c·½((t4−t1)−(t3−t2))` (offsets cancel; all timestamps are each device's own `AudioContext.currentTime`). `RangingCoordinator`: `current_command(online_ids, now)` round-robins ordered pairs every `ROUND_PERIOD_S`; `submit(device_id, entries)` pairs the two halves by round and computes a distance, gating implausible values (>1000 m, negative, non-finite); `distances()` returns the robust **median** per pair; `distance_matrix(ids)` packs them (NaN/weight-0 where unmeasured) for MDS. The browser side (`app.js`) makes a Hann-windowed linear chirp, keeps a `RING_SECONDS=2.5` rolling mic buffer, and **cross-correlates** with the chirp template (`XCORR_MIN_SCORE=0.10`) to timestamp arrivals; initiator plays at t1 then listens for the reply (t4); responder detects (t2) and replies after `RESPONDER_TURNAROUND_S=0.25` (t3).

## Open problems & debugging (you can't reach the phone — use `--tunnel --debug` log + the in-app status chip; the user is the tester)

1. **"zero devices" on the phone** — long-lived SSE is frequently buffered/dropped over Cloudflare tunnels & mobile browsers. **Fixed** with the `/api/state` polling fallback (`app.js` `startPolling()`, every ~0.4–0.5 s). Both run; polling is what actually feeds phones behind the tunnel. If devices still don't appear, confirm polling is firing and `state()` lists them.
2. **Ranging distances not computing** — fixed a responder-timing bug; chirp moved to near-ultrasonic 18–20 kHz, **untested on hardware**; browser timing is rough. If the status chip says "no ping/reply heard": lower `DEFAULT_CHIRP` in `webapp/ranging.py` toward ~16–18 kHz (some speakers roll off above 20 kHz; Nyquist is ~22 kHz at 44.1 kHz) and/or raise chirp gain in `app.js` `playChirpAt` (currently 0.9). Also consider lowering `XCORR_MIN_SCORE` if replies are weak.
3. **"detection zero"** — real detection only fires on loud in-band sound: threshold = `max(noise_floor × DETECT_MARGIN(1.8), DETECT_MIN_LEVEL(0.012))` over `target_band_hz=[120,4000]`. The **local level meter is the mic-alive indicator**; iOS may suspend `AudioContext` until a user tap. To make detection more sensitive, lower `DETECT_MARGIN`/`DETECT_MIN_LEVEL` in `app.js` and retest.

## Gotchas

- **https required for mic** — `getUserMedia` is secure-context only; plain HTTP on LAN won't grant the mic.
- **iOS AudioContext** starts suspended; resumed on first gesture — until then no level/ranging.
- Stdlib server: no middleware/CORS/router — routes are matched by hand in `server.py`; static serving is path-traversal-guarded to `static/`.
- No build step: `index.html`/`app.js`/`app.css` are served as-is; keep `app.js` valid (`node --check app.js`) so the page doesn't break the detection path.
- Two contracts document the intended design: `docs/webapp_contract.md` and `docs/webapp_ranging_contract.md` (the implementation has drifted slightly — e.g. `positioning:"gps"|"none"`, chirp band 18–20 kHz).
