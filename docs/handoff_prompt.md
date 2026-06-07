# DroneTracking — New-Session Handoff Prompt

> Paste everything below into a fresh session (running on the user's Mac).

---

You are continuing an existing project at `/Users/itsikshapira/Developer/ItsikProjects/DroneTracking`
(git repo, branch `main`, pushed to github.com/itsik1/DroneTracking). Orient first:
`git log --oneline -20`, then skim `README.md`, `docs/webapp_contract.md`, and
`docs/webapp_ranging_contract.md`.

## What the project is
A distributed **acoustic drone-localization** system: ordinary devices localize themselves
**without GPS** (acoustic ranging), detect & track drones acoustically (TDOA), georeference to
lat/lon — all **without synchronized clocks**. It began as a Python simulation and now also has a
**zero-install browser app** for real phones/laptops, which is the current focus.

## Environment (important)
- Python 3.9, venv at `.venv`. ALWAYS invoke `.venv/bin/python` — the user's shell `python`/`python3`
  is the system one and does NOT have the package. (Or `source .venv/bin/activate`.)
- Tests: `.venv/bin/python -m pytest -q` (~330 tests, all green). `-m "not slow"` for the fast subset.
- Sim run: `.venv/bin/python -m dronetracking.run --scenario scenarios/field_5dev.yaml`
- Web app: `.venv/bin/python -m dronetracking.webapp --tunnel` (prints an https link + QR for phones).

## What's already built (simulation, fully tested)
Phases 1–9 of the vision, behind a **ground-truth firewall** (`estimation/` must NOT import `sim/`;
enforced by `tests/test_no_truth_leak.py`). Packages: `sim/` (synthetic world: positions, clocks with
offset+drift, drone trajectories; emits only measurable Observations), `estimation/`
(ranging→DistanceMatrix→**MDS** relative layout, clock_sync, tdoa, tracking incl. `online_tracker`,
georeference, joint_clock, detection [matched filter], separation [multi-source], multi_target),
`eval/`, `viz/` (folium maps), `studies/` (accuracy sweeps), `network/` (Ph1 discovery/transport),
`sources/` (DeviceFeed: simulated/recorded/socket + live skeleton), `device/` (on-device capture agent),
`live/` (streaming engine + coordinator). Sim accuracy: device-loc ~cm, tracking cm–dm, georef
GPS-anchor-bound (~2 m). Honest finding: vertical observability ≪ horizontal.

## CURRENT FOCUS — the browser web app (`dronetracking.webapp`) on REAL devices
Each device opens a URL, grants **mic + location**; a stdlib HTTP coordinator does what the connected
devices allow (adaptive by count: 1=detect, 2=detect+distance, 3+=relative layout + energy source fix).
- `webapp/server.py` — stdlib HTTP. `POST /api/join`, `POST /api/report`, `GET /api/events` (SSE),
  `GET /api/state` (polling fallback), static files. `--debug` logs each device report.
- `webapp/session.py` — the brain. Places devices by GPS; **energy-based source localization**
  (level ∝ 1/distance, sync-free); turns acoustic-ranging distances into a **relative MDS layout**.
  Emits state `{devices, source, network, computed, command, distances, relative, note}`.
- `webapp/ranging.py` — `sds_twr_distance` + `RangingCoordinator` (schedules two-way ranging rounds).
- `webapp/static/{index.html,app.js,app.css}` — the browser node: getUserMedia mic (level + detection +
  spectrogram), geolocation, Leaflet map, "Place me" manual placement, "Share" QR, **and acoustic
  ranging** (chirp + cross-correlation matched filter; two-way SDS-TWR on the server's round command).
- `webapp/__main__.py` — CLI: `--tunnel` (runs cloudflared, prints link + QR), `--https` (self-signed),
  `--debug`, `--no-qr`.
- Protocol: report → `{device_id, t_client_ms, gps|null, audio|null, ranging?[]}`;
  state → adds `command.ranging`, `distances[]`, `relative{device_ids,xy_m}`. SDS-TWR:
  `distance = c*((t4-t1)-(t3-t2))/2`, all times each device's own AudioContext clock.

## THE OPEN PROBLEM (debug this with the user)
The user tests with a **phone + Mac laptop over a Cloudflare tunnel**. Status, newest first:
1. **[just fixed — VERIFY]** Phone showed "zero devices" while the PC saw the phone. Cause: SSE
   buffered/dropped by the tunnel/mobile. Fix: added `GET /api/state` + client polling
   (`app.js` `startPolling`). Confirm the phone now lists both devices.
2. **Acoustic ranging distances "not calculating" / devices "don't recognize each other."** Fixed a
   real timing bug (responder now detects-early and replies promptly so the reply lands in the
   initiator's listening window). Chirp moved to **near-ultrasonic 18–20 kHz** (was audible). This
   path is **UNTESTED on real hardware** and browser audio timing is jittery (~meter accuracy at best).
3. **"Detection is zero":** the network detecting-count was the dead-stream bug (fixed). Real detection
   only fires for a loud in-band sound; the local **level meter** is the mic-alive indicator.

**KEY DIAGNOSTIC:** `.venv/bin/python -m dronetracking.webapp --tunnel --debug` logs every join +
report (incl. ranging timestamps `rng:init t1/t4`, `rng:resp t2/t3`). From the user's run, determine:
(a) are BOTH devices reporting? (polling fix), (b) is the phone's mic `lvl` > 0? (capture working —
iOS may suspend the AudioContext until a tap), (c) are ranging entries being POSTed by both? If the
in-app status chip says "no ping/reply heard," the ultrasonic chirp isn't being heard between devices →
lower `DEFAULT_CHIRP` in `webapp/ranging.py` (try 16000–18000 Hz) and/or raise the chirp gain in
`app.js` `playChirpAt`, then re-test.

You (the assistant) **cannot reach the phone or the tunnel** — debug via the `--debug` server log + the
in-app status chip; the user is the hands-on tester. Iterate from their reported observations.

## Realities / constraints
- Phones need **HTTPS** for mic + geolocation → `--tunnel` or `--https`. Laptops usually have **no GPS**
  (use "Place me" or acoustic ranging for their position).
- **2 devices → distance only** (1-D); 3+ → 2-D relative layout. Sub-meter precise localization needs
  ≥4 time-synced sensors (the Python pipeline, not the browser).
- Browser acoustic ranging is experimental; phone speaker/mic response varies near 20 kHz.

## How the user likes to work (from memory)
- Build the **complete, capability-adaptive** thing (do what it can with whatever devices connect);
  don't pare down to a limited demo and don't ask which slice to build.
- Use **parallel sub-agents** for big builds; proceed **autonomously** (no permission prompts).
- **Commit + push per logical chunk**, Conventional Commits; end commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Be honest about what's tested vs what needs real hardware.

## Your first steps
1. `git log --oneline -20`; read `README.md` + `docs/webapp_ranging_contract.md`.
2. Confirm green: `.venv/bin/python -m pytest -q`.
3. Ask the user to run `.venv/bin/python -m dronetracking.webapp --tunnel --debug`, open the link on
   phone + laptop, grant **mic + location on both**, then report: the `--debug` terminal lines, and on
   the phone — the "Connected devices" list, whether the level meter moves when they clap, and the
   "Relative location" status chip text.
4. Fix the next issue from that evidence (most likely: chirp band/volume tuning, or iOS mic-capture).
