/* Drone Tracking — browser sensor node.
 *
 * Speaks the webapp wire protocol exactly:
 *   POST /api/join   {name?}            -> {device_id, params:{report_interval_s, target_band_hz:[lo,hi]}}
 *   POST /api/report {device_id, t_client_ms, gps|null, audio|null} -> {ok:true}
 *   GET  /api/events                    -> SSE; each message = one state snapshot
 *
 * Field names are load-bearing and MUST match:
 *   gps   = {lat, lon, accuracy_m}
 *   audio = {level, detected, confidence, peak_hz}   (level = linear RMS 0..1, NOT dB)
 *
 * No build step. Standard browser APIs + Leaflet from CDN.
 */
"use strict";

(() => {
  // ------------------------------------------------------------------ config
  const DEFAULT_PARAMS = { report_interval_s: 0.5, target_band_hz: [120, 4000] };
  const STALE_MS = 5000;            // server prunes ~5s; we dim local markers similarly
  const FFT_SIZE = 2048;            // AnalyserNode frequency resolution
  const NOISE_ALPHA = 0.05;         // EMA factor for the running noise floor
  const DETECT_MARGIN = 1.8;        // band energy must exceed floor * margin to count as "detected"
  const DETECT_MIN_LEVEL = 0.012;   // absolute floor so silence never trips detection
  const SPECTRO_MIN_HZ = 0;
  const SPECTRO_MAX_HZ = 8000;      // spectrogram display range

  // ---- acoustic ranging (SDS-TWR) -----------------------------------------
  // Protocol: state.command.ranging = {round, initiator, responder,
  //   chirp:{f0,f1,dur_s}}. We POST half-exchanges in report.ranging:
  //   init -> {round, role:"init", t1, t4}; resp -> {round, role:"resp", t2, t3}.
  // distance = c * ((t4-t1) - (t3-t2)) / 2. All times are AudioContext.currentTime (s).
  const RESPONDER_TURNAROUND_S = 0.25;   // B waits this after hearing (t2) before replying (t3);
                                         // long enough to clear A's own-chirp reverb before the reply
  const RING_SECONDS = 2.5;              // rolling mic capture window we cross-correlate against
  const CAPTURE_FRAME = 2048;            // ScriptProcessor block size (mono samples per callback)
  const REPLY_WAIT_S = 0.8;              // initiator: listen this long (after the turnaround) for the reply
  const INCOMING_WAIT_S = 2.2;           // responder: poll up to this long for the incoming chirp
  const POLL_MS = 60;                    // responder detection poll cadence (detect early -> reply promptly)
  const XCORR_MIN_SCORE = 0.10;          // normalized correlation peak must clear this to count as an arrival
  const RANGING_STATUS_MS = 2600;        // how long the ranging status text stays lit after activity

  // ------------------------------------------------------------------ state
  const state = {
    deviceId: null,
    params: { ...DEFAULT_PARAMS },
    started: false,

    // audio
    audioCtx: null,
    analyser: null,
    micStream: null,
    freqData: null,          // Uint8Array(frequencyBinCount), 0..255
    timeData: null,          // Uint8Array(fftSize) for RMS
    sampleRate: 48000,
    noiseFloor: 0,           // running band-energy noise floor (EMA)
    audio: null,             // latest {level, detected, confidence, peak_hz}
    micState: "off",         // off | on | denied

    // location
    gps: null,               // latest {lat, lon, accuracy_m} from the Geolocation API
    gpsWatchId: null,
    gpsState: "off",         // off | on | denied
    manualPos: null,         // {lat, lon, accuracy_m} set by tapping the map (overrides gps)
    placing: false,          // tap-to-place mode active

    // networking
    reportTimer: null,
    es: null,                // EventSource
    sseConnected: false,
    sseRetry: 0,
    lastSnapshot: null,

    // acoustic ranging (SDS-TWR) — rolling mic capture + chirp matched-filter
    ranging: {
      ring: null, ringLen: 0, write: 0, samplesWritten: 0, headTime: 0,
      proc: null, lastRound: -1, busy: false, outbox: [], statusUntil: 0,
    },
  };

  // ------------------------------------------------------------------ DOM
  const $ = (id) => document.getElementById(id);
  const els = {
    connDot: $("connDot"), connText: $("connText"), deviceIdText: $("deviceIdText"),
    map: $("map"),
    panel: $("panel"), panelToggle: $("panelToggle"), panelTitle: $("panelTitle"),
    myRole: $("myRole"), capMic: $("capMic"), capGps: $("capGps"),
    levelFill: $("levelFill"), levelText: $("levelText"), peakText: $("peakText"), detText: $("detText"),
    spectro: $("spectro"),
    statDevices: $("statDevices"), statGps: $("statGps"), statDetect: $("statDetect"),
    compPos: $("compPos"), compSrc: $("compSrc"),
    sourceInfo: $("sourceInfo"), sourceErr: $("sourceErr"), sourceConf: $("sourceConf"),
    note: $("note"),
    deviceList: $("deviceList"),
    rangeStatus: $("rangeStatus"), distList: $("distList"), relCanvas: $("relCanvas"),
    placeBtn: $("placeBtn"), centerBtn: $("centerBtn"), qrBtn: $("qrBtn"),
    qrModal: $("qrModal"), qrCanvas: $("qrCanvas"), qrUrl: $("qrUrl"), qrClose: $("qrClose"),
    gate: $("gate"), nameInput: $("nameInput"), startBtn: $("startBtn"),
    gateError: $("gateError"),
    toasts: $("toasts"),
  };

  // ------------------------------------------------------------------ utils
  const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
  const fmtHz = (hz) => (hz == null ? "—" : hz >= 1000 ? (hz / 1000).toFixed(1) + " kHz" : Math.round(hz) + " Hz");

  function toast(msg, kind = "", ttl = 3600) {
    const el = document.createElement("div");
    el.className = "toast" + (kind ? " " + kind : "");
    el.textContent = msg;
    els.toasts.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity .3s, transform .3s";
      el.style.opacity = "0";
      el.style.transform = "translateY(8px)";
      setTimeout(() => el.remove(), 320);
    }, ttl);
  }

  // ------------------------------------------------------------------ join
  async function join(name) {
    const res = await fetch("/api/join", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(name ? { name } : {}),
    });
    if (!res.ok) throw new Error("join failed: HTTP " + res.status);
    const data = await res.json();
    state.deviceId = data.device_id;
    // Merge server params over defaults (be defensive about shape).
    const p = data.params || {};
    state.params = {
      report_interval_s:
        typeof p.report_interval_s === "number" && p.report_interval_s > 0
          ? p.report_interval_s
          : DEFAULT_PARAMS.report_interval_s,
      target_band_hz:
        Array.isArray(p.target_band_hz) && p.target_band_hz.length === 2
          ? p.target_band_hz.map(Number)
          : DEFAULT_PARAMS.target_band_hz.slice(),
    };
    els.deviceIdText.textContent = state.deviceId;
    return data;
  }

  // ------------------------------------------------------------------ microphone
  async function startMic() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("getUserMedia unavailable (needs https on phones)");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,   // we want the raw acoustic level, not VoIP processing
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    state.micStream = stream;

    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    if (ctx.state === "suspended") {
      try { await ctx.resume(); } catch (_) { /* resumed on first gesture below */ }
    }
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = FFT_SIZE;
    analyser.smoothingTimeConstant = 0.6;
    src.connect(analyser);
    // Note: we deliberately do NOT connect analyser to ctx.destination (no echo/feedback).

    state.audioCtx = ctx;
    state.analyser = analyser;
    state.sampleRate = ctx.sampleRate || 48000;
    state.freqData = new Uint8Array(analyser.frequencyBinCount);
    state.timeData = new Uint8Array(analyser.fftSize);
    state.micState = "on";

    startRangingCapture(ctx, src);  // rolling mic capture for acoustic ranging
    setupSpectro();
    requestAnimationFrame(audioFrame);
  }

  // Map an FFT bin index <-> frequency in Hz.
  const binToHz = (i) => (i * state.sampleRate) / FFT_SIZE;
  const hzToBin = (hz) => Math.round((hz * FFT_SIZE) / state.sampleRate);

  function audioFrame() {
    const an = state.analyser;
    if (!an) return;

    // --- RMS level in [0,1] from time-domain samples (linear amplitude, not dB) ---
    an.getByteTimeDomainData(state.timeData);
    let sumSq = 0;
    const N = state.timeData.length;
    for (let i = 0; i < N; i++) {
      const v = (state.timeData[i] - 128) / 128; // -> [-1, 1]
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / N);
    const level = clamp01(rms);

    // --- band energy within target_band_hz ---
    an.getByteFrequencyData(state.freqData); // 0..255 per bin
    const [loHz, hiHz] = state.params.target_band_hz;
    const loBin = Math.max(1, hzToBin(loHz));
    const hiBin = Math.min(state.freqData.length - 1, hzToBin(hiHz));

    let bandSum = 0, bandCount = 0, peakBin = loBin, peakVal = -1;
    for (let i = loBin; i <= hiBin; i++) {
      const m = state.freqData[i] / 255; // normalize 0..1
      bandSum += m * m;                  // power-ish
      bandCount++;
      if (m > peakVal) { peakVal = m; peakBin = i; }
    }
    const bandEnergy = bandCount ? Math.sqrt(bandSum / bandCount) : 0;
    const peakHz = peakVal > 0.04 ? binToHz(peakBin) : 0;

    // --- adaptive detection: running noise floor (EMA) + margin ---
    // Update the floor faster when current energy is below it (track quiet), slower otherwise.
    if (state.noiseFloor === 0) state.noiseFloor = bandEnergy;
    const a = bandEnergy < state.noiseFloor ? NOISE_ALPHA * 2 : NOISE_ALPHA;
    state.noiseFloor = (1 - a) * state.noiseFloor + a * bandEnergy;
    const threshold = Math.max(state.noiseFloor * DETECT_MARGIN, DETECT_MIN_LEVEL);
    const detected = bandEnergy > threshold && level > DETECT_MIN_LEVEL;

    // confidence in [0,1]: how far above threshold the band energy sits.
    let confidence = 0;
    if (threshold > 0) confidence = clamp01((bandEnergy - threshold) / (threshold + 1e-6));
    if (!detected) confidence = clamp01(confidence * 0.4);

    state.audio = {
      level: round4(level),
      detected,
      confidence: round4(confidence),
      peak_hz: round1(peakHz),
    };

    renderLocalAudio(state.audio);
    drawSpectro(state.freqData);
    requestAnimationFrame(audioFrame);
  }

  const round4 = (x) => Math.round(x * 1e4) / 1e4;
  const round1 = (x) => Math.round(x * 10) / 10;

  // ------------------------------------------------------------------ spectrogram
  let spectroCtx = null, spectroW = 0, spectroH = 0, spectroDpr = 1;
  function setupSpectro() {
    const cv = els.spectro;
    spectroDpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    spectroW = Math.max(120, Math.floor(rect.width));
    spectroH = Math.floor(rect.height) || 90;
    cv.width = Math.floor(spectroW * spectroDpr);
    cv.height = Math.floor(spectroH * spectroDpr);
    spectroCtx = cv.getContext("2d", { alpha: false });
    spectroCtx.scale(spectroDpr, spectroDpr);
    spectroCtx.fillStyle = "#060912";
    spectroCtx.fillRect(0, 0, spectroW, spectroH);
  }

  // Scrolling spectrogram: shift left by 1px, draw newest column at the right.
  function drawSpectro(freq) {
    if (!spectroCtx) return;
    const ctx = spectroCtx;
    const img = ctx.getImageData(spectroDpr, 0, (spectroW - 1) * spectroDpr, spectroH * spectroDpr);
    ctx.putImageData(img, 0, 0);

    const x = spectroW - 1;
    const loBin = Math.max(1, hzToBin(SPECTRO_MIN_HZ));
    const hiBin = Math.min(freq.length - 1, hzToBin(SPECTRO_MAX_HZ));
    const span = Math.max(1, hiBin - loBin);
    for (let y = 0; y < spectroH; y++) {
      // bottom = low freq, top = high freq
      const frac = 1 - y / spectroH;
      const bin = loBin + Math.floor(frac * span);
      const v = freq[bin] / 255; // 0..1
      ctx.fillStyle = magnitudeColor(v);
      ctx.fillRect(x, y, 1, 1);
    }

    // Overlay the target-band edges as faint guide lines.
    ctx.fillStyle = "rgba(77,163,255,0.25)";
    for (const hz of state.params.target_band_hz) {
      const frac = clamp01((hz - SPECTRO_MIN_HZ) / (SPECTRO_MAX_HZ - SPECTRO_MIN_HZ));
      const yy = Math.round((1 - frac) * spectroH);
      ctx.fillRect(0, yy, spectroW, 1);
    }
  }

  // Blue -> green -> yellow -> red heat ramp.
  function magnitudeColor(v) {
    v = clamp01(v);
    const r = clamp01(1.5 * v - 0.3) * 255;
    const g = clamp01(1.4 * v) * 255;
    const b = clamp01(0.7 - v) * 255 + (v < 0.05 ? 12 : 0);
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  }

  // ------------------------------------------------------------------ geolocation
  function startGps() {
    if (!("geolocation" in navigator)) {
      state.gpsState = "denied";
      renderCaps();
      toast("Location unavailable on this device", "warn");
      return;
    }
    state.gpsWatchId = navigator.geolocation.watchPosition(
      (pos) => {
        const c = pos.coords;
        state.gps = {
          lat: c.latitude,
          lon: c.longitude,
          accuracy_m: typeof c.accuracy === "number" ? round1(c.accuracy) : null,
        };
        if (state.gpsState !== "on") {
          state.gpsState = "on";
          renderCaps();
          toast("Location locked", "good");
        }
      },
      (err) => {
        // Permission denied or position unavailable -> degrade gracefully, keep sending gps:null.
        state.gps = null;
        if (err && err.code === err.PERMISSION_DENIED) {
          state.gpsState = "denied";
          toast("Location denied — detection still works", "warn");
        } else if (state.gpsState !== "on") {
          state.gpsState = "off";
          toast("Waiting for location fix…", "warn", 2500);
        }
        renderCaps();
      },
      { enableHighAccuracy: true, maximumAge: 2000, timeout: 15000 }
    );
  }

  // ------------------------------------------------------------------ report loop
  function startReporting() {
    const intervalMs = Math.max(100, state.params.report_interval_s * 1000);
    if (state.reportTimer) clearInterval(state.reportTimer);
    state.reportTimer = setInterval(sendReport, intervalMs);
    sendReport(); // fire one immediately
  }

  async function sendReport() {
    if (!state.deviceId) return;
    const body = {
      device_id: state.deviceId,
      t_client_ms: Date.now(),
      gps: state.manualPos || state.gps, // manual tap overrides GPS (e.g. a laptop with no GPS)
      audio: state.audio, // {level, detected, confidence, peak_hz} or null
    };
    // Attach any completed ranging half-exchanges (SDS-TWR), then clear the outbox.
    if (state.ranging.outbox.length) {
      body.ranging = state.ranging.outbox.splice(0, state.ranging.outbox.length);
    }
    try {
      await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        keepalive: true, // let it flush even if page is backgrounding
      });
    } catch (_) {
      // Transient network hiccup; the SSE connection indicator covers overall health.
    }
  }

  // ------------------------------------------------------------------ SSE / events
  function startEvents() {
    if (state.es) { try { state.es.close(); } catch (_) {} }
    const es = new EventSource("/api/events");
    state.es = es;

    es.onopen = () => {
      state.sseConnected = true;
      state.sseRetry = 0;
      renderConn();
    };
    es.onmessage = (ev) => {
      if (!ev.data) return;
      let snap;
      try { snap = JSON.parse(ev.data); } catch (_) { return; }
      state.lastSnapshot = snap;
      renderSnapshot(snap);
    };
    es.onerror = () => {
      // EventSource auto-reconnects; reflect the gap and back off our own re-open as a safety net.
      state.sseConnected = false;
      renderConn();
      if (es.readyState === EventSource.CLOSED) {
        const delay = Math.min(1000 * Math.pow(2, state.sseRetry++), 15000);
        setTimeout(startEvents, delay);
      }
    };
  }

  // ------------------------------------------------------------------ Leaflet map
  let map = null;
  let tileLayer = null;
  const devLayers = new Map(); // id -> {marker}
  let sourceMarker = null, sourceCircle = null;
  let lastFitCount = 0;

  function initMap() {
    map = L.map("map", { zoomControl: true, attributionControl: true })
      .setView([32.0809, 34.7806], 13); // Tel Aviv-ish default until positions arrive
    tileLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }).addTo(map);
    map.on("click", onMapClick); // tap-to-place when in placing mode
    // Recompute spectrogram size if the layout settled after map init.
    setTimeout(() => { if (state.analyser) setupSpectro(); }, 200);
  }

  // Color a device marker by level (quiet -> blue, loud -> red).
  function levelColor(level) {
    const v = clamp01(level);
    const r = clamp01(1.6 * v) * 255;
    const g = clamp01(1.3 - Math.abs(v - 0.5) * 1.6) * 255;
    const b = clamp01(1.1 - 1.8 * v) * 255;
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  }

  function deviceIcon(dev) {
    const isMe = dev.id === state.deviceId;
    const size = 18 + Math.round(clamp01(dev.level) * 14); // grows with level
    const cls = "dev-marker" + (dev.detected ? " detected" : "") + (isMe ? " me" : "");
    const html =
      `<div class="${cls}" style="width:${size}px;height:${size}px;background:${levelColor(dev.level)}"></div>`;
    return L.divIcon({
      className: "dev-divicon",
      html,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
  }

  function renderSnapshot(snap) {
    // --- status panel: network counts ---
    const net = snap.network || {};
    els.statDevices.textContent = net.n_devices ?? 0;
    els.statGps.textContent = net.n_gps ?? 0;
    els.statDetect.textContent = net.n_detecting ?? 0;

    // --- computed tags ---
    const comp = snap.computed || {};
    setTag(els.compPos, comp.positioning || "none");
    setTag(els.compSrc, comp.source || "none");

    // --- note ---
    els.note.textContent = snap.note || "";

    // --- my role line (derived from my capabilities + what the network is doing) ---
    els.myRole.textContent = describeRole(snap);

    // --- devices on the map ---
    const devices = Array.isArray(snap.devices) ? snap.devices : [];
    renderDeviceList(devices); // textual list so every device sees who's connected (even with no position)
    const seen = new Set();
    const positioned = [];

    for (const dev of devices) {
      seen.add(dev.id);
      const hasPos = dev.lat != null && dev.lon != null && dev.online !== false;
      if (!hasPos) {
        // No position (or offline) -> ensure no stale marker remains.
        removeDevLayer(dev.id);
        continue;
      }
      const latlng = [dev.lat, dev.lon];
      positioned.push(latlng);

      let entry = devLayers.get(dev.id);
      if (!entry) {
        const marker = L.marker(latlng, { icon: deviceIcon(dev), zIndexOffset: dev.id === state.deviceId ? 500 : 0 });
        marker.addTo(map);
        const label = (dev.name || dev.id) + (dev.id === state.deviceId ? " (you)" : "");
        marker.bindTooltip(label, { permanent: true, direction: "top", className: "dev-label", offset: [0, -8] });
        entry = { marker };
        devLayers.set(dev.id, entry);
      } else {
        entry.marker.setLatLng(latlng);
        entry.marker.setIcon(deviceIcon(dev));
        const label = (dev.name || dev.id) + (dev.id === state.deviceId ? " (you)" : "");
        const tip = entry.marker.getTooltip();
        if (tip && tip.getContent() !== label) entry.marker.setTooltipContent(label);
      }
    }
    // Drop markers for devices no longer present.
    for (const id of Array.from(devLayers.keys())) {
      if (!seen.has(id)) removeDevLayer(id);
    }

    // --- source marker + error circle ---
    renderSource(snap.source);

    // --- auto-fit once we have positions ---
    const fitPoints = positioned.slice();
    if (snap.source && snap.source.lat != null) fitPoints.push([snap.source.lat, snap.source.lon]);
    // Re-fit whenever the set of positioned points grows (a new device joins / gets placed).
    if (fitPoints.length > lastFitCount) {
      fitToPoints(fitPoints);
      lastFitCount = fitPoints.length;
    }

    // --- acoustic ranging: act on the round command + render distances / relative layout ---
    try {
      if (snap.command && snap.command.ranging) handleRanging(snap.command.ranging);
      renderDistances(Array.isArray(snap.distances) ? snap.distances : []);
      renderRelative(snap.relative || null);
    } catch (_) { /* ranging is best-effort; never break the rest of the UI */ }
    updateRangeStatus();
  }

  // --- connected-devices list (shows everyone, positioned or not) ---
  function renderDeviceList(devices) {
    const list = els.deviceList;
    if (!list) return;
    if (!devices.length) {
      list.innerHTML = '<div class="dev-item off"><span class="dev-name">no devices yet…</span></div>';
      return;
    }
    const frag = document.createDocumentFragment();
    for (const d of devices) {
      const me = d.id === state.deviceId;
      const online = d.online !== false;
      const pos = d.lat != null && d.lon != null;
      const row = document.createElement("div");
      row.className = "dev-item" + (me ? " me" : "") + (online ? "" : " off");

      const dot = document.createElement("span");
      dot.className = "dev-dot";
      dot.style.background = d.detected ? "#ff8a3d" : levelColor(d.level || 0);

      const name = document.createElement("span");
      name.className = "dev-name";
      name.textContent = (d.name || d.id) + (me ? " (you)" : "");

      const badges = document.createElement("span");
      badges.className = "dev-badges";
      badges.innerHTML =
        `<span class="dev-badge ${d.has_mic ? "on" : ""}">mic</span>` +
        `<span class="dev-badge ${pos ? "on" : ""}">${pos ? "pos" : "no pos"}</span>` +
        (d.detected ? `<span class="dev-badge det">hears it</span>` : "");

      row.appendChild(dot);
      row.appendChild(name);
      row.appendChild(badges);
      frag.appendChild(row);
    }
    list.innerHTML = "";
    list.appendChild(frag);
  }

  // --- manual placement / recenter / QR ---
  const round6 = (x) => Math.round(x * 1e6) / 1e6;

  function onMapClick(e) {
    if (!state.placing || !e || !e.latlng) return;
    state.manualPos = { lat: round6(e.latlng.lat), lon: round6(e.latlng.lng), accuracy_m: 0 };
    state.placing = false;
    els.placeBtn.classList.remove("active");
    els.placeBtn.textContent = "📍 Move me";
    if (state.gpsState !== "on") { state.gpsState = "on"; renderCaps(); }
    toast("Placed on the map", "good");
    lastFitCount = 0;     // allow a re-fit to include the new point
    sendReport();         // reflect immediately
  }

  function togglePlace() {
    state.placing = !state.placing;
    els.placeBtn.classList.toggle("active", state.placing);
    els.placeBtn.textContent = state.placing ? "tap the map…" : (state.manualPos ? "📍 Move me" : "📍 Place me");
    if (state.placing) toast("Tap the map where this device is", "", 4000);
  }

  function centerOnMe() {
    const p = state.manualPos || state.gps;
    if (p && map) map.setView([p.lat, p.lon], 18);
    else toast("No position yet — enable location, or tap “Place me”", "warn");
  }

  function openQR() {
    const url = window.location.href;
    els.qrUrl.textContent = url;
    els.qrCanvas.innerHTML = "";
    if (window.QRCode) {
      try {
        new window.QRCode(els.qrCanvas, {
          text: url, width: 220, height: 220,
          correctLevel: window.QRCode.CorrectLevel ? window.QRCode.CorrectLevel.M : undefined,
        });
      } catch (_) { qrFallback(url); }
    } else {
      qrFallback(url);
    }
    els.qrModal.classList.remove("hidden");
  }
  function qrFallback(url) {
    const img = document.createElement("img");
    img.width = 220; img.height = 220; img.alt = "QR";
    img.src = "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=" + encodeURIComponent(url);
    els.qrCanvas.appendChild(img);
  }
  function closeQR() { els.qrModal.classList.add("hidden"); }

  function removeDevLayer(id) {
    const e = devLayers.get(id);
    if (e) { try { map.removeLayer(e.marker); } catch (_) {} devLayers.delete(id); }
  }

  function renderSource(source) {
    if (!source || source.lat == null || source.lon == null) {
      els.sourceInfo.classList.add("hidden");
      if (sourceMarker) { map.removeLayer(sourceMarker); sourceMarker = null; }
      if (sourceCircle) { map.removeLayer(sourceCircle); sourceCircle = null; }
      return;
    }
    const ll = [source.lat, source.lon];
    const errM = typeof source.error_m === "number" && isFinite(source.error_m) ? source.error_m : 0;

    if (!sourceMarker) {
      const icon = L.divIcon({
        className: "source-divicon",
        html: '<div class="source-icon">🎯</div>',
        iconSize: [30, 30],
        iconAnchor: [15, 15],
      });
      sourceMarker = L.marker(ll, { icon, zIndexOffset: 1000 }).addTo(map);
      sourceMarker.bindTooltip("Estimated source", { direction: "top", className: "dev-label", offset: [0, -12] });
    } else {
      sourceMarker.setLatLng(ll);
    }

    if (!sourceCircle) {
      sourceCircle = L.circle(ll, {
        radius: Math.max(errM, 1),
        color: "#ff8a3d", weight: 2, fillColor: "#ff8a3d", fillOpacity: 0.12,
      }).addTo(map);
    } else {
      sourceCircle.setLatLng(ll);
      sourceCircle.setRadius(Math.max(errM, 1));
    }

    els.sourceErr.textContent = "±" + Math.round(errM) + " m";
    els.sourceConf.textContent = "conf " + (typeof source.confidence === "number" ? source.confidence.toFixed(2) : "—");
    els.sourceInfo.classList.remove("hidden");
  }

  function fitToPoints(points) {
    try {
      if (points.length === 1) {
        map.setView(points[0], 17);
      } else {
        map.fitBounds(L.latLngBounds(points).pad(0.25), { maxZoom: 18 });
      }
    } catch (_) { /* ignore bad bounds */ }
  }

  // ------------------------------------------------------------------ panel renderers
  function setTag(el, value) {
    el.textContent = value;
    el.className = "tag " + value;
  }

  function describeRole(snap) {
    const mic = state.micState === "on";
    const gps = state.gpsState === "on";
    const me = (snap.devices || []).find((d) => d.id === state.deviceId);
    const detecting = me && me.detected;
    const caps = [];
    if (mic) caps.push(detecting ? "listening (signal!)" : "listening");
    if (gps) caps.push("positioned");
    if (!mic && state.micState === "denied") caps.push("no mic");
    if (!gps && state.gpsState === "denied") caps.push("no gps");
    return caps.length ? caps.join(" · ") : "joining…";
  }

  function renderLocalAudio(a) {
    els.levelFill.style.width = Math.round(clamp01(a.level) * 100) + "%";
    els.levelText.textContent = "level " + a.level.toFixed(3);
    els.peakText.textContent = "peak " + fmtHz(a.peak_hz);
    if (a.detected) {
      els.detText.textContent = "DETECTED";
      els.detText.classList.add("on");
    } else {
      els.detText.textContent = "idle";
      els.detText.classList.remove("on");
    }
  }

  function renderCaps() {
    setCap(els.capMic, state.micState, "🎙 mic");
    setCap(els.capGps, state.gpsState, "📍 gps");
  }
  function setCap(el, st, label) {
    el.textContent = label;
    el.className = "cap " + (st === "on" ? "cap-on" : st === "denied" ? "cap-denied" : "cap-off");
  }

  function renderConn() {
    if (state.sseConnected) {
      els.connDot.className = "brand-dot is-on";
      els.connText.textContent = "live";
      els.connText.className = "pill";
    } else {
      els.connDot.className = "brand-dot is-off";
      els.connText.textContent = "reconnecting…";
      els.connText.className = "pill";
    }
  }

  // ------------------------------------------------------------------ panel toggle
  els.panelToggle.addEventListener("click", () => {
    const collapsed = els.panel.classList.toggle("collapsed");
    els.panelToggle.setAttribute("aria-expanded", String(!collapsed));
    if (!collapsed) setTimeout(() => { if (state.analyser) setupSpectro(); }, 50);
  });

  // ------------------------------------------------------------------ startup
  async function start() {
    els.startBtn.disabled = true;
    els.gateError.classList.add("hidden");
    const name = (els.nameInput.value || "").trim();

    // 1) Join the session (must succeed before anything else).
    try {
      await join(name);
    } catch (e) {
      els.gateError.textContent = "Couldn't reach the server. " + (e.message || "") + " — retry?";
      els.gateError.classList.remove("hidden");
      els.startBtn.disabled = false;
      return;
    }

    // 2) Reveal the app; start the event stream and report loop right away
    //    so a device that grants nothing still appears (online, no caps).
    els.gate.classList.add("hidden");
    state.started = true;
    initMap();
    startEvents();
    startReporting();

    // 3) Microphone (best-effort; needs https on phones, needs a user gesture — we're inside one).
    try {
      await startMic();
      renderCaps();
      toast("Microphone on", "good");
    } catch (e) {
      state.micState = (e && (e.name === "NotAllowedError" || e.name === "SecurityError")) ? "denied" : "off";
      renderCaps();
      const msg =
        state.micState === "denied"
          ? "Mic blocked. On a phone use the https link, then allow the mic."
          : "Mic unavailable: " + (e.message || e.name || "unknown");
      toast(msg, "bad", 6000);
    }

    // 4) Location (best-effort; independent of mic).
    startGps();
    renderCaps();

    // 5) iOS sometimes suspends AudioContext until a gesture — resume on next tap.
    const resume = () => {
      if (state.audioCtx && state.audioCtx.state === "suspended") state.audioCtx.resume();
    };
    document.addEventListener("touchend", resume, { passive: true });
    document.addEventListener("click", resume);
  }

  els.startBtn.addEventListener("click", start);
  els.nameInput.addEventListener("keydown", (e) => { if (e.key === "Enter") start(); });

  // Keep canvas crisp across rotation / resize.
  window.addEventListener("resize", () => {
    if (state.analyser && !els.panel.classList.contains("collapsed")) setupSpectro();
    if (map) map.invalidateSize();
  });

  // Re-arm the report cadence promptly when returning to the tab.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.started && state.deviceId) sendReport();
  });

  // ------------------------------------------------------------------ acoustic ranging
  // Two-way SDS-TWR: initiator emits a chirp (t1) and detects the reply (t4); responder
  // detects the chirp (t2) and emits a reply (t3). distance = c*((t4-t1)-(t3-t2))/2.
  // All times are this device's own AudioContext clock (offset cancels in the math).
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function makeChirp(ctx, f0, f1, dur) {
    const sr = ctx.sampleRate, n = Math.max(8, Math.floor(dur * sr));
    const buf = ctx.createBuffer(1, n, sr);
    const d = buf.getChannelData(0);
    const k = (f1 - f0) / dur;
    for (let i = 0; i < n; i++) {
      const t = i / sr;
      const ph = 2 * Math.PI * (f0 * t + 0.5 * k * t * t);
      const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)); // Hann window
      d[i] = Math.sin(ph) * w;
    }
    return buf;
  }

  function startRangingCapture(ctx, srcNode) {
    try {
      const r = state.ranging;
      r.ringLen = Math.ceil(RING_SECONDS * ctx.sampleRate);
      r.ring = new Float32Array(r.ringLen);
      r.write = 0; r.samplesWritten = 0; r.headTime = ctx.currentTime;
      const proc = ctx.createScriptProcessor(CAPTURE_FRAME, 1, 1);
      proc.onaudioprocess = (e) => {
        const inp = e.inputBuffer.getChannelData(0);
        const ring = r.ring, L = r.ringLen;
        for (let i = 0; i < inp.length; i++) { ring[r.write] = inp[i]; r.write = (r.write + 1) % L; }
        r.samplesWritten += inp.length;
        r.headTime = ctx.currentTime; // ~time of the most-recent sample
      };
      const sink = ctx.createGain(); sink.gain.value = 0; // silent: ScriptProcessor needs a sink, no feedback
      srcNode.connect(proc); proc.connect(sink); sink.connect(ctx.destination);
      r.proc = proc;
    } catch (_) { /* ranging capture unavailable -> ranging silently disabled */ }
  }

  const gIndexTime = (g) => state.ranging.headTime - (state.ranging.samplesWritten - g) / state.audioCtx.sampleRate;
  const timeGIndex = (t) => state.ranging.samplesWritten - (state.ranging.headTime - t) * state.audioCtx.sampleRate;

  // Matched-filter: find the best chirp arrival within [tFrom,tTo]; null if below threshold.
  function findChirp(template, tFrom, tTo) {
    const r = state.ranging, ctx = state.audioCtx;
    if (!r.ring || !ctx) return null;
    const M = template.length;
    const newest = r.samplesWritten - 1, oldest = r.samplesWritten - r.ringLen;
    const gFrom = Math.max(Math.floor(timeGIndex(tFrom)), oldest);
    const gTo = Math.min(Math.ceil(timeGIndex(tTo)), newest - M);
    if (gTo <= gFrom) return null;
    const span = (gTo - gFrom) + M + 1;
    const win = new Float32Array(span);
    for (let i = 0; i < span; i++) {
      const g = gFrom + i;
      win[i] = (g < oldest || g > newest) ? 0 : r.ring[((g % r.ringLen) + r.ringLen) % r.ringLen];
    }
    let tnorm = 0; for (let i = 0; i < M; i++) tnorm += template[i] * template[i];
    tnorm = Math.sqrt(tnorm) || 1;
    const lags = gTo - gFrom; let best = -1, bestLag = 0; const coarse = 4;
    const score = (lag) => {
      let dot = 0, e = 0;
      for (let i = 0; i < M; i++) { const s = win[lag + i]; dot += s * template[i]; e += s * s; }
      return dot / (Math.sqrt(e) * tnorm + 1e-9);
    };
    for (let lag = 0; lag <= lags; lag += coarse) { const sc = score(lag); if (sc > best) { best = sc; bestLag = lag; } }
    for (let lag = Math.max(0, bestLag - coarse); lag <= Math.min(lags, bestLag + coarse); lag++) {
      const sc = score(lag); if (sc > best) { best = sc; bestLag = lag; }
    }
    if (best < XCORR_MIN_SCORE) return null;
    return { time: gIndexTime(gFrom + bestLag), score: best };
  }

  function playChirpAt(ctx, buf, at) {
    try {
      const s = ctx.createBufferSource(); s.buffer = buf;
      const g = ctx.createGain(); g.gain.value = 0.9;
      s.connect(g); g.connect(ctx.destination); s.start(at);
      return at;
    } catch (_) { return at; }
  }

  function rangeLog(msg, toastIt) {
    state.ranging.statusMsg = msg;
    state.ranging.statusUntil = Date.now() + RANGING_STATUS_MS;
    if (toastIt) toast("📡 " + msg, "", 2600);
    updateRangeStatus();
  }
  function updateRangeStatus() {
    const el = els.rangeStatus; if (!el) return;
    const active = Date.now() < state.ranging.statusUntil;
    el.textContent = active ? (state.ranging.statusMsg || "ranging…") : "idle";
    el.className = "tag " + (active ? "ranging" : "none");
  }

  // Perform this device's half of a ranging round, then flush via report.
  async function handleRanging(cmd) {
    const r = state.ranging, ctx = state.audioCtx, me = state.deviceId;
    if (!cmd || !ctx || !r.ring || r.busy || cmd.round === r.lastRound) return;
    if (cmd.initiator !== me && cmd.responder !== me) return; // not my round
    r.lastRound = cmd.round; r.busy = true;
    const peer = cmd.initiator === me ? cmd.responder : cmd.initiator;
    try {
      const ch = cmd.chirp || {};
      const buf = makeChirp(ctx, ch.f0 || 18000, ch.f1 || 20000, ch.dur_s || 0.06);
      const template = buf.getChannelData(0).slice();
      const turn = RESPONDER_TURNAROUND_S;

      if (cmd.initiator === me) {
        rangeLog("pinging " + _devName(peer) + "…");
        const t1 = ctx.currentTime + 0.15;
        playChirpAt(ctx, buf, t1);
        await sleep((0.15 + turn + REPLY_WAIT_S) * 1000 + 200);
        // Search only AFTER the turnaround so we don't lock onto our own chirp's reverb.
        const hit = findChirp(template, t1 + turn * 0.7, t1 + turn + REPLY_WAIT_S);
        if (hit) { r.outbox.push({ round: cmd.round, role: "init", t1, t4: hit.time }); rangeLog("reply heard ✓", true); }
        else rangeLog("no reply heard from " + _devName(peer));
      } else {
        rangeLog("listening for " + _devName(peer) + "…");
        // Poll so we detect the incoming chirp EARLY and reply promptly (so the reply
        // lands inside the initiator's listening window).
        let t2 = null;
        const deadline = Date.now() + INCOMING_WAIT_S * 1000;
        while (Date.now() < deadline && t2 === null) {
          await sleep(POLL_MS);
          const hit = findChirp(template, ctx.currentTime - 0.35, ctx.currentTime);
          if (hit) t2 = hit.time;
        }
        if (t2 !== null) {
          const t3 = Math.max(ctx.currentTime + 0.04, t2 + turn);
          playChirpAt(ctx, buf, t3);
          r.outbox.push({ round: cmd.round, role: "resp", t2, t3 });
          rangeLog("heard ping → replied", true);
        } else rangeLog("no ping heard");
      }
    } catch (_) { /* swallow — ranging is best-effort */ }
    finally { r.busy = false; sendReport(); }
  }

  function _devName(id) {
    const d = (state.lastSnapshot && state.lastSnapshot.devices || []).find((x) => x.id === id);
    return (d ? (d.name || id) : id) + (id === state.deviceId ? " (you)" : "");
  }

  function renderDistances(dists) {
    const el = els.distList; if (!el) return;
    if (!dists.length) { el.innerHTML = '<div class="dist-empty">measuring inter-device distances…</div>'; return; }
    el.innerHTML = "";
    for (const x of dists) {
      const row = document.createElement("div");
      row.className = "dist-row";
      row.innerHTML = `<span>${_devName(x.a)} ↔ ${_devName(x.b)}</span>` +
                      `<span class="m">${Number(x.m).toFixed(1)} m</span>`;
      el.appendChild(row);
    }
  }

  function renderRelative(rel) {
    const cv = els.relCanvas; if (!cv) return;
    if (!rel || !Array.isArray(rel.xy_m) || rel.xy_m.length < 2) { cv.classList.remove("show"); return; }
    cv.classList.add("show");
    const ctx = cv.getContext("2d"); const W = cv.width, H = cv.height;
    ctx.fillStyle = "#070b16"; ctx.fillRect(0, 0, W, H);
    const xs = rel.xy_m.map((p) => p[0]), ys = rel.xy_m.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 26, spanX = Math.max(maxX - minX, 1), spanY = Math.max(maxY - minY, 1);
    const sc = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY);
    const tx = (x) => pad + (x - minX) * sc, ty = (y) => H - (pad + (y - minY) * sc);
    for (let i = 0; i < rel.xy_m.length; i++) {
      const id = rel.device_ids[i], me = id === state.deviceId;
      const x = tx(rel.xy_m[i][0]), y = ty(rel.xy_m[i][1]);
      ctx.beginPath(); ctx.arc(x, y, me ? 7 : 5, 0, 2 * Math.PI);
      ctx.fillStyle = me ? "#4da3ff" : "#cfd8ee"; ctx.fill();
      ctx.fillStyle = "#aeb8d4"; ctx.font = "11px system-ui"; ctx.textAlign = "center";
      ctx.fillText(_devName(id), x, y - 10);
    }
    ctx.fillStyle = "#6b7796"; ctx.font = "10px system-ui"; ctx.textAlign = "left";
    ctx.fillText("relative layout (m)", 6, H - 6);
  }

  // Map controls + share
  els.placeBtn.addEventListener("click", togglePlace);
  els.centerBtn.addEventListener("click", centerOnMe);
  els.qrBtn.addEventListener("click", openQR);
  els.qrClose.addEventListener("click", closeQR);
  els.qrModal.addEventListener("click", (e) => { if (e.target === els.qrModal) closeQR(); });

  renderConn();
  renderCaps();
})();
