"""Local live dashboard: stream :class:`Snapshot`s to a browser map over SSE.

A dependency-free (stdlib ``http.server``) server. ``GET /`` serves a Leaflet page;
``GET /events`` runs a fresh :class:`StreamEngine` and pushes one JSON snapshot per
emission as a Server-Sent Event, paced to (scaled) real time. Open the printed URL.

When real hardware arrives, only the engine's data source changes — this server and
page are unchanged.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..sim.scenario import Scenario
from .engine import StreamEngine

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>DroneTracking — live: __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%}#map{height:100vh}
  #hud{position:absolute;top:10px;left:60px;z-index:1000;background:rgba(255,255,255,.9);
       padding:8px 12px;border-radius:6px;font:13px system-ui;box-shadow:0 1px 4px rgba(0,0,0,.3)}
  #hud b{color:#d62728}
  .lg{margin-top:4px;font-size:11px;color:#444}
</style></head>
<body><div id="map"></div>
<div id="hud"><div><b>DroneTracking</b> live — __TITLE__</div>
  <div id="status">connecting…</div>
  <div id="net" class="lg"></div>
  <div class="lg">🔵 device &nbsp; 🟢 GPS anchor / true drone &nbsp; 🔴 tracked drone (+1σ) &nbsp; — mesh links in gray</div></div>
<script>
const map = L.map('map').setView([__LAT__, __LON__], 17);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'© OpenStreetMap'}).addTo(map);
const COLORS=['#d62728','#9467bd','#17becf','#bcbd22','#e377c2','#8c564b'];
let devs=[], anch=[], tstate={}, truem={}, fitted=false;
const es = new EventSource('/events');
es.onmessage = (e)=>{
  const s = JSON.parse(e.data);
  document.getElementById('status').textContent =
    `t = ${s.t.toFixed(1)} s   frame ${s.index+1}/${s.total}   targets: ${s.targets.length}`;
  if(devs.length===0){
    (s.links||[]).forEach(L_=>L.polyline([L_.a,L_.b],
      {color:'#888',weight:1+2*(L_.quality||0),opacity:.25+.5*(L_.quality||0)}).addTo(map));
    s.devices.forEach(d=>devs.push(L.circleMarker([d.lat,d.lon],
      {radius:5,color:'#1f77b4',fill:true,fillColor:'#1f77b4',fillOpacity:.9}).addTo(map).bindTooltip('device '+d.id)));
    s.anchors.forEach(a=>anch.push(L.circleMarker([a.lat,a.lon],
      {radius:7,color:'#2ca02c',weight:3,fill:false}).addTo(map).bindTooltip('GPS anchor '+a.id)));
  }
  if(s.net && s.net.total!==undefined){
    document.getElementById('net').textContent =
      `network: ${s.net.connected?'connected':'PARTITIONED'}  ${s.net.online}/${s.net.total} online`
      + `  · battery ${(s.net.mean_battery*100).toFixed(0)}%  · link ${(s.net.mean_link_quality*100).toFixed(0)}%`;
  }
  s.targets.forEach((t,k)=>{
    const c=COLORS[k%COLORS.length];
    if(!tstate[t.id]){
      tstate[t.id]={trail:L.polyline([],{color:c,weight:3,opacity:.9}).addTo(map),
        marker:L.circleMarker([t.lat,t.lon],{radius:6,color:c,fill:true,fillColor:c,fillOpacity:1}).addTo(map).bindTooltip('drone '+t.id),
        circle:L.circle([t.lat,t.lon],{radius:Math.max(t.r_m,1),color:c,weight:1,fill:true,fillOpacity:.08}).addTo(map)};
    }
    const st=tstate[t.id];
    st.marker.setLatLng([t.lat,t.lon]).setTooltipContent(`drone ${t.id}  alt ${t.alt.toFixed(0)} m`);
    st.circle.setLatLng([t.lat,t.lon]).setRadius(Math.max(t.r_m,1));
    st.trail.addLatLng([t.lat,t.lon]);
  });
  s.true_targets.forEach(tt=>{
    if(!truem[tt.src]) truem[tt.src]=L.circleMarker([tt.lat,tt.lon],
      {radius:4,color:'#2ca02c',fill:true,fillColor:'#2ca02c',fillOpacity:.6}).addTo(map).bindTooltip('true drone '+tt.src);
    else truem[tt.src].setLatLng([tt.lat,tt.lon]);
  });
  if(!fitted && devs.length){ map.fitBounds(L.featureGroup(devs).getBounds().pad(0.6)); fitted=true; }
};
es.addEventListener('done', ()=>{ const e=document.getElementById('status'); e.textContent+='  — run complete'; es.close(); });
es.onerror = ()=>{ document.getElementById('status').textContent='stream ended (reload to replay)'; };
</script></body></html>
"""


def _page(scenario: Scenario) -> str:
    lat, lon = scenario.origin_latlon
    return (_PAGE.replace("__LAT__", repr(float(lat)))
                 .replace("__LON__", repr(float(lon)))
                 .replace("__TITLE__", scenario.name))


def _make_handler(scenario: Scenario, speed: float, detect: bool):
    page = _page(scenario).encode()
    pace = max(scenario.dt_s / max(speed, 1e-6), 0.0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/events"):
                self._stream()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                engine = StreamEngine(scenario, detect=detect)
                for snap in engine.snapshots():
                    payload = json.dumps(snap.to_dict())
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(pace)
                self.wfile.write(b"event: done\ndata: {}\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client navigated away

    return Handler


def serve(scenario: Scenario, host: str = "127.0.0.1", port: int = 8000,
          speed: float = 1.0, detect: bool = False) -> None:
    """Start the live dashboard server (blocks until Ctrl-C)."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(scenario, speed, detect))
    url = f"http://{host}:{port}"
    print(f"Live dashboard for '{scenario.name}' at  {url}")
    print(f"  (playback speed {speed}x; open the URL, Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        httpd.server_close()
