"""Stdlib HTTP/JSON + SSE server for the zero-install browser app.

A single :class:`~http.server.ThreadingHTTPServer` (no new Python deps) speaks the wire
protocol from ``docs/webapp_contract.md``:

- ``POST /api/join``    -> assign a ``device_id``, register it, return ``{device_id, params}``
- ``POST /api/report``  -> hand the *full* report body to the Session, return ``{"ok": true}``
  (the body may carry an optional ``ranging`` array of SDS-TWR half-exchanges; it is
  forwarded verbatim and consumed by ``Session.report``)
- ``GET  /api/events``  -> Server-Sent Events; push ``session.state()`` every
  ``report_interval_s`` (pruning stale devices each tick). The snapshot is serialized
  whole, so any keys the Session adds — including ``command`` / ``distances`` /
  ``relative`` for acoustic ranging — flow through unchanged.
- static files: ``GET /`` -> ``static/index.html``; ``GET /app.js`` / ``GET /app.css``
  -> from ``static/`` (tolerating not-yet-written files with a clean 404)

The server holds a :class:`Session` (agent B's "brain"); it only depends on the small
interface ``upsert_device(id, name) / report(id, payload) / prune(max_age_s) / state()``.
In tests a tiny in-test fake Session is injected; the orchestrator injects the real
:class:`dronetracking.webapp.session.Session` — nothing else changes.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

# Parameters handed to every joining device (the client polls/report at this cadence).
DEFAULT_PARAMS = {
    "report_interval_s": 0.5,
    "target_band_hz": [120, 4000],
}

# Devices unseen for longer than this are pruned from localization each SSE tick.
MAX_DEVICE_AGE_S = 5.0

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _log_report(body: dict) -> None:
    """Print a one-line summary of a device report (for ``--debug``)."""
    did = body.get("device_id", "?")
    gps = "Y" if body.get("gps") else "-"
    audio = body.get("audio") or {}
    lvl = audio.get("level")
    parts = [f"gps={gps}", f"lvl={lvl:.3f}" if isinstance(lvl, (int, float)) else "lvl=-",
             f"det={'Y' if audio.get('detected') else '-'}"]
    for e in (body.get("ranging") or []):
        role = e.get("role")
        rnd = e.get("round")
        if role == "init":
            parts.append(f"rng:init r{rnd} t1={e.get('t1')} t4={e.get('t4')}")
        elif role == "resp":
            parts.append(f"rng:resp r{rnd} t2={e.get('t2')} t3={e.get('t3')}")
    print(f"[{time.strftime('%H:%M:%S')}] report {did} " + " ".join(parts))

# Content types for the (small, fixed) set of static assets we serve.
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml; charset=utf-8",
    ".ico": "image/x-icon",
}

# Map a request path to a file under static/. Only these paths are reachable, which
# also keeps the server safe from path traversal (no user-controlled file paths).
_STATIC_ROUTES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/app.js": "app.js",
    "/app.css": "app.css",
}


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that doesn't dump tracebacks for benign disconnects.

    With HTTP/1.1 keep-alive, a handler thread loops back to read the next request on a
    persistent connection; when a client (browser tab, ``http.client``) simply closes,
    the pending read raises ``ConnectionResetError``/``BrokenPipeError``. Those are
    expected and would otherwise be logged by ``socketserver.handle_error`` as scary
    stack traces. We swallow only those and defer anything else to the base handler.
    """

    daemon_threads = True

    def handle_error(self, request, client_address):  # noqa: D401
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return  # client went away mid-request — nothing to report
        super().handle_error(request, client_address)


class _DeviceRegistry:
    """Thread-safe allocator of short, stable device ids (``d0``, ``d1``, ...)."""

    def __init__(self) -> None:
        self._next = 0
        self._lock = threading.Lock()

    def new_id(self) -> str:
        with self._lock:
            did = f"d{self._next}"
            self._next += 1
            return did


def _make_handler(session, registry: _DeviceRegistry, static_dir: Path, debug: bool = False):
    params_bytes = json.dumps(DEFAULT_PARAMS).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        # Speak HTTP/1.1 so keep-alive SSE streams behave; we always set explicit
        # Content-Length on finite responses and never on the stream.
        protocol_version = "HTTP/1.1"
        server_version = "DroneTrackingWebApp/1.0"

        def log_message(self, *args):  # keep the console clean
            pass

        # --- helpers --------------------------------------------------------
        def _send_json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def _read_json_body(self) -> Optional[dict]:
            """Parse the request body as a JSON object, or None on any problem."""
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except (TypeError, ValueError):
                return None
            raw = self.rfile.read(length) if length > 0 else b""
            if not raw:
                return {}
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return obj if isinstance(obj, dict) else None

        # --- routing --------------------------------------------------------
        def do_POST(self):  # noqa: N802 (http.server API)
            if self.path == "/api/join":
                self._handle_join()
            elif self.path == "/api/report":
                self._handle_report()
            else:
                self._send_error_json(404, "not found")

        def do_GET(self):  # noqa: N802 (http.server API)
            path = self.path.split("?", 1)[0]
            if path == "/api/events":
                self._handle_events()
            elif path == "/api/state":
                self._handle_state()
            elif path in _STATIC_ROUTES:
                self._serve_static(_STATIC_ROUTES[path])
            else:
                self._send_error_json(404, "not found")

        # --- API handlers ---------------------------------------------------
        def _handle_join(self) -> None:
            body = self._read_json_body()
            if body is None:
                self._send_error_json(400, "invalid JSON body")
                return
            name = body.get("name")
            if name is not None and not isinstance(name, str):
                name = str(name)
            device_id = registry.new_id()
            try:
                session.upsert_device(device_id, name)
            except Exception as exc:  # never let a Session bug 500 silently
                self._send_error_json(500, f"session error: {exc}")
                return
            if debug:
                print(f"[{time.strftime('%H:%M:%S')}] join {device_id} {name!r}")
            self._send_json({"device_id": device_id, "params": DEFAULT_PARAMS})

        def _handle_report(self) -> None:
            body = self._read_json_body()
            if body is None:
                self._send_error_json(400, "invalid JSON body")
                return
            device_id = body.get("device_id")
            if not isinstance(device_id, str) or not device_id:
                self._send_error_json(400, "missing device_id")
                return
            try:
                session.report(device_id, body)
            except Exception as exc:
                self._send_error_json(500, f"session error: {exc}")
                return
            if debug:
                _log_report(body)
            self._send_json({"ok": True})

        def _handle_events(self) -> None:
            """Server-Sent Events: one ``session.state()`` snapshot per interval.

            Prunes stale devices each tick. Returns quietly when the client
            disconnects (BrokenPipeError / ConnectionResetError).
            """
            interval = float(DEFAULT_PARAMS["report_interval_s"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            # Disable proxy buffering so events arrive promptly behind nginx etc.
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                # An initial comment flushes headers and confirms the stream is live.
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while not getattr(self.server, "_shutting_down", False):
                    try:
                        session.prune(MAX_DEVICE_AGE_S)
                        snapshot = session.state()
                        payload = json.dumps(snapshot)
                    except Exception as exc:
                        payload = json.dumps({"error": f"session error: {exc}"})
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # client navigated away / closed the tab
            except OSError:
                pass  # socket torn down underneath us (e.g. on shutdown)

        def _handle_state(self) -> None:
            """One-shot state snapshot — the polling fallback for SSE.

            A long-lived ``text/event-stream`` is frequently buffered or dropped by
            tunnels (Cloudflare) and mobile browsers, which leaves a phone seeing nothing.
            A plain JSON GET works everywhere, so the client polls this when SSE is silent.
            """
            try:
                session.prune(MAX_DEVICE_AGE_S)
                snapshot = session.state()
            except Exception as exc:
                self._send_error_json(500, f"session error: {exc}")
                return
            self._send_json(snapshot)

        # --- static files ---------------------------------------------------
        def _serve_static(self, filename: str) -> None:
            file_path = static_dir / filename
            # Resolve and confirm the file is inside static_dir (defense in depth;
            # the route table already restricts filenames to a fixed allowlist).
            try:
                resolved = file_path.resolve()
                resolved.relative_to(static_dir.resolve())
            except (ValueError, OSError):
                self._send_error_json(404, "not found")
                return
            if not resolved.is_file():
                # The frontend agent may not have written this asset yet.
                self._send_error_json(404, "not found")
                return
            try:
                data = resolved.read_bytes()
            except OSError:
                self._send_error_json(404, "not found")
                return
            content_type = _CONTENT_TYPES.get(
                resolved.suffix.lower(), "application/octet-stream"
            )
            self._send_bytes(data, content_type)

    return Handler


def make_server(session, host: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> ThreadingHTTPServer:
    """Build (but do not start) a :class:`ThreadingHTTPServer` bound to ``host:port``.

    Pass ``port=0`` to let the OS assign a free port (read it back from
    ``server.server_address``). The returned server is wired to ``session`` and a fresh
    device-id registry; the caller starts it via ``serve_forever()`` (see :func:`serve`)
    or in a thread (see the tests).
    """
    registry = _DeviceRegistry()
    handler = _make_handler(session, registry, _STATIC_DIR, debug=debug)
    httpd = _QuietThreadingHTTPServer((host, port), handler)
    # Daemonize per-request threads so a hung SSE stream never blocks shutdown.
    httpd.daemon_threads = True
    httpd._shutting_down = False  # SSE loops watch this to exit promptly
    return httpd


def serve(
    session,
    host: str = "0.0.0.0",
    port: int = 8000,
    ssl_context=None,
    url_scheme: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Start the web server and block until interrupted.

    ``session`` is any object implementing ``upsert_device/report/prune/state``.
    ``ssl_context`` (an :class:`ssl.SSLContext`), when given, wraps the listening
    socket for HTTPS — required for microphone access on phones over the LAN.
    """
    httpd = make_server(session, host=host, port=port, debug=debug)
    scheme = url_scheme or ("https" if ssl_context is not None else "http")
    if ssl_context is not None:
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)

    bound_port = httpd.server_address[1]
    _print_startup_banner(host, bound_port, scheme)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        httpd._shutting_down = True
        httpd.shutdown()
        httpd.server_close()


def _lan_ips() -> list:
    """Best-effort list of this host's LAN IPv4 addresses (for the printed URLs)."""
    import socket

    ips = set()
    # Trick: connecting a UDP socket reveals the outbound interface IP (no traffic sent).
    for probe in ("8.8.8.8", "192.168.1.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 80))
            ips.add(s.getsockname()[0])
        except OSError:
            pass
        finally:
            s.close()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    # Drop loopback; it is printed separately.
    return sorted(ip for ip in ips if not ip.startswith("127."))


def _print_startup_banner(host: str, port: int, scheme: str) -> None:
    print(f"DroneTracking web app serving on {scheme}://{host}:{port}")
    print(f"  local:   {scheme}://127.0.0.1:{port}/")
    for ip in _lan_ips():
        print(f"  network: {scheme}://{ip}:{port}/")
    if scheme == "https":
        print("  open the network URL on each phone (accept the self-signed cert).")
    else:
        print(
            "  NOTE: phones need the https/tunnel URL for microphone access — "
            "run with --https (or front this with a tunnel) for mobile mics."
        )
