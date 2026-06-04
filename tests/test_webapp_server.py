"""Tests for the zero-install web server (``dronetracking.webapp.server``).

These tests are self-contained: they use a tiny in-test *fake* ``Session`` that
implements the same interface the real :class:`dronetracking.webapp.session.Session`
exposes (``upsert_device``/``report``/``prune``/``state``). The server is started on an
OS-assigned port (``port=0``) in a daemon thread so the suite stays deterministic and
never leaks a listening socket.

The real wiring (orchestrator) is identical, only swapping the fake for the real
Session::

    from dronetracking.webapp.session import Session
    from dronetracking.webapp.server import serve
    serve(Session(), host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection

import pytest

from dronetracking.webapp.server import make_server


class FakeSession:
    """Minimal stand-in for the real Session, recording everything it is told.

    Implements the contract interface: ``upsert_device``, ``report``, ``prune``,
    ``state``. ``state()`` returns a snapshot whose shape matches the documented
    ``/api/events`` payload so the SSE assertions exercise the real keys.
    """

    def __init__(self) -> None:
        self.devices: dict = {}  # device_id -> name
        self.reports: list = []  # (device_id, payload)
        self.prune_calls = 0
        self.lock = threading.Lock()

    def upsert_device(self, device_id: str, name=None) -> None:
        with self.lock:
            self.devices[device_id] = name

    def report(self, device_id: str, payload: dict) -> None:
        with self.lock:
            self.reports.append((device_id, payload))

    def prune(self, max_age_s: float = 5.0) -> None:
        with self.lock:
            self.prune_calls += 1

    def state(self) -> dict:
        with self.lock:
            devices = [
                {
                    "id": did,
                    "name": name,
                    "lat": None,
                    "lon": None,
                    "has_gps": False,
                    "has_mic": False,
                    "level": 0.0,
                    "detected": False,
                    "confidence": 0.0,
                    "online": True,
                }
                for did, name in self.devices.items()
            ]
        return {
            "devices": devices,
            "source": None,
            "network": {
                "n_devices": len(devices),
                "n_gps": 0,
                "n_detecting": 0,
                "connected": bool(devices),
            },
            "computed": {"positioning": "none", "source": "none"},
            "note": "fake",
        }


@pytest.fixture()
def server():
    """A running server on an OS-assigned port with a fresh FakeSession.

    Yields ``(session, host, port)`` and guarantees shutdown afterwards.
    """
    session = FakeSession()
    httpd = make_server(session, host="127.0.0.1", port=0)
    host, port = httpd.server_address[0], httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield session, host, port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def _post_json(host, port, path, body):
    conn = HTTPConnection(host, port, timeout=5.0)
    try:
        data = json.dumps(body).encode("utf-8")
        conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read().decode("utf-8")
        return resp.status, json.loads(payload) if payload else None
    finally:
        conn.close()


def _get(host, port, path):
    conn = HTTPConnection(host, port, timeout=5.0)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        ctype = resp.getheader("Content-Type")
        return resp.status, ctype, body
    finally:
        conn.close()


def test_join_assigns_device_id_and_params(server):
    _session, host, port = server
    status, body = _post_json(host, port, "/api/join", {"name": "Roof North"})

    assert status == 200
    assert isinstance(body["device_id"], str) and body["device_id"]
    params = body["params"]
    assert params["report_interval_s"] == 0.5
    assert params["target_band_hz"] == [120, 4000]


def test_join_assigns_distinct_ids(server):
    _session, host, port = server
    ids = set()
    for _ in range(3):
        _status, body = _post_json(host, port, "/api/join", {})
        ids.add(body["device_id"])
    assert len(ids) == 3


def test_full_flow_join_report_and_sse_snapshot(server):
    """JOIN 3 devices, POST a report for each (mixing gps/audio presence), then read
    exactly ONE ``data:`` line from /api/events and validate its shape."""
    session, host, port = server

    # --- JOIN three devices ------------------------------------------------
    device_ids = []
    for name in ("alpha", "beta", "gamma"):
        _status, body = _post_json(host, port, "/api/join", {"name": name})
        device_ids.append(body["device_id"])
    assert len(set(device_ids)) == 3
    assert set(session.devices) == set(device_ids)

    # --- POST one report each, mixing capabilities -------------------------
    reports = [
        {  # full: gps + audio
            "device_id": device_ids[0],
            "t_client_ms": 1719,
            "gps": {"lat": 32.08, "lon": 34.78, "accuracy_m": 8.0},
            "audio": {"level": 0.42, "detected": True, "confidence": 0.7, "peak_hz": 180.0},
        },
        {  # audio only, gps denied
            "device_id": device_ids[1],
            "t_client_ms": 1720,
            "gps": None,
            "audio": {"level": 0.10, "detected": False, "confidence": 0.1, "peak_hz": 0.0},
        },
        {  # gps only, no mic
            "device_id": device_ids[2],
            "t_client_ms": 1721,
            "gps": {"lat": 32.081, "lon": 34.781, "accuracy_m": 12.0},
            "audio": None,
        },
    ]
    for r in reports:
        status, body = _post_json(host, port, "/api/report", r)
        assert status == 200
        assert body == {"ok": True}

    # The fake Session received every report, in order, with the bodies intact.
    assert [did for did, _ in session.reports] == device_ids
    by_id = {did: payload for did, payload in session.reports}
    assert by_id[device_ids[0]]["gps"]["lat"] == 32.08
    assert by_id[device_ids[0]]["audio"]["detected"] is True
    assert by_id[device_ids[1]]["gps"] is None
    assert by_id[device_ids[2]]["audio"] is None

    # --- Read exactly ONE SSE data: line from /api/events ------------------
    snapshot = _read_one_sse_event(host, port)

    # Valid JSON with all documented top-level keys.
    for key in ("devices", "source", "network", "computed", "note"):
        assert key in snapshot
    assert isinstance(snapshot["devices"], list)
    assert {d["id"] for d in snapshot["devices"]} == set(device_ids)
    for d in snapshot["devices"]:
        for k in ("id", "name", "has_gps", "has_mic", "level", "detected", "confidence", "online"):
            assert k in d
    net = snapshot["network"]
    assert net["n_devices"] == 3 and net["connected"] is True
    assert set(snapshot["computed"]) == {"positioning", "source"}

    # The SSE loop must have pruned at least once before emitting.
    assert session.prune_calls >= 1


def _read_one_sse_event(host, port, timeout=10.0):
    """Open GET /api/events and return the JSON of the first ``data:`` line.

    Reads the raw socket so we can stop after a single event without waiting for
    the (infinite) stream to end.
    """
    conn = HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", "/api/events", headers={"Accept": "text/event-stream"})
    resp = conn.getresponse()
    assert resp.status == 200
    assert "text/event-stream" in (resp.getheader("Content-Type") or "")

    deadline = time.time() + timeout
    buf = b""
    try:
        while time.time() < deadline:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if b"\n\n" in buf:
                block = buf.split(b"\n\n", 1)[0]
                for line in block.split(b"\n"):
                    if line.startswith(b"data:"):
                        return json.loads(line[len(b"data:"):].strip().decode("utf-8"))
                buf = b""  # not a data block (e.g. a comment); keep reading
        raise AssertionError("no SSE data: event received before timeout")
    finally:
        conn.close()


def test_static_index_served(server):
    """``GET /`` serves the SPA index.html (the frontend agent's file)."""
    _session, host, port = server
    status, ctype, body = _get(host, port, "/")
    # index.html already exists in webapp/static/, so this should be 200 + HTML.
    assert status == 200
    assert "text/html" in ctype
    assert b"<!DOCTYPE html>" in body or b"<!doctype html>" in body


def test_missing_static_asset_is_404(server):
    """Assets that don't exist yet (frontend written in parallel) -> clean 404."""
    _session, host, port = server
    status, _ctype, _body = _get(host, port, "/does-not-exist.js")
    assert status == 404


def test_unknown_api_route_is_404(server):
    _session, host, port = server
    status, _body = _post_json(host, port, "/api/nope", {})
    assert status == 404


def test_app_js_content_type_when_present(server, tmp_path):
    """If app.js exists it is served as JavaScript; otherwise 404 (tolerated)."""
    _session, host, port = server
    status, ctype, _body = _get(host, port, "/app.js")
    if status == 200:
        assert "javascript" in ctype
    else:
        assert status == 404
