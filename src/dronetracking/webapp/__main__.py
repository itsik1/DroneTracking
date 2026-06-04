"""CLI: launch the zero-install browser app.

    python -m dronetracking.webapp                       # HTTP on localhost:8000
    python -m dronetracking.webapp --host 0.0.0.0 --port 8000
    python -m dronetracking.webapp --https               # self-signed cert (for phone mics)
    python -m dronetracking.webapp --https --cert c.pem --key k.pem

Open the printed URL on every device. On a phone the microphone (and, on iOS, the
Geolocation API in many browsers) only works over **https** — use ``--https`` on the LAN
(accept the self-signed cert once) or front the server with an https tunnel.

The server holds the real adaptive :class:`dronetracking.webapp.session.Session`; tests
inject a fake one instead (see ``tests/test_webapp_server.py``).
"""

from __future__ import annotations

import argparse
import sys

from .server import serve


def _build_ssl_context(cert_file, key_file):
    """Return an :class:`ssl.SSLContext` for HTTPS.

    If ``cert_file``/``key_file`` are given, load them. Otherwise generate a throwaway
    self-signed cert (via the ``openssl`` CLI) in a temp dir so phones can use the mic
    without you having to provision a real certificate.
    """
    import ssl

    if cert_file and key_file:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
        return ctx

    cert_file, key_file = _generate_self_signed()
    print(f"Generated self-signed certificate:\n  cert: {cert_file}\n  key:  {key_file}")
    print("  (browsers will warn once — accept it to reach the page; this is expected"
          " for a LAN self-signed cert.)")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ctx


def _generate_self_signed():
    """Shell out to ``openssl`` to mint a self-signed cert/key pair in a temp dir.

    Returns ``(cert_path, key_path)``. Raises a clear error (with the manual one-liner)
    if ``openssl`` is unavailable.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("openssl") is None:
        raise SystemExit(
            "openssl not found on PATH. Generate a cert manually, then pass "
            "--cert/--key:\n"
            "  openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem "
            "-out cert.pem -days 365 -subj '/CN=dronetracking.local'"
        )

    tmp = Path(tempfile.mkdtemp(prefix="dronetracking-tls-"))
    cert_path = tmp / "cert.pem"
    key_path = tmp / "key.pem"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key_path), "-out", str(cert_path),
        "-days", "365", "-subj", "/CN=dronetracking.local",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not cert_path.is_file() or not key_path.is_file():
        raise SystemExit(
            "Failed to generate a self-signed certificate via openssl:\n"
            + (proc.stderr or proc.stdout or "(no output)")
        )
    return str(cert_path), str(key_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dronetracking.webapp",
        description="Zero-install browser app: phones join by URL, grant mic + location, "
                    "and the coordinator localizes the source by acoustic energy.",
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="interface to bind (default 0.0.0.0 = all interfaces / LAN)")
    p.add_argument("--port", type=int, default=8000, help="TCP port (default 8000)")
    p.add_argument("--https", action="store_true",
                   help="serve over HTTPS (required for phone microphone access)")
    p.add_argument("--cert", default=None,
                   help="TLS certificate file (PEM); with --https. If omitted, a "
                        "self-signed cert is generated via openssl.")
    p.add_argument("--key", default=None,
                   help="TLS private key file (PEM); pair with --cert.")
    args = p.parse_args(argv)

    if (args.cert and not args.key) or (args.key and not args.cert):
        p.error("--cert and --key must be provided together")
    if (args.cert or args.key) and not args.https:
        p.error("--cert/--key require --https")

    ssl_context = _build_ssl_context(args.cert, args.key) if args.https else None

    # Import the real Session lazily so the rest of the CLI (and --help) works even if
    # agent B's module isn't importable yet during parallel development.
    try:
        from .session import Session
    except ImportError as exc:  # pragma: no cover - depends on agent B
        print(
            "ERROR: dronetracking.webapp.session.Session is not available yet "
            f"({exc}). The session module is built in parallel; once present, "
            "`python -m dronetracking.webapp` will serve it.",
            file=sys.stderr,
        )
        return 1

    serve(Session(), host=args.host, port=args.port, ssl_context=ssl_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
