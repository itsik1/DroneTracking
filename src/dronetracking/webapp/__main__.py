"""CLI: launch the zero-install browser app.

    python -m dronetracking.webapp                 # HTTP on the LAN (open on the PC)
    python -m dronetracking.webapp --tunnel        # auto-start a Cloudflare https tunnel + show a QR
    python -m dronetracking.webapp --https          # self-signed cert (LAN, for phone mics)

Open the printed URL — or scan the terminal QR — on every device. Phones need an **https**
link for the microphone, so use ``--tunnel`` (easiest) or ``--https`` on the LAN.
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import threading
import time

from .server import make_server, serve


# --------------------------------------------------------------------------- #
# pretty output: prominent URL banner + scannable terminal QR
# --------------------------------------------------------------------------- #
def _print_qr(url: str) -> None:
    try:
        import qrcode  # optional dependency
    except Exception:
        print("  (tip: `pip install qrcode` to print a scannable QR here)\n")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def _announce(url: str, qr: bool = True, note: str | None = None) -> None:
    line = "═" * (len(url) + 4)
    print("\n" + line)
    print(f"  {url}")
    print(line)
    print("  Open this on every device (PC + phones). " + (note or
          "Phones need an https link for the mic — use --tunnel or --https."))
    print("  Tap “🔗 Share” in the page for an on-screen QR too.\n")
    if qr:
        _print_qr(url)
        print()


def _lan_ip() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _primary_url(host: str, port: int, https: bool) -> str:
    scheme = "https" if https else "http"
    h = host
    if host in ("0.0.0.0", "::", ""):
        h = _lan_ip() or "localhost"
    return f"{scheme}://{h}:{port}"


# --------------------------------------------------------------------------- #
# cloudflare quick tunnel
# --------------------------------------------------------------------------- #
def _run_tunnel(port: int):
    """Start `cloudflared` quick tunnel to localhost:port; return (proc, https_url|None)."""
    import shutil
    import subprocess

    if shutil.which("cloudflared") is None:
        print("cloudflared not found. Install it once:  brew install cloudflared\n"
              "  (or run a tunnel yourself:  cloudflared tunnel --url http://localhost:%d)" % port)
        return None, None

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    url = None
    deadline = time.time() + 30
    print("starting Cloudflare tunnel…")
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        m = pat.search(line)
        if m:
            url = m.group(0)
            break
    # Keep draining cloudflared output so its pipe never blocks.
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    return proc, url


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="dronetracking.webapp",
        description="Zero-install browser app: devices join by URL, grant mic + location; "
                    "the coordinator does what the connected devices allow.",
    )
    p.add_argument("--host", default="0.0.0.0", help="bind interface (default 0.0.0.0 = LAN)")
    p.add_argument("--port", type=int, default=8000, help="TCP port (default 8000)")
    p.add_argument("--tunnel", action="store_true",
                   help="auto-start a Cloudflare https tunnel and show its link + QR (easiest for phones)")
    p.add_argument("--https", action="store_true",
                   help="serve HTTPS with a self-signed cert (LAN alternative for phone mics)")
    p.add_argument("--cert", default=None, help="TLS certificate (PEM); with --https")
    p.add_argument("--key", default=None, help="TLS private key (PEM); pair with --cert")
    p.add_argument("--no-qr", action="store_true", help="don't print the terminal QR code")
    p.add_argument("--debug", action="store_true",
                   help="log each device's join/report (incl. ranging timestamps) to the terminal")
    args = p.parse_args(argv)

    if (args.cert and not args.key) or (args.key and not args.cert):
        p.error("--cert and --key must be provided together")
    if (args.cert or args.key) and not args.https:
        p.error("--cert/--key require --https")
    if args.tunnel and args.https:
        p.error("--tunnel already provides https; don't combine it with --https")

    try:
        from .session import Session
    except ImportError as exc:  # pragma: no cover
        print(f"ERROR: webapp.session.Session unavailable ({exc})", file=sys.stderr)
        return 1
    session = Session()

    # --- tunnel mode: run a local HTTP server in a thread, expose it via cloudflared ---
    if args.tunnel:
        httpd = make_server(session, "127.0.0.1", args.port, debug=args.debug)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"local server on http://127.0.0.1:{args.port}")
        proc, url = _run_tunnel(args.port)
        if url:
            _announce(url, qr=not args.no_qr,
                      note="Scan the QR with your phone — it's a public https link, mic works.")
        else:
            print("\nTunnel link not detected. The local server is still up at "
                  f"http://localhost:{args.port} ; start a tunnel manually if needed.\n")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nstopping…")
        finally:
            httpd.shutdown()
            if proc:
                proc.terminate()
        return 0

    # --- direct mode (http on LAN, or --https) ---
    ssl_context = _build_ssl_context(args.cert, args.key) if args.https else None
    _announce(_primary_url(args.host, args.port, args.https), qr=not args.no_qr)
    serve(session, host=args.host, port=args.port, ssl_context=ssl_context, debug=args.debug)
    return 0


# --------------------------------------------------------------------------- #
# self-signed TLS (unchanged)
# --------------------------------------------------------------------------- #
def _build_ssl_context(cert_file, key_file):
    import ssl

    if cert_file and key_file:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
        return ctx
    cert_file, key_file = _generate_self_signed()
    print(f"Generated self-signed certificate:\n  cert: {cert_file}\n  key:  {key_file}")
    print("  (browsers warn once for a LAN self-signed cert — accept it to reach the page.)")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ctx


def _generate_self_signed():
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if shutil.which("openssl") is None:
        raise SystemExit(
            "openssl not found. Generate a cert manually, then pass --cert/--key:\n"
            "  openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem "
            "-out cert.pem -days 365 -subj '/CN=dronetracking.local'"
        )
    tmp = Path(tempfile.mkdtemp(prefix="dronetracking-tls-"))
    cert_path, key_path = tmp / "cert.pem", tmp / "key.pem"
    cmd = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
           "-keyout", str(key_path), "-out", str(cert_path),
           "-days", "365", "-subj", "/CN=dronetracking.local"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not cert_path.is_file():
        raise SystemExit("Failed to generate a self-signed certificate:\n"
                         + (proc.stderr or proc.stdout or "(no output)"))
    return str(cert_path), str(key_path)


if __name__ == "__main__":
    raise SystemExit(main())
