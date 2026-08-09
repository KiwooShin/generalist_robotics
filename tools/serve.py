"""Serve the repository over HTTP with correct UTF-8 headers, optionally on the tailnet."""

import argparse
import functools
import http.server
import socket
import subprocess


class Utf8Handler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that labels text responses as UTF-8.

    Python's stock handler sends bare "text/html", which lets browsers fall back
    to a legacy encoding and mangle non-ASCII characters.
    """

    def guess_type(self, path):
        base = super().guess_type(path)
        if base.startswith("text/") and "charset=" not in base:
            return f"{base}; charset=utf-8"
        return base


def tailscale_ip():
    """Return this machine's Tailscale IPv4 address, or None when unavailable."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    address = result.stdout.strip().splitlines()
    return address[0] if address else None


def resolve_host(requested):
    """Resolve the bind address for a requested host keyword."""
    if requested != "tailscale":
        return requested
    address = tailscale_ip()
    if address is None:
        raise SystemExit("tailscale address unavailable; pass --host explicitly")
    return address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="tailscale",
        help='bind address, or "tailscale" to use this machine\'s tailnet IP (default)',
    )
    parser.add_argument("--port", type=int, default=8765, help="port to listen on")
    args = parser.parse_args()

    host = resolve_host(args.host)
    handler = functools.partial(Utf8Handler, directory=".")
    server = http.server.ThreadingHTTPServer((host, args.port), handler)
    print(f"serving {socket.gethostname()} on http://{host}:{args.port}/research_page.html")
    print("stop with Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
