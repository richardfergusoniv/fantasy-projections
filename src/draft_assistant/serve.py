"""Serve the draft assistant locally."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DRAFT_DIR = os.path.join(REPO_ROOT, "draft_assistant")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DRAFT_DIR, **kwargs)

    def do_GET(self) -> None:
        if self.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve draft assistant")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open browser")
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(DRAFT_DIR, "index.html")):
        raise SystemExit(
            f"Missing {DRAFT_DIR}/index.html — run from repo root after prepare."
        )

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Serving draft assistant at {url}")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
