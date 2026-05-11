"""Small HTTP endpoint exposing the project version."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from typing import Optional

DEFAULT_VERSION = "0.1.0"
PACKAGE_NAME = "ai-english-tutor"


def get_package_version() -> str:
    """Return the installed package version, or the source fallback version."""
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return DEFAULT_VERSION


class VersionRequestHandler(BaseHTTPRequestHandler):
    """Serve JSON for /version and 404 for all other paths."""

    server_version = "AIEnglishTutorVersionEndpoint/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0] != "/version":
            self.send_error(404, "Not Found")
            return

        payload = json.dumps({"version": get_package_version()}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the bot logs focused on bot activity."""
        return


def create_version_server(host: str = "0.0.0.0", port: Optional[int] = None) -> ThreadingHTTPServer:
    resolved_port = port if port is not None else int(os.getenv("PORT", "8000"))
    return ThreadingHTTPServer((host, resolved_port), VersionRequestHandler)


def start_version_endpoint(host: str = "0.0.0.0", port: Optional[int] = None) -> ThreadingHTTPServer:
    """Start the /version endpoint in a daemon thread and return the server."""
    server = create_version_server(host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
