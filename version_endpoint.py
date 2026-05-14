import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from version import get_package_version


class VersionRequestHandler(BaseHTTPRequestHandler):
    server_version = "AIEnglishTutorVersionEndpoint/1.0"

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/version":
            self.send_error(404, "Not Found")
            return

        payload = json.dumps({"version": get_package_version()}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def create_version_server(host: str = "0.0.0.0", port: Optional[int] = None) -> ThreadingHTTPServer:
    resolved_port = port if port is not None else int(os.getenv("VERSION_ENDPOINT_PORT", "8000"))
    return ThreadingHTTPServer((host, resolved_port), VersionRequestHandler)


def start_version_endpoint(host: str = "0.0.0.0", port: Optional[int] = None) -> ThreadingHTTPServer:
    server = create_version_server(host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
