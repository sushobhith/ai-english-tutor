import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Dict, Tuple


APP_NAME = "ai-english-tutor"
API_VERSION = "v2"
VERSION_PATH = "/version/v2"


def build_version_payload() -> Dict[str, str]:
    return {
        "name": APP_NAME,
        "version": API_VERSION,
        "status": "ok",
        "environment": os.getenv("APP_ENV", "development"),
    }


class VersionRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != VERSION_PATH:
            self.send_error(404, "Not Found")
            return

        body = json.dumps(build_version_payload()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_version_server(host: str = "0.0.0.0", port: int = 8080) -> Tuple[HTTPServer, Thread]:
    server = HTTPServer((host, port), VersionRequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
