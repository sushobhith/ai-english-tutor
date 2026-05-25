import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

VERSION: Final = os.getenv('AI_ENGLISH_TUTOR_VERSION', '0.2.0')
VERSION_HOST: Final = os.getenv('VERSION_HOST', '0.0.0.0')
VERSION_PORT: Final = int(os.getenv('VERSION_PORT', '8000'))


def get_version_v2_payload() -> dict[str, str]:
    return {
        'name': 'ai-english-tutor',
        'version': VERSION,
        'apiVersion': 'v2',
    }


class VersionRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/version/v2':
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(get_version_v2_payload()).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def start_version_endpoint(host: str = VERSION_HOST, port: int = VERSION_PORT):
    server = ThreadingHTTPServer((host, port), VersionRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f'Version endpoint listening on http://{host}:{port}/version/v2')
    return server
