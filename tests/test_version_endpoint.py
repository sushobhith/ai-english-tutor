import json
import threading
import unittest
from http.client import HTTPConnection

from version import get_package_version
from version_endpoint import create_version_server


class VersionEndpointTests(unittest.TestCase):
    def test_get_package_version_reads_source_version(self):
        self.assertEqual(get_package_version(), "0.1.0")

    def test_version_endpoint_returns_json_version(self):
        server = create_version_server(host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            host, port = server.server_address
            connection = HTTPConnection(host, port, timeout=2)

            connection.request("GET", "/version")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))

            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "application/json")
            self.assertEqual(payload, {"version": "0.1.0"})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_version_endpoint_returns_404_for_other_paths(self):
        server = create_version_server(host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            host, port = server.server_address
            connection = HTTPConnection(host, port, timeout=2)

            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()

            self.assertEqual(response.status, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
