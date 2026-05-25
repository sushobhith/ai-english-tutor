import json
import unittest
import urllib.error
import urllib.request

from version_server import start_version_server


class VersionServerTests(unittest.TestCase):
    def setUp(self):
        self.server, self.thread = start_version_server("127.0.0.1", 0)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_version_v2_returns_payload(self):
        with urllib.request.urlopen(f"{self.base_url}/version/v2", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(payload["name"], "ai-english-tutor")
        self.assertEqual(payload["version"], "v2")
        self.assertEqual(payload["status"], "ok")

    def test_unknown_path_returns_404(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{self.base_url}/version", timeout=5)

        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
