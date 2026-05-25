import json
import unittest
import urllib.error
import urllib.request

from version_endpoint import get_version_v2_payload, start_version_endpoint


class VersionEndpointTest(unittest.TestCase):
    def test_get_version_v2_payload(self):
        self.assertEqual(
            get_version_v2_payload(),
            {
                'name': 'ai-english-tutor',
                'version': '0.2.0',
                'apiVersion': 'v2',
            },
        )

    def test_version_v2_endpoint_returns_payload(self):
        server = start_version_endpoint(host='127.0.0.1', port=0)
        try:
            host, port = server.server_address[:2]
            with urllib.request.urlopen(f'http://{host}:{port}/version/v2') as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers['Content-Type'], 'application/json')
                self.assertEqual(json.loads(response.read().decode('utf-8')), get_version_v2_payload())
        finally:
            server.shutdown()
            server.server_close()

    def test_unknown_endpoint_returns_404(self):
        server = start_version_endpoint(host='127.0.0.1', port=0)
        try:
            host, port = server.server_address[:2]
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(f'http://{host}:{port}/version')
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
