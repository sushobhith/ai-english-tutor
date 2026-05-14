import json
import urllib.request

from main import get_version_payload, start_version_endpoint


def test_get_version_payload():
    assert get_version_payload() == {'version': '0.1.0'}


def test_version_endpoint_returns_package_version():
    server = start_version_endpoint(host='127.0.0.1', port=0)
    try:
        host, port = server.server_address
        with urllib.request.urlopen(f'http://{host}:{port}/version') as response:
            assert response.status == 200
            assert response.headers['Content-Type'] == 'application/json'
            assert json.loads(response.read().decode('utf-8')) == {'version': '0.1.0'}
    finally:
        server.shutdown()
        server.server_close()
