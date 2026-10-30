"""A POST that is redirected must stay a POST where the status code says so.

urllib refuses 307 and 308 after a POST -- its redirect_request accepts those
codes for GET and HEAD only -- so a service that has moved answers a form
submission with HTTP Error 307 and the caller reads it as an outage. WALTZ does
exactly this. The server here is the smallest thing that reproduces it.
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aggressor_wrappers.core import net as netmod


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the test output clean
        pass

    def do_POST(self):
        if self.path == "/moved":
            self.send_response(307)
            self.send_header("Location", "/final")
            self.end_headers()
            return
        if self.path == "/moved-permanently":
            self.send_response(308)
            self.send_header("Location", "/final")
            self.end_headers()
            return
        if self.path == "/final":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = b"POST:" + body
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_GET(self):
        if self.path == "/final":
            payload = b"GET:"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _post(url: str, opener=None):
    request = urllib.request.Request(url, data=b"seq=ACDEF", method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    if opener is None:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read()
    with opener(request, timeout=10) as response:
        return response.read()


def test_plain_urlopen_raises_on_307_after_post(server):
    """The behaviour being worked around, pinned so the fix is not mistaken
    for a workaround of something that no longer happens."""
    with pytest.raises(urllib.error.HTTPError) as caught:
        _post(f"{server}/moved")
    assert caught.value.code == 307


@pytest.mark.parametrize("path, code", [("/moved", 307), ("/moved-permanently", 308)])
def test_method_and_body_survive(server, path, code):
    body = _post(f"{server}{path}", opener=netmod.urlopen_keeping_method)
    assert body == b"POST:seq=ACDEF", (
        f"after {code} the request arrived as {body!r}; the method or the body "
        f"was dropped"
    )


def test_scheme_upgrade_is_recognised_only_when_nothing_else_changes():
    same = netmod._is_scheme_upgrade
    assert same("http://h/p?q=1", "https://h/p?q=1")
    assert not same("http://h/p", "https://h/other")      # path changed
    assert not same("http://h/p", "https://elsewhere/p")  # host changed
    assert not same("https://h/p", "https://h/p")         # not an upgrade
    assert not same("http://h/p", "http://h/p")           # not an upgrade
