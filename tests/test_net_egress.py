"""Contract tests for the SSRF egress guard (vfairness.net.egress).

The guard shipped without tests, and two defects rode along with it:

  1. ``PinnedIPAdapter`` put ``server_hostname``/``assert_hostname`` into the
     connection-pool kwargs unconditionally. urllib3 does not strip those for a
     plain-HTTP pool, so EVERY ``http://`` request raised
     ``TypeError: HTTPConnection.__init__() got an unexpected keyword argument
     'assert_hostname'`` -- i.e. the documented on-prem/self-hosted endpoint
     path was dead on arrival.
  2. Loopback was refused with no way to opt in, so a local model server
     (Ollama / vLLM) and every local test stub was unreachable.

The refusal matrix below is the security contract: whatever a caller asks for,
cloud-metadata, RFC1918, link-local and unique-local targets stay refused. Only
loopback is unlockable, and only by an explicit opt-in.
"""

import http.server
import json
import threading

import pytest

from vfairness.llm import LLMApiProxy
from vfairness.net.egress import PinnedIPAdapter, SSRFError, validate_endpoint

# Targets that must NEVER be reachable, even with every opt-in turned on.
ALWAYS_REFUSED = [
    "http://169.254.169.254/latest/meta-data/",  # AWS/GCP/Azure IMDS
    "http://[fd00:ec2::254]/",  # IPv6 IMDS
    "http://[::ffff:169.254.169.254]/",  # IPv4-mapped IMDS
    "http://10.0.0.5/v1",  # RFC1918
    "http://192.168.1.10/v1",  # RFC1918
    "http://172.16.0.9/v1",  # RFC1918
    "http://[fc00::1]/v1",  # unique-local
    "http://169.254.10.10/v1",  # link-local
    "file:///etc/passwd",  # non-http scheme
    "ftp://example.com/",  # non-http scheme
    "https://",  # no host
]

LOOPBACK = [
    "http://127.0.0.1:8080/v1",
    "http://localhost:8080/v1",
    "http://[::1]:8080/v1",
    "http://[::ffff:127.0.0.1]:8080/v1",
    "http://127.0.0.53:8080/v1",  # all of 127.0.0.0/8, not just .1
]


@pytest.mark.parametrize("url", ALWAYS_REFUSED)
def test_forbidden_targets_refused_even_with_every_opt_in(url):
    """The opt-ins widen the guard to loopback and cleartext ONLY. They must not
    become a general bypass for metadata / private / link-local targets."""
    with pytest.raises(SSRFError):
        validate_endpoint(url, allow_http=True, allow_loopback=True)


@pytest.mark.parametrize("url", LOOPBACK)
def test_loopback_refused_by_default(url):
    """Default posture is unchanged by the opt-in existing: loopback is refused
    unless a caller deliberately asks for it."""
    with pytest.raises(SSRFError):
        validate_endpoint(url, allow_http=True)


@pytest.mark.parametrize("url", LOOPBACK)
def test_loopback_allowed_only_on_explicit_opt_in(url):
    host, port, ips = validate_endpoint(url, allow_http=True, allow_loopback=True)
    assert host and port and ips


def test_allow_loopback_does_not_imply_allow_http():
    """The two opt-ins are independent; cleartext still needs allow_http."""
    with pytest.raises(SSRFError):
        validate_endpoint("http://127.0.0.1:8080/v1", allow_loopback=True)


def test_pinned_adapter_sets_tls_kwargs_only_for_https():
    """server_hostname/assert_hostname are HTTPSConnection-only. Leaking them
    into a plain-HTTP pool is what broke every http:// endpoint."""
    https = PinnedIPAdapter("api.example.com", "93.184.216.34", tls=True)
    assert https.poolmanager.connection_pool_kw["server_hostname"] == "api.example.com"
    assert https.poolmanager.connection_pool_kw["assert_hostname"] == "api.example.com"

    plain = PinnedIPAdapter("api.example.com", "93.184.216.34", tls=False)
    assert "server_hostname" not in plain.poolmanager.connection_pool_kw
    assert "assert_hostname" not in plain.poolmanager.connection_pool_kw


def _serve():
    """A local OpenAI-shaped stub that echoes a fixed completion."""

    class _Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_plain_http_endpoint_actually_completes_a_request():
    """Regression: the guarded session must work over cleartext HTTP, not raise
    TypeError from TLS-only pool kwargs. This is the self-hosted endpoint path."""
    srv = _serve()
    try:
        url = "http://127.0.0.1:{0}/v1/chat/completions".format(srv.server_address[1])
        proxy = LLMApiProxy(
            endpoint_url=url, api_format="openai", model_name="stub", allow_loopback=True
        )
        out = proxy.send_prompt("ping")
        assert out["text"] == "pong"
    finally:
        srv.shutdown()


def test_proxy_refuses_loopback_without_opt_in():
    """LLMApiProxy keeps the guard on by default: a user-supplied loopback URL is
    still refused at construction time."""
    srv = _serve()
    try:
        url = "http://127.0.0.1:{0}/v1/chat/completions".format(srv.server_address[1])
        with pytest.raises(ValueError, match="egress guard"):
            LLMApiProxy(endpoint_url=url, api_format="openai", model_name="stub")
    finally:
        srv.shutdown()


def test_proxy_refuses_cloud_metadata_even_with_loopback_opt_in():
    with pytest.raises(ValueError, match="egress guard"):
        LLMApiProxy(
            endpoint_url="http://169.254.169.254/latest/meta-data/",
            api_format="openai",
            model_name="stub",
            allow_loopback=True,
        )
