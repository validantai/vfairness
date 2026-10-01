"""Wave-6 audit pins for the SSRF egress guard (vfairness.net.egress).

The CRITICAL finding: the guard validated only the ORIGINAL endpoint URL. requests
follows 3xx redirects by default and PinnedIPAdapter only pins/checks its one vetted
host, so a redirect Location pointing at a DIFFERENT host fell straight through to the
network with no _ip_is_forbidden() check. A single 302 from the (validated) endpoint to
169.254.169.254 / any RFC1918 / loopback host therefore defeated the entire guard and
reached cloud metadata / internal services.

The fix routes every hop through the guard: GuardedSession.get_adapter() re-runs
validate_endpoint() on the original request AND on each redirect target (requests calls
get_adapter via resolve_redirects -> send), refusing any hop whose scheme/host/resolved
IPs fail. This module pins that behaviour.

No external network: the tests spin up loopback http.server stubs, mock the resolver so
the guard SEES the public/private/metadata IPs under test, and redirect the actual TCP
dial to loopback via urllib3's create_connection. The negative battery comes first (the
exact scenario that used to leak), then the does-not-overcorrect cases (a plain public
URL, a normal request, and a redirect to another PUBLIC host must all still work).
"""

import http.server
import json
import socket
import threading

import pytest
import urllib3.util.connection as u3conn

from vfairness.llm import LLMApiProxy
from vfairness.net.egress import SSRFError, guarded_post, validate_endpoint

# Stand-in IPs the guard is made to "resolve" hostnames to.
_PUBLIC_IP = "93.184.216.34"
_METADATA_IP = "169.254.169.254"
_RFC1918_IP = "10.0.0.5"
# 169.254.169.254 written as a bare decimal integer, the classic guard-bypass encoding
# an OS resolver would decode back to the metadata address.
_METADATA_DECIMAL = "2852039166"


class _Handler(http.server.BaseHTTPRequestHandler):
    """A stub whose response is whatever callable the server carries in .action."""

    def do_POST(self):  # noqa: N802 (http.server naming)
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self._dispatch()

    def do_GET(self):  # noqa: N802 (redirects turn POST into GET)
        self._dispatch()

    def _dispatch(self):
        self.server.hits.append(self.path)
        self.server.action(self)

    def log_message(self, *args):  # silence the stub
        pass


def _completion(content="pong"):
    """An OpenAI-shaped 200 body."""

    def _act(h):
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode()
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    return _act


def _redirect(location, status=302):
    def _act(h):
        h.send_response(status)
        h.send_header("Location", location)
        h.end_headers()

    return _act


class _Sandbox:
    """Owns the loopback stub servers plus the resolver / dial patches for one test."""

    def __init__(self, monkeypatch):
        self._servers = []
        self.resolve = {}  # hostname (as the guard sees it) -> ip or list[ip]
        self.resolve_calls = []
        real_gai = socket.getaddrinfo
        real_cc = u3conn.create_connection

        def fake_gai(host, port, *a, **k):
            self.resolve_calls.append(host)
            if host in self.resolve:
                mapped = self.resolve[host]
                ips = [mapped] if isinstance(mapped, str) else list(mapped)
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
                    for ip in ips
                ]
            return real_gai(host, port, *a, **k)

        def fake_cc(address, *a, **k):
            # Every dial is redirected to loopback, preserving the port so each stub
            # server is still reached by its own ephemeral port.
            return real_cc(("127.0.0.1", address[1]), *a, **k)

        monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
        monkeypatch.setattr(u3conn, "create_connection", fake_cc)

    def server(self, action):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        srv.action = action
        srv.hits = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self._servers.append(srv)
        return srv

    def url(self, host, srv, path="/v1/chat/completions"):
        """A URL whose HOST the guard resolves via self.resolve, on srv's real port."""
        return f"http://{host}:{srv.server_address[1]}{path}"

    def shutdown(self):
        for srv in self._servers:
            srv.shutdown()


@pytest.fixture
def sandbox(monkeypatch):
    sb = _Sandbox(monkeypatch)
    try:
        yield sb
    finally:
        sb.shutdown()


# --------------------------------------------------------------------------------------
# NEGATIVE BATTERY (the bug) -- these are the scenarios that used to leak.
# --------------------------------------------------------------------------------------


def test_redirect_from_public_to_metadata_is_refused(sandbox):
    """THE bug: a 302 from the validated public endpoint to the cloud-metadata IP was
    followed with no re-validation. It must now raise SSRFError and never dial the
    metadata host."""
    internal = sandbox.server(_completion("SECRET_FROM_METADATA"))
    entry = sandbox.server(
        _redirect(sandbox.url("metadata.internal", internal, "/latest/meta-data/"))
    )
    sandbox.resolve["trusted.public"] = _PUBLIC_IP
    sandbox.resolve["metadata.internal"] = _METADATA_IP

    proxy = LLMApiProxy(
        endpoint_url=sandbox.url("trusted.public", entry), api_format="openai", model_name="x"
    )
    with pytest.raises(SSRFError):
        proxy.send_prompt("hello")
    assert internal.hits == [], "metadata host must never be contacted"
    assert "metadata.internal" in sandbox.resolve_calls, "the guard must resolve the redirect hop"


def test_redirect_from_public_to_rfc1918_is_refused(sandbox):
    """Same defect for a private (RFC1918) redirect target."""
    internal = sandbox.server(_completion("SECRET_FROM_LAN"))
    entry = sandbox.server(_redirect(sandbox.url("lan.internal", internal, "/admin")))
    sandbox.resolve["trusted.public"] = _PUBLIC_IP
    sandbox.resolve["lan.internal"] = _RFC1918_IP

    proxy = LLMApiProxy(
        endpoint_url=sandbox.url("trusted.public", entry), api_format="openai", model_name="x"
    )
    with pytest.raises(SSRFError):
        proxy.send_prompt("hello")
    assert internal.hits == []


def test_redirect_chain_is_followed_through_public_hops_then_refused_at_private(sandbox):
    """A multi-hop chain: public -> public -> metadata. The intermediate PUBLIC hop is
    followed (proving the chain is honoured), the final private hop is refused, and the
    guard resolves EVERY host (DNS-rebinding shape: re-resolution happens per hop)."""
    metadata = sandbox.server(_completion("SECRET"))
    hop2 = sandbox.server(
        _redirect(sandbox.url("metadata.internal", metadata, "/latest/meta-data/"))
    )
    entry = sandbox.server(_redirect(sandbox.url("hop2.public", hop2, "/next")))
    sandbox.resolve["trusted.public"] = _PUBLIC_IP
    sandbox.resolve["hop2.public"] = _PUBLIC_IP
    sandbox.resolve["metadata.internal"] = _METADATA_IP

    proxy = LLMApiProxy(
        endpoint_url=sandbox.url("trusted.public", entry), api_format="openai", model_name="x"
    )
    with pytest.raises(SSRFError):
        proxy.send_prompt("hello")
    assert hop2.hits, "the intermediate public hop should be followed"
    assert metadata.hits == [], "the final private hop must be refused"
    # Every distinct hop host was run through the guard's resolver.
    for host in ("trusted.public", "hop2.public", "metadata.internal"):
        assert host in sandbox.resolve_calls


def test_direct_private_and_metadata_ips_are_refused():
    """Direct (non-redirect) private/metadata targets stay refused, at both the
    validate_endpoint layer and LLMApiProxy construction."""
    for bad in ("http://10.0.0.5/v1", "http://169.254.169.254/latest/meta-data/", "http://[::1]/"):
        with pytest.raises(SSRFError):
            validate_endpoint(bad, allow_http=True)
    for bad in ("http://10.0.0.5/v1", "http://169.254.169.254/latest/meta-data/"):
        with pytest.raises(ValueError, match="egress guard"):
            LLMApiProxy(endpoint_url=bad, api_format="openai", model_name="x")


def test_decimal_encoded_metadata_ip_is_refused(sandbox):
    """A decimal-integer host that the resolver decodes back to the metadata IP must be
    refused, both directly and as a redirect target."""
    sandbox.resolve[_METADATA_DECIMAL] = _METADATA_IP
    with pytest.raises(SSRFError):
        validate_endpoint(f"http://{_METADATA_DECIMAL}/latest/meta-data/", allow_http=True)

    internal = sandbox.server(_completion("SECRET"))
    entry = sandbox.server(
        _redirect(f"http://{_METADATA_DECIMAL}:{internal.server_address[1]}/latest/meta-data/")
    )
    sandbox.resolve["trusted.public"] = _PUBLIC_IP
    proxy = LLMApiProxy(
        endpoint_url=sandbox.url("trusted.public", entry), api_format="openai", model_name="x"
    )
    with pytest.raises(SSRFError):
        proxy.send_prompt("hello")
    assert internal.hits == []


def test_public_hostname_resolving_to_private_ip_is_refused(sandbox):
    """The guard's primary real-world vector: a public-looking hostname whose A record
    is a private IP. (Closes the audit's untested-hostname-resolution gap.)"""
    sandbox.resolve["evil.example.test"] = _RFC1918_IP
    with pytest.raises(SSRFError, match="non-public"):
        validate_endpoint("http://evil.example.test/v1", allow_http=True)


def test_mixed_public_and_metadata_records_are_refused(sandbox):
    """DNS rebinding: a host that resolves to BOTH a public and a metadata address must
    be refused on the bad address (all-or-nothing), not partially used."""
    sandbox.resolve["rebind.example.test"] = [_PUBLIC_IP, _METADATA_IP]
    with pytest.raises(SSRFError, match="non-public"):
        validate_endpoint("http://rebind.example.test/v1", allow_http=True)


def test_guarded_post_refuses_redirect_to_metadata(sandbox):
    """The standalone guarded_post helper shares the guard and must refuse the same
    redirect bypass, not just LLMApiProxy."""
    internal = sandbox.server(_completion("SECRET"))
    entry = sandbox.server(
        _redirect(sandbox.url("metadata.internal", internal, "/latest/meta-data/"))
    )
    sandbox.resolve["trusted.public"] = _PUBLIC_IP
    sandbox.resolve["metadata.internal"] = _METADATA_IP

    with pytest.raises(SSRFError):
        guarded_post(
            sandbox.url("trusted.public", entry), allow_http=True, json={"x": 1}, timeout=5
        )
    assert internal.hits == []


# --------------------------------------------------------------------------------------
# DOES-NOT-OVERCORRECT -- a public URL, a normal request, and a public redirect must work.
# --------------------------------------------------------------------------------------


def test_plain_public_https_url_is_permitted(sandbox):
    """The guard must still allow an ordinary public https endpoint (mocked resolver)."""
    sandbox.resolve["api.openai.com"] = _PUBLIC_IP
    host, port, ips = validate_endpoint("https://api.openai.com/v1/chat/completions")
    assert host == "api.openai.com"
    assert port == 443
    assert ips == [_PUBLIC_IP]
    # Construction against a public endpoint must not raise.
    LLMApiProxy(
        endpoint_url="https://api.openai.com/v1/chat/completions",
        api_format="openai",
        model_name="gpt-4",
    )


def test_normal_request_without_redirect_completes(sandbox):
    """A public endpoint that answers directly (no redirect) still returns its body."""
    endpoint = sandbox.server(_completion("pong"))
    sandbox.resolve["trusted.public"] = _PUBLIC_IP
    proxy = LLMApiProxy(
        endpoint_url=sandbox.url("trusted.public", endpoint), api_format="openai", model_name="x"
    )
    out = proxy.send_prompt("ping")
    assert out["text"] == "pong"
    assert endpoint.hits


def test_redirect_to_another_public_host_is_followed(sandbox):
    """A 302 to a DIFFERENT public host is legitimate and must still be followed to
    completion; the fix guards redirects, it does not forbid them."""
    final = sandbox.server(_completion("pong"))
    entry = sandbox.server(_redirect(sandbox.url("api2.public", final)))
    sandbox.resolve["api1.public"] = _PUBLIC_IP
    sandbox.resolve["api2.public"] = _PUBLIC_IP

    proxy = LLMApiProxy(
        endpoint_url=sandbox.url("api1.public", entry), api_format="openai", model_name="x"
    )
    out = proxy.send_prompt("ping")
    assert out["text"] == "pong"
    assert final.hits, "the public redirect target should be reached"
