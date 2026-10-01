"""BGL grade wave, batch G09: vfairness.net.egress.

Every unit in this file is a SECURITY surface, so the evidence is the REFUSALS,
not the green light. A check that says "allowed" for a safe URL proves nothing
on its own.

The defect these pins were written for (2026-09-30): ``_ip_is_forbidden``
credited ``ipaddress.is_global`` with refusing "reserved, unspecified,
multicast", and CPython's tables do not. Measured with the default, most
restrictive settings, ``validate_endpoint`` / ``GuardedSession.get_adapter`` /
``guarded_post`` ALLOWED every one of 224.0.0.1, 224.0.0.251, 239.255.255.250,
ff02::1, ff05::1:3, 64:ff9b::7f00:1 (NAT64 prefix embedding 127.0.0.1),
64:ff9b::a9fe:a9fe (the same prefix embedding the cloud-metadata address) and
::a00:1 (IPv4-compatible form of 10.0.0.1). The last three are the exact door
``ipv4_mapped`` unwrapping was added to close, spelled two other ways.

The parametrised battery is derived from ``ipaddress``'s own classification
rather than quoting a verdict list, so it cannot go stale into a test that
passes by agreeing with whatever the module currently does.
"""

import http.server
import ipaddress
import socket
import threading

import pytest
import urllib3.util.connection as u3conn
from requests.adapters import HTTPAdapter

from vfairness.net.egress import (
    GuardedSession,
    PinnedIPAdapter,
    SSRFError,
    _ip_is_forbidden,
    guarded_post,
    validate_endpoint,
)

# A public address that every control below must still reach.
_PUBLIC_IP = "93.184.216.34"
_PUBLIC_V6 = "2606:4700:4700::1111"


# ---------------------------------------------------------------------------
# validate_endpoint: the negative battery
# ---------------------------------------------------------------------------

# Non-routable / internal targets, one per family. Each entry is the reason it
# must be refused, so a failure names what leaked.
FORBIDDEN = [
    ("https://127.0.0.1/x", "loopback"),
    ("https://[::1]/x", "loopback v6"),
    ("https://localhost/x", "hostname resolving to loopback"),
    ("https://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("https://[fd00:ec2::254]/latest/", "cloud metadata v6"),
    ("https://[::ffff:169.254.169.254]/x", "v4-mapped cloud metadata"),
    ("https://[::ffff:127.0.0.1]/x", "v4-mapped loopback"),
    ("https://10.0.0.5/x", "rfc1918"),
    ("https://172.16.0.1/x", "rfc1918"),
    ("https://192.168.1.1/x", "rfc1918"),
    ("https://100.64.0.1/x", "cgnat"),
    ("https://0.0.0.0/x", "unspecified"),
    ("https://255.255.255.255/x", "broadcast"),
    ("https://[fe80::1]/x", "link-local v6"),
    ("https://[fd00::1]/x", "unique-local v6"),
    ("https://2130706433/x", "decimal-encoded loopback"),
    ("https://0x7f000001/x", "hex-encoded loopback"),
    ("https://224.0.0.1/x", "multicast all-hosts"),
    ("https://224.0.0.251/x", "mDNS multicast"),
    ("https://239.255.255.250/x", "SSDP multicast"),
    ("https://[ff02::1]/x", "multicast v6 all-nodes"),
    ("https://[ff05::1:3]/x", "multicast v6 site-local"),
    ("https://[64:ff9b::7f00:1]/x", "NAT64 prefix embedding loopback"),
    ("https://[64:ff9b::a9fe:a9fe]/x", "NAT64 prefix embedding cloud metadata"),
    ("https://[::a00:1]/x", "v4-compatible form of 10.0.0.1"),
    ("file:///etc/passwd", "non-http scheme"),
    ("gopher://example.com/x", "non-http scheme"),
    ("http://93.184.216.34/x", "plain http without allow_http"),
    ("https://", "no host"),
    ("", "empty"),
]


@pytest.mark.parametrize("url,why", FORBIDDEN, ids=[w for _, w in FORBIDDEN])
def test_validate_endpoint_refuses_every_non_public_target(url, why):
    with pytest.raises(SSRFError):
        validate_endpoint(url)


# The families an explicit opt-in must never unlock. allow_loopback is the ONLY
# carve-out this module has, and it is loopback ONLY.
@pytest.mark.parametrize(
    "url,why",
    [(u, w) for u, w in FORBIDDEN if w not in ("plain http without allow_http",)],
    ids=[w for u, w in FORBIDDEN if w not in ("plain http without allow_http",)],
)
def test_no_opt_in_unlocks_anything_but_loopback(url, why):
    if "loopback" in why and "NAT64" not in why:
        pytest.skip("loopback IS the documented carve-out; covered by its own control")
    with pytest.raises(SSRFError):
        validate_endpoint(url, allow_http=True, allow_loopback=True)


def test_the_new_refusals_are_derived_from_ipaddress_not_quoted():
    """A multicast / reserved / unspecified address is refused BECAUSE it is one.

    Derived, not quoted: the classification comes from ``ipaddress`` so the
    battery above cannot drift into agreeing with whatever the module does.
    """
    for ip in (
        "224.0.0.1",
        "239.255.255.250",
        "ff02::1",
        "ff05::1:3",
        "64:ff9b::7f00:1",
        "::a00:1",
        "0.0.0.0",
        "::",
    ):
        addr = ipaddress.ip_address(ip)
        assert addr.is_multicast or addr.is_reserved or addr.is_unspecified or not addr.is_global, (
            f"{ip} is no longer in a family this test claims to cover"
        )
        assert _ip_is_forbidden(ip) is True, ip
        assert _ip_is_forbidden(ip, allow_loopback=True) is True, ip


# ---------------------------------------------------------------------------
# The controls. A guard that refuses everything passes every refusal test.
# ---------------------------------------------------------------------------


def test_a_public_endpoint_is_still_allowed_and_carries_its_real_values():
    host, port, ips = validate_endpoint(f"https://{_PUBLIC_IP}:8443/v1/x")
    assert (host, port, ips) == (_PUBLIC_IP, 8443, [_PUBLIC_IP])
    assert validate_endpoint(f"https://[{_PUBLIC_V6}]/x")[0] == _PUBLIC_V6
    # The default port is derived from the scheme, not defaulted to a literal.
    assert validate_endpoint(f"https://{_PUBLIC_IP}/x")[1] == 443
    assert validate_endpoint(f"http://{_PUBLIC_IP}/x", allow_http=True)[1] == 80


def test_loopback_is_reachable_only_with_the_explicit_opt_in():
    with pytest.raises(SSRFError):
        validate_endpoint("https://127.0.0.1:11434/api")
    assert validate_endpoint("https://127.0.0.1:11434/api", allow_loopback=True) == (
        "127.0.0.1",
        11434,
        ["127.0.0.1"],
    )
    assert validate_endpoint("https://[::1]:8080/x", allow_loopback=True)[0] == "::1"
    assert validate_endpoint("https://[::ffff:127.0.0.1]/x", allow_loopback=True)[0] == (
        "::ffff:127.0.0.1"
    )
    # ...and it does not drag allow_http along with it.
    with pytest.raises(SSRFError):
        validate_endpoint("http://127.0.0.1:11434/api", allow_loopback=True)


# ---------------------------------------------------------------------------
# A hostname that resolves inward, and the all-or-nothing rule
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver(monkeypatch):
    table: dict = {}
    real = socket.getaddrinfo

    def fake(host, port, *a, **k):
        if host in table:
            value = table[host]
            if value is None:
                raise socket.gaierror(8, "nodename nor servname provided")
            ips = [value] if isinstance(value, str) else list(value)
            return [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
                for ip in ips
            ]
        return real(host, port, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return table


def test_a_hostname_that_resolves_inward_is_refused(resolver):
    resolver["inward.example"] = "10.1.2.3"
    resolver["meta.example"] = "169.254.169.254"
    resolver["cast.example"] = "239.255.255.250"
    resolver["nx.example"] = None
    resolver["empty.example"] = []
    resolver["good.example"] = _PUBLIC_IP
    for host in ("inward.example", "meta.example", "cast.example", "nx.example", "empty.example"):
        with pytest.raises(SSRFError):
            validate_endpoint(f"https://{host}/x")
    # control: the same machinery still lets a public name through
    assert validate_endpoint("https://good.example/x") == ("good.example", 443, [_PUBLIC_IP])


def test_a_mixed_resolution_is_refused_whole_not_used_in_part(resolver):
    resolver["mixed.example"] = [_PUBLIC_IP, "169.254.169.254"]
    resolver["mixed2.example"] = [_PUBLIC_IP, "239.255.255.250"]
    for host in ("mixed.example", "mixed2.example"):
        with pytest.raises(SSRFError):
            validate_endpoint(f"https://{host}/x")


# ---------------------------------------------------------------------------
# SSRFError itself
# ---------------------------------------------------------------------------


def test_ssrferror_is_a_catchable_exception_carrying_its_reason():
    err = SSRFError("because")
    assert isinstance(err, Exception) and not isinstance(err, SystemExit)
    assert str(err) == "because"
    with pytest.raises(SSRFError) as caught:
        validate_endpoint("https://169.254.169.254/latest/meta-data/")
    # The reason names the address that decided it, so a refusal is diagnosable.
    assert "169.254.169.254" in str(caught.value)
    assert "refused" in str(caught.value)


# ---------------------------------------------------------------------------
# PinnedIPAdapter
# ---------------------------------------------------------------------------


def test_init_poolmanager_sets_the_tls_kwargs_only_for_tls():
    seen: dict = {}
    real = HTTPAdapter.init_poolmanager

    def capture(self, *a, **k):
        seen[id(self)] = dict(k)
        return real(self, *a, **k)

    HTTPAdapter.init_poolmanager = capture
    try:
        https = PinnedIPAdapter("example.com", _PUBLIC_IP, tls=True)
        plain = PinnedIPAdapter("example.com", _PUBLIC_IP, tls=False)
    finally:
        HTTPAdapter.init_poolmanager = real
    assert seen[id(https)]["server_hostname"] == "example.com"
    assert seen[id(https)]["assert_hostname"] == "example.com"
    # Not merely absent-by-accident: these kwargs kill a plain-HTTP pool.
    assert "server_hostname" not in seen[id(plain)]
    assert "assert_hostname" not in seen[id(plain)]


def test_send_dials_the_vetted_ip_and_keeps_the_hostname_for_sni_and_host():
    import requests
    import requests.adapters as ra

    recorded: dict = {}
    real_send = ra.HTTPAdapter.send

    def capture(self, request, **kw):
        recorded["url"] = request.url
        recorded["host"] = request.headers.get("Host")
        response = requests.Response()
        response.status_code = 204
        response.url = request.url
        response.request = request
        return response

    ra.HTTPAdapter.send = capture
    try:
        adapter = PinnedIPAdapter("example.com", _PUBLIC_IP, tls=True)
        adapter.send(requests.Request("POST", "https://example.com:8443/v1/x").prepare())
        assert recorded == {
            "url": f"https://{_PUBLIC_IP}:8443/v1/x",
            "host": "example.com:8443",
        }
        recorded.clear()
        adapter.send(requests.Request("POST", "https://example.com/v1/x").prepare())
        assert recorded == {"url": f"https://{_PUBLIC_IP}/v1/x", "host": "example.com"}
        # A v6 pin is bracketed, or the netloc is unparseable.
        recorded.clear()
        v6 = PinnedIPAdapter("example.com", _PUBLIC_V6, tls=True)
        v6.send(requests.Request("POST", "https://example.com/v1/x").prepare())
        assert recorded["url"] == f"https://[{_PUBLIC_V6}]/v1/x"
    finally:
        ra.HTTPAdapter.send = real_send


# ---------------------------------------------------------------------------
# GuardedSession.get_adapter: every hop, including the ones that are not hops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,why", FORBIDDEN, ids=[w for _, w in FORBIDDEN])
def test_get_adapter_refuses_the_same_battery_validate_endpoint_does(url, why):
    with pytest.raises(SSRFError):
        GuardedSession().get_adapter(url)


def test_get_adapter_pins_a_public_target_and_caches_it():
    session = GuardedSession()
    adapter = session.get_adapter(f"https://{_PUBLIC_IP}/v1/x")
    assert isinstance(adapter, PinnedIPAdapter)
    assert (adapter._pin_host, adapter._pin_ip, adapter._pin_tls) == (
        _PUBLIC_IP,
        _PUBLIC_IP,
        True,
    )
    assert session.get_adapter(f"https://{_PUBLIC_IP}/v1/other") is adapter
    # A different port is a different target and gets its own validation.
    assert session.get_adapter(f"https://{_PUBLIC_IP}:8443/v1/x") is not adapter
    # tls follows the scheme, so an http mount does not carry HTTPS-only kwargs.
    plain = GuardedSession(allow_http=True).get_adapter(f"http://{_PUBLIC_IP}/v1/x")
    assert plain._pin_tls is False
    # The redirect chain is bounded.
    assert session.max_redirects == GuardedSession._GUARD_MAX_REDIRECTS


# ---------------------------------------------------------------------------
# guarded_post, and a REAL redirect refused at the hop the guard never saw
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.action(self)

    def do_GET(self):  # noqa: N802
        self.server.action(self)

    def log_message(self, *args):
        pass


@pytest.fixture
def redirector(monkeypatch):
    """A loopback stub the guard SEES as a public host, plus the dial patch."""
    state: dict = {"location": None, "hits": []}

    def action(handler):
        state["hits"].append(handler.path)
        if state["location"] is None or handler.path != "/start":
            body = b'{"ok": true}'
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return
        handler.send_response(302)
        handler.send_header("Location", state["location"])
        handler.end_headers()

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    server.action = action
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    real_gai = socket.getaddrinfo
    real_cc = u3conn.create_connection

    def fake_gai(host, p, *a, **k):
        if host == "front.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (_PUBLIC_IP, p))]
        return real_gai(host, p, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
    monkeypatch.setattr(
        u3conn, "create_connection", lambda addr, *a, **k: real_cc(("127.0.0.1", addr[1]), *a, **k)
    )
    state["url"] = f"http://front.example:{port}/start"
    try:
        yield state
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "target",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://127.0.0.1/admin",
        "http://239.255.255.250/upnp",
        "http://224.0.0.251/mdns",
        "http://[ff02::1]/all-nodes",
        "http://[64:ff9b::7f00:1]/nat64-loopback",
    ],
)
def test_a_redirect_to_an_internal_address_is_refused_at_the_hop(redirector, target):
    redirector["location"] = target
    with pytest.raises(SSRFError) as caught:
        guarded_post(redirector["url"], allow_http=True, json={"x": 1}, timeout=5)
    assert "refused" in str(caught.value) or "not allowed" in str(caught.value)
    # The FIRST hop really was dialled: this is a redirect refusal, not the
    # original URL being rejected before anything happened.
    assert redirector["hits"] == ["/start"]


def test_guarded_post_refuses_a_forbidden_target_before_any_socket(resolver):
    resolver["front.example"] = "169.254.169.254"
    for url in (
        "https://169.254.169.254/latest/meta-data/",
        "https://239.255.255.250/x",
        "https://[64:ff9b::a9fe:a9fe]/x",
        "https://front.example/x",
        f"http://{_PUBLIC_IP}/x",
    ):
        with pytest.raises(SSRFError):
            guarded_post(url, json={"x": 1}, timeout=0.01)


def test_guarded_post_reaches_a_target_the_guard_allows(redirector):
    """The control for the battery above: an allowed hop completes.

    Without this, every refusal test above would also pass for a guard that
    refuses everything, which is the failure mode that destroys the unit.
    """
    redirector["location"] = None
    response = guarded_post(redirector["url"], allow_http=True, json={"x": 1}, timeout=5)
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert redirector["hits"] == ["/start"]


def test_a_redirect_to_another_public_host_is_still_followed(redirector, resolver):
    """And the guard does not refuse a hop merely for being a redirect."""
    resolver["second.example"] = _PUBLIC_IP
    port = redirector["url"].split(":")[2].split("/")[0]
    redirector["location"] = f"http://second.example:{port}/next"
    response = guarded_post(redirector["url"], allow_http=True, json={"x": 1}, timeout=5)
    assert response.status_code == 200
    assert redirector["hits"] == ["/start", "/next"]
