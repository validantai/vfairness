"""Egress SSRF guard for the vfairness engine (LLM proxy, judges, sidecar).
Canonical copy of the consumer net_guard, so library-side egress (api_proxy, scorers)
is guarded identically. Zero third-party deps beyond requests. See net_guard tests."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter


class SSRFError(Exception):
    """A requested egress URL is not allowed (non-public target, bad scheme, etc.)."""


# Cloud-metadata endpoints are already caught by is_global (link-local / ULA), but we
# name them explicitly as belt-and-suspenders in case a future is_global change regresses.
_EXPLICIT_BLOCK = frozenset({"169.254.169.254", "fd00:ec2::254", "::ffff:169.254.169.254"})


def _ip_is_forbidden(ip_str: str, allow_loopback: bool = False) -> bool:
    """True if this IP must never be an egress target.

    Uses ipaddress.is_global as the primary rule, PLUS explicit refusals for the
    three families this docstring used to credit is_global with and it does not
    cover. is_global is False for private (RFC1918), loopback, link-local (incl.
    169.254.169.254), unique-local (fc00::/7), CGNAT, documentation and other
    non-routable ranges. IPv4-mapped IPv6 is unwrapped first so ::ffff:127.0.0.1
    cannot smuggle a loopback past the check.

    MULTICAST, RESERVED AND UNSPECIFIED ARE NOT COVERED BY is_global AND ARE
    REFUSED HERE (2026-09-30). The claim above read "reserved, unspecified,
    multicast" for years and only ``unspecified`` is true of CPython's IPv4
    table; the v6 table covers none of the three. Measured on this module with
    the default (most restrictive) settings, every one of these returned
    ALLOWED from ``validate_endpoint``, ``GuardedSession.get_adapter`` and
    therefore ``guarded_post``:

        224.0.0.1        all-hosts multicast      is_global True
        224.0.0.251      mDNS multicast           is_global True
        239.255.255.250  SSDP multicast           is_global True
        ff02::1          v6 all-nodes multicast   is_global True
        ff05::1:3        v6 site-local multicast  is_global True
        64:ff9b::7f00:1  NAT64 well-known prefix embedding 127.0.0.1
        64:ff9b::a9fe:a9fe   the same prefix embedding 169.254.169.254
        ::a00:1          deprecated IPv4-compatible form of 10.0.0.1

    The last three are the door ``ipv4_mapped`` was unwrapped to close, spelled
    two other ways: an embedded loopback / RFC1918 / cloud-metadata address that
    reaches the same internal service on any host with NAT64 or v4-compatible
    addressing configured. They are all inside ``is_reserved``
    (64:ff9b::/96 and ::/96), so the reserved clause closes them without a
    per-prefix unwrap table that could go stale.

    ORDER MATTERS AND IS NOT ARBITRARY. ``::1`` is BOTH is_loopback and
    is_reserved, so a reserved refusal placed above the loopback carve-out
    would refuse an operator's own ``https://[::1]:8080`` model server and
    break the first-class local-endpoint case. The loopback decision is
    therefore taken first, below _EXPLICIT_BLOCK and above the three new
    refusals.

    allow_loopback carves out LOOPBACK ONLY (127.0.0.0/8, ::1) for the
    first-class case of an operator pointing the engine at a model server on
    their own machine (Ollama, vLLM, a test stub). It is opt-in per call and
    never widens to RFC1918, link-local or cloud-metadata: _EXPLICIT_BLOCK is
    enforced above it, so 169.254.169.254 stays refused whatever the caller
    asks for.
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # not parseable -> refuse
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    if str(addr) in _EXPLICIT_BLOCK or ip_str in _EXPLICIT_BLOCK:
        return True
    # Loopback is the ONLY range an explicit opt-in can unlock, and it is decided
    # first because ::1 also matches is_reserved below.
    if addr.is_loopback:
        return not allow_loopback
    if addr.is_multicast or addr.is_reserved or addr.is_unspecified:
        return True
    return not addr.is_global


def _resolve_ips(host: str, port: int) -> list[str]:
    """Resolve host to every A/AAAA address (deduped). Raise SSRFError on failure."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError(f"cannot resolve host '{host}': {exc}") from exc
    # sockaddr is (host, port) for AF_INET and (host, port, flow, scope) for AF_INET6,
    # so typeshed types element 0 as str | int; the address is always the str.
    ips = sorted({str(info[4][0]) for info in infos})
    if not ips:
        raise SSRFError(f"host '{host}' did not resolve to any address")
    return ips


def validate_endpoint(
    url: str, allow_http: bool = False, allow_loopback: bool = False
) -> tuple[str, int, list[str]]:
    """Validate an egress URL. Return (host, port, vetted_ips) or raise SSRFError.

    Refuses: a non-http(s) scheme; http unless allow_http; a missing host; a host
    that does not resolve; and any host whose resolved set contains a non-public IP
    (all-or-nothing, so a mixed public/private result is refused, not partially used).

    allow_loopback additionally permits 127.0.0.0/8 and ::1 (a local model server or
    test stub). It does NOT permit any other private range and never permits the
    cloud-metadata addresses. Default False: an untrusted, user-supplied URL must be
    validated with the default, and only a deliberate local-endpoint call opts in.
    """
    parsed = urlparse((url or "").strip())
    scheme = (parsed.scheme or "").lower()
    if scheme == "https" or (allow_http and scheme == "http"):
        pass
    else:
        raise SSRFError(f"scheme '{scheme or '(none)'}' not allowed (https required)")
    host = parsed.hostname
    if not host:
        raise SSRFError("URL has no host")
    port = parsed.port or (443 if scheme == "https" else 80)

    # A bare IP literal is validated directly; a hostname is resolved and every
    # returned address is checked.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    ips = [str(literal)] if literal is not None else _resolve_ips(host, port)

    for ip in ips:
        if _ip_is_forbidden(ip, allow_loopback=allow_loopback):
            raise SSRFError(f"endpoint '{host}' resolves to a non-public address ({ip}); refused")
    return host, port, ips


class PinnedIPAdapter(HTTPAdapter):
    """A requests adapter that connects to a PRE-VETTED IP while keeping the original
    hostname for TLS SNI + certificate verification (so cert checks still pass) and for
    the Host header. Pinning to the already-validated IP defeats DNS rebinding: urllib3
    never re-resolves the hostname, so it cannot be pointed at a private address after
    our check."""

    def __init__(self, host: str, pinned_ip: str, tls: bool = True, **kwargs):
        self._pin_host = host
        self._pin_ip = pinned_ip
        # tls=False for the http:// mount. server_hostname/assert_hostname are
        # HTTPSConnection-only kwargs; urllib3 does not strip them for plain-HTTP
        # pools, so setting them globally made every http:// request die with
        # "HTTPConnection.__init__() got an unexpected keyword argument
        # 'assert_hostname'". Pinning still applies without them; there is simply
        # no certificate to bind to the hostname on a cleartext connection.
        self._pin_tls = tls
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        parsed = urlparse(request.url)
        if parsed.hostname == self._pin_host:
            # Rewrite the connect target to the vetted IP; keep Host + SNI on the name.
            netloc_ip = self._pin_ip if ":" not in self._pin_ip else f"[{self._pin_ip}]"
            if parsed.port:
                netloc_ip += f":{parsed.port}"
            request.url = parsed._replace(netloc=netloc_ip).geturl()
            request.headers["Host"] = (
                self._pin_host if not parsed.port else f"{self._pin_host}:{parsed.port}"
            )
        return super().send(request, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        # server_hostname drives SNI; assert_hostname makes urllib3 verify the peer cert
        # against the real hostname even though we dialed an IP. Both are HTTPS-only:
        # see the note in __init__ for why they must not reach a plain-HTTP pool.
        if self._pin_tls:
            kwargs["server_hostname"] = self._pin_host
            kwargs["assert_hostname"] = self._pin_host
        return super().init_poolmanager(*args, **kwargs)


class GuardedSession(requests.Session):
    """A requests Session that re-runs the SSRF guard on EVERY hop, including the
    targets of HTTP redirects.

    A plain PinnedIPAdapter validates and pins exactly one host: the one
    validate_endpoint() vetted up front. requests, however, follows 3xx redirects by
    default (allow_redirects defaults to True), and a redirect Location can name a host
    the guard never saw. PinnedIPAdapter.send() only rewrites/pins when the request
    host equals its one pinned host; for any OTHER host it falls straight through to
    the network with no _ip_is_forbidden() check. That turned the guard into a one-hop
    speed bump: a single 302 to 169.254.169.254 (or any RFC1918 / loopback host)
    reached cloud metadata / internal services unchecked.

    This session closes that hole by overriding get_adapter(). requests calls
    get_adapter() for the original request AND, via
    resolve_redirects() -> send() -> get_adapter(), for every redirect hop. Each URL
    is run through validate_endpoint() (scheme, host, and EVERY resolved IP including
    IPv6/IPv4-mapped and decimal/hex/octal encodings, refusing private / loopback /
    link-local / cloud-metadata) and served by a PinnedIPAdapter pinned to that hop's
    freshly resolved IP. A hop that fails raises SSRFError, so the redirect is refused
    before any request is dialled to the forbidden target. Because resolution happens
    per hop, a DNS rebind between hops is caught as well. max_redirects is clamped so a
    redirect loop cannot spin the guard indefinitely.
    """

    # Bound redirect chains: every hop is validated, but a chain must not spin.
    _GUARD_MAX_REDIRECTS = 5

    def __init__(
        self,
        *,
        allow_http: bool = False,
        allow_loopback: bool = False,
        max_retries=None,
    ) -> None:
        super().__init__()
        self._guard_allow_http = allow_http
        self._guard_allow_loopback = allow_loopback
        self._guard_max_retries = max_retries
        self.max_redirects = self._GUARD_MAX_REDIRECTS
        # Cache the pinned adapter per (scheme, host, port): the primary host is
        # validated once and reused (the pin already defeats same-host rebinding),
        # while each distinct redirect host is validated the first time it is seen.
        self._guard_adapters: dict[tuple[str, str, int], PinnedIPAdapter] = {}

    def get_adapter(self, url: str) -> PinnedIPAdapter:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)
        key = (scheme, host, port)
        cached = self._guard_adapters.get(key)
        if cached is not None:
            return cached
        # Validate this exact target. This is what makes a redirect to a private /
        # loopback / link-local / cloud-metadata host fail closed: resolve_redirects()
        # routes every hop back through here, so validate_endpoint() sees it too.
        vetted_host, _port, ips = validate_endpoint(
            url,
            allow_http=self._guard_allow_http,
            allow_loopback=self._guard_allow_loopback,
        )
        adapter_kwargs = {}
        if self._guard_max_retries is not None:
            adapter_kwargs["max_retries"] = self._guard_max_retries
        adapter = PinnedIPAdapter(vetted_host, ips[0], tls=(scheme == "https"), **adapter_kwargs)
        self._guard_adapters[key] = adapter
        return adapter

    def close(self) -> None:
        """Close every pinned adapter, then the parent session.

        ONE FAILING close() USED TO STRAND EVERYTHING AFTER IT (2026-09-27). The
        loop had no try/finally, so the first adapter whose ``close()`` raised
        aborted it. Measured on a session holding a raising adapter and one healthy
        one, with a recording adapter mounted on the parent session:

            before: RuntimeError('poolmanager gone') propagated,
                    len(self._guard_adapters) == 2 (nothing cleared),
                    the parent session's adapter was NOT closed (super().close()
                    never reached), so its connection pools leaked
            after:  the same RuntimeError propagates,
                    len(self._guard_adapters) == 0,
                    the parent session's adapter IS closed

        The exception still reaches the caller: a failing close is a real failure
        and swallowing it would be the other defect. What it may not do is decide
        that the remaining sockets stay open. ``HTTPAdapter.close()`` is
        ``poolmanager.clear()`` and does not normally raise, so this is hardening,
        not a live failure.
        """
        failures: list[BaseException] = []
        for adapter in self._guard_adapters.values():
            try:
                adapter.close()
            except Exception as exc:
                failures.append(exc)
        self._guard_adapters.clear()
        try:
            super().close()
        except Exception as exc:
            failures.append(exc)
        if failures:
            first = failures[0]
            if len(failures) > 1:
                first.add_note(
                    f"GuardedSession.close: {len(failures)} close() calls failed. The "
                    f"others were {[repr(exc) for exc in failures[1:]]}. Every adapter "
                    "and the parent session were still closed before this was raised."
                )
            raise first


def guarded_post(
    url: str, *, allow_http: bool = False, allow_loopback: bool = False, **kwargs
) -> requests.Response:
    """A drop-in for requests.post that first validates the URL (SSRF guard) and pins
    the connection to the vetted IP. Raises SSRFError if the URL is not allowed; other
    kwargs (json, headers, timeout, ...) pass straight through to requests.

    Redirects are guarded too: the request runs on a GuardedSession, which re-runs
    validate_endpoint() on every redirect hop and refuses a 3xx Location that points
    at a private / loopback / link-local / cloud-metadata host. Validating only the
    original URL (the requests default) let a single 302 defeat the whole guard."""
    # Eager check on the initial URL so a forbidden or malformed target raises a clear
    # SSRFError before any request is prepared; GuardedSession re-checks each hop after.
    validate_endpoint(url, allow_http=allow_http, allow_loopback=allow_loopback)
    session = GuardedSession(allow_http=allow_http, allow_loopback=allow_loopback)
    try:
        return session.post(url, **kwargs)
    finally:
        session.close()
