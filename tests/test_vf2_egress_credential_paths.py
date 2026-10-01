"""The two credential-bearing egress paths must run through the SSRF guard.

VF-2. `vfairness.net.egress` exists and is proven to refuse hostile targets, but
it was not wired everywhere. Two call sites POSTed a caller-supplied URL with the
caller's credential attached, using a bare `requests.post`:

  * `llm.scorers.LLMJudgeScorer._call_judge` sends `auth_token` as
    `Authorization: Bearer ...` (or `x-api-key`) to `endpoint_url`.
  * `xai.sidecar_cli._endpoint_callable` does the same to the `endpoint_url` in
    a stdin payload that this module's own header calls untrusted input.

Both therefore did two things at once: reached an internal service, and handed
it the credential. `requests` follows 3xx by default, so a single 302 to
169.254.169.254 forwarded the key to cloud metadata.

An advertised security control that does not run on every path is the same
failure class as a false FAIR verdict: the surface implies a check that is not
executed.

What these tests pin, and why in this shape. Asserting "guarded_post is called"
would pass against a guard that refuses nothing, so the refusal itself is
asserted against real hostile targets. And the happy path is asserted too: a
gate that refuses everything is not a working guard, it is a broken client.
"""

from __future__ import annotations

import pytest

from vfairness.net.egress import SSRFError

# Targets the guard must refuse. Cloud metadata is the one that matters most:
# it is the classic credential-exfiltration destination.
HOSTILE = [
    "http://169.254.169.254/latest/meta-data/",  # AWS/GCP/Azure metadata
    "http://127.0.0.1:8080/v1/chat/completions",  # loopback, without the opt-in
    "http://10.0.0.5/v1/chat/completions",  # RFC1918
    "http://192.168.1.10/v1/chat/completions",  # RFC1918
    "http://[::1]:8080/v1/chat/completions",  # IPv6 loopback
    "file:///etc/passwd",  # not http(s) at all
]


# --- llm.scorers.LLMJudgeScorer ---------------------------------------------


@pytest.mark.parametrize("url", HOSTILE)
def test_judge_refuses_hostile_endpoints_without_sending_the_key(url, recwarn):
    """The refusal must happen before the credential leaves the process."""
    from vfairness.llm.scorers import LLMJudgeScorer

    judge = LLMJudgeScorer(endpoint_url=url, auth_token="sk-secret-not-to-be-sent")
    assert judge._call_judge("some text") is None

    messages = " ".join(str(w.message) for w in recwarn)
    assert "egress guard" in messages, (
        f"a refusal must be reported as a policy refusal, not a generic outage; got: {messages}"
    )
    assert "sk-secret-not-to-be-sent" not in messages, "the credential must not leak into a warning"


def test_judge_score_is_nan_not_a_neutral_score_when_refused():
    """Fail-closed: a refused judge must not read as a mid-scale 'no bias found'."""
    from vfairness.llm.scorers import LLMJudgeScorer

    judge = LLMJudgeScorer(endpoint_url="http://169.254.169.254/v1", auth_token="k")
    with pytest.warns(RuntimeWarning):
        value = judge.score("some text")
    assert value != value, "a refused judge must return NaN, never a neutral score"


def test_judge_loopback_needs_the_explicit_opt_in():
    """Off by default; on when asked. A guard that cannot be opted into is unusable."""
    from vfairness.net.egress import validate_endpoint

    url = "http://127.0.0.1:11434/v1/chat/completions"
    with pytest.raises(SSRFError):
        validate_endpoint(url, allow_http=True, allow_loopback=False)
    validate_endpoint(url, allow_http=True, allow_loopback=True)  # must not raise


def test_judge_carries_the_opt_in_through_to_the_guard():
    from vfairness.llm.scorers import LLMJudgeScorer

    assert LLMJudgeScorer(endpoint_url="https://api.openai.com/v1")._allow_loopback is False
    assert (
        LLMJudgeScorer(
            endpoint_url="https://api.openai.com/v1", allow_loopback=True
        )._allow_loopback
        is True
    )


# --- xai.sidecar_cli._endpoint_callable -------------------------------------


@pytest.mark.parametrize("url", HOSTILE)
def test_sidecar_endpoint_refuses_hostile_targets(url):
    """Refused while BUILDING the callable, before any row is serialised."""
    from vfairness.xai.sidecar_cli import _endpoint_callable

    payload = {"endpoint_url": url, "auth_type": "bearer", "auth_token": "secret-token"}
    with pytest.raises(SSRFError):
        _endpoint_callable(payload, ["a", "b"])


def test_sidecar_loopback_needs_the_explicit_opt_in():
    from vfairness.xai.sidecar_cli import _endpoint_callable

    payload = {"endpoint_url": "http://127.0.0.1:9000/predict", "auth_type": "none"}
    with pytest.raises(SSRFError):
        _endpoint_callable(dict(payload), ["a"])
    # With the opt-in the callable is built (no request is made at build time).
    fn = _endpoint_callable(dict(payload, allow_loopback=True), ["a"])
    assert callable(fn)


# --- the control: a legitimate public endpoint must still be accepted --------


def test_a_public_endpoint_is_still_accepted():
    """Without this, every test above would pass on a guard that refuses everything."""
    from vfairness.net.egress import validate_endpoint

    host, port, ips = validate_endpoint("https://api.openai.com/v1/chat/completions")
    assert host == "api.openai.com"
    assert port == 443
    assert ips, "the guard must resolve and return the vetted IPs"


def test_sidecar_builds_a_callable_for_a_public_endpoint():
    """Control for the sidecar half specifically."""
    from vfairness.xai.sidecar_cli import _endpoint_callable

    fn = _endpoint_callable(
        {"endpoint_url": "https://api.openai.com/predict", "auth_type": "none"}, ["a"]
    )
    assert callable(fn)


# --- the wiring itself, so a future refactor cannot silently unwire it -------


def test_neither_call_site_uses_a_bare_requests_post():
    """AST check. A refactor back to requests.post would restore the defect and
    every behavioural test above would keep passing if the guard were merely
    bypassed for a public URL, so the call shape is pinned directly."""
    import ast
    import inspect
    import textwrap

    from vfairness.llm import scorers as scorers_mod
    from vfairness.xai import sidecar_cli as sidecar_mod

    for mod, fn_name in ((scorers_mod, "_call_judge"), (sidecar_mod, "_endpoint_callable")):
        if fn_name == "_call_judge":
            src = inspect.getsource(scorers_mod.LLMJudgeScorer._call_judge)
        else:
            src = inspect.getsource(sidecar_mod._endpoint_callable)
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"post", "get", "put", "request"} and isinstance(
                    node.func.value, ast.Name
                ):
                    assert node.func.value.id != "requests", (
                        f"{fn_name} calls requests.{node.func.attr} directly; "
                        "credential-bearing egress must go through guarded_post"
                    )
