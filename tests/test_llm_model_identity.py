"""LF-14: the model that answered is recorded, never assumed.

Every behavioural finding this library produces is a statement about the model
it was measured on. A stored result whose identity is unknown has no shelf
life; a stored result whose identity was ASSUMED is worse, because it looks
like it has one.

Each test below pins one way that assumption used to be made.
"""

from __future__ import annotations

import http.server
import json
import threading
import warnings

import pytest

from vfairness._not_assessed import NOT_ASSESSED
from vfairness.llm.api_proxy import LLMApiProxy
from vfairness.llm.model_identity import (
    ModelIdentity,
    clean_model_name,
    compare_model_identity,
    describe_model_identity,
    identity_from_run,
)

# ---------------------------------------------------------------------------
# Capture: what the endpoint said, and only that
# ---------------------------------------------------------------------------


@pytest.fixture()
def proxy() -> LLMApiProxy:
    # Loopback, allowed explicitly: nothing here sends a request. The extractor
    # under test is pure, and it must be exercised on the REAL class rather
    # than on a copy of its logic.
    return LLMApiProxy(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions",
        api_format="openai",
        model_name="gpt-4o",
        allow_loopback=True,
    )


def test_reads_the_endpoints_own_echo(proxy: LLMApiProxy) -> None:
    assert proxy._extract_reported_model({"model": "gpt-4-0613"}) == "gpt-4-0613"
    assert proxy._extract_reported_model({"model": "claude-3-opus-20240229"}) == (
        "claude-3-opus-20240229"
    )


def test_never_falls_back_to_the_requested_model(proxy: LLMApiProxy) -> None:
    """The proxy knows the configured name. It must not answer with it.

    Returning the request's own model parameter here would manufacture
    agreement out of nothing: a caller comparing the two would see a match and
    read it as the endpoint confirming what served the request.
    """
    for body in (
        {},
        {"choices": [{"message": {"content": "hi"}}]},
        {"model": None},
        {"model": ""},
        {"model": "   "},
        {"model": 7},
        {"model": ["gpt-4o"]},
    ):
        assert proxy._extract_reported_model(body) is None, body
    # And the configured name is still what the proxy was built with, so the
    # absence above is a real refusal rather than the proxy not knowing it.
    assert proxy._config.model_name == "gpt-4o"


def test_a_non_dict_body_is_none_not_a_crash(proxy: LLMApiProxy) -> None:
    for body in (None, "gpt-4o", 7, ["gpt-4o"]):
        assert proxy._extract_reported_model(body) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Per run: one run must be served by one model
# ---------------------------------------------------------------------------


def _run(*names: object) -> list:
    return [{"prompt": "p", "responses": [{"reported_model": n} for n in names]}]


def test_a_consistent_run_records_the_name_and_does_not_overclaim() -> None:
    identity = identity_from_run(_run("gpt-4o", "gpt-4o", "gpt-4o"), configured_model="gpt-4o")
    assert identity.reported_model == "gpt-4o"
    assert identity.is_consistent is True
    assert identity.n_reporting == 3
    assert "self-report, not an independent check" in identity.coverage_statement


def test_a_run_served_by_two_models_reports_neither_as_the_answer() -> None:
    """The strongest identity signal a run can produce, and only the engine sees it.

    The platform makes one connection request. A run here makes twenty-five per
    prompt. If the name changes partway through, every statistic over those
    responses is a statistic over two systems. Taking the first name seen would
    hide that completely.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identity = identity_from_run(_run("gpt-4o", "gpt-4o-mini"), configured_model="gpt-4o")

    assert identity.reported_model is None
    assert identity.reported_names == ("gpt-4o", "gpt-4o-mini")
    assert identity.is_consistent is False
    assert len(caught) == 1
    assert "DIFFERENT ones" in str(caught[0].message)
    assert "NOT the first name seen" in str(caught[0].message)
    assert "DID NOT SERVE ONE MODEL" in identity.coverage_statement
    assert "should be repeated before it is reported" in identity.coverage_statement


def test_nothing_reported_is_unknown_not_consistent() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identity = identity_from_run(_run(None, "", "   "), configured_model="gpt-4o")

    # `is_consistent` must not answer True here. Three responses that named
    # nothing agree about nothing.
    assert identity.is_consistent is None
    assert identity.reported_model is None
    assert identity.configured_model == "gpt-4o"
    assert "None of the 3 responses carried a model name" in identity.coverage_statement
    assert any("could not check" in str(w.message) for w in caught)


def test_partial_reporting_says_which_responses_were_not_identified() -> None:
    identity = identity_from_run(_run("gpt-4o", None, "gpt-4o"))
    assert identity.reported_model == "gpt-4o"
    assert identity.n_reporting == 2
    assert identity.n_responses == 3
    assert "not evidence that the same model served them" in identity.coverage_statement


def test_accepts_every_shape_send_batch_can_return() -> None:
    single = identity_from_run({"reported_model": "m"})
    flat = identity_from_run([{"reported_model": "m"}])
    batch = identity_from_run(_run("m"))
    assert single.reported_model == flat.reported_model == batch.reported_model == "m"
    assert identity_from_run([]).n_responses == 0


# ---------------------------------------------------------------------------
# Comparison: is a stored finding still about this model?
# ---------------------------------------------------------------------------


def test_a_matching_configured_name_is_not_a_match() -> None:
    """The setting did not move. That is a different claim.

    A provider can serve a different build under a stable name, so a matching
    configured name establishes nothing about what answered. Returning "same"
    here is a could-not-check wearing a verified badge.
    """
    result = compare_model_identity(
        ModelIdentity(configured_model="gpt-4o"),
        ModelIdentity(configured_model="gpt-4o"),
    )
    assert result.state == NOT_ASSESSED
    assert result.basis == "configuration_only"
    assert result.needs_retest is False
    assert "could-not-check, not a match" in result.statement


def test_same_requires_the_endpoints_echo_on_both_sides() -> None:
    result = compare_model_identity(
        ModelIdentity(configured_model="gpt-4o", reported_model="gpt-4o-2026-05-13"),
        ModelIdentity(configured_model="gpt-4o", reported_model="gpt-4o-2026-05-13"),
    )
    assert result.state == "same"
    assert result.basis == "endpoint_report"
    # Even then it does not claim more than a self-report.
    assert "self-report rather than an independent check" in result.statement


def test_a_value_on_one_side_is_reported_never_dropped() -> None:
    """The case that used to vanish.

    The endpoint said gpt-3.5-turbo answered a gpt-4o configuration. It cannot
    go into `changed` because there is nothing to diff it against, and the
    fixed sentence beside it then told the reader the endpoint reported
    nothing.
    """
    result = compare_model_identity(
        ModelIdentity(configured_model="gpt-4o", reported_model="gpt-3.5-turbo"),
        ModelIdentity(configured_model="gpt-4o"),
    )
    assert result.state == NOT_ASSESSED
    assert result.one_sided == ("reported_model",)
    assert "gpt-3.5-turbo" in result.statement
    assert "did not report a model name" not in result.statement


def test_a_real_change_triggers_a_retest_and_case_counts() -> None:
    changed = compare_model_identity(
        ModelIdentity(configured_model="gpt-4o", reported_model="a"),
        ModelIdentity(configured_model="gpt-4o", reported_model="b"),
    )
    assert changed.state == "changed"
    assert changed.needs_retest is True
    assert "does not carry over" in changed.statement

    cased = compare_model_identity(
        ModelIdentity(configured_model="gpt-4o"),
        ModelIdentity(configured_model="GPT-4o"),
    )
    assert cased.state == "changed"


def test_an_unknown_identity_does_not_trigger_a_retest() -> None:
    """An unknown identity is not evidence of a change.

    Treating it as one would re-run every result whose endpoint never named
    itself, forever. The report states the limit instead.
    """
    unknown = compare_model_identity(None, None)
    assert unknown.state == NOT_ASSESSED
    assert unknown.needs_retest is False
    assert unknown.basis == "none"
    assert "not a sign that the model is unchanged" in unknown.statement


# ---------------------------------------------------------------------------
# Description: never print a setting where a reader expects an observation
# ---------------------------------------------------------------------------


def test_describe_keeps_the_two_names_apart() -> None:
    assert describe_model_identity(None) == "model identity not recorded"
    assert "the configured name; the endpoint reported none" in describe_model_identity(
        ModelIdentity(configured_model="gpt-4o")
    )
    assert "the endpoint reported the same name" in describe_model_identity(
        ModelIdentity(configured_model="gpt-4o", reported_model="gpt-4o")
    )
    assert 'but the endpoint reported "gpt-3.5"' in describe_model_identity(
        ModelIdentity(configured_model="gpt-4o", reported_model="gpt-3.5")
    )
    assert "more than one model answered this run" in describe_model_identity(
        ModelIdentity(configured_model="gpt-4o", reported_names=("a", "b"))
    )


def test_clean_model_name_refuses_blanks_and_non_strings() -> None:
    assert clean_model_name("  gpt-4o  ") == "gpt-4o"
    for value in (None, "", "   ", 7, ["gpt-4o"], {"model": "gpt-4o"}):
        assert clean_model_name(value) is None


# ---------------------------------------------------------------------------
# End to end: the field must survive the real request path, not just the
# extractor. A pure-function test proves the logic; it proves nothing about
# whether send_prompt actually puts the value on what it returns.
# ---------------------------------------------------------------------------


def _stub_server(payload: dict):
    class _Stub(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's own name
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # noqa: D102 - silence the stub's stderr
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _send_against(payload: dict) -> dict:
    srv = _stub_server(payload)
    try:
        url = "http://127.0.0.1:{0}/v1/chat/completions".format(srv.server_address[1])
        proxy = LLMApiProxy(
            endpoint_url=url,
            api_format="openai",
            model_name="gpt-4o-CONFIGURED",
            allow_loopback=True,
        )
        return proxy.send_prompt("ping")
    finally:
        srv.shutdown()


_OK = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}


def test_send_prompt_carries_the_endpoints_echo_over_a_real_request() -> None:
    out = _send_against({**_OK, "model": "gpt-4o-2026-05-13"})
    assert out["text"] == "pong"
    assert out["reported_model"] == "gpt-4o-2026-05-13"


def test_a_silent_endpoint_yields_none_over_a_real_request() -> None:
    """The configured name is 'gpt-4o-CONFIGURED' and must appear nowhere."""
    out = _send_against(_OK)
    assert out["text"] == "pong"
    assert out["reported_model"] is None
    assert "CONFIGURED" not in repr(out)


def test_a_batch_of_silent_responses_is_an_unknown_identity_end_to_end() -> None:
    srv = _stub_server(_OK)
    try:
        url = "http://127.0.0.1:{0}/v1/chat/completions".format(srv.server_address[1])
        proxy = LLMApiProxy(
            endpoint_url=url,
            api_format="openai",
            model_name="gpt-4o-CONFIGURED",
            allow_loopback=True,
        )
        results = proxy.send_batch(["ping"], n_runs=3)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            identity = identity_from_run(results, configured_model="gpt-4o-CONFIGURED")
    finally:
        srv.shutdown()

    assert identity.n_responses == 3
    assert identity.reported_model is None
    assert identity.is_consistent is None
    assert "None of the 3 responses carried a model name" in identity.coverage_statement
