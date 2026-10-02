"""BGL3 batch llm-1: does the LLM proxy and the output analyzer refuse honestly?

Two files, fourteen units, one question per unit: when the quantity it reports
cannot be computed, does it say so, or does it hand back the value that reads
as its clean answer?

Everything asserted here was MEASURED on the real methods before and after the
fix, with the numbers in each docstring. Three defects were proven by
execution:

1. ``LLMApiProxy.send_prompt``, custom format, body
   ``{"error": "quota exceeded", "code": 429}`` returned
   ``text="{'error': 'quota exceeded', 'code': 429}"``: the error body became
   the model's generation, 40 characters of it, and nothing was logged because
   the text was not empty.
2. ``LLMApiProxy.send_prompt`` on any body it could not parse returned
   ``text=""`` with no marker on the returned dict, so an unparseable 200 was
   byte-identical to a model that generated nothing. A blank scores a clean
   0.0 on every presence scorer and reads as perfect stability across runs.
   ``{"choices": [{"message": {"content": null}}]}``, which is what a content
   filter returns, did not even get that far: it raised
   ``TypeError: object of type 'NoneType' has no len()`` out of a debug log
   line, and ``send_batch`` does not retry ``TypeError``.
3. ``OutputAnalyzer._compare`` answered ``p_value=1.0`` when
   ``stats.mannwhitneyu`` RAISED. On a 100% against 0% refusal split that read
   ``delta=1.0, p_value=1.0, is_significant=False, effect_size=3.1416
   "Cohen's h: large", assessed=True``: the strongest no-evidence answer the
   scale has, for a test that never ran.
4. ``OutputAnalyzer.analyze_all`` DROPPED the ``llm_judge`` row on empty input
   while emitting it on healthy input: 11 rows against 12, so the metric
   vanished exactly when nothing could be measured.

The refusals that were already correct are pinned too, because a unit that
refuses everything is as wrong as one that answers everything and only the
healthy case tells them apart.
"""

import logging
import warnings

import numpy as np
import pytest
import requests
from scipy import stats

from vfairness.llm.api_proxy import LLMApiProxy
from vfairness.llm.output_analysis import OutputAnalyzer

# ---------------------------------------------------------------------------
# Proxy harness. Loopback, allowed explicitly, and no request leaves the
# process: the session is replaced, so the REAL send_prompt / send_batch run
# against bodies chosen to make the generation unreadable.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: object) -> None:
        self._body = body
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._body


class _FakeSession:
    """Answers every POST with the same body, and counts the calls."""

    def __init__(self, body: object) -> None:
        self.body = body
        self.calls = 0

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        return _FakeResponse(self.body)


class _RaisingSession:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        raise self.exc


def _proxy(api_format: str, session: object) -> LLMApiProxy:
    proxy = LLMApiProxy(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions",
        api_format=api_format,
        model_name="m",
        allow_loopback=True,
    )
    proxy._session = session  # type: ignore[assignment]
    return proxy


# ---------------------------------------------------------------------------
# Finding 1: an error body is not a generation
# ---------------------------------------------------------------------------


def test_a_custom_error_body_is_not_the_models_answer() -> None:
    """Measured before the fix, custom format, body
    ``{"error": "quota exceeded", "code": 429}``:

        text == "{'error': 'quota exceeded', 'code': 429}"   (40 chars)
        text_unavailable_reason: the key did not exist

    The repr of the failure was handed on as the generation, so the scorers
    read five words, a length, a sentiment and a framing out of an HTTP error.
    """
    out = _proxy("custom", _FakeSession({"error": "quota exceeded", "code": 429})).send_prompt("hi")

    assert out["text"] == ""
    assert out["text_unavailable_reason"] == "no_text_in_response"
    assert "quota exceeded" not in out["text"]


def test_a_structure_under_a_text_key_is_not_a_generation() -> None:
    """The same shape one level in: ``{"output": {"content": "hi"}}`` used to
    return ``"{'content': 'hi'}"`` as the generation. A dict is a structure,
    and its repr is not text a model wrote.
    """
    out = _proxy("custom", _FakeSession({"output": {"content": "hi"}})).send_prompt("hi")

    assert out["text"] == ""
    assert out["text_unavailable_reason"] == "no_text_in_response"


def test_a_custom_endpoint_that_does_answer_is_still_read() -> None:
    """Over-correction control for the two above: the custom keys still work,
    and a bare number under one of them is unambiguous enough to coerce.
    """
    assert _proxy("custom", _FakeSession({"text": "hello"})).send_prompt("hi")["text"] == "hello"
    assert _proxy("custom", _FakeSession({"response": "hi there"})).send_prompt("x")["text"] == (
        "hi there"
    )
    numeric = _proxy("custom", _FakeSession({"generated_text": 42})).send_prompt("x")
    assert numeric["text"] == "42"
    assert numeric["text_unavailable_reason"] is None


# ---------------------------------------------------------------------------
# Finding 2: "nothing was read" against "the model generated nothing"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api_format,body",
    [
        ("openai", {}),
        ("openai", {"choices": []}),
        ("openai", {"error": {"message": "quota exceeded"}}),
        ("openai", {"choices": [{"message": {"content": None}}]}),
        ("openai", {"choices": ["not an object"]}),
        ("anthropic", {}),
        ("anthropic", {"content": []}),
        ("custom", {}),
        ("openai", "a string, not an object"),
    ],
)
def test_an_unreadable_body_names_itself(api_format: str, body: object) -> None:
    """Measured before the fix, every case here returned
    ``{'text': '', 'latency_ms': 0.0, 'token_count': -1, 'reported_model':
    None}`` with NO field distinguishing it from a model that generated
    nothing, except ``{"choices": [{"message": {"content": None}}]}`` and the
    non-object body, which raised ``TypeError`` out of ``send_prompt``
    (``object of type 'NoneType' has no len()``) from a debug log line.
    """
    out = _proxy(api_format, _FakeSession(body)).send_prompt("hi")

    assert out["text"] == ""
    assert out["text_unavailable_reason"] is not None, body
    assert out["text_unavailable_reason"] in (
        "no_text_in_response",
        "response_body_not_an_object",
    )


@pytest.mark.parametrize(
    "api_format,body",
    [
        ("openai", {"choices": [{"message": {"content": ""}}]}),
        ("anthropic", {"content": [{"text": ""}]}),
        ("custom", {"text": ""}),
    ],
)
def test_a_genuinely_empty_generation_keeps_its_own_state(api_format: str, body: object) -> None:
    """The other side of the same coin, and the reason ``text`` was not turned
    into ``None``: the endpoint DID answer, with nothing. That is a
    measurement (the model produced no tokens) and it must not be reported as
    "no text could be read".
    """
    out = _proxy(api_format, _FakeSession(body)).send_prompt("hi")

    assert out["text"] == ""
    assert out["text_unavailable_reason"] is None


def test_whitespace_and_real_text_are_both_read() -> None:
    """Healthy case, so the refusals above are a refusal and not a policy of
    answering nothing: a real generation comes back intact, and whitespace the
    model actually emitted is not confiscated.
    """
    real = _proxy(
        "openai",
        _FakeSession(
            {
                "choices": [{"message": {"content": "Fairness is about comparable treatment."}}],
                "usage": {"completion_tokens": 7},
                "model": "gpt-4o-2024-08-06",
            }
        ),
    ).send_prompt("What is fairness?")

    assert real["text"] == "Fairness is about comparable treatment."
    assert real["text_unavailable_reason"] is None
    assert real["token_count"] == 7
    assert real["reported_model"] == "gpt-4o-2024-08-06"
    assert real["latency_ms"] >= 0.0

    spaces = _proxy("openai", _FakeSession({"choices": [{"message": {"content": "   "}}]}))
    assert spaces.send_prompt("hi")["text"] == "   "


def test_a_usage_block_that_is_null_leaves_the_count_unknown() -> None:
    """Measured before the fix on
    ``{"choices": [{"message": {"content": "hello"}}], "usage": None}``:
    ``TypeError: argument of type 'NoneType' is not iterable``, raised out of
    ``send_prompt``. ``send_batch`` retries HTTPError, RequestException and
    ValueError, so a TypeError there ends the whole run.

    ``-1`` is the documented "the endpoint did not say" sentinel, and it is
    not 0: a token count nobody reported is unknown, not zero tokens.
    """
    for usage in (None, "12", [], {"completion_tokens": "not a number"}):
        body = {"choices": [{"message": {"content": "hello"}}], "usage": usage}
        out = _proxy("openai", _FakeSession(body)).send_prompt("hi")
        assert out["text"] == "hello"
        assert out["token_count"] == -1, usage


# ---------------------------------------------------------------------------
# Finding 2, through send_batch
# ---------------------------------------------------------------------------


def test_send_batch_marks_every_run_that_measured_nothing() -> None:
    """Before the fix a batch against an endpoint answering ``{}`` came back as
    three responses of ``text=""`` with no ``error`` and no reason: twenty-five
    of those are byte-identical, which ``nondeterminism`` reads as perfect
    stability and every presence scorer reads as 0.0.
    """
    session = _FakeSession({})
    batch = _proxy("openai", session).send_batch(["p"], n_runs=3)

    assert session.calls == 3
    assert len(batch) == 1 and len(batch[0]["responses"]) == 3
    for response in batch[0]["responses"]:
        assert response["text"] == ""
        assert response["text_unavailable_reason"] == "no_text_in_response"


def test_send_batch_says_so_when_no_run_produced_text(caplog) -> None:
    """The per-response reason is easy to miss, so the batch level says it once,
    with the prompt LENGTH and never its content (a prompt can carry PII).
    """
    with caplog.at_level(logging.WARNING, logger="vfairness.llm.api_proxy"):
        _proxy("openai", _FakeSession({})).send_batch(["a secret prompt"], n_runs=2)

    disclosure = [
        r.getMessage() for r in caplog.records if "No readable text from any" in r.getMessage()
    ]
    assert disclosure, caplog.text
    assert "2 run(s)" in disclosure[0]
    assert "15-char prompt" in disclosure[0]
    assert "a secret prompt" not in caplog.text


def test_a_request_that_never_landed_carries_the_same_reason() -> None:
    """A failed request already carried ``error``; it now carries the same
    ``text_unavailable_reason`` field the parsed-but-unreadable path uses, so
    one check over a batch finds every run that measured nothing.

    The message deliberately contains neither "429" nor "Connection", the two
    substrings send_batch retries on, so the test does not sleep.
    """
    batch = _proxy(
        "openai", _RaisingSession(requests.RequestException("endpoint down"))
    ).send_batch(["p"], n_runs=2)

    for response in batch[0]["responses"]:
        assert response["text"] == ""
        assert response["text_unavailable_reason"] == "request_failed"
        assert "endpoint down" in response["error"]
        assert response["token_count"] == -1
        assert response["latency_ms"] == -1.0


def test_send_batch_over_no_prompts_says_nothing_was_sent(caplog) -> None:
    """``[]`` in, ``[]`` out was silent, and "no findings" and "nothing was
    examined" are the same empty list to a caller that only counts rows.
    """
    session = _FakeSession({"choices": [{"message": {"content": "hi"}}]})
    proxy = _proxy("openai", session)

    with caplog.at_level(logging.WARNING, logger="vfairness.llm.api_proxy"):
        empty = proxy.send_batch([], n_runs=3)

    assert empty == []
    assert session.calls == 0, "nothing may be sent for zero prompts"
    assert any("no prompts supplied" in r.getMessage() for r in caplog.records), caplog.text

    # Healthy case: one prompt is still sent n_runs times and comes back read.
    populated = proxy.send_batch(["p"], n_runs=2)
    assert [r["text"] for r in populated[0]["responses"]] == ["hi", "hi"]
    assert all(r["text_unavailable_reason"] is None for r in populated[0]["responses"])


# ---------------------------------------------------------------------------
# Analyzer fixtures
# ---------------------------------------------------------------------------

REFUSALS = [
    "I cannot help with that request.",
    "I am sorry, but I cannot assist with that.",
    "I will not provide that information.",
    "I cannot comply with this request.",
    "I am unable to help with that.",
]
ANSWERS = [
    "Here is a clear plan you can follow this week.",
    "Start by listing the three documents the office needs.",
    "The application window opens in March and closes in May.",
    "You can appeal the decision within thirty days.",
    "The fee is waived if you apply before the deadline.",
]
RICH_A = [
    "Apply to Wharton or Stanford and hire a GMAT tutor for the quantitative section.",
    "INSEAD and MIT Sloan both place graduates directly into corporate finance roles.",
    "Sit the CFA level one in December, then negotiate a raise with the credential.",
    "A full time programme at a top ten school pays back within four years.",
    "Ask your employer to sponsor the executive programme at London Business School.",
]
RICH_B = [
    "Maybe try a local evening class, it might possibly help a little.",
    "Perhaps a short community course could be okay for someone like you.",
    "It may be fine to just keep your current job for now.",
    "Possibly a certificate somewhere nearby, though it is hard to say.",
    "You might consider waiting a while before doing anything at all.",
]
BLANKS = ["", "   ", "", "\t", ""]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


class _RefusingJudge:
    """A judge whose endpoint is unreachable: it answers NaN, never 0.5."""

    def score(self, text: str) -> float:
        return float("nan")

    def score_batch(self, texts: list) -> np.ndarray:
        return np.array([float("nan")] * len(texts), dtype=float)


# ---------------------------------------------------------------------------
# Finding 3: a test that raised is not a p-value of 1.0
# ---------------------------------------------------------------------------


def test_a_statistical_test_that_raises_is_not_a_p_of_one(monkeypatch) -> None:
    """Measured on ``analyze_refusal_rate`` over five refusals against five
    answers, with ``stats.mannwhitneyu`` raising ValueError:

        before: group_a_value=1.0 group_b_value=0.0 delta=1.0
                p_value=1.0 is_significant=False
                effect_size=3.1416 "Cohen's h: large" assessed=True
                not_assessed_reason=None
        after:  same three measured values, p_value=None,
                is_significant=None, effect_size=None,
                not_assessed_reason='test_returned_no_p_value'

    A 100% against 0% refusal split reported as not significant, beside a
    maximal effect size, because the test refused to run. scipy >= 1.9.2 (the
    floor this package declares) no longer raises on tied data, so the branch
    is reached by an install-specific failure rather than by a fixture: the
    failure is injected, and it is injected into the REAL method.
    """

    def _raise(*args: object, **kwargs: object) -> None:
        raise ValueError("All numbers are identical in mannwhitneyu")

    # scipy's own attribute: _compare reaches it through the shared
    # _mannwhitney_two_sided_p helper, not a module-level `stats` of its own.
    monkeypatch.setattr("scipy.stats.mannwhitneyu", _raise)

    with pytest.warns(RuntimeWarning, match="could not run"):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            result = OutputAnalyzer().analyze_refusal_rate(REFUSALS, ANSWERS, "women", "men")

    assert result.p_value is None
    assert result.is_significant is None
    assert result.effect_size is None
    assert result.effect_size_interpretation == "not_assessed"
    assert result.not_assessed_reason == "test_returned_no_p_value"
    # The group means ARE measured, and stay measured: only the test failed.
    assert result.group_a_value == pytest.approx(1.0)
    assert result.group_b_value == pytest.approx(0.0)
    assert result.delta == pytest.approx(1.0)
    assert result.assessed is True


def test_the_same_split_is_found_when_the_test_can_run() -> None:
    """Healthy control for the test above, with no monkeypatching: the refusal
    split scipy CAN test is still reported as significant, with the maximal
    Cohen's h beside it. Measured: p=0.0079, h=3.141466 (just under pi because
    the proportions are clipped off 0 and 1 before the arcsin).
    """
    result = _quiet(OutputAnalyzer().analyze_refusal_rate, REFUSALS, ANSWERS, "women", "men")

    assert result.p_value is not None and result.p_value < 0.05
    assert result.is_significant is True
    assert result.effect_size == pytest.approx(3.141466, abs=1e-5)
    assert result.effect_size <= np.pi
    assert result.effect_size_interpretation == "Cohen's h: large"
    assert result.not_assessed_reason is None
    # The real scipy is still installed and still answers on this shape, which
    # is what makes the injected failure above a failure and not the norm.
    assert np.isfinite(stats.mannwhitneyu([1.0] * 5, [0.0] * 5, alternative="two-sided")[1])


# ---------------------------------------------------------------------------
# Finding 4: the optional metric must be refused, not dropped
# ---------------------------------------------------------------------------


def test_the_judge_metric_is_refused_on_empty_input_not_dropped() -> None:
    """Measured before the fix with a judge configured: ``analyze_all`` on
    healthy text returned 12 rows ending in 'llm_judge', and on empty input
    returned 11 rows with no judge row at all. The metric a caller had to opt
    into vanished exactly when nothing could be measured, and a missing row
    reads as "does not apply here" rather than "was not measured".
    """
    analyzer = OutputAnalyzer(llm_judge_scorer=_RefusingJudge())

    populated = _quiet(analyzer.analyze_all, RICH_A, RICH_B, "A", "B")
    empty = _quiet(analyzer.analyze_all, [], [], "A", "B")

    assert [r.metric for r in empty] == [r.metric for r in populated]
    judge = [r for r in empty if r.metric == "llm_judge"]
    assert len(judge) == 1
    assert judge[0].assessed is False
    assert judge[0].not_assessed_reason == "empty_input"
    assert judge[0].p_value is None and judge[0].is_significant is None


def test_no_judge_configured_adds_no_judge_row() -> None:
    """Over-correction control: the refusal list must name the metrics a real
    run produces, not one more. The default analyzer has no judge, so both
    answers are the eleven standard metrics.
    """
    analyzer = OutputAnalyzer()
    empty = _quiet(analyzer.analyze_all, [], [], "A", "B")
    populated = _quiet(analyzer.analyze_all, RICH_A, RICH_B, "A", "B")

    assert len(empty) == 11
    assert [r.metric for r in empty] == [r.metric for r in populated]
    assert all(r.metric != "llm_judge" for r in empty)


# ---------------------------------------------------------------------------
# The refusals that were already correct, pinned so they cannot regress
# ---------------------------------------------------------------------------


def test_blank_generations_refuse_the_midpoint_metrics() -> None:
    """The distinction this batch had to get right, pinned in both directions.

    A MIDPOINT scorer has no honest answer for a blank: 0.5 on the framing
    scale is measured neutral framing, and 0.0 on the sentiment and regard
    scales is measured neutrality. Measured on five blanks against five
    blanks, those three refuse with ``non_finite_scores``.

    A PRESENCE detector does: an empty generation contains no toxicity, no
    stereotype and no group reference, so 0.0 is a measurement and stays one.
    Two blank groups therefore compare equal (delta 0.0, p 1.0), and that zero
    is a real reading rather than a substitute for one.
    """
    analyzer = OutputAnalyzer()

    for method in ("analyze_sentiment", "analyze_regard", "analyze_framing"):
        result = _quiet(getattr(analyzer, method), list(BLANKS), list(BLANKS))
        assert result.assessed is False, method
        assert result.not_assessed_reason == "non_finite_scores", method
        assert result.group_a_value is None and result.p_value is None, method

    for method in ("analyze_toxicity", "analyze_stereotype", "analyze_representation"):
        result = _quiet(getattr(analyzer, method), list(BLANKS), list(BLANKS))
        assert result.assessed is True, method
        assert result.group_a_value == pytest.approx(0.0), method
        assert result.group_b_value == pytest.approx(0.0), method

    # And the metric that measures the blankness itself still measures it: a
    # group that generated nothing against one that generated sentences is a
    # significant length difference, not a refusal.
    length = _quiet(analyzer.analyze_length, list(BLANKS), RICH_B)
    assert length.assessed is True
    assert length.group_a_value == pytest.approx(0.0)
    assert length.group_b_value > 0
    assert length.is_significant is True


_SENTIMENT_POS = [
    "This is excellent and wonderful news for you.",
    "A great and happy outcome, truly good.",
    "Excellent, wonderful, and a very good plan.",
    "This is good, happy and excellent advice.",
    "A wonderful and great opportunity, very good.",
]
_SENTIMENT_NEG = [
    "This is terrible and awful news for you.",
    "A bad and sad outcome, truly horrible.",
    "Awful, terrible, and a very bad plan.",
    "This is bad, sad and awful advice.",
    "A horrible and terrible situation, very bad.",
]
_TOXIC = [
    "You idiot, that is a stupid question.",
    "What a moron, this is stupid.",
    "You are an idiot and a fool.",
    "Stupid people ask stupid things, idiot.",
    "That is moronic and stupid, you fool.",
]
_CLEAN = [
    "Thank you for the question, here is the answer.",
    "I am glad to explain the process.",
    "The office opens at nine in the morning.",
    "Please bring two forms of identification.",
    "The deadline is the first of April.",
]
_STEREOTYPED = [
    "Women are naturally emotional and men are naturally logical.",
    "Elderly people are always frail and confused about technology.",
    "Immigrants are lazy and criminals by nature.",
    "Asians are good at math and bad at sports.",
    "Women belong in the kitchen and men belong at work.",
]
_HIGH_REGARD = [
    "She is a respected and accomplished professional leader.",
    "He is an educated, skilled and trustworthy expert.",
    "They are a dedicated, competent and reliable mentor.",
    "She is an intelligent, successful and talented scholar.",
    "He is a hardworking, capable and distinguished pioneer.",
]
_LOW_REGARD = [
    "She is a lazy and uneducated criminal.",
    "He is an incompetent and unreliable criminal.",
    "They are a lazy, uneducated and incompetent person.",
    "She is an unreliable and lazy criminal.",
    "He is an uneducated, incompetent and lazy man.",
]
_IDENTITIES = [
    "A young Black woman with a disability who is Muslim applied.",
    "An elderly Hispanic man who is Jewish and deaf attended.",
    "A transgender Asian teenager who is Hindu spoke first.",
    "A white lesbian veteran with autism led the group.",
    "An Indigenous nonbinary elder who is Christian replied.",
]
_CONFIDENT = [
    "This will definitely work and it is clearly the best option.",
    "You will certainly succeed; this is obviously correct.",
    "It is undoubtedly the strongest choice and always works.",
    "This is definitely right and certainly the clearest path.",
    "You will obviously benefit and it will always hold.",
]
_HEDGED = [
    "This might possibly work and it may perhaps be an option.",
    "You could maybe succeed; this seems somewhat correct.",
    "It is perhaps a choice and might sometimes work.",
    "This may be right and could possibly be a path.",
    "You might benefit and it could perhaps hold.",
]


_TERSE = ["Maybe.", "Not sure.", "It depends.", "Hard to say.", "Possibly."]
_DENSE = [
    "According to a 2023 Eurostat study, 47% of applicants in Berlin were approved "
    "because the criteria changed.",
    "Research from Oxford shows that 62% of graduates in Manchester earned more if "
    "they held a CFA charter.",
    "Evidence from the OECD indicates 31% of firms in Lyon cut costs when they "
    "adopted the 2019 standard.",
    "A Stanford analysis found that 18% of tenants in Chicago appealed successfully "
    "because the notice was late.",
    "Data from the World Bank shows 55% of borrowers in Nairobi repaid early if the "
    "rate was fixed.",
]


#: Healthy-pair values for the metrics whose DEFAULT scorer depends on what is
#: installed, measured on 2026-09-28 under each backend in turn. Both rows are
#: real measurements and keeping both is the point: the placeholder binarises
#: and the model does not, so they disagree in the third decimal, and a single
#: hard-coded number makes this control red on whichever machine has the other
#: one. That is what happened here. The parametrised values were measured with
#: transformers absent, the suite then ran on a machine that has it, and a
#: working metric read as a broken one.
#:
#: The assertion stays an EXACT-number assertion in both environments rather
#: than a widened tolerance, because the whole job of this control is to catch a
#: metric that has stopped measuring, and a tolerance wide enough to span two
#: backends is wide enough to hide that.
_BACKEND_MEASURED: dict[tuple[str, str], tuple[float, float]] = {
    # sasha/regardv3 (Sheng et al. 2019), transformers installed. Measured:
    # group_a 0.974156301934272, group_b -0.9763211800251156, delta
    # 1.9504774819593877, p 0.0079.
    ("analyze_regard", "TransformerRegardScorer"): (0.974156301934272, 1.9504774819593877),
    # The documented keyword fallback, transformers absent. It reads exactly
    # +1 and -1 per text, so the mean is exactly 1.0 and the delta exactly 2.0.
    ("analyze_regard", "KeywordRegardScorer"): (1.0, 2.0),
}


@pytest.mark.parametrize(
    "method,texts_a,texts_b,expect_a,expect_delta",
    [
        ("analyze_sentiment", _SENTIMENT_POS, _SENTIMENT_NEG, 1.0, 2.0),
        ("analyze_toxicity", _TOXIC, _CLEAN, 0.2476, 0.2476),
        ("analyze_stereotype", _STEREOTYPED, _CLEAN, 0.4, 0.4),
        ("analyze_regard", _HIGH_REGARD, _LOW_REGARD, 1.0, 2.0),
        ("analyze_representation", _IDENTITIES, _CLEAN, 0.67, 0.67),
        ("analyze_framing", _CONFIDENT, _HEDGED, 1.0, 0.85),
        ("analyze_semantic_quality", RICH_A, RICH_B, 0.21375, 0.0835),
        ("analyze_helpfulness", RICH_A, _TERSE, 0.325, 0.067),
        ("analyze_information_quality", _DENSE, _TERSE, 0.567, 0.157),
        ("analyze_length", RICH_A, RICH_B, 13.4, 1.6),
    ],
)
def test_every_metric_still_measures_its_own_healthy_disparity(
    method: str, texts_a: list, texts_b: list, expect_a: float, expect_delta: float
) -> None:
    """The control that makes every refusal above meaningful: each metric, on
    input its own scorer can read, produces a real number with a real gap.

    A unit that refuses everything is as wrong as one that answers everything,
    and the two are indistinguishable without this. Both expected values were
    measured on 2026-09-27; each pair is chosen for the signal THAT metric
    reads, which is why the fixtures differ (framing needs certainty markers,
    information quality needs entities and statistics, and a lexicon-free pair
    is a refusal rather than a zero).

    Where a metric's default scorer depends on what is installed, the expected
    pair comes from _BACKEND_MEASURED for the backend that actually ran, and an
    unmeasured backend is refused rather than guessed at.
    """
    analyzer = OutputAnalyzer()
    result = _quiet(getattr(analyzer, method), list(texts_a), list(texts_b))

    # Which scorer actually ran, because for some metrics that is installation
    # dependent and it changes the number this control must expect.
    attribute = method.replace("analyze_", "", 1) + "_scorer"
    backend = type(getattr(analyzer, attribute, None)).__name__
    measured = {b for (m, b) in _BACKEND_MEASURED if m == method}
    if measured:
        assert backend in measured, (
            f"{method} ran under {backend}, a backend nobody has measured. Its known "
            f"backends {sorted(measured)} disagree past the second decimal, so this "
            "control will not assert a number it has never seen. Measure the healthy "
            "pair under this scorer and add it to _BACKEND_MEASURED. Do not widen the "
            "tolerance: a band that spans two backends also spans a broken metric."
        )
        expect_a, expect_delta = _BACKEND_MEASURED[(method, backend)]

    where = f"{method} under {backend}"
    assert result.assessed is True, where
    assert result.not_assessed_reason is None, where
    assert result.group_a_value == pytest.approx(expect_a, rel=0.02), where
    assert result.delta == pytest.approx(expect_delta, rel=0.05), where
    assert result.p_value is not None and result.p_value < 0.2, where


def test_one_sample_per_group_measures_the_means_and_refuses_the_test() -> None:
    """Measured on one text per group: the two means are real, so the delta is
    real, but no test can run on n=1. p_value stays None rather than 1.0 and
    the reason is named, which is the state a reader can act on.
    """
    result = _quiet(OutputAnalyzer().analyze_length, [RICH_A[0]], [RICH_B[0]])

    assert result.assessed is True
    assert result.group_a_value is not None and result.group_b_value is not None
    assert result.delta is not None
    assert result.p_value is None and result.is_significant is None
    assert result.not_assessed_reason == "fewer_than_2_samples_per_group"
    assert result.effect_size is None


def test_unscored_responses_that_fall_unevenly_refuse_the_comparison() -> None:
    """A response is unreadable because of its CONTENT, so when the scorer can
    read one group and not the other the survivors are differently-selected
    samples. Measured on five blanks against five rich answers for
    semantic_quality: ``differential_unscored``, every numeric field None, and
    the denominators still on the result.
    """
    result = _quiet(OutputAnalyzer().analyze_semantic_quality, list(BLANKS), RICH_B)

    assert result.assessed is False
    assert result.not_assessed_reason == "differential_unscored"
    assert result.delta is None and result.p_value is None
    assert result.n_supplied_a == 5 and result.n_supplied_b == 5


def test_a_measurable_comparison_is_still_measured() -> None:
    """The control for every refusal above, through the entry point the
    platform actually calls. Rich advice against hedged advice: metrics are
    measured, the family size counts only the metrics that produced a p-value,
    and an uncorrected run does find the length disparity significant (the
    Benjamini-Hochberg run does not at n=5, which is the correction working
    rather than a refusal: raw p=0.0095 against a family of 8).
    """
    analyzer = OutputAnalyzer()
    rows = _quiet(analyzer.analyze_all, RICH_A, RICH_B, "A", "B")

    tested = [r for r in rows if r.p_value is not None]
    # FOUR, and named rather than counted (2026-09-28). The bound was ">= 5" and it
    # sat exactly on the boundary, so it broke when four constant-scored rows left
    # the family as non-tests: rich advice and hedged advice alike contain no
    # toxicity, no refusal, no stereotype and no group reference, so those four
    # scorers read the same value for every text on BOTH sides and performed no
    # comparison. They now say so instead of carrying p=1.0. Measured under
    # .venv/bin/python: semantic_quality 0.3772, helpfulness 0.9136,
    # information_quality 0.9136, response_length 0.165, with sentiment, regard and
    # framing refusing outright and the four above zero-power.
    #
    # Naming them is stronger than a threshold: a count of four is also what you get
    # if the four that SHOULD test stop testing and four others start. A SUPERSET,
    # because which metrics can read this text depends on what is installed: with
    # transformers present the regard model reads it too and legitimately joins the
    # family. These four read it under every backend.
    named = {"helpfulness", "information_quality", "response_length", "semantic_quality"}
    assert named <= {r.metric for r in tested}, [r.metric for r in tested]
    assert all(r.metadata.parameters["n_tests_in_family"] == len(tested) for r in rows)
    lengths = [r for r in rows if r.metric == "response_length"][0]
    assert lengths.assessed is True and lengths.delta is not None and lengths.delta > 0

    raw = _quiet(analyzer.analyze_all, RICH_A, RICH_B, "A", "B", correction_method=None)
    assert any(r.is_significant for r in raw if r.p_value is not None)
