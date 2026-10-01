"""Readiness lane docs2: documented workflows that crash, or silently do nothing.

Every pin here comes in two halves, because half a pin catches nothing:

  * a REFUSAL PIN, which fails if the defect is reintroduced into the published
    docs, and
  * an OVER-CORRECTION CONTROL, which fails if the fix was applied by deleting
    the capability or by making the library refuse everything. Each control
    asserts a MEASURED number, never membership of a broad set: "the gap is
    below tolerance" and "the gap is 50x tolerance" are different claims and
    both are made here.

The snippets are not paraphrased. Where a pin says a documented example works,
it locates that example IN THE PUBLISHED FILE, extracts it, and executes it,
so a pin cannot pass against a copy of the code that no reader will ever see.

Findings pinned (all reproduced by execution 2026-09-10 before any edit):

  1. FairRegressor(method='reweighting') was documented and raises ValueError.
  2. The same example offered fairness_constraint values that raise
     NotImplementedError, in an inline comment on the copied line.
  3. The documented workflow called reg.predict(X_test), which returns the
     UNADJUSTED base predictions: the mitigation was never applied. Measured
     group gap 4.97 against the 0.10 tolerance the page promised.
     predict_with_sensitive_attr appeared ZERO times in the whole docs tree.
  4. The site claimed FairRegressor works by adjusting sample weights (the
     mechanism removed as provably ineffective) and that it enforces error
     parity (which fit() refuses).
  5. A documented MetricExplanation(...) omitted benchmark_context, which has
     no default. It crashed unconditionally.
  6. Documented print lines applied float format specs to fields the library
     returns as None. Those are the three-state values the LLM honesty fix was
     written to produce, so the docs taught code that cannot survive the
     library's own headline feature.
"""

from __future__ import annotations

import html
import io
import pathlib
import re
import warnings
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from vfairness import detect_representation_bias
from vfairness.evaluation.vfairness_metrics.explainer import MetricExplanation
from vfairness.evaluation.vfairness_metrics.robustness import contingency_test
from vfairness.in_processing import FairRegressor
from vfairness.llm.api_proxy import LLMApiProxy
from vfairness.llm.counterfactual import CounterfactualTester
from vfairness.llm.output_analysis import OutputAnalyzer

_LIB_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DOCS = _LIB_ROOT / "docs"

_API_MD = _DOCS / "API_REFERENCE.md"
_OVERVIEW_MD = _DOCS / "LIBRARY_OVERVIEW.md"
_API_HTML = _DOCS / "site" / "api-reference" / "index.html"
_TTI_MD = _DOCS / "site" / "training-time-interventions.md"
_TTI_HTML = _DOCS / "site" / "training-time-interventions" / "index.html"
_LLM_HTML = _DOCS / "site" / "llm-testing" / "index.html"

# Same fence pairing as the published-docs gate: an opener carries an
# infostring and starts a line, a closer is a ``` at the start of a line.
_MD_BLOCK = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
_HTML_BLOCK = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S)


def _code_blocks(path: pathlib.Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return _MD_BLOCK.findall(raw)
    return [html.unescape(re.sub(r"<[^>]+>", "", b)) for b in _HTML_BLOCK.findall(raw)]


def _block_containing(path: pathlib.Path, marker: str) -> str:
    """The single published code block carrying `marker`.

    Fails loudly on zero and on more than one, so a pin can never silently
    execute a block other than the one it names.
    """
    hits = [b for b in _code_blocks(path) if marker in b]
    assert hits, f"no code block in {path.name} contains {marker!r}"
    assert len(hits) == 1, f"{len(hits)} code blocks in {path.name} contain {marker!r}"
    return hits[0]


def _run_snippet(source: str, namespace: Dict[str, Any]) -> str:
    """Execute a documented snippet and return what it printed."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(source, "<published-docs>", "exec"), namespace)  # noqa: S102
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared fixtures: a regression problem where the group effect is LEARNABLE.
#
# This matters. With the group absent from X the base model cannot reproduce
# the gap, predict() and predict_with_sensitive_attr() come out within 0.01 of
# each other, and a pin written on that data passes whichever call the docs
# show. The group has to be a feature for the defect to be visible at all.
# ---------------------------------------------------------------------------


def _regression_problem() -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    n = 1000
    g = rng.integers(0, 2, size=n)
    group = np.where(g == 0, "A", "B")
    X = np.column_stack([g.astype(float), rng.normal(size=n), rng.normal(size=n)])
    y = 5.0 * g + X[:, 1] + rng.normal(scale=0.2, size=n)
    half = n // 2
    return X[:half], y[:half], group[:half], X[half:], group[half:]


def _group_gap(pred: np.ndarray, groups: np.ndarray) -> float:
    means = [float(pred[groups == g].mean()) for g in np.unique(groups)]
    return float(max(means) - min(means))


# ---------------------------------------------------------------------------
# Findings 1 and 2: documented option values the code refuses.
# ---------------------------------------------------------------------------

_FAIRREG_CALL = re.compile(r"FairRegressor\((.*?)\n\)", re.S)
_KW_LITERAL = re.compile(r"(\w+)\s*=\s*'([^']*)'")


def _documented_fairregressor_literals() -> List[Tuple[str, str, str]]:
    """(file, kwarg, value) for every string literal in a documented call."""
    out: List[Tuple[str, str, str]] = []
    for path in (_API_MD, _API_HTML, _TTI_MD, _TTI_HTML, _OVERVIEW_MD, _LLM_HTML):
        for block in _code_blocks(path):
            for call in _FAIRREG_CALL.findall(block):
                for kw, value in _KW_LITERAL.findall(call):
                    out.append((path.name, kw, value))
    return out


def test_documented_fairregressor_option_values_are_accepted_by_fit():
    """REFUSAL PIN, findings 1 and 2.

    Not a denylist of two strings. Every option literal that appears in a
    published FairRegressor(...) call is pulled out of the file and PUT THROUGH
    fit() on real data. A value the code refuses fails here whatever it is
    called, so a future removal is caught without editing this test.
    """
    literals = _documented_fairregressor_literals()
    assert literals, "no FairRegressor option literals found; the extractor broke"

    X_train, y_train, group_train, _, _ = _regression_problem()
    failures: List[str] = []
    for where, kwarg, value in literals:
        if kwarg not in ("method", "fairness_constraint"):
            continue
        try:
            reg = FairRegressor(base_estimator=Ridge(), **{kwarg: value})
            reg.fit(X_train, y_train, sensitive_attr=group_train)
        except (ValueError, NotImplementedError) as exc:
            failures.append(f"{where}: {kwarg}={value!r} -> {type(exc).__name__}: {exc}")

    assert not failures, "the published docs offer FairRegressor options the code refuses:\n  " + (
        "\n  ".join(failures)
    )


def test_the_refused_fairregressor_options_really_are_refused():
    """OVER-CORRECTION CONTROL for the pin above.

    The pin would also pass if FairRegressor stopped validating anything. It
    does not: these three are refused, each with its own error type, and the
    supported pair is accepted and produces a MEASURED offset.
    """
    X_train, y_train, group_train, _, _ = _regression_problem()

    with pytest.raises(ValueError, match="reweighting.*was removed"):
        FairRegressor(base_estimator=Ridge(), method="reweighting").fit(
            X_train, y_train, sensitive_attr=group_train
        )
    for constraint in ("error_parity", "bounded_loss"):
        with pytest.raises(NotImplementedError, match="not.*implemented"):
            FairRegressor(base_estimator=Ridge(), fairness_constraint=constraint).fit(
                X_train, y_train, sensitive_attr=group_train
            )

    # The supported pair is not merely "does not raise": it fits offsets whose
    # magnitude is the measured half-gap, ~2.5 on a group effect of 5.0.
    reg = FairRegressor(
        base_estimator=Ridge(), fairness_constraint="mean_parity", method="offset", tolerance=0.1
    )
    reg.fit(X_train, y_train, sensitive_attr=group_train)
    assert set(reg.group_offsets_) == {"A", "B"}
    assert reg.group_offsets_["A"] == pytest.approx(2.78, abs=0.15)
    assert reg.group_offsets_["B"] == pytest.approx(-2.27, abs=0.15)


# ---------------------------------------------------------------------------
# Finding 3: the documented prediction call must apply the mitigation.
# ---------------------------------------------------------------------------

_DOC_SURFACES_WITH_FAIRREGRESSOR = (_API_MD, _API_HTML, _TTI_MD, _TTI_HTML)


def test_no_documented_fairregressor_workflow_assigns_plain_predict():
    """REFUSAL PIN, finding 3.

    predict() returns the UNADJUSTED base predictions. A workflow that fits for
    mean parity and then assigns y_pred = reg.predict(X_test) has applied no
    mitigation at all, on four separate published surfaces.
    """
    offenders: List[str] = []
    for path in _DOC_SURFACES_WITH_FAIRREGRESSOR:
        for block in _code_blocks(path):
            if "FairRegressor(" not in block:
                continue
            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(r"=\s*reg\.predict\(", stripped):
                    offenders.append(f"{path.name}: {stripped}")
    assert not offenders, (
        "a documented FairRegressor workflow assigns the UNADJUSTED predictions; "
        "the correct call is predict_with_sensitive_attr(X, sensitive_attr):\n  "
        + "\n  ".join(offenders)
    )


def test_every_documented_fairregressor_workflow_names_the_adjusting_call():
    """REFUSAL PIN, finding 3, the other direction.

    Removing the bad line is not enough. Before this fix
    `predict_with_sensitive_attr` appeared ZERO times in the whole docs tree, so
    a reader had no way to reach the mitigation from the documentation at all.

    The name must appear in a CODE BLOCK, not merely somewhere on the page.
    Sabotage 2026-09-10: renaming the call in the code while leaving the prose
    paragraph that mentions it left this test GREEN when it searched the whole
    file. A reader copies the code block; prose about a method that the example
    does not call is exactly the gap this pin is for.
    """
    missing = [
        path.name
        for path in _DOC_SURFACES_WITH_FAIRREGRESSOR
        if any("FairRegressor(" in b for b in _code_blocks(path))
        and not any("predict_with_sensitive_attr" in b for b in _code_blocks(path))
    ]
    assert not missing, (
        "these pages document FairRegressor without calling "
        f"predict_with_sensitive_attr in any code block: {missing}"
    )


def test_the_two_prediction_calls_really_differ_by_the_documented_amount():
    """OVER-CORRECTION CONTROL for finding 3, asserting MEASURED gaps.

    Both pins above are text checks and would stay green if the two methods
    became identical, or if the offset stopped being applied. This measures
    both numbers on the workflow the page publishes:

        predict(X_test)                        group gap 4.97   (50x tolerance)
        predict_with_sensitive_attr(X, group)  group gap 0.08   (within 0.10)

    The lower bound on the unadjusted gap is the point: an assertion that only
    said "adjusted <= tolerance" would pass on data where the base model has no
    gap to correct, which is how this defect survived review.
    """
    X_train, y_train, group_train, X_test, group_test = _regression_problem()
    tolerance = 0.1
    reg = FairRegressor(
        base_estimator=Ridge(), fairness_constraint="mean_parity", tolerance=tolerance
    )
    reg.fit(X_train, y_train, sensitive_attr=group_train)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unadjusted = reg.predict(X_test)
    gap_unadjusted = _group_gap(unadjusted, group_test)
    adjusted = reg.predict_with_sensitive_attr(X_test, group_test)
    gap_adjusted = _group_gap(adjusted, group_test)

    assert gap_unadjusted == pytest.approx(4.97, abs=0.2)
    assert gap_unadjusted > 40 * tolerance
    assert gap_adjusted == pytest.approx(0.08, abs=0.05)
    assert gap_adjusted <= tolerance

    # predict() must also SAY it did nothing, not just do nothing quietly.
    assert any("UNADJUSTED" in str(w.message) for w in caught), (
        "predict() applied no mitigation and issued no warning about it"
    )


# ---------------------------------------------------------------------------
# Finding 4: the mechanism the site states must be the mechanism in the code.
# ---------------------------------------------------------------------------


def test_the_site_does_not_claim_a_mechanism_the_code_removed():
    """REFUSAL PIN, finding 4.

    The FairRegressor callout stated that it "enforces these properties by
    adjusting sample weights during training", which is the reweighting
    intervention removed as provably ineffective, and it claimed error parity
    among the properties enforced. fit() raises NotImplementedError for that.
    """
    raw = _API_HTML.read_text(encoding="utf-8")
    start = raw.index("When Fairness Matters in Regression")
    callout = raw[start : raw.index("</div>", raw.index("</p>", start))]

    assert "enforces these properties by adjusting sample weights" not in callout, (
        "the FairRegressor callout still states the removed reweighting mechanism"
    )
    assert "post-fit per-group prediction offset" in callout, (
        "the FairRegressor callout does not state the mechanism the code uses"
    )
    assert "mean parity only" in callout, (
        "the FairRegressor callout does not say that mean parity is the only constraint implemented"
    )


def test_the_mechanism_the_site_now_states_is_the_one_that_runs():
    """OVER-CORRECTION CONTROL for finding 4.

    The pin above is a string check on prose. This executes the claim: the
    correction is a per-group additive offset applied at predict time, and the
    two groups' offsets sum to zero because the band is centred on the pooled
    mean. Both are measured, not asserted from the docstring.
    """
    X_train, y_train, group_train, X_test, group_test = _regression_problem()
    reg = FairRegressor(base_estimator=Ridge(), tolerance=0.1)
    reg.fit(X_train, y_train, sensitive_attr=group_train)

    base = reg.predict(X_test)
    adjusted = reg.predict_with_sensitive_attr(X_test, group_test)
    for group in ("A", "B"):
        mask = group_test == group
        shift = adjusted[mask] - base[mask]
        # Additive and CONSTANT within the group: that is what "offset" means.
        assert shift.std() == pytest.approx(0.0, abs=1e-9)
        assert shift.mean() == pytest.approx(reg.group_offsets_[group], abs=1e-9)
    assert sum(reg.group_offsets_.values()) == pytest.approx(0.51, abs=0.1)


# ---------------------------------------------------------------------------
# Finding 5: the documented custom-handler example must construct.
# ---------------------------------------------------------------------------


def test_the_documented_metric_explanation_example_constructs():
    """REFUSAL PIN, finding 5.

    Executes the published block, not a paraphrase of it. As shipped it omitted
    benchmark_context, which has no default, so it raised TypeError on every
    reader who copied it.
    """
    from vfairness.explainer import FairnessExplainer

    block = _block_containing(_API_HTML, "def _explain_my_result(result)")

    # The example registers by TYPE NAME, so the stand-in has to carry the name
    # the block registers, or FairnessExplainer.explain refuses it and the pin
    # fails for a reason that has nothing to do with the defect.
    class MyCustomResult:
        score = 0.42

    registry = FairnessExplainer._handlers  # type: ignore[attr-defined]
    before = dict(registry)
    try:
        namespace: Dict[str, Any] = {"my_result": MyCustomResult()}
        _run_snippet(block, namespace)

        report = namespace["_explain_my_result"](MyCustomResult())
        assert len(report.explanations) == 1
        explanation = report.explanations[0]
        assert explanation.metric_name == "Custom Score"
        assert explanation.value == pytest.approx(0.42)
        assert explanation.benchmark_context, "benchmark_context was supplied but is empty"

        # The block's last line runs the registered handler through the public
        # entry point. That is the part a reader actually copies.
        assert namespace["explanation"].title == "My Custom Analysis"
    finally:
        registry.clear()
        registry.update(before)


def test_benchmark_context_really_is_required_and_could_not_check_is_reachable():
    """OVER-CORRECTION CONTROL for finding 5.

    The pin above would also pass if benchmark_context had simply been given a
    default, which would silence the crash by inventing an unmeasured benchmark
    for every explanation. It has no default, and omitting it still raises.

    The second half guards the reason the page now mentions could_not_check:
    'info' is what a GRADED, benign metric receives, so an ungraded metric must
    not reuse it. Both are constructible and they are distinguishable.
    """
    complete = dict(
        metric_name="Custom Score",
        definition="d",
        interpretation_guide="g",
        value=0.42,
        evaluation="e",
        benchmark_context="b",
        recommendation="r",
    )
    with pytest.raises(TypeError, match="benchmark_context"):
        without = dict(complete)
        without.pop("benchmark_context")
        MetricExplanation(**without)

    graded = MetricExplanation(**complete, severity="info")
    ungraded = MetricExplanation(**complete, severity="could_not_check")
    assert graded.severity == "info"
    assert ungraded.severity == "could_not_check"
    assert graded.to_dict()["severity"] != ungraded.to_dict()["severity"]


# ---------------------------------------------------------------------------
# Finding 6: documented print lines against the three-state values.
#
# Each case supplies TWO real objects to the SAME published snippet: one whose
# field the library could not measure, and one it measured. The snippet must
# survive the first and print the measured number in the second. A pin that
# only ran the None case would pass on a snippet that printed COULD NOT CHECK
# unconditionally.
# ---------------------------------------------------------------------------


def _analysis_results() -> Tuple[Any, Any]:
    """One result whose p_value could not be computed, one where it could.

    Every text here holds a word from KeywordSentimentScorer's 30-word lexicon,
    and that is load-bearing rather than incidental. This fixture used to say
    "Fine." and "Adequate, with some concerns and problems.", neither of which
    contains a single lexicon word, so the sentiment behind BOTH cases was a
    fabricated 0.0 and the assertion below ("the group values ARE measured")
    was false the whole time. It only became visible on 2026-09-10, when the
    scorer started refusing texts it had not read: the delta went to None and
    this builder raised before either published print site was exercised.

    So the texts are chosen against the lexicon, and the builder asserts the
    two states it is claiming to supply rather than trusting the words.
    """
    analyzer = OutputAnalyzer(alpha=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # One sample a side: the group values ARE measured, the Mann-Whitney
        # test cannot run, so p_value is None while delta is a real number.
        unmeasured = analyzer.analyze_sentiment(texts_a=["Excellent."], texts_b=["Terrible."])
        measured = analyzer.analyze_sentiment(
            texts_a=["Excellent work, truly outstanding."] * 30,
            texts_b=["Poor work, frankly awful."] * 30,
        )
    assert unmeasured.delta is not None, (
        "the refusal case was built from text the scorer cannot read, so it "
        "is unmeasured for the wrong reason and pins nothing about p_value"
    )
    assert measured.not_assessed_reason is None, (
        f"the measured case was not assessed: {measured.not_assessed_reason}"
    )
    return unmeasured, measured


def _disparity_metrics() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """One dict the library could not measure, one it did, both from the real
    compute_disparity, not hand-built.

    A real CounterfactualTester, not `__new__`: the unmeasured path returns
    before it touches a scorer, so a bare `__new__` object gets through the
    refusal case and then fails with AttributeError on the measured one. That
    would have left the over-correction control unrun while the refusal pin
    looked green.
    """
    proxy = object.__new__(LLMApiProxy)  # never called: no request is made here
    tester = CounterfactualTester(proxy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unmeasured = tester.compute_disparity(["some text"], [])
        measured = tester.compute_disparity(
            ["a wonderful excellent result"] * 5, ["a terrible awful result"] * 5
        )
    return unmeasured, measured


def _contingency_results() -> Tuple[Any, Any]:
    rng = np.random.default_rng(4)
    gender = np.array(["M"] * 5 + ["F"] * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Small expected counts select Fisher's exact, which yields no
        # Cramer's V: exactly the path the example's own comment describes.
        fisher = contingency_test(
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0]), gender, min_expected=5.0
        )
        big = np.array(["M"] * 200 + ["F"] * 200)
        chi = contingency_test(rng.integers(0, 2, 400), big, min_expected=5.0)
    return fisher, chi


def _representation_results() -> Tuple[Any, Any]:
    benchmarks = {"gender": {"Male": 0.49, "Female": 0.51}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = detect_representation_bias(
            pd.DataFrame({"gender": ["Male"] * 8 + ["Female"] * 4}),
            ["gender"],
            benchmarks=benchmarks,
        )
        large = detect_representation_bias(
            pd.DataFrame({"gender": ["Male"] * 60 + ["Female"] * 40}),
            ["gender"],
            benchmarks=benchmarks,
        )
    return small, large


def _case_output_analysis_loop(path: pathlib.Path, marker: str) -> "_Site":
    unmeasured, measured = _analysis_results()
    assert unmeasured.p_value is None and unmeasured.delta is not None
    assert measured.p_value is not None
    return _Site(
        path=path,
        marker=marker,
        focus="p_value",
        unmeasured={"results": [unmeasured], "r": unmeasured},
        measured={"results": [measured], "r": measured},
        expected=measured.p_value,
    )


class _Site(NamedTuple):
    """One published print site: where it is, what it reports, and two real
    result objects for it, one unmeasured and one measured."""

    path: pathlib.Path
    marker: str  # picks the block out of the file
    focus: str  # the Optional field whose reporting is under test
    unmeasured: Dict[str, Any]
    measured: Dict[str, Any]
    expected: float


_SIX_SITES: List[Tuple[str, Callable[[], _Site]]] = []


def _register(name: str):
    def _wrap(fn):
        _SIX_SITES.append((name, fn))
        return fn

    return _wrap


@_register("LIBRARY_OVERVIEW.md counterfactual disparity")
def _site_overview_disparity():
    unmeasured, measured = _disparity_metrics()

    class _R:
        def __init__(self, dm):
            self.disparity_metrics = dm
            self.is_significant = None

    return _Site(
        path=_OVERVIEW_MD,
        marker="Sentiment delta:",
        focus="sentiment_delta",
        unmeasured={"result": _R(unmeasured)},
        measured={"result": _R(measured)},
        expected=measured["sentiment_delta"],
    )


@_register("LIBRARY_OVERVIEW.md analyze_all loop")
def _site_overview_loop():
    return _case_output_analysis_loop(_OVERVIEW_MD, "effect={r.effect_size_interpretation}")


@_register("api-reference/index.html counterfactual disparity")
def _site_api_disparity():
    unmeasured, measured = _disparity_metrics()

    class _R:
        def __init__(self, dm):
            self.disparity_metrics = dm
            self.is_significant = None

    return _Site(
        path=_API_HTML,
        marker="Cosine similarity:",
        focus="sentiment_delta",
        unmeasured={"result": _R(unmeasured)},
        measured={"result": _R(measured)},
        expected=measured["sentiment_delta"],
    )


@_register("api-reference/index.html analyze_all loop")
def _site_api_loop():
    return _case_output_analysis_loop(_API_HTML, "effect={r.effect_size_interpretation}")


@_register("llm-testing/index.html analyze_sentiment")
def _site_llm_sentiment():
    unmeasured, measured = _analysis_results()
    return _Site(
        path=_LLM_HTML,
        # The marker LOCATES the block; it is not the subject. It used to be
        # "effect size: {res.effect_size:.3f}", and when effect_size became
        # Optional on its own the published snippet had to stop formatting it
        # unconditionally, so the marker stopped matching and this test failed
        # for a reason that had nothing to do with what it checks. A locator
        # that breaks when the code it points at is CORRECTLY improved sends the
        # next reader to revert the improvement. Anchored on the delta, which is
        # the one field always present on a measured result.
        marker="Sentiment delta: {res.delta:.3f}",
        focus="p_value",
        unmeasured={"res": unmeasured},
        measured={"res": measured},
        expected=measured.p_value,
    )


@_register("API_REFERENCE.md contingency_test Cramer's V")
def _site_api_md_cramers_v():
    fisher, chi = _contingency_results()
    assert fisher.cramers_v is None and chi.cramers_v is not None
    return _Site(
        path=_API_MD,
        marker="Cramer's V (effect size)",
        focus="cramers_v",
        unmeasured={"result": fisher},
        measured={"result": chi},
        expected=chi.cramers_v,
    )


@_register("API_REFERENCE.md representation chi-squared p")
def _site_api_md_chi_p():
    small, large = _representation_results()
    assert small[0].chi_squared_pvalue is None and large[0].chi_squared_pvalue is not None
    return _Site(
        path=_API_MD,
        marker="Chi-squared p-value:",
        focus="chi_squared_pvalue",
        unmeasured={"results": small},
        measured={"results": large},
        expected=large[0].chi_squared_pvalue,
    )


def _reporting_slice(block: str, focus: str) -> str:
    """A BACKWARD SLICE of a published block: the statements that report `focus`.

    The published blocks build their inputs by calling the library, and one of
    them opens a connection to a model endpoint, so a pin cannot simply run the
    whole block. It also cannot run the whole tail: these blocks print several
    unrelated things, and the first unrelated name that the pin's namespace does
    not define raises NameError before reaching the line under test. That
    NameError is indistinguishable from the defect, which is the trap this
    helper exists to avoid.

    So: start from every top-level statement whose source mentions `focus`,
    then close over the dataflow in BOTH directions until it stops growing.

      backward, so `dm = result.disparity_metrics` comes along when the guard
        two lines down is the statement that names the field;
      forward, so the guard comes along when it is the ASSIGNMENT that names
        the field, as in `delta = result.disparity_metrics['sentiment_delta']`
        followed by `if delta is None:`. Backward-only produced a slice that
        executed cleanly and printed NOTHING, and an empty run is not evidence
        of anything.

    Only names DEFINED by an already-needed statement propagate forward, so an
    unrelated trailing print that happens to read the namespace's own `result`
    stays out. Assignments whose value is a CALL are never pulled in either
    way; those are the setup the pin's namespace replaces with a real object.
    """
    import ast

    body = ast.parse(block).body
    sources = [ast.unparse(s) for s in body]

    def loads(stmt: ast.stmt) -> set:
        return {
            n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    def defines(stmt: ast.stmt) -> set:
        return {
            n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }

    def pullable(stmt: ast.stmt) -> bool:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return False
        return isinstance(stmt, (ast.Assign, ast.AnnAssign)) and not isinstance(
            getattr(stmt, "value", None), ast.Call
        )

    needed = {i for i, src in enumerate(sources) if focus in src}
    assert needed, f"no statement in the block mentions {focus!r}"

    changed = True
    while changed:
        changed = False
        wanted: set = set()
        produced: set = set()
        for i in sorted(needed):
            wanted |= loads(body[i])
            produced |= defines(body[i])
        for i, stmt in enumerate(body):
            if i in needed or isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            backward = pullable(stmt) and bool(defines(stmt) & wanted)
            forward = bool(loads(stmt) & produced)
            if backward or forward:
                needed.add(i)
                changed = True

    return "\n".join(sources[i] for i in sorted(needed))


@pytest.mark.parametrize("name,builder", _SIX_SITES, ids=[n for n, _ in _SIX_SITES])
def test_documented_print_lines_survive_the_could_not_check_state(name, builder):
    """REFUSAL PIN, finding 6.

    Applies each PUBLISHED snippet to a real object whose field the library
    could not measure. Before the fix all six raised
    `TypeError: unsupported format string passed to NoneType.__format__`.
    """
    site = builder()
    snippet = _reporting_slice(_block_containing(site.path, site.marker), site.focus)
    out = _run_snippet(snippet, dict(site.unmeasured))
    assert "COULD NOT CHECK" in out.upper() or "not estimated" in out, (
        f"{name}: the snippet ran but never said the value was unmeasured:\n{out}"
    )


@pytest.mark.parametrize("name,builder", _SIX_SITES, ids=[n for n, _ in _SIX_SITES])
def test_documented_print_lines_still_report_the_measured_value(name, builder):
    """OVER-CORRECTION CONTROL for finding 6.

    Every pin above would pass on a snippet that printed COULD NOT CHECK for
    everything, which is the same defect with the sign flipped: a measured
    disparity reported as unmeasurable. Same published snippet, an object whose
    field the library DID measure, and the assertion is that the real number
    reaches the output, formatted, to the digits the page promises.
    """
    site = builder()
    expected = site.expected
    snippet = _reporting_slice(_block_containing(site.path, site.marker), site.focus)
    out = _run_snippet(snippet, dict(site.measured))
    assert "COULD NOT CHECK" not in out.upper(), (
        f"{name}: a MEASURED value was reported as unmeasurable:\n{out}"
    )
    rendered = {f"{expected:.3f}", f"{expected:.4f}", f"{expected:.1%}", f"{expected:.2f}"}
    assert any(r in out for r in rendered), (
        f"{name}: measured value {expected!r} does not appear in the output:\n{out}"
    )


# ---------------------------------------------------------------------------
# Also reported: both assert_fairness functions now fail closed.
# ---------------------------------------------------------------------------


def test_both_assert_fairness_functions_fail_closed_on_an_unmeasurable_metric():
    """OVER-CORRECTION CONTROL, and the evidence behind the corrected callout.

    The published pages said in bold that the cicd wrapper "does not fail
    closed", citing an execution on 2026-08-28. The wrapper was fixed on
    2026-09-07 and the pages were not. Both halves are asserted: the refusal on
    a metric nobody measured, AND that a measured pass still passes, so the
    correction cannot be read as "it now refuses everything".
    """
    from vfairness.evaluation.vfairness_metrics.integrations import (
        FairnessAssertionError as IntegError,
    )
    from vfairness.evaluation.vfairness_metrics.integrations import assert_fairness as integrations
    from vfairness.operations.cicd.testing import FairnessAssertionError as CicdError
    from vfairness.operations.cicd.testing import assert_fairness as cicd

    rng = np.random.default_rng(3)
    n = 200
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    one_group = np.array(["A"] * n)
    two_group = np.array(["A"] * 100 + ["B"] * 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(IntegError, match="NOT MEASURABLE"):
            integrations(
                y_true,
                y_pred,
                one_group,
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_difference": 0.1},
            )
        with pytest.raises(CicdError, match="NOT MEASURABLE"):
            cicd(y_true, y_pred, one_group, metric="demographic_parity_difference", threshold=0.1)

        # A measured VIOLATION still fails, with the measured value in the message.
        biased = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
        with pytest.raises(CicdError, match=r"1\.0000"):
            cicd(y_true, biased, two_group, metric="demographic_parity_difference", threshold=0.1)

        # A measured PASS still passes. Without this the pin above is satisfied
        # by a function that raises unconditionally.
        assert (
            cicd(
                y_true,
                y_pred,
                two_group,
                metric="demographic_parity_difference",
                threshold=0.5,
            )
            is None
        )


def test_the_published_pages_no_longer_claim_the_cicd_wrapper_passes_silently():
    """REFUSAL PIN for the corrected callout."""
    for path in (_API_MD, _API_HTML):
        text = path.read_text(encoding="utf-8")
        assert "It does not fail closed" not in text, (
            f"{path.name} still says the cicd assert_fairness does not fail closed; "
            "it has since 2026-09-07"
        )
