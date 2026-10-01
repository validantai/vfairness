"""The analysis surfaces nothing had executed: correlation, resampling, training
analysis, loss history, regularisers, thresholds, and the status API itself.

WHAT WAS FOUND

``_correlation_ratio_with_pvalue`` published ``corr=0.0, pvalue=1.0`` for a feature
with zero variance, under the comment "no variance in the continuous variable: no
association". eta is ss_between / ss_total, so with ss_total == 0 it is 0/0 and
undefined, and the 1.0 is a p-value for an ANOVA that never ran. That is the exact
claim the same function's docstring refuses one branch below ("definitely not
significant" about a test that never happened, the thing that once deleted a
CRITICAL race proxy at the caller's significance gate).

It was found through an INCONSISTENCY, which is why the test below checks both arities
together: a constant feature against a TWO-level protected attribute routes to
``_point_biserial``, where scipy returns nan, so the identical undefined quantity read
as nan for two groups and as a confident measured zero for three.

``SurfaceStatus.share`` returned a fabricated zero, in the library's own status API.
It was ``counts.get(state, 0) / total`` over upper-case keys, while ``__str__`` prints
them lower case, so the obvious reading of "874 checked" was ``share("checked")`` and
that answered **0.0** about the 874 units it had just printed.

WHAT WAS FOUND HONEST, pinned so it stays that way. The fairness regulariser over a
single group is the model of what this library asks for: it keeps the penalty at 0.0
so the training loss is not poisoned by a NaN, and it says so everywhere else at once,
with ``measured=False``, ``dependence_measure=nan``,
``metadata["not_assessed"]="fewer_than_two_groups"`` and a warning. A fresh loss
returns an empty history and ``None`` from ``end_epoch`` rather than a zeroed metric,
and the LLM proxy refuses a loopback endpoint at the egress guard and reports
``ok: False`` with the transport error rather than a latency it never measured.
"""

from __future__ import annotations

import dataclasses
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from vfairness.in_processing.analyzer import FairnessTrainingAnalyzer  # noqa: E402
from vfairness.in_processing.loss_functions import DemographicParityLoss  # noqa: E402
from vfairness.in_processing.regularizers.fairness_regularizers import (  # noqa: E402
    RegularizerType,
    create_regularizer,
)
from vfairness.post_processing.threshold_optimization.analyzer import (  # noqa: E402
    ThresholdAnalyzer,
)
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    CorrelationResult,
    CorrelationType,
    compute_feature_correlations,
)

N = 400


def _rng():
    return np.random.default_rng(11)


# ---------------------------------------------------------------------------
# Correlation against a feature with no variance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "levels", [["a", "b"], ["a", "b", "c"]], ids=["two_groups", "three_groups"]
)
def test_a_constant_feature_has_no_correlation_to_report_at_any_arity(levels):
    """Both arities must refuse. The three-group path used to answer 0.0 / 1.0."""
    rng = _rng()
    frame = pd.DataFrame({"g": rng.choice(levels, N), "const": 1.0, "real": rng.random(N)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = compute_feature_correlations(
            frame, protected_attributes=["g"], feature_columns=["const", "real"]
        )

    assert np.isnan(matrix.correlations.loc["const", "g"]), (
        "a feature with zero variance was given a measured correlation; eta is 0/0 "
        "there and the value is arithmetic, not data"
    )
    assert np.isnan(matrix.pvalues.loc["const", "g"]), (
        "a p-value of 1.0 here claims 'definitely not significant' about a test that never ran"
    )


def test_control_a_feature_with_variance_is_still_measured():
    """The over-correction control: refusing everything would pass the test above."""
    rng = _rng()
    frame = pd.DataFrame({"g": rng.choice(["a", "b", "c"], N), "real": rng.random(N)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = compute_feature_correlations(
            frame, protected_attributes=["g"], feature_columns=["real"]
        )

    assert np.isfinite(matrix.correlations.loc["real", "g"])
    assert np.isfinite(matrix.pvalues.loc["real", "g"])


def test_the_refusal_is_announced_and_not_only_encoded():
    """nan in a cell is a hole a reader might not look at. Say it out loud too."""
    rng = _rng()
    frame = pd.DataFrame({"g": rng.choice(["a", "b", "c"], N), "const": 1.0})

    with pytest.warns(UserWarning, match="COULD NOT BE MEASURED"):
        compute_feature_correlations(frame, protected_attributes=["g"], feature_columns=["const"])


def test_an_unmeasured_cell_fails_the_high_correlation_gate():
    """nan must not be read as a finding, and must not be read as a clean pass
    either: it simply cannot satisfy a threshold."""
    rng = _rng()
    frame = pd.DataFrame({"g": rng.choice(["a", "b", "c"], N), "const": 1.0, "real": rng.random(N)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = compute_feature_correlations(
            frame, protected_attributes=["g"], feature_columns=["const", "real"]
        )

    flagged = {name for name, _attr, _value in matrix.get_high_correlations(0.0)}
    assert "const" not in flagged


# ---------------------------------------------------------------------------
# CorrelationResult, whose optional evidence must not become zero
# ---------------------------------------------------------------------------


def test_a_correlation_result_keeps_the_evidence_it_does_not_have():
    """Optional evidence that was never computed must serialise as None. Zero is a
    measurement, and for mutual information zero means proven independence."""
    result = CorrelationResult(
        feature="zip",
        protected_attribute="race",
        correlation=0.42,
        correlation_type=CorrelationType.PEARSON,
        pvalue=0.01,
        mutual_information=None,
        cramers_v=None,
        sample_size=300,
    )
    payload = result.to_dict()

    declared = {f.name for f in dataclasses.fields(CorrelationResult)}
    assert declared <= set(payload), f"to_dict drops {sorted(declared - set(payload))}"
    assert payload["mutual_information"] is None
    assert payload["cramers_v"] is None
    assert payload["correlation"] == 0.42
    json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Training analysis and its serialisers
# ---------------------------------------------------------------------------


def _analyzer():
    rng = _rng()
    return FairnessTrainingAnalyzer(
        rng.random((N, 4)), rng.integers(0, 2, N), rng.choice(["a", "b"], N)
    )


def test_the_data_summary_describes_the_data_it_was_given():
    summary = _analyzer().get_data_summary()
    assert summary["n_samples"] == N
    assert summary["n_groups"] == 2
    assert set(summary["groups"]) == {"a", "b"}
    assert summary["n_features"] == 4


def test_the_analyzer_explains_itself_before_and_after_an_analysis():
    analyzer = _analyzer()
    assert str(analyzer.get_explanation()).strip()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyzer.full_analysis()
    assert str(analyzer.get_explanation(report)).strip()


def test_the_training_report_serialises_every_field_it_declares():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = _analyzer().full_analysis()

    payload = report.to_dict()
    declared = {f.name for f in dataclasses.fields(report)}
    assert declared <= set(payload), f"to_dict drops {sorted(declared - set(payload))}"

    # to_json must be the same content, not a second implementation that can drift.
    assert json.loads(report.to_json()) == json.loads(json.dumps(payload, default=str))


def test_the_training_report_renders_an_svg_a_browser_would_accept():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = _analyzer().full_analysis().to_svg()

    assert svg.lstrip().startswith("<svg") or "<svg" in svg[:200]
    assert svg.rstrip().endswith("</svg>")
    assert 'role="img"' in svg, "the rendered chart carries no accessible role"


# ---------------------------------------------------------------------------
# Loss history
# ---------------------------------------------------------------------------


def test_a_loss_with_no_epochs_reports_no_history_rather_than_a_zero():
    loss = DemographicParityLoss()

    assert loss.get_training_history() == []
    assert loss.end_epoch() is None, (
        "a loss that has seen no batches returned an epoch metric; a zeroed metric "
        "is indistinguishable from a measured one"
    )


def test_resetting_history_is_idempotent_and_leaves_it_empty():
    loss = DemographicParityLoss()
    loss.reset_history()
    loss.reset_history()
    assert loss.get_training_history() == []


# ---------------------------------------------------------------------------
# Regularisers
# ---------------------------------------------------------------------------


# One member of the public enum has no implementation behind it. That is recorded
# here as a fact about the release, not smoothed over: a caller who picks it out of
# the library's own enum must be told which of "not implemented" and "you mistyped
# it" is true.
NOT_IMPLEMENTED_REGULARISERS = {RegularizerType.MUTUAL_INFORMATION}


def test_every_regulariser_type_is_either_constructible_or_says_it_is_not():
    for kind in RegularizerType:
        if kind in NOT_IMPLEMENTED_REGULARISERS:
            with pytest.raises(NotImplementedError, match="NOT IMPLEMENTED"):
                create_regularizer(kind)
        else:
            assert create_regularizer(kind) is not None


def test_an_advertised_but_absent_regulariser_is_not_reported_as_a_typo():
    """ValueError("Unknown regularizer type") sent the caller hunting for a
    misspelling of a name they had read off the enum."""
    with pytest.raises(NotImplementedError) as caught:
        create_regularizer(RegularizerType.MUTUAL_INFORMATION)

    message = str(caught.value)
    assert "public enum" in message
    assert "hsic" in message.lower(), "no alternative offered for the missing method"


def test_the_set_of_unimplemented_regularisers_has_not_grown():
    """If a new member is added to the enum without an implementation, this fails
    rather than the gap arriving unannounced."""
    constructible = set()
    for kind in RegularizerType:
        try:
            create_regularizer(kind)
            constructible.add(kind)
        except NotImplementedError:
            pass
    assert set(RegularizerType) - constructible == NOT_IMPLEMENTED_REGULARISERS


def test_an_unknown_regulariser_is_refused_rather_than_substituted():
    with pytest.raises(ValueError, match="not a valid RegularizerType"):
        create_regularizer("not_a_regularizer")


def test_a_regulariser_over_one_group_says_it_measured_nothing():
    """The penalty stays 0.0 on purpose: a NaN would poison the training loss. So
    the disclosure has to live everywhere else, and it does."""
    torch = pytest.importorskip("torch")
    regulariser = create_regularizer(RegularizerType.STATISTICAL_PARITY)

    with pytest.warns(UserWarning, match="only 1 of 1 groups"):
        value, metrics = regulariser(torch.rand(200), torch.zeros(200), return_metrics=True)

    assert float(value) == 0.0
    assert metrics.measured is False
    assert np.isnan(metrics.dependence_measure)
    assert metrics.metadata["not_assessed"] == "fewer_than_two_groups"
    assert metrics.group_penalties == {}


def test_control_a_regulariser_over_two_groups_measures():
    torch = pytest.importorskip("torch")
    regulariser = create_regularizer(RegularizerType.STATISTICAL_PARITY)
    rng = _rng()

    value, metrics = regulariser(
        torch.tensor(rng.random(200), dtype=torch.float32),
        torch.tensor(rng.integers(0, 2, 200), dtype=torch.float32),
        return_metrics=True,
    )

    assert metrics.measured is True
    assert np.isfinite(metrics.dependence_measure)
    assert len(metrics.group_penalties) == 2
    assert float(value) >= 0.0


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def test_the_threshold_analyzer_explains_itself():
    rng = _rng()
    analyzer = ThresholdAnalyzer(rng.integers(0, 2, N), rng.random(N), rng.choice(["a", "b"], N))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert str(analyzer.get_explanation()).strip()


# ---------------------------------------------------------------------------
# The status API, which had a fabricated zero of its own
# ---------------------------------------------------------------------------


def test_the_surface_shares_are_real_fractions_that_sum_to_one():
    import vfairness

    surface = vfairness.status()
    shares = {state: surface.share(state) for state in surface.counts}

    for state, share in shares.items():
        expected = surface.counts[state] / surface.total
        assert share == pytest.approx(expected), (
            f"share({state!r}) disagrees with the count it is derived from"
        )
    assert sum(shares.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("spelling", ["checked", "CHECKED", "Checked"])
def test_share_accepts_the_spelling_the_library_itself_prints(spelling):
    """__str__ prints the states lower case, so a lower-case share() is the obvious
    call. It used to answer 0.0 for the very units it had just printed."""
    import vfairness

    surface = vfairness.status()
    assert surface.share(spelling) == pytest.approx(surface.counts["CHECKED"] / surface.total)
    assert surface.share(spelling) > 0.0


@pytest.mark.parametrize("spelling", ["fix pending", "fix_pending", "FIX-PENDING"])
def test_share_accepts_either_separator(spelling):
    """FIX PENDING is the subject because its NAME carries the separator.

    `counts.get(state, 0)` rather than `counts[state]`, changed 2026-09-27. The BGL3
    campaign closed every open defect, so FIX PENDING has no members and the generated
    ledger's `counts` no longer carries the key: this raised KeyError on all three
    spellings. A state with no members is still a state, and zero of 1,567 is a real
    share of it.

    The test still discriminates with the count at zero, which is the point worth
    keeping: a share() that did NOT normalise separators would RAISE on "fix_pending"
    rather than answer 0.0, so each spelling is still being asked the question this test
    was written to ask.
    """
    import vfairness

    surface = vfairness.status()
    expected = surface.counts.get("FIX PENDING", 0) / surface.total
    assert surface.share(spelling) == pytest.approx(expected)


@pytest.mark.parametrize("spelling", ["not checked", "not_checked", "NOT-CHECKED"])
def test_share_accepts_either_separator_on_a_populated_state(spelling, status_with_every_state):
    """The same property against a multi-word state that HAS members.

    Added 2026-09-27 alongside the change above. With FIX PENDING emptied, that test's
    two sides are both 0.0, and a comparison of zero against zero is weaker evidence
    than it looks even though a non-normalising share() would still raise. NOT CHECKED
    carries a separator and a non-zero count, so this one compares a real number.

    NOT CHECKED emptied on 2026-10-01, so the populated state is now injected
    (tests/conftest.py) rather than borrowed from the live ledger.
    """
    import vfairness

    surface = vfairness.status()
    expected = surface.counts["NOT CHECKED"] / surface.total
    assert surface.share(spelling) == pytest.approx(expected)
    assert surface.share(spelling) > 0.0


@pytest.mark.parametrize("bad", ["nope", "passed", "", "verified"])
def test_a_state_that_does_not_exist_is_refused_not_answered_with_zero(bad):
    import vfairness

    with pytest.raises(ValueError, match="not a published status state"):
        vfairness.status().share(bad)


def test_an_unknown_capability_is_refused_by_name():
    import vfairness
    from vfairness.status import UnknownCapabilityError

    with pytest.raises(UnknownCapabilityError, match="not on the public surface"):
        vfairness.status("no_such_capability_xyz")


# ---------------------------------------------------------------------------
# LLM proxy connectivity, which must never claim a latency it did not measure
# ---------------------------------------------------------------------------


def test_a_loopback_endpoint_is_refused_by_the_egress_guard():
    from vfairness.llm.api_proxy import LLMApiProxy

    with pytest.raises(ValueError, match="egress guard"):
        LLMApiProxy(endpoint_url="http://127.0.0.1:9/v1")


def test_a_connection_that_failed_reports_the_failure_and_no_latency():
    """Port 9 (discard) refuses immediately, so this makes no outbound request that
    can succeed and needs no network to be present."""
    from vfairness.llm.api_proxy import LLMApiProxy

    proxy = LLMApiProxy(
        endpoint_url="http://127.0.0.1:9/v1",
        allow_loopback=True,
        timeout=1,
        max_retries=1,
    )
    result = proxy.test_connection()

    assert result["ok"] is False
    assert result["error"], "a failed connection with no error to show"
    assert result["latency_ms"] < 0, (
        "a failed connection reported a non-negative latency, which reads as a measured round trip"
    )


def test_the_text_scorer_contract_is_a_protocol_anyone_can_satisfy():
    """TextScorer is the seam third-party scorers plug into, so its shape is API."""
    from vfairness.llm.scorers import TextScorer

    assert getattr(TextScorer, "_is_protocol", False), (
        "TextScorer stopped being a Protocol, which breaks every structural "
        "implementation that does not inherit from it"
    )
