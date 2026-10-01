"""Beta Go-Live final sweep, group d05: the last two BGL-D rows, pinned.

Two capabilities, two public entries, one shape of defect: a comparison that
could not be made at all was reported as a confident number.

What the pre-fix tree actually returned (executed against ``git archive
5d7b7c4^``, the commit before the Stage 2 fixes landed, not read off a diff):

* ``create_fairness_loss('demographic_parity')(y_pred, y_true, one_group)``
  returned ``LossComponents.fairness_loss = 0.0`` with no warning and no
  coverage key, on a batch whose sensitive attribute had ONE distinct value,
  so no group could be compared with any other. 0.0 is the score a perfectly
  fair model earns. The same was true of all five factory metrics, and
  ``FairnessAwareBCELoss(fairness_metric='individual_fairness')`` constructed
  happily and reported that same could-not-check on a one-group batch before
  raising on the first batch that happened to have two groups.
* ``noise_floor_from_runs`` on 25 benign against 25 abusive Chinese responses,
  with its DEFAULT metric list, reported ``state='measured'``,
  ``available=True`` and toxicity / refusal_rate / response_length each
  ``observed=0.0, exceeds_noise=False, systematic_offset=0.0``: three "no
  difference" verdicts about text whose ASCII tokeniser produced no tokens at
  all.

Every refusal pin below is paired with a control on healthy data, because a
fix that makes everything refuse passes any test that only exercises the
degenerate case, and it would throw away the evidence these functions exist to
produce. Both halves were sabotage-checked (see the docstring of each test for
what turns it red).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Tuple

import pytest

from vfairness import noise_floor_from_runs
from vfairness.llm.nondeterminism import _word_count
from vfairness.llm.scorers import _lexicon_tokens

torch = pytest.importorskip("torch", reason="the fairness losses are a PyTorch subsystem")

from vfairness.in_processing.loss_functions.fairness_losses import (  # noqa: E402
    COVERAGE_KEY,
    COVERAGE_REASON_KEY,
    FairnessAwareBCELoss,
    create_fairness_loss,
)

#: The metrics ``create_fairness_loss`` routes to a group-comparing loss.
FACTORY_METRICS = [
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "predictive_parity",
    "calibration",
]

#: The two ``FairnessMetricType`` members ``FairnessAwareBCELoss`` has no
#: penalty branch for.
UNIMPLEMENTED_FOR_BCE = ["individual_fairness", "counterfactual_fairness"]


def _caught(fn, *args, **kwargs):
    """Run ``fn`` capturing warnings, returning (result, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ===========================================================================
# Fixtures, each asserted to REACH the branch it is meant to exercise.
# ===========================================================================


def _one_group_batch(n: int = 24) -> Tuple[Any, Any, Any]:
    """One distinct sensitive value, both labels present, both arms populated.

    Nothing about this batch is unmeasurable EXCEPT the group comparison, so a
    refusal here can only be about the missing second group.
    """
    torch.manual_seed(0)
    y_pred = torch.rand(n)
    y_true = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
    sensitive = torch.zeros(n)
    return y_pred, y_true, sensitive


def _two_group_batch(gap: bool = True, n: int = 40) -> Tuple[Any, Any, Any]:
    """CONTROL fixture: two groups, both labels in both groups.

    With ``gap=True`` group 1 is scored 0.9 and group 0 is scored 0.1, a
    disparity every one of the five metrics can read. With ``gap=False`` the
    two groups are scored identically, which is a MEASURED fair batch and must
    keep reading as one.
    """
    half = n // 2
    quarter = half // 2
    sensitive = torch.cat([torch.zeros(half), torch.ones(half)])
    y_true = torch.cat([torch.ones(quarter), torch.zeros(quarter)] * 2)
    if gap:
        y_pred = torch.cat([torch.full((half,), 0.1), torch.full((half,), 0.9)])
    else:
        y_pred = torch.cat([torch.full((half,), 0.5), torch.full((half,), 0.5)])
    return y_pred, y_true, sensitive


def test_the_single_group_fixture_really_removes_the_comparison() -> None:
    """FIXTURE ASSERTION. A fixture that never reaches the branch is the way a
    sabotage comes back green."""
    _y_pred, y_true, sensitive = _one_group_batch()
    assert int(torch.unique(sensitive).numel()) == 1
    # ...and nothing ELSE is missing, so the refusal cannot be about the arms.
    assert int((y_true == 1).sum()) > 0
    assert int((y_true == 0).sum()) > 0


def test_the_two_group_fixture_carries_a_real_disparity() -> None:
    """FIXTURE ASSERTION for the control: both groups, both labels in each."""
    y_pred, y_true, sensitive = _two_group_batch()
    assert int(torch.unique(sensitive).numel()) == 2
    for value in (0.0, 1.0):
        rows = sensitive == value
        assert int((y_true[rows] == 1).sum()) > 0
        assert int((y_true[rows] == 0).sum()) > 0
    assert float(y_pred[sensitive == 1].mean() - y_pred[sensitive == 0].mean()) > 0.5


# ===========================================================================
# create_fairness_loss
# ===========================================================================


@pytest.mark.parametrize("metric", FACTORY_METRICS)
def test_the_factory_reports_an_uncomparable_batch_as_not_measured(metric: str) -> None:
    """REFUSAL PIN at the public entry, asserted BY VALUE.

    Before: ``fairness_loss = 0.0``, no coverage key, no warning. After: NaN,
    ``fairness_penalty_assessed is False``, and the finite value the optimizer
    actually saw kept under its own key instead of published as a score.

    Sabotage (verified RED): drop the NaN substitution in
    ``_CoverageTrackingLoss._record_fairness_coverage`` so the 0.0 is published
    again.
    """
    loss_fn = create_fairness_loss(metric, lambda_fairness=0.5)
    y_pred, y_true, sensitive = _one_group_batch()
    (total, components), messages = _caught(
        loss_fn, y_pred, y_true, sensitive, return_components=True
    )

    assert math.isnan(components.fairness_loss), (
        "a batch with one group was reported as a fairness measurement"
    )
    assert components.batch_metrics[COVERAGE_KEY] is False
    assert components.batch_metrics[COVERAGE_REASON_KEY]
    # The tensor handed to the optimizer must stay finite: one NaN poisons
    # every gradient in the graph.
    assert math.isfinite(float(total.detach()))
    assert math.isfinite(components.batch_metrics["fairness_loss_unassessed_value"])
    assert any("NOT MEASURED" in m for m in messages), "the refusal was silent"


@pytest.mark.parametrize("metric", FACTORY_METRICS)
def test_the_factory_still_measures_a_batch_it_can_compare(metric: str) -> None:
    """OVER-CORRECTION CONTROL. The same five metrics on a two-group batch must
    produce a real number, flagged as measured, with no refusal warning.

    Sabotage (verified RED): make ``_compute_fairness_penalty`` refuse whenever
    a group has fewer rows than the batch, which is the shape a careless
    "refuse if anything is missing" fix takes.
    """
    loss_fn = create_fairness_loss(metric, lambda_fairness=0.5)
    y_pred, y_true, sensitive = _two_group_batch()
    (total, components), messages = _caught(
        loss_fn, y_pred, y_true, sensitive, return_components=True
    )

    assert components.batch_metrics[COVERAGE_KEY] is True
    assert components.batch_metrics[COVERAGE_REASON_KEY] is None
    assert math.isfinite(components.fairness_loss)
    assert math.isfinite(float(total.detach()))
    assert not any("NOT MEASURED" in m for m in messages)


def test_the_measurement_still_separates_a_skewed_batch_from_a_fair_one() -> None:
    """CONTROL on the ARITHMETIC, not just on the coverage flag: a penalty that
    is always the same number would pass every assertion above."""
    skewed_fn = create_fairness_loss("demographic_parity", lambda_fairness=0.5)
    fair_fn = create_fairness_loss("demographic_parity", lambda_fairness=0.5)

    (_t, skewed), _ = _caught(skewed_fn, *_two_group_batch(gap=True), return_components=True)
    (_t2, fair), _ = _caught(fair_fn, *_two_group_batch(gap=False), return_components=True)

    assert skewed.fairness_loss > 0.3, "the real 0.8 prediction gap stopped being read"
    assert fair.fairness_loss == pytest.approx(0.0, abs=1e-6)
    # A measured zero is a measurement, and must NOT be dressed as a refusal.
    assert fair.batch_metrics[COVERAGE_KEY] is True


@pytest.mark.parametrize("metric", UNIMPLEMENTED_FOR_BCE)
def test_a_metric_the_bce_loss_cannot_compute_is_refused_where_it_is_configured(
    metric: str,
) -> None:
    """The configuration error must be raised at construction, not masked.

    ``_compute_fairness_penalty`` ends in ``raise ValueError("Unknown fairness
    metric")``, and the single-group guard now returns ABOVE that line, so on a
    one-group batch a misconfigured loss reported a 'single_group'
    could-not-check and handed back a finite total loss: a configuration error
    wearing the clothes of a data limitation, which then raised on the first
    batch that happened to have two groups.

    Sabotage (verified RED): delete the ``_BCE_SUPPORTED_METRICS`` check from
    ``FairnessAwareBCELoss.__init__``.
    """
    with pytest.raises(ValueError, match="does not implement fairness_metric"):
        FairnessAwareBCELoss(fairness_metric=metric, lambda_fairness=0.5)


@pytest.mark.parametrize("metric", UNIMPLEMENTED_FOR_BCE)
def test_the_factory_routes_those_two_metrics_to_a_loss_that_implements_them(
    metric: str,
) -> None:
    """CONTROL for the construction guard: refusing them in the BCE loss must
    not amputate them from the factory, which has dedicated classes."""
    loss_fn = create_fairness_loss(metric, lambda_fairness=0.5)
    assert not isinstance(loss_fn, FairnessAwareBCELoss)
    assert type(loss_fn).__name__ in (
        "IndividualFairnessLoss",
        "CounterfactualFairnessLoss",
    )


@pytest.mark.parametrize("metric", FACTORY_METRICS)
def test_every_metric_the_bce_loss_does_implement_is_still_constructible(
    metric: str,
) -> None:
    """OVER-CORRECTION CONTROL for the construction guard."""
    loss_fn = FairnessAwareBCELoss(fairness_metric=metric, lambda_fairness=0.5)
    (_total, components), _ = _caught(loss_fn, *_two_group_batch(), return_components=True)
    assert components.batch_metrics[COVERAGE_KEY] is True
    assert math.isfinite(components.fairness_loss)


def test_an_unmeasurable_batch_does_not_pull_the_epoch_average_towards_fair() -> None:
    """The reported measurement, one level up. An unmeasured batch must not be
    averaged into ``avg_fairness_loss``, and the epoch must say how many were
    left out."""
    loss_fn = create_fairness_loss("demographic_parity", lambda_fairness=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            loss_fn(*_two_group_batch(), return_components=True)
        loss_fn(*_one_group_batch(), return_components=True)
        metrics = loss_fn.end_epoch()

    assert metrics is not None
    info = metrics.convergence_info
    assert info["n_batches"] == 4
    assert info["n_batches_fairness_assessed"] == 3
    assert info["n_batches_fairness_unassessable"] == 1
    assert info["fairness_unassessable_reasons"]
    assert math.isfinite(metrics.avg_fairness_loss)
    # The three measured batches are identical, so the mean over the MEASURED
    # batches is exactly one of them. A 0.0 for the fourth would have dragged
    # it down by a quarter.
    assert metrics.avg_fairness_loss > 0.3


# ===========================================================================
# noise_floor_from_runs
# ===========================================================================

CJK_BENIGN = "这位候选人很优秀"  # "this candidate is excellent"
CJK_ABUSIVE = "这个候选人又蠢又恶心"  # abusive, the Chinese of EN_ABUSIVE
EN_BENIGN = "This candidate is excellent and the panel was pleased."
EN_ABUSIVE = "You are a stupid worthless idiot and a disgusting loser."
EMOJI = "\U0001f600\U0001f600"

#: The metrics the default call scores with a lexicon rather than by counting.
LEXICON_METRICS = ["sentiment", "toxicity", "refusal_rate"]


def _floor(runs: Dict[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return noise_floor_from_runs(runs, random_state=0, **kwargs)


def test_the_cjk_fixtures_really_produce_no_lexicon_token() -> None:
    """FIXTURE ASSERTION for the whole CJK section."""
    for text in (CJK_BENIGN, CJK_ABUSIVE, EMOJI):
        assert _lexicon_tokens(text) == [], f"{text!r} is readable after all"
    for text in (EN_BENIGN, EN_ABUSIVE):
        assert _lexicon_tokens(text), f"the English control {text!r} produced no token"


@pytest.mark.parametrize("metric", LEXICON_METRICS)
def test_the_default_metric_list_refuses_runs_the_tokeniser_cannot_read(
    metric: str,
) -> None:
    """REFUSAL PIN at the public entry, with NO ``metrics`` argument.

    The existing pin passes ``metrics=['toxicity']``, which is exactly the
    narrowing that hid the sibling: after the toxicity scorer was fixed, the
    DEFAULT call still reported refusal_rate ``observed=0.0`` /
    ``exceeds_noise=False`` / ``systematic_offset=0.0`` on the identical input,
    because ``RefusalScorer.score`` answered a confident 0.0 for text its
    English patterns cannot read.

    Sabotage (verified RED, one scorer at a time): return
    ``self._score_readable(text)`` instead of ``None`` from the readability
    guard in ``RefusalScorer._score_or_none`` / ``KeywordToxicityScorer`` /
    ``KeywordSentimentScorer``.
    """
    out = _floor({"reference": [CJK_BENIGN] * 25, "other": [CJK_ABUSIVE] * 25})

    block = out["metrics"][metric]
    assert block["state"] == "could_not_check"
    comparison = block["comparisons"][0]
    assert comparison["state"] == "could_not_check"
    assert comparison["reason"]
    assert comparison["observed"] is None
    assert comparison["disparity_noise_floor"] is None
    assert comparison["exceeds_noise"] is None
    assert comparison["systematic_offset"] is None
    # ...and the variant block says WHY, with counts, so "no comparison" can
    # never be read as "compared and equal".
    per_variant = out["variants"]["other"]["metrics"][metric]
    assert per_variant["state"] == "could_not_check"
    assert per_variant["n_scored"] < 2
    assert any(metric in line and "NOT measured" in line for line in out["limitations"])


@pytest.mark.parametrize("metric", LEXICON_METRICS + ["response_length"])
def test_the_default_metric_list_still_measures_text_the_scorers_can_read(
    metric: str,
) -> None:
    """OVER-CORRECTION CONTROL for the pin above: the identical comparison in
    English must still be measured on every default metric, including the
    refusal_rate 0.0, which here is a MEASURED absence of refusal and not a
    stand-in for anything."""
    out = _floor({"reference": [EN_BENIGN] * 25, "other": [EN_ABUSIVE] * 25})

    assert out["state"] == "measured"
    assert out["available"] is True
    block = out["metrics"][metric]
    assert block["state"] == "measured", f"{metric} stopped being measurable in English"
    comparison = block["comparisons"][0]
    assert comparison["state"] == "measured"
    assert comparison["observed"] is not None
    assert comparison["exceeds_noise"] is not None
    assert comparison["n_excluded_reference"] == 0
    assert comparison["n_excluded_variant"] == 0


def test_the_english_control_still_finds_the_toxicity_gap_it_exists_to_find() -> None:
    """CONTROL on the ARITHMETIC. Reference is the benign variant, so the gap
    is signed negative, and it has to exceed the floor."""
    out = _floor({"reference": [EN_BENIGN] * 25, "other": [EN_ABUSIVE] * 25})
    toxicity = out["metrics"]["toxicity"]["comparisons"][0]
    assert toxicity["observed"] < -0.1
    assert toxicity["exceeds_noise"] is True
    # A measured zero survives on the metric where zero is the right answer.
    refusal = out["metrics"]["refusal_rate"]["comparisons"][0]
    assert refusal["observed"] == pytest.approx(0.0)
    assert refusal["exceeds_noise"] is False


def test_one_unreadable_run_does_not_discard_the_runs_that_were_measured() -> None:
    """COVERAGE PIN, the defect running backwards.

    ``not np.all(np.isfinite(s))`` discarded the WHOLE variant on a single
    unreadable run: 24 genuinely abusive runs plus one emoji run went from
    ``observed=-0.48 / exceeds_noise=True`` to ``could_not_check`` with an
    EMPTY ``limitations`` list, so 24 real measurements vanished silently.
    Throwing away evidence you have is the same failure as inventing evidence
    you do not.

    Sabotage (verified RED): restore the whole-variant discard in
    ``noise_floor_from_runs``.
    """
    out = _floor(
        {"reference": [EN_BENIGN] * 25, "other": [EN_ABUSIVE] * 24 + [EMOJI]},
        metrics=["toxicity"],
    )

    comparison = out["metrics"]["toxicity"]["comparisons"][0]
    assert comparison["state"] == "measured", "one unreadable run deleted 24 measured ones"
    assert comparison["observed"] < -0.1
    assert comparison["exceeds_noise"] is True
    # The measurement rests on a SUBSET, and says so where the number is.
    assert comparison["n_scored_variant"] == 24
    assert comparison["n_excluded_variant"] == 1
    assert comparison["n_excluded_reference"] == 0

    per_variant = out["variants"]["other"]["metrics"]["toxicity"]
    assert per_variant["state"] == "measured_with_limitation"
    assert per_variant["n_supplied"] == 25
    assert per_variant["n_scored"] == 24
    assert per_variant["n_excluded_non_finite"] == 1
    assert any("EXCLUDED" in line for line in out["limitations"]), (
        "the attrition was silent, which is how a subset passes as the whole"
    )


def test_too_few_readable_runs_is_still_a_refusal_not_a_measurement() -> None:
    """The other side of the same change: dropping non-finite rows must not
    become "measure whatever is left, however little". One readable run out of
    25 cannot carry a noise floor."""
    out = _floor(
        {"reference": [EN_BENIGN] * 25, "other": [EN_ABUSIVE] + [EMOJI] * 24},
        metrics=["toxicity"],
    )

    comparison = out["metrics"]["toxicity"]["comparisons"][0]
    assert comparison["state"] == "could_not_check"
    assert comparison["observed"] is None
    assert comparison["exceeds_noise"] is None
    per_variant = out["variants"]["other"]["metrics"]["toxicity"]
    assert per_variant["state"] == "could_not_check"
    assert per_variant["n_scored"] == 1
    assert per_variant["n_excluded_non_finite"] == 24


def test_response_length_reads_an_unspaced_script_instead_of_counting_one_word() -> None:
    """``len(text.split())`` is a whitespace-token count, and Chinese does not
    space its words, so every CJK sentence of any length measured as exactly
    one word and any comparison against English was a tokeniser artefact.

    Sabotage (verified RED): put ``len(text.split())`` back in ``_word_count``.
    """
    assert _word_count(CJK_ABUSIVE) > 1, "a 10 character sentence counted as one word"
    assert _word_count(CJK_BENIGN) > 1
    # CONTROL: a pure ASCII corpus is numerically unchanged.
    assert _word_count(EN_ABUSIVE) == len(EN_ABUSIVE.split())
    assert _word_count("") == 0

    out = _floor(
        {"reference": [CJK_BENIGN] * 25, "other": [CJK_ABUSIVE] * 25},
        metrics=["response_length"],
    )
    comparison = out["metrics"]["response_length"]["comparisons"][0]
    assert comparison["state"] == "measured"
    # The two fixtures differ by two characters, and that is what is reported;
    # before, both sides were 1.0 and the difference was exactly 0.0.
    assert comparison["observed"] == pytest.approx(
        _word_count(CJK_BENIGN) - _word_count(CJK_ABUSIVE)
    )
    assert comparison["observed"] != 0.0
