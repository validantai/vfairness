"""The five units grading wave 4 claimed and no test actually executed.

HOW THEY WERE FOUND, and it is the point of this file. Wave 4 recorded a
``test_file`` for all 95 units it graded, and a check was then run asking, per row,
whether the NAMED file's coverage actually reaches the unit's body lines. It did not
for five of them, and those five were published as CHECKED on the live status page
with a test beside them that never ran them:

* ``BaseFairnessLoss.end_epoch`` and ``BaseRegularizer.forward`` are BASE
  implementations, and the tests pointed at ``DemographicParityLoss`` and
  ``StatisticalParityRegularizer``, both of which override them. Exercising a
  subclass is not exercising the base, and the base is public API: it is the
  documented extension point anybody writing their own loss or regulariser inherits.
* ``CausalDecomposition.to_dict`` and ``ExperimentRecommendation.to_dict`` were
  inspected for their field lists and never constructed, so nothing called them.
* ``xai.worker.runner.main`` was attributed to a file that does not mention it. It is
  carried by a module-level default in the generator, not by any test.

The generator assigned a default test file per module and silently fell back for
modules it did not know, which is how a row came to name a test that cannot support
it. Six further rows named the wrong one of the five wave-4 files; those were
attributions to correct, not missing evidence.

A recorded ``test_file`` is the evidence for a published CHECKED badge. A row naming
a test that never touches the unit is the same defect this library is being audited
for, one level up: a claim with nothing behind it, in the surface that exists to say
which claims have something behind them.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import pytest

from vfairness.in_processing.loss_functions.base import (
    BaseFairnessLoss,
    LossComponents,
    TrainingMetrics,
)
from vfairness.in_processing.regularizers.fairness_regularizers import BaseRegularizer
from vfairness.operations.experimentation.analysis import (
    CausalDecomposition,
    ExperimentRecommendation,
    RecommendationDecision,
)

# ---------------------------------------------------------------------------
# BaseRegularizer.forward: a base that answered 0.0 would silently disable
# fairness regularisation for any subclass that forgot to implement it
# ---------------------------------------------------------------------------


class _IncompleteRegularizer(BaseRegularizer):
    """A subclass that forgets to implement forward, which is the case that matters."""


def test_an_unimplemented_regulariser_refuses_instead_of_returning_no_penalty():
    torch = pytest.importorskip("torch")
    regulariser = _IncompleteRegularizer(strength=0.1)

    with pytest.raises(NotImplementedError, match="must implement forward"):
        regulariser.forward(torch.rand(20), torch.randint(0, 2, (20,)).float())


def test_the_base_regulariser_refuses_through_the_call_operator_too():
    """`reg(...)` is how every caller invokes it, so the refusal must survive
    whatever __call__ wraps around forward. A wrapper that swallowed the
    NotImplementedError and returned a zero penalty would train a model with the
    fairness term silently switched off, and every fairness metric would look
    exactly as it does for an honestly regularised run."""
    torch = pytest.importorskip("torch")
    regulariser = _IncompleteRegularizer(strength=0.1)

    with pytest.raises(NotImplementedError):
        regulariser(torch.rand(20), torch.randint(0, 2, (20,)).float())


# ---------------------------------------------------------------------------
# BaseFairnessLoss.end_epoch, reached through a DIRECT subclass
# ---------------------------------------------------------------------------


class _DirectLoss(BaseFairnessLoss):
    """Inherits BaseFairnessLoss without going through _CoverageTrackingLoss.

    Every loss the library exports inherits the coverage-tracking override instead,
    which NaNs the epoch average when no batch could be measured. This subclass is
    what an external implementer gets, and it is the only way to execute the base
    aggregation at all.
    """

    def forward(self, *args, **kwargs):  # pragma: no cover - not what is under test
        raise NotImplementedError


def _components(total: float, task: float, fairness: float) -> LossComponents:
    return LossComponents(
        total_loss=total, task_loss=task, fairness_loss=fairness, regularization_loss=0.0
    )


def test_the_base_epoch_aggregation_withholds_a_metric_when_no_batch_was_seen():
    loss = _DirectLoss(track_metrics=True)
    assert loss.end_epoch() is None, (
        "an epoch with no batches produced a TrainingMetrics object; averaging over "
        "nothing is a division by zero, and a zeroed metric reads as a measured one"
    )


def test_the_base_epoch_aggregation_averages_the_batches_it_was_given():
    loss = _DirectLoss(track_metrics=True)
    loss._batch_history = [_components(1.0, 0.8, 0.2), _components(3.0, 2.0, 1.0)]

    metrics = loss.end_epoch()

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.avg_total_loss == pytest.approx(2.0)
    assert metrics.avg_task_loss == pytest.approx(1.4)
    assert metrics.avg_fairness_loss == pytest.approx(0.6)


def test_the_base_epoch_aggregation_clears_the_batch_history_it_consumed():
    """Otherwise the next epoch averages this epoch's batches again."""
    loss = _DirectLoss(track_metrics=True)
    loss._batch_history = [_components(1.0, 0.8, 0.2)]

    loss.end_epoch()

    assert loss._batch_history == []
    assert loss.end_epoch() is None
    assert len(loss.get_training_history()) == 1


def test_tracking_switched_off_reports_nothing_rather_than_zeroes():
    loss = _DirectLoss(track_metrics=False)
    loss._batch_history = [_components(1.0, 0.8, 0.2)]

    assert loss.end_epoch() is None, (
        "with tracking disabled the loss returned metrics anyway, which a caller "
        "cannot distinguish from metrics it asked to be collected"
    )


# ---------------------------------------------------------------------------
# The two experiment serialisers nothing had constructed
# ---------------------------------------------------------------------------


def test_the_causal_decomposition_serialises_every_field_including_its_caveats():
    decomposition = CausalDecomposition(
        total_effect=0.12,
        direct_effect=0.08,
        indirect_effect=0.04,
        mediator="credit_score",
        proportion_mediated=0.333,
        steps_satisfied={"exchangeability": False, "positivity": True},
    )
    payload = decomposition.to_dict()

    declared = {f.name for f in dataclasses.fields(CausalDecomposition)}
    assert declared <= set(payload), f"to_dict drops {sorted(declared - set(payload))}"

    # steps_satisfied is the assumption ledger: a mediation estimate whose
    # exchangeability step failed is not a causal claim, and dropping that flag on
    # the way out would leave a bare effect size looking established.
    assert payload["steps_satisfied"] == {"exchangeability": False, "positivity": True}
    assert payload["proportion_mediated"] == pytest.approx(0.333)
    json.dumps(payload, default=str)


def test_a_decomposition_with_no_defined_proportion_keeps_the_nan():
    """proportion_mediated is indirect/total, so a zero total effect makes it
    undefined. NaN must survive the serialiser rather than becoming 0.0, which
    would read as "nothing was mediated"."""
    payload = CausalDecomposition(
        total_effect=0.0,
        direct_effect=0.0,
        indirect_effect=0.0,
        mediator="credit_score",
        proportion_mediated=float("nan"),
    ).to_dict()

    assert np.isnan(payload["proportion_mediated"])


def test_the_recommendation_serialises_every_field_including_its_caveats():
    recommendation = ExperimentRecommendation(
        decision=RecommendationDecision.EXTEND_EXPERIMENT,
        confidence=0.41,
        reasoning=["the intersectional arm is underpowered"],
        trade_offs={"revenue": "flat"},
        caveats=["three subgroups below the minimum group size"],
    )
    payload = recommendation.to_dict()

    declared = {f.name for f in dataclasses.fields(ExperimentRecommendation)}
    assert declared <= set(payload), f"to_dict drops {sorted(declared - set(payload))}"

    # The caveats are the reason the decision is EXTEND rather than DEPLOY. A reader
    # who sees the decision without them sees a recommendation with no conditions.
    assert payload["caveats"] == ["three subgroups below the minimum group size"]
    assert payload["reasoning"] == ["the intersectional arm is underpowered"]
    json.dumps(payload, default=str)


def test_the_recommendation_decision_survives_as_its_wire_value():
    payload = ExperimentRecommendation(
        decision=RecommendationDecision.KEEP_CONTROL, confidence=0.9, reasoning=["no effect"]
    ).to_dict()
    assert payload["decision"] == RecommendationDecision.KEEP_CONTROL.value


# ---------------------------------------------------------------------------
# The explainability worker entry point
# ---------------------------------------------------------------------------


def test_the_worker_entry_point_configures_logging_and_starts_the_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    """``main`` runs an unbounded worker loop, so the loop is replaced by a recorder.
    That executes main's own body, which is what had never run: a typo in the
    logging setup or a loop that is constructed and never started would have
    surfaced only on a deployment."""
    from vfairness.xai.worker import runner

    started: list[str] = []

    class _Recorder:
        def run(self):
            started.append("ran")

    monkeypatch.setattr(runner, "WorkerLoop", lambda *a, **k: _Recorder())
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    runner.main()

    assert started == ["ran"], "main did not start the worker loop"
    assert logging.getLogger().level == logging.WARNING, (
        "main did not apply LOG_LEVEL, so a deployed worker would log at the default "
        "level whatever the environment asked for"
    )


# ---------------------------------------------------------------------------
# The drift explainer, whose grade rested on a transitive justification
# ---------------------------------------------------------------------------


def _drift_detector():
    from vfairness.operations.monitoring.drift import FairnessDriftDetector

    return FairnessDriftDetector(window_sizes=[20], compute_mmd=False, n_permutations=20)


def test_the_drift_explanation_says_could_not_check_when_no_test_ran():
    """Its grading row said "none: the explainer's own three-state behaviour is
    already pinned by [two other files]". Those files pin the EXPLAINER; neither
    reaches this method, so the justification was transitive and unverified. It turns
    out to be honest, which is worth pinning rather than assuming.
    """
    import pandas as pd

    detector = _drift_detector()
    detector.set_baseline(pd.Series([1.0, 1.0, 1.0]))
    result = detector.check_drift(pd.Series([1.0, 1.0, 1.0]), metric="demographic_parity")

    assert result.drift_detected is None, "the fixture no longer produces a no-verdict result"

    report = detector.get_explanation(result)
    # ASSERT ON THE SUMMARY FIELD, NOT THE RENDERED REPORT. The first version of this
    # test searched str(report) for "COULD NOT CHECK", and that phrase occurs three
    # times in the rendered text, so rewriting the SUMMARY to "Drift NOT DETECTED: the
    # data looks stable" left the test green. The pin could not fail, which is the
    # condition it exists to detect in other people's code.
    assert "COULD NOT CHECK" in report.summary, (
        f"the summary a reader sees first does not say it could not check: {report.summary!r}"
    )
    assert "no verdict" in report.summary
    assert "not detected" not in report.summary.lower(), (
        "the summary claims stability for a run in which no drift test executed"
    )


def test_control_the_drift_explanation_reports_real_drift():
    """The over-correction control: an explainer that always said could-not-check
    would pass the test above."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    detector = _drift_detector()
    detector.set_baseline(pd.Series(rng.normal(0, 1, 200)))
    result = detector.check_drift(pd.Series(rng.normal(1.5, 1, 200)), metric="demographic_parity")

    assert result.drift_detected is True
    text = str(detector.get_explanation(result))
    assert "COULD NOT CHECK" not in text
    assert "Drift" in text
