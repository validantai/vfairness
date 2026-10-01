"""The CI/CD gate boundaries and the experiment designers, executed for the first time.

Twenty-six public units across ``operations.cicd`` and ``operations.experimentation``
had no assertion behind them: the serialisers that turn a gate decision into a PR
comment or a JSON payload, the batch validator, the test-suite runner, the CLI entry
point, and the two functions that assign an experiment's arms.

The gate itself was already graded. Its BOUNDARIES were not, and a boundary is where
a measured could-not-check turns back into a clean pass, so this file asks of each
one: does the reader on the other side still see what the gate actually established?

WHAT WAS FOUND

``assign_clusters`` produced a single-arm design in silence. Over one cluster, at
``treatment_fraction=0.5``, it assigned all 400 rows to treatment, because
``max(1, int(n * fraction))`` had no upper clamp. Stratified, the same expression
sent EVERY singleton stratum to treatment: ten singleton strata at a requested 0.5
delivered 1.0. Nothing in the returned frame recorded that. It now clamps so a
control cluster survives wherever the count allows, and warns both for the strata it
could not randomise and for a design that ends up with one arm.

``IntersectionalGateDecision.to_dict`` declared six dataclass fields and emitted
five: ``summary_decision`` was dropped, so a hierarchical decision written to JSON
lost its summary arm with no key to show it ever had one.

WHAT WAS FOUND HONEST, recorded so it stays that way. The gate fails closed on a
metric it cannot compute (NaN value, blocked, and the reason in
``blocking_reasons`` verbatim rather than an invented number), the report card
carries small-sample warnings into the PR comment where the merge decision is made,
the validator reports ``disparity_not_measurable`` as a could-not-check in so many
words, the test suite returns SKIPPED rather than PASSED for a check it could not
run, and the CLI entry point emits parseable JSON with ``success: false`` on every
bad input instead of raising.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd import task_handlers
from vfairness.operations.cicd.gate import FairnessReportCard, ModelFairnessGate
from vfairness.operations.cicd.testing import FairnessTestSuite, TestStatus
from vfairness.operations.cicd.validator import DataBiasValidator
from vfairness.operations.experimentation.experiment import (
    FairnessExperiment,
    assign_clusters,
    create_factorial_design,
)

THRESHOLDS = {"demographic_parity_difference": 0.05}


def _rng():
    return np.random.default_rng(3)


def _two_group_frame(n: int = 300):
    rng = _rng()
    return rng.integers(0, 2, n), rng.integers(0, 2, n), rng.choice(["a", "b"], n)


# ---------------------------------------------------------------------------
# The gate's serialisers, on a metric that could NOT be computed
# ---------------------------------------------------------------------------


def _single_group_decision():
    """One group in the data. No disparity is definable, so nothing is measurable."""
    rng = _rng()
    n = 200
    return ModelFairnessGate(thresholds=THRESHOLDS).evaluate(
        rng.integers(0, 2, n), rng.integers(0, 2, n), np.array(["a"] * n)
    )


def test_an_unmeasurable_metric_reaches_the_dict_as_nan_not_zero():
    decision = _single_group_decision()
    payload = decision.to_dict()

    evaluation = payload["metric_evaluations"][0]
    assert np.isnan(evaluation["value"]), (
        "a metric that could not be computed was serialised as a number, which a "
        "consumer cannot distinguish from a measurement"
    )
    assert evaluation["passed"] is False
    assert payload["approved"] is False


def test_the_pr_comment_states_why_rather_than_only_that_it_failed():
    """The reviewer must be able to tell "your model is unfair" from "this gate
    measured nothing". The actions are completely different."""
    decision = _single_group_decision()
    markdown = FairnessReportCard(decision, model_name="loan-v3").to_markdown()

    assert "could not be computed" in markdown, (
        "the PR comment showed a FAIL for a metric that was never computed, with no "
        "statement anywhere that it was not computed"
    )
    assert "fails closed" in markdown


def test_the_report_card_carries_small_sample_warnings_to_the_reviewer():
    """A group of five people is the fact that decides a merge."""
    rng = _rng()
    n = 200
    protected = np.array(["a"] * (n - 5) + ["b"] * 5)
    decision = ModelFairnessGate(thresholds=THRESHOLDS).evaluate(
        rng.integers(0, 2, n), rng.integers(0, 2, n), protected
    )

    assert decision.small_sample_warnings, "the fixture no longer trips the guard"
    markdown = FairnessReportCard(decision, model_name="m").to_markdown()
    assert "Small-Sample Warnings" in markdown
    assert "5 samples" in markdown


def test_the_github_payload_is_the_markdown_and_nothing_less():
    decision = _single_group_decision()
    card = FairnessReportCard(decision, model_name="m")
    assert card.to_github_comment_payload() == {"body": card.to_markdown()}


def test_the_report_card_dict_agrees_with_the_decision_it_wraps():
    decision = _single_group_decision()
    payload = FairnessReportCard(decision, model_name="m").to_dict()
    assert payload["approved"] is decision.approved
    assert payload["status"] == decision.status.value
    assert payload["model_name"] == "m"
    assert payload["markdown"]


# ---------------------------------------------------------------------------
# The hierarchical decision, and the field its serialiser used to drop
# ---------------------------------------------------------------------------


def _hierarchical_decision():
    rng = _rng()
    n = 300
    return ModelFairnessGate(thresholds=THRESHOLDS).evaluate_hierarchical(
        rng.integers(0, 2, n),
        rng.integers(0, 2, n),
        {"g": rng.choice(["a", "b"], n), "g2": rng.choice(["x", "y"], n)},
    )


def test_the_hierarchical_dict_carries_every_field_the_dataclass_declares():
    import dataclasses

    decision = _hierarchical_decision()
    payload = decision.to_dict()
    declared = {f.name for f in dataclasses.fields(decision)}

    assert declared <= set(payload), (
        f"the serialiser drops {sorted(declared - set(payload))}; a field absent from "
        f"the dict is a field the consumer cannot know existed"
    )


def test_the_hierarchical_markdown_reports_every_level():
    decision = _hierarchical_decision()
    report = decision.to_markdown_report()
    for level in decision.level_results:
        assert level in report, f"level {level!r} was evaluated but not reported"


# ---------------------------------------------------------------------------
# Validator and test suite
# ---------------------------------------------------------------------------


def test_the_validator_calls_an_unmeasurable_disparity_what_it_is():
    rng = _rng()
    n = 200
    frame = pd.DataFrame({"g": ["a"] * n, "y": rng.integers(0, 2, n)})
    result = DataBiasValidator(protected_attributes=["g"]).validate(frame, outcome_column="y")

    payload = result.to_dict()
    kinds = {issue["issue_type"] for issue in payload["issues"]}
    assert "disparity_not_measurable" in kinds
    assert payload["passed"] is False

    message = next(
        i["message"] for i in payload["issues"] if i["issue_type"] == "disparity_not_measurable"
    )
    assert "could-not-check" in message


def test_validate_batch_reports_one_result_per_frame():
    rng = _rng()
    n = 120
    frame = pd.DataFrame({"g": rng.choice(["a", "b"], n), "y": rng.integers(0, 2, n)})
    results = DataBiasValidator(protected_attributes=["g"]).validate_batch(
        [frame, frame.head(20)], outcome_column="y"
    )

    assert len(results) == 2, "a frame in the batch produced no result at all"
    assert set(results) == {"dataset_0", "dataset_1"}


def test_the_validator_explanation_is_about_the_result_it_was_given():
    rng = _rng()
    n = 120
    frame = pd.DataFrame({"g": rng.choice(["a", "b"], n), "y": rng.integers(0, 2, n)})
    validator = DataBiasValidator(protected_attributes=["g"])
    result = validator.validate(frame, outcome_column="y")
    assert str(validator.get_explanation(result)).strip()


def test_the_gate_explanation_is_about_the_decision_it_was_given():
    gate = ModelFairnessGate(thresholds=THRESHOLDS)
    assert str(gate.get_explanation(_single_group_decision())).strip()


def test_a_check_that_could_not_run_is_skipped_and_never_passed():
    """One group in the frame, so parity is undefined. SKIPPED, not PASSED."""
    rng = _rng()
    n = 200

    class _Constant:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

    results = FairnessTestSuite(protected_attributes=["g"]).test_model(
        _Constant(),
        pd.DataFrame({"g": ["a"] * n, "f": rng.random(n), "y": rng.integers(0, 2, n)}),
        target_column="y",
    )

    assert results, "the suite returned no results at all"
    assert all(r.status is TestStatus.SKIPPED for r in results), (
        f"a check that could not run reported {[r.status for r in results]}; PASSED "
        f"would certify a model against a test that never executed"
    )


def test_control_a_two_group_frame_actually_runs_the_check():
    """The over-correction control: skipping everything would pass the test above."""
    rng = _rng()
    n = 200

    class _Biased:
        def predict(self, X):
            return (np.asarray(X["f"]) > 0.5).astype(int)

    frame = pd.DataFrame(
        {"g": rng.choice(["a", "b"], n), "f": rng.random(n), "y": rng.integers(0, 2, n)}
    )
    results = FairnessTestSuite(protected_attributes=["g"]).test_model(
        _Biased(), frame, target_column="y"
    )
    assert any(r.status is not TestStatus.SKIPPED for r in results)


# ---------------------------------------------------------------------------
# The CLI entry point
# ---------------------------------------------------------------------------


def _run_main(argv: list[str], stdin_text: str) -> tuple[int, str]:
    saved = (sys.argv, sys.stdin, sys.stdout)
    sys.argv, sys.stdin, sys.stdout = argv, io.StringIO(stdin_text), io.StringIO()
    try:
        code = task_handlers.main()
        return code, sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdin, sys.stdout = saved


@pytest.mark.parametrize(
    "argv,stdin_text,expected_code",
    [
        # BGL5, 2026-09-27: an empty payload is a refusal ("No dataset
        # provided"), and a refusal now exits 1 like the other two. It exited 0
        # until the audit, so a CI step could not tell it from a clean run. The
        # subject of this test is unchanged: every answer is parseable JSON with
        # success False and a stated error.
        (["prog"], "", 1),
        (["prog", "bogus_task"], "{}", 1),
        (["prog"], "{not json", 1),
    ],
)
def test_the_cli_always_answers_in_parseable_json(argv, stdin_text, expected_code):
    """The consumer on the other end parses stdout. A traceback, or a bare NaN
    token, is an unparseable answer and reads as an infrastructure failure rather
    than as the refusal it is."""
    code, out = _run_main(argv, stdin_text)

    assert code == expected_code
    payload = json.loads(out)  # raises if NaN/Infinity or a traceback got out
    assert payload["success"] is False
    assert payload["error"], "a failure with no error message"


def test_the_cli_reports_a_missing_protected_attribute_as_a_refusal():
    """Same subject as before (the CLI must call this a refusal, not a pass),
    corrected mechanism. This asserted ``code == 0`` until the BGL5 audit, which
    is the interface a CI step reads: measured 2026-09-27 the run printed
    ``"success": false`` and exited 0, so the refusal was indistinguishable from
    a clean validation to a shell. It exits 1 now, and stdout is unchanged.
    """
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    code, out = _run_main(
        ["prog"],
        json.dumps({"csv_data": frame.to_csv(index=False), "protected_attributes": ["not_here"]}),
    )
    payload = json.loads(out)
    assert code == 1
    assert payload["success"] is False
    assert "cannot be validated" in payload["error"] or "protected" in payload["error"].lower()


# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------


def _frame(n: int = 400, clusters: int = 20):
    rng = _rng()
    return pd.DataFrame(
        {
            "g": rng.choice(["a", "b"], n),
            "y": rng.integers(0, 2, n),
            "cl": rng.integers(0, clusters, n),
        }
    )


def test_cluster_randomisation_over_one_cluster_says_it_is_not_an_experiment():
    frame = _frame().assign(cl=0)

    with pytest.warns(UserWarning, match="SINGLE ARM"):
        out = assign_clusters(frame, "cl", treatment_fraction=0.5)

    assert out["treatment"].nunique() == 1, "the fixture no longer produces one arm"


def test_a_stratum_that_cannot_be_randomised_is_disclosed():
    """Every singleton stratum went to treatment, delivering 1.0 for a requested 0.5."""
    n = 400
    frame = pd.DataFrame(
        {
            "y": [0, 1] * (n // 2),
            "cl": [f"c{i % 10}" for i in range(n)],
            "s": [f"s{i % 10}" for i in range(n)],
        }
    )
    with pytest.warns(UserWarning, match="COULD NOT BE PERFORMED"):
        assign_clusters(frame, "cl", treatment_fraction=0.5, stratify_by=["s"])


def test_cluster_randomisation_leaves_a_control_arm_whenever_it_can():
    """Two clusters is the minimum that can fill both arms, so both must be filled,
    and silently: a warning here would be noise."""
    frame = _frame(clusters=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = assign_clusters(frame, "cl", treatment_fraction=0.5)

    assert set(out["treatment"].unique()) == {0, 1}
    assert out.groupby("cl")["treatment"].nunique().max() == 1, (
        "assignment varied WITHIN a cluster, which defeats the point of cluster "
        "randomisation: spillover is exactly what it exists to avoid"
    )


def test_a_deliberate_full_rollout_is_honoured_and_disclosed():
    """treatment_fraction >= 1.0 is a rollout, not a mistake. Do not clamp it, but
    do say that there is nothing to compare against."""
    frame = _frame(clusters=4)

    with pytest.warns(UserWarning, match="SINGLE ARM"):
        out = assign_clusters(frame, "cl", treatment_fraction=1.0)

    assert set(out["treatment"].unique()) == {1}


def test_cluster_assignment_is_reproducible_under_a_seed():
    frame = _frame()
    first = assign_clusters(frame, "cl", 0.5, random_state=17)["treatment"]
    second = assign_clusters(frame, "cl", 0.5, random_state=17)["treatment"]
    pd.testing.assert_series_equal(first, second)


def test_the_factorial_design_assigns_every_row_to_a_cell():
    frame = _frame()
    out = create_factorial_design(frame, ["g"], random_state=1)

    assert "treatment_cell" in out.columns
    assert len(out) == len(frame)
    assert out["treatment_cell"].notna().all(), "a row was left without a cell"


# ---------------------------------------------------------------------------
# Experiment results
# ---------------------------------------------------------------------------


def _experiment():
    rng = _rng()
    n = 300

    def arm(shift: float):
        return pd.DataFrame(
            {
                "g": rng.choice(["a", "b"], n),
                "y": rng.binomial(1, np.clip(0.4 + shift, 0, 1), n),
                "rev": rng.random(n) * 100,
            }
        )

    return FairnessExperiment(
        arm(0.0),
        arm(0.08),
        protected_attributes=["g"],
        outcome_column="y",
        business_metrics=["rev"],
    )


def test_the_experiment_frame_has_one_row_per_intersection():
    experiment = _experiment()
    frame = experiment.to_dataframe()
    assert not frame.empty
    assert "intersection" in frame.columns
    # The cell holds the tuple of group values, so hash it before asking.
    keys = [tuple(v) if isinstance(v, list) else v for v in frame["intersection"]]
    assert len(keys) == len(set(keys)), f"an intersection appears twice: {keys}"


def test_the_result_frame_and_the_saved_file_agree_with_the_analysis(
    tmp_path: pathlib.Path,
):
    result = _experiment().run_full_analysis()

    frame = result.to_dataframe()
    assert len(frame) == result.n_intersections

    destination = tmp_path / "experiment.json"
    result.save(str(destination))
    saved = json.loads(destination.read_text(encoding="utf-8"))  # raises on NaN tokens
    assert saved, "save() wrote an empty document"
