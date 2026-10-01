"""BGL-3, batch operations-3: monitoring, CI/CD gates, reporting, dashboards.

Fourteen units examined by execution on 2026-09-27 for one defect shape: a
quantity that cannot be computed, replaced by the value that reads as its clean
answer. Six units carried it and were fixed (five distinct causes, one of which
reaches two units); the other eight refuse honestly already and are pinned here
so they cannot stop.

Every test below is either a REFUSAL PIN (the surface must answer
could-not-check) or an OVER-CORRECTION CONTROL (a real measurement must still
be reported, with the numbers it reports today). Each docstring states what was
measured BEFORE the fix, in concrete numbers.

The findings:

1. ``cusum_drift`` on a constant series sitting away from an explicit
   ``target``: sigma is 0, so the excursion has no sigma-unit size, and it
   answered ``has_drift False, max_cusum 0.00`` with no warning, which
   ``sequential_fairness_drift`` turned into ``"stable"``.
2. ``BiasMonitor.get_rolling_average`` averaged whatever was in the history:
   ``True`` became 1.0 and ``False`` 0.0 (perfect parity, from a flag),
   ``None`` raised TypeError out of ``get_summary``, ``inf`` became inf.
3. ``ModelFairnessGate.create_github_check`` reported conclusion ``success``
   for a CONDITIONAL decision, which is the state a metric that could not be
   measured produces, in the one field branch protection reads.
4. ``build_assurance_verdict`` issued "Unqualified opinion: no material
   fairness defect found on the assessed attributes" when NO attribute was
   assessable, as long as ``has_truth`` was True.
5. ``compute_adverse_action_reasons`` with no feature names returned an empty
   ECOA notice with ``complete`` True and no warning.
"""

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pytest

from vfairness.operations.cicd.gate import (
    GateConfig,
    GateStatus,
    ModelFairnessGate,
)
from vfairness.operations.cicd.monitor import BiasMonitor, MonitorConfig
from vfairness.operations.monitoring.sequential import (
    cusum_drift,
    page_hinkley,
    sequential_fairness_drift,
)
from vfairness.operations.reporting import interactive as interactive_mod
from vfairness.operations.reporting.compliance import (
    build_assurance_verdict,
    compute_adverse_action_reasons,
    compute_signed_test_log,
)
from vfairness.operations.reporting.interactive import (
    InteractiveDashboard,
    simulate_threshold_change,
)
from vfairness.operations.reporting.store import MetricsStore, StoredMetricRecord

METRIC = "demographic_parity_difference"


def _caught(fn, *args, **kwargs):
    """Run *fn* and return ``(result, [warning messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ===========================================================================
# 1. sequential.py: cusum_drift, page_hinkley, sequential_fairness_drift
# ===========================================================================


def _stepped_series():
    """A 17 sigma step at t=50, the fixture the module's own pins use."""
    rng = np.random.default_rng(3)
    return np.concatenate([rng.normal(0.10, 0.01, 50), rng.normal(0.27, 0.01, 50)])


class TestCusumRefusesAnOffTargetConstantSeries:
    """MEASURED BEFORE THE FIX (2026-09-27)::

        cusum_drift([0.9] * 50, target=0.1)
            -> has_drift False, max_cusum 0.0, mean 0.9, warnings []
        sequential_fairness_drift([0.9] * 50, {"target": 0.1})
            -> classification "stable"

    A process pinned 0.8 above the reference level the caller declared, for
    fifty consecutive windows, reported with the calmest reading on the scale
    and no warning, returning the identical dict to a series genuinely sitting
    on its target.
    """

    def test_a_constant_series_away_from_its_target_is_not_stable(self):
        result, messages = _caught(cusum_drift, [0.9] * 50, 0.1)

        assert result["has_drift"] is None, "False is the calm verdict, not a refusal"
        assert math.isnan(result["max_cusum"]), "0.0 is the calmest reading on the scale"
        assert result["drift_index"] is None
        assert all(math.isnan(v) for v in result["cusum_pos"])
        assert all(math.isnan(v) for v in result["cusum_neg"])
        # The measured level and the target are both named, so a reader can see
        # the excursion the detector could not size.
        assert any("0.9" in m and "0.1" in m for m in messages), messages
        assert any("could not check" in m for m in messages), messages
        # The windows were read, and the counts still say so.
        assert result["n_measured"] == 50 and result["n_windows"] == 50
        assert result["mean"] == pytest.approx(0.9)

    def test_the_wrapper_classifies_it_not_assessed_rather_than_stable(self):
        result, messages = _caught(sequential_fairness_drift, [0.9] * 50, {"target": 0.1})

        assert result["classification"] == "not_assessed"
        assert result["cusum"]["has_drift"] is None
        assert messages, "the refusal must reach the caller, not only the dict"

    # OVER CORRECTION CONTROLS

    def test_control_a_flat_series_with_no_target_is_still_stable(self):
        """No target means the reference IS the series mean, so a flat series
        is on its reference and "no drift" is a measurement."""
        result, messages = _caught(cusum_drift, [0.9] * 50)

        assert result["has_drift"] is False
        assert result["max_cusum"] == 0.0
        assert result["cusum_pos"] == [0.0] * 50
        assert messages == [], messages

    def test_control_a_flat_series_sitting_on_its_explicit_target_is_still_stable(self):
        result, messages = _caught(cusum_drift, [0.9] * 50, 0.9)

        assert result["has_drift"] is False
        assert result["max_cusum"] == 0.0
        assert messages == [], messages

    def test_control_a_real_step_is_still_detected_with_a_target_supplied(self):
        """The refusal must not swallow a series that DOES vary. Target 0.1 is
        the pre step level, so the step is a genuine excursion from it."""
        result, messages = _caught(cusum_drift, _stepped_series(), 0.1)

        assert result["has_drift"] is True
        assert result["drift_index"] is not None
        assert result["max_cusum"] > 5.0
        assert messages == [], messages

    def test_control_the_wrapper_still_calls_a_step_abrupt_drift(self):
        result, _ = _caught(sequential_fairness_drift, _stepped_series(), {"target": 0.1})
        assert result["classification"] == "abrupt_drift"


class TestPageHinkleyKeepsItsThreeStates:
    """Verified CORRECT, not changed. Recorded here because this batch graded
    it: an empty series answers has_drift None with a nan min_ph and a warning
    naming 0 of 0 windows, a flat series answers False, and a 17 sigma step
    answers True at index 71."""

    def test_an_unmeasurable_series_is_refused_and_named(self):
        empty, messages = _caught(page_hinkley, [])
        assert empty["has_drift"] is None
        assert math.isnan(empty["min_ph"])
        assert any("only 0 of 0" in m for m in messages), messages

        all_nan, messages = _caught(page_hinkley, [float("nan")] * 10)
        assert all_nan["has_drift"] is None
        assert all_nan["n_measured"] == 0 and all_nan["n_windows"] == 10
        assert any("only 0 of 10" in m for m in messages), messages

    def test_control_a_step_is_still_detected_and_a_flat_series_is_not(self):
        assert page_hinkley(_stepped_series())["has_drift"] is True
        assert page_hinkley([0.25] * 50)["has_drift"] is False


# ===========================================================================
# 2. monitor.py: get_rolling_average
# ===========================================================================


def _monitor(compute_metrics_fn=None, window_size=10):
    return BiasMonitor(
        baseline_metrics={METRIC: 0.5},
        config=MonitorConfig(
            min_samples_for_alert=1,
            window_size=window_size,
            metrics_to_monitor=[METRIC],
        ),
        compute_metrics_fn=compute_metrics_fn,
    )


def _two_group_batch(n=100, rate_a=0.9, rate_b=0.4):
    a = [1] * int(n * rate_a) + [0] * (n - int(n * rate_a))
    b = [1] * int(n * rate_b) + [0] * (n - int(n * rate_b))
    return np.array(a + b), np.array([1] * (2 * n)), np.array(["A"] * n + ["B"] * n)


def _one_group_batch(n=100):
    return np.array([1] * n), np.array([1] * n), np.array(["A"] * n)


def _log(monitor, batch, batch_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return monitor.log_batch(batch[0], batch[1], batch[2], batch_id=batch_id)


class TestRollingAverageOnlyAveragesMeasurements:
    """MEASURED BEFORE THE FIX (2026-09-27), all through the documented
    ``compute_metrics_fn`` parameter::

        {metric: True}   -> history [True]       -> rolling average 1.0
        {metric: False}  -> history [False]      -> rolling average 0.0
        {metric: None}   -> history [0.5, None]  -> TypeError: unsupported
                            operand type(s) for +: 'float' and 'NoneType'
        {metric: inf}    -> history [inf]        -> rolling average inf

    ``log_batch`` had already routed each of those to drift status None, so the
    monitor knew they were not measurements while this accessor published them
    as clean floats. 0.0 on a disparity scale is perfect parity.
    """

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_flag_is_not_a_rolling_average(self, flag):
        monitor = _monitor(compute_metrics_fn=lambda yt, yp, pa: {METRIC: flag})
        _log(monitor, _two_group_batch(), "b1")

        rolling = monitor.get_rolling_average()[METRIC]
        assert rolling is not None and math.isnan(rolling), f"a bool was averaged: {rolling!r}"
        # The same value is already a could-not-check on the drift channel.
        assert monitor.get_drift_status() == {METRIC: None}

    def test_a_none_window_refuses_instead_of_raising(self):
        values = iter([0.5, None])
        monitor = _monitor(compute_metrics_fn=lambda yt, yp, pa: {METRIC: next(values)})
        _log(monitor, _two_group_batch(), "b1")
        _log(monitor, _two_group_batch(), "b2")

        rolling = monitor.get_rolling_average()[METRIC]
        assert rolling is not None and math.isnan(rolling)
        # get_summary() reads this accessor, so the crash took the whole
        # monitoring summary with it.
        assert math.isnan(monitor.get_summary()["rolling_average"][METRIC])

    def test_an_infinite_window_is_not_an_average(self):
        monitor = _monitor(compute_metrics_fn=lambda yt, yp, pa: {METRIC: float("inf")})
        _log(monitor, _two_group_batch(), "b1")

        rolling = monitor.get_rolling_average()[METRIC]
        assert rolling is not None and math.isnan(rolling), "inf beats every threshold"

    # OVER CORRECTION CONTROLS

    def test_control_two_measured_windows_are_still_the_real_mean(self):
        monitor = _monitor()
        _log(monitor, _two_group_batch(), "b1")
        _log(monitor, _two_group_batch(rate_a=0.7, rate_b=0.4), "b2")

        # Gaps of 0.5 and 0.3, so the mean is 0.4, computed here and not copied
        # from the code.
        assert monitor.get_rolling_average()[METRIC] == pytest.approx(0.4)

    def test_control_the_nan_window_refusal_and_the_unseen_metric_are_unchanged(self):
        """Both states this accessor already had: a window holding a batch the
        monitor could not measure is nan, and a metric never seen is None
        (pinned in tests/test_surface_grade_g018.py, restated because the same
        branch was rewritten)."""
        never_seen = _monitor()
        assert never_seen.get_rolling_average() == {METRIC: None}

        monitor = _monitor()
        _log(monitor, _two_group_batch(), "b1")
        _log(monitor, _one_group_batch(), "b2")
        rolling = monitor.get_rolling_average()[METRIC]
        assert rolling is not None and math.isnan(rolling)


# ===========================================================================
# 3. gate.py: create_github_check, evaluate, evaluate_from_metrics,
#    evaluate_hierarchical
# ===========================================================================


def _fair_arrays(n=100):
    half = [1] * (n // 2) + [0] * (n // 2)
    return (
        np.array(half + half),
        np.array(half + half),
        np.array(["a"] * n + ["b"] * n),
    )


def _gate(**kwargs):
    kwargs.setdefault("metrics", [METRIC])
    kwargs.setdefault("thresholds", {METRIC: 0.1})
    return ModelFairnessGate(**kwargs)


class TestTheGithubCheckKeepsTheThirdState:
    """MEASURED BEFORE THE FIX (2026-09-27). A gate configured with
    ``metrics=['demographic_parity_difference', 'auroc_parity']``, a threshold
    for both and ``blocking_metrics=['demographic_parity_difference']``, run on
    100 + 100 genuinely fair rows (auroc_parity is not one of the five
    ``evaluate`` can compute, so it comes back could-not-check)::

        decision.status              GateStatus.CONDITIONAL
        decision.warnings            ['auroc_parity could not be computed ...']
        create_github_check          conclusion 'success'
        output.title                 'Model Fairness Evaluation'

    Both payload fields were byte-identical to the run where every metric WAS
    measured and passed, so the one machine-readable field branch protection
    keys off was the only surface that had dropped the third state.
    """

    def _conditional_decision(self):
        gate = ModelFairnessGate(
            metrics=[METRIC, "auroc_parity"],
            thresholds={METRIC: 0.1, "auroc_parity": 0.1},
            blocking_metrics=[METRIC],
        )
        decision = gate.evaluate(*_fair_arrays())
        assert decision.status is GateStatus.CONDITIONAL, "fixture drifted"
        assert decision.approved is True, "fixture drifted: non-blocking must not block"
        return gate, decision

    def test_a_could_not_check_metric_does_not_write_a_green_check(self):
        gate, decision = self._conditional_decision()
        check = gate.create_github_check(decision)

        assert check["conclusion"] != "success", "a green tick over an unmeasured metric"
        assert check["conclusion"] == "neutral"
        assert "not every metric could be checked" in check["output"]["title"]
        assert check["status"] == "completed"
        # The narrative half was already honest and must stay so.
        assert "could not be computed" in check["output"]["text"]

    def test_control_a_fully_measured_pass_is_still_success(self):
        gate = _gate()
        decision = gate.evaluate(*_fair_arrays())
        check = gate.create_github_check(decision)

        assert decision.status is GateStatus.APPROVED and decision.warnings == []
        assert check["conclusion"] == "success"
        assert check["output"]["title"] == "Model Fairness Evaluation"

    def test_control_a_blocked_gate_is_still_failure(self):
        gate = _gate()
        y_true, _, protected = _fair_arrays()
        unfair = np.array([1] * 100 + [0] * 100)
        decision = gate.evaluate(y_true, unfair, protected)

        assert decision.approved is False
        assert gate.create_github_check(decision)["conclusion"] == "failure"


class TestGateEntryPointsRefuseWhatTheyCouldNotMeasure:
    """Verified CORRECT, not changed. Recorded because this batch graded the
    three entry points: each returns BLOCKED with a named reason where nothing
    was measurable, rather than approving."""

    def test_evaluate_refuses_a_single_group_empty_data_and_a_five_person_group(self):
        gate = _gate()

        one_group = gate.evaluate(
            np.array([1, 0] * 50), np.array([1, 0] * 50), np.array(["a"] * 100)
        )
        assert one_group.approved is False and one_group.status is GateStatus.BLOCKED
        assert any("could not be computed" in r for r in one_group.blocking_reasons)

        empty = gate.evaluate(np.array([]), np.array([]), np.array([]))
        assert empty.approved is False

        y_true = np.array([1] * 50 + [0] * 50 + [1, 0, 1, 0, 1])
        small = gate.evaluate(y_true, y_true, np.array(["a"] * 100 + ["b"] * 5))
        assert small.approved is False
        assert [(w.group_name, w.sample_size) for w in small.small_sample_warnings] == [("b", 5)]
        assert any("could not be certified" in r for r in small.blocking_reasons)

    def test_evaluate_from_metrics_refuses_a_nan_and_an_empty_configuration(self):
        assert _gate().evaluate_from_metrics({METRIC: float("nan")}).approved is False
        nothing = ModelFairnessGate(config=GateConfig(metrics=[], thresholds={}))
        decision = nothing.evaluate_from_metrics({})
        assert decision.approved is False
        assert any("nothing was checked" in r for r in decision.blocking_reasons)
        # Control: a real pass still passes.
        assert _gate().evaluate_from_metrics({METRIC: 0.02}).approved is True

    def test_evaluate_hierarchical_refuses_when_no_level_ran(self):
        gate = _gate()
        y_true, y_pred, protected = _fair_arrays()

        nothing = gate.evaluate_hierarchical(y_true, y_pred, {})
        assert nothing.approved is False and nothing.status is GateStatus.BLOCKED
        assert "0/0 levels passed" in nothing.summary

        # Control: a fair attribute still approves at every level.
        real = gate.evaluate_hierarchical(y_true, y_pred, {"gender": protected})
        assert real.approved is True
        assert sorted(real.level_results) == ["attr:gender", "overall"]


# ===========================================================================
# 4. compliance.py: build_assurance_verdict, compute_adverse_action_reasons,
#    compute_signed_test_log
# ===========================================================================


class TestAssuranceVerdictWillNotCertifyNothing:
    """MEASURED BEFORE THE FIX (2026-09-27)::

        build_assurance_verdict(per_variable=[], has_truth=True)
        build_assurance_verdict(per_variable=[{"attribute": "race",
            "assessable": False, "reason": "groups too small"}], has_truth=True)

    Both returned overall "Unqualified", blocksDeployment False, findings [],
    unassessed [] and the one line verdict "Unqualified opinion: no material
    fairness defect found on the assessed attributes. Keep monitoring." The set
    of assessed attributes was EMPTY, and the orchestrator's own headline for
    the same state already said "No protected variable had enough grouped data
    to assess."
    """

    @pytest.mark.parametrize(
        "per_variable",
        [
            [],
            [{"attribute": "race", "assessable": False, "reason": "groups too small"}],
        ],
        ids=["no_variable_at_all", "every_variable_unassessable"],
    )
    def test_ground_truth_alone_does_not_make_an_unassessable_run_unqualified(self, per_variable):
        verdict = build_assurance_verdict(per_variable=per_variable, has_truth=True)

        assert verdict["overall"] != "Unqualified", "an all clear over nothing assessed"
        assert verdict["overall"] == "Disclaimer"
        assert "Insufficient assessable data" in verdict["oneLineVerdict"]
        assert "outcome labels alone are not enough" in verdict["oneLineVerdict"]
        assert "no material fairness defect" not in verdict["oneLineVerdict"]

    # OVER CORRECTION CONTROLS

    def test_control_an_assessable_clean_run_is_still_unqualified(self):
        verdict = build_assurance_verdict(
            per_variable=[
                {
                    "attribute": "race",
                    "assessable": True,
                    "gap": 0.01,
                    "significant": False,
                    "worstGroup": "B",
                    "referenceGroup": "W",
                }
            ],
            has_truth=True,
        )
        assert verdict["overall"] == "Unqualified"
        assert verdict["blocksDeployment"] is False
        assert verdict["findings"] == []

    def test_control_a_real_gap_is_still_adverse(self):
        verdict = build_assurance_verdict(
            per_variable=[
                {
                    "attribute": "race",
                    "assessable": True,
                    "gap": 0.23,
                    "significant": True,
                    "worstGroup": "Black",
                    "referenceGroup": "White",
                    "ciLow": 0.1,
                    "ciHigh": 0.3,
                }
            ],
            has_truth=True,
            domain="hiring",
            jurisdiction="US",
        )
        assert verdict["overall"] == "Adverse"
        assert verdict["blocksDeployment"] is True

    def test_control_the_no_ground_truth_disclaimer_sentence_is_unchanged(self):
        """The wording other tests assert on (tests/test_audit6_lane1_
        reporting.py) must survive the new branch."""
        verdict = build_assurance_verdict()
        assert verdict["overall"] == "Disclaimer"
        assert verdict["oneLineVerdict"] == (
            "Insufficient assessable data to issue a fairness opinion. Provide "
            "grouped protected attributes and, ideally, outcome labels."
        )

    def test_control_a_material_finding_without_any_per_variable_row_still_qualifies(self):
        """The Disclaimer branch is gated on there being no material finding
        either, and removing ``not has_truth`` must not change that: a PII
        leakage finding still drives the opinion with per_variable empty."""
        verdict = build_assurance_verdict(
            schema={"pii_leakage": ["ssn", "email"]},
            per_variable=[],
            has_truth=True,
        )
        assert verdict["overall"] == "Qualified"
        assert [f["type"] for f in verdict["findings"]] == ["identity_pii_leakage"]


_SHAP = {
    "income": -0.3,
    "zip_code": -0.25,
    "credit_score": -0.2,
    "age_proxy": -0.15,
    "employment": -0.1,
}


class TestAdverseActionNoticeSaysWhenNothingWasExamined:
    """MEASURED BEFORE THE FIX (2026-09-27)::

        compute_adverse_action_reasons(_SHAP, [], [])   -> [] ; complete True ;
            unattributed_features [] ; repr '[]' ; warnings []

    An ECOA Reg B notice that ranked nothing, asserting through ``complete``
    that "the ranking considered the whole model", and byte-identical to the
    notice built from five fully attributed features none of which was adverse.
    """

    def test_no_feature_names_is_not_a_complete_notice(self):
        reasons, messages = _caught(compute_adverse_action_reasons, _SHAP, [], [])

        assert list(reasons) == []
        assert reasons.complete is False, "nothing was examined"
        assert reasons.features_examined == 0
        assert "no feature was examined" in repr(reasons)
        assert any("NOTHING was examined" in m for m in messages), messages
        assert any("could-not-check" in m for m in messages), messages

    # OVER CORRECTION CONTROLS

    def test_control_a_fully_attributed_notice_is_still_complete(self):
        reasons, messages = _caught(
            compute_adverse_action_reasons, _SHAP, list(_SHAP), ["zip_code", "age_proxy"]
        )

        assert [r["feature"] for r in reasons] == [
            "income",
            "credit_score",
            "employment",
            "zip_code",
        ]
        assert reasons.complete is True
        assert reasons.features_examined == 5
        assert messages == [], messages

    def test_control_a_notice_with_no_adverse_factor_is_still_complete(self):
        """Every feature attributed, none of them adverse: an empty notice that
        IS a measurement, and it must not be confused with the refusal above."""
        positive = {k: abs(v) for k, v in _SHAP.items()}
        reasons, messages = _caught(compute_adverse_action_reasons, positive, list(positive), [])
        assert list(reasons) == []
        assert reasons.complete is True
        assert reasons.features_examined == 5
        assert messages == [], messages

    def test_control_an_unattributed_feature_is_still_named(self):
        without_income = {k: v for k, v in _SHAP.items() if k != "income"}
        reasons, messages = _caught(compute_adverse_action_reasons, without_income, list(_SHAP), [])
        assert reasons.unattributed_features == ["income"]
        assert reasons.complete is False
        assert "INCOMPLETE: unattributed" in repr(reasons)
        assert any("could not be ranked" in m for m in messages), messages


class TestSignedTestLogKeepsTheThirdState:
    """Verified CORRECT, not changed. Recorded because this batch graded it: a
    metric row with no reported verdict is counted as ``not_assessed`` (never
    as a pass or a fail), the count travels INSIDE the hashed summary, and
    ``overall_pass`` is None when the caller reported none."""

    def test_a_metric_with_no_verdict_is_counted_as_not_assessed(self):
        log = compute_signed_test_log(
            {"metrics": [{"name": "dp", "value": 0.4, "threshold": 0.1}]},
            "d" * 8,
            "0.1.0",
        )
        summary = log["test_results_summary"]
        assert summary == {
            "total_metrics": 1,
            "passed": 0,
            "failed": 0,
            "not_assessed": 1,
            "derived": 0,
            "overall_pass": None,
        }
        row = log["metrics_snapshot"][0]
        assert row["passed"] is None
        assert row["verdict_basis"] == "unassessed"
        assert row["verdict_note"]

    def test_control_two_graded_metrics_still_count_one_each(self):
        log = compute_signed_test_log(
            {
                "metrics": [
                    {"name": "dp", "value": 0.02, "threshold": 0.1, "passed": True},
                    {"name": "eo", "value": 0.4, "threshold": 0.1, "passed": False},
                ],
                "overall_pass": False,
            },
            "d" * 8,
            "0.1.0",
        )
        summary = log["test_results_summary"]
        assert (summary["passed"], summary["failed"], summary["not_assessed"]) == (1, 1, 0)
        assert summary["overall_pass"] is False
        assert len(log["content_hash"]) == 64


# ===========================================================================
# 5. interactive.py: create_dash_app, both simulate_threshold_change surfaces
# ===========================================================================


class _Node:
    """Stand in for a dash html/dcc component: records children and kwargs."""

    def __init__(self, children=None, **kwargs):
        self.children = children
        self.kwargs = kwargs

    def text(self):
        parts = []
        child = self.children
        if isinstance(child, str):
            parts.append(child)
        elif isinstance(child, _Node):
            parts.append(child.text())
        elif isinstance(child, (list, tuple)):
            for item in child:
                parts.append(item.text() if isinstance(item, _Node) else str(item))
        return " ".join(p for p in parts if p)


class _Namespace:
    def __getattr__(self, name):
        return _Node


class _RecordingApp:
    def __init__(self, *args, **kwargs):
        self.layout = None
        self.recorded = {}

    def callback(self, *args, **kwargs):
        def decorate(fn):
            self.recorded[fn.__name__] = fn
            return fn

        return decorate


class _FakeDash:
    Dash = _RecordingApp


def _dash_callbacks(monkeypatch, store):
    """Build the REAL Dash app with dash's own component classes stubbed, and
    hand back its registered callbacks. dash is an optional dependency and is
    not installed here, so the layout objects are doubles; the callback bodies
    executed below are the shipped ones. Same harness as
    tests/test_readiness5_consumers5.py."""
    monkeypatch.setattr(interactive_mod, "dash", _FakeDash(), raising=False)
    monkeypatch.setattr(interactive_mod, "html", _Namespace(), raising=False)
    monkeypatch.setattr(interactive_mod, "dcc", _Namespace(), raising=False)
    monkeypatch.setattr(interactive_mod, "Input", lambda *a, **k: ("in", a), raising=False)
    monkeypatch.setattr(interactive_mod, "Output", lambda *a, **k: ("out", a), raising=False)
    monkeypatch.setattr(interactive_mod, "_DASH_AVAILABLE", True, raising=False)
    return InteractiveDashboard(store).create_dash_app().recorded


def _store(rows):
    """``rows`` are ``(value, alert, group)``; ``alert=None`` means the record
    was never compared to a threshold."""
    store = MetricsStore()
    now = datetime.now()
    for value, alert, group in rows:
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=1),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=float(value),
                group=group,
                alert=alert,
            )
        )
    return store


class TestInteractiveSurfacesRefuseAnEmptyWindow:
    """Verified CORRECT, not changed. Recorded because this batch graded all
    three: on an EMPTY store the health banner reads "not assessed" (not a
    score of 100), the what if panel reads "Could not check", and both
    simulation surfaces answer None for every count with a stated reason, while
    a store holding graded rows still reports its real numbers."""

    def test_the_dash_callbacks_refuse_an_empty_store(self, monkeypatch):
        callbacks = _dash_callbacks(monkeypatch, MetricsStore())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            banner = callbacks["update_charts"]("demographic_parity", "30d", ["overall"])[0]
            whatif = callbacks["update_whatif"](0.5, "demographic_parity")

        assert "not assessed" in banner.text()
        assert "not a score of 100" in banner.text()
        assert "Could not check" in whatif.text()
        assert "Projected alerts:" not in whatif.text()

    def test_control_the_dash_callbacks_still_render_measured_numbers(self, monkeypatch):
        store = _store([(0.02, False, "a"), (0.03, False, "a"), (0.95, True, "b")])
        callbacks = _dash_callbacks(monkeypatch, store)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            banner = callbacks["update_charts"]("demographic_parity", "30d", ["a", "b"])[0]
            whatif = callbacks["update_whatif"](0.5, "demographic_parity")

        assert "/100" in banner.text() and "not assessed" not in banner.text()
        assert "Current alerts: 1" in whatif.text()
        assert "Could not check" not in whatif.text()

    def test_both_simulation_surfaces_refuse_a_window_with_no_record(self):
        empty = MetricsStore()
        direct = simulate_threshold_change(empty, "demographic_parity", 0.5)
        wrapped = InteractiveDashboard(empty).simulate_threshold_change("demographic_parity", 0.5)

        for result in (direct, wrapped):
            assert result["current_alerts"] is None
            assert result["projected_alerts"] is None
            assert result["groups_impacted"] is None, "[] would read as no group impacted"
            assert result["change_abs"] is None and result["change_pct"] is None
            assert result["records_simulated"] == 0
            assert "COULD NOT CHECK" in result["not_simulated_reason"]

    def test_control_the_wrapper_still_reports_a_measured_simulation(self):
        store = _store([(0.02, False, "a"), (0.95, True, "b"), (0.9, True, "b")])
        result = InteractiveDashboard(store).simulate_threshold_change("demographic_parity", 0.5)

        assert result["current_alerts"] == 2
        assert result["projected_alerts"] == 2
        assert result["groups_impacted"] == ["b"]
        assert result["records_simulated"] == 3
        assert result["not_simulated_reason"] == ""

    def test_create_dash_app_refuses_rather_than_faking_an_app_without_dash(self):
        """dash is an optional dependency and is absent here, so the honest
        answer is a refusal naming it, not a partial app."""
        if interactive_mod._DASH_AVAILABLE:
            pytest.skip("dash is installed in this environment")
        with pytest.raises(ImportError, match="Dash is required"):
            InteractiveDashboard(MetricsStore()).create_dash_app()
