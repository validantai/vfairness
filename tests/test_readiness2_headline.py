"""Readiness-2 headline: a model that denies a protected class outright must not
read as fair.

Reproduced by execution on 2026-09-10, with the PUBLIC API and DEFAULT settings,
on 2302 applicants whose protected attribute has an ordinary long tail (38 levels
of 29 people, 1102 applicants, 47.9 percent of the data, DENIED OUTRIGHT):

  * ``FairnessAnalyzer.get_report()`` -> fairness_score 1.0, 5 passed, 0 failed.
  * ``assert_fairness``, the documented CI/CD gate, RETURNED
    ``{'demographic_parity_difference': 0.0067, ...}`` and did not raise.
  * ``ModelFairnessGate.evaluate``, which applies no size gate, BLOCKED the same
    run (disparate_impact_ratio 0.00). Two surfaces of one library disagreed
    about one run and the stricter one was right.
  * The regression path dropped the same shape of group with ZERO warnings, at
    every one of its eight GroupManager sites, while classification and ranking
    both warn. Measured there: the excluded tail's MAE is 99,728 against 398 for
    the survivors, a factor of 251, and the report read 4/4 within thresholds.
  * With the tail kept (``min_group_size=1``) the truth is
    demographic_parity_difference 0.532 and disparate_impact_ratio 0.00.

The rule these pins encode: a protected level the size gate removed was NEVER
CHECKED, and a check that did not happen is could-not-check. Could-not-check is
its own state; it is not a pass, and on a release gate it fails.

Every refusal pin below has an OVER-CORRECTION CONTROL next to it that asserts
MEASURED values on fully sampled data, because a gate that refuses everything is
as useless as one that passes everything, and much easier to ship by accident.
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
)
from vfairness.evaluation.vfairness_metrics.integrations import (
    FairnessAssertionError,
    assert_fairness,
    create_fairness_callback,
)
from vfairness.evaluation.vfairness_metrics.regression import (
    compute_regression_effect_sizes,
    get_group_metrics,
    mae_parity_difference,
    mean_prediction_difference,
    pricing_disparity,
    r2_parity_difference,
    residual_bias,
    rmse_parity_difference,
)

# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------


def _denied_class_data():
    """The reviewer's dataset, rebuilt exactly.

    Two majority levels of 600 treated alike, plus 38 tail levels of 29 people
    (1102 rows, 47.9 percent) that the model denies without exception. Every
    tail level sits one person below the default ``min_group_size=30``.
    """
    rng = np.random.default_rng(7)
    y_true: list = []
    y_pred: list = []
    attr: list = []
    for name in ("majority_a", "majority_b"):
        y_true.extend(rng.integers(0, 2, size=600).tolist())
        y_pred.extend((rng.random(600) < 0.5).astype(int).tolist())
        attr.extend([name] * 600)
    for i in range(38):
        y_true.extend(rng.integers(0, 2, size=29).tolist())
        y_pred.extend([0] * 29)
        attr.extend([f"minority_{i:02d}"] * 29)
    return np.array(y_true), np.array(y_pred), np.array(attr, dtype=object)


def _fully_sampled_fair_data(n=400):
    """Two groups of n, predictions equal to the labels: genuinely fair, fully
    measurable, and no group anywhere near the size gate. THE control."""
    sens = np.array(["A"] * n + ["B"] * n)
    y_true = np.array([1, 0] * n)
    y_pred = y_true.copy()
    return y_true, y_pred, sens


def _fully_sampled_unfair_data(n=200):
    """Two large groups, one selected always and one never: a MEASURED, maximal
    demographic-parity breach, with nothing excluded."""
    sens = np.array(["A"] * n + ["B"] * n)
    y_true = np.array([1, 0] * n)
    y_pred = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return y_true, y_pred, sens


def _regression_dropped_group_data():
    """Two groups of 60 predicted exactly, plus a group of 5 predicted at 3x.

    Deterministic on purpose: the survivors' MAE is exactly 0.0 each, so the
    measured mae_parity_difference over survivors is exactly 0.0, while the
    dropped group's MAE is 2x its own mean. Nothing here is noise.
    """
    base = np.arange(1.0, 61.0)
    y_true = np.concatenate([base, base, base[:5]])
    y_pred = np.concatenate([base, base, base[:5] * 3.0])
    sens = np.array(["A"] * 60 + ["B"] * 60 + ["C"] * 5)
    return y_true, y_pred, sens


def _regression_fully_sampled_data():
    """Two groups of 60, group B off by exactly 3.0 and group A exact.

    mae_parity_difference is therefore exactly 3.0, a measured number, and no
    group is dropped.
    """
    base = np.arange(1.0, 61.0)
    y_true = np.concatenate([base, base])
    y_pred = np.concatenate([base, base + 3.0])
    sens = np.array(["A"] * 60 + ["B"] * 60)
    return y_true, y_pred, sens


def _quiet():
    return warnings.catch_warnings()


# ---------------------------------------------------------------------------
# Part 1: the release gate must not certify a level it never checked
# ---------------------------------------------------------------------------


class TestTheGateRefusesAnUncheckedProtectedClass:
    def test_the_denied_class_run_raises(self):
        """THE acceptance repro. Before the fix this RETURNED the three metric
        values and did not raise, on a model that denied 1102 applicants."""
        y_true, y_pred, attr = _denied_class_data()
        with _quiet():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(y_true, y_pred, attr)

        failed = excinfo.value.failed_metrics
        assert set(failed) == {
            "demographic_parity_difference",
            "equalized_odds_difference",
            "equal_opportunity_difference",
        }
        for name, failure in failed.items():
            assert failure["not_measurable"] is True, name
            assert "NOT MEASURABLE" in failure["reason"], name

    def test_the_reason_states_how_much_data_left_the_comparison(self):
        """A count of groups reads like a rounding detail. The share of rows is
        the number that decides whether the verdict means anything, and it is
        exactly what the passing run never showed anyone."""
        y_true, y_pred, attr = _denied_class_data()
        with _quiet():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(y_true, y_pred, attr)

        reason = excinfo.value.failed_metrics["demographic_parity_difference"]["reason"]
        assert "1102 of 2302 rows" in reason
        assert "47.9 percent" in reason
        assert "min_group_size=30" in reason
        # And the value it DID measure is still reported, so the caller can tell
        # what was compared from what was not.
        assert excinfo.value.failed_metrics["demographic_parity_difference"][
            "value"
        ] == pytest.approx(0.006666666666666599)

    def test_the_error_names_the_excluded_groups_and_the_covered_ones(self):
        y_true, y_pred, attr = _denied_class_data()
        with _quiet():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(y_true, y_pred, attr)

        text = str(excinfo.value)
        assert "Excluded by min_group_size=30" in text
        assert "minority_00=29" in text
        assert "and 30 more" in text
        assert "'majority_a', 'majority_b'" in text
        assert "nothing here certifies them" in text

    def test_a_fully_sampled_fair_model_still_passes_with_measured_values(self):
        """OVER-CORRECTION CONTROL. Fully sampled and genuinely fair: the gate
        must still pass, and the values it returns must be the MEASURED zeros,
        not a membership test against some permissive set."""
        y_true, y_pred, sens = _fully_sampled_fair_data()
        values = assert_fairness(y_true, y_pred, sens)
        assert set(values) == {
            "demographic_parity_difference",
            "equalized_odds_difference",
            "equal_opportunity_difference",
        }
        for name, value in values.items():
            assert float(value) == pytest.approx(0.0, abs=1e-12), f"{name}={value}"

    def test_the_report_on_that_same_fair_data_is_still_100_of_100(self):
        """OVER-CORRECTION CONTROL, the other headline surface: a fair model on
        fully sampled data must still score exactly 1.0 with 5 passed metrics."""
        y_true, y_pred, sens = _fully_sampled_fair_data()
        report = FairnessAnalyzer(y_true, y_pred, sens).get_report()
        assessment = report["assessment"]
        assert assessment["fairness_score"] == pytest.approx(1.0)
        assert assessment["assessable"] is True
        assert len(assessment["passed_metrics"]) == 5
        assert assessment["failed_metrics"] == []
        assert assessment["insufficient_evidence_groups"] == []

    def test_a_measured_breach_is_still_a_breach_not_could_not_check(self):
        """THREE STATES. A real violation must not be relabelled as could not
        check: the two carry different reasons and have different fixes."""
        y_true, y_pred, sens = _fully_sampled_unfair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(y_true, y_pred, sens, metrics=["demographic_parity_difference"])
        failure = excinfo.value.failed_metrics["demographic_parity_difference"]
        assert failure["not_measurable"] is False
        assert failure["value"] == pytest.approx(1.0)
        assert "NOT MEASURABLE" not in failure["reason"]
        assert "exceeds threshold" in failure["reason"]

    def test_measuring_the_tail_turns_could_not_check_into_a_measured_breach(self):
        """The escape hatch the error message offers must actually work, and it
        must reach the truth, not merely stop complaining.

        With min_group_size=1 every level is measured: the gate still fails, but
        now as a MEASURED breach at 0.532, the value the size gate was hiding.
        """
        y_true, y_pred, attr = _denied_class_data()
        with _quiet():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(
                    y_true,
                    y_pred,
                    attr,
                    metrics=["demographic_parity_difference"],
                    min_group_size=1,
                )
        failure = excinfo.value.failed_metrics["demographic_parity_difference"]
        assert failure["not_measurable"] is False
        assert failure["value"] == pytest.approx(0.5316666666666666)
        assert demographic_parity_difference(
            y_true, y_pred, attr, min_group_size=1
        ) == pytest.approx(0.5316666666666666)


# ---------------------------------------------------------------------------
# Part 2: the regression path must announce the drop, like classification does
# ---------------------------------------------------------------------------


def _regression_calls(y_true, y_pred, sens):
    """Every public regression entry point that applies the size gate."""
    return {
        "mae_parity_difference": lambda: mae_parity_difference(y_true, y_pred, sens),
        "rmse_parity_difference": lambda: rmse_parity_difference(y_true, y_pred, sens),
        "mean_prediction_difference": lambda: mean_prediction_difference(y_true, y_pred, sens),
        "r2_parity_difference": lambda: r2_parity_difference(y_true, y_pred, sens),
        "residual_bias": lambda: residual_bias(y_true, y_pred, sens),
        "get_group_metrics": lambda: get_group_metrics(y_true, y_pred, sens),
        "compute_regression_effect_sizes": lambda: compute_regression_effect_sizes(
            y_true, y_pred, sens
        ),
        "pricing_disparity": lambda: pricing_disparity(y_pred, sens),
    }


_REGRESSION_ENTRY_POINTS = sorted(_regression_calls(None, None, None))


class TestTheRegressionPathAnnouncesTheDrop:
    @pytest.mark.parametrize("name", _REGRESSION_ENTRY_POINTS)
    def test_every_regression_entry_point_warns_when_a_group_is_dropped(self, name):
        """Before the fix each of these emitted ZERO warnings while removing a
        protected group from the comparison."""
        y_true, y_pred, sens = _regression_dropped_group_data()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _regression_calls(y_true, y_pred, sens)[name]()
        dropped = [str(w.message) for w in record if "below min_group_size" in str(w.message)]
        assert dropped, f"{name} dropped group C in silence: {[str(w.message) for w in record]}"
        assert "'C': 5" in dropped[0]

    def test_the_regression_warning_states_the_share_of_rows(self):
        y_true, y_pred, sens = _regression_dropped_group_data()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            mae_parity_difference(y_true, y_pred, sens)
        text = next(str(w.message) for w in record if "below min_group_size" in str(w.message))
        assert "5 of 125 rows (4.0 percent)" in text
        assert "could not check, not a measured pass" in text

    @pytest.mark.parametrize("name", _REGRESSION_ENTRY_POINTS)
    def test_no_warning_when_every_group_clears_the_gate(self, name):
        """OVER-CORRECTION CONTROL: a warning on every run is a warning nobody
        reads. Fully sampled data must stay silent."""
        y_true, y_pred, sens = _regression_fully_sampled_data()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _regression_calls(y_true, y_pred, sens)[name]()
        assert not [w for w in record if "below min_group_size" in str(w.message)], name

    def test_the_regression_values_are_unchanged_by_the_warning(self):
        """OVER-CORRECTION CONTROL on the numbers: adding disclosure must not
        move a single measurement. Both cases assert the exact value."""
        y_true, y_pred, sens = _regression_fully_sampled_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert mae_parity_difference(y_true, y_pred, sens) == pytest.approx(3.0)
            assert mean_prediction_difference(y_true, y_pred, sens) == pytest.approx(3.0)

            y_true_d, y_pred_d, sens_d = _regression_dropped_group_data()
            # Survivors A and B are predicted exactly, so the measured gap
            # between them is exactly zero. That number is honest about A and B
            # and says nothing at all about C.
            assert mae_parity_difference(y_true_d, y_pred_d, sens_d) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Part 3: the warning has to reach the caller, not this module
# ---------------------------------------------------------------------------


class TestTheDropWarningReachesTheCaller:
    def test_the_warning_is_attributed_to_the_calling_line(self):
        """``stacklevel=3`` points the warning at the code that asked for the
        metric. Without it every one of the 17 call sites reported the same file
        and line inside classification.py."""
        y_true, y_pred, sens = _regression_dropped_group_data()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            mae_parity_difference(y_true, y_pred, sens)
        dropped = [w for w in record if "below min_group_size" in str(w.message)]
        assert dropped
        assert os.path.basename(dropped[0].filename) == os.path.basename(__file__)

    def test_default_filters_show_one_warning_per_metric_not_one_per_run(self, tmp_path):
        """The measured collapse. Python's default filter is once per LOCATION,
        so a shared warn() line inside this library printed ONE stderr line for a
        whole report's worth of drops, pointing at the library. Measured before
        the fix: 4 metrics, 1 printed line. After: 4."""
        script = tmp_path / "four_metrics.py"
        script.write_text(
            "import numpy as np\n"
            "from vfairness.evaluation.vfairness_metrics import (\n"
            "    demographic_parity_difference,\n"
            "    equal_opportunity_difference,\n"
            "    equalized_odds_difference,\n"
            "    predictive_parity_difference,\n"
            ")\n"
            "s = np.array(['A'] * 60 + ['B'] * 60 + ['C'] * 5)\n"
            "yt = np.array([1, 0] * 62 + [1])\n"
            "yp = np.array([0] * 60 + [1] * 60 + [1] * 5)\n"
            "demographic_parity_difference(yt, yp, s)\n"
            "equal_opportunity_difference(yt, yp, s)\n"
            "equalized_odds_difference(yt, yp, s)\n"
            "predictive_parity_difference(yt, yp, s)\n"
        )
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, check=True
        )
        printed = [line for line in proc.stderr.splitlines() if "UserWarning" in line]
        # Count the DROP warnings, which is this test's subject, rather than every
        # UserWarning the four calls raise. predictive_parity_difference also
        # refuses this input on its own account: group A is never predicted
        # positive, so its precision is undefined and the comparison is NOT
        # MEASURABLE. That refusal is correct and belongs on stderr, it is simply
        # not a dropped-group warning. Until 2026-09-17 it was unreachable (the
        # gate counted DEFINED precisions, so one defined rate returned a bare
        # NaN in silence) and this line counted 4 only because of that defect.
        # The attribution assertion below still covers EVERY printed line.
        dropped = [line for line in printed if "below min_group_size" in line]
        assert len(dropped) == 4, proc.stderr
        assert all("classification.py" not in line for line in printed), proc.stderr


# ---------------------------------------------------------------------------
# The sibling: the training callback is the same gate one surface over
# ---------------------------------------------------------------------------


class TestTheTrainingCallbackRefusesTheSameRun:
    """``create_fairness_callback(fail_on_violation=True)`` builds its analyzer
    with the DEFAULT min_group_size=30 and offers no way to change it, so it had
    the identical hole. Before the fix it RETURNED
    ``demographic_parity_difference 0.0067`` on the denied-class run and training
    continued."""

    @staticmethod
    def _named(attr):
        return pd.Series(list(attr), name="ethnicity")

    def test_the_gate_arm_refuses_the_denied_class_run(self):
        y_true, y_pred, attr = _denied_class_data()
        callback = create_fairness_callback(
            "ethnicity",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        with _quiet():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as excinfo:
                callback(y_true, y_pred, self._named(attr))
        text = str(excinfo.value)
        assert "NOT MEASURABLE" in text
        assert "1102 of 2302 rows" in text
        assert "47.9 percent" in text
        assert "unchecked group cannot satisfy a fairness gate" in text

    def test_the_gate_arm_still_passes_fully_sampled_fair_data(self):
        """OVER-CORRECTION CONTROL, and it asserts the MEASURED value, not that
        the call merely returned."""
        y_true, y_pred, sens = _fully_sampled_fair_data()
        callback = create_fairness_callback(
            "ethnicity",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        results = callback(y_true, y_pred, self._named(sens))
        value = getattr(
            results["demographic_parity_difference"],
            "value",
            results["demographic_parity_difference"],
        )
        assert float(value) == pytest.approx(0.0, abs=1e-12)

    def test_the_gate_arm_still_raises_on_a_measured_breach(self):
        """THREE STATES on this surface too: a real breach must read as a breach,
        not as could not check."""
        y_true, y_pred, sens = _fully_sampled_unfair_data()
        callback = create_fairness_callback(
            "ethnicity",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        with pytest.raises(ValueError) as excinfo:
            callback(y_true, y_pred, self._named(sens))
        assert "NOT MEASURABLE" not in str(excinfo.value)
        assert "1.0000" in str(excinfo.value) or "1.0" in str(excinfo.value)

    def test_a_metric_specific_reason_still_wins_over_the_size_gate_one(self):
        """THREE STATES, and the more specific one first. 100 rows in A plus 5 in
        B leaves ONE valid group, so the metric is NaN AND the size gate dropped a
        group. The caller must be told the metric was never measured, which is the
        finding with the sharper fix, not the generic exclusion notice."""
        sens = np.array(["A"] * 100 + ["B"] * 5)
        y_true = np.array([0] * 100 + [1] * 5)
        y_pred = np.array([0] * 100 + [1] * 5)
        callback = create_fairness_callback(
            "ethnicity",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        with _quiet():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError) as excinfo:
                callback(y_true, y_pred, self._named(sens))
        assert "could not be measured" in str(excinfo.value)

    def test_the_monitoring_arm_is_not_turned_into_a_gate(self):
        """OVER-CORRECTION CONTROL. Without fail_on_violation the caller asked
        for no gate, so the excluded tail must not stop the run; the metric
        functions warn about the drop instead."""
        y_true, y_pred, attr = _denied_class_data()
        callback = create_fairness_callback("ethnicity")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            results = callback(y_true, y_pred, self._named(attr))
        assert "demographic_parity_difference" in results
        assert [w for w in record if "below min_group_size" in str(w.message)]


# ---------------------------------------------------------------------------
# The second sibling: the experiment trackers logged only the survivors
# ---------------------------------------------------------------------------


class _FakeMlflow:
    """Records what the logger writes, so the dashboard record can be read back."""

    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.tags = {}

    def active_run(self):
        return object()

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_param(self, key, value):
        self.params[key] = value

    def set_tag(self, key, value):
        self.tags[key] = value


def _log_to_fake_mlflow(monkeypatch, y_true, y_pred, attr, via_analyzer=False):
    """Log one run to a recording stand-in for mlflow.

    ``via_analyzer=True`` exercises the FairnessAnalyzer branch, which differs
    only in that it calls get_report(include_ci=True) itself; that path
    bootstraps every interval, so the large-dataset assertions use the report
    dict and one small case covers the analyzer branch.
    """
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    from vfairness.evaluation.vfairness_metrics.integrations import log_fairness_to_mlflow

    with _quiet():
        warnings.simplefilter("ignore")
        analyzer = FairnessAnalyzer(y_true, y_pred, attr)
        subject = analyzer if via_analyzer else analyzer.get_report()
        log_fairness_to_mlflow(subject, log_artifacts=False)
    return fake


class TestTheTrackerRecordsWhatWasNotMeasured:
    """``<prefix>.groups`` is data_info['valid_groups'], the SURVIVORS. On the
    denied-class run the tracker showed groups = ['majority_a', 'majority_b']
    beside fairness_score 1.0, with the 38 excluded levels recorded nowhere."""

    def test_the_excluded_levels_are_recorded_beside_the_groups(self, monkeypatch):
        y_true, y_pred, attr = _denied_class_data()
        fake = _log_to_fake_mlflow(monkeypatch, y_true, y_pred, attr)
        assert fake.params["fairness.n_groups_excluded"] == 38
        assert fake.params["fairness.n_rows_excluded_by_min_group_size"] == 1102
        assert "minority_00" in fake.params["fairness.groups_excluded"]
        # And the survivors are still recorded, unchanged.
        assert fake.params["fairness.groups"] == "['majority_a', 'majority_b']"

    def test_a_fully_sampled_run_records_zero_rather_than_nothing(self, monkeypatch):
        """OVER-CORRECTION CONTROL and absence-of-evidence control in one: the
        field must be present with a MEASURED zero, not omitted. An absent field
        cannot be told apart from a version that never recorded it."""
        y_true, y_pred, sens = _fully_sampled_fair_data()
        fake = _log_to_fake_mlflow(monkeypatch, y_true, y_pred, sens)
        assert fake.params["fairness.n_groups_excluded"] == 0
        assert fake.params["fairness.n_rows_excluded_by_min_group_size"] == 0
        assert fake.params["fairness.groups_excluded"] == "[]"

    def test_the_analyzer_branch_records_it_too(self, monkeypatch):
        """The other entry point, on data small enough to bootstrap quickly."""
        y_true, y_pred, sens = _regression_dropped_group_data()
        binary_true = (y_true > 30).astype(int)
        binary_pred = (y_pred > 30).astype(int)
        fake = _log_to_fake_mlflow(monkeypatch, binary_true, binary_pred, sens, via_analyzer=True)
        assert fake.params["fairness.n_groups_excluded"] == 1
        assert fake.params["fairness.n_rows_excluded_by_min_group_size"] == 5
        assert fake.params["fairness.groups_excluded"] == "['C']"

    def test_an_unrecorded_exclusion_list_reads_unknown_not_zero(self):
        """A report that never recorded the invalid groups must say so. Claiming
        zero exclusions for a run that did not record any is a false statement
        about provenance, not a harmless default."""
        from vfairness.evaluation.vfairness_metrics.integrations import _excluded_levels_note

        assert _excluded_levels_note({})["n_groups_excluded"] == "unknown"
        assert _excluded_levels_note({"invalid_groups": []})["n_groups_excluded"] == 0


# ---------------------------------------------------------------------------
# Still open, and recorded so the record cannot rot
# ---------------------------------------------------------------------------

# Same convention as tests/test_no_aggregator_fabricates_a_verdict.py: a defect
# that is real but not fixable from this lane's files is written down here, the
# test that would catch it xfails on the entry, and the companion test below
# asserts every entry is STILL BROKEN so a landed fix turns this file red instead
# of leaving a stale excuse behind.
#
# Both entries live in evaluation/vfairness_metrics/report.py, which this lane
# does not own (its files are integrations.py, regression.py, classification.py).
KNOWN_UNFIXED_OUT_OF_LANE = {
    "report_fairness_score_ignores_the_size_gate": (
        "report.classification_fairness_report computes "
        "fairness_score = passed / (passed + failed) over the metrics that "
        "survived the size gate, so the denied-class run scores 1.0 with 5 "
        "passed and 0 failed while insufficient_evidence_groups holds all 38 "
        "excluded levels. The honest could-not-check is computed and then "
        "flattened by the score. Fix belongs in report.py."
    ),
    "provenance_clause_counts_only_missing_values": (
        "report._missing_data_clause is handed validate_inputs' info, which "
        "counts missing-value exclusions only, so the clause reads "
        "'2302 of 2302 rows assessed, 0 excluded' on the run where the size "
        "gate removed 1102 rows from every metric. docs/API_REFERENCE.md tells "
        "readers to trust that clause as the disclosure of how rows were "
        "handled. Fix belongs in report.py, which already has the invalid "
        "groups in data_info."
    ),
}


def _denied_class_report():
    y_true, y_pred, attr = _denied_class_data()
    with _quiet():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y_true, y_pred, attr).get_report()


def test_the_report_score_does_not_ignore_an_unchecked_protected_class():
    """A fairness score computed over the survivors of the size gate is not a
    fairness score for the protected attribute."""
    report = _denied_class_report()
    if "report_fairness_score_ignores_the_size_gate" in KNOWN_UNFIXED_OUT_OF_LANE:
        pytest.xfail(KNOWN_UNFIXED_OUT_OF_LANE["report_fairness_score_ignores_the_size_gate"])
    score = report["assessment"]["fairness_score"]
    assert score is None or score < 1.0


def test_the_provenance_clause_counts_the_size_gate_too():
    """The clause a careful reader would use to catch the other two findings."""
    report = _denied_class_report()
    if "provenance_clause_counts_only_missing_values" in KNOWN_UNFIXED_OUT_OF_LANE:
        pytest.xfail(KNOWN_UNFIXED_OUT_OF_LANE["provenance_clause_counts_only_missing_values"])
    assert "2302 of 2302 rows assessed, 0 excluded" not in report["assessment"]["summary"]


def test_the_out_of_lane_record_is_honest():
    """Every KNOWN_UNFIXED_OUT_OF_LANE entry must still actually be broken.

    Without this the record becomes an excuse: report.py gets fixed, the entry
    stays, and two xfailing tests keep covering nothing. If this test fails,
    that is the good news: delete the entry it names and let the test above run
    for real.
    """
    report = _denied_class_report()
    assessment = report["assessment"]
    stale = []
    if assessment["fairness_score"] != 1.0:
        stale.append("report_fairness_score_ignores_the_size_gate")
    if "2302 of 2302 rows assessed, 0 excluded" not in assessment["summary"]:
        stale.append("provenance_clause_counts_only_missing_values")
    assert not stale, (
        f"These KNOWN_UNFIXED_OUT_OF_LANE entries are stale, the defect is gone: {stale}.\n"
        f"  Remove them from the dict so the paired test asserts for real.\n"
        f"  fairness_score={assessment['fairness_score']!r}\n"
        f"  summary={assessment['summary'][-120:]!r}"
    )
