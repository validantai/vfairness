"""The FairnessAnalyzer report must state the provenance of the data it was GIVEN.

``FairnessAnalyzer.__init__`` cleans the caller's arrays once and keeps the true
``data_info``. ``get_report`` hands those already-cleaned arrays to the report
builder, so the builder's own re-validation finds nothing left to drop. Until
``_apply_true_provenance`` landed, the report therefore asserted, positively, that
a run which excluded 15 of 120 rows had excluded none, and named ``'exclude'`` for
an analyzer that was built with ``as_group``.

That is a FALSE claim, not a missing one, and it is the difference between a
report that can be audited and one that cannot: two runs over the same data under
different strategies reach different verdicts, and nothing in the artifact
explained why.

These tests cover the analyzer path only (the direct report-function path is
pinned in ``test_report_provenance.py``). They check BOTH surfaces that state the
provenance, the structured ``data_info`` and the prose clause at the end of
``assessment["summary"]``, on classification and regression, with rows dropped and
with none dropped, through the cache, and they pin that nothing else moved.
"""

import json
import warnings

import numpy as np
import pytest

from vfairness import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.report import classification_fairness_report


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _provenance(report):
    di = report["data_info"]
    return {
        "original_size": di.get("original_size"),
        "final_size": di.get("final_size"),
        "n_excluded": di.get("n_excluded"),
        "missing_strategy": di.get("missing_strategy"),
        "n_samples": di.get("n_samples"),
    }


@pytest.fixture
def classification_missing_predictions():
    """120 rows, 15 of which carry a missing PREDICTION (dropped by every strategy)."""
    rng = np.random.default_rng(0)
    n = 120
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = rng.integers(0, 2, n).astype(float)
    y_pred[:15] = np.nan
    attr = np.array(["A"] * 60 + ["B"] * 60, dtype=object)
    return y_true, y_pred, attr


@pytest.fixture
def classification_missing_attribute():
    """300 rows, 10 of which carry a missing SENSITIVE ATTRIBUTE.

    This is the pair that separates the strategies: ``exclude`` drops those 10,
    ``as_group`` keeps them as their own group. A report that cannot tell the two
    apart cannot explain why the verdicts differ.
    """
    rng = np.random.default_rng(3)
    n = 300
    y_true = rng.integers(0, 2, n)
    y_pred = rng.integers(0, 2, n)
    attr = np.array(["A"] * 150 + ["B"] * 150, dtype=object)
    attr[140:150] = None
    return y_true, y_pred, attr


class TestExcludeStrategy:
    def test_data_info_states_the_rows_the_user_handed_in(self, classification_missing_predictions):
        y_true, y_pred, attr = classification_missing_predictions
        report = _quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report)
        assert _provenance(report) == {
            "original_size": 120,
            "final_size": 105,
            "n_excluded": 15,
            "missing_strategy": "exclude",
            "n_samples": 105,
        }

    def test_the_summary_clause_states_it_too(self, classification_missing_predictions):
        # data_info alone is not enough: the clause on the verdict line is what a
        # reader sees, and it carried the false sentence
        # "105 of 105 rows assessed, 0 excluded".
        y_true, y_pred, attr = classification_missing_predictions
        summary = _quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report)["assessment"]["summary"]
        assert "105 of 120 rows assessed" in summary
        assert "15 excluded" in summary
        assert "missing_strategy='exclude'" in summary
        assert "105 of 105" not in summary
        assert "0 excluded" not in summary

    def test_attribute_missing_rows_are_counted_as_excluded(self, classification_missing_attribute):
        y_true, y_pred, attr = classification_missing_attribute
        report = _quiet(
            FairnessAnalyzer(y_true, y_pred, attr, missing_strategy="exclude").get_report
        )
        assert _provenance(report) == {
            "original_size": 300,
            "final_size": 290,
            "n_excluded": 10,
            "missing_strategy": "exclude",
            "n_samples": 290,
        }

    def test_intersectional_attributes_report_their_dropped_rows_too(self):
        # The intersectional branch of validate_inputs drops rows whose
        # sensitive-attribute COLUMNS are missing, a different code path from the
        # array branch above, and it lost the same accounting.
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(5)
        n = 400
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        attr = pd.DataFrame({"g": ["A"] * 200 + ["B"] * 200, "r": (["X"] * 100 + ["Y"] * 100) * 2})
        attr.loc[:9, "g"] = None
        report = _quiet(FairnessAnalyzer(y_true, y_pred, attr, min_group_size=20).get_report)
        assert report["data_info"]["is_intersectional"] is True
        assert _provenance(report) == {
            "original_size": 400,
            "final_size": 390,
            "n_excluded": 10,
            "missing_strategy": "exclude",
            "n_samples": 390,
        }
        assert "390 of 400 rows assessed" in report["assessment"]["summary"]


class TestAsGroupStrategy:
    def test_the_strategy_in_force_is_the_one_reported(self, classification_missing_attribute):
        y_true, y_pred, attr = classification_missing_attribute
        report = _quiet(
            FairnessAnalyzer(y_true, y_pred, attr, missing_strategy="as_group").get_report
        )
        prov = _provenance(report)
        assert prov["missing_strategy"] == "as_group"
        # as_group keeps the attribute-missing rows as their own group, so nothing
        # is dropped here and the accounting must say so rather than inherit the
        # exclude run's numbers.
        assert prov == {
            "original_size": 300,
            "final_size": 300,
            "n_excluded": 0,
            "missing_strategy": "as_group",
            "n_samples": 300,
        }
        assert "missing_strategy='as_group'" in report["assessment"]["summary"]

    def test_as_group_still_reports_rows_dropped_for_missing_labels(
        self, classification_missing_predictions
    ):
        # as_group only rescues a missing ATTRIBUTE. Rows with a missing
        # prediction are still dropped, and must still be counted.
        y_true, y_pred, attr = classification_missing_predictions
        report = _quiet(
            FairnessAnalyzer(y_true, y_pred, attr, missing_strategy="as_group").get_report
        )
        assert _provenance(report) == {
            "original_size": 120,
            "final_size": 105,
            "n_excluded": 15,
            "missing_strategy": "as_group",
            "n_samples": 105,
        }

    def test_two_strategies_produce_distinguishable_analyzer_reports(
        self, classification_missing_attribute
    ):
        # The finding's exact scenario, on the analyzer path: same data, two
        # strategies, two verdicts, and until now two artifacts that gave the
        # reader nothing to explain the difference with.
        y_true, y_pred, attr = classification_missing_attribute
        summaries = {
            strategy: _quiet(
                FairnessAnalyzer(y_true, y_pred, attr, missing_strategy=strategy).get_report
            )["assessment"]["summary"]
            for strategy in ("exclude", "as_group")
        }
        assert summaries["exclude"] != summaries["as_group"]
        assert "290 of 300 rows assessed" in summaries["exclude"]
        assert "10 excluded" in summaries["exclude"]
        assert "missing_strategy='exclude'" in summaries["exclude"]
        assert "300 of 300 rows assessed" in summaries["as_group"]
        assert "0 excluded" in summaries["as_group"]
        assert "missing_strategy='as_group'" in summaries["as_group"]


class TestCleanRun:
    def test_a_run_that_dropped_nothing_still_states_the_accounting(self):
        # Absence must never be ambiguous: an auditor has to be able to tell
        # "nothing was excluded" from "this version did not record it", so the
        # clause is present and reads 0, not missing.
        y_true = np.array([1, 0] * 60)
        y_pred = np.array([1, 0] * 60)
        attr = np.array(["A"] * 60 + ["B"] * 60)
        report = _quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report)
        assert _provenance(report) == {
            "original_size": 120,
            "final_size": 120,
            "n_excluded": 0,
            "missing_strategy": "exclude",
            "n_samples": 120,
        }
        summary = report["assessment"]["summary"]
        assert "120 of 120 rows assessed" in summary
        assert "0 excluded" in summary
        assert "missing_strategy='exclude'" in summary


class TestRegressionPath:
    def test_regression_report_carries_the_true_provenance(self):
        rng = np.random.default_rng(1)
        n = 120
        y_true = rng.normal(10, 2, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        y_pred[:12] = np.nan
        attr = np.array(["A"] * 60 + ["B"] * 60, dtype=object)
        report = _quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report)
        assert report["task_type"] == "regression"
        assert _provenance(report) == {
            "original_size": 120,
            "final_size": 108,
            "n_excluded": 12,
            "missing_strategy": "exclude",
            "n_samples": 108,
        }
        summary = report["assessment"]["summary"]
        assert "108 of 120 rows assessed" in summary
        assert "12 excluded" in summary
        # Regression-only field survives the overlay.
        assert "y_std" in report["data_info"]


class TestNumbersCannotContradictEachOther:
    def test_final_size_matches_the_rows_the_metrics_were_computed_over(
        self, classification_missing_predictions
    ):
        # final_size / n_samples describe the ANALYSED rows, so they must keep
        # agreeing with the group sizes the metrics were actually computed from,
        # and with original_size - n_excluded.
        y_true, y_pred, attr = classification_missing_predictions
        analyzer = FairnessAnalyzer(y_true, y_pred, attr)
        report = _quiet(analyzer.get_report)
        di = report["data_info"]
        assert di["final_size"] == di["n_samples"] == analyzer.n_samples == len(analyzer.y_true)
        assert di["original_size"] - di["n_excluded"] == di["final_size"]
        assert sum(di["group_sizes"].values()) == di["final_size"]

    def test_an_irreconcilable_accounting_reports_unknown_not_a_number(
        self, classification_missing_predictions
    ):
        # Fail closed. If the recorded counts cannot be reconciled with the rows
        # the metrics were computed over, the two accounts cannot both be true
        # and we do not know which is, so nothing is published as a count.
        # Publishing either one would be the same false-claim failure this whole
        # repair exists to remove.
        y_true, y_pred, attr = classification_missing_predictions
        analyzer = FairnessAnalyzer(y_true, y_pred, attr)
        analyzer.data_info = {**analyzer.data_info, "original_size": 999}
        report = _quiet(analyzer.get_report)
        di = report["data_info"]
        assert di["original_size"] is None
        assert di["n_excluded"] is None
        assert di["final_size"] == 105  # the analysed rows are still known
        assert di["missing_strategy"] == "exclude"
        summary = report["assessment"]["summary"]
        assert "105 of unknown rows assessed" in summary
        assert "unknown excluded" in summary
        assert "999" not in summary


class TestCachePath:
    def test_cached_report_serves_the_true_provenance_on_every_call(
        self, classification_missing_predictions
    ):
        y_true, y_pred, attr = classification_missing_predictions
        analyzer = FairnessAnalyzer(y_true, y_pred, attr, cache=True)
        first = _quiet(analyzer.get_report)
        second = _quiet(analyzer.get_report)
        assert _provenance(first)["n_excluded"] == 15
        assert _provenance(second) == _provenance(first)
        assert second["assessment"]["summary"] == first["assessment"]["summary"]

    def test_mutating_a_returned_report_cannot_poison_the_cached_provenance(
        self, classification_missing_predictions
    ):
        # get_report deep-copies on store and on return, so a caller editing its
        # own copy must not change what the next caller is told about the run.
        y_true, y_pred, attr = classification_missing_predictions
        analyzer = FairnessAnalyzer(y_true, y_pred, attr, cache=True)
        first = _quiet(analyzer.get_report)
        first["data_info"]["n_excluded"] = 0
        first["data_info"]["original_size"] = 105
        first["assessment"]["summary"] = "tampered"
        second = _quiet(analyzer.get_report)
        assert _provenance(second)["n_excluded"] == 15
        assert _provenance(second)["original_size"] == 120
        assert "105 of 120 rows assessed" in second["assessment"]["summary"]

    def test_the_clause_is_not_applied_twice(self, classification_missing_predictions):
        y_true, y_pred, attr = classification_missing_predictions
        analyzer = FairnessAnalyzer(y_true, y_pred, attr, cache=True)
        for _ in range(3):
            summary = _quiet(analyzer.get_report)["assessment"]["summary"]
            assert summary.count("data provenance:") == 1


class TestNoOvercorrection:
    def test_no_field_other_than_the_provenance_moves(self, classification_missing_predictions):
        # The overlay leaves the builder's INPUT untouched, so every other field
        # must be byte-identical to what the builder produces on its own. This is
        # the property that justified the overlay over re-validating the caller's
        # original arrays.
        y_true, y_pred, attr = classification_missing_predictions
        # task_type is stated rather than inferred: a y_pred carrying NaN has more
        # than two unique values, so auto-detection reads this fixture as
        # regression, and the baseline must be built by the SAME builder.
        analyzer = FairnessAnalyzer(y_true, y_pred, attr, task_type="classification")
        overlaid = _quiet(analyzer.get_report)
        baseline = _quiet(
            classification_fairness_report,
            analyzer.y_true,
            analyzer.y_pred,
            analyzer.sensitive_attr,
            y_prob=analyzer.y_prob,
            min_group_size=analyzer.min_group_size,
        )

        def _strip(report):
            report = json.loads(json.dumps(report, default=str))
            report.pop("data_info")
            report["assessment"].pop("summary")
            report.pop("methodology_version", None)
            return report

        assert _strip(overlaid) == _strip(baseline)

        # ... and within data_info, only the provenance fields differ.
        # `coverage` joined that set on 2026-09-30 (B4 tier-1 audit): the overlay
        # listed three fields and the builder derives coverage from the arrays it was
        # handed, which are already cleaned, so it published the CONSTANT 1.0 in a
        # field named coverage for every exclusion fraction (measured 1.0 at 5 of 80,
        # 40 of 120 and 120 of 200 excluded). The subject of this assertion is
        # unchanged: NOTHING outside the provenance block moves.
        differing = {
            key
            for key in set(overlaid["data_info"]) | set(baseline["data_info"])
            if json.dumps(overlaid["data_info"].get(key), default=str)
            != json.dumps(baseline["data_info"].get(key), default=str)
        }
        assert differing == {"original_size", "n_excluded", "coverage"}
        assert overlaid["data_info"]["coverage"] == pytest.approx(
            overlaid["data_info"]["final_size"] / overlaid["data_info"]["original_size"]
        )
        assert baseline["data_info"]["coverage"] == 1.0, (
            "the builder's own coverage is the constant this overlay exists to replace"
        )

        # ... and the summary differs only in that clause.
        assert (
            overlaid["assessment"]["summary"].split(" (data provenance:")[0]
            == (baseline["assessment"]["summary"].split(" (data provenance:")[0])
        )

    def test_the_report_is_still_json_serialisable(self, classification_missing_predictions):
        y_true, y_pred, attr = classification_missing_predictions
        json.dumps(_quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report), default=str)

    def test_explanations_see_the_corrected_provenance(self, classification_missing_predictions):
        # The overlay runs before the explainer, so anything the explainer reads
        # off the report is the true accounting, not the post-cleaning remainder.
        y_true, y_pred, attr = classification_missing_predictions
        report = _quiet(FairnessAnalyzer(y_true, y_pred, attr, fair_explainer=True).get_report)
        assert "explanations" in report
        assert _provenance(report)["n_excluded"] == 15

    def test_no_em_dash_reaches_the_summary(self, classification_missing_predictions):
        y_true, y_pred, attr = classification_missing_predictions
        summary = _quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report)["assessment"]["summary"]
        assert "—" not in summary and "–" not in summary
