"""VF-6: the report must carry its own missing-data provenance.

Two runs over the SAME data with different ``missing_strategy`` values can reach
different verdicts. If the artifact does not say how many rows were dropped and
which strategy dropped them, the verdict cannot be audited: the reader sees two
different answers and nothing that explains the difference.

State of play, established by execution before this was written:

  * the STRUCTURED provenance was already present. ``report["data_info"]`` carries
    ``original_size``, ``final_size``, ``n_excluded`` and ``missing_strategy`` on
    both the direct report functions and the FairnessAnalyzer path. The tests in
    ``TestStructuredProvenanceWasAlreadyPresent`` pin that (they pass before and
    after this change, and exist so a later refactor cannot quietly drop it).
  * the VERDICT surfaces did not. ``assessment["summary"]``, the one line every
    reader and every summary consumer sees, said "3/5 metrics within thresholds"
    whether 0 or 10 rows had been silently excluded, and ``print_report`` printed
    the excluded COUNT but never the STRATEGY, so the two runs above printed
    artifacts that did not distinguish themselves.

This is provenance only. No metric maths changes, and no report key is added or
removed.
"""

import json
import warnings

import numpy as np
import pytest

from vfairness import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    print_report,
    regression_fairness_report,
)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture
def classification_with_missing_rows():
    """120 rows, 15 of which carry a missing prediction and are excluded."""
    rng = np.random.default_rng(0)
    n = 120
    y_true = rng.integers(0, 2, n).astype(float)
    y_pred = rng.integers(0, 2, n).astype(float)
    y_pred[:15] = np.nan
    attr = np.array(["A"] * 60 + ["B"] * 60, dtype=object)
    return y_true, y_pred, attr


@pytest.fixture
def regression_with_missing_rows():
    rng = np.random.default_rng(1)
    n = 120
    y_true = rng.normal(10, 2, n)
    y_pred = y_true + rng.normal(0, 0.5, n)
    y_pred[:12] = np.nan
    attr = np.array(["A"] * 60 + ["B"] * 60, dtype=object)
    return y_true, y_pred, attr


class TestStructuredProvenanceWasAlreadyPresent:
    """The machine-readable half of VF-6, already closed. Pinned, not fixed."""

    def test_data_info_carries_the_four_provenance_fields(self, classification_with_missing_rows):
        y_true, y_pred, attr = classification_with_missing_rows
        di = _quiet(classification_fairness_report, y_true, y_pred, attr)["data_info"]
        assert di["original_size"] == 120
        assert di["final_size"] == 105
        assert di["n_excluded"] == 15
        assert di["missing_strategy"] == "exclude"


class TestAnalyzerPathLosesTheProvenance:
    """FIXED 2026-08-27: the FairnessAnalyzer path used to report FALSE provenance.

    ``FairnessAnalyzer.__init__`` runs ``validate_inputs`` and stores the CLEANED
    arrays plus the true ``data_info``. ``get_report`` then passes those cleaned
    arrays to the report builder, which re-validates from scratch: there was
    nothing left to exclude, so the report recorded ``n_excluded = 0`` and an
    ``original_size`` that was already the post-exclusion count. The analyzer's
    own ``self.data_info``, which holds the true numbers, was never forwarded.
    Nor was ``self.missing_strategy``, so an analyzer built with ``as_group``
    produced a report that named ``'exclude'``.

    ``FairnessAnalyzer._apply_true_provenance`` now overlays the recorded truth
    onto the built report (structured ``data_info`` AND the prose clause in
    ``assessment["summary"]``) before anything downstream, the cache included,
    reads it. These two tests were recorded as ``xfail(strict=True)`` while the
    defect stood; they turned XPASS the moment the overlay landed, and the
    markers were deleted then, because a strict xfail turns a fixed defect back
    into a failure. Keep them as ordinary tests: they are the regression pins.
    """

    def test_analyzer_reports_the_rows_it_excluded(self, classification_with_missing_rows):
        y_true, y_pred, attr = classification_with_missing_rows
        di = _quiet(FairnessAnalyzer(y_true, y_pred, attr).get_report, include_ci=False)[
            "data_info"
        ]
        assert di["original_size"] == 120
        assert di["n_excluded"] == 15

    def test_analyzer_reports_the_strategy_it_was_given(self):
        n = 300
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        attr = np.array(["A"] * 150 + ["B"] * 150, dtype=object)
        attr[140:150] = None
        analyzer = FairnessAnalyzer(y_true, y_pred, attr, missing_strategy="as_group")
        di = _quiet(analyzer.get_report, include_ci=False)["data_info"]
        assert di["missing_strategy"] == "as_group"


class TestTheVerdictLineCarriesTheProvenance:
    """The half that was missing: the summary never disclosed the exclusions."""

    def test_classification_summary_states_rows_and_strategy(
        self, classification_with_missing_rows
    ):
        y_true, y_pred, attr = classification_with_missing_rows
        summary = _quiet(classification_fairness_report, y_true, y_pred, attr)["assessment"][
            "summary"
        ]
        assert "105 of 120 rows assessed" in summary
        assert "15 excluded" in summary
        assert "missing_strategy='exclude'" in summary

    def test_regression_summary_states_rows_and_strategy(self, regression_with_missing_rows):
        y_true, y_pred, attr = regression_with_missing_rows
        summary = _quiet(regression_fairness_report, y_true, y_pred, attr)["assessment"]["summary"]
        assert "108 of 120 rows assessed" in summary
        assert "12 excluded" in summary
        assert "missing_strategy='exclude'" in summary

    def test_a_clean_run_still_states_the_accounting(self):
        # The clause is ALWAYS present, never only when rows were dropped.
        # If it appeared only on exclusion, its absence would be ambiguous: an
        # auditor could not tell "nothing was excluded" from "this version did
        # not record it", which is the absence-of-evidence reading that
        # provenance exists to remove.
        y_true = np.array([1, 0] * 60)
        y_pred = np.array([1, 0] * 60)
        attr = np.array(["A"] * 60 + ["B"] * 60)
        summary = _quiet(classification_fairness_report, y_true, y_pred, attr)["assessment"][
            "summary"
        ]
        assert "120 of 120 rows assessed" in summary
        assert "0 excluded" in summary
        assert "missing_strategy='exclude'" in summary

    def test_two_strategies_produce_distinguishable_verdict_lines(self):
        # The finding's exact scenario: same data, two strategies, and until now
        # two verdict lines that did not explain their difference.
        rng = np.random.default_rng(3)
        n = 300
        y_true = rng.integers(0, 2, n)
        y_pred = rng.integers(0, 2, n)
        attr = np.array(["A"] * 150 + ["B"] * 150, dtype=object)
        attr[140:150] = None
        y_pred[140:150] = 1

        summaries = {
            strategy: _quiet(
                classification_fairness_report,
                y_true,
                y_pred,
                attr,
                missing_strategy=strategy,
            )["assessment"]["summary"]
            for strategy in ("exclude", "as_group")
        }

        assert summaries["exclude"] != summaries["as_group"]
        assert "290 of 300 rows assessed" in summaries["exclude"]
        assert "10 excluded" in summaries["exclude"]
        assert "missing_strategy='exclude'" in summaries["exclude"]
        assert "300 of 300 rows assessed" in summaries["as_group"]
        assert "missing_strategy='as_group'" in summaries["as_group"]

    def test_not_assessable_summary_also_carries_it(self):
        # A run that certifies nothing still has to account for its rows.
        y_true = np.array([1, 0] * 20)
        y_pred = np.array([1, 0] * 20)
        attr = np.array(["only_one_group"] * 40)
        summary = _quiet(classification_fairness_report, y_true, y_pred, attr)["assessment"][
            "summary"
        ]
        assert summary.startswith("NOT ASSESSABLE")  # unchanged
        assert "40 of 40 rows assessed" in summary
        assert "missing_strategy='exclude'" in summary


class TestPrintedArtifactNamesTheStrategy:
    def test_printed_report_names_rows_and_strategy(self, capsys, classification_with_missing_rows):
        y_true, y_pred, attr = classification_with_missing_rows
        report = _quiet(classification_fairness_report, y_true, y_pred, attr)
        print_report(report)
        out = capsys.readouterr().out
        assert "105 of 120" in out
        assert "excluded 15" in out
        assert "missing_strategy='exclude'" in out

    def test_absent_provenance_prints_unknown_not_zero(self, capsys):
        # A hand-built or truncated report has no provenance to report. Printing
        # "excluded 0" there asserts something nobody measured. Could-not-check
        # is the third state and must be printed as such.
        print_report(
            {
                "task_type": "classification",
                "metrics": {"demographic_parity_difference": 0.01},
                "assessment": {
                    "fairness_score": 1.0,
                    "assessable": True,
                    "passed_metrics": [],
                    "failed_metrics": [],
                    "not_assessable_metrics": [],
                    "summary": "1/1 metrics within thresholds",
                },
                "data_info": {},
            }
        )
        out = capsys.readouterr().out
        assert "excluded 0" not in out
        assert "unknown" in out


class TestNoOvercorrection:
    def test_report_is_still_json_serialisable(self, classification_with_missing_rows):
        y_true, y_pred, attr = classification_with_missing_rows
        json.dumps(_quiet(classification_fairness_report, y_true, y_pred, attr))

    def test_the_verdict_itself_is_unchanged(self, classification_with_missing_rows):
        # Provenance only: the same rows, the same metrics, the same score.
        y_true, y_pred, attr = classification_with_missing_rows
        report = _quiet(classification_fairness_report, y_true, y_pred, attr)
        assert report["assessment"]["fairness_score"] == pytest.approx(0.4)
        assert report["assessment"]["assessable"] is True
        assert report["metrics"]["demographic_parity_difference"] == pytest.approx(0.0611, abs=1e-3)

    def test_the_summary_keeps_its_existing_clauses(self):
        # The metric count leads, and the excluded-group caveat survives.
        g = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 7)
        y_true = np.array([1, 0] * 103 + [1])
        y_pred = np.array([1, 0] * 103 + [0])
        summary = _quiet(classification_fairness_report, y_true, y_pred, g)["assessment"]["summary"]
        assert "metrics within thresholds" in summary
        assert "insufficient evidence" in summary
        assert "C" in summary

    def test_no_em_dash_reaches_the_rendered_summary(self, classification_with_missing_rows):
        # The summary is rasterised into a report panel; an em/en-dash breaks it.
        y_true, y_pred, attr = classification_with_missing_rows
        summary = _quiet(classification_fairness_report, y_true, y_pred, attr)["assessment"][
            "summary"
        ]
        assert "—" not in summary and "–" not in summary
