"""Beta Go-Live stage 1, group 04: the reweighting package must not fabricate parity.

Two critical findings, both reproduced at the public API before the fix:

  prediction_reweighting   PredictionReweighter().fit(y_prob=all NaN) returned
                           adjustments_ == {'a': 1.0, 'b': 1.0} with
                           is_fitted True, and result_ carried
                           original_metrics {'disparity': 0.0} beside
                           {'mean_prob': nan}. 1.0 is the else-branch of
                           `mean > 0`, which NaN fails, so it reads as "this
                           group needed no correction"; 0.0 is
                           np.mean(nan >= 0.5) == np.mean(False), so a group
                           nobody scored reads as "nobody was selected".
                           The only warning emitted was about argument ORDER.

  reweighting_analysis     ReweightingAnalyzer(...).full_analysis() reported
                           demographic_parity_diff 0.0 and
                           fairness_improvement 0.0 for all five methods on
                           BOTH a one-group input (max-min over one value is
                           structurally 0.0) and an all-NaN input, then RANKED
                           the five on that 0.0 and wrote "For maximum fairness
                           improvement, use 'multiplicative' (reduces disparity
                           by 0.000)" with zero warnings.

Three states, never two: measured / failed / could-not-check. Each test below
pins the could-not-check state at the public entry point, and every one has a
CONTROL on healthy data in the same class, because a fix that makes everything
refuse is a worse defect than the one it replaces.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness import PredictionReweighter, ReweightingAnalyzer

METHODS = {
    "multiplicative",
    "additive",
    "rejection_option",
    "calibrated",
    "distribution_matching",
}


def _healthy(n_per_group: int = 100, seed: int = 11):
    """Two groups with a real, large positive-rate gap: A high, B low."""
    rng = np.random.default_rng(seed)
    sens = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = rng.integers(0, 2, 2 * n_per_group)
    y_prob = np.concatenate(
        [
            rng.uniform(0.55, 0.95, n_per_group),
            rng.uniform(0.05, 0.45, n_per_group),
        ]
    )
    return y_true, y_prob, sens


def _caught(fn):
    """Run fn() capturing every warning; return (value, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


# ===========================================================================
# prediction_reweighting: PredictionReweighter.fit
# ===========================================================================


class TestPredictionReweighterThreeStates:
    def test_all_nan_scores_are_not_a_neutral_multiplier(self):
        """BEFORE: adjustments_ {'a': 1.0, 'b': 1.0}, disparity 0.0."""
        y_true = np.random.default_rng(0).integers(0, 2, 60)
        sens = np.array(["a"] * 30 + ["b"] * 30)
        y_prob = np.full(60, np.nan)

        rw, messages = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )

        # (1) No group is handed the neutral multiplier it never earned.
        assert set(rw.adjustments_) == {"a", "b"}
        assert all(math.isnan(v) for v in rw.adjustments_.values()), rw.adjustments_
        assert rw.adjustments_ != {"a": 1.0, "b": 1.0}

        # (2) The positive-rate spread is could-not-check, not perfect parity.
        d = rw.result_.to_dict()
        assert math.isnan(d["original_metrics"]["disparity"])
        assert math.isnan(d["adjusted_metrics"]["disparity"])
        assert math.isnan(d["fairness_improvement"]["disparity_reduction"])

        # (3) A reader is told why, by name -- not only about argument order.
        named = [m for m in messages if "no finite" in m and "'a'" in m]
        assert named, messages

        # (4) The rendered surface says so too: no improvement arrow over a
        #     non-measurement, and no "1.0" sitting among real multipliers.
        summary = rw.result_.summary()
        assert "NOT MEASURED" in summary
        assert "NOT FITTED" in summary
        assert "0.0000 ↑" not in summary

    def test_one_unscored_group_does_not_invent_a_gap_against_it(self):
        """The other direction of the same fabrication.

        Group A scored, group B all NaN. Before the fix B's positive rate was
        0.0 (nan >= 0.5 is False), so the report showed a near-total
        demographic-parity gap against a group nobody scored. Python's
        max()/min() also SKIP a NaN that is not the first element, so a plain
        max-min over {'A': 0.9, 'B': nan} still returns 0.9 - 0.9 = 0.0: both
        the invented gap and the invented parity have to be refused here.
        """
        rng = np.random.default_rng(5)
        sens = np.array(["A"] * 50 + ["B"] * 50)
        y_true = rng.integers(0, 2, 100)
        y_prob = np.concatenate([rng.uniform(0.6, 0.99, 50), np.full(50, np.nan)])

        rw, messages = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )

        assert math.isnan(rw.adjustments_["B"])
        d = rw.result_.to_dict()
        assert math.isnan(d["original_metrics"]["disparity"])
        assert math.isnan(d["fairness_improvement"]["disparity_reduction"])
        assert any("'B'" in m and "no finite" in m for m in messages), messages

    def test_refit_drops_a_group_that_is_no_longer_present(self):
        """adjustments_ was only ever written into, so a stale group survived."""
        y_true, y_prob, sens = _healthy(40, seed=2)
        rw = PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        assert set(rw.adjustments_) == {"A", "B"}

        only_a = sens == "A"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rw.fit(
                y_true=y_true[only_a],
                y_prob=y_prob[only_a],
                sensitive_attr=sens[only_a],
            )
        assert set(rw.adjustments_) == {"A"}, rw.adjustments_

    def test_control_healthy_data_still_fits_and_measures(self):
        """CONTROL. Over-correction check: real data still produces numbers."""
        y_true, y_prob, sens = _healthy()

        rw, messages = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )

        assert messages == []
        assert set(rw.adjustments_) == {"A", "B"}
        assert all(np.isfinite(v) for v in rw.adjustments_.values())
        # A is scored high and B low, so their corrections must differ, and in
        # opposite directions around the neutral 1.0.
        assert rw.adjustments_["A"] < 1.0 < rw.adjustments_["B"]

        d = rw.result_.to_dict()
        assert d["original_metrics"]["disparity"] == pytest.approx(1.0)
        assert np.isfinite(d["adjusted_metrics"]["disparity"])
        assert d["fairness_improvement"]["disparity_reduction"] > 0.0

        summary = rw.result_.summary()
        assert "NOT MEASURED" not in summary
        assert "NOT FITTED" not in summary

    def test_control_a_measured_zero_disparity_is_still_zero(self):
        """CONTROL. Two groups drawn identically really do have no gap.

        This is the assertion that would break if the fix had simply replaced
        every 0.0 with NaN: a measured parity must stay 0.0.
        """
        rng = np.random.default_rng(9)
        sens = np.array(["A", "B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.full(200, 0.8)  # every sample above threshold, in both groups

        rw, messages = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        assert messages == []
        d = rw.result_.to_dict()
        assert d["original_metrics"]["disparity"] == 0.0
        assert not math.isnan(d["original_metrics"]["disparity"])


# ===========================================================================
# reweighting_analysis: ReweightingAnalyzer.full_analysis
# ===========================================================================


class TestReweightingAnalyzerThreeStates:
    def test_single_group_is_not_perfect_parity(self):
        """BEFORE: dp_diff 0.0, all five improvements 0.0, a named best method,
        'reduces disparity by 0.000', and ZERO warnings."""
        rng = np.random.default_rng(3)
        sens = np.array(["A"] * 200)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.random(200)

        report, messages = _caught(
            lambda: ReweightingAnalyzer(y_true, y_prob, sens).full_analysis()
        )

        # (1) Nothing is ranked on a gap that does not exist.
        assert report.comparison_summary == {}
        assert report.method_results == []
        assert report.best_method not in METHODS, report.best_method

        # (2) The methods are disclosed as not measurable, in the field the
        #     report already has for undisclosed gaps.
        assert set(report.failed_methods) == METHODS
        assert all("NotMeasurable" in r for r in report.failed_methods.values())

        # (3) No recommendation quotes the fabricated reduction.
        joined = " ".join(report.recommendations)
        assert "reduces disparity by 0.000" not in joined
        assert "best trade-off" not in joined

        # (4) The reader is told at construction, by name.
        assert any("no between-group disparity exists" in m for m in messages), messages

    def test_single_group_per_method_diff_is_nan(self):
        """analyze_method is a public entry too."""
        rng = np.random.default_rng(3)
        sens = np.array(["A"] * 200)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.random(200)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ReweightingAnalyzer(y_true, y_prob, sens).analyze_method("multiplicative")

        assert math.isnan(result.original_fairness["demographic_parity_diff"])
        assert math.isnan(result.adjusted_fairness["demographic_parity_diff"])
        assert math.isnan(result.trade_off_score)

    def test_all_nan_scores_are_not_perfect_parity(self):
        """BEFORE: group_rates {'A': 0.0, 'B': 0.0}, dp_diff 0.0, five 0.0
        improvements, a named best method; the only warnings were about
        argument order."""
        rng = np.random.default_rng(4)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.full(200, np.nan)

        report, messages = _caught(
            lambda: ReweightingAnalyzer(y_true, y_prob, sens).full_analysis()
        )

        assert report.comparison_summary == {}
        assert report.best_method not in METHODS, report.best_method
        assert set(report.failed_methods) == METHODS
        joined = " ".join(report.recommendations)
        assert "reduces disparity by 0.000" not in joined
        assert "best trade-off" not in joined
        assert any("no finite predicted probability" in m for m in messages), messages

    def test_control_healthy_data_still_ranks_and_recommends(self):
        """CONTROL. A real gap is still measured, ranked and recommended on."""
        y_true, y_prob, sens = _healthy()

        report, messages = _caught(
            lambda: ReweightingAnalyzer(y_true, y_prob, sens).full_analysis()
        )

        assert messages == []
        assert report.failed_methods == {}
        assert set(report.comparison_summary) == METHODS
        assert report.best_method in METHODS
        assert len(report.method_results) == 5

        first = report.method_results[0]
        assert first.original_fairness["demographic_parity_diff"] == pytest.approx(1.0)
        assert first.original_fairness["group_rates"] == {"A": 1.0, "B": 0.0}
        assert any("reduces disparity by" in r for r in report.recommendations)
        assert any("best trade-off" in r for r in report.recommendations)

    def test_control_a_partially_scored_group_still_measures_what_was_scored(self):
        """CONTROL + disclosure. Dropping 10 unscored rows from one group must
        not silently shrink the denominator: the rate is measured over the
        scored rows and the exclusion is warned about by name."""
        rng = np.random.default_rng(6)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
        y_prob[:10] = np.nan  # ten of group A were never scored

        result, messages = _caught(
            lambda: ReweightingAnalyzer(y_true, y_prob, sens).analyze_method("multiplicative")
        )

        rates = result.original_fairness["group_rates"]
        assert rates["A"] == pytest.approx(1.0)  # the 90 scored rows, not 90/100
        assert rates["B"] == pytest.approx(0.0)
        assert result.original_fairness["demographic_parity_diff"] == pytest.approx(1.0)
        assert any("10 of 100" in m for m in messages), messages
