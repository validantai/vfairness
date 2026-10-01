"""Beta Go-Live stage 2b, group s2b09: rows nobody scored are not decisions.

Every pin here was reproduced by execution at a PUBLIC entry point before the
fix, and every one of them has a control on healthy data in the same file,
because a capability that refuses everything passes any test that only checks
the degenerate case.

  ReweightingAnalyzer.analyze_method (post_processing/reweighting/analyzer.py)

    A1 `(y_prob_adjusted >= threshold).astype(int)` cast every row the
       reweighter had REFUSED to score into a hard 0, a measured rejection.
       Measured on 179 rows whose 29-row minority DistributionMatcher declined
       to fit: 29 manufactured rejections, 15 of them counted as CORRECT,
       inside a reported adjusted accuracy of 0.547486.
    A2 The ECE divided each bin's weight by the FULL row count while only the
       scored rows could land in a bin, so the same run reported adjusted_ece
       0.105793 where the 150 scored rows measure 0.126246, and the reported
       ece_change was NEGATIVE (a calibration improvement) where the honest one
       is POSITIVE (a cost).
    A3 An input nothing scored reported accuracy 0.525, which is the share of
       zero labels, and ECE 0.0, which is perfect calibration.
    A4 `analyze_method(threshold=nan)` on data with a real 1.0 gap reported
       demographic_parity_diff 0.0, group_rates {'A': 0.0, 'B': 0.0},
       trade_off_score 0.0 and not one warning.
    A5 `max(0, nan)` is 0 (`nan > 0` is False), so a calibration error that
       could not be computed entered the trade-off as "this method cost no
       calibration" and the method was then ranked and recommended.

  counterfactual_fairness (evaluation/vfairness_metrics/counterfactual_metric.py)

    C1 The stage-2 fix masked on `np.isfinite`, which also excluded +/-inf. An
       infinite score sits decidably on one side of a finite threshold: 100
       clean rows plus 100 rows whose factual 0.2 became a counterfactual +inf
       read flip_rate 0.0 / 'info' / "Robust to the perturbation", where the
       code before that mask read 0.5 / CRITICAL. The refusal deleted a real
       finding, which is worse than the fabrication it replaced.
    C2 `threshold=nan` on data with a genuine 0.30 flip rate returned
       flip_rate 0.0, severity 'info', notes [] and no warning.
    C3 A partial assessment carried the unqualified _grade() verdict; only
       `notes` said the sample had been narrowed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness._not_assessed import NOT_ASSESSED
from vfairness.post_processing.reweighting.analyzer import ReweightingAnalyzer

METHODS = {
    "multiplicative",
    "additive",
    "rejection_option",
    "calibrated",
    "distribution_matching",
}


def _caught(fn, *args, **kwargs):
    """Run fn() capturing every warning; return (value, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


def _refused_minority(n_major: int = 150, n_minor: int = 29, seed: int = 11):
    """A minority too small for DistributionMatcher's min_group_size=30.

    Its 29 rows come back NaN from transform(), so they are the rows the
    analyzer used to decide for.
    """
    rng = np.random.default_rng(seed)
    sens = np.array(["A"] * n_major + ["B"] * n_minor)
    y_true = rng.integers(0, 2, n_major + n_minor)
    y_prob = np.clip(rng.normal(0.5, 0.2, n_major + n_minor), 0.01, 0.99)
    return y_true, y_prob, sens


def _healthy(n_per_group: int = 100, seed: int = 11):
    """Two groups, a real positive-rate gap, every row scored."""
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


# ===========================================================================
# A1/A2  the analyzer stops deciding for rows the reweighter refused
# ===========================================================================


class TestAnalyzerDoesNotDecideForUnscoredRows:
    def test_unscored_rows_are_not_counted_as_rejections(self):
        """BEFORE: adjusted accuracy 0.547486 over 179 rows, 29 of which were
        never scored and were counted as rejections (15 of them as CORRECT
        rejections, because their label happened to be 0)."""
        y_true, y_prob, sens = _refused_minority()

        result, messages = _caught(
            ReweightingAnalyzer(y_true, y_prob, sens).analyze_method,
            "distribution_matching",
        )

        # (0) The fixture reaches the branch under test: the transform really
        #     did refuse 29 of the 179 rows. Without this the rest of the test
        #     would pass on an input that never exercises the fix.
        assert result.adjusted_performance["n_rows"] == 179
        assert result.adjusted_performance["n_decided"] == 150
        assert result.original_performance["n_decided"] == 179, "the input side is intact"

        # (1) The accuracy is the one measured over the rows that were decided,
        #     and specifically NOT the value the 29 manufactured rejections
        #     produced.
        assert result.adjusted_performance["accuracy"] == pytest.approx(0.5533333333333333)
        assert result.adjusted_performance["accuracy"] != pytest.approx(0.547486033519553)

        # (2) A reader of the result object alone can see the narrowing.
        assert result.adjusted_performance["n_decided"] < result.adjusted_performance["n_rows"]
        assert any("29 of 179 row(s) carry no score" in m for m in messages), messages

    def test_the_calibration_error_says_how_many_rows_it_covers(self):
        """BEFORE: adjusted_ece 0.105793 and ece_change -0.004974, a reported
        calibration IMPROVEMENT, because each bin's weight was divided by 179
        while only 150 rows could enter a bin."""
        y_true, y_prob, sens = _refused_minority()

        result, messages = _caught(
            ReweightingAnalyzer(y_true, y_prob, sens).analyze_method,
            "distribution_matching",
        )
        cal = result.calibration_metrics

        assert cal["original_ece_rows_used"] == 179
        assert cal["adjusted_ece_rows_used"] == 150
        assert cal["n_rows"] == 179
        assert cal["adjusted_ece"] == pytest.approx(0.12624589001867362)
        assert cal["adjusted_ece"] != pytest.approx(0.10579264526704493)
        # The sign flips: the narrowed denominator turned a cost into a gain.
        assert cal["ece_change"] > 0
        assert any("calibration error covers" in m for m in messages), messages

    def test_an_input_nothing_scored_is_nan_not_the_share_of_zero_labels(self):
        """BEFORE: accuracy 0.525 (= the share of label 0, because every row
        was decided 0) and ECE 0.0 (= perfectly calibrated), on 200 rows where
        nothing was scored at all."""
        rng = np.random.default_rng(4)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.full(200, np.nan)

        result, messages = _caught(
            ReweightingAnalyzer(y_true, y_prob, sens).analyze_method, "multiplicative"
        )

        assert result.original_performance["n_decided"] == 0
        assert math.isnan(result.original_performance["accuracy"])
        assert result.original_performance["accuracy"] != 0.0  # NaN != anything
        assert math.isnan(result.calibration_metrics["original_ece"])
        assert result.calibration_metrics["original_ece_rows_used"] == 0
        assert math.isnan(result.trade_off_score)
        assert any("carry no score" in m for m in messages), messages

    def test_control_healthy_data_is_measured_ranked_and_silent(self):
        """CONTROL. Every row scored: the numbers are exactly the ones the
        pre-fix arithmetic produced, the coverage fields say 'all of them', and
        nothing warns. A fix that refuses healthy data is a worse defect."""
        y_true, y_prob, sens = _healthy()

        report, messages = _caught(ReweightingAnalyzer(y_true, y_prob, sens).full_analysis)

        assert messages == [], messages
        assert report.failed_methods == {}
        assert set(report.comparison_summary) == METHODS
        assert report.best_method in METHODS
        assert len(report.method_results) == 5

        first = report.method_results[0]
        # The pre-fix accuracy formula, unchanged, on a fully scored input.
        expected = float(np.mean(y_true == (y_prob >= 0.5).astype(int)))
        assert first.original_performance["accuracy"] == pytest.approx(expected)
        assert first.original_performance["n_decided"] == 200
        assert first.original_performance["n_rows"] == 200
        assert first.calibration_metrics["original_ece_rows_used"] == 200
        assert np.isfinite(first.calibration_metrics["original_ece"])
        assert any("reduces disparity by" in r for r in report.recommendations)

    def test_control_a_partially_scored_input_still_measures_what_was_scored(self):
        """CONTROL for A1. 150 of 179 rows were scored, and those 150 are
        measured: the fix narrows the sample, it does not refuse it."""
        y_true, y_prob, sens = _refused_minority()

        result, _ = _caught(
            ReweightingAnalyzer(y_true, y_prob, sens).analyze_method,
            "distribution_matching",
        )

        decided = result.adjusted_performance["accuracy"]
        assert np.isfinite(decided) and 0.0 < decided < 1.0
        assert np.isfinite(result.calibration_metrics["adjusted_ece"])
        assert np.isfinite(result.original_performance["accuracy"])


# ===========================================================================
# A4  a threshold nothing can be decided against
# ===========================================================================


class TestAnalyzerThresholdGuard:
    def test_a_nan_threshold_is_refused_not_read_as_perfect_parity(self):
        """BEFORE: a real 1.0 gap reported as demographic_parity_diff 0.0,
        group_rates {'A': 0.0, 'B': 0.0}, accuracy 0.485, trade_off 0.0, and
        ZERO warnings."""
        rng = np.random.default_rng(3)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.concatenate([rng.uniform(0.6, 0.99, 100), rng.uniform(0.01, 0.4, 100)])
        analyzer = ReweightingAnalyzer(y_true, y_prob, sens)

        with pytest.raises(ValueError, match="not a number"):
            analyzer.analyze_method("multiplicative", threshold=float("nan"))

        # The whole-report entry discloses it per method instead of ranking.
        report, _ = _caught(analyzer.full_analysis, threshold=float("nan"))
        assert report.comparison_summary == {}
        assert set(report.failed_methods) == METHODS
        assert report.best_method not in METHODS
        joined = " ".join(report.recommendations)
        assert "reduces disparity by 0.000" not in joined
        assert "best trade-off" not in joined

        # The gap IS there: the same data at a real threshold measures 1.0.
        measured, _ = _caught(analyzer.analyze_method, "multiplicative", threshold=0.5)
        assert measured.original_fairness["demographic_parity_diff"] == pytest.approx(1.0)

    def test_control_an_infinite_threshold_is_still_measured(self):
        """CONTROL against over-correction. +inf is a decision rule ('accept
        nobody'), not an unmeasurable one: every row is decided, the rates are
        a measured 0.0, and nothing is refused."""
        rng = np.random.default_rng(3)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.concatenate([rng.uniform(0.6, 0.99, 100), rng.uniform(0.01, 0.4, 100)])

        result, _ = _caught(
            ReweightingAnalyzer(y_true, y_prob, sens).analyze_method,
            "multiplicative",
            threshold=float("inf"),
        )
        assert result.original_performance["n_decided"] == 200
        assert result.original_fairness["group_rates"] == {"A": 0.0, "B": 0.0}
        assert np.isfinite(result.original_performance["accuracy"])


# ===========================================================================
# A5  a trade-off with an unmeasurable arm is not a trade-off of 0.0
# ===========================================================================


class TestTradeOffWithAnUnmeasurableArm:
    def test_an_uncomputable_calibration_error_is_not_ranked_as_no_cost(self):
        """BEFORE: scores on a 0-100 scale put no row in a calibration bin, so
        original_ece was 0.0 (perfect calibration, fabricated), ece_change was
        +0.46875 against it, and all five methods were ranked and recommended
        on that comparison. `max(0, nan)` would also have read the missing
        value as 'no calibration cost'."""
        rng = np.random.default_rng(7)
        sens = np.array(["A"] * 80 + ["B"] * 80)
        y_true = rng.integers(0, 2, 160)
        y_prob = np.concatenate([rng.uniform(60, 95, 80), rng.uniform(10, 45, 80)])

        result, _ = _caught(
            ReweightingAnalyzer(y_true, y_prob, sens).analyze_method, "multiplicative"
        )
        # The fixture reaches the branch: no row is in [0, 1].
        assert result.calibration_metrics["original_ece_rows_used"] == 0
        assert math.isnan(result.calibration_metrics["original_ece"])
        assert result.calibration_metrics["original_ece"] != 0.0
        # Decisions were still possible, so this is NOT the all-NaN path.
        assert result.original_performance["n_decided"] == 160
        assert math.isnan(result.trade_off_score)

        report, _ = _caught(ReweightingAnalyzer(y_true, y_prob, sens).full_analysis)
        assert report.comparison_summary == {}
        assert set(report.failed_methods) == METHODS
        assert all("NotMeasurable" in r for r in report.failed_methods.values())
        assert report.best_method not in METHODS


# ===========================================================================
# The reweighter's own fit() disparity, pinned here because the sabotage that
# walked through it found no test holding that line.
# ===========================================================================


def test_rejection_option_fit_reports_nan_disparity_when_nothing_was_scored():
    """An all-NaN score vector leaves no decision to compute a rate from, so
    result_.adjusted_metrics['disparity'] must be NaN. 0.0 on this scale is
    perfect parity."""
    from vfairness.post_processing.reweighting.reweighter import RejectionOptionClassifier

    rng = np.random.default_rng(5)
    sens = np.array(["A"] * 30 + ["B"] * 30)
    y_true = rng.integers(0, 2, 60)

    fitted, _ = _caught(
        RejectionOptionClassifier().fit,
        y_true=y_true,
        y_prob=np.full(60, np.nan),
        sensitive_attr=sens,
    )
    assert math.isnan(fitted.result_.adjusted_metrics["disparity"])
    assert math.isnan(fitted.result_.original_metrics["disparity"])

    # CONTROL: a scored input still measures a real gap.
    y_prob = np.concatenate([np.full(30, 0.9), np.full(30, 0.1)])
    ok, _ = _caught(
        RejectionOptionClassifier().fit,
        y_true=y_true,
        y_prob=y_prob,
        sensitive_attr=sens,
    )
    assert np.isfinite(ok.result_.original_metrics["disparity"])


# ===========================================================================
# C1/C2/C3  counterfactual_fairness
# ===========================================================================


def _cf():
    from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
        counterfactual_fairness,
    )

    return counterfactual_fairness


class TestCounterfactualDecidesWhatItCan:
    def test_an_infinite_counterfactual_is_decided_not_deleted(self):
        """BEFORE (after the isfinite mask): flip_rate 0.0, severity 'info',
        "Robust to the perturbation", on 100 rows that genuinely flip from
        reject to accept. The finding was deleted by the refusal."""
        factual = np.concatenate([np.full(100, 0.7), np.full(100, 0.2)])
        counterfactual = np.concatenate([np.full(100, 0.7), np.full(100, np.inf)])

        result, messages = _caught(_cf(), factual, counterfactual, 0.5)

        # The fixture reaches the branch: half the rows are infinite.
        assert result.n == 200, "an infinite score is decidable and must be assessed"
        assert result.flip_rate == pytest.approx(0.5)
        assert result.severity == "critical"
        assert "Robust to the perturbation" not in result.interpretation
        # And the honest half of the old fix survives: the DISTANCE figures
        # cover only the 100 rows that have a readable one, and say so.
        assert result.mean_abs_diff == pytest.approx(0.0)
        assert any("infinite score" in m for m in messages), messages
        assert result.notes and any(
            "excluded from mean_abs_diff" in n and "cover 100 row(s)" in n for n in result.notes
        ), result.notes

    def test_a_distance_nothing_can_read_is_nan_while_the_flip_is_measured(self):
        """Every assessed row is infinite on one side: the SIDE of the
        threshold is readable (so the flip rate is measured) and the DISTANCE
        is not. mean_abs_diff is NaN, and the verdict a reader sees says so
        rather than printing "mean score change nan"."""
        factual = np.full(50, 0.2)
        counterfactual = np.full(50, np.inf)

        result, _ = _caught(_cf(), factual, counterfactual, 0.5)

        assert result.n == 50
        assert result.flip_rate == pytest.approx(1.0)
        assert result.severity == "critical"
        assert math.isnan(result.mean_abs_diff)
        assert math.isnan(result.max_abs_diff)
        assert "NOT MEASURED" in result.interpretation
        assert "nan" not in result.interpretation

    def test_a_nan_score_is_still_refused(self):
        """CONTROL for the narrowed mask: NaN is still not a decision."""
        result, _ = _caught(_cf(), np.full(200, np.nan), np.full(200, np.nan), 0.5)
        assert result.flip_rate is None
        assert result.severity == NOT_ASSESSED
        assert result.n == 0
        assert "Robust to the perturbation" not in result.interpretation

    def test_a_nan_threshold_is_refused(self):
        """BEFORE: a genuine 0.30 flip rate reported as flip_rate 0.0,
        severity 'info', notes [], no warning, "Robust to the perturbation"."""
        rng = np.random.default_rng(3)
        factual = rng.random(200)
        counterfactual = factual.copy()
        counterfactual[:60] = 1 - factual[:60]

        with pytest.raises(ValueError, match="cannot decide any row"):
            _cf()(factual, counterfactual, float("nan"))

        # The rate IS there at a real threshold.
        assert _cf()(factual, counterfactual, 0.5).flip_rate == pytest.approx(0.30)

    def test_a_partial_assessment_qualifies_the_verdict_a_reader_sees(self):
        """BEFORE: severity and interpretation were the unqualified _grade()
        text, so an assessed 4 of 200 read exactly like an assessed 200 of
        200; only `notes` mentioned the narrowing."""
        factual = np.concatenate([np.array([0.2, 0.2, 0.7, 0.7]), np.full(196, np.nan)])
        counterfactual = np.concatenate([np.array([0.7, 0.7, 0.2, 0.2]), np.full(196, np.nan)])

        result, _ = _caught(_cf(), factual, counterfactual, 0.5)

        assert result.n == 4
        assert result.flip_rate == pytest.approx(1.0)
        assert "COVERAGE" in result.interpretation
        assert "4 of 200" in result.interpretation

    def test_control_a_fully_readable_sample_is_graded_exactly_as_before(self):
        """CONTROL. Nothing excluded: no notes, no coverage clause, the same
        rate and severity, and a genuinely unchanged model still reads robust."""
        rng = np.random.default_rng(3)
        factual = rng.random(200)
        counterfactual = factual.copy()
        counterfactual[:60] = 1 - factual[:60]

        result, messages = _caught(_cf(), factual, counterfactual, 0.5)
        assert result.flip_rate == pytest.approx(0.30)
        assert result.severity == "critical"
        assert result.n == 200
        assert result.notes == []
        assert "COVERAGE" not in result.interpretation
        assert messages == [], messages

        fair, _ = _caught(_cf(), factual, factual.copy(), 0.5)
        assert fair.flip_rate == 0.0
        assert fair.severity == "info"
        assert fair.mean_abs_diff == 0.0


# ===========================================================================
# The canonical could-not-check string (src/vfairness/_not_assessed.py)
# ===========================================================================


class TestEveryRefusalUsesTheCanonicalWord:
    """Three capabilities spell the could-not-check state as their own string
    literal rather than importing NOT_ASSESSED. The values agree today, and
    these pins are what makes a future divergence (a rename, a typo, a
    "not assessed" with a space) fail instead of quietly reading as an
    unknown severity a consumer maps to its default. Each asserts by
    EXECUTION at the capability's own public entry, not by grep.
    """

    def test_counterfactual_fairness_refuses_with_the_canonical_word(self):
        result, _ = _caught(_cf(), np.full(20, np.nan), np.full(20, np.nan), 0.5)
        assert result.severity == NOT_ASSESSED

    def test_permutation_importance_refuses_with_the_canonical_word(self):
        pytest.importorskip("sklearn")
        from vfairness.evaluation.vfairness_metrics.attribution import (
            FeatureAttributionExplainer,
        )

        rng = np.random.default_rng(0)
        X = rng.random((50, 3))
        explainer = FeatureAttributionExplainer(lambda Z: Z[:, 0] * 2.0 + Z[:, 1] * 0.5)
        result, _ = _caught(explainer.global_importance, X, y=np.ones(50))

        assert result.contributions
        assert {c.direction for c in result.contributions} == {NOT_ASSESSED}
        assert all(math.isnan(c.importance) for c in result.contributions)

    def test_multivariate_proxy_leakage_refuses_with_the_canonical_word(self):
        pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
        import pandas as pd

        from vfairness.preprocessing.bias_detection.proxy import (
            LEAKAGE_NOT_ASSESSED,
            multivariate_proxy_leakage,
        )

        assert LEAKAGE_NOT_ASSESSED == NOT_ASSESSED

        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "f1": rng.random(20),
                "f2": rng.random(20),
                "gender": ["F"] * 20,  # one class: nothing to reconstruct
            }
        )
        records, _ = _caught(multivariate_proxy_leakage, frame, ["gender"])
        assert records, "the request must still produce a record"
        assert records[0].severity == NOT_ASSESSED
        assert records[0].auc is None
        assert records[0].systemic_leakage is None


# ===========================================================================
# The label-free attribution path measured no direction, and said 'neutral'
# ===========================================================================


class TestAttributionDoesNotInventADirection:
    def _explainer(self):
        from vfairness.evaluation.vfairness_metrics.attribution import (
            FeatureAttributionExplainer,
        )

        return FeatureAttributionExplainer(
            lambda Z: 2.0 * np.asarray(Z)[:, 0] + 0.5 * np.asarray(Z)[:, 1],
            feature_names=["x0", "x1", "x2"],
        )

    def test_the_label_free_path_does_not_report_a_direction_it_never_computed(self):
        """BEFORE: global_importance(X), the DEFAULT call, permutes a column and
        measures the mean ABSOLUTE score change, so it computes no sign at all,
        and then reported direction 'neutral' for every feature. Measured on
        y = 2*x0 + 0.5*x1: feature x0 came back importance 0.734 with the
        verdict 'neutral', which says the dominant feature has no directional
        effect."""
        rng = np.random.default_rng(0)
        X = rng.random((60, 3))

        result, _ = _caught(self._explainer().global_importance, X)

        assert result.method == "permutation_numpy", "this is the label-free path"
        assert {c.direction for c in result.contributions} == {NOT_ASSESSED}
        assert "neutral" not in {c.direction for c in result.contributions}
        # The magnitude IS measured, and ranks correctly: the refusal is about
        # the sign only, not about the importance.
        assert result.top(1)[0].feature == "x0"
        assert result.top(1)[0].importance > 0
        assert any("not measured on this path" in n for n in result.notes)

    def test_a_direction_read_off_a_nan_delta_is_not_neutral(self):
        from vfairness.evaluation.vfairness_metrics.attribution import _direction

        assert _direction(float("nan")) == NOT_ASSESSED
        assert _direction(float("inf")) == NOT_ASSESSED
        # CONTROL: real deltas still get their real verdicts.
        assert _direction(0.5) == "increase"
        assert _direction(-0.5) == "decrease"
        assert _direction(0.0) == "neutral"

    def test_control_a_signed_path_still_reports_increase_and_decrease(self):
        """CONTROL. The local occlusion path DOES compute a sign, and must keep
        reporting it: the fix must not turn every direction into a refusal."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        result, _ = _caught(
            self._explainer().explain_decision, np.array([2.5, 0.1, 1.0]), background=X
        )
        directions = {c.feature: c.direction for c in result.contributions}
        assert directions["x0"] == "increase"
        assert NOT_ASSESSED not in directions.values()


# ===========================================================================
# DistributionMatcher's min_group_size floor, pinned behaviourally
# ===========================================================================


class TestDistributionMatcherRefusalIsPinned:
    def test_a_small_group_with_several_distinct_scores_is_refused(self):
        """The floor was only ever exercised by single-row fixtures, where the
        'one distinct score' branch did the refusing. This group has 5 rows and
        5 DISTINCT scores, so only min_group_size can refuse it. Without the
        floor its 5 rows are rewritten up to the pooled distribution and a
        disparity reduction is reported for a group nobody fitted."""
        from vfairness.post_processing.reweighting.reweighter import DistributionMatcher

        rng = np.random.default_rng(2)
        sens = np.array(["A"] * 195 + ["B"] * 5)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.concatenate([rng.uniform(0.2, 0.9, 195), np.linspace(0.02, 0.10, 5)])
        assert np.unique(y_prob[195:]).size == 5, "the fixture must defeat the unique-value branch"

        matcher = DistributionMatcher()
        out, messages = _caught(
            lambda: matcher.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens).transform(
                y_prob, sens
            )
        )

        assert np.all(np.isnan(out[195:])), out[195:]
        assert np.all(np.isfinite(out[:195])), "group A was fitted and must survive"
        assert "B" in matcher.unfittable_groups_
        assert any("min_group_size" in m for m in messages), messages

    def test_an_all_unscored_input_comes_back_all_nan(self):
        """No all-NaN pin existed for this class, unlike its two siblings."""
        from vfairness.post_processing.reweighting.reweighter import DistributionMatcher

        rng = np.random.default_rng(2)
        sens = np.array(["A"] * 100 + ["B"] * 100)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.full(200, np.nan)

        matcher = DistributionMatcher()
        out, messages = _caught(
            lambda: matcher.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens).transform(
                y_prob, sens
            )
        )
        assert out.size == 200
        assert np.all(np.isnan(out))
        assert messages

    def test_control_two_healthy_groups_are_fitted_and_rewritten(self):
        """CONTROL. A group above the floor is still mapped, not refused."""
        from vfairness.post_processing.reweighting.reweighter import DistributionMatcher

        y_true, y_prob, sens = _healthy()
        matcher = DistributionMatcher()
        out, messages = _caught(
            lambda: matcher.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens).transform(
                y_prob, sens
            )
        )
        assert np.all(np.isfinite(out))
        assert matcher.unfittable_groups_ == {}
        assert messages == [], messages
