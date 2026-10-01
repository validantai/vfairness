"""Audit 6, wave 4, lane E: statistics PROVENANCE honesty.

Three findings, all reproduced by execution on 2026-09-09 before any edit:

* **S-09** ``_statistics.select_method(sample_sizes, prefer_bayesian=...)``
  accepted ``prefer_bayesian`` and never read it: the same
  ``('mixed', {...})`` came back for True and for False. It cannot be
  implemented, because no Bayesian estimator for a disparity metric exists in
  this library (F18), so it is REFUSED.
* **S-09b, the part that reaches a document.** ``select_method`` returns a
  RECOMMENDATION ('bayesian' / 'mixed'), and ``report.py`` writes that label
  into the report as ``statistical_validation.method_used`` while every
  interval in the same report was built by a stratified bootstrap. The library
  side is pinned here; the report side is recorded in
  :func:`test_report_provenance_must_name_the_method_that_ran`.
* **S-16a** ``FairExplAIner._get_benchmark_context`` never read ``value``, so a
  demographic_parity_difference of 0.001 and of 0.85 produced the identical
  sentence. It now names the band the measured value falls in, and says so
  when it cannot place the value. Its sibling in the same lines: absent band
  edges were filled in with literals, so Cohen's d was given the benchmarks
  0.05 / 0.10 / 0.15, none of which appears in its definition (its bands are
  0.2 / 0.5 / 0.8).
* **S-16b** ``get_group_metrics_with_ci(method='bootstrap', random_state=...)``
  never bootstrapped and never read the seed: two different seeds returned
  byte-identical bounds and ``result.method`` said ``'wilson_score'``. The
  signature no longer claims a bootstrap.

Every refusal below is paired with an over-correction control asserting the
MEASURED numbers the honest path still returns.
"""

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    compute_metric_with_ci,
    select_method,
)
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
    get_group_metrics_with_ci,
)
from vfairness.evaluation.vfairness_metrics.explainer import (
    STATISTICAL_MEASURES,
    FairExplAIner,
)
from vfairness.evaluation.vfairness_metrics.report import classification_fairness_report
from vfairness.exceptions import ConfigurationError


def _mixed_size_data():
    """Group A of 20 rows (small) and group B of 200 rows, fixed seed."""
    rng = np.random.default_rng(7)
    n_a, n_b = 20, 200
    sens = np.array(["A"] * n_a + ["B"] * n_b)
    y_true = rng.integers(0, 2, n_a + n_b)
    y_pred = np.concatenate([rng.binomial(1, 0.3, n_a), rng.binomial(1, 0.6, n_b)])
    return y_true, y_pred, sens


# S-09: prefer_bayesian is refused, not ignored


class TestSelectMethodPreferBayesian:
    def test_prefer_bayesian_is_refused(self):
        """REFUSAL PIN: the inert argument now raises instead of being dropped."""
        with pytest.raises(ConfigurationError) as exc:
            select_method({"a": 20, "b": 100}, prefer_bayesian=True)
        message = str(exc.value)
        # The refusal must name what IS supported, not just say no.
        assert "no Bayesian estimator for a disparity metric" in message
        assert "bayesian_difference_ci" in message
        assert "prefer_bayesian=False" in message

    def test_prefer_bayesian_refused_positionally_too(self):
        with pytest.raises(ConfigurationError):
            select_method({"a": 20}, True)

    def test_recommendation_still_returns_its_real_answer(self):
        """OVER-CORRECTION CONTROL: exact labels, not membership of a set."""
        assert select_method({"a": 20, "b": 40, "c": 120}) == (
            "mixed",
            {"a": "bayesian", "b": "bootstrap_enhanced", "c": "bootstrap"},
        )
        assert select_method({"A": 100, "B": 100}) == (
            "bootstrap",
            {"A": "bootstrap", "B": "bootstrap"},
        )
        assert select_method({"A": 20, "B": 25}) == (
            "bayesian",
            {"A": "bayesian", "B": "bayesian"},
        )

    def test_explicit_false_is_still_accepted(self):
        """The refusal must not swallow the documented default."""
        assert select_method({"a": 20, "b": 100}, prefer_bayesian=False) == (
            "mixed",
            {"a": "bayesian", "b": "bootstrap"},
        )
        assert select_method({"a": 20, "b": 100}, False) == select_method({"a": 20, "b": 100})


# S-09b: the recorded provenance must be the method that actually ran


class TestProvenanceOfDisparityIntervals:
    def test_disparity_ci_never_reports_bayesian_however_it_was_asked(self):
        """The interval names the bootstrap that ran, for every accepted method."""
        y_true, y_pred, sens = _mixed_size_data()
        for asked in ("auto", "bootstrap", "bayesian"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = compute_metric_with_ci(
                    demographic_parity_difference,
                    y_true,
                    y_pred,
                    sens,
                    n_bootstrap=200,
                    method=asked,
                    random_state=42,
                    min_group_size=10,
                )
            assert res.method == "stratified_bootstrap_fold_debiased", asked
            assert res.method != "bayesian"
            assert res.n_bootstrap > 0, asked

    def test_report_provenance_must_name_the_method_that_ran(self):
        """report.py records select_method's RECOMMENDATION as method_used.

        The unconditional half of this test is the library truth: every
        interval in the report was built by the stratified bootstrap. The
        conditional half records the OPEN defect in ``report.py`` lines 623 and
        1022, which this lane may not edit: ``method_used`` is written from
        ``select_method`` and reads 'mixed' (per-group 'bayesian') for exactly
        this data. When report.py is fixed to record
        ``StatisticalResult.method``, the imperative xfail stops firing and the
        final assertion takes over.
        """
        y_true, y_pred, sens = _mixed_size_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_fairness_report(
                y_true,
                y_pred,
                sens,
                min_group_size=10,
                include_ci=True,
                n_bootstrap=200,
                random_state=42,
            )

        actually_ran = {ci["method"] for ci in report["metrics_with_ci"].values()}
        assert actually_ran == {"stratified_bootstrap_fold_debiased"}

        claimed = report["statistical_validation"]["method_used"]
        per_group = report["statistical_validation"]["per_group_methods"]
        if (
            claimed in {"mixed", "bayesian", "bootstrap_enhanced"}
            or "bayesian" in per_group.values()
        ):
            pytest.xfail(
                f"OPEN, report.py:623/1022 (not editable from lane E): "
                f"method_used={claimed!r}, per_group_methods={per_group!r}, while "
                f"every interval carries method={sorted(actually_ran)!r}. Fix: write "
                f"the method each StatisticalResult reports, and keep the size-based "
                f"select_method output under a name that says it is a recommendation."
            )
        assert claimed in actually_ran


# S-16a: the benchmark context names the band the MEASURED value falls in


class TestBenchmarkContextUsesTheValue:
    @staticmethod
    def _explainer():
        return FairExplAIner()

    def test_passing_and_catastrophic_values_no_longer_read_alike(self):
        """REFUSAL PIN for the original defect: the two must differ, by band."""
        ex = self._explainer()
        mdef = ex.metrics_definitions["demographic_parity_difference"]
        good = ex._get_benchmark_context("demographic_parity_difference", 0.001, mdef)
        bad = ex._get_benchmark_context("demographic_parity_difference", 0.85, mdef)
        assert good != bad
        assert "The measured value of 0.0010 falls in the EXCELLENT band." in good
        assert "The measured value of 0.8500 falls in the CRITICAL band." in bad

    @pytest.mark.parametrize(
        "value,band",
        [
            (0.001, "EXCELLENT"),
            (0.07, "ACCEPTABLE"),
            (0.12, "CONCERNING"),
            (0.18, "HIGH CONCERN"),
            (0.85, "CRITICAL"),
            (-0.85, "CRITICAL"),  # magnitude, so the sign hole stays closed
        ],
    )
    def test_lower_is_better_bands(self, value, band):
        """OVER-CORRECTION CONTROL: every band is reachable and named exactly."""
        ex = self._explainer()
        mdef = ex.metrics_definitions["demographic_parity_difference"]
        out = ex._get_benchmark_context("demographic_parity_difference", value, mdef)
        assert f"falls in the {band} band." in out
        # The printed benchmark numbers are still the metric's own.
        assert "Excellent (≤0.05), Acceptable (≤0.10), Concerning (>0.15)." in out

    @pytest.mark.parametrize(
        "value,band,severity",
        [
            (0.95, "EXCELLENT", "info"),
            (0.85, "ACCEPTABLE", "low"),
            (0.75, "CONCERNING", "medium"),
            (0.40, "CRITICAL", "critical"),
        ],
    )
    def test_higher_is_better_band_agrees_with_the_verdict(self, value, band, severity):
        """The band named must be the band the verdict beside it grades."""
        ex = self._explainer()
        mdef = ex.metrics_definitions["demographic_parity_ratio"]
        out = ex._get_benchmark_context("demographic_parity_ratio", value, mdef)
        assert f"falls in the {band} band." in out
        assert "Excellent (≥0.90), Acceptable (≥0.80), Concerning (<0.70)." in out
        _, graded = ex._evaluate_value("demographic_parity_ratio", value, 0.8, mdef)
        assert graded == severity

    def test_unmeasured_value_is_not_given_a_band(self):
        """REFUSAL PIN: NaN is could-not-place, never the best or the worst band."""
        ex = self._explainer()
        mdef = ex.metrics_definitions["demographic_parity_difference"]
        out = ex._get_benchmark_context("demographic_parity_difference", float("nan"), mdef)
        assert "was NOT placed in a band" in out
        assert "the value is not a number" in out
        for band in ("EXCELLENT", "ACCEPTABLE", "CONCERNING", "HIGH CONCERN", "CRITICAL"):
            assert f"falls in the {band} band" not in out

    def test_absent_bands_are_not_invented(self):
        """SIBLING PIN: Cohen's d has no excellent/acceptable/concerning edges.

        Measured 2026-09-09 before the fix, this call announced "Industry
        Benchmarks: Excellent (=<0.05), Acceptable (=<0.10), Concerning
        (>0.15)" for a metric whose real bands are 0.2 / 0.5 / 0.8.
        """
        ex = self._explainer()
        out = ex._get_benchmark_context("cohens_d", 0.35, STATISTICAL_MEASURES["cohens_d"])
        assert "Industry Benchmarks" not in out
        for invented in ("0.05", "0.10", "0.15", "0.90", "0.80", "0.70"):
            assert invented not in out
        assert "no excellent/acceptable/concerning bands" in out
        assert "large, medium, negligible, small" in out

    def test_band_reaches_the_public_explanation(self):
        """The mount, not only the helper: explain_metric carries the band."""
        ex = self._explainer()
        exp = ex.explain_metric("demographic_parity_difference", 0.85)
        assert "falls in the CRITICAL band." in exp.benchmark_context
        assert exp.severity == "critical"


# S-16b: no bootstrap here, so the signature stops claiming one


class TestGroupMetricsWithCiSignature:
    @staticmethod
    def _data():
        rng = np.random.default_rng(3)
        n = 200
        sens = np.array(["A"] * n + ["B"] * n)
        y_true = rng.integers(0, 2, 2 * n)
        y_pred = rng.integers(0, 2, 2 * n)
        return y_true, y_pred, sens

    def test_bootstrap_literal_is_refused(self):
        """REFUSAL PIN: 'bootstrap' used to be accepted and served Wilson."""
        y_true, y_pred, sens = self._data()
        with pytest.raises(ConfigurationError) as exc:
            get_group_metrics_with_ci(y_true, y_pred, sens, method="bootstrap")
        message = str(exc.value)
        assert "No bootstrap runs in this function" in message
        assert "demographic_parity_difference_with_ci" in message

    def test_random_state_is_refused(self):
        """REFUSAL PIN: nothing here is stochastic, so a seed cannot be honoured."""
        y_true, y_pred, sens = self._data()
        with pytest.raises(ConfigurationError) as exc:
            get_group_metrics_with_ci(y_true, y_pred, sens, random_state=1)
        assert "no resampling to seed" in str(exc.value)

    def test_wilson_path_returns_its_measured_answer(self):
        """OVER-CORRECTION CONTROL: exact point estimate and exact bounds."""
        y_true, y_pred, sens = self._data()
        res = get_group_metrics_with_ci(y_true, y_pred, sens)
        pr = res["A"]["positive_rate"]
        observed = float(np.mean(y_pred[sens == "A"]))
        assert observed == pytest.approx(0.505)
        assert pr.point_estimate == pytest.approx(observed)
        assert pr.lower_bound == pytest.approx(0.43625515335599574, abs=1e-12)
        assert pr.upper_bound == pytest.approx(0.5735563120641731, abs=1e-12)
        assert pr.method == "wilson_score"
        assert pr.n_bootstrap == 0
        # 'wilson' names what 'auto' already ran for a group this size.
        forced = get_group_metrics_with_ci(y_true, y_pred, sens, method="wilson")
        assert forced["A"]["positive_rate"].lower_bound == pytest.approx(pr.lower_bound, abs=1e-12)
        assert forced["A"]["positive_rate"].upper_bound == pytest.approx(pr.upper_bound, abs=1e-12)

    def test_auto_still_routes_a_small_group_to_the_beta_posterior(self):
        """OVER-CORRECTION CONTROL: the small-group credible interval survives."""
        y_pred = np.array([1] * 5 + [0] * 15 + [0, 1] * 30)
        y_true = np.array([1] * 10 + [0] * 10 + [0, 1] * 30)
        sens = np.array(["g0"] * 20 + ["g1"] * 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = get_group_metrics_with_ci(y_true, y_pred, sens, min_group_size=10)
        small = res["g0"]["positive_rate"]
        assert small.method == "bayesian_beta_binomial"
        assert small.point_estimate == pytest.approx(0.25)  # OBSERVED rate, 5/20
        assert small.metadata["posterior_mean"] == pytest.approx(6 / 22)
        assert small.lower_bound == pytest.approx(0.1128094039219408, abs=1e-12)
        assert small.upper_bound == pytest.approx(0.4716598276546214, abs=1e-12)
        large = res["g1"]["positive_rate"]
        assert large.method == "wilson_score"
        assert large.point_estimate == pytest.approx(0.5)
        assert large.lower_bound == pytest.approx(0.37732489885874304, abs=1e-12)
        assert large.upper_bound == pytest.approx(0.622675101141257, abs=1e-12)


# S-16a sibling: an interval that was never built is not read as a measurement


class TestConfidenceIntervalExplanationThreeStates:
    """``explain_confidence_interval`` graded NaN bounds as a measurement.

    Every comparison against NaN is False, so ``nan <= 0 <= nan`` was False and
    the "excludes zero" branch fired: measured 2026-09-09,
    ``explain_confidence_interval(m, nan, nan, nan)`` returned severity 'info'
    with "The interval excludes zero, suggesting the observed disparity is
    statistically significant." ``compute_metric_with_ci`` returns exactly
    those NaN bounds when no rows survive the exclusions and when the bootstrap
    fails, so that prose reached a report.
    """

    @staticmethod
    def _explainer():
        return FairExplAIner()

    def test_nan_interval_is_could_not_check(self):
        """REFUSAL PIN."""
        exp = self._explainer().explain_confidence_interval(
            "demographic_parity_difference", float("nan"), float("nan"), float("nan")
        )
        assert exp.severity == "could_not_check"
        assert exp.evaluation.startswith("COULD NOT CHECK")
        assert "statistically significant" not in exp.evaluation
        assert "precise" not in exp.evaluation.split("how precise it is")[0]

    @pytest.mark.parametrize(
        "point,lower,upper",
        [
            (0.12, float("nan"), 0.20),
            (float("nan"), 0.04, 0.20),
            (0.12, 0.04, float("inf")),
            (0.12, "0.04", 0.20),
        ],
    )
    def test_any_missing_endpoint_refuses(self, point, lower, upper):
        exp = self._explainer().explain_confidence_interval(
            "demographic_parity_difference", point, lower, upper
        )
        assert exp.severity == "could_not_check"

    def test_measured_interval_still_gets_its_real_reading(self):
        """OVER-CORRECTION CONTROL: the honest path is untouched."""
        ex = self._explainer()
        excludes = ex.explain_confidence_interval("demographic_parity_difference", 0.12, 0.04, 0.20)
        assert excludes.severity == "info"
        assert "[0.0400, 0.2000]" in excludes.evaluation
        assert "excludes zero" in excludes.evaluation
        straddles = ex.explain_confidence_interval(
            "demographic_parity_difference", 0.02, -0.05, 0.09
        )
        assert "includes zero" in straddles.evaluation
        assert "[-0.0500, 0.0900]" in straddles.evaluation

    def test_absent_bound_in_a_report_is_not_read_as_zero(self):
        """REFUSAL PIN: a missing key used to default to a readable 0.0000."""
        report = {
            "metrics": {},
            "metrics_with_ci": {
                "demographic_parity_difference": {"point_estimate": 0.12},
            },
            "statistical_validation": {"confidence_level": 0.95},
        }
        out = self._explainer().explain_report(report)
        ci = out["statistical"]["demographic_parity_difference_ci"]
        assert ci["severity"] == "could_not_check"
        assert "0.0000" not in ci["evaluation"]

    def test_confidence_level_comes_from_the_interval_that_was_built(self):
        """The per-interval level wins over the run-level summary."""
        report = {
            "metrics": {},
            "metrics_with_ci": {
                "demographic_parity_difference": {
                    "point_estimate": 0.12,
                    "lower_bound": 0.04,
                    "upper_bound": 0.20,
                    "confidence_level": 0.99,
                    "interval_type": "confidence",
                },
            },
            "statistical_validation": {"confidence_level": 0.80},
        }
        out = self._explainer().explain_report(report)
        ci = out["statistical"]["demographic_parity_difference_ci"]
        assert "The 99% confidence interval [0.0400, 0.2000]" in ci["evaluation"]
        assert "80%" not in ci["evaluation"]


class TestReportProvenanceIsWhatRanNotWhatWasRecommended:
    """S-09b, which lane E recorded as BLOCKED because report.py was outside
    its file list. ``statistical_validation.method_used`` was written from
    ``select_method``, a SIZE-BASED RECOMMENDATION that answers 'bayesian' for
    small groups, while every disparity interval runs a stratified bootstrap.
    A report therefore named a statistical method that never ran."""

    @staticmethod
    def _small_group_report():
        import numpy as np

        from vfairness.evaluation.vfairness_metrics.report import (
            classification_fairness_report,
        )

        rng = np.random.default_rng(3)
        n = 26
        sens = np.array(["a"] * 13 + ["b"] * 13)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return classification_fairness_report(
                rng.integers(0, 2, n),
                rng.integers(0, 2, n),
                sens,
                include_ci=True,
                n_bootstrap=60,
                min_group_size=5,
            )

    def test_method_used_names_the_method_that_actually_ran(self):
        sv = self._small_group_report()["statistical_validation"]
        assert "bayesian" not in sv["method_used"], sv["method_used"]
        assert "bootstrap" in sv["method_used"], sv["method_used"]

    def test_the_recommendation_is_still_reported_but_labelled_as_one(self):
        """The size heuristic is useful information. It is kept, under a name
        that says what it is, instead of masquerading as provenance."""
        sv = self._small_group_report()["statistical_validation"]
        assert sv["size_based_recommendation"] == "bayesian"
        assert sv["per_group_size_recommendation"] == {"a": "bayesian", "b": "bayesian"}

    def test_every_method_that_ran_is_listed(self):
        sv = self._small_group_report()["statistical_validation"]
        assert sv["methods_that_ran"], sv
        assert all("bayesian" not in m for m in sv["methods_that_ran"]), sv["methods_that_ran"]

    def test_control_large_groups_are_unaffected(self):
        """Over-correction control: where the recommendation and the reality
        already agreed, nothing changes."""
        import numpy as np

        from vfairness.evaluation.vfairness_metrics.report import (
            classification_fairness_report,
        )

        rng = np.random.default_rng(5)
        n = 600
        sens = np.array(["a"] * 300 + ["b"] * 300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = classification_fairness_report(
                rng.integers(0, 2, n),
                rng.integers(0, 2, n),
                sens,
                include_ci=True,
                n_bootstrap=60,
                min_group_size=5,
            )
        sv = rep["statistical_validation"]
        assert sv["size_based_recommendation"] == "bootstrap"
        assert "bootstrap" in sv["method_used"]
