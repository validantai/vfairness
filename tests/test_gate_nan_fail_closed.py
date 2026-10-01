"""Wave-3 pins: a metric that could not be measured must FAIL the release gate,
and a report with no score must still render.

Two defects, both found on real data:

1. ``assert_fairness`` treated a NaN metric as a PASS. Every comparison in the
   gate is False for NaN (``abs(nan) > t``, ``nan < t``), so a run that measured
   nothing returned normally and satisfied a release gate. Reproduced with 100
   rows of group A plus 5 of group B at the default ``min_group_size=30``: only
   one group clears the gate, there is no pair to compare, the metric is NaN,
   and the old code returned ``{'demographic_parity_difference': nan}`` without
   raising. ``create_fairness_callback(fail_on_violation=True)`` and the
   ``ci_lower``/``ci_upper`` NaN path had the identical hole.

2. All three visualization entry points crashed on a not-assessable report with
   ``TypeError: '>=' not supported between instances of 'NoneType' and 'float'``.
   ``.get("fairness_score", 0)`` does not help, because the key EXISTS with the
   value None, so the default never fires.

Fail closed everywhere: unmeasurable is could-not-check, never a pass. And a
chart must never substitute a number for an absent score, because 0% reads as a
measured total failure and 100% as a clean certificate. Neither was measured.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.integrations import (
    FairnessAssertionError,
    assert_fairness,
    create_fairness_callback,
)
from vfairness.evaluation.vfairness_metrics.report import classification_fairness_report

# Data builders


def _unmeasurable_data(seed=0):
    """One group of 100 and one of 5.

    At the default ``min_group_size=30`` only group A clears the gate, so there
    is no pair to compare and every between-group metric returns NaN.
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = 100, 5
    sens = np.array(["A"] * n_a + ["B"] * n_b)
    y_true = rng.integers(0, 2, size=n_a + n_b)
    y_pred = rng.integers(0, 2, size=n_a + n_b)
    return y_true, y_pred, sens


def _measured_fair_data(n=400):
    """Two large groups with identical, perfect predictions: genuinely fair and
    fully measurable. This is the over-correction control."""
    sens = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_true = np.array([1, 0] * (n // 2))
    y_pred = y_true.copy()
    return y_true, y_pred, sens


def _measured_unfair_data(n=200):
    """Two large groups, one predicted all-positive and one all-negative: a
    measured, maximal demographic-parity violation."""
    sens = np.array(["A"] * n + ["B"] * n)
    y_true = np.array([1, 0] * n)
    y_pred = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return y_true, y_pred, sens


def _silence_group_warnings():
    return warnings.catch_warnings()


# Defect 1: assert_fairness


class TestAssertFairnessFailsClosedOnNaN:
    def test_unmeasurable_metric_raises(self):
        """THE negative case: the gate must refuse a metric it could not measure."""
        y_true, y_pred, sens = _unmeasurable_data()
        with _silence_group_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(
                    y_true,
                    y_pred,
                    sens,
                    metrics=["demographic_parity_difference"],
                    thresholds={"demographic_parity_difference": 0.1},
                    min_group_size=30,
                )

        err = excinfo.value
        # The metric is named, so the user knows which one to act on.
        assert "demographic_parity_difference" in str(err)
        assert "NOT MEASURABLE" in str(err)
        # Three states, never two: the failure is marked as could-not-check
        # rather than being indistinguishable from a measured violation.
        failure = err.failed_metrics["demographic_parity_difference"]
        assert failure["not_measurable"] is True
        assert math.isnan(failure["value"])

    def test_ratio_metric_is_also_refused_when_unmeasurable(self):
        """A NaN ratio passed the minimum check too (nan < t is False)."""
        y_true, y_pred, sens = _unmeasurable_data()
        with _silence_group_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(
                    y_true,
                    y_pred,
                    sens,
                    metrics=["demographic_parity_ratio"],
                    thresholds={"demographic_parity_ratio": 0.8},
                    min_group_size=30,
                )
        assert excinfo.value.failed_metrics["demographic_parity_ratio"]["not_measurable"] is True

    def test_message_explains_the_group_size_cause(self):
        y_true, y_pred, sens = _unmeasurable_data()
        with _silence_group_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(FairnessAssertionError) as excinfo:
                assert_fairness(
                    y_true,
                    y_pred,
                    sens,
                    metrics=["demographic_parity_difference"],
                    min_group_size=30,
                )
        text = str(excinfo.value)
        assert "Not measurable" in text
        assert "min_group_size" in text

    def test_measured_fair_data_still_passes(self):
        """Over-correction control: a genuinely fair, measurable run must not
        start failing because of the fail-closed rule."""
        y_true, y_pred, sens = _measured_fair_data()
        values = assert_fairness(
            y_true,
            y_pred,
            sens,
            metrics=["demographic_parity_difference", "equal_opportunity_difference"],
            min_group_size=30,
        )
        assert values
        for name, value in values.items():
            assert math.isfinite(value), f"{name} should be measurable here"

    def test_measured_violation_still_fails_and_is_not_marked_unmeasurable(self):
        y_true, y_pred, sens = _measured_unfair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["demographic_parity_difference"],
                thresholds={"demographic_parity_difference": 0.1},
                min_group_size=30,
            )
        failure = excinfo.value.failed_metrics["demographic_parity_difference"]
        assert failure["not_measurable"] is False
        assert math.isfinite(failure["value"])


class TestAssertFairnessCIPathFailsClosed:
    """The ci_must_exclude_zero check silently did not run when the interval was
    NaN, and the metric passed on the strength of a check that never happened."""

    @staticmethod
    def _patch_metric(monkeypatch, value, ci):
        from vfairness.evaluation.vfairness_metrics import analyzer as analyzer_mod

        def fake_compute_all_metrics(self, **kwargs):
            return {
                "demographic_parity_difference": analyzer_mod.MetricResult(
                    metric_name="demographic_parity_difference",
                    value=value,
                    confidence_interval=ci,
                )
            }

        monkeypatch.setattr(
            analyzer_mod.FairnessAnalyzer, "compute_all_metrics", fake_compute_all_metrics
        )

    def test_unavailable_ci_fails_when_ci_must_exclude_zero(self, monkeypatch):
        self._patch_metric(monkeypatch, 0.01, (float("nan"), float("nan")))
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["demographic_parity_difference"],
                include_ci=True,
                ci_must_exclude_zero=True,
                n_bootstrap=10,
            )
        failure = excinfo.value.failed_metrics["demographic_parity_difference"]
        assert failure["not_measurable"] is True
        assert "NOT MEASURABLE" in failure["reason"]

    def test_absent_ci_bound_also_fails(self, monkeypatch):
        """A None bound is could-not-check too, and np.isnan(None) would itself
        have raised TypeError rather than failing the gate cleanly."""
        self._patch_metric(monkeypatch, 0.01, (None, None))
        y_true, y_pred, sens = _measured_fair_data()
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(
                y_true,
                y_pred,
                sens,
                metrics=["demographic_parity_difference"],
                include_ci=True,
                ci_must_exclude_zero=True,
                n_bootstrap=10,
            )
        assert excinfo.value.failed_metrics["demographic_parity_difference"]["not_measurable"]

    def test_available_ci_excluding_zero_still_passes(self, monkeypatch):
        """Over-correction control for the CI path."""
        self._patch_metric(monkeypatch, 0.01, (0.005, 0.02))
        y_true, y_pred, sens = _measured_fair_data()
        values = assert_fairness(
            y_true,
            y_pred,
            sens,
            metrics=["demographic_parity_difference"],
            include_ci=True,
            ci_must_exclude_zero=True,
            n_bootstrap=10,
        )
        assert values["demographic_parity_difference"] == pytest.approx(0.01)


# Defect 1b: create_fairness_callback


class TestFairnessCallbackFailsClosed:
    def test_unmeasurable_metric_raises(self):
        y_true, y_pred, sens = _unmeasurable_data()
        callback = create_fairness_callback(
            "group",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        with _silence_group_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="could not be measured"):
                callback(y_true, y_pred, sens)

    def test_measured_fair_data_still_passes(self):
        y_true, y_pred, sens = _measured_fair_data()
        callback = create_fairness_callback(
            "group",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        results = callback(y_true, y_pred, sens)
        assert math.isfinite(results["demographic_parity_difference"])

    def test_measured_violation_still_raises(self):
        y_true, y_pred, sens = _measured_unfair_data()
        callback = create_fairness_callback(
            "group",
            thresholds={"demographic_parity_difference": 0.1},
            fail_on_violation=True,
        )
        with pytest.raises(ValueError, match="Fairness violation"):
            callback(y_true, y_pred, sens)

    def test_ratio_direction_is_not_inverted(self):
        """A ratio threshold is a MINIMUM. The callback compared
        ``abs(ratio) > minimum``, which graded a maximal ratio violation as a
        pass and a healthy ratio as a violation."""
        y_true, y_pred, sens = _measured_unfair_data()
        callback = create_fairness_callback(
            "group",
            thresholds={"demographic_parity_ratio": 0.8},
            fail_on_violation=True,
        )
        with pytest.raises(ValueError, match="minimum"):
            callback(y_true, y_pred, sens)

    def test_healthy_ratio_is_not_a_violation(self):
        """Over-correction control for the direction fix."""
        y_true, y_pred, sens = _measured_fair_data()
        callback = create_fairness_callback(
            "group",
            thresholds={"demographic_parity_ratio": 0.8},
            fail_on_violation=True,
        )
        results = callback(y_true, y_pred, sens)
        assert results["demographic_parity_ratio"] >= 0.8


# Defect 2: the three visualization entry points


@pytest.fixture(scope="module")
def not_assessable_report():
    y_true, y_pred, sens = _unmeasurable_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(y_true, y_pred, sens, min_group_size=30)


@pytest.fixture(scope="module")
def measured_report():
    y_true, y_pred, sens = _measured_fair_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(y_true, y_pred, sens, min_group_size=30)


@pytest.fixture(autouse=True, scope="module")
def _agg_backend():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestVisualizationRendersCouldNotCheck:
    def test_report_really_has_no_score(self, not_assessable_report):
        """Pin the precondition: without this the tests below prove nothing."""
        assert not_assessable_report["assessment"]["fairness_score"] is None

    def test_plot_fairness_metrics_renders(self, not_assessable_report, tmp_path):
        from vfairness.evaluation.vfairness_metrics.visualization import plot_fairness_metrics

        ax = plot_fairness_metrics(not_assessable_report)
        assert ax is not None
        texts = [t.get_text() for t in ax.texts]
        assert any("NOT ASSESSABLE" in t for t in texts)
        # No fabricated number stands in for the missing score.
        assert not any("Overall Score: 0%" in t or "Overall Score: 100%" in t for t in texts)
        out = tmp_path / "metrics.png"
        ax.get_figure().savefig(out)
        assert out.stat().st_size > 0

    def test_plot_fairness_report_renders(self, not_assessable_report, tmp_path):
        from vfairness.evaluation.vfairness_metrics.visualization import plot_fairness_report

        fig = plot_fairness_report(not_assessable_report)
        assert fig is not None
        title = fig._suptitle.get_text()
        assert "NOT ASSESSABLE" in title
        assert "0%" not in title and "100%" not in title
        out = tmp_path / "report.png"
        fig.savefig(out)
        assert out.stat().st_size > 0

    def test_create_fairness_dashboard_renders(self, not_assessable_report):
        pytest.importorskip("plotly")
        from vfairness.evaluation.vfairness_metrics.visualization import create_fairness_dashboard

        fig = create_fairness_dashboard(not_assessable_report)
        assert fig is not None
        title = fig.layout.title.text
        assert "NOT ASSESSABLE" in title
        assert "0%" not in title and "100%" not in title

    def test_measured_report_still_shows_its_score(self, measured_report, tmp_path):
        """Over-correction control: a measured run keeps its number."""
        from vfairness.evaluation.vfairness_metrics.visualization import (
            plot_fairness_metrics,
            plot_fairness_report,
        )

        assert measured_report["assessment"]["fairness_score"] is not None

        ax = plot_fairness_metrics(measured_report)
        assert any("Overall Score:" in t.get_text() for t in ax.texts)
        assert not any("NOT ASSESSABLE" in t.get_text() for t in ax.texts)

        fig = plot_fairness_report(measured_report)
        assert "Score:" in fig._suptitle.get_text()
        assert "NOT ASSESSABLE" not in fig._suptitle.get_text()
        out = tmp_path / "measured.png"
        fig.savefig(out)
        assert out.stat().st_size > 0
