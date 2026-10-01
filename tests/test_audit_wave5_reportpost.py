"""Audit wave 5 pins: reporting + post-processing lows.

Each test pins one adversarially confirmed finding so it cannot regress:

1. simulate_threshold_change hid a 0 -> N alert increase as 0.0% change.
2. Standalone what-if slider stepped finer than the precomputed grid.
3. 'Most frequently alerting metric' was picked by mean value and reported
   the global alert count.
4. metric_compliance docstring contradicted the doubled-penalty formula.
5. Risk-register doctest asserted an impossible risk_score (17).
6. IntersectionalCalibrator.get_group_statistics returned None.
7. TemperatureScaling comment claimed a line search it never did.
8. Threshold-independent ECE was recomputed inside the threshold loop.
9. mitigation_pareto carried a no-op NaN guard on an integer length.
"""

import doctest
import inspect
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.reporting.interactive import simulate_threshold_change
from vfairness.operations.reporting.reports import ReportTier, _narrate_metrics
from vfairness.operations.reporting.store import MetricsStore, StoredMetricRecord


def _store_with_values(values, alerts=None):
    """Small store with one metric; alerts default to False."""
    store = MetricsStore()
    now = datetime.now()
    alerts = alerts or [False] * len(values)
    for v, a in zip(values, alerts):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=1),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=float(v),
                group="a",
                alert=bool(a),
            )
        )
    return store


class TestWhatIfHonesty:
    """The subject here is the PERCENTAGE, not the direction: a change from 0
    to N alerts has no percentage, and 0.0% would hide it.

    Values flipped 2026-09-10 (READINESS-5). They used to sit below the
    threshold, because the simulator projected an alert for every value BELOW
    it, whatever the metric. demographic_parity is a violation MAGNITUDE, so
    0.55 against a bound of 0.9 is a pass, and these fixtures were describing
    the inverted comparison rather than the percentage rule they are named for.
    A parity difference of 0.95 is the unfair reading, so that is what breaches
    here now. Every count and percentage asserted below is unchanged.
    """

    def test_zero_baseline_reports_absolute_change_not_zero_pct(self):
        # 0 current alerts, 4 projected: 0.0% would hide a real increase.
        store = _store_with_values([0.95, 0.96, 0.94, 0.97])
        res = simulate_threshold_change(store, "demographic_parity", 0.9)
        assert res["current_alerts"] == 0
        assert res["projected_alerts"] == 4
        assert res["change_abs"] == 4
        assert res["change_pct"] is None

    def test_zero_baseline_zero_projected_is_zero_pct(self):
        store = _store_with_values([0.55, 0.60])
        res = simulate_threshold_change(store, "demographic_parity", 0.9)
        assert res["projected_alerts"] == 0
        assert res["change_abs"] == 0
        assert res["change_pct"] == 0.0

    def test_nonzero_baseline_keeps_percent(self):
        store = _store_with_values([0.95, 0.96], alerts=[True, False])
        res = simulate_threshold_change(store, "demographic_parity", 0.9)
        assert res["current_alerts"] == 1
        assert res["projected_alerts"] == 2
        assert res["change_abs"] == 1
        assert res["change_pct"] == 100.0

    def test_empty_store_refuses_rather_than_reporting_no_change(self):
        """Renamed and inverted 2026-09-17 (Beta Go-Live Stage 2).

        It used to assert change_abs == 0 and change_pct == 0.0 for a metric
        that does not exist in an EMPTY store. Nothing was simulated, and "no
        change" is a finding: it says the proposed threshold would make no
        difference, which is exactly what an operator would read before deciding
        not to bother. The simulation now refuses, and says which of the two it
        is.
        """
        res = simulate_threshold_change(MetricsStore(), "nope", 0.8)
        for field in ("current_alerts", "projected_alerts", "change_abs", "change_pct"):
            assert res[field] is None, f"{field}={res[field]!r} for a simulation that never ran"
        assert res["records_simulated"] == 0
        assert "COULD NOT CHECK" in res["not_simulated_reason"]
        assert (
            "not a finding that the threshold change would have no effect"
            in (res["not_simulated_reason"])
        )


class TestWhatIfSliderGrid:
    def test_slider_positions_all_have_precomputed_data(self):
        pytest.importorskip("plotly")
        from vfairness.operations.reporting.interactive import InteractiveDashboard

        store = _store_with_values([0.55, 0.60, 0.58, 0.62])
        html = InteractiveDashboard(store).to_standalone_html()

        slider = re.search(r'id="threshold-slider"[^>]*', html).group(0)
        lo = float(re.search(r'min="([\d.]+)"', slider).group(1))
        hi = float(re.search(r'max="([\d.]+)"', slider).group(1))
        step = float(re.search(r'step="([\d.]+)"', slider).group(1))

        whatif = re.search(r"var whatifData = (\{.*?\});", html, re.S).group(1)
        keys = set(re.findall(r'"(0\.\d+)"\s*:\s*\{', whatif))

        # Every reachable slider position must hit a precomputed entry.
        pos = lo
        while pos <= hi + 1e-9:
            assert str(round(pos, 1)) in keys, (
                f"slider position {pos} has no precomputed what-if data"
            )
            pos += step

    def test_js_normalises_float_slider_values(self):
        pytest.importorskip("plotly")
        from vfairness.operations.reporting.interactive import InteractiveDashboard

        store = _store_with_values([0.55, 0.60])
        html = InteractiveDashboard(store).to_standalone_html()
        assert "Math.round(parseFloat(value) * 10) / 10" in html


class TestMostFrequentlyAlertingMetric:
    def test_selected_by_alert_count_not_mean_value(self):
        # A alerts 3 times with low values; B alerts once with a high value.
        # The old mean-value idxmax picked B and reported the global count.
        df = pd.DataFrame(
            {
                "metric_name": ["A", "A", "A", "B", "B"],
                "group": ["g1", "g2", "g1", "g1", "g2"],
                "value": [0.1, 0.1, 0.1, 0.9, 0.5],
                "alert": [True, True, True, True, False],
            }
        )
        txt = _narrate_metrics(df, ReportTier.OPERATIONAL)
        assert "Most frequently alerting metric: A" in txt
        assert "3 of 4 alerts" in txt


class TestHealthScoreDocFormulaAgreement:
    def test_docstring_describes_doubled_penalty(self):
        doc = MetricsStore.compute_health_score.__doc__
        assert "1 - 2 * alert_fraction" in doc
        assert "Fraction of recent metrics without alerts." not in doc

    def test_metric_compliance_component_matches_doubled_formula(self):
        # 50% alerting records: 100 * (1 - 2 * 0.5) = 0, not 50.
        store = _store_with_values([0.5, 0.6, 0.7, 0.8], alerts=[True, True, False, False])
        hs = store.compute_health_score()
        assert hs.components["metric_compliance"] == pytest.approx(0.0)


class TestRiskRegisterDoctest:
    def test_doctest_value_is_reachable(self):
        import vfairness.operations.reporting.compliance as comp

        bias = {
            "findings": [
                {
                    "id": "B001",
                    "category": "demographic_parity",
                    "description": "Gender disparity",
                    "confidence": 0.85,
                    "affected_groups": ["female"],
                }
            ]
        }
        metrics = {
            "metrics": [
                {
                    "name": "demographic_parity_difference",
                    "value": 0.15,
                    "threshold": 0.1,
                    "passed": False,
                    "severity": 4,
                }
            ]
        }
        reg = comp.generate_risk_register_from_audit(bias, metrics, "credit")
        # likelihood round(4.25)=4, severity 4 -> 16 (17 was impossible).
        assert reg[0]["risk_score"] == 16

    def test_doctest_runs_clean(self):
        # Run only this function's doctest (module-wide testmod would drag
        # in unrelated examples that need live monitoring objects).
        import vfairness.operations.reporting.compliance as comp

        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner(verbose=False)
        tests = finder.find(
            comp.generate_risk_register_from_audit,
            globs={"generate_risk_register_from_audit": comp.generate_risk_register_from_audit},
        )
        assert tests
        for t in tests:
            runner.run(t)
        assert runner.failures == 0
        assert runner.tries > 0


class TestGroupStatistics:
    def _fitted(self):
        from vfairness.post_processing.calibration.group_calibrator import IntersectionalCalibrator

        rng = np.random.default_rng(0)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_prob = np.clip(rng.random(n), 0.01, 0.99)
        attrs = pd.DataFrame(
            {
                "g": rng.choice(["x", "y"], n),
                "r": rng.choice(["u", "v"], n),
            }
        )
        cal = IntersectionalCalibrator(method="platt", min_group_size=10)
        cal.fit(y_true, y_prob, attrs)
        return cal, n

    def test_returns_counts_dict(self):
        cal, n = self._fitted()
        stats = cal.get_group_statistics()
        assert isinstance(stats, dict) and stats
        assert set(stats) == set(cal.calibrators_)
        assert all(isinstance(v, int) and v > 0 for v in stats.values())
        assert sum(stats.values()) == n

    def test_raises_when_unfitted(self):
        from vfairness.post_processing.calibration.group_calibrator import IntersectionalCalibrator

        with pytest.raises(RuntimeError):
            IntersectionalCalibrator().get_group_statistics()


class TestTemperatureScalingComment:
    def test_comment_no_longer_claims_line_search(self):
        from vfairness.post_processing.calibration.methods import TemperatureScaling

        # Honesty pin: fit was rewritten (audit3 temp-scaling) to optimize the NLL
        # with a bounded scalar minimizer after the previous fixed-step gradient
        # descent was found to have an inverted gradient sign. The comment must not
        # claim a line-search method it does not use, and must describe minimizing
        # the objective it actually minimizes.
        src = inspect.getsource(TemperatureScaling.fit)
        assert "with line search" not in src
        assert "minimiz" in src.lower()

    def test_fit_still_works(self):
        from vfairness.post_processing.calibration.methods import TemperatureScaling

        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 300)
        y_prob = np.clip(rng.random(300), 0.01, 0.99)
        ts = TemperatureScaling().fit(y_true, y_prob)
        assert ts.temperature_ is not None and ts.temperature_ > 0
        out = ts.transform(y_prob)
        assert np.all((out >= 0) & (out <= 1))


class TestTradeoffEceComputedOnce:
    def _data(self):
        # Deterministic scores spread over [0.05, 0.95] so the fairness
        # violation varies across thresholds while ECE stays fixed.
        n = 120
        y_prob = np.linspace(0.05, 0.95, n)
        y_true = (y_prob > 0.5).astype(int)
        y_true[::7] = 1 - y_true[::7]
        attr = np.array(["m", "f"] * (n // 2))
        return y_true, y_prob, attr

    def test_ece_evaluated_exactly_once(self, monkeypatch):
        import vfairness.post_processing.calibration.metrics as calmetrics
        from vfairness.post_processing.calibration.tradeoffs import (
            analyze_calibration_fairness_tradeoff,
        )

        calls = {"n": 0}
        orig = calmetrics.expected_calibration_error

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(calmetrics, "expected_calibration_error", counting)
        y_true, y_prob, attr = self._data()
        result = analyze_calibration_fairness_tradeoff(y_true, y_prob, attr)
        assert calls["n"] == 1

        eces = [v["ece"] for v in result.metrics_at_thresholds.values()]
        fvs = [v["fairness_violation"] for v in result.metrics_at_thresholds.values()]
        # Calibration axis constant (threshold independent), fairness varies.
        assert len(set(eces)) == 1
        assert len(set(np.round(fvs, 6))) > 1


class TestMitigationParetoGuard:
    def test_noop_nan_guard_removed(self):
        import vfairness.post_processing.calibration.tradeoffs as tr

        src = inspect.getsource(tr.mitigation_pareto)
        assert "n != n" not in src

    def test_empty_inputs_still_unavailable(self):
        from vfairness.post_processing.calibration.tradeoffs import mitigation_pareto

        out = mitigation_pareto(None, [], [])
        assert out["available"] is False
