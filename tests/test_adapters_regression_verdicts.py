"""Verdict tests for :mod:`vfairness.rendering.adapters_regression`.

Closes the coverage half of register finding #22 for the regression-fairness
adapter, which sat at 14.0 percent of 72 statements.

The page prints one word for the whole report, so that word is tested from every
side: EQUITABLE when every disparity metric was measured and passed, BIASED on a
measured breach, and NOT ASSESSABLE when a metric could not be checked at all.
Until 2026-08-27 the word was decided by ``n_pass == n_total``, which is also
true of 0 == 0, so a report that measured nothing certified equity.
"""

import re

import pytest

from vfairness.rendering.adapters_regression import _pf, regression_fairness_to_svg

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


GROUP_METRICS = {
    "Male": {
        "mae": 0.142,
        "rmse": 0.185,
        "r2": 0.823,
        "mean_residual": 0.032,
        "std_residual": 0.178,
        "size": 1200,
    },
    "Female": {
        "mae": 0.168,
        "rmse": 0.212,
        "r2": 0.791,
        "mean_residual": -0.045,
        "std_residual": 0.201,
        "size": 980,
    },
}

ALL_PASSING = {
    "mae_parity": 0.02,
    "rmse_parity": 0.03,
    "mean_pred_diff": 0.01,
    "r2_parity": 0.04,
}

SOME_FAILING = {
    "mae_parity": 0.12,
    "rmse_parity": 0.08,
    "mean_pred_diff": 0.15,
    "r2_parity": 0.06,
}


def _pass_ratio(svg: str) -> str:
    found = re.findall(r">(\d+/\d+)<", svg)
    assert found, "the metrics-pass ratio was not rendered"
    return found[0]


class TestHeadlineVerdict:
    def test_all_metrics_within_threshold_reads_equitable(self):
        svg = regression_fairness_to_svg(GROUP_METRICS, ALL_PASSING)

        assert ">EQUITABLE<" in svg
        assert ">BIASED<" not in svg
        assert _pass_ratio(svg) == "4/4"

    def test_a_single_breach_flips_the_headline_to_biased(self):
        svg = regression_fairness_to_svg(GROUP_METRICS, {**ALL_PASSING, "mae_parity": 0.11})

        assert ">BIASED<" in svg
        assert ">EQUITABLE<" not in svg
        assert _pass_ratio(svg) == "3/4"

    def test_two_breaches_are_named_individually_and_counted(self):
        svg = regression_fairness_to_svg(GROUP_METRICS, SOME_FAILING)

        assert ">BIASED<" in svg
        assert _pass_ratio(svg) == "2/4"
        assert "2 disparity metric(s) exceed threshold" in svg

    def test_a_maximal_disparity_is_never_shown_as_equitable(self):
        """Negative case: every metric at ten times its threshold."""
        svg = regression_fairness_to_svg(
            GROUP_METRICS,
            {k: 1.0 for k in ("mae_parity", "rmse_parity", "mean_pred_diff", "r2_parity")},
        )

        assert ">BIASED<" in svg
        assert ">EQUITABLE<" not in svg
        assert _pass_ratio(svg) == "0/4"

    def test_an_unmeasurable_metric_is_never_counted_as_a_pass(self):
        """A NaN disparity is not a measured breach either, so since 2026-08-27
        the headline is the third state rather than BIASED, and the badge shows
        N/A rather than the raw 'nan' it used to print."""
        svg = regression_fairness_to_svg(GROUP_METRICS, {"mae_parity": float("nan")})

        assert ">EQUITABLE<" not in svg
        assert ">BIASED<" not in svg
        assert ">NOT ASSESSABLE<" in svg
        assert _pass_ratio(svg) == "0/1"
        assert ">nan<" not in svg
        assert ">N/A<" in svg

    def test_the_threshold_is_applied_to_the_magnitude_not_the_signed_value(self):
        svg = regression_fairness_to_svg(GROUP_METRICS, {"mae_parity": -0.15})

        assert ">BIASED<" in svg
        assert ">-0.150<" in svg

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, "PASS"), (0.10, "PASS"), (0.1001, "FAIL"), (-0.05, "PASS"), (-0.5, "FAIL")],
    )
    def test_the_pass_rule_is_absolute_value_at_or_below_the_threshold(self, value, expected):
        assert _pf(value, 0.10) == expected

    def test_a_report_with_no_measured_metrics_is_not_called_equitable(self):
        """Was an OPEN DEFECT until 2026-08-27: the headline was
        `n_pass == n_total`, and with no disparities supplied that is 0 == 0, so
        a report where NOTHING was measured printed a green EQUITABLE. Same
        shape as register findings #8 and #11."""
        svg = regression_fairness_to_svg(GROUP_METRICS, {})

        assert ">EQUITABLE<" not in svg
        assert ">BIASED<" not in svg  # nothing was measured, so nothing failed either
        assert ">NOT ASSESSABLE<" in svg
        assert "no disparity metric was supplied" in svg
        assert "All regression fairness checks pass" not in svg

    def test_an_unmeasurable_metric_is_not_silently_removed_from_the_denominator(self):
        """Was an OPEN DEFECT until 2026-08-27: a disparity supplied as None was
        dropped from the badge list entirely, so it left the denominator too.
        Measured metrics plus one unmeasurable read as '1/1 metrics pass,
        EQUITABLE' and the reader never learned a metric was missing."""
        svg = regression_fairness_to_svg(GROUP_METRICS, {"mae_parity": None, "rmse_parity": 0.05})

        assert _pass_ratio(svg) != "1/1"
        assert _pass_ratio(svg) == "1/2"
        assert ">EQUITABLE<" not in svg
        assert ">NOT ASSESSABLE<" in svg
        assert "1 disparity metric(s) could not be checked (MAE Parity)" in svg

    def test_a_measured_breach_still_wins_over_an_unmeasurable_sibling(self):
        """Could-not-check withdraws an all-clear; it never suppresses a breach
        that WAS measured."""
        svg = regression_fairness_to_svg(GROUP_METRICS, {"mae_parity": None, "rmse_parity": 0.55})

        assert ">BIASED<" in svg
        assert ">NOT ASSESSABLE<" not in svg
        assert _pass_ratio(svg) == "0/2"
        # The unchecked metric is still disclosed, it is just not the headline.
        assert "1 disparity metric(s) could not be checked" in svg

    def test_a_metric_that_was_never_requested_does_not_enter_the_denominator(self):
        """Over-correction control. Supplying two of the four known metrics is
        an ordinary two-metric report, not a report with two holes in it."""
        svg = regression_fairness_to_svg(GROUP_METRICS, {"mae_parity": 0.02, "rmse_parity": 0.03})

        assert ">EQUITABLE<" in svg
        assert ">NOT ASSESSABLE<" not in svg
        assert _pass_ratio(svg) == "2/2"
        assert "could not be checked" not in svg

    def test_a_healthy_report_is_numerically_and_verbally_unchanged(self):
        """Over-correction control. The guard must be invisible on ordinary
        input: same headline, same ratio, same subtitle, same numbers."""
        svg = regression_fairness_to_svg(GROUP_METRICS, ALL_PASSING)

        assert ">EQUITABLE<" in svg
        assert _pass_ratio(svg) == "4/4"
        assert "4 metrics · 2 groups" in svg
        assert ">0.020<" in svg and ">0.030<" in svg and ">0.010<" in svg and ">0.040<" in svg
        assert "NOT ASSESSABLE" not in svg
        assert ">N/A<" not in svg
        assert "All regression fairness checks pass" in svg


class TestPerGroupTable:
    def test_group_rows_carry_their_own_measurements(self):
        svg = regression_fairness_to_svg(GROUP_METRICS, ALL_PASSING)

        assert ">0.142<" in svg and ">0.185<" in svg and ">0.823<" in svg
        assert ">1200<" in svg and ">980<" in svg

    def test_the_residual_sign_is_translated_into_a_direction_the_reader_can_use(self):
        svg = regression_fairness_to_svg(GROUP_METRICS, ALL_PASSING)

        assert "over-predict" in svg
        assert "under-predict" in svg
        assert ">+0.0320<" in svg
        assert ">-0.0450<" in svg

    def test_residual_bias_above_the_trigger_is_called_out_in_the_recommendations(self):
        biased = {
            "A": {"mae": 0.1, "rmse": 0.1, "r2": 0.9, "mean_residual": 0.09, "size": 10},
            "B": {"mae": 0.1, "rmse": 0.1, "r2": 0.9, "mean_residual": -0.09, "size": 10},
        }
        borderline = {
            "A": {"mae": 0.1, "rmse": 0.1, "r2": 0.9, "mean_residual": 0.05, "size": 10},
        }

        assert "Residual bias present" in regression_fairness_to_svg(biased, ALL_PASSING)
        # The trigger is strictly greater than 0.05, so 0.05 itself must not fire.
        assert "Residual bias present" not in regression_fairness_to_svg(borderline, ALL_PASSING)

    def test_a_model_with_no_residual_bias_gets_the_clean_recommendation(self):
        clean = {
            "A": {"mae": 0.1, "rmse": 0.1, "r2": 0.9, "mean_residual": 0.001, "size": 10},
            "B": {"mae": 0.1, "rmse": 0.1, "r2": 0.9, "mean_residual": -0.001, "size": 10},
        }

        svg = regression_fairness_to_svg(clean, ALL_PASSING)

        assert "All regression fairness checks pass" in svg
        assert "Residual bias present" not in svg

    def test_only_the_first_eight_groups_are_tabulated(self):
        many = {
            f"g{i}": {"mae": 0.1, "rmse": 0.1, "r2": 0.9, "mean_residual": 0.0, "size": i}
            for i in range(10)
        }

        svg = regression_fairness_to_svg(many, ALL_PASSING)

        assert ">g0<" in svg
        assert ">g9<" not in svg


class TestEffectSizes:
    def test_a_large_cohens_d_triggers_the_investigate_recommendation(self):
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": 0.62}]
        )

        assert "Large effect size detected between groups" in svg
        assert "M vs F" in svg

    def test_a_small_cohens_d_does_not_trigger_it(self):
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": 0.12}]
        )

        assert "Large effect size detected between groups" not in svg

    def test_an_effect_size_object_is_read_through_to_dict(self):
        class EffectSize:
            def to_dict(self):
                return {"pair": "A vs B", "cohens_d": 0.9}

        svg = regression_fairness_to_svg(GROUP_METRICS, ALL_PASSING, [EffectSize()])

        assert "A vs B" in svg
        assert "Large effect size detected between groups" in svg


class TestDemoPath:
    def test_asking_for_the_demo_renders_the_labelled_demo(self):
        # example=True is now required: a bare regression_fairness_to_svg() used
        # to reach this fixture as a runtime fallback and rendered its invented
        # BIASED verdict and 2/4 pass rate as a real finding.
        svg = regression_fairness_to_svg(example=True)

        assert ">BIASED<" in svg
        assert "4 metrics · 3 groups" in svg
