"""BGL4 audit of batch A-post_processing-2: the four overturns, each executed.

Every test here FAILED against the code as graded and was marked
``xfail(strict=True)`` while the defects were open. BGL5 closed all four on
2026-09-27, so each test is now an ASSERTION OF THE CORRECTED BEHAVIOUR: same
subject, same fixture, same docstring, with the measured AFTER recorded beside
the measured BEFORE. The fixes and their sabotage evidence live in
``tests/test_bgl5_post_processing_2_and_rendering.py``, which also carries the
over-correction controls.

The docstrings carry the measured values, taken from running the real functions
on 2026-09-27, not from reading them.

The four are all the same shape, and it is the shape this wave fixed everywhere
else in the same files: a quantity measured over a SUBSET of the evidence,
published as though it covered all of it.

  A-0  CalibrationAnalyzer.get_explanation      PROVEN -> DEFECT OPEN
  A-3  TemperatureScaling.fit                   PROVEN -> DEFECT OPEN
  A-6  ece_confidence_intervals                 PROVEN -> DEFECT OPEN
  A-10 plot_calibration_disparity          SEMI-PROVEN -> DEFECT OPEN
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")

from vfairness.post_processing.calibration import visualization as calibration_plots
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.post_processing.calibration.methods import IsotonicCalibrator, TemperatureScaling
from vfairness.post_processing.calibration.metrics import ece_confidence_intervals


def _three_groups_one_excluded():
    """150 in 'a', 150 in 'b', 25 in 'c'. 'c' is below min_group_size=30 and is
    catastrophically miscalibrated (every score 0.97 against an outcome of 0)."""
    rng = np.random.default_rng(11)
    na, nb, nc = 150, 150, 25
    n = na + nb + nc
    groups = np.array(["a"] * na + ["b"] * nb + ["c"] * nc)
    y = np.zeros(n, dtype=int)
    prob = np.zeros(n)
    for lo, hi in ((0, na), (na, na + nb)):
        p = rng.random(hi - lo)
        prob[lo:hi] = p
        y[lo:hi] = (rng.random(hi - lo) < p).astype(int)
    prob[na + nb :] = 0.97
    y[na + nb :] = 0
    return y, prob, groups


# ── A-0. The explanation summary states a verdict it did not cover ────────────


def test_a_calibration_explanation_says_which_groups_its_disparity_covers():
    """Measured: with 'c' excluded the summary reads

        'Calibration analysis for 3 groups. ECE = 0.1027. No significant
         calibration disparity.'

    while ``analyze_disparity()`` on the same object reports ece_disparity
    0.0297 over ['a', 'b'] with excluded_groups ['c']. Lowering the gate to 20
    so 'c' enters makes the SAME function say 'Significant calibration
    disparity across groups' off ece_disparity 0.9378. The pin for this grade
    covers total refusal (one group, two rows) and full measurement (two groups
    of 150); it never tried a partial comparison.

    AFTER (BGL5, measured on the same fixture): a clean spread measured over a
    subset resolves to the third state rather than to False, so the summary reads

        'Calibration analysis for 3 groups. ECE = 0.1027. Calibration disparity
         COULD NOT CHECK: no significance verdict was measured, so this is not a
         finding of no disparity.'

    at severity 'medium', and the coverage itself is named in the report's own
    recommendations, which now LEAD with "NOT ASSESSED: groups ['c'] were
    excluded (fewer than 30 samples)". The count in "for 3 groups" is the number
    of groups in the data and is true; what may not stand beside it is a clean
    verdict over two of them. A determinate breach on the same partial evidence
    keeps its True, which the BGL5 control asserts.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = CalibrationAnalyzer(y, prob, groups, min_group_size=30)
        report = analyzer.full_analysis()
        partial = analyzer.get_explanation(report)
        whole = CalibrationAnalyzer(y, prob, groups, min_group_size=20).get_explanation()

    assert "Significant calibration disparity" in whole.summary, (
        "the premise is gone: including 'c' no longer changes the verdict"
    )
    assert "No significant calibration disparity" not in partial.summary, (
        f"a disparity measured over 2 of 3 groups summarised as {partial.summary!r}"
    )
    assert "COULD NOT CHECK" in partial.summary, partial.summary
    assert report.has_significant_disparity is None
    assert any("NOT ASSESSED" in r and "'c'" in r for r in partial.recommendations), (
        f"the summary withholds the verdict but no line names the group it left "
        f"out: {partial.recommendations}"
    )


def test_the_excluded_group_survives_the_explanations_recommendation_truncation():
    """``_explain_calibration`` keeps ``report.recommendations[:5]``. Measured on
    100 'a' + 100 'b' + 25 'c' where 'b' is badly miscalibrated, the report's
    own recommendations put the exclusion notice at index 5:

        [0] STRATEGY: group_specific_isotonic (Priority: medium)
        [1] Apply group-specific calibration to reduce ECE disparity from 0.822.
        [2] CRITICAL: High calibration disparity (0.822). ...
        [3] Group 'b' has poor calibration (ECE=0.900). ...
        [4] Large worst-case disparity (MCE diff=0.702). ...
        [5] NOT ASSESSED: groups ['c'] were excluded (fewer than 30 samples). ...

    so the ExplanationReport a reader meets carries no trace at all that 'c'
    exists, beside a summary that says 'Calibration analysis for 3 groups'.

    AFTER (BGL5): the coverage notices are written FIRST, the ordering
    ``_identify_critical_issues`` already used for exactly this reason, so the
    same nine recommendations come back with the exclusion notice at index 0 and
    it survives the [:5] in the explainer and the [:5] in ``summary()``. The
    truncation itself is unchanged and is recorded as a separate deferral: a
    disclosure-aware truncation belongs in ``explainer.py``, which this batch does
    not own.
    """
    rng = np.random.default_rng(3)
    na, nb, nc = 100, 100, 25
    n = na + nb + nc
    groups = np.array(["a"] * na + ["b"] * nb + ["c"] * nc)
    y = np.zeros(n, dtype=int)
    prob = np.zeros(n)
    p = rng.random(na)
    prob[:na] = p
    y[:na] = (rng.random(na) < p).astype(int)
    prob[na : na + nb] = 0.9
    prob[na + nb :] = 0.5
    y[na + nb :] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = CalibrationAnalyzer(y, prob, groups, min_group_size=30)
        report = analyzer.full_analysis()
        explanation = analyzer.get_explanation(report)

    assert any("NOT ASSESSED" in r for r in report.recommendations), (
        "the premise is gone: the report no longer names the excluded group"
    )
    assert any("NOT ASSESSED" in r for r in explanation.recommendations), (
        f"the exclusion notice was truncated out of the explanation: {explanation.recommendations}"
    )


# ── A-3. n_samples counts rows the weighted fit never used ────────────────────


def test_temperature_scaling_says_how_many_rows_carried_weight():
    """``CalibrationFitResult.n_samples`` is documented as 'Number of samples used
    for fitting'. Measured with 3 of 60 rows carrying weight: temperature
    0.864515, n_samples 60, parameters {'temperature': ...} only, fit_metrics
    nll 0.386, ZERO record warnings and ZERO UserWarnings.

    ``IsotonicCalibrator.fit`` on the identical input publishes
    ``n_weighted_rows: 5`` plus '55 of 60 rows carry sample_weight 0 and were
    excluded from the isotonic fit; the map below is fitted from 5
    observation(s), not from 60', and ``PlattScaling.fit`` publishes
    ``n_weighted_rows`` too. The identifiability guard this grade credits only
    fires when the WEIGHTED rows carry fewer than two distinct scores, so the
    partially weightless case never reaches it.

    AFTER (BGL5): the record publishes ``n_weighted_rows: 3`` and the note "57 of
    60 rows carry sample_weight 0 and contribute nothing to the weighted NLL; the
    temperature below is fitted from 3 observation(s), not from 60". The nll is
    the weighted mean of the objective that was minimised (0.27708874982683834)
    rather than an unweighted mean over all 60 rows (0.38612468225821545), and
    ``n_samples`` is documented as the count of rows SUPPLIED. A fit at uniform
    weight is unchanged, nll bit-identical, warnings still [].
    """
    rng = np.random.RandomState(3)
    scores = rng.rand(60)
    y = (scores + rng.normal(0, 0.2, 60) > 0.5).astype(int)
    weights = np.zeros(60)
    weights[np.argsort(scores)[np.linspace(0, 59, 3).astype(int)]] = 1.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        temperature = TemperatureScaling().fit(y, scores, weights)
        isotonic = IsotonicCalibrator().fit(y, scores, weights)

    assert np.isfinite(temperature.temperature_), "the premise is gone: no temperature was fitted"
    assert "n_weighted_rows" in isotonic.fit_result.parameters, (
        "the premise is gone: the isotonic sibling no longer discloses it"
    )

    record = temperature.fit_result
    disclosed = (
        record.parameters.get("n_weighted_rows") == 3
        or record.n_samples == 3
        or any("weight" in note for note in record.warnings)
        or any("weight" in str(w.message) for w in caught)
    )
    assert disclosed, (
        f"a temperature fitted from 3 of 60 rows published n_samples="
        f"{record.n_samples} with parameters {record.parameters} and "
        f"warnings {record.warnings}"
    )


# ── A-6. Two resamples is published as a measured interval ────────────────────


def test_two_resamples_is_not_a_measured_confidence_interval():
    """The graded fix refuses n_bootstrap=0 (ValueError) and n_bootstrap=1 (NaN
    bounds, unstable None), recording the n=1 defect as 'an interval that does
    not even contain the point estimate'. That description reproduces verbatim
    at n_bootstrap=2. Measured on the file's own ``_two_groups()`` fixture,
    group A:

        n_bootstrap=50 -> ci [0.12840, 0.22510] width 0.09669 unstable True
        n_bootstrap=2  -> ci [0.10708, 0.13726] width 0.03018 unstable False,
                          not_assessed None, zero warnings,
                          around a point estimate of 0.15737 the interval
                          does not contain

    AFTER (BGL5): the refusal floor is the ORDER-STATISTIC count for the
    requested level, 2/alpha - 1 = 39 draws at ci=0.95, not the hard-coded 2. At
    n_bootstrap=2 the bounds, the width, the se and the stability verdict are NaN
    / None with the reason and the required count in ``not_assessed`` and a
    UserWarning; 38 is refused and 39 is measured; n_bootstrap=50 is untouched,
    ci [0.1284045572347945, 0.22509658050493972], unstable True.
    """
    rng = np.random.default_rng(0)
    n_per_group = 100
    n = 2 * n_per_group
    prob = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < prob).astype(int)
    groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two = ece_confidence_intervals(y, prob, groups, n_bootstrap=2, random_state=0)["A"]
    fifty = ece_confidence_intervals(y, prob, groups, n_bootstrap=50, random_state=0)["A"]

    assert fifty["unstable"] is True, "the premise is gone: 50 resamples no longer say unstable"
    contains_point = two["ci_lower"] <= two["ece"] <= two["ci_upper"]
    refused = two["unstable"] is None or two["not_assessed"] is not None or caught
    assert contains_point or refused, (
        f"a 95% interval from 2 resamples came back {[two['ci_lower'], two['ci_upper']]} "
        f"around an ECE of {two['ece']}, with unstable={two['unstable']!r}, "
        f"not_assessed={two['not_assessed']!r} and {len(caught)} warning(s), while 50 "
        f"resamples give width {fifty['ci_width']:.5f} and unstable=True"
    )


# ── A-10. The disparity chart removes the group that could unbalance it ───────


def test_the_disparity_chart_names_the_group_it_did_not_compare():
    """``plot_group_calibration`` in this same module puts the omitted group ON
    THE CHART, with the reasoning 'the plot answered "is calibration equal
    across groups?" by quietly removing the group that could have made it
    unequal', and ``create_calibration_dashboard`` annotates '2 of 3 groups; x
    excluded' for the same reason (BGL3-PP3, the sibling graded PROVEN in this
    very batch). ``plot_calibration_disparity`` has neither mark.

    Measured on 150 'a' + 150 'b' + 25 'c', the figure titled 'Calibration
    Disparity Analysis' carried exactly these texts:

        TITLE:Expected Calibration Error, TICK:a, TICK:b,
        LEG:Good calibration threshold, TITLE:Maximum Calibration Error,
        TICK:a, TICK:b, TITLE:Brier Score, TICK:a, TICK:b

    while the result it was drawn from holds excluded_groups ['c'] and 'c' has
    an ECE of 0.97.

    AFTER (BGL5): the same figure carries "Not measured (below min_group_size):
    c" in a box on the first panel, a suptitle reading "Calibration Disparity
    Analysis (2 of 3 groups compared; c excluded)", and a UserWarning. A
    one-group result is marked "Disparity not measured: a comparison needs at
    least two groups", and a complete two-group comparison renders exactly the
    texts it rendered before, with no warning at all.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CalibrationAnalyzer(y, prob, groups, min_group_size=30).analyze_disparity()
        figure = calibration_plots.plot_calibration_disparity(result)

    assert result.excluded_groups == ["c"], (
        f"the premise is gone: excluded_groups is {result.excluded_groups}"
    )

    rendered = []
    for axis in figure.axes:
        rendered += [t.get_text() for t in axis.texts]
        rendered.append(axis.get_title())
        rendered += [label.get_text() for label in axis.get_xticklabels()]
        legend = axis.get_legend()
        if legend is not None:
            rendered += [t.get_text() for t in legend.get_texts()]

    labelled = [t.get_text() for a in figure.axes for t in a.get_xticklabels()]
    disclosed = "c" in labelled or any(
        "exclud" in text.lower() or "not measured" in text.lower() for text in rendered if text
    )
    assert disclosed, (
        f"a chart titled 'Calibration Disparity Analysis' over three groups drew two "
        f"bars and never named the third: {[t for t in rendered if t]}"
    )
