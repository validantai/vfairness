"""BGL5 pins for the seven overturned grades in A-post_processing-2 and A-rendering-1.

Every test here EXECUTES the fixed unit on the input the auditor used, asserts
the corrected behaviour with the measured numbers, and is followed by an
over-correction CONTROL that asserts healthy input still gets its real
measurement, with the value. Each pin was sabotaged (the fix reverted in the
source) and reported red before being trusted; the sabotage and what it printed
are recorded in /tmp/claude-501/bgl/fix/.

A-post_processing-2, all four one shape: a quantity measured over a SUBSET,
published as the whole.

  A-0  CalibrationAnalyzer.get_explanation          a clean disparity verdict
                                                    over 2 of 3 groups, and an
                                                    exclusion notice a [:5]
                                                    truncation could drop
  A-3  TemperatureScaling.fit                       n_samples 60 for a fit from 3
  A-6  ece_confidence_intervals                     a 95% interval from 2 draws
  A-10 plot_calibration_disparity                   a chart that removed the
                                                    group that unbalances it

A-rendering-1, all three a chart disagreeing with its own data source.

  R-8  alert_timeline_to_svg                        a half-checked window tallied
                                                    as CLEAN at severity info
  R-10 monitoring_dashboard_to_svg                  an action asserting a
                                                    measurement nobody made, and
                                                    the dropped stratum
  R-11 temporal_analysis_to_svg                     a "+nan" slope on the card

READ THE DRAWN VALUES, never the axis furniture: an empty matplotlib axes still
carries a "0.0" tick and the SVG font name "Verdana" contains a substring that
looks like an n/a marker, so a check that reads tick labels or font names gives a
false reading in BOTH directions. The renderer pins read <text> node CONTENTS and
the matplotlib pins read artists and text objects.
"""

from __future__ import annotations

import json
import re
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from vfairness.post_processing.calibration import visualization as calibration_plots
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.post_processing.calibration.methods import TemperatureScaling
from vfairness.post_processing.calibration.metrics import ece_confidence_intervals

pd = pytest.importorskip("pandas")
pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness.operations.monitoring.tracker import (  # noqa: E402
    FairnessMonitor,
    FairnessMonitorConfig,
    TemporalFairnessAnalyzer,
)
from vfairness.rendering import explain as EX  # noqa: E402
from vfairness.rendering.adapters_monitoring import (  # noqa: E402
    alert_timeline_to_svg,
    monitoring_dashboard_to_svg,
    temporal_analysis_to_svg,
)

# ── fixtures, the auditor's own ───────────────────────────────────────────────


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


def _two_groups_and_a_miscalibrated_third():
    """100 'a' + 100 'b' + 25 'c', with 'b' badly miscalibrated, so the measured
    spread is a BREACH (0.822) while 'c' is still excluded."""
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
    return y, prob, groups


def _sixty_rows():
    rng = np.random.RandomState(3)
    scores = rng.rand(60)
    y = (scores + rng.normal(0, 0.2, 60) > 0.5).astype(int)
    return y, scores


def _two_groups_for_intervals():
    rng = np.random.default_rng(0)
    n = 200
    prob = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < prob).astype(int)
    groups = np.array(["A"] * 100 + ["B"] * 100)
    return y, prob, groups


def _figure_texts(figure):
    """Every string DRAWN on a matplotlib figure: text artists, titles, tick
    labels, legend entries and the suptitle.

    Tick labels are included only because the group names ARE tick labels here;
    nothing in these pins concludes anything from a tick's numeric value, which
    is the reading that lies in both directions (an empty axes still carries a
    "0.0" tick).
    """
    out = []
    for axis in figure.axes:
        out += [t.get_text() for t in axis.texts]
        out.append(axis.get_title())
        out += [label.get_text() for label in axis.get_xticklabels()]
        legend = axis.get_legend()
        if legend is not None:
            out += [t.get_text() for t in legend.get_texts()]
    out.append(figure._suptitle.get_text() if figure._suptitle else "")
    return [t for t in out if t]


# ── A-0. A clean disparity verdict over a subset is withheld ──────────────────


def test_a_disparity_measured_over_two_of_three_groups_states_no_clean_verdict():
    """A-0. The summary said "Calibration analysis for 3 groups. ECE = 0.1027. No
    significant calibration disparity." over a comparison of TWO groups, with the
    third excluded and holding an ECE of 0.97.

    Measured on 150 'a' + 150 'b' + 25 'c' at min_group_size=30: excluded_groups
    ['c'], ece_disparity 0.029657395451042123 over ['a', 'b'],
    has_significant_disparity False. The SAME data at min_group_size=20, which
    lets 'c' in, gives 0.9377549481374269 and "Significant calibration disparity
    across groups": the group left out is the one that flips the verdict.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = CalibrationAnalyzer(y, prob, groups, min_group_size=30)
        result = analyzer.analyze_disparity()
        explanation = analyzer.get_explanation()
        whole = CalibrationAnalyzer(y, prob, groups, min_group_size=20).analyze_disparity()

    # The premise, asserted rather than assumed.
    assert result.excluded_groups == ["c"], result.excluded_groups
    assert result.ece_disparity == pytest.approx(0.029657395451042123)
    assert whole.ece_disparity == pytest.approx(0.9377549481374269)
    assert whole.has_significant_disparity is True, "including 'c' must still flip it"

    # Three states: this one is could-not-check, not a finding of no disparity.
    assert result.has_significant_disparity is None
    assert result.to_dict()["has_significant_disparity"] is None
    assert "No significant calibration disparity" not in explanation.summary
    assert "COULD NOT CHECK" in explanation.summary, explanation.summary
    assert explanation.severity != "info", explanation.severity


def test_control_a_complete_two_group_comparison_still_states_its_verdict():
    """CONTROL for A-0, on the SAME NUMBER. The first 300 rows are 'a' and 'b'
    only, so nothing is excluded and ece_disparity is the identical
    0.029657395451042123 the partial case measured. It must still come back
    False, with the summary reading "No significant calibration disparity.": the
    fix keys on COVERAGE, never on the value, and a refusal here would destroy
    the verdict for every complete comparison inside the threshold.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = CalibrationAnalyzer(y[:300], prob[:300], groups[:300], min_group_size=30)
        result = analyzer.analyze_disparity()
        explanation = analyzer.get_explanation()

    assert result.excluded_groups == []
    assert result.ece_disparity == pytest.approx(0.029657395451042123)
    assert result.has_significant_disparity is False
    assert "No significant calibration disparity." in explanation.summary
    assert result.recommendations == ["LOW: Calibration disparity is acceptable (0.030)."]


def test_control_a_measured_breach_on_partial_evidence_keeps_its_finding():
    """CONTROL for A-0, the other direction. A spread already over the threshold
    cannot be argued away by a group nobody measured, and withholding it would be
    the reverse fabrication. Measured on 100 'a' + 100 'b' + 25 'c' with 'b'
    miscalibrated: ece_disparity 0.822, excluded_groups ['c'],
    has_significant_disparity True, both before the fix and after.
    """
    y, prob, groups = _two_groups_and_a_miscalibrated_third()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CalibrationAnalyzer(y, prob, groups, min_group_size=30).analyze_disparity()

    assert result.excluded_groups == ["c"]
    assert result.ece_disparity == pytest.approx(0.822, abs=0.001)
    assert result.has_significant_disparity is True


def test_the_excluded_group_survives_the_recommendation_truncation():
    """A-0, second leg. ``explainer._explain_calibration`` keeps
    ``report.recommendations[:5]`` and ``CalibrationReport.summary()`` prints
    ``[:5]``. Measured on 100 'a' + 100 'b' + 25 'c' with 'b' miscalibrated, the
    exclusion notice was at index 5 of 9, so the ExplanationReport a reader meets
    carried NO trace that 'c' exists beside a summary saying "Calibration
    analysis for 3 groups". It is index 0 now and survives every truncation.
    """
    y, prob, groups = _two_groups_and_a_miscalibrated_third()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = CalibrationAnalyzer(y, prob, groups, min_group_size=30)
        report = analyzer.full_analysis()
        explanation = analyzer.get_explanation(report)

    assert report.recommendations[0].startswith("NOT ASSESSED: groups ['c'] were excluded")
    # The surface a reader actually meets, not the producer's list.
    assert any("NOT ASSESSED" in r for r in explanation.recommendations), (
        explanation.recommendations
    )
    assert "NOT ASSESSED" in report.summary()
    # And the disparity result's own list, which other consumers truncate too.
    assert report.disparity_analysis.recommendations[0].startswith("NOT ASSESSED")


def test_control_a_fully_measured_analysis_leads_with_its_strategy():
    """CONTROL for the reordering: it must MOVE coverage notices, never invent
    one. On the two complete groups the report has no NOT ASSESSED line at all
    and its first recommendation is still the strategy, "STRATEGY: monitor_only
    (Priority: low)".
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = CalibrationAnalyzer(
            y[:300], prob[:300], groups[:300], min_group_size=30
        ).full_analysis()

    assert not any("NOT ASSESSED" in r for r in report.recommendations)
    assert report.recommendations[0] == "STRATEGY: monitor_only (Priority: low)"


# ── A-3. n_samples counted rows the weighted fit never used ───────────────────


def test_temperature_scaling_publishes_how_many_rows_carried_weight():
    """A-3. ``CalibrationFitResult.n_samples`` was documented as "Number of
    samples used for fitting". Measured with 3 of 60 rows carrying weight:
    n_samples 60, parameters {'temperature': 0.8645148917436684} ONLY,
    fit_metrics nll 0.38612468225821545 (an UNWEIGHTED mean over all 60,
    including the 57 the fit excluded), record warnings [] and zero UserWarnings.

    Both siblings in the same file disclose it on the identical input, so this
    mirrors them: n_weighted_rows 3, the note naming the 57 dropped rows, and a
    weighted nll of 0.27708874982683834.
    """
    y, scores = _sixty_rows()
    weights = np.zeros(60)
    weights[np.argsort(scores)[np.linspace(0, 59, 3).astype(int)]] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = TemperatureScaling().fit(y, scores, weights).fit_result

    assert record.n_samples == 60
    assert record.parameters["n_weighted_rows"] == 3
    assert record.parameters["temperature"] == pytest.approx(0.8645148917436684)
    joined = " ".join(record.warnings)
    assert "57 of 60 rows carry sample_weight 0" in joined, record.warnings
    assert "fitted from 3 observation(s), not from 60" in joined, record.warnings
    # The number published under the name of the minimised objective is now the
    # objective's own weighted mean, not an unweighted mean over all 60 rows.
    assert record.fit_metrics["nll"] == pytest.approx(0.27708874982683834)
    assert record.fit_metrics["nll"] != pytest.approx(0.38612468225821545)


def test_control_a_fit_at_uniform_weight_is_unchanged_and_silent():
    """CONTROL for A-3, the exact assertion the existing pin makes: the same 60
    rows at weight 1.0 must still fit, with NO note (an unconditional note would
    make every record read as partial) and with the identical statistics.
    Measured: temperature 0.3385169755179897, n_weighted_rows 60, nll
    0.309467169222508, which is bit-identical to the unweighted mean it replaced.
    """
    y, scores = _sixty_rows()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = TemperatureScaling().fit(y, scores, np.ones(60)).fit_result

    assert caught == [], [str(w.message) for w in caught]
    assert record.warnings == []
    assert record.n_samples == 60
    assert record.parameters["n_weighted_rows"] == 60
    assert record.parameters["temperature"] == pytest.approx(0.3385169755179897)
    assert record.fit_metrics["nll"] == 0.309467169222508


# ── A-6. Two resamples is not a measured confidence interval ──────────────────


def test_two_resamples_is_not_published_as_a_confidence_interval():
    """A-6. The graded fix refuses n_bootstrap 0 and 1 and never tried 2.
    Measured on the two-group fixture, group A, random_state=0:

        n_bootstrap=2  -> ci [0.10708156858467521, 0.13726336220604668],
                          width 0.03018, se 0.02247, unstable False,
                          not_assessed None, zero warnings, around an ECE of
                          0.1573694620635577 the interval does NOT contain

    which is verbatim the defect recorded for n_bootstrap=1. A percentile bound
    is an order statistic, so a 95% interval needs 2/alpha - 1 = 39 draws.
    """
    y, prob, groups = _two_groups_for_intervals()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two = ece_confidence_intervals(y, prob, groups, n_bootstrap=2, random_state=0)["A"]

    assert two["ece"] == pytest.approx(0.1573694620635577), "the point estimate still stands"
    assert np.isnan(two["ci_lower"]) and np.isnan(two["ci_upper"])
    assert np.isnan(two["ci_width"]) and np.isnan(two["se"])
    assert two["unstable"] is None
    assert two["n_bootstrap_effective"] == 2
    assert "fewer than the 39" in two["not_assessed"], two["not_assessed"]
    assert any("no percentile interval could be formed" in str(w.message) for w in caught)


def test_the_resample_floor_is_the_order_statistic_count_not_a_round_number():
    """A-6, the boundary, both sides, so the floor is a rule and not a fixture.
    38 draws are refused and 39 are measured, because 0.025 * (39 + 1) == 1.
    """
    y, prob, groups = _two_groups_for_intervals()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thirty_eight = ece_confidence_intervals(y, prob, groups, n_bootstrap=38, random_state=0)[
            "A"
        ]
        thirty_nine = ece_confidence_intervals(y, prob, groups, n_bootstrap=39, random_state=0)["A"]

    assert thirty_eight["unstable"] is None and np.isnan(thirty_eight["ci_lower"])
    assert thirty_nine["unstable"] is True
    assert thirty_nine["ci_lower"] == pytest.approx(0.12652, abs=1e-4)
    assert thirty_nine["not_assessed"] is None


def test_control_fifty_resamples_still_measure_the_real_interval():
    """CONTROL for A-6. A refusal that also fired here would destroy the
    function. Measured, unchanged by the fix: 50 resamples on group A give
    ci_lower 0.1284045572347945, ci_upper 0.22509658050493972, width
    0.09669202327014523, se 0.028670584958780408, unstable True, not_assessed
    None and not one warning.
    """
    y, prob, groups = _two_groups_for_intervals()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fifty = ece_confidence_intervals(y, prob, groups, n_bootstrap=50, random_state=0)["A"]

    assert fifty["ci_lower"] == pytest.approx(0.1284045572347945)
    assert fifty["ci_upper"] == pytest.approx(0.22509658050493972)
    assert fifty["ci_width"] == pytest.approx(0.09669202327014523)
    assert fifty["se"] == pytest.approx(0.028670584958780408)
    assert fifty["unstable"] is True
    assert fifty["not_assessed"] is None
    assert fifty["ci_lower"] <= fifty["ece"] <= fifty["ci_upper"], (
        "a measured interval contains its own point estimate"
    )


# ── A-10. The disparity chart names the group it did not compare ──────────────


def test_the_disparity_chart_names_the_group_it_did_not_compare():
    """A-10. Measured before, the figure titled "Calibration Disparity Analysis"
    over three groups carried exactly these texts and named 'c' nowhere:

        TITLE:Expected Calibration Error, TICK:a, TICK:b,
        LEG:Good calibration threshold, TITLE:Maximum Calibration Error,
        TICK:a, TICK:b, TITLE:Brier Score, TICK:a, TICK:b

    while the result it was drawn from held excluded_groups ['c'] and 'c' has an
    ECE of 0.97. Two siblings in the same module were fixed for precisely this.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = CalibrationAnalyzer(y, prob, groups, min_group_size=30).analyze_disparity()
        figure = calibration_plots.plot_calibration_disparity(result)

    assert result.excluded_groups == ["c"], result.excluded_groups
    drawn = _figure_texts(figure)
    assert "Not measured (below min_group_size): c" in drawn, drawn
    assert any("2 of 3 groups compared; c excluded" in t for t in drawn), drawn
    assert any("plot_calibration_disparity" in str(w.message) for w in caught)


def test_a_one_group_disparity_chart_says_the_disparity_was_not_measured():
    """A-10, the second input class: one group only, so ece_disparity is NaN and
    has_significant_disparity is None. Measured before, the three panels drew one
    bar each with no "not measured" mark of any kind, under the same title.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CalibrationAnalyzer(y[:150], prob[:150], groups[:150], min_group_size=30)
        disparity = result.analyze_disparity()
        figure = calibration_plots.plot_calibration_disparity(disparity)

    assert np.isnan(disparity.ece_disparity)
    assert disparity.has_significant_disparity is None
    drawn = _figure_texts(figure)
    assert any("Disparity not measured" in t for t in drawn), drawn


def test_control_a_complete_disparity_chart_carries_no_caveat():
    """CONTROL for A-10. Over-annotation would make every chart read as partial.
    Two complete groups render exactly the texts they rendered before, with no
    exclusion box, no "not measured" mark, no suptitle suffix and no warning.
    """
    y, prob, groups = _three_groups_one_excluded()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        disparity = CalibrationAnalyzer(
            y[:300], prob[:300], groups[:300], min_group_size=30
        ).analyze_disparity()
        figure = calibration_plots.plot_calibration_disparity(disparity)

    drawn = _figure_texts(figure)
    assert drawn == [
        "Expected Calibration Error",
        "a",
        "b",
        "Good calibration threshold",
        "Maximum Calibration Error",
        "a",
        "b",
        "Brier Score",
        "a",
        "b",
        "Calibration Disparity Analysis",
    ], drawn


# ── the rendering half: read the rendered artifact, never the markup ──────────

_TEXT_NODE = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_META = re.compile(r"<metadata[^>]*><!\[CDATA\[(.*?)\]\]></metadata>", re.S)

# The chip the temporal card draws its trend verdict in, read off the ONE text
# node the template draws it in. A document-wide search for the colour matches
# other slate and green tones, and a search for the word matches the degradation
# table's own chip further down the page.
_CHIP = re.compile(
    r'<text\b[^>]*font-size="7\.5"[^>]*fill="(#[0-9a-fA-F]{6})"[^>]*>'
    r"(STABLE|INCREASING|DECREASING|NOT TRACKED|NOT FITTED)</text>"
)

# The shipped artifact is already skinned, so the card's #059669 has become the
# Blanco pass green by the time anyone reads the SVG.
_BLANCO_PASS = "#41ba1b"


def canvas(svg: str) -> str:
    """Everything a sighted reader reads, as one string: the CONTENTS of the
    <text> nodes. Never the raw markup, where a token inside an attribute or a
    font name is not something a reader sees."""
    out = []
    for raw in _TEXT_NODE.findall(svg):
        stripped = _TAG.sub("", raw).strip()
        if stripped:
            out.append(stripped)
    return " ".join(out)


def meta(svg: str) -> dict:
    match = _META.search(svg)
    assert match, "every rendered chart carries a vfairness-explanation block"
    return json.loads(match.group(1))


def _partially_checked_window():
    """One REAL monitoring window: half its metrics checked, one group dropped.

    ``group_region`` has two large groups, so both its metrics are computed and
    compared to a guardrail. ``group_gender`` carries an "Other" stratum of 7
    rows, below the default ``min_samples=30``: the tracker drops it, records it
    in ``excluded_groups``, and leaves both gender metrics NaN with no entry in
    ``alerts``. So this window applied a guardrail to two of its four metrics.
    """
    rng = np.random.default_rng(1)
    n_a, n_b, n_drop = 300, 300, 7
    total = n_a + n_b + n_drop
    frame = pd.DataFrame(
        {
            "prediction": np.concatenate(
                [
                    (rng.random(n_a) < 0.70).astype(int),
                    (rng.random(n_b) < 0.69).astype(int),
                    np.zeros(n_drop, dtype=int),
                ]
            ),
            "label": (rng.random(total) < 0.5).astype(int),
            "group_gender": ["Male"] * n_a + ["Female"] * n_b + ["Other"] * n_drop,
            "group_region": ["North"] * (total // 2) + ["South"] * (total - total // 2),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = FairnessMonitor(config=FairnessMonitorConfig()).update_and_check(frame)
    assert window.excluded_groups == {"group_gender": {"Other": 7}}, window.excluded_groups
    assert len(window.metrics) == 4 and len(window.alerts) == 2, (window.metrics, window.alerts)
    return window


def _fully_checked_window():
    """The CONTROL window: one protected column, two large groups, so every
    metric it computed was compared to a guardrail and nothing was excluded."""
    rng = np.random.default_rng(2)
    n_a = n_b = 300
    total = n_a + n_b
    frame = pd.DataFrame(
        {
            "prediction": np.concatenate(
                [
                    (rng.random(n_a) < 0.70).astype(int),
                    (rng.random(n_b) < 0.69).astype(int),
                ]
            ),
            "label": (rng.random(total) < 0.5).astype(int),
            "group_region": ["North"] * n_a + ["South"] * n_b,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = FairnessMonitor(config=FairnessMonitorConfig()).update_and_check(frame)
    assert window.excluded_groups == {}, window.excluded_groups
    assert len(window.metrics) == 2 and len(window.alerts) == 2, (window.metrics, window.alerts)
    return window


def _partially_checked_window_with_no_excluded_group():
    """A window partly checked for the OTHER reason: a CUSTOM metric.

    The tracker computes custom metrics and deliberately writes them into
    ``metrics`` and never into ``alerts`` ("they carry no threshold and no
    direction, so no alert determination is possible for them", R-2). Nothing is
    excluded here, so the severity floor this window earns can come only from the
    partial check itself, never from ``explain._excluded_clause``, which floors
    the severity too and which the fixture above trips at the same time.
    """
    rng = np.random.default_rng(4)
    n_a = n_b = 300
    total = n_a + n_b
    frame = pd.DataFrame(
        {
            "prediction": np.concatenate(
                [
                    (rng.random(n_a) < 0.70).astype(int),
                    (rng.random(n_b) < 0.69).astype(int),
                ]
            ),
            "label": (rng.random(total) < 0.5).astype(int),
            "group_region": ["North"] * n_a + ["South"] * n_b,
        }
    )
    monitor = FairnessMonitor(
        config=FairnessMonitorConfig(),
        custom_metrics={"mean_score": lambda df: float(df["prediction"].mean())},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = monitor.update_and_check(frame)
    assert window.excluded_groups == {}, window.excluded_groups
    assert len(window.metrics) == 3 and len(window.alerts) == 2, (window.metrics, window.alerts)
    return window


def _analyzer(values):
    """A REAL TemporalFairnessAnalyzer holding one metric over len(values) days."""
    analyzer = TemporalFairnessAnalyzer()
    base = pd.Timestamp("2026-01-01")
    for offset, value in enumerate(values):
        analyzer.update_daily_metrics(
            base + pd.Timedelta(days=offset), {"demographic_parity_difference": value}
        )
    return analyzer


# ── R-8. A half-checked window is not clean ──────────────────────────────────


def test_a_half_checked_window_is_not_tallied_as_clean():
    """R-8. ``checked = len(alerts) > 0`` was a WINDOW-level two-state test over a
    PER-METRIC record. Measured on the real window above, alert_timeline_to_svg
    rendered "1 monitoring events / 0 with alerts / 1 clean", TOTAL EVENTS 1 /
    WITH ALERTS 0 / CLEAN 1, the row's STATUS pill "OK" in emerald, and published
    "0 of 1 recent monitoring windows raised alerts (1 clean)." at severity info:
    the same artifact a window watched end to end produces.
    """
    window = _partially_checked_window()
    svg = alert_timeline_to_svg([window])
    drawn = canvas(svg)
    published = meta(svg)

    assert "CLEAN 1" not in drawn, drawn[:300]
    assert "CLEAN 0" in drawn, drawn[:300]
    assert "2 of 4 checked" in drawn, drawn[:300]
    assert "PARTIAL" in drawn, drawn[:300]
    assert published["severity"] == "medium", published["severity"]
    assert "not covered by this finding" in published["finding"], published["finding"]
    assert "2 metric(s) across 1 partly checked window(s)" in published["finding"]


def test_the_partial_severity_floor_comes_from_the_partial_check_itself():
    """R-8, the half the fixture above cannot isolate.

    That window also carries an excluded stratum, and ``_excluded_clause`` floors
    the severity too, so "severity == medium" there could not have DISAGREED with
    a fix that only added the sentence. This window is partly checked with NOTHING
    excluded: a real custom metric the tracker computes and never guardrails.
    Measured: metrics 3, alerts 2, excluded_groups {}, canvas "2 of 3 checked",
    finding "... 1 metric(s) across 1 partly checked window(s) were never
    compared to a guardrail ...", severity medium where the old code said info.
    """
    window = _partially_checked_window_with_no_excluded_group()
    svg = alert_timeline_to_svg([window])
    published = meta(svg)

    assert "Excluded:" not in published["finding"], published["finding"]
    assert "1 metric(s) across 1 partly checked window(s)" in published["finding"]
    assert published["severity"] == "medium", published["severity"]
    assert "2 of 3 checked" in canvas(svg)
    assert "CLEAN 0" in canvas(svg)


def test_an_alerting_window_keeps_its_finding_and_still_states_its_coverage():
    """R-8, the direction a coverage rule can get wrong. A window that DID raise
    an alert and left two metrics unchecked must keep the alert (an ungraded row
    elsewhere does not make a real finding less true) AND still say what it did
    not cover, which is why partly-checked windows are counted whatever their
    verdict rather than only when they would otherwise have been called clean.

    Measured on the same two-column frame at alert_threshold=0.99, so both region
    metrics breach: WITH ALERTS 1, CLEAN 0, the row's STATUS "ALERT", and the
    finding "1 of 1 recent monitoring windows raised alerts (0 clean). 2 metric(s)
    across 1 partly checked window(s) were never compared to a guardrail and are
    not covered by this finding. Excluded: Other (n=7, group_gender)." at severity
    high, which the partial floor must not lower.
    """
    rng = np.random.default_rng(1)
    n_a, n_b, n_drop = 300, 300, 7
    total = n_a + n_b + n_drop
    frame = pd.DataFrame(
        {
            "prediction": np.concatenate(
                [
                    (rng.random(n_a) < 0.70).astype(int),
                    (rng.random(n_b) < 0.69).astype(int),
                    np.zeros(n_drop, dtype=int),
                ]
            ),
            "label": (rng.random(total) < 0.5).astype(int),
            "group_gender": ["Male"] * n_a + ["Female"] * n_b + ["Other"] * n_drop,
            "group_region": ["North"] * (total // 2) + ["South"] * (total - total // 2),
        }
    )
    monitor = FairnessMonitor(config=FairnessMonitorConfig(alert_threshold=0.99))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = monitor.update_and_check(frame)

    assert window.any_alert is True and len(window.alerts) == 2
    assert len(window.metrics) == 4

    svg = alert_timeline_to_svg([window])
    published = meta(svg)
    assert "WITH ALERTS 1" in canvas(svg), canvas(svg)[:250]
    assert "CLEAN 0" in canvas(svg)
    assert published["severity"] == "high", published["severity"]
    assert "1 of 1 recent monitoring windows raised alerts (0 clean)." in published["finding"]
    assert "2 metric(s) across 1 partly checked window(s)" in published["finding"]


def test_control_a_fully_checked_clean_window_still_reads_clean():
    """CONTROL for R-8. A window every one of whose metrics WAS compared to a
    guardrail, and none of which fired, must still be counted clean and graded
    info: withholding it would be the reverse fabrication. Measured on two large
    region groups: 2 metrics, 2 alerts, canvas "CLEAN 1" with the OK pill, and
    the finding "0 of 1 recent monitoring windows raised alerts (1 clean)." at
    severity info, with no ungraded tail at all.
    """
    window = _fully_checked_window()
    svg = alert_timeline_to_svg([window])
    drawn = canvas(svg)
    published = meta(svg)

    assert "CLEAN 1" in drawn, drawn[:300]
    assert "PARTIAL" not in drawn, drawn[:300]
    assert published["severity"] == "info", published["severity"]
    assert published["finding"] == ("0 of 1 recent monitoring windows raised alerts (1 clean)."), (
        published["finding"]
    )


# ── R-10. The dropped stratum, and an action that asserts a measurement ───────


def test_both_monitoring_renderers_name_the_stratum_the_tracker_dropped():
    """R-10 (b). ``WindowMetrics.excluded_groups`` is the tracker's record of the
    strata ``min_samples`` kept OUT of every rate on the page, and no renderer in
    adapters_monitoring read it (``grep -rn excluded_groups
    src/vfairness/rendering/`` answered adapters_fairness.py and adapters.py
    only). Measured on the real window, excluded_groups={'group_gender':
    {'Other': 7}} while the canvas drew "Group Positive Rates Female 69.7% Male
    71.0%" as though those were the groups, and the string 'Other' appeared
    nowhere in the canvas text or in the metadata JSON of either chart.
    """
    window = _partially_checked_window()

    dashboard = monitoring_dashboard_to_svg(window)
    assert "Other (n=7, group_gender)" in canvas(dashboard), canvas(dashboard)[:400]
    assert "Excluded: Other (n=7, group_gender)." in meta(dashboard)["finding"]

    # The timeline has no rates table, so it discloses on the accessible layer.
    timeline = alert_timeline_to_svg([window])
    everything = canvas(timeline) + " " + json.dumps(meta(timeline))
    assert "Other" in everything, everything[:400]


def test_control_a_window_that_dropped_nothing_carries_no_exclusion_clause():
    """CONTROL for R-10 (b). The clause must come from the record, not from the
    renderer: the complete window names no excluded stratum anywhere, and its
    finding is the plain measured sentence.
    """
    window = _fully_checked_window()
    for svg in (monitoring_dashboard_to_svg(window), alert_timeline_to_svg([window])):
        published = meta(svg)
        assert "Excluded:" not in published["finding"], published["finding"]
        assert "not measured" not in canvas(svg), canvas(svg)[:300]


def test_a_withheld_alert_verdict_gets_no_action_that_asserts_a_measurement():
    """R-10 (a). ``explain._is_could_not_check`` is a STARTSWITH test and four
    finders deliberately write the subject first, so the action swap in
    build_explanation could not fire for them. Measured end to end through
    FairnessMonitor(metrics_to_track=['equalized_odds']) on a frame with a
    prediction column and NO label column, so zero fairness metrics are computed
    while the group rates are real:

      finding        Alert status: COULD NOT CHECK, this window recorded no alert
                     verdict, ... across 0 fairness metric(s) on 600 samples.
      recommendation Triage the alerting metrics; if several alert at once, treat
                     it as an incident.

    Zero metrics were monitored, so there are no alerting metrics to triage.
    """
    rng = np.random.default_rng(1)
    n = 600
    frame = pd.DataFrame(
        {
            "prediction": (rng.random(n) < 0.7).astype(int),
            "group_gender": ["Female"] * (n // 2) + ["Male"] * (n - n // 2),
        }
    )
    monitor = FairnessMonitor(config=FairnessMonitorConfig(metrics_to_track=["equalized_odds"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        window = monitor.update_and_check(frame)

    # The premise, asserted rather than assumed.
    assert window.metrics == {} and window.alerts == {}
    assert window.any_alert is None
    assert window.group_rates, "the group rates are real, which is what keeps the chart-level "
    "not-assessable guard from firing"

    published = meta(monitoring_dashboard_to_svg(window))
    assert "COULD NOT CHECK" in published["finding"]
    assert "Triage the alerting metrics" not in published["recommendation"]
    assert "certifies nothing" in published["recommendation"], published["recommendation"]


def test_control_a_measured_dashboard_keeps_its_curated_action():
    """CONTROL for R-10 (a), half one: an action swap that fired on a measured
    chart would replace every recommendation in the library with "re-run it".
    The complete window keeps "Triage the alerting metrics; if several alert at
    once, treat it as an incident." at severity info.
    """
    published = meta(monitoring_dashboard_to_svg(_fully_checked_window()))
    assert published["finding"] == "0 of 2 fairness metrics alerting on 600 samples."
    assert published["recommendation"].startswith("Triage the alerting metrics")
    assert published["severity"] == "info"


@pytest.mark.parametrize(
    "finding,withholds",
    [
        # The four subject-first findings: the VERDICT itself is withheld.
        ("Alert status: COULD NOT CHECK, this window recorded no alert verdict.", True),
        ("Best reweighting method: COULD NOT CHECK, 'x' is named with no graded score.", True),
        ("Intersectional disparity: COULD NOT CHECK, no compound gap was reported", True),
        ("COULD NOT CHECK: nothing on this chart was assessed.", True),
        # Mid-sentence markers on charts that DID produce a verdict. Each one is
        # deliberate and documented at its finder, and each must keep the curated
        # action: telling a reader to re-run an audit that returned findings, or a
        # calibration report whose ECE was measured, is the over-correction.
        ("3 critical issue(s) found. COULD NOT CHECK: no overall bias-risk score.", False),
        ("ECE 0.103 (miscalibrated); COULD NOT CHECK: no ECE disparity was measured.", False),
        (
            "Mediator 'x' reports no proportion mediated (COULD NOT CHECK), so how much of "
            "the gap runs through it is not established; 4/4 Baron-Kenny steps satisfied.",
            False,
        ),
        ("0 of 2 fairness metrics alerting on 607 samples.", False),
        ("", False),
    ],
)
def test_the_withheld_verdict_test_reads_only_a_bare_subject_label(finding, withholds):
    """The predicate itself, in both directions, on the real strings. A test that
    only checked the True cases would pass for a predicate that matches the word
    anywhere, which is the over-correction the mid-sentence rows forbid.
    """
    assert EX._withholds_the_verdict(finding) is withholds, finding


# ── R-11. The temporal card's unfitted trend ─────────────────────────────────


def test_the_temporal_card_does_not_print_a_nan_slope_for_an_unfitted_trend():
    """R-11. ``detect_trend`` answers ("not_assessed", nan) for a metric the
    analyzer HOLDS but cannot fit a line through, and warns while doing it. The
    trends TABLE renders that honestly; the metric summary CARD, the largest
    element on the canvas, read the statistic straight off get_metric_summary, so
    it drew "Slope/day +nan" (NaN is not None, so the None test never fired) and
    a green range bar. Measured on three daily readings: summary
    {'mean': 0.06, 'std': 0.01, 'min': 0.05, 'max': 0.07,
     'trend_direction': 'not_assessed', 'trend_slope': nan, 'n_days': 3}.

    The statistics ARE measured for such a metric, so they stay: only the trend
    verdict and the slope are withheld.
    """
    unfitted = _analyzer([0.05, 0.06, 0.07])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = unfitted.get_metric_summary("demographic_parity_difference")
        svg = temporal_analysis_to_svg(unfitted)

    # The premise, asserted rather than assumed.
    assert summary["trend_direction"] == "not_assessed", summary
    assert summary["trend_slope"] != summary["trend_slope"], summary  # NaN

    drawn = canvas(svg)
    assert "nan" not in drawn, drawn[:300]
    assert "Slope/day not measured" in drawn, drawn[:300]
    # The statistics the analyzer DID measure are still on the card.
    assert "0.060" in drawn and "0.070" in drawn, drawn[:300]


def test_the_temporal_card_data_marks_an_unfitted_trend_as_not_fitted():
    """R-11, the adapter's own record, because the CHIP is still drawn by the
    template. The card dict now carries trend_fitted False, the NOT FITTED label
    and the neutral slate for both the chip colour and the range bar, so the one
    remaining green element has a correct value sitting right beside it.
    """
    from vfairness.rendering import adapters_monitoring as AM

    captured = {}
    real = AM.render_svg
    try:
        AM.render_svg = lambda name, data, *a, **k: (captured.update(data), real(name, data))[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temporal_analysis_to_svg(_analyzer([0.05, 0.06, 0.07]))
    finally:
        AM.render_svg = real

    card = captured["metric_summaries"][0]
    assert card["tracked"] is True, "the analyzer does hold this metric"
    assert card["trend_fitted"] is False
    assert card["trend_label"] == "NOT FITTED"
    assert card["trend_color"] == AM.NOT_MEASURED_COLOR
    assert card["range_color"] == AM.NOT_MEASURED_COLOR
    assert card["slope_display"] == "not measured"
    assert card["mean"] == pytest.approx(0.06), "the measured statistics are untouched"


def test_control_a_flat_thirty_day_metric_still_gets_its_measured_stable_trend():
    """CONTROL for R-11. A metric with enough history to be fitted and genuinely
    flat SHOULD get the green STABLE chip and a real slope. Measured on 30 days
    at 0.05: trend_direction 'stable', trend_slope 0.0, the chip
    ('#41ba1b', 'STABLE') and "Slope/day +0.0000" on the canvas.
    """
    flat = _analyzer([0.05] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = flat.get_metric_summary("demographic_parity_difference")
        svg = temporal_analysis_to_svg(flat)

    assert summary["trend_direction"] == "stable"
    assert summary["trend_slope"] == pytest.approx(0.0)
    assert _CHIP.findall(svg)[0] == (_BLANCO_PASS, "STABLE")
    assert "Slope/day +0.0000" in canvas(svg)


def test_the_temporal_card_chip_does_not_certify_an_unfitted_trend_as_stable():
    """R-11 CLOSED, both halves.

    This was a strict xfail deferred to whoever owned the template. It is closed
    now and the assertion is unchanged; only the marker is gone.

    Measured before, on a real TemporalFairnessAnalyzer over 3 days where
    detect_trend warns and returns ('not_assessed', nan): the card read "Slope/day
    not measured" with a slate range bar and its data dict said NOT FITTED, and the
    CHIP read ('#41ba1b', 'STABLE'), byte-identical to the chip a genuinely flat
    30-day metric gets. STABLE is the calmest reading this card has, and no trend
    had been fitted. A card that contradicts its own table is read at the chip,
    which is the largest thing on it.

    The cause was in the template, not the adapter: a Jinja cascade whose
    {% else %} arm mapped EVERY direction other than increasing/decreasing to
    STABLE in the pass green, so no adapter value could change it. The template now
    reads m.trend_label and m.trend_color, and the chip background reads
    m.trend_fitted. A template is the wrong place for a verdict: it has no access
    to whether the fit happened, only to the word that came out of it.
    """
    unfitted = _analyzer([0.05, 0.06, 0.07])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = temporal_analysis_to_svg(unfitted)
    assert _CHIP.findall(svg)[0] != (_BLANCO_PASS, "STABLE"), canvas(svg)[:280]


def test_the_control_a_genuinely_flat_trend_still_reads_stable():
    """OVER-CORRECTION CONTROL. A real fitted flat trend must keep its chip.

    Without this, a template that drew "NOT FITTED" for everything would pass the
    test above. A 30-day flat metric is a measured stability and it is the answer a
    reader is entitled to.
    """
    flat = _analyzer([0.06] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = temporal_analysis_to_svg(flat)
    colour, label = _CHIP.findall(svg)[0]
    assert label == "STABLE", canvas(svg)[:280]
    assert colour == _BLANCO_PASS, colour
    assert "Slope/day +0.0000" in canvas(svg)
