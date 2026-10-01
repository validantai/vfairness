"""BGL5 wave-4 pins for the six overturned calibration grades.

Every grade in this file was published PROVEN, then overturned by an independent
auditor who found the SIBLING DOOR: the earlier fix closed one route into a
fabrication and left a second route into the same fabrication open, in the same
unit, with the same consequence for the reader.

  visualization.plot_calibration_disparity   the earlier fix keyed on
                                             ``excluded_groups``, a COUNT gate, so
                                             a group that CLEARED min_group_size
                                             and had no computable calibration
                                             error was drawn on all three panels
                                             with no mark, and its fabricated
                                             group_mce of 0.0 was drawn as a real
                                             bar: perfect worst-case calibration
                                             for a group nobody measured.

  tradeoffs.analyze_calibration_fairness_tradeoff   the earlier fix floored the
                                             BASE RATE's labelled denominator and
                                             left the per-threshold FPR/FNR arms
                                             with a floor of ONE labelled row, so
                                             a 120-row group holding one negative
                                             label published a rate, including a
                                             0.0 "perfect parity".

  tradeoffs.calibration_vs_error_parity      the earlier fix gave the GROUP a
                                             30-row floor and left the TPR/FPR
                                             arms with a floor of one labelled
                                             row, so one row flipped
                                             conflict_exists in both directions.

  tradeoffs.impossibility_diagnostics        the function warned that a group was
                                             EXCLUDED from the base-rate
                                             comparison and then decided the
                                             verdict from it, because the caller
                                             passes the dict and not the gate.

  tradeoffs.mitigation_pareto                the earlier fix filtered NaN labels
                                             and the object/string label path was
                                             exempt by construction, so a missing
                                             label became a ground-truth NEGATIVE
                                             and the mitigation published a 0.0
                                             accuracy cost where the real one is
                                             -50.

  metrics.ece_confidence_intervals           the earlier fix gave three states to
                                             six fields on the row and left
                                             near_null a two-state bool, and
                                             ``nan <= x`` is False, so a group
                                             with no point estimate published
                                             near_null FALSE, which asserts REAL
                                             miscalibration.

READ THE DRAWN VALUES, never the axis furniture: an empty matplotlib axes still
carries a "0.0" tick. The chart pins here read text artists and bar heights, and
one of them rasterises the figure twice, as drawn and with the disclosure artists
removed, and requires the two PNGs to differ. That is the only form of raster
comparison that could disagree here: three different states rendering a
byte-identical canvas is a failure this campaign has already seen, but so is a
digest that differs for a reason unrelated to the fix, which the first draft of
that pin did (see its own docstring).
"""

from __future__ import annotations

import hashlib
import io
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from vfairness.post_processing.calibration import visualization as calibration_plots
from vfairness.post_processing.calibration.metrics import (  # noqa: E402
    calibration_disparity,
    ece_confidence_intervals,
)
from vfairness.post_processing.calibration.tradeoffs import (  # noqa: E402
    _MIN_LABELS_PER_RATE,
    analyze_calibration_fairness_tradeoff,
    calibration_vs_error_parity,
    impossibility_diagnostics,
    mitigation_pareto,
)

pd = pytest.importorskip("pandas")


# ── shared fixtures ───────────────────────────────────────────────────────────


def _figure_texts(figure):
    """Every string DRAWN on a matplotlib figure.

    Tick labels are included only because the group names ARE tick labels here;
    nothing in these pins concludes anything from a tick's numeric value.
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


def _bar_heights(figure):
    """The height of every bar actually drawn, per panel."""
    return [[patch.get_height() for patch in axis.patches] for axis in figure.axes]


def _rasterise(figure) -> str:
    """sha256 of the rendered PNG.

    A data dict can differ while the CANVAS a person looks at is byte-identical.
    That has happened in this campaign (three distinct states, one sha256), so the
    disclosure is proven against the raster, not against the values behind it.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=50)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class _StandInDisparity:
    """The three dicts and the exclusion list ``plot_calibration_disparity`` reads.

    A stand-in, not a mock of behaviour: the renderer's whole job here is to draw
    these values, and building them directly is the only way to hold every panel
    fixed while varying ONLY whether B's value was measured. The end-to-end pin
    above runs the real ``calibration_disparity``.
    """

    def __init__(self, group_ece, group_mce, group_brier, excluded_groups=None):
        self.group_ece = group_ece
        self.group_mce = group_mce
        self.group_brier = group_brier
        self.excluded_groups = list(excluded_groups or [])


def _two_groups_one_unlabelled(n=200, seed=7):
    """``n`` rows of 'A' carrying real labels beside ``n`` rows of 'B' carrying
    none at all. Both clear min_group_size; only A can be measured."""
    rng = np.random.default_rng(seed)
    y_a = rng.integers(0, 2, n).astype(float)
    p_a = np.clip(y_a * 0.4 + 0.3 + rng.normal(0, 0.05, n), 0.01, 0.99)
    y_b = np.full(n, np.nan)
    p_b = np.clip(rng.uniform(0.0, 1.0, n), 0.01, 0.99)
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * n + ["B"] * n),
    )


def _two_groups_both_labelled(n=200, seed=7):
    """The CONTROL twin of the fixture above: identical shape, both groups
    genuinely labelled, so both carry a real measurement."""
    rng = np.random.default_rng(seed)
    y_a = rng.integers(0, 2, n).astype(float)
    p_a = np.clip(y_a * 0.4 + 0.3 + rng.normal(0, 0.05, n), 0.01, 0.99)
    y_b = rng.integers(0, 2, n).astype(float)
    p_b = np.clip(y_b * 0.2 + 0.4 + rng.normal(0, 0.05, n), 0.01, 0.99)
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * n + ["B"] * n),
    )


# ── A-10 sibling. The disparity chart marks the group it COULD NOT measure ────


class TestTheDisparityChartMarksTheGroupItCouldNotMeasure:
    """The earlier fix keyed on ``excluded_groups``, which holds only what the
    SIZE gate removed. A group inside the gate whose calibration error is NaN
    left no trace, and its fabricated MCE of 0.0 was drawn as a real bar."""

    def test_an_unmeasurable_group_is_marked_on_every_panel(self):
        """MEASURED BEFORE THE FIX, on this exact input:

        group_ece        {'A': 0.3006, 'B': nan}
        group_mce        {'A': 0.4248, 'B': 0.0}
        group_brier      {'A': 0.0927, 'B': nan}
        excluded_groups  []
        DRAWN TEXTS      'A' and 'B' on all three panels, suptitle
                         "Calibration Disparity Analysis", no mark of any kind
        warnings         []
        bar heights p2   [0.4248, 0.0]      <- the fabrication, drawn
        """
        y, prob, groups = _two_groups_one_unlabelled()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_disparity(y, prob, groups, min_group_size=30)
            figure = calibration_plots.plot_calibration_disparity(result)

        # The premise, asserted rather than assumed: B is NOT size-excluded, and
        # its MCE really does arrive as a fabricated 0.0 rather than a NaN.
        assert list(result.excluded_groups) == [], result.excluded_groups
        assert not np.isfinite(result.group_ece["B"])
        assert result.group_mce["B"] == 0.0, result.group_mce

        drawn = _figure_texts(figure)
        assert "Calibration not measured: B" in drawn, drawn
        # One "not measured" mark per panel, all three panels.
        assert drawn.count("not measured") == 3, drawn
        assert any("1 of 2 groups compared" in t and "B" in t for t in drawn), drawn
        assert any("Disparity not measured" in t for t in drawn), drawn

        # THE FABRICATED BAR IS GONE. Panel 2 no longer draws height 0.0 for B.
        heights = _bar_heights(figure)
        assert heights[1][0] == pytest.approx(0.42478638388197565)
        assert not np.isfinite(heights[1][1]), heights[1]
        for panel in heights:
            assert not np.isfinite(panel[1]), heights

        messages = " ".join(str(w.message) for w in caught)
        assert "plot_calibration_disparity" in messages, messages
        assert "NO calibration error could be computed" in messages, messages

    def test_the_disclosure_reaches_the_pixels_a_reader_looks_at(self):
        """THE RASTER PIN: the marks must be INK, not just text artists.

        A text artist can exist in ``ax.texts`` and still reach no pixels: clipped
        out, zero alpha, or placed outside the axes. Asserting the data behind a
        drawing is how three distinct states came to render a byte-identical canvas
        elsewhere in this campaign. So this rasterises the figure twice, once as
        drawn and once with every "not measured" artist REMOVED, and the two PNGs
        must differ: that difference IS the disclosure, measured in pixels.

        It is also built so it could have disagreed. Comparing the unmeasurable
        canvas against a perfectly-calibrated one does NOT work as a pin here and
        was tried first: a bar of height 0.0 drawn with ``edgecolor="black"`` leaves
        a horizontal rule that a NaN bar does not, so those two digests differ for
        a reason that has nothing to do with the fix, and the comparison stayed
        green with the whole fix reverted.
        """
        unmeasurable = _StandInDisparity(
            group_ece={"A": 0.30, "B": float("nan")},
            group_mce={"A": 0.42, "B": 0.0},
            group_brier={"A": 0.09, "B": float("nan")},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figure = calibration_plots.plot_calibration_disparity(unmeasurable)

        as_drawn = _rasterise(figure)

        removed = 0
        for axis in figure.axes:
            for artist in list(axis.texts):
                if "not measured" in artist.get_text():
                    artist.remove()
                    removed += 1
        # Three panel marks, the named disclosure box, and the banner saying the
        # disparity itself was not measured (one measured group is not a comparison).
        assert removed == 5, removed

        without_the_marks = _rasterise(figure)
        assert as_drawn != without_the_marks, (
            "the 'not measured' artists reach no pixels, so the canvas a reader "
            "looks at carries no disclosure"
        )

    def test_control_a_fully_measured_chart_carries_no_caveat(self):
        """THE OVER-CORRECTION CONTROL, with the healthy case's REAL numbers.

        A guard that marks everything passes every refusal test. Two groups that
        both carry a measurement must render the texts they rendered before, with
        no mark, no box, no suptitle suffix and no warning, and the bars must be
        the real measured values.
        """
        y, prob, groups = _two_groups_both_labelled()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = calibration_disparity(y, prob, groups, min_group_size=30)
            figure = calibration_plots.plot_calibration_disparity(result)

        assert _figure_texts(figure) == [
            "Expected Calibration Error",
            "A",
            "B",
            "Good calibration threshold",
            "Maximum Calibration Error",
            "A",
            "B",
            "Brier Score",
            "A",
            "B",
            "Calibration Disparity Analysis",
        ], _figure_texts(figure)

        # The REAL numbers, on the canvas, not merely "it did not raise".
        heights = _bar_heights(figure)
        assert heights[0][0] == pytest.approx(result.group_ece["A"])
        assert heights[0][1] == pytest.approx(result.group_ece["B"])
        assert heights[1][0] == pytest.approx(result.group_mce["A"])
        assert heights[2][1] == pytest.approx(result.group_brier["B"])
        assert all(np.isfinite(v) for panel in heights for v in panel), heights


# ── A-6 sibling. near_null is a three-state, like every field beside it ───────


class TestNearNullCannotBeMeasuredWithoutAPointEstimate:
    """``near_null`` was ``bool(point <= noise_floor + 2 * noise_sd)`` and
    ``nan <= anything`` is False, so a group with NO point estimate published
    near_null FALSE, which the docstring defines as "the observed ECE IS
    distinguishable from estimation noise", i.e. real miscalibration. There is no
    observed ECE. The function's OTHER return path already writes None."""

    def test_the_arithmetic_that_produced_the_fabrication(self):
        """The mechanism, pinned on its own so the reason cannot be lost: the
        comparison this field was built on answers False for a NaN point."""
        assert bool(float("nan") <= 0.10120908735531589 + 2.0 * 0.01) is False

    def test_a_group_with_no_point_estimate_reports_near_null_as_none(self):
        """MEASURED BEFORE THE FIX, 200 rows in two 100-row groups, A labelled and
        B carrying no ground truth, n_bootstrap=60:

            A: ece 0.1162, ci [0.0855, 0.2124], unstable True, near_null True
            B: ece NAN, ci [nan, nan], unstable NONE, not_assessed '...',
               AND near_null FALSE with noise_floor 0.1012

        Every other field on B's row correctly said could-not-check.
        """
        y, prob, groups = _two_groups_one_unlabelled(n=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ece_confidence_intervals(
                y, prob, protected_attr=groups, n_bootstrap=60, random_state=0
            )

        rows = result if isinstance(result, dict) else {}
        assert "B" in rows, list(rows)
        b = rows["B"]

        # The premise: this really is the unmeasurable row.
        assert not np.isfinite(b["ece"]), b
        assert b["unstable"] is None, b
        assert b["not_assessed"], b

        # THE FIX: the field beside them agrees with them.
        assert b["near_null"] is None, b

        # THE SIBLING DOOR, in the one field on this row a reader reads: the
        # disclosure used to end "the point estimate stands on its own" while the
        # point estimate was NaN, so it reported that only the interval was lost.
        assert "stands on its own" not in b["not_assessed"], b["not_assessed"]
        assert "NO ECE could be computed" in b["not_assessed"], b["not_assessed"]

    def test_control_a_measured_group_still_gets_a_real_near_null_verdict(self):
        """THE OVER-CORRECTION CONTROL, with the healthy row's REAL numbers.

        None for everything passes every refusal test. Group A in the same call,
        beside the unmeasurable B, must still publish a measured interval and a
        near_null verdict by identity, and that verdict must be the one its own
        numbers imply: an ECE of 0.288 against a noise floor of 0.082 is real
        miscalibration, so near_null is FALSE and this control is the direction
        that would break if the guard refused everything.
        """
        y, prob, groups = _two_groups_one_unlabelled(n=100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ece_confidence_intervals(
                y, prob, protected_attr=groups, n_bootstrap=60, random_state=0
            )
        a = result["A"]

        assert a["ece"] == pytest.approx(0.288161868435946, rel=1e-9), a
        assert a["noise_floor"] == pytest.approx(0.08189304670859397, rel=1e-9), a
        assert a["ci_lower"] == pytest.approx(0.2795819670754449, rel=1e-9), a
        assert a["ci_upper"] == pytest.approx(0.2956275520764294, rel=1e-9), a
        assert a["not_assessed"] is None, a
        assert a["unstable"] is False, a
        # A REAL verdict, by identity, and the one its numbers imply: the ECE is
        # 3.5x the noise floor, so it is NOT indistinguishable from noise.
        assert a["near_null"] is False, a
        assert a["ece"] > a["noise_floor"] + 2.0 * 0.01, a


# ── shared tradeoff fixtures ──────────────────────────────────────────────────


def _two_groups_b_has_one_negative(b_negative_score):
    """Two 120-row groups, B holding exactly ONE negative-labelled row.

    Used for the fpr_parity arm: A has 48 negatives, B has one. The score given to
    B's single negative row is the only thing that varies.
    """
    y_a = np.array([0] * 48 + [1] * 72, dtype=float)
    p_a = np.concatenate([np.full(48, 0.80), np.full(72, 0.90)])
    y_b = np.array([0] + [1] * 119, dtype=float)
    p_b = np.concatenate([[b_negative_score], np.full(119, 0.90)])
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * 120 + ["B"] * 120),
    )


def _two_groups_b_has_one_positive(b_positive_score):
    """Two 120-row groups, A with 12 positive labels and B with exactly ONE.

    Tuned so the calibration-gap and FPR arms stay inside the 0.05 tolerance and
    the base rates differ (0.0917), which is what makes the TPR arm the single
    thing deciding ``conflict_exists``.
    """
    y_a = np.array([1] * 12 + [0] * 108, dtype=float)
    p_a = np.full(120, 0.10)
    y_b = np.array([1] + [0] * 119, dtype=float)
    p_b = np.concatenate([[b_positive_score], np.full(119, 0.00825)])
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * 120 + ["B"] * 120),
    )


def _healthy_two_groups():
    """The CONTROL frame: two 120-row groups, both carrying plenty of labels of
    both classes, with a REAL and unequal FPR."""
    y_a = np.array([0] * 60 + [1] * 60, dtype=float)
    # A's negatives: 30 of 60 above 0.5 -> fpr 0.5
    p_a = np.concatenate([np.full(30, 0.80), np.full(30, 0.20), np.full(60, 0.90)])
    y_b = np.array([0] * 60 + [1] * 60, dtype=float)
    # B's negatives: 6 of 60 above 0.5 -> fpr 0.1
    p_b = np.concatenate([np.full(6, 0.80), np.full(54, 0.20), np.full(60, 0.90)])
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * 120 + ["B"] * 120),
    )


# ── The per-threshold fairness rate needs LABELS, not one row ─────────────────


class TestThePerThresholdRateNeedsLabelsNotOneRow:
    """The earlier fix floored the BASE RATE's labelled denominator and its own
    comment states the rule: "the gate counts ROWS and a base rate needs LABELS".
    The identical sentence is true one field over and was not applied there: the
    per-threshold FPR/FNR denominator had a floor of ONE."""

    def test_a_perfect_parity_from_one_negative_row_is_not_measured(self):
        """MEASURED BEFORE THE FIX, the FALSE-CLEAN direction: A's 48 negatives all
        scored above the threshold and B's ONE negative row likewise.

            group_metrics at 0.50  {'A': 1.0, 'B': 1.0}
            fairness_violation     0.0     <- PERFECT FPR PARITY as a measurement
            pareto frontier        the 0.0 became a frontier point at every one of
                                   17 thresholds
            warnings               []
        """
        y, prob, groups = _two_groups_b_has_one_negative(0.80)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze_calibration_fairness_tradeoff(
                y, prob, groups, fairness_metric="fpr_parity"
            )

        at_half = result.metrics_at_thresholds[
            min(result.metrics_at_thresholds, key=lambda t: abs(t - 0.5))
        ]
        # The premise: B really does hold exactly one negative label, and 120 rows.
        assert result.rate_label_counts["B"] == 1, result.rate_label_counts
        assert result.rate_label_counts["A"] == 48, result.rate_label_counts

        assert np.isnan(at_half["group_metrics"]["B"]), at_half["group_metrics"]
        assert np.isnan(at_half["fairness_violation"]), at_half
        assert result.groups_without_measured_rate == ["B"], result
        # The disclosure travels on the RESULT, not only in a warning a caller can
        # drop, and on to_dict, which is what a report reads.
        assert result.to_dict()["groups_without_measured_rate"] == ["B"]
        assert result.to_dict()["rate_label_counts"]["B"] == 1
        # The reader-facing line says so too.
        assert any("NOT ASSESSED" in r and "fpr_parity" in r for r in result.recommendations), (
            result.recommendations
        )

        messages = " ".join(str(w.message) for w in caught)
        assert "analyze_calibration_fairness_tradeoff" in messages
        assert f"fewer than {_MIN_LABELS_PER_RATE} label(s)" in messages, messages

    def test_the_verdict_no_longer_turns_on_that_one_row(self):
        """The property that matters: the published number must be INVARIANT to
        the score of a single labelled row. Before the fix the same two frames gave
        fairness_violation 0.8333 and 0.0 (measured); after, both are not measured.
        """
        low = analyze_calibration_fairness_tradeoff(
            *_two_groups_b_has_one_negative(0.05), fairness_metric="fpr_parity"
        )
        high = analyze_calibration_fairness_tradeoff(
            *_two_groups_b_has_one_negative(0.80), fairness_metric="fpr_parity"
        )
        for result in (low, high):
            at_half = result.metrics_at_thresholds[
                min(result.metrics_at_thresholds, key=lambda t: abs(t - 0.5))
            ]
            assert np.isnan(at_half["fairness_violation"]), at_half

    def test_the_floor_sits_above_the_metric_dispatch(self):
        """PUT THE GUARD ABOVE THE DISPATCH. fnr_parity reads the POSITIVE labels,
        so a floor written inside the fpr_parity branch would have left this one
        publishing a rate from one row. Same frame, other arm.
        """
        y, prob, groups = _two_groups_b_has_one_positive(0.9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_calibration_fairness_tradeoff(
                y, prob, groups, fairness_metric="fnr_parity"
            )
        at_half = result.metrics_at_thresholds[
            min(result.metrics_at_thresholds, key=lambda t: abs(t - 0.5))
        ]
        assert result.rate_label_counts["B"] == 1, result.rate_label_counts
        assert np.isnan(at_half["group_metrics"]["B"]), at_half["group_metrics"]
        assert np.isnan(at_half["fairness_violation"]), at_half

    def test_control_demographic_parity_still_measures_every_group(self):
        """THE OVER-CORRECTION CONTROL for the dispatch. demographic_parity is a
        rate over PREDICTIONS, so it needs no labels at all and must NOT be refused
        for a thin label column: the same frame that refuses fpr_parity publishes a
        REAL demographic-parity gap here.
        """
        y, prob, groups = _two_groups_b_has_one_negative(0.80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_calibration_fairness_tradeoff(
                y, prob, groups, fairness_metric="demographic_parity"
            )
        at_half = result.metrics_at_thresholds[
            min(result.metrics_at_thresholds, key=lambda t: abs(t - 0.5))
        ]
        # Every row of both groups is scored at or above 0.5, so both selection
        # rates are exactly 1.0 and the REAL measured gap is 0.0.
        assert at_half["group_metrics"] == {"A": 1.0, "B": 1.0}, at_half["group_metrics"]
        assert at_half["fairness_violation"] == 0.0, at_half
        assert result.rate_label_counts == {}, result.rate_label_counts
        assert result.groups_without_measured_rate == [], result

    def test_control_a_healthy_frame_publishes_its_real_fpr_gap(self):
        """THE OVER-CORRECTION CONTROL, with the healthy case's REAL number. A
        guard that refuses everything passes every refusal test above. Two groups
        with 60 negative labels each must still publish a measured FPR gap, and it
        must be the right one: A 30/60 = 0.5 against B 6/60 = 0.1, so 0.4.
        """
        y, prob, groups = _healthy_two_groups()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_calibration_fairness_tradeoff(
                y, prob, groups, fairness_metric="fpr_parity"
            )
        at_half = result.metrics_at_thresholds[
            min(result.metrics_at_thresholds, key=lambda t: abs(t - 0.5))
        ]
        assert result.rate_label_counts == {"A": 60, "B": 60}, result.rate_label_counts
        assert result.groups_without_measured_rate == [], result
        assert at_half["group_metrics"]["A"] == pytest.approx(0.5)
        assert at_half["group_metrics"]["B"] == pytest.approx(0.1)
        assert at_half["fairness_violation"] == pytest.approx(0.4)


# ── The TPR/FPR arms need labels, and one row must not flip the verdict ───────


class TestTheErrorParityArmsNeedLabelsNotOneRow:
    """The earlier fix gave the GROUP a 30-row floor because a one-ROW group could
    contribute a base rate of 0.0 or 1.0. The arms it feeds kept a floor of ONE, so
    a 120-row group with exactly one positive label cleared the gate and
    contributed a TPR of exactly 0.0 or 1.0, which is the same thing one level up."""

    def test_one_labelled_row_no_longer_flips_conflict_exists(self):
        """MEASURED BEFORE THE FIX, the same data twice, differing only in the
        score of B's single positive row:

            row at 0.018  tpr {'A': 0.0, 'B': 0.0}, tpr_disparity 0.0,
                          calibration_gap_disparity 0.0, base_rate_disparity
                          0.09167, conflict_exists FALSE (a clean bill),
                          not_assessed [], excluded_groups [], warnings []
            row at 0.9    tpr {'A': 0.0, 'B': 1.0}, tpr_disparity 1.0,
                          conflict_exists TRUE, not_assessed [], warnings []
        """
        verdicts = {}
        for score in (0.018, 0.9):
            y, prob, groups = _two_groups_b_has_one_positive(score)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = calibration_vs_error_parity(y, prob, groups)
            verdicts[score] = result

            # The premise: B clears the 30-row gate on ROWS and holds one label.
            assert result["group_metrics"]["B"]["n_samples"] == 120
            assert result["group_metrics"]["B"]["n_positive_labels"] == 1
            assert result["excluded_groups"] == []

            assert np.isnan(result["group_metrics"]["B"]["tpr"]), result["group_metrics"]
            assert np.isnan(result["tpr_disparity"]), result
            assert "tpr_disparity" in result["not_assessed"], result["not_assessed"]
            assert result["conflict_exists"] is None, result
            assert "B" in result["rates_from_too_few_labels"], result
            assert result["rates_from_too_few_labels"]["B"]["n_positive_labels"] == 1
            assert result["min_labels_per_rate"] == _MIN_LABELS_PER_RATE

            messages = " ".join(str(w.message) for w in caught)
            assert "calibration_vs_error_parity" in messages
            assert "labels of one class" in messages.lower(), messages

        # THE PROPERTY: the headline verdict is the same either way now. Before, it
        # was False for one and True for the other.
        assert verdicts[0.018]["conflict_exists"] is verdicts[0.9]["conflict_exists"] is None, {
            s: v["conflict_exists"] for s, v in verdicts.items()
        }

    def test_control_a_healthy_frame_publishes_its_real_tpr_and_fpr(self):
        """THE OVER-CORRECTION CONTROL, with the healthy case's REAL numbers. Two
        groups with 60 labels of each class: A's 60 positives all scored 0.90 so
        tpr 1.0, A's negatives 30/60 above so fpr 0.5, B's fpr 6/60 = 0.1. The arms
        must be measured, the labelled counts must be reported, and the thin-label
        disclosure must be EMPTY, or every result would read as partial.
        """
        y, prob, groups = _healthy_two_groups()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_vs_error_parity(y, prob, groups)

        metrics = result["group_metrics"]
        assert metrics["A"]["n_positive_labels"] == 60
        assert metrics["A"]["n_negative_labels"] == 60
        assert metrics["A"]["tpr"] == pytest.approx(1.0)
        assert metrics["B"]["tpr"] == pytest.approx(1.0)
        assert metrics["A"]["fpr"] == pytest.approx(0.5)
        assert metrics["B"]["fpr"] == pytest.approx(0.1)
        assert result["tpr_disparity"] == pytest.approx(0.0)
        assert result["fpr_disparity"] == pytest.approx(0.4)
        assert result["not_assessed"] == [], result["not_assessed"]
        assert result["rates_from_too_few_labels"] == {}, result
        assert result["conflict_exists"] is not None
        assert not any(
            "too few labels" in str(w.message) or "labels of one class" in str(w.message).lower()
            for w in caught
        ), [str(w.message) for w in caught]


# ── The impossibility verdict and its own exclusion warning must agree ────────


def _three_groups_c_is_small():
    """A and B of 120 rows at base rate 0.5, plus C of 10 rows at base rate 0.0.

    C is BELOW impossibility_diagnostics' own default gate of 30 and is the only
    thing that could move the base-rate disparity off 0.0.
    """
    rng = np.random.default_rng(5)

    def group(n, rate, name):
        y = np.array([1] * int(round(n * rate)) + [0] * (n - int(round(n * rate))), dtype=float)
        p = np.clip(y * 0.3 + 0.35 + rng.normal(0, 0.05, n), 0.01, 0.99)
        return y, p, np.array([name] * n)

    parts = [group(120, 0.5, "A"), group(120, 0.5, "B"), group(10, 0.0, "C")]
    return (
        np.concatenate([x[0] for x in parts]),
        np.concatenate([x[1] for x in parts]),
        np.concatenate([x[2] for x in parts]),
    )


class TestTheImpossibilityExclusionWarningIsTrueByConstruction:
    """The function warned that a group was EXCLUDED from the base-rate comparison
    and then decided the verdict from it, because its only caller in the main path
    passed the base_rates DICT and not min_group_size, so the rates arrived built
    under the caller's gate while this function gated at its own default of 30."""

    def test_a_group_below_this_functions_gate_does_not_decide_the_verdict(self):
        """MEASURED BEFORE THE FIX, a caller-supplied dict holding a group of 10
        rows handed to the default gate of 30:

            base_rates {'A': 0.5, 'B': 0.5, 'C': 0.0}, base_rate_disparity 0.5,
            base_rates_differ True, impossibility_applies TRUE,
            n_groups_compared 3, excluded_groups ['C']

        and the warning printed on that same run, verbatim: "1 group(s) had fewer
        than 30 samples and were EXCLUDED from the base-rate comparison (C); the
        verdict rests on the 3 that remain." C was not excluded: it was the only
        thing that moved the disparity from 0.0 to 0.5. "3 that remain" out of 3
        total gave it away, and n_groups_compared counted C while excluded_groups
        named it as excluded.
        """
        y, prob, groups = _three_groups_c_is_small()
        supplied = {"A": 0.5, "B": 0.5, "C": 0.0}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = impossibility_diagnostics(y, prob, groups, supplied)

        # C is gone from the comparison, so the two fields agree with each other.
        assert diagnosis["excluded_groups"] == ["C"], diagnosis
        assert "C" not in diagnosis["base_rates"], diagnosis["base_rates"]
        assert diagnosis["n_groups_compared"] == 2, diagnosis
        assert diagnosis["base_rate_disparity"] == pytest.approx(0.0)
        assert diagnosis["base_rates_differ"] is False
        assert diagnosis["impossibility_applies"] is False

        # The warning is now TRUE: "the verdict rests on the 2 that remain".
        exclusion = [
            str(w.message) for w in caught if "EXCLUDED from the base-rate" in str(w.message)
        ]
        assert len(exclusion) == 1, exclusion
        assert "the verdict rests on the 2 that remain" in exclusion[0], exclusion[0]
        assert "the verdict rests on the 3" not in exclusion[0], exclusion[0]

    def test_the_verdict_is_identical_with_and_without_the_excluded_group(self):
        """The property the warning claims: a group it calls excluded can have no
        effect on the verdict. Before the fix, adding C flipped
        impossibility_applies from False to True.
        """
        y, prob, groups = _three_groups_c_is_small()
        keep = groups != "C"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            without_c = impossibility_diagnostics(
                y[keep], prob[keep], groups[keep], {"A": 0.5, "B": 0.5}
            )
            with_c = impossibility_diagnostics(y, prob, groups, {"A": 0.5, "B": 0.5, "C": 0.0})

        for key in (
            "base_rate_disparity",
            "base_rates_differ",
            "impossibility_applies",
            "n_groups_compared",
        ):
            assert with_c[key] == without_c[key] or (
                isinstance(with_c[key], float)
                and np.isnan(with_c[key])
                and np.isnan(without_c[key])
            ), (key, with_c[key], without_c[key])

    def test_the_main_trade_off_path_passes_its_own_gate_through(self):
        """The other half of the same defect: the caller now passes min_group_size,
        so the two gates are ONE gate. At min_group_size=5 the 10-row group C is
        legitimately inside the comparison, and the result must say so without
        contradicting itself: no group is both counted and named as excluded.
        """
        y, prob, groups = _three_groups_c_is_small()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyze_calibration_fairness_tradeoff(y, prob, groups, min_group_size=5)
        diagnosis = result.impossibility_diagnosis

        assert diagnosis["n_groups_compared"] == 3, diagnosis
        assert diagnosis["excluded_groups"] == [], diagnosis
        assert "C" in diagnosis["base_rates"], diagnosis["base_rates"]
        # No self-contradicting exclusion warning, because there is no exclusion.
        assert not [
            str(w.message) for w in caught if "EXCLUDED from the base-rate" in str(w.message)
        ], [str(w.message) for w in caught]

    def test_control_a_real_difference_is_still_found(self):
        """THE OVER-CORRECTION CONTROL, with the REAL number. Dropping everything
        would pass every test above. Two groups of 120 rows whose base rates
        genuinely differ, both well above the gate, must still be compared and must
        report the real disparity: 0.8 against 0.2, so 0.6, and the theorem applies.
        """
        y_a = np.array([1] * 96 + [0] * 24, dtype=float)
        y_b = np.array([1] * 24 + [0] * 96, dtype=float)
        y = np.concatenate([y_a, y_b])
        prob = np.clip(y * 0.3 + 0.35, 0.01, 0.99)
        groups = np.array(["A"] * 120 + ["B"] * 120)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            diagnosis = impossibility_diagnostics(y, prob, groups)

        assert diagnosis["base_rates"]["A"] == pytest.approx(0.8)
        assert diagnosis["base_rates"]["B"] == pytest.approx(0.2)
        assert diagnosis["base_rate_disparity"] == pytest.approx(0.6)
        assert diagnosis["base_rates_differ"] is True
        assert diagnosis["is_degenerate"] is False
        assert diagnosis["impossibility_applies"] is True
        assert diagnosis["n_groups_compared"] == 2
        assert diagnosis["excluded_groups"] == []
        assert not [str(w.message) for w in caught], [str(w.message) for w in caught]


# ── An unlabelled row in a TEXT label column is not a ground-truth negative ───


def _two_groups_scored_apart(label_column):
    """Two 100-row groups, both fully scored, A high and B low, with whatever
    label column the caller hands in. Group A is genuinely labelled throughout."""
    p_a = np.linspace(0.55, 0.95, 100)
    p_b = np.linspace(0.05, 0.45, 100)
    return (
        label_column,
        np.concatenate([p_a, p_b]),
        np.array(["A"] * 100 + ["B"] * 100),
    )


def _labels_for_group_a():
    return (np.linspace(0.55, 0.95, 100) >= 0.75).astype(int)


class TestAnUnlabelledTextRowIsNotAGroundTruthNegative:
    """The earlier fix filtered NaN labels and its own source declared the
    object/string path exempt ("``_isnan`` returns all-False for a non-float label
    array"). That path resolved a text column with ``.isin(<positives>)``, so
    anything outside the positive set became a ground-truth NEGATIVE, which is
    verbatim the defect the fix's own heading declared closed."""

    def test_a_text_column_missing_label_is_excluded_not_counted_as_a_negative(self):
        """MEASURED BEFORE THE FIX, on identical data varying ONLY the spelling of
        the missing label:

            FLOAT labels, B NaN        -> available False, nRowsUnlabelled 100
                                          (the path the earlier fix covers)
            OBJECT labels, B 'unknown' -> available TRUE, labelAware TRUE,
                                          nRowsUnlabelled 0, nGroupsMeasurable 2
                                          of 2, baselineGap 1.0, bestGap 0.0,
                                          summary "... at 0.0 points of accuracy
                                          cost", warnings []
            CONTROL, B really labelled -> the same gaps, cost -50.0 points

        The 0.0-point cost is the fabrication: 100 rows nobody labelled were
        counted as correctly rejected, and the published cost of the mitigation
        differed by FIFTY accuracy points from the same frame with real labels.
        """
        labels = np.array([str(v) for v in _labels_for_group_a()] + ["unknown"] * 100, dtype=object)
        result = mitigation_pareto(*_two_groups_scored_apart(labels))

        assert result["available"] is False, result
        assert result["nRowsUnlabelled"] == 100, result
        assert result["nRowsUnscored"] == 0, result
        assert result["nGroupsMeasurable"] == 1, result
        assert result["nGroups"] == 2, result
        assert "not a ground-truth negative" in result["reason"], result["reason"]
        # The refusal is ACTIONABLE: it names the spelling it did not recognise, so
        # a caller whose column uses a negative word this library does not know can
        # tell that apart from genuinely missing labels.
        assert "'unknown'" in result["reason"], result["reason"]
        # And no 0.0-cost summary was published at all.
        assert "accuracy cost" not in result.get("summary", ""), result

    def test_every_spelling_of_absence_lands_in_the_same_place(self):
        """Absence is not emptiness: None, float nan, the empty string and the
        literal string 'None' are four different doors into the same fabrication,
        and ``str(x)`` on an absent value MINTS content that is not a label.
        """
        for spelling in (None, float("nan"), "", "None", "n/a", "pending"):
            labels = np.array(
                [str(v) for v in _labels_for_group_a()] + [spelling] * 100, dtype=object
            )
            result = mitigation_pareto(*_two_groups_scored_apart(labels))
            assert result["available"] is False, (spelling, result)
            assert result["nRowsUnlabelled"] == 100, (spelling, result)

    def test_control_a_genuinely_labelled_text_column_keeps_its_real_cost(self):
        """THE OVER-CORRECTION CONTROL, with the healthy case's REAL number.

        Refusing every text label column would pass every test above. The SAME
        object column with group B genuinely labelled must still be available and
        must publish its real accuracy cost of -50.0 points, which is the figure
        the fabricated 0.0 was standing in for.
        """
        labels = np.array(
            [str(v) for v in _labels_for_group_a()]
            + [str(v) for v in (np.linspace(0.05, 0.45, 100) >= 0.25).astype(int)],
            dtype=object,
        )
        result = mitigation_pareto(*_two_groups_scored_apart(labels))

        assert result["available"] is True, result
        assert result["labelAware"] is True, result
        assert result["nRowsUnlabelled"] == 0, result
        assert result["nGroupsMeasurable"] == 2, result
        assert result["baselineGap"] == pytest.approx(1.0)
        assert result["bestGap"] == pytest.approx(0.0)
        assert "at -50.0 points of accuracy cost" in result["summary"], result["summary"]

    def test_control_the_negative_vocabulary_words_are_read_as_negatives(self):
        """THE SECOND OVER-CORRECTION CONTROL. The new negative vocabulary must
        actually resolve to a ground-truth 0, not to "no label": a yes/no column is
        fully labelled and must be measured, with nRowsUnlabelled 0.
        """
        labels = np.array(
            ["yes" if v else "no" for v in _labels_for_group_a()]
            + ["yes" if v else "no" for v in (np.linspace(0.05, 0.45, 100) >= 0.25).astype(int)],
            dtype=object,
        )
        result = mitigation_pareto(*_two_groups_scored_apart(labels))

        assert result["available"] is True, result
        assert result["labelAware"] is True, result
        assert result["nRowsUnlabelled"] == 0, result
        assert "at -50.0 points of accuracy cost" in result["summary"], result["summary"]

    def test_a_mixed_text_column_no_longer_dies_into_the_generic_except(self):
        """The auditor's SECONDARY finding on this unit. With the missing label
        spelled as None in an object column, the run died into the generic except
        and returned reason "Trade-off unavailable: '<' not supported between
        instances of 'NoneType' and 'str'" with nRowsUnlabelled, nRowsUnscored,
        nGroups and nGroupsMeasurable ALL None, which breaks the promise the two
        other return paths make that those counts are "always present, on every
        return path". The cause was ``np.unique``, which SORTS, so a mixed object
        column raised before any count was reached.

        The distinct-label test now runs on the normalised strings, so this input
        travels the normal path. It carries every count, and because fewer than two
        of its values are recognisable labels it says out loud that no label was
        read, rather than leaving labelAware=False to read as "none was supplied".
        """
        labels = np.array(["yes"] * 100 + [None] * 100, dtype=object)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mitigation_pareto(*_two_groups_scored_apart(labels))

        assert "Trade-off unavailable" not in result.get("reason", ""), result
        for key in ("nRowsUnscored", "nRowsUnlabelled", "nGroups", "nGroupsMeasurable"):
            assert result[key] is not None, (key, result)
        assert result["labelAware"] is False, result

        messages = " ".join(str(w.message) for w in caught)
        assert "mitigation_pareto" in messages, messages
        assert "NOT treated as a negative outcome" in messages, messages
        assert "labelAware=False" in messages, messages

    def test_the_disclosure_counts_are_present_on_the_except_path(self):
        """And when something genuinely does raise, the counts are still KEYS, held
        at None where they were never reached rather than absent. None is the honest
        answer for a count that did not happen; 0 would state that nothing was
        dropped, and an absent key forces every reader into ``.get(k, 0)``.
        """

        # A sensitive_attr that cannot be compared for uniqueness: the group array
        # is where this one fails, after the score arm is validated.
        class _Exploding:
            def __array__(self, dtype=None, copy=None):
                raise RuntimeError("group array is unreadable")

        result = mitigation_pareto(
            np.array([0, 1] * 50, dtype=float),
            np.linspace(0.1, 0.9, 100),
            _Exploding(),
        )
        assert result["available"] is False, result
        assert "Trade-off unavailable" in result["reason"], result
        for key in ("nRowsUnscored", "nRowsUnlabelled", "nGroups", "nGroupsMeasurable"):
            assert key in result, (key, result)
            assert result[key] is None, (key, result)
