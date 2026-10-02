"""Surface grade G03: the calibration containers, serialisers and three plots.

Batch G03 graded 27 units in ``vfairness.post_processing.calibration``: the
result dataclasses, their ``to_dict`` serialisers, the three warning categories,
``BaseCalibrator``, and ``plot_reliability_diagram`` /
``plot_group_calibration`` / ``plot_pareto_frontier``. Seven defects were
reproduced by execution and are pinned here.

G03-1  ``calibration_disparity`` computed ``max(ece_values) - min(ece_values)``
       over per-group values that can hold a NaN. Builtin max/min do NOT
       propagate a NaN that is not first in the iterable, so a comparison in
       which one of two groups was never measured came back as a FINITE 0.0:
       PERFECT PARITY, past every ``math.isfinite`` guard a consumer can write,
       because the value those guards inspect is finite. On 240 rows with group
       B unlabelled: group_ece {'A': 0.58, 'B': nan}, ece_disparity 0.0,
       has_significant_disparity False, recommendations leading with "LOW:
       Calibration disparity is acceptable (0.000).", excluded_groups [], and
       ZERO warnings. With the groups in the other order the same data gave NaN,
       so the verdict depended on which group appeared first. This is the root
       cause the 2026-09-27 ``recommend_calibration_strategy`` guard names in its
       own comment as out of that batch's scope.

G03-2  ``CalibrationDisparityResult.has_significant_disparity`` keyed its
       coverage test on ``excluded_groups``, the row-COUNT gate, so a group
       INSIDE the gate with no computable calibration error left the clean
       verdict a plain ``False``. ``plot_calibration_disparity`` was fixed for
       this same input on 2026-09-30 and derives the list from the VALUES; the
       verdict beside the chart did not.

G03-3  ``CalibrationMetricResult`` tested "was this measured?" with a local
       ``v is not None and math.isfinite(float(v))`` instead of the
       repository's canonical ``_triage.is_measured``, and disagreed with it in
       both directions: a ``bool`` counted as a measured group worth 1.0
       (``n_groups_compared`` 2, ``max_group_disparity`` 0.98), a numeric STRING
       was counted by the COUNT and then crashed the ARITHMETIC, and a blank
       string, the literal "None", ``pd.NA`` and ``pd.NaT`` each raised out of
       ``float()`` inside ``to_dict``.

G03-4  ``IntegratedCalibrationResult.has_significant_disparity`` mirrored only
       the first half of its ECE sibling: a clean ``False`` over a SUBSET of the
       groups. 180 'a' + 180 'b' well calibrated beside 15 excluded 'c' scored
       0.9 against 0 gave ici_disparity 0.0 and False; letting 'c' in gives
       0.89976 and True.

G03-5  ``MulticalibrationResult.is_multicalibrated`` published the CLEAN
       ``True`` over a subset, through two doors. The size gate, and the
       evidence gate no count can see: a group that cleared ``min_group_size``
       and had no cell reaching ``min_cell`` carried ``group_max`` NaN with
       ``excluded_groups []``, ``alpha 3.33e-16``, ``is_multicalibrated True``
       and NOT ONE WARNING.

G03-6  ``plot_pareto_frontier`` re-implements, inline, the dominance test that
       ``ParetoPoint.dominates`` and ``compute_pareto_frontier`` were fixed for
       on 2026-09-27, and kept the defect: a point with a non-finite coordinate
       is nothing's inferior, so it went into the red "Pareto Optimal" scatter,
       and the call then died with matplotlib's bare "Axis limits cannot be NaN
       or Inf".

G03-7  ``plot_group_calibration`` masked groups with ``protected_attr ==
       group``, and ``attr == nan`` is False for EVERY row, so a level holding
       HALF the dataset was drawn as "Not measured (n<10): nan (n=0)" while its
       rows stayed inside the pooled curve. ``GroupManager._level_mask`` was
       fixed for exactly this and its rule is reused now. An object-dtype
       attribute with one float NaN raised a bare TypeError out of np.unique.

EVERY pin carries an OVER-CORRECTION CONTROL asserting the healthy case's REAL
number or verdict. A guard that refuses everything passes every refusal test and
destroys the unit.
"""

from __future__ import annotations

import hashlib
import io
import math
import warnings

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# matplotlib is the optional [viz] extra, so the module must still import
# without it (the lowest-versions CI job installs no extras). Only the tests
# that draw are marked needs_matplotlib and skip; the rest still run.
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
    plt = None
else:
    matplotlib.use("Agg")  # no display in CI; must precede pyplot
    import matplotlib.pyplot as plt
needs_matplotlib = pytest.mark.skipif(
    matplotlib is None, reason="needs the optional [viz] extra (matplotlib)"
)

from vfairness._not_assessed import (  # noqa: E402
    MIN_ROWS_PER_GROUP_FOR_CALIBRATION as MIN_GROUP_CURVE,
)
from vfairness.post_processing.calibration import visualization as cal_plots  # noqa: E402
from vfairness.post_processing.calibration.analyzer import CalibrationReport  # noqa: E402
from vfairness.post_processing.calibration.group_calibrator import (  # noqa: E402
    GroupCalibrationResult,
    IntersectionalProvenanceWarning,
    UnknownGroupWarning,
    UnmeasurableGroupWarning,
)
from vfairness.post_processing.calibration.methods import (  # noqa: E402
    BaseCalibrator,
    CalibrationFitResult,
    PlattScaling,
)
from vfairness.post_processing.calibration.metrics import (  # noqa: E402
    BrierDecomposition,
    CalibrationCurveResult,
    CalibrationDisparityResult,
    CalibrationMetricResult,
    calibration_disparity,
    integrated_calibration_index,
    multicalibration,
)
from vfairness.post_processing.calibration.tradeoffs import (  # noqa: E402
    CalibrationRecommendation,
    ParetoPoint,
    TradeoffAnalysisResult,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    if plt is not None:
        plt.close("all")


def _messages(caught) -> list[str]:
    return [str(w.message) for w in caught]


def _figure_texts(artist) -> list[str]:
    """Every string a reader can see on one axes: annotations and legend."""
    out = [t.get_text() for t in artist.texts]
    legend = artist.get_legend()
    if legend is not None:
        out += [t.get_text() for t in legend.get_texts()]
    return out


# ===========================================================================
# Fixtures. Deterministic, no rng, so every number in a docstring is
# reproducible from the arrays alone.
# ===========================================================================


def _unlabelled_second_group(n_per_group: int = 120, order: str = "AB"):
    """Group A labelled, group B carrying no ground truth at all.

    Both groups clear the 30-row size gate, so this is NOT the
    fewer-than-two-groups path: two groups are compared and one of them has
    nothing to contribute. ``order`` swaps which group the data presents first,
    which is what the old ``max()``/``min()`` verdict depended on.
    """
    labels_a = np.tile(np.array([1.0, 0.0, 1.0, 1.0, 0.0]), n_per_group // 5)
    scores = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), n_per_group // 5)
    blank = np.full(n_per_group, np.nan)
    if order == "AB":
        y = np.concatenate([labels_a, blank])
        g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    else:
        y = np.concatenate([blank, labels_a])
        g = np.array(["B"] * n_per_group + ["A"] * n_per_group)
    return y, np.concatenate([scores, scores]), g


def _both_groups_labelled(n_per_group: int = 120):
    """The over-correction control: the same shape and scores, B labelled too.

    Base rates 0.6 and 0.2, so there is a real measured gap and every refusal
    below has to give way to a number.
    """
    labels_a = np.tile(np.array([1.0, 0.0, 1.0, 1.0, 0.0]), n_per_group // 5)
    labels_b = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), n_per_group // 5)
    scores = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), n_per_group // 5)
    return (
        np.concatenate([labels_a, labels_b]),
        np.concatenate([scores, scores]),
        np.array(["A"] * n_per_group + ["B"] * n_per_group),
    )


def _calibrated_block(n_per_level: int, levels=(0.2, 0.5, 0.8)):
    """A deterministic, essentially perfectly calibrated block of rows."""
    p, y = [], []
    for level in levels:
        positives = int(round(level * n_per_level))
        p.extend([level] * n_per_level)
        y.extend([1] * positives + [0] * (n_per_level - positives))
    return np.array(y), np.array(p)


def _two_clean_groups_and_a_small_awful_third():
    """180 'a' + 180 'b' essentially perfect, 15 'c' scored 0.9 against 0.

    'c' is below the default gate of 30 and above a gate of 10, so the SAME data
    read at two gates says two different things, and the group left out is the
    one that decides it.
    """
    y_a, p_a = _calibrated_block(60)
    y_b, p_b = _calibrated_block(60)
    n_c = 15
    return (
        np.concatenate([y_a, y_b, np.zeros(n_c, int)]),
        np.concatenate([p_a, p_b, np.full(n_c, 0.9)]),
        np.array(["a"] * len(y_a) + ["b"] * len(y_b) + ["c"] * n_c),
    )


def _two_clean_groups_and_a_thinly_spread_third():
    """The EVIDENCE gate, which no row count can see.

    40 'd' rows spread one per prediction bin, so 'd' clears min_group_size and
    no (group x bin) cell reaches min_cell: it contributes no calibration
    evidence at all while ``excluded_groups`` stays empty.
    """
    y_a, p_a = _calibrated_block(60)
    y_b, p_b = _calibrated_block(60)
    n_d = 40
    return (
        np.concatenate([y_a, y_b, np.zeros(n_d, int)]),
        np.concatenate([p_a, p_b, np.linspace(0.01, 0.99, n_d)]),
        np.array(["a"] * len(y_a) + ["b"] * len(y_b) + ["d"] * n_d),
    )


# ===========================================================================
# G03-1 / G03-2. The disparity that max()/min() reported as perfect parity,
# and the verdict beside it.
# ===========================================================================


class TestTheDisparityKnowsWhichGroupsItMeasured:
    def test_an_unmeasured_group_makes_the_disparity_a_refusal_not_a_zero(self):
        """MEASURED BEFORE THE FIX on this exact input:

        group_ece        {'A': 0.58, 'B': nan}
        group_mce        {'A': 0.9,  'B': 0.0}      <- a fabricated 0.0
        excluded_groups  []
        ece_disparity    0.0                        <- perfect parity
        has_significant_disparity                False
        recommendations[0]  'LOW: Calibration disparity is acceptable (0.000).'
        warnings         []
        """
        y, p, g = _unlabelled_second_group()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_disparity(y, p, g, min_group_size=30)

        # The premise, asserted rather than assumed: B is NOT size-excluded and
        # its own ECE really is unmeasurable, so this is the second door.
        assert list(result.excluded_groups) == [], result.excluded_groups
        assert not np.isfinite(result.group_ece["B"])
        assert result.group_ece["A"] > 0.5, result.group_ece

        # THE FABRICATION IS GONE, in the number and in the verdict.
        assert not np.isfinite(result.ece_disparity), result.ece_disparity
        assert result.has_significant_disparity is None
        assert result.to_dict()["has_significant_disparity"] is None
        assert result.groups_without_measured_calibration == ["B"]
        assert result.to_dict()["groups_without_measured_calibration"] == ["B"]

        # An all-clear sentence may not be written over it, and the disclosure
        # has to be audible.
        assert not any("acceptable" in r for r in result.recommendations), result.recommendations
        assert any("MEASURED calibration error" in m for m in _messages(caught)), _messages(caught)

    def test_the_answer_does_not_depend_on_which_group_came_first(self):
        """``max``/``min`` keep the running extreme, so which way they fell was
        decided by dict insertion order, i.e. by the order the groups happened to
        appear in the data. Measured before the fix: ece_disparity 0.0 with A
        first and nan with B first, from the same 240 rows.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_first = calibration_disparity(*_unlabelled_second_group(order="AB"))
            b_first = calibration_disparity(*_unlabelled_second_group(order="BA"))

        assert not np.isfinite(a_first.ece_disparity)
        assert not np.isfinite(b_first.ece_disparity)
        assert a_first.has_significant_disparity is b_first.has_significant_disparity is None

    def test_the_named_extremes_are_not_the_group_nobody_measured(self):
        """``max(group_ece, key=...)`` compares raw values and every comparison
        against a NaN is False, so before the fix BOTH extremes fell to whichever
        group arrived first: 'A' was named best AND worst calibrated on
        {'A': 0.58, 'B': nan}, and with the rows the other way round both came
        back 'B', the group nobody measured.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b_first = calibration_disparity(*_unlabelled_second_group(order="BA"))
        assert b_first.most_miscalibrated_group != "B", b_first.most_miscalibrated_group
        assert b_first.least_miscalibrated_group != "B", b_first.least_miscalibrated_group

    def test_control_a_complete_comparison_still_reports_its_number(self):
        """OVER-CORRECTION CONTROL. Both groups labelled, base rates 0.6 and
        0.2: the disparity must be a real finite number, the verdict a bool, the
        coverage list empty, and nothing may warn about coverage.
        """
        y, p, g = _both_groups_labelled()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calibration_disparity(y, p, g, min_group_size=30)

        assert np.isfinite(result.ece_disparity)
        eces = list(result.group_ece.values())
        assert result.ece_disparity == pytest.approx(max(eces) - min(eces))
        assert isinstance(result.has_significant_disparity, bool)
        assert result.groups_without_measured_calibration == []
        assert result.most_miscalibrated_group != result.least_miscalibrated_group
        assert not any("MEASURED calibration error" in m for m in _messages(caught))
        assert not any("NOT ASSESSED" in r for r in result.recommendations), result.recommendations

    def test_a_clean_spread_over_two_of_three_groups_is_a_refusal(self):
        """THE REACHABLE DOOR for the coverage clause in the verdict property.

        With only TWO groups and one of them unmeasured the disparity itself is
        NaN, so the verdict is already None from the finiteness guard and the
        coverage clause never runs: a sabotage of that clause stays GREEN on the
        two-group input above, and was reported as such rather than counted.

        Three groups is where it bites. 'A' and 'B' are both essentially
        perfectly calibrated, so ``ece_disparity`` is a FINITE number well under
        0.05 with ``excluded_groups []``, and before the fix the object published
        that as ``has_significant_disparity False``: a clean bill of health over
        a comparison that covered two of the three groups on the axis, where the
        third could have decided it either way.
        """
        y_a, p_a = _calibrated_block(20)
        y_b, p_b = _calibrated_block(20)
        n_c = 60
        y = np.concatenate([y_a.astype(float), y_b.astype(float), np.full(n_c, np.nan)])
        p = np.concatenate([p_a, p_b, np.tile(np.array([0.1, 0.5, 0.9]), n_c // 3)])
        g = np.array(["A"] * len(y_a) + ["B"] * len(y_b) + ["C"] * n_c)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibration_disparity(y, p, g, min_group_size=30)

        # The premise: a finite, clean spread, no size exclusion, one uncovered.
        assert list(result.excluded_groups) == [], result.excluded_groups
        assert result.groups_without_measured_calibration == ["C"]
        assert np.isfinite(result.ece_disparity), result.ece_disparity
        assert result.ece_disparity <= 0.05, result.ece_disparity

        assert result.has_significant_disparity is None
        assert result.to_dict()["has_significant_disparity"] is None
        assert any("NOT ASSESSED" in r and "C" in r for r in result.recommendations), (
            result.recommendations
        )
        assert not any(
            r.startswith("LOW: Calibration disparity is acceptable (") and "excluded" not in r
            for r in result.recommendations
        ), result.recommendations

    def test_control_a_measured_breach_on_partial_evidence_keeps_its_finding(self):
        """The reverse fabrication. A spread already over the threshold cannot be
        argued away by a group nobody measured, so a True must survive.
        """
        # Three groups: two measured with a real 0.4-ish gap, one unmeasurable.
        y, p, g = _both_groups_labelled(n_per_group=120)
        y = np.concatenate([y, np.full(60, np.nan)])
        p = np.concatenate([p, np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 12)])
        g = np.concatenate([g, np.array(["C"] * 60)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calibration_disparity(y, p, g, min_group_size=30)

        assert result.groups_without_measured_calibration == ["C"]
        assert np.isfinite(result.ece_disparity), "two groups were measured: keep the bound"
        assert result.ece_disparity > 0.05
        assert result.has_significant_disparity is True, "a determinate breach must stand"


# ===========================================================================
# G03-3. The canonical measured test, and the six doors absence arrives through.
# ===========================================================================

_ABSENCE_DOORS = {
    "None": None,
    "float nan": float("nan"),
    "np.float32 nan": np.float32("nan"),
    "blank string": "",
    "literal 'None'": "None",
    "pd.NA": pd.NA,
    "pd.NaT": pd.NaT,
    "bool True": True,
    "bool False": False,
    "numeric string": "0.9",
}


class TestTheMetricResultUsesTheCanonicalMeasuredTest:
    @pytest.mark.parametrize("door", sorted(_ABSENCE_DOORS))
    def test_no_door_crashes_the_serialiser_or_mints_a_disparity(self, door):
        """MEASURED BEFORE THE FIX, with {'a': 0.02, 'b': <door>}:

        blank string / literal 'None'  ValueError out of float(), from to_dict()
        pd.NA / pd.NaT                 TypeError out of float(), from to_dict()
        bool True                      n_groups_compared 2, disparity 0.98
        numeric string '0.9'           n_groups_compared 2, then TypeError
                                       comparing str to float in the ARITHMETIC

        The count coerced where the arithmetic could not, so the two halves of
        one object disagreed about whether the same value was a measurement.
        """
        result = CalibrationMetricResult(
            "expected_calibration_error", 0.04, {"a": 0.02, "b": _ABSENCE_DOORS[door]}, 10, 10
        )
        payload = result.to_dict()  # must not raise on any door
        assert payload["n_groups_compared"] == 1, door
        assert not math.isfinite(payload["max_group_disparity"]), door
        assert not math.isfinite(payload["measured_subset_disparity"]), door
        assert result.metric_name == "expected_calibration_error"

    def test_control_two_real_values_still_give_the_real_gap(self):
        """OVER-CORRECTION CONTROL. A guard that calls everything unmeasured
        passes every test above and deletes the class.
        """
        result = CalibrationMetricResult(
            "expected_calibration_error", 0.04, {"a": 0.02, "b": 0.09}, 100, 10
        )
        payload = result.to_dict()
        assert payload["n_groups_compared"] == 2
        assert payload["max_group_disparity"] == pytest.approx(0.07)
        assert payload["measured_subset_disparity"] == pytest.approx(0.07)
        assert payload["is_well_calibrated"] is True

    def test_control_a_finite_numpy_scalar_is_a_measurement(self):
        """The dangerous direction: a real measurement reported as a
        could-not-check. ``np.float32`` is not a Python float subclass, so an
        ``isinstance(v, (int, float))`` test would have discarded these.
        """
        result = CalibrationMetricResult(
            "expected_calibration_error",
            np.float32(0.04),
            {"a": np.float32(0.02), "b": np.int64(0), "c": np.float64(0.09)},
            100,
            10,
        )
        payload = result.to_dict()
        assert payload["n_groups_compared"] == 3
        assert payload["max_group_disparity"] == pytest.approx(0.09, abs=1e-6)

    def test_control_the_partial_evidence_is_kept_when_two_groups_remain(self):
        """A LOWER BOUND is still a finding. With three groups and one NaN the
        0.8 gap between the other two is determinate, and dropping it would be
        the reverse fabrication.
        """
        result = CalibrationMetricResult(
            "expected_calibration_error", 0.4, {"a": 0.1, "b": 0.9, "c": float("nan")}, 300, 10
        )
        payload = result.to_dict()
        assert not math.isfinite(payload["max_group_disparity"]), "the whole is unknown"
        assert payload["measured_subset_disparity"] == pytest.approx(0.8), "the part is not"
        assert payload["n_groups_compared"] == 2


# ===========================================================================
# G03-4 / G03-5. Two more between-group verdicts that were clean over a subset.
# ===========================================================================


class TestTheIciVerdictKnowsWhatItCovered:
    def test_a_clean_ici_spread_over_a_subset_is_a_refusal(self):
        """MEASURED BEFORE THE FIX: excluded_groups ['c'], group_ici
        {'a': 0.00024, 'b': 0.00024}, ici_disparity 0.0,
        has_significant_disparity False, and to_dict() serialised that False.
        """
        y, p, g = _two_clean_groups_and_a_small_awful_third()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            partial = integrated_calibration_index(y, p, g, min_group_size=30)

        assert partial.excluded_groups == ["c"], partial.excluded_groups
        assert np.isfinite(partial.ici_disparity) and partial.ici_disparity < 0.05
        assert partial.has_significant_disparity is None
        assert partial.to_dict()["has_significant_disparity"] is None
        assert partial.uncovered_groups == ["c"]
        assert partial.to_dict()["uncovered_groups"] == ["c"]

    def test_the_excluded_group_is_the_one_that_decides_it(self):
        """The same rows at a gate that lets 'c' in. If this stopped flipping the
        verdict, the refusal above would be about nothing."""
        y, p, g = _two_clean_groups_and_a_small_awful_third()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            whole = integrated_calibration_index(y, p, g, min_group_size=10)

        assert whole.excluded_groups == []
        assert whole.ici_disparity > 0.05
        assert whole.has_significant_disparity is True

    def test_control_a_complete_comparison_is_still_graded_both_ways(self):
        """OVER-CORRECTION CONTROL, both polarities, nothing excluded."""
        y_clean, p_clean = _calibrated_block(60)
        g_clean = np.array(["a", "b"] * (len(y_clean) // 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = integrated_calibration_index(y_clean, p_clean, g_clean, min_group_size=30)
        assert clean.excluded_groups == [] and clean.uncovered_groups == []
        assert clean.has_significant_disparity is False

        y_bad, p_bad = _calibrated_block(60)
        g_bad = np.array(["a"] * (len(y_bad) // 2) + ["b"] * (len(y_bad) - len(y_bad) // 2))
        p_bad = p_bad.copy()
        p_bad[g_bad == "b"] = 0.95
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bad = integrated_calibration_index(y_bad, p_bad, g_bad, min_group_size=30)
        assert bad.uncovered_groups == []
        assert bad.has_significant_disparity is True


class TestTheMulticalibrationVerdictKnowsWhatItAudited:
    def test_a_clean_alpha_over_a_size_excluded_group_is_a_refusal(self):
        """MEASURED BEFORE THE FIX: excluded_groups ['c'], group_max
        {'a': 0.0, 'b': 0.0}, alpha 3.33e-16, is_multicalibrated True.
        """
        y, p, g = _two_clean_groups_and_a_small_awful_third()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            partial = multicalibration(y, p, g, n_bins=10, min_group_size=30, min_cell=10)
            whole = multicalibration(y, p, g, n_bins=10, min_group_size=10, min_cell=10)

        assert partial.excluded_groups == ["c"]
        assert partial.alpha < 0.05, "the audit really did find nothing on the subset"
        assert partial.is_multicalibrated is None
        assert partial.to_dict()["is_multicalibrated"] is None
        # and the group left out is the one that decides it
        assert whole.alpha > 0.05 and whole.is_multicalibrated is False

    def test_a_group_with_no_evaluated_cell_is_the_door_a_count_gate_misses(self):
        """MEASURED BEFORE THE FIX, on 40 rows spread one per bin beside two
        clean 180-row groups: excluded_groups [] (it cleared the size gate),
        group_max {'a': 0.0, 'b': 0.0, 'd': nan}, alpha 3.33e-16,
        is_multicalibrated True, and NOT ONE WARNING. A guard keyed on
        ``excluded_groups`` alone passes this input.
        """
        y, p, g = _two_clean_groups_and_a_thinly_spread_third()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = multicalibration(y, p, g, n_bins=20, min_group_size=30, min_cell=10)

        assert result.excluded_groups == [], "the premise: no count gate fired"
        assert not np.isfinite(result.group_max["d"]), result.group_max
        assert result.alpha < 0.05, "the premise: the evaluated cells really were clean"
        assert result.is_multicalibrated is None
        assert result.to_dict()["is_multicalibrated"] is None
        assert result.uncovered_groups == ["d"]
        assert result.to_dict()["uncovered_groups"] == ["d"]
        assert any("NO evaluated" in m for m in _messages(caught)), _messages(caught)

    def test_control_a_full_audit_is_still_graded_both_ways(self):
        """OVER-CORRECTION CONTROL. Every group inside the gate with evaluated
        cells: a clean audit must still say True and a violation must still say
        False.
        """
        y_clean, p_clean = _calibrated_block(120)
        g_clean = np.array(["a", "b"] * (len(y_clean) // 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean = multicalibration(y_clean, p_clean, g_clean, n_bins=5, min_cell=10)
        assert clean.uncovered_groups == []
        assert clean.is_multicalibrated is True
        assert not any("NO evaluated" in m for m in _messages(caught))

        p_bad = p_clean.copy()
        p_bad[g_clean == "b"] = 0.95
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bad = multicalibration(y_clean, p_bad, g_clean, n_bins=5, min_cell=10)
        assert bad.alpha > 0.05
        assert bad.is_multicalibrated is False, "a real violation must survive any coverage gate"


# ===========================================================================
# G03-6. The Pareto chart that marked an unrankable point optimal.
# ===========================================================================

_CAL = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.09, 0.13])
_FAIR = np.array([0.20, 0.12, 0.08, 0.06, 0.05, 0.18, 0.19])


def _pareto_png(cal, fair, **kwargs) -> str:
    """sha256 of one rendering, on a FRESH figure that is closed again."""
    figure, axes = plt.subplots(figsize=(6, 4))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal_plots.plot_pareto_frontier(
                calibration_errors=cal, fairness_violations=fair, ax=axes, **kwargs
            )
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=60)
    finally:
        plt.close(figure)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _frontier_offsets(cal, fair):
    """The points actually drawn in each labelled scatter collection."""
    figure, axes = plt.subplots(figsize=(6, 4))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal_plots.plot_pareto_frontier(
                calibration_errors=cal, fairness_violations=fair, ax=axes
            )
        drawn = {
            c.get_label(): [tuple(xy) for xy in c.get_offsets().tolist()] for c in axes.collections
        }
        return drawn, _figure_texts(axes), _messages(caught)
    finally:
        plt.close(figure)


class TestTheParetoChartRanksOnlyWhatItCanCompare:
    @needs_matplotlib
    def test_an_unrankable_point_is_not_drawn_as_pareto_optimal(self):
        """MEASURED BEFORE THE FIX, with the fifth fairness violation set to NaN:
        the "Pareto Optimal" collection was built INCLUDING (0.15, nan), and the
        call then died at the very end with matplotlib's bare
        ``ValueError: Axis limits cannot be NaN or Inf`` out of set_ylim.
        """
        fair = _FAIR.copy()
        fair[4] = np.nan
        drawn, texts, messages = _frontier_offsets(_CAL, fair)

        optimal = drawn.get("Pareto Optimal", [])
        dominated = drawn.get("Dominated Points", [])
        assert (0.15, np.nan) not in optimal
        assert not any(math.isnan(y) for _, y in optimal), optimal
        assert not any(math.isnan(y) for _, y in dominated), (
            "'Dominated' is a finding too: it says something beats it in both"
        )
        assert any("Not ranked (1 of 7)" in t for t in texts), texts
        assert any("no dominance comparison" in m for m in messages), messages

    @needs_matplotlib
    def test_every_point_unrankable_does_not_die_inside_matplotlib(self):
        """The 2026-09-27 input class: one group with no negative labels, so the
        whole fairness axis is NaN. Before the fix every point came back on the
        frontier and then the call raised on the axis limits.
        """
        drawn, texts, messages = _frontier_offsets(_CAL, np.full(len(_CAL), np.nan))
        assert drawn.get("Pareto Optimal", []) == []
        assert any("No trade-off point could be ranked" in t for t in texts), texts
        assert any("NOTHING could be ranked" in m for m in messages), messages

    @needs_matplotlib
    def test_control_the_real_frontier_is_unchanged(self):
        """OVER-CORRECTION CONTROL, on fully measured points: the frontier must
        be exactly the non-dominated set, and the other three points must be
        drawn as dominated.
        """
        drawn, texts, messages = _frontier_offsets(_CAL, _FAIR)
        optimal = set(drawn["Pareto Optimal"])
        dominated = set(drawn["Dominated Points"])
        assert optimal == {(0.05, 0.20), (0.08, 0.12), (0.10, 0.08), (0.12, 0.06), (0.15, 0.05)}
        assert dominated == {(0.09, 0.18), (0.13, 0.19)}
        assert not any("Not ranked" in t for t in texts), texts
        assert messages == [], messages

    @needs_matplotlib
    def test_three_states_render_three_different_images(self):
        """A drawing is only pinned by its own bytes. Three DISTINCT states:
        fully measured, one point unrankable, nothing rankable.
        """
        one_nan = _FAIR.copy()
        one_nan[4] = np.nan
        measured = _pareto_png(_CAL, _FAIR)
        partial = _pareto_png(_CAL, one_nan)
        none_at_all = _pareto_png(_CAL, np.full(len(_CAL), np.nan))
        assert len({measured, partial, none_at_all}) == 3, (
            measured[:12],
            partial[:12],
            none_at_all[:12],
        )

    @needs_matplotlib
    def test_control_the_raster_harness_is_not_vacuous(self):
        """If the same call twice differed, or a wired parameter did not change
        the bytes, the hash pin above would prove nothing.
        """
        assert _pareto_png(_CAL, _FAIR) == _pareto_png(_CAL, _FAIR)
        assert _pareto_png(_CAL, _FAIR, frontier_color="#00ff00") != _pareto_png(_CAL, _FAIR)


# ===========================================================================
# G03-7. The protected-attribute level that IS a missing value.
# ===========================================================================


def _scores_and_labels(n: int = 200):
    scores = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), n // 5)
    labels = np.tile(np.array([1, 0, 1, 1, 0]), n // 5)
    return labels, scores


class TestTheGroupChartCountsAMissingLevelHonestly:
    @needs_matplotlib
    def test_a_missing_level_states_its_real_row_count(self):
        """MEASURED BEFORE THE FIX, on 100 zeros beside 100 NaNs in a float
        attribute: the chart read "Not measured (n<10): nan (n=0)" and the
        warning said "fewer than 10 samples", for a level holding HALF the rows,
        while those rows stayed inside the pooled "Overall" curve.
        """
        y, p = _scores_and_labels(200)
        attr = np.concatenate([np.zeros(100), np.full(100, np.nan)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_group_calibration(y, p, attr)

        texts = _figure_texts(axes)
        assert any("Missing attribute (not a group)" in t and "n=100" in t for t in texts), texts
        assert not any("n=0" in t for t in texts), texts
        assert any("ARE a missing value, not a group" in m for m in _messages(caught))
        assert not any(f"fewer than {MIN_GROUP_CURVE} samples" in m for m in _messages(caught))
        # the measured level still gets its curve
        assert any("0.0 (ECE=" in t for t in texts), texts

    @needs_matplotlib
    def test_an_object_dtype_attribute_with_one_missing_cell_does_not_crash(self):
        """Before the fix ``np.unique`` sorted a float NaN against strings and
        raised ``TypeError: '<' not supported between instances of 'float' and
        'str'``: a bare crash on an ordinary column with one blank cell.
        """
        y, p = _scores_and_labels(200)
        attr = np.array(["a"] * 100 + [np.nan] * 100, dtype=object)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_group_calibration(y, p, attr)
        texts = _figure_texts(axes)
        assert any("Missing attribute" in t and "n=100" in t for t in texts), texts
        assert any("a (ECE=" in t for t in texts), texts
        assert any("ARE a missing value" in m for m in _messages(caught))

    @pytest.mark.parametrize("blank", ["", "None", "nan", "<NA>"])
    @needs_matplotlib
    def test_the_string_spellings_of_absence_are_not_demographic_groups(self, blank):
        """``str(x)`` on an absent value MINTS content, and a CSV read is where
        these spellings come from. ``pd.NA`` once minted a group named '<NA>'
        elsewhere in this repository.
        """
        y, p = _scores_and_labels(200)
        attr = np.array(["a"] * 100 + [blank] * 100, dtype=object)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_group_calibration(y, p, attr)
        assert any("Missing attribute" in t for t in _figure_texts(axes)), _figure_texts(axes)
        assert any("ARE a missing value" in m for m in _messages(caught))

    @needs_matplotlib
    def test_control_two_real_groups_are_drawn_and_nothing_warns(self):
        """OVER-CORRECTION CONTROL: a chart that cries "missing" for data it
        measured is worse than one that says nothing, and a reader who sees the
        box every time stops reading it.
        """
        y, p = _scores_and_labels(200)
        attr = np.array(["a"] * 100 + ["b"] * 100)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_group_calibration(y, p, attr)
        texts = _figure_texts(axes)
        assert any("a (ECE=" in t for t in texts) and any("b (ECE=" in t for t in texts), texts
        assert not any("Missing attribute" in t or "Not measured" in t for t in texts), texts
        assert _messages(caught) == [], _messages(caught)

    @needs_matplotlib
    def test_control_a_genuinely_small_group_is_still_called_small(self):
        """The two disclosures must stay distinguishable: one needs more rows,
        the other needs a value in the column."""
        y, p = _scores_and_labels(200)
        attr = np.array(["a"] * 195 + ["b"] * 5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_group_calibration(y, p, attr)
        texts = _figure_texts(axes)
        assert any(f"Not measured (n<{MIN_GROUP_CURVE}): b (n=5)" in t for t in texts), texts
        assert not any("Missing attribute" in t for t in texts), texts
        assert any("no calibration curve was drawn" in m for m in _messages(caught))


# ===========================================================================
# The reliability diagram: the one number this chart writes on the canvas.
# ===========================================================================


class TestTheReliabilityDiagramCaption:
    @needs_matplotlib
    def test_zero_rows_is_not_captioned_as_perfect_calibration(self):
        y, p = np.array([]), np.array([])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_reliability_diagram(y, p)
        caption = [t for t in _figure_texts(axes) if t.startswith("ECE")]
        assert caption and "0.000" not in caption[0], caption
        assert "nan" in caption[0].lower(), caption
        assert any("would read as perfect calibration" in m for m in _messages(caught))

    @needs_matplotlib
    def test_control_a_measurable_ece_is_still_a_number(self):
        """DERIVED, not quoted: the caption must be the ECE the metric computes
        on this data, whatever that number turns out to be."""
        from vfairness.post_processing.calibration.metrics import calibration_curve

        y, p = _scores_and_labels(200)
        expected = calibration_curve(y, p).overall_ece
        assert np.isfinite(expected), "the control needs a measurable case"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            axes = cal_plots.plot_reliability_diagram(y, p)
        caption = [t for t in _figure_texts(axes) if t.startswith("ECE")]
        assert caption == [f"ECE = {expected:.3f}"], caption
        assert _messages(caught) == [], _messages(caught)


# ===========================================================================
# The report summary's verdict line.
# ===========================================================================


def _report(diagnosis, base_rate_disparity=0.02) -> CalibrationReport:
    disparity = CalibrationDisparityResult(
        0.02,
        0.03,
        0.01,
        {"a": 0.02, "b": 0.04},
        {"a": 0.03, "b": 0.05},
        {"a": 0.10, "b": 0.11},
        "b",
        "a",
        2,
    )
    tradeoff = TradeoffAnalysisResult(
        [], ParetoPoint(0.04, 0.02), diagnosis, [], base_rate_disparity
    )
    return CalibrationReport(
        timestamp="t",
        data_info={"n_samples": 240},
        protected_attribute="g",
        n_groups=2,
        overall_metrics={"ece": 0.04, "mce": 0.06, "brier": 0.12},
        group_metrics={},
        disparity_analysis=disparity,
        brier_decomposition=BrierDecomposition(0.2774, 0.0, 0.0, 0.0, 200, 10),
        tradeoff_analysis=tradeoff,
        recommendation=None,
        is_well_calibrated=True,
        has_significant_disparity=False,
        critical_issues=[],
        recommendations=[],
    )


class TestTheSummaryRendersEveryVerdictTheSameWay:
    def test_an_absent_impossibility_verdict_is_not_a_no(self):
        """MEASURED BEFORE THE FIX: with ``impossibility_diagnosis={}`` the line
        read "Impossibility Applies: False", a verdict on a theorem nobody
        applied, from ``.get('impossibility_applies', False)``.
        """
        line = [x for x in _report({}).summary().splitlines() if "Impossibility" in x]
        assert line == ["Impossibility Applies: Not assessable (not measured)"], line

    def test_a_three_state_none_reads_like_every_other_three_state(self):
        """``impossibility_diagnostics`` returns None for an unmeasurable base
        rate comparison, and this line printed the bare token "None" while every
        sibling verdict in the same method printed "Not assessable".
        """
        summary = _report({"impossibility_applies": None}, float("nan")).summary()
        assert "Impossibility Applies: Not assessable (not measured)" in summary
        assert "Impossibility Applies: None" not in summary
        assert "Base Rate Disparity: nan%" not in summary
        assert "Base Rate Disparity: not measured (undefined on this data)" in summary

    def test_an_undefined_skill_score_is_named_not_printed_as_a_sentinel(self):
        """``skill_score`` is deliberately NaN when every label is one class:
        there is no climatology to be more skilful than."""
        summary = _report({"impossibility_applies": True}).summary()
        assert "Skill Score: nan" not in summary
        assert "Skill Score: not measured (undefined on this data)" in summary

    def test_control_both_real_verdicts_still_read_yes_and_no(self):
        """OVER-CORRECTION CONTROL. A renderer that answers "not assessable" to
        everything passes all three tests above."""
        assert "Impossibility Applies: Yes" in _report({"impossibility_applies": True}).summary()
        assert "Impossibility Applies: No" in _report({"impossibility_applies": False}).summary()
        assert "Base Rate Disparity: 2.0%" in _report({"impossibility_applies": True}).summary()


# ===========================================================================
# NOT A MEASUREMENT. Executed, and checked not to mint a value.
# ===========================================================================


class TestTheContainersDoNotMintAValue:
    def test_the_warning_categories_carry_no_verdict_and_can_be_escalated(self):
        for category, base in (
            (IntersectionalProvenanceWarning, UserWarning),
            (UnmeasurableGroupWarning, UserWarning),
            (UnknownGroupWarning, RuntimeWarning),
        ):
            assert issubclass(category, base), category
            assert category.__doc__, f"{category.__name__} must say what it discloses"
            # no state of its own: it cannot carry a fabricated number
            assert not [a for a in vars(category) if not a.startswith("__")], vars(category)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warnings.warn("probe", category)
            assert caught[0].category is category
            with warnings.catch_warnings():
                warnings.simplefilter("error", category)
                with pytest.raises(category):
                    warnings.warn("probe", category)

    def test_the_base_calibrator_refuses_transform_before_fit(self):
        """``BaseCalibrator`` is abstract and its concrete gate is shared: the
        empty-data and non-finite-weight refusals are stated ONCE, above the
        dispatch, so no subclass can escape them."""
        with pytest.raises(TypeError):
            BaseCalibrator()  # abstract: fit and transform are not implemented

        calibrator = PlattScaling()
        assert calibrator.is_fitted is False
        assert calibrator.fit_result is None
        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.transform(np.array([0.5]))

        with pytest.raises(ValueError, match="0 samples"):
            calibrator.fit(np.array([]), np.array([]))
        assert calibrator.is_fitted is False, "a refused fit may not report itself fitted"

        with pytest.raises(ValueError, match="non-finite"):
            calibrator.fit(
                np.array([0, 1, 0, 1]),
                np.array([0.1, 0.9, 0.2, 0.8]),
                sample_weight=np.array([1.0, np.nan, 1.0, 1.0]),
            )

    def test_the_single_class_note_survives_on_the_record_not_only_on_stderr(self):
        """A warning is discarded by any caller that does not capture warnings."""
        calibrator = PlattScaling()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibrator.fit(np.zeros(40, int), np.linspace(0.05, 0.95, 40))
        assert calibrator.fit_result is not None
        assert calibrator.fit_result.warnings, calibrator.fit_result.to_dict()
        assert any("SINGLE class" in w for w in calibrator.fit_result.warnings)
        assert any("SINGLE class" in m for m in _messages(caught))

    def test_the_fit_result_serialiser_is_a_faithful_copy(self):
        result = CalibrationFitResult("platt", 60, {"a": 1.0}, {"log_loss": 0.4}, ["note"])
        assert result.to_dict() == {
            "method": "platt",
            "n_samples": 60,
            "parameters": {"a": 1.0},
            "fit_metrics": {"log_loss": 0.4},
            "warnings": ["note"],
        }
        assert CalibrationFitResult("platt", 0).to_dict()["warnings"] == []

    def test_the_curve_container_serialises_an_empty_curve_as_empty(self):
        empty = CalibrationCurveResult(
            np.array([]), np.array([]), np.array([]), np.linspace(0, 1, 11), float("nan")
        )
        payload = empty.to_dict()
        assert payload["prob_true"] == [] and payload["bin_counts"] == []
        assert not math.isfinite(payload["overall_ece"]), "0.0 would be perfect calibration"

        real = CalibrationCurveResult(
            np.array([0.1, 0.9]),
            np.array([0.15, 0.85]),
            np.array([50, 50]),
            np.linspace(0, 1, 3),
            0.05,
        )
        assert real.to_dict()["overall_ece"] == 0.05
        assert real.to_dict()["bin_counts"] == [50, 50]

    def test_an_undefined_brier_skill_score_is_nan_not_zero(self):
        """0.0 is a REAL attainable score ("exactly as skilful as climatology"),
        so it cannot double as the refusal."""
        single_class = BrierDecomposition(0.2774, 0.0, 0.0, 0.0, 200, 10)
        assert not math.isfinite(single_class.to_dict()["skill_score"])
        healthy = BrierDecomposition(0.2, 0.01, 0.05, 0.24, 100, 10)
        assert healthy.to_dict()["skill_score"] == pytest.approx(1 - 0.2 / 0.24)

    def test_a_fresh_pareto_point_claims_nothing(self):
        point = ParetoPoint(0.1, 0.2)
        assert point.is_pareto_optimal is None, "nothing has been compared yet"
        assert point.is_rankable is True
        unrankable = ParetoPoint(0.1, float("nan"))
        assert unrankable.is_rankable is False
        assert unrankable.dominates(point) is None and point.dominates(unrankable) is None

    def test_the_group_calibration_result_carries_its_own_caveat(self):
        """An EMPTY ``unmeasurable_groups`` means every group got a real varying
        map, so the field has to be serialised or the disclosure exists only for
        a caller holding the dataclass."""
        result = GroupCalibrationResult(
            "isotonic",
            2,
            ["a", "b"],
            {"a": 0.1, "b": float("nan")},
            {"a": 0.02, "b": 0.0},
            {"a": 0.08, "b": float("nan")},
            float("nan"),
            unmeasurable_groups={"b": "non_finite_output"},
        )
        payload = result.to_dict()
        assert payload["unmeasurable_groups"] == {"b": "non_finite_output"}
        assert not math.isfinite(payload["overall_improvement"])
        clean = GroupCalibrationResult(
            "isotonic",
            2,
            ["a", "b"],
            {"a": 0.1, "b": 0.2},
            {"a": 0.02, "b": 0.03},
            {"a": 0.08, "b": 0.17},
            0.12,
        )
        assert clean.to_dict()["unmeasurable_groups"] == {}

    def test_the_recommendation_serialiser_keeps_the_coverage_caveat(self):
        recommendation = CalibrationRecommendation(
            "monitor_only", "low", "ok", None, None, ["c"], False
        )
        payload = recommendation.to_dict()
        assert payload["not_assessed_groups"] == ["c"]
        assert payload["disparity_assessed"] is False

    def test_the_tradeoff_serialiser_reports_not_assessed_rather_than_minimal(self):
        unmeasured = TradeoffAnalysisResult(
            [], ParetoPoint(0.04, float("nan")), {}, [], float("nan")
        )
        payload = unmeasured.to_dict()
        assert payload["tradeoff_severity"] == "not assessed"
        assert not math.isfinite(payload["base_rate_disparity"])
        assert payload["rate_label_counts"] == {} and payload["groups_without_measured_rate"] == []
        measured = TradeoffAnalysisResult([], ParetoPoint(0.04, 0.02), {}, [], 0.2)
        assert measured.to_dict()["tradeoff_severity"] == "severe"
