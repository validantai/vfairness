"""BGL grade-1, batch G01: the 60 never-graded units of ``vfairness_metrics``.

Every test here was written from an EXECUTION, on healthy input and on the
degenerate input where the thing the unit claims to measure does not exist. The
five defects it pins were all found the same way, and each one is the same
mechanism seen from a different side: a value that could not be measured,
published where a reader takes it for one.

  1. ``AttributionResult.top`` ranked a NaN importance FIRST, so the one feature
     the library had explicitly refused to measure was reported as the single
     strongest driver of the model.
  2. ``compute_max_ratio`` returned its untouched ``1.0`` initialiser, perfect
     parity, when no pair of groups could be divided at all, including for a
     group that is never selected.
  3. ``GroupManager`` minted an intersectional group out of ``pd.NA`` / ``None``
     / ``nan`` in silence, while its own sibling path warns about the same rows.
  4. ``GroupManager``'s size floor let a non-measurement threshold past an order
     comparison, so every group fell out of BOTH the valid and the invalid list
     and ``get_invalid_groups`` returned its vacuous ``[]`` without a word.
  5. ``MetricExplanation`` published ``np.float32(nan)``, ``<NA>``, ``NaT`` and
     ``inf`` as the card's value, directly under an ``evaluation`` string that
     said the metric had never been computed.

and one found next door while proving (3):

  6. ``missing_strategy='as_group'`` was SILENTLY IGNORED for an intersectional
     sensitive attribute: half the dataset was dropped under the one strategy
     that exists to avoid dropping it.

Every refusal test has a CONTROL beside it asserting the healthy case's real
number, because a guard that refuses everything passes every refusal test and
destroys the unit.
"""

from __future__ import annotations

import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics import _metric_direction as MD
from vfairness.evaluation.vfairness_metrics import _statistics as ST
from vfairness.evaluation.vfairness_metrics import counterfactual_metric as CF
from vfairness.evaluation.vfairness_metrics import discovery as DISC
from vfairness.evaluation.vfairness_metrics import explanation_diagnostics as EDIAG
from vfairness.evaluation.vfairness_metrics import fairness_decomposition as FDEC
from vfairness.evaluation.vfairness_metrics import integrations as INTEG
from vfairness.evaluation.vfairness_metrics import intersectional as INTER
from vfairness.evaluation.vfairness_metrics import ranking as RANK
from vfairness.evaluation.vfairness_metrics import recourse as RECO
from vfairness.evaluation.vfairness_metrics import report_types as RTYPES
from vfairness.evaluation.vfairness_metrics import robustness as ROB
from vfairness.evaluation.vfairness_metrics import visualization as VIZ
from vfairness.evaluation.vfairness_metrics._grouping import (
    GroupManager,
    compute_max_difference,
    compute_max_ratio,
)
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer, MetricResult
from vfairness.evaluation.vfairness_metrics.attribution import (
    AttributionResult,
    FeatureAttributionExplainer,
    FeatureContribution,
)
from vfairness.evaluation.vfairness_metrics.explainer import MetricExplanation

NAN = float("nan")
INF = float("inf")


def _warned(records, fragment):
    """True when any captured warning mentions ``fragment`` (case-insensitive)."""
    return any(fragment.lower() in str(r.message).lower() for r in records)


# 1. AttributionResult.top must never rank a could-not-check


def _constant_column_model():
    """A model whose column 0 is CONSTANT, so its importance cannot be measured.

    Shuffling a constant column always returns it to its original arrangement,
    which ``global_importance`` detects and reports as ``importance=nan`` with
    ``direction='not_assessed'``. Column 1 is the real driver, by a wide margin
    over column 2, so the CORRECT ranking is unambiguous and is asserted below
    rather than quoted.
    """
    rng = np.random.default_rng(0)
    n = 60
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = (2.0 * X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)

    def predict(a):
        return 1.0 / (1.0 + np.exp(-(2.0 * a[:, 1] + 0.5 * a[:, 2])))

    return predict, X, y


class TestTopNeverRanksAnUnmeasuredFeature:
    def test_the_public_entry_point_does_not_report_an_unmeasured_top_driver(self):
        """The defect, through ``global_importance``, not through the container."""
        predict, X, y = _constant_column_model()
        explainer = FeatureAttributionExplainer(
            predict, feature_names=["constant_col", "income", "age"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explainer.global_importance(X, y=y, n_repeats=5, random_state=0)

        by_name = {c.feature: c for c in result.contributions}
        # The premise: this really is the refusal state, and the OTHER two really
        # were measured. Without this the test below could pass vacuously.
        assert math.isnan(by_name["constant_col"].importance)
        assert by_name["constant_col"].direction == "not_assessed"
        assert is_measured(by_name["income"].importance)
        assert is_measured(by_name["age"].importance)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top = result.top(1)
            top_all = result.top(3)

        assert [c.feature for c in top] == ["income"], (
            "the feature whose importance could NOT be measured was ranked first"
        )
        # Derived, not quoted: the ranking must be the measured ones in
        # descending order, and the unmeasured one must be absent.
        assert [c.feature for c in top_all] == ["income", "age"]
        assert all(is_measured(c.importance) for c in top_all)
        # CONTROL: nothing was deleted from the result itself, so a reader who
        # wants the full list still has it.
        assert len(result.contributions) == 3

    def test_the_exclusion_is_said_out_loud(self):
        predict, X, y = _constant_column_model()
        explainer = FeatureAttributionExplainer(
            predict, feature_names=["constant_col", "income", "age"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explainer.global_importance(X, y=y, n_repeats=5, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.top(1)
        assert _warned(caught, "not a measurement")
        assert _warned(caught, "constant_col")

    @pytest.mark.parametrize("nan_position", [0, 1, 2])
    def test_input_order_cannot_move_a_refusal_into_a_top_slot(self, nan_position):
        """A NaN sort key loses every comparison, so Timsort leaves it in place.

        That is why the defect depended on the COLUMN ORDER and why this is
        parametrised: at position 0 the refusal was returned as the top driver,
        at position 2 the same data ranked correctly by accident.
        """
        measured = [
            FeatureContribution("income", 0.40, "increase", 0.40),
            FeatureContribution("age", 0.10, "increase", 0.10),
        ]
        contribs = list(measured)
        contribs.insert(nan_position, FeatureContribution("zip", NAN, "not_assessed", None))
        result = AttributionResult("global", "permutation", contributions=contribs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert [c.feature for c in result.top(5)] == ["income", "age"]

    @pytest.mark.parametrize(
        "importance",
        [NAN, INF, -INF, np.float32("nan"), np.float64("nan"), np.float32("inf"), None],
    )
    def test_every_unmeasurable_importance_is_kept_out_of_the_ranking(self, importance):
        """Keyed on ``is_measured``, so np.float32 and the infinities are covered.

        ``isinstance(x, float)`` is False for np.float32 and True for an
        infinity, which is why the predicate is the repo's canonical one.
        """
        result = AttributionResult(
            "global",
            "permutation",
            contributions=[
                FeatureContribution("bad", importance, "not_assessed", None),
                FeatureContribution("good", 0.25, "increase", 0.25),
            ],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert [c.feature for c in result.top(5)] == ["good"]

    def test_no_measured_importance_means_an_empty_ranking_not_a_fabricated_one(self):
        result = AttributionResult(
            "global",
            "permutation",
            contributions=[FeatureContribution(f, NAN, "not_assessed", None) for f in "abc"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert result.top(3) == []
        assert _warned(caught, "not a measurement")

    def test_control_a_healthy_result_ranks_by_its_real_numbers_in_silence(self):
        """The over-correction guard: a guard that refuses everything is wrong."""
        contribs = [
            FeatureContribution("age", 0.10, "increase", 0.10),
            FeatureContribution("income", 0.40, "increase", 0.40),
            FeatureContribution("zip", 0.25, "decrease", -0.25),
        ]
        result = AttributionResult("global", "permutation", contributions=contribs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            top = result.top(2)
        assert [c.feature for c in top] == ["income", "zip"]
        assert top[0].importance == pytest.approx(0.40)
        assert [str(r.message) for r in caught] == [], "a clean result must not warn"

    def test_control_numpy_scalar_importances_are_real_evidence(self):
        """The defect running backwards: refusing np.float32 would DISCARD data."""
        result = AttributionResult(
            "global",
            "permutation",
            contributions=[
                FeatureContribution("a", np.float32(0.1), "increase", 0.1),
                FeatureContribution("b", np.float64(0.9), "increase", 0.9),
                FeatureContribution("c", np.int64(0), "neutral", 0),
            ],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert [c.feature for c in result.top(3)] == ["b", "a", "c"]
        assert [str(r.message) for r in caught] == []


# 2. compute_max_ratio: 1.0 is perfect parity, never a could-not-check


class TestMaxRatioRefusesWhatItCouldNotDivide:
    def test_two_undefined_groups_are_not_perfect_parity(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ratio, hi, lo = compute_max_ratio({"a": NAN, "b": NAN})
        assert math.isnan(ratio), "1.0 on a ratio scale is PERFECT PARITY"
        assert (hi, lo) == ("", "")
        assert _warned(caught, "not")
        assert _warned(caught, "1.0")

    def test_the_sibling_answers_the_same_way_on_the_same_data(self):
        """The two helpers must not disagree; that asymmetry WAS the defect."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diff, _, _ = compute_max_difference({"a": NAN, "b": NAN})
            ratio, _, _ = compute_max_ratio({"a": NAN, "b": NAN})
        assert math.isnan(diff) and math.isnan(ratio)

    def test_a_group_that_is_never_selected_is_disclosed_not_absorbed(self):
        """A rate of 0.0 cannot be a denominator, which is the SHARPEST case.

        A group with no selections at all is the most extreme disparity this
        statistic can express, and dropping it can only make the number read
        fairer.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ratio, _, _ = compute_max_ratio({"a": 0.5, "b": 0.0})
        assert _warned(caught, "lower bound") or _warned(caught, "not measurable")
        assert _warned(caught, "zero") or _warned(caught, "never selected")
        # Whatever it returns, it must not be a silent 1.0.
        assert math.isnan(ratio) or ratio == pytest.approx(1.0)

    def test_only_zero_valued_groups_cannot_be_divided_at_all(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ratio, _, _ = compute_max_ratio({"a": 0.0, "b": 0.0})
        assert math.isnan(ratio)
        assert _warned(caught, "not measurable")

    @pytest.mark.parametrize("absent", [None, "0.5", b"x", pd.NA])
    def test_the_absence_doors_refuse_instead_of_crashing(self, absent):
        """``np.isnan(None)`` raises; the sibling has used ``_as_float`` for months."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ratio, _, _ = compute_max_ratio({"a": absent, "b": 0.5})
        assert math.isnan(ratio)
        assert _warned(caught, "not measurable")

    def test_control_a_healthy_pair_still_returns_its_real_ratio(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ratio, hi, lo = compute_max_ratio({"a": 0.8, "b": 0.4})
        # Derived: 0.8 / 0.4 up to the epsilon guard against a zero denominator.
        assert ratio == pytest.approx(2.0, rel=1e-6)
        assert (hi, lo) == ("a", "b")
        assert [str(r.message) for r in caught] == []

    def test_control_one_undefined_group_among_three_still_measures_the_rest(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio, hi, lo = compute_max_ratio({"a": 0.9, "b": 0.3, "c": NAN})
        assert ratio == pytest.approx(3.0, rel=1e-6)
        assert (hi, lo) == ("a", "b")


# 3. GroupManager: a group minted out of an absent value must be disclosed


def _frame_with_absent_race():
    return pd.DataFrame(
        {
            "sex": ["F"] * 20 + ["M"] * 20,
            "race": ["white"] * 10 + [pd.NA] * 10 + ["black"] * 10 + [None] * 10,
        }
    )


class TestGroupManagerDisclosesAMintedIntersection:
    def test_pd_na_and_none_mint_group_names_and_that_is_now_said(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager = GroupManager(_frame_with_absent_race(), min_group_size=5)
            groups = manager.groups
            sizes = manager.get_group_sizes()

        # The premise: the labels really are built out of the absence, and the
        # rows really are carried (not dropped), so the disclosure is the half
        # that has to exist.
        minted = [g for g in groups if "<NA>" in g or g.endswith("_None") or "nan" in g]
        assert len(minted) == 2, groups
        assert sum(sizes[g] for g in minted) == 20

        assert _warned(caught, "MISSING value in at least one")
        assert _warned(caught, "not a measured result for any real group")
        # The distinct SPELLINGS matter: they split one absent category across
        # several groups, each one able to fall under the size floor alone.
        assert _warned(caught, "<NA>") and _warned(caught, "None")

    @pytest.mark.parametrize(
        "absent", [pd.NA, None, np.nan, float("nan"), pd.NaT], ids=lambda v: type(v).__name__
    )
    def test_every_absence_door_reaches_the_disclosure(self, absent):
        frame = pd.DataFrame({"sex": ["F"] * 10 + ["M"] * 10, "race": ["w"] * 10 + [absent] * 10})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GroupManager(frame, min_group_size=2)
        assert _warned(caught, "MISSING value in at least one")

    def test_the_2d_numpy_path_discloses_it_too(self):
        attr = np.array([[1.0, 2.0]] * 10 + [[np.nan, 2.0]] * 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GroupManager(attr, min_group_size=2)
        assert _warned(caught, "MISSING value in at least one")

    def test_the_simple_path_keeps_its_own_disclosure(self):
        """The sibling this fix was modelled on must not have moved."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GroupManager(np.array([1.0] * 20 + [np.nan] * 10), min_group_size=5)
        assert _warned(caught, "MISSING sensitive attribute")

    def test_control_a_clean_intersectional_frame_is_silent_and_unchanged(self):
        frame = pd.DataFrame(
            {"sex": ["F"] * 20 + ["M"] * 20, "race": ["w"] * 10 + ["b"] * 10 + ["w"] * 20}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager = GroupManager(frame, min_group_size=5)
            sizes = manager.get_group_sizes()
        assert sorted(sizes) == ["F_b", "F_w", "M_w"]
        assert sizes["F_w"] == 10 and sizes["F_b"] == 10 and sizes["M_w"] == 20
        assert manager.is_intersectional is True
        assert [str(r.message) for r in caught] == []


# 4. GroupManager: a size floor that is not a measurement grades nothing


class TestGroupManagerRefusesAnUnusableSizeFloor:
    @pytest.mark.parametrize("floor", [NAN, INF, -INF, None, True, "5"])
    def test_a_non_measurement_floor_cannot_return_a_silent_empty_invalid_list(self, floor):
        """``nan <= 1`` is False, so NaN walked past the vacuity guard.

        ``size < nan`` is then False for every group, so the answer was the same
        vacuous ``[]`` the ``min_group_size <= 1`` case is disclosed for.
        """
        manager = GroupManager(np.array(["a"] * 40 + ["b"] * 2), min_group_size=floor)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            invalid = manager.get_invalid_groups()
        assert invalid == []
        assert _warned(caught, "NEVER APPLIED")

    @pytest.mark.parametrize("floor", [NAN, None, True, "5"])
    def test_the_valid_side_stops_claiming_the_groups_are_too_small(self, floor):
        manager = GroupManager(np.array(["a"] * 40 + ["b"] * 2), min_group_size=floor)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            valid = manager.get_valid_groups()
        assert valid == []
        assert _warned(caught, "not a usable size floor")
        assert not _warned(caught, "All groups are below"), (
            "nothing was below anything: no comparison happened"
        )

    def test_get_info_still_reports_the_groups_it_has(self):
        manager = GroupManager(np.array(["a"] * 40 + ["b"] * 2), min_group_size=NAN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = manager.get_info()
        assert info["n_groups"] == 2
        assert info["n_valid_groups"] == 0 and info["n_invalid_groups"] == 0

    def test_control_a_real_floor_still_partitions_the_groups_in_silence(self):
        manager = GroupManager(np.array(["a"] * 40 + ["b"] * 2), min_group_size=30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            valid = manager.get_valid_groups()
            invalid = manager.get_invalid_groups()
        assert valid == ["a"] and invalid == ["b"]
        assert [str(r.message) for r in caught] == []

    def test_control_the_vacuous_floor_of_one_keeps_its_own_disclosure(self):
        manager = GroupManager(np.array(["a"] * 40 + ["b"] * 2), min_group_size=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert manager.get_invalid_groups() == []
        assert _warned(caught, "vacuous")

    def test_control_a_numpy_integer_floor_is_a_measurement(self):
        manager = GroupManager(np.array(["a"] * 40 + ["b"] * 2), min_group_size=np.int64(30))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert manager.get_valid_groups() == ["a"]
            assert manager.get_invalid_groups() == ["b"]
        assert [str(r.message) for r in caught] == []


# 5. MetricExplanation: the card's value must agree with its own severity


class TestMetricExplanationDoesNotPublishAnAbsentValue:
    @pytest.mark.parametrize(
        "absent",
        [
            NAN,
            np.float64("nan"),
            np.float32("nan"),
            np.float16("nan"),
            INF,
            -INF,
            np.float32("inf"),
            None,
            pd.NA,
            pd.NaT,
        ],
        ids=repr,
    )
    def test_no_absence_door_reaches_the_value_field_or_the_printed_card(self, absent):
        card = MetricExplanation("demographic_parity_difference", "d", "g", absent, "e", "b", "r")
        assert card.to_dict()["value"] is None
        assert "N/A (insufficient data)" in str(card)
        # Serialisable, strictly: np.float32(nan) raised TypeError here and
        # ``inf`` produced the non-JSON token ``Infinity``.
        json.dumps(card.to_dict(), allow_nan=False)

    @pytest.mark.parametrize(
        "absent",
        [np.float32("nan"), pd.NA, INF, None],
        ids=["np.float32 nan", "pd.NA", "inf", "None"],
    )
    def test_the_two_halves_of_the_card_agree_through_the_public_entry_point(self, absent):
        from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            card = FairExplAIner().explain_metric("demographic_parity_difference", absent)
        # The severity half was already right; the value half contradicted it.
        assert card.severity == "could_not_check"
        assert card.to_dict()["value"] is None
        assert "N/A (insufficient data)" in str(card)

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.1234, "0.1234"),
            (np.float64(0.1234), "0.1234"),
            (np.float32(0.5), "0.5000"),
            (0.0, "0.0000"),
        ],
    )
    def test_control_a_measured_value_is_still_shown_as_a_number(self, value, expected):
        card = MetricExplanation("m", "d", "g", value, "e", "b", "r")
        assert card.to_dict()["value"] is not None
        assert expected in str(card)

    def test_control_zero_is_a_measurement_and_is_not_blanked(self):
        """Perfect parity is a RESULT. A guard that refuses it destroys the unit."""
        card = MetricExplanation("demographic_parity_difference", "d", "g", 0.0, "e", "b", "r")
        assert card.to_dict()["value"] == 0.0

    @pytest.mark.parametrize("value", ["balanced", 5, {"a": 1}, ["x"], True])
    def test_control_non_numeric_content_is_left_exactly_as_it_was(self, value):
        """``value`` is typed Any; refusing every falsy thing would delete content."""
        card = MetricExplanation("m", "d", "g", value, "e", "b", "r")
        assert card.to_dict()["value"] == value


# 6. missing_strategy='as_group' must mean the same thing for one attribute or two


def _mixed_missing_inputs():
    rng = np.random.default_rng(3)
    n = 120
    y_true = (rng.random(n) < 0.5).astype(int)
    y_pred = y_true.copy()
    frame = pd.DataFrame(
        {
            "sex": ["F"] * 60 + ["M"] * 60,
            "race": ["w"] * 30 + [np.nan] * 30 + ["b"] * 30 + [None] * 30,
        }
    )
    flat = np.array(["w"] * 40 + [np.nan] * 40 + ["b"] * 40, dtype=object)
    return y_true, y_pred, frame, flat


class TestAsGroupMeansTheSameThingForAnIntersection:
    def test_an_intersectional_as_group_run_keeps_every_row(self):
        y_true, y_pred, frame, _ = _mixed_missing_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = FairnessAnalyzer(
                y_true, y_pred, frame, missing_strategy="as_group"
            ).get_report()
        info = report["data_info"]
        assert info["n_excluded"] == 0, "as_group must not drop the rows it exists to keep"
        assert info["final_size"] == info["original_size"] == len(y_true)
        # The level is NAMED by this library, never a str()-minted 'nan'/'<NA>'.
        groups = list(info["group_sizes"])
        assert any("__missing__" in g for g in groups), groups
        assert not any("nan" in g or "<NA>" in g or "None" in g for g in groups), groups

    def test_the_one_attribute_and_two_attribute_paths_now_agree(self):
        y_true, y_pred, frame, flat = _mixed_missing_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat_info = FairnessAnalyzer(
                y_true, y_pred, flat, missing_strategy="as_group"
            ).get_report()["data_info"]
            wide_info = FairnessAnalyzer(
                y_true, y_pred, frame, missing_strategy="as_group"
            ).get_report()["data_info"]
        assert flat_info["n_excluded"] == wide_info["n_excluded"] == 0
        assert any("__missing__" in str(g) for g in flat_info["group_sizes"])
        assert any("__missing__" in str(g) for g in wide_info["group_sizes"])

    def test_control_exclude_still_excludes_and_still_says_so(self):
        y_true, y_pred, frame, _ = _mixed_missing_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = FairnessAnalyzer(
                y_true, y_pred, frame, missing_strategy="exclude"
            ).get_report()
        info = report["data_info"]
        assert info["n_excluded"] == 60
        assert info["final_size"] == 60
        assert "missing_strategy='exclude'" in report["assessment"]["summary"]

    def test_control_error_still_refuses(self):
        from vfairness.exceptions import InvalidDataError

        y_true, y_pred, frame, _ = _mixed_missing_inputs()
        with pytest.raises(InvalidDataError):
            FairnessAnalyzer(y_true, y_pred, frame, missing_strategy="error")


# 7. vacuous_bound_reason: reused rather than reinvented, and pinned


class TestVacuousBoundReason:
    @pytest.mark.parametrize(
        "metric,bound,role",
        [
            ("demographic_parity_difference", 1.2, MD.BoundRole.THRESHOLD),
            ("demographic_parity_difference", 60.0, MD.BoundRole.THRESHOLD),
            ("disparate_impact_ratio", 0.0, MD.BoundRole.THRESHOLD),
            ("demographic_parity_difference", -1.0, MD.BoundRole.REQUIRED_IMPROVEMENT),
            ("demographic_parity_difference", -1e9, MD.BoundRole.REQUIRED_IMPROVEMENT),
            ("demographic_parity_difference", 1e9, MD.BoundRole.ALLOWED_DEGRADATION),
        ],
    )
    def test_an_unbreachable_bound_is_refused_with_a_reason(self, metric, bound, role):
        reason = MD.vacuous_bound_reason(metric, bound, role)
        assert reason is not None and "lie in" in reason

    @pytest.mark.parametrize(
        "metric,bound,role",
        [
            ("demographic_parity_difference", 0.1, MD.BoundRole.THRESHOLD),
            ("disparate_impact_ratio", 0.8, MD.BoundRole.THRESHOLD),
            ("demographic_parity_difference", 0.05, MD.BoundRole.REQUIRED_IMPROVEMENT),
            ("demographic_parity_difference", 0.01, MD.BoundRole.ALLOWED_DEGRADATION),
        ],
    )
    def test_control_a_usable_bound_grades_and_is_not_refused(self, metric, bound, role):
        assert MD.vacuous_bound_reason(metric, bound, role) is None

    def test_an_unrecognised_role_fails_closed(self):
        reason = MD.vacuous_bound_reason(
            "demographic_parity_difference", 60.0, "required_baseline_floor"
        )
        assert reason is not None and "COULD NOT BE CHECKED" in reason

    @pytest.mark.parametrize("bound", [NAN, INF, None, True, "0.9"])
    def test_a_bound_that_is_not_a_measurement_is_left_to_the_caller(self, bound):
        """One bound must not collect two different refusals."""
        assert (
            MD.vacuous_bound_reason("demographic_parity_difference", bound, MD.BoundRole.THRESHOLD)
            is None
        )

    def test_a_metric_with_no_declared_range_makes_no_claim(self):
        assert MD.vacuous_bound_reason("brier_score", 60.0, MD.BoundRole.THRESHOLD) is None

    def test_a_numpy_bound_is_judged_like_a_python_one(self):
        assert (
            MD.vacuous_bound_reason(
                "demographic_parity_difference", np.float32(1.2), MD.BoundRole.THRESHOLD
            )
            is not None
        )


# 8. min_attainable_p_mannwhitney: fail closed, and never flatter a dead design


class TestMinAttainablePMannWhitney:
    def test_a_bigger_design_has_a_lower_floor(self):
        """The ``mde(inf, 100)`` failure mode: an unassessable design reading BETTER."""
        small = ST.min_attainable_p_mannwhitney(3, 3)
        medium = ST.min_attainable_p_mannwhitney(30, 30)
        large = ST.min_attainable_p_mannwhitney(100, 1000)
        assert large < medium < small

    def test_a_three_by_three_design_can_barely_reach_five_percent(self):
        floor = ST.min_attainable_p_mannwhitney(3, 3)
        assert 0.0 < floor < 1.0
        # Derived, not quoted: the point of the function is that a tiny design's
        # best possible p sits next to alpha, so "not significant" says nothing.
        assert floor > 0.01

    def test_a_single_pair_cannot_reach_any_alpha(self):
        assert ST.min_attainable_p_mannwhitney(1, 1) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "n_a,n_b", [(0, 5), (5, 0), (-3, 5), (NAN, 5), (5, NAN), (INF, 5), (5, INF)]
    )
    def test_an_unusable_sample_size_is_could_not_check_and_never_a_number(self, n_a, n_b):
        assert ST.min_attainable_p_mannwhitney(n_a, n_b) is None

    @pytest.mark.parametrize("n", [3, 5])
    def test_the_floor_is_the_smallest_any_method_could_reach(self, n):
        """The fail-safe DIRECTION, which a monotonicity check cannot see.

        The docstring's load-bearing claim is that the tie-corrected asymptotic
        floor is LOWER than the exact one, so assuming the exact one would call
        a detectable design dead. Measured at n=3: the exact two-sided floor is
        0.1000, which cannot reach alpha=0.05 at all, while the real attainable
        floor is 0.0469, which can. Taking the largest candidate instead of the
        smallest therefore refuses a design that has power, and only this
        comparison reddens on that change.
        """
        from scipy import stats as scipy_stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact = float(
                scipy_stats.mannwhitneyu(
                    list(range(1, n + 1)),
                    list(range(n + 1, 2 * n + 1)),
                    alternative="two-sided",
                    method="exact",
                )[1]
            )
        floor = ST.min_attainable_p_mannwhitney(n, n)
        assert floor is not None
        assert floor < exact, "the floor must be the smallest attainable p, not the largest"

    def test_a_three_by_three_design_can_reach_five_percent_but_only_just(self):
        """Derived from the function's own purpose, not quoted from today's run."""
        floor = ST.min_attainable_p_mannwhitney(3, 3)
        assert floor < 0.05 < ST.min_attainable_p_mannwhitney(2, 2)

    def test_control_numpy_integers_are_accepted(self):
        assert ST.min_attainable_p_mannwhitney(np.int64(20), np.int64(20)) == pytest.approx(
            ST.min_attainable_p_mannwhitney(20, 20)
        )


# 9. FairnessAnalyzer.report_to_svg: an EXPORT must carry its own could-not-check


def _svg_text(svg):
    return " ".join(re.findall(r">([^<>]+)<", svg))


class TestReportToSvg:
    def _analyzer(self, kind):
        rng = np.random.default_rng(3)
        n = 120
        y_true = (rng.random(n) < 0.5).astype(int)
        y_pred = y_true.copy()
        groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
        if kind == "single_group":
            groups = np.array(["a"] * n)
        elif kind == "n_equals_2":
            y_true, y_pred, groups = y_true[:2], y_pred[:2], np.array(["a", "b"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return FairnessAnalyzer(y_true, y_pred, groups)

    def test_a_healthy_run_renders_a_real_score(self):
        svg = self._analyzer("healthy").report_to_svg()
        assert svg.lstrip().startswith("<svg")
        text = _svg_text(svg)
        assert re.search(r"\b100\s*/\s*100\b", text)
        assert "NOT ASSESSABLE" not in text

    @pytest.mark.parametrize("kind", ["single_group", "n_equals_2"])
    def test_a_run_that_measured_nothing_says_so_on_the_canvas(self, kind):
        svg = self._analyzer(kind).report_to_svg()
        text = _svg_text(svg)
        assert "NOT ASSESSABLE" in text
        assert "could not check" in text.lower()
        assert not re.search(r"\b\d+\s*/\s*100\b", text), (
            "a score printed over an assessment that never ran"
        )

    def test_the_svg_is_written_where_it_was_asked_for(self, tmp_path):
        out = tmp_path / "report.svg"
        svg = self._analyzer("healthy").report_to_svg(save_path=str(out))
        assert out.read_text() == svg


# 10. Every result container in this batch: execute it, and check it does not
#     quietly mint a value where it could not measure one.


def _refusal_and_healthy():
    """``(label, refusal_instance, healthy_instance)`` for each container.

    The refusal instance is built in the shape its own producer documents for a
    could-not-check; the healthy one carries real numbers.
    """
    recourse = RECO.Recourse(
        [RECO.FeatureChange("income", 100.0, 200.0)], 1, 0.2, 0.7, "raise income"
    )
    return [
        (
            "PermutationTestResult",
            ROB.PermutationTestResult(NAN, NAN, np.array([]), 0, None, None, "none"),
            ROB.PermutationTestResult(
                0.2, 0.01, np.array([0.1, 0.2, 0.3]), 999, True, False, "positive"
            ),
        ),
        (
            "ContingencyTestResult",
            ROB.ContingencyTestResult(
                "fisher_exact", NAN, NAN, np.zeros((2, 2)), np.zeros((2, 2)), None
            ),
            ROB.ContingencyTestResult(
                "chi_square",
                4.2,
                0.04,
                np.array([[10, 5], [5, 10]]),
                np.full((2, 2), 7.5),
                True,
            ),
        ),
        (
            "RobustMetricsResult",
            ROB.RobustMetricsResult("g", 3, 0.5, 0.5, 0.5, 0.5, None, NAN, trim_effective=False),
            ROB.RobustMetricsResult("g", 300, 0.5, 0.48, 0.49, 0.5, False, 0.04),
        ),
        (
            "SensitivityResult",
            ROB.SensitivityResult(NAN, np.array([]), NAN, NAN, NAN, None, None, "flip", 0, 20),
            ROB.SensitivityResult(
                0.2, np.array([0.19, 0.21]), 0.2, 0.01, 0.01, True, 0.9, "flip", 20, 0
            ),
        ),
        (
            "SubgroupAuditResult",
            ROB.SubgroupAuditResult({}, [], None, NAN, None, 0, 0, 5, 2),
            ROB.SubgroupAuditResult({"a": {"m": 0.1}}, [("a", 0.3)], "a", 0.3, True, 4, 1),
        ),
        (
            "ColumnRole",
            DISC.ColumnRole("c", "unknown", NAN, "could not read the column"),
            DISC.ColumnRole("sex", "protected", 0.9123, "name match", category="sex"),
        ),
        (
            "FairnessViolation",
            DISC.FairnessViolation(
                "a", "demographic_parity_difference", NAN, 0.1, "critical", "m", "f", NAN
            ),
            DISC.FairnessViolation(
                "a", "demographic_parity_difference", 0.3, 0.1, "critical", "m", "f", 0.3
            ),
        ),
        (
            "ProtectedAttributeCandidate",
            DISC.ProtectedAttributeCandidate("c", NAN, "could not read", 0, []),
            DISC.ProtectedAttributeCandidate(
                "sex", 0.95, "name match", 2, ["m", "f"], category="demographic"
            ),
        ),
        (
            "MetricResult",
            MetricResult("demographic_parity_difference", NAN),
            MetricResult("demographic_parity_difference", 0.05, is_fair=True, verdict="fair"),
        ),
        (
            "StatisticalResult",
            ST.StatisticalResult(NAN, NAN, NAN, ST.IntervalType.CONFIDENCE),
            ST.StatisticalResult(0.2, 0.1, 0.3, ST.IntervalType.CONFIDENCE),
        ),
        (
            "MultipleTestingResult",
            ST.MultipleTestingResult(
                np.array([NAN]),
                np.array([NAN]),
                np.array([False]),
                "fdr_bh",
                0.05,
                0,
                tested_mask=np.array([False]),
                n_not_tested=1,
            ),
            ST.MultipleTestingResult(
                np.array([0.01]),
                np.array([0.01]),
                np.array([True]),
                "fdr_bh",
                0.05,
                1,
                tested_mask=np.array([True]),
                n_not_tested=0,
            ),
        ),
        (
            "FeatureContribution",
            FeatureContribution("zip", NAN, "not_assessed", None),
            FeatureContribution("income", 0.4, "increase", 0.4),
        ),
        (
            "AttributionResult",
            AttributionResult(
                "global",
                "permutation",
                contributions=[FeatureContribution("zip", NAN, "not_assessed", None)],
                notes=["no shuffle rearranged the column"],
            ),
            AttributionResult(
                "global",
                "permutation",
                contributions=[FeatureContribution("income", 0.4, "increase", 0.4)],
            ),
        ),
        (
            "CounterfactualFairnessResult",
            CF.CounterfactualFairnessResult(
                0, None, NAN, NAN, None, "not_assessed", "x", notes=["no scoreable row"]
            ),
            CF.CounterfactualFairnessResult(100, 0.05, 0.02, 0.2, 0.5, "low", "x"),
        ),
        (
            "FairnessDecompositionResult",
            FDEC.FairnessDecompositionResult(
                "demographic_parity",
                "sex",
                "m",
                "f",
                NAN,
                NAN,
                {},
                {"f1": None},
                [],
                NAN,
                0,
                "not_assessed",
                "x",
                notes=["n"],
                additivity_verified=None,
            ),
            FDEC.FairnessDecompositionResult(
                "demographic_parity",
                "sex",
                "m",
                "f",
                0.2,
                0.5,
                {"f1": 0.2},
                {"f1": 0.8},
                ["f1"],
                0.0,
                100,
                "high",
                "x",
                n_rows_supplied=100,
                n_advantaged=50,
                n_disadvantaged=50,
                additivity_verified=True,
            ),
        ),
        (
            "GroupAdvantage",
            INTER.GroupAdvantage(
                "g",
                0.5,
                10,
                1.0,
                1.0,
                0.0,
                "info",
                false_positive_rate=NAN,
                unmeasured={"false_positive_rate": "no actual negatives"},
            ),
            INTER.GroupAdvantage("g", 0.5, 100, 1.0, 0.9, 0.1, "low", 0.5, 0.1, 0.0),
        ),
        (
            "RankingFairnessResult",
            RANK.RankingFairnessResult("ndkl", NAN, {}, None, None, 0.1),
            RANK.RankingFairnessResult("ndkl", 0.05, {"a": 0.5, "b": 0.5}, {"a": 1.0}, True, 0.1),
        ),
        (
            "FeatureChange",
            RECO.FeatureChange("f", NAN, NAN),
            RECO.FeatureChange("income", 100.0, 200.0),
        ),
        ("Recourse", RECO.Recourse([], 0, NAN, NAN, ""), recourse),
        (
            "RecourseResult",
            RECO.RecourseResult(None, NAN, 0.5, "above_threshold", [], 10, notes=["unscorable"]),
            RECO.RecourseResult(
                True, 0.3, 0.5, "above_threshold", [recourse], 10, n_scorable_candidates=10
            ),
        ),
        (
            "MetricExplanation",
            MetricExplanation("m", "d", "g", NAN, "e", "b", "r", severity="could_not_check"),
            MetricExplanation("m", "d", "g", 0.05, "e", "b", "r", severity="low"),
        ),
    ]


_CONTAINERS = _refusal_and_healthy()


@pytest.mark.parametrize("label,refusal,healthy", _CONTAINERS, ids=[c[0] for c in _CONTAINERS])
def test_a_container_to_dict_round_trips_and_mints_nothing(label, refusal, healthy):
    """Execute both shapes and check no neutral value was minted.

    The rule: every field of the REFUSAL instance must be None, NaN, an empty
    container, a string, an int count, or a non-numeric marker. It must never be
    a number that reads as a measured result where the corresponding healthy
    field carries one. ``0.0`` and ``True`` are the two that do read that way.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refused = refusal.to_dict()
        measured = healthy.to_dict()
    assert isinstance(refused, dict) and isinstance(measured, dict)
    assert set(refused) == set(measured), "the two shapes must serialise the same keys"
    # Both must survive an ordinary json round trip (NaN is permitted: it is the
    # deliberate could-not-check magnitude these containers document).
    json.loads(json.dumps(refused))
    json.loads(json.dumps(measured))

    verdict_like = (
        "is_fair",
        "found",
        "flag",
        "significant_at_05",
        "significant_at_01",
        "gerrymandering_detected",
        "is_robust",
        "outlier_influence_detected",
        "detectable_at_05",
        "additivity_verified",
    )
    for key in verdict_like:
        if key in refused:
            assert refused[key] is not True, f"{label}.{key} certified a verdict it never reached"

    score_like = (
        "flip_rate",
        "robustness_score",
        "fairness_score",
        "value",
        "importance",
        "confidence",
        "divergence_ratio",
        "worst_disparity",
        "p_value",
        "point_estimate",
        "total_disparity",
    )
    for key in score_like:
        if key in refused and isinstance(refused[key], float):
            assert math.isnan(refused[key]) or refused[key] != 0.0 or key in {"value"}, (
                f"{label}.{key} published 0.0 for a quantity it could not measure"
            )


@pytest.mark.parametrize("label,refusal,healthy", _CONTAINERS, ids=[c[0] for c in _CONTAINERS])
def test_to_dict_is_the_same_answer_twice(label, refusal, healthy):
    """A to_dict with a side effect would make the refusal state unstable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert repr(refusal.to_dict()) == repr(refusal.to_dict())
        assert repr(healthy.to_dict()) == repr(healthy.to_dict())


class TestTheThreeStateFieldsSurviveSerialisation:
    """A None dropped at the to_dict boundary is a correct measurement no reader sees."""

    def test_ranking_is_fair_none_is_not_a_pass(self):
        assert RANK.RankingFairnessResult("ndkl", NAN, {}).to_dict()["is_fair"] is None

    def test_recourse_found_none_is_not_a_no(self):
        result = RECO.RecourseResult(None, NAN, 0.5, "above_threshold", [], 10)
        assert result.to_dict()["found"] is None
        assert result.to_dict()["n_scorable_candidates"] == 0

    def test_counterfactual_flip_rate_none_is_not_a_zero(self):
        result = CF.CounterfactualFairnessResult(0, None, NAN, NAN, None, "not_assessed", "x")
        assert result.to_dict()["flip_rate"] is None
        assert result.to_dict()["severity"] == "not_assessed"

    def test_decomposition_proxy_score_none_is_not_a_cleared_feature(self):
        result = FDEC.FairnessDecompositionResult(
            "demographic_parity",
            "sex",
            "m",
            "f",
            0.2,
            0.5,
            {"f1": 0.2},
            {"f1": None},
            [],
            0.0,
            100,
            "high",
            "x",
        )
        payload = result.to_dict()
        assert payload["proxy_scores"]["f1"] is None
        assert "f1" not in payload["flagged_proxies"]

    def test_group_advantage_names_the_field_it_could_not_measure(self):
        cell = INTER.GroupAdvantage(
            "g",
            0.5,
            10,
            1.0,
            1.0,
            0.0,
            "info",
            false_positive_rate=NAN,
            unmeasured={"false_positive_rate": "no actual negatives"},
        )
        payload = cell.to_dict()
        assert math.isnan(payload["false_positive_rate"])
        assert payload["unmeasured"]["false_positive_rate"]

    def test_multiple_testing_tested_mask_distinguishes_untested_from_not_rejected(self):
        payload = ST.MultipleTestingResult(
            np.array([0.01, NAN]),
            np.array([0.02, NAN]),
            np.array([True, False]),
            "fdr_bh",
            0.05,
            1,
            tested_mask=np.array([True, False]),
            n_not_tested=1,
        ).to_dict()
        assert payload["rejection_mask"] == [True, False]
        assert payload["tested_mask"] == [True, False]
        assert payload["n_not_tested"] == 1

    def test_statistical_result_is_not_significant_on_an_uncomputable_interval(self):
        assert (
            ST.StatisticalResult(NAN, NAN, NAN, ST.IntervalType.CONFIDENCE).is_significant is False
        )

    def test_statistical_result_carries_the_null_it_tested(self):
        ratio = ST.StatisticalResult(1.0, 0.9, 1.1, ST.IntervalType.CONFIDENCE, null_value=1.0)
        assert ratio.is_significant is False
        assert ratio.to_dict()["null_value"] == 1.0

    def test_metric_result_assessable_travels_with_the_value(self):
        assert MetricResult("m", NAN).to_dict()["assessable"] is False
        assert MetricResult("m", 0.05).to_dict()["assessable"] is True

    def test_metric_result_default_is_not_a_passing_fairness_result(self):
        payload = MetricResult("m", NAN).to_dict()
        assert payload["is_fair"] is False and payload["verdict"] == "not_computed"

    def test_adversarial_probe_flag_none_is_not_a_clean_explainer(self):
        probe = EDIAG.AdversarialProbeResult(
            None, NAN, NAN, NAN, "no finite gap", not_run_because="no seed produced a finite gap"
        )
        assert probe.flag is None
        assert probe.not_run_because
        assert EDIAG.AdversarialProbeResult(True, 0.9, 0.3, 0.8, "flagged").not_run_because == ""

    def test_attribution_result_top_on_an_empty_run_is_empty(self):
        assert AttributionResult("global", "permutation").top(5) == []

    def test_permutation_result_emits_its_whole_design_power_disclosure(self):
        """An emptied null is a real outcome; 0.0 for its mean is not.

        ``np.mean`` of an empty array is NaN, and the guard that produces it
        deliberately returns NaN rather than absorbing the RuntimeWarning into a
        number: a null mean of 0.0 reads as a null distribution centred on no
        effect, which is a measurement of a distribution that does not exist.
        """
        payload = ROB.PermutationTestResult(
            NAN, NAN, np.array([]), 0, None, None, "none", groups_omitted=("g2",)
        ).to_dict()
        assert math.isnan(payload["null_mean"]) and math.isnan(payload["null_std"])
        assert payload["significant_at_05"] is None and payload["significant_at_01"] is None
        assert payload["min_attainable_p_value"] is None
        assert payload["detectable_at_05"] is None
        assert payload["groups_omitted"] == ["g2"]

    def test_contingency_result_emits_its_whole_design_power_disclosure(self):
        """A "not significant" from a design that could never reach alpha is not a finding."""
        payload = ROB.ContingencyTestResult(
            "fisher_exact",
            NAN,
            NAN,
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            None,
            min_attainable_p_value=0.0625,
            detectable_at_05=False,
            design_note="no data at these group sizes could have reached 0.05",
            groups_omitted=("g3",),
        ).to_dict()
        for key in ("min_attainable_p_value", "detectable_at_05", "design_note", "groups_omitted"):
            assert key in payload, f"{key} is the disclosure and it was dropped"
        assert payload["detectable_at_05"] is False
        assert payload["min_attainable_p_value"] == 0.0625
        assert payload["design_note"]
        assert payload["groups_omitted"] == ["g3"]
        assert payload["significant_at_05"] is None

    def test_robust_metrics_result_keeps_could_not_tell_apart_from_no(self):
        payload = ROB.RobustMetricsResult(
            "g", 3, 0.5, 0.5, 0.5, 0.5, None, NAN, trim_effective=False
        ).to_dict()
        assert payload["outlier_influence_detected"] is None, (
            "False asserts the robust estimate agrees with the standard one"
        )
        assert math.isnan(payload["divergence_ratio"])
        assert payload["trim_effective"] is False
        # CONTROL: a real negative finding is still a False, not a None.
        measured = ROB.RobustMetricsResult("g", 300, 0.5, 0.48, 0.49, 0.5, False, 0.04).to_dict()
        assert measured["outlier_influence_detected"] is False
        assert measured["trim_effective"] is True

    def test_sensitivity_result_keeps_its_unmeasurable_draw_count(self):
        payload = ROB.SensitivityResult(
            NAN, np.array([]), NAN, NAN, NAN, None, None, "flip", 0, 20
        ).to_dict()
        assert payload["is_robust"] is None
        assert payload["robustness_score"] is None, "0.0 on this scale means FRAGILE"
        assert payload["n_iterations_run"] == 0 and payload["n_unmeasurable"] == 20

    def test_subgroup_audit_result_keeps_the_subgroups_it_dropped(self):
        payload = ROB.SubgroupAuditResult({}, [], None, NAN, None, 0, 0, 5, 2).to_dict()
        assert payload["gerrymandering_detected"] is None
        assert payload["worst_subgroup"] is None
        assert math.isnan(payload["worst_disparity"]), (
            "0.0 is perfect parity across every subgroup, the strongest all-clear here"
        )
        assert payload["n_subgroups_too_small"] == 5
        assert payload["n_subgroups_unmeasurable"] == 2, (
            "the count of subgroups whose metric was undefined is the disclosure"
        )
        assert payload["n_subgroups_analyzed"] == 0


# 11. Not-a-measurement units: a warning class, an exception, TypedDicts, palettes


def test_proxy_scan_incomplete_warning_is_a_warning_and_carries_its_message():
    assert issubclass(DISC.ProxyScanIncompleteWarning, RuntimeWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("could not assess 2 columns", DISC.ProxyScanIncompleteWarning)
    assert isinstance(caught[0].message, DISC.ProxyScanIncompleteWarning)
    assert "could not assess" in str(caught[0].message)


def test_fairness_assertion_error_is_an_assertion_error_that_keeps_its_evidence():
    failed = {"demographic_parity_difference": {"value": 0.3, "threshold": 0.1}}
    every = {"demographic_parity_difference": 0.3}
    with pytest.raises(AssertionError) as info:
        raise INTEG.FairnessAssertionError("2 metrics failed", failed, every)
    assert str(info.value) == "2 metrics failed"
    assert info.value.failed_metrics == failed
    assert info.value.all_metrics == every


@pytest.mark.parametrize(
    "name,required,optional",
    [
        ("MetricStatusEntry", {"metric", "value", "threshold", "status"}, set()),
        ("InsufficientEvidenceGroup", {"group", "n", "tier", "verdict", "reason"}, set()),
        (
            "AssessmentReport",
            {
                "fairness_score",
                "assessable",
                "passed_metrics",
                "failed_metrics",
                "not_assessable_metrics",
                "insufficient_evidence_groups",
                "summary",
            },
            set(),
        ),
        (
            "DataInfo",
            set(),
            {
                "original_size",
                "final_size",
                "n_samples",
                "n_excluded",
                "n_groups",
                "valid_groups",
                "invalid_groups",
                "group_sizes",
                "is_intersectional",
                "missing_strategy",
                "y_std",
            },
        ),
        ("ExplanationsReport", set(), {"metrics", "statistical", "summary"}),
        (
            "FairnessReport",
            {
                "task_type",
                "methodology_version",
                "metrics",
                "group_stats",
                "assessment",
                "data_info",
                "thresholds_used",
            },
            {
                "residual_bias",
                "metrics_with_ci",
                "effect_sizes",
                "statistical_validation",
                "explanations",
            },
        ),
    ],
)
def test_a_report_typed_dict_declares_exactly_the_documented_contract(name, required, optional):
    """These are static annotations: they must not coerce or default anything.

    The public API promise in ``docs/API_STABILITY.md`` is the key set, so the
    key set is what is pinned. A required key silently becoming optional is a
    contract change no runtime check would catch.
    """
    typed = getattr(RTYPES, name)
    assert set(typed.__required_keys__) == required
    assert set(typed.__optional_keys__) == optional


def test_a_report_typed_dict_preserves_the_could_not_check_it_was_given():
    """``fairness_score=None`` is the third state and must not become 0.0."""
    report = RTYPES.AssessmentReport(
        fairness_score=None,
        assessable=False,
        passed_metrics=[],
        failed_metrics=[],
        not_assessable_metrics=[],
        insufficient_evidence_groups=[],
        summary="NOT ASSESSABLE",
    )
    assert report["fairness_score"] is None
    assert report["assessable"] is False
    entry = RTYPES.MetricStatusEntry(
        metric="demographic_parity_difference", value=NAN, threshold=None, status="NOT_ASSESSABLE"
    )
    assert entry["threshold"] is None and math.isnan(entry["value"])


def test_the_two_get_available_styles_are_the_same_answer():
    """The package re-export and the implementation must not drift."""
    import vfairness.evaluation.vfairness_metrics as pkg

    styles = VIZ.get_available_styles()
    assert styles == pkg.get_available_styles()
    assert sorted(styles) == sorted(pkg.get_palettes())
    assert styles, "an empty style list would make every figure fall back silently"


def test_get_palettes_returns_the_palettes_the_styles_name():
    palettes = VIZ.PALETTES
    for style in VIZ.get_available_styles():
        assert style in palettes
        assert "groups" in palettes[style] and palettes[style]["groups"]


def _require_a_plot_backend():
    """plotly and matplotlib are OPTIONAL extras, so a figure test has to skip.

    Only the calls that actually DRAW need this. The label helper below is pure
    string work and is asserted unconditionally, so the disclosure rule is still
    checked in an environment with no plotting backend at all.
    """
    if not (VIZ._check_plotly() or VIZ._check_matplotlib()):
        pytest.skip("preview_palette needs plotly or matplotlib, both optional extras")


@pytest.mark.parametrize("style", VIZ.get_available_styles())
def test_preview_palette_draws_the_style_it_was_asked_for_in_silence(style):
    _require_a_plot_backend()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure = VIZ.preview_palette(style)
    assert figure is not None
    assert not _warned(caught, "not a known visualization style")


def test_preview_palette_says_in_the_figure_that_it_drew_another_palette():
    """A warning does not travel in a saved PNG; the headline has to."""
    if VIZ._check_plotly() or VIZ._check_matplotlib():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VIZ.preview_palette("colorblind_safe_v2")
        assert _warned(caught, "not a known visualization style")
    headline, disclosure = VIZ._palette_preview_labels("colorblind_safe_v2")
    assert "colorblind_safe_v2" not in headline
    assert "colorblind_safe_v2" in disclosure and "not a known style" in disclosure
