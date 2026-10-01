"""Readiness pins for `evaluation.vfairness_metrics.discovery`.

THE DEFECT CLASS: an unmeasured value replaced by a neutral default and then
counted, compared or reported as if it were a measurement. Two findings of that
shape lived here, and both produced the most reassuring verdict the library can
emit over a group the model had selected nobody from.

R-2  `discover_intersectional_groups` read ONLY `result["max_disparity"]` from
     `identify_privileged_groups` and discarded that function's
     `excluded_groups` and `zero_selection_alerts`. Measured 2026-09-09 on 1200
     rows, race {white:1000, black:200} x sex {M,F}, cells white_M 0.528/n=509,
     white_F 0.485/n=491, black_M 0.511/n=180, black_F 0.0/n=20: the layer below
     reported black_F in BOTH transparency lists, and the layer above reported
     n_violations=0, n_not_assessable=0, not_assessable=[], max_disparity=0.0,
     intersectional_issues=[] and "No significant fairness violations detected.
     Continue monitoring." with ZERO warnings on any channel. The true cell gap
     is 0.53.

C-03b `scan_fairness_violations` recorded `not_assessable` only when FEWER THAN
     TWO groups met `min_group_size`. When some groups qualified and one did
     not, the excluded group was dropped from every metric and nothing was
     recorded. Measured on 1000 applicants {white:500, black:300, asian:175,
     native:25} with native never selected: violations=[], not_assessable=[],
     n_not_assessable=0, max_disparity=0.0, demographic_parity_difference=0.0180
     PASS, disparate_impact_ratio=0.9639 PASS, against a true worst gap of 0.498
     and a true four-fifths ratio of 0.0.

`n_not_assessable=0` and `not_assessable=[]` are POSITIVE ASSERTIONS that
nothing was skipped. Every pin below therefore checks the FIELD, not only the
warning: the assessment that found these ran under `python -W ignore`.

Every pin is paired with an over-correction control that asserts MEASURED
numbers, so "make everything could-not-check" cannot pass this file.
"""

import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    discover_intersectional_groups,
    identify_proxy_features,
    rank_fairness_issues,
    scan_fairness_violations,
)
from vfairness.evaluation.vfairness_metrics.intersectional import identify_privileged_groups

ALL_CLEAR = "No significant fairness violations detected. Continue monitoring."


def _frame(spec: List[Tuple[str, str, int, int]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a race x sex frame from (race, sex, n, n_positive) cells."""
    rows = []
    for race, sex, n, n_pos in spec:
        for i in range(n):
            rows.append({"race": race, "sex": sex, "y_pred": 1 if i < n_pos else 0})
    df = pd.DataFrame(rows)
    return df[["race", "sex"]], df["y_pred"].to_numpy()


def _one_attribute(spec: List[Tuple[str, int, int]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a single-attribute frame from (group, n, n_positive) cells."""
    rows = []
    for group, n, n_pos in spec:
        for i in range(n):
            rows.append({"race": group, "y_pred": 1 if i < n_pos else 0})
    df = pd.DataFrame(rows)
    return df[["race"]], df["y_pred"].to_numpy()


# The exact frame the finding was measured on.
_NEVER_SELECTED_CELL = [
    ("white", "M", 509, 269),
    ("white", "F", 491, 238),
    ("black", "M", 180, 92),
    ("black", "F", 20, 0),
]
# Same shape, but every cell large enough to be measured: the intersection is a
# real 0.53 gap and nothing is skipped.
_MEASURABLE_CELL = [
    ("white", "M", 509, 269),
    ("white", "F", 491, 238),
    ("black", "M", 100, 51),
    ("black", "F", 100, 0),
]
# Perfectly fair: every cell rate is exactly 0.5, every cell far above the gate.
_CLEAN = [
    ("white", "M", 300, 150),
    ("white", "F", 300, 150),
    ("black", "M", 200, 100),
    ("black", "F", 200, 100),
]

_PARTIAL_SUBSET = [("white", 500, 249), ("black", 300, 144), ("asian", 175, 86), ("native", 25, 0)]
_ALL_MEASURABLE = [("white", 500, 249), ("black", 300, 144), ("asian", 175, 86), ("native", 200, 0)]
_ALL_MEASURABLE_CLEAN = [("white", 500, 250), ("black", 300, 150), ("asian", 200, 100)]


def _rank(df, y_pred, columns, **kwargs):
    """rank_fairness_issues with warnings suppressed, i.e. `python -W ignore`.

    The FIELDS have to carry the answer on their own; the assessment that found
    both defects ran exactly like this.

    `**kwargs` reaches `y_true`, which the two all-clear controls below pass: an
    all-clear is a statement about FAIRNESS, and since 2026-09-29 it is withheld
    from a run that never attempted the three metrics needing labels.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rank_fairness_issues(
            df,
            y_pred,
            protected_columns=columns,
            auto_detect=False,
            min_group_size=30,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# R-2: an intersectional cell that is never selected must not vanish
# ---------------------------------------------------------------------------


class TestR2NeverSelectedIntersectionalCell:
    def test_the_layer_below_really_does_report_the_dropped_cell(self):
        """Pins the premise: the honesty exists one layer down, and the defect
        was the discard at the call site, not a missing measurement."""
        df, y_pred = _frame(_NEVER_SELECTED_CELL)
        cell = (df["race"].astype(str) + "_" + df["sex"].astype(str)).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            low = identify_privileged_groups(y_pred, y_pred, cell, min_group_size=30)

        assert [c["group"] for c in low["excluded_groups"]] == ["black_F"]
        assert low["excluded_groups"][0]["size"] == 20
        assert low["excluded_groups"][0]["positive_count"] == 0
        assert [a["group"] for a in low["zero_selection_alerts"]] == ["black_F"]
        # The disparity it CAN compute is over the three surviving cells only.
        assert low["max_disparity"] == pytest.approx(0.0437621789, abs=1e-9)

    def test_discover_carries_the_dropped_cell_to_its_caller(self):
        df, y_pred = _frame(_NEVER_SELECTED_CELL)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["race", "sex"], y_pred, min_group_size=30)

        assert isinstance(out, list), "the return type changed and callers would break"
        # The combination itself is under min_disparity, so there is no result
        # dict to hang the cell on: this is exactly why it needs its own channel.
        assert list(out) == []
        assert len(out.not_assessable) == 1
        entry = out.not_assessable[0]
        assert "black_F" in entry
        assert "n=20" in entry
        assert "NEVER SELECTED" in entry
        assert [a["group"] for a in out.zero_selection_alerts] == ["black_F"]
        assert out.zero_selection_alerts[0]["size"] == 20
        assert out.zero_selection_alerts[0]["analysed"] is False
        assert any("NEVER SELECTED" in str(c.message) for c in caught)

    def test_rank_refuses_the_all_clear_and_names_the_cell(self):
        """The worst user-visible verdict in the assessment. Under `-W ignore`
        the fields are the only channel left, so they are what is pinned."""
        df, y_pred = _frame(_NEVER_SELECTED_CELL)
        analysis = _rank(df, y_pred, ["race", "sex"])
        summary = analysis["summary"]

        assert summary["n_not_assessable"] == 1
        assert summary["not_assessable"] and "black_F" in summary["not_assessable"][0]
        assert summary["n_zero_selection_alerts"] == 1
        assert summary["zero_selection_alerts"][0]["group"] == "black_F"
        assert summary["zero_selection_alerts"][0]["size"] == 20
        # Both attributes WERE scanned; neither was skipped outright.
        assert summary["n_attributes_checked"] == 2
        assert summary["n_violations"] == 0
        assert math.isnan(summary["max_disparity"]), (
            f"max_disparity was {summary['max_disparity']!r}; 0.0 is the value a "
            "perfectly fair run gets, and a cell nobody measured is not that"
        )

        text = " ".join(analysis["recommendations"])
        assert ALL_CLEAR not in text
        assert "NEVER SELECTED" in text
        assert "black_F" in text
        assert "COULD NOT CHECK" in text

    def test_control_a_measurable_intersection_reports_the_real_number(self):
        """OVER-CORRECTION CONTROL. Same shape, black_F at n=100 instead of 20,
        so the cell is measured rather than dropped. The verdict must be a
        MEASURED 0.5285 gap with nothing marked not-assessable, not another
        could-not-check."""
        df, y_pred = _frame(_MEASURABLE_CELL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = discover_intersectional_groups(df, ["race", "sex"], y_pred, min_group_size=30)

        assert out.not_assessable == []
        assert len(out) == 1
        found = out[0]
        # white_M 269/509 = 0.528487..., black_F 0/100 = 0.0
        assert found["disparity"] == pytest.approx(0.528487229, abs=1e-9)
        assert found["privileged_group"] == "white_M"
        assert found["disadvantaged_group"] == "black_F"
        assert found["n_groups"] == 4
        assert found["cells_not_assessable"] == []
        assert found["disparity_over_subset"] is False
        # The cell IS in the analysis, and is still flagged: analysed, n=100.
        assert [a["group"] for a in found["zero_selection_alerts"]] == ["black_F"]
        assert found["zero_selection_alerts"][0]["analysed"] is True

        analysis = _rank(df, y_pred, ["race", "sex"])
        assert analysis["summary"]["n_not_assessable"] == 0
        assert analysis["summary"]["not_assessable"] == []
        # summary.max_disparity ranks the SINGLE-attribute violations: race is
        # white 507/1000 = 0.507 against black 51/200 = 0.255, a measured 0.252.
        assert analysis["summary"]["max_disparity"] == pytest.approx(0.252, abs=1e-9)
        assert analysis["intersectional_issues"][0]["disparity"] == pytest.approx(
            0.528487229, abs=1e-9
        )
        assert ALL_CLEAR not in " ".join(analysis["recommendations"])

    def test_control_a_genuinely_clean_run_still_gets_the_all_clear(self):
        """OVER-CORRECTION CONTROL. Every cell rate is exactly 0.5 and every
        cell is far above the gate, so there is nothing to report and nothing
        that could not be checked. The measured numbers are asserted, not
        membership of a set."""
        df, y_pred = _frame(_CLEAN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = discover_intersectional_groups(df, ["race", "sex"], y_pred, min_group_size=30)
        assert list(out) == []
        assert out.not_assessable == []
        assert out.zero_selection_alerts == []
        assert not [c for c in caught if "discover_intersectional_groups" in str(c.message)]

        # y_true == y_pred: a perfect model, so all four registered metrics are
        # computed and each is exactly 0.0. See _rank.
        analysis = _rank(df, y_pred, ["race", "sex"], y_true=y_pred)
        summary = analysis["summary"]
        assert summary["n_violations"] == 0
        assert summary["n_not_assessable"] == 0
        assert summary["not_assessable"] == []
        assert summary["n_zero_selection_alerts"] == 0
        assert summary["n_attributes_checked"] == 2
        assert summary["max_disparity"] == 0.0
        assert summary["n_metrics_not_attempted"] == 0
        assert analysis["recommendations"] == [ALL_CLEAR]


# ---------------------------------------------------------------------------
# C-03b: some groups qualify, one does not, and the one that does not vanishes
# ---------------------------------------------------------------------------


class TestC03bPartiallyScannedAttribute:
    def test_scan_records_the_excluded_group_it_could_not_measure(self):
        df, y_pred = _one_attribute(_PARTIAL_SUBSET)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            violations = scan_fairness_violations(
                df, y_pred, protected_columns=["race"], auto_detect=False, min_group_size=30
            )

        assert isinstance(violations, list), "the return type changed and callers would break"
        assert list(violations) == []
        assert len(violations.not_assessable) == 1
        entry = violations.not_assessable[0]
        assert entry.startswith("race:")
        assert "'native'" in entry
        assert "n=25" in entry
        assert "NEVER SELECTED" in entry
        # 'race' was scanned over a subset, so it is NOT one of the columns that
        # were skipped outright: those two states must stay apart.
        assert violations.skipped_columns == []
        assert [a["group"] for a in violations.zero_selection_alerts] == ["native"]
        assert violations.zero_selection_alerts[0]["size"] == 25
        assert any("SUBSET" in str(c.message) for c in caught)

    def test_rank_refuses_the_all_clear_for_a_partially_scanned_attribute(self):
        df, y_pred = _one_attribute(_PARTIAL_SUBSET)
        analysis = _rank(df, y_pred, ["race"])
        summary = analysis["summary"]

        assert summary["n_not_assessable"] == 1
        assert "'native'" in summary["not_assessable"][0]
        assert summary["n_zero_selection_alerts"] == 1
        assert summary["zero_selection_alerts"][0]["group"] == "native"
        # The attribute WAS checked, over three of its four groups. Counting it
        # as unchecked would be the opposite error.
        assert summary["n_attributes_checked"] == 1
        assert summary["n_attributes_requested"] == 1
        assert summary["n_violations"] == 0
        assert math.isnan(summary["max_disparity"])

        text = " ".join(analysis["recommendations"])
        assert ALL_CLEAR not in text
        assert "NEVER SELECTED" in text
        assert "COULD NOT CHECK" in text

    def test_control_the_same_data_with_a_measurable_native_group(self):
        """OVER-CORRECTION CONTROL. native at n=200 instead of 25: the group is
        analysed, the disparity is a MEASURED 0.498, and nothing is
        not-assessable. Pins that the fix did not turn measurable data into a
        could-not-check."""
        df, y_pred = _one_attribute(_ALL_MEASURABLE)
        analysis = _rank(df, y_pred, ["race"])
        summary = analysis["summary"]

        assert summary["n_not_assessable"] == 0
        assert summary["not_assessable"] == []
        assert summary["n_violations"] == 1
        assert summary["max_disparity"] == pytest.approx(0.498, abs=1e-9)
        assert summary["critical_violations"] == 1
        assert summary["n_attributes_checked"] == 1
        violation = analysis["violations"][0]
        assert violation["metric"] == "demographic_parity_difference"
        assert violation["value"] == pytest.approx(0.498, abs=1e-9)
        assert violation["disadvantaged_group"] == "native"
        assert violation["privileged_group"] == "white"
        # 'native' was ANALYSED, so it is not a dropped cell: the 0.498 gap
        # carries it, and there is nothing to alert about separately here.
        assert summary["n_zero_selection_alerts"] == 0
        text = " ".join(analysis["recommendations"])
        assert ALL_CLEAR not in text
        assert "CRITICAL" in text

    def test_control_a_clean_single_attribute_still_gets_the_all_clear(self):
        """OVER-CORRECTION CONTROL. Three groups, all above the gate, all at
        rate exactly 0.5."""
        df, y_pred = _one_attribute(_ALL_MEASURABLE_CLEAN)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            violations = scan_fairness_violations(
                df, y_pred, protected_columns=["race"], auto_detect=False, min_group_size=30
            )
        assert list(violations) == []
        assert violations.not_assessable == []
        assert violations.zero_selection_alerts == []
        assert not [c for c in caught if "SUBSET" in str(c.message)]

        # y_true == y_pred: a perfect model, so all four registered metrics are
        # computed and each is exactly 0.0. See _rank.
        analysis = _rank(df, y_pred, ["race"], y_true=y_pred)
        summary = analysis["summary"]
        assert summary["n_violations"] == 0
        assert summary["n_not_assessable"] == 0
        assert summary["n_zero_selection_alerts"] == 0
        assert summary["n_attributes_checked"] == 1
        assert summary["max_disparity"] == 0.0
        assert summary["n_metrics_not_attempted"] == 0
        assert analysis["recommendations"] == [ALL_CLEAR]

    def test_control_an_excluded_group_that_was_selected_is_not_a_zero_alert(self):
        """A small group that IS selected is still not-assessable (it left every
        metric) but it is NOT a zero-selection alert. Pins the two states apart
        instead of collapsing everything small into the loudest bucket."""
        df, y_pred = _one_attribute([("white", 500, 249), ("black", 300, 144), ("native", 25, 12)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            violations = scan_fairness_violations(
                df, y_pred, protected_columns=["race"], auto_detect=False, min_group_size=30
            )
        assert len(violations.not_assessable) == 1
        assert "'native'" in violations.not_assessable[0]
        assert "n=25" in violations.not_assessable[0]
        assert "NEVER SELECTED" not in violations.not_assessable[0]
        assert violations.zero_selection_alerts == []

        analysis = _rank(df, y_pred, ["race"])
        assert analysis["summary"]["n_not_assessable"] == 1
        assert analysis["summary"]["n_zero_selection_alerts"] == 0
        text = " ".join(analysis["recommendations"])
        assert "COULD NOT CHECK" in text
        assert "NEVER SELECTED" not in text
        assert ALL_CLEAR not in text

    def test_sibling_an_unreadable_proxy_column_reaches_the_summary(self):
        """SIBLING of the same shape, found in this file: `n_proxy_warnings: 0`
        was a positive assertion that the proxy scan was complete, while
        `identify_proxy_features` returned a bare list and its `unassessable`
        columns lived only on the warning channel. A JSON column stored as list
        objects is a PERFECT proxy here (w -> [1,0], b -> [0,1]) that no
        association measure can read."""
        race = np.array(["w"] * 200 + ["b"] * 200)
        y_pred = np.where(race == "w", 1, 0)
        df = pd.DataFrame(
            {
                "race": race,
                "json_feature": [[1, 0] if r == "w" else [0, 1] for r in race],
                "noise": np.random.default_rng(0).random(400),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proxies = identify_proxy_features(df, "race")
        assert isinstance(proxies, list)
        assert list(proxies) == []
        assert len(proxies.not_assessable) == 1
        assert "json_feature" in proxies.not_assessable[0]

        analysis = _rank(df, y_pred, ["race"])
        assert analysis["summary"]["n_proxy_warnings"] == 0
        assert analysis["summary"]["n_not_assessable"] == 1
        assert "json_feature" in analysis["summary"]["not_assessable"][0]
        assert "COULD NOT CHECK" in " ".join(analysis["recommendations"])

    def test_control_a_readable_proxy_is_measured_not_marked_unassessable(self):
        """OVER-CORRECTION CONTROL for the sibling. The same frame with a
        numeric column that splits exactly on race: a MEASURED correlation of
        1.0, one proxy warning, and nothing not-assessable."""
        race = np.array(["w"] * 200 + ["b"] * 200)
        y_pred = np.where(race == "w", 1, 0)
        df = pd.DataFrame(
            {
                "race": race,
                "readable_proxy": np.where(race == "w", 1.0, 0.0),
                "noise": np.random.default_rng(0).random(400),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proxies = identify_proxy_features(df, "race")
        assert proxies.not_assessable == []
        assert [p["column"] for p in proxies] == ["readable_proxy"]
        assert proxies[0]["abs_correlation"] == pytest.approx(1.0, abs=1e-12)
        assert proxies[0]["risk_level"] == "high"
        assert not [c for c in caught if "identify_proxy_features" in str(c.message)]

        analysis = _rank(df, y_pred, ["race"])
        assert analysis["summary"]["n_proxy_warnings"] == 1
        assert analysis["summary"]["n_not_assessable"] == 0
        assert analysis["summary"]["not_assessable"] == []
        assert "PROXY RISK" in " ".join(analysis["recommendations"])

    def test_a_column_skipped_entirely_is_still_counted_as_unchecked(self):
        """Regression guard on the count that C-03 fixed: an attribute with
        fewer than two qualifying groups is not "checked over a subset", it is
        not checked at all."""
        df, y_pred = _one_attribute([("white", 280, 280), ("native", 20, 0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            violations = scan_fairness_violations(
                df, y_pred, protected_columns=["race"], auto_detect=False, min_group_size=30
            )
        assert violations.skipped_columns == ["race"]

        analysis = _rank(df, y_pred, ["race"])
        assert analysis["summary"]["n_attributes_checked"] == 0
        assert analysis["summary"]["n_not_assessable"] == 1
        assert math.isnan(analysis["summary"]["max_disparity"])
        assert ALL_CLEAR not in " ".join(analysis["recommendations"])
