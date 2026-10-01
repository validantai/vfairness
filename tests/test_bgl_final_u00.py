"""Beta Go-Live final batch u00: the two capabilities the census NEVER REACHED.

Both were carried as UNPROVEN, which is a positive statement that nothing was
known about them, not a clean bill. Both have now been run, and both FABRICATED:
they handed back the most reassuring verdict they can emit for input where
nothing had been measured at all.

WHAT WAS MEASURED, 2026-09-17
-----------------------------
`scan_fairness_violations` (ledger row auto_discovery). Nine degenerate inputs.
Six were already honest: one group only, zero rows, an all-NaN attribute, an
undefined TPR (y_true single class), and both saturated-prediction frames. Three
were not, and each returned `violations=[]` with `not_assessable=[]`,
`skipped_columns=[]` and nothing on any field a reader could tell it from a clean
scan by:

  S1  a requested column that is not in the DataFrame. `protected_columns=['sex']`
      against a frame holding a real 0.60 selection-rate gap on `race`:
      `rank_fairness_issues` reported n_attributes_checked=1, n_not_assessable=0,
      max_disparity=0.0 and "No significant fairness violations detected.
      Continue monitoring." A renamed or misspelt column bought a clean bill of
      health for an attribute nobody had looked at.
  S2  `auto_detect=True` on a frame the detector finds no candidate in.
      `columns_to_check` is empty, the scan loop never runs, and an examination
      of NOTHING is returned as an empty violation list.
  S3  a metric that raises. The RuntimeWarning was the only channel, so under
      `python -W ignore` (how the assessment that found the sibling defects ran)
      a metric that crashed was indistinguishable from one that passed. A
      three-class `y_pred`, which every metric here refuses, made the whole scan
      compute nothing and still report max_disparity=0.0 and the all-clear.

`CausalFairnessGraph` (ledger row causal_fairness_graph). Six graphs. Four of
them cannot be traced at all, because no variable carries `protected=True`, or
none carries `outcome=True`, or the only protected variable is also the only
outcome. Every one returned `discrimination_paths() == []`,
`has_direct_discrimination() is False` and `summary()['severity'] == 'info'`:
byte for byte the answer a graph gets when it HAS both roles and genuinely no
pathway between them. The most realistic of the four is a graph built only from
`add_edge` calls, because `add_edge` creates missing variables with every role
defaulting to False: `add_edge('gender', 'hiring')` on its own produces a graph
containing literal direct discrimination and grades it "info, no direct
discrimination".

THE FIX, in both: three states, never two. A refusal is recorded on a FIELD
(`not_assessable`, `assessable`, a None verdict, a NaN max_disparity), not only
in a warning, because the field is the only channel left under `-W ignore`.

Every pin below is paired with a healthy-data control that still asserts a
MEASURED number or a MEASURED verdict, so "make everything could-not-check"
cannot pass this file.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    rank_fairness_issues,
    scan_fairness_violations,
)
from vfairness.operations.causal.graph import CausalFairnessGraph

ALL_CLEAR = "No significant fairness violations detected. Continue monitoring."


def _one_attribute(spec: List[Tuple[str, int, int]]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a single-attribute frame from (group, n, n_positive) cells."""
    rows = []
    for group, n, n_pos in spec:
        for i in range(n):
            rows.append({"race": group, "y_pred": 1 if i < n_pos else 0})
    df = pd.DataFrame(rows)
    return df[["race"]], df["y_pred"].to_numpy()


# white 400/500 = 0.80, black 100/500 = 0.20, a real and findable 0.60 gap, and
# both groups an order of magnitude above the min_group_size=30 default. Every
# threshold in the signature is checked by the healthy control below: if the
# fixture were simply too small, THAT test would fail rather than the refusals
# looking honest for free.
_REAL_GAP = [("white", 500, 400), ("black", 500, 100)]
# Same shape, both rates exactly 0.50: a genuinely clean run.
_CLEAN = [("white", 500, 250), ("black", 500, 250)]


def _rank(df, y_pred, columns, **kwargs):
    """rank_fairness_issues with warnings suppressed, i.e. `python -W ignore`.

    The FIELDS have to carry the answer on their own.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rank_fairness_issues(
            df, y_pred, protected_columns=columns, auto_detect=False, **kwargs
        )


def _scan(df, y_pred, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = scan_fairness_violations(df, y_pred, **kwargs)
    return out, [str(c.message) for c in caught]


# ---------------------------------------------------------------------------
# auto_discovery: the healthy control, first, so the fixture is proven to reach
# the code with data before any refusal is read as honesty.
# ---------------------------------------------------------------------------


class TestScanHealthyControl:
    def test_a_real_disparity_is_measured_and_graded(self):
        df, y_pred = _one_attribute(_REAL_GAP)
        out, caught = _scan(df, y_pred, protected_columns=["race"], auto_detect=False)

        assert len(out) == 1
        v = out[0]
        assert v.attribute == "race"
        assert v.metric == "demographic_parity_difference"
        assert v.disparity == pytest.approx(0.6, abs=1e-9)
        assert v.severity == "critical"
        assert v.privileged_group == "white"
        assert v.disadvantaged_group == "black"
        # Nothing refused: the fixture really is big enough.
        assert out.not_assessable == []
        assert out.skipped_columns == []
        # The ONE warning a label-free scan carries since 2026-09-29: this fixture
        # has no y_true, so three of the four registered metrics were never
        # attempted, and that is now said out loud on a field
        # (`metrics_not_attempted`) and in this warning. Nothing about this
        # fixture's SIZE or COVERAGE warns, which is what the control is for.
        assert [c for c in caught if "WITHOUT labels" not in c] == []
        assert out.metrics_not_attempted == [
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "predictive_parity_difference",
        ]

        summary = _rank(df, y_pred, ["race"])["summary"]
        assert summary["n_attributes_checked"] == 1
        assert summary["n_not_assessable"] == 0
        # The measured 0.60 is untouched by the scope disclosure: a real finding
        # is never withdrawn by something else the run could not ask.
        assert summary["max_disparity"] == pytest.approx(0.6, abs=1e-9)
        assert summary["critical_violations"] == 1
        assert summary["n_metrics_not_attempted"] == 3

    def test_a_genuinely_clean_run_still_gets_the_all_clear(self):
        """OVER-CORRECTION CONTROL. A measured 0.0 is a measurement, and it must
        keep the all-clear. This is the assertion that fails if the fixes below
        are widened into refusing everything.

        Labels were added to the fixture on 2026-09-29, and nothing else changed.
        `y_true == y_pred` is a perfect model, so all FOUR registered metrics are
        computed and every one of them is exactly 0.0, which is what "a genuinely
        clean run" has to mean for an all-clear about FAIRNESS: without y_true
        only demographic parity is ever attempted, and the all-clear is now
        withheld for a run that never asked the other three questions (a model
        perfect for one group and inverted for another has identical selection
        rates). The control is strictly stronger: it now also proves the scope
        disclosure does NOT fire when the questions were asked.
        """
        df, y_pred = _one_attribute(_CLEAN)
        out, caught = _scan(
            df, y_pred, y_true=y_pred, protected_columns=["race"], auto_detect=False
        )

        assert list(out) == []
        assert out.not_assessable == []
        assert out.skipped_columns == []
        assert out.metrics_not_attempted == []
        assert caught == []

        analysis = _rank(df, y_pred, ["race"], y_true=y_pred)
        assert analysis["summary"]["max_disparity"] == 0.0
        assert analysis["summary"]["n_not_assessable"] == 0
        assert analysis["summary"]["n_metrics_not_attempted"] == 0
        assert analysis["recommendations"] == [ALL_CLEAR]


# ---------------------------------------------------------------------------
# S1: a requested column the DataFrame does not have
# ---------------------------------------------------------------------------


class TestS1RequestedColumnAbsent:
    def test_scan_records_the_column_it_never_looked_at(self):
        df, y_pred = _one_attribute(_REAL_GAP)
        out, caught = _scan(df, y_pred, protected_columns=["sex"], auto_detect=False)

        assert list(out) == []
        assert out.skipped_columns == ["sex"], (
            "the column was skipped silently; skipped_columns is what "
            "n_attributes_checked is computed from"
        )
        assert len(out.not_assessable) == 1
        assert "sex" in out.not_assessable[0]
        assert "not in the DataFrame" in out.not_assessable[0]
        assert any("was NOT scanned" in m for m in caught)

    def test_rank_refuses_the_all_clear_for_a_column_it_never_had(self):
        """The user-visible surface, under `-W ignore`. The frame carries a real
        0.60 gap on `race`; the caller asked about `sex`, which does not exist."""
        df, y_pred = _one_attribute(_REAL_GAP)
        summary = _rank(df, y_pred, ["sex"])["summary"]

        assert summary["n_attributes_requested"] == 1
        assert summary["n_attributes_checked"] == 0, (
            "one attribute was requested and none was examined; counting it as "
            "checked is the fabricated half of the verdict"
        )
        assert summary["n_not_assessable"] == 1
        assert "sex" in summary["not_assessable"][0]
        assert math.isnan(summary["max_disparity"]), (
            f"max_disparity was {summary['max_disparity']!r}; 0.0 is the value a "
            "perfectly fair run gets, and an attribute nobody read is not that"
        )

        recs = " ".join(_rank(df, y_pred, ["sex"])["recommendations"])
        assert ALL_CLEAR not in recs
        assert "COULD NOT CHECK" in recs

    def test_the_present_column_in_the_same_request_is_still_measured(self):
        """OVER-CORRECTION CONTROL. One absent column must not suppress the real
        finding on the column that IS there."""
        df, y_pred = _one_attribute(_REAL_GAP)
        out, _ = _scan(df, y_pred, protected_columns=["race", "sex"], auto_detect=False)

        assert [v.attribute for v in out] == ["race"]
        assert out[0].disparity == pytest.approx(0.6, abs=1e-9)
        assert out.skipped_columns == ["sex"]

        summary = _rank(df, y_pred, ["race", "sex"])["summary"]
        assert summary["n_attributes_checked"] == 1
        assert summary["n_attributes_requested"] == 2
        # The scan's own entry, plus the two the intersectional pass already
        # recorded for the same absent column, so this is >= 1 rather than == 1.
        assert summary["n_not_assessable"] >= 1
        assert any("not in the DataFrame" in e for e in summary["not_assessable"]), summary[
            "not_assessable"
        ]
        # A real violation exists, so max_disparity is the MEASURED number, not NaN.
        assert summary["max_disparity"] == pytest.approx(0.6, abs=1e-9)
        assert summary["critical_violations"] == 1


# ---------------------------------------------------------------------------
# S2: auto-detection found no candidate, so nothing was scanned
# ---------------------------------------------------------------------------


def _no_candidate_frame() -> Tuple[pd.DataFrame, np.ndarray]:
    """A frame with no column the protected-attribute detector will nominate."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "widget_id": np.arange(400),
            "sprocket_len": rng.rand(400),
            "torque_nm": rng.rand(400) * 10,
        }
    )
    y_pred = np.r_[np.ones(200), np.zeros(200)].astype(int)
    return df, y_pred


class TestS2NothingToScan:
    def test_detection_really_does_find_nothing_here(self):
        """Pins the premise, so this test cannot pass for the wrong reason."""
        from vfairness.evaluation.vfairness_metrics.discovery import (
            detect_protected_attributes,
        )

        df, _ = _no_candidate_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert detect_protected_attributes(df, min_confidence=0.3) == []

    def test_scan_says_it_examined_nothing(self):
        df, y_pred = _no_candidate_frame()
        out, caught = _scan(df, y_pred, auto_detect=True)

        assert list(out) == []
        assert len(out.not_assessable) == 1
        assert "no attribute was scanned at all" in out.not_assessable[0]
        assert "auto-detection" in out.not_assessable[0]
        assert any("scanned NO attribute" in m for m in caught)

    def test_an_auto_detected_frame_is_still_really_scanned(self):
        """OVER-CORRECTION CONTROL. The same call on a frame the detector DOES
        nominate a column in must produce the measured 0.60 gap, not a refusal."""
        df, y_pred = _one_attribute(_REAL_GAP)
        out, _ = _scan(df, y_pred, auto_detect=True)

        assert out.not_assessable == []
        assert [v.attribute for v in out] == ["race"]
        assert out[0].disparity == pytest.approx(0.6, abs=1e-9)


# ---------------------------------------------------------------------------
# S3: a metric that raised is not a metric that passed
# ---------------------------------------------------------------------------


class TestS3MetricRaised:
    def test_scan_records_the_metric_that_crashed(self):
        """A three-class `y_pred`. Every metric in the registry refuses it, so
        the scan computes nothing at all."""
        df, y_pred = _one_attribute(_REAL_GAP)
        y_pred = y_pred.copy()
        y_pred[:50] = 2

        out, caught = _scan(df, y_pred, protected_columns=["race"], auto_detect=False)

        assert list(out) == []
        assert len(out.not_assessable) == 1
        entry = out.not_assessable[0]
        assert entry.startswith("race/demographic_parity_difference:")
        assert "raised" in entry
        assert "InvalidDataError" in entry
        # The column itself was reachable, so it is NOT counted as skipped whole.
        assert out.skipped_columns == []
        assert any("could not be computed, not because it passed" in m for m in caught)

    def test_rank_refuses_the_all_clear_when_every_metric_crashed(self):
        df, y_pred = _one_attribute(_REAL_GAP)
        y_pred = y_pred.copy()
        y_pred[:50] = 2

        analysis = _rank(df, y_pred, ["race"])
        summary = analysis["summary"]

        assert summary["n_violations"] == 0
        assert summary["n_not_assessable"] == 1
        assert math.isnan(summary["max_disparity"])
        text = " ".join(analysis["recommendations"])
        assert ALL_CLEAR not in text
        assert "COULD NOT CHECK" in text

    def test_a_length_mismatch_is_recorded_too(self):
        df, y_pred = _one_attribute(_REAL_GAP)
        out, _ = _scan(df, y_pred[:500], protected_columns=["race"], auto_detect=False)

        assert list(out) == []
        assert len(out.not_assessable) == 1
        assert "Inconsistent array lengths" in out.not_assessable[0]


# ---------------------------------------------------------------------------
# The six degenerate inputs that were ALREADY honest. Pinned so a later change
# cannot quietly turn one of them back into a clean-looking empty list.
# ---------------------------------------------------------------------------


class TestScanDegenerateInputsAlreadyHonest:
    def test_one_group_only(self):
        df, y_pred = _one_attribute([("white", 500, 400)])
        out, _ = _scan(df, y_pred, protected_columns=["race"], auto_detect=False)
        assert out.skipped_columns == ["race"]
        assert "only 1 group(s)" in out.not_assessable[0]

    def test_zero_rows(self):
        df = pd.DataFrame({"race": pd.Series(dtype=object)})
        out, _ = _scan(df, np.array([]), protected_columns=["race"], auto_detect=False)
        assert out.skipped_columns == ["race"]
        assert "only 0 group(s)" in out.not_assessable[0]

    def test_protected_column_all_nan(self):
        df = pd.DataFrame({"race": [np.nan] * 400})
        y_pred = np.r_[np.ones(200), np.zeros(200)].astype(int)
        out, _ = _scan(df, y_pred, protected_columns=["race"], auto_detect=False)
        assert out.skipped_columns == ["race"]
        assert out.not_assessable

    def test_undefined_tpr_is_recorded_as_uncomputed(self):
        df, y_pred = _one_attribute(_REAL_GAP)
        y_true = np.zeros(len(y_pred), dtype=int)
        out, _ = _scan(
            df,
            y_pred,
            y_true=y_true,
            protected_columns=["race"],
            auto_detect=False,
            metrics=["equal_opportunity_difference"],
        )
        assert list(out) == []
        assert "could not be computed" in out.not_assessable[0]
        assert "nan" in out.not_assessable[0]

    @pytest.mark.parametrize("n_pos", [0, 500])
    def test_a_saturated_prediction_is_a_measured_zero_not_a_refusal(self, n_pos):
        """A model that selects everybody, or nobody, really does have a zero
        selection-rate gap. That 0.0 is a MEASUREMENT and must not be converted
        into a could-not-check: an early return of that kind once swallowed a
        genuine zero-outcome-rate finding elsewhere in this library."""
        df, y_pred = _one_attribute([("white", 500, n_pos), ("black", 500, n_pos)])
        out, caught = _scan(df, y_pred, protected_columns=["race"], auto_detect=False)
        assert list(out) == []
        assert out.not_assessable == []
        assert out.skipped_columns == []
        # The ONE warning a label-free scan carries since 2026-09-29: this fixture
        # has no y_true, so three of the four registered metrics were never
        # attempted, and that is now said out loud on a field
        # (`metrics_not_attempted`) and in this warning. Nothing about this
        # fixture's SIZE or COVERAGE warns, which is what the control is for.
        assert [c for c in caught if "WITHOUT labels" not in c] == []


# ---------------------------------------------------------------------------
# causal_fairness_graph
# ---------------------------------------------------------------------------


def _hiring_graph() -> CausalFairnessGraph:
    """gender -> hiring direct, gender -> education -> hiring indirect,
    gender -> zip -> hiring proxy."""
    g = CausalFairnessGraph()
    g.add_variable("gender", protected=True)
    g.add_variable("education", mediator=True)
    g.add_variable("zip", proxy=True)
    g.add_variable("hiring", outcome=True)
    g.add_edge("gender", "education")
    g.add_edge("education", "hiring")
    g.add_edge("gender", "hiring")
    g.add_edge("gender", "zip")
    g.add_edge("zip", "hiring")
    return g


def _traced_but_clean_graph() -> CausalFairnessGraph:
    """BOTH roles declared and genuinely no pathway between them. This is the
    graph whose answer the four unassessable ones used to be identical to."""
    g = CausalFairnessGraph()
    g.add_variable("gender", protected=True)
    g.add_variable("hiring", outcome=True)
    g.add_variable("tenure")
    g.add_edge("tenure", "hiring")
    return g


class TestCausalGraphHealthyControl:
    def test_a_real_structure_is_traced_and_classified(self):
        g = _hiring_graph()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            paths = g.discrimination_paths()
            summary = g.summary()
            verdict = g.has_direct_discrimination()

        kinds = {tuple(p.path): p.kind for p in paths}
        assert kinds[("gender", "hiring")] == "direct"
        assert kinds[("gender", "education", "hiring")] == "indirect"
        assert kinds[("gender", "zip", "hiring")] == "proxy"
        assert verdict is True
        assert summary["severity"] == "high"
        assert summary["counts"] == {"direct": 1, "indirect": 1, "proxy": 1}
        assert summary["assessable"] is True
        assert summary["not_assessable"] == []
        assert g.not_assessable() == []
        assert caught == []

    def test_a_traced_graph_with_no_pathway_keeps_its_clean_verdict(self):
        """OVER-CORRECTION CONTROL. A graph that CAN be read and holds no
        pathway is a measurement of "no pathway", and must stay False / info."""
        g = _traced_but_clean_graph()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            paths = g.discrimination_paths()
            summary = g.summary()
            verdict = g.has_direct_discrimination()

        assert list(paths) == []
        assert paths.not_assessable == []
        assert verdict is False, "False here is a real finding, not a default"
        assert summary["severity"] == "info"
        assert summary["assessable"] is True
        assert summary["has_direct_discrimination"] is False
        assert g.not_assessable() == []
        assert caught == []


def _unassessable_graphs():
    empty = CausalFairnessGraph()

    no_protected = CausalFairnessGraph()
    no_protected.add_variable("gender")  # role forgotten
    no_protected.add_variable("hiring", outcome=True)
    no_protected.add_edge("gender", "hiring")

    no_outcome = CausalFairnessGraph()
    no_outcome.add_variable("gender", protected=True)
    no_outcome.add_variable("hiring")
    no_outcome.add_edge("gender", "hiring")

    # The realistic one: `add_edge` creates missing variables with every role
    # False, so a graph authored entirely from edges declares no role at all.
    edges_only = CausalFairnessGraph()
    edges_only.add_edge("gender", "hiring")

    same_node = CausalFairnessGraph()
    same_node.add_variable("gender", protected=True, outcome=True)

    return [
        ("empty graph", empty),
        ("no protected role", no_protected),
        ("no outcome role", no_outcome),
        ("built only from add_edge", edges_only),
        ("protected is also the outcome", same_node),
    ]


@pytest.mark.parametrize(
    "label,graph", _unassessable_graphs(), ids=lambda v: v if isinstance(v, str) else ""
)
class TestCausalGraphRefusesWhatItCannotTrace:
    def test_the_verdict_is_none_never_false(self, label, graph):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verdict = graph.has_direct_discrimination()
        assert verdict is None, (
            f"{label}: has_direct_discrimination() returned {verdict!r}. False is "
            "the answer a graph earns by being traced and found free of direct "
            "pathways, and this one was never traced"
        )

    def test_the_summary_grades_it_unknown_and_says_why(self, label, graph):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = graph.summary()
        assert summary["severity"] == "unknown", (
            f"{label}: severity was {summary['severity']!r}; 'info' is the grade "
            "a traced graph with no pathway gets"
        )
        assert summary["assessable"] is False
        assert summary["not_assessable"], f"{label}: no reason was recorded"
        assert summary["has_direct_discrimination"] is None

    def test_the_pathway_list_carries_the_reason_and_warns(self, label, graph):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            paths = graph.discrimination_paths()
        assert list(paths) == []
        assert paths.not_assessable, f"{label}: an empty list with no reason on it"
        assert any("could not be assessed" in str(c.message) for c in caught), (
            f"{label}: no warning on any channel"
        )

    def test_the_summary_is_json_friendly(self, label, graph):
        import json

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            json.dumps(graph.summary())


def test_the_edges_only_graph_becomes_assessable_once_roles_are_declared():
    """OVER-CORRECTION CONTROL and the mechanism, in one. The SAME two nodes,
    with the two roles added, produce a measured DIRECT discrimination finding."""
    g = CausalFairnessGraph()
    g.add_edge("gender", "hiring")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert g.has_direct_discrimination() is None

    g.add_variable("gender", protected=True)
    g.add_variable("hiring", outcome=True)
    g.add_edge("gender", "hiring")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert g.has_direct_discrimination() is True
        summary = g.summary()
    assert summary["severity"] == "high"
    assert summary["counts"]["direct"] == 1
    assert summary["assessable"] is True
    assert caught == []
