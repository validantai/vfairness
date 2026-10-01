"""Surface grading batch g016: the vfairness MCP tool layer (``vfairness.mcp.tools``).

These tools ARE the boundary an LLM/agent client reads. The dict each one
returns is the whole of what crosses it: a ``UserWarning`` raised inside the
library goes to the server process's stderr and reaches no client at all. So a
refusal that exists only as a warning is, at this layer, no refusal.

Five defects were measured here on 2026-09-17 and are pinned below.

* ``.astype(str)`` on a protected attribute turned a row that carries NO
  attribute into a group literally named "None", which then took part in every
  comparison. The library's own ``missing_strategy`` could not reach it,
  because by the time the array arrived nothing was missing any more.
  Measured: ``audit_agent`` published a 0.450 action-rate gap against that
  phantom group where the only real comparison was 0.000;
  ``analyze_intersectional`` published "max subgroup disparity 0.500" for the
  cell ``None_M`` where a_M against b_M was 0.000; ``measure_fairness``
  reported ``n_excluded: 0`` under ``missing_strategy='exclude'``.
* ``audit_agent`` hoisted the metric's worst pair into its summary without the
  metric's own interpretability verdict, so one row per group produced
  "Largest cross-group action-rate gap: 'escalate' (1.000)" over two groups it
  had already rated ``tier: invalid``.
* ``triage_dataset`` said "Bias audit complete" while the report it was
  wrapping said ``assessment_coverage: 'none'``.
* ``suggest_mitigation`` said "Suggested 0 pre-processing mitigation(s)" while
  all ten of the feature screens behind that 0 had refused for sample size.
* ``explain_decision`` dropped every feature whose association was NaN out of
  ``drivers`` and called the remainder "Top drivers".

Every pin is followed by an over-correction control asserting the MEASURED
value still arrives unchanged, because the cheapest way to pass a refusal test
is to refuse everything.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from vfairness.mcp import tools


def _call(fn, *args, **kwargs):
    """Run with warnings ENABLED and recorded, so a refusal carried only in a
    warning is visible here instead of silently swallowed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# Fixtures. All deterministic: no RNG anywhere, so every asserted number is
# reproducible and hand-computable.
# ---------------------------------------------------------------------------


@pytest.fixture()
def missing_attribute_frame() -> pd.DataFrame:
    """100 rows of group 'a', 100 of 'b', 100 that carry NO group at all.

    Hand-computed on the rows that DO carry a group: selection rate a = 80/100
    = 0.80, b = 40/100 = 0.40, so the demographic parity difference is 0.40 and
    the ratio is 0.50. The 100 attribute-less rows are 50/50, which is why a
    phantom group made of them lands between the two real ones and changes the
    headline.
    """
    return pd.DataFrame(
        {
            "race": ["a"] * 100 + ["b"] * 100 + [None] * 100,
            "y": [1, 0] * 150,
            "p": [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60 + [1] * 50 + [0] * 50,
        }
    )


@pytest.fixture()
def agent_log_with_attribute_less_rows() -> pd.DataFrame:
    """40 rows group A, 40 group B, 40 with no group.

    Hand-computed: A escalates 20/40 = 0.50, B escalates 20/40 = 0.50, so the
    true cross-group action-rate gap is 0.00 for both actions. The 40
    attribute-less rows escalate 2/40 = 0.05, so a phantom group made of them
    manufactures a 0.45 gap.
    """
    return pd.DataFrame(
        {
            "group": ["A"] * 40 + ["B"] * 40 + [None] * 40,
            "action": (
                ["escalate"] * 20
                + ["resolve"] * 20
                + ["escalate"] * 20
                + ["resolve"] * 20
                + ["escalate"] * 2
                + ["resolve"] * 38
            ),
        }
    )


@pytest.fixture()
def hiring_frame() -> pd.DataFrame:
    """A healthy frame with a real, findable disparity. 60 rows per cell.

    Hand-computed selection rates: (m, north) 48/60 = 0.80, (m, south) 36/60 =
    0.60, (f, north) 18/60 = 0.30, (f, south) 6/60 = 0.10. Marginals therefore
    are gender m 84/120 = 0.70 against f 24/120 = 0.20 (a 0.50 gap) and region
    north 66/120 = 0.55 against south 42/120 = 0.35 (a 0.20 gap).
    """
    cells = {("m", "north"): 48, ("m", "south"): 36, ("f", "north"): 18, ("f", "south"): 6}
    rows: List[Dict[str, Any]] = []
    for (gender, region), hired in cells.items():
        for i in range(60):
            rows.append(
                {
                    "gender": gender,
                    "region": region,
                    "hired": 1 if i < hired else 0,
                    "prediction": 1 if i < hired else 0,
                    "tenure": float(i % 12),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# REFUSAL PINS: a row that carries no protected attribute is not a group
# ---------------------------------------------------------------------------


def test_measure_fairness_excludes_rows_with_no_protected_attribute(missing_attribute_frame):
    """REFUSAL PIN. A missing attribute must not become the group "None".

    Before the fix: group_stats carried a third group named "None" of size 100,
    and data_info said n_excluded 0 under missing_strategy 'exclude', which is
    an affirmative claim that no row was dropped.
    """
    out, _ = _call(tools.measure_fairness, missing_attribute_frame, ["race"], "p", "y")
    stats = out["per_attribute"]["race"]["group_stats"]

    assert set(stats) == {"a", "b"}
    for phantom in ("None", "nan", "<NA>", "NaN"):
        assert phantom not in stats

    info = out["per_attribute"]["race"]["data_info"]
    assert info["n_excluded"] == 100
    assert info["final_size"] == 200
    assert info["valid_groups"] == ["a", "b"]


def test_measure_fairness_still_measures_the_real_disparity(missing_attribute_frame):
    """OVER-CORRECTION CONTROL. Dropping the unmeasured rows must not drop the
    measurement: 0.80 against 0.40 is a 0.40 difference and a 0.50 ratio,
    computed here from the fixture, not read back from the code."""
    out, _ = _call(tools.measure_fairness, missing_attribute_frame, ["race"], "p", "y")
    metrics = out["per_attribute"]["race"]["metrics"]

    assert metrics["demographic_parity_difference"] == pytest.approx(0.40)
    assert metrics["demographic_parity_ratio"] == pytest.approx(0.50)
    assert out["per_attribute"]["race"]["group_stats"]["a"]["positive_rate"] == pytest.approx(0.80)
    assert out["per_attribute"]["race"]["group_stats"]["b"]["positive_rate"] == pytest.approx(0.40)
    assert "demographic parity difference 0.400" in out["summary"]


def test_a_group_genuinely_named_none_is_still_measured():
    """OVER-CORRECTION CONTROL. The refusal is about MISSINGNESS, not the word.

    A dataset whose group column really holds the string "None" (a category, a
    dropdown default, a literal export) must still be measured as three groups.
    A fix that filtered on the rendered name would silently delete a real one.
    """
    df = pd.DataFrame(
        {
            "race": ["a"] * 100 + ["b"] * 100 + ["None"] * 100,
            "y": [1, 0] * 150,
            "p": [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60 + [1] * 50 + [0] * 50,
        }
    )
    out, _ = _call(tools.measure_fairness, df, ["race"], "p", "y")
    stats = out["per_attribute"]["race"]["group_stats"]

    assert set(stats) == {"a", "b", "None"}
    assert stats["None"]["size"] == 100
    assert stats["None"]["positive_rate"] == pytest.approx(0.50)
    assert out["per_attribute"]["race"]["data_info"]["n_excluded"] == 0


def test_audit_agent_gap_is_not_drawn_against_attribute_less_rows(
    agent_log_with_attribute_less_rows,
):
    """REFUSAL PIN. Measured before the fix: "Largest cross-group action-rate
    gap: 'escalate' (0.450)" for the pair (A, "None"), with that phantom group
    marked interpretable. The only real comparison, A against B, is 0.000."""
    out, _ = _call(tools.audit_agent, agent_log_with_attribute_less_rows, "group", "action")

    for matrix in out["per_action"].values():
        assert matrix["groups"] == ["A", "B"]
        assert matrix["max_difference"] == pytest.approx(0.0)
        assert matrix["rates"]["A"]["rate"] == pytest.approx(0.5)
        assert matrix["rates"]["B"]["rate"] == pytest.approx(0.5)
    assert "0.450" not in out["summary"]
    assert "0.000" in out["summary"]


def test_analyze_intersectional_cell_needs_every_attribute_recorded():
    """REFUSAL PIN. Measured before the fix: "max subgroup disparity 0.500"
    against the cell "None_M", where the two real cells a_M and b_M are both
    at a 0.50 selection rate and the true max disparity is 0.000."""
    n = 60
    df = pd.DataFrame(
        {
            "race": ["a"] * n + ["b"] * n + [None] * n,
            "sex": ["M"] * (3 * n),
            "y": ([1] * 30 + [0] * 30) * 3,
            "p": ([1] * 30 + [0] * 30) * 2 + [0] * n,
        }
    )
    out, _ = _call(tools.analyze_intersectional, df, ["race", "sex"], "p", "y")
    inter = out["result"]["intersectional_analysis"]

    assert sorted(g["group"] for g in inter["all_groups"]) == ["a_M", "b_M"]
    assert inter["max_disparity"] == pytest.approx(0.0)
    assert out["result"]["data_info"]["n_excluded"] == 60
    assert "0.500" not in out["summary"]
    assert out["coverage"]["max_disparity_measured"] is True


# ---------------------------------------------------------------------------
# REFUSAL PINS: "complete" / "0 found" must not be said over an absent check
# ---------------------------------------------------------------------------


def test_audit_agent_discloses_a_gap_over_groups_the_metric_calls_invalid():
    """REFUSAL PIN. One row per group: the metric falls back to comparing
    groups it has itself rated uninterpretable, and the 1.000 that comes out
    was headlined bare. The number is kept (it is the only evidence there is)
    and the state it was drawn in is now stated.

    WHERE THE NUMBER LIVES CHANGED on 2026-09-27 and the subject did not.
    selection_rate_disparity_matrix stopped putting an uninterpretable pair's
    number in ``max_difference``, answering NaN with
    ``headline_basis="no_interpretable_pair"`` and preserving the value under
    ``headline_over_all_groups``. That is right for a metric whose max_difference
    is read as a headline.

    This test asserted the old field name, so it went red while the behaviour it is
    about was intact everywhere except one place: audit_agent stopped reading the
    fallback, so the SUMMARY lost the disclaimed headline entirely and said "NO
    cross-group gap could be computed" instead, with interpretable_groups null
    rather than 0. That was a real regression, and it is fixed in mcp/tools.py; the
    assertion here moves to the field that now holds the number, and asserts that
    the metric's own max_difference stays refused, which is the upstream decision
    this surface must not undo.
    """
    df = pd.DataFrame({"group": ["A", "B"], "action": ["escalate", "resolve"]})
    out, _ = _call(tools.audit_agent, df, "group", "action")

    matrix = out["per_action"]["escalate"]
    assert matrix["max_difference"] is None, (
        "the metric refused this headline; audit_agent must not re-publish it in the "
        "metric's own field"
    )
    assert matrix["headline_basis"] == "no_interpretable_pair"
    assert matrix["headline_over_all_groups"]["max_difference"] == pytest.approx(1.0)
    assert "1.000" in out["summary"], out["summary"]
    assert "NOT INTERPRETABLE" in out["summary"]
    assert out["coverage"]["gap_is_interpretable"] is False
    assert out["coverage"]["interpretable_groups"] == 0


def test_audit_agent_says_when_no_cross_group_gap_could_be_computed():
    """REFUSAL PIN. One group means no pair exists. The absence of a gap is
    not a finding that the agent treats groups alike, and the summary now says
    so instead of reporting only the action count."""
    df = pd.DataFrame({"group": ["A"] * 50, "action": ["escalate"] * 25 + ["resolve"] * 25})
    out, _ = _call(tools.audit_agent, df, "group", "action")

    assert out["coverage"]["gap_measured"] is False
    assert out["coverage"]["gap_is_interpretable"] is None
    assert "NO cross-group gap could be computed" in out["summary"]
    for matrix in out["per_action"].values():
        assert matrix["max_difference"] is None


def test_audit_agent_healthy_gap_is_measured_and_not_disclaimed():
    """OVER-CORRECTION CONTROL. A real, interpretable disparity must arrive as
    a plain number with no caveat, and the headline must name the action with
    the WIDEST gap. Three actions, so the widest is unambiguous rather than a
    tie between an action and its complement. Hand-computed rates: escalate
    0.10 against 0.60 (gap 0.50), resolve 0.60 against 0.30 (gap 0.30), defer
    0.30 against 0.10 (gap 0.20)."""
    df = pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "action": (
                ["escalate"] * 10
                + ["resolve"] * 60
                + ["defer"] * 30
                + ["escalate"] * 60
                + ["resolve"] * 30
                + ["defer"] * 10
            ),
        }
    )
    out, _ = _call(tools.audit_agent, df, "group", "action")

    assert out["per_action"]["escalate"]["max_difference"] == pytest.approx(0.50)
    assert out["per_action"]["resolve"]["max_difference"] == pytest.approx(0.30)
    assert out["per_action"]["defer"]["max_difference"] == pytest.approx(0.20)
    assert out["coverage"]["gap_measured"] is True
    assert out["coverage"]["gap_is_interpretable"] is True
    assert out["coverage"]["interpretable_groups"] == 2
    assert "Largest cross-group action-rate gap: 'escalate' (0.500)." in out["summary"]
    assert "NOT INTERPRETABLE" not in out["summary"]
    assert "NO cross-group gap" not in out["summary"]


def test_analyze_intersectional_refuses_when_no_cell_is_measurable(hiring_frame):
    """REFUSAL PIN. Every cell under min_group_size means nothing was ranked.
    Measured before the fix: "Intersectional analysis complete." over zero
    measured cells, which reads as an analysis that found no disparity."""
    out, _ = _call(
        tools.analyze_intersectional,
        hiring_frame.head(20),
        ["gender", "region"],
        "prediction",
        "hired",
    )

    assert out["coverage"]["max_disparity_measured"] is False
    assert out["result"]["intersectional_analysis"]["max_disparity"] is None
    assert "NOT ASSESSED" in out["summary"]
    assert "complete" not in out["summary"]


def test_analyze_intersectional_healthy_reports_the_measured_disparity(hiring_frame):
    """OVER-CORRECTION CONTROL. Hand-computed from the fixture: the widest and
    narrowest cells are (m, north) at 48/60 = 0.80 and (f, south) at 6/60 =
    0.10, so the max subgroup disparity is exactly 0.70."""
    out, _ = _call(
        tools.analyze_intersectional,
        hiring_frame,
        ["gender", "region"],
        "prediction",
        "hired",
    )

    assert out["coverage"]["max_disparity_measured"] is True
    assert out["result"]["intersectional_analysis"]["max_disparity"] == pytest.approx(0.70)
    assert "Intersectional analysis complete; max subgroup disparity 0.700." == out["summary"]


def test_triage_dataset_does_not_say_complete_when_nothing_was_assessed():
    """REFUSAL PIN. The report already separates "the module RAN" from "the
    module reached a verdict", and warns when nothing did. Measured before the
    fix on this frame: assessment_coverage 'none', overall_risk_score 0.0, and
    a tool summary of "Bias audit complete: 1 finding(s)"."""
    df = pd.DataFrame({"g": ["a"] * 120, "hired": [1] * 60 + [0] * 60, "tenure": list(range(120))})
    out, _ = _call(tools.triage_dataset, df, ["g"], "hired")

    assert out["coverage"]["assessment_coverage"] == "none"
    assert out["coverage"]["findings_are_a_measurement"] is False
    assert out["coverage"]["attributes_not_assessed"] == ["g"]
    assert out["summary"].startswith("Bias audit ASSESSED NOTHING:")
    assert "NOT A CLEAN BILL OF HEALTH" in out["summary"]
    # The finding that WAS raised must survive the disclosure.
    assert out["finding_counts"]["representation_findings"] == 1


def test_triage_dataset_partial_coverage_is_its_own_state(hiring_frame):
    """REFUSAL PIN plus a do-not-delete-the-finding control. One assessable
    attribute and one entirely null one is neither 'complete' nor 'none', and
    the real attribute's findings must be untouched by saying so."""
    frame = hiring_frame.copy()
    frame["nullattr"] = [None] * len(frame)
    both, _ = _call(tools.triage_dataset, frame, ["gender", "nullattr"], "hired")
    alone, _ = _call(tools.triage_dataset, hiring_frame, ["gender"], "hired")

    assert both["coverage"]["assessment_coverage"] == "partial"
    assert both["coverage"]["attributes_not_assessed"] == ["nullattr"]
    assert both["coverage"]["findings_are_a_measurement"] is False
    assert "Bias audit INCOMPLETE:" in both["summary"]
    # Every finding the assessable attribute produced on its own is still here.
    for bucket, n in alone["finding_counts"].items():
        assert both["finding_counts"][bucket] >= n


def test_triage_dataset_healthy_still_says_complete(hiring_frame):
    """OVER-CORRECTION CONTROL. A fully assessed audit keeps the old wording
    and the old counts, and says its findings ARE a measurement."""
    out, _ = _call(tools.triage_dataset, hiring_frame, ["gender", "region"], "hired")

    assert out["coverage"]["assessment_coverage"] == "complete"
    assert out["coverage"]["execution_coverage"] == "complete"
    assert out["coverage"]["attributes_not_assessed"] == []
    assert out["coverage"]["findings_are_a_measurement"] is True
    assert out["summary"].startswith("Bias audit complete:")
    assert "NOT A CLEAN BILL" not in out["summary"]
    # The caller's own target was tested as an OUTCOME for both attributes.
    outcome_rows = sorted(
        f["protected_attribute"]
        for f in out["report"]["disparity_findings"]
        if f["feature"] == "hired" and f["disparity_type"] == "outcome"
    )
    assert outcome_rows == ["gender", "region"]


def test_suggest_mitigation_zero_recommendations_is_not_an_all_clear(hiring_frame):
    """REFUSAL PIN. Measured before the fix on a 5-row frame: "Suggested 0
    pre-processing mitigation(s) from feature analysis", while all ten feature
    screens behind that 0 had refused for sample size."""
    out, _ = _call(tools.suggest_mitigation, hiring_frame.head(5), ["gender", "region"], "hired")

    assert out["coverage"]["feature_screen_complete"] is False
    assert out["coverage"]["recommendations_are_a_measurement"] is False
    assert out["coverage"]["n_screens_not_run"] > 0
    assert "COULD NOT CHECK" in out["summary"]


def test_suggest_mitigation_full_screen_carries_no_caveat(hiring_frame):
    """OVER-CORRECTION CONTROL. When every screen ran, the summary is the
    plain count and the coverage says the recommendations are a measurement."""
    out, _ = _call(tools.suggest_mitigation, hiring_frame, ["gender", "region"], "hired")

    assert out["coverage"]["feature_screen_complete"] is True
    assert out["coverage"]["n_screens_not_run"] == 0
    assert out["coverage"]["recommendations_are_a_measurement"] is True
    assert "COULD NOT CHECK" not in out["summary"]
    assert "UNRECORDED" not in out["summary"]
    assert out["summary"].startswith("Suggested ")


def test_explain_decision_lists_the_features_it_could_not_assess():
    """REFUSAL PIN. A constant column and a column with fewer usable rows than
    the correlation floor both come back NaN. Measured before the fix: they
    vanished from `drivers` and the summary read "Top drivers of 'prediction':
    real_driver (0.98)." with no sign that two features were never assessed."""
    n = 200
    pred = [1, 0] * (n // 2)
    df = pd.DataFrame(
        {
            "prediction": pred,
            # A perfect copy of the prediction: association exactly 1.0.
            "real_driver": [float(v) for v in pred],
            "const": ["X"] * n,
            "sparse": [1.0] * 5 + [np.nan] * (n - 5),
        }
    )
    out, _ = _call(tools.explain_decision, df, "prediction")

    unmeasured = {d["feature"]: d for d in out["features_not_measured"]}
    assert set(unmeasured) == {"const", "sparse"}
    for row in unmeasured.values():
        assert row["association"] is None
        assert "unassessed" in row["reason"]
    assert "could NOT be assessed" in out["summary"]
    # OVER-CORRECTION CONTROL, in the same call: the measurable feature is
    # still measured, and a perfect copy of the prediction correlates at 1.0.
    assert [d["feature"] for d in out["drivers"]] == ["real_driver"]
    assert out["drivers"][0]["association"] == pytest.approx(1.0)


def test_explain_decision_named_feature_absent_from_the_frame_is_disclosed():
    """REFUSAL PIN. A feature the caller NAMES that is not a column is dropped
    before any statistic is attempted, so its silence used to be
    indistinguishable from "measured, and it drives nothing"."""
    n = 200
    pred = [1, 0] * (n // 2)
    df = pd.DataFrame({"prediction": pred, "real_driver": [float(v) for v in pred]})
    out, _ = _call(
        tools.explain_decision, df, "prediction", feature_columns=["real_driver", "ghost"]
    )

    unmeasured = {d["feature"]: d["reason"] for d in out["features_not_measured"]}
    assert set(unmeasured) == {"ghost"}
    assert "never screened" in unmeasured["ghost"]
    assert [d["feature"] for d in out["drivers"]] == ["real_driver"]


def test_explain_decision_clean_frame_reports_no_unmeasured_features(hiring_frame):
    """OVER-CORRECTION CONTROL. When every feature was assessable the
    disclosure must be empty and the summary must carry no caveat, or the pin
    above would pass for a tool that declared everything unmeasurable."""
    out, _ = _call(tools.explain_decision, hiring_frame, "prediction")

    assert out["features_not_measured"] == []
    assert "could NOT be assessed" not in out["summary"]
    assert out["summary"].startswith("Top drivers of 'prediction':")
    assert {d["feature"] for d in out["drivers"]} == {"gender", "region", "hired", "tenure"}


# ---------------------------------------------------------------------------
# REFUSAL PIN: a missing LABEL is not a third outcome class
# ---------------------------------------------------------------------------


@pytest.fixture()
def string_labels_with_missing_predictions() -> pd.DataFrame:
    """300 rows, a string prediction column, 50 of them with no prediction.

    Hand-computed over the 250 rows that DO carry a prediction. Category codes
    are built alphabetically from the present values, so "approve" is 0 and
    "deny" is 1 and the measured positive rate is the deny rate: group a is
    30/150 = 0.20, group b is 50/100 = 0.50, a difference of exactly 0.30.
    """
    return pd.DataFrame(
        {
            "race": ["a"] * 150 + ["b"] * 150,
            "y": [1, 0] * 150,
            "p": (
                ["approve"] * 120 + ["deny"] * 30 + ["approve"] * 50 + ["deny"] * 50 + [None] * 50
            ),
        }
    )


def test_a_missing_string_label_is_not_a_third_class(string_labels_with_missing_predictions):
    """REFUSAL PIN. pd.Categorical uses -1 for "no category", and returning
    that sentinel made every row with NO prediction a third prediction VALUE.

    Measured before the fix: codes [-1, 0, 1], three distinct whole numbers,
    which made FairnessAnalyzer auto-detect REGRESSION. measure_fairness then
    answered with mae_parity_difference 0.333 / r2_parity_difference 2.667 /
    mean_prediction_difference 0.200, a battery computed over a class nobody
    recorded, with n_excluded reported as 0.
    """
    codes = tools._numeric_labels(string_labels_with_missing_predictions["p"])
    assert sorted(np.unique(codes[~np.isnan(codes)])) == [0.0, 1.0]
    assert np.isnan(codes).sum() == 50

    out, _ = _call(
        tools.measure_fairness, string_labels_with_missing_predictions, ["race"], "p", "y"
    )
    metrics = out["per_attribute"]["race"]["metrics"]
    # A classification battery, not a regression one over an invented class.
    assert "demographic_parity_difference" in metrics
    for regression_metric in ("mae_parity_difference", "r2_parity_difference"):
        assert regression_metric not in metrics
    assert out["per_attribute"]["race"]["data_info"]["n_excluded"] == 50


def test_the_real_disparity_survives_dropping_the_unlabelled_rows(
    string_labels_with_missing_predictions,
):
    """OVER-CORRECTION CONTROL. 0.20 against 0.50 is a 0.30 difference,
    computed from the fixture above rather than read back from the code."""
    out, _ = _call(
        tools.measure_fairness, string_labels_with_missing_predictions, ["race"], "p", "y"
    )
    stats = out["per_attribute"]["race"]["group_stats"]

    assert stats["a"]["size"] == 150
    assert stats["b"]["size"] == 100
    assert stats["a"]["positive_rate"] == pytest.approx(0.20)
    assert stats["b"]["positive_rate"] == pytest.approx(0.50)
    assert out["per_attribute"]["race"]["metrics"][
        "demographic_parity_difference"
    ] == pytest.approx(0.30)


def test_a_complete_string_column_encodes_exactly_as_before():
    """OVER-CORRECTION CONTROL. The fix touches only the -1 sentinel. A column
    with nothing missing must encode to the same codes it always did:
    alphabetical categories, "approve" 0 and "deny" 1."""
    codes = tools._numeric_labels(pd.Series(["deny", "approve", "approve", "deny"]))
    assert list(codes) == [1.0, 0.0, 0.0, 1.0]
    numeric = tools._numeric_labels(pd.Series([1, 0, 1]))
    assert list(numeric) == [1, 0, 1]


# ---------------------------------------------------------------------------
# detect_proxies and measure_fairness: refusals that already existed, pinned
# so they cannot regress out of the payload again
# ---------------------------------------------------------------------------


@pytest.fixture()
def proxy_frame() -> pd.DataFrame:
    """A perfect proxy plus a constant column nothing can be correlated with.

    Hand-computed: 'zipc' is 1 for every 'a' and 2 for every 'b', a perfect
    two-by-two association, so its measured strength is 1.0. 'const' has no
    variation at all, so every pair through it is 0/0 and undefined.
    """
    return pd.DataFrame(
        {
            "race": ["a"] * 50 + ["b"] * 50,
            "zipc": [1] * 50 + [2] * 50,
            "const": ["X"] * 100,
            "noise": list(range(100)),
        }
    )


def test_detect_proxies_will_not_call_a_partial_search_complete(proxy_frame):
    """REFUSAL PIN. An empty or short chain list is a finding only when the
    search covered the frame. The pair through the constant column is 0/0, and
    a chain running through it would not appear, so the word is refused."""
    out, _ = _call(tools.detect_proxies, proxy_frame, ["race"])
    coverage = out["proxy_chain_coverage"]["race"]

    assert coverage["complete"] is False
    assert coverage["n_pairs_not_computed"] == 3
    assert "COULD NOT CHECK" in coverage["coverage_note"]
    assert out["summary"].startswith("Proxy analysis INCOMPLETE:")
    # The correlation that could not be taken is null, never 0.0.
    assert out["correlations"]["correlations"]["race"]["const"] is None


def test_detect_proxies_still_measures_the_real_proxy(proxy_frame):
    """OVER-CORRECTION CONTROL. Refusing the undefined pair must not cost the
    measured one: a column that is 1 for every 'a' and 2 for every 'b' is a
    perfect proxy and must come back at association 1.0."""
    out, _ = _call(tools.detect_proxies, proxy_frame, ["race"])

    assert out["correlations"]["correlations"]["race"]["zipc"] == pytest.approx(1.0)
    assert out["correlations"]["correlations"]["race"]["noise"] > 0.5


def test_detect_proxies_says_complete_when_the_search_did_cover_the_frame(proxy_frame):
    """OVER-CORRECTION CONTROL. Drop the constant column and every pair is
    computable, so the word 'complete' must come back rather than a blanket
    caveat that would make the pin above pass for free."""
    out, _ = _call(tools.detect_proxies, proxy_frame.drop(columns=["const"]), ["race"])
    coverage = out["proxy_chain_coverage"]["race"]

    assert coverage["complete"] is True
    assert coverage["n_pairs_not_computed"] == 0
    assert coverage["coverage_note"] == ""
    assert out["summary"].startswith("Proxy analysis complete:")


def test_measure_fairness_refuses_a_one_group_comparison():
    """REFUSAL PIN. One group means no between-group comparison happened, so
    every disparity metric is null with a stated reason and the tool summary
    carries no headline number a reader could act on."""
    df = pd.DataFrame({"g": ["a"] * 200, "y": [1, 0] * 100, "p": [1] * 100 + [0] * 100})
    out, _ = _call(tools.measure_fairness, df, ["g"], "p", "y")
    report = out["per_attribute"]["g"]

    assert all(v is None for v in report["metrics"].values())
    assert report["assessment"]["assessable"] is False
    assert report["assessment"]["fairness_score"] is None
    reasons = {r["metric"]: r["status"] for r in report["assessment"]["not_assessable_metrics"]}
    assert reasons["demographic_parity_difference"] == "NOT_ASSESSABLE"
    assert "demographic parity difference" not in out["summary"]
