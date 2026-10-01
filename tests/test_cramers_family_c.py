"""Cramer's V family, group C: the remaining naive copies.

One defect, four sites, all the same shape::

    cramers_v = sqrt(chi2 / (n * (min(r, c) - 1)))  if min_dim > 0 else 0.0

``0.0`` is the NO ASSOCIATION reading. It is what a clean table looks like, so
publishing it for a table on which V is not defined at all hands the reader a
clean bill of health for a pair nobody measured. The three states are
measured / failed / could-not-check, and the could-not-check here is NaN plus a
UserWarning naming the table shape.

Sites pinned (all four, whether or not they reproduced at a public entry):

* ``preprocessing/bias_detection/statistical.py:869`` in the PUBLIC
  ``compute_effect_sizes``. REPRODUCED 2026-09-17: 200 rows whose outcome was
  the constant "approved" crosstab to a 2x1 table and the function returned
  ``{"cramers_v": {"value": 0.0, "interpretation": "negligible"}}`` with no
  warning at all.
* ``preprocessing/bias_detection/statistical.py:556`` in
  ``_test_categorical_disparity``, reached from the public
  ``analyze_statistical_disparities``. NOT reproduced: the
  ``contingency.shape < 2`` guard returns None first. The two sites were
  byte-identical copies of one expression in two different functions, so both
  now go through one shared helper, ``_cramers_v_from_chi2``.
* ``evaluation/vfairness_metrics/robustness.py:957`` in the PUBLIC
  ``contingency_test``. NOT reproduced: the C-02 degenerate-table guard above
  it returns ``test_used="none"`` before scipy is called.
* ``multi_agent/delegation.py:216`` ``DelegationRoutingAuditor._cramers_v``.
  REPRODUCED at the helper (``_cramers_v(5.0, 100, (1, 3))`` returned 0.0 with
  zero warnings) and NOT reachable from ``analyze``, which raises ValueError
  below two distinct routes or two distinct groups.

Every refusal pin is paired with a healthy-data CONTROL asserting that a real,
findable association is still measured to its exact value, computed here from
the definition of the statistic rather than copied from what the code returns.
A detector that refuses everything passes every degenerate test and finds
nothing, which is worse than the defect.
"""

import math
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics import robustness as robustness_mod
from vfairness.evaluation.vfairness_metrics.robustness import contingency_test
from vfairness.multi_agent.delegation import DelegationRoutingAuditor
from vfairness.preprocessing.bias_detection import statistical as statistical_mod
from vfairness.preprocessing.bias_detection.statistical import (
    _cramers_v_from_chi2,
    analyze_statistical_disparities,
    compute_effect_sizes,
)

# ---------------------------------------------------------------------------
# Expected values, computed here from the definition of the statistic.
# ---------------------------------------------------------------------------


def _chi2_2x2_yates(a: int, b: int, c: int, d: int) -> float:
    """Yates-corrected Pearson chi-square of [[a, b], [c, d]].

    scipy applies the continuity correction by default on a 2x2 table, and
    both statistical.py call sites take scipy's default, so this is the
    statistic their V is built on. Written out here so the expected number is
    derived, never lifted from a run of the code under test.
    """
    n = a + b + c + d
    num = n * (abs(a * d - b * c) - n / 2.0) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return num / den


def _chi2_uncorrected(table: np.ndarray) -> float:
    """Plain Pearson chi-square, sum((o - e)^2 / e) with e from the margins."""
    table = np.asarray(table, dtype=float)
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / table.sum()
    return float(np.sum((table - expected) ** 2 / expected))


def _v(chi2: float, n: int, shape) -> float:
    return math.sqrt(chi2 / (n * (min(shape) - 1)))


# The 2x2 outcome table used by every statistical.py control:
#   group A: 90 approved / 10 denied, group B: 30 approved / 70 denied.
_A_POS, _A_NEG, _B_POS, _B_NEG = 90, 10, 30, 70
_EXPECTED_V_2X2 = _v(_chi2_2x2_yates(_A_POS, _A_NEG, _B_POS, _B_NEG), 200, (2, 2))


def _decisive_frame() -> pd.DataFrame:
    """200 rows carrying a large, real, findable outcome disparity."""
    rows = (
        [("A", "approved")] * _A_POS
        + [("A", "denied")] * _A_NEG
        + [("B", "approved")] * _B_POS
        + [("B", "denied")] * _B_NEG
    )
    return pd.DataFrame(rows, columns=["race", "outcome"])


def _constant_outcome_frame() -> pd.DataFrame:
    """200 rows, two groups, and an outcome column that never varies.

    Crosstabs to 2x1. There is nothing for the outcome to vary AGAINST, so V is
    undefined, not zero.
    """
    return pd.DataFrame({"race": ["A"] * 100 + ["B"] * 100, "outcome": ["approved"] * 200})


def _warned(record, *needles: str) -> bool:
    text = " ".join(str(w.message) for w in record)
    return all(needle in text for needle in needles)


# ---------------------------------------------------------------------------
# SITE 1 (REPRODUCED): statistical.compute_effect_sizes, the public entry.
# ---------------------------------------------------------------------------


class TestComputeEffectSizesRefusesAnUndefinedTable:
    def test_constant_outcome_is_not_measurable_not_negligible(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = compute_effect_sizes(_constant_outcome_frame(), "outcome", "race")

        entry = out["effect_sizes"]["cramers_v"]
        assert math.isnan(entry["value"]), entry
        # The word a reader actually sees. "negligible" was the old answer and
        # it is a finding of no association.
        assert entry["interpretation"] == "not_measurable", entry
        assert entry["interpretation"] != "negligible"
        assert _warned(rec, "not defined", "2x1", "NOT measured"), [str(w.message) for w in rec]

    def test_constant_numeric_outcome_takes_the_same_branch(self):
        # nunique() == 1 sends a numeric column down the categorical arm, so
        # the same undefined table arrives by a second route. Fixing one site
        # must not leave its sibling fabricating.
        df = pd.DataFrame({"race": ["A"] * 100 + ["B"] * 100, "score": np.ones(200)})
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = compute_effect_sizes(df, "score", "race")
        entry = out["effect_sizes"]["cramers_v"]
        assert math.isnan(entry["value"])
        assert entry["interpretation"] == "not_measurable"
        assert _warned(rec, "not defined", "NOT measured")

    def test_control_a_real_association_is_still_measured_exactly(self):
        """CONTROL. The finding below the new guard is still reachable."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = compute_effect_sizes(_decisive_frame(), "outcome", "race")

        entry = out["effect_sizes"]["cramers_v"]
        assert entry["value"] == pytest.approx(_EXPECTED_V_2X2, abs=1e-12)
        assert entry["value"] == pytest.approx(0.6021662284341979, abs=1e-12)
        assert entry["interpretation"] == "large"
        # The odds ratio below it still computes: the refusal did not become an
        # early return that swallows everything after it.
        assert out["effect_sizes"]["odds_ratio"]["value"] == pytest.approx(21.0)
        assert not [w for w in rec if "not defined" in str(w.message)]


# ---------------------------------------------------------------------------
# SITE 2: the shared helper, and statistical._test_categorical_disparity via
# the public analyze_statistical_disparities.
# ---------------------------------------------------------------------------


class TestSharedCramersHelper:
    """One helper now backs both statistical.py copies.

    They were two different functions carrying one byte-identical expression,
    not one function duplicated. A third copy is one feature away, so the
    refusal lives in the helper rather than at the call sites.
    """

    @pytest.mark.parametrize(
        "chi2, n, shape, needle",
        [
            (10.0, 200, (2, 1), "2x1"),
            (10.0, 200, (1, 4), "1x4"),
            (10.0, 0, (2, 2), "0 observation"),
            (float("nan"), 200, (2, 2), "finite chi-square"),
        ],
    )
    def test_undefined_tables_return_nan_and_warn(self, chi2, n, shape, needle):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            v = _cramers_v_from_chi2(chi2, n, shape, "pin")
        assert math.isnan(v)
        assert _warned(rec, needle, "NOT measured", "could-not-check")

    def test_control_helper_measures_a_defined_table_exactly(self):
        chi2 = _chi2_2x2_yates(_A_POS, _A_NEG, _B_POS, _B_NEG)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            v = _cramers_v_from_chi2(chi2, 200, (2, 2), "pin")
        assert v == pytest.approx(_EXPECTED_V_2X2, abs=1e-12)
        assert rec == []


class TestBothStatisticalCopiesShareOneHelper:
    """The two statistical.py sites are two FUNCTIONS, not one duplicated.

    :556 lives in ``_test_categorical_disparity`` and :869 in
    ``compute_effect_sizes``. They carried the same expression independently,
    which is exactly how one gets fixed while the other keeps fabricating, and
    a third copy is one feature away. Both now route through
    ``_cramers_v_from_chi2``; these two pins are what stops the expression from
    being inlined back beside them, because the guarded branch in
    ``_test_categorical_disparity`` is unreachable and so a behaviour pin alone
    cannot see it regress.
    """

    @staticmethod
    def _spy(monkeypatch):
        calls = []
        real = statistical_mod._cramers_v_from_chi2

        def spy(chi2, n, shape, where):
            calls.append((int(n), tuple(int(x) for x in shape), where))
            return real(chi2, n, shape, where)

        monkeypatch.setattr(statistical_mod, "_cramers_v_from_chi2", spy)
        return calls

    def test_compute_effect_sizes_routes_through_the_helper(self, monkeypatch):
        calls = self._spy(monkeypatch)
        out = compute_effect_sizes(_decisive_frame(), "outcome", "race")
        assert (200, (2, 2), "compute_effect_sizes") in calls, calls
        assert out["effect_sizes"]["cramers_v"]["value"] == pytest.approx(
            _EXPECTED_V_2X2, abs=1e-12
        )

    def test_categorical_disparity_routes_through_the_helper(self, monkeypatch):
        calls = self._spy(monkeypatch)
        results = analyze_statistical_disparities(_decisive_frame(), ["race"], ["outcome"])
        assert (200, (2, 2), "_test_categorical_disparity") in calls, calls
        assert [r.effect_size for r in results if r.feature == "outcome"] == [
            pytest.approx(_EXPECTED_V_2X2, abs=1e-12)
        ]


class TestAnalyzeStatisticalDisparities:
    def test_constant_outcome_yields_no_fabricated_effect_size(self):
        """The degenerate table never reaches the statistic.

        ``_test_categorical_disparity`` returns None on a table with fewer than
        two rows or two columns, so site :556 is not reachable from here. This
        pins that: nothing is reported, and in particular nothing is reported
        carrying effect_size 0.0.
        """
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            results = analyze_statistical_disparities(
                _constant_outcome_frame(), ["race"], ["outcome"]
            )
        assert [r.effect_size for r in results] == []
        assert not [w for w in rec if "not defined" in str(w.message)]

    def test_control_a_real_disparity_is_still_reported_exactly(self):
        """CONTROL. Effect size, its type, its grade and the raw gap."""
        results = analyze_statistical_disparities(_decisive_frame(), ["race"], ["outcome"])
        assert results, "the decisive frame must still produce a finding"
        outcome = [r for r in results if r.feature == "outcome"]
        assert outcome, [r.feature for r in results]
        r = outcome[0]
        assert r.effect_size == pytest.approx(_EXPECTED_V_2X2, abs=1e-12)
        assert r.effect_size_type == "Cramér's V"
        assert r.effect_interpretation.value == "large"
        # 0.90 minus 0.30 on the resolved positive label.
        assert r.disparity_magnitude == pytest.approx(0.6, abs=1e-12)
        assert r.privileged_group == "A"
        assert r.disadvantaged_group == "B"


# ---------------------------------------------------------------------------
# SITE 3: robustness.contingency_test.
# ---------------------------------------------------------------------------


class TestContingencyTestCramersV:
    def test_degenerate_table_never_reaches_the_statistic(self):
        """The C-02 guard above :957 still refuses, and still says so.

        This is why :957 did not reproduce. It is pinned because the guard and
        the statistic are ten lines apart: if the guard is ever narrowed, the
        expression below it must refuse rather than answer 0.0.
        """
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = contingency_test(np.array([0, 1, 0, 1, 1, 0]), np.array(["A"] * 6))
        assert res.test_used == "none"
        assert res.cramers_v is None
        assert res.cramers_v != 0.0
        assert math.isnan(res.p_value)
        assert res.significant_at_05 is None
        assert _warned(rec, "COULD NOT CHECK")

    def test_non_finite_chi_square_refuses_rather_than_answering_zero(self):
        """Reach the branch itself by injection.

        The only way into ``else`` at :957 with the C-02 guard in place is a
        chi-square that is not a number, so scipy is replaced for one call.
        With the old expression this returned 0.0, which is NO ASSOCIATION
        between the predictions and the group.
        """
        real = robustness_mod.scipy_stats

        def fake_chi2_contingency(table, *args, **kwargs):
            chi2, p, dof, expected = real.chi2_contingency(table, *args, **kwargs)
            return float("nan"), p, dof, expected

        shim = types.SimpleNamespace(
            chi2_contingency=fake_chi2_contingency,
            fisher_exact=real.fisher_exact,
        )
        y_pred = np.array([1] * 10 + [0] * 10 + [1] * 5 + [0] * 15 + [1] * 18 + [0] * 2)
        attr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)

        original = robustness_mod.scipy_stats
        robustness_mod.scipy_stats = shim
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                res = contingency_test(y_pred, attr)
        finally:
            robustness_mod.scipy_stats = original

        assert res.cramers_v is not None
        assert math.isnan(res.cramers_v), res.cramers_v
        assert res.cramers_v != 0.0
        assert _warned(rec, "not defined", "2x3", "NOT measured")

    def test_control_three_groups_measure_exactly(self):
        """CONTROL. A real association is measured, and is still significant."""
        y_pred = np.array([1] * 10 + [0] * 10 + [1] * 5 + [0] * 15 + [1] * 18 + [0] * 2)
        attr = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = contingency_test(y_pred, attr)

        # rows are (negative, positive) by group A, B, C.
        table = np.array([[10, 15, 2], [10, 5, 18]])
        expected_v = _v(_chi2_uncorrected(table), 60, (2, 3))
        assert expected_v == pytest.approx(0.5381099233077657, abs=1e-12)

        assert res.test_used == "chi_square"
        assert res.cramers_v == pytest.approx(expected_v, abs=1e-12)
        assert res.degrees_of_freedom == 2
        assert res.significant_at_05 is not None
        assert bool(res.significant_at_05) is True
        assert not [w for w in rec if "not defined" in str(w.message)]


# ---------------------------------------------------------------------------
# SITE 4: DelegationRoutingAuditor._cramers_v.
# ---------------------------------------------------------------------------


class TestDelegationCramersV:
    @pytest.mark.parametrize(
        "chi2, n, shape, needle",
        [
            (5.0, 100, (1, 3), "1x3"),  # the reproduction from the brief
            (5.0, 100, (3, 1), "3x1"),
            (5.0, 0, (2, 2), "0 decision"),
            (float("nan"), 100, (2, 2), "finite chi-square"),
        ],
    )
    def test_undefined_routing_table_returns_nan_and_warns(self, chi2, n, shape, needle):
        auditor = DelegationRoutingAuditor()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            v = auditor._cramers_v(chi2, n, shape)
        assert math.isnan(v), v
        assert v != 0.0
        assert _warned(rec, needle, "NOT measured", "could not check")

    def test_analyze_still_refuses_degenerate_input_at_the_door(self):
        """Why site :216 is not reachable from the public entry.

        Both guards raise. The refusal added below them must not become the
        reason these stop raising: a ValueError names the problem, a NaN does
        not.
        """
        auditor = DelegationRoutingAuditor()
        with pytest.raises(ValueError, match="2 distinct routes"):
            auditor.analyze(["x"] * 6, ["A", "B"] * 3)
        with pytest.raises(ValueError, match="2 distinct demographic groups"):
            auditor.analyze(["x", "y"] * 3, ["A"] * 6)

    def test_control_perfect_segregation_still_measures_one(self):
        """CONTROL. The strongest finding this auditor has is still raised."""
        auditor = DelegationRoutingAuditor()
        routes = ["junior"] * 30 + ["senior"] * 30
        demographics = ["A"] * 30 + ["B"] * 30
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = auditor.analyze(routes, demographics)

        # Uncorrected chi-square of [[30, 0], [0, 30]] is n, so V is exactly 1.
        expected_v = _v(_chi2_uncorrected(np.array([[30, 0], [0, 30]])), 60, (2, 2))
        assert expected_v == pytest.approx(1.0, abs=1e-12)

        assert res.cramers_v == pytest.approx(expected_v, abs=1e-12)
        assert not math.isnan(res.cramers_v)
        assert res.is_significant is True
        assert res.detectable is True
        assert not [w for w in rec if "not measured" in str(w.message).lower()]

    def test_control_independent_routing_measures_a_genuine_zero(self):
        """CONTROL, the other direction.

        A measured 0.0 must still be reported as 0.0. Refusing something the
        data can answer throws evidence away, and is worse than the defect.
        """
        auditor = DelegationRoutingAuditor()
        routes = ["junior", "senior"] * 30
        demographics = (["A"] * 2 + ["B"] * 2) * 15
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = auditor.analyze(routes, demographics)

        assert res.contingency_table == {
            "junior": {"A": 15, "B": 15},
            "senior": {"A": 15, "B": 15},
        }
        assert res.cramers_v == 0.0
        assert not math.isnan(res.cramers_v)
        assert res.is_significant is False
        assert not [w for w in rec if "not measured" in str(w.message).lower()]
