"""Beta Go-Live stage 1, group g03: the two classification capabilities that answered a
CONFIDENT NUMBER for data in which no decision and no score was ever recorded.

Both defects have the same shape. An unrecorded entry is NaN, and under IEEE-754 both
``nan == 1`` and ``nan >= threshold`` are silently False, so a comparison written against the
raw array reads every missing row as a recorded NEGATIVE:

* ``conditional_demographic_disparity`` (the CJEU objective-justification mirror the EU-hiring
  profile SEALS on) returned **0.0 = "no disparity beyond the legitimate factor"** on 2000 rows
  in which nothing was decided, with no warning, and its sealed CI variant returned
  point_estimate = lower = upper = 0.0 so a TOST upper-bound gate passed trivially at any
  margin.
* ``net_benefit_parity`` (the EU-healthcare clinical-utility gate) returned **0.0** with every
  probability NaN, byte-identical to what a real, equal, useful model returns, and the
  library's own ``check_threshold`` graded that non-measurement PASS.

Every assertion below is at the PUBLIC entry point (``vfairness.<name>``), and each refusal is
paired with a CONTROL on healthy data proving the metric still measures.
"""

import warnings

import numpy as np
import pytest

import vfairness
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    ThresholdOutcome,
    check_threshold,
)

# ---------------------------------------------------------------------------
# Fixtures: the exact data the defect was reproduced on.
# ---------------------------------------------------------------------------


def _hiring_fixture():
    """2000 rows, two protected groups, two legitimate strata, a real selection gap."""
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.choice(["F", "M"], n)
    s = rng.choice(["S1", "S2"], n)
    y = (rng.random(n) < np.where(a == "M", 0.686, 0.252)).astype(float)
    return y, a, s


def _clinical_fixture():
    """Two groups of 60, both comfortably above the default min_group_size=30."""
    y_true = np.array([1] * 50 + [0] * 10 + [1] * 50 + [0] * 10)
    a = np.array(["a"] * 60 + ["b"] * 60)
    return y_true, a


# ---------------------------------------------------------------------------
# conditional_demographic_disparity
# ---------------------------------------------------------------------------


def test_cdd_control_measures_a_real_disparity():
    """CONTROL. With the decisions actually recorded the metric measures, silently."""
    y, a, s = _hiring_fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = vfairness.conditional_demographic_disparity(y, a, s)
    assert np.isfinite(measured)
    assert measured == pytest.approx(0.4167208299339526, abs=1e-9)
    assert not [w for w in caught if "unrecorded" in str(w.message)], (
        "healthy, fully recorded data must not raise the unrecorded-decisions warning"
    )
    assert check_threshold("conditional_demographic_disparity", measured, 0.1)[0] is (
        ThresholdOutcome.FAIL
    )


def test_cdd_refuses_when_no_decision_was_recorded():
    """THREE STATES. Nothing was decided -> could-not-check (NaN), never 0.0.

    0.0 from this function is a legal all-clear: "no disparity beyond the legitimate
    factor". It may not be issued for a comparison that read no decision at all.
    """
    _y, a, s = _hiring_fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vfairness.conditional_demographic_disparity(np.full(len(a), np.nan), a, s)

    assert np.isnan(result), f"expected NaN (insufficient evidence), got {result!r}"
    assert result != 0.0
    assert any("unrecorded" in str(w.message) for w in caught), (
        "the refusal must name its reason in a UserWarning"
    )
    assert any("2000 of 2000" in str(w.message) for w in caught), (
        "the warning must say HOW MANY decisions were unrecorded"
    )
    # The caller-facing gate must route it to could-not-check, not to a pass.
    outcome, _ = check_threshold("conditional_demographic_disparity", result, 0.1)
    assert outcome is ThresholdOutcome.COULD_NOT_CHECK


def test_cdd_does_not_count_an_unrecorded_decision_as_a_rejection():
    """The mechanism, pinned on a hand fixture with no randomness.

    Stratum X: A is selected 10/10, B is rejected 10/10 -> a full 1.0 within-stratum gap.
    Stratum Y: identical rows, except B's decisions were never recorded.

    Counting a NaN as a rejection makes Y look like a second full-disparity stratum and
    returns 1.0 over 40 rows "assessed". Y is in fact unassessable: only X may weigh in, and
    the caller must be told that half the population was dropped.
    """
    a = np.array(["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10)
    strata = np.array(["X"] * 20 + ["Y"] * 20)
    yp = np.array([1.0] * 10 + [0.0] * 10 + [1.0] * 10 + [np.nan] * 10)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = vfairness.conditional_demographic_disparity(yp, a, strata, min_group_size=1)

    # X alone: a 1.0 gap over the 20 rows that were assessable.
    assert got == pytest.approx(1.0, abs=1e-12)
    assert any("10 of 40" in str(w.message) for w in caught)

    # And with B unrecorded in BOTH strata there is no stratum left to assess at all.
    yp_worse = np.array([1.0] * 10 + [np.nan] * 10 + [1.0] * 10 + [np.nan] * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(
            vfairness.conditional_demographic_disparity(yp_worse, a, strata, min_group_size=1)
        )


def test_cdd_sealed_ci_gate_cannot_pass_on_unrecorded_decisions():
    """The sealed EU-hiring gate is the CI UPPER BOUND. On an all-unrecorded array it used to
    be 0.0, so a TOST equivalence gate passed at any margin. It must be NaN."""
    _y, a, s = _hiring_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = vfairness.conditional_demographic_disparity_with_ci(
            np.full(len(a), np.nan), a, s, n_bootstrap=100, random_state=0
        )
    assert np.isnan(res.point_estimate), f"point_estimate {res.point_estimate!r} is not NaN"
    assert res.upper_bound is None or np.isnan(res.upper_bound), (
        f"a TOST upper-bound gate would read {res.upper_bound!r} as a measurement"
    )


# ---------------------------------------------------------------------------
# net_benefit_parity
# ---------------------------------------------------------------------------


def test_net_benefit_parity_control_measures_scored_groups():
    """CONTROL. Real scores still measure: an equal, useful model reads 0.0 parity, and a
    genuinely useless group reads a real 2/3 gap."""
    y_true, a = _clinical_fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        equal = vfairness.net_benefit_parity(y_true, np.full(120, 0.9), a)
        useless_b = vfairness.net_benefit_parity(
            y_true, np.concatenate([np.full(60, 0.9), np.full(60, 0.0)]), a
        )
    assert equal == pytest.approx(0.0, abs=1e-12)
    assert useless_b == pytest.approx(2 / 3, abs=1e-9)
    assert not [w for w in caught if "SCORED" in str(w.message)], (
        "fully scored data must not raise the unscored-group warning"
    )

    # And the ordinary discrimination signal still fires (mirrors the spine test).
    rng = np.random.default_rng(6)
    n = 4000
    grp = rng.choice(["M", "F"], n)
    y = (rng.uniform(0, 1, n) < 0.4).astype(int)
    prob = np.where(
        grp == "M", np.clip(y * 0.5 + rng.uniform(0, 0.5, n), 0, 1), rng.uniform(0, 1, n)
    )
    assert vfairness.net_benefit_parity(y, prob, grp, threshold=0.4) > 0.1


def test_net_benefit_parity_refuses_when_nothing_was_scored():
    """THREE STATES. No probability was produced for anyone -> NaN, and the library's own
    gate must say could-not-check rather than PASS."""
    y_true, a = _clinical_fixture()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vfairness.net_benefit_parity(y_true, np.full(120, np.nan), a)

    assert np.isnan(result), f"expected NaN (insufficient evidence), got {result!r}"
    assert result != 0.0
    assert any("SCORED" in str(w.message) for w in caught)

    outcome, detail = check_threshold("net_benefit_parity", result, 0.1)
    assert outcome is ThresholdOutcome.COULD_NOT_CHECK
    assert "could not be measured" in detail


def test_net_benefit_parity_refuses_when_one_group_was_never_scored():
    """An unscored group must not be mistaken for a CLINICALLY USELESS one.

    Before the fix both read 0.6666666666666667. The useless-model reading is a real
    measurement and must survive (asserted in the control above); the unscored one must not.
    """
    y_true, a = _clinical_fixture()
    unscored_b = np.concatenate([np.full(60, 0.9), np.full(60, np.nan)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vfairness.net_benefit_parity(y_true, unscored_b, a)

    assert np.isnan(result), f"expected NaN, got {result!r} (the useless-model value)"
    assert any("'b'" in str(w.message) for w in caught), (
        "the warning must name the group whose net benefit is undefined"
    )


def test_net_benefit_parity_holds_the_scored_subset_to_min_group_size():
    """A 60-row group with 3 scored rows is not a 60-row group for this estimand: the size
    floor applies to what was actually scored, or the refusal just moves to a smaller n."""
    y_true, a = _clinical_fixture()
    prob = np.concatenate([np.full(60, 0.9), np.full(60, np.nan)])
    prob[60:63] = 0.0  # three of b's sixty subjects were scored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(vfairness.net_benefit_parity(y_true, prob, a))
        # ... while a fully scored b of the same size is measured, not refused.
        prob_ok = np.concatenate([np.full(60, 0.9), np.full(60, 0.0)])
        assert np.isfinite(vfairness.net_benefit_parity(y_true, prob_ok, a))


# ---------------------------------------------------------------------------
# Beta Go-Live Stage 1 audit, 2026-09-11. A NEW critical, same defect class as
# the one fixed above and found only because that fix removed the number hiding
# it: a stratum that lost a valid group's cell was still scored, on whichever
# groups survived, and given full population weight.
# ---------------------------------------------------------------------------

import numpy as _np2
import pytest as _pytest2

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    check_threshold as _check_threshold,
)
from vfairness.evaluation.vfairness_metrics.classification import (
    conditional_demographic_disparity as _cdd,
)

_PATTERN = _np2.arange(100) % 2  # a 50% approval rate, deterministic


def _two_strata(a_rows_in_y: int):
    """'a' approves at 100%, 'b' and 'c' at 50%. 'a' is valid overall (58 rows)
    but holds only ``a_rows_in_y`` rows in stratum y."""
    a = _np2.array(
        ["a"] * 58 + ["b"] * 100 + ["c"] * 100 + ["a"] * a_rows_in_y + ["b"] * 100 + ["c"] * 100
    )
    s = _np2.array(["x"] * 258 + ["y"] * (200 + a_rows_in_y))
    yp = _np2.concatenate(
        [
            _np2.ones(58, int),
            _PATTERN,
            _PATTERN,
            _np2.ones(a_rows_in_y, int),
            _PATTERN,
            _PATTERN,
        ]
    )
    return yp, a, s


def test_a_stratum_missing_a_valid_groups_cell_is_excluded_not_scored() -> None:
    """The measured disparity is 0.50 in the stratum where all three groups were
    examined. Scoring the other stratum on the two groups that survived diluted
    it to 0.2804 using a comparison that never looked at group 'a'."""
    yp, a, s = _two_strata(a_rows_in_y=2)
    with _pytest2.warns(UserWarning, match="EXCLUDED because a valid group"):
        value = _cdd(yp, a, s)
    assert value == _pytest2.approx(0.50, abs=1e-9), (
        f"got {value}; 0.2804 is the diluted answer this test exists to prevent"
    )


def test_the_dilution_could_flip_the_grade() -> None:
    """Why it is critical: the diluted value sits below a threshold the true one
    fails, so the defect turns a real finding into a pass."""
    yp, a, s = _two_strata(a_rows_in_y=2)
    with _pytest2.warns(UserWarning):
        value = _cdd(yp, a, s)
    assert _check_threshold("conditional_demographic_disparity", value, 0.40)[0].value == "fail"
    # the diluted 0.2804 would have passed the same threshold
    assert _check_threshold("conditional_demographic_disparity", 0.2804, 0.40)[0].value == "pass"


def test_control_a_fully_assessed_run_is_unchanged_and_warns_nothing() -> None:
    """Over-correction control: when every stratum has every valid group, the
    answer and the silence must both be preserved."""
    yp, a, s = _two_strata(a_rows_in_y=58)
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        value = _cdd(yp, a, s)
    assert value == _pytest2.approx(0.50, abs=1e-9)
    assert not [c for c in caught if "EXCLUDED because a valid group" in str(c.message)]


def test_control_no_assessable_stratum_is_nan_not_zero() -> None:
    """If excluding the partial strata leaves nothing, the answer is could-not-
    check. A 0.0 here would read as perfect conditional parity.

    The fixture matters and the first version of it was wrong: with only 2 rows
    of 'a' in total, 'a' was not a VALID group at all, so the function returned
    NaN from the `len(valid_groups) < 2` guard at the top and never reached the
    branch under test. The sabotage caught it by staying green. Here 'a' has 40
    rows, comfortably valid overall, but they are spread 4-per-stratum across ten
    strata, so every stratum is partial and none is assessable.
    """
    import math as _m

    n_strata = 10
    a_parts, s_parts, yp_parts = [], [], []
    for k in range(n_strata):
        a_parts += ["a"] * 4 + ["b"] * 50 + ["c"] * 50
        s_parts += [f"s{k}"] * 104
        yp_parts.append(_np2.ones(4, int))
        yp_parts.append(_np2.arange(50) % 2)
        yp_parts.append(_np2.arange(50) % 2)
    a = _np2.array(a_parts)
    s = _np2.array(s_parts)
    yp = _np2.concatenate(yp_parts)

    # the fixture must actually reach the branch: 'a' is valid overall
    from vfairness.evaluation.vfairness_metrics._grouping import GroupManager

    assert "a" in set(GroupManager(a, min_group_size=30).get_valid_groups()), (
        "fixture does not exercise the branch: 'a' is not a valid group, so the "
        "function returns NaN from the <2-valid-groups guard instead"
    )

    with _pytest2.warns(UserWarning, match="EXCLUDED because a valid group"):
        value = _cdd(yp, a, s)
    assert _m.isnan(value), f"got {value}; nothing was assessable, so this must be NaN"
