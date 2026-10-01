"""Permutation importance may not be divided by a denominator nobody measured.

WHY. ``FeatureAttributionExplainer._scored_permutation`` scores the model with
R^2 and reports each feature's importance as the DROP in that score when the
feature's column is shuffled. R^2's denominator is the target's total sum of
squares, so when the target has no variation there is no denominator and no
proportion of it can be explained.

The function used to pick ``sklearn.metrics.r2_score`` when it imported and fall
back to a local NumPy R^2 whose denominator was

    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2)) or 1e-12

``0.0 or 1e-12`` is ``1e-12`` in Python, so a genuinely zero sum of squares was
replaced by one part in a trillion and the quotient was published as a feature
importance. Nothing warned. The two arms did not even agree with each other:
measured on this repo at n = 200 with y = np.linspace(1e-200, 2e-200, 200),
whose 200 distinct values every constancy test correctly passes while every
squared deviation underflows to zero,

    NumPy arm:   f0 8.79e+12, f1 2.56e+13, f2 2.14e+12, notes [], 0 warnings
    sklearn arm: f0 0.0,      f1 0.0,      f2 0.0,      notes [], 0 warnings

so one arm fabricated a magnitude, the other fabricated a neutral value, and
which one a reader got depended on whether sklearn happened to import.

The same family, one step further out and live on the DEFAULT path: a target
that is constant only to the resolution of the arithmetic that produced it.
``np.unique`` is exact, so y = (base + 100.1) - base for
base = np.linspace(1e6, 1e8, 200) has five distinct values and walked past the
constant-target guard with ss_tot 2.28e-15, and the published importances were
f0 3.86e+15, f1 1.13e+16, f2 9.42e+14, all 'increase', notes [], 0 warnings.

The fix is one R^2 predicate, ``regression._r2_or_not_measured``, the same one
both R^2 sites in ``regression.py`` were moved onto on the same day, applied
ABOVE the point where the arms used to be chosen: to the target on its own, to
the base score every importance is a drop from, and to each permuted score.

The controls matter as much as the refusals here. A guard that refuses
everything passes every refusal test, so the healthy fixture's REAL numbers are
asserted, and a genuinely terrible model (R^2 about -5) must keep its measured
importances rather than be swept up by the magnitude band.
"""

from __future__ import annotations

import math
import sys
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.attribution import FeatureAttributionExplainer

WEIGHTS = np.array([2.0, 0.5, 0.1])
N = 200


def _model(A):
    return np.asarray(A, dtype=float) @ WEIGHTS


def _features(seed=7, sort_first_column=False):
    X = np.random.default_rng(seed).normal(0, 1, (N, 3))
    if sort_first_column:
        X = X[np.argsort(X[:, 0])]
    return X


def _run(predict, X, y, n_repeats=3, random_state=0):
    """Return (result, [warning messages]) with warnings captured, never swallowed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FeatureAttributionExplainer(predict, ["f0", "f1", "f2"]).global_importance(
            X, y=y, n_repeats=n_repeats, random_state=random_state
        )
    return result, [str(w.message) for w in caught]


def _all_refused(result):
    return all(math.isnan(c.importance) for c in result.contributions) and all(
        c.direction == "not_assessed" for c in result.contributions
    )


def _underflowing_target():
    """200 distinct target values whose total sum of squares is exactly 0.0.

    Every squared deviation is about 1e-401 and underflows, so this is the input
    that made ``or 1e-12`` fire. It is NOT a constant target: n_unique is 200 and
    its relative spread is 0.5, which is why no spread test can catch it and the
    sum of squares has to be tested for what it is.
    """
    y = np.linspace(1e-200, 2e-200, N)
    assert np.unique(y).size == N
    assert float(np.sum((y - np.mean(y)) ** 2)) == 0.0
    return y


def _near_constant_target():
    """100.1 carrying the rounding of the 1e8 arithmetic that produced it.

    The base must SPAN BINADES or the noise does not exist: np.spacing is constant
    inside one binade, so (base + c) - base rounds identically for every element
    and the spread comes out exactly 0.0, which the old exact test already caught.
    """
    base = np.linspace(1e6, 1e8, N)
    y = (base + 100.1) - base
    assert np.unique(y).size > 1, "the fixture must PASS the exact constancy test"
    assert 0.0 < float(np.ptp(y)) < 1e-7
    return y


# ---------------------------------------------------------------------------
# REFUSALS
# ---------------------------------------------------------------------------


def test_a_target_whose_sum_of_squares_is_zero_has_no_importance_to_report():
    """BEFORE: f0 8.79e+12 / f1 2.56e+13 / f2 2.14e+12 on the NumPy arm, 0.0 on
    the sklearn arm, both silent. The denominator was 1e-12, a number no part of
    the data produced."""
    result, caught = _run(_model, _features(), _underflowing_target())

    assert _all_refused(result), [c.importance for c in result.contributions]
    assert result.notes, "notes is the field that exists for this disclosure"
    assert "sum of squares is zero" in result.notes[0]
    assert len(caught) == 1, caught
    assert "not measured" in caught[0] or "NOT measured" in caught[0]


def test_the_refusal_does_not_depend_on_whether_sklearn_imports(monkeypatch):
    """The two arms are gone, so they cannot disagree. Blocking
    ``sklearn.metrics`` used to select the ``or 1e-12`` arm; it must now change
    nothing at all, on the refusal AND on the healthy number."""
    y_zero = _underflowing_target()
    X = _features()
    with_sklearn, warned_with = _run(_model, X, y_zero)

    monkeypatch.setitem(sys.modules, "sklearn.metrics", None)
    without_sklearn, warned_without = _run(_model, X, y_zero)

    assert _all_refused(with_sklearn) and _all_refused(without_sklearn)
    assert with_sklearn.notes == without_sklearn.notes
    assert warned_with == warned_without

    # And the healthy case is the same number to the bit either way, which is the
    # control on this pin: the fix removed an arm, it did not remove R^2.
    y_ok = _features() @ WEIGHTS
    healthy_without, _ = _run(_model, X, y_ok)
    monkeypatch.undo()
    healthy_with, _ = _run(_model, X, y_ok)
    assert [c.importance for c in healthy_without.contributions] == [
        c.importance for c in healthy_with.contributions
    ]
    assert all(math.isfinite(c.importance) for c in healthy_with.contributions)


def test_a_target_constant_to_the_resolution_of_its_own_arithmetic_is_refused():
    """BEFORE: f0 3.86e+15, f1 1.13e+16, f2 9.42e+14, all 'increase', notes [],
    ZERO warnings, on the DEFAULT sklearn path. Caught by the magnitude of the
    base score, because no tolerance derived from the target alone can recognise
    noise inherited from arithmetic three orders of magnitude larger."""
    result, caught = _run(_model, _features(), _near_constant_target())

    assert _all_refused(result), [c.importance for c in result.contributions]
    assert result.notes and "not a measurement" in result.notes[0]
    assert "every importance is a DROP from that score" in result.notes[0]
    assert len(caught) == 1, caught


def test_a_perfect_fit_on_a_target_with_no_variation_is_not_a_perfect_fit():
    """THE CASE NO MAGNITUDE BAND CAN EVER SEE, and the reason the target's own
    resolution test cannot be dropped in favour of one.

    y = 100.0 + 1e-13 * arange(200)/200 has EIGHT distinct values, so the exact
    ``np.unique`` test passes it, and its entire variation is 9.95e-14 out of 100,
    which is rounding at that magnitude. Predicted exactly, ss_res is 0.0 and R^2
    is 1.0: a perfect fit, comfortably inside every plausible band.

    Measured on HEAD against the fixed file on this fixture:
        HEAD  -> f0 1.949561403508772 'increase', f1 0.0 'neutral',
                 f2 0.0 'neutral', notes [], ZERO warnings
        FIXED -> all three NaN 'not_assessed', one warning, one note
    HEAD published a named finding for f0 AND an exact neutral verdict for the
    other two, out of a denominator of 1.84e-25.
    """
    y = 100.0 + 1e-13 * (np.arange(N) / N)
    assert np.unique(y).size > 1, "the fixture must PASS the exact constancy test"
    X = np.random.default_rng(3).normal(0, 1, (N, 3))
    X[:, 0] = y  # so the model can predict the target exactly

    result, caught = _run(lambda A: np.asarray(A, dtype=float)[:, 0], X, y)

    assert _all_refused(result), [c.importance for c in result.contributions]
    assert "constant target" in result.notes[0]
    assert "rounding noise at this magnitude" in result.notes[0]
    assert len(caught) == 1, caught


def test_an_exactly_constant_target_keeps_the_wording_it_already_had():
    """The 2026-09-16 refusal is not disturbed by the new axes beside it."""
    result, caught = _run(_model, _features(), np.full(N, 100.0))

    assert _all_refused(result)
    assert "constant target" in result.notes[0]
    assert "1 distinct finite target value(s) in 200 rows" in result.notes[0]
    assert len(caught) == 1


def test_a_target_with_non_finite_rows_is_refused_with_the_count():
    """BEFORE: the sklearn arm raised ValueError out of sklearn's own validator
    and the NumPy arm returned NaN for every feature with notes EMPTY. Neither is
    this function's three-state disclosure."""
    y = _features() @ WEIGHTS
    y[:100] = np.nan
    result, caught = _run(_model, _features(), y)

    assert _all_refused(result)
    assert "100 of 200 target value(s) are not finite" in result.notes[0]
    assert "NOT silently dropped" in result.notes[0]
    assert len(caught) == 1


def test_a_permuted_score_that_is_not_a_measurement_is_not_averaged_in():
    """BOTH DIRECTIONS IN ONE TEST.

    A model that cannot cope with a reordering of column 0 returns a non-finite
    prediction for every shuffle of it. BEFORE: the drop was
    ``base_score - -inf`` = inf and ``float(abs(inf))`` was published as f0's
    importance with notes EMPTY. AFTER: f0 is NaN and named in a note, while f1
    and f2, whose shuffles never disturb column 0, keep their REAL measured
    numbers.
    """
    X = _features(sort_first_column=True)
    y = X @ WEIGHTS

    def order_sensitive(A):
        A = np.asarray(A, dtype=float)
        out = A @ WEIGHTS
        if not np.all(np.diff(A[:, 0]) >= 0):
            out = out.copy()
            out[0] = np.inf
        return out

    result, caught = _run(order_sensitive, X, y)
    by_name = {c.feature: c for c in result.contributions}

    assert math.isnan(by_name["f0"].importance)
    assert by_name["f0"].direction == "not_assessed"
    assert by_name["f0"].signed_value is None

    assert by_name["f1"].importance == pytest.approx(0.10446320004864425, rel=1e-9)
    assert by_name["f2"].importance == pytest.approx(0.004319115308246431, rel=1e-9)

    assert any("was not a measurement" in note for note in result.notes)
    assert any("f0: a permuted R^2 came back -inf" in note for note in result.notes)
    assert any("f0" in message for message in caught)
    # and it must NOT be described as a shuffle that never rearranged the column,
    # which is a different cause with a different remedy.
    assert not any("never actually" in note for note in result.notes)


# ---------------------------------------------------------------------------
# CONTROLS. A guard that refuses everything passes every refusal test above.
# ---------------------------------------------------------------------------


def test_a_healthy_target_keeps_its_real_numbers_unchanged():
    """The values measured on HEAD before the fix, BIT-IDENTICAL after it.

    Established by loading HEAD's attribution.py beside the fixed one and running
    both on this fixture: 1.8275907665671631 / 0.10806639310398036 /
    0.003878175699399596 from each, notes [] and zero warnings from each. The fix
    removed an arm of a dispatch; it did not change R^2.
    """
    X = _features()
    result, caught = _run(_model, X, X @ WEIGHTS)
    by_name = {c.feature: c.importance for c in result.contributions}

    assert by_name["f0"] == pytest.approx(1.8275907665671631, rel=1e-12)
    assert by_name["f1"] == pytest.approx(0.10806639310398036, rel=1e-12)
    assert by_name["f2"] == pytest.approx(0.003878175699399596, rel=1e-12)
    assert by_name["f0"] > by_name["f1"] > by_name["f2"]
    assert all(c.direction == "increase" for c in result.contributions)
    assert result.notes == []
    assert caught == []


def test_a_genuinely_terrible_model_still_gets_measured_importances():
    """The magnitude band refuses |R^2| >= 1e6, i.e. a squared error a million
    times the target's own variation. A model with R^2 about -5 is merely bad,
    and bad is a measurement: refusing it would trade one fabrication for a
    refusal of the real thing."""
    X = _features(sort_first_column=True)
    y_unrelated = np.random.default_rng(7).normal(0, 1, N)
    result, caught = _run(_model, X, y_unrelated)

    assert all(math.isfinite(c.importance) for c in result.contributions), [
        c.importance for c in result.contributions
    ]
    # HEAD's own number on this fixture, bit-identical after the fix.
    assert result.contributions[0].importance == pytest.approx(0.46441076435086764, rel=1e-12)
    assert result.notes == []
    assert caught == []


def test_the_label_free_path_is_untouched():
    """``global_importance`` without y never computed an R^2, so none of this
    reaches it, and its numbers must not move."""
    X = _features()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = FeatureAttributionExplainer(_model, ["f0", "f1", "f2"]).global_importance(
            X, n_repeats=3, random_state=0
        )
    assert result.method == "permutation_numpy"
    assert all(math.isfinite(c.importance) for c in result.contributions)
    assert result.top(1)[0].feature == "f0"
    assert [str(w.message) for w in caught] == []
