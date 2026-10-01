"""Both Cohen's d arms of compute_regression_effect_sizes, not just the residual.

WHY THIS FILE EXISTS, and it is a record of a verification failure as much as a
defect. On 2026-09-29 the RESIDUAL arm was given an operand-scaled tolerance,
because a within-group-constant error is constant only up to rounding and was
publishing an effect size of -6.9e15. That fix was verified by running the
function and watching cohens_d_residuals become nan.

ONE OF TWO ARMS WAS CHECKED. `d_pred = cohens_d(pred1, pred2)` sits one line
above and receives no tolerance at all, and it cannot receive the residual's:
that bound is derived from the operands the residual is subtracted FROM, and the
predictions are handed in already computed, so nothing in the data says what
produced them.

Measured 2026-09-30, base spanning 1e6 to 1e8, predictions `(base + 0.1) - base`
against `(base + 0.2) - base`, which are 0.1 and 0.2 carrying 7.5e-9 of spread
inherited from 1e8 arithmetic:

    cohens_d_predictions = -37430695.85, interpretation 'large', ZERO warnings

THE FIXTURE'S BASE MUST SPAN BINADES, and this is why two earlier attempts to
reproduce it failed. `np.spacing` is constant within a binade, so `(base + c) -
base` rounds identically for every element and the spread comes out EXACTLY zero,
which the existing guard refuses correctly and for the wrong reason. Cross a
power-of-two boundary and the spread appears.

The value is PUBLISHED rather than refused, because it is arithmetically correct
for the arrays supplied and refusing it would discard a real computation on a
guess about provenance. What was corrected is the LABEL and the silence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.regression import compute_regression_effect_sizes


def _spanning_binades(n: int = 40):
    """Predictions of 0.1 and 0.2 carrying rounding noise from 1e8 arithmetic."""
    base = np.linspace(1e6, 1e8, n)
    p1, p2 = (base + 0.1) - base, (base + 0.2) - base
    assert np.ptp(p1) > 0.0, "the fixture is exactly constant, so it proves nothing"
    y_pred = np.concatenate([p1, p2])
    y_true = y_pred + np.random.default_rng(0).normal(0.0, 1.0, 2 * n)
    return y_true, y_pred, np.array(["A"] * n + ["B"] * n)


def _only(out):
    return list(out.values())[0]


def test_a_degenerate_denominator_is_not_graded_large():
    y_true, y_pred, attr = _spanning_binades()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r = _only(compute_regression_effect_sizes(y_true, y_pred, attr))
    assert abs(r["cohens_d_predictions"]) > 1e6, r["cohens_d_predictions"]
    assert r["interpretation"] != "large", (
        "an effect size of tens of millions was graded with the strongest band this "
        "field carries; the denominator is resolution, not variation"
    )
    assert "not interpretable" in r["interpretation"], r["interpretation"]


def test_the_caller_is_told_and_not_left_to_read_it():
    y_true, y_pred, attr = _spanning_binades()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_regression_effect_sizes(y_true, y_pred, attr)
    assert any("negligible beside the between-group gap" in str(w.message) for w in caught), [
        str(w.message)[:90] for w in caught
    ]


@pytest.mark.parametrize(
    "gap,expected_d,expected",
    [(20.0, -1.7184, "large"), (5.0, -0.3561, "small")],
    ids=["a_real_large_effect", "a_real_small_effect"],
)
def test_control_a_real_effect_is_still_graded_silently(gap, expected_d, expected):
    """OVER-CORRECTION CONTROL. A guard that refuses everything passes the two
    tests above, so this asserts the NUMBER and the band it earns.

    The two bands are the ones this fixture actually produces, measured rather
    than assumed: the generator stream advances between the two cases, so a gap
    of 5.0 in the means lands at d = -0.3561, 'small'. Two earlier versions of
    this test were red: the first asserted 'medium' because that is what a gap of
    5.0 sounds like, and the second used a number measured from a loop that shared
    ONE generator across both cases, while parametrising gives each case a fresh
    one. The band is scenery; what the control establishes is that a real effect
    is still graded and still silent."""
    n, rng = 40, np.random.default_rng(3)
    pred = np.concatenate([rng.normal(100, 10, n), rng.normal(100 + gap, 10, n)])
    y_true = pred + rng.normal(0.0, 2.0, 2 * n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = _only(compute_regression_effect_sizes(y_true, pred, np.array(["A"] * n + ["B"] * n)))
    assert r["cohens_d_predictions"] == pytest.approx(expected_d, abs=1e-4)
    assert r["interpretation"] == expected, (r["cohens_d_predictions"], r["interpretation"])
    assert "not interpretable" not in r["interpretation"]
    assert not caught, [str(w.message)[:80] for w in caught]


def test_control_the_residual_arm_still_refuses_what_it_was_fixed_for():
    """The 2026-09-29 fix must survive this change. A within-group-constant error
    still reaches nan rather than 6.9e15."""
    n, rng = 40, np.random.default_rng(0)
    pred = rng.normal(100, 10, n)
    y_pred = np.concatenate([pred, pred])
    y_true = np.concatenate([pred + 0.0, pred + 50.0])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r = _only(compute_regression_effect_sizes(y_true, y_pred, np.array(["A"] * n + ["B"] * n)))
    assert not np.isfinite(r["cohens_d_residuals"]), r["cohens_d_residuals"]
