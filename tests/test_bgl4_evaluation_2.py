"""BGL-4 AUDIT RECORD, batch A-evaluation-2. Six DEFECTS, each executed here.

WHAT THIS FILE IS. It is not a pin of correct behaviour. Every assertion below
records the behaviour the library has TODAY, which the audit judged WRONG, so
that the finding is reproducible by running a test rather than by reading a
report. Each test names the honest answer in its docstring.

WHEN ONE OF THESE IS FIXED, THE TEST GOES RED. That is the point: invert the
assertion then and move it into the module's real pin file. Do NOT loosen an
assertion here to get green, and do not delete one without recording where the
finding went.

ALL FIVE WERE FIXED ON 2026-09-27 (BGL-5), so every assertion below has been
INVERTED in place: each test now asserts the CORRECTED behaviour and would go red
if the defect came back. Nothing was deleted and no subject was changed, so the
test names still record which finding each one came from, and the docstrings still
describe the defect and name the honest answer, which is now the asserted answer.
The full pins, with the over-correction controls and the exact measured values,
are in tests/test_bgl5_evaluation_2.py; these are the audit's own reproductions,
kept as the second, independent way each fix can fail.

The six, and the grade each one refutes:

  D1/D2  bootstrap_over_index   graded PROVEN on the claim that every degenerate
                                resampling class was swept. The ``groups=``
                                (stratified) branch was never swept at all: it
                                has no length check, so a groups array shorter
                                than ``n`` yields an interval built from a
                                silently truncated slice of the rows, which does
                                not even contain the point estimate it is
                                published beside; and one row per stratum yields
                                a zero-width interval with no warning.
  D3     minimum_detectable_effect  graded PROVEN. The pins cover n < 2, which
                                now refuses. ``min(mde, 1.0)`` still
                                manufactures a finite floor of exactly 1.0,
                                silently, for arms whose arithmetic floor
                                exceeds 1.0, i.e. for designs that can detect
                                nothing at all. The function's own comment calls
                                1.0 the lie it was fixed for.
  D4     power_warning          graded PROVEN on "a subgroup with too few rows
                                is could-not-check, not low power". True for
                                n = 0 and n = 1, which the pins cover. At n = 2
                                it reports the exact sentence the pin's own
                                docstring names as the defect.
  D5     validate_probabilities graded SEMI-PROVEN, "no fabrication seen". The
                                NaN guard is gated on a floating dtype, so an
                                object-dtype column of probabilities carrying
                                NaN is accepted, and a public consumer then
                                publishes per-bin calibration numbers over rows
                                that have no probability.
  D6     selection_rate_disparity_matrix  graded PROVEN. The 0/0 refusal it was
                                graded for does hold. With fewer than two
                                INTERPRETABLE groups the worst-pair pool falls
                                back to every group, so a two-row group whose
                                own tier is "invalid" sets the headline
                                min_ratio to 0.0, in silence.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.evaluation.vfairness_metrics.classification as C
from vfairness.evaluation.vfairness_metrics._statistics import (
    bootstrap_over_index,
    minimum_detectable_effect,
    power_warning,
)
from vfairness.evaluation.vfairness_metrics._validation import validate_probabilities


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


# ---------------------------------------------------------------------------
# D1  bootstrap_over_index: groups is never length-checked against n
# ---------------------------------------------------------------------------


def test_defect_bootstrap_over_index_builds_an_interval_from_a_truncated_slice():
    """HONEST ANSWER: refuse, because ``groups`` does not describe the ``n`` rows.

    ``bootstrap_over_index`` builds ``idx = np.arange(n)`` itself and hands it to
    ``stratified_bootstrap_ci`` with the caller's ``groups``. That function takes
    its strata from ``np.where(groups == g)[0]``, so a groups array of length 5
    against n = 20 can only ever resample indices 0..4. The point estimate is
    computed over all 20 rows and the interval over 5 of them, and the result
    still reports ``sample_size = 20``. No length check and no warning exist.

    INVERTED 2026-09-27: the length check was added and this now pins the
    refusal. Before: point_estimate 9.5, interval [0.845, 3.155] (the mean of
    rows 0..4), sample_size 20, warnings []. After: InvalidDataError naming both
    lengths.
    """
    from vfairness.exceptions import InvalidDataError

    data = np.arange(20, dtype=float)
    with pytest.raises(InvalidDataError) as excinfo:
        bootstrap_over_index(
            20,
            lambda idx: float(np.mean(data[idx])),
            groups=np.array(["a"] * 5),
            n_bootstrap=50,
            random_state=3,
        )
    assert "5" in str(excinfo.value) and "20" in str(excinfo.value), str(excinfo.value)


def test_defect_bootstrap_over_index_publishes_a_zero_width_stratified_interval():
    """HONEST ANSWER: NaN bounds, as the ``len(data) < 2`` branch already gives.

    A stratum of one row resamples to itself on every draw, so every bootstrap
    replicate is identical and the percentile interval has zero width. A
    zero-width confidence interval asserts the statistic is known exactly. The
    guard that catches this for the WHOLE sample (``len(data) < 2``) cannot see
    it per stratum.

    INVERTED 2026-09-27: the per-stratum guard was added. Before: lower_bound ==
    upper_bound == 0.5, standard_error 0.0, warnings []. After: nan bounds, no
    standard error, and a UserWarning, with the point estimate kept because it
    was measured.
    """
    data = np.arange(2, dtype=float)
    result, messages = _caught(
        lambda: bootstrap_over_index(
            2,
            lambda idx: float(np.mean(data[idx])),
            groups=np.array(["a", "b"]),
            n_bootstrap=50,
            random_state=3,
        )
    )
    assert math.isnan(result.lower_bound) and math.isnan(result.upper_bound)
    assert result.standard_error is None
    assert result.point_estimate == pytest.approx(0.5)
    assert any("single row" in m for m in messages), f"the zero width is silent: {messages}"


# ---------------------------------------------------------------------------
# D3  minimum_detectable_effect: the 1.0 clamp is still a silent finite floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n1,n2", [(2, 2), (2, 3), (3, 3), (2, 5), (2, 100)])
def test_defect_minimum_detectable_effect_clamps_an_unattainable_floor_to_one(n1, n2):
    """HONEST ANSWER: NaN and a warning, the same as the two branches beside it.

    The arithmetic floor for each of these designs is greater than 1.0, i.e. not
    even a total 100-point gap could be detected, so no attainable minimum
    detectable effect exists. ``return float(min(mde, 1.0))`` reports 1.0 and
    says nothing, and 1.0 is FINITE, which is precisely why the could-not-check
    path in ``power_warning`` cannot fire (see D4).

    INVERTED 2026-09-27: the clamp was removed. Before: 1.0 with no warning for
    all five arm pairs. After: nan and a warning for all five.
    """
    z_alpha, z_power = 1.959963984540054, 0.8416212335729143
    raw = (z_alpha + z_power) * math.sqrt(0.25 * (1 / n1 + 1 / n2))
    assert raw > 1.0, "fixture no longer selects a clamped design"

    value, messages = _caught(lambda: minimum_detectable_effect(n1, n2))
    assert math.isnan(value), f"({n1}, {n2}) still returns a floor: {value}"
    assert value != 1.0
    assert messages, "the refusal is silent again"


# ---------------------------------------------------------------------------
# D4  power_warning: could-not-check collapsed into the low-power band at n=2
# ---------------------------------------------------------------------------


def test_defect_power_warning_grades_a_design_that_can_detect_nothing():
    """HONEST ANSWER: "COULD NOT BE ASSESSED", which is what n=0 and n=1 get.

    tests/test_tier1_statistics_disclosure.py names this very sentence as the
    defect it fixed: 'It used to read "Very low statistical power (n=0). This
    test can only detect disparities larger than 100%", a floor computed from the
    clamped 1.0.' The pins stop at n = 1. At n = 2 the sentence is still
    produced, from the same clamped 1.0.

    INVERTED 2026-09-27: the clamp that produced the 100% floor is gone, so n=2
    now reads the same as n=0 and n=1. Before: "Very low statistical power (n=2).
    This test can only detect disparities larger than 100%". After: "Statistical
    power COULD NOT BE ASSESSED (n=2) ...".
    """
    verdict = power_warning(2, 100)
    assert verdict is not None
    assert "COULD NOT BE ASSESSED" in verdict, f"n=2 is graded into a band again: {verdict}"
    assert "Very low statistical power (n=2)" not in verdict
    assert "larger than 100%" not in verdict
    # The neighbours it now agrees with.
    assert "COULD NOT BE ASSESSED" in power_warning(1, 100)
    assert "COULD NOT BE ASSESSED" in power_warning(0, 100)


# ---------------------------------------------------------------------------
# D5  validate_probabilities: the NaN guard is gated on a floating dtype
# ---------------------------------------------------------------------------


def test_defect_validate_probabilities_accepts_nan_in_an_object_dtype_column():
    """HONEST ANSWER: the same InvalidDataError the float dtype gets.

    The guard reads ``np.issubdtype(probs.dtype, np.floating) and
    np.any(np.isnan(probs))``. An object-dtype array skips it, and then
    ``probs < 0`` / ``probs > 1`` are False for NaN, which is the exact mechanism
    the comment above that guard says it was written to close.

    INVERTED 2026-09-27: the guard is no longer dtype-gated (pd.isna). Before:
    the object column returned None with no warning. After: the same
    InvalidDataError the float column gets.
    """
    floats = np.array([0.5, np.nan, 0.7])
    with pytest.raises(Exception) as excinfo:
        validate_probabilities(floats, "y_prob")
    assert "NaN" in str(excinfo.value)

    objects = np.array([0.5, np.nan, 0.7], dtype=object)
    with pytest.raises(Exception) as object_excinfo:
        validate_probabilities(objects, "y_prob")
    assert "NaN" in str(object_excinfo.value)
    assert type(object_excinfo.value) is type(excinfo.value), (
        "the two dtypes must refuse the same way"
    )


def test_defect_calibration_publishes_bins_over_rows_with_no_probability():
    """The consumer half of D5: numbers computed over unscored rows.

    ``expected_calibration_error`` validates with the function above. With the
    NaN column in float dtype it refuses. With the identical values in object
    dtype it returns a result whose per-bin table sweeps every unscored row into
    the last bin and reports an accuracy for it. Only numpy RuntimeWarnings are
    emitted; nothing says a fifth of the rows carry no probability.

    INVERTED 2026-09-27: both dtypes now refuse. Before, the object column
    returned bin counts [22, 20, 18, 22, 13, 14, 20, 20, 10, 41] with
    accuracies[-1] = 0.4878 over the 20 rows that had no probability, and no
    UserWarning. After: InvalidDataError naming the 20.
    """
    from vfairness import expected_calibration_error

    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, n)
    probs = np.clip(rng.random(n), 0.01, 0.99)
    probs[:20] = np.nan

    with pytest.raises(Exception) as excinfo:
        expected_calibration_error(y_true, probs)
    assert "NaN" in str(excinfo.value)

    as_object = pd.Series(list(probs), dtype=object)
    with pytest.raises(Exception) as object_excinfo:
        expected_calibration_error(y_true, as_object)
    assert "20 NaN value(s)" in str(object_excinfo.value), (
        "the per-bin table is published over unscored rows again"
    )


# ---------------------------------------------------------------------------
# D6  selection_rate_disparity_matrix: an "invalid"-tier group sets the headline
# ---------------------------------------------------------------------------


def test_defect_selection_rate_headline_comes_from_an_uninterpretable_group():
    """HONEST ANSWER: NaN, or at minimum a warning naming the fallback.

    The worst-pair pool is ``interp if len(interp) >= 2 else names``, under a
    comment reading "a 4-person group's rate is noise and must not define the
    headline disparity". With one interpretable group the fallback puts every
    group back in, so a TWO-row group whose own tier is "invalid" and whose
    ``interpretable`` flag is False sets min_ratio to 0.0, the starkest
    adverse-impact reading the statistic has, with no warning at all.
    ``scripts/pulse_bias_recall_harness.py::_flagged_disparity`` flags any
    attribute whose min_ratio is below 0.8 without consulting that flag. That
    harness is internal tooling and is NOT in the published repository, so the
    name above is here to say where the fabricated 0.0 became a false finding,
    not as a file to open. The behaviour this test pins is public and is
    documented in docs/API_REFERENCE.md.

    INVERTED 2026-09-27: the pool no longer falls back to every group. Before:
    min_ratio 0.0 for the pair ('B', 'A'), max_difference 0.8, warnings []. After:
    min_ratio and max_difference nan, headline_basis 'no_interpretable_pair', a
    UserWarning naming B, and the old values under headline_over_all_groups.
    """
    y_pred = np.array([1] * 40 + [0] * 10 + [0] * 2)
    sens = np.array(["A"] * 50 + ["B"] * 2)
    out, messages = _caught(lambda: C.selection_rate_disparity_matrix(y_pred, sens))

    assert out["rates"]["B"]["n"] == 2
    assert out["rates"]["B"]["interpretable"] is False
    assert out["rates"]["B"]["tier"] == "invalid"
    assert math.isnan(out["min_ratio"]), (
        f"min_ratio is {out['min_ratio']!r}: the two-row group set the headline again"
    )
    assert out["min_ratio_pair"] is None
    assert out["headline_basis"] == "no_interpretable_pair"
    # The value that used to be the headline is disclosed, not destroyed.
    assert out["headline_over_all_groups"]["min_ratio"] == pytest.approx(0.0)
    assert out["headline_over_all_groups"]["min_ratio_pair"] == ("B", "A")
    assert any("fewer than two groups are interpretable" in m for m in messages), (
        f"the fallback is silent again: {messages}"
    )
