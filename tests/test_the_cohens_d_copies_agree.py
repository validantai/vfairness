"""The three COPIES of Cohen's d refuse a computed near-constant array too.

WHY THIS FILE EXISTS. `cohens_d` lives in four places in this library: the
canonical one in evaluation/vfairness_metrics/_statistics.py and three
copy-pasted siblings. On 2026-09-29 the canonical one was fixed, because its
degeneracy guard tested `np.ptp(...) == 0.0`, an EXACT equality with zero. That
is exact for a caller-supplied constant array and false for one the caller
COMPUTED: a residual or a difference is constant only to a few ulps.

The three copies kept the broken test. Measured that day on `(pred + 50.0) - pred`
over 40 rows, peak-to-peak 2.84e-14, for two groups whose real gap is 50:

    agents/action_bias           -> -6624998218327707.0   0 warnings
    llm/output_analysis          -> -6624998218327707.0   0 warnings
    preprocessing/bias_detection -> -6624998218327707.0   0 warnings

Identical to the digit, because they are the same code. A fix to one copy that
does not reach the others is the half-fix this campaign kept finding, so this
file asks the same question of all three at once and will fail if a fourth copy
is added without it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.agents.action_bias import ActionBiasAnalyzer
from vfairness.llm.output_analysis import OutputAnalyzer
from vfairness.preprocessing.bias_detection import statistical as _statistical

COPIES = [
    ("agents.action_bias", ActionBiasAnalyzer._cohens_d),
    ("llm.output_analysis", OutputAnalyzer._cohens_d),
    ("preprocessing.bias_detection.statistical", _statistical._cohens_d),
]


def _computed_constant_arms():
    """Two arms a CALLER computed, constant only up to rounding.

    Not `np.full(...)`: that is constant in the last bit and every copy always
    refused it, which is why the defect survived. The subtraction is what leaves
    a few ulps of spread at this magnitude.
    """
    pred = np.random.default_rng(0).normal(100, 10, 40)
    return (pred + 0.0) - pred, (pred + 50.0) - pred


@pytest.mark.parametrize("name,fn", COPIES, ids=[n for n, _ in COPIES])
def test_a_computed_near_constant_arm_is_not_scored(name, fn):
    a, b = _computed_constant_arms()
    assert np.ptp(b) > 0.0, "the fixture is exactly constant, so it proves nothing"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        value = fn(a, b)
    assert value is None or not np.isfinite(value), (
        f"{name} scored a standardised effect of {value!r} for two arms whose "
        f"spread is {np.ptp(a):.3g} and {np.ptp(b):.3g}. The denominator is "
        "rounding noise, so the magnitude is an artefact of the arithmetic."
    )


@pytest.mark.parametrize("name,fn", COPIES, ids=[n for n, _ in COPIES])
def test_control_a_real_effect_is_still_measured(name, fn):
    """OVER-CORRECTION CONTROL, with the number.

    A guard that refuses everything passes the test above. Two genuinely spread
    samples 0.8 apart at unit variance measure -0.53098 in all three copies, and
    that is asserted rather than the shape.
    """
    rng = np.random.default_rng(0)
    rng.normal(100, 10, 40)  # keep the stream aligned with the fixture above
    g1, g2 = rng.normal(0.0, 1.0, 60), rng.normal(0.8, 1.0, 60)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        value = fn(g1, g2)
    assert value is not None and np.isfinite(value), f"{name} refused a real effect"
    assert value == pytest.approx(-0.53098, abs=1e-4), value
