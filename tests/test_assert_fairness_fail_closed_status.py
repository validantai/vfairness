"""`assert_fairness` is a release gate, so could-not-check must never pass.

Until 2026-09-07 it raised only on `TestStatus.FAILED` and let every other
status through. Both cases that reach it are `SKIPPED` with `actual_value=None`
and they are indistinguishable at that point:

  * a metric that is undefined on the data, e.g. `equal_opportunity_difference`
    where no group carries a positive label; and
  * A METRIC NAME THAT DOES NOT EXIST.

So a typo in the metric name made the gate pass, silently, on any model. The
README already promised the opposite ("refuses a metric whose value is NaN ...
fail closed"), which is the shape this project treats as most serious: a surface
claiming a check it does not perform.
"""

import numpy as np
import pytest

from vfairness.operations.cicd.testing import FairnessAssertionError, assert_fairness

# No positive labels anywhere, so any TPR-based metric is undefined.
_Y_UNDEFINED = np.array([0] * 40)
_P_UNDEFINED = np.array([0] * 40)
_G = np.array(["a"] * 20 + ["b"] * 20)

# Measurable, and comfortably inside its threshold.
_Y_OK = np.array([0, 1] * 20)
_P_OK = np.array([0, 1] * 20)


@pytest.mark.parametrize(
    "metric",
    ["equal_opportunity_difference", "equalized_odds_difference"],
    ids=["undefined-tpr", "undefined-odds"],
)
def test_a_metric_that_could_not_be_measured_does_not_pass(metric):
    with pytest.raises(FairnessAssertionError) as e:
        assert_fairness(_Y_UNDEFINED, _P_UNDEFINED, _G, metric=metric, threshold=0.1)
    assert "NOT MEASURABLE" in str(e.value)


def test_a_metric_name_that_does_not_exist_does_not_pass():
    """The one that matters most: this is a TYPO, and it used to approve
    everything. A gate that green-lights on a misspelling is worse than no gate,
    because the pipeline reports success."""
    with pytest.raises(FairnessAssertionError) as e:
        assert_fairness(_Y_UNDEFINED, _P_UNDEFINED, _G, metric="not_a_real_metric", threshold=0.1)
    assert "NOT MEASURABLE" in str(e.value)


def test_a_measured_metric_inside_its_threshold_still_passes():
    """OVER-CORRECTION CONTROL. The fix must not turn the gate into one that
    refuses everything, which would be just as useless and much more obvious."""
    assert_fairness(_Y_OK, _P_OK, _G, metric="demographic_parity_difference", threshold=0.5)


def test_a_measured_metric_over_its_threshold_still_fails_as_a_breach():
    """And a real breach must still read as a BREACH, not as could-not-check:
    the two carry different messages on purpose."""
    y = np.array([0, 1] * 20)
    p = np.concatenate([np.ones(20, dtype=int), np.zeros(20, dtype=int)])
    with pytest.raises(FairnessAssertionError) as e:
        assert_fairness(y, p, _G, metric="demographic_parity_difference", threshold=0.01)
    assert "NOT MEASURABLE" not in str(e.value)
