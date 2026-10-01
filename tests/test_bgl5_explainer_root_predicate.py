"""An infinity is not a measurement, at the DIRECT explainer API as well.

WHY. `FairExplAIner._could_not_check_reason` is the one predicate that decides
whether a metric value gets graded or refused. It read

    unmeasurable = bool(np.isnan(value))

so NaN was the only unmeasurable class it knew and AN INFINITY WAS GRADED.
Measured at the direct API on 2026-09-27, before the change:

    explain_metric("demographic_parity_difference", inf)  -> "critical"
    explain_metric("demographic_parity_difference", -inf) -> "critical"
    explain_metric("demographic_parity_difference", nan)  -> "could_not_check"

and through the analyzer, on a RATIO metric, the same infinity read "Excellent!
The ratio of inf indicates near-perfect parity". The grade an infinity received
depended on which metric it arrived at. That is the tell: it was never a reading
of anything, it was whichever band the ladder happened to put it in.

A fix for the analyzer entry points landed first, above the one dispatch both of
those use, which closed the surface most callers reach and left this one open. So
this file is about the predicate itself, at the API a caller can import.

`is_measured` is the repo-wide predicate for exactly this question and was already
imported by the same module, used twenty lines below in `_finite_or_none`. Those
two answered DIFFERENTLY about the same input, which is how the hole survived: one
class disagreed with itself and nothing compared them.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

# One difference metric and one ratio metric, because the DEFECT presented
# differently on each and a single-metric test would have read as a one-off.
METRICS = ("demographic_parity_difference", "demographic_parity_ratio")
NOT_MEASUREMENTS = (
    np.inf,
    -np.inf,
    np.nan,
    np.float32("inf"),
    np.float64("nan"),
    True,
    np.True_,
    None,
)


@pytest.fixture
def explainer():
    return FairExplAIner()


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("value", NOT_MEASUREMENTS, ids=repr)
def test_a_value_that_is_not_a_measurement_is_refused_at_the_direct_api(explainer, metric, value):
    result = explainer.explain_metric(metric, value)
    assert result.severity == "could_not_check", (
        f"{metric}({value!r}) was graded {result.severity!r}. A value the repo-wide "
        "predicate rejects must not receive a fairness band at any entry point."
    )


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "value", [0.1, 0.9, 0.0, np.float32(0.1), np.float64(0.9), np.int64(0)], ids=repr
)
def test_the_over_correction_control_a_real_number_still_gets_a_band(explainer, metric, value):
    """A fix that refuses everything passes every refusal test above.

    A measured 0.0 is included deliberately: 0.0 is the value the whole campaign
    is about, and a perfect parity that a model genuinely achieved must still be
    reported as a measurement rather than refused for looking convenient.
    """
    result = explainer.explain_metric(metric, value)
    assert result.severity != "could_not_check", (
        f"{metric}({value!r}) was refused. A real number, including a genuine 0.0 "
        "and a numpy scalar, must still be graded."
    )
    assert result.severity in {"info", "low", "medium", "high", "critical"}, result.severity


@pytest.mark.parametrize("value", NOT_MEASUREMENTS + (0.1, 0.0, np.float32(0.1)), ids=repr)
def test_the_predicate_and_the_repo_wide_one_cannot_disagree(explainer, value):
    """The relation that would have prevented this defect from existing.

    The module already imported is_measured and used it in a sibling method. The
    two answering differently about one input is the whole bug, so it is the thing
    asserted rather than either answer on its own.
    """
    refused = (
        explainer._could_not_check_reason("demographic_parity_difference", value) == "unmeasurable"
    )
    assert refused == (not is_measured(value)), (
        f"for {value!r}: the explainer says unmeasurable={refused} and is_measured "
        f"says {is_measured(value)}. One class disagreeing with itself about one "
        "input is how an infinity came to be graded."
    )
