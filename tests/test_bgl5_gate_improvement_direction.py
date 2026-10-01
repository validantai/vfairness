"""The gate may not decide pass or fail from a direction it does not know.

WHY. ``improvement_amount`` returns None when a metric's better-direction is not
declared: whether a value went the right way is simply not established. Both gate
entry points SUBSTITUTED a magnitude comparison for that None,

    if improvement is None:
        improvement = abs(baseline_value) - abs(value)

and then used the substituted number to decide ``passed``. A release gate was
making a pass or fail decision out of a direction it did not know, and PUBLISHING
the substituted magnitude as the metric row's ``improvement``.

The substitution is not merely unmeasured, IT ASSUMES SMALLER IS BETTER, so for
any metric where larger is better the sign is backwards and a real improvement
reads as a degradation. Measured 2026-09-27 against the original file, with no
warning in either case:

    value 0.2, baseline 0.3  ->  improvement published as  0.09999999999999998
    value 0.3, baseline 0.2  ->  improvement published as -0.09999999999999998

and after, both publish None, with the unevaluated requirement named.

FOUR SITES, ONE HELPER. The substitution appeared twice in ``evaluate`` and twice
in ``evaluate_from_metrics``, and the degradation half's own comment records that
those two paths had already diverged once, when only ``evaluate`` enforced the
margin at all. Four copies of a rule is four chances to fix three of them, so this
file exercises BOTH entry points for both requirements.

SCOPE, stated rather than implied. For a metric with no declared direction the
THRESHOLD check above already fails the gate, so the substituted number was not
usually what flipped the verdict. It was still published in the metric row, the
markdown table and the GitHub check payload as though somebody had measured it,
and the requirement it was standing in for was reported as met. This file pins the
published value and the reported requirement, which is where the claim was, and it
asserts the guard is REACHED rather than assuming it.
"""

from __future__ import annotations

import pytest

from vfairness.operations.cicd.gate import GateConfig, ModelFairnessGate

UNDECLARED = "a_metric_nobody_declared"
DECLARED = "demographic_parity_difference"

#: Both doors into the same four copies of the rule. Named once so a test cannot
#: silently cover one of them.
ENTRIES = ["from_metrics", "evaluate"]


def _decide(entry: str, metric: str, value: float, baseline: float, **cfg):
    # Built through GateConfig because allow_degradation_margin is a config field
    # and is NOT a parameter of the convenience constructor, so the degradation
    # half of this defect is only reachable this way. A first version of this file
    # passed it to ModelFairnessGate() and got a TypeError, which is worth the
    # comment: the two requirements are configured through different doors and it
    # is the less-travelled door that carried the unfixed copy.
    config = GateConfig(metrics=[metric], thresholds={metric: 0.9}, **cfg)
    if entry == "from_metrics":
        gate = ModelFairnessGate(config=config)
        decision = gate.evaluate_from_metrics({metric: value}, baseline_metrics={metric: baseline})
    else:
        import numpy as np

        # compute_metrics_fn, not the built-in computation. evaluate() derives its
        # metrics from the arrays, and UNDECLARED is by definition not one it can
        # derive, so without this the row under test does not exist and the helper
        # raises StopIteration. That absence is exactly why the entry parametrisation
        # was a one-element list for two weeks while the docstring claimed both doors:
        # the evaluate half was unreachable, not merely unexercised.
        gate = ModelFairnessGate(
            config=config,
            compute_metrics_fn=lambda *_a, _m=metric, _v=value: {_m: _v},
        )
        rng = np.random.default_rng(0)
        n = 120
        decision = gate.evaluate(
            y_true=rng.integers(0, 2, n),
            y_pred=rng.integers(0, 2, n),
            protected_attr=np.array(["a"] * (n // 2) + ["b"] * (n // 2)),
            baseline_metrics={metric: baseline},
        )
    row = next(r for r in decision.metric_evaluations if r.metric_name == metric)
    return decision, row


def _all_text(decision, row) -> str:
    return " ".join([str(row.message or "")] + [str(w) for w in (decision.warnings or [])])


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("value,baseline", [(0.2, 0.3), (0.3, 0.2), (-0.5, 0.1)])
def test_an_unknown_direction_publishes_no_improvement_number(entry, value, baseline):
    decision, row = _decide(
        entry, UNDECLARED, value, baseline, require_improvement=True, improvement_margin=0.01
    )
    # NaN, not None: None is this dataclass's "no baseline comparison was asked
    # for", and collapsing an unevaluable requirement into it would lose the
    # distinction. That is the convention the baseline guard in the same function
    # already uses, and a first version of the fix got it wrong.
    assert row.improvement != row.improvement, (
        f"the gate published improvement={row.improvement!r} for a metric whose "
        "better direction is not declared. abs(baseline) - abs(value) is a magnitude "
        "guess that assumes smaller is better, and None would read as 'no comparison "
        "was required'."
    )
    assert not decision.approved, "an unevaluable requirement must not approve"


@pytest.mark.parametrize("value,baseline", [(0.2, 0.3), (0.3, 0.2)])
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_unevaluated_requirement_is_reported_not_dropped(entry, value, baseline):
    """The guard is REACHED, and says so where a reader looks.

    Asserted rather than assumed: for an undeclared metric the threshold check
    above already fails the gate, so this guard writes into warnings rather than
    into the blocking message, and a test that looked only at the message would
    pass while the requirement went unreported.
    """
    decision, row = _decide(
        entry,
        UNDECLARED,
        value,
        baseline,
        require_improvement=True,
        improvement_margin=0.01,
    )
    text = _all_text(decision, row)
    assert "COULD NOT BE EVALUATED" in text, text
    assert "required improvement" in text, text


@pytest.mark.parametrize("value,baseline", [(0.2, 0.3), (0.3, 0.2)])
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_degradation_half_is_closed_too(entry, value, baseline):
    decision, row = _decide(
        entry,
        UNDECLARED,
        value,
        baseline,
        require_improvement=False,
        allow_degradation_margin=0.01,
    )
    assert row.improvement != row.improvement, (
        f"improvement={row.improvement!r}; an unevaluable degradation margin must be "
        "NaN, not None, because None means no comparison was required. This block "
        "does not otherwise write the field, so None was what it published."
    )
    text = _all_text(decision, row)
    assert "COULD NOT BE EVALUATED" in text, text
    assert "allowed degradation" in text, text


@pytest.mark.parametrize(
    "value,baseline,expect_approved",
    [(0.2, 0.3, True), (0.3, 0.2, False)],
)
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_over_correction_control_a_declared_direction_still_measures(
    entry, value, baseline, expect_approved
):
    """A fix that refused every baseline comparison would pass everything above.

    The numbers are asserted, not just the shape: a declared metric moving from
    0.3 to 0.2 is a measured improvement of 0.1 and the gate approves, and the same
    move backwards is a measured failure to improve and it blocks.
    """
    decision, row = _decide(
        entry,
        DECLARED,
        value,
        baseline,
        require_improvement=True,
        improvement_margin=0.01,
    )
    assert row.improvement == pytest.approx(baseline - value, abs=1e-9), row.improvement
    assert decision.approved is expect_approved, (decision.status, row.message)
    assert "COULD NOT BE EVALUATED" not in _all_text(decision, row)


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_control_declared_direction_degradation_still_measures(entry):
    """A declared metric degrading past the margin is still blocked, by measurement.

    Asserted on the DECISION rather than on row.improvement. A first version of this
    control asserted improvement == -0.2 and failed: with require_improvement=False
    the improvement block never runs, so the field stays at its "nothing was asked
    for" value. That is pre-existing behaviour and correct; the degradation
    requirement's own verdict is what this test is about.
    """
    decision, row = _decide(
        entry,
        DECLARED,
        0.4,
        0.2,
        require_improvement=False,
        allow_degradation_margin=0.01,
    )
    assert not decision.approved
    assert "degraded beyond allowed margin" in str(row.message), row.message
    assert "COULD NOT BE EVALUATED" not in _all_text(decision, row)


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_control_a_configured_nothing_still_reads_as_nothing_asked(entry):
    """The state my fix nearly destroyed: no baseline requirement at all.

    With neither requirement configured the field must stay None, meaning nobody
    asked. If this returned NaN the three states would have collapsed the other
    way, and every unconfigured gate would look like a failed check.
    """
    decision, row = _decide(
        entry,
        DECLARED,
        0.4,
        0.2,
        require_improvement=False,
        allow_degradation_margin=None,
    )
    assert row.improvement is None, row.improvement
    assert "COULD NOT BE EVALUATED" not in _all_text(decision, row)
