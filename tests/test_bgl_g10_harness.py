"""G10 grading pins: vfairness.multi_agent.harness (HarnessTrace + the as_* accessors).

THE HARNESS COUNTS RUNS, so the question for every accessor is whether a run that
errored, returned nothing or was never captured can sit in a denominator as though
it had participated. Executed on both a healthy shape and the degenerate shapes.

Two defects pinned, both reproduced by execution:

D5 THE ALIGNMENT PRECONDITION WAS IN NEITHER CALLER. ``as_compositionality_input``
   and ``system_bias`` both mask a values array with a boolean built from
   ``sample_groups``, and neither checked the two were the same length. On a trace
   whose per-agent array was shorter, ``as_compositionality_input`` raised the raw
   numpy ``IndexError: boolean index did not match indexed array``, naming neither
   the agent nor the trace, while ALL THREE sibling accessors name the mismatch
   through their analyzer. That unnamed IndexError is the exact symptom the
   2026-09-27 note in ``record_sample`` says it wanted gone; the door it closed was
   the recording API, and ``HarnessTrace`` is a public dataclass reached directly.
D6 AN EMPTY PER-AGENT DICT READ AS A CLEAN SHEET. ``record_sample`` accepts
   ``component_outputs={}`` on purpose, so a run whose per-agent capture was never
   wired returned ``{}`` from ``as_compositionality_input`` in silence. A caller
   iterating that dict renders a page with no biased agent on it.

Every refusal pin below is paired with a control asserting the healthy run's REAL
number, because a guard that refuses everything passes every refusal test.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.multi_agent.collusion import AdversarialCollusionDetector
from vfairness.multi_agent.compositionality import CompositionalityAnalyzer
from vfairness.multi_agent.delegation import DelegationRoutingAuditor
from vfairness.multi_agent.emergent import EmergentBiasDetector
from vfairness.multi_agent.harness import HarnessTrace, MultiAgentRunHarness
from vfairness.multi_agent.negotiation import NegotiationFairnessTracker


def _biased_harness(n: int = 20, agent_gap: float = 0.8) -> MultiAgentRunHarness:
    """A run where group 0 is favoured by a known, hand-computable margin."""
    harness = MultiAgentRunHarness()
    for i in range(n):
        group = i % 2
        value = 0.9 if group == 0 else 0.9 - agent_gap
        harness.record_sample(
            group=group,
            component_outputs={"a": value},
            system_output=value,
            pre_interaction={"a": 0.5},
            post_interaction={"a": value},
        )
    return harness


def _hand_built(**fields) -> MultiAgentRunHarness:
    """A harness whose trace was assembled directly, which the public API allows:
    ``HarnessTrace`` is exported and ``harness.trace`` is a documented attribute."""
    harness = MultiAgentRunHarness()
    harness.trace = HarnessTrace(**fields)
    return harness


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ------------------------------------------------------- an empty harness refuses


def test_an_empty_harness_refuses_every_accessor_that_can():
    harness = MultiAgentRunHarness()
    for accessor, expected in (
        ("as_compositionality_input", "at least 2 distinct groups"),
        ("as_collusion_inputs", "pre_interaction and post_interaction"),
        ("as_delegation_inputs", "No routing decisions recorded"),
        ("as_negotiation_inputs", "at least 3 recorded turns"),
    ):
        with pytest.raises(ValueError) as caught:
            getattr(harness, accessor)()
        assert expected in str(caught.value), accessor


def test_an_empty_harness_hands_emergent_nothing_and_the_detector_refuses_it():
    """``as_emergent_inputs`` is a pass-through by design, so the refusal has to be
    reachable at the consumer. It is: nothing here reads as a clean zero."""
    components, system, groups = MultiAgentRunHarness().as_emergent_inputs()
    assert components == {}
    assert system.size == 0 and groups.size == 0
    with pytest.raises(ValueError, match="at least 2 unique values"):
        EmergentBiasDetector().analyze(components, system, groups)


# --------------------------------------------------------- D5: alignment refusal


def test_a_misaligned_agent_array_is_refused_by_name_not_by_a_numpy_indexerror():
    """D5. Before: ``IndexError: boolean index did not match indexed array along
    axis 0; size of axis is 3 but size of corresponding boolean axis is 10``."""
    harness = _hand_built(
        sample_groups=[0, 1] * 5,
        component_outputs={"a": [0.9, 0.1, 0.9]},
        system_outputs=[0.9, 0.1] * 5,
        n_samples=10,
    )
    with pytest.raises(ValueError) as caught:
        _quiet(harness.as_compositionality_input)
    message = str(caught.value)
    assert "as_compositionality_input[a]" in message, "the agent must be named"
    assert "3 value(s) against 10 group label(s)" in message
    assert not isinstance(caught.value, IndexError)


def test_the_alignment_guard_sits_above_the_dispatch_so_system_bias_refuses_too():
    """PUT THE GUARD ABOVE THE DISPATCH: both accessors share this precondition,
    and a guard in one would have left the other leaking the numpy error."""
    harness = _hand_built(
        sample_groups=[0, 1] * 5,
        component_outputs={"a": [0.9, 0.1] * 5},
        system_outputs=[0.9, 0.1, 0.9],
        n_samples=10,
    )
    with pytest.raises(ValueError, match=r"system_bias: recorded 3 value\(s\) against 10"):
        _quiet(harness.system_bias)


@pytest.mark.parametrize("n_values", [0, 1, 3, 9, 11, 20])
def test_any_misalignment_in_either_direction_is_refused(n_values):
    harness = _hand_built(
        sample_groups=[0, 1] * 5,
        component_outputs={"a": [0.5] * n_values},
        system_outputs=[0.5] * 10,
        n_samples=10,
    )
    with pytest.raises(ValueError, match="cannot be aligned"):
        _quiet(harness.as_compositionality_input)


def test_control_an_aligned_trace_still_measures_its_real_gap():
    """CONTROL for D5. The gap is DERIVED from the trace, not quoted."""
    harness = _biased_harness(n=20, agent_gap=0.8)
    groups = np.asarray(harness.trace.sample_groups)
    values = np.asarray(harness.trace.component_outputs["a"], dtype=float)
    expected = abs(values[groups == 0].mean() - values[groups == 1].mean())
    assert harness.as_compositionality_input() == {"a": pytest.approx(expected)}
    assert harness.system_bias() == pytest.approx(expected)
    assert expected == pytest.approx(0.8), "the fixture's own margin, as a sanity check"


# ------------------------------------------------ D6: an empty per-agent dict


def test_recorded_samples_with_no_agents_say_the_dict_is_a_could_not_check():
    """D6. ``{}`` here means "no agent was measured", never "no agent is biased"."""
    harness = MultiAgentRunHarness()
    for i in range(10):
        harness.record_sample(group=i % 2, component_outputs={}, system_output=float(i % 2))
    assert harness.trace.n_samples == 10

    with pytest.warns(UserWarning) as caught:
        biases = harness.as_compositionality_input()
    assert biases == {}, "the return shape must not change; only the disclosure is new"
    message = str(caught[0].message)
    assert "NO per-agent outputs" in message
    assert "could-not-check" in message
    assert "NOT a finding that no agent is biased" in message

    # And the verdict layer still refuses it, so the empty dict never became a grade.
    with pytest.raises(ValueError, match="At least one component bias score is required"):
        CompositionalityAnalyzer().analyze(biases, _quiet(harness.system_bias))


def test_control_a_run_with_agents_does_not_warn_about_missing_ones():
    """CONTROL for D6: the new warning must fire only on the empty case."""
    harness = _biased_harness()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert harness.as_compositionality_input() == {"a": pytest.approx(0.8)}


# --------------------------------------------- unmeasurable runs stay unmeasurable


def test_an_agent_that_returned_nothing_for_one_group_is_nan_and_says_so():
    """A group with no finite output must not be dropped and replaced by a
    confident finite gap computed from the groups that survived."""
    harness = MultiAgentRunHarness()
    for i in range(20):
        group = i % 2
        harness.record_sample(
            group=group,
            component_outputs={"a": 0.6 if group == 0 else 0.5, "b": 0.5 if group == 0 else np.nan},
            system_output=0.6 if group == 0 else 0.5,
        )
    with pytest.warns(UserWarning, match="as_compositionality_input\\[b\\]"):
        biases = harness.as_compositionality_input()
    assert biases["a"] == pytest.approx(0.1), "the measurable agent keeps its real number"
    assert math.isnan(biases["b"]), "the unmeasurable agent must not get a finite gap"

    result = _quiet(CompositionalityAnalyzer().analyze, biases, _quiet(harness.system_bias))
    assert result.scenario == "not_assessed", "a NaN component must not grade 'consistent'"
    assert result.metadata.parameters["n_components_unmeasurable"] == 1


def test_a_negotiation_of_three_unmeasurable_turns_does_not_become_a_verdict():
    """The floor counts TURNS; the trend needs finite disparities. Three recorded
    turns that all returned nothing clear the floor, so the refusal has to survive
    into the verdict, and it does."""
    harness = MultiAgentRunHarness()
    for _ in range(3):
        harness.record_turn(float("nan"), float("nan"))
    group_a, group_b = harness.as_negotiation_inputs()
    assert len(group_a) == len(group_b) == 3
    with pytest.warns(UserWarning, match="finite disparity"):
        result = NegotiationFairnessTracker().analyze(group_a, group_b)
    assert result.trend == "not_assessed"
    assert result.is_significant is None
    assert not math.isfinite(result.mean_disparity)


def test_control_a_real_negotiation_trend_is_still_measured():
    harness = MultiAgentRunHarness()
    for i in range(8):
        harness.record_turn(0.9 - 0.1 * i, 0.2)
    group_a, group_b = harness.as_negotiation_inputs()
    result = _quiet(NegotiationFairnessTracker().analyze, group_a, group_b)
    assert result.mean_disparity == pytest.approx(
        float(np.mean(np.abs(np.asarray(group_a) - np.asarray(group_b))))
    )
    assert result.trend != "not_assessed"


def test_a_collusion_run_whose_agents_returned_nothing_reports_could_not_check():
    harness = MultiAgentRunHarness()
    for i in range(10):
        harness.record_sample(
            group=i % 2,
            component_outputs={"a": np.nan},
            system_output=np.nan,
            pre_interaction={"a": np.nan},
            post_interaction={"a": np.nan},
        )
    pre, post, groups = harness.as_collusion_inputs()
    assert set(pre) == set(post) == {"a"}
    assert len(pre["a"]) == len(post["a"]) == len(groups) == 10
    result = _quiet(AdversarialCollusionDetector().analyze, pre, post, groups)
    assert result.is_collusion is None, "an unmeasurable run must not read as 'no collusion'"
    assert result.is_significant is None
    assert not math.isfinite(result.collusion_score)


def test_control_a_real_collusion_run_is_still_detected():
    """CONTROL: the accessor must still carry a genuine post-interaction gap."""
    harness = MultiAgentRunHarness()
    rng = np.random.default_rng(0)
    for i in range(30):
        group = i % 2
        harness.record_sample(
            group=group,
            component_outputs={"a": 0.5},
            system_output=0.5,
            pre_interaction={"a": 0.5 + rng.normal(0, 0.01)},
            post_interaction={"a": (0.9 if group == 0 else 0.1) + rng.normal(0, 0.01)},
        )
    result = _quiet(AdversarialCollusionDetector().analyze, *harness.as_collusion_inputs())
    assert result.is_collusion is True
    assert result.collusion_score == pytest.approx(0.8, abs=0.05)


# --------------------------------------------------- the accessors' preconditions


def test_collusion_refuses_a_run_where_only_one_side_of_the_interaction_exists():
    harness = MultiAgentRunHarness()
    for i in range(4):
        harness.record_sample(
            group=i % 2, component_outputs={"a": 0.5}, system_output=0.5, pre_interaction={"a": 0.4}
        )
    assert harness.trace.pre_interaction_outputs and not harness.trace.post_interaction_outputs
    with pytest.raises(ValueError, match="pre_interaction and post_interaction"):
        harness.as_collusion_inputs()


def test_delegation_returns_both_lists_and_the_auditor_refuses_a_single_contrast():
    harness = MultiAgentRunHarness()
    harness.record_routing(route="agent_a", demographic="x")
    routes, demographics = harness.as_delegation_inputs()
    assert routes == ["agent_a"] and demographics == ["x"]
    with pytest.raises(ValueError, match="At least 2 distinct routes"):
        DelegationRoutingAuditor().analyze(routes, demographics)

    one_group = MultiAgentRunHarness()
    for i in range(20):
        one_group.record_routing(route="a" if i % 2 else "b", demographic="only")
    with pytest.raises(ValueError, match="At least 2 distinct demographic groups"):
        DelegationRoutingAuditor().analyze(*one_group.as_delegation_inputs())


def test_delegation_accessor_returns_copies_not_the_live_trace_lists():
    """A caller mutating what it got back must not rewrite the recorded trace."""
    harness = MultiAgentRunHarness()
    harness.record_routing(route="a", demographic="x")
    routes, demographics = harness.as_delegation_inputs()
    routes.append("invented")
    demographics.append("invented")
    assert harness.trace.routing_decisions == ["a"]
    assert harness.trace.routing_demographics == ["x"]


def test_negotiation_accessor_returns_copies_not_the_live_trace_lists():
    harness = MultiAgentRunHarness()
    for _ in range(3):
        harness.record_turn(0.5, 0.4)
    group_a, group_b = harness.as_negotiation_inputs()
    group_a.append(99.0)
    group_b.append(99.0)
    assert len(harness.trace.per_turn_group_a) == 3
    assert len(harness.trace.per_turn_group_b) == 3


@pytest.mark.parametrize("absent", [None, float("nan"), "", "  ", "None", "nan", pd.NA, pd.NaT])
def test_a_routing_label_that_was_never_recorded_cannot_enter_the_delegation_inputs(absent):
    """The sibling defect's pin, kept here because it is the denominator that
    matters: an unrecorded label became a GROUP and carried p = 1.08e-05."""
    harness = MultiAgentRunHarness()
    with pytest.raises(ValueError, match="not a recorded label"):
        harness.record_routing(route="a", demographic=absent)
    with pytest.raises(ValueError, match="not a recorded label"):
        harness.record_routing(route=absent, demographic="x")
    assert harness.trace.routing_decisions == []
    assert harness.trace.routing_demographics == []
    with pytest.raises(ValueError, match="No routing decisions recorded"):
        harness.as_delegation_inputs()


def test_np_ma_masked_is_refused_too_and_needs_its_own_identity_test():
    assert bool(pd.isna(np.ma.masked)) is False
    harness = MultiAgentRunHarness()
    with pytest.raises(ValueError, match="not a recorded label"):
        harness.record_routing(route="a", demographic=np.ma.masked)


@pytest.mark.parametrize(
    "label", ["NA", "N/A", "null", "missing", "unknown", "None of the above", 0]
)
def test_control_a_real_label_that_merely_looks_absent_is_still_recorded(label):
    """DO NOT OVER-CORRECT. 'NA' is a region, 0 is a perfectly good binary group,
    and 'None of the above' is a real answer."""
    harness = MultiAgentRunHarness()
    harness.record_routing(route="a", demographic=label)
    routes, demographics = harness.as_delegation_inputs()
    assert routes == ["a"] and demographics == [str(label)]


def test_emergent_accessor_returns_the_labels_as_recorded():
    """Documented behaviour: a non-binary run comes back with every label, and the
    binary consumer refuses it rather than silently comparing the first two."""
    harness = MultiAgentRunHarness()
    for i in range(30):
        group = i % 3
        harness.record_sample(
            group=group,
            component_outputs={"a": 0.9 if group < 2 else 0.05},
            system_output=0.9 if group < 2 else 0.05,
        )
    _components, _system, groups = harness.as_emergent_inputs()
    assert sorted(np.unique(groups).tolist()) == [0, 1, 2], "no label may be dropped here"
    with pytest.warns(UserWarning, match="3 distinct group labels"):
        gap = harness.system_bias()
    values = np.asarray(harness.trace.system_outputs, dtype=float)
    means = [values[groups == u].mean() for u in np.unique(groups)]
    assert gap == pytest.approx(max(means) - min(means))


# ------------------------------------------------------------------ HarnessTrace


def test_trace_defaults_are_empty_and_per_instance():
    trace = HarnessTrace()
    assert trace.sample_groups == [] and trace.system_outputs == []
    assert trace.component_outputs == {} and trace.pre_interaction_outputs == {}
    assert trace.post_interaction_outputs == {}
    assert trace.routing_decisions == [] and trace.routing_demographics == []
    assert trace.per_turn_group_a == [] and trace.per_turn_group_b == []
    assert trace.n_samples == 0
    HarnessTrace().sample_groups.append(1)
    HarnessTrace().component_outputs["leak"] = [1.0]
    assert HarnessTrace().sample_groups == []
    assert HarnessTrace().component_outputs == {}


def test_trace_carries_a_real_run_metadata_and_serialises_strictly():
    """The trace is a record, not a measurement, but it must not mint one on the
    way out: an unmeasurable sample has to reach the envelope as null."""
    harness = MultiAgentRunHarness()
    for i in range(4):
        harness.record_sample(
            group=i % 2, component_outputs={"a": np.nan if i else 0.5}, system_output=0.5
        )
    trace = harness.trace
    assert trace.n_samples == 4 == len(trace.sample_groups) == len(trace.system_outputs)
    assert len(trace.component_outputs["a"]) == 4

    def _reject(name):
        raise AssertionError(f"non-JSON constant {name!r} in the trace envelope")

    parsed = json.loads(trace.to_json(), parse_constant=_reject)
    assert parsed["n_samples"] == 4
    assert parsed["component_outputs"]["a"] == [0.5, None, None, None]
    assert parsed["metadata"]["library_version"]
    # to_dict keeps the NaN for in-process consumers that use np.isfinite.
    assert not np.isfinite(trace.to_dict()["component_outputs"]["a"][1])
