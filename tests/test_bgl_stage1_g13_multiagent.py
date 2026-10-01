"""Beta Go-Live stage 1, group g13: MultiAgentRunHarness group coverage.

The defect
----------
``MultiAgentRunHarness.as_compositionality_input`` and
``MultiAgentRunHarness.system_bias`` each took ``np.unique(groups)[0]``
and ``[1]`` and contrasted only those two labels. ``record_sample``
appends ``int(group)`` with no cardinality check, so a run with three
groups was accepted in silence and every sample outside the first two
labels entered NEITHER mask. On the reproduction below -- groups 0 and 1
at ~0.90 and group 2 at ~0.05, forty samples each -- the public entry
returned 0.0037 while the true worst pairwise gap was 0.8365, with zero
warnings. The forty most-harmed samples were the ones discarded, so the
failure was silent UNDER-reporting: the direction that hides harm.

The fix, and the three states this file pins at the public entry
-----------------------------------------------------------------
- MEASURED: every observed group is measured. Two groups give exactly
  the old value; more than two give the worst-case pairwise gap over all
  of them, plus a ``UserWarning`` naming the cardinality so a caller can
  tell the two apart without reading the source.
- COULD-NOT-CHECK: a group whose outputs are not finite makes the gap
  NaN. Before the fix such a group was simply dropped and a confident
  finite number was returned in its place.
- REFUSED: fewer than two distinct groups still raises ``ValueError``.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness import MultiAgentRunHarness


def _true_worst_gap(values, groups):
    """Worst pairwise gap computed from the RAW trace, independently of
    the code under test."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    means = [float(values[groups == u].mean()) for u in np.unique(groups)]
    return max(means) - min(means)


def _three_group_harness():
    """The reproduction from the finding: the harmed group is the third
    label, so it is exactly the one the old masks dropped."""
    rng = np.random.default_rng(3)
    h = MultiAgentRunHarness()
    for g, m in [(0, 0.90), (1, 0.90), (2, 0.05)]:
        for _ in range(40):
            v = float(np.clip(rng.normal(m, 0.05), 0, 1))
            h.record_sample(
                group=g,
                component_outputs={"agent_a": v, "agent_b": v},
                system_output=v,
            )
    return h


class TestThirdGroupIsMeasured:
    """MEASURED state: no observed group is silently discarded."""

    def test_system_bias_reports_the_harmed_third_group(self):
        h = _three_group_harness()
        truth = _true_worst_gap(h.trace.system_outputs, h.trace.sample_groups)
        assert truth > 0.8, truth  # the fixture really does carry the harm

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            measured = h.system_bias()

        # The value a caller sees is the real worst-case gap, not the
        # near-zero contrast of the two unharmed groups.
        assert measured == pytest.approx(truth, abs=1e-12), measured
        assert measured > 0.5, (
            f"system_bias() returned {measured!r}; the forty group-2 samples "
            f"were dropped again (true worst gap {truth!r})"
        )
        assert [str(w.message) for w in caught if w.category is UserWarning], (
            "a three-group run must disclose its cardinality; got no UserWarning"
        )
        assert "3 distinct group labels" in str(caught[0].message)

    def test_compositionality_input_reports_the_harmed_third_group(self):
        h = _three_group_harness()
        truth = _true_worst_gap(h.trace.component_outputs["agent_a"], h.trace.sample_groups)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            biases = h.as_compositionality_input()

        assert set(biases) == {"agent_a", "agent_b"}
        for name, val in biases.items():
            assert val == pytest.approx(truth, abs=1e-12), (name, val)
            assert val > 0.5, (name, val)
        assert any(w.category is UserWarning for w in caught)

    def test_not_an_artefact_of_the_label_values(self):
        """Relabelling 0/1/2 as 1/2/3 must not change the answer: the old
        code was indexing np.unique, so a different label set gave a
        different (and equally wrong) number."""
        rng = np.random.default_rng(3)
        h = MultiAgentRunHarness()
        for g, m in [(1, 0.90), (2, 0.90), (3, 0.05)]:
            for _ in range(40):
                v = float(np.clip(rng.normal(m, 0.05), 0, 1))
                h.record_sample(group=g, component_outputs={"a": v}, system_output=v)
        truth = _true_worst_gap(h.trace.system_outputs, h.trace.sample_groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            measured = h.system_bias()
        assert measured == pytest.approx(truth, abs=1e-12), measured
        assert measured > 0.5, measured


class TestCouldNotCheck:
    """COULD-NOT-CHECK state: a group nobody could measure is NaN, never a
    finite number computed from the groups that survived."""

    def test_non_finite_group_yields_nan_at_both_entries(self):
        h = MultiAgentRunHarness()
        for g, m in [(0, 0.9), (1, 0.5), (2, float("nan"))]:
            for _ in range(10):
                h.record_sample(group=g, component_outputs={"a": m}, system_output=m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sb = h.system_bias()
            comp = h.as_compositionality_input()

        # Pre-fix this returned abs(0.9 - 0.5) == 0.4: a confident value
        # standing in for a group that was never measured.
        assert math.isnan(sb), sb
        assert sb != pytest.approx(0.4), sb
        assert math.isnan(comp["a"]), comp


class TestControls:
    """CONTROL: healthy data still measures correctly and quietly."""

    def test_healthy_binary_run_is_unchanged_and_silent(self):
        h = MultiAgentRunHarness()
        for i in range(20):
            g = i % 2
            v = 0.8 if g == 0 else 0.3
            h.record_sample(
                group=g,
                component_outputs={"a": v, "b": v / 2},
                system_output=v,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sb = h.system_bias()
            comp = h.as_compositionality_input()

        assert sb == pytest.approx(0.5, abs=1e-12), sb
        assert comp["a"] == pytest.approx(0.5, abs=1e-12), comp
        assert comp["b"] == pytest.approx(0.25, abs=1e-12), comp
        assert [w for w in caught if w.category is UserWarning] == [], (
            "a plain binary run must not warn"
        )

    def test_binary_run_matches_the_explicit_two_group_contrast(self):
        """The generalisation must reduce exactly to abs(mean_0 - mean_1)
        on two groups, on ragged group sizes and non-round values."""
        rng = np.random.default_rng(11)
        h = MultiAgentRunHarness()
        for g, m, n in [(0, 0.62, 17), (1, 0.41, 23)]:
            for _ in range(n):
                v = float(rng.normal(m, 0.09))
                h.record_sample(group=g, component_outputs={"a": v}, system_output=v)
        arr = np.asarray(h.trace.system_outputs, dtype=float)
        grp = np.asarray(h.trace.sample_groups)
        expected = abs(arr[grp == 0].mean() - arr[grp == 1].mean())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert h.system_bias() == pytest.approx(expected, abs=1e-12)
        assert [w for w in caught if w.category is UserWarning] == []

    def test_fewer_than_two_groups_still_refuses(self):
        h = MultiAgentRunHarness()
        for _ in range(5):
            h.record_sample(group=0, component_outputs={"a": 1.0}, system_output=1.0)
        with pytest.raises(ValueError, match="Need at least 2 distinct groups"):
            h.system_bias()
        with pytest.raises(ValueError, match="Need at least 2 distinct groups"):
            h.as_compositionality_input()


def test_guard_sits_above_both_accessors():
    """Rule 3 of the stage-1 fix rules: the two accessors share one
    precondition, so neither may be fixed alone. Both must move together
    on the same three-group trace."""
    h = _three_group_harness()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sb = h.system_bias()
        comp = h.as_compositionality_input()
    assert sb > 0.5 and all(v > 0.5 for v in comp.values()), (sb, comp)
    # one disclosure per public call, not one per agent
    assert len([w for w in caught if w.category is UserWarning]) == 2
