"""BGL4 audit of batch agents_multi-2: two overturns, now CLOSED.

WRITTEN BY AN AUDITOR, NOT A FIXER. Every assertion here originally encoded the
DEFECTIVE behaviour observed on 2026-09-27, so the file was GREEN against the
defect and did not break the shared suite. The auditor's instruction was: when
either defect is fixed these tests MUST go red, and the right response is to
INVERT the assertion, never to delete the case.

INVERTED 2026-09-27 by the BGL-5 fix wave, exactly as instructed. Each test
keeps its name, its subject and its docstring; every assertion now states the
HONEST ANSWER that docstring names, and the observed pre-fix values are kept
beside it as the record of what changed. ``test_defect_`` is left in the test
names on purpose: it says which defect the case belongs to. The fix's own pins,
with their sabotage results and over-correction controls, are in
tests/test_bgl5_agents_multi.py.

Both grades under audit were recorded PROVEN, meaning "shown to refuse honestly
when its quantity cannot be measured, pinned by a named test, and the pin was
sabotaged to show it can fail". Neither claim survives a second input.

1. ``CompositionalityAnalyzer.analyze`` guards the DATA against NaN and not the
   ``threshold`` it compares the data against. The module's own comment says the
   defect it was fixed for was that "every ``>`` and ``<`` against NaN is False,
   so control fell through to the final ``else`` and reported scenario
   'consistent'". That mechanism is still live: it now arrives through the
   threshold instead of through the component scores, and produces BOTH a
   fabricated absence ('consistent' on a measured divergence of 0.9) and a
   fabricated finding ('amplification' on a measured divergence of exactly 0.0),
   in silence, with no ``not_assessed`` anywhere.

2. ``EmergentBiasDetector.analyze`` validates the group count on one side only
   (``len(unique_groups) < 2`` raises) and silently compares the first two
   sorted labels when more are supplied. On three groups whose entire bias lives
   in the third, it reports ``system_bias=0.0``, ``is_emergent=False``,
   ``p_value=1.0`` and ``n_samples_unmeasurable=0`` with no warning at all: a
   measured-looking census of a comparison that never looked at a third of the
   data.

A third finding is about the PIN rather than the code and cannot be expressed
here, because it needs the module loaded with one branch removed:
``test_emergent_refuses_when_no_sample_carries_a_system_measurement`` stays GREEN
when the ``if not math.isfinite(system_bias):`` branch it is evidence for is
deleted, because the ``n_measured == 0`` branch below it refuses with the same
three fields the test asserts. See the audit report for the harness output. It is
closed in tests/test_bgl5_agents_multi.py by
``test_emergent_system_bias_refusal_says_the_system_outputs_are_the_cause``,
which asserts the DIAGNOSIS only that branch produces and goes red when it is
deleted.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.multi_agent.compositionality import CompositionalityAnalyzer
from vfairness.multi_agent.emergent import EmergentBiasDetector

NAN = float("nan")
INF = float("inf")


def _capture(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, " || ".join(str(w.message) for w in caught)


class TestCompositionalityThresholdIsNotGuarded:
    """The unmeasurable-input guard covered the data and not the tolerance.

    Closed 2026-09-27: the tolerance is validated three lines below the
    aggregation_method check, so each case below now refuses instead of
    classifying, and each keeps the measured control that proves the data path
    is untouched."""

    def test_defect_a_non_finite_threshold_reports_consistent_on_a_0_9_divergence(self):
        """HONEST ANSWER: scenario='not_assessed', because a comparison against
        a tolerance that is not a number was not made. OBSERVED: 'consistent',
        which the class docstring glosses as "System and component biases are
        aligned", sitting beside its own divergence field of 0.9. The same input
        at the default threshold returns 'novel_emergence'.

        INVERTED: the BGL-5 fix refuses the call, the way the sibling modules
        reject a non-finite alpha in __init__ and the way this very method
        already rejects an unsupported aggregation_method, so no result object
        is produced to carry a scenario at all."""
        with pytest.raises(ValueError, match="finite tolerance greater than 0"):
            CompositionalityAnalyzer().analyze({"a": 0.0, "b": 0.0}, 0.9, threshold=NAN)
        # The same data, measured tolerance: the finding the NaN threshold hid,
        # and the proof the fix did not simply refuse everything.
        healthy, msg = _capture(
            lambda: CompositionalityAnalyzer().analyze({"a": 0.0, "b": 0.0}, 0.9)
        )
        assert healthy.scenario == "novel_emergence"
        assert healthy.divergence == 0.9
        assert msg == ""

    def test_defect_an_infinite_threshold_makes_every_other_scenario_unreachable(self):
        """HONEST ANSWER: 'not_assessed'. OBSERVED: 'consistent' for a system
        bias of 1e9 over components of exactly 0. No data of any magnitude can
        reach 'amplification', 'reduction' or 'novel_emergence' at this
        threshold, so the verdict is vacuous: it could not have disagreed.

        INVERTED: ValueError. The same system bias of 1e9 over components of
        exactly 0 is still classified at a measured tolerance, so the refusal is
        of the threshold and not of the data."""
        with pytest.raises(ValueError, match="finite tolerance greater than 0"):
            CompositionalityAnalyzer().analyze({"a": 0.0, "b": 0.0}, 1e9, threshold=INF)
        healthy, msg = _capture(
            lambda: CompositionalityAnalyzer().analyze({"a": 0.0, "b": 0.0}, 1e9, threshold=0.05)
        )
        assert healthy.scenario == "novel_emergence"
        assert healthy.divergence == 1e9
        assert msg == ""

    def test_defect_a_negative_threshold_manufactures_an_amplification_finding(self):
        """HONEST ANSWER: ValueError, the way the sibling modules reject an
        out-of-range alpha in __init__, or 'not_assessed'. OBSERVED:
        'amplification' on a divergence of EXACTLY 0.0, which is the definition
        of consistent, in silence. The fabrication runs in both directions.

        INVERTED to the ValueError the docstring names first. The identical
        inputs at the default tolerance still return 'consistent' with a
        measured divergence of 0.0, which is the answer the negative threshold
        overwrote."""
        with pytest.raises(ValueError, match="finite tolerance greater than 0"):
            CompositionalityAnalyzer().analyze({"a": 0.05, "b": 0.05}, 0.05, threshold=-5.0)
        healthy, msg = _capture(
            lambda: CompositionalityAnalyzer().analyze({"a": 0.05, "b": 0.05}, 0.05)
        )
        assert healthy.scenario == "consistent"
        assert healthy.divergence == 0.0
        assert msg == ""


class TestEmergentSilentlyDropsEveryGroupAfterTheSecond:
    """``len(unique_groups) < 2`` was checked; more than two was not.

    Closed 2026-09-27: more than two distinct labels is refused by name, so no
    sample can be dropped from both group masks without being counted."""

    def test_defect_three_groups_report_no_emergent_bias_from_the_two_flat_ones(self):
        """HONEST ANSWER: refuse, or measure every group and say which pair the
        verdict belongs to. OBSERVED: the first two sorted labels are compared
        and the third is dropped with no warning, no count and no field. Group 2
        carries a 1.0 system gap and a 0.4 component gap; the result says
        system_bias=0.0, is_emergent=False, p_value=1.0 and
        n_samples_unmeasurable=0.

        INVERTED: the detector refuses, which is the first of the two honest
        answers the docstring names. It is documented binary and has no field in
        which to say which pair a verdict belongs to, so it declines to publish
        one."""
        groups = np.array([0, 1, 2] * 60)
        component = np.where(groups == 2, 0.90, np.where(groups == 1, 0.51, 0.50))
        system = np.where(groups == 2, 1.0, 0.0)

        # The gap that exists in the data and used to be absent from the result.
        assert system[groups == 0].mean() == 0.0
        assert system[groups == 2].mean() == 1.0

        with pytest.raises(ValueError, match="exactly 2 unique values"):
            EmergentBiasDetector().analyze({"a": component}, system, groups)

        # OVER-CORRECTION CONTROL, in the same case: the pair that CAN be
        # compared is still measured, and it is the one carrying the real gap.
        pair = groups != 1
        result, msg = _capture(
            lambda: EmergentBiasDetector().analyze(
                {"a": component[pair]}, system[pair], groups[pair]
            )
        )
        assert result.system_bias == 1.0
        assert result.is_emergent is True
        assert result.max_component_bias == pytest.approx(0.40)
        assert result.amplification_factor == pytest.approx(2.5)
        assert result.n_samples_unmeasurable == 0
        assert result.metadata.parameters["n_samples"] == 120
        assert result.metadata.parameters["n_samples_measured"] == 120
        assert msg == ""

    def test_defect_group_labels_that_are_nan_are_dropped_without_a_word(self):
        """HONEST ANSWER: a sample whose GROUP is not measurable is an exclusion
        and belongs on the same disclosure the unit already keeps for a sample
        whose SYSTEM OUTPUT is not measurable. OBSERVED: np.unique sorts NaN
        last, the two NaN-labelled samples match neither mask, and
        n_samples_unmeasurable stays 0 while n_samples_measured claims all
        200.

        INVERTED: a NaN label is a third distinct value to np.unique, so the
        same guard refuses it and names it in the message. No sample can now be
        dropped from both masks while the count field claims it was measured."""
        groups = np.array([0.0, 1.0] * 99 + [NAN, NAN])
        binary = np.nan_to_num(groups)
        component = np.where(binary == 1, 0.55, 0.50)
        system = np.where(binary == 1, 1.0, 0.0)

        assert math.isnan(groups[-1])
        with pytest.raises(ValueError, match="exactly 2 unique values") as excinfo:
            EmergentBiasDetector().analyze({"a": component}, system, groups)
        assert "nan" in str(excinfo.value)

        # OVER-CORRECTION CONTROL: the 198 samples whose group IS measurable
        # still get their real emergence verdict.
        ok = np.isfinite(groups)
        result, msg = _capture(
            lambda: EmergentBiasDetector().analyze({"a": component[ok]}, system[ok], groups[ok])
        )
        assert result.is_emergent is True
        assert result.system_bias == 1.0
        assert result.max_component_bias == pytest.approx(0.05)
        assert result.n_samples_unmeasurable == 0
        assert result.metadata.parameters["n_samples_measured"] == 198
        assert msg == ""
