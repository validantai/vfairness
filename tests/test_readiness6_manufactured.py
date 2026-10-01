"""READINESS-6: findings MANUFACTURED by a denominator that was small, not zero.

Every other detector in this audit hunts a fabricated PASS. This is the opposite
direction and the more expensive one. A fabricated pass costs a missed defect. A
fabricated FINDING costs a remediation programme aimed at floating-point noise,
and it arrives with a number that looks MORE certain the smaller the real
difference is, which is the property that makes it survive review.

Both sites here clamp a scale estimate to a tiny epsilon. That reads as a
divide-by-zero guard and is one, but what it guards is the CRASH: it converts an
undefined standardised statistic into an enormous finite one, and both functions
return a verdict.

Measured before the fix.

``sequential_fairness_test`` on two CONSTANT arms::

    gap 0.5     -> reject_h0_groups_differ
    gap 0.01    -> reject_h0_groups_differ
    gap 0.001   -> reject_h0_groups_differ
    gap 0.0001  -> reject_h0_groups_differ

while the SAME gaps with ordinary noise returned
``continue_insufficient_evidence`` for everything below 0.5. The test declared a
difference from a gap five thousand times smaller than the one it correctly
refuses to call on real data.

``FairnessDriftDetector.run_sprt`` on a CONSTANT stream of 40 values::

    offset 0.2    -> ('drift',  1 obs, log_lambda = +2.0e10)
    offset 0.01   -> ('stable', 1 obs, log_lambda = -1.8e10)
    offset 0.001  -> ('stable', 1 obs, log_lambda = -2.0e10)

Both directions manufactured, and ``'stable'`` is the worse: an all-clear on a
monitored metric, declared from one observation, with a likelihood ratio of
twenty billion behind it. It also invented ``sigma = 0.1`` for any stream too
short to estimate one from, which sets the scale every increment is measured in.

Constant inputs are not exotic. The pulse path scores LLM output at temperature
0, where repeated runs return identical text, and a metric that has not moved is
exactly when somebody relies on the monitor to say so.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import sequential_fairness_test
from vfairness.operations.monitoring.drift import FairnessDriftDetector

CONSTANT_GAPS = [0.5, 0.1, 0.01, 0.001, 0.0001]


def _detector():
    return FairnessDriftDetector.__new__(FairnessDriftDetector)


def _sprt(stream, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return FairnessDriftDetector.run_sprt(_detector(), stream, **kw)


# ------------------------------------------------------ sequential_fairness_test


@pytest.mark.parametrize("gap", CONSTANT_GAPS)
def test_constant_arms_do_not_manufacture_a_difference(gap):
    result = sequential_fairness_test(np.full(30, 0.5), np.full(30, 0.5 + gap))
    assert result["decision"] == "could_not_check", (
        f"a gap of {gap} between two arms with NO variance was decided as {result['decision']!r}"
    )
    assert result["available"] is False
    assert result["standardisedDifference"] is None


@pytest.mark.parametrize("gap", CONSTANT_GAPS)
def test_the_refusal_still_reports_the_real_difference(gap):
    """The gap IS real and certain; only its standardisation is undefined.
    Refusing without reporting it would discard the finding entirely, which is
    the discard-evidence direction."""
    result = sequential_fairness_test(np.full(30, 0.5), np.full(30, 0.5 + gap))
    assert result["meanDifference"] == pytest.approx(gap)
    assert "REAL AND CERTAIN" in result["interpretation"]


def test_identical_constant_arms_are_a_measured_no_difference():
    """OVER-CORRECTION CONTROL. Every observation identical in both arms is an
    OBSERVED absence of difference, not a could-not-check. Refusing here would
    throw away a real confirmation, and at temperature 0 it is the commonest
    genuinely fair shape."""
    result = sequential_fairness_test(np.full(30, 0.5), np.full(30, 0.5))
    assert result["available"] is True
    assert result["decision"] == "accept_h0_no_difference"
    assert result["standardisedDifference"] == 0.0


def test_ordinary_data_decides_exactly_as_before():
    """OVER-CORRECTION CONTROL: the test must still work."""
    rng = np.random.default_rng(0)
    big = sequential_fairness_test(rng.normal(0.5, 0.1, 30), rng.normal(1.0, 0.1, 30))
    assert big["decision"] == "reject_h0_groups_differ"
    assert big["available"] is True

    rng = np.random.default_rng(0)
    tiny = sequential_fairness_test(rng.normal(0.5, 0.1, 30), rng.normal(0.501, 0.1, 30))
    assert tiny["decision"] == "continue_insufficient_evidence"


def test_one_constant_arm_is_not_enough_to_refuse():
    """OVER-CORRECTION CONTROL. A pooled SD needs only ONE arm to vary. Refusing
    whenever either arm is constant would discard a large class of real
    comparisons."""
    rng = np.random.default_rng(0)
    result = sequential_fairness_test(np.full(30, 0.5), rng.normal(1.0, 0.1, 30))
    assert result["available"] is True
    assert result["decision"] == "reject_h0_groups_differ"


# ------------------------------------------------------------------ run_sprt


@pytest.mark.parametrize("offset", [0.2, 0.05, 0.01, 0.001])
def test_a_constant_stream_yields_neither_drift_nor_stable(offset):
    status, _, llr = _sprt([0.5 + offset] * 40, null_value=0.5, alternative_value=0.7)
    assert status == "could_not_check", (
        f"a constant stream offset by {offset} was decided as {status!r} from data with no variance"
    )
    assert math.isnan(llr), "a likelihood ratio was reported for a test that did not run"


def test_a_stream_too_short_for_a_sigma_does_not_invent_one():
    """`else 0.1` set the scale every increment is measured in, from a number
    nobody observed."""
    status, _, llr = _sprt([0.5, 0.9], null_value=0.5, alternative_value=0.7)
    assert status == "could_not_check"
    assert math.isnan(llr)


def test_the_refusals_say_so_out_loud():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FairnessDriftDetector.run_sprt(
            _detector(), [0.5] * 40, null_value=0.4, alternative_value=0.7
        )
    messages = " ".join(str(w.message) for w in caught)
    assert "could_not_check" in messages
    assert "NOT 'stable'" in messages, "the all-clear it is NOT giving must be named"


def test_a_real_stream_still_decides_and_with_a_sane_magnitude():
    """OVER-CORRECTION CONTROL, and the tell that the clamp is gone: the
    likelihood ratios were of order 1e10 and should now be of order 10."""
    rng = np.random.default_rng(0)
    drifted, _, llr_drift = _sprt(
        list(rng.normal(0.7, 0.05, 40)), null_value=0.5, alternative_value=0.7
    )
    assert drifted == "drift"
    assert abs(llr_drift) < 1e4, f"log_lambda {llr_drift} still carries the clamp"

    rng = np.random.default_rng(1)
    stable, _, llr_stable = _sprt(
        list(rng.normal(0.5, 0.05, 40)), null_value=0.5, alternative_value=0.7
    )
    assert stable == "stable"
    assert abs(llr_stable) < 1e4


def test_a_near_constant_stream_is_still_measured():
    """OVER-CORRECTION CONTROL. One ulp of variation is still variation: the
    refusal must not swallow data that merely has a small spread."""
    stream = [0.5] * 39 + [0.5 + 1e-9]
    status, _, _ = _sprt(stream, null_value=0.5, alternative_value=0.7)
    assert status != "could_not_check", "a stream with real variation was refused"


# ============================================================================
# The rest of the clamp triage.
#
# I told the other lane I had read each of the remaining 15 clamps. I had read
# four. Doing it properly turned up three more live defects and one duplicate
# implementation, which is the argument for finishing a triage rather than
# reporting one.
# ============================================================================


def _constant_experiment(n, gap):
    import pandas as pd

    from vfairness.operations.experimentation.experiment import (
        ExperimentConfig,
        FairnessExperiment,
    )

    rows = [{"arm": "control", "g": "A", "y": 0.5} for _ in range(n)]
    rows += [{"arm": "treatment", "g": "A", "y": 0.5 + gap} for _ in range(n)]
    df = pd.DataFrame(rows)
    return FairnessExperiment(
        df[df.arm == "control"],
        df[df.arm == "treatment"],
        ["g"],
        "y",
        config=ExperimentConfig(min_group_size=10),
    )


@pytest.mark.parametrize("gap", [0.5, 0.01, 0.001, 0.0001])
def test_the_power_analyzer_sprt_does_not_manufacture_a_treatment_effect(gap):
    """The THIRD site of the identical clamp, after _statistics and drift.

    Measured before the fix at 30 per arm: gap 0.0001 gave reject_null,
    stopped_early, with a log-likelihood ratio of 2,999,999.7, while the same
    gap with ordinary noise correctly gave `continue`.
    """
    from vfairness.operations.experimentation.power import FairnessPowerAnalyzer, PowerConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        analyzer = FairnessPowerAnalyzer(
            _constant_experiment(30, gap), PowerConfig(min_group_size=10)
        )
        result = analyzer.sequential_test(effect_size=0.2)[("A",)]

    assert result.decision.value == "could_not_check", (
        f"a gap of {gap} between two arms with NO variance was decided as "
        f"{result.decision.value!r} with llr {result.log_likelihood_ratio!r}"
    )
    assert math.isnan(result.log_likelihood_ratio)
    assert result.stopped_early is False


def test_the_power_analyzer_still_confirms_identical_arms():
    """OVER-CORRECTION CONTROL."""
    from vfairness.operations.experimentation.power import FairnessPowerAnalyzer, PowerConfig

    analyzer = FairnessPowerAnalyzer(_constant_experiment(30, 0.0), PowerConfig(min_group_size=10))
    result = analyzer.sequential_test(effect_size=0.2)[("A",)]
    assert result.decision.value == "accept_null"
    assert result.log_likelihood_ratio == 0.0


def test_the_power_analyzer_still_decides_on_real_data():
    """OVER-CORRECTION CONTROL."""
    import pandas as pd

    from vfairness.operations.experimentation.experiment import (
        ExperimentConfig,
        FairnessExperiment,
    )
    from vfairness.operations.experimentation.power import FairnessPowerAnalyzer, PowerConfig

    rng = np.random.default_rng(5)
    rows = [{"arm": "control", "g": "A", "y": float(v)} for v in rng.normal(0.5, 0.1, 30)]
    rows += [{"arm": "treatment", "g": "A", "y": float(v)} for v in rng.normal(1.0, 0.1, 30)]
    df = pd.DataFrame(rows)
    experiment = FairnessExperiment(
        df[df.arm == "control"],
        df[df.arm == "treatment"],
        ["g"],
        "y",
        config=ExperimentConfig(min_group_size=10),
    )
    analyzer = FairnessPowerAnalyzer(experiment, PowerConfig(min_group_size=10))
    assert analyzer.sequential_test(effect_size=0.2)[("A",)].decision.value == "reject_null"


# ------------------------------------------------- the duplicate faithfulness AUC


def test_the_two_removal_curve_auc_implementations_agree():
    """`removal_curve_auc` exists TWICE, both exported, and only one received
    the READINESS-6 fixes. Measured on identical inputs, a model with large
    offsetting weights so the base prediction is small and the drops are large::

        weights                canonical   the twin
        [10, -9.9, 0]             1.0000      33.83
        [100, -99.99, 0]          1.0000    3333.83
        [1000, -999.999, 0]       1.0000  333333.83

    The twin's own docstring promised "[0, 1]". It returned an unbounded ratio
    that GROWS as the base prediction shrinks: the number looks more emphatic
    the less prediction there was to explain.
    """
    from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
        removal_curve_auc as canonical,
    )
    from vfairness.xai.diagnostics.faithfulness import removal_curve_auc as twin

    x = np.ones(3)
    attributions = np.array([0.6, 0.3, 0.1])
    background = np.zeros(3)
    for weights in ([1.0, 0.5, 0.25], [10.0, -9.9, 0.0], [1000.0, -999.999, 0.0]):
        w = np.array(weights)

        def predict(X, w=w):
            return np.asarray(X, dtype=float) @ w

        a = canonical(predict, x, attributions, background, 3)
        b = twin(
            predict_fn=predict,
            x=x,
            attributions=attributions,
            background_mean=background,
            n_steps=3,
        )
        assert a == b, f"weights {weights}: canonical {a!r} vs twin {b!r}"
        assert 0.0 <= b <= 1.0, f"the twin returned {b!r}, outside the [0, 1] it documents"


def test_both_removal_curve_implementations_refuse_an_empty_background():
    from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
        removal_curve_auc as canonical,
    )
    from vfairness.xai.diagnostics.faithfulness import removal_curve_auc as twin

    x = np.ones(3)
    attributions = np.array([0.6, 0.3, 0.1])
    background = np.full(3, np.nan)

    def predict(X):
        return np.asarray(X, dtype=float) @ np.array([1.0, 0.5, 0.25])

    assert math.isnan(canonical(predict, x, attributions, background, 3))
    assert math.isnan(
        twin(
            predict_fn=predict,
            x=x,
            attributions=attributions,
            background_mean=background,
            n_steps=3,
        )
    )


# ---------------------------------------------------------------- vision skew/NDKL


def test_an_absent_group_has_no_skew_rather_than_a_large_negative_one():
    """Skew is ln(observed/desired), genuinely -infinity for a group that never
    appears. The clamp replaced that with ln(1e-9 / 0.25) = -19.3, a number
    decided entirely by the epsilon: at 1e-6 the same absence scores -12.4.
    maxSkew and minSkew are graded, so a threshold on them was being compared
    against the choice of epsilon."""
    from vfairness.vision import skew

    result = skew(
        ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
        reference={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
    )
    assert result["perGroup"]["d"] is None
    assert result["absentGroups"] == ["d"]
    assert result["mostUnderrepresented"] != "d", (
        "an absent group was ranked as the most underrepresented on an invented magnitude"
    )
    assert "did not appear at all" in result["note"]


def test_ordinary_imbalance_is_measured_exactly_as_before():
    """OVER-CORRECTION CONTROL: real under- and over-representation must still
    produce numbers."""
    from vfairness.vision import skew

    result = skew(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    assert result["available"] is True
    assert result["perGroup"]["a"] > 0 and result["perGroup"]["c"] < 0
    assert result["mostOverrepresented"] == "a"
    assert result["mostUnderrepresented"] == "c"
    assert result["absentGroups"] == []
    assert "note" not in result


def test_skew_refuses_when_no_group_is_measurable():
    from vfairness.vision import skew

    result = skew(["a"] * 10, reference={"b": 1.0})
    assert result["available"] is False
    assert "no skew is defined" in result["reason"]


def test_ndkl_refuses_a_target_that_excludes_a_group_the_ranking_contains():
    """The KL divergence is genuinely infinite there. The clamp made it a large
    finite number whose size is an artefact of 1e-9, and NDKL is graded."""
    from vfairness.vision import ndkl

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = ndkl(["a", "a", "b", "b"], reference={"a": 1.0, "b": 0.0})
    assert math.isnan(value)
    assert any("ZERO share" in str(w.message) for w in caught)


def test_ndkl_is_unchanged_on_ordinary_rankings():
    """OVER-CORRECTION CONTROL."""
    from vfairness.vision import ndkl

    skewed = ndkl(["a", "a", "a", "b"])
    balanced = ndkl(["a", "b", "a", "b"])
    assert 0.0 < balanced < skewed
    # A zero-share group that does NOT appear is not a problem.
    assert ndkl(["a", "a", "a"], reference={"a": 1.0, "z": 0.0}) == pytest.approx(0.0)


def test_total_exclusion_is_still_flagged_critical():
    """The FIX ABOVE ALMOST DELETED A FINDING, and this is the control that
    caught it.

    Removing the clamp from `skew` removed the only route by which a completely
    excluded group reached `representation_severity`: it arrived as a large
    negative minSkew, which is a magnitude decided by the epsilon. The existing
    `test_representation_severity_flags_total_exclusion` went red, correctly,
    and the answer was to flag the FACT rather than restore the number.

    Total exclusion is now critical categorically, which is stronger than the
    threshold it replaces: it no longer depends on ln(1e-9 / 0.25) happening to
    land below -0.5.
    """
    from vfairness.vision import representation_severity, skew

    reference = {g: 1.0 / 6 for g in "ABCDEF"}
    excluded = skew(["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20 + ["E"] * 20, reference)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert representation_severity(excluded) == "critical"
    assert any("do not appear in the set at all" in str(w.message) for w in caught)


def test_a_mildly_skewed_set_with_no_exclusion_is_not_critical():
    """OVER-CORRECTION CONTROL for the clause above: only ABSENCE triggers it."""
    from vfairness.vision import representation_severity, skew

    balanced = skew(["A"] * 34 + ["B"] * 33 + ["C"] * 33)
    assert balanced["absentGroups"] == []
    assert representation_severity(balanced) == "pass"


# ------------------------------------------------------------- Cohen's d, x3


COHENS_D_CASES = [
    # (label, value, n, gap) -- values and sizes chosen so that BOTH sides of
    # the exact-zero miss are covered: 0.5 at n=5 has exactly zero variance,
    # 0.9 at n=20 does not.
    ("0.5 at n=5", 0.5, 5, 0.7),
    ("0.5 at n=20", 0.5, 20, 0.7),
    ("0.9 at n=20", 0.9, 20, 0.7),
    ("0.1 at n=25", 0.1, 25, 0.7),
    ("1/3 at n=10", 1.0 / 3.0, 10, 0.7),
]


@pytest.mark.parametrize("label,value,n,gap", COHENS_D_CASES)
def test_cohens_d_refuses_two_constant_separated_arms(label, value, n, gap):
    """Cohen's d existed in THREE implementations answering differently.

    `agents.action_bias._cohens_d` was fixed on 2026-09-08 and pinned with
    group A at 10.0 and group B at 100.0, n=10. Both values are exactly
    representable in binary, so the variance is exactly 0.0, the
    `pooled_std == 0.0` guard fires, and the pin passed. Measured 2026-09-10
    with ordinary values the fix did not apply at all::

        constant arms, gap 0.7, n=5   -> nan      (guard fires)
        constant arms, gap 0.7, n=20  -> -4.3e15  (guard misses)

    -4.3e15 is FINITE, so every isfinite guard downstream passes it and an
    interpretation table labels it "large".
    """
    from vfairness.agents.action_bias import ActionBiasAnalyzer
    from vfairness.evaluation.vfairness_metrics._statistics import cohens_d

    a = np.full(n, value)
    b = np.full(n, value + gap)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shared = cohens_d(a, b)
        agents = ActionBiasAnalyzer._cohens_d(a, b)
    assert math.isnan(shared), f"{label}: _statistics.cohens_d returned {shared!r}"
    assert math.isnan(agents), f"{label}: agents._cohens_d returned {agents!r}"


@pytest.mark.parametrize("label,value,n,gap", COHENS_D_CASES)
def test_identical_constant_arms_are_a_measured_zero_effect(label, value, n, gap):
    """OVER-CORRECTION CONTROL. Zero variance AND equal means is a genuine zero
    effect, and at temperature 0 it is the commonest fair shape."""
    from vfairness.agents.action_bias import ActionBiasAnalyzer
    from vfairness.evaluation.vfairness_metrics._statistics import cohens_d

    a = np.full(n, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert cohens_d(a, a.copy()) == 0.0, label
        assert ActionBiasAnalyzer._cohens_d(a, a.copy()) == 0.0, label


def test_ordinary_cohens_d_is_untouched_in_both():
    """OVER-CORRECTION CONTROL."""
    from vfairness.agents.action_bias import ActionBiasAnalyzer
    from vfairness.evaluation.vfairness_metrics._statistics import cohens_d

    a = np.array([1.0, 2, 3, 4, 5])
    b = np.array([3.0, 4, 5, 6, 7])
    assert cohens_d(a, b) == pytest.approx(-1.2649, abs=1e-4)
    assert ActionBiasAnalyzer._cohens_d(a, b) == pytest.approx(-1.2649, abs=1e-4)


def test_one_constant_arm_still_yields_an_effect_size():
    """OVER-CORRECTION CONTROL: only BOTH arms constant makes d undefined."""
    from vfairness.evaluation.vfairness_metrics._statistics import cohens_d

    rng = np.random.default_rng(0)
    value = cohens_d(np.full(20, 0.5), rng.normal(1.0, 0.1, 20))
    assert math.isfinite(value) and value < 0
