"""The last five Tier-1 rows: every branch, not one input class.

All five were recorded DEFECT OPEN as "evidence insufficient", which is a
different statement from the rest of the register: no fabrication was ever seen in
them, and an independent audit overturned the PROVEN claim because the pins covered
less input than the function accepts. The remedy is evidence, so this file sweeps
every branch of each one rather than the one path a fixture happened to take.

Executing that sweep found three real gaps anyway, and each is fixed with its pin
below. None of them is a wrong number; all three are a SILENT answer that reads as
a clean bill:

  get_invalid_groups    returned []: "no group is too small": both for an empty
                        sensitive attribute, where no group was examined, and for
                        min_group_size <= 0, under which no group can EVER be too
                        small so the answer is vacuous. Same shape as the design-
                        power problem this package already guards for significance
                        tests: a check whose threshold puts it out of reach cannot
                        fire for any data.
  interpret_effect_size said "negligible effect (lower in group 1)" for d exactly
                        0.0. A direction is a claim about which group is worse off
                        and there is none at zero.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._statistics import (
    bootstrap_over_index,
    interpret_effect_size,
)
from vfairness.preprocessing.bias_detection.detector import (
    AUDIT_MODULES,
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    COVERAGE_UNASSESSED,
    COVERAGE_UNRECORDED,
    BiasAuditReport,
)


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


def _report(**over) -> BiasAuditReport:
    base = dict(
        timestamp="2026-09-25T00:00:00",
        dataset_info={"n_rows": 100},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return BiasAuditReport(**base)


# ── assessment_coverage: every branch ─────────────────────────────────────────


@pytest.mark.parametrize(
    "assessed,observations,expected",
    [
        ({"gender": True}, None, COVERAGE_COMPLETE),
        ({"gender": True, "age": True}, None, COVERAGE_COMPLETE),
        ({"gender": True, "age": False}, None, COVERAGE_PARTIAL),
        ({"gender": False}, None, COVERAGE_NONE),
        ({"gender": False, "age": False}, None, COVERAGE_NONE),
        ({}, None, COVERAGE_NONE),
        # The fallback, for reports predating `attribute_assessed`: a count is
        # weaker evidence than a verdict and is used only when nothing better exists.
        (None, {"gender": 40}, COVERAGE_COMPLETE),
        (None, {"gender": 0}, COVERAGE_NONE),
        (None, {"gender": 40, "age": 0}, COVERAGE_PARTIAL),
        (None, {}, COVERAGE_NONE),
        # Neither recorded: unrecorded, which withholds rather than granting.
        (None, None, COVERAGE_UNRECORDED),
    ],
)
def test_assessment_coverage_covers_every_branch(assessed, observations, expected):
    report = _report(attribute_assessed=assessed, attribute_observations=observations)
    assert report.assessment_coverage() == expected


def test_a_verdict_beats_a_row_count_when_the_two_disagree():
    """The BGL-S2 defect: a 2-row frame had observations and no verdict, and the
    count said "complete". The verdict must win wherever both exist."""
    report = _report(attribute_assessed={"gender": False}, attribute_observations={"gender": 2})
    assert report.assessment_coverage() == COVERAGE_NONE


# ── execution_coverage: every branch ──────────────────────────────────────────


@pytest.mark.parametrize(
    "modules,assessed,expected",
    [
        (None, {"gender": True}, COVERAGE_UNRECORDED),
        ([], {"gender": True}, COVERAGE_NONE),
        (["historical"], {"gender": True}, COVERAGE_PARTIAL),
        (list(AUDIT_MODULES[:3]), {"gender": True}, COVERAGE_PARTIAL),
        (list(AUDIT_MODULES), {"gender": True}, COVERAGE_COMPLETE),
        (list(AUDIT_MODULES), {"gender": False}, COVERAGE_UNASSESSED),
        (list(AUDIT_MODULES), {}, COVERAGE_UNASSESSED),
    ],
)
def test_execution_coverage_covers_every_branch(modules, assessed, expected):
    report = _report(modules_run=modules, attribute_assessed=assessed)
    assert report.execution_coverage() == expected


def test_an_unrecognised_module_name_does_not_buy_coverage():
    """`modules_run` is free-form. Three real modules plus two invented names must
    not read as complete: overstated coverage is the fabricated all-clear in
    miniature."""
    report = _report(
        modules_run=[*AUDIT_MODULES[:3], "bogus_a", "bogus_b"],
        attribute_assessed={"gender": True},
    )
    assert report.execution_coverage() == COVERAGE_PARTIAL


def test_the_two_coverage_answers_never_contradict_each_other():
    """ "Complete" execution requires something to have been assessed, so it can
    never sit beside an assessment coverage of none."""
    for modules in (None, [], ["historical"], list(AUDIT_MODULES)):
        for assessed in ({"gender": True}, {"gender": False}, {"gender": True, "age": False}):
            report = _report(modules_run=modules, attribute_assessed=assessed)
            if report.execution_coverage() == COVERAGE_COMPLETE:
                assert report.assessment_coverage() != COVERAGE_NONE


# ── get_invalid_groups: the silent empty list ─────────────────────────────────


def test_no_groups_at_all_is_disclosed_not_reported_as_all_groups_big_enough():
    groups = np.array([], dtype=object)
    value, msgs = _caught(lambda: GroupManager(groups, min_group_size=5).get_invalid_groups())
    assert value == []
    assert any("nothing was checked" in m for m in msgs), "an empty audit answered in silence"


@pytest.mark.parametrize("threshold", [0, -1, -30])
def test_a_threshold_that_disables_the_check_is_disclosed(threshold):
    groups = np.array(["a"] * 40 + ["b"] * 2)
    value, msgs = _caught(
        lambda: GroupManager(groups, min_group_size=threshold).get_invalid_groups()
    )
    assert value == []
    assert any("vacuous" in m for m in msgs), (
        f"min_group_size={threshold} cannot flag any group and said nothing"
    )


@pytest.mark.parametrize(
    "groups,threshold,expected",
    [
        (["a"] * 40 + ["b"] * 40, 5, []),
        (["a"] * 40 + ["b"] * 2, 5, ["b"]),
        (["a", "b", "c", "d"], 5, ["a", "b", "c", "d"]),
        (["a"] * 40, 5, []),
        (["a"] * 40 + ["b"] * 40, 10**6, ["a", "b"]),
    ],
)
def test_control_get_invalid_groups_still_names_the_groups_that_are_too_small(
    groups, threshold, expected
):
    value, msgs = _caught(
        lambda: GroupManager(np.array(groups), min_group_size=threshold).get_invalid_groups()
    )
    assert sorted(value) == sorted(expected)
    assert not msgs, f"a measurable case emitted {msgs}"


# ── interpret_effect_size: no direction at exactly zero ───────────────────────


@pytest.mark.parametrize("d", [0.0, -0.0])
def test_exactly_zero_claims_no_direction(d):
    text = interpret_effect_size(d, "cohens_d")
    assert "lower in group 1" not in text and "higher in group 1" not in text
    assert "equal" in text


@pytest.mark.parametrize(
    "d,expect",
    [(0.9, "higher in group 1"), (-0.9, "lower in group 1"), (0.3, "higher in group 1")],
)
def test_control_a_real_effect_keeps_its_direction(d, expect):
    assert expect in interpret_effect_size(d, "cohens_d")


def test_control_a_measured_zero_ratio_is_still_total_separation():
    """A ratio of 0.0 is a measurement, not an absence, and must not be swept into
    the new zero branch: it means group 1 has zero risk while group 2 does not."""
    assert "total separation" in interpret_effect_size(0.0, "risk_ratio")
    assert "total separation" in interpret_effect_size(0.0, "odds_ratio")


# ── bootstrap_over_index: every degenerate resampling class ───────────────────


@pytest.mark.parametrize("n", [0, 1])
def test_bootstrap_refuses_when_there_is_nothing_to_resample(n):
    result = bootstrap_over_index(n, lambda idx: 1.0, n_bootstrap=50, random_state=3)
    assert np.isnan(result.lower_bound) and np.isnan(result.upper_bound), (
        f"n={n} produced an interval from nothing to resample"
    )


def test_bootstrap_refuses_an_interval_when_the_statistic_is_never_measurable():
    result, msgs = _caught(
        lambda: bootstrap_over_index(50, lambda idx: float("nan"), n_bootstrap=50, random_state=3)
    )
    assert np.isnan(result.point_estimate)
    assert np.isnan(result.lower_bound) and np.isnan(result.upper_bound)
    assert msgs, "a statistic that never returned a number refused in silence"


def test_control_bootstrap_still_brackets_a_real_statistic():
    """The interval must bracket the SAMPLE statistic, which is what it resamples.

    The first version of this control asserted it bracketed the POPULATION mean of
    10.0 and failed: this seeded sample's own mean is 9.736, about 1.9 standard
    errors away, and a 95% interval around the sample is not required to reach the
    population value. That was a defect in the test, not in the bootstrap, and it
    is the kind that gets "fixed" by loosening the library. Asserted here against
    the estimand the function is actually given.
    """
    rng = np.random.default_rng(7)
    data = rng.normal(10.0, 2.0, 200)
    sample_mean = float(np.mean(data))
    result = bootstrap_over_index(
        200, lambda idx: float(np.mean(data[idx])), n_bootstrap=400, random_state=5
    )
    assert result.lower_bound < result.point_estimate < result.upper_bound
    assert result.lower_bound < sample_mean < result.upper_bound, (
        f"the interval {result.lower_bound:.3f}..{result.upper_bound:.3f} missed the "
        f"sample mean {sample_mean:.3f} it was built from"
    )
    # Tight enough to be a measurement rather than a shrug: ~4 standard errors wide.
    assert (result.upper_bound - result.lower_bound) < 8 * 2.0 / np.sqrt(200)
