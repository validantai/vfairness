"""Readiness-3, repr2 lane: one result said "I could not assess this" AND named a group to act on.

THE DEFECT. With no population benchmark available, ``_analyze_single_attribute``
took its ``else`` branch, which scored every group holding under 10% of the rows
against ``1 / len(props)`` and appended the result to ``underrepresented_groups``.
It wrote nothing into ``representation_ratios``, so ``_determine_severity`` saw an
empty ratio map and graded the run INSUFFICIENT_DATA. Measured on 60 alpha /
35 beta / 5 gamma in a column named ``cohort`` (no built-in benchmark matches
that name):

    detect_representation_bias(df, ["cohort"], include_intersectional=False)
      severity          : insufficient_data
      benchmark_source  : None
      representation_ratios : {}
      underrepresented  : [{'group': 'gamma', 'ratio': 0.15000000000000002,
                            'actual_proportion': 0.05, 'count': 5}]
      recommendations   : ['Underrepresented groups requiring attention: gamma']
      warnings          : none at all

Two surfaces of one object disagreeing about one run. ``severity`` says the run
reached no verdict; ``underrepresented_groups`` and the recommendation hand the
reader a named group to act on. ``operations/pulse/orchestrator.py`` renders
``underrepresentedGroups`` from that list without consulting the severity beside
it, so the contradiction reaches a reader.

WHICH SIDE WAS WRONG, MEASURED. The ``ratio`` is not a measurement of its
subject. gamma holds exactly 5% of the rows in all three frames below and only
the split of the OTHER groups changes:

    60/35/5              gamma share 0.050  ratio 0.1500
    40/35/20/5           gamma share 0.050  ratio 0.2000
    35/30/20/10/5        gamma share 0.050  ratio 0.2500

The denominator is a uniform distribution over whatever labels the column
happens to contain, which is a benchmark invented from the data's own shape. So
the findings did NOT survive without a benchmark: the list must be empty and the
severity was right. The entries also carried neither ``expected_proportion`` nor
``deficit``, which the documented contract and every benchmarked entry do carry.

Nothing measured was dropped. Every group's share is still in
``group_distributions`` (control below asserts 0.05 in all three frames), and the
skew is now stated in a warning rather than left silent, because an empty result
with no warning is what a clean run looks like.

SIBLING, same shape, opposite answer. An INSUFFICIENT_DATA result could also
carry a populated ``intersectional_findings`` list with a 'high' severity entry.
Those ARE measurements that survive with no benchmark: they are scored against
the product of the data's own marginals. They stay, and the scope of ``severity``
is now stated on the field and in the recommendation the reader sees.
"""

import warnings

import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.representation import (
    RepresentationSeverity,
    detect_representation_bias,
)

# The exact frame from the reproduction above.
_SKEWED = ["alpha"] * 60 + ["beta"] * 35 + ["gamma"] * 5


def _run(labels, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (result,) = detect_representation_bias(
            pd.DataFrame({"cohort": labels}),
            ["cohort"],
            include_intersectional=False,
            **kwargs,
        )
    return result


# ── refusal pins: the object may not assert both at once ──────────────────


def test_an_unassessed_result_carries_no_group_to_act_on():
    """The reproduction, pinned. insufficient_data and a named group cannot coexist."""
    result = _run(_SKEWED)

    assert result.severity == RepresentationSeverity.INSUFFICIENT_DATA
    assert result.benchmark_source is None
    assert dict(result.representation_ratios) == {}

    # The contradiction itself.
    assert result.underrepresented_groups == [], (
        "a result whose severity says no verdict was reached handed the reader "
        f"{[g['group'] for g in result.underrepresented_groups]} to act on"
    )
    assert result.overrepresented_groups == []


def test_no_recommendation_names_a_group_when_nothing_was_compared():
    """The prose surface must not contradict the severity either."""
    result = _run(_SKEWED)

    joined = " ".join(result.recommendations)
    assert "requiring attention" not in joined
    assert "gamma" not in joined
    assert "NOT assessed" in joined, result.recommendations
    # An empty list is what an untroubled result looks like; say it instead.
    assert result.recommendations, "insufficient_data left the reader nothing at all"


def test_the_invented_ratio_moved_with_the_other_groups_and_is_gone():
    """gamma's share is identical in all three frames; the old ratio was not."""
    frames = {
        "three": _SKEWED,
        "four": ["alpha"] * 40 + ["beta"] * 35 + ["delta"] * 20 + ["gamma"] * 5,
        "five": ["alpha"] * 35 + ["beta"] * 30 + ["delta"] * 20 + ["eps"] * 10 + ["gamma"] * 5,
    }
    for name, labels in frames.items():
        result = _run(labels)
        assert result.underrepresented_groups == [], name
        # OVER-CORRECTION CONTROL: the measured share survives, unchanged and
        # identical across all three, which is what a real measurement does.
        assert result.group_distributions["gamma"] == 0.05, name
        assert result.sample_size == 100, name


def test_the_skew_is_said_out_loud_rather_than_dropped_in_silence():
    """Removing the finding must not turn a skewed column into a silent clean run."""
    with pytest.warns(UserWarning, match="UNBENCHMARKED") as caught:
        detect_representation_bias(
            pd.DataFrame({"cohort": _SKEWED}), ["cohort"], include_intersectional=False
        )
    message = " ".join(str(w.message) for w in caught)
    assert "gamma" in message
    assert "NOT a finding of adequate representation" in message


def test_severity_and_findings_agree_across_the_no_benchmark_grid():
    """The invariant, not just the one frame: no ratios means no findings."""
    grid = [
        _SKEWED,
        ["alpha"] * 95 + ["gamma"] * 5,
        ["alpha"] * 50 + ["beta"] * 45 + ["gamma"] * 3 + ["delta"] * 2,
        ["alpha"] * 34 + ["beta"] * 33 + ["gamma"] * 33,
    ]
    for labels in grid:
        result = _run(labels)
        if result.severity == RepresentationSeverity.INSUFFICIENT_DATA:
            assert not result.underrepresented_groups, labels[:3]
            assert not result.overrepresented_groups, labels[:3]
        if not result.representation_ratios:
            assert result.severity == RepresentationSeverity.INSUFFICIENT_DATA


# ── over-correction controls: measured values, so an empty fix fails ───────


def test_a_benchmarked_run_still_reports_the_group_with_its_numbers():
    """3000 male / 7000 female against 50/50. Exact values, so silencing fails here."""
    df = pd.DataFrame({"gender": ["male"] * 3000 + ["female"] * 7000})
    (result,) = detect_representation_bias(
        df,
        ["gender"],
        benchmarks={"gender": {"male": 0.5, "female": 0.5}},
        include_intersectional=False,
    )

    assert result.severity == RepresentationSeverity.HIGH
    assert result.benchmark_source == "provided"
    assert dict(result.representation_ratios) == {"female": 1.4, "male": 0.6}

    (under,) = result.underrepresented_groups
    assert under["group"] == "male"
    assert under["ratio"] == pytest.approx(0.6)
    assert under["actual_proportion"] == pytest.approx(0.3)
    assert under["expected_proportion"] == pytest.approx(0.5)
    assert under["count"] == 3000
    assert under["deficit"] == 2000

    (over,) = result.overrepresented_groups
    assert over["group"] == "female"
    assert over["ratio"] == pytest.approx(1.4)
    assert over["surplus"] == 1999

    assert result.chi_squared_statistic == pytest.approx(1600.0)
    assert result.chi_squared_pvalue == 0.0
    assert result.chi_squared_significant is True
    assert "Underrepresented groups requiring attention: male" in result.recommendations


def test_the_new_warning_does_not_fire_on_an_unskewed_column():
    """A blanket warning would be its own defect: 60/40 trips no 10%/30% rule."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (result,) = detect_representation_bias(
            pd.DataFrame({"cohort": ["alpha"] * 60 + ["beta"] * 40}),
            ["cohort"],
            include_intersectional=False,
        )
    assert [str(w.message) for w in caught] == []
    assert result.severity == RepresentationSeverity.INSUFFICIENT_DATA
    assert result.group_distributions == {"alpha": 0.6, "beta": 0.4}
    assert len(result.recommendations) == 1


def test_intersectional_findings_are_measurements_and_are_not_emptied():
    """The sibling that answered the other way: these need no benchmark, so they stay.

    Scored against the product of the data's own marginals, so the numbers below
    are asserted exactly. The scope of ``severity`` is stated in the
    recommendation instead, which is what removes the disagreement here.
    """
    df = pd.DataFrame({"a": ["x"] * 24 + ["y"] * 4, "b": ["p"] * 4 + ["q"] * 20 + ["p"] * 4})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = detect_representation_bias(
            df, ["a", "b"], min_group_size=3, include_intersectional=True
        )

    by_attr = {r.attribute: r for r in results}
    result = by_attr["a"]
    assert result.severity == RepresentationSeverity.INSUFFICIENT_DATA
    assert result.underrepresented_groups == []

    (finding,) = result.intersectional_findings
    assert finding["intersection"] == "x_p"
    assert finding["ratio"] == pytest.approx(0.5833333333333334)
    assert finding["actual_proportion"] == pytest.approx(0.14285714285714285)
    assert finding["expected_proportion"] == pytest.approx(0.24489795918367344)
    assert finding["count"] == 4
    assert finding["severity"] == "high"

    assert "NOT assessed" in " ".join(result.recommendations)


def test_too_few_rows_keeps_its_own_explanation_and_stays_empty():
    """The other insufficient_data path was already consistent; keep it that way."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (result,) = detect_representation_bias(
            pd.DataFrame({"cohort": ["alpha"] * 8 + ["gamma"] * 1}),
            ["cohort"],
            include_intersectional=False,
        )
    assert result.severity == RepresentationSeverity.INSUFFICIENT_DATA
    assert result.sample_size == 9
    assert result.underrepresented_groups == []
    assert result.overrepresented_groups == []
    (recommendation,) = result.recommendations
    assert recommendation.startswith("Insufficient data: only 9 rows")
