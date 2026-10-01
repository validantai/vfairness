"""Check 2 (honest on broken data): TextFairnessAnalyzer on a classifier that ties.

Found by scripts/broken_data_check.py, spec ``TextFairnessAnalyzer.analyze[severity]``
in scripts/broken_data_specs/llm.py, as OVER-REFUSES on the ``constant_scores`` and
``one_label_only`` worlds: 200 texts per identity term, every one scored 0.0.

Before: gap +0.000, p=1.000, severity 'not_assessed', and a warning that the
Mann-Whitney test "could NOT have reached 0.05 for any scores at all". The
detectability floor was read off the most extreme rearrangement of the OBSERVED
values, and when every value is tied that rearrangement is the data itself, so
its p of 1.0 was taken for the design's floor. For 200 against 200 the design
floor is about 1e-88.

After: a tied design's floor is the design's own (from the two sample sizes),
the reading is graded, and a small tied design (2 against 2, floor 0.19) is still
not assessed.
"""

from __future__ import annotations

import warnings

import pytest

from vfairness.llm.text_fairness import TextFairnessAnalyzer


def _analyze(scores_by_group):
    texts = {g: [f"{g} text {i}" for i in range(len(s))] for g, s in scores_by_group.items()}
    lookup = {t: v for g in texts for t, v in zip(texts[g], scores_by_group[g])}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TextFairnessAnalyzer(lambda ts: [lookup[t] for t in ts]).analyze(texts)
    return result, [str(w.message) for w in caught]


def test_healthy_control_is_measured_and_graded_exactly():
    result, _ = _analyze({"women": [0.9] * 30 + [0.8] * 30, "men": [0.1] * 30 + [0.2] * 30})
    assert result.severity == "critical"
    assert result.worst_group == "women"
    assert result.max_gap == pytest.approx(0.35)
    assert result.p_value is not None and result.p_value < 1e-10


def test_identical_scores_on_a_large_design_are_a_graded_no_bias_reading():
    result, caught = _analyze({"women": [0.0] * 200, "men": [0.0] * 200})
    assert result.max_gap == 0.0
    assert result.p_value == 1.0
    assert result.severity == "info", (result.severity, result.interpretation)
    assert result.interpretation.startswith("No material identity-term bias detected")
    assert not any("could NOT have reached" in m for m in caught), caught


def test_identical_scores_on_a_tiny_design_are_still_not_assessed():
    result, caught = _analyze({"women": [0.5, 0.5], "men": [0.5, 0.5]})
    assert result.severity == "not_assessed"
    assert any("could NOT have reached" in m for m in caught)
