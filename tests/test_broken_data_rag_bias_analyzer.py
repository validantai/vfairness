"""Check 2 pin: RAGBiasAnalyzer.full_analysis on a group that was never queried.

Found 2026-10-01 by scripts/broken_data_check.py (world ``single_group``): every
query belonged to group A, so group B's queries, retrieved documents and outputs
were all empty, and full_analysis reported retrieval_disparity 1.0,
output_disparity 1.0 and both bias flags True with no warning. That is the
strongest bias finding the scale has, about a group that never entered the
pipeline. The method receives the queries, so it can tell this case apart from
a group that was queried and served nothing; now it refuses it.
"""

from __future__ import annotations

import math
import warnings

import pytest

from vfairness.agents.rag_bias import RAGBiasAnalyzer


def _run(*args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = RAGBiasAnalyzer().full_analysis(*args)
    return result, [str(w.message) for w in caught]


def test_a_group_with_no_queries_is_not_measured():
    queries_a = [f"q{i}" for i in range(10)]
    docs_a = [{"id": f"d{i % 3}"} for i in range(10)]
    outputs_a = ["your application is approved"] * 10

    result, messages = _run(queries_a, [], docs_a, [], outputs_a, [])
    assert math.isnan(result.retrieval_disparity)
    assert math.isnan(result.output_disparity)
    assert result.is_retrieval_biased is None
    assert result.is_output_biased is None
    assert any("group_b submitted no queries" in m for m in messages)

    # Mirror image: group A never queried.
    result, _ = _run([], queries_a, [], docs_a, [], outputs_a)
    assert result.is_retrieval_biased is None
    assert math.isnan(result.retrieval_disparity)


def test_healthy_control_is_measured_exactly():
    """Both groups queried: the disparities are the Jaccard and trigram values."""
    docs_a = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
    docs_b = [{"id": "d2"}, {"id": "d4"}]
    out_a = ["approved"]
    out_b = ["declined"]
    result, _ = _run(["q1"], ["q2"], docs_a, docs_b, out_a, out_b)

    # Independently recomputed: 1 shared id of 4 distinct.
    assert result.retrieval_disparity == pytest.approx(1.0 - 1.0 / 4.0)
    tri = lambda s: {s[i : i + 3] for i in range(len(s) - 2)}  # noqa: E731
    ta, tb = tri("approved"), tri("declined")
    assert result.output_disparity == pytest.approx(1.0 - len(ta & tb) / len(ta | tb))
    assert result.is_retrieval_biased is True
    assert result.is_output_biased is True


def test_a_queried_group_served_nothing_is_still_a_finding():
    """The bare-stage reading is kept where the group WAS queried."""
    result, _ = _run(["q1"], ["q2"], [{"id": "d1"}], [], ["approved"], [""])
    assert result.retrieval_disparity == 1.0
    assert result.output_disparity == 1.0
    assert result.is_retrieval_biased is True


def test_callers_that_record_no_queries_keep_the_old_behaviour():
    """[] for both query lists means "not recorded", as existing callers pass it."""
    result, _ = _run([], [], [{"id": "d1"}], [{"id": "d1"}], ["same"], ["same"])
    assert result.retrieval_disparity == 0.0
    assert result.is_retrieval_biased is False
