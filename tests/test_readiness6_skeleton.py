"""READINESS-6: a causal graph that omits what it could not measure.

`causal_skeleton._assoc` returns an association strength in [0, 1], and that
number alone decides whether a proxy edge is drawn::

    if _assoc(df[p], df[f]) >= _PROXY_MIN:   # draw the edge

It returned ``0.0`` in five different refusals, and 0.0 here is a FINDING: it
says these two columns were compared and are unrelated. So a candidate proxy
that could not be assessed at all was silently absent from the causal graph,
indistinguishable from one that was measured and cleared.

Measured on this repo before the fix::

    only 8 overlapping rows        -> 0.0
    a constant numeric column      -> 0.0
    a constant categorical column  -> 0.0
    all values missing on one side -> 0.0
    two INDEPENDENT normals        -> 0.0649

The unmeasurable cases scored BETTER than the honestly-measured one.

The overview sentence is where it reached a reader. With no proxies found it
said "No strong proxy mediators detected; any disparity is a more direct effect
of the protected attribute(s)". The clause after the semicolon asserts where a
disparity comes from, and over columns that were never assessed it is built out
of nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse.causal_skeleton import _MIN_OVERLAP, _assoc, build_causal_skeleton

UNMEASURABLE = [
    (
        "fewer overlapping rows than the floor",
        pd.Series(list("aabbaabb")),
        pd.Series([1, 1, 0, 0] * 2),
    ),
    ("a constant numeric column", pd.Series([1.0] * 30), pd.Series(np.arange(30, dtype=float))),
    ("a constant categorical column", pd.Series(["x"] * 30), pd.Series(list("ab") * 15)),
    ("one side entirely missing", pd.Series([np.nan] * 30), pd.Series(np.arange(30, dtype=float))),
]


@pytest.mark.parametrize("label,a,b", UNMEASURABLE)
def test_an_unmeasurable_pair_is_not_reported_as_unrelated(label, a, b):
    assert _assoc(a, b) is None, (
        f"{label} returned {_assoc(a, b)!r}; 0.0 means 'compared and unrelated', "
        f"which is a finding this data cannot support"
    )


def test_a_genuinely_weak_association_is_still_measured():
    """OVER-CORRECTION CONTROL. Two independent columns ARE measurable and their
    association is a real, small number. Refusing here would be worse than the
    defect: it would delete the only honest reading in the file."""
    rng = np.random.default_rng(0)
    value = _assoc(pd.Series(rng.normal(size=200)), pd.Series(rng.normal(size=200)))
    assert value is not None
    assert 0.0 <= value < 0.3


def test_a_strong_association_is_still_measured():
    """OVER-CORRECTION CONTROL."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    value = _assoc(pd.Series(base), pd.Series(base * 2 + rng.normal(0, 0.1, 200)))
    assert value is not None and value > 0.9


def test_the_overlap_floor_is_the_named_constant():
    """A magic 10 inline is a refusal nobody can grep for."""
    n = _MIN_OVERLAP
    short = _assoc(pd.Series(list("ab") * ((n - 1) // 2)), pd.Series([1, 0] * ((n - 1) // 2)))
    assert short is None
    long_enough = _assoc(pd.Series(list("ab") * n), pd.Series([1, 0] * n))
    assert long_enough is not None


# ------------------------------------------------------------ the graph itself


def _all_unmeasurable():
    return pd.DataFrame(
        {
            "gender": ["m", "f"] * 25,
            "hired": [1, 0] * 25,
            "const": ["x"] * 50,
            "empty": [np.nan] * 50,
        }
    )


def test_features_that_could_not_be_assessed_are_recorded_not_dropped():
    sk = build_causal_skeleton(
        _all_unmeasurable(), ["gender"], "hired", candidate_features=["const", "empty"]
    )
    assert sorted(sk.unassessed) == ["const", "empty"]
    assert "unassessed" in sk.to_dict(), "the third state must survive serialisation"
    assert sorted(sk.to_dict()["unassessed"]) == ["const", "empty"]


def test_the_overview_refuses_to_say_where_a_disparity_comes_from():
    """The sentence a reader actually gets."""
    sk = build_causal_skeleton(
        _all_unmeasurable(), ["gender"], "hired", candidate_features=["const", "empty"]
    )
    assert "could not be measured" in sk.overview
    assert "NOT a finding that no proxy exists" in sk.overview
    assert "more direct effect" not in sk.overview, (
        "the overview still asserts where the disparity comes from, over "
        "columns that were never assessed"
    )


def _one_real_proxy(n=400):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "gender": np.where(g == 1, "m", "f"),
            "postcode": g * 5 + rng.normal(0, 1, n),
            "hired": (g * 2 + rng.normal(0, 1, n) > 1).astype(int),
            "const": ["x"] * n,
        }
    )


def test_a_real_proxy_is_still_found():
    """OVER-CORRECTION CONTROL, the one that matters most: the graph must still
    do its job."""
    sk = build_causal_skeleton(
        _one_real_proxy(), ["gender"], "hired", candidate_features=["postcode", "const"]
    )
    assert [n["id"] for n in sk.nodes if n["kind"] == "proxy"] == ["postcode"]
    assert any(e["source"] == "gender" and e["target"] == "postcode" for e in sk.edges)
    assert "proxy path(s) found" in sk.overview


def test_the_shortfall_is_disclosed_even_when_a_proxy_was_found():
    """A found proxy does not excuse silence about what was not looked at."""
    sk = build_causal_skeleton(
        _one_real_proxy(), ["gender"], "hired", candidate_features=["postcode", "const"]
    )
    assert sk.unassessed == ["const"]
    assert "could not be assessed" in sk.overview


def test_a_fully_assessed_run_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL. No disclosure where there is nothing to
    disclose, or every report grows a paragraph of noise."""
    sk = build_causal_skeleton(
        _one_real_proxy(), ["gender"], "hired", candidate_features=["postcode"]
    )
    assert sk.unassessed == []
    assert "could not be assessed" not in sk.overview


def test_a_clean_run_with_no_proxies_reads_exactly_as_before():
    """OVER-CORRECTION CONTROL for the OTHER overview branch.

    Added because a sabotage that forced the no-proxy branch to disclose
    unconditionally came back GREEN: every existing control exercised the
    proxy-FOUND branch, so the commonest real outcome, a clean run where the
    candidate features were assessed and none is a proxy, was unpinned.
    """
    rng = np.random.default_rng(3)
    n = 400
    g = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "gender": np.where(g == 1, "m", "f"),
            # Measurable, and unrelated to gender and to the outcome.
            "noise": rng.normal(size=n),
            "hired": rng.integers(0, 2, n),
        }
    )
    sk = build_causal_skeleton(df, ["gender"], "hired", candidate_features=["noise"])

    assert sk.unassessed == [], "the fixture must be fully assessable to control anything"
    assert [n_["id"] for n_ in sk.nodes if n_["kind"] == "proxy"] == []
    assert "No strong proxy mediators detected" in sk.overview
    assert "could not be assessed" not in sk.overview, (
        "a fully assessed clean run grew a could-not-check disclosure; that noise "
        "is how a real disclosure stops being read"
    )


# ------------------------------------------------------------ the tradeoffs list


def _recommendations(impossibility_applies):
    import inspect

    from vfairness.post_processing.calibration.tradeoffs import (
        ParetoPoint,
        _generate_tradeoff_recommendations,
    )

    fields = list(inspect.signature(ParetoPoint).parameters)
    point = ParetoPoint(**{f: 0.05 for f in fields})
    return _generate_tradeoff_recommendations(
        base_rate_disparity=0.05,
        current_point=point,
        pareto_points=[point],
        fairness_metric="equalized_odds",
        impossibility_diagnosis={"impossibility_applies": impossibility_applies},
    )


def test_an_unknown_impossibility_verdict_is_said_out_loud():
    """`.get(key, False)` does NOT fire its default when the key is PRESENT
    holding None, and `impossibility_applies` is documented Optional[bool]
    exactly so it can be None. `if None:` is falsy, so a could-not-check
    produced the same SILENCE on this list as a measured "does not apply", and
    silence here reads as "we checked and there is nothing to document"."""
    joined = " ".join(_recommendations(None))
    assert "COULD NOT BE CHECKED" in joined, joined
    assert "NOT a finding that" in joined


def test_a_measured_impossibility_still_says_so():
    """OVER-CORRECTION CONTROL."""
    joined = " ".join(_recommendations(True))
    assert "IMPOSSIBILITY THEOREM APPLIES" in joined
    assert "COULD NOT BE CHECKED" not in joined


def test_a_measured_non_application_stays_silent():
    """OVER-CORRECTION CONTROL. A measured False must NOT grow a
    could-not-check line, or the disclosure becomes noise everybody skips."""
    joined = " ".join(_recommendations(False))
    assert "IMPOSSIBILITY THEOREM" not in joined
