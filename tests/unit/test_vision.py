"""Unit tests for vfairness.vision representation-fairness math.

Exercises the pure-numpy Skew / NDKL / bias-amplification layer (Geyik et al.
2019, Seshadri et al. 2023). The model-backed demographic classification needs
the vision sidecar and is intentionally out of scope here.
"""

from vfairness.vision import (
    bias_amplification,
    ndkl,
    representation_severity,
    skew,
)


def test_skew_uniform_is_near_zero():
    result = skew(["a", "b", "a", "b"])
    assert result["available"] is True
    assert abs(result["maxSkew"]) < 1e-3
    assert abs(result["minSkew"]) < 1e-3


def test_skew_flags_over_and_under_represented_groups():
    result = skew(["a", "a", "a", "b"])
    assert result["mostOverrepresented"] == "a"
    assert result["mostUnderrepresented"] == "b"
    assert result["maxSkew"] > 0 > result["minSkew"]


def test_skew_empty_reports_unavailable():
    assert skew([])["available"] is False


def test_ndkl_balanced_ranking_stays_small():
    value = ndkl(["a", "b", "a", "b"])
    assert value >= 0.0
    assert value < 0.5


def test_ndkl_single_group_ranking_diverges():
    value = ndkl(["a", "a", "a", "a"], reference={"a": 0.5, "b": 0.5})
    assert value > 0.0


def test_ndkl_empty_refuses_rather_than_scoring_perfect():
    """Was `assert ndkl([]) == 0.0` until the fifth-iteration audit.

    0.0 is the BEST attainable NDKL, so returning it for an empty input made
    "nothing was measured" outrank every real ranking: a genuinely skewed list
    scores 0.6275 and an empty one scored 0.0. The original intent of this test,
    that an empty input must not raise, is preserved below.
    """
    import math
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = ndkl([])
    assert math.isnan(value)
    assert caught
    # A real ranking still scores, and scores worse than any refusal can.
    assert ndkl([["a", "a", "a", "a", "b"]]) > 0.0


def test_bias_amplification_zero_when_matching_reference():
    out = bias_amplification(["a", "a", "b", "b"], reference={"a": 0.5, "b": 0.5})
    assert out["available"] is True
    assert out["maxAmplification"] == 0.0


def test_bias_amplification_detects_amplified_group():
    out = bias_amplification(["a", "a", "a", "a"], reference={"a": 0.5, "b": 0.5})
    assert out["worstGroup"] == "a"
    assert out["maxAmplification"] > 0.0


def test_representation_severity_thresholds_key_off_magnitude():
    assert representation_severity(0.1) == "pass"
    assert representation_severity(0.3) == "warn"
    assert representation_severity(0.6) == "critical"
    assert representation_severity(-0.6) == "critical"
