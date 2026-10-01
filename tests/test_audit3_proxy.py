"""Third-iteration audit regression tests for the proxy / correlation
detectors.

CRITICAL defect (both twin detectors, bias_detection.proxy and
feature_engineering.correlation): a CONTINUOUS feature that near-perfectly
encodes a CATEGORICAL protected attribute (income / credit-score proxying for
race, the textbook proxy case) was silently missed. The decision statistic
was Cramér's V, which requires BOTH sides categorical; a continuous feature
treated as categorical makes every distinct float its own level, so the
Bergsma (2013) bias correction drives V to exactly 0 and the proxy is dropped.
The fix routes the mixed continuous-vs-categorical case to the correlation
ratio (eta) with a one-way ANOVA p-value.

MEDIUM defect (feature_engineering.correlation.compute_feature_correlations,
method='auto'): the same continuous proxy reported as 'weak' because it used
point-biserial (valid only for a BINARY categorical side) on the arbitrary
category codes of a 3+ level nominal attribute.

Each defect case is pinned FIRST, then a does-not-overcorrect case that keeps a
genuinely-unrelated continuous feature UNflagged and preserves the correct
measure for the categorical-categorical and binary cases.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vfairness.preprocessing.bias_detection.proxy import (
    compute_proxy_correlations,
)
from vfairness.preprocessing.bias_detection.proxy import (
    identify_proxy_variables as bd_identify_proxy_variables,
)
from vfairness.preprocessing.feature_engineering.correlation import (
    compute_feature_correlations,
)
from vfairness.preprocessing.feature_engineering.correlation import (
    identify_proxy_variables as fe_identify_proxy_variables,
)


def _continuous_proxy_frame(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """race is a 3-category nominal attribute; credit_rating is a near-perfect
    CONTINUOUS proxy (group mean + small noise, eta^2 ~0.97); unrelated is a
    genuinely independent continuous feature (eta^2 ~0)."""
    rng = np.random.default_rng(seed)
    groups = rng.choice(["White", "Black", "Asian"], size=n, p=[0.4, 0.3, 0.3])
    means = {"White": 750.0, "Black": 580.0, "Asian": 760.0}
    credit = np.array([means[g] for g in groups]) + rng.normal(0, 15, size=n)
    unrelated = rng.normal(500, 100, size=n)
    return pd.DataFrame({"race": groups, "credit_rating": credit, "unrelated": unrelated})


def _by_feature(results):
    return {r.feature: r for r in results}


# ── CRITICAL: continuous proxy of a categorical attribute must be flagged ────


def test_bias_detection_flags_continuous_proxy_of_categorical_attr():
    """Defect: bias_detection.identify_proxy_variables returned 0 proxies for a
    continuous proxy of a categorical protected attribute (Cramér's V == 0)."""
    df = _continuous_proxy_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = bd_identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )
    by_feat = _by_feature(results)
    assert "credit_rating" in by_feat, "continuous proxy of a categorical attribute was not flagged"
    r = by_feat["credit_rating"]
    assert r.correlation_type == "Correlation ratio (eta)"
    assert r.correlation > 0.9
    assert r.risk_level.value in ("critical", "high")


def test_feature_engineering_flags_continuous_proxy_of_categorical_attr():
    """Defect twin: feature_engineering.identify_proxy_variables returned 0."""
    df = _continuous_proxy_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = fe_identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )
    by_feat = _by_feature(results)
    assert "credit_rating" in by_feat
    r = by_feat["credit_rating"]
    assert r.correlation_type == "Correlation ratio (eta)"
    assert r.correlation > 0.9
    assert r.risk_level.value in ("critical", "high")


def test_compute_proxy_correlations_uses_eta_for_mixed_case():
    """compute_proxy_correlations must report the correlation ratio, not the
    degenerate Cramér's V, for a continuous feature vs a categorical attr."""
    df = _continuous_proxy_frame()
    out = compute_proxy_correlations(df, "credit_rating", "race")
    assert out["primary_correlation_type"] == "Correlation ratio (eta)"
    assert out["primary_correlation"] > 0.9


# ── MEDIUM: compute_feature_correlations must not report the proxy as weak ───


def test_compute_feature_correlations_reports_strong_for_continuous_proxy():
    """Defect: compute_feature_correlations(method='auto') reported the
    near-perfect continuous proxy as 'weak' (|corr| ~0.03) via point-biserial
    on 3 arbitrary category codes; it must now read as a strong association."""
    df = _continuous_proxy_frame()
    m = compute_feature_correlations(df, protected_attributes=["race"], method="auto")
    assert abs(m.correlations.loc["credit_rating", "race"]) > 0.8


# ── does-not-overcorrect: an unrelated continuous feature stays UNflagged ────


def test_unrelated_continuous_feature_not_flagged_either_detector():
    df = _continuous_proxy_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bd = bd_identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )
        fe = fe_identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )
    assert "unrelated" not in _by_feature(bd)
    assert "unrelated" not in _by_feature(fe)


def test_compute_feature_correlations_unrelated_stays_weak():
    df = _continuous_proxy_frame()
    m = compute_feature_correlations(df, protected_attributes=["race"], method="auto")
    assert abs(m.correlations.loc["unrelated", "race"]) < 0.3


# ── does-not-overcorrect: categorical x categorical still uses Cramér's V ─────


def test_categorical_categorical_still_uses_cramers_v():
    rng = np.random.default_rng(7)
    n = 500
    race = rng.choice(["W", "B", "A"], n)
    # zip strongly tracks race but both sides are categorical.
    zipc = np.where(
        race == "W",
        rng.choice(["1", "2"], n, p=[0.9, 0.1]),
        np.where(
            race == "B",
            rng.choice(["3", "4"], n, p=[0.9, 0.1]),
            rng.choice(["5", "6"], n, p=[0.9, 0.1]),
        ),
    )
    df = pd.DataFrame({"race": race, "zipc": zipc})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = bd_identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )
    by_feat = _by_feature(results)
    assert "zipc" in by_feat
    assert by_feat["zipc"].correlation_type == "Cramér's V"
    assert by_feat["zipc"].correlation > 0.7


# ── does-not-overcorrect: binary attribute still uses point-biserial ─────────


def test_binary_attribute_continuous_proxy_still_detected():
    """A binary categorical side is genuinely dichotomous, so point-biserial
    stays valid; the continuous proxy must still read as strong and an
    unrelated numeric feature must stay weak."""
    rng = np.random.default_rng(11)
    n = 500
    gender = rng.choice(["M", "F"], n)
    salary = np.where(gender == "M", 80000.0, 50000.0) + rng.normal(0, 3000, n)
    noise = rng.normal(0, 1, n)
    df = pd.DataFrame({"gender": gender, "salary": salary, "noise": noise})
    m = compute_feature_correlations(df, protected_attributes=["gender"], method="auto")
    assert abs(m.correlations.loc["salary", "gender"]) > 0.8
    assert abs(m.correlations.loc["noise", "gender"]) < 0.3
