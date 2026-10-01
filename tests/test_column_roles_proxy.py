"""Regression test: correlation-based proxy detection in the typology gate.

`classify_column_roles` precomputes, per declared protected attribute, which
other columns correlate with it (via `identify_proxy_features`) and flags them
as ``proxy_candidate``. A latent bug passed the whole target list where a single
column name was expected, so the call always raised, was swallowed, and proxy
detection was a silent no-op. This test pins the working behavior so it cannot
regress.
"""

import numpy as np
import pandas as pd

from vfairness.evaluation.vfairness_metrics.discovery import classify_column_roles


def _dataset(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.integers(0, 2, n)
    # `feat_x` agrees with gender ~80% of the time: correlation well above the
    # 0.3 proxy threshold, low cardinality (not a near-unique id), and a neutral
    # name (not a proxy-name hint), so it can ONLY be caught by correlation.
    flip = rng.random(n) < 0.2
    feat_x = np.where(flip, 1 - gender, gender)
    # `noise` is an independent binary column: uncorrelated, not a proxy.
    noise = rng.integers(0, 2, n)
    return pd.DataFrame({"gender": gender, "feat_x": feat_x, "noise": noise})


def test_correlated_column_is_flagged_as_proxy_candidate():
    df = _dataset()
    result = classify_column_roles(df, declared_protected=["gender"])

    roles = {r["column"]: r for r in result["roles"]}

    # The declared protected attribute stays protected.
    assert roles["gender"]["role"] == "protected"
    assert "gender" in result["protected"]

    # The correlated feature is caught by correlation-based proxy detection.
    assert "feat_x" in result["proxy_candidates"]
    assert roles["feat_x"]["role"] == "proxy_candidate"
    assert "correlates with protected attribute gender" in roles["feat_x"]["reason"]
    assert roles["feat_x"]["evidence"].get("correlation") is not None

    # The uncorrelated column is NOT caught by correlation-based detection.
    # (It may be flagged by an unrelated heuristic, but it carries no
    # correlation signal, which is the branch this fix restored.)
    assert "correlates with protected attribute" not in roles["noise"]["reason"]
    assert roles["noise"]["evidence"].get("correlation") is None
