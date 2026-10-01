"""
Lundberg decomposition tests. The 1e-6 identity is the spine; if it
breaks the run is invalid and must fail loudly. These tests lock the
identity, the proxy-score math, and the typical-fixture invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from vfairness.xai.decomposition import (
    lundberg_fairness_decomposition,
    proxy_score,
)


def _synth_shap(seed: int = 7) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic SHAP matrix + group labels with a designed proxy feature."""
    rng = np.random.default_rng(seed)
    n_samples = 400
    feature_names = ["income", "credit_util", "postal_code", "savings"]
    # postal_code is the planted proxy: skewed by group.
    shap = rng.normal(loc=0.0, scale=0.05, size=(n_samples, len(feature_names)))
    groups = rng.integers(0, 2, size=n_samples)
    shap[groups == 0, 2] += 0.04  # the proxy
    return shap, groups, feature_names


def test_identity_holds_on_clean_input() -> None:
    shap, groups, names = _synth_shap()
    decomp = lundberg_fairness_decomposition(
        shap_values=shap,
        group_labels=groups,
        feature_names=names,
        metric="demographic_parity",
        protected_attribute="gender",
        subject_id="subject-1",
        audit_artifact_id="audit-1",
    )
    # ``assert_identity`` runs inside the function; this is a belt-and-braces check.
    assert decomp.identity_residual <= 1e-6


def test_planted_proxy_is_flagged() -> None:
    shap, groups, names = _synth_shap()
    decomp = lundberg_fairness_decomposition(
        shap_values=shap,
        group_labels=groups,
        feature_names=names,
        metric="demographic_parity",
        protected_attribute="gender",
        subject_id="subject-1",
        audit_artifact_id="audit-1",
        proxy_threshold=0.20,
    )
    # postal_code was planted with a 0.04 group-skew over noise of 0.05;
    # it should be the largest per-feature contribution and pass the threshold.
    largest = max(decomp.per_feature.items(), key=lambda kv: abs(kv[1]))[0]
    assert largest == "postal_code"
    assert "postal_code" in decomp.flagged_proxies


def test_identity_breaks_on_unit_mismatch() -> None:
    """Worker should refuse to write a row when the SHAP units are wrong.

    We simulate this by giving the function shap_values that, when
    summed, do NOT match the metric on the model output. The decomposer
    catches the mismatch via the 1e-6 assertion.
    """
    shap, groups, names = _synth_shap()
    # Halve all shap values; the per-feature contributions remain
    # additive, so the identity is still preserved here. We force a
    # mismatch instead by manually constructing a bad decomposition.
    from vfairness.xai.schemas import FairnessDecomposition

    bad = FairnessDecomposition(
        id="bad",
        subject_id="subject-1",
        metric="demographic_parity",
        protected_attribute="gender",
        total_disparity=0.10,
        per_feature={"a": 0.02, "b": 0.03},
        proxy_scores={"a": 0.4, "b": 0.6},
        flagged_proxies=[],
        audit_artifact_id="audit-1",
    )
    with pytest.raises(ValueError, match="identity broken"):
        bad.assert_identity()


def test_proxy_score_sums_to_one_when_nonzero() -> None:
    # total_disparity= was removed here (R6-3, 2026-09-10): it was required,
    # positional, and read nowhere. The refusal is pinned in
    # tests/test_readiness6_claims.py.
    p = proxy_score({"a": 0.04, "b": 0.06})
    assert pytest.approx(sum(p.values()), abs=1e-9) == 1.0
    assert p["b"] > p["a"]


def test_proxy_score_zero_input() -> None:
    p = proxy_score({"a": 0.0, "b": 0.0})
    assert p == {"a": 0.0, "b": 0.0}
