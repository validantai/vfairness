"""Cross-library parity tests for the core group-fairness metrics.

Validates that vfairness produces the SAME numbers as the de-facto industry
references -- fairlearn (Microsoft) and AI Fairness 360 (IBM) -- on identical
data. This is differential testing against reference implementations: if all
three independent codebases agree, the vfairness metric code is correct per the
established standard, not merely self-consistent.

The reference libraries are optional; each comparison skips cleanly via
``pytest.importorskip`` when the library is absent, so this file never breaks a
minimal environment. To run the full comparison:

    pip install fairlearn aif360
    pytest tests/test_cross_library_parity.py -v

Definition alignment (verified empirically, machine precision):
  demographic_parity_difference : max(SR) - min(SR)
      == fairlearn.demographic_parity_difference
      == abs(AIF360 statistical_parity_difference)        [binary]
  demographic_parity_ratio      : min(SR) / max(SR)
      == fairlearn.demographic_parity_ratio
      == AIF360 disparate_impact (privileged = higher-SR group)  [binary]
  equal_opportunity_difference  : max(TPR) - min(TPR)
      == fairlearn MetricFrame(true_positive_rate).difference
      == abs(AIF360 equal_opportunity_difference)         [binary]
  equalized_odds_difference     : max(TPR spread, FPR spread)
      == fairlearn.equalized_odds_difference
      AIF360 has no max-form metric (average_odds_difference AVERAGES the TPR/FPR
      gaps rather than taking the max), so for the apples-to-apples check we
      reconstruct it from AIF360's per-group TPR/FPR.

vfairness is called with ``min_group_size=1`` so it never drops a group, matching
the no-filtering behaviour of fairlearn / AIF360.
"""

import numpy as np
import pytest

from vfairness.evaluation import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)

# Counts come from exact integer arithmetic in every library, so agreement is to
# machine precision; 1e-9 leaves headroom for float division ordering only.
ATOL = 1e-9
_VF_KW = dict(min_group_size=1)
_METRICS = ("dpd", "dpr", "eo", "eqo")


def _make(seed, n=2000, n_groups=2, gap=0.3):
    """Synthetic data with a graded selection-rate gap across groups.

    y_true is drawn independently of y_pred so TPR/FPR-based metrics are
    non-trivial. Group sizes are large enough that every group has both label
    classes and a non-zero selection rate (so disparate impact is well defined).
    """
    rng = np.random.default_rng(seed)
    sens = rng.integers(0, n_groups, n)
    base = 0.5 - gap / 2
    span = gap / max(n_groups - 1, 1)
    p_pred = base + sens * span  # group 0 lowest SR ... group n-1 highest SR
    y_pred = (rng.random(n) < p_pred).astype(int)
    y_true = (rng.random(n) < 0.5).astype(int)
    return y_true, y_pred, sens


def _vfairness(y_true, y_pred, sens):
    return {
        "dpd": demographic_parity_difference(y_true, y_pred, sens, **_VF_KW),
        "dpr": demographic_parity_ratio(y_true, y_pred, sens, **_VF_KW),
        "eo": equal_opportunity_difference(y_true, y_pred, sens, **_VF_KW),
        "eqo": equalized_odds_difference(y_true, y_pred, sens, **_VF_KW),
    }


def _fairlearn(y_true, y_pred, sens):
    flm = pytest.importorskip("fairlearn.metrics")
    tpr_frame = flm.MetricFrame(
        metrics=flm.true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sens
    )
    return {
        "dpd": flm.demographic_parity_difference(y_true, y_pred, sensitive_features=sens),
        "dpr": flm.demographic_parity_ratio(y_true, y_pred, sensitive_features=sens),
        "eo": tpr_frame.difference(method="between_groups"),
        "eqo": flm.equalized_odds_difference(y_true, y_pred, sensitive_features=sens),
    }


def _aif360(y_true, y_pred, sens):
    pytest.importorskip("aif360")
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import ClassificationMetric

    # privileged = the higher-selection-rate group, so disparate_impact comes out
    # as min/max in [0, 1], matching vfairness's demographic_parity_ratio.
    rates = {int(g): float(y_pred[sens == g].mean()) for g in np.unique(sens)}
    priv, unpriv = max(rates, key=rates.get), min(rates, key=rates.get)

    def _ds(labels):
        return BinaryLabelDataset(
            df=pd.DataFrame({"label": labels, "grp": sens}),
            label_names=["label"],
            protected_attribute_names=["grp"],
            favorable_label=1,
            unfavorable_label=0,
        )

    cm = ClassificationMetric(
        _ds(y_true),
        _ds(y_pred),
        unprivileged_groups=[{"grp": unpriv}],
        privileged_groups=[{"grp": priv}],
    )
    tpr_gap = abs(cm.true_positive_rate(privileged=False) - cm.true_positive_rate(privileged=True))
    fpr_gap = abs(
        cm.false_positive_rate(privileged=False) - cm.false_positive_rate(privileged=True)
    )
    return {
        "dpd": abs(cm.statistical_parity_difference()),
        "dpr": cm.disparate_impact(),
        "eo": abs(cm.equal_opportunity_difference()),
        "eqo": max(tpr_gap, fpr_gap),
    }


def _assert_parity(vf, other, lib, ctx=""):
    for k in _METRICS:
        assert vf[k] == pytest.approx(other[k], abs=ATOL), (
            f"{k}{ctx}: vfairness={vf[k]!r} {lib}={other[k]!r}"
        )


@pytest.mark.parametrize("seed", [1, 7, 42, 100, 2024])
def test_parity_vs_fairlearn_binary(seed):
    """vfairness == fairlearn (Microsoft) on a binary protected attribute."""
    yt, yp, s = _make(seed, n_groups=2)
    _assert_parity(_vfairness(yt, yp, s), _fairlearn(yt, yp, s), "fairlearn")


@pytest.mark.parametrize("seed", [3, 11, 99])
@pytest.mark.parametrize("n_groups", [3, 4])
def test_parity_vs_fairlearn_multigroup(seed, n_groups):
    """vfairness == fairlearn for >2 groups (max-min / worst-case definitions)."""
    yt, yp, s = _make(seed, n_groups=n_groups)
    _assert_parity(_vfairness(yt, yp, s), _fairlearn(yt, yp, s), "fairlearn", f" (g={n_groups})")


@pytest.mark.parametrize("seed", [1, 7, 42, 100, 2024])
def test_parity_vs_aif360_binary(seed):
    """vfairness == AI Fairness 360 (IBM) on a binary protected attribute."""
    yt, yp, s = _make(seed, n_groups=2)
    _assert_parity(_vfairness(yt, yp, s), _aif360(yt, yp, s), "aif360")


def test_three_way_agreement_snapshot():
    """All three libraries agree on one fixed dataset (documents the numbers)."""
    yt, yp, s = _make(7, n_groups=2)
    vf, fl, aif = _vfairness(yt, yp, s), _fairlearn(yt, yp, s), _aif360(yt, yp, s)
    for k in _METRICS:
        spread = max(vf[k], fl[k], aif[k]) - min(vf[k], fl[k], aif[k])
        assert spread < ATOL, f"{k}: vf={vf[k]} fairlearn={fl[k]} aif360={aif[k]}"
