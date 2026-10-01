"""Golden-file regression tests for the core fairness metrics.

VI_A_GR_028: the core-metric tests elsewhere use loose bounds or trivial
inputs. This module pins values two ways:

1. REGRESSION pins (``GOLDEN`` below): computed once BY THE CODE UNDER TEST
   on ``tests/fixtures/recruitment_fairness_dataset.csv``. They are honest
   drift detectors, not independent ground truth: if the implementation was
   wrong when they were generated, they pin the wrong value. Note also that
   this fixture is DEGENERATE for equalized odds: every group has FPR = 0
   (no negative-truth row is ever invited; verified 2026-08-02), so the FPR
   branch of ``equalized_odds_difference`` is never exercised here and its
   golden value collapses to the equal-opportunity (TPR) value.
2. HAND-DERIVED ground truth (``TestHandDerivedConfusionMatrices``): a small
   synthetic fixture whose per-group confusion matrices are written out
   explicitly, with every expected metric derived by hand as an exact
   fraction, independently of the implementation. It deliberately has a
   nonzero FPR gap LARGER than the TPR gap so the FPR branch of equalized
   odds decides the result.

Everything is deterministic:
    - the fixture is read in file order (no sampling, no shuffling),
    - derived label / score / position arrays are pure functions of columns,
    - metrics take no randomness.

If a metric intentionally changes, re-pin the regression constants in the
SAME change and note why in the commit message. The hand-derived values must
NEVER be re-pinned to match the code: if they fail, the code is wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vfairness import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics import classification as C
from vfairness.evaluation.vfairness_metrics import ranking as K
from vfairness.evaluation.vfairness_metrics import regression as R

FIXTURE = Path(__file__).parent / "fixtures" / "recruitment_fairness_dataset.csv"

# Tolerance for the pinned golden values. Tight enough to catch a real drift,
# loose enough to survive platform float rounding.
ABS = 1e-9

# Golden REGRESSION values: self-generated (computed from the fixture with
# the code under test), so they detect drift but do not prove correctness.
# Correctness is covered by TestHandDerivedConfusionMatrices below.
GOLDEN = {
    # Classification parity differences / ratios
    "cls_demographic_parity_difference": 0.043808421776,
    "cls_demographic_parity_ratio": 0.357635206787,
    "cls_equalized_odds_difference": 0.048889372822,
    "cls_equal_opportunity_difference": 0.048889372822,
    # Regression parity
    "reg_mae_parity_difference": 0.031437273638,
    "reg_rmse_parity_difference": 0.032052401153,
    "reg_mean_prediction_difference": 0.036633649183,
    "reg_r2_parity_difference": 0.406122948426,
    # Ranking exposure
    "rank_exposure_parity_difference": 0.005838709330,
    "rank_exposure_parity_ratio": 0.945072092609,
    # Analyzer report dict
    "analyzer_report_dpd": 0.043808421776,
    "analyzer_report_fairness_score": 0.8,
}


@pytest.fixture(scope="module")
def arrays():
    """Deterministic label / score / group / position arrays from the fixture."""
    assert FIXTURE.exists(), f"missing golden fixture: {FIXTURE}"
    df = pd.read_csv(FIXTURE)
    sens = df["gender"].to_numpy()
    y_true_cls = (df["true_qualification_score"] >= 0.5).astype(int).to_numpy()
    y_pred_cls = (
        (df["invite_decision"].astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()
    )
    y_true_reg = df["true_qualification_score"].astype(float).to_numpy()
    y_pred_reg = df["model_score"].astype(float).to_numpy()
    # Ranking positions: 0 = top, ranked by model_score descending, stable
    # tie-break by original row order for reproducibility.
    order = np.argsort(-y_pred_reg, kind="stable")
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(len(order))
    return {
        "sens": sens,
        "y_true_cls": y_true_cls,
        "y_pred_cls": y_pred_cls,
        "y_true_reg": y_true_reg,
        "y_pred_reg": y_pred_reg,
        "positions": positions,
    }


def test_fixture_is_stable(arrays):
    """Row count and group set are pinned so a fixture edit is loud."""
    assert len(arrays["sens"]) == 2700
    assert set(np.unique(arrays["sens"])) == {"Female", "Male", "Non-binary"}


def test_classification_golden(arrays):
    a = arrays
    assert C.demographic_parity_difference(
        a["y_true_cls"], a["y_pred_cls"], a["sens"]
    ) == pytest.approx(GOLDEN["cls_demographic_parity_difference"], abs=ABS)
    assert C.demographic_parity_ratio(a["y_true_cls"], a["y_pred_cls"], a["sens"]) == pytest.approx(
        GOLDEN["cls_demographic_parity_ratio"], abs=ABS
    )
    assert C.equalized_odds_difference(
        a["y_true_cls"], a["y_pred_cls"], a["sens"]
    ) == pytest.approx(GOLDEN["cls_equalized_odds_difference"], abs=ABS)
    assert C.equal_opportunity_difference(
        a["y_true_cls"], a["y_pred_cls"], a["sens"]
    ) == pytest.approx(GOLDEN["cls_equal_opportunity_difference"], abs=ABS)


def test_regression_golden(arrays):
    a = arrays
    assert R.mae_parity_difference(a["y_true_reg"], a["y_pred_reg"], a["sens"]) == pytest.approx(
        GOLDEN["reg_mae_parity_difference"], abs=ABS
    )
    assert R.rmse_parity_difference(a["y_true_reg"], a["y_pred_reg"], a["sens"]) == pytest.approx(
        GOLDEN["reg_rmse_parity_difference"], abs=ABS
    )
    assert R.mean_prediction_difference(
        a["y_true_reg"], a["y_pred_reg"], a["sens"]
    ) == pytest.approx(GOLDEN["reg_mean_prediction_difference"], abs=ABS)
    assert R.r2_parity_difference(a["y_true_reg"], a["y_pred_reg"], a["sens"]) == pytest.approx(
        GOLDEN["reg_r2_parity_difference"], abs=ABS
    )


def test_ranking_golden(arrays):
    a = arrays
    assert K.exposure_parity_difference(a["positions"], a["sens"]) == pytest.approx(
        GOLDEN["rank_exposure_parity_difference"], abs=ABS
    )
    assert K.exposure_parity_ratio(a["positions"], a["sens"]) == pytest.approx(
        GOLDEN["rank_exposure_parity_ratio"], abs=ABS
    )


def test_analyzer_report_golden(arrays):
    a = arrays
    analyzer = FairnessAnalyzer(
        a["y_true_cls"],
        a["y_pred_cls"],
        a["sens"],
        task_type="classification",
        min_group_size=30,
    )
    report = analyzer.get_report()
    assert report["task_type"] == "classification"
    assert set(report["metrics"].keys()) == {
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "predictive_parity_difference",
    }
    assert report["metrics"]["demographic_parity_difference"] == pytest.approx(
        GOLDEN["analyzer_report_dpd"], abs=ABS
    )
    assert report["assessment"]["fairness_score"] == pytest.approx(
        GOLDEN["analyzer_report_fairness_score"], abs=ABS
    )
    # The analyzer method and the module-level function must agree exactly.
    assert float(analyzer.demographic_parity_difference()) == pytest.approx(
        C.demographic_parity_difference(a["y_true_cls"], a["y_pred_cls"], a["sens"]),
        abs=ABS,
    )


# ---------------------------------------------------------------------------
# Hand-derived ground truth (independent of the implementation)
# ---------------------------------------------------------------------------
#
# Two groups of 40 rows each (>= the default min_group_size of 30), with the
# per-group confusion matrix chosen so every metric is an exact fraction:
#
#   Group A (n=40):  TP=12  FN=4   FP=12  TN=12
#     positives predicted = TP+FP = 24  -> selection rate = 24/40 = 0.60
#     TPR = TP/(TP+FN) = 12/16 = 0.75
#     FPR = FP/(FP+TN) = 12/24 = 0.50
#     PPV = TP/(TP+FP) = 12/24 = 0.50
#
#   Group B (n=40):  TP=11  FN=5   FP=3   TN=21
#     positives predicted = TP+FP = 14  -> selection rate = 14/40 = 0.35
#     TPR = 11/16 = 0.6875
#     FPR = 3/24  = 0.125
#     PPV = 11/14
#
# Expected metric values (exact, by hand):
#   demographic_parity_difference = 0.60 - 0.35            = 0.25
#   demographic_parity_ratio      = 0.35 / 0.60             = 7/12
#   equal_opportunity_difference  = 0.75 - 0.6875           = 0.0625 (TPR gap)
#   FPR gap                       = 0.50 - 0.125            = 0.375
#   equalized_odds_difference     = max(0.0625, 0.375)      = 0.375
#     (the FPR gap DECIDES the result here, so the FPR branch is exercised,
#      unlike the recruitment fixture above where FPR = 0 in every group)
#   predictive_parity_difference  = 11/14 - 1/2 = 8/28      = 2/7


def _hand_fixture():
    """Materialize the confusion matrices above as flat label arrays."""
    rows = []

    def add_group(name, tp, fn, fp, tn):
        rows.extend([(name, 1, 1)] * tp)  # true=1, pred=1
        rows.extend([(name, 1, 0)] * fn)  # true=1, pred=0
        rows.extend([(name, 0, 1)] * fp)  # true=0, pred=1
        rows.extend([(name, 0, 0)] * tn)  # true=0, pred=0

    add_group("A", tp=12, fn=4, fp=12, tn=12)
    add_group("B", tp=11, fn=5, fp=3, tn=21)
    sens = np.array([r[0] for r in rows])
    y_true = np.array([r[1] for r in rows])
    y_pred = np.array([r[2] for r in rows])
    return y_true, y_pred, sens


class TestHandDerivedConfusionMatrices:
    """Exact hand-computed expectations; see the derivation block above."""

    def test_fixture_confusion_matrices_are_as_documented(self):
        y_true, y_pred, sens = _hand_fixture()
        for grp, (tp, fn, fp, tn) in {
            "A": (12, 4, 12, 12),
            "B": (11, 5, 3, 21),
        }.items():
            m = sens == grp
            assert int(((y_true == 1) & (y_pred == 1) & m).sum()) == tp
            assert int(((y_true == 1) & (y_pred == 0) & m).sum()) == fn
            assert int(((y_true == 0) & (y_pred == 1) & m).sum()) == fp
            assert int(((y_true == 0) & (y_pred == 0) & m).sum()) == tn

    def test_demographic_parity_difference_hand_value(self):
        y_true, y_pred, sens = _hand_fixture()
        assert C.demographic_parity_difference(y_true, y_pred, sens) == pytest.approx(0.25, abs=ABS)

    def test_demographic_parity_ratio_hand_value(self):
        y_true, y_pred, sens = _hand_fixture()
        assert C.demographic_parity_ratio(y_true, y_pred, sens) == pytest.approx(7 / 12, abs=ABS)

    def test_equal_opportunity_difference_hand_value(self):
        y_true, y_pred, sens = _hand_fixture()
        assert C.equal_opportunity_difference(y_true, y_pred, sens) == pytest.approx(
            0.0625, abs=ABS
        )

    def test_equalized_odds_difference_is_the_fpr_gap(self):
        # 0.375 is the FPR gap; the TPR gap is only 0.0625. Any value other
        # than 0.375 means the FPR branch is broken or ignored.
        y_true, y_pred, sens = _hand_fixture()
        assert C.equalized_odds_difference(y_true, y_pred, sens) == pytest.approx(0.375, abs=ABS)

    def test_predictive_parity_difference_hand_value(self):
        y_true, y_pred, sens = _hand_fixture()
        assert C.predictive_parity_difference(y_true, y_pred, sens) == pytest.approx(2 / 7, abs=ABS)

    def test_recruitment_fixture_fpr_degeneracy_is_documented_truth(self):
        """Pin the degeneracy claim itself: every group's FPR is 0 in the
        recruitment fixture. If the fixture is ever regenerated with nonzero
        FPRs, this fails and the docstring above must be updated."""
        df = pd.read_csv(FIXTURE)
        sens = df["gender"].to_numpy()
        y_true = (df["true_qualification_score"] >= 0.5).astype(int).to_numpy()
        y_pred = (
            (df["invite_decision"].astype(str).str.strip().str.lower() == "yes")
            .astype(int)
            .to_numpy()
        )
        for grp in np.unique(sens):
            neg = (sens == grp) & (y_true == 0)
            assert int((neg & (y_pred == 1)).sum()) == 0, (
                f"group {grp} now has false positives; fixture no longer "
                "degenerate for equalized odds"
            )
