"""Third-iteration audit fixes: eval-core-stats.

Pins two defects reproduced by execution in
docs/audits/third-iteration-audit-2026-08-22.md:

HIGH (intersectional.py): the over/under-prediction finding attached a
significance test of the WRONG hypothesis. The finding claims a within-group
over/under-prediction relative to the group's OWN base rate
(delta = prediction_rate - ground_truth_rate), but the test compared the
group's prediction rate to the POPULATION selection rate, so
`statistically_significant` was decoupled from the claim. Fixed to test
pred_rate vs the group's own gt_rate at n = size.

MEDIUM (discovery.py): scan_fairness_violations labelled
privileged/disadvantaged by selection rate for EVERY metric, inverting the
attribution for error-based metrics (TPR/FPR/PPV). Fixed to identify the pair
from the rate the triggering metric is built from.
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import scan_fairness_violations
from vfairness.evaluation.vfairness_metrics.intersectional import (
    _proportion_test,
    generate_structured_findings,
)

# --------------------------------------------------------------------------- #
# HIGH: over/under-prediction significance tests the group's OWN base rate     #
# --------------------------------------------------------------------------- #


def _group(name, pred, gt, size, fpr=0.1):
    return {
        "group": name,
        "positive_rate": pred,
        "ground_truth_rate": gt,
        "prediction_delta": pred - gt,
        "false_positive_rate": fpr,
        "size": size,
    }


class TestOverPredictionHypothesis:
    def test_defect_massive_within_group_over_prediction_is_significant(self):
        """Group A grossly over-predicts its OWN base rate (gt=0.06, pred=0.475)
        while its selection rate (~0.5) sits right on the population's. Under the
        old code (pred vs overall selection rate) this deterministic, critical
        over-prediction was marked NOT significant (p~0.54). It must now be
        confirmed significant."""
        groups = [
            _group("A", 0.475, 0.06, 200, fpr=0.44),
            _group("B", 0.5125, 0.5125, 200),
            _group("C", 0.5125, 0.5125, 200),
        ]
        inter = {
            "all_groups": groups,
            "overall_rate": 0.5,
            "max_disparity": 0.0,
            "max_ground_truth_disparity": 0.0,
        }
        findings = generate_structured_findings(inter)
        over_a = [f for f in findings if f["type"] == "over_prediction" and f["groups"] == ["A"]]
        assert len(over_a) == 1
        assert over_a[0]["statistically_significant"] is True
        assert over_a[0]["p_value"] < 1e-6

    def test_p_value_is_pred_vs_own_gt_not_vs_population(self):
        """Pin the exact hypothesis: p_value == two-proportion test of
        pred_rate vs the group's OWN gt_rate at n = size, and NOT the old
        pred_rate vs population-selection-rate comparison."""
        groups = [
            _group("b", 0.50, 0.30, 100),  # within-group over-prediction
            _group("a", 0.30, 0.30, 100),  # calibrated
        ]
        inter = {
            "all_groups": groups,
            "overall_rate": 0.90,  # deliberately unrelated stored baseline
            "max_disparity": 0.0,
            "max_ground_truth_disparity": 0.0,
        }
        findings = generate_structured_findings(inter)
        over_b = [f for f in findings if f["type"] == "over_prediction" and f["groups"] == ["b"]]
        assert len(over_b) == 1

        correct_p = round(_proportion_test(0.50, 100, 0.30, 100), 6)  # pred vs own gt
        # The old (buggy) baseline: pred vs the pooled population selection rate.
        total_n = 200
        overall_pred_rate = (0.50 * 100 + 0.30 * 100) / total_n  # 0.40
        wrong_p = round(_proportion_test(0.50, 100, overall_pred_rate, total_n), 6)

        assert over_b[0]["p_value"] == correct_p
        assert over_b[0]["p_value"] != wrong_p

    def test_does_not_overcorrect_calibrated_groups_yield_no_findings(self):
        """A dataset where every group is well calibrated (pred == gt) must
        produce zero over/under-prediction findings even when the between-group
        SELECTION rates differ hugely. The new within-group test must not
        manufacture findings from a between-group selection disparity."""
        groups = [
            _group("hi", 0.90, 0.90, 200),
            _group("lo", 0.10, 0.10, 200),
        ]
        inter = {
            "all_groups": groups,
            "overall_rate": 0.50,
            "max_disparity": 0.0,
            "max_ground_truth_disparity": 0.0,
        }
        findings = generate_structured_findings(inter)
        oup = [f for f in findings if f["type"] in ("over_prediction", "under_prediction")]
        assert oup == []

    def test_does_not_overcorrect_modest_gap_stays_non_significant(self):
        """A small within-group over-prediction that is genuinely NOT
        significant must stay non-significant, even for a group whose selection
        rate is far from the population (the old code would have confirmed it
        purely from that between-group gap)."""
        groups = [
            _group("g", 0.30, 0.24, 40),  # delta 0.06 (just over trigger), small n
            _group("h", 0.90, 0.90, 400),  # pulls the population selection rate up
            _group("i", 0.90, 0.90, 400),
        ]
        inter = {
            "all_groups": groups,
            "overall_rate": 0.75,
            "max_disparity": 0.0,
            "max_ground_truth_disparity": 0.0,
        }
        findings = generate_structured_findings(inter)
        over_g = [f for f in findings if f["type"] == "over_prediction" and f["groups"] == ["g"]]
        assert len(over_g) == 1  # the delta > 0.05 trigger still fires
        # pred (0.30) vs its own gt (0.24) at n=40 is not significant...
        assert over_g[0]["statistically_significant"] is False
        # ...whereas pred (0.30) vs the population selection rate (~0.84) is
        # extreme, which is exactly the false confirmation the fix removes.
        assert round(_proportion_test(0.30, 40, 0.24, 40), 6) > round(
            _proportion_test(0.30, 40, 0.84, 840), 6
        )


# --------------------------------------------------------------------------- #
# MEDIUM: violation group attribution uses the metric's own rate               #
# --------------------------------------------------------------------------- #


def _make_arm(n_pos, tp, n_neg, fp):
    """Build (y_true, y_pred) for one group from a confusion-matrix spec."""
    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_pred = np.array([1] * tp + [0] * (n_pos - tp) + [1] * fp + [0] * (n_neg - fp))
    return y_true, y_pred


def _two_group_frame(spec_a, spec_b):
    yta, ypa = _make_arm(*spec_a)
    ytb, ypb = _make_arm(*spec_b)
    y_true = np.concatenate([yta, ytb])
    y_pred = np.concatenate([ypa, ypb])
    grp = np.array(["A"] * len(yta) + ["B"] * len(ytb))
    df = pd.DataFrame({"grp": grp})
    return df, y_pred, y_true


class TestViolationGroupAttribution:
    def test_defect_equal_opportunity_labels_by_tpr_not_selection(self):
        """A has the HIGHER selection rate (0.60) but the LOWER TPR (0.62);
        B has the lower selection rate (0.515) but TPR 1.0. For equal
        opportunity the disadvantaged group is the one with the lower TPR (A).
        The old code labelled A privileged and B disadvantaged (inverted)
        because A's selection rate is higher."""
        # A: 100 pos / TP 62 (TPR .62); 300 neg / FP 178  -> selection .60
        # B: 100 pos / TP 100 (TPR 1.0); 300 neg / FP 106  -> selection .515
        df, y_pred, y_true = _two_group_frame((100, 62, 300, 178), (100, 100, 300, 106))
        vios = scan_fairness_violations(
            df,
            y_pred,
            y_true,
            protected_columns=["grp"],
            metrics=["equal_opportunity_difference"],
            threshold=0.05,
        )
        assert len(vios) == 1
        v = vios[0]
        assert v.disadvantaged_group == "A"  # lowest TPR
        assert v.privileged_group == "B"  # highest TPR

    def test_does_not_overcorrect_demographic_parity_still_uses_selection(self):
        """Demographic parity is a selection-rate metric; its attribution must
        stay selection-based (higher selection = privileged). The fix must not
        change the answer for the metric that was already correct."""
        # A selection .60, B selection .515 (labels are irrelevant here).
        df, y_pred, y_true = _two_group_frame((100, 62, 300, 178), (100, 100, 300, 106))
        vios = scan_fairness_violations(
            df,
            y_pred,
            y_true,
            protected_columns=["grp"],
            metrics=["demographic_parity_difference"],
            threshold=0.05,
        )
        assert len(vios) == 1
        v = vios[0]
        assert v.privileged_group == "A"  # highest selection rate
        assert v.disadvantaged_group == "B"

    def test_predictive_parity_labels_by_ppv_opposite_to_selection(self):
        """PPV points opposite to selection here: A has LOWER selection (0.25)
        and LOWER PPV (0.50); B has HIGHER selection (0.50) and HIGHER PPV
        (0.90). Under the favourable-positive convention a lower PPV means more
        unwarranted positive predictions (leniency), so A is privileged and B
        disadvantaged. Selection-based labelling would invert this."""
        # A: 100 pred-pos, TP 50 -> PPV .50, selection 100/400 = .25
        # B: 200 pred-pos, TP 180 -> PPV .90, selection 200/400 = .50
        # A: 100 pos / TP 50; 300 neg / FP 50
        # B: 200 pos / TP 180; 200 neg / FP 20
        df, y_pred, y_true = _two_group_frame((100, 50, 300, 50), (200, 180, 200, 20))
        vios = scan_fairness_violations(
            df,
            y_pred,
            y_true,
            protected_columns=["grp"],
            metrics=["predictive_parity_difference"],
            threshold=0.05,
        )
        assert len(vios) == 1
        v = vios[0]
        assert v.privileged_group == "A"  # lower PPV
        assert v.disadvantaged_group == "B"  # higher PPV

    def test_equalized_odds_labels_from_the_driving_arm(self):
        """FPR gap (0.5) drives over the TPR gap (0.4), and the FPR ordering is
        opposite to the TPR ordering. Labels must come from the FPR arm (higher
        FPR = privileged): B privileged, A disadvantaged. Using the TPR arm
        would give the opposite (A privileged)."""
        # A: TPR .9, FPR .1  ->  200 pos / TP 180 ; 200 neg / FP 20
        # B: TPR .5, FPR .6  ->  200 pos / TP 100 ; 200 neg / FP 120
        df, y_pred, y_true = _two_group_frame((200, 180, 200, 20), (200, 100, 200, 120))
        vios = scan_fairness_violations(
            df,
            y_pred,
            y_true,
            protected_columns=["grp"],
            metrics=["equalized_odds_difference"],
            threshold=0.05,
        )
        assert len(vios) == 1
        v = vios[0]
        assert v.value == pytest.approx(0.5, abs=1e-9)  # FPR gap is the max
        assert v.privileged_group == "B"  # highest FPR (driving arm)
        assert v.disadvantaged_group == "A"
