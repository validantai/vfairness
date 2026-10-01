"""
Tests for ranking fairness metrics.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness import (
    RankingFairnessResult,
    attention_weighted_rank_fairness,
    exposure_parity_difference,
    exposure_parity_ratio,
    get_ranking_group_metrics,
    normalized_discounted_kl_divergence,
)


class TestExposureParity:
    """Test exposure parity metrics."""

    def test_perfect_parity(self):
        """Test exposure parity with perfectly interleaved groups."""
        # Alternating groups in ranking
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        groups = np.array(["A", "B", "A", "B", "A", "B", "A", "B"])

        diff = exposure_parity_difference(rankings, groups, min_group_size=2)
        # Should be close to 0 (both groups get similar exposure)
        assert diff < 0.2

    def test_maximum_disparity(self):
        """Test exposure parity with one group ranked higher."""
        # Group A all at top, Group B all at bottom
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        diff = exposure_parity_difference(rankings, groups, min_group_size=2)
        # Should be significant (A gets more exposure)
        assert diff > 0.25  # Adjusted threshold (normalized exposure gives ~0.29)

    def test_exposure_ratio(self):
        """Test exposure parity ratio."""
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        ratio = exposure_parity_ratio(rankings, groups, min_group_size=2)
        # Ratio should be less than 1 (B gets less exposure)
        assert ratio < 1.0
        assert ratio >= 0.0

    def test_80_percent_rule(self):
        """Test 80% rule for exposure."""
        # Slight imbalance
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        groups = np.array(["A", "B", "A", "B", "A", "B", "A", "B", "B", "B"])

        ratio = exposure_parity_ratio(rankings, groups, min_group_size=2)
        # Check if passes or fails 80% rule
        assert isinstance(ratio, float)

    def test_exposure_types(self):
        """Test different exposure calculation types."""
        rankings = np.array([0, 1, 2, 3, 4, 5])
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        log_diff = exposure_parity_difference(
            rankings, groups, min_group_size=2, exposure_type="log"
        )
        linear_diff = exposure_parity_difference(
            rankings, groups, min_group_size=2, exposure_type="linear"
        )
        geo_diff = exposure_parity_difference(
            rankings, groups, min_group_size=2, exposure_type="geometric"
        )

        # All should be positive (A ranked higher)
        assert log_diff > 0
        assert linear_diff > 0
        assert geo_diff > 0


class TestAttentionWeightedFairness:
    """Test attention-weighted ranking fairness."""

    def test_basic_attention_fairness(self):
        """Test basic attention-weighted fairness."""
        rankings = np.array([0, 1, 2, 3, 4, 5])
        groups = np.array(["A", "A", "B", "B", "A", "B"])

        result = attention_weighted_rank_fairness(rankings, groups, min_group_size=2)

        assert isinstance(result, RankingFairnessResult)
        assert result.metric_name == "attention_weighted_rank_fairness"
        assert result.value >= 0
        assert "A" in result.group_exposures
        assert "B" in result.group_exposures

    def test_attention_models(self):
        """Test different attention models."""
        rankings = np.array([0, 1, 2, 3, 4, 5])
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        pos_result = attention_weighted_rank_fairness(
            rankings, groups, min_group_size=2, attention_model="position"
        )
        cascade_result = attention_weighted_rank_fairness(
            rankings, groups, min_group_size=2, attention_model="cascade"
        )

        # Both should show A gets more attention
        assert pos_result.group_attentions["A"] > pos_result.group_attentions["B"]
        assert cascade_result.group_attentions["A"] > cascade_result.group_attentions["B"]


class TestNDKL:
    """Test Normalized Discounted KL-Divergence."""

    def test_ndkl_uniform_distribution(self):
        """Test NDKL with uniform target distribution."""
        # Perfect representation
        rankings = np.array([0, 1, 2, 3, 4, 5])
        groups = np.array(["A", "B", "A", "B", "A", "B"])

        ndkl = normalized_discounted_kl_divergence(rankings, groups, min_group_size=2)
        # Should be close to 0 (matches uniform)
        assert ndkl < 0.5

    def test_ndkl_custom_target(self):
        """Test NDKL with custom target distribution."""
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])

        # Target: 60% A, 40% B
        ndkl = normalized_discounted_kl_divergence(
            rankings, groups, target_distribution={"A": 0.6, "B": 0.4}, min_group_size=2
        )
        assert isinstance(ndkl, float)
        assert ndkl >= 0

    def test_ndkl_top_k(self):
        """Test NDKL for top-k positions only."""
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        groups = np.array(["A", "A", "A", "B", "B", "B", "B", "B", "B", "B"])

        ndkl_full = normalized_discounted_kl_divergence(rankings, groups, min_group_size=2)
        ndkl_top5 = normalized_discounted_kl_divergence(rankings, groups, min_group_size=2, top_k=5)

        # Top-5 should show more bias (A overrepresented)
        assert isinstance(ndkl_top5, float)
        assert ndkl_top5 > ndkl_full


class TestGroupMetrics:
    """Test per-group ranking metrics."""

    def test_get_ranking_group_metrics(self):
        """Test getting per-group metrics."""
        rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        groups = np.array(["A", "A", "B", "B", "A", "A", "B", "B"])

        metrics = get_ranking_group_metrics(rankings, groups, min_group_size=2)

        assert "A" in metrics
        assert "B" in metrics

        for group in ["A", "B"]:
            assert "count" in metrics[group]
            assert "avg_position" in metrics[group]
            assert "avg_exposure" in metrics[group]
            assert "min_position" in metrics[group]
            assert "max_position" in metrics[group]
            assert "median_position" in metrics[group]

    def test_group_metrics_values(self):
        """Test that group metrics have sensible values."""
        rankings = np.array([0, 1, 2, 3])
        groups = np.array(["A", "A", "B", "B"])

        metrics = get_ranking_group_metrics(rankings, groups, min_group_size=2)

        # Group A has positions 0, 1 - better ranks
        assert metrics["A"]["avg_position"] < metrics["B"]["avg_position"]
        assert metrics["A"]["avg_exposure"] > metrics["B"]["avg_exposure"]


class TestRankingFairnessResult:
    """Test RankingFairnessResult dataclass."""

    def test_result_creation(self):
        """Test creating RankingFairnessResult."""
        result = RankingFairnessResult(
            metric_name="test_metric",
            value=0.15,
            group_exposures={"A": 0.6, "B": 0.4},
            group_attentions={"A": 0.55, "B": 0.45},
            is_fair=False,
            threshold=0.1,
        )
        assert result.metric_name == "test_metric"
        assert result.value == 0.15
        assert not result.is_fair

    def test_result_to_dict(self):
        """Test to_dict method."""
        result = RankingFairnessResult(
            metric_name="test",
            value=0.1,
            group_exposures={"A": 0.5, "B": 0.5},
            is_fair=True,
            threshold=0.1,
        )
        d = result.to_dict()
        assert d["metric_name"] == "test"
        assert d["value"] == 0.1
        assert d["is_fair"] is True


class TestEdgeCases:
    """Test edge cases for ranking fairness."""

    def test_single_group(self):
        """A ranking with one group has no between-group parity to report.

        REWRITTEN. This assertion used to read ``diff == 0.0``, i.e. it pinned
        the release-blocking defect as if it were the intended contract: a
        comparison that never ran, reported as perfect parity. 0.0 is
        indistinguishable downstream from a MEASURED perfect parity, and
        adapters_ranking renders it onto a green PASS badge. NaN is the honest
        answer and routes through the not-assessable path.
        """
        rankings = np.array([0, 1, 2, 3])
        groups = np.array(["A", "A", "A", "A"])

        diff = exposure_parity_difference(rankings, groups, min_group_size=2)
        assert math.isnan(diff)
        assert diff != 0.0

    def test_min_group_size(self):
        """A group dropped by the size gate leaves nothing to compare.

        REWRITTEN, same reason as :meth:`test_single_group`: the old assertion
        was ``diff == 0.0  # Only one valid group``, which states the defect as
        the contract. It is now NaN, and the drop is disclosed as a warning as
        well, exactly as the classification metrics disclose theirs.
        """
        rankings = np.array([0, 1, 2, 3, 4, 5])
        groups = np.array(["A", "A", "A", "A", "B", "B"])

        # With min_group_size=3, B should be excluded
        with pytest.warns(UserWarning, match="min_group_size"):
            diff = exposure_parity_difference(rankings, groups, min_group_size=3)
        assert math.isnan(diff)
        assert diff != 0.0

    def test_scores_instead_of_positions(self):
        """Test that scores are converted to positions."""
        # Scores where higher is better
        scores = np.array([100, 90, 80, 70, 60, 50])
        groups = np.array(["A", "A", "A", "B", "B", "B"])

        diff = exposure_parity_difference(scores, groups, min_group_size=2)
        # A has higher scores, should get more exposure
        assert diff > 0


# ---------------------------------------------------------------------------
# The release blocker: ranking metrics returned PERFECT-PARITY sentinels when a
# group was dropped by the size gate.
#
# The incident, measured on this repo: 12 ranked items, group A holds positions
# 0-7, group B (4 items) holds 8-11. With the DEFAULT min_group_size=5, B is
# dropped and every ranking metric reported perfect parity:
#
#   reported : difference 0.0, ratio 1.0, ndkl 0.0, attention 0.0, is_fair True
#   truth    : difference 0.209, ratio 0.576 (a four-fifths FAIL), ndkl 0.578,
#              attention 0.941 against its own threshold of 0.1
#   warnings : []   <- not one, unlike classification
#
# Group B received 12.4 percent of the attention and the module certified the
# ranking as fair. classification.py states the convention these metrics were
# violating; ranking.py now imports its _warn_dropped_groups rather than
# carrying a second copy of it.
# ---------------------------------------------------------------------------

INCIDENT_RANKINGS = np.arange(12)
INCIDENT_GROUPS = np.array(["A"] * 8 + ["B"] * 4)

# The TRUE readings, recoverable by lowering min_group_size to 4.
TRUE_EXPOSURE_DIFFERENCE = 0.20936408399822987
TRUE_EXPOSURE_RATIO = 0.5763430618481035
TRUE_NDKL = 0.5782551230280996
TRUE_ATTENTION_DIFFERENCE = 0.9411945920182281


def _quiet(fn, *args, **kwargs):
    """Run fn with the dropped-group warnings silenced; they are asserted
    separately and the subject of these tests is the VALUE."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


class TestADroppedGroupIsNeverPerfectParity:
    def test_exposure_parity_difference_is_unmeasurable(self):
        got = _quiet(exposure_parity_difference, INCIDENT_RANKINGS, INCIDENT_GROUPS)
        assert math.isnan(got)
        assert got != 0.0, "returned the perfect-difference sentinel 0.0"

    def test_exposure_parity_ratio_is_unmeasurable(self):
        got = _quiet(exposure_parity_ratio, INCIDENT_RANKINGS, INCIDENT_GROUPS)
        assert math.isnan(got)
        assert got != 1.0, "returned the perfect-ratio sentinel 1.0"

    def test_ndkl_is_unmeasurable(self):
        got = _quiet(normalized_discounted_kl_divergence, INCIDENT_RANKINGS, INCIDENT_GROUPS)
        assert math.isnan(got)
        assert got != 0.0

    def test_attention_fairness_withholds_both_the_value_and_the_verdict(self):
        """The worst of the four: it attached a VERDICT to the sentinel, and the
        SVG adapter prints a reported verdict in preference to anything else."""
        result = _quiet(attention_weighted_rank_fairness, INCIDENT_RANKINGS, INCIDENT_GROUPS)

        assert math.isnan(result.value)
        assert result.is_fair is None, "is_fair must be could-not-check, never True"
        assert result.is_fair is not True
        assert result.to_dict()["is_fair"] is None

    def test_a_result_nobody_graded_does_not_default_to_fair(self):
        """The dataclass default itself was the defect in miniature."""
        blank = RankingFairnessResult(
            metric_name="x", value=float("nan"), group_exposures={"A": 0.5}
        )
        assert blank.is_fair is None

    @pytest.mark.parametrize(
        "fn",
        [
            exposure_parity_difference,
            exposure_parity_ratio,
            normalized_discounted_kl_divergence,
            attention_weighted_rank_fairness,
            get_ranking_group_metrics,
        ],
        ids=lambda f: f.__name__,
    )
    def test_every_entry_point_discloses_the_drop_as_a_warning(self, fn):
        """The NaN is the machine-readable signal, the warning is the human one.
        Not one of these five emitted a warning before: the drop was completely
        silent, unlike every classification metric."""
        with pytest.warns(UserWarning, match="min_group_size"):
            fn(INCIDENT_RANKINGS, INCIDENT_GROUPS)


class TestTheTruthIsStillRecoverable:
    """CONTROL A: the fix is not a blanket NaN. Keep the group, get the number.

    These are the values the module SHOULD have reported for the incident data,
    and each is the opposite end of the scale from the sentinel it replaced.
    """

    def test_lowering_the_gate_measures_the_real_exposure_gap(self):
        got = exposure_parity_difference(INCIDENT_RANKINGS, INCIDENT_GROUPS, min_group_size=4)
        assert got == pytest.approx(TRUE_EXPOSURE_DIFFERENCE)

    def test_lowering_the_gate_measures_a_four_fifths_failure(self):
        got = exposure_parity_ratio(INCIDENT_RANKINGS, INCIDENT_GROUPS, min_group_size=4)
        assert got == pytest.approx(TRUE_EXPOSURE_RATIO)
        assert got < 0.8, "the ranking the sentinel certified is a four-fifths FAIL"

    def test_lowering_the_gate_measures_the_real_ndkl(self):
        got = normalized_discounted_kl_divergence(
            INCIDENT_RANKINGS, INCIDENT_GROUPS, min_group_size=4
        )
        assert got == pytest.approx(TRUE_NDKL)

    def test_lowering_the_gate_reports_the_ranking_as_unfair(self):
        result = attention_weighted_rank_fairness(
            INCIDENT_RANKINGS, INCIDENT_GROUPS, min_group_size=4
        )
        assert result.value == pytest.approx(TRUE_ATTENTION_DIFFERENCE)
        assert result.is_fair is False, "measured and failing, not could-not-check"
        # The shut-out group's share of the attention, which the sentinel hid.
        assert result.group_attentions["B"] == pytest.approx(0.12417897955150485)


class TestNoOverCorrection:
    """CONTROL B: fully measured input behaves EXACTLY as before.

    Every number below was produced by the code as it stood BEFORE the fix and
    must be reproduced bit for bit, including the genuine zeros: the convention
    forbids the sentinel where no comparison happened, never a measured 0.0.
    """

    EVEN_RANKINGS = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    EVEN_GROUPS = np.array(["A", "B"] * 4)
    SKEWED_RANKINGS = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    SKEWED_GROUPS = np.array(["A"] * 4 + ["B"] * 4)

    def test_a_measured_run_is_unchanged(self):
        assert exposure_parity_difference(
            self.SKEWED_RANKINGS, self.SKEWED_GROUPS, min_group_size=2
        ) == pytest.approx(0.2924370267958062, abs=1e-15)
        assert exposure_parity_ratio(
            self.SKEWED_RANKINGS, self.SKEWED_GROUPS, min_group_size=2
        ) == pytest.approx(0.5433536754396464, abs=1e-15)
        assert normalized_discounted_kl_divergence(
            self.SKEWED_RANKINGS, self.SKEWED_GROUPS, min_group_size=2
        ) == pytest.approx(0.47394384004311607, abs=1e-15)

    def test_a_measured_verdict_is_still_a_real_boolean(self):
        result = attention_weighted_rank_fairness(
            self.SKEWED_RANKINGS, self.SKEWED_GROUPS, min_group_size=2
        )
        assert result.value == pytest.approx(1.0661410424879547, abs=1e-15)
        assert result.is_fair is False
        assert result.is_fair is not None

        even = attention_weighted_rank_fairness(
            self.EVEN_RANKINGS, self.EVEN_GROUPS, min_group_size=2
        )
        assert even.value == pytest.approx(0.46692947875602286, abs=1e-15)
        assert even.is_fair is False

    def test_a_measured_run_emits_no_dropped_group_warning(self):
        """The warning is for a DROP. Healthy input must stay silent, or every
        healthy run trains its reader to ignore the signal."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exposure_parity_difference(self.EVEN_RANKINGS, self.EVEN_GROUPS, min_group_size=2)
            exposure_parity_ratio(self.EVEN_RANKINGS, self.EVEN_GROUPS, min_group_size=2)
            normalized_discounted_kl_divergence(
                self.EVEN_RANKINGS, self.EVEN_GROUPS, min_group_size=2
            )
            attention_weighted_rank_fairness(self.EVEN_RANKINGS, self.EVEN_GROUPS, min_group_size=2)
            get_ranking_group_metrics(self.EVEN_RANKINGS, self.EVEN_GROUPS, min_group_size=2)

    def test_a_measured_perfect_parity_is_still_reported_as_zero(self):
        """The other direction of the convention: a 0.0 that was actually
        MEASURED must survive, and so must a measured ratio of exactly 1.0.

        Construction: four positions under the LINEAR discount, whose exposures
        are 1.0, 0.75, 0.5, 0.25. Group A holds the outer pair and group B the
        inner pair, so both average 0.625 exactly. This is real parity between
        two groups that were both present and both measured, and it is the case
        the sentinel was indistinguishable from.
        """
        rankings = np.array([0, 1, 2, 3])
        groups = np.array(["A", "B", "B", "A"])

        diff = exposure_parity_difference(
            rankings, groups, min_group_size=2, exposure_type="linear"
        )
        assert diff == 0.0
        assert not math.isnan(diff), "a MEASURED zero must never be turned into NaN"

        ratio = exposure_parity_ratio(rankings, groups, min_group_size=2, exposure_type="linear")
        assert ratio == 1.0
        assert not math.isnan(ratio)
