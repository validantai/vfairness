"""Behavioural tests for :mod:`vfairness.post_processing.ranking.rerank`.

Closes the coverage half of register finding #22 for this module. Before this
file the module sat at 6.8 percent line coverage: only the imports and the
``def`` lines had ever executed, while ``exposure_parity_rerank`` is re-exported
from ``vfairness.__all__`` and is documented as a post-processing intervention
with a *provable* utility budget.

Every test here asserts a value, an ordering or a refusal. The budget claim in
the module docstring ("the produced order is always a valid permutation whose
NDCG respects the utility budget") is checked as an arithmetic property, not as
"the call returned something".
"""

import math

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.ranking import get_ranking_group_metrics
from vfairness.post_processing.ranking.rerank import exposure_parity_rerank

# A ranking whose top five slots are monopolised by group A. Any exposure-aware
# re-ranker has to move at least one B item up to change the parity difference.
SKEWED_SCORES = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.50, 0.49, 0.48, 0.47, 0.46])
SKEWED_GROUPS = np.array(["A"] * 5 + ["B"] * 5)


def _optimal_order(scores):
    return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")]


class TestOrderIsAValidRanking:
    def test_reranked_order_is_a_permutation_of_every_item(self):
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS)
        order = result["reranked_order"]

        assert sorted(order) == list(range(len(SKEWED_SCORES)))
        assert result["n_items"] == len(SKEWED_SCORES)

    def test_group_share_is_the_population_share(self):
        groups = np.array(["A"] * 3 + ["B"] * 7)
        result = exposure_parity_rerank(np.linspace(1.0, 0.1, 10), groups)

        assert result["group_share"] == {"A": 0.3, "B": 0.7}

    def test_the_call_is_deterministic(self):
        first = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS)
        second = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS)

        assert first["reranked_order"] == second["reranked_order"]
        assert first["ndcg_after"] == second["ndcg_after"]


class TestUtilityBudgetIsHonoured:
    """The budget is the module's central promise, so it is checked by value."""

    @pytest.mark.parametrize("budget", [0.02, 0.05, 0.1, 0.3])
    def test_ndcg_after_never_falls_below_the_promised_floor(self, budget):
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=budget)

        floor = (1.0 - budget) * result["ndcg_before"]
        assert result["ndcg_after"] >= floor - 1e-9, (
            f"budget {budget}: ndcg_after {result['ndcg_after']} broke the floor {floor}"
        )

    def test_reported_utility_loss_is_the_ndcg_difference_and_never_negative(self):
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.1)

        assert result["utility_loss"] == pytest.approx(result["ndcg_before"] - result["ndcg_after"])
        assert result["utility_loss"] >= 0.0
        # The reference order is the ideal, so ndcg_before is exactly 1.
        assert result["ndcg_before"] == pytest.approx(1.0)

    def test_a_zero_budget_forbids_any_reordering(self):
        """Negative case: with no utility to spend, parity must NOT improve."""
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.0)

        assert result["reranked_order"] == _optimal_order(SKEWED_SCORES)
        assert result["ndcg_after"] == pytest.approx(1.0)
        assert result["exposure_parity_diff_after"] == pytest.approx(
            result["exposure_parity_diff_before"]
        )

    def test_a_larger_budget_buys_at_least_as_much_parity(self):
        tight = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.01)
        loose = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.2)

        assert loose["exposure_parity_diff_after"] <= tight["exposure_parity_diff_after"] + 1e-9


class TestParityActuallyImproves:
    def test_a_group_starved_ranking_gets_measurably_fairer(self):
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.1)

        before = result["exposure_parity_diff_before"]
        after = result["exposure_parity_diff_after"]
        assert before > 0.25, "fixture no longer starts from a starved ranking"
        assert after < before / 2, f"parity barely moved: {before} to {after}"

    def test_the_starved_group_gains_exposure_and_the_favoured_group_gives_some_up(self):
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.1)

        assert result["exposure_after"]["B"] > result["exposure_before"]["B"]
        assert result["exposure_after"]["A"] < result["exposure_before"]["A"]

    def test_reported_exposure_matches_the_order_that_was_returned(self):
        """The numbers must describe the returned order, not some other one."""
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, max_utility_loss=0.1)

        positions = np.zeros(len(SKEWED_SCORES), dtype=int)
        positions[np.asarray(result["reranked_order"], dtype=int)] = np.arange(len(SKEWED_SCORES))
        recomputed = get_ranking_group_metrics(positions, SKEWED_GROUPS, min_group_size=1)

        for group, reported in result["exposure_after"].items():
            assert reported == pytest.approx(float(recomputed[group]["avg_exposure"]))

    def test_residual_gap_of_an_interleaved_ranking_is_reported_not_hidden(self):
        """Six slots under a log discount cannot reach perfect parity, and the
        result must say so rather than report a zero gap."""
        scores = np.array([0.9, 0.89, 0.88, 0.87, 0.86, 0.85])
        groups = np.array(["A", "B", "A", "B", "A", "B"])

        result = exposure_parity_rerank(scores, groups, max_utility_loss=0.1)

        assert result["exposure_parity_diff_before"] == pytest.approx(0.1563, abs=5e-4)
        assert result["exposure_parity_diff_after"] == pytest.approx(0.0897, abs=5e-4)
        assert result["exposure_parity_diff_after"] > 0.0


class TestDegenerateInputs:
    def test_a_single_group_is_left_in_utility_order(self):
        """REWRITTEN. The last assertion used to be
        ``exposure_parity_diff_after == pytest.approx(0.0)``, which pinned the
        release-blocking ranking defect: with one group there is no pair to
        compare, so ``exposure_parity_difference`` returned the perfect-parity
        sentinel 0.0 and this intervention reported that it had achieved perfect
        exposure parity for a ranking it never even measured. It is NaN now, on
        both sides of the intervention, and the ORDER assertions are what carry
        the behaviour this test is really about.
        """
        result = exposure_parity_rerank(SKEWED_SCORES, np.array(["A"] * 10))

        assert result["reranked_order"] == _optimal_order(SKEWED_SCORES)
        assert result["ndcg_after"] == pytest.approx(1.0)
        assert math.isnan(result["exposure_parity_diff_after"])
        assert math.isnan(result["exposure_parity_diff_before"])
        assert result["exposure_parity_diff_after"] != 0.0

    def test_all_zero_scores_do_not_divide_by_zero(self):
        result = exposure_parity_rerank(np.zeros(6), np.array(["A", "B"] * 3))

        assert result["ndcg_before"] == 1.0
        assert result["ndcg_after"] == 1.0
        assert sorted(result["reranked_order"]) == list(range(6))

    def test_linear_exposure_is_supported_and_changes_the_exposure_scale(self):
        log_result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, exposure_type="log")
        lin_result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, exposure_type="linear")

        assert lin_result["exposure_before"] != log_result["exposure_before"]
        assert sorted(lin_result["reranked_order"]) == list(range(len(SKEWED_SCORES)))


class TestRefusals:
    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS[:3])

    def test_an_unsupported_exposure_discount_is_refused(self):
        with pytest.raises(ValueError, match="log.*linear"):
            exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, exposure_type="quadratic")

    def test_the_geometric_discount_named_in_the_signature_is_accepted(self):
        result = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, exposure_type="geometric")

        assert sorted(result["reranked_order"]) == list(range(len(SKEWED_SCORES)))

    def test_the_geometric_discount_is_computed_and_not_aliased_onto_log(self):
        """Accepting the value is not enough: it has to reach both layers.

        A guard widened to let 'geometric' through while the body still applied
        the log discount would pass the test above and silently report log-decay
        results under a geometric label. The discount is used twice, once to
        REPORT exposure and once to DRIVE the greedy, so both are pinned: an
        alias in either layer alone changes only one of these.
        """
        geo = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, exposure_type="geometric")
        log = exposure_parity_rerank(SKEWED_SCORES, SKEWED_GROUPS, exposure_type="log")

        # Reported exposure: 0.85 ** position, averaged over the five top slots
        # group A holds in the utility-optimal order:
        # (1 + .85 + .7225 + .614125 + .52200625) / 5.
        assert geo["exposure_before"] != log["exposure_before"]
        assert geo["exposure_before"]["A"] == pytest.approx(0.74172625)

        # The greedy itself: a steeper discount buys parity at a different price,
        # so the order and the utility it costs both differ from the log run.
        assert geo["reranked_order"] != log["reranked_order"]
        assert geo["reranked_order"] == [0, 5, 6, 1, 2, 7, 3, 8, 4, 9]
        assert geo["ndcg_after"] == pytest.approx(0.900986, abs=1e-6)
