"""BGL5: the seven grades the BGL4 audit overturned in batches A-in_processing-3
and A-mcp-1, closed and pinned.

Every test here is the CORRECTED behaviour, so it goes red when the defect comes
back. Each one carries the number the defect produced before the fix and the
number it produces now, and each class ends with an over-correction control that
asserts the real measurement on healthy input, because a fix that refuses
everything passes every refusal test and destroys the library.

The seven:

1. ``adversarial_convergence_diagnostics`` set ``oscillation_score = 0.0`` for a
   trajectory with fewer than two usable rounds, and that 0.0 then GATED the
   classification, so "oscillating" was unreachable by construction for any
   one-round game and the verdict said "plateaued" off one point. ``n_rounds=1``
   is a supported call, so the input arrives from the public API.
2. ``sklearn_adversarial_debiasing`` neither refused nor disclosed a three-class
   y, while the adversary saw one of three output columns and every predictor log
   loss was swallowed into a silent NaN by a bare ``except Exception``.
3. ``CounterfactualFairnessLoss.forward`` guarded only at 100 percent
   bit-identity, so a counterfactual identical on 7 of 8 rows reported an
   assessed penalty diluted eightfold; and ``distance_metric='kl'`` returned a
   NEGATIVE penalty, paying the model for counterfactual unfairness.
4. ``IndividualFairnessLoss.forward`` turned the undefined cosine similarity of a
   zero-norm row into the MAXIMUM distance, which exempted every pair touching it
   from the Lipschitz check and scored the batch perfectly fair.
5. ``triage_dataset`` carried the per-ATTRIBUTE coverage across the MCP boundary
   and not the per-COMPARISON coverage, so "Bias audit complete" and
   ``findings_are_a_measurement: True`` came back over a run in which the library
   itself warned that 2 of 5 comparisons had no effect size, one of them the
   caller's own target column.
6. ``detect_proxies(df, [])`` answered "Proxy analysis complete" over zero
   searches, and ``suggest_mitigation(df, [], 'y')`` published
   ``recommendations_are_a_measurement: True`` over zero screens.
7. The recorded over-correction control for the four MCP analysis tools executed
   ZERO body lines of three of them and grepped for two literal phrases, so an
   unconditionally disclosing copy of any of them left the file green.
   ``TestTheControlReachesAllFourTools`` is the control that does the job.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from vfairness.in_processing.diagnostics import (
    adversarial_convergence_diagnostics,
    sklearn_adversarial_debiasing,
)
from vfairness.in_processing.loss_functions.counterfactual import (
    CounterfactualFairnessLoss,
    IndividualFairnessLoss,
)
from vfairness.mcp import tools as T


def _messages(caught) -> list[str]:
    return [str(w.message) for w in caught]


# ===========================================================================
# 1. adversarial_convergence_diagnostics: no successive pair, no oscillation
# ===========================================================================


class TestAOneRoundGameHasNoOscillationToMeasure:
    """MEASURED BEFORE, ``adversarial_convergence_diagnostics([0.6], [0.62], None,
    0.5)``: oscillation_score 0.0, classification "plateau", n_rounds_unusable 0,
    verdict "Adversary accuracy plateaued at 62.0% against a 50.0% majority-class
    baseline (skill 0.24) without dropping to chance. Debiasing stalled; the
    attribute stays partly recoverable", and NOT ONE WARNING.

    0.0 is the perfectly-settled end of the oscillation scale, computed from one
    point, and "plateaued" is a claim about a shape.
    """

    def test_a_single_usable_round_reports_no_oscillation_score(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = adversarial_convergence_diagnostics([0.6], [0.62], None, 0.5)

        assert result["oscillation_score"] is None
        assert result["n_rounds_usable"] == 1
        assert result["n_rounds_unusable"] == 0
        # The skill IS measured from the final round, so it is reported.
        assert result["adversary_skill"] == pytest.approx(0.24)
        assert result["classification"] == "not_assessed"
        assert "NOT ASSESSED" in result["verdict"]
        assert "plateaued" not in result["verdict"]
        assert "Debiasing stalled" not in result["verdict"]
        assert any("only 1 usable round(s)" in m for m in _messages(caught)), _messages(caught)

    def test_converged_and_diverged_stay_reachable_from_one_round(self):
        """The fix must not refuse what one round DOES establish. A final accuracy
        at chance is "converged" and one far above it is "diverged"; both rest on
        the final round alone. Only the in-between band, which is the trajectory
        finding "plateau", is withheld."""
        seen = {}
        for acc in (0.51, 0.62, 0.90, 0.99):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = adversarial_convergence_diagnostics([0.6], [acc], None, 0.5)
            assert r["oscillation_score"] is None, acc
            assert r["classification"] != "plateau", acc
            seen[acc] = r["classification"]
        assert seen == {
            0.51: "converged",
            0.62: "not_assessed",
            0.90: "diverged",
            0.99: "diverged",
        }, seen

    def test_a_trajectory_whose_only_usable_round_is_the_last_says_both_things(self):
        """Three NaN rounds then one usable one. Before: the unusable ROUNDS were
        warned about and the fabricated oscillation_score 0.0 was not."""
        nan = float("nan")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = adversarial_convergence_diagnostics([0.6] * 4, [nan, nan, nan, 0.62], None, 0.5)
        messages = _messages(caught)
        assert r["n_rounds_unusable"] == 3
        assert r["n_rounds_usable"] == 1
        assert r["oscillation_score"] is None
        assert any("carry no usable adversary accuracy" in m for m in messages), messages
        assert any("only 1 usable round(s)" in m for m in messages), messages

    def test_it_is_reachable_end_to_end_because_n_rounds_1_is_allowed(self):
        """``sklearn_adversarial_debiasing`` refuses only ``n_rounds < 1``, so
        ``n_rounds=1`` is a supported call and produces exactly this input. Before:
        oscillation_score 0.0 and no warning anywhere in the chain."""
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 3))
        s = (rng.random(n) < 0.5).astype(int)
        X[:, 0] += 2.0 * s
        y = (X[:, 1] + 0.5 * X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = sklearn_adversarial_debiasing(X, y, s, n_rounds=1)
            diag = adversarial_convergence_diagnostics(
                run["adversary_loss_history"],
                run["adversary_acc_history"],
                run["predictor_loss_history"],
                run["majority_share"],
            )
        assert len(run["adversary_acc_history"]) == 1
        assert diag["oscillation_score"] is None
        assert any("only 1 usable round(s)" in m for m in _messages(caught))

    def test_control_a_real_zero_and_the_four_verdicts_are_unchanged(self):
        """OVER-CORRECTION CONTROL. A ten-round trajectory of identical accuracies
        has NINE successive differences, every one of them genuinely 0.0, and that
        0.0 must survive: it is a measurement of a settled game. All four
        classifications stay reachable with their historical thresholds."""
        cases = (
            ([0.54] * 10, "converged", 0.0),
            ([0.60] * 10, "plateau", 0.0),
            ([0.70] * 10, "diverged", 0.0),
            ([0.5, 0.9, 0.5, 0.9, 0.5, 0.9], "oscillating", 0.4),
        )
        for history, want, want_osc in cases:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = adversarial_convergence_diagnostics([0.6] * len(history), history, None, 0.5)
            assert out["classification"] == want, (history, out["classification"])
            assert out["oscillation_score"] == pytest.approx(want_osc), history
            assert out["n_rounds_usable"] == len(history)
            assert _messages(caught) == [], history
        # Two usable rounds are enough for a successive difference, so the
        # refusal really is about "fewer than two" and not about "few".
        two = adversarial_convergence_diagnostics([0.6, 0.6], [0.62, 0.80], None, 0.5)
        assert two["oscillation_score"] == pytest.approx(0.18)
        assert two["classification"] == "oscillating"


# ===========================================================================
# 2. sklearn_adversarial_debiasing: a non-binary target
# ===========================================================================


class TestANonBinaryTargetIsRefused:
    """MEASURED BEFORE, 600 rows, default_rng(0), x1/x2 standard normal, sensitive
    = (x2 > 0), y = 2 where x2 > 1.0 else 1 where x1 > 0 else 0, n_rounds=5:
    insufficient_data False, not_assessed_reason '', predictor_fitted True,
    predictor_loss_history [nan, nan, nan, nan, nan], warnings NONE, and the
    pipeline verdict "Adversary accuracy is 52.8% against a 52.0% majority-class
    baseline (skill 0.02), so the protected attribute is no longer recoverable
    from the model output beyond chance. Debiasing converged."

    An adversary given the FULL three-column output on the same data reaches 71.0%
    accuracy, skill 0.396, which is above _DIVERGED_SKILL 0.30, i.e. "debiasing
    did not succeed". The adversary in the loop only ever saw
    ``predict_proba(Xs)[:, 1]``.
    """

    @staticmethod
    def _three_class_leak(seed=0, n=600, thr=1.0):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        sensitive = (x2 > 0).astype(int)
        y = np.where(x2 > thr, 2, np.where(x1 > 0, 1, 0))
        return np.column_stack([x1, x2]), y, sensitive

    def test_a_three_class_target_is_refused_not_reported_as_converged(self):
        X, y, sensitive = self._three_class_leak()
        assert len(np.unique(y)) == 3
        with pytest.raises(ValueError, match="y holds 3 classes"):
            sklearn_adversarial_debiasing(X, y, sensitive, n_rounds=5)

    def test_the_refusal_names_what_the_adversary_would_have_measured(self):
        """A refusal a caller cannot act on sends them back to the same input."""
        X, y, sensitive = self._three_class_leak()
        with pytest.raises(ValueError) as exc:
            sklearn_adversarial_debiasing(X, y, sensitive, n_rounds=5)
        text = str(exc.value)
        assert "predict_proba(X)[:, 1]" in text
        assert "one of 3 output columns" in text
        assert "Binarize y" in text

    def test_a_swallowed_log_loss_failure_is_disclosed_instead_of_left_as_nan(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The other half of the same defect: lines 231-232 were a bare
        ``except Exception`` that appended NaN in silence, and that silence was the
        ONLY signal the three-class input was out of contract. With the input class
        refused above, the swallow is exercised through the mechanism itself."""
        import sklearn.metrics as skm

        real = skm.log_loss

        def _explode(y_true, y_prob, **kwargs):
            if list(kwargs.get("labels") or []) == [0, 1]:
                raise ValueError("synthetic log_loss failure")
            return real(y_true, y_prob, **kwargs)

        monkeypatch.setattr(skm, "log_loss", _explode)

        rng = np.random.default_rng(3)
        X = rng.normal(size=(120, 2))
        y = (X[:, 0] > 0).astype(int)
        s = (X[:, 1] > 0).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = sklearn_adversarial_debiasing(X, y, s, n_rounds=3)
        assert all(not math.isfinite(v) for v in run["predictor_loss_history"])
        assert any(
            "could not compute a log loss" in m and "synthetic log_loss failure" in m
            for m in _messages(caught)
        ), _messages(caught)

    def test_control_a_binary_target_still_plays_the_whole_game(self):
        """OVER-CORRECTION CONTROL. The same 600 rows with a BINARY y must still
        fit, still record a finite loss every round and still stay silent."""
        X, _, sensitive = self._three_class_leak()
        y = (X[:, 0] > 0).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = sklearn_adversarial_debiasing(X, y, sensitive, n_rounds=5)
        assert _messages(caught) == []
        assert run["insufficient_data"] is False
        assert run["predictor_fitted"] is True
        assert len(run["predictor_loss_history"]) == 5
        assert all(math.isfinite(v) for v in run["predictor_loss_history"])
        assert run["predictor_loss_history"][0] == pytest.approx(0.08458622866900328, abs=1e-9)
        assert all(math.isfinite(a) for a in run["adversary_acc_history"])


# ===========================================================================
# 3. CounterfactualFairnessLoss.forward: a counterfactual that moved SOME rows
# ===========================================================================


def _cf_batch():
    y_pred = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    y_true = torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0])
    sens = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return y_pred, y_true, sens


class _Sigmoid(torch.nn.Module):
    """The BGL3 fixture's model, so the generated arm keeps its recorded number."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.lin(x)).squeeze(-1)


def _generated_cf_batch(seed: int = 0):
    torch.manual_seed(seed)
    feats = torch.randn(8, 3)
    model = _Sigmoid()
    y_pred = model(feats)
    y_true = torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0])
    sens = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return feats, model, y_pred, y_true, sens


class TestAPartiallyMovedCounterfactualIsDisclosed:
    """MEASURED BEFORE, 8 rows, two groups, lambda_fairness 1.0, a counterfactual
    identical to the factual except row 0 moved by 0.8: fairness_loss
    0.07999999076128006, fairness_penalty_assessed True,
    fairness_unassessable_reason None, no 'fairness_penalty_partial' key, warnings
    NONE. 0.64 is the squared distance of the ONE row that was compared; the seven
    rows nobody compared each contributed a zero distance, the best attainable
    score, and pulled the reported penalty down eightfold.
    """

    def test_one_moved_row_in_eight_measures_the_row_it_compared(self):
        y_pred, y_true, sens = _cf_batch()
        cf = y_pred.clone()
        cf[0] = 0.1  # |d| = 0.8, so d^2 = 0.64

        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, sens, y_pred_counterfactual=cf, return_components=True
            )

        metrics = components.batch_metrics
        assert components.fairness_loss == pytest.approx(0.64, abs=1e-6)
        assert metrics["fairness_penalty_partial"] is True
        assert metrics["fairness_rows_compared"] == 1
        assert metrics["fairness_rows_total"] == 8
        # Still a measurement, of the part that WAS compared.
        assert metrics["fairness_penalty_assessed"] is True
        assert any(
            "7 of 8 row(s) have a counterfactual prediction bit-identical" in m
            for m in _messages(caught)
        ), _messages(caught)

    def test_the_kl_metric_can_no_longer_pay_the_model_for_unfairness(self):
        """MEASURED BEFORE, same batch, distance_metric='kl', counterfactual =
        y_pred * 0.5: fairness_loss -0.1732867956161499, assessed True, total_loss
        0.12571436166763306 BELOW task_loss 0.29900115728378296, warnings NONE. The
        documented quantity is E[|y - y_cf|^2] >= 0 and this file's own comment
        calls 0.0 "the best attainable counterfactual-fairness score", so the run
        reported a score better than the best attainable one."""
        y_pred, y_true, sens = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0, distance_metric="kl")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, sens, y_pred_counterfactual=y_pred * 0.5, return_components=True
            )

        # The full Bernoulli KL of the same input, computed independently here.
        p = (y_pred * 0.5).double()
        q = y_pred.double()
        expected = float(
            (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).mean()
        )
        assert expected > 0.0
        assert components.fairness_loss == pytest.approx(expected, abs=1e-6)
        assert components.fairness_loss == pytest.approx(0.20716696977615356, abs=1e-6)
        assert components.total_loss > components.task_loss
        assert components.batch_metrics["fairness_penalty_assessed"] is True
        assert _messages(caught) == []

    def test_kl_refuses_a_vector_that_is_not_a_probability(self):
        """A Bernoulli KL needs both sides in [0, 1]; outside it the quantity does
        not exist, so it is a could-not-check and not a penalty of any size."""
        y_pred, y_true, sens = _cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0, distance_metric="kl")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, sens, y_pred_counterfactual=y_pred * 3.0, return_components=True
            )
        assert np.isnan(components.fairness_loss)
        assert components.batch_metrics["fairness_unassessable_reason"] == "kl_needs_probabilities"
        assert components.batch_metrics["fairness_loss_unassessed_value"] == pytest.approx(0.0)
        assert any("Bernoulli probabilities" in m for m in _messages(caught))

    def test_control_a_fully_moved_counterfactual_is_untouched(self):
        """OVER-CORRECTION CONTROL, three arms, each with the number the BGL3 pin
        recorded: a supplied counterfactual that moved EVERY row, the l2 reading of
        the kl fixture, and the generated arm."""
        y_pred, y_true, sens = _cf_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, moved = CounterfactualFairnessLoss(lambda_fairness=1.0)(
                y_pred, y_true, sens, y_pred_counterfactual=y_pred + 0.2, return_components=True
            )
        assert _messages(caught) == []
        assert moved.fairness_loss == pytest.approx(0.04, abs=1e-6)
        assert moved.batch_metrics["fairness_penalty_assessed"] is True
        # No row was left behind, so the row mask is not built at all and the
        # healthy path reaches _compute_fairness_penalty exactly as before.
        assert "fairness_penalty_partial" not in moved.batch_metrics
        assert "fairness_rows_compared" not in moved.batch_metrics

        _, l2 = CounterfactualFairnessLoss(lambda_fairness=1.0, distance_metric="l2")(
            y_pred, y_true, sens, y_pred_counterfactual=y_pred * 0.5, return_components=True
        )
        assert l2.fairness_loss == pytest.approx(0.08124999701976776, abs=1e-8)
        _, l1 = CounterfactualFairnessLoss(lambda_fairness=1.0, distance_metric="l1")(
            y_pred, y_true, sens, y_pred_counterfactual=y_pred * 0.5, return_components=True
        )
        assert l1.fairness_loss == pytest.approx(0.25, abs=1e-8)

    def test_control_the_generated_arm_still_measures_its_own_number(self):
        """OVER-CORRECTION CONTROL. The row mask is built from the SUPPLIED
        counterfactual only; the generated arm, on the BGL3 fixture, must still
        report 0.006912893150001764 with no partial flag and no warning."""
        features, model, y_pred, y_true, sens = _generated_cf_batch()
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred,
                y_true,
                sens,
                features=features,
                model=model,
                return_components=True,
            )
        assert _messages(caught) == []
        assert components.batch_metrics["fairness_penalty_assessed"] is True
        assert "fairness_penalty_partial" not in components.batch_metrics
        assert components.fairness_loss == pytest.approx(0.0069128931, abs=1e-7)


# ===========================================================================
# 4. IndividualFairnessLoss.forward: a cosine similarity that does not exist
# ===========================================================================


def _eight_individuals():
    y_pred = torch.tensor([0.95] * 4 + [0.05] * 4)
    y_true = torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0])
    sens = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return y_pred, y_true, sens


class TestCosineOnAZeroNormRowIsNotPerfectFairness:
    """MEASURED BEFORE, the control fixture of this very class (eight individuals
    scored 0.95 four times and 0.05 four times), similarity_metric='cosine', which
    is the option the class docstring's own Example passes:

        features=torch.zeros(8, 3) -> fairness_loss 0.0,
            fairness_penalty_assessed True, fairness_unassessable_reason None,
            n_pairs_compared 28, warnings NONE
        one zero row among seven nonzero ones -> 0.38571426272392273, assessed,
            n_pairs_compared 28, warnings NONE

    The cosine of a zero vector is 0/0. ``F.normalize`` returns the zero vector,
    so the distance becomes 1 - 0 = 1, the MAXIMUM, and with lipschitz_constant
    1.0 and predictions in [0, 1] no prediction gap can exceed that allowance: the
    pair is exempted from the Lipschitz check. The same individuals with nonzero
    features measure 0.514285683631897.
    """

    def test_all_zero_features_are_refused_not_scored_perfect(self):
        y_pred, y_true, sens = _eight_individuals()
        loss_fn = IndividualFairnessLoss(lambda_fairness=1.0, similarity_metric="cosine")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(
                y_pred, y_true, sens, features=torch.zeros(8, 3), return_components=True
            )
        metrics = components.batch_metrics
        assert np.isnan(components.fairness_loss)
        assert metrics["fairness_penalty_assessed"] is False
        assert metrics["fairness_unassessable_reason"] == "feature_similarity_undefined"
        assert metrics["n_pairs_compared"] == 0
        assert metrics["n_rows_without_feature_distance"] == 8
        # The finite 0.0 the optimizer saw is kept, never reported as the penalty.
        assert metrics["fairness_loss_unassessed_value"] == pytest.approx(0.0)
        assert any("cosine similarity is 0/0" in m for m in _messages(caught))

    def test_a_single_zero_row_excludes_its_own_pairs_and_says_so(self):
        """One zero row among seven identical nonzero ones. Its seven pairs have no
        distance, so they leave the criterion instead of being recorded at the
        maximum distance; the 21 pairs that remain measure 12 * 0.9 / 21."""
        y_pred, y_true, sens = _eight_individuals()
        feats = torch.ones(8, 3)
        feats[0] = 0.0
        loss_fn = IndividualFairnessLoss(lambda_fairness=1.0, similarity_metric="cosine")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(y_pred, y_true, sens, features=feats, return_components=True)
        metrics = components.batch_metrics
        assert components.fairness_loss == pytest.approx(12 * 0.9 / 21, abs=1e-5)
        assert metrics["n_pairs_compared"] == 21
        assert metrics["n_rows_without_feature_distance"] == 1
        assert metrics["fairness_penalty_partial"] is True
        assert any("7 pair(s) touching them were left out" in m for m in _messages(caught))

    def test_the_nearest_neighbour_branch_excludes_them_too(self):
        """A branch the BGL3 pin never reached (counterfactual.py:706-718 was
        unexecuted). An undefined row must not be ranked as the most distant
        neighbour of everybody; it leaves every neighbourhood, and the divisor is
        the pairs actually summed."""
        y_pred, y_true, sens = _eight_individuals()
        feats = torch.ones(8, 3)
        feats[0] = 0.0
        loss_fn = IndividualFairnessLoss(
            lambda_fairness=1.0, similarity_metric="cosine", n_neighbors=2
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(y_pred, y_true, sens, features=feats, return_components=True)
        metrics = components.batch_metrics
        # 7 rows with a distance, 2 neighbours each.
        assert metrics["n_pairs_compared"] == 14
        assert metrics["fairness_penalty_partial"] is True
        assert math.isfinite(components.fairness_loss)
        assert any("cosine similarity is 0/0" in m for m in _messages(caught))

    def test_control_the_measured_violation_is_unchanged_on_every_metric(self):
        """OVER-CORRECTION CONTROL, against the hand computation the BGL3 pin uses:
        16 cross pairs at 0.9 and 12 same pairs at 0.0 over 28 upper-triangle pairs
        = 14.4 / 28 = 0.5142857. Identical under cosine, euclidean and a custom
        metric, all silent, all 28 pairs."""
        y_pred, y_true, sens = _eight_individuals()
        for metric, kwargs in (
            ("cosine", {}),
            ("euclidean", {}),
            ("custom", {"custom_similarity_fn": lambda x: torch.cdist(x, x)}),
        ):
            loss_fn = IndividualFairnessLoss(
                lambda_fairness=1.0, similarity_metric=metric, **kwargs
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, components = loss_fn(
                    y_pred, y_true, sens, features=torch.ones(8, 3), return_components=True
                )
            assert components.fairness_loss == pytest.approx(16 * 0.9 / 28, abs=1e-5), metric
            metrics = components.batch_metrics
            assert metrics["fairness_penalty_assessed"] is True, metric
            assert metrics["n_pairs_compared"] == 28, metric
            assert "fairness_penalty_partial" not in metrics, metric
            assert _messages(caught) == [], metric

    def test_control_the_neighbour_branch_on_healthy_features_is_unchanged(self):
        """OVER-CORRECTION CONTROL for the divisor change: with every distance
        defined, the pairs summed ARE n * n_neighbors, so the penalty is the same
        number the old code produced and nothing is flagged."""
        y_pred, y_true, sens = _eight_individuals()
        torch.manual_seed(1)
        feats = torch.randn(8, 3)
        loss_fn = IndividualFairnessLoss(lambda_fairness=1.0, n_neighbors=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, components = loss_fn(y_pred, y_true, sens, features=feats, return_components=True)
        metrics = components.batch_metrics
        assert metrics["n_pairs_compared"] == 8 * 2
        assert metrics["fairness_penalty_assessed"] is True
        assert "fairness_penalty_partial" not in metrics
        assert _messages(caught) == []

    def test_control_the_three_older_refusals_still_fire(self):
        """OVER-CORRECTION CONTROL. The new guard must not shadow the ones that
        were already there."""
        y_pred, y_true, sens = _eight_individuals()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, no_features = IndividualFairnessLoss(lambda_fairness=1.0)(
                y_pred, y_true, sens, return_components=True
            )
            _, one_row = IndividualFairnessLoss(lambda_fairness=1.0)(
                y_pred[:1], y_true[:1], sens[:1], features=torch.ones(1, 3), return_components=True
            )
            reassigned = IndividualFairnessLoss(lambda_fairness=1.0)
            reassigned.n_neighbors = 0
            _, no_pairs = reassigned(
                y_pred, y_true, sens, features=torch.ones(8, 3), return_components=True
            )
        assert no_features.batch_metrics["fairness_unassessable_reason"] == "features_not_provided"
        assert one_row.batch_metrics["fairness_unassessable_reason"] == "fewer_than_two_samples"
        assert no_pairs.batch_metrics["fairness_unassessable_reason"] == "no_pairs_compared"


# ===========================================================================
# 5. triage_dataset: the per-COMPARISON coverage crosses the boundary
# ===========================================================================

_N = 60


def _outcome_column_is_constant() -> pd.DataFrame:
    """Two real groups of 30, real feature spread, and an outcome nobody varies:
    a model that approves everybody. The feature distributions are measurable and
    the outcome disparity the caller asked about is not."""
    return pd.DataFrame(
        {
            "race": ["a"] * 30 + ["b"] * 30,
            "f1": np.linspace(0, 1, _N),
            "f2": np.arange(_N) * 1.0,
            "score": np.linspace(0, 1, _N),
            "y": [1] * _N,
            "pred": [1] * _N,
        }
    )


def _fully_measurable() -> pd.DataFrame:
    """400 rows, two groups of 200, discrete features with a real disparity.

    Every comparison the audit attempts on this frame succeeds, which is what
    makes it a control: a tool that disclosed unconditionally would fail here.
    """
    rng = np.random.RandomState(0)
    n = 400
    race = np.array(["White"] * 200 + ["Black"] * 200)
    tier = np.where(race == "White", rng.binomial(1, 0.7, n), rng.binomial(1, 0.3, n))
    region = rng.randint(0, 3, n)
    y = np.where(tier == 1, rng.binomial(1, 0.8, n), rng.binomial(1, 0.3, n))
    return pd.DataFrame(
        {
            "race": race,
            "tier": tier,
            "region": region,
            "y": y,
            "pred": y,
            "score": np.where(y == 1, 0.8, 0.2),
        }
    )


class TestAnUnmeasuredComparisonReachesTheMcpClient:
    """MEASURED BEFORE, ``triage_dataset(constant_outcome_frame, ['race'], 'y')``:
    summary "Bias audit complete: 4 finding(s) across 1 protected attribute(s).
    Outcome column: 'y', as named by the caller.", coverage
    {'assessment_coverage': 'complete', 'execution_coverage': 'complete',
    'attributes_not_assessed': [], 'findings_are_a_measurement': True}, and the
    tokens 'not_measurable', 'UNMEASURED' and 'could not' occurring ZERO times
    anywhere in the returned dict, while the library warned "2 of 5 comparison(s)
    have an UNMEASURED effect size (effect_size is NaN, effect_interpretation is
    not_measurable) ... This is not a small effect and must not be read as one".

    The two unmeasured comparisons were y and pred, one of them the outcome column
    the caller named and this tool echoes back, and both were absent from
    disparity_findings rather than listed as unassessed.
    """

    def test_the_payload_carries_the_librarys_own_sentence(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.triage_dataset(_outcome_column_is_constant(), ["race"], "y")
        coverage = out["coverage"]
        assert coverage["comparison_coverage"] == "partial"
        assert coverage["findings_are_a_measurement"] is False
        joined = " | ".join(coverage["could_not_check"])
        assert "UNMEASURED effect size" in joined
        assert "2 of 5 comparison(s)" in joined
        assert "must not be read as one" in joined
        # The per-ATTRIBUTE answer is kept beside it rather than replaced by it.
        assert coverage["assessment_coverage"] == "complete"
        assert coverage["attributes_not_assessed"] == []

    def test_the_summary_an_agent_reads_first_says_it_too(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.triage_dataset(_outcome_column_is_constant(), ["race"], "y")
        assert "COULD NOT measure" in out["summary"]
        assert "coverage.could_not_check" in out["summary"]

    def test_capturing_the_warnings_does_not_swallow_them(self):
        """The disclosure is carried across the boundary, not moved there: a caller
        who does watch warnings must still see exactly what the library raised."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            T.triage_dataset(_outcome_column_is_constant(), ["race"], "y")
        assert any("UNMEASURED effect size" in m for m in _messages(caught)), _messages(caught)

    def test_control_a_fully_measurable_frame_still_reports_a_complete_audit(self):
        """OVER-CORRECTION CONTROL, with the real numbers. If this reads as
        anything less than a complete measurement, the disclosure above is noise
        and the field has stopped discriminating."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.triage_dataset(_fully_measurable(), ["race"], "y")
        coverage = out["coverage"]
        assert coverage["comparison_coverage"] == "complete"
        assert coverage["n_could_not_check"] == 0
        assert coverage["could_not_check"] == []
        assert coverage["findings_are_a_measurement"] is True
        assert out["summary"].startswith("Bias audit complete: 6 finding(s)")
        assert "COULD NOT" not in out["summary"]
        assert out["finding_counts"]["disparity_findings"] == 4
        assert out["finding_counts"]["proxy_findings"] == 1


# ===========================================================================
# 6. Zero searches and zero screens are not complete ones
# ===========================================================================


class TestZeroSearchesIsNotACompleteSearch:
    """MEASURED BEFORE, on a healthy frame:

        detect_proxies(df, [])        -> "Proxy analysis complete: 0 proxy
            chain(s) across 0 protected attribute(s)." with
            proxy_chain_coverage {}
        suggest_mitigation(df, [], 'y') -> "Suggested 0 pre-processing
            mitigation(s) from feature analysis across 0 protected
            attribute(s)." with coverage {'feature_screen_complete': True,
            'n_screens_not_run': 0, 'recommendations_are_a_measurement': True}

    With no attribute the per-attribute loop never runs, so nothing can land in
    the incomplete list and the complete branch fires over zero work. The sibling
    triage_dataset refuses the identical input, so two surfaces in one file
    disagreed about it.
    """

    def test_detect_proxies_with_no_attribute_says_it_assessed_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.detect_proxies(_fully_measurable(), [])
        assert "ASSESSED NOTHING" in out["summary"]
        assert "not evidence that no proxy chain exists" in out["summary"]
        assert out["coverage"]["chains_are_a_measurement"] is False
        assert out["coverage"]["n_attributes_searched"] == 0

    def test_suggest_mitigation_with_no_attribute_is_not_a_measurement(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.suggest_mitigation(_fully_measurable(), [], "y")
        assert "ASSESSED NOTHING" in out["summary"]
        assert out["coverage"]["recommendations_are_a_measurement"] is False
        assert out["coverage"]["n_attributes_screened"] == 0

    def test_control_one_real_attribute_still_searches_and_screens(self):
        """OVER-CORRECTION CONTROL with the measured values: 3 proxy chains and 4
        mitigations on the same frame, both reported as measurements."""
        df = _fully_measurable()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proxies = T.detect_proxies(df, ["race"])
            mitigation = T.suggest_mitigation(df, ["race"], "y", "score")
        assert proxies["summary"] == (
            "Proxy analysis complete: 3 proxy chain(s) across 1 protected attribute(s)."
        )
        assert proxies["coverage"]["chains_are_a_measurement"] is True
        assert len(proxies["proxy_chains"]["race"]) == 3
        assert mitigation["summary"] == (
            "Suggested 4 pre-processing mitigation(s) from feature analysis across "
            "1 protected attribute(s)."
        )
        assert mitigation["coverage"]["recommendations_are_a_measurement"] is True
        assert mitigation["coverage"]["n_attributes_screened"] == 1


# ===========================================================================
# 7. The over-correction control that actually reaches all four tools
# ===========================================================================

# Every phrase any of these four tools uses to say it could not measure
# something. The control below asserts that NONE of them appears on a frame where
# everything IS measurable, which is what makes an unconditionally disclosing copy
# of ANY of the four fail. The recorded control for three of these tools never
# called them at all, and the one it did call was grepped for two phrases only
# ("assessed nothing", "not a clean bill"), so an unconditional discloser passed
# it and the whole file stayed at 14 passed.
_CRIES_WOLF = (
    "assessed nothing",
    "not a clean bill",
    "incomplete",
    "could not",
    "not measured",
    "unmeasured",
    "unassessed",
    "not evidence",
    "unrecorded",
    "absence",
)


class TestTheControlReachesAllFourTools:
    """The recorded sabotage for detect_proxies, suggest_mitigation and
    explain_decision named a test that executes ZERO body lines of them
    (``test_an_mcp_tool_on_measurable_data_does_not_cry_wolf`` calls only
    triage_dataset). Measured by the BGL4 audit: an unconditionally disclosing copy
    of each of the four left the whole file at 14 passed.
    """

    @staticmethod
    def _call(tool: str):
        df = _fully_measurable()
        args = {
            "triage_dataset": (df, ["race"], "y"),
            "detect_proxies": (df, ["race"]),
            "suggest_mitigation": (df, ["race"], "y", "score"),
            "explain_decision": (df, "pred", ["tier", "region"]),
        }[tool]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(T, tool)(*args)

    @pytest.mark.parametrize(
        "tool",
        ["triage_dataset", "detect_proxies", "suggest_mitigation", "explain_decision"],
    )
    def test_no_tool_cries_wolf_on_a_fully_measurable_frame(self, tool):
        out = self._call(tool)
        summary = out["summary"].lower()
        said = [phrase for phrase in _CRIES_WOLF if phrase in summary]
        assert not said, f"{tool} disclosed {said} on measurable data: {out['summary']!r}"

    @pytest.mark.parametrize(
        "tool,expected",
        [
            ("triage_dataset", "Bias audit complete: 6 finding(s)"),
            ("detect_proxies", "Proxy analysis complete: 3 proxy chain(s)"),
            ("suggest_mitigation", "Suggested 4 pre-processing mitigation(s)"),
            ("explain_decision", "Top drivers of 'pred': tier (0.41), region (-0.05)."),
        ],
    )
    def test_each_tool_is_actually_reached_and_measures_something(self, tool, expected):
        """A control that does not execute the unit proves nothing about it, so
        each tool is pinned to a value only IT can produce."""
        assert expected in self._call(tool)["summary"]

    def test_explain_decision_still_names_what_it_could_not_assess(self):
        """The re-pointed pin for explain_decision, whose graded evidence cited a
        test id that does not exist (pytest: "ERROR: not found", 0 items) plus one
        that executes 0 body lines of the unit. This half of the pair is the
        refusal; the parametrised control above is the other half."""
        constant = _outcome_column_is_constant()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.explain_decision(constant, "pred", ["f1", "f2"])
        assert "could NOT be assessed" in out["summary"]
        assert {f["feature"] for f in out["features_not_measured"]} >= {"f1", "f2"}
        assert out["drivers"] == []

    def test_explain_decision_measures_the_features_it_can(self):
        """OVER-CORRECTION CONTROL for the test above, with the real numbers."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = T.explain_decision(_fully_measurable(), "pred", ["tier", "region"])
        assert out["features_not_measured"] == []
        assert out["drivers"][0]["feature"] == "tier"
        assert out["drivers"][0]["association"] == pytest.approx(0.414474, abs=1e-5)


def test_the_disclosure_survives_the_real_mcp_boundary():
    """A correct field that a serialiser drops is still a defect, one layer up, so
    the fix is followed to the surface a client actually reads: the live FastMCP
    tool call, not the python function.

    Measured through ``await mcp.call_tool('triage_dataset', ...)`` on the same
    constant-outcome frame: the text content carries 'UNMEASURED effect size',
    '2 of 5 comparison(s)' and 'COULD NOT measure', and the structured payload's
    result.coverage carries comparison_coverage 'partial', n_could_not_check 5 and
    findings_are_a_measurement False. Before the fix the whole payload contained
    none of those tokens.
    """
    pytest.importorskip("mcp.server.fastmcp")
    import asyncio

    csv = _outcome_column_is_constant().to_csv(index=False)

    async def _call():
        from vfairness.mcp.server import mcp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return await mcp.call_tool(
                "triage_dataset",
                {"protected_attributes": ["race"], "target_column": "y", "csv": csv},
            )

    result = asyncio.run(_call())
    content, structured = result if isinstance(result, tuple) else (result, None)
    text = content[0].text if content else ""
    assert "UNMEASURED effect size" in text
    assert "COULD NOT measure" in text
    assert isinstance(structured, dict)
    coverage = structured["result"]["coverage"]
    assert coverage["comparison_coverage"] == "partial"
    assert coverage["findings_are_a_measurement"] is False
    assert coverage["n_could_not_check"] >= 1


# ===========================================================================
# 8. server.main holds no measurement
# ===========================================================================


def test_server_main_takes_no_input_and_returns_no_value():
    """The A-mcp-1 grade for ``vfairness.mcp.server.main`` was SEMI-PROVEN, i.e.
    "the evidence does not cover enough input to settle it". There is no input:
    main() takes no parameter, holds no quantity and returns None, so no input
    class exists to sweep and no value can be wrong. Corrected grade: NOT A
    MEASUREMENT. This records that in executable form, so the day main() grows a
    parameter or a return value the grade is asked again."""
    pytest.importorskip("mcp.server.fastmcp")
    from vfairness.mcp import server

    signature = inspect.signature(server.main)
    assert list(signature.parameters) == []
    assert signature.return_annotation in (None, "None")
    source = inspect.getsource(server.main)
    assert "return" not in source
