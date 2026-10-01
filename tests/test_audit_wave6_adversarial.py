"""
Audit wave 6: adversarial unit pins.

Pins the fixes for three second-iteration audit findings (2026-08-22):

1. CRITICAL, in_processing/loss_functions/adversarial.py:
   AdversarialDebiasingLoss with use_gradient_reversal=False returned the
   adversary's loss with a POSITIVE sign, so gradient descent on
   task + lambda * penalty trained the predictor to HELP the adversary
   (bias amplification). Fixed: the non-GRL penalty is the NEGATED
   adversary loss against a frozen adversary (min L_task - lambda *
   L_adversary, Zhang et al. 2018).

2. HIGH, same file: the gradient reversal layer was constructed with scale
   lambda_fairness while BaseFairnessLoss.forward multiplies the penalty by
   lambda again, so the predictor saw lambda squared (default 0.1 became
   0.01); and the docstring claimed a single backward pass trains the
   adversary while nothing ever stepped it. Fixed: reversal scale 1.0
   (lambda applied exactly once) and update_adversary documented as
   required in both modes. FairRepresentationLoss carried the identical
   double-scaling and is fixed the same way.

3. HIGH, in_processing/diagnostics.py: adversarial_convergence_diagnostics
   classified on raw adversary accuracy with fixed thresholds (0.55/0.65),
   which is base-rate blind: a majority-class adversary scores the majority
   share with zero information, so any split more imbalanced than 65/35
   could never read "converged" (false FAIL). Fixed: classification on
   skill = (acc - p) / (1 - p) over the majority-share baseline p, with the
   balanced case (p = 0.5) numerically unchanged. The sklearn debiasing
   loop's reweighting also UPWEIGHTED attribute-recoverable samples
   (sharpening the leak instead of suppressing it); fixed to downweight.

Negative cases (the exact scenarios that used to produce the wrong result)
come first in each block; does-not-overcorrect cases follow.
"""

import numpy as np
import pytest

from vfairness.in_processing.diagnostics import (
    adversarial_convergence_diagnostics,
    sklearn_adversarial_debiasing,
)

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Shared torch helpers (only referenced inside @needs_torch tests)
# ---------------------------------------------------------------------------


def _s_informative_setup(seed=0, n=100, pretrain_steps=300):
    """Leaf logits whose sigmoid leaks the group, plus a pretrained adversary.

    Returns (loss_fn factory args are left to the caller) the tensors and a
    function that pretrains a given loss_fn's adversary on the detached
    predictions so its loss surface is informative for sign tests.
    """
    torch.manual_seed(seed)
    s = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    logits = (torch.randn(n) * 0.5 + 1.5 * s.float()).clone().requires_grad_(True)
    y_true = (torch.rand(n) < 0.5).float()

    def pretrain(loss_fn):
        preds = torch.sigmoid(logits).detach()
        for _ in range(pretrain_steps):
            loss_fn.update_adversary(preds, s)

    return logits, y_true, s, pretrain


def _adversary_bce(loss_fn, y_pred, s):
    """Adversary BCE on given predictions with the adversary held fixed."""
    with torch.no_grad():
        inp = y_pred.unsqueeze(1) if y_pred.dim() == 1 else y_pred
        out = loss_fn.adversary(inp)
        return torch.nn.functional.binary_cross_entropy(out.squeeze(), s.float()).item()


# ---------------------------------------------------------------------------
# Finding 1 (CRITICAL): non-GRL sign inversion
# ---------------------------------------------------------------------------


@needs_torch
class TestNonGrlSign:
    def test_descent_step_increases_adversary_loss(self):
        """NEGATIVE CASE. Old code: a descent step on the fairness term
        DECREASED the adversary's loss through y_pred (predictor trained to
        help the adversary). Fixed code must INCREASE it."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        logits, y_true, s, pretrain = _s_informative_setup()
        lf = AdversarialDebiasingLoss(
            lambda_fairness=1.0,
            use_gradient_reversal=False,
            adversary_hidden_dims=[8],
            track_metrics=False,
        )
        pretrain(lf)

        y_pred = torch.sigmoid(logits)
        before = _adversary_bce(lf, y_pred.detach(), s)

        # Exactly forward's fairness term: effective_lambda * penalty.
        pen = lf._compute_fairness_penalty(y_pred, y_true, s)
        (lf.lambda_fairness * pen).backward()
        with torch.no_grad():
            stepped = torch.sigmoid(logits - 0.5 * logits.grad)
        after = _adversary_bce(lf, stepped, s)

        assert after > before, (
            "descent on the fairness term must push predictions AWAY from "
            "the adversary (adv BCE up), got %.6f -> %.6f" % (before, after)
        )

    def test_penalty_leaves_adversary_weights_untouched(self):
        """The predictor update plays against a FIXED adversary: the negated
        penalty must not send (anti-training) gradients into the adversary's
        own weights, and requires_grad must be restored afterwards."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        logits, y_true, s, pretrain = _s_informative_setup()
        lf = AdversarialDebiasingLoss(
            lambda_fairness=1.0,
            use_gradient_reversal=False,
            adversary_hidden_dims=[8],
            track_metrics=False,
        )
        pretrain(lf)
        # update_adversary leaves its own training grads behind; clear them so
        # any grad seen below can only come from the penalty path.
        lf.adversary_optimizer.zero_grad()
        pen = lf._compute_fairness_penalty(torch.sigmoid(logits), y_true, s)
        pen.backward()

        for p in lf.adversary.parameters():
            assert p.requires_grad, "requires_grad must be restored after the penalty"
            assert p.grad is None or torch.all(p.grad == 0), (
                "the frozen-adversary penalty must not write gradients into the adversary's weights"
            )

    def test_end_to_end_adversary_accuracy_drops_toward_base_rate(self):
        """NEGATIVE CASE, end to end. Old code amplified the leak (audit:
        demographic gap +0.2878 -> +0.3445). With debiasing on, a fresh probe
        adversary's accuracy on the protected attribute must DROP toward the
        base rate versus off, and the demographic gap must shrink.

        Executed configuration (torch 2.8.0 scratch env, 2026-08-22): probe
        accuracy 0.5950 (off) -> 0.5517 (lambda=2), gap 0.2031 -> 0.1196; the
        pre-fix code left the probe unchanged and grew the gap."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        torch.manual_seed(0)
        n = 600
        s = (torch.rand(n) < 0.5).long()
        x1 = torch.randn(n)
        x2 = s.float() + 0.6 * torch.randn(n)  # proxy feature for s
        y = ((x1 + 0.9 * s.float() + 0.4 * torch.randn(n)) > 0.45).float()
        X = torch.stack([x1, x2], dim=1)

        def train(lam):
            torch.manual_seed(1)
            model = nn.Linear(2, 1)
            lf = AdversarialDebiasingLoss(
                lambda_fairness=lam,
                use_gradient_reversal=False,
                adversary_hidden_dims=[16],
                n_adversary_steps=2,
                track_metrics=False,
            )
            opt = torch.optim.Adam(model.parameters(), lr=0.05)
            for _ in range(200):
                opt.zero_grad()
                y_pred = torch.sigmoid(model(X).squeeze(1))
                loss = lf(y_pred, y, s)
                loss.backward()
                opt.step()
                # Documented usage: update the adversary every iteration.
                lf.update_adversary(y_pred.detach(), s)
            with torch.no_grad():
                return torch.sigmoid(model(X).squeeze(1))

        def probe_acc(preds, seed=5):
            # Fresh torch probe adversary trained on the final predictions:
            # measures how recoverable the attribute remains.
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
            opt = torch.optim.Adam(net.parameters(), lr=0.02)
            x = preds.detach().unsqueeze(1)
            for _ in range(500):
                opt.zero_grad()
                out = net(x).squeeze(1)
                nn.functional.binary_cross_entropy_with_logits(out, s.float()).backward()
                opt.step()
            with torch.no_grad():
                return ((net(x).squeeze(1) > 0).long() == s).float().mean().item()

        def dp_gap(preds):
            hard = (preds > 0.5).float()
            return abs(float(hard[s == 1].mean()) - float(hard[s == 0].mean()))

        base_rate = max(float(s.float().mean()), 1.0 - float(s.float().mean()))
        preds_off = train(0.0)
        preds_on = train(2.0)

        acc_off = probe_acc(preds_off)
        acc_on = probe_acc(preds_on)

        assert acc_on < acc_off - 0.02, (
            "debiasing must reduce attribute recoverability: probe acc "
            "%.4f (on) vs %.4f (off)" % (acc_on, acc_off)
        )
        assert abs(acc_on - base_rate) < abs(acc_off - base_rate), (
            "probe accuracy must move toward the base rate %.4f" % base_rate
        )
        assert dp_gap(preds_on) < dp_gap(preds_off), (
            "debiasing must shrink the demographic gap: %.4f (on) vs "
            "%.4f (off)" % (dp_gap(preds_on), dp_gap(preds_off))
        )


# ---------------------------------------------------------------------------
# Finding 2 (HIGH): GRL double-scaled lambda; adversary trainable as documented
# ---------------------------------------------------------------------------


@needs_torch
class TestGrlScaleAndAdversaryTraining:
    def _shared_adversary_losses(self, lambdas):
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        fns = []
        ref = None
        for lam in lambdas:
            lf = AdversarialDebiasingLoss(
                lambda_fairness=lam,
                use_gradient_reversal=True,
                adversary_hidden_dims=[8],
                track_metrics=False,
            )
            if ref is None:
                ref = lf.adversary.state_dict()
            else:
                lf.adversary.load_state_dict(ref)
            fns.append(lf)
        return fns

    def test_lambda_is_applied_exactly_once(self):
        """NEGATIVE CASE. Old code scaled the reversed gradient by lambda in
        the GRL AND multiplied the penalty by lambda in forward, so the
        fairness gradient was quadratic in lambda (0.5 -> 0.25x). It must be
        linear: the fairness gradient at lambda=0.5 is 0.5x the one at 1.0."""
        logits, y_true, s, pretrain = _s_informative_setup()
        lf1, lf05, lf0 = self._shared_adversary_losses([1.0, 0.5, 0.0])
        pretrain(lf1)
        lf05.adversary.load_state_dict(lf1.adversary.state_dict())
        lf0.adversary.load_state_dict(lf1.adversary.state_dict())

        def grad_of(lf):
            y_pred = torch.sigmoid(logits)
            (g,) = torch.autograd.grad(lf(y_pred, y_true, s), logits)
            return g

        g_task = grad_of(lf0)  # lambda=0: task gradient only
        fair_1 = grad_of(lf1) - g_task
        fair_05 = grad_of(lf05) - g_task

        assert fair_1.abs().max().item() > 1e-6, "fairness gradient must be nonzero"
        assert torch.allclose(fair_05, 0.5 * fair_1, rtol=1e-4, atol=1e-8), (
            "fairness gradient must scale LINEARLY with lambda "
            "(lambda applied exactly once, not lambda squared)"
        )

    def test_grl_descent_step_still_increases_adversary_loss(self):
        """Does-not-overcorrect: the GRL path's predictor gradient stays
        reversed (a descent step must still push the adversary's loss UP)."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        logits, y_true, s, pretrain = _s_informative_setup()
        lf = AdversarialDebiasingLoss(
            lambda_fairness=1.0,
            use_gradient_reversal=True,
            adversary_hidden_dims=[8],
            track_metrics=False,
        )
        pretrain(lf)

        y_pred = torch.sigmoid(logits)
        before = _adversary_bce(lf, y_pred.detach(), s)
        pen = lf._compute_fairness_penalty(y_pred, y_true, s)
        (lf.lambda_fairness * pen).backward()
        with torch.no_grad():
            stepped = torch.sigmoid(logits - 0.5 * logits.grad)
        after = _adversary_bce(lf, stepped, s)

        assert after > before

    def test_shared_backward_gives_training_gradient_for_adversary(self):
        """The (corrected) docstring promise: in GRL mode the shared backward
        computes correctly-signed gradients on the adversary's weights, so
        stepping them reduces the adversary's loss."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        logits, y_true, s, _ = _s_informative_setup()
        lf = AdversarialDebiasingLoss(
            lambda_fairness=1.0,
            use_gradient_reversal=True,
            adversary_hidden_dims=[8],
            track_metrics=False,
        )
        y_pred = torch.sigmoid(logits)
        before = _adversary_bce(lf, y_pred.detach(), s)
        loss = lf(y_pred, y_true, s)
        loss.backward()
        with torch.no_grad():
            for p in lf.adversary.parameters():
                if p.grad is not None:
                    p -= 0.05 * p.grad
        after = _adversary_bce(lf, y_pred.detach(), s)

        assert after < before, (
            "stepping adversary weights along -grad from the shared backward "
            "must train the adversary (BCE down), got %.6f -> %.6f" % (before, after)
        )

    def test_update_adversary_trains_in_both_modes(self):
        """Documented usage in both modes: update_adversary reduces the
        adversary's loss on fixed, attribute-informative predictions."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
        )

        for grl in (True, False):
            logits, _, s, _ = _s_informative_setup(seed=3)
            lf = AdversarialDebiasingLoss(
                lambda_fairness=1.0,
                use_gradient_reversal=grl,
                adversary_hidden_dims=[8],
                track_metrics=False,
            )
            preds = torch.sigmoid(logits).detach()
            first = lf.update_adversary(preds, s)
            for _ in range(200):
                last = lf.update_adversary(preds, s)
            assert last < first, "update_adversary must train the adversary (grl=%s)" % grl

    def test_reversal_scale_is_one_in_both_classes(self):
        """The GRL scale must be 1.0 so lambda is applied exactly once by the
        forward that multiplies the penalty (AdversarialDebiasingLoss and the
        same-family FairRepresentationLoss)."""
        from vfairness.in_processing.loss_functions.adversarial import (
            AdversarialDebiasingLoss,
            FairRepresentationLoss,
        )

        lf = AdversarialDebiasingLoss(lambda_fairness=0.3, use_gradient_reversal=True)
        assert lf.gradient_reversal.lambda_ == 1.0
        fr = FairRepresentationLoss(lambda_fairness=0.3)
        assert fr.gradient_reversal.lambda_ == 1.0


# ---------------------------------------------------------------------------
# Finding 3 (HIGH): base-rate-blind convergence verdict; reweighting direction
# ---------------------------------------------------------------------------


class TestConvergenceVerdictBaseRate:
    def _independent_imbalanced(self, n=4000, seed=7):
        rng = np.random.default_rng(seed)
        s = (rng.random(n) < 0.25).astype(int)  # 75/25 split
        X = rng.normal(size=(n, 3))
        # y depends only on X: the attribute is entirely unrecoverable.
        y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
        return X, y, s

    def test_unrecoverable_attribute_is_not_a_false_fail(self):
        """NEGATIVE CASE. 75/25 split, X and y independent of the attribute:
        the adversary sits exactly at the majority share (zero information).
        Old code classified this 'diverged' and reported 'debiasing did not
        succeed'. It must read converged."""
        X, y, s = self._independent_imbalanced()
        res = sklearn_adversarial_debiasing(X, y, s, n_rounds=5)
        diag = adversarial_convergence_diagnostics(
            res["adversary_loss_history"],
            res["adversary_acc_history"],
            res["predictor_loss_history"],
            majority_share=res["majority_share"],
        )
        assert res["majority_share"] > 0.65, "scenario must be more imbalanced than 65/35"
        assert diag["classification"] == "converged"
        assert diag["adversary_skill"] < 0.05
        assert "did not succeed" not in diag["verdict"]

    def test_recoverable_attribute_still_flagged(self):
        """Does-not-overcorrect: balanced groups with a genuinely recoverable
        attribute must still be flagged as not converged."""
        rng = np.random.default_rng(7)
        n = 3000
        s = (rng.random(n) < 0.5).astype(int)
        x = rng.normal(size=n) + s  # x ~ N(s, 1): attribute recoverable
        y = (x + 0.5 * rng.normal(size=n) > 0.5).astype(int)
        res = sklearn_adversarial_debiasing(x.reshape(-1, 1), y, s, n_rounds=10)
        diag = adversarial_convergence_diagnostics(
            res["adversary_loss_history"],
            res["adversary_acc_history"],
            majority_share=res["majority_share"],
        )
        assert diag["classification"] in ("diverged", "plateau", "oscillating")
        assert diag["adversary_skill"] > 0.30

    def test_balanced_default_thresholds_unchanged(self):
        """Without majority_share the historical balanced-baseline thresholds
        apply unchanged (0.55 converged / 0.65 diverged on raw accuracy)."""
        cases = [([0.54] * 10, "converged"), ([0.70] * 10, "diverged"), ([0.60] * 10, "plateau")]
        for hist, want in cases:
            diag = adversarial_convergence_diagnostics([0.6] * 10, hist)
            assert diag["classification"] == want
            assert diag["majority_share"] == 0.5

    def test_majority_share_validation(self):
        with pytest.raises(ValueError, match="majority_share"):
            adversarial_convergence_diagnostics([0.6], [0.6], majority_share=1.5)
        with pytest.raises(ValueError, match="majority_share"):
            adversarial_convergence_diagnostics([0.6], [0.6], majority_share=0.0)

    def test_single_group_degenerate(self):
        """One protected group only: majority share 1.0, no crash and no
        divide-by-zero.

        SUBJECT UNCHANGED, MECHANISM CORRECTED (BGL3 in_processing-1,
        2026-09-27). The no-crash / no-divide-by-zero subject is still what this
        pins and still passes. What it USED to pin beside that was
        ``adversary_skill == 0.0`` and a finite final accuracy, and those two
        values were the fabrication: skill 0.0 is below _CONVERGED_SKILL, so a
        single-group run was classified "converged" and worded "the protected
        attribute is no longer recoverable from the model output beyond chance.
        Debiasing converged.". Measured before the fix on these 200 rows:
        adversary_acc_history [1.0, 1.0, 1.0], loss [0.0, 0.0, 0.0],
        classification "converged", and no warning anywhere. With one group
        there is no attribute to recover, so the skill denominator (1 - p) is
        zero and the quantity does not exist.
        """
        X, y, _ = self._independent_imbalanced(n=200)
        with pytest.warns(UserWarning, match="single group"):
            res = sklearn_adversarial_debiasing(X, y, np.zeros(200, dtype=int), n_rounds=3)
        assert res["majority_share"] == 1.0
        assert res["insufficient_data"] is True
        with pytest.warns(UserWarning, match="not_assessed"):
            diag = adversarial_convergence_diagnostics(
                res["adversary_loss_history"],
                res["adversary_acc_history"],
                majority_share=res["majority_share"],
            )
        # No crash, no divide-by-zero: the original subject.
        assert isinstance(diag, dict)
        # Three states: neither a pass nor a fail.
        assert diag["classification"] == "not_assessed"
        assert diag["adversary_skill"] is None
        assert diag["final_adversary_accuracy"] is None


class TestReweightingDirection:
    def test_reweighting_reduces_recoverability_on_proxy_leak(self):
        """NEGATIVE CASE. Proxy scenario where reweighting CAN debias: y
        depends on x1 and s, x2 proxies s, so the predictor's output leaks s
        through x2. The corrected downweighting must drive adversary accuracy
        toward the majority-share baseline; the old upweighting stalled well
        above it (and in the audit's runs got WORSE at higher learning
        rates)."""
        rng = np.random.default_rng(11)
        n = 4000
        s = (rng.random(n) < 0.5).astype(int)
        x1 = rng.normal(size=n)
        y = (x1 + 0.9 * s + 0.4 * rng.normal(size=n) > 0.45).astype(int)
        x2 = s + 0.5 * rng.normal(size=n)
        X = np.column_stack([x1, x2])

        res = sklearn_adversarial_debiasing(X, y, s, n_rounds=50)
        h = res["adversary_acc_history"]

        assert h[0] - h[-1] >= 0.03, (
            "reweighting must reduce adversary accuracy over rounds, got "
            "%.4f -> %.4f" % (h[0], h[-1])
        )
        assert h[-1] - res["majority_share"] <= 0.05, (
            "final adversary accuracy %.4f must approach the majority-share "
            "baseline %.4f" % (h[-1], res["majority_share"])
        )

    def test_reweighting_harmless_when_attribute_unrecoverable(self):
        """Does-not-overcorrect: with an unrecoverable attribute the
        reweighting must not degrade the predictor."""
        rng = np.random.default_rng(7)
        n = 2000
        s = (rng.random(n) < 0.25).astype(int)
        X = rng.normal(size=(n, 3))
        y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
        res = sklearn_adversarial_debiasing(X, y, s, n_rounds=10)
        # Predictor log loss must not blow up across rounds.
        pl = res["predictor_loss_history"]
        assert pl[-1] <= pl[0] + 0.05
        # Adversary stays pinned at the baseline the whole time.
        assert max(res["adversary_acc_history"]) - res["majority_share"] < 0.02
