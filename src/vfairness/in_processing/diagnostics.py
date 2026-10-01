"""
Fair-Training Convergence Diagnostics (torch-free).

CS-T-44: Adversarial debiasing convergence diagnostics without PyTorch.

The classic adversarial debiasing setup (Zhang et al. 2018) trains a predictor
and an adversary in an alternating min-max game: the predictor learns the task
while a gradient-reversal signal pushes it to make its outputs uninformative
about a protected attribute; the adversary tries to recover that attribute from
the predictor output. torch is not available on the consumer runtime, so this
module emulates that game with scikit-learn only.

Each round:
    1. Fit a predictor (LogisticRegression) on (X, y) with per-sample weights.
    2. Fit an adversary (LogisticRegression) that recovers the protected
       attribute from the predictor output alone, recording its log loss and
       accuracy.
    3. Emulate gradient reversal by DOWNWEIGHTING samples whose protected
       attribute the adversary recovers confidently, penalizing the y-loss
       contribution of the attribute-recoverable score region so the predictor
       relies less on attribute-informative directions in the next round.
       (The original implementation upweighted those samples, which sharpens
       the attribute signal instead of suppressing it; 2026-08-22 audit.)

The recorded adversary-accuracy trajectory is then classified as converged /
oscillating / diverged / plateau. "Chance" for an adversary is the MAJORITY
GROUP SHARE, not 0.5: a majority-class guesser scores the majority share with
zero information, so adversary skill is measured as the accuracy EXCESS over
that baseline. A final accuracy near the majority share means the attribute is
no longer recoverable from the model output, which is the goal of debiasing.

Limitation (documented, not hidden): sample reweighting can only reduce
recoverability that the predictor is able to trade away, e.g. a proxy feature
whose coefficient can shrink. When the leak survives any monotone rescoring of
the score (e.g. a single feature that carries both the task and the attribute),
NO reweighting changes the sample ordering, so adversary accuracy cannot drop;
the diagnostics then honestly report a plateau/divergence.

References:
    - Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating Unwanted
      Biases with Adversarial Learning. AAAI/ACM AIES.

This module uses numpy and scikit-learn only (no torch, imblearn, fairlearn or
aif360); it needs no dependency beyond the package's own core set.
"""

import math
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

# Rows above this are subsampled (seeded) so the alternating game stays
# interactive; the same subset is used for training and evaluation every round.
_MAX_TRAIN_ROWS = 20000

# The classification a trajectory gets when the game recorded nothing a verdict
# could be read off. It is NOT one of the four measured outcomes: "plateau" is a
# finding ("debiasing stalled, the attribute stays partly recoverable") and it
# was what an empty and an all-NaN trajectory both came back as.
_NOT_ASSESSED = "not_assessed"


def sklearn_adversarial_debiasing(
    X: Any,
    y: Any,
    sensitive: Any,
    n_rounds: int = 50,
    learning_rate: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Torch-free alternating adversarial debiasing game.

    Args:
        X: Feature matrix (2D array-like of shape (n_samples, n_features)).
        y: Binary target labels. MORE THAN TWO CLASSES IS REFUSED (ValueError),
            because the adversary is measured on ``predict_proba(X)[:, 1]``
            alone: see the Raises section.
        sensitive: Protected-attribute labels (binary or categorical).
        n_rounds: Number of predictor / adversary rounds to play.
        learning_rate: Scales the per-round reweighting strength.
        seed: Random seed for subsampling (the estimators are deterministic).

    Returns:
        Dict with:
            predictions: list of final predictor labels (original label space).
            adversary_loss_history: list[float], adversary log loss per round.
            adversary_acc_history: list[float], adversary accuracy per round.
            predictor_loss_history: list[float], predictor log loss per round.
            n_rounds: int, the number of rounds played.
            majority_share: float, largest protected-group share of the rows
                actually trained on. This is the zero-information adversary
                accuracy baseline; pass it to
                adversarial_convergence_diagnostics so the verdict is not
                base-rate blind.
            insufficient_data: bool, True when the game could not be played as
                described, so nothing here supports a debiasing claim. Read it
                before reading the histories.
            not_assessed_reason: str, what was degenerate when
                insufficient_data is True, empty otherwise.
            predictor_fitted: bool, whether a predictor was actually trained.
                False means ``predictions`` is a constant fallback and no model
                looked at the features.
            degenerate_constant_predictions: bool, True when ``predictions``
                holds ONE distinct value. A constant output is trivially
                uninformative about the protected attribute, so the adversary
                scores the majority-group share by construction and the skill
                0.00 that follows is not evidence of debiasing. This is keyed on
                the predictor OUTPUT and is therefore reachable with a BINARY
                target, which ``insufficient_data``'s single-class test is not:
                measured 2026-09-30 on 400 rows of featureless X, a binary target
                produced predictor_fitted True, insufficient_data False and the
                verdict "Debiasing converged" for one distinct prediction across
                every row. Pass it to
                :func:`adversarial_convergence_diagnostics`, which cannot tell a
                hidden attribute from a model that made no decision without it.
            n_prediction_classes: int, the distinct-label count the flag above
                was derived from.

    Raises:
        ValueError: if n_rounds is below 1. The loop then runs zero times, no
            predictor is ever fitted, and every history comes back empty while
            ``predictions`` is still populated with a constant label. Measured
            before this guard on 200 rows with n_rounds=0: 200 predictions all
            equal to the first class, three empty histories and no warning, a
            completed fit reported by a function that never trained anything.
        ValueError: if y holds more than two classes. The adversary only ever
            sees ``predict_proba(X)[:, 1]``, one output column, so on a
            three-class target it measures a third of the model output and the
            attribute leaks through the rest. Measured before this guard on 600
            rows with three classes: skill 0.02 and the verdict "Debiasing
            converged", against skill 0.396 for an adversary given the full
            output, plus five NaN predictor losses and no warning.

    Warns:
        UserWarning: when a round could not compute a log loss and recorded NaN.
            The two ``log_loss`` calls used to sit under a bare ``except
            Exception`` that appended NaN in silence.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss

    if int(n_rounds) < 1:
        raise ValueError(
            "sklearn_adversarial_debiasing: n_rounds={0!r} plays no round at all, so no "
            "predictor is fitted and no adversary is measured. The dict this used to "
            "return still carried one prediction per row (a constant label from an "
            "untrained fallback) beside three empty histories, which reads as a "
            "completed fit. Pass n_rounds >= 1.".format(n_rounds)
        )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y)
    sensitive = np.asarray(sensitive)

    # Encode target and sensitive to contiguous integer codes so log_loss and
    # accuracy have a stable, explicit label set regardless of the input dtype.
    y_classes, y_codes = np.unique(y, return_inverse=True)
    s_classes, s_codes = np.unique(sensitive, return_inverse=True)

    # A NON-BINARY y is refused, the same way n_rounds < 1 is, and for the same
    # reason: the game below cannot be played on it, and every number it hands
    # back is then a measurement of something else.
    #
    # The adversary is only ever shown `predictor.predict_proba(Xs)[:, 1]`, ONE
    # column of the model output. For a binary y that column is the whole output
    # (the other is 1 - p). For three classes it is one third of it, and the
    # attribute leaks through the columns nobody looked at.
    #
    # Measured 2026-09-27 on 600 rows, default_rng(0), x1/x2 standard normal,
    # sensitive = (x2 > 0), y = 2 where x2 > 1.0 else 1 where x1 > 0 else 0,
    # n_rounds=5:
    #   before -> insufficient_data False, not_assessed_reason '',
    #             predictor_fitted True, predictor_loss_history
    #             [nan, nan, nan, nan, nan] (every round swallowed by the bare
    #             except below, silently), warnings NONE, and the pipeline
    #             verdict "Adversary accuracy is 52.8% against a 52.0%
    #             majority-class baseline (skill 0.02), so the protected
    #             attribute is no longer recoverable from the model output beyond
    #             chance. Debiasing converged." An adversary given the FULL
    #             three-column output on the same data reaches 71.0% accuracy,
    #             skill 0.396, above _DIVERGED_SKILL 0.30, i.e. "debiasing did
    #             not succeed".
    #   after  -> ValueError naming the 3 classes, before any history is built.
    if len(y_classes) > 2:
        raise ValueError(
            "sklearn_adversarial_debiasing: y holds {0} classes {1!r}, but this game is "
            "binary. The adversary is measured on predict_proba(X)[:, 1] alone, which is "
            "one of {0} output columns, so the attribute leaks through the columns it "
            "never sees: measured on 600 rows with 3 classes, this returned skill 0.02 "
            "and the verdict 'Debiasing converged' while an adversary given the full "
            "output recovered the attribute at skill 0.396. The predictor log loss was "
            "NaN in every round as well, because log_loss(labels=[0, 1]) raises on a "
            "non-binary target. Binarize y (one-vs-rest) and run the game per "
            "class.".format(len(y_classes), y_classes.tolist())
        )

    n = X.shape[0]

    # Cap rows for responsiveness (seeded, so the trajectory is reproducible).
    if n > _MAX_TRAIN_ROWS:
        sel = rng.choice(n, size=_MAX_TRAIN_ROWS, replace=False)
        X = X[sel]
        y_codes = y_codes[sel]
        s_codes = s_codes[sel]
        n = _MAX_TRAIN_ROWS

    # Standardize features so LogisticRegression converges quickly.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    single_y_class = len(y_classes) < 2
    single_s_class = len(s_classes) < 2

    # THREE STATES, NEVER TWO. Both degeneracies below used to be absorbed in
    # silence and both then produced the strongest all-clear the pipeline can
    # give. Measured 2026-09-27 on 200 rows, n_rounds=3, feeding
    # adversarial_convergence_diagnostics with this function's own
    # majority_share:
    #
    #   one protected group   -> adversary_acc_history [1.0, 1.0, 1.0],
    #       adversary_loss_history [0.0, 0.0, 0.0], majority_share 1.0, no
    #       warning, and the verdict "the protected attribute is no longer
    #       recoverable from the model output beyond chance. Debiasing
    #       converged." Nothing was recovered because there was nothing to
    #       recover: one group is not an attribute a classifier can get wrong.
    #   one target class      -> no predictor is fitted at all (the branch below
    #       sets predictor=None and emits a constant score), the adversary reads
    #       the majority share off that constant, skill 0.00, and the same
    #       "Debiasing converged" sentence comes back for a run that trained no
    #       model. A constant output is trivially uninformative about the
    #       attribute; that is the degenerate solution, not the goal.
    #
    # The histories become NaN rather than 1.0 / 0.0 so the verdict layer cannot
    # read a number off them, and the reason travels with the result.
    not_assessed_reasons: List[str] = []
    if single_s_class:
        not_assessed_reasons.append(
            "sensitive holds a single group ({0!r}), so the adversary has no "
            "attribute to recover and its accuracy is not a measurement".format(
                s_classes[0] if len(s_classes) else None
            )
        )
    if single_y_class:
        not_assessed_reasons.append(
            "y holds a single class ({0!r}), so no predictor is trained and the "
            "score every round is a constant".format(y_classes[0] if len(y_classes) else None)
        )
    if not_assessed_reasons:
        warnings.warn(
            "sklearn_adversarial_debiasing: "
            + "; ".join(not_assessed_reasons)
            + ". Reporting insufficient_data=True; the adversary history is NaN where "
            "nothing was measurable, so the convergence verdict is not_assessed rather "
            "than 'debiasing converged'.",
            UserWarning,
            stacklevel=2,
        )

    # Majority group share of the rows actually trained on: the accuracy a
    # zero-information (majority-class) adversary attains. Computed after the
    # subsample so it matches the recorded accuracy histories.
    if single_s_class:
        majority_share = 1.0
    else:
        majority_share = float(np.bincount(s_codes).max()) / float(n)

    w = np.ones(n, dtype=float)

    adversary_loss_history: List[float] = []
    adversary_acc_history: List[float] = []
    predictor_loss_history: List[float] = []

    predictor = None

    # Why a loss round came back NaN, if any did. Both log_loss calls below used
    # to sit under a BARE `except Exception` that appended NaN and said nothing,
    # and that silence was the only signal that a three-class y was outside the
    # contract (measured 2026-09-27: five NaN predictor losses, zero warnings,
    # and the verdict "Debiasing converged"). The refusal above closes that input
    # class; this closes the SWALLOW, so any other cause is disclosed instead of
    # arriving as a hole in the history.
    loss_failures: List[str] = []

    for _round in range(int(n_rounds)):
        # Predictor step: recover y from X under the current sample weights.
        if single_y_class:
            # Degenerate target: a constant output carries no task signal.
            p_hat = np.full(n, float(y_codes[0]))
            predictor = None
        else:
            predictor = LogisticRegression(max_iter=1000, solver="lbfgs")
            predictor.fit(Xs, y_codes, sample_weight=w)
            p_hat = predictor.predict_proba(Xs)[:, 1]

        p_hat_clipped = np.clip(p_hat, 1e-7, 1.0 - 1e-7)
        try:
            predictor_loss_history.append(float(log_loss(y_codes, p_hat_clipped, labels=[0, 1])))
        except Exception as exc:  # noqa: BLE001 - recorded below, never swallowed
            predictor_loss_history.append(float("nan"))
            loss_failures.append("predictor log loss: {0}: {1}".format(type(exc).__name__, exc))

        # Adversary step: recover the protected attribute from p_hat alone.
        if single_s_class:
            # Only one group present, so there is no recovery task. This used to
            # append 1.0 / 0.0, a perfect adversary against a 100% baseline,
            # which the skill normalization then turned into 0.00 and reported as
            # "debiasing converged" (see the block above the loop). NaN is the
            # measurement: unknown, not perfect and not chance.
            adversary_acc_history.append(float("nan"))
            adversary_loss_history.append(float("nan"))
            continue

        adv = LogisticRegression(max_iter=1000, solver="lbfgs")
        adv.fit(p_hat.reshape(-1, 1), s_codes)
        adv_proba = adv.predict_proba(p_hat.reshape(-1, 1))
        adv_pred = adv.classes_[np.argmax(adv_proba, axis=1)]

        adversary_acc_history.append(float(accuracy_score(s_codes, adv_pred)))
        try:
            adversary_loss_history.append(float(log_loss(s_codes, adv_proba, labels=adv.classes_)))
        except Exception as exc:  # noqa: BLE001 - recorded below, never swallowed
            adversary_loss_history.append(float("nan"))
            loss_failures.append("adversary log loss: {0}: {1}".format(type(exc).__name__, exc))

        # Gradient-reversal emulation via sample reweighting.
        # For each sample take the probability the adversary assigns to its TRUE
        # group. Where that is high, the protected attribute is recoverable from
        # the predictor output, so DOWNWEIGHT those samples: their y-loss
        # contribution is penalized, and the refit predictor relies less on the
        # attribute-informative score region next round. (The original code
        # UPWEIGHTED them, which makes the predictor fit the recoverable region
        # HARDER and sharpens the attribute signal - the emulation's sign was
        # backwards and the loop never debiased; 2026-08-22 audit.) Normalize
        # so the effective sample size stays n (prevents weight drift /
        # collapse).
        class_to_col = {c: j for j, c in enumerate(adv.classes_.tolist())}
        cols = np.array([class_to_col[int(c)] for c in s_codes])
        adv_prob_correct = adv_proba[np.arange(n), cols]
        w = w / (1.0 + float(learning_rate) * adv_prob_correct)
        total = w.sum()
        if total > 0:
            w = w * (n / total)

    # A NaN in a loss history is a hole, and a reader cannot tell a hole from a
    # measurement without being told why it is there.
    if loss_failures:
        seen: List[str] = []
        for reason in loss_failures:
            if reason not in seen:
                seen.append(reason)
        warnings.warn(
            "sklearn_adversarial_debiasing: {0} of {1} round(s) could not compute a log "
            "loss and recorded NaN instead of a measurement: {2}. Those rounds are NOT "
            "evidence of a low loss.".format(len(loss_failures), int(n_rounds), "; ".join(seen)),
            UserWarning,
            stacklevel=2,
        )

    # Final predictor labels, mapped back to the original label space.
    if predictor is None:
        preds = np.full(n, y_classes[0])
    else:
        pred_codes = predictor.predict(Xs)
        preds = y_classes[pred_codes]

    # KEYED ON THE OUTPUT, NOT ON THE INPUT (BGL4 audit wave 4, 2026-09-30).
    # The single-target-class refusal above states its own reason: "no predictor
    # is fitted at all ... the adversary reads the majority share off that
    # constant, skill 0.00, and the same Debiasing converged sentence comes back
    # for a run that trained no model. A constant output is trivially
    # uninformative about the attribute; that is the degenerate solution, not the
    # goal." It is keyed on len(y_classes) < 2, a property of the INPUT, while
    # the quantity that decides it is whether the predictor OUTPUT is constant.
    #
    # Measured 2026-09-30, 400 rows, n_rounds=5, sensitive = (x2 > 0), with a
    # BINARY target and features carrying no signal at all:
    #
    #   predictor_fitted TRUE, insufficient_data FALSE, not_assessed_reason "",
    #   ONE distinct prediction for all 400 rows, acc_hist [0.52]*5, skill 0.0,
    #   classification "converged", ZERO WARNINGS, and the verdict "Adversary
    #   accuracy is 52.0% against a 52.0% majority-class baseline (skill 0.00),
    #   so the protected attribute is no longer recoverable from the model
    #   output. Debiasing converged."
    #
    # That is the guard's own forbidden sentence, reproduced with a binary
    # target. The collapse WAS in the returned dict, as a predictions list
    # holding one distinct value, and no field reported it: predictor_fitted is
    # True (a predictor WAS fitted), insufficient_data is False and
    # not_assessed_reason is empty. Controls on the same fixture, both silent and
    # both unchanged: a real group-correlated target reached skill 0.29 with 2
    # distinct predictions, and a pure-noise target reached skill 0.84 "diverged".
    #
    # The histories are NOT overwritten here. The adversary accuracy really was
    # measured; what is false is the CLAIM built on it, so the measurement stays
    # and the claim is withdrawn through insufficient_data and the flag, which is
    # exactly how the single-target-class sibling above is handled.
    prediction_classes = np.unique(preds) if preds.size else np.asarray([])
    n_prediction_classes = int(len(prediction_classes))
    degenerate_constant_predictions = bool(preds.size and n_prediction_classes < 2)
    if degenerate_constant_predictions and not single_y_class:
        not_assessed_reasons.append(
            "the predictor collapsed to a CONSTANT output ({0!r} for all {1} row(s)), so "
            "the adversary was asked to recover the attribute from a number that never "
            "varies: its accuracy is the majority-group share by construction and the "
            "skill 0.00 that follows is not evidence of debiasing".format(
                prediction_classes[0] if n_prediction_classes else None, n
            )
        )
        warnings.warn(
            "sklearn_adversarial_debiasing: the predictor collapsed to a CONSTANT output "
            "({0} distinct prediction(s) over {1} row(s)), so nothing here supports a "
            "debiasing claim: a constant score is trivially uninformative about the "
            "protected attribute, which is the degenerate solution and not the goal. "
            "Reporting insufficient_data=True with "
            "degenerate_constant_predictions=True; the adversary history is left as "
            "measured, because it WAS measured, and it is the verdict built on it that is "
            "withdrawn. Pass degenerate_constant_predictions to "
            "adversarial_convergence_diagnostics so its verdict says so "
            "too.".format(n_prediction_classes, n),
            UserWarning,
            stacklevel=2,
        )

    return {
        "predictions": preds.tolist(),
        # True means `predictions` holds ONE distinct value, so every fairness
        # and skill number in this dict is vacuous rather than good. Read it
        # beside insufficient_data, and pass it to
        # adversarial_convergence_diagnostics, which cannot tell a hidden
        # attribute from a model that made no decision without it.
        "degenerate_constant_predictions": degenerate_constant_predictions,
        "n_prediction_classes": n_prediction_classes,
        "adversary_loss_history": adversary_loss_history,
        "adversary_acc_history": adversary_acc_history,
        "predictor_loss_history": predictor_loss_history,
        "n_rounds": int(n_rounds),
        "majority_share": majority_share,
        "insufficient_data": bool(not_assessed_reasons),
        "not_assessed_reason": "; ".join(not_assessed_reasons),
        # False means `predictions` is the constant fallback, not a model's
        # output: the single-class-target branch never fits a predictor.
        "predictor_fitted": predictor is not None,
    }


# Classification thresholds. The CS-T-44 spec fixed them on RAW accuracy
# (converged < 0.55, diverged > 0.65), which silently assumed balanced groups:
# a majority-class adversary scores the majority share with ZERO information,
# so for any split more imbalanced than 65/35 the "converged" verdict was
# unreachable by construction (2026-08-22 audit false-FAIL). Classification is
# now on SKILL, the accuracy excess over the majority-share baseline normalized
# by its achievable range: skill = (acc - p) / (1 - p). At the balanced
# baseline p = 0.5 these skill thresholds are exactly the spec's raw-accuracy
# thresholds (0.55 -> 0.10, 0.65 -> 0.30).
_CONVERGED_SKILL = 0.10  # final skill below this => attribute hidden
_DIVERGED_SKILL = 0.30  # final skill above this => still recoverable
_OSCILLATION = 0.05  # mean successive |acc diff| above this => unsettled game


def _adversary_skill(final_acc: float, majority_share: float) -> float:
    """Adversary accuracy in excess of the majority-class baseline, normalized.

    Returns (acc - p) / (1 - p) where p is the majority group share, i.e. the
    accuracy of a zero-information majority-class guesser. 0.0 means no skill
    beyond chance, 1.0 means perfect recovery.

    Degenerate p >= 1 (a single protected group) yields NaN, not 0.0. It used to
    yield 0.0 with a comment calling it "no attribute signal recoverable beyond
    triviality", and 0.0 is below _CONVERGED_SKILL, so a single-group run was
    classified "converged" and worded "the protected attribute is no longer
    recoverable from the model output beyond chance. Debiasing converged."
    Measured 2026-09-27 on 200 rows with one group. The denominator 1 - p is
    zero there: the achievable range the skill is normalized by does not exist,
    which is an unmeasurable quantity and not a zero one.
    """
    p = float(majority_share)
    if p >= 1.0 - 1e-12:
        return float("nan")
    return (float(final_acc) - p) / (1.0 - p)


def _adversarial_verdict(
    classification: str,
    final_acc: float,
    oscillation_score: Optional[float],
    improvement: float,
    majority_share: float,
    skill: float,
) -> str:
    """Plain-language interpretation of the convergence classification."""
    pct = float(final_acc) * 100.0
    base_pct = float(majority_share) * 100.0
    if classification == _NOT_ASSESSED:
        # Reached only from the measured path, where the FINAL accuracy and the
        # skill are real numbers and the trajectory SHAPE is the part that is
        # unknown. Measured 2026-09-27 on
        # adversarial_convergence_diagnostics([0.6], [0.62], None, 0.5), one
        # round, before: classification "plateau", oscillation_score 0.0,
        # verdict "Adversary accuracy plateaued at 62.0% ... Debiasing stalled;
        # the attribute stays partly recoverable", no warning. "Plateaued" and
        # "stalled" are claims about a shape read off one point.
        return (
            "Adversary accuracy is {0:.1f}% against a {1:.1f}% majority-class baseline "
            "(skill {2:.2f}), between the converged and diverged thresholds. Whether the "
            "game PLATEAUED or OSCILLATED is NOT ASSESSED: fewer than two usable rounds "
            "were recorded, so no successive accuracy difference exists and the "
            "oscillation score is None. This says nothing about whether debiasing "
            "stalled.".format(pct, base_pct, float(skill))
        )
    if classification == "converged":
        return (
            "Adversary accuracy is {0:.1f}% against a {1:.1f}% majority-class "
            "baseline (skill {2:.2f}), so the protected attribute is no "
            "longer recoverable from the model output beyond chance. "
            "Debiasing converged.".format(pct, base_pct, float(skill))
        )
    if classification == "diverged":
        return (
            "Adversary still recovers the protected attribute at {0:.1f}% "
            "accuracy against a {1:.1f}% majority-class baseline (skill "
            "{2:.2f}). The model output stays informative about the "
            "attribute; debiasing did not succeed.".format(pct, base_pct, float(skill))
        )
    if classification == "oscillating":
        # Only reachable with a MEASURED oscillation score: the caller cannot
        # reach this classification without a successive pair to average, which
        # is the whole point of the None above.
        assert oscillation_score is not None
        return (
            "Adversary accuracy oscillates (score {0:.3f}) around {1:.1f}%. "
            "The adversarial game did not settle; a lower learning rate may "
            "help.".format(float(oscillation_score), pct)
        )
    return (
        "Adversary accuracy plateaued at {0:.1f}% against a {1:.1f}% "
        "majority-class baseline (skill {2:.2f}) without dropping to chance. "
        "Debiasing stalled; the attribute stays partly "
        "recoverable.".format(pct, base_pct, float(skill))
    )


def adversarial_convergence_diagnostics(
    adversary_loss_history: List[float],
    adversary_acc_history: List[float],
    predictor_loss_history: Optional[List[float]] = None,
    majority_share: Optional[float] = None,
    degenerate_constant_predictions: Optional[bool] = None,
) -> Dict[str, Any]:
    """Classify an adversarial-debiasing trajectory as converged / oscillating /
    diverged / plateau.

    Args:
        adversary_loss_history: Per-round adversary log loss.
        adversary_acc_history: Per-round adversary accuracy.
        predictor_loss_history: Optional per-round predictor log loss.
        majority_share: Largest protected-group share, i.e. the accuracy of a
            zero-information majority-class adversary (the ``majority_share``
            key of :func:`sklearn_adversarial_debiasing`). ALWAYS pass this
            for imbalanced groups: without it a 0.5 baseline is assumed
            (balanced groups), which reproduces the historical CS-T-44
            raw-accuracy thresholds but misreads any imbalanced split - e.g.
            a 75/25 split where the attribute is entirely unrecoverable still
            read as "debiasing did not succeed" because a majority guesser
            already scores 75%.
        degenerate_constant_predictions: Whether the PREDICTOR whose output the
            adversary was scored against produced a constant value (the
            ``degenerate_constant_predictions`` key of
            :func:`sklearn_adversarial_debiasing`). ALWAYS pass this. A collapsed
            predictor and a perfectly debiased one produce the SAME histories,
            an adversary sitting at the majority-group share with skill 0.00, so
            this function cannot tell them apart from the trajectory and the
            caller is the only one who can. True withholds the verdict as
            ``not_assessed``; None means the question was never asked and is
            reported as such in the returned dict, because a constant model is
            trivially uninformative about the attribute, which is the degenerate
            solution and not the goal.

    Returns:
        Dict with classification, final_adversary_accuracy, final_adversary_loss,
        adversary_skill, majority_share, oscillation_score,
        adversary_loss_trajectory, adversary_accuracy_trajectory,
        predictor_loss_trajectory, n_rounds_unusable, n_loss_rounds_unusable,
        n_gapped_pairs, n_rounds_usable and verdict.

        A round recorded as ``None`` counts exactly as one recorded as NaN
        (BGL4 wave 4): it is an unmeasurable round, not a deletion, so the
        trajectories keep one entry per recorded round and ``n_rounds_unusable``
        counts it. ``n_loss_rounds_unusable`` is the same count for the LOSS
        series, which nothing used to test, and ``final_adversary_loss`` is None
        rather than NaN when the last round's loss is not a measurement.
        ``n_gapped_pairs`` is how many adjacent round pairs the oscillation score
        had to leave out: a mean absolute SUCCESSIVE difference cannot be taken
        across a gap, so the window is not closed up, and with no admissible
        adjacent pair ``oscillation_score`` is None and the shape-dependent
        classifications are withheld.

        THREE STATES, NEVER TWO. ``classification`` is ``"not_assessed"``, with
        ``adversary_skill`` and ``oscillation_score`` both None, when the
        trajectory carries no verdict: an empty history, a final round whose
        adversary accuracy is not a number, or a majority share of 1.0 (a single
        protected group, so the skill scale is zero). That is neither a pass nor
        a fail, and it is not "plateau", which is the FINDING that debiasing
        stalled. ``n_rounds_unusable`` says how many recorded rounds the
        classification could not use and ``n_rounds_usable`` how many it used.

        ``oscillation_score`` is also None, on its own, when FEWER THAN TWO
        usable rounds were recorded (``n_rounds_usable`` < 2, which ``n_rounds=1``
        produces): no successive pair exists, so there is no difference to
        average, and 0.0 would be the perfectly-settled end of that scale. The
        final accuracy and the skill ARE measured there, so "converged" and
        "diverged" stay reachable; the in-between band is ``"not_assessed"``
        rather than "plateau", because "plateaued" and "oscillates" are both
        claims about a trajectory's shape.

    Thresholds: classification is on adversary SKILL, the accuracy excess over
    the majority-share baseline p normalized by its achievable range,
    skill = (acc - p) / (1 - p). Skill below 0.10 => "converged" (the
    attribute can no longer be recovered beyond chance, the debiasing goal);
    above 0.30 => "diverged" (debiasing failed, the attribute is still
    recoverable); a high mean successive accuracy difference =>
    "oscillating"; an in-between, near-flat trajectory that never reaches
    chance => "plateau". At p = 0.5 these are exactly the CS-T-44 raw
    thresholds (converged < 0.55, diverged > 0.65), so behavior for balanced
    groups is unchanged.
    """

    # ABSENCE IS NOT EMPTINESS, AND None IS NOT A DELETION (BGL4 audit wave 4,
    # 2026-09-30). These three lines FILTERED None out, which removed the round
    # from the history entirely instead of recording it as unmeasurable, so the
    # count below never saw it and the trajectory handed to a reader was shorter
    # than the game. The comment below considered exactly this channel and
    # dismissed it on the grounds that sklearn_adversarial_debiasing records NaN,
    # but adversary_acc_history is a public List[float] argument and any caller
    # loop that skipped a round supplies None. Measured 2026-09-30, the SAME
    # five-round game written two ways, majority_share 0.5:
    #
    #   [0.6, nan, nan, nan, 0.62] -> usable 2, unusable 3, 1 warning,
    #                                 trajectory keeps 5 entries  (disclosed)
    #   [0.6, None, None, None, 0.62] -> usable 2, unusable ZERO, 0 WARNINGS,
    #                                 trajectory [0.6, 0.62]      (vanished)
    #
    # None is now mapped to NaN rather than dropped, so the six doors of absence
    # arrive at the one machinery that already discloses them and the trajectory
    # keeps one entry per recorded round.
    def _as_round(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            # A blank string, pd.NA or the literal 'None' is an absence too, and
            # str()-ing it into the history would mint content.
            return float("nan")

    acc = [_as_round(a) for a in (adversary_acc_history or [])]
    loss = [_as_round(x) for x in (adversary_loss_history or [])]
    # Only None was dropped here, and None is not how an unmeasurable round
    # arrives: sklearn_adversarial_debiasing records NaN. Every comparison
    # against NaN is False, so an all-NaN trajectory fell past the converged
    # test, past the oscillation test and past the diverged test and landed on
    # the "plateau" default. Measured 2026-09-27 on five NaN rounds:
    # classification "plateau", verdict "Adversary accuracy plateaued at nan%
    # ... Debiasing stalled; the attribute stays partly recoverable", no
    # warning. That is a finding about a game nobody could read.
    n_rounds_unusable = sum(1 for a in acc if not math.isfinite(a))
    # ONE OF TWO RECORDED SERIES (BGL4 audit wave 4). n_rounds_unusable counted
    # non-finite ACCURACY only, and the loss half of the trajectory was never
    # tested at all: measured acc [0.6, 0.62] with adversary_loss_history
    # [nan, nan] returned final_adversary_loss nan, n_rounds_unusable 0 and zero
    # warnings, so a reader was handed a NaN loss with nothing saying it is a
    # hole. final_adversary_loss is None (the third state this dict already
    # spells for that field on the not-assessed path) when the last round's loss
    # is not a measurement.
    n_loss_rounds_unusable = sum(1 for x in loss if not math.isfinite(x))
    if predictor_loss_history:
        pred_loss = [_as_round(p) for p in predictor_loss_history]
    else:
        pred_loss = []

    # Baseline: accuracy of a zero-information majority-class adversary.
    # None keeps the historical balanced-groups assumption (p = 0.5).
    if majority_share is None:
        baseline = 0.5
    else:
        baseline = float(majority_share)
        if not (0.0 < baseline <= 1.0):
            raise ValueError(
                "majority_share must be in (0, 1], got {0!r} (it is the "
                "largest protected-group share)".format(majority_share)
            )

    # The three not-assessed cases, each proven to come back as a measured
    # verdict before this block existed. "plateau" is one of the four FINDINGS
    # ("debiasing stalled, the attribute stays partly recoverable"), and an
    # oscillation_score of 0.0 is "perfectly settled", so both were answers to
    # questions nobody asked. _NOT_ASSESSED is neither a pass nor a fail.
    final_acc = float(acc[-1]) if acc else float("nan")
    skill = _adversary_skill(final_acc, baseline) if acc else float("nan")
    if not acc:
        not_assessed = "no adversary trajectory was recorded"
    elif degenerate_constant_predictions is True:
        # A CONSTANT MODEL OUTPUT IS NOT A HIDDEN ATTRIBUTE (BGL4 audit wave 4).
        # This function sees only the histories, and a collapsed predictor and a
        # perfectly debiased one produce the SAME ones: an adversary at the
        # majority-group share, skill 0.00, "converged". The two cannot be told
        # apart from here, so the caller has to say which it was, and
        # sklearn_adversarial_debiasing now returns exactly this flag. Measured
        # 2026-09-30 on 400 rows with featureless X: one distinct prediction for
        # every row and the verdict "the protected attribute is no longer
        # recoverable from the model output. Debiasing converged." Placed above
        # the finiteness checks because it disqualifies the verdict whatever the
        # numbers say.
        not_assessed = (
            "the caller reports the predictor collapsed to a constant output, so the "
            "adversary was scored against a number that never varies: its accuracy is the "
            "majority-group share by construction and the skill that follows is not "
            "evidence that the attribute was hidden"
        )
    elif not math.isfinite(final_acc):
        not_assessed = (
            "the final round's adversary accuracy is not a number ({0} of {1} round(s) "
            "recorded no usable accuracy)".format(n_rounds_unusable, len(acc))
        )
    elif not math.isfinite(skill):
        not_assessed = (
            "the majority-group share is {0:.4f}, so the skill scale (1 - p) it would be "
            "normalized by is zero: with a single protected group there is no attribute "
            "to recover".format(baseline)
        )
    else:
        not_assessed = ""

    if not_assessed:
        warnings.warn(
            "adversarial_convergence_diagnostics: {0}, so the trajectory cannot be "
            "classified. Reporting classification={1!r} (could not check), NOT "
            "'plateau' or 'converged'.".format(not_assessed, _NOT_ASSESSED),
            UserWarning,
            stacklevel=2,
        )
        return {
            "classification": _NOT_ASSESSED,
            "final_adversary_accuracy": final_acc if acc and math.isfinite(final_acc) else None,
            "final_adversary_loss": None,
            "adversary_skill": None,
            "majority_share": baseline,
            # None, not 0.0: an unrecorded game is not a settled one.
            "oscillation_score": None,
            "adversary_loss_trajectory": loss,
            "adversary_accuracy_trajectory": acc,
            "predictor_loss_trajectory": pred_loss,
            "n_rounds_unusable": n_rounds_unusable,
            # The loss series is counted separately because it is a SECOND
            # recorded series and n_rounds_unusable never covered it.
            "n_loss_rounds_unusable": n_loss_rounds_unusable,
            # None, not 0: the oscillation window was never reached on this path,
            # so nothing counted its gapped pairs.
            "n_gapped_pairs": None,
            # True / False / None (the caller did not report it). Reading None as
            # False is what let a collapsed predictor wear "Debiasing converged".
            "degenerate_constant_predictions": degenerate_constant_predictions,
            "n_rounds_usable": sum(1 for a in acc if math.isfinite(a)),
            "verdict": (
                "Convergence diagnostics NOT ASSESSED: {0}. This says nothing about "
                "whether debiasing converged, stalled or failed.".format(not_assessed)
            ),
        }

    # None, not NaN: this field's third state on the not-assessed path above is
    # already None, and a NaN handed to a reader as final_adversary_loss is a
    # hole nothing identifies as one. See n_loss_rounds_unusable above for the
    # measured run this closes.
    final_loss: Optional[float]
    if loss and math.isfinite(loss[-1]):
        final_loss = float(loss[-1])
    else:
        final_loss = None
    if n_loss_rounds_unusable:
        warnings.warn(
            "adversarial_convergence_diagnostics: {0} of {1} recorded round(s) carry no "
            "usable adversary LOSS. final_adversary_loss is {2!r} and the loss half of the "
            "trajectory is returned with those rounds as NaN: they are NOT evidence of a "
            "low loss. The classification and the verdict rest on the ACCURACY series "
            "only.".format(n_loss_rounds_unusable, len(loss), final_loss),
            UserWarning,
            stacklevel=2,
        )

    # Oscillation over the recent window: mean absolute successive difference.
    # Over the USABLE rounds only. One NaN in the window made every difference
    # touching it NaN and the mean with it, and `nan > _OSCILLATION` is False, so
    # a trajectory that could not be checked for oscillation was reported as
    # settled and fell through to the next test.
    usable = [a for a in acc if math.isfinite(a)]
    if n_rounds_unusable:
        warnings.warn(
            "adversarial_convergence_diagnostics: {0} of {1} recorded round(s) carry no "
            "usable adversary accuracy and were left out of the oscillation score. The "
            "classification below rests on the {2} usable round(s).".format(
                n_rounds_unusable, len(acc), len(usable)
            ),
            UserWarning,
            stacklevel=2,
        )
    # A SUCCESSIVE DIFFERENCE MUST BE BETWEEN SUCCESSIVE ROUNDS (BGL4 audit
    # wave 4, 2026-09-30). Compacting the window to `usable` before differencing
    # takes the difference ACROSS the gaps and calls it successive. The count of
    # unusable rounds was disclosed above; the SHAPE claim built on top of the
    # gaps was not withdrawn. Measured 2026-09-30, majority_share 0.5:
    #
    #   [0.6, None, 0.95, None, 0.6] -> oscillation_score 0.35, classification
    #       "oscillating", verdict "Adversary accuracy oscillates (score 0.350)
    #       around 60.0%", 0 warnings. Those 0.35 differences are between rounds
    #       1 and 3 and rounds 3 and 5.
    #   [0.6, nan, nan, nan, 0.62] -> oscillation_score 0.020, reported as a
    #       successive difference between round 1 and round 5.
    #
    # The window is now over the RECORDED rounds and a difference is admitted
    # only when both ends of the adjacent pair are usable. With no admissible
    # pair the score is None (could not check), which the dispatch below already
    # routes to _NOT_ASSESSED rather than to "plateau", the FINDING that
    # debiasing stalled. `improvement` stays over the usable rounds, which is an
    # honest first-to-last statement and is not a successive-difference claim.
    recorded_window = acc[-min(10, len(acc)) :]
    diffs = [
        abs(recorded_window[i] - recorded_window[i - 1])
        for i in range(1, len(recorded_window))
        if math.isfinite(recorded_window[i]) and math.isfinite(recorded_window[i - 1])
    ]
    n_gapped_pairs = max(len(recorded_window) - 1, 0) - len(diffs)
    window = usable[-min(10, len(usable)) :]
    oscillation_score: Optional[float]
    if diffs:
        oscillation_score = float(sum(diffs) / len(diffs))
        if n_gapped_pairs:
            warnings.warn(
                "adversarial_convergence_diagnostics: {0} of {1} adjacent round pair(s) in "
                "the window have an unusable end and were left out of the oscillation "
                "score, which is the mean absolute SUCCESSIVE difference and is therefore "
                "computed over the {2} adjacent pair(s) that remain, NOT across the gaps "
                "(a difference between round 1 and round 5 is not a successive "
                "one).".format(n_gapped_pairs, max(len(recorded_window) - 1, 0), len(diffs)),
                UserWarning,
                stacklevel=2,
            )
    elif len(window) >= 2:
        # Two or more usable rounds exist but no two of them are ADJACENT, so
        # there is no successive pair to average. 0.0 would be the perfectly
        # settled end of that scale and the compacted mean was a shape claim
        # about rounds that never followed one another.
        oscillation_score = None
        warnings.warn(
            "adversarial_convergence_diagnostics: {0} usable round(s) were recorded but no "
            "two of them are ADJACENT, so no successive accuracy difference exists and "
            "oscillation_score is None (COULD NOT CHECK), not the {1:.3f} a mean over the "
            "compacted window used to report as a successive difference. 'oscillating' and "
            "'plateau' are both claims about the trajectory's shape and neither is readable "
            "across a gap.".format(
                len(window),
                sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
                / (len(window) - 1),
            ),
            UserWarning,
            stacklevel=2,
        )
    else:
        # None, not 0.0: with fewer than two usable rounds there is no
        # successive pair, so no oscillation exists to average, and 0.0 is the
        # PERFECTLY SETTLED end of that scale. n_rounds=1 is a supported call
        # (sklearn_adversarial_debiasing refuses only n_rounds < 1), so this
        # arrives from the public API.
        #
        # Measured 2026-09-27 on adversarial_convergence_diagnostics([0.6],
        # [0.62], None, 0.5):
        #   before -> oscillation_score 0.0, classification "plateau",
        #             n_rounds_unusable 0, verdict "Adversary accuracy plateaued
        #             at 62.0% ... Debiasing stalled", warnings NONE
        #   after  -> oscillation_score None, classification "not_assessed",
        #             adversary_skill 0.24 (still measured), one warning naming
        #             the single usable round
        # The 0.0 also GATED the classification through
        # `elif oscillation_score > _OSCILLATION`, so "oscillating" was
        # unreachable by construction for every one-round game: probed at 0.51,
        # 0.62, 0.90 and 0.99 it was never once reached.
        oscillation_score = None
        warnings.warn(
            "adversarial_convergence_diagnostics: only {0} usable round(s) were "
            "recorded, so no successive accuracy difference exists and "
            "oscillation_score is None (COULD NOT CHECK), not 0.0, which is the "
            "settled end of that scale. 'oscillating' cannot be reached from fewer "
            "than two rounds, so its absence here is not evidence that the game "
            "settled.".format(len(window)),
            UserWarning,
            stacklevel=2,
        )

    # Positive improvement means the adversary weakened over the window.
    improvement = float(window[0] - window[-1])

    # Order matters. "converged" and "diverged" rest on the FINAL round's
    # accuracy, which one usable round does supply, so both stay reachable.
    # "oscillating" and "plateau" are claims about the trajectory's SHAPE, and
    # neither is readable without a successive pair: the in-between skill band
    # is therefore not_assessed rather than "plateau", which is the finding
    # "debiasing stalled".
    if skill < _CONVERGED_SKILL:
        classification = "converged"
    elif oscillation_score is not None and oscillation_score > _OSCILLATION:
        classification = "oscillating"
    elif skill > _DIVERGED_SKILL:
        classification = "diverged"
    elif oscillation_score is None:
        classification = _NOT_ASSESSED
    else:
        classification = "plateau"

    verdict = _adversarial_verdict(
        classification, final_acc, oscillation_score, improvement, baseline, skill
    )

    return {
        "classification": classification,
        "final_adversary_accuracy": final_acc,
        "final_adversary_loss": final_loss,
        "adversary_skill": skill,
        "majority_share": baseline,
        "oscillation_score": oscillation_score,
        "adversary_loss_trajectory": loss,
        "adversary_accuracy_trajectory": acc,
        "predictor_loss_trajectory": pred_loss,
        # How much of the trajectory the classification could NOT use. 0 in the
        # normal case; a reader has to be told rather than infer a full game.
        "n_rounds_unusable": n_rounds_unusable,
        # And the SECOND recorded series, which n_rounds_unusable never covered:
        # how many rounds carry no usable adversary loss.
        "n_loss_rounds_unusable": n_loss_rounds_unusable,
        # How many adjacent round pairs in the oscillation window had an
        # unusable end. Read beside oscillation_score: a successive difference
        # cannot be taken across a gap, so these pairs are left out rather than
        # closed up.
        "n_gapped_pairs": n_gapped_pairs,
        # What the caller reported about the predictor's own output: True means a
        # constant model, False a model that decided, None that the question was
        # never asked. A "converged" classification below is only evidence of
        # debiasing when this is False, because a collapsed predictor produces
        # the identical trajectory. THREE STATES, and None is the could-not-check.
        "degenerate_constant_predictions": degenerate_constant_predictions,
        # And how many rounds it COULD use. An oscillation_score of None is
        # explained by this count: fewer than two usable rounds carry no
        # successive difference to average.
        "n_rounds_usable": len(usable),
        "verdict": verdict,
    }
