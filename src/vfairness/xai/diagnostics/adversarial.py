"""
Adversarial probe. Reference: Slack, Hilgard, Jia, Singh & Lakkaraju
(2020) AIES, pp. 180-186. Replicated by Chauhan et al. (2025) arXiv:
2508.11053 (SHLIME).

The probe flags an OOD-detection layer that hides the model's true
behaviour from perturbation-based explainers (KernelSHAP, LIME). It
returns ``adversarial_flag = True`` plus a reason string when the
explanation is suspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from vfairness._not_assessed import warn_not_assessed


def _positive_mean(predict_fn: Callable[[np.ndarray], np.ndarray], arr: np.ndarray) -> float:
    """Mean positive-class prediction, robust to 1-D or 2-D proba output."""
    out = np.asarray(predict_fn(arr)).reshape(len(arr), -1)
    return float(out[:, -1].mean())


#: Spread a background column must carry, RELATIVE to that column's own
#: magnitude, before the perturbation draw can be said to have moved.
#:
#: Relative and not an equality with zero, on purpose. The first version of this
#: guard (and the twin's, written the same day) asked ``np.ptp(col) == 0.0``, and
#: a background one part in 1e12 away from constant is not exactly zero: measured
#: on 50 copies of one row plus a 1e-12 jitter, the per-column spread was
#: 4.0e-12 .. 5.3e-12 against column magnitudes of 1.3 .. 2.5, so the exact test
#: said "this background has spread" and the probe reported the clean verdict for
#: a scaffold. A relative spread of 1e-9 means the draw agrees with the instance
#: to nine significant digits, and a real sample clears it by orders of
#: magnitude: the varying fixture in tests/test_bgl3_xai_1.py measures 1.0 .. 2.4.
_MIN_RELATIVE_SPREAD = 1e-9


def _unperturbable_background(background: np.ndarray) -> str:
    """Why this background makes the probe's own comparison impossible, or "".

    ``_probe_gap`` draws its cloud as
    ``rng.normal(loc=x, scale=background.std(axis=0) + 1e-9)``. The ``+ 1e-9`` is
    a division guard, and it turns a background with no spread into a
    usable-looking scale: every "perturbed" point is then x to within a
    nanometre, x's nearest neighbours are copies of x, and the gap is 0.0 BY
    CONSTRUCTION for any model whatsoever. A check whose answer cannot depend on
    the model is not a check.

    Uses ``np.ptp``, a subtraction of two values that are really in the data, and
    never ``np.var(...) == 0``: an accumulated variance is exactly 0.0 only for
    some n and some values, so a variance guard passes for a round fixture and
    fails on real data.

    Returns the empty string when the background CAN be perturbed, so a caller
    reads it as a reason and not as a boolean.
    """
    bg = np.asarray(background, dtype=float)
    if bg.ndim != 2 or bg.shape[0] == 0 or bg.shape[1] == 0:
        return (
            f"the background is not a usable 2-D sample (shape {bg.shape}), so there is "
            "no cloud to draw around x and nothing on the manifold to compare it against"
        )
    spread = np.ptp(bg, axis=0)
    magnitude = np.max(np.abs(bg), axis=0)
    usable = np.isfinite(spread) & np.isfinite(magnitude)
    floor = _MIN_RELATIVE_SPREAD * np.maximum(magnitude, np.finfo(float).tiny)
    # PARTLY FROZEN IS STILL UNPERTURBABLE ALONG THE FROZEN COLUMNS (BGL6 F04,
    # 2026-09-28). `if n_moving: return` cleared as soon as ONE column moved, and
    # the seven fixtures both twins are pinned on are each either FULLY frozen or
    # fully real, so they agree here and agree on the answer that lets the probe run
    # with no power along the frozen column. Measured on 3 columns where column 0 is
    # constant: sigma for that column is 1e-9, the division guard rather than
    # spread, so every drawn point keeps x[0] and the gap along it is exactly 0.0
    # for ANY model. A model deliberately scaffolded on column 0 came back
    # flag=False, confidence=1.0, byte-identical to a clean model, and the consumer
    # published that as a clean verdict.
    #
    # A clean verdict over part of the input space reads as a clean verdict over the
    # input space, and no disclosure can separate the two models here, because the
    # probe genuinely learned nothing about that column for either of them. So a
    # frozen column is a could-not-check, named, and the caller is told what to do
    # about it. This refuses more often than before, on purpose: a constant column
    # in a background is common, and the honest answer for it is "this probe cannot
    # speak about that feature", not a clean bill for the model.
    moving = usable & (spread > floor)
    n_moving = int(np.count_nonzero(moving))
    if n_moving == int(spread.size):
        return ""
    if n_moving:
        frozen = np.flatnonzero(~moving).tolist()
        shown = ", ".join(str(i) for i in frozen[:10])
        more = "" if len(frozen) <= 10 else f" and {len(frozen) - 10} more"
        return (
            f"{len(frozen)} of {int(spread.size)} background column(s) carry no usable "
            f"spread (index {shown}{more}), so the draw never moves x along them and the "
            f"gap along them is exactly 0.0 for every model. The probe has no power over "
            f"those features, and a verdict that does not say so reads as a verdict about "
            f"all of them. Supply a background whose every column varies, or drop the "
            f"constant columns from both x and the background before probing"
        )
    n_features = int(spread.size)
    n_not_finite = int(n_features - np.count_nonzero(usable))
    relative = spread[usable] / np.maximum(magnitude[usable], np.finfo(float).tiny)
    largest = float(np.max(relative)) if relative.size else float("nan")
    return (
        f"not one of the {n_features} background column(s) carries enough spread for the "
        f"draw to leave the instance (the widest is {largest:.2e} of its own column's "
        f"magnitude over {bg.shape[0]} row(s), against a floor of "
        f"{_MIN_RELATIVE_SPREAD:.0e}"
        + (f"; {n_not_finite} column(s) are not a finite number" if n_not_finite else "")
        + "), so the perturbation cloud is copies of x and the gap is 0.0 for every model: "
        "the probe could not have fired on any scaffold"
    )


def _probe_gap(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_perturbations: int,
    seed: int,
    k_neighbors: int = 50,
) -> float:
    """One probe draw.

    Compares the model's mean prediction on synthetic Gaussian
    perturbations around *x* (which tend to leave the data manifold)
    against its mean prediction on *x*'s nearest on-manifold neighbours
    from the background. An honest model behaves similarly on both, the
    perturbations are just a noisier local sample, so the gap is small.
    A Slack-style OOD scaffold behaves differently off-manifold, opening
    a large gap.

    Using the *local* neighbours (not the global background mean) is what
    makes this discriminate scaffolding from the benign case where *x*
    simply sits in a high- or low-prediction region.
    """
    rng = np.random.default_rng(seed)
    sigma = background.std(axis=0) + 1e-9
    perturbed = rng.normal(loc=x, scale=sigma, size=(n_perturbations, x.shape[0]))

    # BGL5 A-xai-1, 2026-09-27. A gap of 0.0 is FINITE, so every finiteness
    # filter downstream counted a draw that never moved as a draw that measured
    # nothing happening. Measured on the scaffolded fixture (|x1 - x0| < 0.5),
    # same instance, same manifold, only the background changed:
    #     400 varying rows              -> 0.64  (the scaffold is found)
    #     400 copies of one of them     -> 0.0   BEFORE, nan NOW
    #     those copies + a 1e-12 jitter -> 0.0   BEFORE, nan NOW
    # The refusal is stated once, relative to the data's own magnitude, and both
    # public probes ask the same question of the background before they draw; this
    # is asked here too, by the SAME helper, so a direct caller of this private
    # helper cannot be handed the fabricated 0.0 and the two answers cannot
    # disagree. It is deliberately NOT a test on the realised cloud: the sigma
    # floor is an ABSOLUTE 1e-9, so a frozen background still moves the draw by
    # about 4e-9 (measured), which a tolerance on the deviation would have to
    # compare against an absolute number to catch. The background's spread
    # relative to its own magnitude is the scale-free question.
    deviation = float(np.max(np.abs(perturbed - x))) if perturbed.size else float("nan")
    if not np.isfinite(deviation) or _unperturbable_background(background):
        return float("nan")

    # x's on-manifold local reference: its k nearest background points.
    k = min(k_neighbors, len(background))
    dists = np.linalg.norm(background - x, axis=1)
    neighbors = background[np.argsort(dists)[:k]]

    return abs(_positive_mean(predict_fn, perturbed) - _positive_mean(predict_fn, neighbors))


def slack_adversarial_probe(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_perturbations: int = 200,
    threshold: float = 0.3,
    seed: int = 7,
) -> tuple[Optional[bool], str]:
    """Heuristic Slack-style probe (single draw).

    Sample perturbations around x; if the prediction distribution on
    perturbed inputs is materially different from on the background,
    that suggests an OOD scaffolding is being triggered to mask the
    real classifier from the explainer.

    Conservative: returns ``(True, reason)`` when the absolute mean
    prediction difference exceeds ``threshold``. The Engineer panel
    surfaces the flag; the run is not blocked.

    Three states, never two: ``True`` flagged, ``False`` measured and clean,
    ``None`` when the draw produced no finite gap and the probe therefore did
    not run.
    """
    # BGL5 A-xai-1, 2026-09-27. Both guards below sit ABOVE the draw, because the
    # comparison they protect is the last line of this function and a guard under
    # it cannot fire. Measured on the scaffolded fixture (|x1 - x0| < 0.5), x =
    # background[0], threshold 0.3:
    #     400 varying rows, threshold 0.3  -> (True, 'Slack probe: ... by 0.640 ...')
    #     400 copies of one row            -> BEFORE (False, ''), 0 warnings
    #                                         NOW   (None, 'COULD NOT CHECK: ...'), 1 warning
    #     those copies + a 1e-12 jitter    -> BEFORE (False, ''), 0 warnings
    #                                         NOW   (None, 'COULD NOT CHECK: ...'), 1 warning
    #     400 varying rows, threshold=nan  -> BEFORE (False, ''), 0 warnings, over a
    #                                         MEASURED gap of 0.640
    #                                         NOW   (None, 'COULD NOT CHECK: ...'), 1 warning
    # `gap > threshold` is False for a NaN THRESHOLD exactly as it was for a NaN
    # gap: the BGL3 fix below guarded one operand of the comparison its own
    # comment names and left the other, so a caller passing a threshold it had
    # itself failed to compute was told the explainer is clean.
    #
    # AND THE SAME DEFECT WITH THE SIGN FLIPPED, which is the other half of the
    # same bound. B4 tier-1 audit, 2026-09-30. `_probe_gap` returns `abs(...)`, so
    # every measured gap is >= 0 by construction and `gap > threshold` for a
    # NEGATIVE threshold cannot be False for any data in the statistic's range.
    # Measured on this tree, a 400-row real background, x = background[0], a flat
    # CLEAN model returning 0.4 everywhere, threshold -0.5:
    #     BEFORE (True, 'Slack probe: perturbation-mean prediction differs from
    #             background by 0.000 (threshold -0.5). Likely OOD-scaffolding
    #             detection. Treat SHAP/LIME outputs as suspect ...'), 0 warnings
    #     AFTER  (None, 'COULD NOT CHECK: ...'), 1 warning
    # A named scaffolding finding, with its remediation advice, off a gap of
    # exactly 0.000 on a model that has nothing to hide. A bound is unusable when
    # the comparison it controls cannot come out both ways, and that is a property
    # of the bound AND the statistic's range together, so `isfinite` alone leaves
    # the finite vacuous one open. The predicate and the wording are the twin's,
    # vfairness.evaluation.vfairness_metrics.explanation_diagnostics, fixed the
    # same day: two surfaces answering one question must not answer it twice.
    #
    # Zero is NOT refused, deliberately and after measuring it: `gap > 0.0` is
    # False for a gap of exactly 0.0, so the comparison is still falsifiable and a
    # caller asking for zero tolerance is asking for something coherent. It is
    # worth knowing that in practice the clean model above measures a gap of
    # 1.11e-16 rather than 0.0 and so does flag at threshold 0.0; that is the
    # caller's policy, not a fabricated verdict, and the twin draws the line in
    # the same place.
    if not np.isfinite(threshold) or float(threshold) < 0.0:
        unusable = (
            f"the threshold supplied is {threshold!r}, not a finite number, so no gap "
            "however large could exceed it"
            if not np.isfinite(threshold)
            else (
                f"the threshold supplied is {threshold!r}, and a probe gap is an absolute "
                "difference so it is never below zero: every gap exceeds this threshold, "
                "including a gap of exactly 0.0, so the comparison could not have come out "
                "clean for any model"
            )
        )
        warn_not_assessed(
            "slack_adversarial_probe",
            measured=0,
            total=1,
            unit="threshold(s) supplied are a usable bound",
            requirement=(
                "a finite threshold of at least zero, because `gap > threshold` cannot "
                "come out both ways otherwise"
            ),
            reporting="flag=None (could not check)",
            instead_of=(
                "flag=False, which asserts the explainer is clean, or flag=True over a gap of 0.0"
            ),
        )
        return None, (
            f"COULD NOT CHECK: {unusable}. This is not a finding about the explainer either way."
        )
    cannot_perturb = _unperturbable_background(background)
    if cannot_perturb:
        warn_not_assessed(
            "slack_adversarial_probe",
            measured=0,
            total=1,
            unit="probe draw(s) could leave the instance",
            requirement=f"a verdict needs one, and {cannot_perturb}",
            reporting="flag=None (could not check)",
            instead_of="flag=False, which asserts the explainer is clean",
        )
        return None, (
            f"COULD NOT CHECK: {cannot_perturb}. This is not a finding that the explainer is sound."
        )
    gap = _probe_gap(
        predict_fn=predict_fn,
        x=x,
        background=background,
        n_perturbations=n_perturbations,
        seed=seed,
    )
    # BGL3 xai-1, 2026-09-27. `gap > threshold` is False for a NaN gap, so a draw
    # that produced NOTHING fell through to the clean verdict. Measured here, a
    # flat clean model beside a model returning all-NaN predictions:
    #     clean  -> (False, '') and 0 warnings
    #     outage -> (False, '') and 0 warnings
    # byte-identical, so no caller could tell an explainer the probe had cleared
    # from one the probe never ran on. The multi-seed twin below took exactly this
    # fix as READINESS-6 on 2026-09-10 and this single-draw entry point, which is
    # exported from vfairness.xai.diagnostics in its own right, was left behind.
    if not np.isfinite(gap):
        warn_not_assessed(
            "slack_adversarial_probe",
            measured=0,
            total=1,
            unit="probe draw(s) produced a finite gap",
            requirement="a verdict needs one",
            reporting="flag=None (could not check)",
            instead_of="flag=False, which asserts the explainer is clean",
        )
        return None, (
            "COULD NOT CHECK: the probe draw produced no finite gap, so the probe "
            "never ran. This is not a finding that the explainer is sound."
        )
    if gap > threshold:
        return True, (
            f"Slack probe: perturbation-mean prediction differs from background by {gap:.3f} "
            f"(threshold {threshold}). Likely OOD-scaffolding detection. "
            "Treat SHAP/LIME outputs as suspect; corroborate with TreeSHAP or DiCE."
        )
    return False, ""


@dataclass(frozen=True)
class AdversarialProbeResult:
    """Confidence-bounded verdict from the multi-seed probe.

    ``flag`` is the majority decision; ``confidence`` is the fraction of
    seeds that agreed with it (a unanimous verdict scores 1.0). These map
    onto ``XaiDiagnostics.adversarial_flag`` / ``adversarial_reason``.
    """

    #: ``None`` when no seed produced a finite gap: the probe did not run,
    #: which is neither a clean explainer nor a flagged one. READINESS-6.
    flag: Optional[bool]
    confidence: float
    mean_gap: float
    fired_fraction: float
    reason: str


def multi_seed_adversarial_probe(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_perturbations: int = 200,
    threshold: float = 0.3,
    n_seeds: int = 10,
    base_seed: int = 7,
) -> AdversarialProbeResult:
    """Production probe (#P2-08): run the Slack draw across ``n_seeds``
    and bound the verdict by agreement.

    A single perturbation draw is noisy: one unlucky seed can flip the
    flag. Running ``n_seeds`` independent draws and reporting the
    majority verdict plus the agreement fraction turns the heuristic into
    a confidence-bounded diagnostic. On a clean model the gaps stay below
    ``threshold`` for every seed (flag False); on a Slack-style
    OOD-scaffolded model they exceed it for (nearly) every seed
    (flag True, confidence -> 1.0).
    """
    # BGL5 A-xai-1, 2026-09-27. The READINESS-6 filter below keeps every FINITE
    # gap, and a gap of exactly 0.0 is finite, so a draw that could not move
    # counted as a seed that measured nothing happening. Measured on the
    # scaffolded fixture (|x1 - x0| < 0.5), x = background[0], n_seeds=4:
    #     400 varying rows      -> flag True  confidence 1.0 mean_gap 0.7075
    #     400 copies of one row -> BEFORE flag False confidence 1.0 mean_gap 0.0,
    #                              fired 0.0, reason '', 0 warnings
    #                              NOW    flag None, confidence/mean_gap/fired nan,
    #                              reason 'COULD NOT CHECK: ...', 1 warning
    #     + a 1e-12 jitter      -> identical to the line above, before and after
    #     varying, threshold=nan-> BEFORE flag False confidence 1.0 mean_gap 0.7075
    #                              (a MEASURED 0.7075 reported as clean), 0 warnings
    #                              NOW    flag None, nan, 1 warning
    # flag False with confidence 1.0 is this function's own words for maximum
    # certainty that the explainer is clean, and both inputs above made it
    # impossible for the probe to say anything else.
    #
    # The twin in vfairness.evaluation.vfairness_metrics.explanation_diagnostics
    # carries the same defect and is owned by another batch; both copies refuse
    # the frozen background, and only this one refuses the 1e-12 jitter, because
    # the twin's guard tests `np.ptp(...) > 0.0` exactly.
    #
    # A THRESHOLD EVERY GAP CROSSES IS THE SAME DEFECT AS ONE NO GAP CROSSES.
    # B4 tier-1 audit, 2026-09-30. `_probe_gap` returns `abs(...)`, so every gap
    # is >= 0 and `gaps > threshold` for a NEGATIVE threshold cannot be False for
    # any data. Measured on this tree, 400-row real background, x = background[0],
    # n_seeds=4, threshold -0.5:
    #     the flat CLEAN model -> BEFORE flag True, confidence 1.0,
    #                             fired_fraction 1.0, mean_gap 0.0, ZERO warnings
    #                             AFTER  flag None, confidence/mean_gap/fired nan,
    #                             1 warning
    #     the scaffolded model -> BEFORE flag True, confidence 1.0, mean_gap 0.141
    #                             (right answer, unfalsifiable reasoning)
    # flag True with confidence 1.0 over a mean gap of 0.0 is this function's own
    # words for maximum certainty, spent on a verdict the data could not have
    # contradicted. The guard stays ABOVE the background dispatch below, where the
    # non-finite half already was, so no seed loop and no predict_fn call happens
    # for a bound that cannot decide anything. Predicate and wording are shared
    # with the twin in
    # vfairness.evaluation.vfairness_metrics.explanation_diagnostics, fixed the
    # same day: its comment named this copy, and this one named it back.
    if not np.isfinite(threshold) or float(threshold) < 0.0:
        unusable = (
            f"the threshold supplied is {threshold!r}, not a finite number, so no gap "
            "however large could exceed it"
            if not np.isfinite(threshold)
            else (
                f"the threshold supplied is {threshold!r}, and a probe gap is an absolute "
                "difference so it is never below zero: every gap exceeds this threshold, "
                "including a gap of exactly 0.0, so the comparison could not have come out "
                "clean for any model"
            )
        )
        warn_not_assessed(
            "multi_seed_adversarial_probe",
            measured=0,
            total=int(max(1, n_seeds)),
            unit="seed(s) could be judged against a usable threshold",
            requirement=(
                "a finite threshold of at least zero, because `gaps > threshold` cannot "
                "come out both ways otherwise"
            ),
            reporting="flag=None (could not check), confidence and mean_gap NaN",
            instead_of=(
                "flag=False with confidence 1.0, which asserts the explainer is clean, or "
                "flag=True with confidence 1.0 over a gap of 0.0"
            ),
        )
        return AdversarialProbeResult(
            flag=None,
            confidence=float("nan"),
            mean_gap=float("nan"),
            fired_fraction=float("nan"),
            reason=(
                f"COULD NOT CHECK: {unusable}. This is not a finding about the explainer "
                "either way."
            ),
        )
    cannot_perturb = _unperturbable_background(background)
    if cannot_perturb:
        warn_not_assessed(
            "multi_seed_adversarial_probe",
            measured=0,
            total=int(max(1, n_seeds)),
            unit="seed(s) could draw a cloud that leaves the instance",
            requirement=f"a verdict needs one, and {cannot_perturb}",
            reporting="flag=None (could not check), confidence and mean_gap NaN",
            instead_of="flag=False with confidence 1.0, which asserts the explainer is clean",
        )
        return AdversarialProbeResult(
            flag=None,
            confidence=float("nan"),
            mean_gap=float("nan"),
            fired_fraction=float("nan"),
            reason=(
                f"COULD NOT CHECK: {cannot_perturb}. This is not a finding that the "
                "explainer is sound."
            ),
        )
    gaps = np.array(
        [
            _probe_gap(
                predict_fn=predict_fn,
                x=x,
                background=background,
                n_perturbations=n_perturbations,
                seed=base_seed + i,
            )
            for i in range(max(1, n_seeds))
        ]
    )
    # READINESS-6, 2026-09-10. `gaps > threshold` is False for a NaN gap, so a
    # seed whose probe produced NOTHING counted as a seed that fired nothing.
    # With every seed unmeasurable that gave fired_fraction 0.0, flag False and
    # confidence 1.0, which asserts MAXIMUM certainty the explainer is clean
    # over a model that returned no predictions at all.
    #
    # Measured that day, a flat clean model against a model returning all-NaN:
    #     clean   -> flag=False confidence=1.0 mean_gap=0.0 reason='' 0 warnings
    #     outage  -> flag=False confidence=1.0 mean_gap=nan reason='' 0 warnings
    # The only difference is a mean_gap no consumer is obliged to read.
    #
    # This function exists in TWO independent copies, here and in the other
    # package, and both carried this. Fixed in both, and pinned to agree.
    measured = gaps[np.isfinite(gaps)]
    n_unmeasurable = int(gaps.size - measured.size)
    if measured.size == 0:
        warn_not_assessed(
            "multi_seed_adversarial_probe",
            measured=0,
            total=int(gaps.size),
            unit="seed(s) whose probe produced a finite gap",
            requirement="at least one",
            reporting="flag=None (could not check), confidence and mean_gap NaN",
            instead_of="flag=False with confidence 1.0, which asserts the explainer is clean",
        )
        return AdversarialProbeResult(
            flag=None,
            confidence=float("nan"),
            mean_gap=float("nan"),
            fired_fraction=float("nan"),
            reason=(
                "COULD NOT CHECK: none of the "
                f"{int(gaps.size)} seed(s) produced a finite gap, so the probe "
                "never ran. This is not a finding that the explainer is sound."
            ),
        )
    if n_unmeasurable:
        warn_not_assessed(
            "multi_seed_adversarial_probe",
            measured=int(measured.size),
            total=int(gaps.size),
            unit="seed(s) whose probe produced a finite gap",
            requirement="all of them, for the agreement fraction to mean what it says",
            reporting=f"a verdict over the {measured.size} that remain",
            instead_of="counting an unmeasurable seed as one that did not fire",
        )
    fired = measured > threshold
    fired_fraction = float(fired.mean())
    flag = fired_fraction >= 0.5
    confidence = fired_fraction if flag else 1.0 - fired_fraction
    mean_gap = float(measured.mean())
    if flag:
        reason = (
            f"Slack probe fired on {int(fired.sum())}/{len(measured)} seeds "
            f"(mean gap {mean_gap:.3f} > threshold {threshold}; confidence {confidence:.2f}). "
            "Likely OOD-scaffolding detection: treat SHAP/LIME outputs as suspect; "
            "corroborate with TreeSHAP or DiCE."
        )
    else:
        reason = ""
    return AdversarialProbeResult(
        flag=flag,
        confidence=round(confidence, 4),
        mean_gap=round(mean_gap, 6),
        fired_fraction=round(fired_fraction, 4),
        reason=reason,
    )
