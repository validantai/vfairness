"""
Explanation trustworthiness diagnostics.

Answers "how much should you trust this explanation?" alongside the attribution
itself, so every per-decision XAI result carries a quality signal rather than a
bare list of contributions. Three complementary, model-agnostic, numpy-only
checks:

  * faithfulness  -- ROAR-style removal-curve AUC (Hooker et al. 2019 NeurIPS;
    Yeh et al. 2019). Mask the top features by |attribution| and watch how fast
    the prediction degrades. A faithful explanation degrades it quickly.
    Normalised to [0, 1]; higher is better.
  * stability     -- mean per-feature sigma of the attribution across repeated
    runs (Alvarez-Melis & Jaakkola 2018). Lower is better (more reproducible).
  * adversarial   -- Slack et al. (2020) AIES OOD-scaffolding probe, run across
    several seeds and bounded by agreement. Flags when perturbation-based
    explainers (SHAP/LIME) may be looking at a different model than production.

These mirror the implementations in ``vfairness.xai.diagnostics`` but live here,
in the already-deployed evaluation package, so the per-decision handlers can use
them without pulling in the heavier xai explainer subsystem. Pure numpy.

Not imported from the package __init__:

    from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
        diagnose_local_attribution,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vfairness._not_assessed import warn_not_assessed

# numpy>=2 renamed trapz -> trapezoid; support both so this runs on any host.
# getattr for both names keeps this valid under numpy 2 stubs (which drop trapz).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def removal_curve_auc(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    background_mean: np.ndarray,
    n_steps: Optional[int] = None,
) -> float:
    """ROAR-style removal-curve AUC, normalised and CLAMPED to [0, 1]
    (higher = faithful).

    The raw AUC is the area under |p0 - p_masked| divided by |p0|. When
    masking swings the prediction PAST zero (or p0 is near zero) the drop
    can exceed |p0| and the ratio blows past 1, which would break the
    documented [0, 1] contract and the strong/moderate/weak grade bands
    downstream. Any drop of |p0| or more already means "prediction fully
    destroyed", so values above 1 carry no extra faithfulness signal and
    are clamped.

    Returns NaN when the masking ORDER does not exist, which is the case for three
    kinds of attribution vector: one where any entry is not a finite number (the
    explainer could not fill it in); one where every ``|attribution|`` is the same
    to within floating point tolerance (nothing separates the features); and one
    where features ARE tied in part and the score moves when the tied features are
    rearranged, which is measured rather than assumed. ``np.argsort`` answers the
    first two with the identity permutation, i.e. the caller's column order, so a
    score computed from either measures the arrangement and not the explanation.
    """
    attr = np.asarray(attributions, dtype=float).ravel()
    n_unattributed = int(np.count_nonzero(~np.isfinite(attr)))
    if n_unattributed:
        # BGL3 evaluation-4, 2026-09-27. THE ORDER IS THE MEASUREMENT HERE, and
        # a non-finite attribution is not a small contribution, it is a feature
        # this run could not attribute (the same reading
        # ``xai.explainers.shap_adapter._unattributed`` already documents).
        # ``np.abs(nan) > x`` is False for every x, so ``np.argsort`` sorts the
        # unattributable features LAST and, when EVERY entry is non-finite,
        # returns the identity permutation: the caller's own feature order. The
        # removal curve then scored the faithfulness of a ranking that does not
        # exist, and nothing said so.
        #
        # Measured on this repo before this guard, a linear margin model with
        # weights [0.5, -0.25, 1.0, 0.75], x = [1, 2, -1, 0.5] and a zero
        # background:
        #     attributions = [nan, nan, nan, nan] -> 0.725, 0 warnings
        #     attributions = w * x (the true ones) -> 1.0
        # 0.725 grades "strong" in ``_grade_faithfulness``, so an explanation
        # nobody could rank was reported as a highly faithful one.
        #
        # The guard sits ABOVE the argsort and above every ``predict_fn`` call,
        # so the twin in ``xai.diagnostics.faithfulness`` (which delegates here)
        # inherits it instead of the fabrication moving one caller along.
        warn_not_assessed(
            "removal_curve_auc",
            measured=int(attr.size - n_unattributed),
            total=int(attr.size),
            unit="attribution(s) are a finite number",
            requirement=(
                "the removal curve masks features in order of |attribution|, so it "
                "needs all of them to have an order at all"
            ),
            reporting="NaN",
            instead_of=(
                "a faithfulness score for whatever order np.argsort returned, which "
                "puts the unattributable features last and, when every entry is "
                "non-finite, is the caller's own feature order"
            ),
        )
        return float("nan")
    abs_attr = np.abs(attr)
    # BGL6 F04-1, 2026-09-29. THE TIE TEST WAS EXACT FLOAT EQUALITY, and one ulp
    # is not equality. ``np.unique`` saw TWO values in [1.0, 1.0, 1.0, 1.0+eps],
    # so the guard below did not fire, although the top THREE magnitudes were
    # still exactly equal and nothing ordered them but their position in the
    # caller's array: measured before this change, that vector scored 0.775,
    # 0.975 and 0.575 for three arrangements of the SAME data and the SAME
    # explanation, 2.2e-16 away from the vector that is refused outright.
    #
    # Tie blocks are therefore cut on DISTINGUISHABILITY, with a tolerance that
    # scales with the vector. Still never with a variance or == test on an
    # accumulated statistic: np.var of a constant array is exactly 0.0 only at
    # some n, so such a guard passes for a fixture and fails on real data.
    scale = float(np.max(abs_attr)) if abs_attr.size else 0.0
    atol = 1e-12 + 1e-9 * scale
    order = np.argsort(-abs_attr, kind="stable")
    ranked = abs_attr[order]
    separated = np.diff(ranked) < -atol if ranked.size > 1 else np.zeros(0, dtype=bool)
    block = np.concatenate([np.zeros(1, dtype=int), np.cumsum(separated)]).astype(int)
    n_blocks = int(block[-1]) + 1 if block.size else 0
    if attr.size > 1 and n_blocks == 1:
        # BGL5 A-evaluation-4, 2026-09-27. THE FINITENESS TEST WAS THE WRONG
        # TEST. What this function needs is an ORDER, and the guard above only
        # asked whether the numbers exist. ``np.argsort(-np.abs(attr))`` returns
        # the IDENTITY permutation for a CONSTANT vector exactly as it does for
        # an all-NaN one, so a tied explanation was scored in the caller's own
        # column order, silently.
        #
        # Measured on this repo before this guard, the same linear margin model
        # with weights [0.5, -0.25, 1.0, 0.75], x = [1, 2, -1, 0.5] and a zero
        # background:
        #     attributions = [0.0, 0.0, 0.0, 0.0]    -> 0.725, 0 warnings
        #     attributions = [1.0, 1.0, 1.0, 1.0]    -> 0.725, 0 warnings
        #     attributions = [-3.0, -3.0, -3.0, -3.0] -> 0.725, 0 warnings
        # 0.725 grades "strong" in ``_grade_faithfulness``, and it is a function
        # of the ARRANGEMENT and nothing else: permuting the columns to
        # [3, 2, 1, 0] (with x permuted alongside, so the data is identical)
        # turned the same tied explanation into 0.975.
        # After: nan, with the warning below.
        #
        # Tested with np.unique on the RAW |attribution| values, never with a
        # variance test: np.var of a constant array is exactly 0.0 only at some
        # n, so a variance guard passes for a fixture and fails on real data.
        # This package's own ranking._undefined_order_reason refuses the
        # identical situation in the identical words ("all N ranking scores are
        # identical, so no ranking order is defined").
        #
        # A PARTIAL tie is deliberately NOT refused HERE: one distinct value
        # still orders the features it separates, and the true attributions of
        # this repo's own control vector (w * x = [0.5, -0.5, -1.0, 0.375]) tie
        # two features at |0.5|, so refusing or warning there would withdraw a
        # faithfulness score that is measured and correct. What a partial tie
        # gets instead is the ambiguity MEASUREMENT below, which withdraws the
        # score only when the arrangement demonstrably moves it.
        warnings.warn(
            f"removal_curve_auc: all {attr.size} |attribution| values are identical "
            f"to within floating point tolerance ({float(ranked[0])} to "
            f"{float(ranked[-1])}, tolerance {atol:.3g}), so there is no order of "
            f"|attribution| to mask features BY, and the order IS the measurement "
            f"here. Returning NaN (could not check), NOT a faithfulness score for "
            f"np.argsort's identity permutation, which is the caller's own column "
            f"order: the same tied vector measured 0.725 in one column order and "
            f"0.975 in another.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    n_features = x.shape[-1]
    n_steps = n_features if n_steps is None else min(n_steps, n_features)

    def _auc_for(masking_order: np.ndarray) -> float:
        masked = x.astype(float).copy()
        p0 = float(np.asarray(predict_fn(masked.reshape(1, -1))).ravel()[0])
        drops = [0.0]
        for i in range(n_steps):
            masked[masking_order[i]] = background_mean[masking_order[i]]
            p = float(np.asarray(predict_fn(masked.reshape(1, -1))).ravel()[0])
            drops.append(abs(p0 - p))
        raw = float(_trapz(drops, dx=1.0 / max(1, n_steps)) / max(abs(p0), 1e-9))
        if not np.isfinite(raw):
            # A curve that could not be computed is NOT "maximally unfaithful".
            # The clamp below turns nan into 0.0 (`nan > 0.0` is False, so `max`
            # returns the 0.0), and 0.0 is a real score that
            # ``_grade_faithfulness`` grades "weak". Measured 2026-09-10: an empty
            # background makes ``background_mean`` nan, every masked prediction
            # nan and the AUC nan, and the caller was handed
            # ``faithfulness: 0.0, faithfulness_grade: "weak"`` for an explanation
            # nobody measured. nan says nothing was measured; the [0, 1] clamp
            # still applies to every finite value.
            return float("nan")
        return float(min(1.0, max(0.0, raw)))

    auc = _auc_for(order)
    if n_blocks < int(ranked.size):
        # BGL6 F04-1, 2026-09-29. A tie block wider than one feature means the
        # masking sequence inside it is chosen by the caller's column order and
        # by nothing in the explanation, so whether the published number is a
        # measurement is itself a question to be MEASURED rather than assumed.
        # The same curve is walked again with every tie block reversed, which is
        # the opposite extreme of the arrangements argsort could have returned:
        # equal under both means the number does not depend on the arrangement,
        # different means it does and the difference IS the fabrication.
        flipped = np.concatenate([order[block == b][::-1] for b in range(n_blocks)])
        sizes = [int(np.count_nonzero(block == b)) for b in range(n_blocks)]
        n_tied = int(sum(size for size in sizes if size > 1))
        alt = _auc_for(flipped)
        both_unmeasurable = not np.isfinite(auc) and not np.isfinite(alt)
        if not both_unmeasurable and (
            not np.isfinite(auc) or not np.isfinite(alt) or abs(auc - alt) > 1e-9
        ):
            warnings.warn(
                f"removal_curve_auc: {n_tied} of "
                f"{int(ranked.size)} |attribution| values are tied, and the score "
                f"MOVES with the arrangement of the tied features ({auc} in the "
                f"caller's column order, {alt} with the tied blocks reversed, on "
                f"identical data and an identical explanation). Returning NaN "
                f"(could not check), NOT either arm: nothing in the explanation "
                f"chooses between them.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
    return auc


def attribution_stability(reruns: List[np.ndarray]) -> float:
    """Mean per-feature standard deviation across repeated runs (lower = stable)."""
    if len(reruns) < 2:
        # 0.0 sigma is PERFECTLY stable, the best score on this scale, and it
        # was what a caller got for supplying nothing to compare. Stability is
        # variation ACROSS runs; one run has none to measure.
        warnings.warn(
            f"attribution_stability: {len(reruns)} rerun(s) supplied, and stability is "
            f"variation ACROSS runs, so nothing was measured. Returning nan, not 0.0, "
            f"which would read as perfectly stable.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    sigmas = np.vstack(reruns).std(axis=0)
    n_unmeasurable = int(np.count_nonzero(~np.isfinite(sigmas)))
    result = float(sigmas.mean()) if sigmas.size else float("nan")
    if not np.isfinite(result):
        # BGL3 evaluation-4, 2026-09-27. The VALUE was already honest: one
        # non-finite per-feature sigma makes the mean NaN, and NaN is not 0.0.
        # It was SILENT, though, so a caller reading only the number could not
        # tell it from a run that raised nothing because nothing was wrong.
        # Measured before this warning: attribution_stability([[nan, 1.0, 2.0],
        # [nan, 1.1, 1.9]]) -> nan with 0 warnings, and the all-NaN pair the
        # same. The single-rerun branch above has warned since 2026-09-10; these
        # two paths reach the same NaN by a different route and said nothing.
        #
        # The count is the useful part: it names how much of the vector had a
        # sigma at all. Deliberately NOT averaged over the measurable features
        # only. Dropping a feature can only move this scale DOWNWARD, and lower
        # is "more reproducible", so a partial mean would report a more stable
        # explanation than was measured, in a float with no field beside it to
        # carry the coverage.
        warn_not_assessed(
            "attribution_stability",
            measured=int(sigmas.size - n_unmeasurable),
            total=int(sigmas.size),
            unit="feature(s) have a finite standard deviation across the reruns",
            requirement=(
                "the score is the MEAN over every feature, so one unmeasurable "
                "feature leaves the mean undefined"
            ),
            reporting="nan",
            instead_of="a number covering only part of the attribution vector",
        )
        return float("nan")
    return result


def _positive_mean(predict_fn: Callable[[np.ndarray], np.ndarray], arr: np.ndarray) -> float:
    out = np.asarray(predict_fn(arr)).reshape(len(arr), -1)
    return float(out[:, -1].mean())


#: IMPORTED from the twin rather than re-declared, because two copies of a
#: threshold is how these two guards came to disagree in the first place. A single
#: definition means a change to the floor reaches both surfaces or neither.
from vfairness.xai.diagnostics.adversarial import (  # noqa: E402
    _MIN_RELATIVE_SPREAD,
)


def _background_cannot_be_perturbed(background: np.ndarray) -> Optional[str]:
    """Why this background makes the probe's own comparison impossible.

    BGL5 A-evaluation-4, 2026-09-27. ``_probe_gap`` draws its perturbation cloud
    as ``rng.normal(loc=x, scale=background.std(axis=0) + 1e-9)``. That ``+ 1e-9``
    is a division guard, and it silently turns a background with NO SPREAD into a
    usable-looking scale: every "perturbed" point is then x itself, x's nearest
    neighbours are x as well, and the gap is EXACTLY 0.0 by construction for any
    model whatsoever. A check whose result cannot depend on the data is the
    design-power shape this package already refuses elsewhere.

    Measured before this guard, x = [1.0, 2.0], background = np.tile(x, (30, 1)),
    n_perturbations=200, n_seeds=6, under warnings.simplefilter("error"):

        a blatantly SCAFFOLDED model (0.95 off the manifold, 0.05 on it)
            before -> AdversarialProbeResult(flag=False, confidence=1.0,
                      mean_gap=0.0, fired_fraction=0.0, reason=''), 0 warnings
        the clean flat model, same background
            before -> byte-identical to the line above

    flag=False with confidence 1.0 is the pair this function's own comment names
    as the thing that must never be produced. It ran in both directions too: with
    a ONE-row background the cloud collapses as well and the same scaffolded model
    came back flag=True, confidence=1.0, mean_gap=0.9, also silently.

    After: flag=None, confidence/mean_gap/fired_fraction NaN, a reason opening
    COULD NOT CHECK, and a UserWarning.

    Uses ``np.ptp``, a subtraction of two values that are actually in the data,
    never ``np.var(...) == 0``: an accumulated variance is exactly 0.0 only for
    some n and some values, so a variance guard passes for a fixture and fails on
    real data. Returns None when the background CAN be perturbed.
    """
    bg = np.asarray(background, dtype=float)
    if bg.ndim != 2 or bg.shape[0] == 0 or bg.shape[1] == 0:
        return (
            f"the background is not a usable 2D sample (shape {bg.shape}), so there "
            f"is no cloud to draw around x and nothing to compare it against"
        )
    # RELATIVE to each column's own magnitude, not an equality with zero, and
    # this twin was the one left behind. Two fix agents closed the same defect in
    # the same hour on 2026-09-27, one here and one in
    # xai/diagnostics/adversarial.py, and they chose different tests: that file
    # took a relative floor and this one kept `spread > 0.0`. Measured on the same
    # three backgrounds, 50 rows of 3 columns:
    #
    #   frozen (50 copies of one row)   both refuse
    #   frozen + a 1e-12 jitter         xai refuses, THIS ONE DID NOT
    #   a real sample                   neither refuses
    #
    # A background one part in 1e12 away from constant is not exactly zero: the
    # per-column spread measured 4.0e-12 to 5.3e-12 against column magnitudes of
    # 1.3 to 2.5, so the exact test said "this background has spread" while
    # `sigma = background.std(axis=0) + 1e-9` still collapsed the cloud onto x and
    # the gap came back 0.0 for any model, including a deliberately scaffolded
    # one. Two sibling guards disagreeing about one input is the shape that let
    # the original hole survive, so the floor is shared rather than re-derived.
    spread = np.ptp(bg, axis=0)
    magnitude = np.max(np.abs(bg), axis=0)
    n_features = int(spread.size)
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
        return None
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
    n_not_finite = int(n_features - np.count_nonzero(usable))
    relative = spread[usable] / np.maximum(magnitude[usable], np.finfo(float).tiny)
    largest = float(np.max(relative)) if relative.size else float("nan")
    return (
        f"not one of the {n_features} background column(s) carries enough spread for "
        f"the probe to perturb x along (the widest is {largest:.2e} of its own "
        f"column's magnitude over {bg.shape[0]} row(s), against a floor of "
        f"{_MIN_RELATIVE_SPREAD:.0e}"
        + (f"; {n_not_finite} are not a finite number" if n_not_finite else "")
        + "), so the perturbation cloud is copies of x and the gap is 0.0 for every "
        "model: the check could not have fired for any data"
    )


def _probe_gap(predict_fn, x, background, n_perturbations, seed, k_neighbors=50) -> float:
    rng = np.random.default_rng(seed)
    sigma = background.std(axis=0) + 1e-9
    perturbed = rng.normal(loc=x, scale=sigma, size=(n_perturbations, x.shape[0]))
    k = min(k_neighbors, len(background))
    dists = np.linalg.norm(background - x, axis=1)
    neighbors = background[np.argsort(dists)[:k]]
    return abs(_positive_mean(predict_fn, perturbed) - _positive_mean(predict_fn, neighbors))


@dataclass(frozen=True)
class AdversarialProbeResult:
    #: ``None`` when no seed produced a finite gap: the probe did not run, which
    #: is neither a clean explainer nor a flagged one. READINESS-6, 2026-09-10.
    #: The twin of this class in xai/diagnostics/adversarial.py carries the same
    #: change; the two implementations are independent and both had the defect.
    flag: Optional[bool]
    confidence: float
    mean_gap: float
    fired_fraction: float
    reason: str
    #: When ``flag is None``, the CAUSE on its own, as a clause a consumer can
    #: embed in its own sentence. Added BGL5 A-evaluation-4 because
    #: ``diagnose_local_attribution`` was writing its own guess at the cause
    #: ("none of its N seed(s) produced a finite gap") for every refusal,
    #: including the ones that have nothing to do with the seeds. Empty string
    #: whenever the probe did run.
    not_run_because: str = ""


def multi_seed_adversarial_probe(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_perturbations: int = 200,
    threshold: float = 0.3,
    n_seeds: int = 8,
    base_seed: int = 7,
) -> AdversarialProbeResult:
    """Slack OOD-scaffolding probe across seeds, bounded by agreement.

    Refused, with ``flag=None``, when the background carries no spread for the
    perturbation cloud to use: the gap is then 0.0 by construction for every
    model and the probe cannot fire for any data. See
    :func:`_background_cannot_be_perturbed` for the measured before and after.

    Refused the same way when ``threshold`` is not a usable bound: NaN or
    infinite, so no gap however large can exceed it, or negative, so every gap
    exceeds it and the probe can never come out clean. Both are comparisons that
    cannot come out both ways, which is could-not-check and not a verdict. See
    the guard at the top of the body for the measured numbers.
    """
    # A THRESHOLD NO GAP CAN CROSS, AND A THRESHOLD EVERY GAP CROSSES.
    #
    # BGL-W4, 2026-09-30. THE SIBLING DOOR, and the twin's own comment named it:
    # "The twin in vfairness.evaluation.vfairness_metrics.explanation_diagnostics
    # carries the same defect and is owned by another batch"
    # (xai/diagnostics/adversarial.py). That twin refuses a non-finite threshold
    # at the top of its body; this copy had no threshold test at all.
    #
    # `measured > threshold` is False for EVERY gap when the threshold is NaN or
    # +inf, so fired_fraction is 0.0, flag is False and confidence is 1.0, which
    # this function's own comments call maximum certainty that the explainer is
    # clean. Measured on a 400-row real background, x = background[0], n_seeds=4,
    # a blatantly scaffolded model (0.9 off the manifold, 0.1 on it):
    #
    #   threshold=nan   -> flag False, confidence 1.0, mean_gap 0.8, fired 0.0,
    #                      0 warnings, while the twin returned flag None + 1 warning
    #   threshold=+inf  -> identical
    #   the clean flat model, threshold=nan -> flag False, confidence 1.0,
    #                      mean_gap 0.0: the same verdict as the scaffolded one
    #
    # A MEASURED gap of 0.8 was published as a clean explainer. That is the
    # READINESS-6 defect this function already records for a NaN GAP, arriving
    # through the other operand.
    #
    # AND THE OTHER DIRECTION IS THE SAME DEFECT WITH THE SIGN FLIPPED, which
    # neither twin refused. `_probe_gap` returns `abs(...)`, so every measured gap
    # is >= 0 by construction, and `gap > threshold` for a NEGATIVE threshold
    # cannot be False for any data in the statistic's range. Measured on the same
    # background with threshold=-0.5: the CLEAN flat model came back flag True,
    # confidence 1.0, fired_fraction 1.0 over a mean_gap of 0.0, an unfalsifiable
    # positive finding. A bound is unusable when the comparison it controls cannot
    # be false, which is a property of the bound AND the statistic's range
    # together, so testing only `isfinite` leaves the finite vacuous one open.
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
            not_run_because=unusable,
        )

    # THE GUARD SITS ABOVE THE SEED LOOP, so no predict_fn call and no gap is
    # made from a cloud that is just copies of x.
    unperturbable = _background_cannot_be_perturbed(background)
    if unperturbable is not None:
        warn_not_assessed(
            "multi_seed_adversarial_probe",
            measured=0,
            total=int(max(1, n_seeds)),
            unit="seed(s) could perturb x away from the background at all",
            requirement=unperturbable,
            reporting="flag=None (could not check), confidence and mean_gap NaN",
            instead_of=(
                "flag=False with confidence 1.0, which asserts the explainer is clean, "
                "or flag=True from a cloud that never left x"
            ),
        )
        return AdversarialProbeResult(
            flag=None,
            confidence=float("nan"),
            mean_gap=float("nan"),
            fired_fraction=float("nan"),
            reason=(
                f"COULD NOT CHECK: {unperturbable}, so the probe never ran. This is "
                "not a finding that the explainer is sound."
            ),
            not_run_because=unperturbable,
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
            not_run_because=(
                f"none of its {int(gaps.size)} seed(s) produced a finite gap, so no "
                f"comparison was ever made"
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
    reason = (
        (
            f"Slack probe fired on {int(fired.sum())}/{len(measured)} seeds "
            f"(mean gap {mean_gap:.3f} > threshold {threshold}; confidence {confidence:.2f}). "
            "Treat SHAP/LIME outputs as suspect; corroborate with TreeSHAP or DiCE."
        )
        if flag
        else ""
    )
    return AdversarialProbeResult(
        flag=flag,
        confidence=round(confidence, 4),
        mean_gap=round(mean_gap, 6),
        fired_fraction=round(fired_fraction, 4),
        reason=reason,
    )


# The Slack probe compares a perturbed cloud around ``x`` against ``x``'s
# nearest background neighbours. Below this many background rows there is
# nothing to compare it against and the probe does not run.
#
# A ROW COUNT IS NOT THE WHOLE QUESTION (BGL5 A-evaluation-4, 2026-09-27).
# Thirty IDENTICAL rows clear this gate while carrying nothing to compare
# against, and that input published adversarial_flag False with confidence 1.0
# for a blatantly scaffolded model. The spread question is asked inside
# ``multi_seed_adversarial_probe`` (see _background_cannot_be_perturbed), above
# its own seed loop, so this consumer inherits the refusal through the
# ``probe.flag is None`` branch below rather than carrying a second copy of the
# rule that could drift from it.
_MIN_PROBE_BACKGROUND = 5


def _record_probe_not_run(out: Dict[str, Any], why: str) -> None:
    """Record that the adversarial probe did NOT run, in fields that survive.

    ``adversarial_flag = False`` is a verdict: the probe ran and found no
    scaffolding. It was also what a caller got when the probe never ran at
    all, so an empty background produced the same negative a real 120-row
    probe produces on a clean model, silently and with no note. The flag is
    tri-state now: True (probe fired), False (probe ran and cleared the
    model), None (COULD NOT CHECK).

    The state is written into ``adversarial_flag``, ``adversarial_reason`` and
    ``notes``, which are the three fields ``vfairness.xai.schemas``
    ``XaiDiagnostics`` actually has. ``adversarial_confidence`` is NOT one of
    them, so a consumer that builds that dataclass from this dict drops it,
    and a could-not-check carried only there never reaches the reader.
    """
    out["adversarial_flag"] = None
    out["adversarial_confidence"] = None
    out["adversarial_reason"] = f"COULD NOT CHECK: adversarial probe did not run ({why})."
    out["notes"].append(f"adversarial probe did not run ({why})")
    warnings.warn(
        f"diagnose_local_attribution: the adversarial probe did not run ({why}), so "
        f"adversarial_flag is None (could not check), not False, which would read as "
        f"'the probe ran and found no scaffolding'.",
        UserWarning,
        stacklevel=3,
    )


def _grade_faithfulness(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    if v >= 0.5:
        return "strong"
    if v >= 0.2:
        return "moderate"
    return "weak"


def diagnose_local_attribution(
    predict: Callable[[np.ndarray], np.ndarray],
    x_row: Sequence[float],
    attributions: Sequence[float],
    background: Any,
    reruns: Optional[List[Sequence[float]]] = None,
    n_steps: int = 8,
    n_seeds: int = 6,
) -> Dict[str, Any]:
    """Trustworthiness diagnostics for one local attribution.

    Returns a JSON-safe dict matching the platform XaiDiagnostics shape:
    ``faithfulness`` (0..1, higher better), ``stability`` (>=0, lower better),
    ``adversarial_flag`` / ``adversarial_reason`` / ``adversarial_confidence``,
    and ``notes``. Each metric degrades to None on failure; this never raises,
    so a diagnostics problem never breaks the underlying attribution.

    ``adversarial_flag`` has THREE states, never two: ``True`` the probe fired,
    ``False`` the probe ran and cleared the model, ``None`` COULD NOT CHECK, no
    probe ran (see :func:`_record_probe_not_run`, which also fills
    ``adversarial_reason`` and ``notes`` so the state survives the mapping onto
    ``XaiDiagnostics``). ``faithfulness_grade`` is likewise ``None`` whenever
    ``faithfulness`` is: an ungraded explanation is not a weak one.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: explanation_diagnostics. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    x = np.asarray(x_row, dtype=float).ravel()
    bg = np.asarray(background, dtype=float)
    attr = np.asarray(attributions, dtype=float).ravel()
    out: Dict[str, Any] = {
        "faithfulness": None,
        "faithfulness_grade": None,
        "stability": None,
        # None, not False. False is the verdict "the probe ran and found no
        # scaffolding"; this dict is built before any probe has run.
        "adversarial_flag": None,
        "adversarial_reason": None,
        "adversarial_confidence": None,
        "notes": [],
    }

    try:
        auc = removal_curve_auc(predict, x, attr, bg.mean(axis=0), n_steps=n_steps)
        if np.isfinite(auc):
            out["faithfulness"] = round(auc, 4)
            out["faithfulness_grade"] = _grade_faithfulness(out["faithfulness"])
        else:
            # Nothing to grade. Leaving both None keeps "not measured" distinct
            # from the measured 0.0 that grades "weak".
            out["notes"].append("faithfulness unavailable (removal curve not computable)")
    except Exception as exc:  # pragma: no cover - best-effort
        out["notes"].append(f"faithfulness unavailable ({type(exc).__name__})")

    try:
        if reruns and len(reruns) >= 2:
            out["stability"] = round(
                attribution_stability([np.asarray(r, dtype=float).ravel() for r in reruns]), 6
            )
    except Exception as exc:  # pragma: no cover - best-effort
        out["notes"].append(f"stability unavailable ({type(exc).__name__})")

    n_background = int(bg.shape[0]) if bg.ndim == 2 else 0
    try:
        if n_background >= _MIN_PROBE_BACKGROUND:
            probe = multi_seed_adversarial_probe(predict, x, bg, n_seeds=n_seeds)
            if probe.flag is None:
                # BGL3 evaluation-4, 2026-09-27. ``bool(None)`` is False, so the
                # ONE tri-state this function documents was collapsed back to two
                # on the only path where the probe actually runs.
                # ``multi_seed_adversarial_probe`` was given flag=None in
                # READINESS-6 precisely so a probe that measured nothing could
                # say so, and this line overwrote it with the verdict "the probe
                # ran and found no scaffolding".
                #
                # Measured before this branch, a model returning all-NaN
                # predictions over a 6-row background, n_seeds=3:
                #     adversarial_flag False, confidence nan, notes []
                # against a genuinely clean model:
                #     adversarial_flag False, confidence 1.0,  notes []
                # The two differ only in a confidence NaN and a reason string,
                # and ``XaiDiagnostics`` has no confidence field at all (see
                # _record_probe_not_run), so the reader got the same False.
                # BGL5 A-evaluation-4, 2026-09-27. The cause is the PROBE's to
                # state, not this function's to guess. This line used to write
                # "none of its N seed(s) produced a finite gap" for every refusal,
                # and once the probe learned to refuse a background with no spread
                # that sentence was a cause nobody measured, in the field a reader
                # acts on. Measured on the zero-spread background below, before:
                # adversarial_reason named the seeds, which HAD all produced a
                # finite gap of exactly 0.0.
                _record_probe_not_run(
                    out,
                    probe.not_run_because
                    or f"the probe reported it did not run over {n_seeds} seed(s)",
                )
            else:
                out["adversarial_flag"] = bool(probe.flag)
                out["adversarial_reason"] = probe.reason or None
                out["adversarial_confidence"] = probe.confidence
        else:
            # The skip was silent and left the fabricated False standing.
            _record_probe_not_run(
                out,
                f"the background holds {n_background} usable row(s), fewer than "
                f"{_MIN_PROBE_BACKGROUND}",
            )
    except Exception as exc:  # pragma: no cover - best-effort
        _record_probe_not_run(out, f"it raised {type(exc).__name__}")

    return out
