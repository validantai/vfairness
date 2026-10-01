"""Paired inference for counterfactual runs (LF-20).

WHY THIS MODULE EXISTS
----------------------
A counterfactual run asks the SAME prompt twice, changing only the demographic
attribute. That is a matched-pair design, and the pairing is the whole point of
it: the prompt is held constant so that a difference in the two responses can be
charged to the attribute rather than to the prompt.

The library tested that data with :func:`scipy.stats.mannwhitneyu`, which is an
UNPAIRED rank-sum test. It throws the pairing away, and in doing so it charges
prompt-to-prompt variance, usually the largest source of spread in a template
set, to the group difference. It is both weaker than the design deserves and
answering a different null than the design poses.

So: McNemar on discordant pairs for binary per-pair outcomes (refused or not,
selected or not), and a paired sign-flip permutation for continuous per-pair
deltas (sentiment, toxicity, length).

WHAT EVERY FUNCTION HERE REFUSES TO DO
--------------------------------------
Return a number for a test that did not run. ``p_value`` is ``nan`` in that
case, never ``1.0``. This is not a style preference: ``_statistics._testable``
excludes a NaN p from a correction family and is BLIND to a finite ``1.0``
sentinel, because no mechanical check can tell a sentinel 1.0 from a measured
one. A test that could not run must not put a number in the family at all.

Report a "not significant" that the design could never have contradicted.
McNemar's evidence is only its DISCORDANT pairs, so five of them cannot reach
alpha = 0.05 however lopsided they are, and a sign-flip over four pairs cannot
go below 0.125. Both floors are checked through
:func:`~vfairness.evaluation.vfairness_metrics._statistics.detectability` before
a result is offered as testable.

Drop a pair silently. A pair is usable only when BOTH sides are usable, and how
many were dropped is on every record, because unreadable responses are
content-driven and dropping them makes the survivors a differently selected set.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .._not_assessed import NOT_ASSESSED
from ..evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_mcnemar,
    min_attainable_p_sign_flip,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NOT_ASSESSED",
    "PairedResult",
    "mcnemar_paired_test",
    "sign_flip_paired_test",
    "usable_binary_pairs",
    "usable_numeric_pairs",
]

#: The record every test here returns. A plain dict on purpose: it is merged
#: into result payloads that already travel as dicts, and a dataclass would just
#: be unpacked at every call site.
PairedResult = Dict[str, Any]


def _blank(test: str, n_pairs_supplied: int) -> PairedResult:
    """A record that has not been tested. Every numeric field is nan, not 0."""
    return {
        "test": test,
        "tested": False,
        "p_value": float("nan"),
        "statistic": float("nan"),
        "n_pairs_supplied": int(n_pairs_supplied),
        "n_pairs_usable": 0,
        "n_pairs_dropped": int(n_pairs_supplied),
        "min_attainable_p": None,
        "detectable": None,
        "state": NOT_ASSESSED,
        "reason": "",
    }


def _as_binary(value: Any) -> Optional[bool]:
    """A per-pair binary outcome, or None when the value is not one.

    Deliberately narrow. ``bool(x)`` would turn NaN into True, an empty string
    into False and any non-empty string into True, all of which are verdicts
    invented from unusable input.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value) in (0.0, 1.0):
            return bool(value)
    return None


def usable_binary_pairs(
    reference: Sequence[Any],
    variant: Sequence[Any],
) -> Tuple[list[bool], list[bool], int]:
    """The pairs where BOTH sides are a usable binary outcome, and how many were not.

    Pairing is positional, so a length mismatch is truncated to the shorter and
    the remainder counts as dropped: silently zipping to the shorter length
    would hide a producer that returned different numbers of responses per arm.
    """
    n_supplied = max(len(reference), len(variant))
    ref_out: list[bool] = []
    var_out: list[bool] = []
    for i in range(min(len(reference), len(variant))):
        a, b = _as_binary(reference[i]), _as_binary(variant[i])
        if a is None or b is None:
            continue
        ref_out.append(a)
        var_out.append(b)
    return ref_out, var_out, n_supplied


def usable_numeric_pairs(
    reference: Sequence[Any],
    variant: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """The pairs where BOTH sides are finite numbers, and how many were supplied."""
    n_supplied = max(len(reference), len(variant))
    ref_out: list[float] = []
    var_out: list[float] = []
    for i in range(min(len(reference), len(variant))):
        try:
            a, b = float(reference[i]), float(variant[i])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        ref_out.append(a)
        var_out.append(b)
    return np.asarray(ref_out, dtype=float), np.asarray(var_out, dtype=float), n_supplied


def mcnemar_paired_test(
    reference: Sequence[Any],
    variant: Sequence[Any],
    *,
    outcome: str = "outcome",
    alpha: float = 0.05,
    n_family: int = 1,
) -> PairedResult:
    """Exact two-sided McNemar over matched binary outcomes.

    Args:
        reference: Per-pair outcome for the reference arm.
        variant: Per-pair outcome for the varied arm, positionally paired.
        outcome: What is being counted, for the reason text ("refusal").
        alpha: The bar a single test must clear.
        n_family: How many hypotheses are corrected together. The design floor
            is checked against ``alpha / n_family``, so a test that could clear
            0.05 alone but not its share of a family is reported as such.

    Returns:
        A record whose ``p_value`` is ``nan`` unless ``tested`` is True. The
        discordant counts are always present, because they are the evidence and
        a reader needs them whether or not the test ran.
    """
    record = _blank("mcnemar_exact", max(len(reference), len(variant)))
    ref, var, n_supplied = usable_binary_pairs(reference, variant)
    record["n_pairs_supplied"] = n_supplied
    record["n_pairs_usable"] = len(ref)
    record["n_pairs_dropped"] = n_supplied - len(ref)

    # b: reference yes, variant no. c: reference no, variant yes. Concordant
    # pairs contribute nothing to McNemar and are counted only for the reader.
    b = sum(1 for a, x in zip(ref, var) if a and not x)
    c = sum(1 for a, x in zip(ref, var) if x and not a)
    record["n_discordant_reference_only"] = b
    record["n_discordant_variant_only"] = c
    record["n_discordant"] = b + c
    record["n_concordant"] = len(ref) - b - c

    if not ref:
        record["reason"] = (
            f"no pair had a usable {outcome} on both sides "
            f"({n_supplied} pair(s) supplied), so no test could run"
        )
        return record

    floor = min_attainable_p_mcnemar(b + c)
    record["min_attainable_p"] = floor
    detectable, note = detectability(floor, n_family=n_family, alpha=alpha)
    record["detectable"] = detectable
    if detectable is not True:
        record["reason"] = (
            f"the {outcome} comparison found {b + c} discordant pair(s) out of {len(ref)}. "
            f"McNemar's evidence is ONLY its discordant pairs, so the number of pairs does not "
            f"rescue this. {note}"
        )
        return record

    try:
        from scipy import stats as _st

        result = _st.binomtest(b, b + c, 0.5, alternative="two-sided")
        p_value = float(result.pvalue)
    except Exception as exc:
        logger.debug("mcnemar_paired_test: the exact binomial raised", exc_info=True)
        record["reason"] = f"the exact McNemar binomial raised {exc!r}, so nothing was tested"
        return record

    if not np.isfinite(p_value):
        record["reason"] = f"the exact McNemar binomial returned {p_value}, so nothing was tested"
        return record

    record.update(
        {
            "tested": True,
            "p_value": p_value,
            "statistic": float(b - c),
            "state": "",
            "reason": "",
        }
    )
    if record["n_pairs_dropped"]:
        # A measured result, on a subset. Saying so is the condition on which
        # reporting the subset is honest.
        record["reason"] = (
            f"measured on {len(ref)} of {n_supplied} pair(s): the rest had no usable "
            f"{outcome} on one or both sides. Responses are unusable for reasons that "
            f"depend on their content, so the pairs that remain are not a random subset."
        )
    return record


def sign_flip_paired_test(
    reference: Sequence[Any],
    variant: Sequence[Any],
    *,
    metric: str = "score",
    alpha: float = 0.05,
    n_family: int = 1,
    n_resamples: Optional[int] = None,
    seed: int = 0,
) -> PairedResult:
    """Paired sign-flip permutation over per-pair deltas.

    The null is that the sign of each pair's delta is exchangeable, which is
    exactly the null the design poses: if the attribute does nothing, flipping
    which arm is called "reference" changes nothing.

    Args:
        reference: Per-pair value for the reference arm.
        variant: Per-pair value for the varied arm, positionally paired.
        metric: What is being compared, for the reason text.
        alpha: The bar a single test must clear.
        n_family: Hypotheses corrected together, as in :func:`mcnemar_paired_test`.
        n_resamples: ``None`` enumerates exactly (only for small pair counts);
            an int samples that many sign vectors with the ``(count + 1) /
            (B + 1)`` estimator.
        seed: Fixed by default, so a run is reproducible. A sampled permutation
            test that cannot be reproduced cannot be audited.
    """
    record = _blank("sign_flip_permutation", max(len(reference), len(variant)))
    ref, var, n_supplied = usable_numeric_pairs(reference, variant)
    record["n_pairs_supplied"] = n_supplied
    record["n_pairs_usable"] = int(len(ref))
    record["n_pairs_dropped"] = int(n_supplied - len(ref))

    if len(ref) == 0:
        record["reason"] = (
            f"no pair had a finite {metric} on both sides "
            f"({n_supplied} pair(s) supplied), so no test could run"
        )
        return record

    deltas = var - ref
    observed = float(np.mean(deltas))
    record["mean_delta"] = observed

    if float(np.max(np.abs(deltas))) == 0.0:
        # Every pair identical. The test DID run in substance and its answer is
        # the largest p there is, which is a measurement rather than a sentinel.
        #
        # BGL3-LLM3 (2026-09-27). The DESIGN FLOOR IS CHECKED HERE TOO. This
        # branch used to hard-code "detectable": True and skip
        # ``detectability`` altogether, which is the one claim in the record
        # that identical data cannot support: whether the design had the power
        # to find a difference is a fact about the pair COUNT, not about what
        # the arms happened to contain. Measured over three identical pairs:
        #
        #   min_attainable_p 0.25, detectable True, tested True, p_value 1.0
        #
        # while the SAME three pairs carrying a real delta of 0.1 were refused
        # by the path below with detectable False and the NOT DETECTABLE note.
        # So an arm with nothing to find was credited with power that the same
        # arm was denied the moment there was something to find, and
        # ``operations/pulse/llm_probe.py`` copies ``detectable`` straight into
        # its published comparison record, where False is documented to mean
        # "this design could not have detected one".
        floor = min_attainable_p_sign_flip(len(ref), n_resamples)
        record["min_attainable_p"] = floor
        detectable, note = detectability(floor, n_family=n_family, alpha=alpha)
        record["detectable"] = detectable
        identical = f"every pair had an identical {metric}, so the delta is exactly zero"
        if detectable is not True:
            # Same disposition as the path below: a design that could not have
            # rejected contributes no p to the family. The zero delta is still
            # reported, in mean_delta and in the reason, because it is a fact
            # about the values that were read.
            record["reason"] = f"{identical}. {note}"
            return record
        record.update(
            {
                "tested": True,
                "p_value": 1.0,
                "statistic": 0.0,
                "state": "",
                "reason": identical,
            }
        )
        return record

    # Enumeration is only honest up to a point: 2**20 sign vectors is a million.
    exact = n_resamples is None
    if exact and len(ref) > 20:
        record["reason"] = (
            f"exact enumeration over {len(ref)} pairs would need 2**{len(ref)} sign vectors. "
            f"Pass n_resamples to sample instead; nothing was tested."
        )
        return record

    floor = min_attainable_p_sign_flip(len(ref), n_resamples)
    record["min_attainable_p"] = floor
    detectable, note = detectability(floor, n_family=n_family, alpha=alpha)
    record["detectable"] = detectable
    if detectable is not True:
        record["reason"] = (
            f"the paired {metric} comparison over {len(ref)} pair(s)"
            + (f" with {n_resamples} resamples" if n_resamples is not None else "")
            + f" could not reach {alpha} for any values. {note}"
        )
        return record

    try:
        if exact:
            n = len(ref)
            signs = 1.0 - 2.0 * ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(
                float
            )
            means = signs @ deltas / n
        else:
            rng = np.random.default_rng(seed)
            b = int(n_resamples or 0)
            signs = rng.choice([-1.0, 1.0], size=(b, len(ref)))
            means = signs @ deltas / len(ref)
        at_least_as_extreme = int(np.sum(np.abs(means) >= abs(observed) - 1e-12))
        total = int(means.shape[0])
        # The (count + 1) / (B + 1) estimator for the sampled case. Exact
        # enumeration already contains the observed assignment, so it is the
        # plain proportion there.
        p_value = (
            at_least_as_extreme / total if exact else (at_least_as_extreme + 1.0) / (total + 1.0)
        )
    except Exception as exc:
        logger.debug("sign_flip_paired_test: the permutation raised", exc_info=True)
        record["reason"] = f"the paired {metric} permutation raised {exc!r}, so nothing was tested"
        return record

    if not np.isfinite(p_value):
        record["reason"] = (
            f"the paired {metric} permutation returned {p_value}, so nothing was tested"
        )
        return record

    record.update(
        {
            "tested": True,
            "p_value": float(p_value),
            "statistic": observed,
            "state": "",
            "reason": "",
        }
    )
    if record["n_pairs_dropped"]:
        record["reason"] = (
            f"measured on {len(ref)} of {n_supplied} pair(s): the rest had no finite "
            f"{metric} on one or both sides. Responses are unusable for reasons that "
            f"depend on their content, so the pairs that remain are not a random subset."
        )
    return record
