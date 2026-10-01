"""vfairness.vision -- image / text-to-image fairness.

Two layers, deliberately separated for honest verifiability:

1. Representation fairness MATH (pure numpy, no models, fully testable):
   MaxSkew / MinSkew / NDKL (Geyik et al. 2019) and bias-amplification
   (Seshadri et al. 2023) over demographic LABELS. These work on any
   list of group labels -- whatever produced them.

2. Demographic CLASSIFICATION (FairFace / CLIP). Requires torch + model
   weights, which the Python-3.14 consumer cannot load. It runs in the
   Python-3.9 vision sidecar (mirrors the existing ML-sidecar pattern via
   env vars). When the sidecar / models are unavailable, classification
   degrades to ``available: False`` with a clear reason and a metadata-
   only fallback -- it NEVER fabricates demographics.

The skew/NDKL stats are exported and verified; the model layer is an
explicit, isolated, optional step.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np

# Representation fairness math (Geyik, Ambler, Kenthapadi & Mehrotra 2019;
# Seshadri, Singh & Elazar 2023). Pure numpy, no models, no infra.


def _dist(labels: Sequence[str]) -> Dict[str, float]:
    labels = [str(x) for x in labels if x is not None]
    n = len(labels)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    for x in labels:
        out[x] = out.get(x, 0.0) + 1.0
    return {k: v / n for k, v in out.items()}


def skew(labels: Sequence[str], reference: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Per-group Skew_g = ln(observed_g / desired_g), plus MaxSkew/MinSkew.

    ``reference`` is the desired distribution (default: uniform over the
    observed groups). Returns per-group skew and the max/min: the standard
    fairness-in-retrieval / generated-set diversity measure.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: representation_skew. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    obs = _dist(labels)
    if not obs:
        return {"available": False, "reason": "no labels"}

    # A SINGLE OBSERVED GROUP AND NO REFERENCE IS NOT A MEASUREMENT.
    #
    # 2026-09-11. The default reference below is uniform over the groups that
    # were OBSERVED. With one observed group that reference is {g: 1.0}, the
    # observed share is also 1.0, and the skew is exactly ln(1) = 0.0: the best
    # attainable score, with no warning.
    #
    # Measured before this fix, through the grading function a caller actually
    # reads:
    #
    #     skew(["White_male"] * 100)          -> severity 'pass'
    #     skew(["a"]*40 + ["b"]*35 + ["c"]*25) -> severity 'warn'
    #     skew(["a"]*90 + ["b"]*5  + ["c"]*5)  -> severity 'critical'
    #
    # So a set of 100 images containing ONE demographic scored BETTER than a
    # genuinely balanced set. Not merely a false all-clear: inverted, with the
    # most homogeneous possible input receiving the best possible grade, which
    # is the exact opposite of what a representation-fairness measure is for.
    #
    # The measure is not broken; the DEFAULT is vacuous. Skew is meaningful only
    # against a desired distribution that says something, and "uniform over
    # whatever I happened to see" says nothing when only one thing was seen.
    # Supplying a real reference works correctly and is the fix a caller wants:
    # the same 100 images against a four-group reference grade 'critical' and
    # name the three absent groups.
    if reference is None and len(obs) < 2:
        only = next(iter(obs))
        warnings.warn(
            f"skew: every observation is {only!r} and no reference distribution was "
            f"supplied, so there is nothing to compare against. The default reference "
            f"is uniform over the OBSERVED groups, which a single group satisfies by "
            f"definition and would score a perfect 0.0. Returning available=False. "
            f"A homogeneous set is not evidence of fair representation; supply the "
            f"desired distribution and this becomes measurable.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "available": False,
            "reason": (
                f"only one group was observed ({only!r}) and no reference distribution "
                f"was supplied, so representation skew is undefined. This is NOT a pass."
            ),
            "observed": {k: round(v, 4) for k, v in obs.items()},
            "perGroup": {only: None},
            "absentGroups": [],
            "unreferencedGroups": [],
        }

    groups = sorted(set(obs) | set(reference or {}))
    ref = reference or {g: 1.0 / len(groups) for g in groups}
    # READINESS-6, 2026-09-10. `max(x, 1e-9)` on BOTH sides of this log
    # manufactures a magnitude. Skew_g = ln(observed/desired) is genuinely
    # -infinity for a group that appears ZERO times, and +infinity for a group
    # the reference gives zero share. The clamp replaced each infinity with a
    # finite number that is a pure artefact of the epsilon: at a desired share
    # of 0.5, an absent group scored ln(1e-9 / 0.5) = -20.03, and had the
    # epsilon been 1e-6 the same absence would have scored -13.1.
    #
    # The DIRECTION was right, which is what made it survive: the absent group
    # really is the most underrepresented one. The MAGNITUDE was invented, and
    # `maxSkew` / `minSkew` are graded numbers, so a threshold on them was being
    # compared against the choice of epsilon.
    #
    # Three states. A group that is absent, or that the reference excludes, gets
    # None and is named in its own list, because "this group never appeared" is
    # a stronger and more useful finding than any number, and it is not the same
    # claim as "this group appeared 1e-9 times".
    per: Dict[str, Optional[float]] = {}
    absent: List[str] = []
    unreferenced: List[str] = []
    for g in groups:
        o = obs.get(g, 0.0)
        d = ref.get(g, 0.0)
        if o <= 0.0:
            per[g] = None
            absent.append(g)
        elif d <= 0.0:
            per[g] = None
            unreferenced.append(g)
        else:
            per[g] = round(math.log(o / d), 4)

    measured = {g: v for g, v in per.items() if v is not None}
    if not measured:
        return {
            "available": False,
            "reason": (
                "no group has both a non-zero observed share and a non-zero desired "
                "share, so no skew is defined for any of them"
            ),
            "perGroup": per,
            "absentGroups": absent,
            "unreferencedGroups": unreferenced,
            "observed": {k: round(v, 4) for k, v in obs.items()},
        }

    result: Dict[str, Any] = {
        "available": True,
        "perGroup": per,
        "observed": {k: round(v, 4) for k, v in obs.items()},
        # Over the MEASURED groups only. An absent group is not a large negative
        # skew, it is an absence, and it is reported as one below.
        "maxSkew": round(max(measured.values()), 4),
        "minSkew": round(min(measured.values()), 4),
        "mostOverrepresented": max(measured, key=lambda g: measured[g]),
        "mostUnderrepresented": min(measured, key=lambda g: measured[g]),
        "absentGroups": absent,
        "unreferencedGroups": unreferenced,
    }
    if absent:
        result["note"] = (
            f"{len(absent)} group(s) did not appear at all ({', '.join(absent)}). "
            f"Their skew is undefined rather than large-and-negative, and they are "
            f"excluded from maxSkew/minSkew. A group that never appears is a stronger "
            f"finding than any skew value; read absentGroups first."
        )
    return result


class NDKLValue(float):
    """An NDKL score that also says HOW MANY of the supplied rankings it rests on.

    BGL-S2 (2026-09-16): :func:`ndkl` averages over the non-empty rankings and
    used to return a bare ``float``, so a batch had nowhere to say how much of
    it was measured. Measured before the fix: ``ndkl([good] + [[]] * 9, ref)``
    returned 0.0506, byte-identical to ``ndkl([good], ref)``, while the same
    ten queries with none empty returned 0.6289. Against the 0.10 gate that is
    a PASS badge standing in for a FAIL, from 10% coverage, with no warning and
    no field that could carry the shortfall.

    It subclasses ``float`` on purpose: every existing caller keeps working
    (arithmetic, comparisons, ``json.dumps``, ``float()``), and a caller that
    wants the coverage reads ``n_rankings_measured`` / ``n_rankings_empty``
    without reading the source.
    """

    __slots__ = ("n_rankings_supplied", "n_rankings_measured", "n_rankings_empty")

    n_rankings_supplied: int
    n_rankings_measured: int
    n_rankings_empty: int

    def __new__(
        cls, value: float, *, supplied: int = 0, measured: int = 0, empty: int = 0
    ) -> "NDKLValue":
        obj = super().__new__(cls, value)
        obj.n_rankings_supplied = int(supplied)
        obj.n_rankings_measured = int(measured)
        obj.n_rankings_empty = int(empty)
        return obj


def ndkl(rankings: Sequence[Sequence[str]], reference: Optional[Dict[str, float]] = None) -> float:
    """Normalised Discounted Cumulative KL divergence (Geyik 2019): how far
    the prefix distributions of a ranked list drift from the desired
    distribution, position-discounted. ``rankings`` may be a single ranked
    list (passed as one sequence) or several.

    Returns an :class:`NDKLValue`, a ``float`` that also carries
    ``n_rankings_supplied`` / ``n_rankings_measured`` / ``n_rankings_empty``.
    Empty rankings are dropped from the mean and COUNTED; when any were
    dropped the shortfall is warned about as well, because the score is a
    measurement of the rankings that had content and says nothing about the
    rest. ``nan`` (also an :class:`NDKLValue`) is returned when nothing was
    measurable.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: representation_ndkl. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    ranking_lists: Sequence[Sequence[str]]
    if rankings and isinstance(rankings[0], str):
        ranking_lists = [cast(Sequence[str], rankings)]  # single list
    else:
        ranking_lists = rankings
    total = 0.0
    cnt = 0
    n_empty = 0
    for rk in ranking_lists:
        rk = [str(x) for x in rk]
        if not rk:
            # BGL-S2 (2026-09-16): counted, not merely skipped. See NDKLValue.
            n_empty += 1
            continue
        groups = sorted(set(rk) | set(reference or {}))
        # Same vacuous default as `skew`, same consequence. 2026-09-11:
        # ndkl(["a"] * 10) returned 0.0, the BEST attainable score, with no
        # warning, because the default reference is uniform over the groups
        # observed and a single observed group matches it exactly. A ranking
        # containing one demographic is not a perfectly diverse ranking.
        if reference is None and len(set(rk)) < 2:
            warnings.warn(
                f"ndkl: every item in this ranking is {next(iter(set(rk)))!r} and no "
                f"reference distribution was supplied, so divergence from the desired "
                f"distribution is undefined. The default reference is uniform over the "
                f"OBSERVED groups, which one group satisfies exactly and would score "
                f"0.0, the best attainable NDKL. Returning nan. Supply the desired "
                f"distribution to make this measurable.",
                UserWarning,
                stacklevel=2,
            )
            return NDKLValue(float("nan"), supplied=len(ranking_lists), measured=0, empty=n_empty)
        ref = reference or {g: 1.0 / len(groups) for g in groups}
        # READINESS-6, 2026-09-10. `q = max(ref.get(g, 0.0), 1e-9)` manufactures
        # a magnitude, the same way the clamp in `skew` above did. A group that
        # the desired distribution gives ZERO share to, but that actually
        # appears in the ranking, makes the KL term genuinely infinite: the
        # ranking contains something the target says should never appear. The
        # clamp replaced that infinity with p * ln(p / 1e-9), roughly 20 * p,
        # and the size of it is decided entirely by the epsilon.
        #
        # NDKL is a graded number, so a threshold on it was being compared
        # against the choice of 1e-9. Refuse instead: a target that excludes a
        # group the ranking contains is not a "very divergent" ranking, it is a
        # question NDKL cannot answer.
        #
        # `p`'s clamp was already dead: the `if pref.get(g, 0.0) > 0` below means
        # p is only ever used when it is strictly positive. It is dropped rather
        # than left to imply a guard that does nothing.
        excluded = [g for g in groups if ref.get(g, 0.0) <= 0.0 and g in set(rk)]
        if excluded:
            warnings.warn(
                f"ndkl: the desired distribution gives ZERO share to "
                f"{len(excluded)} group(s) that appear in the ranking "
                f"({', '.join(sorted(excluded))}), so the KL divergence is infinite "
                f"and NDKL has no value. Returning nan, NOT a large finite number: "
                f"the previous clamp made the magnitude an artefact of an epsilon. "
                f"Either give those groups a non-zero desired share, or treat their "
                f"presence as the finding.",
                UserWarning,
                stacklevel=2,
            )
            return NDKLValue(float("nan"), supplied=len(ranking_lists), measured=0, empty=n_empty)

        Z = sum(1.0 / math.log2(i + 1) for i in range(1, len(rk) + 1))
        s = 0.0
        for i in range(1, len(rk) + 1):
            pref = _dist(rk[:i])
            kl = 0.0
            for g in groups:
                p = pref.get(g, 0.0)
                if p > 0:
                    kl += p * math.log(p / ref[g])
            s += (1.0 / math.log2(i + 1)) * kl
        total += s / Z if Z > 0 else 0.0
        cnt += 1
    if cnt == 0:
        # 0.0 is the BEST possible NDKL, so an empty or all-empty input used to
        # score better than every real ranking. Measured 2026-09-08: ndkl([])
        # and ndkl([[]]) both returned 0.0 while a genuinely skewed list scored
        # 0.6275, i.e. "no data" ranked as perfectly fair. The pulse probe
        # already refused to default this metric to zero on failure; the
        # function itself did it anyway.
        warnings.warn(
            "ndkl: no non-empty ranking was supplied, so no divergence was computed. "
            "Returning nan, not 0.0, because 0.0 is the best attainable score.",
            UserWarning,
            stacklevel=2,
        )
        return NDKLValue(float("nan"), supplied=len(ranking_lists), measured=0, empty=n_empty)
    if n_empty:
        warnings.warn(
            f"ndkl: {n_empty} of {len(ranking_lists)} supplied ranking(s) were empty and "
            f"were dropped from the mean. The score returned is the mean over the {cnt} "
            f"ranking(s) that had content and says NOTHING about the empty ones; read "
            f"n_rankings_measured / n_rankings_empty on the returned value before "
            f"comparing it against a threshold.",
            UserWarning,
            stacklevel=2,
        )
    return NDKLValue(
        round(total / cnt, 4),
        supplied=len(ranking_lists),
        measured=cnt,
        empty=n_empty,
    )


def bias_amplification(generated: Sequence[str], reference: Dict[str, float]) -> Dict[str, Any]:
    """Per-group proportion difference: Δ_g = P_generated(g) − P_reference(g),
    where P_generated is the group's share of the generated set and
    P_reference the real-world share (e.g. BLS occupation statistics).
    Positive Δ = the model AMPLIFIES the real-world imbalance. This is the
    direct amplification measure of Seshadri, Singh & Elazar (2023); it is
    a raw proportion difference, NOT a difference of log-skew values as
    computed by :func:`skew`.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: representation_bias_amplification. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    gen = _dist(generated)

    # AMPLIFICATION IS A COMPARISON, so an empty reference measures nothing.
    # 2026-09-11: bias_amplification(["a"] * 10, {}) returned available=True with
    # maxAmplification 1.0, computed as 1.0 minus a real-world share nobody
    # supplied. The direction happens to be alarming rather than reassuring, but
    # the number is still made of nothing, and a caller cannot tell it from a
    # measured total amplification.
    if not reference:
        warnings.warn(
            "bias_amplification: no reference distribution was supplied. Amplification "
            "is the generated share MINUS the real-world share, so with no real-world "
            "share there is nothing to subtract and nothing to report. Returning "
            "available=False rather than a difference against zero.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "available": False,
            "reason": ("no reference distribution was supplied, so amplification is undefined"),
            "perGroup": {g: None for g in sorted(gen)},
            "worstGroup": None,
            "maxAmplification": None,
        }

    groups = sorted(set(gen) | set(reference))

    # BGL-S2 (2026-09-16): `reference.get(g, 0.0)` invented a real-world share
    # of 0.0 for any group the caller supplied no reference share for. The
    # WHOLE of that group's generated share then became "amplification", was
    # compared against the 0.1 gate, and was printed to the reader as a
    # measured finding. Measured before the fix:
    # bias_amplification(['m']*50 + ['x']*50, {'m': 0.5}) returned
    # {"available": true, "perGroup": {"m": 0.0, "x": 0.5},
    #  "worstGroup": "x", "maxAmplification": 0.5} with no warning, and the
    # pulse probe rendered BIA-001 severity 'warn' naming 'x'. The reference
    # {'m': 0.5} asserts NOTHING about 'x'.
    #
    # A reference share that is PRESENT and zero is a different claim: the
    # real world says this group should not appear, so the amplification IS
    # measured. The distinction is membership, never the value, because
    # `.get(key, default)` cannot tell "absent" from "present and 0.0".
    # `skew()` above already refuses the identical input this way; this is the
    # same three states, in the same vocabulary.
    per: Dict[str, Optional[float]] = {}
    unreferenced: List[str] = []
    for g in groups:
        raw = reference.get(g, None)
        try:
            share = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            share = float("nan")
        if g not in reference or not math.isfinite(share):
            per[g] = None
            unreferenced.append(g)
        else:
            per[g] = round(gen.get(g, 0.0) - share, 4)

    measured = {g: v for g, v in per.items() if v is not None}

    if not gen:
        # An empty generated set measures nothing about any group, so the
        # per-group numbers are not the leftover reference shares either.
        # Matches the `not reference` branch above rather than reporting
        # available=False beside a filled-in maxAmplification.
        return {
            "available": False,
            "reason": (
                "the generated set is empty, so there is no generated share to compare "
                "against the reference and amplification is undefined"
            ),
            "perGroup": {g: None for g in groups},
            "worstGroup": None,
            "maxAmplification": None,
            "unreferencedGroups": unreferenced,
        }

    if not measured:
        return {
            "available": False,
            "reason": (
                f"no generated group has a reference share: the reference distribution "
                f"says nothing about {', '.join(unreferenced)}, so there is nothing to "
                f"subtract and no amplification is defined"
            ),
            "perGroup": per,
            "worstGroup": None,
            "maxAmplification": None,
            "unreferencedGroups": unreferenced,
        }

    worst = max(measured, key=lambda g: abs(measured[g]))
    result: Dict[str, Any] = {
        "available": True,
        "perGroup": per,
        "worstGroup": worst,
        # Over the MEASURED groups only. A group the reference never mentions
        # is not an amplification of 1.0 minus nothing, it is an unanswered
        # question, and it is reported as one below.
        "maxAmplification": round(max(abs(v) for v in measured.values()), 4),
        "unreferencedGroups": unreferenced,
    }
    if unreferenced:
        result["note"] = (
            f"{len(unreferenced)} generated group(s) have no reference share "
            f"({', '.join(unreferenced)}). Amplification is undefined for them rather "
            f"than equal to their whole generated share, and they are excluded from "
            f"worstGroup/maxAmplification. Supply their real-world share to measure "
            f"them; read unreferencedGroups first."
        )
    return result


def representation_severity(
    max_skew: "float | Mapping[str, Any]",
    min_skew: Optional[float] = None,
) -> str:
    """Worst-absolute-skew severity per the Pulse spec (0.22 flag, 0.5 fail).

    Representation fairness bounds BOTH over- and under-representation:
    MaxSkew is the largest unfair ADVANTAGE, MinSkew the largest unfair
    DISADVANTAGE (Geyik et al. 2019). Gating on MaxSkew alone is blind to a
    group that is severely under-represented or excluded (a large negative
    MinSkew) -- the dominant representation harm for generated / representative
    image sets (Seshadri et al. 2023). So the gate keys on the worst absolute
    skew, ``max(|MaxSkew|, |MinSkew|)``.

    Pass the full :func:`skew` result dict (preferred), or ``max_skew`` and
    ``min_skew`` explicitly. A bare ``max_skew`` float is still accepted and
    thresholds on its magnitude only, for backward compatibility.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: representation_severity. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    # A REFUSAL IS NOT A PASS. `.get("maxSkew", 0.0) or 0.0` turned a missing
    # key, a None and an explicit `available: False` all into 0.0, and 0.0 is
    # the cleanest band this function has. Measured 2026-09-08: the result dict
    # of `classify_face_demographics([])`, whose own docstring says it "NEVER
    # returns fabricated demographics" and which correctly answered
    # `available: False, distribution: None`, was graded "pass" here. The
    # measurement layer refused and the grading layer overrode it.
    #
    # Keyed on ABSENCE, never on the value: a measured skew of exactly 0.0 is a
    # real, and excellent, result and still returns "pass".
    if isinstance(max_skew, Mapping):
        if max_skew.get("available") is False:
            warnings.warn(
                "representation_severity: the supplied result reports available=False, "
                "so no skew was measured. Returning 'not_assessed', not 'pass'.",
                UserWarning,
                stacklevel=2,
            )
            return "not_assessed"
        # TOTAL EXCLUSION, keyed on the FACT rather than on a magnitude.
        # READINESS-6, 2026-09-10. This case used to arrive here as a large
        # negative minSkew, because `skew` clamped an absent group's observed
        # share to 1e-9 and took a logarithm of it. Removing that fabrication
        # (correctly: the number was an artefact of the epsilon) also removed
        # the only route by which a completely excluded group reached this
        # gate, and `test_representation_severity_flags_total_exclusion` went
        # red. The test was right: total exclusion IS critical, and it was
        # being detected through an invented number.
        #
        # A group that never appears is now reported by `skew` as a named
        # absence, and banded here on that, which is stronger than the
        # threshold it replaces: "critical" no longer depends on 1e-9 landing
        # far enough below 0.5 after a log.
        absent = max_skew.get("absentGroups") or []
        if absent:
            warnings.warn(
                f"representation_severity: {len(absent)} group(s) do not appear in the "
                f"set at all ({', '.join(sorted(absent))}). Total exclusion is the most "
                f"severe representation harm there is, so this is 'critical' regardless "
                f"of the skews of the groups that ARE present.",
                UserWarning,
                stacklevel=2,
            )
            return "critical"

        raw_mx, raw_mn = max_skew.get("maxSkew"), max_skew.get("minSkew")
        if raw_mx is None and raw_mn is None:
            warnings.warn(
                "representation_severity: the supplied mapping carries neither maxSkew "
                "nor minSkew, so there is nothing to band. Returning 'not_assessed', "
                "not 'pass'.",
                UserWarning,
                stacklevel=2,
            )
            return "not_assessed"
        # READINESS-5, 2026-09-10. ONE key missing used to substitute 0.0 for
        # it and band on the other, silently. Measured that day:
        #     {"maxSkew": 0.10, "minSkew": -0.90} -> 'critical'
        #     {"maxSkew": 0.10}                   -> 'pass'   (0 warnings)
        # This function's own docstring says gating on MaxSkew alone "is blind
        # to a group that is severely under-represented or excluded (a large
        # negative MinSkew) -- the dominant representation harm", and then it
        # did exactly that whenever MinSkew was absent. The asymmetry ran the
        # dangerous way: a missing MAXskew still yields 'critical', because 0.0
        # cannot hide a large |minSkew|, so only the reassuring direction was
        # reachable. The two sibling guards above already warn and return
        # 'not_assessed' for available=False and for BOTH keys missing; this
        # third absence case returned a band in silence.
        #
        # `a = max(|mx|, |mn|)` is MONOTONE in the missing value, so a band
        # computed from the skew we do have is a LOWER BOUND: more evidence can
        # only raise it. That is what makes a partial answer safe here. A band
        # of 'warn' or 'critical' is therefore reported as it stands, and only
        # a 'pass' has to be withheld, because the skew nobody measured could
        # be anything. Deciding that needs the band, so it happens after it.
        partial_key = None
        if raw_mx is None:
            partial_key = "maxSkew"
        elif raw_mn is None:
            partial_key = "minSkew"
        mx = float(raw_mx) if raw_mx is not None else 0.0
        mn = float(raw_mn) if raw_mn is not None else 0.0
    else:
        # The bare-float path is DOCUMENTED to threshold on magnitude only, for
        # backward compatibility, so it keeps its behaviour and is not treated
        # as a partial mapping.
        partial_key = None
        mx = float(max_skew)
        mn = float(min_skew) if min_skew is not None else 0.0

    if not (math.isfinite(mx) and math.isfinite(mn)):
        warnings.warn(
            f"representation_severity: skew values are not finite (maxSkew={mx}, "
            f"minSkew={mn}), so no band was assigned. Returning 'not_assessed'.",
            UserWarning,
            stacklevel=2,
        )
        return "not_assessed"

    a = max(abs(mx), abs(mn))
    band = "critical" if a >= 0.5 else ("warn" if a >= 0.22 else "pass")
    if partial_key is not None and band == "pass":
        # A lower bound of "pass" is not a pass: the unmeasured skew could be
        # anything, and it is the one that carries the exclusion harm.
        warnings.warn(
            f"representation_severity: the supplied mapping carries no "
            f"{partial_key!r}, so only one of the two skews was banded. The "
            f"skew that WAS supplied lands in 'pass', but the missing one could "
            f"be anything, so this is 'not_assessed', not 'pass'. Supply both, "
            f"or pass the full skew() result which always carries them.",
            UserWarning,
            stacklevel=2,
        )
        return "not_assessed"
    return band


# Demographic classification (FairFace / CLIP): sidecar-gated, never fakes.


def classify_face_demographics(
    image_paths: Sequence[str],
    *,
    attribute: str = "race",
) -> Dict[str, Any]:
    """Classify faces with FairFace. Requires the Python-3.9 vision sidecar
    (env ``VFAIRNESS_VISION_SIDECAR``) OR torch + the FairFace weights in
    this process. Degrades to ``available: False`` with a reason and a
    metadata-only note -- it NEVER returns fabricated demographics.
    """
    sidecar = os.environ.get("VFAIRNESS_VISION_SIDECAR")
    if sidecar:
        try:
            import json
            import subprocess

            payload = json.dumps({"image_paths": list(image_paths), "attribute": attribute})
            res = subprocess.run(
                [sidecar, "--task", "fairface"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if res.returncode == 0 and res.stdout.strip():
                # G12, 2026-09-30. ``available: True`` WAS ASSERTED FROM THE EXIT
                # STATUS, not from the reply. Measured with a stub sidecar
                # exiting 0 on two images:
                #   stdout '{}'      -> {'available': True}
                #   stdout 'null'    -> available False, reason "vision sidecar
                #                       unreachable: 'NoneType' object is not a
                #                       mapping"
                #   stdout '[1,2]'   -> the same, blaming unreachability
                # The first one is the defect: two images went in, no
                # classification came back, and the field every consumer
                # branches on said the classification was available. The other
                # two reached the right answer through a TypeError raised by the
                # ``**`` unpacking and caught by the ``except`` below, so the
                # reason named the wrong cause. An empty reply is the loudest
                # outcome here, not the mildest, and this function's own
                # docstring promises it never hands back demographics it does
                # not have.
                parsed = json.loads(res.stdout)
                if not isinstance(parsed, dict) or not parsed:
                    return {
                        "available": False,
                        "reason": (
                            f"vision sidecar returned no classification for "
                            f"{len(list(image_paths))} image(s): it exited 0 with a "
                            f"{type(parsed).__name__} carrying nothing to read. This is "
                            f"COULD NOT CHECK, not an absence of faces."
                        ),
                        "fallback": "metadata_only",
                    }
                return {"available": True, **parsed}
            return {
                "available": False,
                "reason": (
                    f"vision sidecar error: exit {res.returncode}, "
                    f"{'empty' if not res.stdout.strip() else 'unread'} stdout"
                    + (f": {res.stderr[:200]}" if res.stderr.strip() else " and no stderr")
                ),
            }
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"vision sidecar unreachable: {e}"}
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        return {
            "available": False,
            "reason": (
                "Face demographic classification needs the Python-"
                "3.9 vision sidecar (FairFace + torch). Set "
                "VFAIRNESS_VISION_SIDECAR. No demographics are "
                "guessed without it."
            ),
            "fallback": "metadata_only",
        }
    return {
        "available": False,
        "reason": ("torch present but FairFace weights not bundled; use the vision sidecar."),
        "fallback": "metadata_only",
    }


__all__ = [
    "skew",
    "ndkl",
    # NDKLValue is the RETURN TYPE of ndkl, importable by name, and is
    # deliberately not listed here: __all__ is read by
    # tests/test_registry_completeness.py as the capability surface, and a
    # result type is not a capability. See the class for what it carries.
    "bias_amplification",
    "representation_severity",
    "classify_face_demographics",
]
