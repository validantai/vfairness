"""
Text Classification Fairness (identity-term bias), Workstream E.

Measures whether a text classifier (toxicity, sentiment, moderation, ...)
assigns systematically different scores to text mentioning different identity
groups, the canonical "unintended bias in toxicity classification" problem
(Dixon et al., 2018; Borkan et al., 2019, the Jigsaw civil-comments metrics).

Given a classifier ``score_fn(list[str]) -> list[float]`` and texts grouped by
the identity term they mention, it reports:
  * per-group mean score,
  * the worst gap vs the overall mean (Subgroup AUC-style disparity, simplified
    to mean-score disparity so it needs no labels),
  * a Mann-Whitney U significance test for the worst group vs the rest,
  * a severity grade + plain-language interpretation.

Dependency-light: numpy + scipy only. ``score_fn`` can wrap any model / API.
NOT imported from the package __init__. Import explicitly:

    from vfairness.llm.text_fairness import TextFairnessAnalyzer
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vfairness.evaluation.vfairness_metrics._statistics import (
    _mannwhitney_two_sided_p,
    detectability,
    min_attainable_p_mannwhitney,
)

#: Significance threshold this analyzer grades against, and the bar a design
#: has to be able to clear for "no material bias" to mean anything.
ALPHA = 0.05


@dataclass
class GroupScore:
    group: str
    n: int
    mean_score: float
    gap_vs_overall: float  # mean_score - overall_mean

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TextFairnessResult:
    overall_mean: float
    groups: List[GroupScore]
    worst_group: Optional[str]
    max_gap: float  # signed gap of the worst group
    p_value: Optional[float]  # worst group vs rest (Mann-Whitney U)
    severity: str
    interpretation: str
    notes: List[str] = field(default_factory=list)
    #: Identity terms that were SUPPLIED with no text, so nothing was scored
    #: for them and they are absent from ``groups``, from ``overall_mean`` and
    #: from the worst-group search. Empty on a complete run. BGL stage 5,
    #: 2026-09-27: see ``TextFairnessAnalyzer.analyze`` for what their silent
    #: removal cost.
    groups_not_scored: List[str] = field(default_factory=list)
    #: How many identity terms the caller supplied, scored or not. The
    #: denominator for ``groups``, which holds only the scored ones.
    n_groups_supplied: int = 0

    def to_dict(self) -> Dict[str, Any]:
        # This is a hand-written key list, which does not fail when a field is
        # added: it just stops carrying it. The two coverage fields below are the
        # only machine-readable trace of the identity terms nothing was scored
        # for, so dropping them here would put the measurement in front of a
        # consumer with the caveat removed, which is the defect they were added
        # to fix. Caught RED by tests/test_serialiser_honesty.py the moment they
        # were declared, and pinned in tests/test_bgl3_llm_4.py.
        return {
            "overall_mean": self.overall_mean,
            "worst_group": self.worst_group,
            "max_gap": self.max_gap,
            "p_value": self.p_value,
            "severity": self.severity,
            "interpretation": self.interpretation,
            "notes": list(self.notes),
            "groups": [g.to_dict() for g in self.groups],
            "groups_not_scored": list(self.groups_not_scored),
            "n_groups_supplied": self.n_groups_supplied,
        }


class TextFairnessAnalyzer:
    """Identity-term bias analyzer for a text classifier.

    Parameters
    ----------
    score_fn : callable
        ``score_fn(list[str]) -> sequence[float]`` returning one score per text
        (e.g. toxicity probability in [0, 1]).

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: text_fairness_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, score_fn: Callable[[Sequence[str]], Sequence[float]]):
        if not callable(score_fn):
            raise TypeError("score_fn must be callable: score_fn(texts) -> scores")
        self._score_fn = score_fn

    def analyze(
        self,
        texts_by_group: Dict[str, Sequence[str]],
        higher_is_worse: bool = True,
    ) -> TextFairnessResult:
        """Score each group's texts and quantify disparity.

        texts_by_group : {identity_term: [texts mentioning it]}.
        higher_is_worse : True for toxicity (high score = harm); False for
            sentiment/regard (low score = harm). Controls which direction the
            "worst" group is picked from.

        An identity term supplied with NO text cannot be scored, and it is
        reported as such (``groups_not_scored``, a note, a warning, and a
        severity that refuses to say "no material bias") rather than removed.
        See the comment on the filter below for what it read as before.
        """
        clean = {g: list(t) for g, t in texts_by_group.items() if len(t) > 0}
        # BGL stage 5, 2026-09-27. This filter was the whole handling: an
        # identity term supplied with no text left the analysis with no record,
        # no note and no warning, and the term most likely to arrive empty is
        # the one the classifier or the harness choked on. Everything downstream
        # then described a SMALLER design as if it were the one that was asked
        # for: `overall_mean` is the mean of the survivors, so every
        # `gap_vs_overall` moves, and `worst_group` is the worst of whoever
        # remained. Measured 2026-09-27 on three identity terms, 'nonbinary'
        # supplied empty, the other two scoring 0.10 to 0.13 and 0.14 to 0.17:
        #   severity 'info', "No material identity-term bias detected for
        #   'men' (gap +0.020, p=0.029)", notes [], and no warning at all
        # over a run in which a third of the identity terms was never scored.
        # The measurement on the survivors is real, so it is kept; what changes
        # is that a clean reading may no longer be reported as one when part of
        # the design produced nothing.
        not_scored = [g for g, t in texts_by_group.items() if len(t) == 0]
        if len(clean) < 2:
            raise ValueError("Need at least two non-empty groups to compare.")

        scores: Dict[str, "np.ndarray"] = {}
        for g, txts in clean.items():
            s = np.asarray(self._score_fn(txts), dtype=float).ravel()
            if len(s) != len(txts):
                raise ValueError(f"score_fn returned {len(s)} scores for {len(txts)} '{g}' texts.")
            scores[g] = s

        all_scores = np.concatenate(list(scores.values()))
        overall = float(np.mean(all_scores))

        groups = [
            GroupScore(
                group=g,
                n=len(s),
                mean_score=float(np.mean(s)),
                gap_vs_overall=float(np.mean(s) - overall),
            )
            for g, s in scores.items()
        ]

        # Worst group = largest harmful deviation.
        if higher_is_worse:
            worst = max(groups, key=lambda gs: gs.gap_vs_overall)
        else:
            worst = min(groups, key=lambda gs: gs.gap_vs_overall)

        # Significance: worst group vs everyone else.
        rest = np.concatenate([s for g, s in scores.items() if g != worst.group])
        p_value = self._mwu(scores[worst.group], rest)
        detectable, note = self._detectability(scores[worst.group], rest)

        severity, interpretation = self._grade(
            worst.gap_vs_overall, p_value, worst.group, detectable, note, not_scored
        )
        notes: List[str] = []
        if note:
            notes.append(note)
        if not_scored:
            coverage = (
                f"{len(not_scored)} of {len(texts_by_group)} identity term(s) were "
                f"supplied with no text and were NOT SCORED "
                f"({', '.join(sorted(not_scored))}). They are absent from the group "
                f"means, from the overall mean every gap is measured against, and from "
                f"the worst-group search, so this is a comparison of the "
                f"{len(clean)} terms that produced text."
            )
            notes.append(coverage)
            warnings.warn(
                f"TextFairnessAnalyzer.analyze: {coverage} A gap measured over part of "
                f"a design is not a gap over the design, so a reading with nothing "
                f"material in it is graded 'not_assessed' rather than 'no material "
                f"identity-term bias'.",
                UserWarning,
                stacklevel=2,
            )
        return TextFairnessResult(
            overall_mean=overall,
            groups=groups,
            worst_group=worst.group,
            max_gap=worst.gap_vs_overall,
            p_value=p_value,
            severity=severity,
            interpretation=interpretation,
            notes=notes,
            groups_not_scored=sorted(not_scored),
            n_groups_supplied=len(texts_by_group),
        )

    @staticmethod
    def _detectability(a: "np.ndarray", b: "np.ndarray"):
        """Could the Mann-Whitney test ever reach ALPHA on data shaped like this?

        DISCRETE FLOOR (readiness 6, 2026-09-10). The rank-sum p-value has a
        floor set by the two sample sizes: with 2 texts against 6 and no ties,
        the most extreme possible separation gives 2 / C(8,2) = 0.0714, so no
        classifier behaviour whatsoever could reach 0.05. Measured 2026-09-10 on
        exactly that shape, one identity group scoring 0.95/0.90 against six
        others between 0.15 and 0.20: gap +0.562, perfectly separated, graded
        "info" and captioned "No material identity-term bias detected for
        'women' (gap +0.562, p=0.071)". The bigger the untestable gap, the more
        reassuring the sentence.

        The floor is computed by running the REAL test on the most extreme
        rearrangement of the OBSERVED values (the n_a largest against the rest),
        not from the sizes alone. That keeps the tie structure, and the tie
        structure is what decides which method scipy resolves to and therefore
        what the floor is: the same 2-vs-6 design floors at 0.0153 when the
        scores are tied (scipy uses the tie-corrected asymptotic test there) and
        at 0.0714 when they are distinct (exact test). Reading the floor off the
        data answers the question that matters (could THIS comparison have
        fired), and never calls a design dead that its own data could have made
        detectable. See ``min_attainable_p_mannwhitney`` in
        ``evaluation/vfairness_metrics/_statistics.py`` for the design-level
        version and the fail-safe reasoning.
        """
        if len(a) < 2 or len(b) < 2:
            # _mwu already refused and warned; that is a did-not-run, not a
            # could-not-fire, and _grade routes it to not_assessed on its own.
            return None, ""
        try:
            from scipy.stats import mannwhitneyu

            pooled = np.sort(
                np.concatenate([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
            )
            n_a = len(a)
            floor_val: Optional[float]
            if float(np.ptp(pooled)) == 0.0:
                # Check 2 (honest on broken data), 2026-10-01. EVERY SCORE TIED
                # IS A FINDING, NOT A DEAD DESIGN. With one value throughout,
                # every rearrangement of the observed values is the observed
                # arrangement, so the "most extreme" rearrangement above is the
                # data itself and its p of 1.0 was read as the design's floor.
                # Measured on 200 texts a side, the classifier scoring every one
                # of them 0.0: gap +0.000, p=1.000, severity 'not_assessed', and
                # a warning that the test "could NOT have reached 0.05 for any
                # scores at all", which is false for 200 against 200 (the design
                # floor is about 1e-88). A classifier that scored both groups
                # identically on 400 texts is the strongest no-bias reading the
                # test can give, and it was withheld. Tied data says nothing
                # about the floor, so the floor here is the DESIGN's, from the
                # two sample sizes, exactly as CounterfactualTester._test_metric
                # does for the same case. A small tied design (2 against 2,
                # floor 0.19) is still not detectable and still not assessed.
                floor_val = min_attainable_p_mannwhitney(n_a, len(pooled) - n_a)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, floor = mannwhitneyu(pooled[-n_a:], pooled[:-n_a], alternative="two-sided")
                floor_val = float(floor)
        except Exception:  # pragma: no cover - defensive
            floor_val = None
        verdict, note = detectability(floor_val, n_family=1, alpha=ALPHA)
        if verdict is not True:
            warnings.warn(
                f"TextFairnessAnalyzer: with {len(a)} text(s) in the worst group against "
                f"{len(b)} in the rest, the Mann-Whitney test could NOT have reached "
                f"{ALPHA} for any scores at all. {note} Grading this comparison "
                f"'not_assessed', NOT 'no material bias'.",
                UserWarning,
                stacklevel=3,
            )
        return verdict, note

    @staticmethod
    def _mwu(a: "np.ndarray", b: "np.ndarray") -> Optional[float]:
        """None means the test did not run. It never means "not significant"."""
        if len(a) < 2 or len(b) < 2:
            warnings.warn(
                f"TextFairnessAnalyzer: the Mann-Whitney test needs at least 2 texts on "
                f"each side and got {len(a)} vs {len(b)}, so significance was NOT "
                f"tested. The measured gap still stands; the p-value does not exist.",
                UserWarning,
                stacklevel=3,
            )
            return None
        # EVERY SCORE TIED is the exact permutation p of 1.0, a measurement,
        # where scipy 1.18's default path returns nan; the shared helper holds
        # that rule and its evidence. Whether the design could have fired at all
        # is _detectability's job, and a small tied design is still not assessed
        # there. Same reading as CounterfactualTester._test_metric.
        try:
            p = _mannwhitney_two_sided_p(a, b)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"TextFairnessAnalyzer: the Mann-Whitney test raised {exc!r}, so "
                f"significance was NOT tested.",
                UserWarning,
                stacklevel=3,
            )
            return None
        if p is None:
            # A nan p is not a p-value. Passed on, `nan < 0.05` is False and it
            # would be graded as "not significant"; it is a test that did not run.
            warnings.warn(
                "TextFairnessAnalyzer: the Mann-Whitney test returned no finite "
                "p-value, so significance was NOT tested.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return p

    @staticmethod
    def _grade(
        gap: float,
        p_value: Optional[float],
        group: str,
        detectable: Optional[bool] = True,
        detectability_note: str = "",
        groups_not_scored: Optional[Sequence[str]] = None,
    ):
        mag = abs(gap)
        # BGL stage 5, 2026-09-27. Appended to every caption rather than only to
        # the one below, because a critical finding measured over part of a
        # design is still a finding measured over part of a design.
        unscored = list(groups_not_scored or ())
        coverage_note = (
            (
                f" COVERAGE: {len(unscored)} identity term(s) "
                f"({', '.join(sorted(unscored))}) were supplied with no text and were "
                f"never scored, so neither this gap nor the mean it is measured "
                f"against covers them."
            )
            if unscored
            else ""
        )

        # `p_value is None` means the test DID NOT RUN, and it used to be folded
        # into `significant = p_value is not None and p_value < 0.05`, i.e.
        # straight onto the not-significant side. Measured 2026-09-08 with one
        # group of a single text scoring 0.95 against six others at 0.05: a gap
        # of +0.771 was graded "info" and captioned "No material identity-term
        # bias detected for 'women'". The bigger the untested gap, the more
        # reassuring the sentence became.
        if p_value is None:
            return (
                "not_assessed",
                f"Identity-term bias for '{group}' was NOT ASSESSED: the measured gap is "
                f"{gap:+.3f}, but no significance test could be run on it (a group needs "
                f"at least 2 texts on each side). This is not a finding of no bias."
                + coverage_note,
            )

        # A test that RAN but could never have reached ALPHA is not a
        # not-significant finding either: see _detectability above for the
        # measurement. Its own not_assessed branch, with its own sentence.
        if detectable is not True:
            state = "could not be computed" if detectable is None else "cannot reach"
            return (
                "not_assessed",
                f"Identity-term bias for '{group}' was NOT ASSESSED: the measured gap is "
                f"{gap:+.3f} (p={p_value:.3f}), but the smallest p-value this comparison "
                f"could produce {state} the {ALPHA} threshold, so no classifier behaviour "
                f"could have been flagged here. {detectability_note} This is not a finding "
                f"of no bias; add texts to the smaller group." + coverage_note,
            )

        significant = p_value < ALPHA
        if not significant or mag < 0.05:
            caption = (
                f"No material identity-term bias detected for '{group}' "
                f"(gap {gap:+.3f}" + (f", p={p_value:.3f}" if p_value is not None else "") + ")."
            )
            if unscored:
                # The all-clear is the ONE caption that may not survive a
                # partial design: it is a positive claim that every identity
                # term was compared and none of them differed, and the terms
                # that produced no text were never in the comparison.
                return (
                    "not_assessed",
                    f"Identity-term bias was NOT ASSESSED for the whole design: nothing "
                    f"material was found among the terms that were scored (gap "
                    f"{gap:+.3f} for '{group}', p={p_value:.3f})." + coverage_note,
                )
            return ("info", caption)
        if mag >= 0.25:
            sev = "critical"
        elif mag >= 0.15:
            sev = "high"
        elif mag >= 0.10:
            sev = "medium"
        else:
            sev = "low"
        return (
            sev,
            f"Texts mentioning '{group}' score {gap:+.3f} vs the overall mean "
            f"(p={p_value:.3f}). The classifier treats this identity group "
            f"systematically differently, a disparate-impact risk." + coverage_note,
        )
