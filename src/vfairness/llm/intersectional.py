"""Intersectional fairness testing for LLMs.

Tests bias at the intersection of multiple protected attributes
(e.g., Black women vs. White men), which can reveal compounded
discrimination invisible to single-axis analysis.

References:
    - Buolamwini & Gebru (2018): Gender Shades
    - Kearns et al. (2018): Preventing Fairness Gerrymandering
    - UW Hiring Study (2024): Intersectional harm against Black men
"""

import logging
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np

from ._base import RunMetadata
from .output_analysis import OutputAnalysisResult, OutputAnalyzer

logger = logging.getLogger(__name__)


#: The SERIALISED reprs of absence that a protected-attribute LEVEL arrives as
#: when a design is read out of a CSV, a JSON export or a DataFrame column: by
#: the time it reaches this module it is a plain ``str`` and no test of the
#: object can see it. Taken from the same list
#: ``operations.pulse.traces._serialised_absence_strings`` builds from pandas'
#: own ``STR_NA_VALUES`` (measured members on this repo: '', '<NA>', 'NA',
#: 'NaN', 'NaT', 'None', 'NULL', 'N/A', 'n/a', 'nan', 'null' and the Excel
#: '#N/A' / '-1.#IND' family), so the two cannot disagree about what absence
#: looks like as text. It is a copy rather than an import ON PURPOSE: that
#: module lives under ``operations`` and pulls the pulse subsystem in with it,
#: and ``vfairness.llm`` is deliberately light at import time.
_ABSENT_LEVEL_STRINGS = frozenset(
    {
        "",
        "#n/a",
        "#n/a n/a",
        "#na",
        "-1.#ind",
        "-1.#qnan",
        "-nan",
        "1.#ind",
        "1.#qnan",
        "<na>",
        "n/a",
        "na",
        "nan",
        "nat",
        "none",
        "null",
    }
)


def _level_is_absent(value: object) -> bool:
    """Is this attribute LEVEL no level at all, by every door absence arrives through?

    The object doors (``None``, float ``nan``, ``pd.NA``, ``pd.NaT``) are tested
    without importing pandas, the way ``_names._is_missing`` does it: NaN is the
    only value unequal to itself, and ``pd.NA != pd.NA`` answers with another NA
    whose ``bool()`` raises, so the ``TypeError`` IS the missing answer.

    The text door is :data:`_ABSENT_LEVEL_STRINGS`, matched on the stripped,
    case-folded value, because a design is usually read from a file.

    A level that is a real, visible token is left exactly as it is, including
    levels like "missing", "unknown" and "prefer not to say", which are groups
    somebody named and are frequently the most interesting ones. Only the
    absence markers are refused.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in _ABSENT_LEVEL_STRINGS
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


@dataclass
class IntersectionalGroup:
    """A specific intersection of protected attributes.

    Attributes:
        attributes: Mapping of attribute names to their values,
            e.g., ``{"race": "Black", "gender": "female"}``.
        label: Short human-readable label, e.g., ``"Black_female"``.
    """

    attributes: dict[str, str]
    label: str

    @staticmethod
    def from_attributes(
        attribute_values: dict[str, list[str]],
    ) -> list["IntersectionalGroup"]:
        """Generate all intersectional groups from attribute value lists.

        Computes the Cartesian product of all attribute values.

        Args:
            attribute_values: Mapping from attribute names to possible
                values, e.g.,
                ``{"race": ["Black", "White"], "gender": ["male", "female"]}``.
                A level that is an ABSENCE MARKER rather than a level (``None``,
                float ``nan``, ``pd.NA``, a blank string, or the text 'None' /
                '<NA>' / 'nan' a CSV or JSON export writes) is dropped from its
                axis with a warning naming it; see :func:`_level_is_absent`.

        Returns:
            List of all intersectional groups (4 in the example above), and an
            EMPTY list when the design names no intersection to enumerate. That
            case is never silent: it warns, for the reasons below.
        """
        keys = list(attribute_values.keys())
        # G12, 2026-09-30. AN INTERSECTIONAL GROUP WAS MINTED OUT OF AN ABSENT
        # VALUE. The guard below tests whether an AXIS is empty; it never looked
        # at the VALUES, and `"_".join(combo)` pastes whatever is in them into a
        # label. Measured before this change, every one of these SILENT:
        #   {"race": ["Black", ""],     "gender": ["female"]}
        #     -> a second group, label '_female',     attributes race=''
        #   {"race": ["Black", "None"], "gender": ["female"]}
        #     -> label 'None_female',   attributes race='None'
        #   {"race": ["Black", "<NA>"], "gender": ["female"]}  -> '<NA>_female'
        #   {"race": ["Black", "nan"],  "gender": ["female"]}  -> 'nan_female'
        #   {"race": ["Black", "  "],   "gender": ["female"]}  -> '  _female'
        # and the OBJECT doors (None, float nan, pd.NA) raised a bare
        # `TypeError: sequence item 0: expected str instance` out of join(),
        # which blames this function for the caller's design and is not a stated
        # refusal. A design is routinely read from a CSV or a JSON export, where
        # pandas writes exactly those strings, so half a dataset lands in two
        # protected groups nobody chose, the analyzer compares them as
        # intersections, and max_disparity / has_intersectional_bias are then
        # computed over a group that is not a group. The '_female' label is the
        # worst of them: its blank half renders as nothing at all in a report.
        #
        # THE SAME REASONING THE EMPTY-AXIS BRANCH BELOW ALREADY GIVES ("a group
        # with an empty label would be counted as an intersection by every caller
        # that keys its outputs by the label") applies to a value that RENDERS
        # blank; that branch just tested the wrong half of the design.
        #
        # Absent LEVELS are dropped from their axis and NAMED, rather than the
        # whole design being refused: a race axis of ["Black", "White", ""] still
        # has a real 2-level design in it, and refusing that would destroy the
        # unit to close the door. When dropping empties an axis, the existing
        # branch below refuses the design, which is the honest end state.
        usable: dict[str, list[str]] = {}
        absent_by_axis: dict[str, list[str]] = {}
        for k in keys:
            kept: list[str] = []
            dropped: list[str] = []
            for v in attribute_values[k] or []:
                if _level_is_absent(v):
                    dropped.append(repr(v))
                else:
                    kept.append(str(v))
            usable[k] = kept
            if dropped:
                absent_by_axis[k] = dropped
        if absent_by_axis:
            listed = "; ".join(
                f"{axis}: {', '.join(vals)}" for axis, vals in sorted(absent_by_axis.items())
            )
            n_dropped = sum(len(v) for v in absent_by_axis.values())
            warnings.warn(
                f"IntersectionalGroup.from_attributes: {n_dropped} supplied attribute "
                f"level(s) are an ABSENCE MARKER, not a level, and were DROPPED from the "
                f"design ({listed}). An intersection built on one of them is a protected "
                f"group nobody chose: the label pastes the marker in ('None_female', "
                f"'<NA>_female') or renders blank ('_female'), and every caller that keys "
                f"its outputs by the label would then count it as an intersection and put "
                f"it in max_disparity. If those rows matter, give the level a name you "
                f"chose before building the design.",
                UserWarning,
                stacklevel=2,
            )
        empty_axes = [k for k in keys if not usable[k]]
        # BGL stage 5, 2026-09-27. `product()` of NOTHING yields one empty
        # tuple, so an empty design manufactured a single group with
        # attributes={} and label="": a group nobody asked for, which a caller
        # then keys its outputs by (`outputs[g.label]`) and a report counts as
        # an intersection. The mirror case erases the design instead: one axis
        # with no values makes the product empty, so the whole design vanished
        # and "nothing was requested" and "no groups exist" became the same
        # answer. Measured before this change, both with ZERO warnings:
        #   from_attributes({})
        #     -> [IntersectionalGroup(attributes={}, label='')]
        #   from_attributes({"race": ["Black", "White"], "gender": []})
        #     -> []
        # Returning [] for both is the honest answer (a design with no axis, or
        # with an axis holding no value, has no intersections), and the warning
        # is what stops it reading as a design that was enumerated.
        if not keys or empty_axes:
            if not keys:
                detail = "no attribute was supplied"
            else:
                emptied = sorted(a for a in empty_axes if a in absent_by_axis)
                detail = (
                    # "no values" is kept verbatim: tests/test_bgl3_llm_4.py
                    # locates this refusal by that phrase, and the subject it
                    # pins (an axis with nothing to enumerate is refused, loudly)
                    # is unchanged by G12.
                    f"the attribute(s) {', '.join(sorted(empty_axes))} hold no values this "
                    f"design can use"
                    + (
                        f" (every level supplied for {', '.join(emptied)} was an absence "
                        f"marker, see the warning above)"
                        if emptied
                        else ""
                    )
                )
            warnings.warn(
                f"IntersectionalGroup.from_attributes: {detail}, so there is no "
                f"intersection to enumerate and NO group was built. Returning an empty "
                f"list, not a placeholder group: an analysis over these groups measures "
                f"nothing, and a group with an empty label would be counted as an "
                f"intersection by every caller that keys its outputs by the label.",
                UserWarning,
                stacklevel=2,
            )
            return []
        value_lists = [usable[k] for k in keys]
        groups = []
        for combo in product(*value_lists):
            attrs = dict(zip(keys, combo))
            label = "_".join(combo)
            groups.append(IntersectionalGroup(attributes=attrs, label=label))
        return groups


@dataclass
class IntersectionalResult:
    """Results of an intersectional fairness analysis.

    Attributes:
        groups: The intersectional groups that were compared.
        pairwise_results: ``OutputAnalysisResult`` for every pair of groups.
        most_disadvantaged: Group with the lowest mean metric value, or
            ``None`` if no results are available.
        most_advantaged: Group with the highest mean metric value, or
            ``None`` if no results are available.
        max_disparity: Largest absolute delta across the pairwise comparisons
            that were ASSESSED, and ``nan`` when none was. Read it against
            ``n_pairs_not_assessed``: a maximum over part of a design is not a
            maximum over the design.
        n_significant_pairs: Number of pairs with statistically
            significant differences (after Bonferroni correction). Counted out
            of the assessed pairs only, for the same reason.
        total_pairs: Total number of pairwise comparisons the design HAS,
            assessed or not. Every pair of the supplied ``groups`` is present
            in ``pairwise_results``, so this is ``n*(n-1)/2`` and can be read
            against the design a caller passed in.
        n_pairs_not_assessed: How many of those pairs nothing was measured on
            (``assessed=False``). ``0`` on a complete run. BGL-D, 2026-09-11.
            READ IT WITH ``total_pairs``: ``0`` beside ``total_pairs`` of ``0``
            means the design held no comparison at all (fewer than two groups),
            not that a complete run found nothing, and ``analyze`` warns in that
            case (BGL5 A-llm-3).
    """

    groups: list[IntersectionalGroup]
    pairwise_results: list[OutputAnalysisResult]
    most_disadvantaged: Optional[IntersectionalGroup]
    most_advantaged: Optional[IntersectionalGroup]
    max_disparity: float
    n_significant_pairs: int
    total_pairs: int
    n_pairs_not_assessed: int = 0

    @property
    def has_intersectional_bias(self) -> Optional[bool]:
        """``True`` if any pairwise comparison is significant.

        ``None`` when no pair could be tested at all (LF-06): a set of
        comparisons in which every test was unassessable is not evidence of
        no intersectional bias, and used to read as exactly that.

        ``None`` also when nothing came back significant but at least one pair
        of the design was never compared (BGL-D, 2026-09-11). ``False`` is a
        positive claim that every intersection was checked and none of them
        differed; it may only be made when every pair actually produced a test.
        A significant pair still answers ``True``, because a finding found is a
        finding however much of the rest is missing.
        """
        if self.n_significant_pairs > 0:
            return True
        if not any(r.p_value is not None for r in self.pairwise_results):
            return None
        if any(r.p_value is None for r in self.pairwise_results):
            return None
        return False


class IntersectionalAnalyzer:
    """Analyze LLM fairness across intersections of protected attributes.

    Runs pairwise output comparisons between all intersectional groups
    and applies Bonferroni correction to control the family-wise error
    rate.

    Args:
        analyzer: An ``OutputAnalyzer`` instance.  A default one is
            created if not provided.
        alpha: Significance level for individual tests *before*
            Bonferroni correction.  Default ``0.05``.

    Example:
        >>> groups = IntersectionalGroup.from_attributes(
        ...     {"race": ["Black", "White"], "gender": ["male", "female"]}
        ... )
        >>> outputs = {g.label: llm_responses[g.label] for g in groups}
        >>> result = IntersectionalAnalyzer().analyze(outputs, groups)
        >>> print(result.has_intersectional_bias)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: llm_intersectional_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        analyzer: Optional[OutputAnalyzer] = None,
        alpha: float = 0.05,
    ) -> None:
        self.analyzer = analyzer or OutputAnalyzer(alpha=alpha)
        self.alpha = alpha

    def analyze(
        self,
        outputs_by_group: dict[str, list[str]],
        groups: list[IntersectionalGroup],
        metric: str = "sentiment",
    ) -> IntersectionalResult:
        """Run pairwise analysis across all intersectional groups.

        Args:
            outputs_by_group: Maps each group label to a list of LLM
                output texts.
            groups: The intersectional groups being compared.
            metric: Which analysis to run. Any ``OutputAnalyzer.analyze_<metric>``
                method: ``'sentiment'``, ``'toxicity'``, ``'refusal_rate'``,
                ``'helpfulness'``, ``'stereotype'``, ``'length'`` and the other
                metrics that class defines. This used to document ``'refusal'``,
                which resolves to nothing, so a caller following the docstring
                landed in the unknown-metric branch below.

        Returns:
            ``IntersectionalResult`` with pairwise comparisons,
            Bonferroni-corrected p-values, and summary statistics.
        """
        pairwise_results: list[OutputAnalysisResult] = []
        empty_labels: list[str] = []

        # BGL5 A-llm-3 (2026-09-27). TWO ROUTES REACHED THE RECORD THE UNKNOWN
        # METRIC HOIST BELOW EXISTS TO PREVENT, and neither went through the
        # metric lookup.
        #
        # (a) A DESIGN WITH NO PAIR. Measured on
        #     groups = from_attributes({"race": ["Black","White"], "gender": []})
        #     -> [] (from_attributes warns), then analyze({}, []):
        #       total_pairs 0, n_pairs_not_assessed 0, n_significant_pairs 0,
        #       max_disparity nan, has_intersectional_bias None,
        #       warnings from analyze: []
        #     A single group did the same and from_attributes does not warn for
        #     it at all. n_pairs_not_assessed is documented as "0 on a complete
        #     run", so the record said the run was complete while nothing was
        #     measured, and analyze said nothing. It now warns, naming the group
        #     count. There is genuinely no pair to record as not assessed, so
        #     the count stays 0 and the WARNING is the disclosure.
        #
        # (b) TWO GROUPS SHARING ONE LABEL. Measured on
        #     from_attributes({"race": ["Black","Black"], "gender": ["female"]})
        #     -> two groups both labelled 'Black_female', then analyze with one
        #     cell of three texts:
        #       total_pairs 1, n_pairs_not_assessed 0, max_disparity 0.0,
        #       has_intersectional_bias False
        #     which is documented as "a positive claim that every intersection
        #     was checked and none of them differed", made from ONE cell compared
        #     against ITSELF: `outputs_by_group.get(label)` returns the same list
        #     for both sides. A duplicate label pair is now recorded as not
        #     assessed with reason 'duplicate_group_label', so
        #     has_intersectional_bias reports None, and the duplicates are named
        #     in a warning.
        labels = [g.label for g in groups]
        duplicate_labels = sorted({lbl for lbl in labels if labels.count(lbl) > 1})
        if len(groups) < 2:
            warnings.warn(
                f"IntersectionalAnalyzer.analyze: a design of {len(groups)} group(s) has "
                f"no pair to compare, so NOTHING was measured. total_pairs 0 and "
                f"n_pairs_not_assessed 0 here mean the design contained no comparison, "
                f"NOT that a complete run found nothing; max_disparity is nan and "
                f"has_intersectional_bias reports None (could not check), NOT False. An "
                f"intersectional analysis needs at least two groups.",
                UserWarning,
                stacklevel=2,
            )
        # G12, 2026-09-30. THE CELLS AND THE DESIGN DO NOT HAVE TO PARTITION THE
        # SAME DATA, and only one direction of the mismatch was disclosed. A
        # design label with no cell is caught below (`empty_labels`), and a run
        # whose labels match NO group is caught by the same branch because every
        # cell then reads empty. The remaining door is a cell that carries REAL
        # OUTPUTS under a label the design does not contain: measured on a
        # complete 2x2 design plus one extra key 'Purple_alien' holding six
        # texts, total_pairs 6, n_pairs_not_assessed 0,
        # has_intersectional_bias False -- documented as "a positive claim that
        # every intersection was checked" -- and not one warning about the group
        # whose outputs were dropped. A label typo in a declared design does
        # exactly this, and it drops the group's data, not the design's.
        # A WARNING ONLY: the pairs that WERE compared are real comparisons, so
        # no verdict changes; what was missing is the disclosure.
        unused_labels = sorted(set(outputs_by_group) - set(labels))
        if unused_labels:
            warnings.warn(
                f"IntersectionalAnalyzer.analyze: {len(unused_labels)} label(s) in "
                f"outputs_by_group ({', '.join(unused_labels)}) name no group in the "
                f"supplied design, so their outputs were NEVER COMPARED to anything and "
                f"appear in no pair. The verdict below covers the {len(groups)} group(s) "
                f"of the design only. Check the labels agree with the design (a typo in "
                f"one label drops that group's data, not the design).",
                UserWarning,
                stacklevel=2,
            )
        if duplicate_labels:
            warnings.warn(
                f"IntersectionalAnalyzer.analyze: {len(duplicate_labels)} group label(s) "
                f"({', '.join(duplicate_labels)}) are carried by more than one of the "
                f"{len(groups)} supplied groups, so outputs_by_group holds ONE cell for "
                f"each of them and any pair between two of them would compare a cell "
                f"against itself. Those pairs are recorded as NOT ASSESSED "
                f"('duplicate_group_label'), never as a measured absence of disparity. "
                f"Give every intersection a distinct label.",
                UserWarning,
                stacklevel=2,
            )

        # BGL stage 5, 2026-09-27. The lookup was INSIDE the pair loop and its
        # failure branch did a bare `continue`, so an unrecognised metric name
        # dropped every pair of the design and the result reported
        # total_pairs=0, n_pairs_not_assessed=0, n_significant_pairs=0: the
        # field documented as "0 on a complete run" said the run was complete
        # while nothing whatsoever had been measured. Measured on a 2x2 design
        # (6 pairs) with metric="nonexistent": total_pairs=0,
        # n_pairs_not_assessed=0, max_disparity=nan, and the only disclosure was
        # a logger line, which a caller capturing warnings never sees. The
        # lookup is one decision about the whole run, so it belongs above the
        # dispatch, and its failure is now recorded pair by pair like any other
        # not-assessed pair.
        analyze_fn = getattr(self.analyzer, f"analyze_{metric}", None)
        if analyze_fn is None:
            logger.warning(
                "Unknown metric '%s'; skipping (valid: sentiment, toxicity, refusal_rate, helpfulness, stereotype, length)",
                metric,
            )
            warnings.warn(
                f"IntersectionalAnalyzer.analyze: '{metric}' is not a metric this "
                f"analyzer can run (no OutputAnalyzer.analyze_{metric}), so NOTHING was "
                f"measured. Every pair is recorded as not assessed and "
                f"has_intersectional_bias reports None (could not check), NOT False. "
                f"Valid names include sentiment, toxicity, refusal_rate, helpfulness, "
                f"stereotype and length.",
                UserWarning,
                stacklevel=2,
            )

        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                texts_a = outputs_by_group.get(g1.label, [])
                texts_b = outputs_by_group.get(g2.label, [])
                # BGL-D, 2026-09-11. This used to `continue`, deleting every
                # pair an empty cell belonged to with no record, no reason and
                # no warning. In a 2x2 design one empty cell is 3 of the 6
                # pairs, and the cell most likely to be empty is the one the
                # model refused or the harness choked on: the harmed one.
                # Measured on that fixture, with Black_female empty:
                # total_pairs=3 (the design has 6), n_significant_pairs=0,
                # max_disparity=0.0, has_intersectional_bias=False -- a positive
                # finding of NO intersectional bias for an analysis in which a
                # quarter of the design was never compared to anything -- and
                # most_disadvantaged == most_advantaged == 'Black_male'.
                # `.get(label, [])` above MANUFACTURES the empty list, so this
                # is the first and only place that can see it; there is no
                # upstream guard to lean on. The pair now enters the results as
                # not assessed, which makes total_pairs the real total and lets
                # has_intersectional_bias refuse.
                # A pair of two groups carrying the SAME label is one cell
                # against itself (see (b) above), so it is never measured.
                same_label = g1.label == g2.label
                if analyze_fn is None or same_label or not texts_a or not texts_b:
                    if analyze_fn is not None:
                        for g, texts in ((g1, texts_a), (g2, texts_b)):
                            if not texts and g.label not in empty_labels:
                                empty_labels.append(g.label)
                    pairwise_results.append(
                        OutputAnalysisResult(
                            group_a=g1.label,
                            group_b=g2.label,
                            metric=metric,
                            group_a_value=None,
                            group_b_value=None,
                            delta=None,
                            effect_size=None,
                            p_value=None,
                            is_significant=None,
                            sample_size=0,
                            effect_size_interpretation="not_assessed",
                            metadata=RunMetadata(
                                parameters={
                                    "n_supplied_a": len(texts_a),
                                    "n_supplied_b": len(texts_b),
                                }
                            ),
                            assessed=False,
                            not_assessed_reason=(
                                "unknown_metric"
                                if analyze_fn is None
                                else "duplicate_group_label"
                                if same_label
                                else "empty_input"
                            ),
                            n_supplied_a=len(texts_a),
                            n_supplied_b=len(texts_b),
                        )
                    )
                    continue

                result = analyze_fn(texts_a, texts_b)
                # Override group names to reflect intersectional labels
                result = OutputAnalysisResult(
                    group_a=g1.label,
                    group_b=g2.label,
                    metric=result.metric,
                    group_a_value=result.group_a_value,
                    group_b_value=result.group_b_value,
                    delta=result.delta,
                    effect_size=result.effect_size,
                    p_value=result.p_value,
                    is_significant=result.is_significant,
                    sample_size=result.sample_size,
                    effect_size_interpretation=result.effect_size_interpretation,
                    assessed=result.assessed,
                    not_assessed_reason=result.not_assessed_reason,
                )
                pairwise_results.append(result)

        # Bonferroni correction for multiple comparisons. LF-06: a pair whose
        # test could not run (p_value None) is not a member of the family; it
        # stays None rather than becoming "not significant".
        testable = [r for r in pairwise_results if r.p_value is not None]
        if len(testable) > 1:
            n_comparisons = len(testable)
            corrected: list[OutputAnalysisResult] = []
            for r in pairwise_results:
                adj_p = None if r.p_value is None else min(r.p_value * n_comparisons, 1.0)
                corrected.append(
                    OutputAnalysisResult(
                        group_a=r.group_a,
                        group_b=r.group_b,
                        metric=r.metric,
                        group_a_value=r.group_a_value,
                        group_b_value=r.group_b_value,
                        delta=r.delta,
                        effect_size=r.effect_size,
                        p_value=adj_p,
                        is_significant=(None if adj_p is None else bool(adj_p < self.alpha)),
                        sample_size=r.sample_size,
                        effect_size_interpretation=r.effect_size_interpretation,
                        assessed=r.assessed,
                        not_assessed_reason=r.not_assessed_reason,
                    )
                )
            pairwise_results = corrected

        n_sig = sum(1 for r in pairwise_results if r.is_significant is True)
        measured_deltas = [abs(r.delta) for r in pairwise_results if r.delta is not None]
        # No measured pair means no maximum, not a maximum of zero.
        max_disp = max(measured_deltas) if measured_deltas else float("nan")

        # Determine most advantaged / disadvantaged by mean metric value
        group_means: dict[str, float] = {}
        for g in groups:
            relevant = [r for r in pairwise_results if g.label in (r.group_a, r.group_b)]
            if relevant:
                vals = []
                for r in relevant:
                    v = r.group_a_value if r.group_a == g.label else r.group_b_value
                    if v is not None:
                        vals.append(v)
                if vals:
                    group_means[g.label] = float(np.mean(vals))

        # BGL-D. A ranking needs two ends. With one group mean, or with every
        # mean tied, `max()` and `min()` both return the FIRST key, so the same
        # group was handed back as most advantaged AND most disadvantaged -- a
        # ranking with no content, presented as a ranking. Observed on the
        # dropped-cell fixture above: both ends read 'Black_male'.
        most_adv_label: Optional[str] = None
        most_dis_label: Optional[str] = None
        if len(group_means) >= 2 and max(group_means.values()) > min(group_means.values()):
            most_adv_label = max(group_means, key=lambda k: group_means[k])
            most_dis_label = min(group_means, key=lambda k: group_means[k])

        n_not_assessed = sum(1 for r in pairwise_results if not r.assessed)
        if empty_labels:
            warnings.warn(
                f"IntersectionalAnalyzer.analyze: no outputs for "
                f"{', '.join(sorted(empty_labels))}. {n_not_assessed} of "
                f"{len(pairwise_results)} pairs were NOT ASSESSED and are excluded "
                f"from n_significant_pairs and max_disparity; "
                f"has_intersectional_bias reports None (could not check), NOT False.",
                UserWarning,
                stacklevel=2,
            )

        return IntersectionalResult(
            groups=groups,
            pairwise_results=pairwise_results,
            most_disadvantaged=next((g for g in groups if g.label == most_dis_label), None),
            most_advantaged=next((g for g in groups if g.label == most_adv_label), None),
            max_disparity=max_disp,
            n_significant_pairs=n_sig,
            total_pairs=len(pairwise_results),
            n_pairs_not_assessed=n_not_assessed,
        )
