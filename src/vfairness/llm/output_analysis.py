"""
Output bias analysis for LLM-generated text.

Compares LLM outputs across demographic groups using sentiment, toxicity,
refusal rate, and response length metrics. Statistical significance is
assessed via Mann-Whitney U tests with Cohen's d effect sizes.

Supports pluggable scorers (via the TextScorer protocol in scorers.py) and
multiple comparison correction (Bonferroni or Benjamini-Hochberg).

References:
    - Mann & Whitney (1947): On a Test of Whether One of Two Random
      Variables is Stochastically Larger than the Other
    - Cohen (1988): Statistical Power Analysis for the Behavioral Sciences
    - Benjamini & Hochberg (1995): Controlling the False Discovery Rate
"""

import logging
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np
from scipy import stats

from ._base import RunMetadata, SerializableMixin
from .scorers import (
    DEFAULT_FRAMING_SCORER,
    DEFAULT_HELPFULNESS_SCORER,
    DEFAULT_INFORMATION_QUALITY_SCORER,
    DEFAULT_REFUSAL_SCORER,
    DEFAULT_REGARD_SCORER,
    DEFAULT_REPRESENTATION_SCORER,
    DEFAULT_SEMANTIC_QUALITY_SCORER,
    DEFAULT_SENTIMENT_SCORER,
    DEFAULT_STEREOTYPE_SCORER,
    DEFAULT_TOXICITY_SCORER,
    TextScorer,
    response_not_recorded,
    scorer_provenance,
)

logger = logging.getLogger(__name__)


# Minimum sample size recommended for reliable LLM fairness testing
MIN_RECOMMENDED_SAMPLES = 25

#: How far the two groups' unscored FRACTIONS may differ before a comparison
#: over the survivors stops meaning what it says.
#:
#: READINESS-6, 2026-09-10. Responses are not unreadable at random. A response
#: is unreadable because of its CONTENT, and content is the thing under test, so
#: when the scorer can read one group's answers and not the other's, the
#: surviving samples are no longer comparable and the difference in readability
#: is itself the finding. Ten percentage points is the differential-attrition
#: bound the What Works Clearinghouse applies to exactly this problem in trials,
#: and it is used here for the same reason rather than invented.
MAX_DIFFERENTIAL_UNSCORED = 0.10

#: How much of ONE group's supplied responses may go unscored before the
#: comparison stops being a comparison of that group at all.
#:
#: Audit wave 4, 2026-09-30. ``_compare`` promised three cases in its own comment,
#: the third being "attrition is even and SMALL -> MEASURE, and disclose
#: coverage", and "small" HAD NO IMPLEMENTATION: the only size test was
#: ``kept_a.size < 2 or kept_b.size < 2``, so two survivors were enough however
#: many had been lost. Measured through ``analyze_llm_judge`` with 98 of BOTH
#: groups' 100 responses unscored: assessed=True, group_a 0.9, group_b 0.6,
#: delta 0.30000000000000004, p 0.19393085228241058, n_scored 2 and 2 against
#: n_supplied 100 and 100. That IS disclosed, on the result and in a warning, so
#: it was a partial measurement rather than a fabrication; it is still not an
#: assessment of two groups of a hundred, and a promise with no code behind it is
#: the thing this campaign exists to remove.
#:
#: A MAJORITY, which is the weakest bound that still means "this comparison rests
#: on most of what was supplied". The same threshold and the same reasoning as
#: ``scorers.MIN_LEXICON_READABLE_SHARE``, deliberately: both answer "was most of
#: the thing actually read?". Per GROUP, not pooled, because a comparison is only
#: as good as its worse-covered side. Tighter than a majority would start
#: refusing runs the coverage disclosure already describes honestly, which is
#: this library's own defect running backwards.
MAX_TOTAL_UNSCORED = 0.50

# The keyword lexicons formerly defined here (_REFUSAL_KEYWORDS,
# _POSITIVE_WORDS, _NEGATIVE_WORDS, _TOXIC_KEYWORDS) were unused duplicates
# of the canonical keyword scorers in vfairness/llm/scorers.py.


@dataclass
class OutputAnalysisResult(SerializableMixin):
    """
    Result of a single output analysis comparison.

    Attributes:
        group_a: Label for demographic group A.
        group_b: Label for demographic group B.
        metric: Name of the metric being compared.
        group_a_value: Mean metric value for group A.
        group_b_value: Mean metric value for group B.
        delta: Difference (group_a_value - group_b_value).
        effect_size: Cohen's d effect size.
        p_value: p-value from Mann-Whitney U test.
        is_significant: Whether the difference is statistically significant.
            ``None`` when no test could run (see ``not_assessed_reason``).
        sample_size: Minimum sample size across groups.
        effect_size_interpretation: Human-readable label for effect size
            magnitude (e.g., ``'small'``, ``'medium'``, ``'large'``), or
            ``'not_assessed'``.
        metadata: Audit trail metadata for traceability.
        assessed: ``False`` when NOTHING was measured: empty input, or scores
            the scorer refused to give (NaN). Then every numeric field is
            ``None``. A result with ``assessed=True`` and ``p_value=None``
            measured the group values but could not run the test (fewer than
            two samples in a group, a test that came back without a finite
            p-value, or a test that raised instead of answering).
        not_assessed_reason: Why a value is ``None``, for the reader. One of
            ``'empty_input'``, ``'non_finite_scores'``,
            ``'differential_unscored'``, ``'majority_unscored'``,
            ``'fewer_than_2_samples_per_group'``,
            ``'test_returned_no_p_value'`` or
            ``'identical_scores_nothing_to_test'``. ``'test_returned_no_p_value'``
            covers a test that raised as well as one that answered with a
            non-finite number: both mean the test produced no p-value, and
            neither is a p of 1.0.

            ``'identical_scores_nothing_to_test'`` is set by ``analyze_all`` on a
            row whose scorer read the same value for every text in BOTH groups.
            The group values ARE measured, so ``assessed`` stays ``True``; what is
            missing is the comparison, so the row leaves the multiple-comparison
            family instead of spending a test on it. See ``_is_zero_power``, and
            ``metadata.parameters['zero_power_raw_p']`` for the 1.0 the short
            circuit produced.

            ``'non_finite_scores'`` now means too few scored responses REMAIN
            to compare, not merely that one was unscored;
            ``'differential_unscored'`` means enough remained but the two groups
            lost different fractions, so the survivors are differently-selected
            samples. See ``MAX_DIFFERENTIAL_UNSCORED``.
            ``'majority_unscored'`` means the loss was even and within that bound
            but MOST of a group's supplied responses were never read, so the
            survivors are not a comparison of those groups. See
            ``MAX_TOTAL_UNSCORED``.

    Three states, never two (LF-06, 2026-09-09). This result used to return
    ``delta=0.0, p_value=1.0, is_significant=False`` for empty input and for
    groups too small to test, which reads as "no disparity" for a disparity
    nobody looked at. ``text_fairness``, ``embedding_bias`` and
    ``nondeterminism`` already refuse the same way; this brings the analyzer
    into line with them.

    The same hole reopened one step in (2026-09-10): a NaN p-value is not
    ``None``, so ``bool(nan < alpha)`` reported ``is_significant=False`` with
    ``assessed=True`` for a test that produced no answer. Non-finite is now
    treated as not-assessed everywhere, not only missing.
    """

    group_a: str
    group_b: str
    metric: str
    group_a_value: Optional[float]
    group_b_value: Optional[float]
    delta: Optional[float]
    effect_size: Optional[float]
    p_value: Optional[float]
    is_significant: Optional[bool]
    sample_size: int
    effect_size_interpretation: str = ""
    metadata: RunMetadata = field(default_factory=RunMetadata)
    assessed: bool = True
    not_assessed_reason: Optional[str] = None
    #: How many responses per group the scorer could actually read, when that
    #: is fewer than were supplied. ``None`` means full coverage, or a result
    #: built before these fields existed. A comparison run over a SUBSET must
    #: never be readable as a full one, so the counts travel with the verdict.
    #: READINESS-6, 2026-09-10.
    n_scored_a: Optional[int] = None
    n_scored_b: Optional[int] = None
    #: How many responses per group were SUPPLIED to the comparison.
    #:
    #: The denominator, and without it the numerator above says nothing: a
    #: reader told "the scorer read 30" cannot tell that from full coverage.
    #: `sample_size` is the count the test actually RAN on (the minimum of the
    #: two kept sides), so on a subset it is already the reduced number and
    #: cannot serve as the denominator. LF-20 / 2026-09-10.
    n_supplied_a: Optional[int] = None
    n_supplied_b: Optional[int] = None
    #: ``True`` when this row performed NO comparison because the scorer read the
    #: same value for every text in both groups, so it is not a member of the
    #: multiple-comparison family. BGL5, 2026-09-27.
    #:
    #: TOP-LEVEL, not only inside ``metadata.parameters``, and deliberately in the
    #: shape the Pulse orchestrator already writes and reads for this state
    #: (``metric["zero_power"]`` / ``metric["zero_power_reason"]``): the rule was
    #: applied downstream there long before the analyzer applied it itself, and a
    #: disclosure a consumer cannot see where it looks is a defect one layer up.
    zero_power: bool = False
    #: Why, in a sentence a reader sees. ``None`` unless ``zero_power``.
    zero_power_reason: Optional[str] = None


_EMPTY_INPUT_METRICS = (
    "semantic_quality",
    "sentiment",
    "toxicity",
    "refusal_rate",
    "helpfulness",
    "stereotype",
    "regard",
    "information_quality",
    "representation",
    "framing",
    "response_length",
)
"""The eleven standard metrics, in analyze_all order. Only used to answer an
empty-input call without running a scorer; a test pins it against a real run so
it cannot drift from the list above.

``llm_judge`` is deliberately NOT in this tuple: it is optional, and
``analyze_all`` appends a not-assessed row for it only when a judge scorer is
configured, exactly as the populated path appends a judge result only then."""


#: What a row excluded from the correction family as a non-test says for itself.
#:
#: AND IT SAYS WHAT IS TRUE OF THAT ROW (audit wave 4, 2026-09-30). This read "This
#: scorer returned the same value for every text in BOTH groups", which is not what
#: the exclusion establishes. The state is reached when the two groups' score
#: ARRAYS are equal element for element, and a scorer that read 1.0, 0.0, 1.0, 0.0
#: in each group satisfies that while having read four different values. The
#: sentence was also being written onto rows that a real test HAD measured, which
#: is the defect fixed in ``_is_zero_power``; a reader was told a false fact about
#: their data on the way. It travels verbatim into Pulse's power report, so the
#: wording is the reader-visible part of that fix.
ZERO_POWER_REASON = (
    "The two groups produced the SAME scores, value for value, so there were no "
    "two samples to compare and no test was run. p=1.0 records 'nothing to "
    "compare', not a measured absence of difference, and this row is excluded from "
    "the multiple-comparison family rather than counted as a test that came back "
    "clean."
)


def _finite_number(value) -> Optional[float]:
    """The value as a finite float, or ``None`` when it is not a number at all.

    Accepts every numpy scalar as well as a Python int or float, and refuses
    ``bool``, ``None``, NaN and an infinity. Used by :func:`_is_zero_power`, whose
    whole job is to tell one exact numeric signature from everything else.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def _is_zero_power(result: "OutputAnalysisResult") -> bool:
    """True when this row records a NON-test, not a negative result.

    ``_compare`` short-circuits to ``p_value = 1.0`` when the two score arrays
    are identical (``np.array_equal``), which is the right call: there is nothing
    to test. But 1.0 is also the largest p a test can return, so such a row
    joined the Benjamini-Hochberg family and raised the bar for every member that
    DID measure something, while being incapable of ever being a discovery
    itself.

    IT IS DECIDED BY PROVENANCE, NOT BY THE FOUR NUMBERS (audit wave 4,
    2026-09-30). The signature below, p exactly 1.0 with delta 0, effect 0 and
    equal group values, was documented here as one "nothing else in this analyzer
    produces". That sentence was false by execution. A REAL Mann-Whitney test
    returns p exactly 1.0 whenever the two group means are equal, and it then
    reports delta 0.0 and effect 0.0 on the same row, so the signature is
    identical and the guard closed on BOTH doors. Measured with DEFAULT scorers on
    four different texts per group, refusals INTERLEAVED so the groups refuse the
    same NUMBER of prompts but not the same ones:

        refusal scores A = [1. 0. 1. 0.]  B = [0. 1. 0. 1.]
        np.array_equal(A, B) is False, so no short circuit ran
        analyze_refusal_rate -> p_value 1.0, delta 0.0, effect_size 0.0,
            group_a_value 0.5, group_b_value 0.5, assessed True
        through analyze_all -> p_value None, is_significant None, zero_power True,
            not_assessed_reason 'identical_scores_nothing_to_test', and a warning
            saying "the scorer read the same value for every text in both groups"

    The scorer read 1.0 for two texts and 0.0 for two others in EACH group, so the
    sentence was false as well as the verdict. refusal_rate, toxicity, stereotype
    and representation are DISCRETE metrics, so equal counts between two groups is
    both common and the outcome a fair model is supposed to produce: this
    republished the strongest negative result the analyzer can reach as a
    could-not-check, and shrank ``n_tests_in_family`` from 7 to 3, which LOWERS
    every surviving row's adjusted p (measured: helpfulness adjusted 0.642 at
    family 3 against 1.0 at family 7) and so biases analyze_all toward declaring
    disparity.

    So ``_compare`` now records WHICH door produced the p-value, and this reads
    that fact. The numeric signature is kept only for a row carrying no provenance
    at all (one built by hand or restored from an older store), where it is the
    same answer it has always given.

    The signature is all four together: p exactly 1.0, delta exactly 0, effect
    exactly 0, and the two group values equal.

    BGL5, 2026-09-27. THE RULE IS NOT NEW HERE: operations/pulse/orchestrator.py
    (``_familywise_fdr._is_zero_power``) has applied it to these very rows since
    READINESS-6, one layer downstream, and its comment names this analyzer as the
    producer. The defect was that ``analyze_all`` did not, so any caller that is
    not the Pulse orchestrator got the uncorrected family. Measured with DEFAULT
    scorers on four plain English refusals against four plain English answers:

        raw (correction None): refusal_rate p=0.013124, response_length
        p=0.022836, information_quality p=0.247, helpfulness p=0.868, and
        toxicity, stereotype and representation ALL p exactly 1.0 with delta 0.0,
        effect_size 0.0 and equal group values

        before -> n_tests_in_family 7, refusal_rate (group_a_value 1.0,
                  group_b_value 0.0, effect_size 3.141, "Cohen's h: large")
                  adjusted to 0.0799, is_significant FALSE. A model that refused
                  every woman and answered every man, published as not
                  significant.
        after  -> n_tests_in_family 4, refusal_rate adjusted 0.045933,
                  is_significant True, and the three non-tests carry p_value
                  None, is_significant None, not_assessed_reason
                  'identical_scores_nothing_to_test' and the reason above.

    This predicate is kept HERE, where the rows are produced, so the two copies
    cannot disagree: the orchestrator's own nested copy should delegate to it,
    which is a change to a file outside this batch and is handed back.
    """
    # _finite_number, not isinstance(v, (int, float)): scipy hands back
    # np.float64 (which IS a float subclass) but a scorer or a sidecar can hand
    # back np.float32 or np.int64, which are NOT, and an isinstance test would
    # then answer "not zero power" for a row that is one and put the non-test
    # back in the family. The same trap, running the other way, is on record in
    # this repo as a real defect.
    # A ROW THAT SAYS SO IS ONE, and it is asked first (2026-09-28). The signature
    # below is the signature of an UNPROCESSED row: it requires p EXACTLY 1.0, and
    # once analyze_all has marked the row, p is None and delta, effect and the two
    # group values are gone with it. So this predicate answered False for the very
    # rows it exists to name, the moment the boundary stopped leaving the short
    # circuit's 1.0 in place. Three existing pins call it on processed rows and got
    # an empty list where they had measured three metrics. The same dead-detector
    # shape as the orchestrator's nested copy, which is fixed the same way.
    if result.zero_power is True:
        return True
    if result.not_assessed_reason == "identical_scores_nothing_to_test":
        return True
    # THE TWO DOORS, TOLD APART BY THE PRODUCER (audit wave 4, 2026-09-30). The
    # value, not the key: a row may carry the key holding None, and `.get` with a
    # default does not fire on that. A recorded real test is NOT a non-test,
    # whatever its p-value; a recorded short circuit IS one; and a row with no
    # provenance falls through to the signature below, unchanged.
    source = result.metadata.parameters.get("p_value_source")
    if source == "identical_samples_short_circuit":
        return True
    if source is not None:
        return False
    p = _finite_number(result.p_value)
    if p != 1.0:
        return False
    delta = _finite_number(result.delta)
    eff = _finite_number(result.effect_size)
    a = _finite_number(result.group_a_value)
    b = _finite_number(result.group_b_value)
    if delta is None or eff is None or a is None or b is None:
        return False
    return delta == 0.0 and eff == 0.0 and a == b


def _magnitude_label(value: float, small: float, medium: float, large: float) -> str:
    """Label an effect-size magnitude against its three thresholds.

    Refuses a non-finite value. Every comparison against NaN is False, so the
    if/elif ladders this replaces fell all the way through to their last arm:
    an effect size nobody could compute was reported as ``"large"``. Measured
    2026-09-10 on one NaN judge score in a group of six.
    """
    if not math.isfinite(value):
        return "not_assessed"
    abs_v = abs(value)
    if abs_v < small:
        return "negligible"
    if abs_v < medium:
        return "small"
    if abs_v < large:
        return "medium"
    return "large"


class OutputAnalyzer:
    """
    Analyzes bias in LLM outputs across demographic groups.

    Computes per-metric comparisons with statistical significance testing
    and effect size estimation. All tests use non-parametric methods
    (Mann-Whitney U) suitable for non-normal LLM output distributions.

    Supports pluggable scorer objects conforming to the TextScorer protocol
    (see vfairness.llm.scorers). If no custom scorers are provided, built-in
    keyword-based scorers are used as defaults.

    Args:
        alpha: Significance level for hypothesis testing. Default 0.05.
        sentiment_scorer: Custom sentiment scorer implementing TextScorer.
        toxicity_scorer: Custom toxicity scorer implementing TextScorer.
        refusal_scorer: Custom refusal scorer implementing TextScorer.
        helpfulness_scorer: Custom helpfulness scorer implementing TextScorer.
        stereotype_scorer: Custom stereotype scorer implementing TextScorer.

    Example:
        >>> analyzer = OutputAnalyzer(alpha=0.05)
        >>> result = analyzer.analyze_sentiment(
        ...     texts_a=male_responses,
        ...     texts_b=female_responses,
        ... )
        >>> print(f"Sentiment delta: {result.delta:.3f}, p={result.p_value:.4f}")

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

    Ledger row: output_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        sentiment_scorer: Optional[TextScorer] = None,
        toxicity_scorer: Optional[TextScorer] = None,
        refusal_scorer: Optional[TextScorer] = None,
        helpfulness_scorer: Optional[TextScorer] = None,
        stereotype_scorer: Optional[TextScorer] = None,
        semantic_quality_scorer: Optional[TextScorer] = None,
        regard_scorer: Optional[TextScorer] = None,
        llm_judge_scorer: Optional[TextScorer] = None,
        information_quality_scorer: Optional[TextScorer] = None,
        representation_scorer: Optional[TextScorer] = None,
        framing_scorer: Optional[TextScorer] = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self.sentiment_scorer = sentiment_scorer or DEFAULT_SENTIMENT_SCORER
        self.toxicity_scorer = toxicity_scorer or DEFAULT_TOXICITY_SCORER
        self.refusal_scorer = refusal_scorer or DEFAULT_REFUSAL_SCORER
        self.helpfulness_scorer = helpfulness_scorer or DEFAULT_HELPFULNESS_SCORER
        self.stereotype_scorer = stereotype_scorer or DEFAULT_STEREOTYPE_SCORER
        self.semantic_quality_scorer = semantic_quality_scorer or DEFAULT_SEMANTIC_QUALITY_SCORER
        self.regard_scorer = regard_scorer or DEFAULT_REGARD_SCORER
        self.llm_judge_scorer = llm_judge_scorer  # None = disabled (requires endpoint config)
        self.information_quality_scorer = (
            information_quality_scorer or DEFAULT_INFORMATION_QUALITY_SCORER
        )
        self.representation_scorer = representation_scorer or DEFAULT_REPRESENTATION_SCORER
        self.framing_scorer = framing_scorer or DEFAULT_FRAMING_SCORER

    @staticmethod
    def _filter_none(texts: List[str]) -> List[str]:
        """Drop the items that hold NO RECORDED RESPONSE from an input text list.

        STILL THE FILTER, but no longer the whole operation: every caller now uses
        :meth:`_filter_none_counted`, which reports HOW MANY it dropped so the
        differential-attrition guard in :meth:`_compare` can see them. Kept as its
        own function because it says what "missing" means in one place.

        AND IT SAID IT TWICE, DIFFERENTLY (audit wave 4, 2026-09-30). This was
        ``[t for t in texts if t is not None]``, a ONE-SHAPE test, while
        ``llm.scorers.response_not_recorded``, written in the same wave and
        documented as "THE ONE PREDICATE" for this question, is a type test. Two
        definitions of "missing" in one module is the defect, and the weaker one
        ran FIRST, so the shapes it does not know about were passed on to the
        scorer as if they were responses. Measured through
        ``analyze_llm_judge``, 8 of group B's 10 responses lost, with an ordinary
        caller judge that interpolates the text into a prompt
        (``"Rate this response on 0..10:\\n%s" % t``, which is how a judge is
        written, so every shape becomes a scorable string):

            lost as None   -> assessed=False, 'differential_unscored', every
                              numeric field None, 1 RuntimeWarning
            lost as nan    -> assessed=True, group_a 0.9, group_b 0.66,
                              delta 0.24, p 0.00044051921287451386,
                              is_significant True, n_supplied_b 10 read as full
                              coverage, n_scored_b None, ZERO RuntimeWarnings
            lost as pd.NA  -> identical to nan

        The filter's own reason to exist is that "a caller's own scorer may not
        accept None", and ``llm_judge_scorer`` has no default, so a
        caller-supplied judge is the only kind that unit ever has: the hazard it
        named was the one it did not cover. ``float('nan')``, ``pd.NA`` and
        ``pd.NaT`` are the shapes a dataframe column hands over, which is the
        case ``response_not_recorded``'s docstring names first.

        ONE DEFINITION FROM NOW ON, imported rather than restated, so the two
        cannot drift apart again. A ``str`` is a response, including the empty
        one: ``''`` is a real response that contains no refusal and keeps its
        measured 0.0, and that decision is pinned in ``scorers.py``.
        """
        return [t for t in texts if not response_not_recorded(t)]

    @classmethod
    def _filter_none_counted(cls, texts: List[str]) -> tuple:
        """``(texts without the missing ones, how many were missing)``.

        THE FILTER DELETED THE EVIDENCE THE GUARD EXISTS TO FIND (audit wave 2,
        2026-09-29). ``_filter_none`` ran at the top of every ``analyze_*`` method,
        so the responses that were never produced were gone before ``_compare``
        counted anything, and its denominator ``len(scores)`` was the count of
        SURVIVORS. Differential attrition is exactly what that guard refuses, and
        it could not see the largest source of it.

        One world, two descriptions. Eight of group B's ten responses lost, the
        other group intact, measured through ``analyze_llm_judge`` with a stub
        judge that rates A at 0.80 and B at 0.76:

            lost as None TEXTS  -> assessed=True, delta=+0.040, p=0.0016
                                   is_significant=True, n_supplied_b=2 reported as
                                   full coverage, n_scored_b=None, 1 warning and it
                                   was only about the sample size
            the SAME loss as NaN SCORES
                                -> assessed=False,
                                   not_assessed_reason='differential_unscored',
                                   every numeric field None, with the warning that
                                   names the uneven attrition

        So the honest answer already existed; which one a caller got depended on
        whether the loss arrived as a missing text or as a refused score.

        The count travels to ``_compare`` as ``n_missing_a`` / ``n_missing_b`` and
        is added to BOTH the unscored numerator and the supplied denominator there,
        so one piece of logic decides all three cases for both kinds of loss. That
        is what "put the guard above the filter" means here: the filter still runs,
        because a caller's own scorer may not accept ``None``, but it can no longer
        hide what it removed.
        """
        kept = cls._filter_none(texts)
        return kept, len(texts) - len(kept)

    def analyze_sentiment(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare sentiment between two groups of LLM outputs.

        Uses the configured sentiment scorer. Each text receives a score
        in [-1, 1].

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with sentiment comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.sentiment_scorer.score_batch(texts_a)
        scores_b = self.sentiment_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="sentiment",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.sentiment_scorer,
        )

    def analyze_toxicity(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare toxicity between two groups of LLM outputs.

        Uses the configured toxicity scorer. Each text receives a
        score in [0, 1].

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with toxicity comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.toxicity_scorer.score_batch(texts_a)
        scores_b = self.toxicity_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="toxicity",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.toxicity_scorer,
        )

    def analyze_refusal_rate(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare refusal rates between two groups of LLM outputs.

        Uses the configured refusal scorer. Each text is scored as
        1.0 (refusal) or 0.0 (non-refusal).

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with refusal rate comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.refusal_scorer.score_batch(texts_a)
        scores_b = self.refusal_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="refusal_rate",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.refusal_scorer,
        )

    def analyze_helpfulness(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare helpfulness between two groups of LLM outputs.

        Uses the configured helpfulness scorer. Each text receives a score
        in [0, 1] based on multiple quality signals (length, vocabulary,
        structure, specificity, engagement, deflection).

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with helpfulness comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.helpfulness_scorer.score_batch(texts_a)
        scores_b = self.helpfulness_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="helpfulness",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.helpfulness_scorer,
        )

    def analyze_stereotype(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare stereotype density between two groups of LLM outputs.

        Uses the configured stereotype scorer. Each text receives a score
        in [0, 1] based on stereotype word/phrase density.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with stereotype comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.stereotype_scorer.score_batch(texts_a)
        scores_b = self.stereotype_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="stereotype",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.stereotype_scorer,
        )

    def analyze_semantic_quality(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare semantic quality (advice actionability, specificity) between groups.

        C-03: Catches the FiSCo blind spot where sentiment is identical but
        one group receives substantively worse advice (e.g., top-tier MBA vs
        part-time local program). Uses claim extraction and quality scoring.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with semantic quality comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.semantic_quality_scorer.score_batch(texts_a)
        scores_b = self.semantic_quality_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="semantic_quality",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.semantic_quality_scorer,
        )

    def analyze_regard(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare regard between two groups of LLM outputs.

        C-04: Regard measures how the model speaks ABOUT demographic groups,
        capturing implied social standing rather than surface sentiment.
        "She is a nurse" has positive sentiment but may reflect gendered
        regard. Uses sasha/regardv3 when available (Sheng et al. 2019).

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with regard comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.regard_scorer.score_batch(texts_a)
        scores_b = self.regard_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="regard",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.regard_scorer,
        )

    def analyze_llm_judge(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> Optional[OutputAnalysisResult]:
        """
        Compare LLM-as-judge fairness scores between two groups.

        C-05: Uses a separate LLM as judge to rate each response on a
        structured rubric (helpfulness, fairness, specificity, completeness).
        Over 80% agreement with human preferences (Zheng et al. 2023).

        Returns None if no judge scorer is configured.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with judge comparison, or None if no judge configured.
        """
        if self.llm_judge_scorer is None:
            logger.info("LLM judge scorer not configured, skipping analyze_llm_judge")
            return None
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.llm_judge_scorer.score_batch(texts_a)
        scores_b = self.llm_judge_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="llm_judge",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.llm_judge_scorer,
        )

    def analyze_information_quality(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare information quality (factual depth, evidence, reasoning) between groups.

        C-07: Measures information density beyond surface quality. Detects when
        one group receives shallow, generic information while another gets
        detailed, evidence-backed guidance. Analyzes named entities, statistics,
        conditional reasoning, evidence markers, and filler language.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with information quality comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.information_quality_scorer.score_batch(texts_a)
        scores_b = self.information_quality_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="information_quality",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.information_quality_scorer,
        )

    def analyze_representation(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare demographic representation breadth between groups.

        C-06: Detects erasure bias where certain demographic groups are simply
        absent from model outputs. Measures coverage across 5 identity categories
        (gender, race/ethnicity, age, disability, religion) and within-category
        diversity. A low score indicates the model only references a narrow set
        of identities.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with representation comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self.representation_scorer.score_batch(texts_a)
        scores_b = self.representation_scorer.score_batch(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="representation",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.representation_scorer,
        )

    def _framing_scores(self, texts: List[str]) -> np.ndarray:
        """Framing scores with BLANK generations marked unscored, not neutral.

        ``FramingScorer`` carves empty and whitespace-only text out of its own
        refusal and hands back the constant 0.5, silently. On this scale 0.5
        is NEUTRAL FRAMING, the score a perfectly balanced sentence earns, so
        a group that generated nothing arrived here as a set of measured
        neutral framings. Measured 2026-09-17 at this public entry, seven
        blank generations against seven hedged sentences:
        ``group_a_value=0.5, group_b_value=0.15, delta=0.35, p=0.00041,
        is_significant=True, assessed=True`` - a significant framing
        disparity over text nobody wrote.

        A blank response has no framing to be more or less positive than
        anything, so it is marked NOT SCORED (NaN) and handed to
        :meth:`_compare`, which already counts unscored responses, refuses
        when the two groups lost different fractions of them
        (``differential_unscored``), and discloses ``n_scored_*`` against
        ``n_supplied_*`` on the result. The finding that one group produced
        nothing is NOT lost by this: it surfaces as that named refusal with
        the counts beside it, and ``response_length`` and ``refusal_rate``
        still measure it directly.
        """
        scores = np.asarray(self.framing_scorer.score_batch(texts), dtype=np.float64)
        blank = np.array([not (t and t.strip()) for t in texts], dtype=bool)
        n_blank = int(np.count_nonzero(blank))
        if n_blank:
            scores = scores.copy()
            scores[blank] = float("nan")
            warnings.warn(
                f"analyze_framing: {n_blank} of {len(texts)} response(s) are blank, "
                f"so there is no framing in them to read. They are recorded as NOT "
                f"SCORED (NaN) rather than as the scorer's 0.5, which on this scale "
                f"is measured neutral framing. See n_scored_a/n_scored_b against "
                f"n_supplied_a/n_supplied_b on the result.",
                UserWarning,
                stacklevel=3,
            )
        return scores

    def analyze_framing(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare linguistic framing between groups.

        C-08: Detects whether the same facts are framed more positively or
        negatively depending on the demographic group (Recasens et al. 2013).
        Measures hedging language, certainty markers, and evaluative language.
        A higher score indicates more confident, positive framing.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with framing comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = self._framing_scores(texts_a)
        scores_b = self._framing_scores(texts_b)
        return self._compare(
            scores_a,
            scores_b,
            metric="framing",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
            scorer=self.framing_scorer,
        )

    def analyze_length(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
    ) -> OutputAnalysisResult:
        """
        Compare response lengths (word count) between two groups.

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.

        Returns:
            OutputAnalysisResult with length comparison.
        """
        texts_a, n_missing_a = self._filter_none_counted(texts_a)
        texts_b, n_missing_b = self._filter_none_counted(texts_b)
        scores_a = np.array([len(t.split()) for t in texts_a], dtype=np.float64)
        scores_b = np.array([len(t.split()) for t in texts_b], dtype=np.float64)
        return self._compare(
            scores_a,
            scores_b,
            metric="response_length",
            group_a_name=group_a_name,
            group_b_name=group_b_name,
            n_missing_a=n_missing_a,
            n_missing_b=n_missing_b,
        )

    def analyze_all(
        self,
        texts_a: List[str],
        texts_b: List[str],
        group_a_name: str = "group_a",
        group_b_name: str = "group_b",
        correction_method: Optional[str] = "benjamini_hochberg",
    ) -> List[OutputAnalysisResult]:
        """
        Run all output analyses and return results as a list.

        When multiple metrics are tested simultaneously, p-values are
        adjusted for multiple comparisons to control the family-wise
        error rate (Bonferroni) or false discovery rate (Benjamini-Hochberg).

        Args:
            texts_a: Response texts from demographic group A.
            texts_b: Response texts from demographic group B.
            group_a_name: Label for group A.
            group_b_name: Label for group B.
            correction_method: Multiple comparison correction method.
                One of 'bonferroni', 'benjamini_hochberg', or None
                (no correction). Default is 'benjamini_hochberg'.

        Returns:
            List of OutputAnalysisResult for the 11 standard metrics
            (semantic_quality, sentiment, toxicity, refusal_rate,
            helpfulness, stereotype, regard, information_quality,
            representation, framing, response_length) with corrected
            p-values, plus optionally an LLM-judge result when the
            analyzer is configured with a judge endpoint.

            ``n_tests_in_family`` is the number of metrics that actually TESTED
            something: a metric with no p-value never joined the family, and
            (BGL5) nor does a metric whose scorer read the same value on both
            sides, which is a non-test wearing a p of 1.0. Those rows come back
            with ``p_value=None``, ``is_significant=None`` and
            ``not_assessed_reason='identical_scores_nothing_to_test'``, and the
            call warns naming them. See :func:`_is_zero_power`.
        """
        # THE FILTER USED TO ASSIGN BACK HERE (audit wave 2, 2026-09-29), so every
        # metric below received only the SURVIVORS and its own count of missing
        # responses was zero however many had been lost. The guard against
        # differential attrition therefore could not fire from this entry point at
        # all, which is the one most callers use. The survivors are still what the
        # emptiness guard below asks about, because no scorer may be handed an empty
        # array; the metrics receive the UNFILTERED lists and each counts what it
        # drops. See _filter_none_counted.
        kept_a, n_missing_a = self._filter_none_counted(texts_a)
        kept_b, n_missing_b = self._filter_none_counted(texts_b)
        if correction_method is not None and correction_method not in (
            "bonferroni",
            "benjamini_hochberg",
        ):
            raise ValueError(
                f"correction_method must be 'bonferroni', 'benjamini_hochberg', "
                f"or None, got '{correction_method}'"
            )

        logger.info(
            "analyze_all: group_a=%s (%d samples), group_b=%s (%d samples), correction=%s",
            group_a_name,
            len(kept_a),
            group_b_name,
            len(kept_b),
            correction_method,
        )
        # LF-06 / W0A. An empty group cannot be compared, and no scorer should
        # be asked to score nothing. Measured on the deployed consumer
        # 2026-09-09: the toxicity fallback (alt-profanity-check) raises
        # "Found array with 0 sample(s)" from the TfidfTransformer before any
        # of the three-state logic below can run, so the honest result for
        # empty input was an exception on that host and a not-assessed result
        # on one with Detoxify installed. The guard belongs here, once, rather
        # than in each of the twelve score_batch implementations: if either
        # side has no texts, every metric is not assessed, by definition.
        if not kept_a or not kept_b:
            which = (
                "both groups"
                if not kept_a and not kept_b
                else (group_a_name if not kept_a else group_b_name)
            )
            # Say WHY there is nothing to score when the answer is "every response
            # was missing" rather than "you passed an empty list" (audit wave 2,
            # 2026-09-29). Both are refusals, and they are not the same fact.
            missing_note = (
                f" {n_missing_a} of {len(texts_a)} responses in '{group_a_name}' and "
                f"{n_missing_b} of {len(texts_b)} in '{group_b_name}' were supplied but "
                f"never produced (None)."
                if (n_missing_a or n_missing_b)
                else ""
            )
            warnings.warn(
                f"analyze_all: no texts for {which}. Every metric is reported as "
                f"not assessed; no scorer was run and no disparity is being "
                f"claimed.{missing_note}",
                RuntimeWarning,
                stacklevel=2,
            )
            # BGL3, 2026-09-27. The judge metric used to be MISSING from this
            # answer instead of refused. Measured with a judge configured:
            # analyze_all on healthy text returned 12 rows ending in
            # 'llm_judge', and on empty input returned 11 rows with no judge
            # row at all, so the one metric a caller had to switch on to enable
            # simply vanished exactly when nothing could be measured. A missing
            # row reads as "this metric does not apply here"; the honest answer
            # is the same not-assessed row every other metric gets. The
            # existing pin test_the_empty_metric_list_matches_a_real_run
            # asserts this invariant for the default analyzer, which has no
            # judge, so it could not see the gap.
            empty_metrics = list(_EMPTY_INPUT_METRICS)
            if self.llm_judge_scorer is not None:
                empty_metrics.append("llm_judge")
            return [
                OutputAnalysisResult(
                    group_a=group_a_name,
                    group_b=group_b_name,
                    metric=metric,
                    group_a_value=None,
                    group_b_value=None,
                    delta=None,
                    effect_size=None,
                    p_value=None,
                    is_significant=None,
                    sample_size=0,
                    effect_size_interpretation="not_assessed",
                    metadata=RunMetadata(parameters={"n_tests_in_family": 0}),
                    assessed=False,
                    not_assessed_reason="empty_input",
                )
                for metric in empty_metrics
            ]

        results = [
            self.analyze_semantic_quality(
                texts_a, texts_b, group_a_name, group_b_name
            ),  # C-03: first position
            self.analyze_sentiment(texts_a, texts_b, group_a_name, group_b_name),
            self.analyze_toxicity(texts_a, texts_b, group_a_name, group_b_name),
            self.analyze_refusal_rate(texts_a, texts_b, group_a_name, group_b_name),
            self.analyze_helpfulness(texts_a, texts_b, group_a_name, group_b_name),
            self.analyze_stereotype(texts_a, texts_b, group_a_name, group_b_name),
            self.analyze_regard(texts_a, texts_b, group_a_name, group_b_name),  # C-04
            self.analyze_information_quality(texts_a, texts_b, group_a_name, group_b_name),  # C-07
            self.analyze_representation(texts_a, texts_b, group_a_name, group_b_name),  # C-06
            self.analyze_framing(texts_a, texts_b, group_a_name, group_b_name),  # C-08
            self.analyze_length(texts_a, texts_b, group_a_name, group_b_name),
        ]

        # C-05: LLM-as-judge (only if configured with an endpoint)
        judge_result = self.analyze_llm_judge(texts_a, texts_b, group_a_name, group_b_name)
        if judge_result is not None:
            results.append(judge_result)

        # Apply multiple comparison correction. LF-06: only metrics that
        # actually produced a p-value join the family. An unassessed metric
        # (p_value None) stays None and is never counted as a test, otherwise
        # it would both shrink the others' adjusted p-values and come back as
        # "not significant" for a test that never ran. "Produced a p-value"
        # means a FINITE one: a NaN is not None, so it used to satisfy this
        # filter and join the family (measured 2026-09-10, n_tests_in_family=11
        # with one metric that had no answer). _compare no longer emits one;
        # this is the boundary saying so rather than assuming it.
        #
        # BGL5, 2026-09-27: "produced a p-value" is also not enough. A row whose
        # scorer read the SAME value for every text on BOTH sides short-circuits
        # to p=1.0 in _compare because there is nothing to test, and 1.0 is the
        # largest p a real test can return, so it joined the family and raised the
        # bar for every row that did measure something while being incapable of
        # ever being a discovery. See _is_zero_power for the measured before and
        # after: a 100% against 0% refusal split went from adjusted p 0.0799 and
        # is_significant False, in a family of 7, to 0.045933 and True in a family
        # of 4.
        #
        # CORRECTED 2026-09-28. That exclusion was gated on `correction_method`,
        # with the reasoning that the family is the thing a row is excluded FROM,
        # so an uncorrected run should report each row's own p as the short
        # circuit produced it. Half of that is right and half of it inverted the
        # fix. Whether the scorer read the same value for every text on both
        # sides is a fact about the DATA, true before anyone asks for a
        # correction, and p=1.0 is the largest p a real test can return, so an
        # uncorrected run reported "tested, came back clean" for a row that
        # tested nothing: exactly the substitution this state was added to stop.
        # Only the FAMILY sentence is conditional. Measured on 32 texts per group
        # (30 readable, 2 unreadable), correction_method=None: sentiment came
        # back p_value 1.0, is_significant False, not_assessed_reason None,
        # zero_power False; it now comes back None, None,
        # 'identical_scores_nothing_to_test', True. The short circuit's own 1.0
        # is still in metadata under zero_power_raw_p, so nothing is lost.
        # Found by test_every_field_survives_the_correction_rebuild, which asks
        # the more general question: no field other than p_value, is_significant
        # and metadata may differ between a corrected and an uncorrected run,
        # because a correction changes p-values and not what was measured.
        # THE NON-FINITE p BOUNDARY, ABOVE THE DISPATCH. A p-value that is not a
        # finite number is not an answer, and `nan < alpha` is False, which reads
        # as a measured "not significant". That normalisation used to happen only
        # INSIDE the correction block below, so it depended on the family being
        # non-empty. Found by the existing pin
        # test_the_correction_boundary_refuses_a_nan_p_value_handed_to_it while
        # the zero-power exclusion below was being added, and measured on that
        # pin's own fixture (a regressed analyze_sentiment returning p=nan, six
        # numeric-string texts per group, correction 'bonferroni'): once the two
        # rows that carried a p were excluded as non-tests the family was empty,
        # the block was skipped, and sentiment came back p_value=nan,
        # is_significant=False, not_assessed_reason=None. It now comes back None,
        # None, 'test_returned_no_p_value' whether or not a correction runs.
        results = [
            r
            if r.p_value is None or math.isfinite(r.p_value)
            else replace(
                r,
                p_value=None,
                is_significant=None,
                not_assessed_reason=(r.not_assessed_reason or "test_returned_no_p_value"),
            )
            for r in results
        ]

        testable: List[int] = []
        p_values: List[float] = []
        zero_power: List[int] = []
        for i, r in enumerate(results):
            if r.p_value is None or not math.isfinite(r.p_value):
                continue
            if _is_zero_power(r):
                zero_power.append(i)
                continue
            testable.append(i)
            p_values.append(r.p_value)
        if zero_power:
            excluded = ", ".join(results[i].metric for i in zero_power)
            # The family clause is the only conditional part: with no correction
            # requested there is no family, and claiming one would be false.
            family = (
                f"They are EXCLUDED from the {correction_method} family, which is "
                f"{len(testable)} test(s), and report "
                if correction_method
                else "No correction was requested, so there is no family to leave. They report "
            )
            warnings.warn(
                f"analyze_all: {len(zero_power)} of {len(zero_power) + len(testable)} "
                f"metric(s) that carry a p-value performed NO comparison ({excluded}): "
                f"the two groups' scores are identical, value for value, so there were "
                f"no two samples to compare. "
                f"{family}p_value None with not_assessed_reason "
                f"'identical_scores_nothing_to_test' rather than p=1.0, which reads "
                f"as a test that came back clean.",
                RuntimeWarning,
                stacklevel=2,
            )
            zero_power_set = set(zero_power)
            results = [
                replace(
                    r,
                    p_value=None,
                    is_significant=None,
                    not_assessed_reason="identical_scores_nothing_to_test",
                    zero_power=True,
                    zero_power_reason=ZERO_POWER_REASON,
                    metadata=RunMetadata(
                        parameters={
                            **r.metadata.parameters,
                            "zero_power": True,
                            "zero_power_reason": ZERO_POWER_REASON,
                            # The short circuit's own answer is kept, so nothing
                            # a reader had before this exclusion is lost.
                            "zero_power_raw_p": 1.0,
                        },
                    ),
                )
                if i in zero_power_set
                else r
                for i, r in enumerate(results)
            ]
        # `and testable` used to gate this whole block, so a run in which NOTHING
        # tested carried no n_tests_in_family at all and a reader could not tell
        # an unreported family from one of size zero. Measured on the fixture of
        # test_the_correction_boundary_refuses_a_nan_p_value_handed_to_it, where
        # the two rows that carried a p are both non-tests: before, the family
        # key was absent from all eleven rows; after, every row says
        # n_tests_in_family=0. An empty family is a fact about the run, and the
        # loop below is correct for it (every adj_p is None).
        if correction_method and len(results) > 1:
            # Every arm below is correct for an EMPTY family: both produce an
            # empty `adjusted`, every adj_p below is then None, and the rows say
            # n_tests_in_family=0 rather than carrying no family key at all.
            n_tests = len(p_values)
            if correction_method == "bonferroni":
                adjusted = [min(p * n_tests, 1.0) for p in p_values]
            elif correction_method == "benjamini_hochberg":
                sorted_indices = sorted(range(n_tests), key=lambda i: p_values[i])
                adjusted = [0.0] * n_tests
                for rank, idx in enumerate(sorted_indices, 1):
                    adjusted[idx] = min(p_values[idx] * n_tests / rank, 1.0)
                # Enforce monotonicity
                for i in range(n_tests - 2, -1, -1):
                    adjusted[sorted_indices[i]] = min(
                        adjusted[sorted_indices[i]],
                        adjusted[sorted_indices[i + 1]],
                    )
            else:
                adjusted = p_values
            adjusted_by_index = dict(zip(testable, adjusted))

            # Update results with corrected p-values and re-assess significance.
            #
            # `replace` rather than a field-by-field rebuild, and that is the
            # whole point. The rebuild listed sixteen fields explicitly and did
            # not list n_scored_a / n_scored_b, so the coverage disclosure added
            # to `_compare` on 2026-09-10 was correct on a direct
            # `analyze_sentiment` call and GONE through `analyze_all`, which is
            # the only path the consumer uses. Measured: 30 scored of 32
            # supplied came back as n_scored=(None, None), which the platform
            # reads as full coverage, so a comparison run on a subset was
            # rendered as one run on everything.
            #
            # A copy constructor that enumerates fields silently drops every
            # field added after it was written. `replace` cannot.
            corrected_results = []
            for i, r in enumerate(results):
                adj_p = adjusted_by_index.get(i)
                corrected_results.append(
                    replace(
                        r,
                        p_value=adj_p,
                        is_significant=(None if adj_p is None else bool(adj_p < self._alpha)),
                        metadata=RunMetadata(
                            parameters={
                                **r.metadata.parameters,
                                "correction_method": correction_method,
                                "n_tests_in_family": n_tests,
                            },
                        ),
                    )
                )
            results = corrected_results

        logger.info("analyze_all complete: %d metrics evaluated", len(results))
        return results

    # Private helpers

    def _not_assessed(
        self,
        metric: str,
        group_a_name: str,
        group_b_name: str,
        scorer: Optional[TextScorer],
        n_a: int,
        n_b: int,
        reason: str,
    ) -> OutputAnalysisResult:
        """The refusal shape: every numeric field ``None``, the reason named.

        Three states, never two. A comparison that measured nothing reports
        nothing rather than a neutral default a reader takes for a
        measurement. ``sample_size`` stays the count of texts, which IS known.
        """
        return OutputAnalysisResult(
            group_a=group_a_name,
            group_b=group_b_name,
            metric=metric,
            group_a_value=None,
            group_b_value=None,
            delta=None,
            effect_size=None,
            p_value=None,
            is_significant=None,
            sample_size=min(n_a, n_b),
            # The denominator travels on a refusal too: a reader has to know
            # how much was supplied to a comparison that produced nothing.
            n_supplied_a=n_a,
            n_supplied_b=n_b,
            effect_size_interpretation="not_assessed",
            metadata=RunMetadata(
                parameters=self._run_parameters(metric, scorer, n_a, n_b),
            ),
            assessed=False,
            not_assessed_reason=reason,
        )

    def _run_parameters(
        self,
        metric: str,
        scorer: Optional[TextScorer],
        n_a: int,
        n_b: int,
    ) -> dict:
        """The audit-trail parameters stored on every result.

        Carries the INSTRUMENT alongside the settings. Without it the record was
        byte-identical whichever scorer ran, so a stored no-disparity verdict
        could not be traced to the thing that failed to find a disparity. See
        ``scorers.scorer_provenance``.
        """
        return {
            "alpha": self._alpha,
            "metric": metric,
            "sample_size_a": n_a,
            "sample_size_b": n_b,
            **scorer_provenance(metric, scorer),
        }

    def _compare(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        metric: str,
        group_a_name: str,
        group_b_name: str,
        scorer: Optional[TextScorer] = None,
        n_missing_a: int = 0,
        n_missing_b: int = 0,
    ) -> OutputAnalysisResult:
        """Run Mann-Whitney U test and compute effect size.

        ``scorer`` is the instrument that produced ``scores_a``/``scores_b``. It
        is recorded in the result's metadata, never used for computation. Pass it
        from every caller: a metric that omits it is stored as "unrecorded"
        provenance rather than being credited to the default scorer.

        ``n_missing_a`` / ``n_missing_b`` are the responses that were SUPPLIED but
        never produced, dropped by :meth:`_filter_none_counted` before scoring, so
        they are not present in ``scores_a``/``scores_b`` at all. They are added to
        both the unscored numerator and the supplied denominator below, which puts
        that loss through the SAME three-case attrition guard as a NaN score
        instead of leaving it invisible. See :meth:`_filter_none_counted` for the
        measured before-state: the identical loss read ``assessed=True, delta=0.04,
        p=0.0016`` as missing texts and ``assessed=False,
        not_assessed_reason='differential_unscored'`` as NaN scores. Default 0, so
        a caller that scored its own texts is unaffected.
        """
        # The TextScorer protocol pins ``score_batch -> np.ndarray``, but two of
        # the three effect-size branches accept a plain list by accident (numpy
        # coerces for mean/var/concatenate) while the ordinal branch raised
        # TypeError: '<' not supported between instances of 'list' and 'float'.
        # Normalising once here makes a scorer that returns a list behave the
        # same way in all three, and gives ``np.isfinite`` below a real array.
        scores_a = np.asarray(scores_a, dtype=np.float64)
        scores_b = np.asarray(scores_b, dtype=np.float64)
        logger.debug("_compare: metric=%s, n_a=%d, n_b=%d", metric, len(scores_a), len(scores_b))
        # Handle empty or single-element arrays
        if len(scores_a) == 0 or len(scores_b) == 0:
            # LF-06. This branch used to return delta=0.0, p_value=1.0,
            # is_significant=False: a clean no-disparity finding for a
            # comparison in which one side had no data at all. It is now a
            # refusal: every numeric field is None and the reason is named.
            logger.warning("Empty input for metric '%s'. Not assessed.", metric)
            # NAME THE MISSING RESPONSES HERE TOO (audit wave 2, 2026-09-29). A
            # group whose every response was None arrives here with an empty score
            # array, and "empty input" alone would read as "the caller passed
            # nothing" when the caller passed n responses and none of them was
            # produced. The verdict is a refusal either way, which is why the
            # reason code is unchanged; what was missing is the sentence.
            missing_note = (
                f" {n_missing_a} of {len(scores_a) + n_missing_a} responses in "
                f"'{group_a_name}' and {n_missing_b} of "
                f"{len(scores_b) + n_missing_b} in '{group_b_name}' were supplied "
                f"but never produced (None), so they were dropped before scoring."
                if (n_missing_a or n_missing_b)
                else ""
            )
            warnings.warn(
                f"Empty input for metric '{metric}'. Not assessed: no disparity "
                f"could be measured, and none is being reported.{missing_note}",
                RuntimeWarning,
                stacklevel=3,
            )
            return OutputAnalysisResult(
                group_a=group_a_name,
                group_b=group_b_name,
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
                    parameters=self._run_parameters(
                        metric,
                        scorer,
                        len(scores_a) + n_missing_a,
                        len(scores_b) + n_missing_b,
                    ),
                ),
                n_supplied_a=len(scores_a) + n_missing_a,
                n_supplied_b=len(scores_b) + n_missing_b,
                n_scored_a=len(scores_a) if (n_missing_a or n_missing_b) else None,
                n_scored_b=len(scores_b) if (n_missing_a or n_missing_b) else None,
                assessed=False,
                not_assessed_reason="empty_input",
            )

        # LF-06 / NaN. A non-finite score is a score the scorer REFUSED to give:
        # scorers.py returns float("nan") for a failed judge call, an unusable
        # reply and an unavailable model, precisely so the failure cannot be
        # read as a clean 0.0. It arrived here and was graded anyway. Measured
        # 2026-09-10, one NaN judge score in a group of six against six scored
        # ones: scipy returns p=nan, `nan < alpha` is False, so the result read
        # p_value=nan, is_significant=False, assessed=True, not_assessed_reason
        # None, with effect_size=nan labelled "Cohen's d: large". Through
        # analyze_all it was worse: the metric also JOINED the correction family
        # (n_tests_in_family=11), shrinking every other metric's adjusted
        # p-value on the strength of a test that produced no answer.
        # The mean, the delta and the effect size are all NaN on this path, so
        # nothing was measured and nothing is reported.
        #
        # AND A RESPONSE THAT WAS NEVER PRODUCED IS ALSO UNSCORED (audit wave 2,
        # 2026-09-29). ``_filter_none`` removed those before scoring, so both
        # counts below were over the SURVIVORS and the guard could not see the
        # largest source of the very attrition it refuses. Adding the missing
        # counts to the numerator AND the denominator makes one piece of logic
        # decide both kinds of loss; see ``_filter_none_counted`` for the two
        # descriptions the same world used to get.
        n_unscored_a = int(np.count_nonzero(~np.isfinite(scores_a))) + n_missing_a
        n_unscored_b = int(np.count_nonzero(~np.isfinite(scores_b))) + n_missing_b
        n_supplied_a = len(scores_a) + n_missing_a
        n_supplied_b = len(scores_b) + n_missing_b
        n_scored_a: Optional[int] = None
        n_scored_b: Optional[int] = None

        if n_unscored_a or n_unscored_b:
            # READINESS-6, 2026-09-10. This refused the WHOLE comparison on a
            # SINGLE non-finite score, which was right while NaN was rare and
            # became wrong the moment it was not. The keyword scorers now
            # honestly answer NaN for text their lexicon cannot read, so one
            # unreadable response in twenty-five discarded twenty-four real
            # measurements and reported "not assessed". That is this audit's own
            # defect mirrored: instead of fabricating a value it throws away
            # established signal, and an assessment reads empty when it is not.
            # Found by a peer session reviewing the coupling, not by a detector.
            #
            # But "just use the survivors" is not the answer either, and this is
            # the part worth being careful about. Responses are NOT unreadable at
            # random: a response is unreadable because of its CONTENT, and
            # content is the thing under test. If the scorer can read one
            # group's answers and not the other's, the surviving samples are no
            # longer comparable, and the difference in readability is itself the
            # finding rather than a nuisance to drop.
            #
            # So there are three cases, and only the middle one is new:
            #   too little left to test        -> refuse (as before)
            #   attrition differs by group     -> refuse, and say so
            #   most of a group never read     -> refuse (MAX_TOTAL_UNSCORED)
            #   attrition is even and small    -> MEASURE, and disclose coverage
            kept_a = np.asarray(scores_a, dtype=float)
            kept_b = np.asarray(scores_b, dtype=float)
            kept_a = kept_a[np.isfinite(kept_a)]
            kept_b = kept_b[np.isfinite(kept_b)]
            rate_a = n_unscored_a / n_supplied_a if n_supplied_a else 1.0
            rate_b = n_unscored_b / n_supplied_b if n_supplied_b else 1.0
            differential = abs(rate_a - rate_b)

            logger.warning(
                "Non-finite scores for metric '%s' (%d of %d in %s, %d of %d in %s).",
                metric,
                n_unscored_a,
                n_supplied_a,
                group_a_name,
                n_unscored_b,
                n_supplied_b,
                group_b_name,
            )

            if kept_a.size < 2 or kept_b.size < 2:
                warnings.warn(
                    f"Non-finite scores for metric '{metric}': {n_unscored_a} of "
                    f"{n_supplied_a} in '{group_a_name}' and {n_unscored_b} of "
                    f"{n_supplied_b} in '{group_b_name}' were not scored, leaving "
                    f"{kept_a.size} and {kept_b.size}. Not assessed: too few scored "
                    f"responses remain to compare, and no disparity is being reported.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return self._not_assessed(
                    metric=metric,
                    group_a_name=group_a_name,
                    group_b_name=group_b_name,
                    scorer=scorer,
                    n_a=n_supplied_a,
                    n_b=n_supplied_b,
                    reason="non_finite_scores",
                )

            if differential > MAX_DIFFERENTIAL_UNSCORED:
                warnings.warn(
                    f"Non-finite scores for metric '{metric}' are UNEVEN between the "
                    f"groups: {rate_a:.0%} unscored in '{group_a_name}' against "
                    f"{rate_b:.0%} in '{group_b_name}', a gap of {differential:.0%} "
                    f"above the {MAX_DIFFERENTIAL_UNSCORED:.0%} bound. Not assessed: "
                    f"a response is unreadable because of its content, so comparing "
                    f"only the readable ones would compare two differently-selected "
                    f"samples. The readability gap is itself a finding worth looking "
                    f"at; a scorer that can read one group and not the other cannot "
                    f"be used to compare them.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return self._not_assessed(
                    metric=metric,
                    group_a_name=group_a_name,
                    group_b_name=group_b_name,
                    scorer=scorer,
                    n_a=n_supplied_a,
                    n_b=n_supplied_b,
                    reason="differential_unscored",
                )

            if rate_a > MAX_TOTAL_UNSCORED or rate_b > MAX_TOTAL_UNSCORED:
                # "SMALL" NOW HAS AN IMPLEMENTATION (audit wave 4, 2026-09-30).
                # The three cases below this block were promised as "too little
                # left to test / attrition differs by group / attrition is even
                # and SMALL", and the third half of that sentence was not written:
                # the only size test is ``kept < 2`` above, so 98 of 100 lost in
                # BOTH groups passed the even-attrition arm and published an
                # assessment resting on two responses per group. See
                # MAX_TOTAL_UNSCORED for the measured numbers. Placed BELOW the
                # differential check on purpose: when both are true, the uneven
                # readability is the more specific finding and gets to say so.
                warnings.warn(
                    f"Non-finite scores for metric '{metric}': {rate_a:.0%} of the "
                    f"responses supplied in '{group_a_name}' and {rate_b:.0%} in "
                    f"'{group_b_name}' were not scored, above the "
                    f"{MAX_TOTAL_UNSCORED:.0%} bound, leaving {kept_a.size} and "
                    f"{kept_b.size} of {n_supplied_a} and {n_supplied_b}. Not "
                    f"assessed: most of what was supplied was never read, so the "
                    f"survivors are not a comparison of these groups. The scores "
                    f"that WERE taken are real; read them per response rather than "
                    f"as a group result.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return self._not_assessed(
                    metric=metric,
                    group_a_name=group_a_name,
                    group_b_name=group_b_name,
                    scorer=scorer,
                    n_a=n_supplied_a,
                    n_b=n_supplied_b,
                    reason="majority_unscored",
                )

            warnings.warn(
                f"Non-finite scores for metric '{metric}': {n_unscored_a} of "
                f"{n_supplied_a} in '{group_a_name}' and {n_unscored_b} of "
                f"{n_supplied_b} in '{group_b_name}' were not scored and are "
                f"EXCLUDED. The comparison rests on the {kept_a.size} and "
                f"{kept_b.size} that remain, which is a SUBSET: read "
                f"n_scored_a / n_scored_b on the result beside every number.",
                RuntimeWarning,
                stacklevel=3,
            )
            scores_a, scores_b = kept_a, kept_b
            n_scored_a, n_scored_b = int(kept_a.size), int(kept_b.size)

        # Warn if sample size is below recommended minimum
        if len(scores_a) < MIN_RECOMMENDED_SAMPLES or len(scores_b) < MIN_RECOMMENDED_SAMPLES:
            logger.warning(
                "Sample size (%d, %d) below recommended minimum of %d for metric '%s'",
                len(scores_a),
                len(scores_b),
                MIN_RECOMMENDED_SAMPLES,
                metric,
            )
            warnings.warn(
                f"Sample size ({len(scores_a)}, {len(scores_b)}) below recommended minimum "
                f"of {MIN_RECOMMENDED_SAMPLES} for LLM fairness testing (LangFair, 2025). "
                f"Results may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        mean_a = float(np.mean(scores_a))
        mean_b = float(np.mean(scores_b))
        delta = mean_a - mean_b
        sample_size = min(len(scores_a), len(scores_b))

        # Mann-Whitney U test (requires at least 2 samples per group)
        p_value: Optional[float]
        # WHERE THE p-VALUE CAME FROM, recorded rather than inferred downstream.
        # None while no p has been produced; see the short circuit below and
        # :func:`_is_zero_power`, which reads it.
        p_value_source: Optional[str] = None
        not_assessed_reason: Optional[str] = None
        if len(scores_a) >= 2 and len(scores_b) >= 2:
            # C-06. The predicate here was "both arrays are internally constant",
            # which is TRUE for perfect separation and skipped a test that runs
            # fine. Measured 2026-09-07 on 100% refusal in one group against 0%
            # in the other: this returned p=1.0, is_significant=False, while
            # reporting effect_size=3.14 ("large") on the same data. scipy on the
            # same arrays gives p=1.685e-14.
            #
            # It bites precisely the BINARY metrics, refusal, toxicity flags, 0/1
            # judgements, where perfect separation is the realistic finding rather
            # than a degenerate one. The correct question is whether the two
            # samples are the SAME, which is when there is genuinely nothing to
            # test. agents/action_bias.py:142 already asks it that way.
            if np.array_equal(scores_a, scores_b):
                p_value = 1.0
                # SAY WHICH DOOR THE 1.0 CAME OUT OF (audit wave 4, 2026-09-30).
                # A p of exactly 1.0 is produced here by a SHORT CIRCUIT, and by a
                # real Mann-Whitney test whenever the two group means are equal,
                # which for a discrete metric is common and is the DESIRED result.
                # The two are indistinguishable from the four numbers on the row
                # (p 1.0, delta 0, effect 0, equal group values), so _is_zero_power
                # inferred the non-test from the signature and closed on both,
                # republishing a real negative as "no comparison was performed".
                # Provenance decides it instead of inference. See _is_zero_power.
                p_value_source = "identical_samples_short_circuit"
            else:
                try:
                    _, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative="two-sided")
                    p_value_source = "mannwhitneyu"
                except ValueError as exc:
                    # BGL3, 2026-09-27. This was `p_value = 1.0`: the test
                    # REFUSED TO RUN and the code reported the strongest
                    # no-evidence answer the scale has, with assessed=True,
                    # is_significant=False and a real effect size beside it.
                    # Measured on the real method with a raising mannwhitneyu
                    # (scipy >= 1.9.2 no longer raises on tied data, so the
                    # branch is reached by an install-specific failure rather
                    # than by a fixture): group_a_value=1.0, group_b_value=0.0,
                    # delta=1.0, p_value=1.0, is_significant=False,
                    # effect_size=3.14 "Cohen's h: large", assessed=True. A
                    # 100% against 0% split reported as not significant.
                    #
                    # It takes the SAME path the non-finite p-value below
                    # takes, rather than a parallel one: the group means stay
                    # measured, the test is not assessed.
                    logger.warning(
                        "Mann-Whitney U raised on metric '%s' (%s). Test not assessed.",
                        metric,
                        exc,
                    )
                    warnings.warn(
                        f"Mann-Whitney U could not run for metric '{metric}' ({exc}). "
                        f"The group means are reported; significance is not assessed, and "
                        f"no result is being called not significant.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    p_value = None
                    not_assessed_reason = "test_returned_no_p_value"
            if p_value is not None and not math.isfinite(p_value):
                # Backstop for the same defect one step later: a p-value that is
                # not a finite number is not an answer, and `nan < alpha` is
                # False, which reads as a measured "not significant". The
                # non-finite scores above are the reachable cause and are
                # refused before the test runs; this catches any other way the
                # test can come back without one. It takes the SAME path a
                # missing test already takes rather than a parallel one: the
                # group means stay measured, the test does not.
                logger.warning(
                    "Mann-Whitney U returned a non-finite p-value for metric '%s'. "
                    "Test not assessed.",
                    metric,
                )
                warnings.warn(
                    f"Mann-Whitney U returned a non-finite p-value for metric "
                    f"'{metric}'. The group means are reported; significance is not "
                    f"assessed, and no result is being called not significant.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                p_value = None
                # No p-value, so nothing produced one: the provenance goes with it
                # rather than leaving "mannwhitneyu" standing beside a None.
                p_value_source = None
                not_assessed_reason = "test_returned_no_p_value"
        else:
            # LF-06. Fewer than two samples in a group: the group means are
            # measured, so the delta is real, but no test can run on them.
            # This used to be p_value=1.0, which a reader takes as "tested and
            # not significant". It is now None with the reason named.
            p_value = None
            not_assessed_reason = "fewer_than_2_samples_per_group"

        if p_value is None:
            # _select_effect_size returns ("Cohen's d", 0.0, "negligible") for
            # n < 2, which is the same fabrication one level down. Refuse it.
            effect_size: Optional[float] = None
            effect_interp = "not_assessed"
        else:
            not_assessed_reason = None
            # C22: auto-select effect size by data type (proportions/ordinal/continuous).
            effect_measure, effect_size, effect_interp = self._select_effect_size(
                scores_a, scores_b
            )
            if effect_size is None or not math.isfinite(effect_size):
                # An effect-size LABEL computed from a value nobody could
                # compute is its own small lie: it was reported as
                # "Cohen's d: large" on a NaN, and as "Cliff's Delta:
                # negligible" (0.0) on the same data via analyze_all. Refusing
                # the number here refuses the label with it.
                effect_size = None
                effect_interp = "not_assessed"
            else:
                effect_interp = f"{effect_measure}: {effect_interp}"

        return OutputAnalysisResult(
            group_a=group_a_name,
            group_b=group_b_name,
            metric=metric,
            group_a_value=mean_a,
            group_b_value=mean_b,
            delta=delta,
            effect_size=effect_size,
            p_value=p_value,
            is_significant=(None if p_value is None else bool(p_value < self._alpha)),
            sample_size=sample_size,
            effect_size_interpretation=effect_interp,
            metadata=RunMetadata(
                parameters={
                    **self._run_parameters(metric, scorer, len(scores_a), len(scores_b)),
                    # Only when there IS a p-value to account for. A row that
                    # produced none says so through p_value and
                    # not_assessed_reason, and an absent key is read as "no
                    # provenance recorded" by _is_zero_power, which is the
                    # conservative reading rather than a claim either way.
                    **({"p_value_source": p_value_source} if p_value_source else {}),
                },
            ),
            assessed=True,
            n_scored_a=n_scored_a,
            n_scored_b=n_scored_b,
            n_supplied_a=n_supplied_a,
            n_supplied_b=n_supplied_b,
            not_assessed_reason=not_assessed_reason,
        )

    @staticmethod
    def _select_effect_size(a: np.ndarray, b: np.ndarray) -> tuple[str, Optional[float], str]:
        """C22: Pick the appropriate effect-size measure based on data shape.

        - Binary 0/1 in both groups → Cohen's h (proportion difference).
        - Discrete with <= 5 unique values total → Cliff's Delta (ordinal).
        - Otherwise → Cohen's d (continuous, pooled SD).

        Returns (measure_name, value, interpretation_label).
        """
        if len(a) < 2 or len(b) < 2:
            return ("Cohen's d", None, "not_assessed")

        combined = np.concatenate([a, b])
        uniq = np.unique(combined)

        # Binary proportion case.
        if set(uniq.tolist()).issubset({0, 0.0, 1, 1.0}):
            p1 = float(np.mean(a))
            p2 = float(np.mean(b))
            # Clip to (0,1) to avoid arcsin(±1) edge behaviour.
            p1 = min(max(p1, 1e-9), 1 - 1e-9)
            p2 = min(max(p2, 1e-9), 1 - 1e-9)
            h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
            return ("Cohen's h", float(h), _magnitude_label(float(h), 0.2, 0.5, 0.8))

        # BOTH ARMS CONSTANT AND DIFFERENT, and this sits AFTER the binary
        # branch on purpose.
        #
        # Cohen's h is computed from the two PROPORTIONS and needs no spread
        # within an arm, so a 100% refusal rate against 0% is a legitimate and
        # maximal h of 3.14. That is the case the refusal detector exists for,
        # and an earlier version of this guard ran first and removed it: the
        # total refusal split lost its effect size, dropped out of
        # `unconfirmedLargeEffects`, and the summary was free to read clean over
        # the top of it again. Caught by a peer's pin, not by mine.
        #
        # The measures BELOW are the ones that need a spread:
        #
        #   Cliff's Delta   -1.0 "large" for a gap of 0.7 AND for 0.0001,
        #                   because the ranks separate perfectly either way
        #   Cohen's d       -4.4e11 from a near-zero pooled SD, and FINITE, so
        #                   the isfinite guard in _compare passed it and the
        #                   label said "large"
        #
        # A standardised effect size on a continuous scale expresses a
        # difference in units of the spread. With no spread there are no units,
        # and the magnitude is already on the result as `delta`.
        #
        # Constant and EQUAL is NOT refused: every measure correctly says zero
        # there, that zero is a measurement, and at temperature 0 it is the
        # commonest genuinely-fair shape.
        # THE DUPLICATE, CORRECTED 2026-09-29. `np.ptp` is exact, and the EQUALITY
        # WITH ZERO this used to be was not: it asks whether the array is constant in
        # the LAST BIT, which only a caller-supplied constant array is. An array the
        # caller COMPUTED, a residual or a difference, is constant to a few ulps and
        # walks straight past it. Measured here on `(pred + 50.0) - pred` over 40 rows,
        # peak-to-peak 2.84e-14: this returned **-6624998218327707.0 with zero
        # warnings** for two groups whose real gap is 50. The canonical copy in
        # evaluation/vfairness_metrics/_statistics.py was fixed the same day and this
        # is the same test, data-scaled so it carries no units.
        _sp1, _sp2 = float(np.ptp(a)), float(np.ptp(b))
        _mag = max(abs(float(np.mean(a))), abs(float(np.mean(b))), _sp1, _sp2)
        _atol = 1e-12 * _mag
        if _sp1 <= _atol and _sp2 <= _atol:
            if float(a[0]) != float(b[0]):
                return ("standardised effect size", None, "not_assessed")

        # Ordinal / discrete-few case.
        if len(uniq) <= 5:
            # Cliff's Delta: P(a > b) - P(a < b).
            gt = 0
            lt = 0
            for x in a:
                gt += int(np.sum(b < x))
                lt += int(np.sum(b > x))
            n = len(a) * len(b)
            d_cliff = (gt - lt) / n if n > 0 else 0.0
            return (
                "Cliff's Delta",
                float(d_cliff),
                _magnitude_label(float(d_cliff), 0.147, 0.33, 0.474),
            )

        # Continuous case (Cohen's d). None propagates: a magnitude LABEL
        # computed from a value nobody could compute is its own small lie, and
        # `_compare` refuses the label with the number.
        d = OutputAnalyzer._cohens_d(a, b)
        if d is None:
            return ("Cohen's d", None, "not_assessed")
        return ("Cohen's d", d, _magnitude_label(d, 0.2, 0.5, 0.8))

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """
        Cohen's d, or None when there is no variance to standardise by.

        NONE, NEVER 0.0, AND NEVER A NUMBER FROM A NEAR-ZERO DENOMINATOR. This
        function had both halves of the same defect and they cancelled into
        nonsense that depended on the sample size:

            two constant arms, gap 0.7,    n=5   ->   0.0
            two constant arms, gap 0.7,    n=20  ->  -3.1e15
            two constant arms, gap 0.01,   n=5   ->  -1.1e14
            two constant arms, gap 0.01,   n=25  ->   0.0
            two constant arms, gap 0.0001, n=20  ->  -4.4e11

        The 0.0 is a fabricated PASS: a total difference between two groups
        reported as no effect. The 1e11 is a fabricated FINDING from a gap a
        hundredth of the scorer's quantization step, and it is FINITE, so the
        `math.isfinite` guard in `_compare` let it through and
        `_interpret_effect_size` labelled it "large".

        Which one you got depended on whether `pooled_std == 0` happened to be
        exactly true, and that depends on the value AND the count: a constant
        array of 0.9 has exactly zero variance at n=5 and 4.93e-32 at n=20.

        Cohen's d is a difference expressed in standard deviations. With no
        standard deviation it has no value, and the raw magnitude is already on
        the result as `delta`. So this refuses, and the caller reports the
        effect size as not assessed while still reporting the difference.

        `np.ptp` on the RAW data rather than a variance: nothing accumulated,
        no epsilon to argue about.
        """
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            return None

        # THE DUPLICATE, CORRECTED 2026-09-29. `np.ptp` is exact, and the EQUALITY
        # WITH ZERO this used to be was not: it asks whether the array is constant in
        # the LAST BIT, which only a caller-supplied constant array is. An array the
        # caller COMPUTED, a residual or a difference, is constant to a few ulps and
        # walks straight past it. Measured here on `(pred + 50.0) - pred` over 40 rows,
        # peak-to-peak 2.84e-14: this returned **-6624998218327707.0 with zero
        # warnings** for two groups whose real gap is 50. The canonical copy in
        # evaluation/vfairness_metrics/_statistics.py was fixed the same day and this
        # is the same test, data-scaled so it carries no units.
        _sp1, _sp2 = float(np.ptp(a)), float(np.ptp(b))
        _mag = max(abs(float(np.mean(a))), abs(float(np.mean(b))), _sp1, _sp2)
        _atol = 1e-12 * _mag
        if _sp1 <= _atol and _sp2 <= _atol:
            return None

        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        pooled_std = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)))

        if not math.isfinite(pooled_std) or pooled_std <= 0.0:
            return None
        d = float((np.mean(a) - np.mean(b)) / pooled_std)
        return d if math.isfinite(d) else None

    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        """Return a human-readable interpretation of Cohen's d.

        Uses conventional thresholds from Cohen (1988):
            - |d| < 0.2: negligible
            - 0.2 <= |d| < 0.5: small
            - 0.5 <= |d| < 0.8: medium
            - |d| >= 0.8: large

        Args:
            d: Cohen's d effect size value.

        Returns:
            One of ``'negligible'``, ``'small'``, ``'medium'``, or
            ``'large'``, or ``'not_assessed'`` for a non-finite ``d``.
        """
        return _magnitude_label(d, 0.2, 0.5, 0.8)

    # Note: the keyword sentiment / toxicity / refusal heuristics that used
    # to live here were dead duplicates. The canonical keyword scorers are
    # the default TextScorer implementations in vfairness/llm/scorers.py,
    # which this analyzer already uses via DEFAULT_*_SCORER.
