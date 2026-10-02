"""
Correspondence testing for AI agent fairness.

Implements paired-artifact testing methodology adapted from employment
discrimination audit studies (e.g., Bertrand & Mullainathan, 2004) for
use with AI agents. Creates matched artifacts that differ only on a
protected attribute, then measures outcome disparities.
"""

import copy
import logging
import math
import uuid
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from vfairness.evaluation.vfairness_metrics._statistics import (
    _mannwhitney_two_sided_p,
    detectability,
)
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)


def _is_measured(value: object) -> bool:
    """Is this outcome a real number a test can use?

    READINESS-6. The filter this replaces was ``isinstance(v, float) and
    np.isnan(v)``, which is true of a Python float NaN and of nothing else, so
    ``None`` and every numpy scalar walked past it into ``np.asarray(...,
    dtype=float)`` and became NaN there. Anything that is not a finite number is
    an outcome nobody recorded, whatever its type.
    """
    if value is None:
        return False
    # A bool needs no special case: float(True) is 1.0 and is finite, which is
    # the right reading of a recorded yes/no outcome. Spelling that out because
    # the first attempt at this function DID special-case it and let None
    # through as a result, which is the very value it exists to catch.
    try:
        # `float(value)` is deliberately attempted on an arbitrary object: the
        # point is to accept anything that CAN be a number (Python floats and
        # ints, numpy scalars, Decimal) and reject everything else, and the
        # except clause below is the rejection. mypy cannot see that, hence the
        # ignore; the behaviour is pinned in tests/test_readiness4_agents.py.
        return bool(np.isfinite(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass
class CorrespondenceResult(SerializableMixin):
    """Result of a correspondence test between two demographic groups.

    Attributes:
        test_id: Unique identifier for this test run.
        artifact_type: Type of artifact tested (e.g., 'resume', 'profile').
        demographic_a: Label for the first demographic group.
        demographic_b: Label for the second demographic group.
        outcome_a: Aggregated outcome for group A (e.g., callback rate).
        outcome_b: Aggregated outcome for group B.
        disparity_metric: Computed disparity between groups.
        sample_size: Number of paired observations.
        is_significant: Whether the disparity is statistically significant, or
            ``None`` when no test could be run (see ``n_unmeasurable``). Never
            ``False`` for a comparison that did not happen.
        p_value: P-value from the statistical test, NaN when none was run.
        n_unmeasurable: Outcomes dropped because they carried no value. Counted
            and reported rather than silently removed, because the sample size
            beside it is the count that REMAINED.
        min_attainable_p: The smallest p-value this test's ARM SIZES can
            produce, ``None`` when it was not computed. G12, 2026-09-30.
        detectable: Could this design ever have reached ``alpha``? ``False``
            means no data, however extreme, could have made it significant, so
            ``is_significant=False`` beside it is an absence of power and not
            evidence that the groups were treated alike. ``None`` is
            could-not-check. Named as the three sibling modules name it
            (``CollusionResult``, ``DelegationResult``, ``NegotiationResult``),
            so one vocabulary covers the design floor across the library.
        detectability_note: The sentence a reader needs when ``detectable`` is
            not ``True``; empty when the design has power.
    """

    test_id: str
    artifact_type: str
    demographic_a: str
    demographic_b: str
    outcome_a: float
    outcome_b: float
    disparity_metric: float
    sample_size: int
    is_significant: Optional[bool]
    p_value: float
    n_unmeasurable: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    metadata: RunMetadata = field(default_factory=RunMetadata)
    min_attainable_p: Optional[float] = None
    detectable: Optional[bool] = None
    detectability_note: str = ""


class CorrespondenceTester:
    """Correspondence testing framework for AI agent bias detection.

    Creates paired artifacts that vary only on a demographic attribute,
    submits them to an agent, and statistically tests for outcome disparities.

    Args:
        alpha: Significance level for hypothesis tests.

    Example:
        >>> tester = CorrespondenceTester(alpha=0.05)
        >>> artifacts = tester.create_paired_artifacts(
        ...     base_artifact={"name": "PLACEHOLDER", "experience": 5},
        ...     demographic_field="name",
        ...     values=["John Smith", "Jamal Washington"],
        ... )
        >>> result = tester.analyze_outcomes(
        ...     outcomes_a=[1, 0, 1, 1, 0],
        ...     outcomes_b=[0, 0, 1, 0, 0],
        ...     artifact_type="resume",
        ... )

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: correspondence_tester. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.alpha = alpha
        self.random_seed = random_seed
        logger.info("CorrespondenceTester initialized: alpha=%.3f", alpha)

    def create_paired_artifacts(
        self,
        base_artifact: dict,
        demographic_field: str,
        values: List[str],
    ) -> List[dict]:
        """Generate matched artifacts varying only on the demographic field.

        Creates deep copies of the base artifact, substituting the
        demographic field with each provided value. All other fields
        remain identical to isolate the effect of the demographic attribute.

        Args:
            base_artifact: Template artifact dictionary.
            demographic_field: Key in the artifact to vary.
            values: Demographic values to substitute.

        Returns:
            List of artifact dicts, one per value, each with a unique
            ``_artifact_id`` and ``_demographic_value`` metadata field.

        Raises:
            ValueError: If fewer than 2 values are provided, or if the values
                supplied hold fewer than 2 DISTINCT values, which is a paired
                design with no contrast in it.
        """
        if len(values) < 2:
            raise ValueError("At least 2 demographic values are required.")
        # BGL-3 (2026-09-27): the guard above counts ENTRIES, and a
        # correspondence study is built on the pair differing on the protected
        # attribute and on nothing else. Measured that day,
        # values=["John Smith", "John Smith"] passed it and returned two
        # artifacts whose 'name' was byte for byte identical, with no warning and
        # no error, so the study that followed varied NOTHING while reporting a
        # disparity on the demographic field. On that design a 0.0 disparity from
        # analyze_outcomes is byte for byte what MEASURED equal treatment
        # returns, and the attribute it names was never manipulated.
        #
        # Repeated values ALONGSIDE a second distinct one are left alone on
        # purpose: replicate arms are a normal part of this method, and only the
        # total absence of a contrast makes the design unmeasurable. repr() is
        # used for the distinct count so an unhashable value cannot turn this
        # guard into a TypeError.
        if len({repr(v) for v in values}) < 2:
            raise ValueError(
                f"At least 2 DISTINCT demographic values are required: all "
                f"{len(values)} supplied values are {values[0]!r}, so the artifacts "
                f"would differ on nothing and no outcome difference could be "
                f"attributed to '{demographic_field}'."
            )
        if demographic_field not in base_artifact:
            raise ValueError(
                f"demographic_field '{demographic_field}' not found in base_artifact. "
                f"Available keys: {list(base_artifact.keys())}"
            )

        artifacts = []
        for value in values:
            artifact = copy.deepcopy(base_artifact)
            artifact[demographic_field] = value
            artifact["_artifact_id"] = str(uuid.uuid4())
            artifact["_demographic_value"] = value
            artifacts.append(artifact)

        return artifacts

    def analyze_outcomes(
        self,
        outcomes_a: List[float],
        outcomes_b: List[float],
        artifact_type: str = "resume",
    ) -> CorrespondenceResult:
        """Analyze outcome disparities between two demographic groups.

        Uses chi-square test for categorical (binary) outcomes and
        Mann-Whitney U test for numeric outcomes.

        Args:
            outcomes_a: Outcomes for demographic group A.
            outcomes_b: Outcomes for demographic group B.
            artifact_type: Label for the type of artifact tested.

        Returns:
            CorrespondenceResult with disparity metrics and significance.

        Raises:
            ValueError: If outcome lists are empty.
        """
        logger.info(
            "analyze_outcomes: artifact_type=%s, n_a=%d, n_b=%d",
            artifact_type,
            len(outcomes_a),
            len(outcomes_b),
        )
        if len(outcomes_a) == 0 or len(outcomes_b) == 0:
            raise ValueError("Outcome lists must not be empty.")

        # Validate outcomes are numeric (not strings)
        for i, v in enumerate(outcomes_a):
            if isinstance(v, str):
                raise TypeError(f"outcomes_a[{i}] is a string ('{v}'). Outcomes must be numeric.")
        for i, v in enumerate(outcomes_b):
            if isinstance(v, str):
                raise TypeError(f"outcomes_b[{i}] is a string ('{v}'). Outcomes must be numeric.")

        # Drop outcomes that carry no value, and COUNT them.
        #
        # READINESS-6, 2026-09-10. This was `isinstance(v, float) and
        # np.isnan(v)`, which catches a Python float NaN and nothing else.
        # `None` is not a float and a numpy scalar is not a Python float, so
        # both survived the filter, became NaN inside `np.asarray(..., float)`,
        # and carried NaN through the mean, the disparity and the test. The
        # verdict below was `p_value < self.alpha`, and `nan < 0.05` is False.
        #
        # Measured that day on a maximal callback disparity, group A called back
        # 40 of 40 and group B 0 of 40:
        #     all measured        disparity=1.0 p=1.86e-23 significant=True  n=40
        #     one None in B       disparity=nan p=nan      significant=False n=40
        #     one np.float32 nan  disparity=nan p=nan      significant=False n=40
        # One unusable row turned a total hiring disparity into "not
        # significant", and the sample size still claimed all 40. This is a
        # correspondence study, the resume-audit method, so that verdict is the
        # whole output.
        #
        # `four_fifths_rule` in this same file already answers None for a ratio
        # it could not compute and says why; this path was missed.
        n_before_a, n_before_b = len(outcomes_a), len(outcomes_b)
        outcomes_a = [v for v in outcomes_a if _is_measured(v)]
        outcomes_b = [v for v in outcomes_b if _is_measured(v)]
        n_unmeasurable = (n_before_a - len(outcomes_a)) + (n_before_b - len(outcomes_b))
        if n_unmeasurable:
            warnings.warn(
                f"analyze_outcomes: {n_unmeasurable} outcome(s) carried no value "
                f"(None, NaN or non-numeric) and were EXCLUDED. The comparison rests "
                f"on the {len(outcomes_a)} and {len(outcomes_b)} that remain, and "
                f"sample_size reports those, not the {n_before_a} and {n_before_b} "
                f"supplied.",
                UserWarning,
                stacklevel=2,
            )
        if len(outcomes_a) == 0 or len(outcomes_b) == 0:
            raise ValueError("Outcome lists are empty after filtering NaN values.")

        MIN_RECOMMENDED_SAMPLES = 30
        if len(outcomes_a) < MIN_RECOMMENDED_SAMPLES or len(outcomes_b) < MIN_RECOMMENDED_SAMPLES:
            # `warnings` is imported at module scope. A local `import warnings`
            # here made the name local to this whole function, so every earlier
            # use of it in the same body raised UnboundLocalError.
            logger.warning(
                "Sample size (%d, %d) below recommended minimum of %d",
                len(outcomes_a),
                len(outcomes_b),
                MIN_RECOMMENDED_SAMPLES,
            )
            warnings.warn(
                f"Sample size ({len(outcomes_a)}, {len(outcomes_b)}) below recommended minimum "
                f"of {MIN_RECOMMENDED_SAMPLES}. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        arr_a = np.asarray(outcomes_a, dtype=float)
        arr_b = np.asarray(outcomes_b, dtype=float)

        # Determine if outcomes are categorical (binary) or numeric.
        unique_values = np.unique(np.concatenate([arr_a, arr_b]))
        is_binary = (
            np.array_equal(unique_values, np.array([0.0, 1.0]))
            or np.array_equal(unique_values, np.array([0.0]))
            or np.array_equal(unique_values, np.array([1.0]))
        )

        if is_binary:
            rate_a = float(np.mean(arr_a))
            rate_b = float(np.mean(arr_b))
            disparity = rate_a - rate_b

            # Chi-square test on 2x2 contingency table.
            table = np.array(
                [
                    [np.sum(arr_a == 1), np.sum(arr_a == 0)],
                    [np.sum(arr_b == 1), np.sum(arr_b == 0)],
                ]
            )

            # Use Fisher's exact test for small samples.
            total = table.sum()
            if total < 40 or np.any(table < 5):
                _, p_value = stats.fisher_exact(table)
            else:
                chi2, p_value, _, _ = stats.chi2_contingency(table, correction=True)
        else:
            rate_a = float(np.mean(arr_a))
            rate_b = float(np.mean(arr_b))
            disparity = rate_a - rate_b

            # Mann-Whitney U test for numeric outcomes.
            if np.array_equal(arr_a, arr_b):
                p_value = 1.0
            else:
                # The shared helper: a fully tied pair of different lengths is the
                # exact p of 1.0, where scipy 1.18 answers nan. Any other missing
                # p stays nan, which the three-state block below reports as
                # could-not-check with its own warning.
                measured = _mannwhitney_two_sided_p(arr_a, arr_b)
                p_value = float("nan") if measured is None else measured

        sample_size = min(len(arr_a), len(arr_b))

        # G12, 2026-09-30. A COMPARISON THAT COULD NOT HAVE FIRED REPORTED
        # is_significant=False, which on this field means "tested, and the
        # groups were treated alike". Both tests used here are EXACT and
        # condition on the arm sizes, so their p has a combinatorial FLOOR: the
        # most extreme outcome, complete separation, has two-sided p
        # 2 / C(n_a + n_b, n_a). Measured on this method before the change:
        #   analyze_outcomes([1.0], [0.0])
        #     -> outcome_a 1.0, outcome_b 0.0, disparity_metric 1.0 (the largest
        #        disparity the scale has), p_value 1.0, is_significant False
        # 1.0 is the floor at one observation per arm, so no data whatsoever
        # could have produced a finding, and the only disclosure was a
        # "below recommended minimum of 30" warning about reliability, which is
        # a different statement from "this test could not fire".
        # Three sibling modules already carry this exact vocabulary; the shared
        # helper is reused rather than a second floor invented.
        min_attainable_p: Optional[float] = None
        try:
            min_attainable_p = min(1.0, 2.0 / float(math.comb(len(arr_a) + len(arr_b), len(arr_a))))
        except (ValueError, OverflowError, ZeroDivisionError):
            # A combinatorial count too large to read as a float means the floor
            # is far below any alpha; leaving it None would report a
            # could-not-check for a design with abundant power, so say so.
            logger.debug(
                "analyze_outcomes: C(%d, %d) is not readable as a float; the exact "
                "floor is far below alpha at this size",
                len(arr_a) + len(arr_b),
                len(arr_a),
                exc_info=True,
            )
            min_attainable_p = 0.0
        detectable, detectability_note = detectability(
            min_attainable_p, n_family=1, alpha=self.alpha
        )
        if detectable is not True:
            warnings.warn(
                f"analyze_outcomes: with {len(arr_a)} and {len(arr_b)} measured "
                f"outcome(s) the smallest p-value this exact test can produce is "
                f"{min_attainable_p}, so a 'not significant' reading is an absence of "
                f"statistical power rather than evidence that the groups were treated "
                f"alike. {detectability_note} Read result.detectable beside "
                f"is_significant.",
                UserWarning,
                stacklevel=2,
            )

        # Bootstrap confidence interval on disparity.
        ci = self._bootstrap_ci(
            arr_a,
            arr_b,
            stat_fn=lambda a, b: float(np.mean(a) - np.mean(b)),
        )

        # A test that produced no p-value did not find "no significance": it
        # found nothing. Three states, never two. READINESS-6.
        p_float = float(p_value)
        significant: Optional[bool] = bool(p_float < self.alpha) if np.isfinite(p_float) else None
        if significant is None:
            warnings.warn(
                "analyze_outcomes: the statistical test produced no p-value, so "
                "is_significant is None (could not check), NOT False. Nothing here "
                "says the groups were treated alike.",
                UserWarning,
                stacklevel=2,
            )

        logger.info(
            "analyze_outcomes complete: disparity=%s, p_value=%s, significant=%s",
            disparity,
            p_float,
            significant,
        )
        return CorrespondenceResult(
            test_id=str(uuid.uuid4()),
            artifact_type=artifact_type,
            demographic_a="group_a",
            demographic_b="group_b",
            outcome_a=rate_a,
            outcome_b=rate_b,
            disparity_metric=disparity,
            sample_size=sample_size,
            is_significant=significant,
            p_value=p_float,
            n_unmeasurable=n_unmeasurable,
            confidence_interval=ci,
            min_attainable_p=min_attainable_p,
            detectable=detectable,
            detectability_note=detectability_note,
            metadata=RunMetadata(
                parameters={
                    "alpha": self.alpha,
                    "artifact_type": artifact_type,
                    "sample_size_a": len(outcomes_a),
                    "sample_size_b": len(outcomes_b),
                },
            ),
        )

    def _bootstrap_ci(self, values_a, values_b, stat_fn, n_bootstrap=1000, confidence=0.95):
        """Bootstrap CI for any statistic comparing two groups."""
        rng = np.random.default_rng(self.random_seed)
        stats_list = []
        for _ in range(n_bootstrap):
            boot_a = rng.choice(values_a, size=len(values_a), replace=True)
            boot_b = rng.choice(values_b, size=len(values_b), replace=True)
            stats_list.append(stat_fn(boot_a, boot_b))
        alpha = 1 - confidence
        return (
            float(np.percentile(stats_list, 100 * alpha / 2)),
            float(np.percentile(stats_list, 100 * (1 - alpha / 2))),
        )

    def four_fifths_rule(self, rate_a: float, rate_b: float) -> dict:
        """Apply the four-fifths (80%) rule for disparate impact.

        The EEOC four-fifths rule states that a selection rate for any
        group that is less than 80% of the rate of the highest group
        constitutes evidence of adverse impact.

        The verdict has three states, never two: True (adverse impact),
        False (no adverse impact) and None (could not check). None is
        reported whenever no impact ratio could be formed, and the ratio
        is nan in exactly those cases.

        Args:
            rate_a: Selection/positive outcome rate for group A.
            rate_b: Selection/positive outcome rate for group B.

        Returns:
            Dictionary with:
                - ``ratio``: Disparate impact ratio (lower / higher), or nan
                  when no ratio could be formed.
                - ``adverse_impact``: True if ratio < 0.8, False if not, and
                  None when the ratio was not measured.
                - ``favored_group``: ``"group_a"`` or ``"group_b"`` for the
                  higher rate, ``"neither"`` when the rates are equal, and
                  ``"undetermined"`` when a rate is not a finite number.
        """
        if not math.isfinite(rate_a) or not math.isfinite(rate_b):
            # This is the EEOC four-fifths verdict, so a non-finite rate must
            # not reach the comparison: `nan < 0.8` is False, and False here
            # is the finding "no adverse impact" from a rate nobody measured.
            warnings.warn(
                f"CorrespondenceTester.four_fifths_rule: selection rates "
                f"({rate_a!r}, {rate_b!r}) are not both finite, so the impact "
                f"ratio was not measured. Reporting ratio=nan and "
                f"adverse_impact=None (could not check), not False, which is "
                f"the EEOC finding of no adverse impact.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "ratio": float("nan"),
                "adverse_impact": None,
                "favored_group": "undetermined",
            }

        if rate_a > rate_b:
            favored = "group_a"
        elif rate_b > rate_a:
            favored = "group_b"
        else:
            favored = "neither"

        higher = max(rate_a, rate_b)
        lower = min(rate_a, rate_b)

        if higher <= 0.0:
            # Nobody was selected in either group, so the ratio is 0/0. An
            # earlier pass called ratio 1.0 with favored_group "neither"
            # defensible because neither group was favoured. Only the second
            # half of that holds: the rates ARE equal, so "neither" is a
            # measurement and is kept. The ratio and the verdict are not: 1.0
            # with adverse_impact False is byte for byte what MEASURED parity
            # (0.5 vs 0.5) returns, so an audit in which nobody at all was
            # selected was indistinguishable from one that found equal
            # treatment. Measured 2026-09-10.
            warnings.warn(
                f"CorrespondenceTester.four_fifths_rule: neither group has a "
                f"positive selection rate ({rate_a!r}, {rate_b!r}), so the "
                f"impact ratio is undefined (its denominator is 0). Reporting "
                f"ratio=nan and adverse_impact=None (could not check), not "
                f"ratio=1.0 with adverse_impact=False, which is what MEASURED "
                f"parity returns.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "ratio": float("nan"),
                "adverse_impact": None,
                "favored_group": favored,
            }

        ratio = lower / higher

        return {
            "ratio": float(ratio),
            "adverse_impact": bool(ratio < 0.8),
            "favored_group": favored,
        }
