"""
Action and delegation bias measurement for AI agents.

Measures whether AI agents take different actions or delegate to
different downstream services depending on the demographic group.
For example, an agent might approve requests from one demographic
while routing another to manual review.
"""

import logging
import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from vfairness.evaluation.vfairness_metrics._statistics import (
    _mannwhitney_two_sided_p,
    detectability,
    min_attainable_p_fisher,
)
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)

#: Smallest EXPECTED cell count at which the asymptotic chi-square is trusted
#: (Cochran's rule).
#:
#: WHY IT IS WRITTEN TWICE. ``multi_agent.delegation.MIN_EXPECTED_CELL`` holds
#: the same number for the same reason, and one copy being raised while the
#: other is forgotten is the failure mode ``vfairness._not_assessed`` documents
#: at length. Importing it would make ``vfairness.agents`` pull in
#: ``vfairness.multi_agent`` for a single float, so the two copies are instead
#: pinned equal by ``tests/test_bgl3_agents_multi_2.py``, which goes red the day
#: they diverge.
MIN_EXPECTED_CELL = 5.0


@dataclass
class ActionBiasResult(SerializableMixin):
    """Result of an action bias analysis.

    Attributes:
        action_type: Type of action analyzed.
        outcome_a: Mean outcome for demographic group A.
        outcome_b: Mean outcome for demographic group B.
        disparity: Absolute difference in outcomes.
        effect_size: Cohen's d effect size.
        is_significant: Whether disparity is statistically significant, or
            None when no significance test could be run (three states, never
            two: significant / not significant / could not check).
        p_value: P-value from the Mann-Whitney U test, or nan when the test
            did not run. This is the channel the significance verdict is
            derived from; without it ``is_significant`` was the only word a
            caller had on a test that never happened. It defaults to nan
            because "not measured" is the only safe thing an unset p can mean.
    """

    action_type: str
    outcome_a: float
    outcome_b: float
    disparity: float
    effect_size: float
    is_significant: Optional[bool]
    p_value: float = float("nan")
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    metadata: RunMetadata = field(default_factory=RunMetadata)


class ActionBiasAnalyzer:
    """Analyzer for action and delegation bias in AI agents.

    Compares the actions taken and delegation patterns between
    demographic groups to identify differential treatment.

    Args:
        alpha: Significance level for hypothesis tests.

    Example:
        >>> analyzer = ActionBiasAnalyzer(alpha=0.05)
        >>> actions_a = [
        ...     {"action": "approve", "score": 0.9},
        ...     {"action": "approve", "score": 0.85},
        ... ]
        >>> actions_b = [
        ...     {"action": "review", "score": 0.6},
        ...     {"action": "approve", "score": 0.7},
        ... ]
        >>> result = analyzer.analyze_outcomes(actions_a, actions_b, "score")

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: action_bias_analyzer. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, alpha: float = 0.05, random_seed: int = 42) -> None:
        self.alpha = alpha
        self.random_seed = random_seed
        logger.info("ActionBiasAnalyzer initialized: alpha=%.3f", alpha)

    def analyze_outcomes(
        self,
        actions_a: list[dict],
        actions_b: list[dict],
        outcome_field: str,
    ) -> ActionBiasResult:
        """Analyze outcome disparities between demographic groups.

        Extracts the specified outcome field from action records and
        tests for statistically significant differences.

        Args:
            actions_a: Action records for demographic group A.
            actions_b: Action records for demographic group B.
            outcome_field: Key in action dicts containing the outcome value.

        Returns:
            ActionBiasResult with disparity and significance metrics. When no
            significance test could be run, ``p_value`` is nan and
            ``is_significant`` is None (could not check), never False.

        Raises:
            ValueError: If action lists are empty or outcome_field is missing.
        """
        logger.info(
            "analyze_outcomes: outcome_field=%s, n_a=%d, n_b=%d",
            outcome_field,
            len(actions_a),
            len(actions_b),
        )
        if len(actions_a) == 0 or len(actions_b) == 0:
            raise ValueError("Action lists must not be empty.")

        MIN_RECOMMENDED_SAMPLES = 30
        if len(actions_a) < MIN_RECOMMENDED_SAMPLES or len(actions_b) < MIN_RECOMMENDED_SAMPLES:
            # No local `import warnings` here: it made `warnings` a LOCAL name
            # for the whole method, so any later warnings.warn raised
            # UnboundLocalError whenever both groups held 30 or more records.
            logger.warning(
                "Sample size (%d, %d) below recommended minimum of %d",
                len(actions_a),
                len(actions_b),
                MIN_RECOMMENDED_SAMPLES,
            )
            warnings.warn(
                f"Sample size ({len(actions_a)}, {len(actions_b)}) below recommended minimum "
                f"of {MIN_RECOMMENDED_SAMPLES}. Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Raise the documented ValueError (not a bare KeyError) when the
        # outcome field is absent from any record.
        for records, label in ((actions_a, "actions_a"), (actions_b, "actions_b")):
            for record in records:
                if outcome_field not in record:
                    raise ValueError(
                        f"outcome_field '{outcome_field}' missing from a record in {label}."
                    )

        values_a = np.array([a[outcome_field] for a in actions_a], dtype=float)
        values_b = np.array([a[outcome_field] for a in actions_b], dtype=float)

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        disparity = abs(mean_a - mean_b)

        # Cohen's d effect size.
        effect_size = self._cohens_d(values_a, values_b)

        # Statistical test: Mann-Whitney U for robustness.
        if len(values_a) < 2 or len(values_b) < 2:
            # p = 1.0 is the MOST non-significant value there is, so a test that
            # never ran read as a test that found nothing.
            warnings.warn(
                f"ActionBiasAnalyzer: the Mann-Whitney test needs at least 2 values per "
                f"group and got {len(values_a)} vs {len(values_b)}, so significance was "
                f"NOT tested. Reporting p_value=nan and is_significant=None "
                f"(could not check), not 1.0 and False.",
                UserWarning,
                stacklevel=2,
            )
            p_value = float("nan")
        elif np.array_equal(values_a, values_b):
            # Genuinely identical samples: a real p of 1.0, not a stand-in.
            p_value = 1.0
        else:
            # The shared helper, not a bare mannwhitneyu: on scipy 1.18 a fully
            # tied pair of DIFFERENT lengths (array_equal is False) came back
            # p=nan and read as could-not-check with no warning; its exact p is
            # 1.0. Any other nan p is still could-not-check, and now says so.
            measured = _mannwhitney_two_sided_p(values_a, values_b)
            if measured is None:
                warnings.warn(
                    f"ActionBiasAnalyzer: the Mann-Whitney test on {outcome_field!r} "
                    f"returned no p-value (a non-finite outcome, or a degenerate input "
                    f"scipy answers with nan), so significance was NOT tested. "
                    f"Reporting p_value=nan and is_significant=None (could not check), "
                    f"not a 'not significant' reading.",
                    UserWarning,
                    stacklevel=2,
                )
                p_value = float("nan")
            else:
                p_value = measured

        # Determine action type from the most common action.
        action_counts = Counter(a.get("action", "unknown") for a in actions_a + actions_b)
        action_type = action_counts.most_common(1)[0][0]

        # Bootstrap CI on effect size.
        ci = self._bootstrap_ci_effect_size(values_a, values_b)

        # `nan < alpha` is False, so a test that explicitly REFUSED to run
        # reported is_significant=False, which on this field means "tested, and
        # the disparity is not significant". The p-value it refused on was not
        # carried on the result at all, so is_significant was the only
        # significance channel a caller had and it said the wrong thing.
        # Measured 2026-09-10 on a 1-vs-1 comparison.
        is_significant: Optional[bool]
        if math.isnan(p_value):
            is_significant = None
        else:
            is_significant = bool(p_value < self.alpha)

        return ActionBiasResult(
            action_type=action_type,
            outcome_a=mean_a,
            outcome_b=mean_b,
            disparity=disparity,
            effect_size=float(effect_size),
            is_significant=is_significant,
            p_value=float(p_value),
            confidence_interval=ci,
            metadata=RunMetadata(
                parameters={
                    "alpha": self.alpha,
                    "outcome_field": outcome_field,
                    "sample_size_a": len(actions_a),
                    "sample_size_b": len(actions_b),
                },
            ),
        )

    def analyze_delegation(
        self,
        routing_a: list[str],
        routing_b: list[str],
    ) -> dict:
        """Compare delegation/routing patterns between demographic groups.

        Tests whether agents route requests from different demographics
        to different downstream handlers at different rates.

        Args:
            routing_a: Delegation targets for demographic group A.
            routing_b: Delegation targets for demographic group B.

        Returns:
            Dictionary with:
                - ``per_target``: Per-target rate comparison. Each entry carries
                  ``rate_a``, ``rate_b``, ``p_value``, ``is_significant``
                  (None when the design could not have reached ``alpha`` for ANY
                  routing), ``min_attainable_p`` and ``detectable``.
                - ``overall_chi2``: Overall chi-square statistic, or None when
                  no overall test was possible.
                - ``overall_p_value``: P-value for the overall distribution
                  test, or None when no valid test could be run.
                - ``is_significant``: Whether the overall difference is
                  significant. None is could-not-check, never "no difference".
                - ``test_used``: ``"chi2"``, ``"fisher"`` or ``None``.
                - ``min_attainable_p`` / ``detectable`` /
                  ``detectability_note``: the design's own power for the overall
                  test, and what a "not significant" reading does NOT mean when
                  it has none.

        Raises:
            ValueError: If routing lists are empty.
        """
        if len(routing_a) == 0 or len(routing_b) == 0:
            raise ValueError("Routing lists must not be empty.")

        count_a = Counter(routing_a)
        count_b = Counter(routing_b)
        all_targets = sorted(set(count_a.keys()) | set(count_b.keys()))

        total_a = len(routing_a)
        total_b = len(routing_b)

        # DISCRETE FLOOR (2026-09-27). Fisher's exact conditions on both
        # margins, so the smallest p a per-target test can return is fixed by
        # the two group sizes before any routing is looked at. Measured
        # 2026-09-27 on 3 group-A tasks all routed to "junior" against 3 group-B
        # tasks all routed to "senior", which is total segregation and the
        # strongest routing bias this method can express:
        #
        #   per_target junior  p = 0.1, is_significant = False
        #   per_target senior  p = 0.1, is_significant = False, no warning
        #
        # 0.1 is the SMALLEST p 3 against 3 can produce, so no arrangement of
        # that data could have said True, and False there is an absence of
        # power reported as a measured absence of bias. At 1 decision per group
        # the floor is 1.0. This is the defect DelegationRoutingAuditor
        # (multi_agent/delegation.py) was fixed for on 2026-09-10, on the same
        # numbers; the same two shared helpers answer it here.
        target_floor = min_attainable_p_fisher(total_a, total_b)
        target_detectable, target_note = detectability(target_floor, n_family=1, alpha=self.alpha)

        # Per-target analysis.
        per_target = {}
        for target in all_targets:
            n_a = count_a.get(target, 0)
            n_b = count_b.get(target, 0)

            rate_a = n_a / total_a
            rate_b = n_b / total_b

            table = np.array(
                [
                    [n_a, total_a - n_a],
                    [n_b, total_b - n_b],
                ]
            )

            if table.sum() < 40 or np.any(table < 5):
                _, p_val = stats.fisher_exact(table)
            else:
                _, p_val, _, _ = stats.chi2_contingency(table, correction=True)

            per_target[target] = {
                "rate_a": float(rate_a),
                "rate_b": float(rate_b),
                "p_value": float(p_val),
                # None, never False: at these group sizes the verdict may be one
                # the test could not have reached whatever the routing was.
                "is_significant": (bool(p_val < self.alpha) if target_detectable is True else None),
                "min_attainable_p": target_floor,
                "detectable": target_detectable,
            }

        if target_detectable is not True:
            warnings.warn(
                f"ActionBiasAnalyzer.analyze_delegation: per-target routing significance "
                f"was NOT ASSESSED for {total_a} vs {total_b} decisions. {target_note} "
                f"Reporting is_significant=None for all {len(all_targets)} target(s) "
                f"(could not check), NOT False. The measured rates still stand.",
                UserWarning,
                stacklevel=2,
            )

        # Overall distribution test on the full (2 x targets) table.
        observed_a = np.array([count_a.get(t, 0) for t in all_targets])
        observed_b = np.array([count_b.get(t, 0) for t in all_targets])
        contingency = np.array([observed_a, observed_b])

        overall_chi2: Optional[float]
        overall_p: Optional[float]
        overall_floor: Optional[float]
        test_used: Optional[str]
        overall_reason = ""
        if contingency.shape[1] < 2:
            # ONE observed target. A distribution over a single category cannot
            # differ between groups, so there is no test here at all: the table
            # has 0 degrees of freedom and scipy will not return a p for it. The
            # ``overall_chi2 = 0.0, overall_p = 1.0`` this replaces was the p of
            # a test that never ran, and on that field 1.0 is the strongest "no
            # difference" the scale has. Same ruling as
            # operations/pulse/agent_probe._cramers_v, which refuses a
            # single-level table rather than answering 0.0 (2026-09-17).
            #
            # None and NOT nan, for a reason measured 2026-09-27. This dict is
            # published verbatim as `actionDistribution` by
            # operations/pulse/agent_probe, and the agent lane's `run_pulse`
            # return is NOT passed through `orchestrator._jsonify`, which is what
            # coerces non-finite floats to null elsewhere. With nan here,
            # `run_pulse(30 A-episodes + 30 B-episodes that only ever call one
            # tool, source_kind="agent")` produced a payload on which
            # `json.dumps(out, allow_nan=False)` raised ValueError("Out of range
            # float values are not JSON compliant: nan"), and that nan was the
            # ONLY non-finite float in the whole result. Exactly the BGL-S2c
            # incident recorded in orchestrator.py. null is the value that
            # library's own rule prescribes for an unmeasured quantity crossing
            # a JSON boundary (see agent_probe._finite_or_none).
            overall_chi2 = None
            overall_p = None
            overall_floor = None
            test_used = None
            overall_reason = (
                f"COULD NOT CHECK: only one routing target ({all_targets[0]!r}) was "
                f"observed across both groups, so there is no distribution over "
                f"targets for the groups to differ on. The per-target comparison "
                f"above is the measured result."
            )
        else:
            # COCHRAN'S RULE, on EXPECTED counts. The asymptotic chi-square was
            # run here unconditionally, and on a sparse table it does not merely
            # lose power, it reports the WRONG p. Measured 2026-09-27 on 12
            # group-A tasks split over two routes against 1 group-B task on a
            # third:
            #
            #   overall_chi2 13.0, overall_p_value 0.0015, is_significant True
            #
            # on a table whose smallest expected count is 0.077, while the exact
            # conditional p for those margins is 0.0769, which is also the
            # smallest p they can produce at all. A fabricated FINDING: the same
            # defect running backwards. delegation.py measured the identical
            # numbers on the identical shape on 2026-09-10.
            expected = np.asarray(stats.contingency.expected_freq(contingency), dtype=float)
            min_expected = float(expected.min())
            if min_expected >= MIN_EXPECTED_CELL:
                chi2_stat, overall_p, _, _ = stats.chi2_contingency(contingency)
                overall_chi2 = float(chi2_stat)
                # The asymptotic p is continuous, so this design has no discrete
                # floor and can reach any alpha in principle. 0.0 says that to
                # `detectability` explicitly rather than leaving it unasked.
                overall_floor = 0.0
                test_used = "chi2"
            elif contingency.shape[1] == 2:
                # Two targets: Fisher's exact is valid exactly where the
                # asymptotic test is not, and on a 2 x 2 it tests the same
                # hypothesis the per-target test above does. Uncorrected
                # chi-square is kept as the descriptive statistic, as in
                # DelegationRoutingAuditor.
                chi2_stat, _, _, _ = stats.chi2_contingency(contingency, correction=False)
                overall_chi2 = float(chi2_stat)
                _, overall_p = stats.fisher_exact(contingency)
                overall_p = float(overall_p)
                overall_floor = target_floor
                test_used = "fisher"
            else:
                # More than two targets AND cells too thin for the asymptotic
                # test. scipy's Fisher is 2 x 2 only, so there is no valid test
                # to run here; the exact-by-permutation chi-square that belongs
                # in this branch already exists as
                # DelegationRoutingAuditor.analyze, which is the one to use for
                # a categorical routing audit on sparse data.
                chi2_stat, _, _, _ = stats.chi2_contingency(contingency, correction=False)
                overall_chi2 = float(chi2_stat)
                overall_p = None
                overall_floor = None
                test_used = None
                overall_reason = (
                    f"COULD NOT CHECK: the smallest expected cell count is "
                    f"{min_expected:.3g}, below the {MIN_EXPECTED_CELL:g} the asymptotic "
                    f"chi-square needs, and Fisher's exact does not extend past a 2 x 2, "
                    f"so no valid overall test exists for this "
                    f"{contingency.shape[0]} x {contingency.shape[1]} table. Use "
                    f"DelegationRoutingAuditor.analyze, whose permutation chi-square "
                    f"holds the margins fixed and needs no asymptotic assumption."
                )

        overall_detectable, overall_note = detectability(
            overall_floor, n_family=1, alpha=self.alpha
        )
        is_significant: Optional[bool]
        if overall_p is not None and overall_detectable is True:
            is_significant = bool(overall_p < self.alpha)
        else:
            is_significant = None
            warnings.warn(
                f"ActionBiasAnalyzer.analyze_delegation: the overall routing-distribution "
                f"test was NOT ASSESSED over {len(all_targets)} target(s) and "
                f"{total_a} + {total_b} decisions. {overall_reason or overall_note} "
                f"Reporting overall_p_value={overall_p} and is_significant=None (could "
                f"not check), not a significance verdict either way.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "per_target": per_target,
            "overall_chi2": overall_chi2,
            "overall_p_value": None if overall_p is None else float(overall_p),
            "is_significant": is_significant,
            "test_used": test_used,
            "min_attainable_p": overall_floor,
            "detectable": overall_detectable,
            "detectability_note": overall_reason or overall_note,
        }

    def _bootstrap_ci_effect_size(self, values_a, values_b, n_bootstrap=1000, confidence=0.95):
        """Bootstrap CI for Cohen's d effect size."""
        rng = np.random.default_rng(self.random_seed)
        effects = []
        for _ in range(n_bootstrap):
            boot_a = rng.choice(values_a, size=len(values_a), replace=True)
            boot_b = rng.choice(values_b, size=len(values_b), replace=True)
            effects.append(self._cohens_d(boot_a, boot_b))
        alpha = 1 - confidence
        return (
            float(np.percentile(effects, 100 * alpha / 2)),
            float(np.percentile(effects, 100 * (1 - alpha / 2))),
        )

    @staticmethod
    def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n_a = len(group_a)
        n_b = len(group_b)

        if n_a < 2 or n_b < 2:
            # 0.0 is "no effect", a measurement. Cohen's d needs at least two
            # observations per group to have a variance at all.
            warnings.warn(
                f"ActionBiasAnalyzer: Cohen's d needs at least 2 observations per group "
                f"and got {n_a} vs {n_b}, so no effect size was computed. Returning nan, "
                f"not 0.0.",
                UserWarning,
                stacklevel=3,
            )
            return float("nan")

        var_a = np.var(group_a, ddof=1)
        var_b = np.var(group_b, ddof=1)

        # Pooled standard deviation.
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = np.sqrt(pooled_var)

        mean_gap = float(np.mean(group_a) - np.mean(group_b))
        # READINESS-6, 2026-09-10. The guard below WAS `pooled_std == 0.0`, and
        # that exact float test only fires for values that are exactly
        # representable in binary, at some lengths. This fix was made on
        # 2026-09-08 and pinned with group A at 10.0 and group B at 100.0, n=10:
        # both powers-of-two-friendly, so the variance is exactly 0.0 and the
        # guard fires. Measured 2026-09-10 with ordinary values, the fix simply
        # did not apply:
        #
        #   constant arms, gap 0.7, n=5   -> nan   (guard fires)
        #   constant arms, gap 0.7, n=20  -> -4.3e15 (guard misses)
        #   0.9 vs 1.6,          n=20     -> -3.1e15 (guard misses)
        #
        # A constant array of a value that is not exactly representable
        # accumulates a tiny non-zero variance, and it does so at SOME lengths
        # and not others: np.var of twenty 0.9s is 4.93e-32, of twenty-five,
        # exactly 0.0. So neither a different value nor a larger n reliably
        # exposes it, and the fixture that was chosen exposed neither.
        #
        # np.ptp asks the question on the RAW data, where it is exact.
        # THE DUPLICATE, CORRECTED 2026-09-29. `np.ptp` is exact, and the EQUALITY
        # WITH ZERO this used to be was not: it asks whether the array is constant in
        # the LAST BIT, which only a caller-supplied constant array is. An array the
        # caller COMPUTED, a residual or a difference, is constant to a few ulps and
        # walks straight past it. Measured here on `(pred + 50.0) - pred` over 40 rows,
        # peak-to-peak 2.84e-14: this returned **-6624998218327707.0 with zero
        # warnings** for two groups whose real gap is 50. The canonical copy in
        # evaluation/vfairness_metrics/_statistics.py was fixed the same day and this
        # is the same test, data-scaled so it carries no units.
        _sp1, _sp2 = float(np.ptp(group_a)), float(np.ptp(group_b))
        _mag = max(abs(float(np.mean(group_a))), abs(float(np.mean(group_b))), _sp1, _sp2)
        _atol = 1e-12 * _mag
        if _sp1 <= _atol and _sp2 <= _atol:
            if mean_gap == 0.0:
                # Both groups constant AND equal: a genuine zero effect.
                return 0.0
            # Both groups constant but SEPARATED. Measured 2026-09-08 with group
            # A always 10.0 and group B always 100.0, perfect separation and the
            # strongest bias signal there is: this returned 0.0, "no effect",
            # because the standardising denominator is zero. The standardised
            # effect is undefined here, not absent, and the raw gap is stated so
            # the caller is not left with nothing.
            warnings.warn(
                f"ActionBiasAnalyzer: both groups have zero variance but differ in mean "
                f"by {mean_gap:.4f}, so Cohen's d is undefined (its denominator is 0). "
                f"Returning nan rather than 0.0, which would read as 'no effect' for "
                f"what is in fact perfect separation.",
                UserWarning,
                stacklevel=3,
            )
            return float("nan")

        return mean_gap / pooled_std
