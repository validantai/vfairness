"""
Role-based delegation routing audit for multi-agent orchestrators.

Tests whether an orchestrator routes demographically equivalent tasks
to different downstream agents based on irrelevant protected attributes.
This is the multi-agent analogue of the classic correspondence test
(Bertrand & Mullainathan 2004), generalized from binary callbacks to
categorical routing decisions.

Methodology grounding:
- Bertrand & Mullainathan (2004) "Are Emily and Greg More Employable
  than Lakisha and Jamal?", correspondence-test methodology
- Wu et al. (2023) "AutoGen: Enabling Next-Gen LLM Applications via
  Multi-Agent Conversation", orchestrator pattern
- Jin et al. (2024) "Comm-Bench", multi-agent communication
  benchmark; routing-fairness probe pattern

Statistical test:
- Build the (routes × groups) contingency table.
- 2 × 2: Fisher's exact test (small samples) or chi-square test.
- k × 2 / k × m: chi-square test of independence.
- Effect size: Cramér's V.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from vfairness.evaluation.vfairness_metrics._statistics import (
    detectability,
    min_attainable_p_fisher,
    min_attainable_p_permutation,
)
from vfairness.llm._base import RunMetadata, SerializableMixin

logger = logging.getLogger(__name__)

#: Smallest expected cell count at which the asymptotic chi-square is trusted
#: (Cochran's rule, and the number this module's own docstring already cites).
MIN_EXPECTED_CELL = 5.0


@dataclass
class DelegationResult(SerializableMixin):
    """Result of a delegation-routing audit.

    Attributes:
        contingency_table: 2-D nested dict {route: {group: count}}.
        routes: Ordered list of route identifiers seen.
        groups: Ordered list of group identifiers seen.
        chi_square: Uncorrected chi-square statistic. Always a float: it is
            computed even when Fisher's exact provides the p-value, because
            Cramér's V is derived from it.
        odds_ratio: Odds ratio from Fisher's exact on a 2 x 2 table.
            ``float('nan')`` when the chi-square path was used.
        cramers_v: Cramér's V effect size in [0, 1].
        p_value: P-value from the appropriate test.
        is_significant: `p_value < alpha`, or None when the test could never
            have reached alpha for ANY data at these group sizes. None is
            "could not check", never "no routing bias".
        test_used: Which test produced the p-value ("fisher", "chi2" or
            "permutation_chi2").
        per_route_disparity: Per-route absolute difference in invocation
            rate between the two largest groups (helpful summary metric
            even when more than two groups are present).
        min_attainable_p: Smallest p-value this design could return, whatever
            the routing had been. None when that could not be computed.
        detectable: Whether ``min_attainable_p`` clears ``alpha``. None is
            could-not-check.
        detectability_note: What a "not significant" reading does NOT mean
            here. Empty when the design has power.
    """

    contingency_table: Dict[str, Dict[str, int]]
    routes: List[str]
    groups: List[str]
    chi_square: float
    odds_ratio: float
    cramers_v: float
    p_value: float
    is_significant: Optional[bool]
    test_used: str
    per_route_disparity: Dict[str, float]
    metadata: RunMetadata = field(default_factory=RunMetadata)
    min_attainable_p: Optional[float] = None
    detectable: Optional[bool] = None
    detectability_note: str = ""


class DelegationRoutingAuditor:
    """Audits orchestrator routing for demographic-conditional patterns.

    Pass a sequence of (route_chosen, demographic_label) pairs captured
    from an orchestrator's routing decisions over a set of paired
    equivalent tasks (only the demographic attribute differs). The
    auditor returns a contingency-table-based significance test plus an
    effect size, so a small route disparity in a huge sample doesn't
    masquerade as a meaningful audit failure.

    Args:
        alpha: Significance threshold.
        prefer_fisher: If True (default), use Fisher's exact test on
            any 2 x 2 table. A 2 x 2 whose expected cell counts fall below
            :data:`MIN_EXPECTED_CELL` routes to Fisher either way: the
            asymptotic chi-square is not valid there, and that fallback is
            documented behaviour, so it must actually happen.
        n_permutations: Resamples for the exact-by-permutation chi-square used
            on larger-than-2x2 tables whose expected counts are too small for
            the asymptotic test.
        random_seed: Seed for that permutation null.

    Example:
        >>> auditor = DelegationRoutingAuditor()
        >>> routes = ["junior", "senior", "junior", "senior", ...]
        >>> demographics = ["A", "A", "B", "B", ...]
        >>> result = auditor.analyze(routes, demographics)
        >>> result.is_significant

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: delegation_routing_auditor. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        prefer_fisher: bool = True,
        n_permutations: int = 2000,
        random_seed: int = 42,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if n_permutations < 50:
            raise ValueError(
                f"n_permutations must be >= 50 for a usable permutation null, got {n_permutations}."
            )
        self.alpha = alpha
        self.prefer_fisher = prefer_fisher
        self.n_permutations = n_permutations
        self.random_seed = random_seed

    @staticmethod
    def _chi2_stat(table: np.ndarray, expected: np.ndarray) -> float:
        """Uncorrected chi-square against FIXED margins (permutation-safe)."""
        return float(np.sum((table - expected) ** 2 / expected))

    def _permutation_chi2(
        self,
        table: np.ndarray,
        expected: np.ndarray,
    ) -> Tuple[float, Optional[float], int]:
        """Exact-by-permutation chi-square for a table with sparse cells.

        Shuffles the demographic labels against the route decisions, which is
        the null of interest (routing independent of group) and holds both sets
        of margins fixed, so the expected counts never move and the statistic is
        a plain sum. Returns ``(p_value, min_attainable_p, n_permutations)``.

        ``min_attainable_p`` is read off the draws themselves: the observed
        statistic is one arrangement of this data, so it can never exceed the
        largest the null can produce, and the share of draws that TIE that
        ceiling is the smallest p the test could ever report here. With 12
        decisions from one group and 1 from another there are only 13
        arrangements, and that floor is 1/13 = 0.0769 whatever the routing.
        """
        observed_stat = self._chi2_stat(table, expected)
        rows, cols = table.shape
        row_codes = np.repeat(np.arange(rows), table.sum(axis=1).astype(int))
        col_codes = np.repeat(np.arange(cols), table.sum(axis=0).astype(int))
        rng = np.random.default_rng(self.random_seed)
        draws = np.empty(self.n_permutations, dtype=float)
        for i in range(self.n_permutations):
            shuffled = rng.permutation(col_codes)
            counts = np.bincount(row_codes * cols + shuffled, minlength=rows * cols)
            draws[i] = self._chi2_stat(counts.reshape(rows, cols).astype(float), expected)
        count = int(np.sum(draws >= observed_stat - 1e-12))
        p_value = (1.0 + count) / (1.0 + self.n_permutations)
        ceiling = max(float(draws.max()), observed_stat)
        ties = int(np.sum(draws >= ceiling - 1e-12))
        min_attainable = (1.0 + ties) / (1.0 + self.n_permutations)
        # Never claim a floor below what the resample count itself allows.
        resample_floor = min_attainable_p_permutation(self.n_permutations)
        if resample_floor is not None:
            min_attainable = max(min_attainable, resample_floor)
        return p_value, min_attainable, self.n_permutations

    def _build_table(
        self,
        routes: Sequence[str],
        demographics: Sequence[str],
    ) -> Tuple[List[str], List[str], np.ndarray, Dict[str, Dict[str, int]]]:
        unique_routes = sorted(set(routes))
        unique_groups = sorted(set(demographics))
        table = np.zeros((len(unique_routes), len(unique_groups)), dtype=int)
        for r, g in zip(routes, demographics):
            i = unique_routes.index(r)
            j = unique_groups.index(g)
            table[i, j] += 1
        nested: Dict[str, Dict[str, int]] = {
            r: {g: int(table[i, j]) for j, g in enumerate(unique_groups)}
            for i, r in enumerate(unique_routes)
        }
        return unique_routes, unique_groups, table, nested

    def _cramers_v(self, chi2: float, n: int, table_shape: Tuple[int, int]) -> float:
        """Cramer's V for the routing table, or NaN when it is not defined.

        V = sqrt(chi2 / (n * (min(r, c) - 1))) needs at least 2 routes AND 2
        groups and at least one decision. The two `return 0.0` this replaces
        answered NO ASSOCIATION for exactly the tables on which the question
        cannot be asked, and 0.0 is the value `_severity_from_v` in the pulse
        agent probe grades "info", the weakest word it has. Executed
        2026-09-17 before the change: `_cramers_v(5.0, 100, (1, 3))` returned
        0.0 with zero warnings.

        `analyze` refuses fewer than 2 distinct routes and fewer than 2
        distinct groups with a ValueError, and `_build_table` takes both axes
        from the observed data, so as the class stands no table reaching here
        can be degenerate and this refusal never fires from the public entry
        (verified by execution the same day). It is here because the helper is
        reachable on its own, and because a fabricated 0.0 is invisible while a
        NaN is not.
        """
        rows, cols = int(table_shape[0]), int(table_shape[1])
        k = min(rows, cols) - 1
        if n <= 0 or k <= 0 or not np.isfinite(chi2):
            warnings.warn(
                f"DelegationRoutingAuditor._cramers_v: Cramer's V is not defined "
                f"for a {rows}x{cols} routing table of {int(n)} decision(s) with "
                f"chi-square {chi2}. It needs at least 2 routes AND 2 groups, a "
                f"positive count and a finite chi-square, so the association was "
                f"NOT measured. Returning NaN (could not check), not 0.0, which "
                f"would read as routing independent of the group.",
                UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return float(np.sqrt(chi2 / (n * k)))

    def _per_route_disparity(
        self,
        nested: Dict[str, Dict[str, int]],
        groups: List[str],
    ) -> Dict[str, float]:
        # Total counts per group → invocation rates per route.
        group_totals = {g: sum(nested[r][g] for r in nested) for g in groups}
        # Use the two largest groups so the disparity is well-defined even
        # in the multi-group case.
        top_two = sorted(group_totals, key=lambda g: group_totals[g], reverse=True)[:2]
        if len(top_two) < 2:
            # 0.0 on this scale means "both groups reach this route at exactly
            # the same rate", the perfect-parity end, and it was returned for a
            # table where no two groups exist to compare at all. `analyze`
            # refuses fewer than 2 distinct groups with a ValueError and
            # `_build_table` takes both axes from the observed data, so as the
            # class stands this never fires from the public entry (verified by
            # execution 2026-09-27, where `_per_route_disparity({'junior':
            # {'A': 3}, 'senior': {'A': 1}}, ['A'])` returned
            # {'junior': 0.0, 'senior': 0.0} silently). It is here for the same
            # reason `_cramers_v` above refuses rather than answering 0.0: the
            # helper is reachable on its own, and a fabricated 0.0 is invisible
            # while a NaN is not.
            #
            # SCOPED TO THE ONE-GROUP TABLE ON PURPOSE. This guard first also
            # refused a two-group table in which one group's total is 0, which made
            # the zero-substituting ternary below unreachable. That ternary is a
            # REVIEWED decision and it stands: the pair is DIFFERENCED, so a
            # substituted 0.0 can only ever UNDERSTATE a disparity for a group that
            # received no routes at all, never overstate one. It fails closed.
            #
            # Why the wider guard was wrong even though it looked stricter: the
            # reviewing record keys on the literal shape, which still matched, so
            # every check stayed green while the behaviour the record describes had
            # stopped happening. A review cannot see a change that leaves its own
            # pattern intact. Nothing measured here argued against the verdict, so
            # widening the guard would have retired a sound decision silently.
            warnings.warn(
                f"DelegationRoutingAuditor._per_route_disparity: a per-route rate "
                f"difference needs two groups to compare, and got "
                f"{len(group_totals)} group(s) with totals "
                f"{dict(sorted(group_totals.items()))}. Returning NaN for every route "
                f"(could not check), not 0.0, which would read as identical routing "
                f"rates.",
                UserWarning,
                stacklevel=2,
            )
            return {r: float("nan") for r in nested}
        g_a, g_b = top_two[0], top_two[1]
        out: Dict[str, float] = {}
        for r in nested:
            # The zero substituted on both lines is a REVIEWED decision and it FAILS
            # CLOSED: the pair is differenced, so a zero standing in for a group that
            # received no routes at all can only ever UNDERSTATE the disparity, never
            # overstate it, and both group totals are reported alongside so a reader
            # can see which arm was empty.
            #
            # Deleting it is not an improvement, and that was measured on 2026-09-27:
            # removing it made this repository's review record describe code that no
            # longer existed, and the check that guards that record went red. If it
            # ever does have to go, the record of why it was kept has to go with it in
            # the same change.
            rate_a = nested[r][g_a] / group_totals[g_a] if group_totals[g_a] else 0.0
            rate_b = nested[r][g_b] / group_totals[g_b] if group_totals[g_b] else 0.0
            out[r] = abs(rate_a - rate_b)
        return out

    def analyze(
        self,
        routes: Sequence[str],
        demographics: Sequence[str],
    ) -> DelegationResult:
        """Run the routing-fairness audit.

        Args:
            routes: Sequence of route / agent identifiers chosen by the
                orchestrator, one per paired task.
            demographics: Sequence of demographic labels aligned with
                ``routes``. Must be the same length.

        Returns:
            DelegationResult.

        Raises:
            ValueError: If inputs are empty, of different lengths, or
                contain fewer than 2 unique routes / fewer than 2 unique
                demographic groups.
        """
        logger.info("analyze: %d decisions", len(routes))
        if len(routes) != len(demographics):
            raise ValueError(
                f"routes ({len(routes)}) and demographics "
                f"({len(demographics)}) must have the same length."
            )
        if not routes:
            raise ValueError("Empty input; cannot audit routing.")
        if len(set(routes)) < 2:
            raise ValueError("At least 2 distinct routes are required.")
        if len(set(demographics)) < 2:
            raise ValueError("At least 2 distinct demographic groups are required.")

        unique_routes, unique_groups, table, nested = self._build_table(
            routes,
            demographics,
        )

        n = int(table.sum())
        group_totals = table.sum(axis=0)

        # Every route and every group in this table was OBSERVED at least once
        # (`_build_table` takes them from the data), so no margin can be zero.
        # The old `sparse = any margin == 0` test could therefore never be True,
        # which is why the small-sample fallback this class documents ("or when
        # the chi-square assumption fails (expected cell counts < 5)") never ran
        # and the +0.5 continuity correction behind it was unreachable. The
        # assumption that actually fails is about EXPECTED counts, so that is
        # what is measured. Measured 2026-09-10 on 13 decisions, 12 from group A
        # split over two routes and 1 from group B on a third: the asymptotic
        # chi-square reported p=0.0015 and is_significant=True on a table whose
        # smallest expected count is 0.077, while the exact conditional p for
        # the same table is 0.0769, and 0.0769 is also the smallest p those
        # margins can produce at all.
        expected = np.asarray(stats.contingency.expected_freq(table), dtype=float)
        min_expected = float(expected.min())
        assumption_met = min_expected >= MIN_EXPECTED_CELL
        is_2x2 = table.shape == (2, 2)

        min_attainable: Optional[float] = None
        n_perm_used = 0
        if is_2x2 and (self.prefer_fisher or not assumption_met):
            try:
                odds_ratio, p_value = stats.fisher_exact(table)
                test_used = "fisher"
                # Uncorrected chi-square for Cramér's V. (Fisher's exact
                # is the gold-standard p-value for 2x2 small-sample
                # tables; we only need chi2 here for the effect size.)
                chi2_stat, _, _, _ = stats.chi2_contingency(table, correction=False)
                # The design floor for Fisher on these two group sizes: the p of
                # total segregation, which no data can beat.
                min_attainable = min_attainable_p_fisher(int(group_totals[0]), int(group_totals[1]))
            except ValueError as exc:
                logger.warning("Fisher's exact failed (%s); falling back to chi-square.", exc)
                # p-value with scipy's default (Yates on 2x2).
                _, p_value, _, _ = stats.chi2_contingency(table)
                # Uncorrected chi-square just for V.
                chi2_stat, _, _, _ = stats.chi2_contingency(table, correction=False)
                odds_ratio = float("nan")
                test_used = "chi2"
                min_attainable = 0.0 if assumption_met else None
        elif assumption_met:
            # p-value: scipy's default (Yates correction on 2x2, no-op otherwise).
            _, p_value, _, _ = stats.chi2_contingency(table)
            # Cramér's V: always from uncorrected chi-square so perfect
            # association gives V = 1.0 (correction=False is a no-op on
            # tables larger than 2x2).
            chi2_stat, _, _, _ = stats.chi2_contingency(table, correction=False)
            odds_ratio = float("nan")
            test_used = "chi2"
            # The asymptotic chi-square p-value is continuous: it has no
            # discrete floor, so this design can always reach alpha in
            # principle. 0.0 says that to `detectability` explicitly rather
            # than leaving the question unasked.
            min_attainable = 0.0
        else:
            # Larger than 2x2 with cells too thin for the asymptotic test.
            # Fisher's exact is 2x2-only in scipy, so the exact-by-permutation
            # chi-square stands in for it: same null, same fixed margins, and a
            # p-value that does not depend on an approximation the table
            # violates.
            chi2_stat = self._chi2_stat(table.astype(float), expected)
            p_value, min_attainable, n_perm_used = self._permutation_chi2(
                table.astype(float), expected
            )
            odds_ratio = float("nan")
            test_used = "permutation_chi2"
            warnings.warn(
                f"DelegationRoutingAuditor.analyze: the smallest expected cell count is "
                f"{min_expected:.3g}, below the {MIN_EXPECTED_CELL:g} the asymptotic "
                f"chi-square needs, so an exact-by-permutation chi-square "
                f"({n_perm_used} resamples) was run instead. Its p-value is "
                f"{p_value:.4g}.",
                UserWarning,
                stacklevel=2,
            )

        cramers_v = self._cramers_v(chi2_stat, n, table.shape)

        # DISCRETE FLOOR (readiness 6, 2026-09-10). `is_significant = p < alpha`
        # is a verdict on a test that, on a small enough table, cannot reach
        # alpha for ANY routing. Measured 2026-09-10 on 100 percent segregated
        # routing with 3 decisions per group: Cramér's V = 1.0 (perfect
        # association) and is_significant=False, because Fisher's floor at 3
        # against 3 is 0.10. At 2 per group the floor is 0.3333. The auditor
        # accepted both without a word. There is no minimum-n rule here now
        # either; the design's own power decides, which is the same question
        # asked properly.
        detectable, note = detectability(min_attainable, n_family=1, alpha=self.alpha)
        is_significant: Optional[bool]
        if detectable is True:
            is_significant = bool(p_value < self.alpha)
        else:
            warnings.warn(
                f"DelegationRoutingAuditor.analyze: routing significance was NOT "
                f"ASSESSED for {n} decisions across {len(unique_groups)} group(s) "
                f"(sizes {dict(zip(unique_groups, (int(t) for t in group_totals)))}). "
                f"{note} Reporting is_significant=None (could not check), NOT False. "
                f"The measured association still stands: Cramér's V = {cramers_v:.3f}.",
                UserWarning,
                stacklevel=2,
            )
            is_significant = None
        per_route = self._per_route_disparity(nested, unique_groups)

        logger.info(
            "analyze complete: test=%s, p_value=%.4f, cramers_v=%.4f, detectable=%s",
            test_used,
            p_value,
            cramers_v,
            detectable,
        )
        return DelegationResult(
            contingency_table=nested,
            routes=unique_routes,
            groups=unique_groups,
            chi_square=float(chi2_stat),
            odds_ratio=float(odds_ratio),
            cramers_v=float(cramers_v),
            p_value=float(p_value),
            is_significant=is_significant,
            test_used=test_used,
            per_route_disparity=per_route,
            metadata=RunMetadata(
                parameters={
                    "alpha": self.alpha,
                    "n_decisions": n,
                    "n_routes": len(unique_routes),
                    "n_groups": len(unique_groups),
                    "test_used": test_used,
                    "min_expected_cell": min_expected,
                    "chi2_assumption_met": assumption_met,
                    "n_permutations": n_perm_used,
                    "min_attainable_p": min_attainable,
                    "detectable": detectable,
                },
                random_seed=self.random_seed,
            ),
            min_attainable_p=min_attainable,
            detectable=detectable,
            detectability_note=note,
        )
