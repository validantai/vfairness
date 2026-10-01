"""
Reweighting Analysis Tools for vfairness.

This module provides comprehensive analysis tools for understanding the
impact of different reweighting strategies on fairness and model performance.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
)

from .reweighter import (
    BaseReweighter,
    CalibratedEqualizer,
    DistributionMatcher,
    PredictionReweighter,
    RejectionOptionClassifier,
    _compute_group_positive_rates,
    _group_disparity,
)


@dataclass
class ReweightingImpactResult:
    """
    Result of analyzing a reweighting method.

    Attributes:
        method: Reweighting method analyzed
        original_fairness: Fairness metrics before reweighting
        adjusted_fairness: Fairness metrics after reweighting
        original_performance: Performance before reweighting
        adjusted_performance: Performance after reweighting
        calibration_metrics: Impact on calibration
        trade_off_score: Overall trade-off score
    """

    method: str
    # Holds a scalar 'demographic_parity_diff' plus a nested 'group_rates'
    # dict, so the value type is intentionally heterogeneous.
    original_fairness: Dict[str, Any]
    adjusted_fairness: Dict[str, Any]
    original_performance: Dict[str, float]
    adjusted_performance: Dict[str, float]
    calibration_metrics: Dict[str, float]
    trade_off_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "method": self.method,
            "original_fairness": self.original_fairness,
            "adjusted_fairness": self.adjusted_fairness,
            "original_performance": self.original_performance,
            "adjusted_performance": self.adjusted_performance,
            "calibration_metrics": self.calibration_metrics,
            "trade_off_score": self.trade_off_score,
        }


@dataclass
class ReweightingAnalysisReport:
    """
    Comprehensive reweighting analysis report.

    Attributes:
        method_results: Results for each analyzed method
        best_method: The method with the highest trade-off score among the
            survivors. IT IS THE WINNER OF A COMPARISON, NOT A CLAIM THAT IT
            HELPED. Read ``metadata['no_method_reduced_disparity']`` beside it:
            when that is True, not one evaluated method reduced the
            demographic-parity gap and applying this one is not an improvement
            (G10, 2026-09-30).
        comparison_summary: Summary comparing methods
        recommendations: Analysis recommendations
        failed_methods: Methods that raised during analysis, mapped to the
            failure reason; best_method is chosen only among survivors
        metadata: Run context. Always carries
            ``no_method_reduced_disparity`` (bool),
            ``best_fairness_improvement`` and
            ``original_demographic_parity_diff`` on a report from
            ``full_analysis``, so a consumer never has to read an absent key
            as good news.
    """

    method_results: List[ReweightingImpactResult]
    best_method: str
    comparison_summary: Dict[str, Dict[str, float]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    failed_methods: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "method_results": [r.to_dict() for r in self.method_results],
            "best_method": self.best_method,
            "comparison_summary": self.comparison_summary,
            "recommendations": self.recommendations,
            "failed_methods": self.failed_methods,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Reweighting Analysis Report", "=" * 50]

        lines.append(f"\nAnalyzed {len(self.method_results)} reweighting methods")
        lines.append(f"Recommended method: {self.best_method}")

        # G10, 2026-09-30. The line right under "Recommended method" is where a
        # reader decides what to apply, so the disclosure belongs here and not
        # only in `recommendations` further down. `is True` rather than a truthy
        # test: the key is absent on a report built by hand, and absence is not
        # the same claim as False.
        if self.metadata.get("no_method_reduced_disparity") is True:
            lines.append(
                "  NOT A MITIGATION: no evaluated method reduced the "
                "demographic-parity gap (best change "
                f"{float(self.metadata.get('best_fairness_improvement', float('nan'))):+.4f} "
                "against an original gap of "
                f"{float(self.metadata.get('original_demographic_parity_diff', float('nan'))):.4f})."
                " The name above is the winner of the comparison between these "
                "methods, not a method that improved fairness."
            )

        if self.failed_methods:
            lines.append(
                f"\nWARNING: {len(self.failed_methods)} method(s) failed and "
                "were excluded from the comparison:"
            )
            for method, reason in self.failed_methods.items():
                lines.append(f"  {method}: {reason}")

        lines.append("\nMethod Comparison:")
        for method, metrics in self.comparison_summary.items():
            lines.append(f"\n  {method}:")
            for metric, value in metrics.items():
                lines.append(f"    {metric}: {value:.4f}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


def _baseline_disparity(method_results: List["ReweightingImpactResult"]) -> float:
    """The original demographic-parity gap every method was measured against.

    G10, 2026-09-30. Read off the results rather than recomputed, so the number a
    recommendation quotes is the same number the methods were compared on. Every
    surviving result carries the SAME original gap (it is the pre-adjustment
    frame), so the first one is authoritative; NaN when there is no result or the
    gap was never measurable, which the caller reports as such rather than as a
    zero gap.
    """
    for result in method_results:
        gap = result.original_fairness.get("demographic_parity_diff")
        if gap is not None and np.isfinite(gap):
            return float(gap)
    return float("nan")


def _decisions(scores: np.ndarray, threshold: float, where: str) -> np.ndarray:
    """Threshold scores into decisions, refusing the rows nothing scored.

    BGL-S2B (2026-09-17). ``np.nan >= 0.5`` is False, so ``(scores >=
    threshold).astype(int)`` turned every row a reweighter had REFUSED to score
    into a hard ``0``: a confident rejection, indistinguishable from a measured
    0.4. Executed on 179 rows whose 29-row minority DistributionMatcher
    declined to fit, this manufactured 29 rejections, 15 of which were then
    counted as CORRECT, inside a reported accuracy of 0.5475.

    Three states, never two. When every row carries a score the return is the
    int array of 0/1 decisions it has always been. When any row does not, the
    return is a float array carrying 0.0, 1.0 and NaN, and the NaN rows are
    named in a warning: not scored is not denied.

    The undecidable rows are the NaNs, not every non-finite value.
    ``+/-inf >= threshold`` is a true comparison against a finite threshold, so
    an infinite score IS decided here; excluding it would delete a real
    finding by turning a certain acceptance into a could-not-check.
    """
    values = np.asarray(scores, dtype=float)
    undecidable = np.isnan(values)
    n_undecided = int(np.count_nonzero(undecidable))
    if n_undecided == 0:
        return (values >= threshold).astype(int)
    warnings.warn(
        f"{where}: {n_undecided} of {values.size} row(s) carry no score, so no "
        f"decision was made for them. They are NaN in the decision vector, not 0: "
        f"a 0 here is a measured rejection and these rows were never scored. They "
        f"are excluded from accuracy rather than counted as rejections.",
        UserWarning,
        stacklevel=3,
    )
    decisions = np.full(values.shape, float("nan"), dtype=float)
    decided = ~undecidable
    decisions[decided] = (values[decided] >= threshold).astype(float)
    return decisions


def _n_decided(y_pred: np.ndarray) -> int:
    """How many rows of a decision vector actually carry a decision."""
    return int(np.count_nonzero(~np.isnan(np.asarray(y_pred, dtype=float))))


def _labelled_mask(y_true: np.ndarray) -> np.ndarray:
    """Rows that carry a ground-truth label at all.

    BGL5 (2026-09-27). ``coerce_to_array(y_true).astype(int)`` turns a MISSING
    label into a real one: on this platform NaN casts to 0 (on others to
    INT_MIN), behind a single numpy "invalid value encountered in cast"
    RuntimeWarning that nothing in this module reads. Measured on 160 rows of
    which 40 carried no label: ``analyze_method("multiplicative")`` reported
    accuracy 0.75 with ``n_decided`` 160 and ``n_rows`` 160, and original_ece
    0.2 with ``original_ece_rows_used`` 160, so the coverage fields asserted
    that all 160 rows had been graded. Accuracy over the 120 rows that DO carry
    a label is 1.0.

    ``pd.isna`` is the dtype-agnostic test (float nan, None and pd.NA alike) and
    is all-False for an int array, so a fully labelled input is unchanged. It is
    the same test :func:`validate_probabilities` uses for the score column,
    which is how the unscored row was closed; this is the unlabelled row.
    """
    import pandas as pd

    return ~np.asarray(pd.isna(np.asarray(y_true)), dtype=bool)


def _ece_usable_mask(y_prob: np.ndarray) -> np.ndarray:
    """Rows a calibration error can be computed from: a probability in [0, 1].

    A NaN (never scored) and a score outside the unit interval fall into no
    bin, so neither contributes evidence about calibration.
    """
    values = np.asarray(y_prob, dtype=float)
    return np.isfinite(values) & (values >= 0.0) & (values <= 1.0)


def _compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy over the rows a decision was actually made for.

    BGL-S2B (2026-09-17). ``y_pred`` carries NaN for every row :func:`_decisions`
    refused (see there). ``y_true == np.nan`` is False, so an undecided row was
    scored as WRONG when its label was 1 and as RIGHT when its label was 0,
    which is an accuracy statement about decisions that were never made.
    Undecided rows leave both sides of the ratio, and an input with no decided
    row at all returns NaN (could not check), never 0.0.
    """
    decisions = np.asarray(y_pred, dtype=float)
    decided = ~np.isnan(decisions)
    if not bool(decided.any()):
        return float("nan")
    return float(np.mean(np.asarray(y_true)[decided] == decisions[decided]))


def _compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over the rows carrying a usable probability.

    BGL-S2B (2026-09-17). A row whose probability is NaN, or outside [0, 1],
    lands in no bin, but ``bin_size`` divided by the FULL row count, so those
    rows quietly pulled the reported error toward 0 while it still read as an
    ECE over the whole sample: 29 unscored rows out of 179 turned a measured
    0.1262 into a reported 0.1058. The bin weights are normalised over the rows
    actually binned, and an input with no usable probability returns NaN (could
    not check), never 0.0. An all-usable input is unchanged.

    The count that was used is not visible in this float, so the caller
    (:meth:`ReweightingAnalyzer.analyze_method`) publishes it beside the value
    and warns when the sample was narrowed.
    """
    usable = _ece_usable_mask(y_prob)
    n_usable = int(np.count_nonzero(usable))
    if n_usable == 0:
        return float("nan")

    y_true_usable = np.asarray(y_true)[usable]
    y_prob_usable = np.asarray(y_prob, dtype=float)[usable]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            # Close the top bin: an open upper edge silently dropped every
            # sample with probability exactly 1.0 from the ECE.
            mask = (y_prob_usable >= bin_edges[i]) & (y_prob_usable <= bin_edges[i + 1])
        else:
            mask = (y_prob_usable >= bin_edges[i]) & (y_prob_usable < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = np.mean(y_true_usable[mask])
            bin_conf = np.mean(y_prob_usable[mask])
            bin_size = mask.sum() / n_usable
            ece += bin_size * abs(bin_acc - bin_conf)

    return ece


class ReweightingAnalyzer:
    """
    Comprehensive reweighting analysis for fairness-aware predictions.

    This analyzer compares different reweighting methods and their impact
    on fairness, accuracy, and calibration, helping practitioners choose
    the best approach for their use case.

    Example:
        >>> analyzer = ReweightingAnalyzer(y_true, y_prob, gender)
        >>> report = analyzer.full_analysis()
        >>> print(report.summary())
        >>>
        >>> # Compare specific methods
        >>> comparison = analyzer.compare_methods(['multiplicative', 'rejection_option'])

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: reweighting_analysis. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ):
        """
        Initialize the reweighting analyzer.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sensitive_attr: Sensitive attribute for grouping
            sample_weight: Optional sample weights
        """
        y_true_raw = coerce_to_array(y_true)
        self.y_prob = coerce_to_array(y_prob)
        self.sensitive_attr = coerce_to_array(sensitive_attr)
        check_consistent_length(y_true_raw, self.y_prob, self.sensitive_attr)

        if sample_weight is not None:
            self.sample_weight = coerce_to_array(sample_weight)
        else:
            self.sample_weight = np.ones(len(y_true_raw))

        # A row nothing LABELLED cannot be graded, so it leaves the analysis here,
        # once, rather than being cast to a label and then counted as correct or
        # wrong. See _labelled_mask for the measurement: 40 unlabelled rows of 160
        # were graded inside a reported accuracy of 0.75 over n_decided 160, and
        # the accuracy over the 120 labelled rows is 1.0. Excluding is what
        # `handle_missing_values(strategy="exclude")` does elsewhere in this
        # package, and the count is published rather than dropped: it is in the
        # warning below, in `n_rows_unlabelled_excluded` on every performance dict
        # analyze_method returns, and in the full_analysis metadata.
        labelled = _labelled_mask(y_true_raw)
        self.n_rows_unlabelled_excluded = int(np.count_nonzero(~labelled))
        if self.n_rows_unlabelled_excluded:
            n_total = int(labelled.size)
            warnings.warn(
                f"ReweightingAnalyzer: {self.n_rows_unlabelled_excluded} of {n_total} "
                f"row(s) carry no ground-truth label and are EXCLUDED from this "
                f"analysis. Every accuracy, calibration error and base rate below "
                f"covers the {n_total - self.n_rows_unlabelled_excluded} labelled "
                f"row(s) only; a missing label used to be cast to a real one (NaN "
                f"becomes 0 on this platform), so those rows were graded as decisions "
                f"nobody could check. Groups made up entirely of unlabelled rows are "
                f"not in the comparison at all.",
                UserWarning,
                stacklevel=2,
            )
            y_true_raw = y_true_raw[labelled]
            self.y_prob = self.y_prob[labelled]
            self.sensitive_attr = self.sensitive_attr[labelled]
            self.sample_weight = self.sample_weight[labelled]

        self.y_true = y_true_raw.astype(int)

        self.unique_groups = sorted([str(g) for g in np.unique(self.sensitive_attr)])

        # A between-group disparity does not exist with fewer than two groups:
        # max-min over one value is structurally 0.0 for every possible input,
        # and 0.0 is the perfect-parity reading. Say so at construction; the
        # disparities themselves come back NaN from _group_disparity, and
        # full_analysis() then refuses to rank or recommend on them
        # (BGL-D, 2026-09-11).
        if len(self.unique_groups) < 2:
            warnings.warn(
                f"ReweightingAnalyzer: sensitive_attr holds {len(self.unique_groups)} "
                f"group(s) {self.unique_groups}, so no between-group disparity exists "
                f"to measure or reduce. Every demographic_parity_diff and "
                f"fairness_improvement is NaN (could-not-check), not 0.0, and no "
                f"method can be ranked on fairness.",
                UserWarning,
                stacklevel=2,
            )

    def analyze_method(
        self, method: Union[str, BaseReweighter], threshold: float = 0.5, **method_kwargs
    ) -> ReweightingImpactResult:
        """
        Analyze the impact of a single reweighting method.

        Args:
            method: Reweighting method name or instance
            threshold: Decision threshold for predictions
            **method_kwargs: Additional arguments for the method

        Returns:
            ReweightingImpactResult with detailed analysis
        """
        # The guard goes ABOVE the dispatch, because every branch below shares
        # this precondition. A non-finite threshold makes every comparison
        # False, so each group's positive rate comes back 0.0 and the gap
        # between them 0.0, which on this scale is the perfect-parity reading.
        # Measured on data with a real 1.0 gap, analyze_method(threshold=nan)
        # reported demographic_parity_diff 0.0, group_rates {'A': 0.0,
        # 'B': 0.0}, trade_off_score 0.0 and NOT ONE warning (BGL-S2B,
        # 2026-09-17). Refusing the argument is the only honest answer: there is
        # no subset of the data this threshold could decide.
        #
        # NaN only, deliberately not `not np.isfinite`. A threshold of +inf
        # ("accept nobody") or -inf ("accept everybody") decides every row, and
        # the 0.0 or 1.0 rates that follow ARE measurements of that rule;
        # refusing them would delete a real finding.
        if np.isnan(threshold):
            raise ValueError(
                f"threshold={threshold!r} is not a number, so no row can be decided "
                f"against it and no positive rate, disparity, accuracy or trade-off "
                f"exists to report. Every comparison with NaN is False, so every such "
                f"value would come back 0.0, which reads as perfect parity rather "
                f"than as a measurement that was never taken."
            )

        reweighter: BaseReweighter
        if isinstance(method, str):
            if method == "multiplicative":
                reweighter = PredictionReweighter(method="multiplicative", **method_kwargs)
            elif method == "additive":
                reweighter = PredictionReweighter(method="additive", **method_kwargs)
            elif method == "rejection_option":
                reweighter = RejectionOptionClassifier(**method_kwargs)
            elif method == "calibrated":
                reweighter = CalibratedEqualizer(**method_kwargs)
            elif method == "distribution_matching":
                reweighter = DistributionMatcher(**method_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            method_name = method
        else:
            reweighter = method
            method_name = reweighter.__class__.__name__

        reweighter.fit(y_true=self.y_true, y_prob=self.y_prob, sensitive_attr=self.sensitive_attr)
        y_prob_adjusted = reweighter.transform(self.y_prob, self.sensitive_attr)

        # Not `(scores >= threshold).astype(int)`: that cast every row the
        # reweighter refused to score into a measured rejection, and then
        # graded it (BGL-S2B, 2026-09-17).
        y_pred_original = _decisions(
            self.y_prob, threshold, "ReweightingAnalyzer.analyze_method (original scores)"
        )
        y_pred_adjusted = _decisions(
            y_prob_adjusted,
            threshold,
            f"ReweightingAnalyzer.analyze_method ({method_name} adjusted scores)",
        )

        original_rates = _compute_group_positive_rates(self.y_prob, self.sensitive_attr, threshold)
        adjusted_rates = _compute_group_positive_rates(
            y_prob_adjusted, self.sensitive_attr, threshold
        )

        # NaN, not 0.0, when there is no between-group comparison to make:
        # one group, or a group with no finite score (BGL-D, 2026-09-11).
        original_fairness: Dict[str, Any] = {
            "demographic_parity_diff": _group_disparity(original_rates),
            "group_rates": original_rates,
        }
        adjusted_fairness: Dict[str, Any] = {
            "demographic_parity_diff": _group_disparity(adjusted_rates),
            "group_rates": adjusted_rates,
        }

        # Every accuracy and ECE below is published beside the number of rows
        # it was computed over. A narrowed sample is still a measurement; a
        # narrowed sample reported as if it covered everybody is not.
        n_rows = int(np.size(self.y_true))
        original_performance: Dict[str, float] = {
            "accuracy": _compute_accuracy(self.y_true, y_pred_original),
            "n_decided": _n_decided(y_pred_original),
            "n_rows": n_rows,
            # Always present, 0 on a fully labelled input. n_rows is the GRADED
            # sample, so without this count a reader cannot tell it from the
            # sample they passed in.
            "n_rows_unlabelled_excluded": self.n_rows_unlabelled_excluded,
        }
        adjusted_performance: Dict[str, float] = {
            "accuracy": _compute_accuracy(self.y_true, y_pred_adjusted),
            "n_decided": _n_decided(y_pred_adjusted),
            "n_rows": n_rows,
            "n_rows_unlabelled_excluded": self.n_rows_unlabelled_excluded,
        }

        original_ece = _compute_calibration_error(self.y_true, self.y_prob)
        adjusted_ece = _compute_calibration_error(self.y_true, y_prob_adjusted)
        n_original_binned = int(np.count_nonzero(_ece_usable_mask(self.y_prob)))
        n_adjusted_binned = int(np.count_nonzero(_ece_usable_mask(y_prob_adjusted)))

        if n_original_binned < n_rows or n_adjusted_binned < n_rows:
            warnings.warn(
                f"ReweightingAnalyzer.analyze_method ({method_name}): the calibration "
                f"error covers {n_original_binned} of {n_rows} row(s) before the "
                f"adjustment and {n_adjusted_binned} of {n_rows} after it; the rest "
                f"carry no probability in [0, 1] and land in no bin. The bin weights "
                f"are normalised over the rows that were binned, so each ECE measures "
                f"those rows and not the whole sample, and the counts are reported in "
                f"calibration_metrics.",
                UserWarning,
                stacklevel=2,
            )

        calibration_metrics: Dict[str, float] = {
            "original_ece": original_ece,
            "adjusted_ece": adjusted_ece,
            "ece_change": adjusted_ece - original_ece,
            "original_ece_rows_used": n_original_binned,
            "adjusted_ece_rows_used": n_adjusted_binned,
            "n_rows": n_rows,
        }

        # Trade-off score (higher is better)
        fairness_improvement = (
            original_fairness["demographic_parity_diff"]
            - adjusted_fairness["demographic_parity_diff"]
        )
        accuracy_loss = original_performance["accuracy"] - adjusted_performance["accuracy"]
        # max(0, nan) is 0: `nan > 0` is False, so max returns its first
        # argument and a calibration error that could not be computed entered
        # the trade-off as "this method cost no calibration". NaN propagates
        # instead, and full_analysis then declines to rank the method
        # (BGL-S2B, 2026-09-17).
        ece_change = calibration_metrics["ece_change"]
        calibration_loss = max(0.0, float(ece_change)) if np.isfinite(ece_change) else float("nan")

        # Weighted trade-off (fairness improvement minus penalties)
        trade_off_score = fairness_improvement - 0.5 * accuracy_loss - 0.3 * calibration_loss

        return ReweightingImpactResult(
            method=method_name,
            original_fairness=original_fairness,
            adjusted_fairness=adjusted_fairness,
            original_performance=original_performance,
            adjusted_performance=adjusted_performance,
            calibration_metrics=calibration_metrics,
            trade_off_score=trade_off_score,
        )

    def compare_methods(
        self,
        methods: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, ReweightingImpactResult]:
        """
        Compare multiple reweighting methods.

        Args:
            methods: List of method names to compare
            threshold: Decision threshold

        Returns:
            Dictionary mapping method names to results
        """
        if methods is None:
            methods = ["multiplicative", "additive", "rejection_option", "calibrated"]

        results = {}
        for method in methods:
            try:
                results[method] = self.analyze_method(method, threshold)
            except Exception as e:
                # A print() disappears in non-interactive runs; warn so the
                # caller can see the comparison is missing this method.
                warnings.warn(f"Failed to analyze {method}: {e}")

        return results

    def full_analysis(
        self,
        threshold: float = 0.5,
    ) -> ReweightingAnalysisReport:
        """
        Perform comprehensive reweighting analysis.

        Args:
            threshold: Decision threshold

        Returns:
            ReweightingAnalysisReport with complete analysis
        """
        methods = [
            "multiplicative",
            "additive",
            "rejection_option",
            "calibrated",
            "distribution_matching",
        ]

        method_results = []
        comparison_summary = {}
        failed_methods: Dict[str, str] = {}

        for method in methods:
            try:
                result = self.analyze_method(method, threshold)
                fairness_improvement = (
                    result.original_fairness["demographic_parity_diff"]
                    - result.adjusted_fairness["demographic_parity_diff"]
                )
                # A method whose fairness effect could not be measured is not a
                # method that improved fairness by 0.000. Route it to the
                # existing disclosure field so it is excluded from the ranking
                # and from _generate_recommendations, which would otherwise
                # name a "best" method and quote the fabricated reduction
                # (BGL-D, 2026-09-11).
                if not np.isfinite(fairness_improvement):
                    reason = (
                        "NotMeasurable: the demographic-parity gap could not be "
                        "measured on this data (fewer than two groups, or a group "
                        "with no finite score), so the method's fairness effect is "
                        "NaN, not 0.0, and it cannot be compared or ranked."
                    )
                    failed_methods[method] = reason
                    warnings.warn(f"Cannot measure fairness effect of {method}: {reason}")
                    continue
                # The gap was measurable, but the other two arms of the
                # trade-off may not have been: no row could be decided, or no
                # row carried a probability a calibration error can be computed
                # from. Ranking on a NaN trade-off score is ordering by
                # whichever entry max() happened to see first (BGL-S2B).
                if not np.isfinite(result.trade_off_score):
                    reason = (
                        "NotMeasurable: the demographic-parity gap was measured, but "
                        "the accuracy or the calibration error was not (no row could "
                        "be decided, or no row carried a probability in [0, 1]), so "
                        "the trade-off score is NaN, not 0.0, and this method cannot "
                        "be compared or ranked against the others."
                    )
                    failed_methods[method] = reason
                    warnings.warn(f"Cannot measure the trade-off of {method}: {reason}")
                    continue
                method_results.append(result)
                comparison_summary[method] = {
                    "fairness_improvement": fairness_improvement,
                    "accuracy_change": (
                        result.adjusted_performance["accuracy"]
                        - result.original_performance["accuracy"]
                    ),
                    "calibration_change": result.calibration_metrics["ece_change"],
                    "trade_off_score": result.trade_off_score,
                }
            except Exception as e:
                # Record and warn instead of print(): silently dropping a
                # method let the report recommend a "best" method without
                # disclosing that alternatives were never evaluated.
                failed_methods[method] = f"{type(e).__name__}: {e}"
                warnings.warn(f"Failed to analyze {method}: {e}")

        if method_results:
            best_method = max(method_results, key=lambda r: r.trade_off_score).method
        else:
            best_method = "None (all methods failed)"

        # G10, 2026-09-30. THE DISCLOSURE HAS TO BE REACHABLE FROM THE OBJECT, not
        # only from the prose. `best_method` is typed `str` and documented as the
        # highest trade-off score, and a caller that reads it and applies that
        # method would have applied a no-op. The flag is published beside it, the
        # way `parameters['degenerate_constant_predictions']` is published beside
        # `constraint_satisfied` in FairnessTrainingAnalyzer.compare_methods; the
        # value of `best_method` is deliberately NOT changed, because it still
        # correctly names the winner of the comparison that was made.
        best_reduction = (
            max(v["fairness_improvement"] for v in comparison_summary.values())
            if comparison_summary
            else float("nan")
        )
        no_method_reduced_disparity = bool(comparison_summary) and not (best_reduction > 0)
        if no_method_reduced_disparity:
            warnings.warn(
                f"ReweightingAnalyzer.full_analysis: NOT ONE of the "
                f"{len(comparison_summary)} evaluated method(s) reduced the "
                f"demographic-parity gap (best change {float(best_reduction):+.4f} against "
                f"an original gap of {_baseline_disparity(method_results):.4f}). "
                f"best_method={best_method!r} names the winner of the comparison between "
                f"them, NOT a method that improved fairness; read "
                f"metadata['no_method_reduced_disparity'] beside it.",
                UserWarning,
                stacklevel=2,
            )

        recommendations = self._generate_recommendations(method_results, comparison_summary)

        if failed_methods:
            recommendations.append(
                f"NOT EVALUATED: {sorted(failed_methods)} failed during "
                "analysis; the recommendation only compares the methods "
                "that succeeded."
            )

        return ReweightingAnalysisReport(
            method_results=method_results,
            best_method=best_method,
            comparison_summary=comparison_summary,
            recommendations=recommendations,
            metadata={
                "n_samples": len(self.y_true),
                "n_groups": len(self.unique_groups),
                "groups": self.unique_groups,
                "threshold": threshold,
                # n_samples is the GRADED sample. Always present, 0 when nothing
                # was excluded, so a reader can reconcile it with their input.
                "n_rows_unlabelled_excluded": self.n_rows_unlabelled_excluded,
                # G10. Always present (False, never absent, on a run where a
                # method did reduce the gap) so a consumer can test it without
                # a `.get` default that would read an absent key as good news.
                "no_method_reduced_disparity": no_method_reduced_disparity,
                "best_fairness_improvement": float(best_reduction),
                "original_demographic_parity_diff": _baseline_disparity(method_results),
            },
            failed_methods=failed_methods,
        )

    def _generate_recommendations(
        self,
        method_results: List[ReweightingImpactResult],
        comparison_summary: Dict,
    ) -> List[str]:
        """Generate analysis recommendations."""
        recommendations = []

        if not method_results:
            recommendations.append("No methods could be analyzed successfully.")
            return recommendations

        best_fairness = max(
            comparison_summary.keys(), key=lambda m: comparison_summary[m]["fairness_improvement"]
        )
        best_reduction = float(comparison_summary[best_fairness]["fairness_improvement"])
        # G10, 2026-09-30. A NEUTERED MITIGATION REPORTS SUCCESS, NOT FAILURE.
        # This sentence was unconditional, so when NOT ONE method reduced the
        # disparity it still read as an instruction to apply one. Reproduced on
        # deterministic data (y_prob = tile([0.2, 0.8]), y_true = tile([0, 1]),
        # two groups of 200, so every method's fairness_improvement is exactly
        # 0.000): "For maximum fairness improvement, use 'multiplicative'
        # (reduces disparity by 0.000)." followed by "Overall recommendation:
        # 'multiplicative' provides the best trade-off between fairness,
        # accuracy, and calibration." A reader is told to apply a mitigation
        # that mitigated nothing. The template is worse in the negative
        # direction, which `max` over negatives reaches: a run measured the same
        # day gave rejection_option a fairness_improvement of -0.49, and had it
        # been the least-bad row the sentence would have printed "reduces
        # disparity by -0.490" for a method that WIDENED the gap.
        #
        # The numbers were always right; the READING was not. So the numbers are
        # kept and the sentence is made to say what they mean, including the
        # ORIGINAL gap, because "nothing reduced it" is the correct outcome on
        # data that had nothing to reduce and a finding on data that did.
        baseline_gap = _baseline_disparity(method_results)
        if best_reduction > 0:
            recommendations.append(
                f"For maximum fairness improvement, use '{best_fairness}' "
                f"(reduces disparity by {best_reduction:.3f})."
            )
        else:
            nothing_to_fix = np.isfinite(baseline_gap) and baseline_gap <= 1e-12
            baseline_text = (
                f"an original demographic-parity gap of {baseline_gap:.3f}"
                if np.isfinite(baseline_gap)
                else "an original demographic-parity gap that could not be measured"
            )
            recommendations.append(
                f"NO METHOD REDUCED THE DISPARITY: the best change any of them made was "
                f"{best_reduction:+.3f} ('{best_fairness}'), against {baseline_text}. "
                + (
                    "That gap is already at zero, so there is nothing here for a "
                    "reweighter to reduce and applying one is not an improvement."
                    if nothing_to_fix
                    else "Applying any of them would leave that gap in place or widen "
                    "it, so none of them is a mitigation for this data."
                )
            )

        best_accuracy = max(
            comparison_summary.keys(), key=lambda m: comparison_summary[m]["accuracy_change"]
        )
        if comparison_summary[best_accuracy]["accuracy_change"] >= 0:
            recommendations.append(f"'{best_accuracy}' preserves accuracy best (no accuracy loss).")
        else:
            recommendations.append(
                f"All methods reduce accuracy somewhat. '{best_accuracy}' has the "
                f"smallest loss ({abs(comparison_summary[best_accuracy]['accuracy_change']):.3f})."
            )

        methods_hurting_calibration = [
            m for m in comparison_summary if comparison_summary[m]["calibration_change"] > 0.05
        ]
        if methods_hurting_calibration:
            recommendations.append(
                f"Warning: {', '.join(methods_hurting_calibration)} significantly "
                "degrade calibration. Consider this if probability estimates are important."
            )

        best_overall = max(
            comparison_summary.keys(), key=lambda m: comparison_summary[m]["trade_off_score"]
        )
        if best_reduction > 0:
            recommendations.append(
                f"Overall recommendation: '{best_overall}' provides the best "
                "trade-off between fairness, accuracy, and calibration."
            )
        else:
            # G10. The same correction on the line a reader is most likely to act
            # on. `max` always returns a winner, and a winner of a comparison
            # between methods that all changed nothing is not a recommendation.
            recommendations.append(
                f"NO OVERALL RECOMMENDATION: '{best_overall}' has the highest "
                f"trade-off score ({float(comparison_summary[best_overall]['trade_off_score']):+.3f}), "
                "but that only ranks these methods against each other. Since none of "
                "them reduced the disparity, it is NOT evidence that applying it "
                "improves fairness."
            )

        return recommendations

    def get_explanation(self, report=None):
        """Generate educational explanations for the reweighting analysis.

        Parameters
        ----------
        report : ReweightingAnalysisReport, optional
            Pre-computed report. If *None*, :meth:`full_analysis` is run.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        if report is None:
            report = self.full_analysis()
        return FairnessExplainer.explain(report)
