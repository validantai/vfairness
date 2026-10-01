"""
vfairness.rendering.explain: self-explaining, accessible SVG output.

Every SVG vfairness renders carries its own explanation, automatically:

  * an **on-canvas** explanation, auto-generated when the caller supplies none,
    written as concept -> how to read -> the finding on *this* chart -> what to
    do; and
  * an **accessible + machine-readable** layer: ``role="img"``, a ``<title>``, a
    one-line ``<desc>`` ("what it shows + the key finding + severity"), and a
    ``<metadata>`` JSON block carrying the structured explanation. A screen
    reader, a report pipeline, or an assistant can read the finding without
    OCR-ing the pixels.

Design
------
A curated per-chart knowledge base (``CHART_META``) supplies the stable,
data-independent content: title, concept, how-to-read, recommended action. A
per-chart finding extractor (``_FINDERS``) reads the template's *data dict*, the
same values the chart's badge is built from: to produce a data-specific finding
and a severity that agrees with the badge by construction (so the explanation can
never contradict the badge, e.g. "0.12 (MEDIUM)" next to a MINIMAL badge).

Everything is defensive: a missing key, a novel template, or a malformed value
degrades to concept-only. It never raises: an explanation must never break a
render.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# Severity vocabulary (ordered).
_SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_VALID_SEVERITY = set(_SEVERITY_ORDER)


def _norm_sev(sev: Any) -> str:
    s = str(sev or "info").lower()
    return s if s in _VALID_SEVERITY else "info"


# ── Curated per-chart knowledge base ────────────────────────────────────────
# title:   human name (used for <title> and the desc lead)
# concept: what the chart is / measures (one sentence)
# reading: how to interpret it (one sentence)
# action:  what to do about it (one short clause)
CHART_META: Dict[str, Dict[str, str]] = {
    "bias_audit": {
        "title": "Bias Audit",
        "concept": "A pre-deployment audit that combines historical patterns, representation balance, statistical disparities and proxy exposure into one dataset bias-risk score.",
        "reading": "The 0-1 risk score bands as minimal (<0.25), low, medium, then high (>=0.75); the tiles break the score into its four sources and the table lists the issues driving it.",
        "action": "Address the flagged critical issues before training; aim for an overall score below 0.25.",
    },
    "calibration_report": {
        "title": "Calibration Report",
        "concept": "Whether the model's predicted probabilities match observed outcome frequencies, overall and per demographic group.",
        "reading": "An Expected Calibration Error (ECE) near 0 is well calibrated; a large ECE disparity between groups means some groups receive less trustworthy probabilities.",
        "action": "Recalibrate (e.g. Platt or temperature scaling), per group if the disparity is significant.",
    },
    "fairness_report": {
        "title": "Fairness Metric Dashboard",
        "concept": "A grid of group-fairness metrics (demographic parity, equal opportunity, equalized odds, predictive parity), each scored pass or fail against a threshold.",
        "reading": "The 0-100 fairness score aggregates the metrics; each card shows the metric value versus its threshold and a fair/unfair verdict.",
        "action": "Investigate the metrics marked unfair; consider reweighting or threshold adjustment.",
    },
    "cicd_pipeline": {
        "title": "CI/CD Fairness Gate",
        "concept": "The automated gate that runs fairness tests on a model candidate and blocks promotion when a metric breaches its threshold.",
        "reading": "Each test shows its metric value against the threshold and pass/fail; the gate verdict and blocking reasons summarise whether the model may ship.",
        "action": "Fix the blocking tests, or retrain with fairness constraints, before merging.",
    },
    "radar_chart": {
        "title": "Fairness Radar",
        "concept": "An at-a-glance polygon of several fairness metrics on one shared fairness axis: the farther a vertex sits from the centre, the fairer that metric. A single dashed ring marks the pass threshold.",
        "reading": "Vertices outside the dashed ring pass; a vertex pulled inward toward the centre is a larger violation on that metric, so a fair model draws a large round shape and failures cave inward. Dot labels show the raw metric value. Overall status is Fair, Marginal or Unfair.",
        "action": "Focus on the spokes that fall inside the threshold ring.",
    },
    "disparity_heatmap": {
        "title": "Disparity Heatmap",
        "concept": "Fairness-metric disparity for each demographic group, shown as a colour-graded grid.",
        "reading": "Stronger-tinted cells mark larger gaps; the summary bands the maximum disparity as low (<0.1), moderate (<0.2) or high.",
        "action": "Examine the darkest cells to find the most disadvantaged group-metric pairs.",
    },
    "metrics_bar_chart": {
        "title": "Fairness Metrics Bar Chart",
        "concept": "Each fairness metric drawn as a horizontal bar against its threshold line.",
        "reading": "Bars that stay under the dashed threshold pass; bars that exceed it fail.",
        "action": "Bring the failing metrics under threshold via mitigation.",
    },
    "group_comparison": {
        "title": "Group Comparison",
        "concept": "The selected outcome rate for each demographic group, side by side.",
        "reading": "Differing bar heights show which groups receive favourable outcomes more often; the max-min gap is banded low, moderate or high.",
        "action": "If the gap is moderate or high, investigate the driver and consider rebalancing.",
    },
    "effect_sizes": {
        "title": "Effect Sizes",
        "concept": "Cohen's d for each group difference - the practical magnitude of a disparity beyond mere statistical significance.",
        "reading": "|d| bands: negligible (<0.2), small (<0.5), medium (<0.8), large (>=0.8); longer lollipops are larger effects.",
        "action": "Prioritise medium and large effects; small or negligible ones may be real but practically minor.",
    },
    "confidence_intervals": {
        "title": "Confidence Intervals",
        "concept": "A forest plot of each fairness metric with its confidence interval.",
        "reading": "An interval that excludes the no-difference line is a statistically reliable disparity; one that crosses it is not.",
        "action": "Treat metrics whose intervals exclude zero as real disparities to address.",
    },
    "reliability_diagram": {
        "title": "Reliability Diagram",
        "concept": "A calibration curve comparing predicted probability to observed frequency, with a confidence histogram.",
        "reading": "Points on the diagonal are perfectly calibrated; above it is under-confident, below it is over-confident. ECE and MCE summarise the gap.",
        "action": "Recalibrate if ECE exceeds roughly 0.05.",
    },
    "group_calibration": {
        "title": "Group Calibration",
        "concept": "Per-group calibration curves overlaid to reveal group-specific miscalibration.",
        "reading": "Curves that diverge from the diagonal, or from each other, mean the model is better calibrated for some groups than others.",
        "action": "Apply group-aware recalibration when the ECE disparity is significant.",
    },
    "calibration_disparity": {
        "title": "Calibration Disparity",
        "concept": "A comparison of Expected Calibration Error across demographic groups.",
        "reading": "The ECE disparity and ratio quantify how much better some groups are calibrated than others; the worst and best groups are named.",
        "action": "Recalibrate the worst-calibrated group toward parity.",
    },
    "pareto_frontier": {
        "title": "Calibration-Fairness Pareto Frontier",
        "concept": "Each candidate model plotted by calibration error against fairness violation, with the non-dominated frontier highlighted.",
        "reading": "Frontier points are the best achievable trade-offs; points inside the frontier are dominated - worse on both axes.",
        "action": "Pick a frontier configuration that matches your calibration/fairness priority.",
    },
    "correlation_heatmap": {
        "title": "Feature-Attribute Correlation Heatmap",
        "concept": "Correlations between model features and protected attributes - a map of proxy-discrimination risk.",
        "reading": "Cells above the |r| threshold flag features that could act as proxies even if the protected attribute is never used directly.",
        "action": "Decorrelate or drop the high-correlation features.",
    },
    "correlation_matrix": {
        "title": "Correlation Matrix",
        "concept": "An N-by-N feature correlation matrix with an automatically chosen method per pair (Pearson, Spearman, Cramer's V, point-biserial).",
        "reading": "Off-diagonal cells above the |r| threshold flag redundant or proxy relationships; the dot marks the method used.",
        "action": "Review the high-correlation pairs for redundancy or proxy risk.",
    },
    "proxy_risk": {
        "title": "Proxy Risk Assessment",
        "concept": "Features ranked by how strongly they correlate with a protected attribute - their potential to enable indirect discrimination.",
        "reading": "Each feature is rated high, medium or low proxy risk by its correlation strength and confidence.",
        "action": "Remove or transform the high-risk proxies before modelling.",
    },
    "transformation_comparison": {
        "title": "Transformation Comparison",
        "concept": "Feature correlations with protected attributes before versus after a mitigation transformation.",
        "reading": "A larger reduction means more proxy signal was removed; effectiveness is banded effective, moderate or minimal.",
        "action": "Keep transformations that achieve a large reduction while preserving predictive power.",
    },
    "intersectional_analysis": {
        "title": "Intersectional Analysis",
        "concept": "An outcome rate for each cell at the intersection of two protected attributes.",
        "reading": "The most extreme cells reveal compound advantage or disadvantage that single-attribute analysis misses; max disparity is banded low, moderate or high.",
        "action": "Investigate the most disadvantaged intersection, not just each attribute alone.",
    },
    "intersectional_disparity": {
        "title": "Intersectional Disparity",
        "concept": "Subgroups at intersecting attributes ranked by outcome rate, contrasting the most and least advantaged.",
        "reading": "The gap in percentage points between the privileged and disadvantaged subgroup is banded by severity; delta marks the prediction versus ground-truth gap.",
        "action": "Target the most disadvantaged intersection; verify the gap exceeds either single-attribute gap.",
    },
    "training_report": {
        "title": "Fairness Training Report",
        "concept": "A summary of fairness-aware training: the baseline's accuracy and violation and the recommended mitigation method.",
        "reading": "A satisfied constraint means the baseline already meets the fairness bound; otherwise the recommended method trades a little accuracy for fairness.",
        "action": "Adopt the recommended method and re-check the constraint.",
    },
    "training_analysis_report": {
        "title": "Training Analysis Report",
        "concept": "A full-page view of the accuracy-fairness trade-off with method comparisons and action items.",
        "reading": "Read the baseline metrics, the base-rate disparity and the recommendation together; a violated constraint plus high base-rate disparity signals a data problem.",
        "action": "Apply the recommended method and collect more data for under-represented groups.",
    },
    "method_comparison": {
        "title": "Method Comparison",
        "concept": "Competing fairness-training methods compared on accuracy and fairness violation.",
        "reading": "Each method shows its accuracy and whether it satisfies the fairness constraint; the best maximises accuracy among those that satisfy it.",
        "action": "Choose the constraint-satisfying method with the least accuracy loss.",
    },
    "tradeoff_analysis": {
        "title": "Accuracy-Fairness Trade-off",
        "concept": "How much accuracy must be traded to tighten the fairness constraint, with the Pareto frontier marked.",
        "reading": "Each point is a constraint strength; frontier points are efficient trade-offs, and steeper drops mean fairness costs more accuracy.",
        "action": "Pick the constraint strength that meets your fairness target at acceptable accuracy.",
    },
    "threshold_optimization_report": {
        "title": "Threshold Optimisation",
        "concept": "Group-specific decision thresholds tuned to equalise outcome rates - a post-processing mitigation.",
        "reading": "Feasible means the target parity is achievable; the disparity reduction and accuracy change quantify the trade.",
        "action": "Deploy the group-specific thresholds if feasible and the accuracy cost is acceptable.",
    },
    "reweighting_comparison_report": {
        "title": "Reweighting Comparison",
        "concept": "Sample-reweighting methods compared on how well each improves fairness while preserving accuracy and calibration.",
        "reading": "The best method has the highest trade-off score; weigh fairness improvement against accuracy and calibration change.",
        "action": "Adopt the best-scoring method and monitor calibration after deployment.",
    },
    "fairness_detailed_report": {
        "title": "Detailed Fairness Report",
        "concept": "An executive fairness report combining metric-level and group-level findings into one assessment.",
        "reading": "A score below 50 (Unfair) signals significant concern, 50-80 is Marginal, above 80 is Fair; review both the metric table and group statistics.",
        "action": "Act on the key findings and recommendations for the failing metrics.",
    },
    "monitoring_dashboard": {
        "title": "Monitoring Dashboard",
        "concept": "A live snapshot of fairness metrics for the latest monitoring window.",
        "reading": "Each metric shows its current value and whether it breached its guardrail; any alert means production fairness has drifted out of bounds.",
        "action": "Triage the alerting metrics; if several alert at once, treat it as an incident.",
    },
    "drift_report": {
        "title": "Fairness Drift Report",
        "concept": "Multi-scale detection of whether model fairness is degrading over time.",
        "reading": "The 0-1 drift score bands as low, moderate (>=0.3) or high (>=0.6); short, medium and long scales separate transient noise from systematic shifts.",
        "action": "If medium or long-scale drift is detected, investigate the cause and consider retraining.",
    },
    "alert_timeline": {
        "title": "Alert Timeline",
        "concept": "A chronological view of fairness alerts across recent monitoring windows.",
        "reading": "Marked windows raised alerts; a rising or clustered pattern signals a recurring or escalating problem.",
        "action": "Investigate clusters of alerts rather than isolated blips.",
    },
    "temporal_analysis": {
        "title": "Temporal Analysis",
        "concept": "Fairness metrics tracked over time, with trend and seasonal or weekly pattern detection.",
        "reading": "An increasing trend in a disparity metric, or weekly degradation, points to distribution or concept drift.",
        "action": "Address metrics with a rising trend or recurring weekly degradation.",
    },
    "experiment_results": {
        "title": "Experiment Results",
        "concept": "A/B test results with per-intersection treatment effects and confidence intervals.",
        "reading": "A significant overall effect plus detected heterogeneity means the treatment affects subgroups differently.",
        "action": "Inspect heterogeneous subgroups before rollout; a benefit on average can hide harm to a group.",
    },
    "experiment_recommendation": {
        "title": "Experiment Recommendation",
        "concept": "A ship, extend or stop recommendation synthesised from effect, power and heterogeneity evidence.",
        "reading": "The decision and confidence summarise the evidence; extend or investigate means it is not yet conclusive.",
        "action": "Follow the recommendation; if extending, power up the under-powered intersections first.",
    },
    "power_analysis": {
        "title": "Statistical Power Analysis",
        "concept": "The statistical power to detect a meaningful effect within each intersection.",
        "reading": "Under-powered intersections (power below the ~0.8 target) can hide real harms; each shows its required sample size.",
        "action": "Collect more data for the under-powered intersections before drawing conclusions.",
    },
    "causal_decomposition": {
        "title": "Causal Decomposition",
        "concept": "Mediation analysis splitting a disparity into direct and indirect (mediated) effects, validated by the Baron-Kenny steps.",
        "reading": "The proportion mediated shows how much of the gap runs through the mediator; fewer than 4/4 steps weakens the causal claim, and temporal stability checks robustness.",
        "action": "If mediation is strong and stable, intervene on the mediator; otherwise gather more evidence.",
    },
    "workflow_overview": {
        "title": "Workflow Integration Overview",
        "concept": "How vfairness plugs into the ML lifecycle: experiment tracking, version-control hooks and CI fixtures.",
        "reading": "This view is informational - it maps where fairness checks attach across development, not a pass/fail result.",
        "action": "Wire the shown integrations into your pipeline to automate fairness checks.",
    },
    "hierarchical_gate": {
        "title": "Hierarchical Fairness Gate",
        "concept": "A fairness gate evaluated at three levels: overall, per single attribute, and intersectional.",
        "reading": "The gate is approved only if all levels pass; small-sample warnings flag intersections with too little data to judge.",
        "action": "If blocked, address the failing level; if warned, collect more data for those intersections.",
    },
    "report_card": {
        "title": "Fairness Report Card",
        "concept": "A pull-request-ready summary of a model's fairness evaluation: gate verdict, blocking metrics and values versus baseline.",
        "reading": "Approved means the model may ship; blocking reasons list what must be fixed first.",
        "action": "Resolve every blocking reason before merging or registering the model.",
    },
    "robustness_testing": {
        "title": "Robustness Testing",
        "concept": "Whether fairness conclusions hold under permutation tests, input perturbations and fine-grained subgroup analysis.",
        "reading": "The 0-1 score bands as pass (>=0.8), marginal (>=0.5) or fail; flagged subgroups and gerrymandering signals mark fragile conclusions.",
        "action": "Investigate flagged subgroups; a fail means the fairness result is not robust.",
    },
    "ranking_fairness": {
        "title": "Ranking Fairness",
        "concept": "Whether demographic groups receive equitable exposure in ranked results (exposure parity, NDKL, attention fairness).",
        "reading": "Each metric passes when it stays under its threshold; failing metrics mean some groups are systematically under-exposed.",
        "action": "Re-rank or apply exposure constraints for the failing metrics.",
    },
    "data_validation": {
        "title": "Data Validation",
        "concept": "Automated pre-training data-quality checks, categorised by severity and linked to affected groups.",
        "reading": "Critical and error issues block training, warnings need review, info is advisory; the verdict is pass only if no critical issues remain.",
        "action": "Resolve critical and error issues before fitting the model.",
    },
    "auto_discovery": {
        "title": "Protected-Attribute Auto-Discovery",
        "concept": "An automated scan for likely protected attributes and their intersections, flagging fairness violations.",
        "reading": "Each candidate attribute and violation is rated by severity; high-severity violations mark the biggest disparities found.",
        "action": "Confirm the discovered attributes and prioritise the high-severity violations.",
    },
    "regression_fairness": {
        "title": "Regression Fairness",
        "concept": "Whether a regression model produces equitable error distributions across groups (MAE, RMSE, R-squared parity and effect sizes).",
        "reading": "Each disparity metric passes under its threshold; Cohen's d effect sizes show the practical magnitude of any error gap.",
        "action": "Address the failing metrics and the largest error-gap effect sizes.",
    },
    "reporting_dashboard": {
        "title": "Fairness Reporting Dashboard",
        "concept": "A multi-tier fairness health dashboard (executive, operational, technical) with a health score, metric grid and alerts.",
        "reading": "The health score bands as green, yellow or red; breaches and active alerts show where fairness is slipping.",
        "action": "Escalate red status and critical alerts; work the metric breaches down.",
    },
}

_GENERIC_META = {
    "title": "vfairness Visualization",
    "concept": "A vfairness fairness analysis result.",
    "reading": "Colour marks the four semantic tones: green (pass), terracotta (warning or failure), ocean (informational) and slate (neutral).",
    "action": "Review the highlighted values against their thresholds.",
}


# ── Data-specific finding extractors (finding text + severity) ──────────────
# Each returns (finding: str|None, severity: str). Severity is derived from the
# same values the chart's badge uses, so the two never disagree. All are called
# through _finding_for() which swallows any exception.


def _g(d: Any, key: str, default: Any = None) -> Any:
    """dict.get that tolerates non-dict input."""
    return d.get(key, default) if isinstance(d, dict) else default


def _f(v: Any, spec: str = ".3f") -> str:
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return "?"


def _count_true(items: Any, key: str) -> int:
    return sum(1 for it in (items or []) if isinstance(it, dict) and it.get(key))


def _band_idx(x: float, boundaries: Tuple[float, ...]) -> int:
    return sum(1 for b in boundaries if x >= b)


# ── could-not-check, the third state ────────────────────────────────────────
# The finders below read the same data dict the badge is built from, so when the
# badge is suppressed the prose must be suppressed with it. "5 of 5 fairness
# metric(s) passed" written underneath a NOT ASSESSABLE banner is the sentence a
# reader believes, because it is the only full sentence on the chart.
_COULD_NOT_CHECK = "COULD NOT CHECK"

# Severity for a could-not-check finding. Deliberately not "info": an
# unsupported fairness claim is not benign news, and not "high" either, because
# nothing was measured to be wrong.
_UNKNOWN_SEV = "medium"

# The static per-chart `action` tells the reader to fix failing metrics. That is
# the wrong instruction for a chart that measured nothing, so build_explanation
# swaps in this one whenever the data dict says the run was not assessable.
_NOT_ASSESSABLE_ACTION = (
    "Give every group enough samples (or lower min_group_size) and re-run; "
    "as it stands this chart neither passes nor fails, it certifies nothing."
)

# The safety net used by build_explanation for a chart whose data dict says
# not-assessable while its finder still produced a result-shaped sentence.
# Deliberately generic: it is reached only when the per-chart finder has NOT
# written its own could-not-check clause, and the adapter's own
# `not_assessable_reason` is appended after it by _could_not_check.
_GENERIC_NOT_ASSESSABLE = (
    "nothing on this chart was assessed, so it neither passes nor fails and certifies nothing."
)

# The action for a chart whose FINDING came back could-not-check while its data
# dict carries no `not_assessable` flag. That combination is real: an
# unrecognised decision string leaves adapters_experimentation with
# `decision_known=True` (something WAS supplied) while
# _fr_experiment_recommendation refuses to read it out, and the curated action
# then said "Follow the recommendation" underneath a sentence stating that no
# recommendation was made. It is deliberately not _NOT_ASSESSABLE_ACTION: that
# one names group sample sizes, which is the wrong remedy for a chart whose
# problem is a value nobody graded.
_COULD_NOT_CHECK_ACTION = (
    "Establish what this chart could not check and re-run it; as it stands this "
    "result is neither a pass nor a failure and certifies nothing."
)


def _is_could_not_check(finding: Any) -> bool:
    """True when a finding already leads with the could-not-check marker.

    This is the test the not-assessable net uses to decide whether to REPLACE a
    finder's sentence, so it stays a leading-marker test: several finders put the
    marker mid-sentence precisely so their own zeros are not leaked onto an empty
    canvas (see the note in _fr_mediation). For the ACTION line, which asks a
    different question, see :func:`_withholds_the_verdict`.
    """
    return str(finding or "").strip().upper().startswith(_COULD_NOT_CHECK)


# A finding whose VERDICT is a could-not-check, written as "<subject>: COULD NOT
# CHECK, ...". The label before the colon names what was not established; it
# carries no measurement of its own, so the whole sentence withholds the verdict
# just as a leading marker does.
_LABEL_THEN_MARKER = re.compile(
    r"^[A-Za-z][A-Za-z '/-]{0,48}:\s*" + re.escape(_COULD_NOT_CHECK),
    re.IGNORECASE,
)


def _withholds_the_verdict(finding: Any) -> bool:
    """True when the finding states no verdict, so the curated action cannot stand.

    BGL5 (2026-09-27). ``_is_could_not_check`` is a STARTSWITH test and four
    finders deliberately write the subject first, so the action swap in
    ``build_explanation`` never fired for them. MEASURED end to end through
    ``FairnessMonitor(config=FairnessMonitorConfig(metrics_to_track=
    ['equalized_odds'])).update_and_check(frame)`` on a frame with a prediction
    column and no label column, which gives ``metrics={}``, ``alerts={}``,
    ``any_alert=None`` and real group positive rates:

      finding        Alert status: COULD NOT CHECK, this window recorded no alert
                     verdict, so nothing on it fired and nothing on it is clear,
                     across 0 fairness metric(s) on 600 samples.
      recommendation Triage the alerting metrics; if several alert at once, treat
                     it as an incident.
      severity       medium

    Zero metrics were monitored, so there are no alerting metrics to triage, and
    ``test_empty_input_action_does_not_imply_a_measurement`` exists to forbid
    exactly that ("on a run that measured nothing that instruction is not merely
    useless, it asserts that a measurement happened"). It passed only because the
    EMPTY fixture trips the chart-level not_assessable flag, which the group rates
    keep from firing here. Coverage of ``build_explanation`` from its own four
    named tests showed lines 2499-2501, the ``elif _is_could_not_check(finding)``
    branch, MISSING.

    A MID-SENTENCE marker is not this state and must not match: a chart with four
    satisfied Baron-Kenny steps and one missing gauge, or a measured ECE beside an
    unmeasured disparity, still has a verdict its curated action fits, and
    ``_fr_bias_audit`` leads with a critical-issue count for the express reason
    that the generic action would tell a reader to re-run an audit that DID
    return findings. Those keep the curated action, which is why this reads the
    text before the marker: only a bare subject label counts.
    """
    text = str(finding or "").strip()
    if not text:
        return False
    return bool(text.upper().startswith(_COULD_NOT_CHECK) or _LABEL_THEN_MARKER.match(text))


# Leads the <desc> of any chart whose data dict is a built-in demonstration
# fixture (`is_example`). Applied in build_explanation, see the note there.
_EXAMPLE_PREFIX = "EXAMPLE ONLY, synthetic demonstration data, not a real evaluation. "


def _not_assessable(d: Any) -> bool:
    """True when the chart's own data dict says the run could not be assessed."""
    return bool(_g(d, "not_assessable"))


def _excluded_clause(d: Any) -> str:
    """' Excluded: a (n=3), b (n=1).' when groups were dropped, else ''."""
    groups = _g(d, "excluded_groups") or []
    names = [str(g) for g in groups if g]
    return f" Excluded: {', '.join(names)}." if names else ""


def _could_not_check(d: Any, what: str) -> Tuple[str, str]:
    """Build a (finding, severity) pair that never reads as a pass or a fail."""
    reason = str(_g(d, "not_assessable_reason") or "").strip()
    text = f"{_COULD_NOT_CHECK}: {what}"
    if reason:
        text += f" {reason}"
    # The reason already names the dropped groups when the engine supplied one;
    # only add the clause when it did not, so the sentence is not doubled.
    if "Excluded:" not in text:
        text += _excluded_clause(d)
    return (text, _UNKNOWN_SEV)


def _pass_tail(p: Any, t: Any, fails: list, ungraded: int = 0) -> str:
    """The clause that closes a "p of t metrics pass" sentence.

    "; all pass" is an all-clear over the WHOLE chart, so it may only be
    written when the counts say every metric passed. It used to be written
    whenever the badge list named no failure, and that list is TRUNCATED before
    it reaches here (adapters_ranking keeps badges[:6], adapters_regression
    badges[:4]) while the counts are taken over all of them. Seven ranking
    metrics with the only failure in seventh place produced the accessible
    <desc> "Ranking fairness: 6/7 metrics pass across 1 groups; all pass"
    beside an UNFAIR headline: the canvas graded the breach, and the sentence
    that is all a screen-reader user gets called it a clean sheet.

    A count that does not add up is reported as the shortfall it is, without
    naming metrics this list cannot see.

    *ungraded* closes the same hole from the other side, which is part (b) of the
    headline rule. ``p == t`` is a clean sheet over the GRADED subset, and the
    counts add up perfectly whenever a metric was never compared to a threshold
    at all, because such a metric is in neither number. "; all pass" written
    beside that is an all-clear over rows nobody checked, so the words are
    withheld and the caller's ungraded clause carries the sentence instead.
    """
    if fails:
        return "; failing: " + ", ".join(fails)
    try:
        n_pass, n_total = int(p), int(t)
    except (TypeError, ValueError):
        return ""
    if n_total and n_pass == n_total:
        return "; every graded metric passes" if ungraded > 0 else "; all pass"
    if n_total and n_pass < n_total:
        return f"; {n_total - n_pass} metric(s) do not pass and are not named here"
    return ""


# ── Partial runs: the canvas grades a subset, so the sentence must say so ───
# THE HEADLINE RULE, decided 2026-08-28, because two adapters had drifted into
# two different answers for the same situation.
#
# Earlier waves closed the FULLY empty case in these finders: a data dict
# carrying `not_assessable` now yields a could-not-check clause, enforced for
# every chart by the sweep in tests/test_explain_matches_canvas.py. The PARTIAL
# case was missed, and it is the commoner one. A chart that graded three of five
# rows wrote a sentence shaped exactly like a chart that graded all five, while
# the canvas beside it had already learned to say otherwise (the monitoring
# dashboard separates `n_checked` from `n_metrics`, the discovery scanner counts
# `n_ungraded`, the power analysis counts `n_ungraded`, the radar names the
# metrics it could not plot). Whichever surface is more confident is the one a
# reader believes, and the <desc> is the whole artifact for a screen-reader
# user. The four parts:
#
#   a) grade and band over the GRADED subset, so a partly measured run still
#      gives a useful verdict on what WAS measured. Do not withhold everything
#      because one row is ungraded.
#   b) never write an unqualified all-clear while anything is ungraded.
#   c) state the ungraded count in the FINDING. Naming it only in the chart's
#      subtitle leaves the accessible layer the more confident of the two.
#   d) an ungraded row never enters a numerator or a denominator that implies
#      it was measured.


def _n(d: Any, key: str, default: int = 0) -> int:
    """A non-negative count from the data dict; *default* for anything else.

    ``True`` is an ``int`` in Python and would count as 1, so bools are
    refused rather than silently counted as a row.
    """
    v = _g(d, key)
    if v is None or isinstance(v, bool):
        return default
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return default


# The marker every partial-run clause ends with. One fixed phrase, so the guard
# in tests/test_explain_partial_runs.py can check the rule rather than checking
# ten separate wordings, and so a reader meets the same sentence on every chart.
_NOT_COVERED = "and are not covered by this finding."


def _ungraded_tail(n: int, what: str) -> str:
    """Part (c): the clause naming the rows the chart withheld a grade from.

    *what* is a verb phrase without its subject count and without a full stop,
    e.g. ``"metric(s) were never compared to a threshold"``.

    Returns "" for a fully graded run, so a healthy chart's sentence stays
    byte-identical to what it was before this rule existed. That is deliberate:
    the over-correction control for this wave is that healthy input renders
    unchanged, and an empty string is how that is guaranteed rather than hoped.
    """
    if n <= 0:
        return ""
    return f" {n} {what} {_NOT_COVERED}"


def _floor_sev(severity: Any, floor: str = _UNKNOWN_SEV) -> str:
    """Raise a severity to at least *floor*, never lower it.

    Part (b) in its machine-readable form. The prose clause on its own is not
    enough: ``ChartExplanation.metadata()["severity"]`` is what a report
    pipeline, an assistant or a downstream dashboard reads INSTEAD of the
    sentence, and "info" there is the same unqualified all-clear the sentence
    is forbidden to make. A measured breach keeps its higher severity, because
    an ungraded row elsewhere does not make a real finding less true.

    The floor is ``_UNKNOWN_SEV`` for the reason given at that constant: an
    unsupported fairness claim is not benign news, and it is not "high" either,
    because nothing was measured to be wrong. A partial run carries exactly
    that unresolved unknown, so it bands the same way.
    """
    s, f = _norm_sev(severity), _norm_sev(floor)
    return s if _SEVERITY_ORDER[s] >= _SEVERITY_ORDER[f] else f


def _f_band(v: Any, boundaries: Tuple[float, ...], prec: int = 2, max_prec: int = 6) -> str:
    """Format *v* so the displayed number stays in the same band as *v*.

    The band label printed next to a value is derived from the RAW value
    (by the badge), so rounding for display must never carry the number
    across a band boundary: 0.2499 shown as '0.25' beside a MINIMAL badge
    (band ends below 0.25) reads as a contradiction. Precision is raised
    until the displayed number parses back into the raw value's band.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "?"
    raw_band = _band_idx(x, boundaries)
    for p in range(prec, max_prec + 1):
        s = format(x, f".{p}f")
        if _band_idx(float(s), boundaries) == raw_band:
            return s
    return format(x, f".{max_prec}f")


def _fr_bias_audit(d):
    # The canvas qualifies a partial audit on its own subtitle line ("Only 2 of
    # 4 audit modules ran"), so the description has to carry the same clause or
    # it ends up the MORE confident of the two: "score 0.00 (MINIMAL)" with
    # nothing said about the modules that never looked. Absent for any data
    # dict that does not report coverage, which leaves the sentence unchanged.
    coverage = str(_g(d, "coverage", "") or "")
    tail = ""
    if coverage == "partial":
        tail = (
            f" Only {_g(d, 'n_modules_run', '?')} of {_g(d, 'n_modules', '?')} audit modules "
            "ran, so this score leaves the rest of the audit unmeasured."
        )
    # The PER-FINDING partial, which is a different population from the module
    # coverage above. A finding that reported no severity is in no module
    # average and in no badge (adapters._module_score returns it as an ungraded
    # count), and the canvas states it in the headline band. This sentence
    # reported the score and the critical count with nothing said about the
    # findings behind neither, which is the same silence the coverage clause
    # was written to close one level up.
    ungraded = _n(d, "n_ungraded_findings")
    tail += _ungraded_tail(ungraded, "finding(s) reported no severity and are in no module score")
    # Bands match engine._risk_label: MINIMAL < 0.25 <= LOW < 0.50 <= MEDIUM
    # < 0.75 <= HIGH. _f_band keeps the shown score inside the badge's band.
    #
    # THE UNKNOWN LABEL, and it is a state this chart really produces.
    # `engine._risk_label` answers "N/A" for a score that is None or NaN
    # (`_bands.risk_band`, whose own docstring says an uncomputable score must
    # not read as a pass), and adapters.bias_audit_to_svg withholds the BADGE
    # for exactly that case through `score_assessable` while deliberately
    # leaving `not_assessable` False, because the audit's findings are real and
    # the <desc> must keep reporting them. So the canvas prints NOT ASSESSABLE
    # where the percentage would be, and this map sent the same "N/A" through
    # `.get(..., "info")`: an audit carrying a HIGH critical issue and no
    # aggregate score was announced at the all-clear severity, and the score
    # itself came out of `_f_band(None, ...)` as a bare question mark, which is
    # the withheld-value shape rather than a statement that there is none.
    #
    # `score_assessable` is the adapter's own flag and is preferred; a dict that
    # predates it falls back to the two signals the flag is derived from, so a
    # caller who never sent either is unchanged.
    n_critical = _g(d, "n_critical", 0)
    label = str(_g(d, "overall_label", "")).upper()
    score_known = _g(d, "score_assessable")
    if score_known is None:
        score_known = _g(d, "overall_score") is not None and label != "N/A"
    sev = {
        "MINIMAL": "info",
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
        "CRITICAL": "critical",
    }.get(label, "info")
    if not score_known:
        # Part (a): grade over what WAS measured. A critical issue is a real
        # finding and outranks the missing aggregate; with no issue either, the
        # unresolved unknown is all that is left, and it is not benign news.
        sev = "high" if n_critical else _UNKNOWN_SEV
    if ungraded:
        sev = _floor_sev(sev)
    if not score_known:
        # The critical count leads, so the sentence is not read as a
        # could-not-check over the whole audit: `_is_could_not_check` matches on
        # the LEADING marker, and swapping in the generic action here would tell
        # a reader to re-run an audit that did return findings, instead of to
        # address them.
        return (
            f"{n_critical} critical issue(s) found. {_COULD_NOT_CHECK}: this audit reports "
            f"no overall bias-risk score, so no aggregate risk was computed and none is "
            f"claimed here.{tail}",
            sev,
        )
    return (
        f"Overall bias-risk score {_f_band(_g(d, 'overall_score'), (0.25, 0.50, 0.75))} "
        f"({_g(d, 'overall_label', '')}); {n_critical} critical issue(s).{tail}",
        sev,
    )


def _fr_calibration_report(d):
    ok = _g(d, "is_well_calibrated")
    sev = (
        "info"
        if ok and not _g(d, "has_disparity")
        else ("high" if _g(d, "has_disparity") else "medium")
    )
    # THREE STATES. `'well calibrated' if ok else 'miscalibrated'` is a two-way
    # branch over an Optional[bool]: None is falsy, so a verdict the producer
    # explicitly WITHHELD was published as the word "miscalibrated". Measured
    # 2026-09-30 on a report with is_well_calibrated=None and a finite ECE of
    # 0.047: "ECE 0.047 (miscalibrated)" at severity MEDIUM. The analyzer's own
    # text summary renders the same None as "Not assessable (not measured)".
    #
    # ONE mechanism, the value itself. A separate `*_known` flag beside it was
    # tried and withdrawn: sabotaging the adapter's three-state value then left
    # every pin green, because the flag carried the fact on its own, and two
    # mechanisms for one fact is how they drift. A data dict that OMITS the key
    # takes this branch too, which is correct: an absent verdict is not a
    # negative one, and the two-way branch used to publish it as "miscalibrated".
    if ok is None:
        cal_word = f"{_COULD_NOT_CHECK.lower()}, no calibration verdict was reported"
        sev = _floor_sev(sev)
    else:
        cal_word = "well calibrated" if ok else "miscalibrated"
    lead = f"ECE {_f(_g(d, 'ece'))} ({cal_word}); "
    # The PARTIAL run, and the literal malformed sentence it produced. The
    # adapter sends `ece_disparity=None` whenever fewer than two groups reported
    # a calibration error (adapters.calibration_report_to_svg:
    # `disparity_assessable`), the canvas prints its own reason for that, and
    # `_f(None)` printed a bare question mark: "ECE disparity ? across groups."
    # A question mark is not a measurement, it is not a state a reader can act
    # on, and it left the overall verdict ("well calibrated") reading at
    # severity INFO as though the groups had been compared and agreed.
    disparity_known = _g(d, "disparity_assessable")
    if disparity_known is None:
        disparity_known = _g(d, "ece_disparity") is not None
    if not disparity_known:
        reason = str(_g(d, "disparity_reason") or "").strip() or (
            "fewer than two groups reported a calibration error."
        )
        return (
            f"{lead}{_COULD_NOT_CHECK}: no ECE disparity was measured across groups, {reason}",
            _floor_sev(sev, _UNKNOWN_SEV),
        )
    # The disparity WAS measured, but only over the groups that reported an
    # error. The ones that did not are named rather than left to be assumed
    # covered by "across groups".
    unmeasured = 0
    if _g(d, "n_measured_groups") is not None:
        unmeasured = max(0, _n(d, "n_groups") - _n(d, "n_measured_groups"))
    tail = _ungraded_tail(unmeasured, "group(s) reported no calibration error")
    # The third population: a key in group_metrics that is the absence of a
    # group. It cannot be named (there is no name), so it is counted; `str()` on
    # it minted a row labelled "None" with a real-looking ECE and a GOOD badge.
    missing_keys = _n(d, "n_missing_group_keys")
    tail += _ungraded_tail(missing_keys, "group key(s) carried no group label")
    if unmeasured or missing_keys:
        sev = _floor_sev(sev)
    return (f"{lead}ECE disparity {_f(_g(d, 'ece_disparity'))} across groups.{tail}", sev)


def _fr_fairness_report(d):
    p, f, s = _g(d, "n_passed", 0), _g(d, "n_failed", 0), _g(d, "fairness_score")
    # Could-not-check only when there is genuinely no verdict. A data dict that
    # simply omits the score while carrying pass/fail counts HAS been assessed,
    # and the pre-existing "unknown score + failures is high" rule still applies
    # to it (test_fairness_report_missing_score_with_failures_stays_high).
    if _not_assessable(d) or (s is None and not p and not f):
        return _could_not_check(
            d,
            "this dashboard has no fairness score because no metric was verified "
            "against a threshold.",
        )
    # Explicit None checks: `s or 100` treated a legitimate score of 0 as
    # missing, so "Fairness score 0/100" reported severity "info".
    try:
        s_num = None if s is None else float(s)
    except (TypeError, ValueError):
        s_num = None
    low = s_num is not None and s_num < 40
    mid = s_num is not None and s_num < 70
    # A known score below 40 is high on its own; an unknown score with failed
    # metrics stays high (the pre-fix behaviour for missing scores).
    sev = "high" if low or (f and s_num is None) else ("medium" if f or mid else "info")
    # The PARTIAL run. `n_unknown` counts the cards the adapter put in its third
    # state (adapters._metric_cards: `COULD_NOT_CHECK`), the canvas draws each
    # of them in slate rather than green, and the score is computed over the
    # graded ones only. "Fairness score 92/100 with 3 metric(s) fair, 0 unfair"
    # said none of that: it is the sentence a fully checked dashboard writes.
    unknown = _n(d, "n_unknown")
    tail = _ungraded_tail(unknown, "metric(s) were never compared to a threshold")
    # The other half of the same dashboard. The metric CARDS are only one of the
    # two populations on it: the group bars are the other, and a group that
    # reported no rate is not plotted at all. The adapter names them on the
    # canvas ("Not plotted, no rate reported: ... These groups were not
    # measured.") and counts them in `n_unmeasured_groups`, and a group that
    # silently vanishes from a fairness dashboard leaves the reader believing
    # every stratum was looked at, which is the omission the invented 0.0% bar
    # used to be.
    unmeasured_groups = _n(d, "n_unmeasured_groups")
    tail += _ungraded_tail(unmeasured_groups, "group(s) reported no rate and are not plotted")
    # THE THIRD POPULATION: a key that carried no group LABEL at all. It cannot
    # be named (there is no name; that is the point), and `str(key)` on it MINTS
    # one, so the adapter now drops it and counts it. Measured 2026-09-30 with
    # ``{None: {"positive_rate": 0.4, "size": 10}, "b": {...}}``: the dashboard
    # drew a bar labelled "None" at 40% next to "n=10", and with pd.NA one
    # labelled "<NA>". Dropping it without saying so would leave the reader
    # believing every stratum in the data is on the panel, which is the same
    # omission as the invented bar; the canvas says it and this is the
    # accessible half.
    missing_keys = _n(d, "n_missing_group_keys")
    tail += _ungraded_tail(missing_keys, "group key(s) carried no group label")
    if unknown or unmeasured_groups or missing_keys:
        sev = _floor_sev(sev)
    return (
        f"Fairness score {_g(d, 'fairness_score', '?')}/100 with {p} metric(s) fair, "
        f"{f} unfair.{tail}",
        sev,
    )


def _fr_cicd(d):
    passed = _g(d, "all_passed")
    gate = _g(d, "gate", {})
    val = _g(d, "validation", {})
    # The PARTIAL run, part (d) in its BOOLEAN form. `all_passed` is an AND over
    # every test row, and a check the suite SKIPPED carries `passed=None`, which
    # is falsy, so ONE ungraded check dragged the whole conjunction to False and
    # this sentence read "Pipeline failed" at severity HIGH. The canvas had
    # already been corrected the other way: with no graded failure its tile
    # stays green and reads "PASS / N passed, M failed, K not run"
    # (cicd_pipeline.svg withholds the word ALL rather than inventing a
    # failure). So a screen-reader user was told the pipeline failed while a
    # sighted reader saw it pass, and a fabricated breach is not the safer
    # error: it sends someone after a failure that was never measured, and when
    # they cannot find it the report is what they stop believing.
    #
    # The counts are read from the SAME keys the template branches on, and the
    # rows are the fallback so a dict predating the split still counts right.
    tests = _g(d, "tests", []) or []
    ungraded = _n(d, "n_tests_ungraded", default=sum(1 for t in tests if _g(t, "passed") is None))
    graded_fail = sum(1 for t in tests if _g(t, "passed") is False)
    # A real breach anywhere still wins: an ungraded row elsewhere does not make
    # a measured failure less true.
    breached = bool(graded_fail) or _g(gate, "approved") is False or _g(val, "passed") is False
    if passed:
        verdict = "Pipeline passed"
        sev = "info"
    elif breached:
        verdict = "Pipeline failed"
        sev = "critical" if _g(gate, "approved") is False else "high"
    elif ungraded:
        # Part (a): a useful verdict over the graded subset, and part (b): never
        # the unqualified "passed" while a check did not run.
        verdict = "Pipeline passed every check that ran"
        sev = _UNKNOWN_SEV
    else:
        verdict = "Pipeline failed"
        sev = "high"
    tail = _ungraded_tail(ungraded, "fairness check(s) did not run")
    if ungraded:
        sev = _floor_sev(sev)
    return (
        f"{verdict}; gate {_g(gate, 'status', 'N/A')}, "
        f"{_g(val, 'n_errors', 0)} validation error(s).{tail}",
        sev,
    )


def _fr_radar(d):
    st = _g(d, "status_text", "?")
    if _not_assessable(d):
        return _could_not_check(d, "no metric could be plotted, so there is no polygon to read.")
    # The PARTIAL run. A metric that could not be placed on the axis is not on
    # the chart at all, and the canvas already withdraws its green badge for one
    # (adapters_fairness.radar_chart_to_svg swaps the Fair badge for NOT
    # ASSESSABLE when a metric has no known better-direction, and names the
    # unplotted metrics in the subtitle). The <desc> said only "Overall fairness
    # status: Fair."
    #
    # The default severity moves off "info" for the same reason: NOT ASSESSABLE
    # is a real value of `status_text` and it missed this map, so the withheld
    # verdict was read out at the all-clear severity.
    unmeasured = _n(d, "n_unmeasured")
    sev = {"Fair": "info", "Marginal": "medium", "Unfair": "high"}.get(st, _UNKNOWN_SEV)
    tail = _ungraded_tail(unmeasured, "metric(s) could not be measured and were not plotted")
    if unmeasured:
        sev = _floor_sev(sev)
    return (f"Overall fairness status: {st}.{tail}", sev)


def _fr_disparity_heatmap(d):
    if _not_assessable(d):
        return _could_not_check(
            d, "no disparity was measured across the groups shown on this grid."
        )
    sev = {"✓": "info", "⚠": "medium", "✗": "high"}.get(_g(d, "summary_icon"), "info")
    # The PARTIAL run, and it is the count itself that was wrong. `n_rows` and
    # `n_cols` are the rows and the columns the grid DRAWS, and a cell whose
    # value is None is drawn "N/A" in slate: the group is on the canvas and
    # nothing about it was measured. "across 3 group(s) and 4 metric(s)" beside
    # a grid on which one of those groups is blank claims a coverage a sighted
    # reader can see it does not have, and the sentence is the whole artifact
    # for everyone else.
    #
    # Counted from the SAME cells the template draws, cell by cell, exactly as
    # _fr_corr_heatmap counts its uncomputed pairs, so the two surfaces cannot
    # drift apart. Only when the rows are actually carried: a dict that predates
    # them keeps the numbers it always printed, which is the over-correction
    # control in code.
    rows = _g(d, "rows", []) or []
    n_rows, n_cols = _g(d, "n_rows", "?"), _g(d, "n_cols", "?")
    tail = ""
    if rows:
        measured_cols = set()
        measured_rows = 0
        width = 0
        for r in rows:
            cells = _g(r, "cells", []) or []
            width = max(width, len(cells))
            hit = False
            for i, c in enumerate(cells):
                if _g(c, "value") is not None:
                    hit = True
                    measured_cols.add(i)
            measured_rows += 1 if hit else 0
        blank_rows = max(0, len(rows) - measured_rows)
        blank_cols = max(0, width - len(measured_cols))
        n_rows, n_cols = measured_rows, len(measured_cols)
        tail = _ungraded_tail(blank_rows, "group(s) reported no value on any metric")
        tail += _ungraded_tail(blank_cols, "metric(s) reported no value for any group")
        if blank_rows or blank_cols:
            sev = _floor_sev(sev)
    return (
        f"{_g(d, 'summary_text', '?')} across {n_rows} group(s) and {n_cols} metric(s).{tail}",
        sev,
    )


def _fr_metrics_bar(d):
    p, t = _g(d, "n_passed", 0), _g(d, "n_total", 0)
    u = _g(d, "n_unknown", 0) or 0
    # `t == 0` used to fall into the "info" branch and print "0 of 0 fairness
    # metric(s) passed", which reads as a clean sheet for a run that checked
    # nothing. Nothing checked is could-not-check, not good news.
    if _not_assessable(d) or not t:
        return _could_not_check(d, "not one metric on this chart was verified against a threshold.")
    # This finder already named the unchecked metrics, which is why its wording
    # became the shared one. What it still did was grade a partial run "info":
    # p == t over the CHECKED subset is a clean sheet for what was measured, and
    # the machine-readable severity is the surface that says so without the
    # qualifying clause attached. See _floor_sev.
    tail = _ungraded_tail(u, "metric(s) were never compared to a threshold")
    sev = "info" if p == t else ("high" if p == 0 else "medium")
    if u:
        sev = _floor_sev(sev)
    return (f"{p} of {t} checked fairness metric(s) passed.{tail}", sev)


def _fr_group_comparison(d):
    if not (_g(d, "bars") or []):
        # A chart with nothing plotted on it. The severity stays "info" (pinned
        # by test_fr_group_comparison_none_disparity_is_info: an empty chart must
        # never claim a 'High' disparity), but the wording now says outright that
        # nothing was compared rather than merely reporting an absence.
        return (f"{_COULD_NOT_CHECK}: No group data available to compare.", "info")
    if _not_assessable(d):
        return _could_not_check(
            d, "fewer than two comparable groups are shown, so no gap was measured."
        )
    if _g(d, "max_disparity") is None:
        return ("No group data available to compare.", "info")
    lbl = _g(d, "disparity_label", "?")
    # The PARTIAL run. The gap is banded over the groups that WERE measured
    # (part (a)) and its operands are named from those groups only, so the
    # numbers are honest. What the sentence did not say is who was left out: a
    # group that reported no rate is not on the plot and not in the gap, the
    # adapter counts it (`n_ungraded_groups`) and names it in the subtitle, and
    # "Low disparity (0.050) in Rate" beside that silence reads as a verdict
    # over the whole attribute at severity INFO.
    ungraded = _n(d, "n_ungraded_groups")
    sev = {"Low": "info", "Moderate": "medium", "High": "high"}.get(lbl, "info")
    tail = _ungraded_tail(
        ungraded, f"group(s) reported no {str(_g(d, 'metric_label', '')).lower() or 'rate'}"
    )
    if ungraded:
        sev = _floor_sev(sev)
    return (
        f"{lbl} disparity ({_f(_g(d, 'max_disparity'))}) in {_g(d, 'metric_label', '')}; "
        f"worst {_g(d, 'worst_group', '?')} vs best {_g(d, 'best_group', '?')}.{tail}",
        sev,
    )


def _fr_effect_sizes(d):
    it = _g(d, "max_effect_interpretation", "?")
    if _not_assessable(d) or _g(d, "max_effect") is None:
        return _could_not_check(d, "no effect size could be computed on this data.")
    # Bands match adapters_fairness._effect_interpretation: Negligible < 0.2
    # <= Small < 0.5 <= Medium < 0.8 <= Large. _f_band keeps the shown |d|
    # inside the interpretation's band (0.1999 must not display as 0.20).
    #
    # The PARTIAL run: "Largest effect |d|=0.11 (Negligible)" is a maximum, and
    # a maximum taken over four of six metrics is not the maximum. The adapter
    # counts the ones it could not compute (`n_unmeasured`) and names them in
    # the subtitle; the sentence beside it claimed the whole set.
    unmeasured = _n(d, "n_unmeasured")
    sev = {"Negligible": "info", "Small": "low", "Medium": "medium", "Large": "high"}.get(
        it, "info"
    )
    tail = _ungraded_tail(unmeasured, "metric(s) had no computable effect size")
    if unmeasured:
        sev = _floor_sev(sev)
    return (
        f"Largest effect |d|={_f_band(_g(d, 'max_effect'), (0.2, 0.5, 0.8))} ({it}); "
        f"average |d|={_f(_g(d, 'avg_effect'), '.2f')}.{tail}",
        sev,
    )


def _fr_confidence_intervals(d):
    n = _g(d, "n_significant", 0) or 0
    # The significance count is a claim ABOUT confidence intervals. When the
    # adapter fabricated a value +/- 0.02 band, this line turned the fabricated
    # count into confident prose ("4 of 5 metric(s) show a significant
    # disparity") on data the library's own bootstrap called NOT significant.
    # With no computed interval there is no count to report.
    n_ci = _g(d, "n_ci_computed")
    if _not_assessable(d) or n_ci == 0:
        return _could_not_check(
            d,
            "no confidence interval was computed for any metric here, so no "
            "significance is claimed.",
        )
    # The PARTIAL run. The denominator is already the computed-interval
    # population (part (a) of the headline rule, and the SIGNIFICANT tile counts
    # the same way), but a metric whose value came out undefined never reached
    # the plot at all. The canvas names those in its subtitle; this sentence
    # reported "0 of 3 ... show a significant disparity" at severity "info",
    # which is a measured all-clear over a set that was not fully measured.
    unmeasured = _n(d, "n_unmeasured")
    sev = "info" if n == 0 else "high"
    tail = _ungraded_tail(unmeasured, "metric(s) came out undefined, so no interval was drawn")
    if unmeasured:
        sev = _floor_sev(sev)
    return (
        f"{n} of {_g(d, 'n_total', '?')} metric(s) with a computed interval show a "
        f"significant disparity.{tail}",
        sev,
    )


def _fr_reliability(d):
    # The canvas withholds its verdict when nothing could be binned
    # (adapters_calibration sets `not_assessable` and the subtitle reads NOT
    # ASSESSABLE), but this finder read `verdict_text`, `ece` and `mce` straight
    # out of the same dict and their ADAPTER DEFAULTS are 'WELL CALIBRATED' and
    # 0. So the paragraph beside the withheld banner, and the accessible <desc>
    # that is all a screen-reader user gets, read "Model is WELL CALIBRATED
    # (ECE 0.000, MCE 0.000)": a perfect-calibration certificate for a run that
    # binned nothing, and the more confident of the two states is the one a
    # reader believes. Both derive from ONE signal now.
    if _not_assessable(d):
        return _could_not_check(
            d, "this diagram does not establish whether the model is calibrated."
        )
    e = _g(d, "ece", 0) or 0
    return (
        f"Model is {_g(d, 'verdict_text', 'N/A')} (ECE {_f(e)}, MCE {_f(_g(d, 'mce', 0))}).",
        "low" if e < 0.05 else ("medium" if e < 0.1 else "high"),
    )


def _fr_group_calibration(d):
    mx = _g(d, "max_ece", 0) or 0
    sev = "high" if mx >= 0.1 else ("medium" if _g(d, "has_disparity") or mx >= 0.05 else "low")
    # The PARTIAL run. `n_groups` counts every group SUPPLIED, and the spread is
    # taken over the ones that reported a calibration error
    # (adapters_calibration.group_calibration_to_svg: `n_measured_groups`, named
    # in the subtitle as "N of M group(s) could not be measured and are
    # excluded"). "across 5 groups" therefore put groups nobody measured into
    # the denominator of a disparity, which is part (d) of the headline rule.
    n_groups, n_measured = _n(d, "n_groups"), _n(d, "n_measured_groups")
    reported = _g(d, "n_measured_groups") is not None
    unmeasured = max(0, n_groups - n_measured) if reported else 0
    shown = n_measured if reported else _g(d, "n_groups", 0)
    tail = _ungraded_tail(unmeasured, "group(s) reported no calibration error")
    # THE ROWS WITH NO GROUP LABEL AT ALL. They are in no curve, so the curve
    # set covers less of the data than the sentence implies, and the group they
    # would form cannot be named because `str()` on an absent value MINTS the
    # name (measured: "worst m vs best None"). The adapter drops them, counts
    # them and says so on the canvas; this is the accessible half.
    no_label_rows = _n(d, "n_missing_group_rows")
    tail += _ungraded_tail(no_label_rows, "row(s) carry no group label and are in no curve")
    if unmeasured or no_label_rows:
        sev = _floor_sev(sev)
    return (
        f"Calibration disparity {_f(_g(d, 'ece_disparity', 0))} across {shown} groups; "
        f"worst {_g(d, 'worst_group', 'N/A')} vs best {_g(d, 'best_group', 'N/A')}.{tail}",
        sev,
    )


def _fr_calibration_disparity(d):
    disp = _g(d, "ece_disparity", 0) or 0
    ratio = _g(d, "disparity_ratio", 0) or 0
    sev = (
        "high"
        if disp > 0.03 or ratio > 3
        else ("medium" if disp > 0.01 or _g(d, "has_significant_disparity") else "low")
    )
    # The VERDICT's third state, as opposed to the data's.
    # calibration_disparity.svg grew a fourth branch for a run where every group
    # reported an error and the spread was measured but NO caller ever judged it
    # significant: the panel reads "DISPARITY VERDICT NOT SUPPLIED ... this is
    # not a finding of no disparity". This finder read the same missing key
    # through `_g(d, "has_significant_disparity")`, where absent is falsy and
    # indistinguishable from a measured False, and a small spread then graded
    # "low", the calmest band this chart has, beside that banner. The severity
    # is floored to the unresolved-unknown band, and the sentence says which of
    # the two it is; the measured spread is still reported, because the numbers
    # are real and it is only the verdict that is missing.
    verdict_known = _g(d, "has_significant_disparity_known")
    if verdict_known is None:
        verdict_known = _g(d, "has_significant_disparity") is not None
    if not verdict_known:
        sev = _floor_sev(sev)
    # The PARTIAL run, same shape as _fr_group_calibration: worst and best are
    # comparative, and they compare only the groups that reported an error
    # (adapters_calibration.calibration_disparity_to_svg: `n_measured_groups`).
    unmeasured = 0
    if _g(d, "n_measured_groups") is not None:
        unmeasured = max(0, _n(d, "n_groups") - _n(d, "n_measured_groups"))
    tail = _ungraded_tail(unmeasured, "group(s) reported no calibration error")
    # The same third population one surface along: a KEY in the mapping that is
    # the absence of a group. `str(key)` minted a name and this panel RANKED it
    # (measured: "worst b vs best None" at severity HIGH).
    missing_keys = _n(d, "n_missing_group_keys")
    tail += _ungraded_tail(missing_keys, "group key(s) carried no group label")
    if unmeasured or missing_keys:
        sev = _floor_sev(sev)
    verdict_clause = (
        ""
        if verdict_known
        else f" {_COULD_NOT_CHECK}: no significance verdict was supplied for this spread, "
        "so it is not a finding of no disparity."
    )
    return (
        f"ECE disparity {_f(disp)} ({_f(ratio, '.1f')}x); worst {_g(d, 'worst_group', 'N/A')} vs "
        f"best {_g(d, 'best_group', 'N/A')}.{verdict_clause}{tail}",
        sev,
    )


def _fr_pareto(d):
    # THE PARTIAL RUN, the same shape _fr_corr_heatmap below already closed.
    # `n_points` is the count that SURVIVED the finite-coordinate filter, so
    # "2 of 2 configurations are Pareto-optimal" reads as full coverage of
    # everything the caller supplied. Measured 2026-09-30 on three
    # configurations with one non-finite: the canvas said "1 of 3
    # configuration(s) have non-finite coordinates and are excluded" and this
    # sentence said "2 of 2 configurations are Pareto-optimal; best trade-off:
    # a" at severity info, so the reader who gets only the <desc> was told a
    # frontier had been searched over a set that was silently one smaller.
    # TOTAL loss was already refused by _pareto_not_assessable; this is the
    # partial case, which is the commoner one.
    n_dropped = _n(d, "n_dropped") or 0
    tail = _ungraded_tail(n_dropped, "configuration(s) have non-finite coordinates")
    sev = _floor_sev("info") if n_dropped else "info"
    return (
        f"{_g(d, 'n_frontier', 0)} of {_g(d, 'n_points', 0)} configurations are Pareto-optimal; "
        f"best trade-off: {_g(d, 'best_label', 'N/A') or 'N/A'}.{tail}",
        sev,
    )


def _fr_corr_heatmap(d):
    n = _g(d, "n_high_corr", 0) or 0
    # The PARTIAL run. A cell that could not be computed cannot raise
    # `n_high_corr`, so the fewer pairs were tested the cleaner this sentence
    # reads: "0 correlation(s) exceed |r| >= 0.30 across 3 features" is a
    # proxy-risk all-clear covering pairs nobody tested. The adapter refuses the
    # same all-clear on the canvas by naming the uncomputed pairs in its
    # subtitle (adapters_feature_engineering.correlation_heatmap_to_svg).
    #
    # Counted from the SAME rows the grid draws, cell by cell, rather than from
    # a separate tally: a None value is exactly what the template renders as
    # N/A in slate, so the two cannot drift apart.
    cells = [c for row in (_g(d, "rows", []) or []) for c in (_g(row, "cells", []) or [])]
    uncomputed = sum(1 for c in cells if _g(c, "value") is None)
    sev = "high" if n > 2 else ("medium" if n > 0 else "low")
    tail = _ungraded_tail(uncomputed, "feature-attribute pair(s) carried no correlation")
    if uncomputed:
        sev = _floor_sev(sev)
    return (
        f"{n} feature-attribute correlation(s) exceed |r| >= {_f(_g(d, 'threshold', 0), '.2f')} "
        f"across {_g(d, 'n_features', 0)} features.{tail}",
        sev,
    )


def _fr_corr_matrix(d):
    n = _g(d, "n_high_corr", 0) or 0
    # The PARTIAL run, the same one _fr_corr_heatmap closed, on the chart that
    # sat directly beside it and was left as it was. A pair that could not be
    # computed cannot raise `n_high_corr`, so the fewer pairs were tested the
    # cleaner this sentence read: "0 feature pair(s) exceed |r| >= 0.30 across 5
    # features" at severity "low" is a redundancy and proxy all-clear over pairs
    # nobody correlated. The adapter already refuses that all-clear on the
    # canvas ("N of M pair(s) not computed, so not tested" in the subtitle), and
    # the cells are drawn N/A in slate where a coefficient would be.
    #
    # Counted from the SAME cells the grid draws rather than from a separate
    # tally, so the two cannot drift apart.
    cells = [c for row in (_g(d, "rows", []) or []) for c in (_g(row, "cells", []) or [])]
    uncomputed = sum(1 for c in cells if _g(c, "value") is None)
    sev = "high" if n > 4 else ("medium" if n > 0 else "low")
    tail = _ungraded_tail(uncomputed, "feature pair(s) carried no correlation")
    if uncomputed:
        sev = _floor_sev(sev)
    return (
        f"{n} feature pair(s) exceed |r| >= {_f(_g(d, 'threshold', 0), '.2f')} across "
        f"{_g(d, 'n_features', 0)} features ({_g(d, 'method_display', 'N/A')}).{tail}",
        sev,
    )


def _fr_proxy_risk(d):
    counts = _g(d, "risk_counts", []) or []
    # CRITICAL is a distinct badge row (no longer folded into HIGH), so it is
    # counted separately here and drives a 'critical' severity.
    n_crit = sum(_g(c, "count", 0) or 0 for c in counts if _g(c, "label") == "CRITICAL")
    n_high = sum(_g(c, "count", 0) or 0 for c in counts if _g(c, "label") == "HIGH")
    has_med = any(_g(c, "label") == "MEDIUM" and _g(c, "count", 0) for c in counts)
    sev = "critical" if n_crit else ("high" if n_high else ("medium" if has_med else "low"))
    crit_clause = f" ({n_crit} critical)" if n_crit else ""
    # The PARTIAL run. `features` holds the RATED candidates only (the adapter
    # keeps the rest in `unrated_features` and draws them a NOT RATED pill), so
    # the denominator is the graded subset and part (d) was met. Part (c) was
    # not, and this is the chart where silence reads loudest: "0 high-risk proxy
    # variable(s) among 3 features" at severity "low" is the strongest all-clear
    # a proxy scan can give, and a candidate that arrived without a risk level
    # was neither cleared nor flagged, it was never scored.
    unrated = _n(d, "n_unrated", default=len(_g(d, "unrated_features", []) or []))
    tail = _ungraded_tail(unrated, "candidate feature(s) carried no proxy risk level")
    if unrated:
        sev = _floor_sev(sev)
    return (
        f"{n_crit + n_high} high-risk proxy variable(s){crit_clause} among "
        f"{len(_g(d, 'features', []) or [])} features.{tail}",
        sev,
    )


def _fr_transformation(d):
    r = _g(d, "avg_reduction", 0) or 0
    # The PARTIAL run. The denominator is already the measured subset (the
    # adapter sets `n_total` to `n_measured`, never `len(features)`, and says
    # so), and transformation_comparison.svg prints the unmeasured count beside
    # its badge, so parts (a) and (d) were met. Part (c) was not: "Transformation
    # is Effective ... 3/3 features improved" at severity "low" is an all-clear
    # over a run that compared three pairs of five, and a feature whose before or
    # after correlation was never reported is not a feature this transformation
    # left alone, it is one nobody looked at.
    unmeasured = _n(d, "n_unmeasured")
    # THE SECOND SILENCE, for a different reason and on the same headline row.
    # A feature that started at |r| = 0 HAS both halves of its pair, so it is
    # not `n_unmeasured`, and (0 - after) / 0 is undefined, so it has no
    # proportional reduction either: adapters_feature_engineering counts it as
    # `n_ratio_ungraded`, keeps it out of the average and out of the `n_total`
    # denominator, and transformation_comparison.svg ADDS THE TWO TOGETHER into
    # one NOT MEASURED pill ("2 feature(s), not graded") beside the band. This
    # sentence read only the first of the two, so a run whose whole silence was
    # ratio-ungraded said nothing at all: "Transformation is Effective ... 1/1
    # features improved" at severity "low", the all-clear, beside a headline
    # pill that already said a feature was not graded.
    ratio_ungraded = _n(d, "n_ratio_ungraded")
    sev = "low" if r >= 0.3 else ("medium" if r >= 0.1 else "high")
    tail = _ungraded_tail(unmeasured, "feature(s) carried no before/after pair to compare")
    # Worded to read grammatically into the fixed _NOT_COVERED suffix, which
    # closes every partial clause this library writes.
    tail += _ungraded_tail(
        ratio_ungraded,
        "feature(s) started at zero correlation, so have no proportional reduction to report",
    )
    if unmeasured or ratio_ungraded:
        sev = _floor_sev(sev)
    return (
        f"Transformation is {_g(d, 'summary_text', 'N/A')}: proxy correlation reduced by "
        f"{r * 100:.0f}% on average; {_g(d, 'n_improved', 0)}/{_g(d, 'n_total', 0)} "
        f"features improved.{tail}",
        sev,
    )


def _fr_intersectional_analysis(d):
    m = _g(d, "max_disparity", 0) or 0
    # The PARTIAL run. The spread is a max minus a min over the POPULATED cells
    # (part (a), and the adapter's own CRITICAL note explains why an unmeasured
    # operand may not name a breach), but "across gender x race" is a claim
    # about the whole grid, and a grid with three of eight cells populated is
    # not that grid. intersectional_analysis.svg prints "N of M cells, not
    # graded" beside its badge; this sentence claimed the full crossing at
    # severity "low", which is the all-clear.
    unmeasured = _n(d, "n_unmeasured")
    sev = "low" if m < 0.1 else ("medium" if m < 0.2 else "high")
    tail = _ungraded_tail(unmeasured, "intersection cell(s) carried no outcome rate")
    if unmeasured:
        sev = _floor_sev(sev)
    return (
        f"Feature '{_g(d, 'feature_name', 'N/A')}' shows max intersectional disparity {_f(m)} "
        f"across {_g(d, 'x_attr', '')} x {_g(d, 'y_attr', '')}.{tail}",
        sev,
    )


def _fr_intersectional_disparity(d):
    # THE WITHHELD GAP, printed as a question mark. adapters_feature_engineering
    # sends `disparity_pp` as None when no gap was reported ("percentage points,
    # withheld when no gap was reported") and `severity_graded` False when
    # nobody banded it; intersectional_disparity.svg draws NOT GRADED in slate
    # for the second, which its own note calls "not a point on the severity
    # scale", and prints N/A for the first through its |f1 filter.
    # `_f(None, '.1f')` here rendered it "?", so the accessible layer read "Max
    # intersectional disparity ?pp between N/A and N/A (not graded)": a question
    # mark where a measurement would be, which a reader cannot tell from a
    # rendering fault. That is the shape rule (d) exists to stop.
    #
    # The severity fallback stays "medium" rather than "info" for the reason it
    # always had: an unbanded gap is not benign news. What changes is that the
    # absence is said in words and the ungraded band floors the grade.
    sev = str(_g(d, "severity", "medium")).lower()
    sev = sev if sev in _VALID_SEVERITY else "medium"
    graded = _g(d, "severity_graded")
    if graded is None:
        graded = True
    pp = _g(d, "disparity_pp")
    disadv = _g(_g(d, "disadvantaged", {}), "group", "N/A")
    priv = _g(_g(d, "privileged", {}), "group", "N/A")
    between = f"between {disadv} and {priv}"
    if pp is None:
        # The subject leads and the marker follows, per the note in _fr_causal:
        # a leading marker suppresses build_explanation's not-assessable net,
        # and this chart's empty case relies on it. The two group names are
        # dropped when neither was supplied, because "between N/A and N/A" is
        # the same withheld-value shape one sentence further along.
        named = "N/A" not in (disadv, priv)
        return (
            f"Intersectional disparity: {_COULD_NOT_CHECK}, no compound gap was reported"
            f"{' ' + between if named else ''}.",
            _floor_sev(sev),
        )
    band = (
        f"({_g(d, 'severity', 'N/A')})"
        if graded
        else f"({_COULD_NOT_CHECK}: no severity band was reported for this gap)"
    )
    return (
        f"Max intersectional disparity {_f(pp, '.1f')}pp {between} {band}.",
        sev if graded else _floor_sev(sev),
    )


def _fr_training_report(d):
    """The accessible sentence for both training pages.

    THE CANVAS'S OWN WITHHOLDING FLAGS ARE READ, NOT RE-DERIVED (2026-09-30).
    ``adapters_training._baseline_state`` decides per cell whether a baseline
    number may be printed, and this finder ignored that decision and formatted
    the raw values with a ``0`` default. Two shapes of the same defect were
    executed on this tree, both against a canvas that correctly said "Accuracy
    and fairness violation were not measured on this run":

        accuracy nan, accuracy_measured False
            -> "<desc>... Baseline accuracy nan, violation 0.020 (constraint
               satisfied) ... (severity: LOW)"
        accuracy None, violation None
            -> "<desc>... Baseline accuracy ?, violation ? (constraint
               satisfied) ... (severity: LOW)"

    The description must never be more confident than the badge beside it, and
    severity LOW over a baseline nobody could score is the machine-readable half
    of the same claim: ``ChartExplanation.metadata()["severity"]`` is what a
    report pipeline reads INSTEAD of the sentence. A partial baseline now names
    which cell is missing and its severity is floored, and a fully measured
    baseline keeps the sentence and the severity it has always had.
    """
    issues = _g(d, "issues", []) or []
    sat = _g(d, "baseline_satisfied", False)
    # `is not False` rather than truthiness: these keys are absent from a
    # hand-built dict, and an absent flag must not withhold a number the dict
    # does carry. _baseline_state always writes both.
    acc_measured = _g(d, "baseline_accuracy_measured") is not False
    viol_measured = _g(d, "baseline_violation_measured") is not False
    acc = _g(d, "baseline_accuracy")
    viol = _g(d, "baseline_violation")
    acc_txt = _f(acc) if acc_measured and acc is not None else _COULD_NOT_CHECK
    viol_txt = _f(viol) if viol_measured and viol is not None else _COULD_NOT_CHECK
    withheld = acc_txt == _COULD_NOT_CHECK or viol_txt == _COULD_NOT_CHECK
    if any(str(_g(i, "severity", "")).lower() == "critical" for i in issues):
        sev = "critical"
    elif not sat or any(str(_g(i, "severity", "")).lower() == "high" for i in issues):
        sev = "high"
    else:
        sev = _norm_sev(_g(_g(d, "recommendation", {}), "priority", "low"))
    if withheld:
        sev = _floor_sev(sev)
    return (
        f"Baseline accuracy {acc_txt}, violation "
        f"{viol_txt} (constraint {'satisfied' if sat else 'violated'}); "
        f"recommended: {_g(_g(d, 'recommendation', {}), 'method', 'N/A')}.",
        sev,
    )


def _fr_method_comparison(d):
    methods = _g(d, "methods", []) or []
    ok = _count_true(methods, "satisfied")
    # The PARTIAL run, part (d), and the worst shape of it: an ungraded row was
    # counted OUT of the numerator and INTO the denominator at the same time.
    # `satisfied` is None for a method whose run reported no constraint result
    # (adapters_training._method_state leaves `graded` False and the canvas
    # badges the row NOT CHECKED in slate), and `_count_true` reads None as "did
    # not satisfy", so a comparison that evaluated two methods out of three read
    # "2 of 3 method(s) satisfy the fairness constraint": the third is reported
    # as a method that was tested and fell short. method_comparison.svg already
    # names the same count beside its headline (`n_ungraded`), so the two
    # surfaces now count the same population.
    ungraded = _n(d, "n_ungraded", default=sum(1 for m in methods if _g(m, "graded") is False))
    graded = max(0, len(methods) - ungraded)
    # A VACUOUS verdict is inside `ungraded` and it is NOT "no constraint result
    # reported": the method reported one, and it is satisfied by construction
    # because the model collapsed to a constant prediction. Saying it reported
    # nothing sends the reader looking for a run that did not finish. Added
    # 2026-09-30 with adapters_training's `n_vacuous`.
    vacuous = _n(d, "n_vacuous", default=sum(1 for m in methods if _g(m, "vacuous") is True))
    vacuous = min(vacuous, ungraded)
    if ungraded and not graded:
        if vacuous == ungraded:
            return _could_not_check(
                d,
                f"every one of the {ungraded} method(s) collapsed to a constant prediction, "
                "so none of their constraint results measures anything.",
            )
        if vacuous:
            return _could_not_check(
                d,
                f"not one of the {ungraded} method(s) produced a constraint result that "
                f"measures anything: {vacuous} collapsed to a constant prediction and "
                f"{ungraded - vacuous} reported no constraint result at all.",
            )
        return _could_not_check(
            d,
            f"not one of the {ungraded} method(s) reported a constraint result, so none "
            "was evaluated and none was cleared.",
        )
    sev = "low" if ok else ("high" if graded else "info")
    if vacuous == ungraded:
        tail = _ungraded_tail(
            ungraded, "method(s) collapsed to a constant prediction, so their verdicts are vacuous"
        )
    elif vacuous:
        tail = _ungraded_tail(
            ungraded,
            f"method(s) are not graded ({vacuous} collapsed to a constant prediction, "
            f"{ungraded - vacuous} reported no constraint result)",
        )
    else:
        tail = _ungraded_tail(ungraded, "method(s) reported no constraint result")
    if ungraded:
        sev = _floor_sev(sev)
    return (f"{ok} of {graded} method(s) satisfy the fairness constraint.{tail}", sev)


def _fr_tradeoff(d):
    pts = _g(d, "points", []) or []
    # This finder contradicted its own canvas twice over.
    #
    #   * SEVERITY was the literal constant "info", so "0 of 5 configurations
    #     satisfy the constraint" was read out at the all-clear grade beside a
    #     scatter of five red dots. `ChartExplanation.metadata()["severity"]` is
    #     what a report pipeline reads INSTEAD of the sentence, and "info" there
    #     says a sweep that met the fairness constraint nowhere is fine news.
    #   * adapters_training already puts the withheld count in the TITLE of this
    #     chart, in the headline band ("Accuracy-Fairness Trade-off: 2 of 9
    #     points not plotted"), because the subtitle here is fixed prose. The
    #     sentence underneath spoke for the plotted subset as though it were the
    #     whole sweep, so the accessible layer contradicted the only headline
    #     the canvas has.
    #
    # Part (d) as well: `_count_true` reads a None verdict as "did not satisfy",
    # so a configuration whose constraint was never evaluated was reported as
    # one that WAS evaluated and fell short. The real adapter filters those out
    # before they reach the template (counting them into `n_withheld_points`
    # instead), so the graded population here is normally every point; a
    # hand-built dict that still carries one is kept out of both sides.
    graded = [p for p in pts if isinstance(_g(p, "satisfied"), bool)]
    withheld = _n(d, "n_withheld_points") + (len(pts) - len(graded))
    if not graded:
        return _could_not_check(
            d,
            "no configuration on this plot reported a constraint verdict, so no "
            "constraint strength was cleared and none was rejected.",
        )
    ok = _count_true(graded, "satisfied")
    sev = "info" if ok == len(graded) else ("high" if ok == 0 else "medium")
    tail = _ungraded_tail(
        withheld, "trade-off point(s) reported no complete result and are not plotted"
    )
    if withheld:
        sev = _floor_sev(sev)
    return (
        f"{ok} of {len(graded)} configurations satisfy the constraint; "
        f"{len(_g(d, 'pareto_points', []) or [])} on the Pareto frontier.{tail}",
        sev,
    )


def _fr_threshold(d):
    fi = _g(d, "fairness_improvement", {}) or {}
    # `is_feasible` defaulted to True here, so a dict carrying no feasibility
    # result at all read "Threshold optimisation is feasible" beside a green
    # badge. Feasibility is a result, not an assumption.
    if _g(d, "is_feasible") is None:
        return _could_not_check(
            d, "no feasibility result was supplied, so no threshold was optimised."
        )
    red = _g(fi, "reduction_pct", 0) or 0
    feas = _g(d, "is_feasible")
    sev = "high" if not feas else ("info" if red >= 50 else ("low" if red >= 20 else "medium"))
    return (
        f"Threshold optimisation ({_g(d, 'constraint_type', '?')}) "
        f"{'is feasible' if feas else 'is INFEASIBLE'}; cuts disparity by {red:.0f}% "
        f"with accuracy change {_g(_g(d, 'accuracy_change', {}), 'change', 0):+.3f}.",
        sev,
    )


def _fr_reweighting(d):
    # This finder had no third state at all, and it makes the strongest claim on
    # the page: BEST METHOD is a recommendation.
    #
    #   * `_g(d, "original_disparity", 0) or 0` turned an UNREPORTED disparity
    #     into 0.000, which is the best possible reading of that metric, and
    #     banded it "low", the calmest severity here. adapters_post_processing
    #     leaves the key None when no method reported one.
    #   * reweighting_comparison_report.svg paints the emerald stripe and the
    #     emerald name only when `best_method_graded`, and appends "(not graded
    #     here)" in slate otherwise, because a name with nothing measured behind
    #     it is not a recommendation. This sentence named it either way.
    #   * `n_ungraded` counts the supplied method results that reported no
    #     fairness, accuracy, calibration or trade-off number; the subtitle at
    #     y=122 says so, and the denominator here is `len(methods)`, which is
    #     already the graded subset, so part (d) held and part (c) did not.
    od = _g(d, "original_disparity")
    n_methods = len(_g(d, "methods", []) or [])
    ungraded = _n(d, "n_ungraded")
    graded = _g(d, "best_method_graded")
    if graded is None:
        graded = True
    best = _g(d, "best_method", "N/A")
    # Subject first, marker after: a leading marker would suppress
    # build_explanation's not-assessable net, and this chart's empty case leans
    # on it to keep the placeholder name off the canvas entirely.
    best_clause = (
        f"Best reweighting method is '{best}'"
        if graded
        else f"Best reweighting method: {_COULD_NOT_CHECK}, '{best}' is named with no graded "
        f"row behind it, so nothing here recommends it"
    )
    disparity_clause = (
        f"an original disparity of {_f(od)}"
        if od is not None
        else "an original disparity that was never reported"
    )
    sev = "medium" if od is None else ("high" if od >= 0.1 else ("medium" if od >= 0.05 else "low"))
    tail = _ungraded_tail(ungraded, "supplied method result(s) reported nothing to score")
    if ungraded or not graded or od is None:
        sev = _floor_sev(sev)
    return (
        f"{best_clause} against {disparity_clause} across {n_methods} methods.{tail}",
        sev,
    )


def _fr_detailed_report(d):
    # This finding is the ONLY full sentence a reader gets twice: once in the
    # visible Explanation paragraph and once in the accessible <desc>, which is
    # all a screen-reader user gets. So it must never be more confident than the
    # banner beside it.
    #
    # fairness_detailed_report.svg downgrades its banner to COULD NOT CHECK as
    # soon as one metric row was never compared to a threshold, but this
    # function was written from the score alone: the badge read COULD NOT CHECK
    # while the <desc> read "Overall fairness is FAIR with a score of 90/100".
    # The downgrade is re-derived here from the SAME signal the template uses,
    # the "COULD NOT CHECK" prefix the adapter writes into a row's
    # interpretation line, so the two cannot drift apart.
    metrics = _g(d, "metrics") or []
    unchecked = sum(
        1
        for m in metrics
        if _COULD_NOT_CHECK in str(_g(m, "interpretation", "") or "").upper()
        or str(_g(m, "state", "") or "") == "could_not_check"
    )
    score = _g(d, "fairness_score")
    try:
        score_text = f"{float(score):.0f}/100"
    except (TypeError, ValueError):
        score_text = "not available"

    if unchecked or score is None:
        n = len(metrics)
        return (
            f"{_COULD_NOT_CHECK}: {unchecked} of {n} metric(s) were never compared to a "
            f"threshold, so the score ({score_text}) does not cover them and this report "
            f"certifies nothing about the whole model.",
            _UNKNOWN_SEV,
        )

    a = str(_g(d, "overall_assessment", "")).upper()
    return (
        f"Overall fairness is {_g(d, 'overall_assessment', '?')} with a score of {score_text}.",
        {"FAIR": "low", "MARGINAL": "medium", "UNFAIR": "high"}.get(a, "medium"),
    )


def _fr_monitoring(d):
    n = _g(d, "n_alerts", 0) or 0
    # THREE STATES ON THE BADGE, because the canvas beside this sentence has
    # three. `{% if any_alert %}` sends every falsy value to the emerald OK
    # branch, so adapters_monitoring sends `any_alert_recorded` alongside it and
    # monitoring_dashboard.svg draws NOT RECORDED in slate for a window that
    # never said whether anything fired, replacing the "0 alerts" pill with "no
    # guardrail applied" for the reason written at that line: a count of firings
    # reads as a measurement. This finder read only the bool, where absent is
    # falsy and indistinguishable from a measured False, so the accessible layer
    # said "0 of N fairness metrics alerting" at severity INFO, which is the
    # all-clear, beside a badge that had already withheld the verdict.
    #
    # Derived from the adapter's own flag; a dict that predates it falls back to
    # the key being present, so a healthy render is unchanged.
    recorded = _g(d, "any_alert_recorded")
    if recorded is None:
        recorded = _g(d, "any_alert") is not None
    sev = (
        "info"
        if not _g(d, "any_alert")
        else ("critical" if n >= 3 else ("high" if n >= 2 else "medium"))
    )
    if not recorded:
        sev = _floor_sev(sev)
    # The PARTIAL run, part (d). `n_metrics` counts every metric that was
    # COMPUTED; a metric with no entry in the alert record was never compared to
    # a guardrail, so it cannot alert. Putting it in the denominator made the
    # fraction read cleaner the LESS was checked: "0 of 4 fairness metrics
    # alerting" for a window that applied a guardrail to one of them.
    #
    # The adapter already keeps the two apart for this exact reason
    # (adapters_monitoring.monitoring_dashboard_to_svg: "N metrics monitored
    # counted every metric that was COMPUTED"), and its comment says `n_checked`
    # is kept as a separate key rather than narrowing `n_metrics` BECAUSE this
    # finder reads `n_metrics` as the population size. This is that read.
    n_metrics = _n(d, "n_metrics")
    # An absent `n_checked` means the dict predates the split, so the sentence
    # is left exactly as it was rather than inventing an unchecked count.
    n_checked = _n(d, "n_checked", default=n_metrics)
    unchecked = max(0, n_metrics - n_checked)
    if n_metrics and not n_checked:
        return _could_not_check(
            d,
            f"not one of the {n_metrics} metric(s) on this dashboard was compared to a "
            "guardrail, so no alert could fire.",
        )
    tail = _ungraded_tail(unchecked, "metric(s) were never compared to a guardrail")
    if unchecked:
        sev = _floor_sev(sev)
    if not recorded:
        # The COUNT goes too, for the reason the template's own note gives: "0
        # alerts" is a count of firings, and a count reads as a measurement.
        # Subject first, marker after, so build_explanation's not-assessable net
        # still owns the sentence on a dashboard that monitored nothing at all.
        return (
            f"Alert status: {_COULD_NOT_CHECK}, this window recorded no alert verdict, so "
            f"nothing on it fired and nothing on it is clear, across {n_checked} fairness "
            f"metric(s) on {_g(d, 'sample_count', 0)} samples.{tail}",
            sev,
        )
    return (
        f"{n} of {n_checked} fairness metrics alerting on "
        f"{_g(d, 'sample_count', 0)} samples.{tail}",
        sev,
    )


def _fr_drift(d):
    s = _g(d, "overall_drift_score", 0) or 0
    sev = (
        "high"
        if s >= 0.6
        else ("medium" if s >= 0.3 else ("low" if _g(d, "drift_detected") else "info"))
    )
    # Three states on the VERDICT, because the badge beside this sentence has
    # three. adapters_monitoring sends `overall_verdict_recorded` precisely
    # because `{% if drift_detected %}` routes every falsy value to the green
    # STABLE branch and None is falsy; its comment says "keep both keys". This
    # finder read only the bool, so a result object that recorded no overall
    # verdict was described as "Drift not detected" at severity "info": the
    # calmest words this chart has, for a question nobody answered.
    recorded = _g(d, "overall_verdict_recorded")
    if recorded is None:
        recorded = _g(d, "drift_detected") is not None
    if recorded:
        verdict = "DETECTED" if _g(d, "drift_detected") else "not detected"
    else:
        verdict = f"{_COULD_NOT_CHECK}: no overall verdict was recorded"
        sev = _floor_sev(sev)
    # The PARTIAL run. The overall score is the MAXIMUM over the scales that
    # reported one, so a scale that reported nothing cannot raise it and the
    # fewer scales reported, the calmer this number reads. The canvas names them
    # in the one header line it has ("N of M scale(s) did not fully report").
    partial = _n(d, "n_partial_scales")
    tail = _ungraded_tail(partial, "temporal scale(s) did not fully report")
    if partial:
        sev = _floor_sev(sev)
    return (
        f"Drift {verdict} on '{_g(d, 'metric_name', '?')}'; overall drift score {_f(s)}.{tail}",
        sev,
    )


def _fr_alert_timeline(d):
    a, t = _g(d, "n_alerted", 0) or 0, _g(d, "n_total", 0) or 0
    # The PARTIAL run, part (d). A window that applied no guardrail cannot raise
    # an alert, so it does not belong in the denominator of "N of M windows
    # raised alerts": the more windows monitored nothing, the cleaner that
    # fraction read. The adapter keeps such a window out of `n_clean` for the
    # same reason and counts it in `n_unmonitored`
    # (adapters_monitoring.alert_timeline_to_svg), so this reads the same split.
    unmonitored = _n(d, "n_unmonitored")
    monitored = max(0, t - unmonitored)
    if unmonitored and not monitored:
        return _could_not_check(
            d, "no window in this history applied a guardrail, so none is clean and none alerted."
        )
    # The PARTIAL run one level down, per METRIC (BGL5, 2026-09-27). The clause
    # above withdraws a whole window that applied no guardrail; a window that
    # applied one to SOME of its metrics used to be counted as clean, and this
    # sentence read "0 of 1 recent monitoring windows raised alerts (1 clean)."
    # at severity info for a window two of whose four metrics were never compared
    # to anything. The adapter now keeps such a window out of `n_clean` and sends
    # the per-metric shortfall (alert_timeline_to_svg: `n_metrics_unchecked`),
    # which is the same count monitoring_dashboard_to_svg has always sent and
    # which _fr_monitoring already names on the dashboard drawn from the SAME
    # object. An absent key means the dict predates the split, so the sentence is
    # left exactly as it was rather than inventing an unchecked count.
    partial_windows = _n(d, "n_partial")
    unchecked_metrics = _n(d, "n_unchecked_metrics")
    sev = "info" if a == 0 else ("high" if monitored and a / monitored >= 0.5 else "medium")
    tail = _ungraded_tail(unmonitored, "window(s) applied no guardrail")
    if partial_windows:
        tail += _ungraded_tail(
            unchecked_metrics,
            f"metric(s) across {partial_windows} partly checked window(s) were never "
            "compared to a guardrail",
        )
        # THE NAMES, when the adapter supplies them, as a SEPARATE clause after
        # the shared not-covered marker. A count cannot tell a metric the window
        # genuinely could not compute from a REQUESTED metric whose name has a
        # one-letter typo, and the typo is the case that produces zero warnings
        # anywhere else in the pipeline: nothing validates
        # `FairnessMonitorConfig.metrics_to_track` against the metrics the
        # tracker knows how to compute.
        #
        # APPENDED, NOT INTERPOLATED. The first version of this put the names
        # between "metric(s)" and "across", which broke two existing assertions
        # that quote that sentence contiguously
        # (tests/test_bgl5_post_processing_2_and_rendering.py). Their subject is
        # unaffected by this addition, so the sentence is left alone and the
        # names follow it. At most three are spelled out and the rest counted,
        # because this string is also the <desc>.
        names = [str(n) for n in (_g(d, "unchecked_metric_names") or []) if n]
        if names:
            shown = ", ".join(names[:3])
            if len(names) > 3:
                shown += f" and {len(names) - 3} more"
            tail += f" Never compared: {shown}."
    if unmonitored or partial_windows:
        sev = _floor_sev(sev)
    return (
        f"{a} of {monitored} recent monitoring windows raised alerts "
        f"({_g(d, 'n_clean', 0)} clean).{tail}",
        sev,
    )


def _fr_temporal(d):
    degs = _g(d, "degradations", []) or []
    trends = _g(d, "trends", []) or []
    summaries = _g(d, "metric_summaries", []) or []

    # The PARTIAL run, part (d), on all three counts in one sentence.
    #
    #   * `metric_summaries` holds a card for every metric NAMED, and the
    #     adapter counts the ones with no history at all (`n_untracked`, printed
    #     beside the headline in temporal_analysis.svg), so "Tracked 3 metrics"
    #     claimed a metric nobody tracked;
    #   * a trend row carries `tracked` False and `slope` None when
    #     ``detect_trend`` was never run for it, so "across 3 trend checks"
    #     counted a check that never happened;
    #   * a degradation row carries `ran` False with `degraded` FORCED to False,
    #     because ``detect_weekly_degradation`` answers (False, nan) when it
    #     REFUSES to run, so a check that declined counted into the numerator as
    #     a metric found steady.
    #
    # The rows are counted the way the template draws them, cell by cell, so the
    # two cannot drift apart; `n_untracked` is preferred for the cards because
    # the adapter computes it from the summaries themselves.
    #
    # Both row tests default to GRADED, so only a row that says outright it was
    # not fitted is withdrawn: a dict that predates the split describes exactly
    # what it did before, which is the over-correction control in code.
    def _trend_graded(t):
        if _g(t, "tracked") is False:
            return False
        return not (isinstance(t, dict) and "slope" in t and t["slope"] is None)

    untracked = _n(d, "n_untracked")
    tracked = max(0, len(summaries) - untracked)
    graded_trends = [t for t in trends if _trend_graded(t)]
    ran = [x for x in degs if _g(x, "ran", True)]
    n_deg = sum(1 for x in ran if _g(x, "degraded"))
    ungraded_checks = (len(trends) - len(graded_trends)) + (len(degs) - len(ran))
    if n_deg:
        sev = "high"
    elif any(abs(_g(t, "slope", 0) or 0) >= 0.01 for t in graded_trends):
        sev = "medium"
    else:
        sev = "info"
    tail = _ungraded_tail(untracked, "metric(s) carried no history and were not tracked")
    tail += _ungraded_tail(ungraded_checks, "trend or degradation check(s) never ran")
    if untracked or ungraded_checks:
        sev = _floor_sev(sev)
    return (
        f"Tracked {tracked} metrics; {n_deg} showing weekly degradation across "
        f"{len(graded_trends)} trend checks.{tail}",
        sev,
    )


def _fr_experiment_results(d):
    # The two verdicts on this canvas are three-state, so the sentence under it
    # is too. "is not significant" and the absence of a heterogeneity clause both
    # used to be written from adapter defaults (p = 1.0, detected = False), so
    # this paragraph reported two tests that the experiment never ran.
    het_known = _g(d, "heterogeneity_known", True)
    p_known = _g(d, "overall_p_known", True)
    het = _g(d, "heterogeneity_detected")
    sig = _g(d, "overall_significant")
    # THE HEADLINE NUMBER, the third of the three states on this canvas and the
    # one this sentence still spoke for. experiment_results.svg withholds the
    # 22px estimate entirely without `overall_effect_known` and draws COULD NOT
    # CHECK in slate where it would be; its note calls that number "the last
    # fabrication left on this canvas". The adapter then sends the WORD "not
    # computed" in `overall_effect`, so this f-string read out "Overall
    # treatment effect not computed ... The effect is not significant." at
    # severity INFO: the words were honest and the grade was the all-clear. A
    # hand-built dict with no key at all fell to `'?'`, which is the
    # withheld-value shape rather than a statement that there is none.
    effect_known = _g(d, "overall_effect_known")
    if effect_known is None:
        effect_known = _g(d, "overall_effect") is not None

    if p_known:
        verdict = f"The effect is {'significant' if sig else 'not significant'}."
    else:
        verdict = f"{_COULD_NOT_CHECK}: no p-value was reported, so significance is unknown."
    if not het_known:
        het_clause = f" {_COULD_NOT_CHECK}: no heterogeneity test was reported either."
    elif het:
        het_clause = " Heterogeneous effects were detected across subgroups."
    else:
        het_clause = ""

    severity = "high" if (het_known and het) else ("medium" if (p_known and sig) else "info")
    if not (p_known and het_known and effect_known):
        severity = _UNKNOWN_SEV

    # The PARTIAL run. The two overall verdicts above are three-state, but the
    # per-intersection rows were not mentioned at all, so "across 4
    # intersections" spoke for four rows of which the canvas had graded two: the
    # power panel prints its fraction over the graded population and names the
    # rest (`n_power_ungraded`), and the forest plot leaves a row with no
    # estimate off the axis rather than parking it on the zero line
    # (`forest_omitted`). Both counts are stated here so the sentence covers
    # what the picture covers, and no more.
    n_power_ungraded = _n(d, "n_power_ungraded")
    n_omitted = _n(d, "forest_omitted")
    partial_clause = _ungraded_tail(
        n_power_ungraded, "intersection(s) reported no power verdict"
    ) + _ungraded_tail(
        n_omitted, "intersection(s) reported no effect estimate and were not plotted"
    )
    if partial_clause:
        severity = _floor_sev(severity)
    # Subject first, marker after: a leading marker suppresses
    # build_explanation's not-assessable net, and this chart's empty case needs
    # it, or "across 0 intersections" survives onto a canvas that estimated
    # nothing (tests/test_adapters_experimentation_empty.py pins exactly that).
    effect_clause = (
        f"Overall treatment effect {_g(d, 'overall_effect')}"
        if effect_known
        else f"Overall treatment effect: {_COULD_NOT_CHECK}, none was estimated"
    )
    return (
        f"{effect_clause} across "
        f"{_g(d, 'n_intersections', 0)} intersections "
        f"(p={_g(d, 'overall_p_display', '?')}). {verdict}{het_clause}{partial_clause}",
        severity,
    )


# The four decisions this canvas can badge, and the severity each carries.
# operations.experimentation.analysis.RecommendationDecision is a closed
# vocabulary of exactly these four, and experiment_recommendation.svg draws a
# labelled badge for each; anything else is not a decision this library made.
_RECOMMENDATION_SEVERITY = {
    "INVESTIGATE_FURTHER": "high",
    "EXTEND_EXPERIMENT": "medium",
    "KEEP_CONTROL": "low",
    "DEPLOY_TREATMENT": "low",
}


def _fr_experiment_recommendation(d):
    # The <desc> is the whole artifact for a screen-reader user, and it must
    # never be the MORE confident of the two surfaces. This finder used to
    # title-case whatever string sat under "decision" and read it out as
    # "Recommendation: <X>", ignoring `decision_known` entirely, so:
    #
    #   * an ABSENT decision, which the canvas badges "NO RECOMMENDATION MADE"
    #     in grey, was announced as a recommendation to every reader who cannot
    #     see the canvas (build_explanation's not_assessable net caught the
    #     adapter path, but the sentence was still written first and the finder
    #     itself asserted a decision nobody made); and
    #   * an UNRECOGNISED decision string was echoed verbatim:
    #     experiment_recommendation_to_svg({"decision": "ship it now"}) drew the
    #     template's fall-through badge and the <desc> read "Recommendation:
    #     Ship It Now", a recommendation the canvas never showed and this
    #     library cannot make.
    #
    # A decision is named only when the adapter says one was supplied AND it is
    # one of the four this library grades. Anything else is could-not-check,
    # which is never more confident than the badge beside it.
    dec = str(_g(d, "decision", "") or "").upper()
    known = _g(d, "decision_known")
    if known is None:
        known = bool(dec)
    if not known or dec not in _RECOMMENDATION_SEVERITY:
        return _could_not_check(
            d,
            "no decision this chart can state was supplied, so nothing is recommended "
            "here, for the treatment or for the control.",
        )
    # A confidence of 0 is a real report and still prints as 0%. An ABSENT
    # confidence printed "(confidence %)" beside the canvas's "not reported".
    conf = str(_g(d, "confidence_display", "") or "").strip()
    conf_clause = f"(confidence {conf}%)" if conf else "(confidence not reported)"
    return (
        f"Recommendation: {dec.replace('_', ' ').title()} "
        f"{conf_clause}. {_g(d, 'summary_text', '')}".strip(),
        _RECOMMENDATION_SEVERITY[dec],
    )


def _fr_power(d):
    u, t = _g(d, "n_underpowered", 0) or 0, _g(d, "n_total", 0) or 0
    sev = "info" if u == 0 else ("high" if t and u / t >= 0.5 else "medium")
    # The PARTIAL run. `n_total` is already the GRADED population, so part (a)
    # and part (d) were met; part (c) was not. A row that reported no power and
    # no verdict is simply missing from both sides of "0 of 3 intersections are
    # underpowered", and nothing in the sentence said two more were supplied.
    ungraded = _n(d, "n_ungraded")
    if not ungraded and _g(d, "n_supplied") is not None:
        ungraded = max(0, _n(d, "n_supplied") - t)
    # "avg power %" is what the empty `avg_power_display` printed: a percent
    # sign with no number, which is the malformed shape of a withheld value
    # rather than a statement that it was withheld. An average over no reported
    # power is not zero and it is not a number at all.
    avg = str(_g(d, "avg_power_display", "") or "").strip()
    avg_known = _g(d, "avg_power_known")
    if avg_known is None:
        avg_known = bool(avg)
    avg_clause = f"avg power {avg}%" if avg_known and avg else "avg power not reported"
    tail = _ungraded_tail(ungraded, "intersection(s) reported no power")
    if ungraded or not avg_known:
        sev = _floor_sev(sev)
    return (
        f"{u} of {t} intersections are underpowered ({avg_clause}, "
        f"target {_g(d, 'power_target', '?')}).{tail}",
        sev,
    )


def _fr_causal(d):
    temporal = _g(d, "temporal", None)
    # Explicit None check: `n_ok or 4` treated a legitimate 0 (no Baron-Kenny
    # step passed) as missing and reported severity "low". Bands align with
    # the chart: 0-1 steps -> high, 2-3 -> medium, 4 -> low.
    n_ok = _g(d, "n_steps_ok", None)
    try:
        n_ok = 4 if n_ok is None else int(n_ok)
    except (TypeError, ValueError):
        n_ok = 4
    # `is_stable` defaulted to TRUE, so a temporal block that reported no
    # stability result read as one that ran and found the mediation stable: the
    # branch that raises severity to "high" could never fire on an absent
    # verdict. Only a reported False is a finding of instability; an absent one
    # is picked up by the ungraded clause below.
    temporal_stable = _g(temporal, "is_stable") if temporal else None
    if temporal_stable is False:
        sev = "high"
    elif n_ok <= 1:
        sev = "high"
    elif n_ok < 4:
        sev = "medium"
    else:
        sev = "low"
    # The sentence contradicted ITSELF as well as the canvas. With `n_steps_ok`
    # absent the band above assumed 4 (the `n_ok is None -> 4` line), while the
    # text below printed `_g(d, "n_steps_ok", 0)`, so the reader was told
    # "0/4 Baron-Kenny steps satisfied" at severity "low": the strongest
    # possible failure of the mediation claim, graded as the mildest news, in
    # one sentence. The printed number now comes from the SAME resolved count
    # the band uses, and an absent count says so rather than printing a zero.
    #
    # And the PARTIAL run: causal_decomposition.svg prints "N step(s) not
    # reported" in its subtitle (adapters_experimentation counts
    # `n_steps_ungraded`, a step whose verdict never arrived, which `n_steps_ok`
    # deliberately does not count as satisfied). "3/4 steps satisfied" over a
    # fourth step nobody ran reports a step that WAS run and failed.
    reported = _g(d, "n_steps_ok") is not None
    steps_text = f"{n_ok}/4" if reported else f"{_COULD_NOT_CHECK}: no count of"
    ungraded_steps = _n(d, "n_steps_ungraded")
    tail = _ungraded_tail(ungraded_steps, "Baron-Kenny step(s) reported no verdict")
    # THE GAUGE, which is the largest thing on this canvas and the last defaulted
    # number in this sentence. causal_decomposition.svg draws no needle and
    # prints no percentage without `proportion_mediated_known`; its own comment
    # says why, and it is the sharpest statement of the rule on any template: "a
    # 0% gauge is the finding 'the mediator carries none of the gap', which is
    # the most specific claim this dial can make". `_g(d, "prop_med_pct", 0)`
    # wrote exactly that claim into the accessible layer for a decomposition
    # that reported no proportion at all, beside a dial reading COULD NOT CHECK.
    # An absent proportion also arrives as None from a hand-built dict, and
    # "accounts for None% of the effect" is the rendering-fault shape.
    #
    # Derived from the adapter's own flag; a dict that predates it falls back to
    # the value being present, so a healthy render is unchanged.
    prop_known = _g(d, "proportion_mediated_known")
    if prop_known is None:
        prop_known = _g(d, "prop_med_pct") is not None
    # THE MARKER IS PLACED MID-SENTENCE, and this is a rule for every clause
    # about ONE withheld value on an otherwise populated chart, not a stylistic
    # choice here. `_is_could_not_check` matches on the LEADING marker, and
    # build_explanation reads it twice: a leading marker swaps the whole action
    # line for the generic re-run instruction, which is not what a reader with
    # four satisfied Baron-Kenny steps and one missing gauge should be told, AND
    # it SUPPRESSES the not-assessable net that replaces the entire sentence for
    # a chart that measured nothing at all. Leading with it therefore leaks this
    # sentence's own zeros and N/As onto an empty canvas, which is the opposite
    # of what the clause is for. Lead with the subject; put the marker after it.
    prop_clause = (
        f"accounts for {_g(d, 'prop_med_pct')}% of the effect"
        if prop_known
        else f"reports no proportion mediated ({_COULD_NOT_CHECK}), so how much of the gap "
        f"runs through it is not established here"
    )
    if ungraded_steps or not reported or not prop_known:
        sev = _floor_sev(sev)
    return (
        f"Mediator '{_g(d, 'mediator', '?')}' {prop_clause}; "
        f"{steps_text} Baron-Kenny steps satisfied.{tail}",
        sev,
    )


def _fr_workflow(d):
    return (
        f"Workflow overview: {len(_g(d, 'integrations', []) or [])} tracking integration(s), "
        f"{len(_g(d, 'vcs_tools', []) or [])} VCS tool(s), "
        f"{len(_g(d, 'test_types', []) or [])} test type(s).",
        "info",
    )


def _fr_hierarchical(d):
    # The default was True, so a data dict that never carried a gate decision
    # (a malformed call, a partial dict) produced the sentence "Hierarchical
    # gate APPROVED" in the visible paragraph and in the accessible <desc>.
    # An approval is the single most damaging string this library can emit, so
    # it is written only when the dict actually carries one. Every real adapter
    # sets this key (rendering/adapters_workflow.py), so a healthy render is
    # unchanged.
    appr = _g(d, "approved")
    if appr is None:
        return _could_not_check(d, "no gate decision was supplied, so no level was graded.")
    warns = _g(d, "warnings", []) or []
    # The PARTIAL run, and the empty-denominator all-clear inside it. A gate
    # level with no metric was never evaluated: the canvas badges it NOT CHECKED
    # in slate (adapters_workflow leaves `single_status` / `intersect_status`
    # None, and tests/test_adapters_row_level_regression.py pins that the three
    # badges are neither FAIL nor red), while this sentence wrote "single-attr
    # 0/0, intersectional 0/0 metrics passed" beside APPROVED at severity INFO.
    # Zero of zero passed is not a level that passed.
    #
    # THE DENOMINATOR, and the contradiction that outlived the fix above.
    # hierarchical_gate.svg re-derives each level's total from the ROWS, taking
    # out the metrics the gate never graded ("(n_*_total or 0) - _ug.*"), and
    # prints the ungraded count beside it, precisely so an ungraded metric is in
    # neither the numerator nor the denominator. The counts the adapter supplies
    # in `n_*_total` are the RAW ones, every metric on the level included, so
    # this sentence wrote "single-attr 1/2 metrics passed" beside a canvas
    # reading "PASS 1/1 checks, 1 not checked": the canvas reports a level that
    # passed everything it measured, the <desc> reports a level that measured
    # two and failed one, and a screen-reader user gets only the second.
    #
    # Re-derived here from the SAME rows, by the SAME rule, so the two cannot
    # disagree. `levels` is absent from a dict that carries only the six counts,
    # and then nothing is subtracted and the sentence is exactly what it was.
    _ug = {"overall": 0, "single-attr": 0, "intersectional": 0}
    for lv in _g(d, "levels", []) or []:
        lv_type = str(_g(lv, "type", "") or "").strip().lower()
        # The template's own mapping: "overall" and "single attribute" by name,
        # everything else intersectional. Kept identical on purpose.
        bucket = (
            "overall"
            if lv_type == "overall"
            else ("single-attr" if lv_type == "single attribute" else "intersectional")
        )
        for m in _g(lv, "metrics", []) or []:
            if _g(m, "passed") is None:
                _ug[bucket] += 1
    levels = (
        ("overall", _n(d, "n_overall_pass"), max(0, _n(d, "n_overall_total") - _ug["overall"])),
        (
            "single-attr",
            _n(d, "n_single_pass"),
            max(0, _n(d, "n_single_total") - _ug["single-attr"]),
        ),
        (
            "intersectional",
            _n(d, "n_intersect_pass"),
            max(0, _n(d, "n_intersect_total") - _ug["intersectional"]),
        ),
    )
    n_ungraded_metrics = sum(_ug.values())
    graded = [(name, p, t) for name, p, t in levels if t]
    ungraded = [name for name, _p, t in levels if not t]
    # A dict carrying NO level count at all is a different claim from one whose
    # counts are all zero, and the two must not be collapsed. All-zero counts
    # come from the real adapter and mean the gate evaluated nothing; a dict
    # with none of the six keys simply never reported the breakdown, and the
    # decision it DID carry is still its own. So the breakdown is dropped rather
    # than printed as zeros, and the gap is named either way.
    counted = any(
        _g(d, k) is not None
        for k in (
            "n_overall_pass",
            "n_overall_total",
            "n_single_pass",
            "n_single_total",
            "n_intersect_pass",
            "n_intersect_total",
        )
    )
    if counted and not graded:
        return _could_not_check(
            d,
            "not one of the three gate levels evaluated a metric, so nothing was "
            "graded and this gate approves nothing.",
        )
    sev = "critical" if not appr else ("low" if warns else "info")
    if not counted:
        tail = _ungraded_tail(len(levels), "gate level(s) reported no metric count")
        sev = _floor_sev(sev)
        breakdown = ""
    else:
        tail = _ungraded_tail(
            len(ungraded),
            f"gate level(s) evaluated no metric ({', '.join(ungraded)})",
        )
        # The metrics the levels DID run but never graded, counted from the rows
        # above and stated separately from the levels that ran nothing at all.
        tail += _ungraded_tail(
            n_ungraded_metrics, "metric(s) on a graded level were never compared to a threshold"
        )
        if ungraded or n_ungraded_metrics:
            sev = _floor_sev(sev)
        breakdown = (
            " - " + ", ".join(f"{name} {p}/{t}" for name, p, t in graded) + " metrics passed"
        )
    return (
        f"Hierarchical gate {'APPROVED' if appr else 'BLOCKED'}{breakdown}; "
        f"{len(warns)} small-sample warning(s).{tail}",
        sev,
    )


def _fr_report_card(d):
    # Same defaulted approval as _fr_hierarchical, on the surface that is
    # pasted into a pull request: "Report card for model: approved, 0/0 metrics
    # pass" was written from an absent key. No decision in the dict, no verdict
    # in the prose.
    appr = _g(d, "approved")
    if appr is None:
        return _could_not_check(d, "no gate decision was supplied, so nothing was approved here.")
    metrics = _g(d, "metrics", []) or []
    # Same empty-denominator rule as the gate above, on the surface that is
    # pasted into a pull request: "0/0 metrics pass" beside "approved" is a
    # pass rate for a card that lists no metric. The verdict the caller did
    # supply is still stated; the fraction nobody computed is not.
    sev = "critical" if not appr else ("low" if _g(d, "card_warnings", []) else "info")
    verdict = "approved" if appr else "blocked"
    name = _g(d, "model_name", "model")
    n_blocking = len(_g(d, "blocking_reasons", []) or [])
    if not metrics:
        return (
            f"Report card for {name}: {verdict}, {n_blocking} blocking reason(s). "
            f"{_COULD_NOT_CHECK}: the card lists no metric, so there is no pass rate "
            f"behind that verdict.",
            _floor_sev(sev),
        )
    # THE DENOMINATOR, and this is the surface that gets pasted into a pull
    # request. report_card.svg badges a metric with no verdict NOT GRADED in
    # slate and prints "not measured" where its value would be, keeping it out
    # of the pass rate; `_count_true` reads the same None as "did not pass" and
    # `len(metrics)` puts it back in the denominator, so a card that graded two
    # metrics of three read "1/3 metrics pass" underneath a canvas showing one
    # pass, one fail and one row nobody measured. That is a fabricated breach in
    # a review comment, and a fabricated breach is not the safer error.
    graded = [m for m in metrics if _g(m, "passed") is not None]
    ungraded = len(metrics) - len(graded)
    if not graded:
        return (
            f"Report card for {name}: {verdict}, {n_blocking} blocking reason(s). "
            f"{_COULD_NOT_CHECK}: not one of the {len(metrics)} metric(s) on this card "
            f"was graded, so there is no pass rate behind that verdict.",
            _floor_sev(sev),
        )
    tail = _ungraded_tail(ungraded, "metric(s) on this card were never graded")
    if ungraded:
        sev = _floor_sev(sev)
    return (
        f"Report card for {name}: {verdict}, "
        f"{_count_true(graded, 'passed')}/{len(graded)} metrics pass, "
        f"{n_blocking} blocking reason(s).{tail}",
        sev,
    )


def _fr_robustness(d):
    # `overall_score` defaulted to 0, and 0 is the WORST score this chart can
    # carry: an absent score fell into the `s < 0.5` band and was read out as
    # "score 0.000" at severity HIGH, a fabricated breach for a run that
    # reported no score at all. A fabricated breach is not the safer error; it
    # sends someone after a fragility nobody measured. Absence is answered
    # before any band is applied.
    if _g(d, "overall_score") is None:
        return _could_not_check(
            d, "no robustness score was reported, so this run is neither robust nor fragile."
        )
    s = float(_g(d, "overall_score", 0) or 0)
    sg = _g(d, "subgroup", {}) or {}
    # _f_band keeps the displayed score in the same band as the raw score the
    # label/severity are derived from: 0.795 must not print as "0.80" beside
    # a MARGINAL verdict (0.80 passes the >= 0.8 band).
    #
    # The PARTIAL run. The score is banded over the tests that WERE graded
    # (part (a): adapters_robustness averages the graded permutation tests and
    # names the rest in its recommendations, `n_perm_ungraded`), so a run where
    # one test reported no p-value still reads "Robustness PASS (score 0.86)".
    # That is a fair verdict on what was measured and a silent one about what
    # was not, and silence next to a PASS reads as coverage.
    # The SENSITIVITY half of the same canvas was left out. A sensitivity check
    # that reported no score is ungraded exactly as a permutation test with no
    # p-value is, the adapter counts it (`n_sens_ungraded`), names it in the
    # recommendation panel and prints it beside the headline in
    # robustness_testing.svg, and the sentence covered neither.
    ungraded = _n(d, "n_perm_ungraded")
    ungraded_sens = _n(d, "n_sens_ungraded")
    sev = "high" if s < 0.5 else ("medium" if s < 0.8 else "info")
    tail = _ungraded_tail(ungraded, "permutation test(s) reported no p-value")
    tail += _ungraded_tail(ungraded_sens, "sensitivity check(s) reported no score")
    # The subgroup fraction, part (d) in the shape that prints the withheld
    # value itself. `n_flagged` is None whenever the audit reported no flagged
    # count (the canvas says so in words: "The audit of N subgroup(s) reported
    # no flagged count"), and `_g(sg, "n_flagged", 0)` returns that None because
    # the KEY IS PRESENT, so the sentence read "None/6 subgroups flagged". None
    # over six is not a count, it is a withheld numerator with a denominator
    # still attached, and a reader cannot tell it from a rendering fault.
    #
    # The ABSENT subgroup audit (`sg` is {}), which the note left here by the
    # previous wave described as printing "0/0 subgroups flagged". It did not:
    # `_g(sg, "n_flagged")` returns None, `_n(sg, "n_analyzed")` returns 0, and
    # `withheld` is 0 and therefore falsy, so the else arm ran and the sentence
    # this library emitted was
    #
    #     "Robustness ROBUST (score 0.85); None/0 subgroups flagged."
    #
    # A Python None formatted into a fraction over a zero denominator, on the
    # accessible layer, beside a canvas that draws no subgroup panel at all. It
    # is not a count a reader can act on and it is not distinguishable from a
    # rendering fault, which is the shape rule (d) exists to stop. The absence
    # is now stated in words, in the same third-state wording the withheld case
    # uses two lines above.
    #
    # CLOSED 2026-09-07, with its gating condition, in the same commit:
    # tests/test_audit_wave5_render.py::test_robustness_display_stays_in_verdict_band
    # asserted `ce.severity == "info"` for exactly this dict, which pinned the
    # defect. That test's own purpose is the DISPLAY band ("0.795" must not print
    # as "0.80"), and that half is untouched; only the severity it happened to
    # assert alongside moved.
    n_flagged, n_analyzed = _g(sg, "n_flagged"), _n(sg, "n_analyzed")
    no_audit = not n_analyzed and n_flagged is None
    withheld = n_analyzed and (n_flagged is None or isinstance(n_flagged, bool))
    if no_audit:
        subgroups = "no subgroup audit was reported, so no subgroup is flagged or cleared"
        # W-28, closed 2026-09-07. The sentence said this in words; the SEVERITY
        # a pipeline reads still came from the score band, so "no audit ran" was
        # graded `info`, which is exactly what a COMPLETE audit finding nothing
        # wrong is graded. Measured before the fix, all at score 0.85:
        #   subgroup {}                        -> info    (no audit ran)
        #   subgroup {n_analyzed: 4}           -> medium  (count withheld)
        #   subgroup {n_analyzed: 4, flagged 0} -> info   (real all-clear)
        # The weakest evidence of the three was graded like the strongest.
        sev = _floor_sev(sev)
    elif withheld:
        subgroups = f"the audit of {n_analyzed} subgroup(s) reported no flagged count"
        tail += _ungraded_tail(n_analyzed, "subgroup(s) carry no flagged verdict")
        ungraded_sens = ungraded_sens or n_analyzed
    else:
        subgroups = f"{n_flagged}/{n_analyzed} subgroups flagged"
    if ungraded or ungraded_sens:
        sev = _floor_sev(sev)
    return (
        f"Robustness {_g(d, 'overall_label', '?')} (score {_f_band(s, (0.5, 0.8))}); "
        f"{subgroups}.{tail}",
        sev,
    )


def _fr_ranking(d):
    # Same shape as the regression finder above: the ranking canvas grades its
    # headline into three states (adapters_ranking._headline), so "; all pass"
    # must not be written next to a NOT ASSESSED badge, and "0/0 metrics pass"
    # must not be written at all.
    if _not_assessable(d):
        return _could_not_check(d, "this report does not establish ranking fairness either way.")
    p, t = _g(d, "n_pass", 0), _g(d, "n_total", 0)
    fails = [_g(b, "name", "?") for b in (_g(d, "badges", []) or []) if _g(b, "label") == "FAIL"]
    # The PARTIAL run. The denominator here is already the graded population
    # (adapters_ranking sets `n_total` to `n_graded` on purpose, and says so),
    # so part (d) was met, and ranking_fairness.svg prints `n_ungraded` beside
    # its headline, so part (c) was met on the canvas. What was still written
    # underneath was "; all pass" and severity INFO, which is the unqualified
    # all-clear part (b) forbids: p == t over the graded subset is a clean sheet
    # for what was measured, and nothing in the sentence said the rest was not.
    ungraded = _n(d, "n_ungraded")
    if not ungraded and _g(d, "n_supplied") is not None:
        ungraded = max(0, _n(d, "n_supplied") - _n(d, "n_total"))
    # THE GROUP COUNT, which is the OTHER half of this page and was still the
    # raw one. `len(groups)` counts every group row the canvas DRAWS, and
    # adapters_ranking draws a group that reported no exposure without a bar,
    # keeps it out of the max-exposure scale and out of the exposure gap, and
    # says so in the subtitle at y=122 ("4 groups (1 reported no exposure)").
    # "across 4 groups" beside that claims a population the gap was never taken
    # over, which is part (d) on the group half: the fewer groups reported an
    # exposure, the wider the claim read.
    #
    # Counted from the SAME rows the template draws, by the SAME `measured`
    # flag, exactly as _fr_disparity_heatmap counts its blank rows. Only an
    # explicit False withdraws a row, so a dict whose groups are bare values or
    # predate the flag counts exactly what it always counted, which is the
    # over-correction control in code.
    groups = _g(d, "groups", []) or []
    ungraded_groups = sum(1 for g in groups if _g(g, "measured") is False)
    n_groups = len(groups) - ungraded_groups
    tail = _pass_tail(p, t, fails, ungraded or ungraded_groups)
    sev = "info" if not t or p == t else ("high" if p == 0 else "medium")
    if ungraded or ungraded_groups:
        sev = _floor_sev(sev)
    return (
        f"Ranking fairness: {p}/{t} metrics pass across {n_groups} groups{tail}."
        f"{_ungraded_tail(ungraded, 'ranking metric(s) were never compared to a threshold')}"
        f"{_ungraded_tail(ungraded_groups, 'group(s) reported no exposure')}",
        sev,
    )


def _fr_data_validation(d):
    nc = _g(d, "n_critical", 0) or 0
    nw = _g(d, "n_warning", 0) or 0
    # The PARTIAL run, part (d). The canvas carries three counters and no fourth,
    # so the adapter files an issue of UNRECOGNISED severity under WARNING
    # (adapters_validation: `n_warning = ... + n_unknown`) and then names the
    # count on the canvas, in the summary line ("(N of unrecognised severity)")
    # and again in the recommendation panel. This sentence took the merged
    # counter and said nothing, so an issue that was never graded was read out
    # as a warning the validator raised: the desc was the more confident of the
    # two surfaces again, and it is the only one a screen reader gets.
    #
    # Counted from the issue rows the table draws, which is the same population
    # the adapter counted, so a merged counter cannot hide an ungraded row here.
    issues = _g(d, "issues", []) or []
    ungraded = sum(1 for i in issues if str(_g(i, "bucket", "") or "").upper() == "UNKNOWN")
    graded_warnings = max(0, nw - ungraded)
    # `passed` DEFAULTED TO TRUE, so a validation dict that carried no verdict
    # at all was graded as one that ran and cleared: absent became "info", the
    # all-clear, on a chart whose whole job is to say whether the data may be
    # fitted. The three states are read from the key itself, and the missing
    # one is neither a pass nor a failure.
    passed = _g(d, "passed")
    sev = "critical" if nc > 0 else ("high" if passed is False else ("medium" if nw else "info"))
    tail = _ungraded_tail(ungraded, "issue(s) carry an unrecognised severity and were not graded")
    if passed is None:
        tail += (
            f" {_COULD_NOT_CHECK}: no overall validation verdict was supplied, so this "
            "run neither clears the data for training nor blocks it."
        )
        sev = _floor_sev(sev)
    if ungraded:
        sev = _floor_sev(sev)
    return (
        f"Data validation {_g(d, 'pass_label', '?')}: {nc} critical/error, "
        f"{graded_warnings} warning, {_g(d, 'n_info', 0)} info issue(s).{tail}",
        sev,
    )


def _fr_auto_discovery(d):
    nh, nv = _g(d, "n_high", 0) or 0, _g(d, "n_violations", 0) or 0
    # The PARTIAL run. A violation that reported no severity is not a violation
    # BELOW high severity, it is unplaced, so "(0 high-severity)" over rows
    # nobody graded is an emerald zero the adapter already refuses to print on
    # the canvas: adapters_discovery counts `n_ungraded`, gives the badge a
    # third branch for it, and keeps those rows out of its own "further
    # violation(s) recorded below high severity" line for exactly this reason.
    ungraded = _n(d, "n_ungraded")
    graded = max(0, nv - ungraded)
    high_clause = (
        f"{nh} high-severity" if not ungraded else f"{nh} of {graded} graded high-severity"
    )
    sev = "high" if nh else ("medium" if nv else "info")
    tail = _ungraded_tail(ungraded, "violation(s) reported no severity")
    if ungraded:
        sev = _floor_sev(sev)
    return (
        f"Auto-discovery: {len(_g(d, 'candidates', []) or [])} candidate attribute(s), {nv} "
        f"violation(s) ({high_clause}) across {len(_g(d, 'groups', []) or [])} group(s).{tail}",
        sev,
    )


def _fr_regression(d):
    # regression_fairness.svg already prints NOT ASSESSABLE when a disparity
    # metric was never compared to its threshold, but this finder was written
    # from n_pass/n_total alone. "0/0 disparity metrics pass ...; all pass" is
    # what it wrote underneath that banner, and `not t` even graded it "info".
    # Nothing measured is not an all-clear, so the third state is read from the
    # SAME key the template branches on.
    if _not_assessable(d):
        return _could_not_check(d, "this report does not establish prediction equity either way.")
    p, t = _g(d, "n_pass", 0), _g(d, "n_total", 0)
    badges = _g(d, "badges", []) or []
    fails = [_g(b, "name", "?") for b in badges if _g(b, "label") == "FAIL"]
    # The PARTIAL run, part (d). `n_total` is `len(badges)` in the adapter, which
    # counts every disparity metric SUPPLIED, the unchecked ones included, and
    # `n_pass` counts only the PASS labels. A metric that was never compared to
    # its threshold therefore sat in the denominator as a non-pass: with one
    # PASS, one FAIL and one NO DATA the sentence read "1/3 disparity metrics
    # pass", which reports two breaches where one was measured.
    #
    # This is the only path that reaches the fraction at all: with an unchecked
    # metric and NO failure the adapter sets `not_assessable` and the clause
    # above answers. The ungraded count is recovered from the badge list, and
    # only when that list is COMPLETE, because the adapter truncates it to four
    # while `n_total` counts them all; a truncated list cannot tell us how many
    # of the metrics it does not carry were graded, so the sentence is left as
    # it was rather than corrected with a number that is itself a guess.
    graded_labels = {"PASS", "FAIL"}
    ungraded = 0
    try:
        complete = len(badges) == int(t)
    except (TypeError, ValueError):
        complete = False
    if complete:
        ungraded = sum(1 for b in badges if _g(b, "label") not in graded_labels)
        t = max(0, int(t) - ungraded)
    # THE OTHER TWO POPULATIONS ON THE SAME PAGE, and the reason this finder was
    # still the more confident surface after the badge count was corrected.
    # regression_fairness.svg carries three panels, not one: the disparity
    # badges, the EFFECT SIZE pairs and the per-group RESIDUALS. A pair that
    # reported no effect size was never compared and a group that reported no
    # mean residual was never screened, and adapters_regression counts both
    # (`n_effects_unmeasured`, `n_residual_missing`), names them in the
    # Recommendations panel AND appends them to the subtitle at y=122, which is
    # the headline band: "4 metrics . 2 groups . 1 pair(s) not compared".
    #
    # This sentence read only n_pass/n_total, so the canvas said "1 pair(s) not
    # compared" while the <desc> said "4/4 disparity metrics pass across 2
    # groups; all pass" at severity INFO. Every disparity metric really did
    # pass; the page still measured less than the sentence claimed, and "all
    # pass" is the phrase a reader quotes.
    #
    # The badge fraction is left exactly as it is: those metrics WERE graded,
    # and part (d) is about the numbers, not about the sentence. What changes is
    # the unqualified all-clear (_pass_tail writes "every graded metric passes"
    # instead), the severity floor, and the two counts being stated.
    n_effects_unmeasured = _n(d, "n_effects_unmeasured")
    n_residual_missing = _n(d, "n_residual_missing")
    silent = ungraded + n_effects_unmeasured + n_residual_missing
    tail = _pass_tail(p, t, fails, silent)
    sev = "info" if not t or p == t else ("high" if p == 0 else "medium")
    if silent:
        sev = _floor_sev(sev)
    return (
        f"Regression fairness: {p}/{t} disparity metrics pass across "
        f"{len(_g(d, 'groups', []) or [])} groups{tail}."
        f"{_ungraded_tail(ungraded, 'disparity metric(s) were never compared to a threshold')}"
        f"{_ungraded_tail(n_effects_unmeasured, 'group pair(s) reported no effect size')}"
        f"{_ungraded_tail(n_residual_missing, 'group(s) reported no mean residual')}",
        sev,
    )


def _fr_reporting(d):
    status = str(_g(d, "status_label", "")).upper()
    sev = {"GREEN": "info", "YELLOW": "medium", "RED": "critical"}.get(status, "medium")
    # The PARTIAL run. adapters_reporting already narrows `n_metrics` to the
    # GRADED rows and says so at that key, so part (d) was met and the fraction
    # is honest. Part (c) was not: an unchecked row was named only in the
    # narrative panel, far down the canvas, while "0 breach(es) of 1" plus a
    # GREEN health status at severity "info" is the strongest all-clear this
    # dashboard can give and it is the sentence a screen reader reads out.
    #
    # Counted off the SAME `could_not_check` flag the metric rows and the
    # headline-band chip in reporting_dashboard.svg branch on, so the three
    # surfaces cannot drift apart.
    rows = _g(d, "metrics", []) or []
    unchecked = sum(1 for m in rows if _g(m, "could_not_check"))
    tail = _ungraded_tail(unchecked, "metric(s) were never compared to a threshold")
    if unchecked:
        sev = _floor_sev(sev)
    # "0/0 critical alerts" is the empty-denominator shape, and on this
    # dashboard it appeared on every quiet window: a fraction whose denominator
    # is the number of alerts that fired. A reader cannot tell it from the
    # withheld-numerator shape rule (d) is about, so a window that genuinely
    # raised nothing says that in words instead of in a fraction over zero.
    n_active = _n(d, "n_active_alerts")
    alerts = (
        "no active alert"
        if not n_active
        else f"{_g(d, 'n_critical_alerts', 0)}/{n_active} critical alerts"
    )
    return (
        f"{_g(d, 'report_tier', '')} report: health {_g(d, 'health_score', '?')}/100 ({status}); "
        f"{len(_g(d, 'breaches', []) or [])} breach(es) of {_g(d, 'n_metrics', 0)}, "
        f"{alerts}.{tail}",
        sev,
    )


_FINDERS: Dict[str, Callable[[Any], Tuple[Optional[str], str]]] = {
    "bias_audit": _fr_bias_audit,
    "calibration_report": _fr_calibration_report,
    "fairness_report": _fr_fairness_report,
    "cicd_pipeline": _fr_cicd,
    "radar_chart": _fr_radar,
    "disparity_heatmap": _fr_disparity_heatmap,
    "metrics_bar_chart": _fr_metrics_bar,
    "group_comparison": _fr_group_comparison,
    "effect_sizes": _fr_effect_sizes,
    "confidence_intervals": _fr_confidence_intervals,
    "reliability_diagram": _fr_reliability,
    "group_calibration": _fr_group_calibration,
    "calibration_disparity": _fr_calibration_disparity,
    "pareto_frontier": _fr_pareto,
    "correlation_heatmap": _fr_corr_heatmap,
    "correlation_matrix": _fr_corr_matrix,
    "proxy_risk": _fr_proxy_risk,
    "transformation_comparison": _fr_transformation,
    "intersectional_analysis": _fr_intersectional_analysis,
    "intersectional_disparity": _fr_intersectional_disparity,
    "training_report": _fr_training_report,
    "training_analysis_report": _fr_training_report,
    "method_comparison": _fr_method_comparison,
    "tradeoff_analysis": _fr_tradeoff,
    "threshold_optimization_report": _fr_threshold,
    "reweighting_comparison_report": _fr_reweighting,
    "fairness_detailed_report": _fr_detailed_report,
    "monitoring_dashboard": _fr_monitoring,
    "drift_report": _fr_drift,
    "alert_timeline": _fr_alert_timeline,
    "temporal_analysis": _fr_temporal,
    "experiment_results": _fr_experiment_results,
    "experiment_recommendation": _fr_experiment_recommendation,
    "power_analysis": _fr_power,
    "causal_decomposition": _fr_causal,
    "workflow_overview": _fr_workflow,
    "hierarchical_gate": _fr_hierarchical,
    "report_card": _fr_report_card,
    "robustness_testing": _fr_robustness,
    "ranking_fairness": _fr_ranking,
    "data_validation": _fr_data_validation,
    "auto_discovery": _fr_auto_discovery,
    "regression_fairness": _fr_regression,
    "reporting_dashboard": _fr_reporting,
}


#: What the accessible layer says when the finder for a chart RAISED, so this
#: run produced no finding at all. It is a could-not-check sentence in the
#: library's one spelling, so ``_withholds_the_verdict`` recognises it and
#: ``build_explanation`` swaps the curated action for the honest one.
_FINDER_FAILED = (
    f"{_COULD_NOT_CHECK}: this chart's summary could not be computed from the "
    f"data it was given, so no finding is stated and no verdict is implied."
)


def _finding_for(template: str, data: Any) -> Tuple[Optional[str], str]:
    """Return (finding, severity) for a template, never raising."""
    fn = _FINDERS.get(template)
    if fn is None:
        # generic fallback: use a subtitle if present.
        sub = _g(data, "subtitle") or _g(data, "summary")
        return (str(sub) if sub else None, "info")
    try:
        finding, severity = fn(data)
        finding = str(finding).strip() if finding else None
        return finding, _norm_sev(severity)
    except Exception:
        # A CRASHED FINDER IS NOT GOOD NEWS. Until 2026-09-30 this arm answered
        # ``(None, "info")``, and "info" is the benign end of the vocabulary:
        # ``metadata()["severity"]`` is what a pipeline reads INSTEAD of the
        # sentence, and ``caption()`` falls back to the curated concept prose,
        # so the artifact read like a described chart with nothing to report.
        #
        # MEASURED on this tree, with the metric fields stringified (what a
        # JSON or CSV round trip of a report gives you):
        #   build_explanation("reliability_diagram", {"ece": "0.5", ...})
        #     finding  None
        #     severity info
        #     caption  "Reliability Diagram: A calibration curve comparing
        #               predicted probability to observed frequency, with a
        #               confidence histogram. (severity: INFO)"
        #     action   "Recalibrate if ECE exceeds roughly 0.05."
        # _fr_reliability had died on ``'0.5' < 0.05``. The recommendation names
        # a threshold that was never compared to anything, and INFO is the
        # unqualified all-clear the sentence is forbidden to make.
        # ``intersectional_analysis`` was the same, and ``alert_timeline`` dies
        # on ``'0.5' - 1``; three of the 45 finders on one hostile shape.
        #
        # The severity floor is _UNKNOWN_SEV for the reason given at that
        # constant: not benign, and not "high" either, because nothing was
        # measured to be wrong.
        return _FINDER_FAILED, _UNKNOWN_SEV


# ── Public structured explanation ───────────────────────────────────────────


@dataclass
class ChartExplanation:
    template: str
    title: str
    concept: str
    reading: str
    action: str
    finding: Optional[str]
    severity: str

    def paragraph(self) -> str:
        """The on-canvas explanation: concept -> how to read -> finding -> action."""
        parts = [self.concept, self.reading]
        if self.finding:
            parts.append(self.finding)
        if self.action:
            parts.append(self.action)
        return " ".join(p.strip() for p in parts if p and p.strip())

    def caption(self) -> str:
        """One-line accessible summary: what it shows + key finding + severity."""
        lead = self.finding or _first_sentence(self.concept)
        return f"{self.title}: {lead} (severity: {self.severity.upper()})"

    def metadata(self) -> Dict[str, Any]:
        return {
            "library": "vfairness",
            "chart": self.template,
            "title": self.title,
            "severity": self.severity,
            "concept": self.concept,
            "howToRead": self.reading,
            "finding": self.finding,
            "recommendation": self.action,
        }


def _first_sentence(text: str) -> str:
    text = (text or "").strip()
    dot = text.find(". ")
    return text[: dot + 1] if dot != -1 else text


def build_explanation(template: str, data: Any) -> ChartExplanation:
    """Build a structured explanation for a rendered chart. Never raises."""
    meta = CHART_META.get(template, _GENERIC_META)
    finding, severity = _finding_for(template, data)
    action = str(meta.get("action", _GENERIC_META["action"]))
    # The curated action assumes the chart produced a verdict ("bring the
    # failing metrics under threshold"). On a run that measured nothing that
    # instruction is not just useless, it implies a measurement happened.
    #
    # The finding is forced to the same state HERE, not only inside each
    # finder, because a per-finder rule is a rule that can be forgotten: on
    # 2026-08-27 _fr_reliability and _fr_regression had been left out of the
    # sweep, so a NOT ASSESSABLE reliability diagram carried the paragraph
    # "Model is WELL CALIBRATED (ECE 0.000, MCE 0.000)" and a NOT ASSESSABLE
    # regression report carried "0/0 disparity metrics pass ...; all pass",
    # both from adapter DEFAULTS rather than from anything measured. The badge
    # and the accessible text now derive from one signal for EVERY chart,
    # including any chart added after this line was written. A finder that has
    # already written its own could-not-check clause keeps its wording and its
    # severity; nothing else may outrank the canvas.
    try:
        if _not_assessable(data):
            action = _NOT_ASSESSABLE_ACTION
            if not _is_could_not_check(finding):
                finding, severity = _could_not_check(data, _GENERIC_NOT_ASSESSABLE)
                severity = _norm_sev(severity)
        elif _withholds_the_verdict(finding):
            # The other direction, which the net above cannot see: the FINDER
            # refused to state a result while the data dict carries no
            # not-assessable flag. The curated action assumes a verdict exists
            # ("Follow the recommendation"), and printing it under a sentence
            # that says no recommendation was made makes the action line the
            # more confident of the two.
            #
            # BGL5 (2026-09-27): the test here was `_is_could_not_check`, a
            # STARTSWITH, and four finders write the subject before the marker, so
            # this branch could not fire for them. Measured on a real monitoring
            # window carrying zero fairness metrics, the dashboard published
            # "Alert status: COULD NOT CHECK, this window recorded no alert
            # verdict ..." beside "Triage the alerting metrics; if several alert
            # at once, treat it as an incident." It now reads
            # _COULD_NOT_CHECK_ACTION. `_withholds_the_verdict` matches only a
            # bare subject label before the marker, so a mid-sentence marker on an
            # otherwise measured chart keeps its curated action, and the net above
            # keeps `_is_could_not_check` so it still owns the sentence on a chart
            # that measured nothing at all.
            action = _COULD_NOT_CHECK_ACTION
    except Exception:  # noqa: BLE001 - an explanation must never break a render
        pass
    # A GROUP dropped before any comparison ran is the one stratum the chart
    # says nothing about, and the sentence must say so even when the rest of
    # the run WAS measurable. Until now `_excluded_clause` was reachable only
    # from `_could_not_check`, i.e. only on runs that measured nothing, so the
    # partial case (the commoner one) disclosed nothing at all.
    #
    # Measured on a 130-row report, A=60 and B=60 selected identically while
    # C=10 was never selected at all and fell below min_group_size=30, so every
    # metric was computed over A and B only:
    #   radar        "Overall fairness status: Fair." severity INFO
    #   metrics bar  "5 of 5 checked fairness metric(s) passed." severity INFO
    #   heatmap      "Low disparity across groups across 2 group(s) and 5
    #                 metric(s)." severity INFO
    #   effect sizes "Largest effect |d|=0.00 (Negligible)." severity INFO
    # `report_assessability` had already resolved excluded=['C (n=10)'] and
    # every one of those adapters passed it in as `excluded_groups`, which no
    # template reads: a perfect all-clear produced by removing the only group
    # that could have made it unequal, and none of those four sentences named C
    # (nor did group_comparison's, which discloses on its canvas but not here).
    #
    # The severity floor is part (b) of the headline rule in its
    # machine-readable form: `severity` is what a pipeline reads INSTEAD of the
    # sentence, and "info" there is the unqualified all-clear the sentence is
    # forbidden to make. A measured breach keeps its higher severity.
    try:
        clause = _excluded_clause(data)
        if clause and clause.strip() not in str(finding or ""):
            finding = f"{finding}{clause}" if finding else clause.strip()
            severity = _floor_sev(severity)
    except Exception:  # noqa: BLE001 - an explanation must never break a render
        pass
    # A demonstration render is watermarked EXAMPLE on the canvas, but the
    # <desc> is the whole artifact for a screen-reader user and for anything
    # that parses the SVG, and there the demo fixture read exactly like a
    # measurement: "Report card for model: approved, 3/3 metrics pass". The
    # marker leads the sentence so it survives truncation, and it is added HERE
    # rather than in each finder so a chart added later cannot miss it.
    try:
        if _g(data, "is_example") and finding and not str(finding).startswith(_EXAMPLE_PREFIX):
            finding = f"{_EXAMPLE_PREFIX}{finding}"
    except Exception:  # noqa: BLE001 - an explanation must never break a render
        pass
    return ChartExplanation(
        template=template,
        title=str(meta.get("title", _GENERIC_META["title"])),
        concept=str(meta.get("concept", _GENERIC_META["concept"])),
        reading=str(meta.get("reading", _GENERIC_META["reading"])),
        action=action,
        finding=finding,
        severity=severity,
    )


# ── Accessibility / machine-readable injection ──────────────────────────────

_SVG_OPEN_RE = re.compile(r"<svg\b", re.IGNORECASE)
_SVG_TAG_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)


def inject_accessibility(svg: str, ce: ChartExplanation) -> str:
    """
    Add ``role="img"`` and insert ``<title>``, ``<desc>`` and a ``<metadata>``
    JSON block as the first children of the root ``<svg>``. If *svg* has no
    ``<svg>`` element (e.g. an adapter returned ""), it is returned unchanged.
    """
    if not svg or "<svg" not in svg:
        return svg

    title = html.escape(ce.title)
    desc = html.escape(ce.caption())
    meta_json = json.dumps(ce.metadata(), ensure_ascii=False).replace("]]>", "]]&gt;")
    block = (
        f"<title>{title}</title>"
        f"<desc>{desc}</desc>"
        f'<metadata id="vfairness-explanation"><![CDATA[{meta_json}]]></metadata>'
    )

    # 1. add role="img" to the opening tag if not already present
    def _add_role(m: "re.Match[str]") -> str:
        tag = m.group(0)
        return tag if "role=" in tag else tag[:4] + ' role="img"' + tag[4:]

    svg = _SVG_TAG_RE.sub(_add_role, svg, count=1)

    # 2. insert the a11y block right after the opening <svg ...> tag
    svg = _SVG_TAG_RE.sub(lambda m: m.group(0) + block, svg, count=1)
    return svg
