"""
SVG Rendering Adapters for Experimentation & A/B Testing Module.

This module provides adapters for converting experimentation results
into SVG visualizations.

Adapters Implemented:
    - experiment_results_to_svg: Render FairnessExperiment results (forest plot)
    - experiment_recommendation_to_svg: Render ExperimentAnalysis recommendation
    - power_analysis_to_svg: Render FairnessPowerAnalyzer results
"""

import math
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from .._triage import is_flag
from .adapters_fairness import _is_missing_level
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

# Shown wherever an interval is absent. Never a number: a confidence interval is
# COMPUTED or it is missing, and "[0.0000, 0.0000]" is neither.
CI_NOT_COMPUTED = "CI not computed"

# Shown wherever a statistical TEST is absent, for the same reason. A p-value of
# 1.0 is not "no p-value": it is the most confident "no effect" the scale can
# express, so defaulting to it manufactured a NOT SIGNIFICANT badge and a green
# HOMOGENEOUS card out of an experiment that ran neither test.
P_NOT_COMPUTED = "not computed"
COULD_NOT_CHECK_TEXT = "COULD NOT CHECK"


def _finite(value: Any) -> Optional[float]:
    """Coerce to a real, finite float, or None. NaN, inf, None and junk give None.

    READINESS-6, 2026-09-10. A BOOL is refused, which the sentence above always
    claimed and the code never did: ``float(True)`` is 1.0, a finite number that
    passes every check here. numpy's ``np.bool_`` is the half that gets missed,
    because it is not a Python bool, so a boolean column read out of a DataFrame
    walked straight through. A numeric STRING is still accepted, deliberately:
    these rows arrive from JSON and CSV where "0.5" is a real measurement that
    was serialised, and refusing it would discard evidence.
    """
    if is_flag(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


#: The fields a causal decomposition reports a NUMBER in. Used only to tell an
#: absent decomposition apart from a decomposition that measured zero.
_CAUSAL_EFFECT_FIELDS = (
    "total_effect",
    "direct_effect",
    "indirect_effect",
    "proportion_mediated",
)


def _decomposition_measured(dec: Dict[str, Any]) -> bool:
    """True when the decomposition dict carries at least one measurement.

    A supplied ZERO is a measurement and keeps rendering as 0.0000: an estimated
    indirect effect of exactly 0.0 is the finding "this mediator carries none of
    the gap", and suppressing it would swap one false claim for another. Only an
    ABSENT field is an absence.

    This exists because ``causal_decomposition_to_svg({})`` printed Direct
    0.0000, Indirect 0.0000 and Total 0.0000 under the heading EFFECT
    DECOMPOSITION, drew a 0% mediation gauge, and rendered four red crosses
    under "0/4 Baron-Kenny steps satisfied". Every one of those came from a
    ``.get(key, 0)`` default, so a decomposition that never ran read as a
    definite negative result: mediation tested and not found. That is a
    stronger claim than the data supports, and it is the opposite of the
    absence it actually represents.
    """
    if any(_finite(dec.get(k)) is not None for k in _CAUSAL_EFFECT_FIELDS):
        return True
    steps = dec.get("steps_satisfied")
    if isinstance(steps, dict) and steps:
        return True
    return bool(str(dec.get("mediator") or "").strip())


def _interval(lower: Any, upper: Any) -> Optional[Tuple[float, float]]:
    """The pair of bounds, or None when either bound is absent or unusable.

    Half an interval is not an interval: one computed bound and one invented
    zero is still a fabricated band, so both must be present.

    This exists because ``eff.get("ci_lower", 0)`` defaulted a MISSING bound to
    0. An effect dict with no interval rendered "[0.0000, 0.0000]" under a
    heading reading "95% CONFIDENCE INTERVALS", and fed 0,0 into the forest-plot
    scale so the whisker was drawn on the zero line. Both are claims the data
    never made.
    """
    lo = _finite(lower)
    hi = _finite(upper)
    if lo is None or hi is None:
        return None
    return (lo, hi)


if TYPE_CHECKING:
    from ..operations.experimentation.analysis import (
        CausalDecomposition,
        ExperimentRecommendation,
        TemporalStabilityResult,
    )
    from ..operations.experimentation.experiment import ExperimentResult


def experiment_results_to_svg(
    result: Union["ExperimentResult", Dict[str, Any]],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render A/B test experiment results as an SVG dashboard with forest plot.

    Visualises the overall treatment effect, per-intersection effects table,
    forest plot with confidence intervals, heterogeneity test, and power
    summary.

    Args:
        result: ExperimentResult from FairnessExperiment.run_full_analysis(),
                or a dictionary with the required fields.
        save_path: Optional path to save the SVG file.

    Returns:
        SVG markup string.

    Example:
        >>> from vfairness.operations.experimentation import FairnessExperiment
        >>> from vfairness.rendering import experiment_results_to_svg
        >>>
        >>> exp = FairnessExperiment(control, treatment, ['gender', 'age'], 'outcome')
        >>> result = exp.run_full_analysis()
        >>> svg = experiment_results_to_svg(result, save_path='experiment.svg')
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    # Handle both result objects and dictionaries
    if hasattr(result, "__dataclass_fields__"):
        data = _experiment_result_to_dict(result)
    elif isinstance(result, dict):
        data = result
    else:
        data = {}

    # Extract intersection effects
    effects_raw = data.get("intersection_effects", [])
    effects: List[Dict[str, Any]] = []

    for ie in effects_raw:
        eff: Dict[str, Any]
        if isinstance(ie, dict):
            eff = cast(Dict[str, Any], ie)
        elif hasattr(ie, "__dataclass_fields__"):
            eff = {
                "intersection": ie.intersection,
                "effect": ie.effect,
                "effect_size_d": ie.effect_size_d,
                "ci_lower": ie.ci_lower,
                "ci_upper": ie.ci_upper,
                "p_value": ie.p_value,
                "n_control": ie.n_control,
                "n_treatment": ie.n_treatment,
                "significant": ie.significant,
                "powered": ie.powered,
            }
        else:
            continue

        # `.get(key, default)` DOES NOT FIRE when the key is PRESENT holding
        # None, so a row carrying ``{"intersection": None}`` skipped the
        # ``("unknown",)`` default and `str(None)` MINTED the label "None".
        # Measured 2026-09-30 on power_analysis_to_svg: the canvas drew a row
        # and a bar labelled "None" with a real power percentage beside it, in
        # the font and the position of a real subgroup. pd.NA and pd.NaT print
        # "<NA>" and "NaT" through the same line. An unlabelled row gets the
        # SAME word an absent key already got, so nothing that was working
        # changes.
        name = eff.get("intersection")
        if name is None or _is_missing_level(name):
            name = ("unknown",)
        if isinstance(name, (tuple, list)):
            name = " × ".join(str(x) for x in name)

        # The same three states the panels above this table already have, now
        # applied PER ROW. `eff.get("effect_size_d", eff.get("effect", 0))` was
        # the last confirmed fabrication in this module: an intersection that
        # reported nothing rendered "+0.0000" in the EFFECT column, was PLOTTED
        # as a point on the forest plot's zero line (an effect of exactly no
        # difference, the most specific claim that plot can make), showed N 0,
        # and was badged UNDERPOWERED, which says the subgroup WAS sized and
        # found short of data. None of that was measured, and the zeros then fed
        # the arm totals and the "k of n adequately powered" count beneath.
        #
        # A SUPPLIED zero stays a measurement and still renders as +0.0000: only
        # an absent, unparseable, NaN or infinite value is an absence.
        raw_effect = eff.get("effect_size_d")
        if raw_effect is None:
            raw_effect = eff.get("effect")
        effect_val = _finite(raw_effect)
        effect_known = effect_val is not None

        band = _interval(eff.get("ci_lower"), eff.get("ci_upper"))
        # An absent p-value stays absent, exactly like an absent CI bound: it is
        # never defaulted to 1.0 and never used to derive a significance verdict.
        p_val = _finite(eff.get("p_value"))

        # Two arms, either of which may be absent. A row that reported one arm
        # did report a sample size; a row that reported neither did not report a
        # sample size of 0.
        # The `is not None` is checked first for the type checker's benefit as
        # well as the reader's: `_finite` already rejects None, but a checker
        # cannot see that through a call, so without it the sum below is a
        # sum over a list that might hold None.
        n_parts = [
            v
            for v in (eff.get("n_control"), eff.get("n_treatment"))
            if v is not None and _finite(v) is not None
        ]
        n_known = bool(n_parts)
        n_total = sum(n_parts) if n_parts else 0

        # POWERED and UNDERPOWERED are both verdicts about this subgroup, and
        # there is nothing to derive one from: no reported flag, no verdict.
        raw_powered = eff.get("powered")
        powered_known = raw_powered is not None
        powered = bool(raw_powered)

        # Significance keeps its old derivation from a KNOWN p-value, and gains
        # the third state the derivation could not express.
        raw_significant = eff.get("significant")
        if raw_significant is not None:
            significant = bool(raw_significant)
            significant_known = True
        elif p_val is not None:
            significant = p_val < 0.05
            significant_known = True
        else:
            significant = False
            significant_known = False

        effects.append(
            {
                "name": str(name),
                "effect_known": effect_known,
                "effect": f"{effect_val:+.4f}" if effect_known else P_NOT_COMPUTED,
                "effect_raw": effect_val,
                "has_ci": band is not None,
                "ci_display": (f"[{band[0]:.4f}, {band[1]:.4f}]" if band else CI_NOT_COMPUTED),
                "ci_lower": f"{band[0]:.4f}" if band else "",
                "ci_upper": f"{band[1]:.4f}" if band else "",
                "ci_lower_raw": band[0] if band else None,
                "ci_upper_raw": band[1] if band else None,
                "has_p": p_val is not None,
                "p_value": p_val,
                "p_display": (
                    P_NOT_COMPUTED
                    if p_val is None
                    else (f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001")
                ),
                "n_known": n_known,
                "n_total": n_total,
                "n_display": str(n_total) if n_known else "not reported",
                "significant": significant,
                "significant_known": significant_known,
                "powered": powered,
                "powered_known": powered_known,
                # True when the row reported ANY measurement. A row that
                # reported none of them is not a row of zeros.
                "row_measured": bool(
                    effect_known
                    or band is not None
                    or p_val is not None
                    or n_known
                    or powered_known
                    or significant_known
                ),
            }
        )

    # Compute forest plot scale. A missing bound, and now a missing effect,
    # contribute NOTHING to the scale: feeding either in as 0 both invented a
    # data point and dragged the axis toward zero, so the real effects were
    # squashed against the centre line.
    all_vals = []
    for e in effects:
        if e["effect_known"]:
            all_vals.append(e["effect_raw"])
        if e["has_ci"]:
            all_vals.extend([e["ci_lower_raw"], e["ci_upper_raw"]])
    if all_vals:
        abs_max = max(abs(v) for v in all_vals) * 1.2
    else:
        abs_max = 1.0
    if abs_max < 0.01:
        abs_max = 0.5

    # Overall effect.
    #
    # Three states for the HEADLINE too, not only for the panels beneath it.
    # `data.get("overall_effect", 0)` printed "+0.0000" in the largest type on
    # the canvas, directly beside an honest COULD NOT CHECK p-value badge and an
    # honest COULD NOT CHECK heterogeneity card. That pairing is worse than
    # either failure alone: the two truthful panels make the fabricated point
    # estimate read as the one number that WAS measured, and a reader takes the
    # headline before anything else. A point estimate is computed or it is
    # missing. A supplied 0.0 is a measurement (the treatment moved nothing) and
    # still renders as +0.0000; only an absent, unparseable, NaN or infinite
    # value is an absence.
    overall_effect_val = _finite(data.get("overall_effect"))
    overall_effect_known = overall_effect_val is not None
    overall_ci = data.get("overall_ci")
    if isinstance(overall_ci, (list, tuple)) and len(overall_ci) >= 2:
        overall_band = _interval(overall_ci[0], overall_ci[1])
    else:
        overall_band = None

    # Three states for the overall test, never two. `data.get(..., 1.0)` turned
    # "no test was reported" into "p = 1.0000", and the template rendered a
    # confident grey NOT SIGNIFICANT badge from it. A verdict requires a
    # measurement.
    overall_p = _finite(data.get("overall_p_value"))
    overall_p_known = overall_p is not None
    # Narrow on `is not None` directly rather than through the bool above: a
    # plain bool carries no type information, so a checker cannot see that
    # overall_p is a float here. Same behaviour, expressed checkably.
    overall_significant = bool(overall_p is not None and overall_p < 0.05)

    # Same for the heterogeneity test. `heterogeneity_detected` defaulting to
    # False painted the panel green and captioned it HOMOGENEOUS, which is a
    # positive finding about between-group consistency, for an experiment that
    # never ran the test. The verdict and its p-value are tracked separately: a
    # reported verdict with no p-value is still a verdict.
    het_detected_raw = data.get("heterogeneity_detected")
    heterogeneity_known = het_detected_raw is not None
    heterogeneity_p = _finite(data.get("heterogeneity_p_value"))

    # Count powered intersections. Supplied / graded / powered are three
    # different counts, and the denominator of a power verdict is the GRADED
    # population: "1 of 2 intersections adequately powered" over a second row
    # that carried no power verdict says that second subgroup was sized and
    # found short. The convention matches power_analysis_to_svg below, where
    # `n_total` is the assessed population rather than the number of rows.
    n_powered = sum(1 for e in effects if e["powered"])
    n_power_graded = sum(1 for e in effects if e["powered_known"])
    n_power_ungraded = len(effects) - n_power_graded

    # Compute total sample sizes. Only the rows that REPORTED an arm count are
    # added, and the row's own absence is what decides it: a row contributing 0
    # in silence understated the total and, with no row reporting at all,
    # printed "N CONTROL 0" for two arms nobody counted.
    n_control = 0
    n_treatment = 0
    n_control_known = False
    n_treatment_known = False
    for ie in effects_raw:
        raw_ctrl = ie.get("n_control") if isinstance(ie, dict) else getattr(ie, "n_control", None)
        raw_trt = (
            ie.get("n_treatment") if isinstance(ie, dict) else getattr(ie, "n_treatment", None)
        )
        # The raw value is added, not the coerced float, so a count of 610 keeps
        # printing as "610" rather than "610.0". `is not None` is checked first
        # so a checker can see that too (see the note in the row loop above).
        if raw_ctrl is not None and _finite(raw_ctrl) is not None:
            n_control += raw_ctrl
            n_control_known = True
        if raw_trt is not None and _finite(raw_trt) is not None:
            n_treatment += raw_trt
            n_treatment_known = True

    # Nothing at all was supplied: no estimate, no interval, no test, not one
    # per-intersection row. The canvas then has to say so where a reader sees
    # it, and the accessible description has to say it too (through
    # `not_assessable`, which rendering.explain.build_explanation reads to
    # replace both the finding sentence and the action line). A canvas that
    # carries even one measurement is NOT flagged: its real rows keep their real
    # prose, and only the absent parts read could-not-check.
    # A row that reported nothing does not count as something on the canvas
    # either: `or effects` alone let one empty intersection dict keep the
    # canvas out of the could-not-check state it belongs in.
    nothing_measured = not (
        overall_effect_known
        or any(e["row_measured"] for e in effects)
        or overall_band is not None
        or overall_p_known
        or heterogeneity_known
    )

    # A count is a claim about what was analysed, so it needs the same three
    # states. For an empty call these all read 0, and "0 / 0 intersections
    # adequately powered", "N CONTROL 0" and "INTERSECTIONS 0" are findings: an
    # experiment with no adequately powered subgroup and no samples in either
    # arm. None of that was reported. A REPORTED zero still renders as 0.
    n_intersections_known = data.get("n_intersections") is not None or bool(effects)
    design_known = data.get("design_type") is not None

    # The power panel's denominator. With every supplied row graded this is the
    # experiment's own intersection count, exactly as before. With any row
    # ungraded it falls back to the graded population, and the panel says how
    # many rows were left out; with nothing graded there is no fraction to
    # print at all.
    _reported_n_intersections = data.get("n_intersections")
    if not isinstance(_reported_n_intersections, int) or isinstance(
        _reported_n_intersections, bool
    ):
        _reported_n_intersections = len(effects)
    n_power_denom = (
        _reported_n_intersections if (effects and n_power_ungraded == 0) else n_power_graded
    )

    # Attribute names
    metadata = data.get("metadata", {})
    attr_names = metadata.get("protected_attributes", [])
    if isinstance(attr_names, (list, tuple)):
        attr_str = ", ".join(str(a) for a in attr_names)
    else:
        attr_str = str(attr_names) if attr_names else "N/A"

    design_type_raw = data.get("design_type", "independent")
    if hasattr(design_type_raw, "value"):
        design_type = str(design_type_raw.value)
    else:
        design_type = str(design_type_raw).replace("_", " ").title()

    _excluded_raw = _finite(data.get("n_excluded"))
    _excluded_count = int(_excluded_raw) if _excluded_raw is not None else 0

    template_data = {
        "title": data.get("title", "A/B Test: Experiment Results"),
        "timestamp": data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "design_type": design_type,
        "n_intersections": data.get("n_intersections", len(effects)),
        "n_intersections_known": n_intersections_known,
        "n_control_known": n_control_known,
        "n_treatment_known": n_treatment_known,
        "design_known": design_known,
        "nothing_measured": nothing_measured,
        # Read by rendering.explain.build_explanation, which replaces both the
        # accessible <desc> and the on-canvas action line whenever it is set.
        # Without it the <desc> read "Overall treatment effect +0.0000 across 0
        # intersections" for an empty dict, so the fabricated headline reached
        # every reader who cannot see the canvas, and the action line told them
        # to inspect heterogeneous subgroups of an experiment that never ran.
        "not_assessable": nothing_measured,
        "not_assessable_reason": (
            ""
            if not nothing_measured
            else (
                "No treatment effect, no interval, no p-value, no heterogeneity verdict "
                "and no per-intersection row carrying a measurement were supplied to "
                "experiment_results_to_svg, so nothing was estimated and nothing was "
                "tested on this run. This is not a finding of no effect: no effect was "
                "measured."
            )
        ),
        "attribute_names": attr_str,
        "overall_effect_known": overall_effect_known,
        # P_NOT_COMPUTED, never a formatted zero: this string is also what the
        # accessible description prints for the effect.
        "overall_effect": (
            f"{overall_effect_val:+.4f}" if overall_effect_known else P_NOT_COMPUTED
        ),
        "overall_has_ci": overall_band is not None,
        "overall_ci_display": (
            f"95% CI [{overall_band[0]:.4f}, {overall_band[1]:.4f}]"
            if overall_band
            else CI_NOT_COMPUTED
        ),
        "overall_ci_lower": f"{overall_band[0]:.4f}" if overall_band else "",
        "overall_ci_upper": f"{overall_band[1]:.4f}" if overall_band else "",
        "overall_p_known": overall_p_known,
        "overall_p": overall_p,
        "overall_p_display": (
            P_NOT_COMPUTED
            if overall_p is None
            else (f"{overall_p:.4f}" if overall_p >= 0.0001 else "<0.0001")
        ),
        "overall_significant": overall_significant,
        "effects": effects[:12],
        # Only the rows that reported an effect are drawn. A marker placed at
        # the zero line is the plot's most specific claim ("this subgroup moved
        # by exactly nothing"), so a row with no estimate is left off and
        # counted here instead.
        "forest_effects": [e for e in effects[:12] if e["effect_known"]],
        "forest_omitted": sum(1 for e in effects[:12] if not e["effect_known"]),
        "forest_abs_max": abs_max,
        "forest_min": f"{-abs_max:.2f}",
        "forest_max": f"{abs_max:.2f}",
        "forest_q1": f"{-abs_max / 2:.2f}",
        "forest_q3": f"{abs_max / 2:.2f}",
        "heterogeneity_known": heterogeneity_known,
        "heterogeneity_detected": bool(het_detected_raw),
        "heterogeneity_p_known": heterogeneity_p is not None,
        "heterogeneity_p": heterogeneity_p,
        "heterogeneity_p_display": (
            P_NOT_COMPUTED if heterogeneity_p is None else f"{heterogeneity_p:.4f}"
        ),
        "could_not_check_text": COULD_NOT_CHECK_TEXT,
        "n_powered": n_powered,
        "n_power_graded": n_power_graded,
        "n_power_ungraded": n_power_ungraded,
        "n_power_denom": n_power_denom,
        # The one default on this canvas that is NOT a measurement: the template
        # only ever asks `n_excluded > 0`, to decide whether to draw the
        # "N excluded (insufficient data)" line, and prints no number when it is
        # 0. An absent exclusion count therefore draws no claim either way, so
        # the zero stays. It is coerced because `None > 0` raises inside Jinja,
        # and the except below swallows that into an EMPTY STRING: a partial row
        # must never cost the caller the whole chart.
        "n_excluded": _excluded_count,
        "n_control": n_control,
        "n_treatment": n_treatment,
    }

    try:
        template_data["explanation"] = explanation
        svg = render_svg("experiment_results", template_data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception:
        # READINESS-6, 2026-09-10. This warned and returned "".
        #
        # `raise_jinja2_missing` in engine.py records the decision this reverses,
        # taken 2026-08-28 for the missing-BACKEND case: "their docstrings
        # document the return as 'SVG markup string', so a caller doing
        # open(p, "w").write(svg) wrote a ZERO-BYTE file and read it as this
        # run's report ... An empty string is not markup, and 'could not render'
        # must not be indistinguishable from 'rendered nothing worth showing'."
        # That fix funnelled 16 adapters through a raise and stopped there. The
        # general render-FAILURE path in this wrapper kept the swallow, and
        # render_svg raises through it, so the refusal was caught here anyway.
        #
        # Measured that day on a real failure (incomplete template data):
        #     returned '' (len 0), one warning, and a caller writing it got a
        #     0-byte report; THREE failed renders produced ONE warning, because
        #     Python shows a warning once per process. So a batch job warns once
        #     and then emits nothing, silently, for every report after it.
        #
        # Re-raised bare, which preserves the original type and traceback: the
        # cause is usually a jinja2 UndefinedError naming the missing key, which
        # is far more use to a caller than a warning they will not see.
        raise


def experiment_recommendation_to_svg(
    recommendation: Union["ExperimentRecommendation", Dict[str, Any]],
    experiment_result: Union["ExperimentResult", Dict[str, Any], None] = None,
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render an experiment decision recommendation as an SVG dashboard.

    Visualises the recommended decision (deploy, keep control, extend, or
    investigate), confidence score, reasoning, trade-offs, and caveats.

    Args:
        recommendation: ExperimentRecommendation from
            ExperimentAnalysis.decision_recommendation(), or a dict.
        experiment_result: Optional ExperimentResult to enrich the report
            with experiment metrics.
        save_path: Optional path to save the SVG file.

    Returns:
        SVG markup string.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    # Unpack recommendation
    rec_data: Dict[str, Any]
    if hasattr(recommendation, "__dataclass_fields__"):
        rec = cast("ExperimentRecommendation", recommendation)
        rec_data = {
            "decision": rec.decision.value if hasattr(rec.decision, "value") else str(rec.decision),
            "confidence": rec.confidence,
            "reasoning": list(rec.reasoning) if rec.reasoning else [],
            "trade_offs": dict(rec.trade_offs) if rec.trade_offs else {},
            "caveats": list(rec.caveats) if rec.caveats else [],
        }
    elif isinstance(recommendation, dict):
        rec_data = recommendation
    else:
        rec_data = {}

    # Three states, never two. `rec_data.get("decision", "INVESTIGATE_FURTHER")`
    # manufactured a recommendation out of an absent field, and the template
    # rendered it as the headline badge in red with a red accent bar. It is the
    # most confident sentence on the page and nobody had made it: the whole
    # point of this canvas is the decision, so defaulting it is the one thing
    # this adapter must never do. An absent decision is COULD NOT CHECK.
    raw_decision: Any = rec_data.get("decision")
    enum_value = getattr(raw_decision, "value", None)
    if enum_value is not None:
        raw_decision = enum_value
    decision = str(raw_decision).upper().replace(" ", "_").strip() if raw_decision else ""
    decision_known = bool(decision)

    # A confidence of 0 is a real value (no confidence at all) and must keep
    # rendering as 0%, so the absence is read off the KEY, not off falsiness.
    confidence_value = _finite(rec_data.get("confidence"))
    confidence_known = confidence_value is not None
    confidence = confidence_value if confidence_value is not None else 0.0

    reasoning = rec_data.get("reasoning", [])
    if not isinstance(reasoning, list):
        reasoning = [str(reasoning)]

    trade_offs_raw = rec_data.get("trade_offs", {})
    tradeoffs: List[Dict[str, str]] = []
    if isinstance(trade_offs_raw, dict):
        for k, v in trade_offs_raw.items():
            tradeoffs.append({"name": str(k), "value": str(v)})
    elif isinstance(trade_offs_raw, list):
        for item in trade_offs_raw:
            if isinstance(item, dict):
                tradeoffs.append(
                    {
                        "name": str(item.get("name", item.get("key", ""))),
                        "value": str(item.get("value", item.get("description", ""))),
                    }
                )

    caveats = rec_data.get("caveats", [])
    if not isinstance(caveats, list):
        caveats = [str(caveats)]

    # Enrich from experiment_result if provided
    exp_data: Dict[str, Any] = {}
    if experiment_result is not None:
        if hasattr(experiment_result, "__dataclass_fields__"):
            exp_data = _experiment_result_to_dict(experiment_result)
        elif isinstance(experiment_result, dict):
            exp_data = experiment_result

    # The SAME three states experiment_results_to_svg uses, applied here too.
    # Until 2026-08-27 this function read `exp_data.get("overall_p_value", 1.0)`
    # and `exp_data.get("heterogeneity_detected", False)`, so a recommendation
    # rendered with NO experiment result at all printed "p-value 1.0000" and a
    # green "Heterogeneity: No" under a heading reading EXPERIMENT METRICS. That
    # is exactly the fabrication removed from experiment_results_to_svg and
    # recorded in the P_NOT_COMPUTED note at the top of this module: a p-value of
    # 1.0 is not "no p-value", it is the most confident "no effect" the scale can
    # express, and "no heterogeneity" is a positive finding about between-group
    # consistency. Neither was ever measured on this path.
    #
    # `experiment_metrics_known` is the coarse state: with no experiment_result
    # argument there is no effect, no test, no power count and no sample count,
    # so the panel says so once instead of printing a row of confident zeros.
    experiment_metrics_known = bool(exp_data)

    overall_effect = _finite(exp_data.get("overall_effect"))
    overall_p = _finite(exp_data.get("overall_p_value"))
    # A reported verdict with no p-value is still a verdict, so the verdict and
    # its p-value are tracked separately (as in experiment_results_to_svg).
    het_detected_raw = exp_data.get("heterogeneity_detected")
    heterogeneity_known = het_detected_raw is not None

    # The COUNTS, which the 2026-08-27 pass above left alone. It hardened the
    # effect, the p-value and the heterogeneity verdict, and the very next block
    # kept counting `.get("powered", False)`, `.get("n_control", 0)` and
    # `.get("n_treatment", 0)` off the same rows: an experiment result carrying
    # an effect and nothing else rendered "Powered 0/0" and "0 control · 0
    # treatment samples" beside the three honest panels. Read together those are
    # measurements, and the honest neighbours are what make them credible: no
    # subgroup had enough data, and neither arm had a single sample. Nobody
    # counted any of it.
    #
    # Every total is the same three states experiment_results_to_svg already
    # applies to the identical quantities, and the conventions match it exactly:
    # a row is added only if it REPORTED the value, the denominator of the power
    # fraction is the GRADED population, and a supplied zero stays a
    # measurement.
    effects_raw = exp_data.get("intersection_effects", [])
    n_powered = 0
    n_power_graded = 0
    n_control = 0
    n_treatment = 0
    n_control_known = False
    n_treatment_known = False
    n_rows = 0
    for ie in effects_raw:
        if isinstance(ie, dict):
            raw_powered = ie.get("powered")
            raw_ctrl = ie.get("n_control")
            raw_trt = ie.get("n_treatment")
        elif hasattr(ie, "powered"):
            # Objects without a `powered` attribute are skipped exactly as
            # before: they are not rows of this table.
            raw_powered = getattr(ie, "powered", None)
            raw_ctrl = getattr(ie, "n_control", None)
            raw_trt = getattr(ie, "n_treatment", None)
        else:
            continue
        n_rows += 1
        # POWERED and UNDERPOWERED are both verdicts about a subgroup, and an
        # absent flag supports neither.
        if raw_powered is not None:
            n_power_graded += 1
            if raw_powered:
                n_powered += 1
        # The raw value is added, not the coerced float, so a count of 610 keeps
        # printing as "610" rather than "610.0" (see the same note in
        # experiment_results_to_svg).
        if raw_ctrl is not None and _finite(raw_ctrl) is not None:
            n_control += raw_ctrl
            n_control_known = True
        if raw_trt is not None and _finite(raw_trt) is not None:
            n_treatment += raw_trt
            n_treatment_known = True

    n_power_ungraded = n_rows - n_power_graded

    # "0 intersections" in the subtitle is a claim about what was analysed, so
    # it needs the same three states. A REPORTED zero still renders as 0.
    _reported_n_intersections = exp_data.get("n_intersections")
    n_intersections_known = _reported_n_intersections is not None or n_rows > 0
    if not isinstance(_reported_n_intersections, int) or isinstance(
        _reported_n_intersections, bool
    ):
        _reported_n_intersections = n_rows
    n_intersections = _reported_n_intersections
    # With every supplied row graded this is the experiment's own intersection
    # count, exactly as before; with any row ungraded it falls back to the
    # graded population, and with nothing graded there is no fraction to print.
    n_power_denom = n_intersections if (n_rows and n_power_ungraded == 0) else n_power_graded

    # A design name nobody reported is the same fabrication in prose form: the
    # panel printed "Independent" for an experiment result that named no design.
    # The sibling adapter has carried `design_known` since its own fix.
    design_known = exp_data.get("design_type") is not None
    design_type_raw = exp_data.get("design_type", "independent")
    if hasattr(design_type_raw, "value"):
        design_type = str(design_type_raw.value)
    else:
        design_type = str(design_type_raw).replace("_", " ").title()

    # Summary text
    decision_labels = {
        "DEPLOY_TREATMENT": "Deploy the treatment, it improves fairness outcomes.",
        "KEEP_CONTROL": "Keep the control, treatment does not improve outcomes.",
        "EXTEND_EXPERIMENT": "Extend the experiment, more data is needed for a confident decision.",
        "INVESTIGATE_FURTHER": "Investigate further, heterogeneous effects or anomalies detected.",
    }
    if decision_known:
        summary_text = decision_labels.get(decision, "Review the experiment results carefully.")
    else:
        # NOT the INVESTIGATE_FURTHER label. "Investigate further, heterogeneous
        # effects or anomalies detected" is a statement about the experiment's
        # results, and it was printed for a call that carried no results at all.
        # Kept inside the template's truncate_text(95) so the sentence survives
        # whole: a half-rendered statement of absence is not a statement.
        summary_text = (
            "No decision was supplied: this canvas recommends nothing, "
            "for the treatment or the control."
        )

    template_data = {
        "title": rec_data.get("title", "Experiment Decision Recommendation"),
        "timestamp": rec_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
        # The subtitle described an experiment too: with no result attached it
        # read "Independent design · 0 intersections", naming a design nobody
        # reported and a count nobody took.
        "subtitle": (
            (
                (f"{design_type} design · " if design_known else "")
                + (
                    f"{n_intersections} intersection{'s' if n_intersections != 1 else ''}"
                    if n_intersections_known
                    else "intersection count not reported"
                )
            )
            if experiment_metrics_known
            else "No experiment result supplied"
        ),
        "decision": decision,
        "decision_known": decision_known,
        "confidence": confidence,
        "confidence_known": confidence_known,
        "confidence_display": f"{confidence * 100:.0f}" if confidence_known else "",
        "summary_text": summary_text,
        # Read by rendering.explain.build_explanation, which replaces the
        # accessible <desc> and the action line whenever it is set. Without it
        # the <desc> read "Recommendation: Investigate Further (confidence 0%).
        # Investigate further, heterogeneous effects or anomalies detected." for
        # an empty dict, i.e. the recommendation withheld from the canvas came
        # back as a decision for every reader who cannot see the canvas.
        "not_assessable": not decision_known,
        "not_assessable_reason": (
            ""
            if decision_known
            else (
                "No decision was supplied to experiment_recommendation_to_svg, so no "
                "recommendation was made and no confidence was reported."
                + (
                    " The experiment metrics on this canvas are shown exactly as they "
                    "were supplied and are not graded into a recommendation."
                    if experiment_metrics_known
                    else ""
                )
            )
        ),
        "reasoning": reasoning[:8],
        "tradeoffs": tradeoffs[:6],
        "caveats": caveats[:5],
        "experiment_metrics_known": experiment_metrics_known,
        "overall_effect_known": overall_effect is not None,
        "overall_effect": (
            f"{overall_effect:+.4f}" if overall_effect is not None else P_NOT_COMPUTED
        ),
        "overall_p_known": overall_p is not None,
        "overall_p": overall_p,
        "overall_p_display": (
            P_NOT_COMPUTED
            if overall_p is None
            else (f"{overall_p:.4f}" if overall_p >= 0.0001 else "<0.0001")
        ),
        "heterogeneity_known": heterogeneity_known,
        "heterogeneity_detected": bool(het_detected_raw),
        "could_not_check_text": COULD_NOT_CHECK_TEXT,
        "n_powered": n_powered,
        "n_power_graded": n_power_graded,
        "n_power_ungraded": n_power_ungraded,
        "n_power_denom": n_power_denom,
        "n_intersections": n_intersections,
        "n_intersections_known": n_intersections_known,
        "n_control": n_control,
        "n_control_known": n_control_known,
        "n_treatment": n_treatment,
        "n_treatment_known": n_treatment_known,
        "design_type": design_type,
        "design_known": design_known,
    }

    try:
        template_data["explanation"] = explanation
        svg = render_svg("experiment_recommendation", template_data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception:
        # READINESS-6, 2026-09-10. This warned and returned "".
        #
        # `raise_jinja2_missing` in engine.py records the decision this reverses,
        # taken 2026-08-28 for the missing-BACKEND case: "their docstrings
        # document the return as 'SVG markup string', so a caller doing
        # open(p, "w").write(svg) wrote a ZERO-BYTE file and read it as this
        # run's report ... An empty string is not markup, and 'could not render'
        # must not be indistinguishable from 'rendered nothing worth showing'."
        # That fix funnelled 16 adapters through a raise and stopped there. The
        # general render-FAILURE path in this wrapper kept the swallow, and
        # render_svg raises through it, so the refusal was caught here anyway.
        #
        # Measured that day on a real failure (incomplete template data):
        #     returned '' (len 0), one warning, and a caller writing it got a
        #     0-byte report; THREE failed renders produced ONE warning, because
        #     Python shows a warning once per process. So a batch job warns once
        #     and then emits nothing, silently, for every report after it.
        #
        # Re-raised bare, which preserves the original type and traceback: the
        # cause is usually a jinja2 UndefinedError naming the missing key, which
        # is far more use to a caller than a warning they will not see.
        raise


def power_analysis_to_svg(
    power_results: Union[List, Dict[str, Any]],
    *,
    alpha: float = 0.05,
    power_target: float = 0.80,
    mde: Optional[float] = None,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render power analysis results as an SVG dashboard.

    Visualises per-intersection power, required vs current sample sizes,
    power bar chart, and minimum detectable effect (MDE).

    Args:
        power_results: List of PowerResult objects or dicts, or a summary dict
            from FairnessPowerAnalyzer.get_power_summary() /
            get_detailed_results().
        alpha: Significance level (default 0.05).
        power_target: Target power (default 0.80).
        mde: Minimum detectable effect size (Cohen's d). Optional.
        save_path: Optional path to save the SVG file.

    Returns:
        SVG markup string.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    # Normalise input to list of dicts
    raw_list: Any
    if isinstance(power_results, dict):
        # Summary dict format from get_power_summary()
        raw_list = power_results.get("results", power_results.get("intersections", []))
        if not raw_list and "overall_power" in power_results:
            raw_list = power_results.get("detailed_results", [])
    elif isinstance(power_results, list):
        raw_list = power_results
    else:
        raw_list = []

    intersections: List[Dict[str, Any]] = []
    total_required_n = 0
    total_current_n = 0
    power_sum = 0.0
    # Counted, not assumed. Every total below is a sum over the rows that
    # actually reported the value, so a row that reported none of them cannot
    # drag an average down or leave a required sample size understated in
    # silence.
    n_power_known = 0
    n_required_known = 0
    n_current_known = 0

    for pr in raw_list:
        if hasattr(pr, "__dataclass_fields__"):
            pr_dict = {
                "intersection": pr.intersection,
                "power": pr.power,
                "required_n": pr.required_n,
                "is_powered": pr.is_powered,
                "n_control": pr.n_control,
                "n_treatment": pr.n_treatment,
                "effect_size": pr.effect_size,
            }
        elif isinstance(pr, dict):
            pr_dict = pr
        else:
            continue

        # `.get(key, default)` DOES NOT FIRE when the key is PRESENT holding
        # None, so a row carrying ``{"intersection": None}`` skipped the
        # ``("unknown",)`` default and `str(None)` MINTED the label "None".
        # Measured 2026-09-30 on power_analysis_to_svg: the canvas drew a row
        # and a bar labelled "None" with a real power percentage beside it, in
        # the font and the position of a real subgroup. pd.NA and pd.NaT print
        # "<NA>" and "NaT" through the same line. An unlabelled row gets the
        # SAME word an absent key already got, so nothing that was working
        # changes.
        name = pr_dict.get("intersection")
        if name is None or _is_missing_level(name):
            name = ("unknown",)
        if isinstance(name, (tuple, list)):
            name_str = " × ".join(str(x) for x in name)
            short_name = str(name[0]) if name else "?"
        else:
            name_str = str(name)
            short_name = str(name)[:8]

        # The remainder of the same defect, on the three fields an earlier wave
        # left open. `power` defaulted to 0, `required_n` to 0 and both arm
        # counts to 0, so an intersection that reported none of them rendered
        # "0%" in red with a full-width NEEDS DATA badge, a zero-height red bar
        # on the chart and a required sample size of 0. Read together that is a
        # measured-sounding finding: this subgroup has no power whatsoever, and
        # needs no additional data to fix it. Both halves are false, and the
        # zeros also flowed into the averages and totals above the table.
        #
        # A SUPPLIED zero is a measurement and still renders as 0% / 0, exactly
        # as the effect-size fix below does: only an absent, unparseable, NaN or
        # infinite value is an absence.
        power_val = _finite(pr_dict.get("power"))
        power_known = power_val is not None
        required_val = _finite(pr_dict.get("required_n"))
        required_known = required_val is not None
        n_ctrl = _finite(pr_dict.get("n_control"))
        n_trt = _finite(pr_dict.get("n_treatment"))
        current_known = n_ctrl is not None or n_trt is not None
        current_n = int((n_ctrl or 0) + (n_trt or 0))
        # Zero is a measurement. `f"{x:.3f}" if effect_size else P_NOT_COMPUTED`
        # was a FALSY test, so a measured effect size of exactly 0.0 (the
        # treatment moved this intersection not at all, which is precisely the
        # result a power analysis exists to size) was reported in the table as
        # "not computed", i.e. a real null finding was rendered as an absence.
        # The absence is read off `_finite`, which is None only for a missing,
        # unparseable, NaN or infinite value.
        effect_size = _finite(pr_dict.get("effect_size"))

        # A verdict the analyser reported wins, and is a verdict even without a
        # power value beside it. Otherwise it is derived from a KNOWN power.
        # With neither there is NO verdict: the row is not called underpowered,
        # because "this subgroup needs more data" is a finding and nothing was
        # measured to support it. The old expression could not express that,
        # since its fallback was computed from the defaulted 0.
        #
        # Every branch below narrows on `is not None` directly rather than
        # through the `*_known` bool beside it: a plain bool carries no type
        # information, so a checker cannot see that the value is a float here.
        # Same behaviour, expressed checkably (the note on `overall_p` in
        # experiment_results_to_svg records the same reason).
        raw_powered = pr_dict.get("is_powered")
        if raw_powered is not None:
            is_powered = bool(raw_powered)
            powered_known = True
        elif power_val is not None:
            is_powered = power_val >= power_target
            powered_known = True
        else:
            is_powered = False
            powered_known = False

        intersections.append(
            {
                "name": name_str,
                "short_name": short_name,
                # Kept numeric for the bar geometry and the colour bands, which
                # the template reaches only under `power_known`.
                "power": power_val if power_val is not None else 0.0,
                "power_known": power_known,
                "power_display": (
                    f"{power_val * 100:.0f}" if power_val is not None else P_NOT_COMPUTED
                ),
                "required_known": required_known,
                "required_n": (
                    str(int(required_val)) if required_val is not None else P_NOT_COMPUTED
                ),
                "current_known": current_known,
                "current_n": (str(current_n) if current_known else P_NOT_COMPUTED),
                "effect_size": (
                    f"{effect_size:.3f}" if effect_size is not None else P_NOT_COMPUTED
                ),
                "is_powered": is_powered,
                "powered_known": powered_known,
            }
        )

        if required_val is not None:
            total_required_n += int(required_val)
            n_required_known += 1
        if current_known:
            total_current_n += current_n
            n_current_known += 1
        if power_val is not None:
            power_sum += power_val
            n_power_known += 1

    # Supplied / graded / powered / underpowered are four different counts, and
    # the denominator of a verdict is the GRADED population, never the number of
    # rows: "1 of 2 intersections are underpowered" over a second row that was
    # never graded says the second one is adequately powered.
    #
    # `n_total` is therefore the assessed population, following the same
    # convention as adapters_fairness.fairness_bar_chart_to_svg (`"n_total":
    # n_assessed`, with the unchecked metrics counted separately). It is what
    # rendering.explain._fr_power writes into the accessible description, so the
    # sentence there talks about the rows that carry a verdict and nothing else.
    n_supplied = len(intersections)
    n_total = sum(1 for i in intersections if i["powered_known"])
    n_powered = sum(1 for i in intersections if i["is_powered"])
    n_underpowered = sum(1 for i in intersections if i["powered_known"] and not i["is_powered"])
    # An average of nothing is not 0 percent. It is averaged over the rows that
    # reported a power, and when none did there is no average to report.
    avg_power = power_sum / n_power_known if n_power_known > 0 else 0
    avg_power_known = n_power_known > 0
    # Nothing graded: the whole canvas is a could-not-check, not a clean bill of
    # health. `power_analysis_to_svg([])` rendered "POWERED 0 / 0" behind a GREEN
    # accent (nothing underpowered), "AVG POWER 0%", "TOTAL REQUIRED N 0" and
    # "Total N = 0", with the accessible description reading "0 of 0
    # intersections are underpowered (avg power 0%, target 80%)" at severity
    # INFO. Zero of zero underpowered is not a finding that every intersection
    # is adequately powered.
    measured = n_total > 0

    template_data = {
        "title": "Statistical Power Analysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # The subtitle counts what was SUPPLIED, which is a fact about the call
        # rather than a verdict about the data.
        "subtitle": f"{n_supplied} intersection{'s' if n_supplied != 1 else ''} · α={alpha} · target power={power_target:.0%}",
        "intersections": intersections[:12],
        "measured": measured,
        "n_powered": n_powered,
        "n_underpowered": n_underpowered,
        "n_supplied": n_supplied,
        "n_ungraded": n_supplied - n_total,
        "n_total": n_total,
        # Kept numeric (and 0 when nothing was reported) for the colour bands,
        # which the template reaches only under `avg_power_known`.
        "avg_power": avg_power,
        "avg_power_known": avg_power_known,
        "n_power_known": n_power_known,
        "avg_power_display": f"{avg_power * 100:.0f}" if avg_power_known else "",
        "total_required_known": n_required_known > 0,
        "n_required_known": n_required_known,
        "total_required_n": (f"{total_required_n:,}" if n_required_known > 0 else P_NOT_COMPUTED),
        "n_current_known": n_current_known,
        "total_current_n": (f"{total_current_n:,}" if n_current_known > 0 else P_NOT_COMPUTED),
        "could_not_check_text": COULD_NOT_CHECK_TEXT,
        # Read by rendering.explain.build_explanation, which replaces the
        # accessible <desc> and the on-canvas action line. Without it the
        # description read "0 of 0 intersections are underpowered (avg power 0%,
        # target 80%)" at severity INFO, and the action told the reader to
        # collect more data for the under-powered intersections of an analysis
        # that sized none.
        "not_assessable": not measured,
        "not_assessable_reason": (
            ""
            if measured
            else (
                (
                    "No power result was supplied to power_analysis_to_svg, so no "
                    "intersection was sized and no power was computed."
                    if n_supplied == 0
                    else (
                        f"Not one of the {n_supplied} intersection(s) supplied to "
                        "power_analysis_to_svg carried a power value or a powered verdict, "
                        "so none of them was graded."
                    )
                )
                + " This is not a finding that every intersection is adequately powered."
            )
        ),
        "alpha": f"{alpha}",
        "power_target": f"{power_target:.0%}",
        "mde_value": f"{mde:.4f}" if mde is not None else None,
    }

    try:
        template_data["explanation"] = explanation
        svg = render_svg("power_analysis", template_data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception:
        # READINESS-6, 2026-09-10. This warned and returned "".
        #
        # `raise_jinja2_missing` in engine.py records the decision this reverses,
        # taken 2026-08-28 for the missing-BACKEND case: "their docstrings
        # document the return as 'SVG markup string', so a caller doing
        # open(p, "w").write(svg) wrote a ZERO-BYTE file and read it as this
        # run's report ... An empty string is not markup, and 'could not render'
        # must not be indistinguishable from 'rendered nothing worth showing'."
        # That fix funnelled 16 adapters through a raise and stopped there. The
        # general render-FAILURE path in this wrapper kept the swallow, and
        # render_svg raises through it, so the refusal was caught here anyway.
        #
        # Measured that day on a real failure (incomplete template data):
        #     returned '' (len 0), one warning, and a caller writing it got a
        #     0-byte report; THREE failed renders produced ONE warning, because
        #     Python shows a warning once per process. So a batch job warns once
        #     and then emits nothing, silently, for every report after it.
        #
        # Re-raised bare, which preserves the original type and traceback: the
        # cause is usually a jinja2 UndefinedError naming the missing key, which
        # is far more use to a caller than a warning they will not see.
        raise


# Helper


def _experiment_result_to_dict(result) -> Dict[str, Any]:
    """Convert an ExperimentResult dataclass to a flat dictionary."""
    return {
        "overall_effect": result.overall_effect,
        "overall_ci": result.overall_ci,
        "overall_p_value": result.overall_p_value,
        "intersection_effects": result.intersection_effects,
        "heterogeneity_detected": result.heterogeneity_detected,
        "heterogeneity_p_value": result.heterogeneity_p_value,
        "power_results": result.power_results,
        "n_intersections": result.n_intersections,
        "n_excluded": result.n_excluded,
        "design_type": result.design_type,
        "metadata": result.metadata if hasattr(result, "metadata") else {},
    }


# 4.  CausalDecomposition  →  causal_decomposition.svg


def causal_decomposition_to_svg(
    decomposition: Union["CausalDecomposition", Dict[str, Any], None] = None,
    temporal: Union["TemporalStabilityResult", Dict[str, Any], None] = None,
    *,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render causal decomposition analysis as an SVG dashboard.

    Visualises direct vs indirect effects, proportion mediated,
    Baron-Kenny steps, and optional temporal stability.

    Parameters
    ----------
    decomposition : CausalDecomposition or dict
    temporal : TemporalStabilityResult or dict, optional
    example : bool
        Render the built-in demonstration fixture, watermarked EXAMPLE on the
        canvas.  Off by default and never reached implicitly: see the note
        below.
    explanation : str
    save_path : str, optional

    Returns
    -------
    str  SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old default. Until 2026-08-27 a missing
    *decomposition* silently dispatched to ``_demo_causal`` and this function
    returned a complete, confident mediation analysis: "Partial mediation (38%)
    through 'score_calibration'", a direct effect of 0.0280, an indirect effect
    of 0.0170, a total of 0.0450, four of four Baron-Kenny steps satisfied and a
    five-quarter STABLE temporal trend, none of it measured from anything the
    caller supplied. An SVG is an export format: it leaves the building, an
    auditor reads it, and it outlives the version that produced it, so a
    fabricated causal claim is among the most damaging artifacts this module can
    emit. A caller could not tell that output apart from a real analysis.

    An empty decomposition was the mirror failure: it printed 0.0000 effects and
    four crosses, so a run that never happened read as a definite negative
    result. See :func:`_decomposition_measured`. Both are COULD NOT CHECK, and
    it is stated on the canvas. The demo fixture survives only behind the
    explicit ``example=True``, watermarked.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    supplied = decomposition is not None

    # Unpack decomposition
    dec: Dict[str, Any]
    if decomposition is None:
        if example:
            return _demo_causal(explanation=explanation, save_path=save_path)
        dec = {}
    elif hasattr(decomposition, "__dataclass_fields__"):
        dec = {k: getattr(decomposition, k) for k in decomposition.__dataclass_fields__}
    elif isinstance(decomposition, dict):
        dec = decomposition
    else:
        dec = {}

    # Three states, never two. A zero is a measurement; an absent field is not.
    measured = _decomposition_measured(dec)

    # PER FIELD, not per canvas. `_decomposition_measured` above answers "did
    # this decomposition report anything at all", and the whole-canvas
    # could-not-check branch hangs off it. It does NOT answer "did it report
    # THIS effect", and `float(dec.get(key, 0))` answered that with a zero: a
    # decomposition reporting only a total effect printed "Direct: 0.0000" and
    # "Indirect: 0.0000" under the heading EFFECT DECOMPOSITION and drew a 0%
    # mediation gauge, which is the finding "this mediator carries none of the
    # gap" for a path nobody estimated. The partial row is the case the
    # canvas-level guard cannot see.
    #
    # It also CRASHED on one. `float(None)` raises TypeError, and these five
    # lines sit outside the try that wraps the render, so a decomposition
    # carrying an explicit None for any effect took the exception all the way
    # out to the caller instead of drawing the absence.
    #
    # A supplied zero is a measurement and still renders as 0.0000: an estimated
    # indirect effect of exactly 0.0 is a real finding. Only an absent,
    # unparseable, NaN or infinite value is an absence.
    total_val = _finite(dec.get("total_effect"))
    direct_val = _finite(dec.get("direct_effect"))
    indirect_val = _finite(dec.get("indirect_effect"))
    prop_med_val = _finite(dec.get("proportion_mediated"))
    total_known = total_val is not None
    direct_known = direct_val is not None
    indirect_known = indirect_val is not None
    prop_med_known = prop_med_val is not None
    total = total_val if total_val is not None else 0.0
    direct = direct_val if direct_val is not None else 0.0
    indirect = indirect_val if indirect_val is not None else 0.0
    prop_med = prop_med_val if prop_med_val is not None else 0.0
    raw_mediator = dec.get("mediator")
    mediator_known = bool(str(raw_mediator or "").strip())
    mediator = str(raw_mediator) if mediator_known else "not reported"
    steps = dec.get("steps_satisfied", {})

    # Compute bar widths (stacked, max 280px). An unknown component contributes
    # no segment rather than a minimum-width one: the template draws each half
    # only under its own `*_known` flag.
    abs_total = abs(total) or 1
    direct_w = int(abs(direct) / abs_total * 280)
    indirect_w = int(abs(indirect) / abs_total * 280)

    # Baron-Kenny steps.
    #
    # Three states per STEP. `steps.get(key, False)` gave a step nobody
    # evaluated the same red cross as a step that was tested and failed, which
    # is the strongest claim this panel can make about a path: the mediation
    # ladder was climbed here and it broke. A half-reported ladder
    # ({"step_1": True, "step_2": True}) drew two green ticks and two red
    # crosses, so the two unevaluated steps read as measured negatives.
    #
    # `n_steps_ok` keeps counting only steps reported as satisfied, exactly as
    # before, so an unreported step is still never CREDITED. The denominator
    # stays 4 because the Baron-Kenny ladder is four steps by definition and
    # "0/4 satisfied" is literally true of a ladder nobody climbed; what the
    # ungraded count buys is the canvas saying WHY, instead of showing four
    # crosses that claim four failed tests.
    bk_steps = []
    step_labels = [
        ("step_1", "Treatment → Outcome (c path)"),
        ("step_2", "Treatment → Mediator (a path)"),
        ("step_3", "Mediator → Outcome controlling Treatment (b path)"),
        ("step_4", "Direct effect reduced when mediator included (c' < c)"),
    ]
    for key, label in step_labels:
        raw_step = steps.get(key) if isinstance(steps, dict) else None
        step_known = raw_step is not None
        bk_steps.append(
            {
                "key": key,
                "label": label,
                "satisfied": bool(raw_step),
                "step_known": step_known,
            }
        )
    n_steps_ok = sum(1 for s in bk_steps if s["step_known"] and s["satisfied"])
    n_steps_graded = sum(1 for s in bk_steps if s["step_known"])
    n_steps_ungraded = len(bk_steps) - n_steps_graded

    # Temporal stability
    temporal_data: Optional[Dict[str, Any]] = None
    if temporal is not None:
        if hasattr(temporal, "__dataclass_fields__"):
            td = {k: getattr(temporal, k) for k in temporal.__dataclass_fields__}
        elif isinstance(temporal, dict):
            td = temporal
        else:
            td = {}

        periods = td.get("periods", [])
        effects_over_time = td.get("effects_over_time", [])
        # Three states, never two. `td.get("is_stable", True)` defaulted a
        # MISSING stability verdict to True, so a series with no stability test
        # attached was captioned STABLE in green: a positive finding about
        # temporal consistency, manufactured out of an absent field. It is the
        # same fail-open default as the heterogeneity incident recorded at the
        # top of this module. An absent verdict is could-not-check, and the
        # timeline is drawn in neutral slate rather than in a verdict colour.
        is_stable_raw = td.get("is_stable")
        stability_known = is_stable_raw is not None
        is_stable = bool(is_stable_raw)
        # The slope gets the same treatment as the verdict beside it, and for
        # the same reason: "trend slope: 0.0000" is a measured flat trend, which
        # is a positive finding about temporal consistency, and it was printed
        # for any series that carried no slope. `float(None)` also raised
        # TypeError here, outside the try that wraps the render, so a series
        # with an explicit None slope crashed the caller.
        slope_val = _finite(td.get("trend_slope"))
        trend_slope_known = slope_val is not None
        trend_slope = slope_val if slope_val is not None else 0.0

        # Build timeline points for SVG polyline
        points: List[Dict[str, Any]] = []
        if effects_over_time:
            max_eff = max(abs(e) for e in effects_over_time) or 1
            n = len(effects_over_time)
            for i, eff in enumerate(effects_over_time):
                x = int(56 + i * (560 / max(n - 1, 1)))
                y = int(40 - (eff / max_eff) * 30)
                points.append({"x": x, "y": y, "value": eff})

        if not stability_known:
            trend_color = "#64748b"  # slate-500, neither verdict colour
            trend_label = COULD_NOT_CHECK_TEXT
            # The longer caption needs a wider pill; the right edge is unchanged
            # so the two verdict states render exactly as before.
            pill_x, pill_w = 530, 110
        elif is_stable:
            trend_color = "#059669"
            trend_label = "STABLE"
            pill_x, pill_w = 570, 70
        else:
            trend_color = "#dc2626"
            trend_label = "UNSTABLE"
            pill_x, pill_w = 570, 70

        temporal_data = {
            "periods": periods,
            "points": points,
            "stability_known": stability_known,
            "is_stable": is_stable,
            "trend_slope": trend_slope,
            "trend_slope_known": trend_slope_known,
            "trend_color": trend_color,
            "trend_label": trend_label,
            "pill_x": pill_x,
            "pill_w": pill_w,
            "pill_cx": pill_x + pill_w // 2,
        }

    recs: List[str] = []
    if not measured:
        # An absence, stated as an absence. "Only 0/4 Baron-Kenny steps
        # satisfied" was written here for the same input, which reads as a
        # mediation test that ran and failed.
        recs.append(
            "No effect estimate and no Baron-Kenny step was supplied, so nothing was "
            "decomposed: this canvas reports no mediation either way."
        )
    elif prop_med_known and prop_med > 0.5:
        recs.append(
            f"Mediator '{mediator}' accounts for {prop_med:.0%}, consider mediator intervention."
        )
    elif prop_med_known and prop_med > 0.2:
        recs.append(f"Partial mediation ({prop_med:.0%}) through '{mediator}', monitor both paths.")
    if measured and n_steps_ok < 4:
        recs.append(
            f"Only {n_steps_ok}/4 Baron-Kenny steps satisfied, causal interpretation is limited."
        )
    # WHY it is limited, which the line above cannot distinguish: a ladder that
    # was climbed and broke reads identically to a ladder nobody climbed.
    if measured and n_steps_ungraded > 0:
        recs.append(
            f"{n_steps_ungraded} of the 4 Baron-Kenny steps reported no verdict, so they were "
            "not tested rather than failed."
        )
    # Gated on the VERDICT being known. An absent is_stable used to read as
    # False here as well, so the same missing field could produce a confident
    # "instability detected" finding in one place and a green STABLE caption in
    # the other. Neither was measured.
    if temporal_data and not temporal_data["stability_known"]:
        recs.append(
            "No temporal stability verdict was supplied, so the trend below was not graded."
        )
    elif temporal_data and not temporal_data["is_stable"]:
        recs.append("Temporal instability detected, effects are changing over time.")
    # The all-clear, which may not be unqualified while any part of the
    # decomposition is ungraded. All four steps can be satisfied on a
    # decomposition that estimated no effect at all, and "Causal decomposition
    # is clean" over four "not reported" effects is a verdict on numbers nobody
    # produced. The clean sentence is unchanged when everything was graded.
    #
    # Counted rather than listed, and kept near 120 characters: the
    # recommendation rows are drawn at x=56 in a 644-wide panel with no wrap and
    # no truncate filter, so a longer sentence runs off the right edge of the
    # canvas. Rasterising caught exactly that on the first draft of this line,
    # which is the same trap the summary_text note in
    # experiment_recommendation_to_svg records: a half-rendered statement of
    # absence is not a statement.
    _n_ungraded_effects = sum(
        1 for known in (total_known, direct_known, indirect_known, prop_med_known) if not known
    )
    if not recs:
        if measured and _n_ungraded_effects:
            recs.append(
                f"All reported mediation steps are satisfied, but {_n_ungraded_effects} of the 4 "
                "effect estimates were not reported: incomplete, not clean."
            )
        else:
            recs.append("Causal decomposition is clean, all mediation steps satisfied.")

    template_data = {
        "title": "Causal Decomposition Analysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Drives the template's could-not-check branch, which withholds the flow
        # diagram, the effect bar, the mediation gauge and the Baron-Kenny
        # ticks, and (through rendering.explain.build_explanation) the
        # accessible <desc> and the action line, so the canvas and the screen
        # reader tell one story. Any supplied temporal series is still drawn:
        # it is data the caller measured, and dropping it would be its own
        # falsehood.
        "measured": measured,
        "not_assessable": not measured,
        "not_assessable_reason": (
            ""
            if measured
            else (
                (
                    "The decomposition supplied to causal_decomposition_to_svg carried no "
                    "effect estimate and no Baron-Kenny step, so nothing was decomposed on "
                    "this run."
                )
                if supplied
                else (
                    "No decomposition was supplied to causal_decomposition_to_svg, so no "
                    "direct, indirect or total effect was estimated and no Baron-Kenny "
                    "step was tested."
                )
            )
            + " This is not a finding of no mediation: no mediation was tested."
        ),
        "is_example": False,
        "mediator": mediator,
        "mediator_known": mediator_known,
        "total_effect": total,
        "direct_effect": direct,
        "indirect_effect": indirect,
        # Each effect carries its own state, so the template can withhold one
        # number without withholding the panel that holds it.
        "total_known": total_known,
        "direct_known": direct_known,
        "indirect_known": indirect_known,
        "direct_w": max(4, direct_w),
        "indirect_w": max(4, indirect_w),
        "proportion_mediated": prop_med,
        "proportion_mediated_known": prop_med_known,
        "prop_med_pct": int(prop_med * 100),
        "prop_med_arc": int(prop_med * 180),
        "bk_steps": bk_steps,
        "n_steps_ok": n_steps_ok,
        "n_steps_graded": n_steps_graded,
        "n_steps_ungraded": n_steps_ungraded,
        "could_not_check_text": COULD_NOT_CHECK_TEXT,
        "temporal": temporal_data,
        "recommendations": recs[:4],
        "explanation": explanation,
    }

    try:
        svg = render_svg("causal_decomposition", template_data)
        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)
        return svg
    except Exception:
        # READINESS-6, 2026-09-10. This warned and returned "".
        #
        # `raise_jinja2_missing` in engine.py records the decision this reverses,
        # taken 2026-08-28 for the missing-BACKEND case: "their docstrings
        # document the return as 'SVG markup string', so a caller doing
        # open(p, "w").write(svg) wrote a ZERO-BYTE file and read it as this
        # run's report ... An empty string is not markup, and 'could not render'
        # must not be indistinguishable from 'rendered nothing worth showing'."
        # That fix funnelled 16 adapters through a raise and stopped there. The
        # general render-FAILURE path in this wrapper kept the swallow, and
        # render_svg raises through it, so the refusal was caught here anyway.
        #
        # Measured that day on a real failure (incomplete template data):
        #     returned '' (len 0), one warning, and a caller writing it got a
        #     0-byte report; THREE failed renders produced ONE warning, because
        #     Python shows a warning once per process. So a batch job warns once
        #     and then emits nothing, silently, for every report after it.
        #
        # Re-raised bare, which preserves the original type and traceback: the
        # cause is usually a jinja2 UndefinedError naming the missing key, which
        # is far more use to a caller than a warning they will not see.
        raise


def _demo_causal(*, explanation: Optional[str] = None, save_path: Optional[str] = None) -> str:
    """Generate demo causal decomposition SVG.

    Reachable ONLY through ``causal_decomposition_to_svg(example=True)``. Every
    number below is invented, and ``is_example`` is what marks the canvas as a
    demonstration: it draws the full-width EXAMPLE band under the header and
    leads the accessible description with the same marker.
    """
    bk_steps = [
        {
            "key": "step_1",
            "label": "Treatment → Outcome (c path)",
            "satisfied": True,
            "step_known": True,
        },
        {
            "key": "step_2",
            "label": "Treatment → Mediator (a path)",
            "satisfied": True,
            "step_known": True,
        },
        {
            "key": "step_3",
            "label": "Mediator → Outcome controlling Treatment (b path)",
            "satisfied": True,
            "step_known": True,
        },
        {
            "key": "step_4",
            "label": "Direct effect reduced when mediator included (c' < c)",
            "satisfied": True,
            "step_known": True,
        },
    ]
    temporal_data = {
        "periods": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "points": [
            {"x": 56, "y": 18, "value": 0.042},
            {"x": 196, "y": 15, "value": 0.044},
            {"x": 336, "y": 20, "value": 0.040},
            {"x": 476, "y": 12, "value": 0.047},
            {"x": 616, "y": 16, "value": 0.043},
        ],
        "stability_known": True,
        "is_stable": True,
        "trend_slope": 0.001,
        "trend_slope_known": True,
        "trend_color": "#059669",
        "trend_label": "STABLE",
        "pill_x": 570,
        "pill_w": 70,
        "pill_cx": 605,
    }
    recs = [
        "Partial mediation (38%) through 'score_calibration', monitor both paths.",
        "All 4 Baron-Kenny steps satisfied, causal interpretation is supported.",
        "Temporal trend is stable, effects are consistent over time.",
    ]

    template_data = {
        "title": "Causal Decomposition Analysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # DO NOT REMOVE: the band on the canvas and the marker on the <desc> are
        # the only things separating this fixture from an audit artifact once
        # the SVG is exported.
        "is_example": True,
        "measured": True,
        "mediator": "score_calibration",
        "mediator_known": True,
        "total_effect": 0.045,
        "direct_effect": 0.028,
        "indirect_effect": 0.017,
        "total_known": True,
        "direct_known": True,
        "indirect_known": True,
        "direct_w": 174,
        "indirect_w": 106,
        "proportion_mediated": 0.38,
        "proportion_mediated_known": True,
        "prop_med_pct": 38,
        "prop_med_arc": 68,
        "bk_steps": bk_steps,
        "n_steps_ok": 4,
        "n_steps_graded": 4,
        "n_steps_ungraded": 0,
        "could_not_check_text": COULD_NOT_CHECK_TEXT,
        "temporal": temporal_data,
        "recommendations": recs,
        "explanation": explanation,
    }

    try:
        svg = render_svg("causal_decomposition", template_data)
        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)
        return svg
    except Exception:
        # READINESS-6, 2026-09-10. This warned and returned "".
        #
        # `raise_jinja2_missing` in engine.py records the decision this reverses,
        # taken 2026-08-28 for the missing-BACKEND case: "their docstrings
        # document the return as 'SVG markup string', so a caller doing
        # open(p, "w").write(svg) wrote a ZERO-BYTE file and read it as this
        # run's report ... An empty string is not markup, and 'could not render'
        # must not be indistinguishable from 'rendered nothing worth showing'."
        # That fix funnelled 16 adapters through a raise and stopped there. The
        # general render-FAILURE path in this wrapper kept the swallow, and
        # render_svg raises through it, so the refusal was caught here anyway.
        #
        # Measured that day on a real failure (incomplete template data):
        #     returned '' (len 0), one warning, and a caller writing it got a
        #     0-byte report; THREE failed renders produced ONE warning, because
        #     Python shows a warning once per process. So a batch job warns once
        #     and then emits nothing, silently, for every report after it.
        #
        # Re-raised bare, which preserves the original type and traceback: the
        # cause is usually a jinja2 UndefinedError naming the missing key, which
        # is far more use to a caller than a warning they will not see.
        raise
