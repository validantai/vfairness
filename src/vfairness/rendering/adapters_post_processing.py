"""
SVG Rendering Adapters for Post-Processing Module.

This module provides adapters for converting post-processing analysis results
(threshold optimization, prediction reweighting) into SVG visualizations.

Adapters Implemented:
    - threshold_optimization_to_svg: Render GroupThresholdOptimizer results
    - reweighting_comparison_to_svg: Render ReweightingAnalyzer comparison
    - fairness_detailed_report_to_svg: Render comprehensive FairnessAnalyzer report
"""

import math
import numbers
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from .._triage import unmeasurable_reason
from ..evaluation.vfairness_metrics._metric_direction import (
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    metric_direction,
)
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

if TYPE_CHECKING:
    from ..post_processing.reweighting.analyzer import ReweightingAnalysisReport


def threshold_optimization_to_svg(
    result: Dict[str, Any],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render threshold optimization results as an SVG dashboard.

    This function creates a comprehensive visualization of group-specific
    threshold optimization results, showing:
    - Original vs optimized thresholds per group
    - Positive rate changes before/after
    - Fairness improvement metrics
    - Accuracy trade-off analysis

    Args:
        result: ThresholdOptimizationResult from GroupThresholdOptimizer,
                or a dictionary with the required fields.
        save_path: Optional path to save the SVG file.

    Returns:
        SVG markup string.

    Example:
        >>> from vfairness.post_processing import GroupThresholdOptimizer
        >>> from vfairness.rendering import threshold_optimization_to_svg
        >>>
        >>> optimizer = GroupThresholdOptimizer(constraint='demographic_parity')
        >>> optimizer.fit(y_true, y_prob, gender)
        >>> result = optimizer.result_
        >>> svg = threshold_optimization_to_svg(result, save_path='threshold_opt.svg')
    """
    if not JINJA2_AVAILABLE:
        # This is an EXPORT path. A caller who passed save_path got "" back,
        # no exception, and no file: whatever sat at that path from an earlier
        # run stayed there and was read as this run's result. So the FILE says
        # could-not-check out loud, and it is still written FIRST, before the
        # raise: an exception alone would leave the stale chart in place for a
        # pipeline that catches per-report and carries on.
        #
        # The return value used to be "" here, to match 15 sibling adapters.
        # That was the wrong half of the library to match: the other 28 raise,
        # "" is not the "SVG markup string" this function documents, and a
        # caller writing it out produced a zero-byte report. Both halves now
        # raise (2026-08-28); see engine.raise_jinja2_missing.
        _write_svg(
            save_path,
            _could_not_render_svg(
                "Threshold Optimization Report",
                "Jinja2 is not installed, so this chart could not be rendered.",
            ),
        )
        raise_jinja2_missing()

    # Handle both result objects and dictionaries
    if hasattr(result, "to_dict"):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = result
    else:
        data = {}

    # Extract or compute required fields
    groups = data.get("groups", [])
    if not groups and "group_thresholds" in data:
        # Build groups from group_thresholds dict
        group_thresholds = data.get("group_thresholds", {})
        original_rates = data.get("original_rates", {})
        optimized_rates = data.get("optimized_rates", {})
        group_sizes = data.get("group_sizes", {})

        # The same rule per GROUP ROW. `ThresholdResult.to_dict()` carries
        # group_thresholds and nothing else, so every real optimiser run took
        # these defaults: each row printed ORIG RATE 50.0%, OPT RATE 50.0% and
        # SIZE 0. That reads as a measured before/after (the optimiser moved
        # this group's selection rate by exactly nothing) over a group with no
        # members, and 0.5 is this library's placeholder, not a rate anyone
        # computed. An absent rate is left as None, which the template's pct1
        # filter renders as N/A, and an absent size says it was not reported.
        groups = []
        for name, threshold in group_thresholds.items():
            groups.append(
                {
                    "name": str(name),
                    "threshold": threshold,
                    "original_rate": original_rates.get(name),
                    "optimized_rate": optimized_rates.get(name),
                    "size": group_sizes.get(name),
                }
            )

    # Compute fairness improvement. A disparity the result does not carry is
    # left as None, NOT defaulted to 0: the old defaults turned an empty result
    # into "0.000 -> 0.000" with a green reduction badge and a full-width
    # "100% reduction" scale, a measured-looking trade on a run that measured
    # nothing. An original_disparity that IS supplied as 0 keeps its old
    # reduction of 0: that is a measured "there was nothing to reduce".
    original_disparity = _as_float(data.get("original_disparity"))
    optimized_disparity = _as_float(data.get("optimized_disparity"))

    if original_disparity is None or optimized_disparity is None:
        reduction_pct = None
    elif original_disparity > 0:
        reduction_pct = ((original_disparity - optimized_disparity) / original_disparity) * 100
    else:
        reduction_pct = 0

    fairness_improvement = {
        "original": original_disparity,
        "optimized": optimized_disparity,
        "reduction_pct": reduction_pct,
    }

    # Compute accuracy change. Same rule: 0.000 -> 0.000 and a green "+0.00%"
    # is a claim about an accuracy cost that was never computed.
    original_accuracy = _as_float(data.get("original_accuracy"))
    optimized_accuracy = _as_float(data.get("optimized_accuracy"))

    accuracy_change = {
        "original": original_accuracy,
        "optimized": optimized_accuracy,
        "change": (
            None
            if original_accuracy is None or optimized_accuracy is None
            else optimized_accuracy - original_accuracy
        ),
    }

    # The run's CONFIGURATION, and only what the caller supplied of it. These
    # three cells are read as a record of how the optimiser was set up, so a
    # library default printed in them is a claim about the run: until 2026-08-27
    # `threshold_optimization_to_svg({})` stated CONSTRAINT demographic_parity,
    # TOLERANCE 0.05 and OBJECTIVE accuracy under those headings, beside panels
    # that honestly read NOT MEASURED and NOT ESTABLISHED.
    #
    # It is not only an empty dict. `ThresholdResult.to_dict()` carries none of
    # these keys, so a REAL optimisation run with constraint="equalized_odds"
    # and tolerance=0.10 rendered "demographic_parity" and "0.05" as its own
    # configuration: the defaults can contradict the run they are printed
    # beside, and nothing on the canvas distinguished them.
    #
    # A supplied value renders exactly as before; an absent one is withheld and
    # the canvas says the configuration was not supplied.
    constraint_type = data.get("constraint_type")
    tolerance = data.get("tolerance")
    objective = data.get("objective")

    # Which group rows were actually EVALUATED. A row carrying only a threshold
    # reports no original rate, no optimised rate and no size, so nothing about
    # that group was measured; the headline still counted it as "evaluated",
    # which is (b), (c) and (d) of the headline rule broken at once on the one
    # line a reader takes the scope of the run from. `ThresholdResult.to_dict()`
    # carries group_thresholds and nothing else, so this was every real
    # optimiser run.
    evaluated_groups: list = []
    unevaluated_groups: list = []
    for g in groups:
        measured = any(
            _as_float(g.get(key)) is not None for key in ("original_rate", "optimized_rate", "size")
        )
        (evaluated_groups if measured else unevaluated_groups).append(g)

    # Build template data
    template_data = {
        "title": data.get("title", "Threshold Optimization Analysis"),
        "timestamp": data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "constraint_type": constraint_type,
        "constraint_known": constraint_type is not None,
        "tolerance": tolerance,
        "tolerance_known": _as_float(tolerance) is not None,
        # The same rule, on the one configuration field only the fallback card
        # prints: 0.5 is this library's default decision threshold, not a
        # threshold the caller reported using.
        "original_threshold": data.get("original_threshold"),
        "groups": groups,
        "fairness_improvement": fairness_improvement,
        "accuracy_change": accuracy_change,
        # Feasibility is the optimiser's VERDICT, so a result that does not
        # carry one has not established it. The old default of True painted a
        # green FEASIBLE badge on an empty dict.
        "is_feasible": data.get("is_feasible"),
        "target_rate": data.get("target_rate"),
        "recommendation": data.get(
            "recommendation",
            "Use optimized group-specific thresholds to achieve fairness constraint.",
        ),
        "objective": objective,
        "objective_known": objective is not None,
        # True when the caller supplied NONE of the three: the subtitle then
        # says the configuration was not supplied, rather than leaving three
        # withheld cells to be read as a rendering glitch.
        "config_known": bool(
            constraint_type is not None or tolerance is not None or objective is not None
        ),
        # No default. 100 is the optimiser's library default sweep size, not a
        # number this run reported, and it is the same class as the constraint /
        # tolerance / objective cells above: a configuration claim about a run
        # that never stated it. `threshold_optimization_report.svg` does not draw
        # this key today, so nothing on the canvas changes; the default is
        # removed so that a cell added here later cannot inherit a fabricated
        # sweep size from this line.
        "n_thresholds": data.get("n_thresholds"),
        # The headline says "N groups evaluated", and a group row built from
        # `group_thresholds` alone carries no rate and no size, i.e. it was NOT
        # evaluated. Part (d) of the headline rule: an ungraded row enters no
        # numerator and no denominator, so the headline counts the rows that
        # reported at least one measurement and names the rest.
        "n_groups_evaluated": len(evaluated_groups),
        "n_groups_unevaluated": len(unevaluated_groups),
        "unevaluated_group_names": ", ".join(
            str(g.get("name", "")) for g in unevaluated_groups[:4]
        ),
    }

    # Nothing was optimised, nothing was measured and nothing was decided: no
    # group row, no disparity figure, no accuracy figure, no feasibility
    # verdict. That is could-not-check, and it must not read as a successful
    # zero-cost mitigation.
    if (
        not groups
        and reduction_pct is None
        and accuracy_change["change"] is None
        and template_data["is_feasible"] is None
    ):
        template_data["not_assessable"] = True
        template_data["not_assessable_reason"] = (
            "No group was optimised and no disparity, accuracy or feasibility result was supplied."
        )
        if "recommendation" not in data:
            template_data["recommendation"] = (
                "No threshold optimisation result was supplied, so this report does not "
                "recommend deploying group-specific thresholds. Re-run the optimiser and "
                "render its result."
            )

    try:
        template_data["explanation"] = explanation
        svg = render_svg("threshold_optimization_report", template_data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        # The ``open(save_path, "w")`` above truncates before it writes, so a
        # failure here left the caller with a zero-byte file while this branch
        # returned a perfectly good fallback that never reached the disk. The
        # fallback goes to the file too, so the return value and the artifact
        # on disk say the same thing.
        #
        # CRITICAL, and the reason the card changed on 2026-08-28: what this
        # branch returned was a SILENT DEGRADATION. Executed on this repo, one
        # group supplied as ``{"name": "female"}`` with no ``threshold`` key
        # makes Jinja raise UndefinedError, and the old fallback answered with a
        # plain 800x400 page headed "Threshold Optimization Analysis" that
        # listed the group thresholds and said nothing about the failure. It
        # carried no feasibility badge, no disparity figure and no accuracy
        # change, the three results this report exists to state; it never went
        # through ``inject_accessibility``, so it had no role, no title and no
        # <desc> either. A caller who passed save_path was left holding a file
        # that LOOKS like the report, announced only by a UserWarning, and
        # warnings are filtered by default in exactly the batch pipelines that
        # call this. That is the same family as the empty-string returns removed
        # earlier: the caller is not told.
        #
        # The shared card states COULD NOT CHECK on the canvas AND in the
        # <desc>, so it can be read as neither a pass nor a failure, and it is
        # the SAME card and writer as the Jinja-missing branch above and as
        # method_comparison_to_svg. Do not grow a second copy of it here.
        warnings.warn(f"SVG rendering failed: {e}")
        svg = _could_not_render_svg(
            template_data.get("title", "Threshold Optimization Analysis"),
            f"The chart could not be drawn ({type(e).__name__} while rendering the template).",
            template_data.get("timestamp", ""),
        )
        _write_svg(save_path, svg)
        return svg


def reweighting_comparison_to_svg(
    report: Union["ReweightingAnalysisReport", Dict[str, Any]],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render reweighting method comparison as an SVG dashboard.

    This function creates a comprehensive visualization comparing different
    prediction reweighting methods, showing:
    - Fairness improvement per method: the relative reduction of the
      demographic-parity gap, (original - adjusted) / original, rendered
      as a percentage. The same formula is applied whether the input is a
      ReweightingAnalysisReport object or a plain dict.
    - Accuracy and calibration trade-offs
    - Trade-off score ranking
    - Recommendations

    Args:
        report: ReweightingAnalysisReport from ReweightingAnalyzer.full_analysis(),
                or a dictionary with the required fields.
        save_path: Optional path to save the SVG file.

    Returns:
        SVG markup string.

    Example:
        >>> from vfairness.post_processing import ReweightingAnalyzer
        >>> from vfairness.rendering import reweighting_comparison_to_svg
        >>>
        >>> analyzer = ReweightingAnalyzer(y_true, y_prob, gender)
        >>> report = analyzer.full_analysis()
        >>> svg = reweighting_comparison_to_svg(report, save_path='reweighting.svg')
    """
    if not JINJA2_AVAILABLE:
        # Same export path, same reason as threshold_optimization_to_svg above:
        # writing nothing leaves a stale chart in place of a chart that was
        # never drawn, so the could-not-check file is written before the raise.
        _write_svg(
            save_path,
            _could_not_render_svg(
                "Prediction Reweighting Method Comparison",
                "Jinja2 is not installed, so this chart could not be rendered.",
            ),
        )
        raise_jinja2_missing()

    # Handle both report objects and dictionaries
    if hasattr(report, "to_dict"):
        data = report.to_dict()
    elif isinstance(report, dict):
        data = report
    else:
        data = {}

    # Extract method results.
    # fairness_improvement semantics (both input shapes, dict and object):
    # the RELATIVE reduction of the demographic-parity gap, i.e.
    # (original - adjusted) / original, in [0, 1] for an improvement. The
    # template renders it as a percentage ('+50.0%'). The object path used
    # to emit the ABSOLUTE gap change instead, so the same analysis showed
    # different numbers depending on which input type reached the adapter.
    method_results = data.get("method_results", [])
    methods = []
    ungraded_methods = []

    def _sub(mr: Any, key: str) -> Any:
        """One sub-mapping off either input shape, or None when it is absent.

        A missing ``calibration_metrics`` used to be ``None.get(...)`` on the
        object path, i.e. an AttributeError raised out of the adapter before the
        try/except below could turn it into a could-not-check canvas. Absent is
        not an error here, it is a cell that reported nothing.
        """
        sub = mr.get(key) if isinstance(mr, dict) else getattr(mr, key, None)
        return sub if isinstance(sub, dict) else None

    def _cell(mr: Any, key: str, inner: Optional[str] = None) -> Optional[float]:
        """One reported number, or None. NEVER a default.

        CRITICAL, per ROW. Every cell on this table used to arrive through a
        ``.get(key, 0)`` default, so a method result that carried nothing but
        its own name was written onto the canvas as four measurements: an
        accuracy change of +0.00% (drawn in the emerald "improved" arm, because
        0 >= 0), a calibration change of 0.000 (drawn in the emerald "did not
        get worse" arm, because 0 <= 0), a trade-off score of 0.000 and a
        fairness improvement of +0.0%. Absent and zero are different claims: a
        method that really scored 0.0 still prints, because `is None` and not
        truthiness is what is tested here.
        """
        if inner is None:
            raw = mr.get(key) if isinstance(mr, dict) else getattr(mr, key, None)
        else:
            sub = _sub(mr, key)
            raw = None if sub is None else sub.get(inner)
        return _as_float(raw)

    def _relative_improvement(orig_dp, adj_dp):
        # Both operands must have been REPORTED. (0 - 0) / 0 used to fall into
        # the `else 0` arm and print "+0.0%", which reads as "this method was
        # applied and moved the gap by nothing" for a method that was never
        # applied. A genuine original gap of 0.0 still returns 0, because that
        # pair IS a measurement.
        if orig_dp is None or adj_dp is None:
            return None
        return (orig_dp - adj_dp) / orig_dp if orig_dp > 0 else 0

    def _change(new, old):
        """new - old, or None unless BOTH ends were reported."""
        if new is None or old is None:
            return None
        return new - old

    for mr in method_results:
        name = (
            mr.get("method", "unknown")
            if isinstance(mr, dict)
            else getattr(mr, "method", "unknown")
        )
        row = {
            "name": name,
            "fairness_improvement": _relative_improvement(
                _cell(mr, "original_fairness", "demographic_parity_diff"),
                _cell(mr, "adjusted_fairness", "demographic_parity_diff"),
            ),
            "accuracy_change": _change(
                _cell(mr, "adjusted_performance", "accuracy"),
                _cell(mr, "original_performance", "accuracy"),
            ),
            "calibration_change": _cell(mr, "calibration_metrics", "ece_change"),
            "trade_off_score": _cell(mr, "trade_off_score"),
        }
        # A row that reported NOTHING is not a comparison. It is held out of
        # `methods` entirely, so it takes no place in the subtitle count, in the
        # trade-off maximum that scales every bar, or in the "across N methods"
        # denominator of the accessible <desc> (explain._fr_reweighting reads
        # len(methods)). The canvas still lists it, so the reader can see the
        # method was named and not measured.
        if any(
            row[k] is not None
            for k in (
                "fairness_improvement",
                "accuracy_change",
                "calibration_change",
                "trade_off_score",
            )
        ):
            methods.append(row)
        else:
            ungraded_methods.append({"name": name})

    # The original disparity is a MEASUREMENT, so it is read from the first
    # method result that actually REPORTED one rather than from whichever entry
    # happens to sit first. The `= 0` initialiser survived a first entry with no
    # original_fairness at all, and 0.000 is the best possible reading of a
    # disparity.
    original_disparity = None
    for mr in method_results:
        candidate = _cell(mr, "original_fairness", "demographic_parity_diff")
        if candidate is not None:
            original_disparity = candidate
            break

    # Computed here rather than in a {% set %} inside the template loop, for the
    # reason recorded at adapters_feature_engineering's badge_x: a Jinja
    # assignment made inside a loop is not testable and does not survive the
    # iteration. An ungraded row is not in this maximum either, so it cannot
    # rescale everyone else's bar.
    scores = [m["trade_off_score"] for m in methods if m["trade_off_score"] is not None]
    max_trade_off = max(scores) if scores else 0

    # "BEST METHOD" is a verdict, and a verdict may only stand on a row that was
    # measured. The default used to be methods[0]["name"], which promoted
    # whatever sat first in the list, and the badge is drawn in emerald with a
    # green left stripe whatever it holds. The fallback now names the first
    # GRADED method, and a caller-supplied best method that has no graded row is
    # drawn in slate and labelled, rather than painted as a recommendation.
    graded_names = {m["name"] for m in methods}
    best_method = data.get("best_method") or (methods[0]["name"] if methods else "N/A")

    # Build template data
    template_data = {
        "title": data.get("title", "Prediction Reweighting Method Comparison"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_samples": data.get("metadata", {}).get("n_samples", "N/A"),
        "n_groups": data.get("metadata", {}).get("n_groups", "N/A"),
        "methods": methods,
        # The ungraded rows travel in their OWN list. The subtitle count, the
        # trade-off scale and the <desc> denominator all read `methods`, and
        # part (d) of the headline rule is that an ungraded row enters neither a
        # numerator nor a denominator.
        "ungraded_methods": ungraded_methods,
        "n_ungraded": len(ungraded_methods),
        "max_trade_off": max_trade_off,
        "best_method": best_method,
        "best_method_graded": best_method in graded_names,
        "original_disparity": original_disparity,
        "recommendations": data.get("recommendations", []),
    }

    # Three states, never two. With no method result, nothing was reweighted and
    # nothing was compared. The badges already read N/A, but the explanation and
    # the accessible <desc> still asserted "an original disparity of 0.000 across
    # 0 methods": the 0.000 is the initialiser above, not a measurement, and a
    # disparity of zero is the best possible reading of that metric. The same
    # single flag now drives the canvas and the <desc>, so the two cannot
    # disagree; it is what ``rendering.explain._not_assessable`` reads.
    if not methods:
        template_data["not_assessable"] = True
        if ungraded_methods:
            # Every supplied row is a method NAME and nothing else. Drawing the
            # table would put a full set of column headings over rows that
            # measure nothing, under a subtitle counting methods "compared", so
            # the whole page is could-not-check rather than an honest table
            # under a dishonest headline. Same call as method_comparison_to_svg.
            template_data["not_assessable_reason"] = (
                f"None of the {len(ungraded_methods)} supplied method result(s) reported a "
                "fairness, accuracy, calibration or trade-off number, so no method was "
                "scored and none was compared against another."
            )
            template_data["na_headline"] = "No method was measured"
        else:
            template_data["not_assessable_reason"] = (
                "No reweighting method result was supplied, so no method was applied, no "
                "original disparity was measured and no trade-off was scored."
            )
            template_data["na_headline"] = "No method was compared"

    try:
        template_data["explanation"] = explanation
        svg = render_svg("reweighting_comparison_report", template_data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        # NEVER return "" here, for the reason recorded at the same branch of
        # fairness_detailed_report_to_svg: the ``open(save_path, "w")`` above
        # truncates before it writes, so a caller passing save_path was left
        # with a zero-byte file, no exception, and only a warning to explain
        # it. A zero-byte SVG cannot even be read as could-not-check.
        warnings.warn(f"SVG rendering failed: {e}")
        svg = _could_not_render_svg(
            template_data.get("title", "Prediction Reweighting Method Comparison"),
            f"The chart could not be drawn ({type(e).__name__} while rendering the template).",
            template_data.get("timestamp", ""),
        )
        _write_svg(save_path, svg)
        return svg


def _first_present(*candidates: Any, default: Any = None) -> Any:
    """First candidate that is not None, else ``default``."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return default


def _as_float(candidate: Any) -> Optional[float]:
    """Return ``candidate`` as a float, or None when it is not a measurement.

    Strings, None, booleans and per-group dicts are NOT measurements. Coercing
    them, or substituting a default, is how an absent number becomes a graded
    row: ``metric_info.get("value", 0)`` invented a 0.0, and 0.0 is inside every
    ceiling, so a metric that was never measured rendered a green PASS.

    G12, 2026-09-30. A NON-FINITE NUMBER WAS NOT A MEASUREMENT EITHER, and this
    function said it was: ``isinstance(nan, numbers.Real)`` is True, so NaN and
    the infinities were returned as floats, while the repo's single rule for
    this question, ``_triage.is_measured``, refuses all three. Every caller here
    then compared the value, and ``nan > 0`` is False, so the NaN took the
    "nothing to reduce" arm of an if/else rather than the could-not-check arm.
    Three measured instances, all of them on a verdict surface and all SILENT:

    * ``threshold_optimization_to_svg({... "original_disparity": nan,
      "optimized_disparity": 0.01})`` rendered the fairness-improvement badge as
      "N/A -> 0.010, DOWN 0.0%", and the comment on that very branch says a 0
      there means "there was nothing to reduce", i.e. the run was already fair.
      With BOTH disparities NaN it printed "N/A -> N/A" and "DOWN 0.0%" beside
      it: a measured 0% reduction between two values it had just declared
      unavailable.
    * ``_relative_improvement`` in the method-comparison adapter, the same
      ``if orig_dp > 0 else 0`` one function down, printed "+0.0%".
    * the evaluated/unevaluated group split counted a group whose only number
      was NaN as EVALUATED, on the one line a reader takes the scope of the run
      from.

    Refusing non-finite here fixes all three at their shared root rather than at
    three call sites, which is where the last copy of this defect survived.
    ``check_threshold`` keeps naming the reason (NaN vs infinite) for the metric
    rows; see ``fairness_detailed_report_to_svg``, which now asks
    ``unmeasurable_reason`` for it directly.
    """
    if isinstance(candidate, bool) or not isinstance(candidate, numbers.Real):
        return None
    as_float = float(candidate)
    if not math.isfinite(as_float):
        return None
    return as_float


def _metric_measurement(
    metric_name: str,
    metric_info: Any,
    thresholds_used: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], str]:
    """Pull (value, threshold, interpretation) out of either report shape.

    The engine emits ``metrics`` as ``{name: float}`` and carries the bounds in a
    separate ``thresholds_used`` dict (``FairnessAnalyzer.get_report``,
    ``classification_fairness_report``, ``regression_fairness_report``). Older
    hand-built report dicts nest ``{'value': ..., 'threshold': ...}`` per metric.
    Both are accepted. Reading ONLY the nested shape is what made this adapter
    inoperative on real engine output: every metric failed the isinstance test,
    so the executive dashboard rendered an empty table reading "0 metrics
    evaluated" underneath a confident overall score.

    Either element may come back None, which means could-not-check. There is no
    default threshold: grading a metric against a bound nobody set produces a
    verdict nobody asked for, and on a ratio metric a borrowed 0.1 turns a
    0.28 four-fifths ratio into a PASS.
    """
    interpretation = ""
    threshold_candidate: Any = None
    if isinstance(metric_info, dict):
        value = _as_float(metric_info.get("value"))
        threshold_candidate = metric_info.get("threshold")
        interpretation = str(metric_info.get("interpretation", "") or "")
    else:
        value = _as_float(metric_info)

    if threshold_candidate is None:
        threshold_candidate = thresholds_used.get(metric_name)
    threshold = _as_float(threshold_candidate)
    return value, threshold, interpretation


def fairness_detailed_report_to_svg(
    report: Dict[str, Any],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a comprehensive fairness analysis report as an SVG dashboard.

    This function creates an executive-level visualization of fairness
    analysis results, showing:
    - Overall fairness score and assessment
    - Group statistics and comparisons
    - All fairness metrics with pass/fail status
    - Key findings and recommendations

    Args:
        report: FairnessReport from FairnessAnalyzer.get_report(),
                classification_fairness_report() or regression_fairness_report()
                (metrics as ``{name: float}``, bounds in ``thresholds_used``,
                groups in ``group_stats``, sizes in ``data_info``), or an older
                hand-built dictionary (metrics as
                ``{name: {'value': ..., 'threshold': ...}}``, groups in
                ``group_statistics``, a top-level ``n_samples``). Both shapes
                are read.
        save_path: Optional path to save the SVG file.

    Returns:
        SVG markup string.

    Example:
        >>> from vfairness import FairnessAnalyzer
        >>> from vfairness.rendering import fairness_detailed_report_to_svg
        >>>
        >>> analyzer = FairnessAnalyzer(y_true, y_pred, gender)
        >>> report = analyzer.get_report()
        >>> svg = fairness_detailed_report_to_svg(report, save_path='fairness.svg')
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    # Handle both report objects and dictionaries
    if hasattr(report, "to_dict"):
        data = report.to_dict() if callable(getattr(report, "to_dict")) else report
    elif isinstance(report, dict):
        data = report
    else:
        data = {}

    # Extract assessment info. A missing score must surface as an explicit
    # N/A state: the old fallback of 50 was scaled by 100 and rendered as
    # a '5000/100' score with a 'FAIR' verdict for a report that carried
    # no score at all.
    assessment = data.get("assessment", {}) or {}
    raw_candidate = assessment.get("fairness_score") if isinstance(assessment, dict) else None
    # G12, 2026-09-30. THE HEADLINE VERDICT WAS THE ONE NUMBER ON THIS CANVAS
    # THAT DID NOT USE THIS FILE'S OWN MEASUREMENT RULE. The coercion here was
    # `float(raw_score) if raw_score is not None else None`, i.e. "is it not
    # None", while every metric row beneath it went through `_as_float`, whose
    # docstring says strings and booleans are NOT measurements. Three doors were
    # open at once, every one of them SILENT, measured on this adapter:
    #
    #   fairness_score = nan   -> badge UNFAIR, score rendered "0", no warning.
    #       `nan <= 1` is False and `nan >= 80` / `>= 50` are both False, so a
    #       could-not-check fell all the way through to the WORST measured
    #       verdict the badge has.
    #   fairness_score = True  -> badge FAIR, score 100. `float(True) == 1.0`,
    #       which this repo's `_triage.is_measured` refuses BY NAME for exactly
    #       this reason: a yes/no flag clamps to a perfect score.
    #   fairness_score = inf   -> `int(inf)` raised inside the template, so the
    #       whole report came back as the render-failure card. Had the template
    #       formatted it, `inf >= 80` is True and the badge would have read FAIR.
    #   fairness_score = "0.92" -> badge FAIR at 92, while a metric row carrying
    #       the string "0.02" on the same canvas was refused as "no usable
    #       value". One dashboard, two answers to the same question.
    #
    # `_as_float` is this file's single answer, so the headline now asks it. The
    # third state is already fully built here (score None -> the template's
    # NOT ASSESSABLE branch, plus the warning below), so every one of those four
    # is now disclosed instead of graded, and a real 0.92 still renders FAIR 92.
    raw_score = _as_float(raw_candidate)

    if raw_score is None:
        fairness_score = None
        overall_assessment = "N/A"
    else:
        fairness_score = raw_score * 100 if raw_score <= 1 else raw_score
        # Determine overall assessment
        if fairness_score >= 80:
            overall_assessment = "FAIR"
        elif fairness_score >= 50:
            overall_assessment = "MARGINAL"
        else:
            overall_assessment = "UNFAIR"

    # Extract metrics. Both report shapes are read (see _metric_measurement):
    # the engine's flat {name: float} plus `thresholds_used`, and the older
    # nested {'value': ..., 'threshold': ...} dict.
    metrics_data = data.get("metrics", {}) or {}
    thresholds_used = data.get("thresholds_used", {}) or {}
    data_info = data.get("data_info", {}) or {}
    if not isinstance(data_info, dict):
        data_info = {}
    metrics = []
    for metric_name, metric_info in metrics_data.items():
        value, threshold, interpretation = _metric_measurement(
            metric_name, metric_info, thresholds_used
        )

        if value is None or threshold is None:
            # No measurement, or no bound to measure it against. Fail closed:
            # this is could-not-check, never a pass and never a measured fail.
            missing = "value" if value is None else "threshold"
            outcome = ThresholdOutcome.COULD_NOT_CHECK
            outcome_message = (
                f"{metric_name} carries no usable {missing} in this report, "
                f"so it was never compared to a bound and nothing was graded"
            )
        else:
            # Direction-aware verdict. The old `abs(value) <= threshold` treated
            # every metric as lower-is-better, so a disparate_impact_ratio of
            # 0.00 (the protected group is never selected) rendered a green PASS
            # badge and 1.00 (perfect parity) rendered a red FAIL.
            outcome, outcome_message = check_threshold(metric_name, value, threshold)

        could_not_check = outcome is ThresholdOutcome.COULD_NOT_CHECK
        # Three states, never two. `passed` is a MEASURED result, so it is
        # True or False only when a comparison actually happened; a
        # could-not-check row carries None, because False is the
        # representation of "we checked and it failed".
        passed = None if could_not_check else outcome is ThresholdOutcome.PASS
        state = "could_not_check" if could_not_check else None
        if could_not_check:
            # The template DOES render a third state now (the neutral
            # "NO DATA" badge, the slate row stripe and the "(not applied)"
            # threshold cell). It reads an explicit `state` field first, so
            # declare the state outright rather than leaving the row to be
            # recognised by the "COULD NOT CHECK" prefix in its own prose:
            # a reworded or truncated interpretation line would silently
            # turn the row back into a red measured FAIL. The prefix is
            # kept as well, because it is what the reader sees on the
            # canvas under the metric name.
            interpretation = f"COULD NOT CHECK: {outcome_message}"
            warnings.warn(
                f"fairness_detailed_report: {outcome_message}",
                stacklevel=2,
            )
        row = {
            "name": metric_name.replace("_", " ").title(),
            "value": value,
            "threshold": threshold,
            "passed": passed,
            "interpretation": interpretation,
        }
        # State the direction outright so the printed operator cannot contradict
        # the badge beside it. The template can otherwise only infer it from
        # which inequality happens to hold in this one row, and it has no way to
        # call the shared _metric_direction helper itself.
        direction = metric_direction(metric_name)
        if direction is not MetricDirection.UNKNOWN:
            row["direction"] = direction.value
        if state is not None:
            row["state"] = state
        metrics.append(row)

    # Extract group statistics. `group_stats` is what the engine emits;
    # `group_statistics` is the older hand-built key. Reading only the latter
    # rendered GROUPS 0 and an empty groups table for every real report.
    group_stats = data.get("group_stats") or data.get("group_statistics") or {}
    groups = []
    for group_name, stats in group_stats.items():
        if not isinstance(stats, dict):
            continue
        # A rate the report does not carry is left as None, which the template's
        # pct1 filter renders as "N/A". The old fallback of 0 printed "0.0%" for
        # a rate that was never computed at all: a regression report has no TPR
        # or FPR, and a printed 0.0% reads as a measured zero.
        # The SIZE cell is the same rule as the rate cells beside it, and it was
        # the one that kept its default: a row carrying neither "size" nor
        # "count" printed a bare "0" in the GROUPS table, and "0" there says
        # this stratum is EMPTY, which is precisely the finding an auditor
        # hunts for. Absent is not zero. The template prints "not reported" for
        # None. (The engine's own key is "size"; "count" stays as the legacy
        # fallback.)
        groups.append(
            {
                "name": str(group_name),
                "size": stats.get("size", stats.get("count")),
                "positive_rate": stats.get("positive_rate", stats.get("selection_rate")),
                "tpr": stats.get("tpr", stats.get("true_positive_rate")),
                "fpr": stats.get("fpr", stats.get("false_positive_rate")),
            }
        )

    # Extract pairwise comparisons. The DISPARITY cell is graded on the canvas:
    # the template bands it green below 0.05, amber below 0.10 and red above,
    # and draws a bar whose width is the value. The old `, 0)` default therefore
    # rendered a full green "0.000" with no bar for a pair the report never
    # compared, which is the strongest all-clear this table can express, and it
    # was invented from an absent key. A pair that reported no difference gets
    # no number, no colour and no bar; the template says so.
    pairwise = data.get("pairwise_comparisons", [])
    pairwise_comparisons = []
    n_pairwise_unmeasured = 0
    for comp in pairwise:
        if isinstance(comp, dict):
            disparity = _as_float(_first_present(comp.get("disparity"), comp.get("difference")))
            if disparity is None:
                n_pairwise_unmeasured += 1
            pairwise_comparisons.append(
                {
                    "group_a": comp.get("group_a", comp.get("group1", "")),
                    "group_b": comp.get("group_b", comp.get("group2", "")),
                    "disparity": disparity,
                }
            )

    # Build template data
    template_data = {
        "title": data.get("title", "Fairness Analysis Report"),
        "timestamp": data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "task_type": data.get("task_type", "classification"),
        # The engine records the sample count in `data_info` (n_samples, and
        # final_size before it), not at the top level; reading only the top
        # level printed SAMPLES N/A on every real report.
        "n_samples": _first_present(
            data.get("n_samples"),
            data_info.get("n_samples"),
            data_info.get("final_size"),
            data.get("metadata", {}).get("n_samples")
            if isinstance(data.get("metadata"), dict)
            else None,
            default="N/A",
        ),
        "n_groups": len(groups),
        "attribute_name": data.get(
            "attribute_name", data.get("sensitive_attribute", "sensitive_attr")
        ),
        "fairness_score": fairness_score,
        "overall_assessment": overall_assessment,
        "metrics": metrics,
        "groups": groups,
        "pairwise_comparisons": pairwise_comparisons,
        "key_findings": data.get("key_findings", data.get("findings", [])),
        "recommendations": data.get("recommendations", data.get("action_items", [])),
    }

    if fairness_score is None:
        # Three states, never two. There is no number to print, and
        # `fairness_score|int` would render a confident "0/100", the mirror
        # image of the old fallback of 50 that rendered "5000/100 FAIR" for a
        # report carrying no score at all. The template already holds the third
        # state (its `_no_score` branch prints NOT ASSESSABLE in slate and draws
        # no assessment badge), so None is handed straight to it.
        #
        # This used to short-circuit to a bespoke 956-byte card built by string
        # formatting, bypassing render_svg entirely: it carried no COULD NOT
        # CHECK wording anywhere on the canvas, no severity in its <desc>, and
        # none of the machine-readable explanation metadata every other chart
        # in the library emits. It was the one report an SVG consumer could not
        # tell apart from an ordinary chart. Do not reinstate it.
        # G12: name WHY, because a score that is PRESENT and unusable (nan, a
        # boolean, a string) is a different thing for an operator to fix than a
        # score that is absent, and "is missing" was the only sentence either
        # one produced. The word "missing" is kept in both, because the suite
        # locates this refusal by it.
        why = (
            "is missing"
            if raw_candidate is None
            else (
                f"is present but is not a measurement "
                f"({unmeasurable_reason(raw_candidate) or f'not a number ({type(raw_candidate).__name__})'}), "
                f"so it is treated as missing"
            )
        )
        warnings.warn(
            f"fairness_detailed_report: report['assessment']['fairness_score'] "
            f"{why}; rendering an explicit could-not-check state."
        )

    # Nothing at all was established: no aggregate score, no metric row to
    # compare against a bound, and no group to compare against another. The
    # subtitle's "0 metrics evaluated" and the empty tables are counts of zero,
    # not findings of zero, so the whole verdict stack is replaced by the shared
    # could-not-check panel. This is the SAME flag
    # ``rendering.explain._not_assessable`` reads, so the canvas and the
    # accessible <desc> cannot disagree about which of the three states this is.
    if fairness_score is None and not metrics and not groups:
        template_data["not_assessable"] = True
        template_data["not_assessable_reason"] = (
            "The report carried no fairness score, no metric and no group, so no value "
            "was compared to a threshold and no verdict was reached."
        )
        template_data["na_headline"] = "No metric was checked"

    try:
        template_data["explanation"] = explanation
        svg = render_svg("fairness_detailed_report", template_data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        # NEVER return "" here. This is an export path: the ``open(save_path,
        # "w")`` above truncates before it writes, so a caller passing
        # save_path was left with a zero-byte file, no exception, and only a
        # warning to explain it. A zero-byte SVG is the worst of the three
        # states, because it cannot even be READ as could-not-check: the reader
        # sees a broken image and has no way to tell a failed render from a
        # failed model. Jinja has just raised, so the fallback below is plain
        # markup rather than a template.
        warnings.warn(f"SVG rendering failed: {e}")
        svg = _generate_render_failure_detailed_report_svg(template_data, e)
        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)
        return svg


def _xml_text(value: Any) -> str:
    """Escape a caller-supplied string for use as SVG character data.

    The fallback cards below are built by string formatting, not by Jinja, so
    nothing autoescapes for them: a report titled ``A & B`` produced markup that
    no XML parser would accept, and rsvg-convert refused the whole file.
    """
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _could_not_render_svg(title: str, reason: str, timestamp: str = "") -> str:
    """Could-not-check card for a chart that could not be drawn at all.

    The sibling of ``_generate_render_failure_detailed_report_svg`` for the
    threshold and reweighting surfaces. Deliberately plain markup: it is
    reached when Jinja is missing or has just raised, so nothing here may go
    through a template, and none of the report's values are printed, because
    none of them reached a canvas.

    It states COULD NOT CHECK on the canvas AND in the accessible ``<desc>``,
    so it reads as neither a pass nor a failure.
    """
    title = _xml_text(title)
    reason = _xml_text(reason)
    tail = "This chart is not a pass and not a failure. No value on it was measured, so it certifies nothing."
    return f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 680 210" font-family="'Inter','Segoe UI',system-ui,sans-serif">
    <title>{title}</title>
    <desc>COULD NOT CHECK: {reason} {tail}</desc>
    <rect width="680" height="210" fill="#f8f9fb"/>
    <text x="36" y="48" font-size="18" font-weight="700" fill="#0f172a">{title}</text>
    <rect x="36" y="76" width="644" height="112" rx="10" fill="#ffffff"/>
    <rect x="36" y="76" width="2.5" height="112" rx="1" fill="#64748b"/>
    <rect x="516" y="94" width="144" height="28" rx="14" fill="#f1f5f9"/>
    <text x="588" y="113" text-anchor="middle" font-size="12" font-weight="700" fill="#64748b" letter-spacing="0.4">NOT CHECKED</text>
    <text x="56" y="108" font-size="15" font-weight="700" fill="#64748b">This report was not drawn</text>
    <text x="56" y="132" font-size="10.5" fill="#64748b">{reason}</text>
    <text x="56" y="150" font-size="10.5" fill="#64748b">No value from this report reached the canvas.</text>
    <text x="56" y="174" font-size="9.5" fill="#94a3b8">{tail}</text>
    <text x="644" y="202" text-anchor="end" font-size="9" fill="#b0b8c4">{_xml_text(timestamp)}</text>
</svg>
"""


def _write_svg(save_path: Optional[str], svg: str) -> None:
    """Write *svg* to *save_path*, never raising out of a failure path.

    ``open(save_path, "w")`` TRUNCATES before it writes, so a caller who passed
    save_path and hit a failure path was left holding a zero-byte file with no
    exception to explain it. A zero-byte SVG is the worst of the three states:
    it cannot even be read as could-not-check, because the reader sees a broken
    image and cannot tell a failed render from a failed model.
    """
    if not save_path:
        return
    try:
        with open(save_path, "w") as f:
            f.write(svg)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        warnings.warn(f"Could not write the could-not-render SVG to {save_path}: {exc}")


def _generate_render_failure_detailed_report_svg(data: Dict[str, Any], error: Exception) -> str:
    """Could-not-check card for a detailed report whose template render failed.

    Reached only after ``render_svg`` raised, so it deliberately does not use
    Jinja and does not try to render any of the report's values: whatever went
    wrong, none of them reached the canvas, so none of them may be claimed here.

    It states COULD NOT CHECK on the canvas AND in the accessible ``<desc>``, in
    the same slate ``_could_not_check.svg`` uses, so it reads as neither a pass
    nor a failure. It replaces a bare ``return ""``; see the comment at the call
    site for why an empty string was the worst answer available.
    """
    title = _xml_text(data.get("title", "Fairness Analysis Report"))
    # Two drawn lines, not one. There is no wordwrap filter here (Jinja is what
    # just failed), and a single 131-character line at 10.5px runs past the
    # 680-wide viewBox: the tail of the sentence, which is the part saying that
    # nothing reached the canvas, was clipped off the right edge.
    reason_1 = f"The chart could not be drawn ({_xml_text(type(error).__name__)} while rendering the template)."
    reason_2 = "No fairness score, metric or group reached this canvas."
    return f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 680 210" font-family="'Inter','Segoe UI',system-ui,sans-serif">
    <title>{title}</title>
    <desc>COULD NOT CHECK: {reason_1} {reason_2} This chart is not a pass and not a failure. No value on it was measured, so it certifies nothing.</desc>
    <rect width="680" height="210" fill="#f8f9fb"/>
    <text x="36" y="48" font-size="18" font-weight="700" fill="#0f172a">{title}</text>
    <rect x="36" y="76" width="644" height="112" rx="10" fill="#ffffff"/>
    <rect x="36" y="76" width="2.5" height="112" rx="1" fill="#64748b"/>
    <rect x="516" y="94" width="144" height="28" rx="14" fill="#f1f5f9"/>
    <text x="588" y="113" text-anchor="middle" font-size="12" font-weight="700" fill="#64748b" letter-spacing="0.4">NOT CHECKED</text>
    <text x="56" y="108" font-size="15" font-weight="700" fill="#64748b">This report was not drawn</text>
    <text x="56" y="132" font-size="10.5" fill="#64748b">{reason_1}</text>
    <text x="56" y="150" font-size="10.5" fill="#64748b">{reason_2}</text>
    <text x="56" y="174" font-size="9.5" fill="#94a3b8">This chart is not a pass and not a failure. No value on it was measured, so it certifies nothing.</text>
    <text x="644" y="202" text-anchor="end" font-size="9" fill="#b0b8c4">{_xml_text(data.get("timestamp", ""))}</text>
</svg>
"""


def _generate_fallback_threshold_svg(data: Dict[str, Any]) -> str:
    """Generate a simple fallback SVG when templates are not available.

    RETIRED FROM THE FAILURE PATH, 2026-08-28, and deliberately NOT re-wired.
    ``threshold_optimization_to_svg`` used to answer a failed render with this
    card, and it reads as the report rather than as a failure: it repeats the
    title, lists the group thresholds, and states none of the three results the
    report exists for (feasibility, disparity reduction, accuracy change). See
    the CRITICAL note at that except branch. Kept, not deleted, because the
    formatting guards below record a fixed defect: a withheld tolerance arrives
    as None, and the old ``{...:.2f}`` format would have raised inside the
    caller's except branch, turning a render failure into a crash, while
    printing 0.05 instead would have restored the very default this adapter
    stopped claiming. Anything reached after a failed render must say COULD NOT
    CHECK on the canvas and in the ``<desc>``; use ``_could_not_render_svg``.
    """
    constraint_txt = data.get("constraint_type") or "not supplied"
    tolerance = data.get("tolerance")
    tolerance_txt = f"{tolerance:.2f}" if isinstance(tolerance, numbers.Real) else "not supplied"
    original_threshold = data.get("original_threshold")
    threshold_txt = (
        f"{original_threshold:.3f}"
        if isinstance(original_threshold, numbers.Real)
        else "not supplied"
    )
    groups_text = ""
    y_offset = 150
    for group in data.get("groups", []):
        # Same three states per row, and the same crash-safety: an absent
        # threshold or rate now reaches here as None, which ``:.3f`` cannot
        # format.
        g_threshold = group.get("threshold")
        g_rate = group.get("optimized_rate")
        g_threshold_txt = (
            f"{g_threshold:.3f}" if isinstance(g_threshold, numbers.Real) else "not reported"
        )
        g_rate_txt = f"{g_rate:.3f}" if isinstance(g_rate, numbers.Real) else "not reported"
        groups_text += f"""
        <text x="50" y="{y_offset}" font-size="12">
            {group["name"]}: threshold={g_threshold_txt}, rate={g_rate_txt}
        </text>
        """
        y_offset += 25

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
    <rect width="800" height="400" fill="#f8f9fa"/>
    <text x="400" y="40" font-size="20" font-weight="bold" text-anchor="middle" fill="#333">
        {data.get("title", "Threshold Optimization Report")}
    </text>
    <text x="50" y="80" font-size="14" fill="#666">
        Constraint: {constraint_txt} | Tolerance: {tolerance_txt}
    </text>
    <text x="50" y="110" font-size="14" fill="#666">
        Original Threshold: {threshold_txt}
    </text>
    <text x="50" y="130" font-size="14" font-weight="bold" fill="#333">Group Thresholds:</text>
    {groups_text}
    <text x="400" y="380" font-size="10" text-anchor="middle" fill="#999">
        Generated by vfairness | {data.get("timestamp", "")}
    </text>
</svg>
"""
