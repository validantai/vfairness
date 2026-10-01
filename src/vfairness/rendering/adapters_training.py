"""
SVG Rendering Adapters for Training Module.

This module provides adapters for converting training-related reports
and results into SVG visualizations.

Adapters Implemented:
    - training_report_to_svg: Render FairnessTrainingReport as dashboard
    - training_analysis_report_to_svg: Comprehensive full-page analysis report
    - method_comparison_to_svg: Render method comparisons chart
    - tradeoff_analysis_to_svg: Render accuracy-fairness trade-off plot
"""

import math
import numbers
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .adapters import _fmt_ts

# The could-not-render card and its writer live with the post-processing
# adapters, where they were introduced. Imported rather than copied: a second
# copy of a rule is how one of them drifts, and this one is a safety net.
from .adapters_post_processing import _could_not_render_svg, _write_svg
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

if TYPE_CHECKING:
    from ..in_processing.analyzer import FairnessTrainingReport, MethodComparison


# The severity stamped on a critical issue that arrived as a bare STRING, i.e.
# carrying no severity of its own.
#
# CLOSED 2026-09-07. It WAS the pattern this wave exists to remove, in its
# string-valued shape. MEDIUM is a GRADE, an assessment of how much the issue
# matters, and a bare description string says nothing about that.
# Both training templates draw this value exactly as they draw a REPORTED
# medium: an amber dot and an amber MEDIUM chip (training_report.svg rows
# 249/252/253, training_analysis_report.svg rows 343/346/347). Reproduce with
#
#     training_report_to_svg(report(critical_issues=["some finding"]))
#
# and read the chip: an issue nobody graded is presented as a graded one.
#
# THE FIX, ready to apply in one step: set this to "unrated". It needs no
# template change at all, because "unrated" falls to the neutral slate arm of
# the `high / medium / else` chain both templates already write, and it is
# distinguishable from a genuinely reported "low" by its label, which is the
# cell a reader looks at. Verified by rasterising: the chip reads UNRATED in
# slate and fits its 50px width; a reported severity is untouched, so a run
# whose issues all carry one renders unchanged.
#
# APPLIED 2026-09-07, with its gating condition, in the same commit:
# `tests/test_adapters_training_verdicts.py::TestTrainingReport
# ::test_a_string_issue_is_wrapped_rather_than_dropped` asserted `">MEDIUM<" in
# svg`, which pinned the defect. That assertion pins the WRAPPING (the issue must
# survive rather than be dropped), so it now asserts `">UNRATED<"` and a second
# test asserts a graded issue is still drawn at its own severity.
_UNGRADED_ISSUE_SEVERITY = "unrated"


def _mark_not_assessable(data: Dict[str, Any], reason: str, headline: str) -> None:
    """Arm the third state on one chart's data dict: could-not-check.

    CRITICAL, three states, never two. A count of zero is not a finding of zero.
    ``tradeoff_analysis_to_svg({})`` drew an empty but fully labelled scatter and
    reported "0 of 0 configurations satisfy the constraint; 0 on the Pareto
    frontier" at severity INFO, which is the canvas for a search that ran and
    found nothing wrong. No configuration had been trained.

    Setting all three keys here means the canvas and the accessible ``<desc>``
    cannot disagree: the template swaps its verdict stack for the shared
    ``_could_not_check.svg`` panel, and ``rendering.explain._not_assessable``
    reads the SAME flag to replace the finding sentence and the action line.
    """
    data["not_assessable"] = True
    data["not_assessable_reason"] = reason
    data["na_headline"] = headline


#: Returned by :func:`_disclosure` when the producer wrote no flag at all.
#: A distinct object, not None, because None is a flag the producer CAN write to
#: say "I looked and could not tell", and that is a different answer from
#: "nobody said".
_NO_DISCLOSURE = object()


def _disclosure(container: Any, key: str) -> Any:
    """What the PRODUCER itself said about *key*, without deciding it for them.

    CRITICAL, and this is the whole reason it is not a ``.get`` with a default.
    ``parameters.get("accuracy_measured", True)`` answers MEASURED for a producer
    that never spoke, and ``.get(..., False)`` answers NOT MEASURED for the same
    silence. Both publish a verdict nobody reached. A ``.get`` default also does
    NOT fire when the key is PRESENT holding None, so the default reads as the
    producer's answer in exactly the case the producer said it could not tell.

    Four answers, and the caller must handle each:

        True              the producer measured it
        False            the producer says this is NOT a measurement
        None             the producer looked and could not tell
        _NO_DISCLOSURE   the producer wrote nothing here

    Anything that is not a bool is folded into the could-not-tell answer rather
    than being coerced, because ``bool("false")`` is True and a string is not a
    verdict.
    """
    if not isinstance(container, dict) or key not in container:
        return _NO_DISCLOSURE
    raw = container.get(key)
    return raw if isinstance(raw, bool) else None


def _may_print(flag: Any, number: Optional[float]) -> bool:
    """True only when a producer's disclosure AND the value itself allow printing.

    The producer's flag can only ever WITHHOLD here, never promote: a flag saying
    "measured" over a value that is not a finite number is a contradiction, and
    the renderer draws neither side of it. ``_coord`` supplies *number*, so a
    bool, a NaN and an infinity have already been refused.
    """
    return number is not None and (flag is True or flag is _NO_DISCLOSURE)


def _withheld_reason(
    container: Any, key: str, flag: Any, number: Optional[float], what: str
) -> str:
    """The sentence printed in place of a withheld number.

    The producer's own reason wins when it wrote one, because it knows WHY. The
    fallbacks name which of the three doors was taken, so "not measured" on the
    canvas is never the whole story a reader gets.
    """
    given = container.get(key) if isinstance(container, dict) else None
    reason = str(given).strip() if isinstance(given, str) else ""
    if reason:
        return reason
    if flag is False:
        return f"the producer reports this {what} is not a measurement"
    if flag is None:
        return f"the producer could not tell whether this {what} is a measurement"
    if number is None:
        return f"no finite {what} was reported"
    return ""


def _baseline_state(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Report what the baseline verdict panel is ALLOWED to print.

    CRITICAL, three states, never two. ``constraint_satisfied`` used to be read
    as ``baseline.get("constraint_satisfied", False)``, and False is the branch
    that paints the red VIOLATED chip. A report whose baseline was never
    evaluated therefore rendered "Accuracy 0.0000 | Violation 0.0000 |
    Constraint VIOLATED" at accessible severity HIGH. That is a fabricated
    FAILURE, the mirror image of the fabricated all-clears closed in the other
    waves and just as wrong: it sends a reader to investigate a violation that
    does not exist, and HIGH is the grade a real breach gets.

    A measured zero and an absent measurement are different claims, so every
    test here is ``is not None`` and never truthiness: an accuracy that really
    came out 0.0 still prints 0.0000 and still carries its constraint verdict.

    ``constraint_evaluated`` is deliberately a WHOLE-CHART signal for these two
    reports, not a per-panel one. ``explain._fr_training_report`` writes the
    constraint into the accessible description from the same field, with no
    third branch of its own, and ``build_explanation`` can only replace that
    sentence when the chart's ``not_assessable`` flag is set. A canvas that said
    NOT CHECKED beside a ``<desc>`` still reading "constraint violated" at
    severity HIGH would be the same fabrication moved somewhere a sighted reader
    cannot see it, and the description must never be more confident than the
    badge. So the caller raises the chart-level flag, and both layers change
    together off one signal.

    THE PRODUCER'S OWN FLAG IS READ, NOT RE-DERIVED (2026-09-30). Two separate
    defects lived in the `accuracy is not None` test this replaces, and both were
    executed on this tree:

    1. ``FairnessTrainingAnalyzer.generate_report`` now puts
       ``accuracy_measured`` and ``accuracy_not_measured_reason`` INTO
       ``baseline_metrics`` beside the number. Re-deriving the verdict from the
       value's shape threw that away, and NaN is not None: a baseline carrying
       ``accuracy=nan, accuracy_measured=False`` drew ``<text ...>nan</text>`` as
       a bold measured number and an accessible description reading "Baseline
       accuracy nan, violation 0.020 (constraint satisfied)" at severity LOW.
       Every producer-side check passed; the reader saw the fabrication.
    2. ``baseline_measured`` was an OR over two independent cells and gated a
       template block that prints BOTH, so a baseline with a measured accuracy
       and no violation said "measured" and then took the render down:
       ``{% if baseline_violation > 0.1 %}`` raised ``TypeError: '>' not
       supported between instances of 'NoneType' and 'float'``, the adapter
       swallowed it as a warning and returned an 1179-byte fallback card. One
       unmeasured cell must cost that cell, never the page, so the two cells are
       gated separately now. ``baseline_measured`` stays as the OR because its
       OTHER caller asks a different question ("did this report carry any
       baseline number at all", for the chart-level nothing-at-all test).
    """
    accuracy = _coord(baseline.get("accuracy"))
    violation = _coord(baseline.get("fairness_violation"))
    satisfied = baseline.get("constraint_satisfied")
    accuracy_flag = _disclosure(baseline, "accuracy_measured")
    accuracy_measured = _may_print(accuracy_flag, accuracy)
    violation_measured = violation is not None
    return {
        # Withheld at the boundary as well as gated in the template: a cell the
        # producer disowned must not be reachable by a template that forgets its
        # gate, and `|f4` prints a NaN straight through as the string "nan".
        "baseline_accuracy": accuracy if accuracy_measured else None,
        "baseline_violation": violation if violation_measured else None,
        "baseline_satisfied": satisfied,
        "baseline_accuracy_measured": accuracy_measured,
        "baseline_violation_measured": violation_measured,
        "baseline_accuracy_reason": _withheld_reason(
            baseline,
            "accuracy_not_measured_reason",
            accuracy_flag,
            accuracy,
            "accuracy",
        )
        if not accuracy_measured
        else "",
        "baseline_measured": accuracy_measured or violation_measured,
        "constraint_evaluated": satisfied is not None,
    }


def _recommendation_state(rec: Any) -> Dict[str, Any]:
    """Normalise the recommendation, and say whether one was actually produced.

    CRITICAL, do not go back to reading the attributes unguarded.
    ``training_report_to_svg`` reached straight for
    ``report.recommendation.recommended_method`` and raised AttributeError when
    the analyser produced no recommendation, so the two training renders
    disagreed about the very same report: one crashed, the other invented.
    Neither is right. A missing recommendation is the third state, not a crash
    and not an "N/A" wearing a LOW priority chip, because LOW is a priority
    verdict about advice that was never given.
    """
    method = getattr(rec, "recommended_method", None) if rec is not None else None
    given = bool(method) and str(method).strip().upper() not in {"N/A", "NONE"}
    # ``priority`` keeps its old "low" fallback because
    # ``explain._fr_training_report`` grades the BASELINE from it, and that
    # grade is honest whether or not advice was attached. What was not honest
    # was drawing a LOW chip beside "N/A" on the canvas, so the chip is gated on
    # ``given`` in the template rather than on the value being present.
    return {
        "method": method if given else "N/A",
        "priority": (getattr(rec, "priority", None) or "low") if given else "low",
        "rationale": (getattr(rec, "rationale", None) or "") if rec is not None else "",
        "alternatives": (getattr(rec, "alternative_methods", None) or [])
        if rec is not None
        else [],
        "given": given,
    }


def _method_state(m: Any) -> Dict[str, Any]:
    """One method-comparison ROW, with each cell withheld unless it was reported.

    CRITICAL, three states, never two, and this is the FABRICATED-FAILURE
    direction. ``constraint_satisfied`` reached both training canvases raw, and
    every template branch that decides the row is written ``{% if m.satisfied %}
    ... {% else %} FAIL``, so ``None`` took the FAIL arm: a red row stripe, a red
    violation number and a red FAIL chip for a method whose constraint was never
    evaluated. Executed on this repo, ``MethodComparison("adversarial", 0.79,
    0.06, None)`` rendered PASS beside FAIL on both ``training_report`` and
    ``method_comparison``, and a reader has no way to tell that invented failure
    from a measured one. It sends someone after a violation that does not exist.

    ``accuracy`` and ``fairness_violation`` are withheld the same way, and for a
    second reason as well: ``method_comparison.svg`` multiplies both by a bar
    width, so a ``None`` there did not render a wrong number, it raised
    TypeError, and the caller got a bare empty string back (see
    ``method_comparison_to_svg``).

    Every test is a measurability test and never truthiness: an accuracy that
    really came out 0.0, or a violation that really came out 0.0, is a
    measurement and still prints.

    THE PRODUCER'S OWN DISCLOSURES ARE READ, NOT RE-DERIVED (2026-09-30). This
    row was rebuilt field by field from eight names, and ``MethodComparison``
    has since grown three disclosures this rebuild silently dropped. All three
    were executed on this tree:

    * ``accuracy`` is typed as a plain ``float`` and so CANNOT carry None, which
      is exactly why the producers carry ``parameters['accuracy_measured']``
      beside it and set the number to NaN. ``accuracy is not None`` is True of a
      NaN, so every one of those refusals came back out of here as a claimed
      measurement: with ``accuracy=nan`` and ``parameters['accuracy_measured']
      = False``, ``parameters['accuracy_measured']`` read False while this
      function published True and ``method_comparison.svg`` drew
      ``<text ...>nan</text>`` as the row's accuracy with a blue bar beside a
      green PASS chip. Both ``evaluate_baseline`` and ``compare_methods`` now
      always set that key, so this one line converted every refusal on both arms
      back into a number.
    * ``fairness_violation`` was tested the same way, and a NaN there drew
      ``nan`` in the alert colour under a red FAIL chip: a fabricated breach.
    * ``parameters['degenerate_constant_predictions']`` says the constraint
      verdict is VACUOUS rather than good, because a model that makes no
      decision satisfies every rate-based constraint by construction. The
      producer warns, and ``generate_recommendation`` excludes the method from
      both rankings, and this canvas drew a green PASS chip with nothing beside
      it. A vacuous verdict is not a measured pass, so it is withheld from
      ``satisfied`` (which is what ``explain._count_true`` counts) and named on
      the row and in the headline instead.

    ``_disclosure`` is used rather than ``.get`` for all of them: a ``.get``
    default answers for a producer that never spoke, and does not fire at all
    when the key is PRESENT holding None, which is how these producers say "I
    looked and could not tell".
    """
    params = getattr(m, "parameters", None)
    accuracy = _coord(getattr(m, "accuracy", None))
    violation = _coord(getattr(m, "fairness_violation", None))
    satisfied = getattr(m, "constraint_satisfied", None)

    accuracy_flag = _disclosure(params, "accuracy_measured")
    accuracy_measured = _may_print(accuracy_flag, accuracy)
    violation_measured = violation is not None

    # A vacuous verdict and an absent one are different sentences, so they are
    # carried separately even though both leave the row ungraded.
    vacuous = _disclosure(params, "degenerate_constant_predictions") is True
    # The producer can also disown the verdict outright, independently of
    # `constraint_satisfied`: read the pair together, as its docstring says.
    evaluated_flag = _disclosure(params, "constraint_evaluated")
    graded = (
        satisfied is not None
        and not vacuous
        and (evaluated_flag is True or evaluated_flag is _NO_DISCLOSURE)
    )
    return {
        "name": getattr(m, "method_name", "method"),
        # Withheld at the boundary, not only gated in the template: `|f3` prints
        # a NaN through as the string "nan", so a template branch that forgets
        # its gate must have nothing to print.
        "accuracy": accuracy if accuracy_measured else None,
        "violation": violation if violation_measured else None,
        # None, not the raw verdict, when the verdict is vacuous or disowned:
        # `explain._count_true` counts this field's truthiness into "N of M
        # method(s) satisfy the fairness constraint", and a vacuous pass must not
        # be one of them.
        "satisfied": satisfied if graded else None,
        "time": getattr(m, "training_time", None),
        # `graded` gates the VERDICT chip, the row stripe and every count over
        # the rows; the two `*_measured` flags gate their own NUMBER and bar.
        # They are independent: a method may report a constraint verdict with no
        # accuracy beside it, or an accuracy with no verdict.
        "graded": graded,
        "accuracy_measured": accuracy_measured,
        "violation_measured": violation_measured,
        # The WHY, so "not measured" on the canvas is not the whole story a
        # reader gets. Empty string for a measured cell, so a healthy row
        # renders byte-identical to what it rendered before this rule existed.
        "accuracy_reason": _withheld_reason(
            params, "accuracy_not_measured_reason", accuracy_flag, accuracy, "accuracy"
        )
        if not accuracy_measured
        else "",
        "vacuous": vacuous,
        "vacuous_reason": (
            "this method collapsed to a constant prediction, so it satisfies every "
            "rate-based constraint by construction: the verdict is vacuous, not measured"
        )
        if vacuous
        else "",
    }


def _field(row: Any, name: str) -> Any:
    """Read one field of a configuration row, whatever shape the row is.

    CRITICAL, do not go back to ``row["accuracy"]``. That subscript is what
    raised ``KeyError('accuracy')`` out of ``tradeoff_analysis_to_svg`` for a
    sweep in which ONE configuration had not finished, so a partial run did not
    render a partial chart: it took the caller's process down, and a crash is
    not one of the three states. Missing is answered with None here, and the
    caller decides what may be drawn from that.
    """
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _coord(value: Any) -> Optional[float]:
    """A coordinate this scatter may plot, or None when the row reported none.

    CRITICAL, three states, never two, in the PLOTTING direction. A point is a
    position on two measured axes, so there is no honest place to draw a
    configuration that reported no accuracy or no violation: every pixel of the
    plot area is a claim about both numbers at once. ``.get("accuracy", 0)``
    put such a row at the left edge of the axis (and, through ``_axis_ranges``,
    stretched the axis down to it, moving every REAL point on the chart), and
    ``row["accuracy"]`` raised instead. Both are wrong.

    A bool is refused: ``True`` is an ``int`` in Python and would plot as
    accuracy 1.0, the best value on the axis. NaN and the infinities are
    refused because they are sentinels, and because an infinite coordinate
    raised OverflowError out of ``_axis_ranges`` (``math.ceil(inf / 0.05)``),
    which happens BEFORE the try block that could have explained it: executed
    on the tree at 927abdf, one configuration reporting an infinite accuracy
    took the whole render down.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    num = float(value)
    return num if math.isfinite(num) else None


def _count(value: Any) -> Optional[int]:
    """A reported whole-number count, or None when the row reported none.

    The integer sibling of :func:`_coord`, and it inherits every refusal: a bool
    is not a count, and neither is NaN or an infinity. Used for the group SIZE
    column, where a defaulted 0 reads as a group with no members in the training
    set, which is a finding rather than a blank.
    """
    num = _coord(value)
    return int(num) if num is not None else None


def _optimum(raw: Any) -> Optional[Dict[str, float]]:
    """The marked optimum, or None when it carries no coordinate pair.

    "Best Fair" is a LABELLED VERDICT drawn at a position, so it may only be
    drawn where both of its numbers were reported. ``best_fair.get("accuracy",
    0)`` fed the axis range from a default and ``best_fair["accuracy"]`` then
    raised in the very next statement, three lines apart, over the same absent
    field.
    """
    if raw is None:
        return None
    accuracy = _coord(_field(raw, "accuracy"))
    violation = _coord(_field(raw, "violation"))
    if accuracy is None or violation is None:
        return None
    return {"accuracy": accuracy, "violation": violation}


def _fmt4(value: Any) -> str:
    """Four decimals for the fallback canvas, or a phrase that claims nothing.

    The fallback formatted ``data.get("baseline_accuracy", 0)`` directly, so a
    baseline that was never evaluated printed "Accuracy: 0.0000" there too, and
    a None reaching it raised TypeError inside the very branch whose job is to
    survive a template failure.

    A NaN is refused for the same reason the engine's numeric filters refuse one
    (2026-09-30): ``float("nan")`` parses, so ``f"{float(value):.4f}"`` returns
    the string "nan", and this is the ONE canvas a reader looks at when the chart
    did not draw. A bool is refused because ``True`` would print as 1.0000.
    """
    if isinstance(value, bool):
        return "not measured"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "not measured"
    return f"{num:.4f}" if math.isfinite(num) else "not measured"


def _fmt3(value: Any) -> str:
    """Three decimals for the fallback card's method rows, or a phrase.

    The sibling of ``_fmt4``, kept at three decimals because that is the width
    those rows have always printed; only the absent case changes, and it changes
    from a TypeError to a phrase that claims nothing.

    Inherits every refusal of :func:`_fmt4`, NaN and bools included.
    """
    if isinstance(value, bool):
        return "not measured"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "not measured"
    return f"{num:.3f}" if math.isfinite(num) else "not measured"


def _nice_ceil(val: float, step: float = 0.05) -> float:
    """Round up to nearest multiple of *step*."""
    return math.ceil(val / step) * step


def _nice_floor(val: float, step: float = 0.05) -> float:
    """Round down to nearest multiple of *step*."""
    return math.floor(val / step) * step


def _axis_ranges(
    x_values: List[float],
    y_values: List[float],
) -> Tuple[float, float, float]:
    """Compute tight axis ranges from scatter-plot data.

    Returns ``(x_min, x_range, y_max)`` where y always starts at 0.
    All values are rounded to nice 0.05 boundaries with small padding.
    """
    # Y-axis (violation): always starts at 0
    if y_values:
        y_max = _nice_ceil(max(y_values) * 1.15, 0.05)
        y_max = max(y_max, 0.10)
    else:
        y_max = 1.0

    # X-axis (accuracy)
    if x_values:
        xlo, xhi = min(x_values), max(x_values)
        pad = max(0.02, (xhi - xlo) * 0.10)
        x_min = max(0.0, _nice_floor(xlo - pad, 0.05))
        x_max = min(1.0, _nice_ceil(xhi + pad, 0.05))
        if x_max - x_min < 0.10:
            x_max = min(1.0, x_min + 0.10)
    else:
        x_min, x_max = 0.0, 1.0

    return x_min, x_max - x_min, y_max


def training_report_to_svg(
    report: "FairnessTrainingReport",
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render FairnessTrainingReport as an SVG dashboard.

    Args:
        report: FairnessTrainingReport object
        save_path: Optional path to save the SVG

    Returns:
        SVG markup string

    Notes
    -----
    CRITICAL, three states, never two. Until 2026-08-27 an empty report rendered
    "Accuracy 0.0000 | Violation 0.0000 | Constraint VIOLATED" at accessible
    severity HIGH, and it raised AttributeError outright when *recommendation*
    was None. A constraint nobody evaluated must never be drawn as a constraint
    that failed: see ``_baseline_state`` and ``_recommendation_state``.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    baseline_state = _baseline_state(report.baseline_metrics or {})
    recommendation = _recommendation_state(report.recommendation)

    # Prepare data for template
    data = {
        "title": "Fairness Training Analysis Report",
        "timestamp": _fmt_ts(report.timestamp),
        "task_type": report.task_type,
        # Data info
        "n_samples": report.data_info.get("n_samples", "N/A"),
        "n_groups": report.data_info.get("n_groups", "N/A"),
        "n_features": report.data_info.get("n_features", "N/A"),
        "attribute_name": report.data_info.get("attribute_name", "sensitive_attr"),
        # Baseline metrics: numbers and verdict, each withheld unless measured
        **baseline_state,
        # Method comparisons, each cell withheld unless it was reported.
        # See _method_state for the fabricated FAIL this closes.
        "methods": [_method_state(m) for m in report.method_comparisons]
        if report.method_comparisons
        else [],
        # Recommendation
        "recommendation": recommendation,
        # Critical issues (normalize to dicts)
        # A bare-string issue carries no severity of its own, so it is labelled
        # "unrated" rather than graded. W-19, closed 2026-09-07.
        # See _UNGRADED_ISSUE_SEVERITY.
        "issues": [
            iss
            if isinstance(iss, dict)
            else {"type": "issue", "description": str(iss), "severity": _UNGRADED_ISSUE_SEVERITY}
            for iss in (report.critical_issues or [])[:5]
        ],
        # Action items
        "actions": report.action_items[:5] if report.action_items else [],
        # Trade-off analysis
        "tradeoff": report.tradeoff_analysis if report.tradeoff_analysis else {},
    }

    # The constraint verdict is this report's spine: it is the red VIOLATED chip
    # on the canvas AND the "(constraint violated)" clause of the accessible
    # description, both taken from the same field. Without it there is no
    # verdict to draw, so the whole page becomes could-not-check rather than
    # printing a baseline of zeros under a fabricated failure.
    if not baseline_state["constraint_evaluated"]:
        nothing_at_all = not any(
            [
                baseline_state["baseline_measured"],
                data["methods"],
                data["tradeoff"],
                data["issues"],
                data["actions"],
                recommendation["given"],
            ]
        )
        _mark_not_assessable(
            data,
            "The report carries no baseline metric, no trained method and no trade-off "
            "configuration, so no accuracy was recorded, no fairness violation was "
            "measured and no constraint was evaluated."
            if nothing_at_all
            else "The baseline carries no constraint result, so whether the fairness "
            "constraint holds was never evaluated on this run.",
            "No training run was analysed"
            if nothing_at_all
            else "The fairness constraint was not evaluated",
        )

    try:
        data["explanation"] = explanation
        svg = render_svg("training_report", data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        warnings.warn(f"SVG rendering failed: {e}")
        return _generate_fallback_svg(data)


def method_comparison_to_svg(
    comparisons: List["MethodComparison"],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render method comparisons as a bar chart SVG.

    Args:
        comparisons: List of MethodComparison objects
        save_path: Optional path to save the SVG

    Returns:
        SVG markup string

    Notes
    -----
    CRITICAL, do not restore the early ``return ""``. Until 2026-08-27 an empty
    *comparisons* list returned a zero-length string, and a caller that passed
    ``save_path`` got a 0-byte .svg on disk and no error at all: the file opens
    as a broken image, the pipeline that wrote it reports success, and nothing
    anywhere says why. A silent empty artifact is the worst of the three states,
    because it cannot even be read as could-not-check. Missing input now renders
    the shared COULD NOT CHECK canvas, which says on the face of it that no
    method was trained and no constraint was evaluated.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    methods = [_method_state(m) for m in comparisons]
    n_ungraded = sum(1 for m in methods if not m["graded"])
    # Counted apart from the rest of `n_ungraded` only so the sentences below can
    # say which of the two happened. A vacuous verdict IS ungraded, so it stays
    # inside that count and out of `explain`'s "N of M satisfy" numerator.
    n_vacuous = sum(1 for m in methods if m["vacuous"])
    data: Dict[str, Any] = {
        "title": "Fairness Training Method Comparison",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "methods": methods,
        # The subtitle counts the rows, and a count reads as a measurement, so
        # the ungraded ones are named in it rather than left to be assumed
        # evaluated. This is the decided headline rule (c): the ungraded count
        # goes on the CANVAS beside the headline, not only into the <desc>.
        "n_ungraded": n_ungraded,
        "n_vacuous": n_vacuous,
    }

    if not data["methods"]:
        _mark_not_assessable(
            data,
            "No training method was supplied for comparison, so no accuracy was "
            "recorded, no fairness violation was measured and no constraint was checked.",
            "No method was compared",
        )
    elif n_ungraded == len(methods):
        # Every row is a name and nothing else. The table would draw a full set
        # of column headings over rows that grade nothing, under a subtitle
        # claiming N methods were evaluated, so the whole page is could-not-check
        # rather than an honest-per-row table under a dishonest headline.
        #
        # The sentence names which of the two ways every row failed to be graded.
        # "None reported a constraint result" is FALSE about a run whose methods
        # all collapsed to a constant prediction: they each reported one, and
        # each one is vacuous.
        if n_vacuous == len(methods):
            reason = (
                f"All {len(methods)} supplied method(s) collapsed to a constant prediction, "
                "so every constraint result they reported is satisfied by construction and "
                "none of them measures whether the fairness constraint holds."
            )
            headline = "Every method's verdict is vacuous"
        elif n_vacuous:
            reason = (
                f"None of the {len(methods)} supplied method(s) reported a constraint result "
                f"that measures anything: {n_vacuous} collapsed to a constant prediction and "
                f"{len(methods) - n_vacuous} reported no constraint result at all."
            )
            headline = "No method's constraint was measured"
        else:
            reason = (
                f"None of the {len(methods)} supplied method(s) reported a constraint result, "
                "so whether any of them satisfies the fairness constraint was never evaluated."
            )
            headline = "No method's constraint was evaluated"
        _mark_not_assessable(data, reason, headline)

    try:
        data["explanation"] = explanation
        svg = render_svg("method_comparison", data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        # NEVER return "" here. A zero-length string is the worst of the three
        # states, because it cannot even be read as could-not-check: a caller
        # that writes the return value gets a 0-byte .svg that opens as a broken
        # image, and a caller that passed save_path is left holding whatever the
        # PREVIOUS render left on disk, with success reported either way. Only a
        # UserWarning said otherwise, and warnings are filtered by default in
        # exactly the batch pipelines that call this. Executed on this repo, one
        # MethodComparison with accuracy=None made the template raise TypeError
        # inside its bar-width arithmetic and this branch swallowed it whole.
        # Same failure card and same writer as adapters_post_processing, so the
        # two paths cannot drift; do not grow a second copy here.
        warnings.warn(f"SVG rendering failed: {e}")
        svg = _could_not_render_svg(
            data.get("title", "Fairness Training Method Comparison"),
            f"The chart could not be drawn ({type(e).__name__} while rendering the template).",
            data.get("timestamp", ""),
        )
        _write_svg(save_path, svg)
        return svg


def tradeoff_analysis_to_svg(
    tradeoff_data: Dict[str, Any],
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render accuracy-fairness trade-off as a scatter plot SVG.

    Args:
        tradeoff_data: Trade-off analysis dictionary
        save_path: Optional path to save the SVG

    Returns:
        SVG markup string
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    all_results = tradeoff_data.get("all_results") or []
    pareto_frontier = tradeoff_data.get("pareto_frontier") or []

    # PER ROW, three states, never two, and here the third state is that the row
    # gets no dot. A configuration is plotted only when it reported BOTH
    # coordinates and a constraint verdict:
    #
    #   * accuracy and violation are the position, and there is no position that
    #     means "not measured" (see _coord);
    #   * ``satisfied`` is the dot's COLOUR, and the template writes it
    #     `fill="{% if p.satisfied %}#059669{% else %}#dc2626{% endif %}"`, a
    #     two-state test in which None takes the red arm. A configuration whose
    #     constraint was never evaluated would be drawn as a green-legend
    #     "constraint violated" point, which is a fabricated breach.
    #
    # Withheld rows are counted and named in the headline (below), so a partial
    # sweep still gives a verdict over what it did measure without the reader
    # taking the chart for the whole run.
    plotted: List[Dict[str, Any]] = []
    n_unplotted = 0
    for r in all_results:
        accuracy = _coord(_field(r, "accuracy"))
        violation = _coord(_field(r, "violation"))
        satisfied = _field(r, "satisfied")
        if accuracy is None or violation is None or not isinstance(satisfied, bool):
            n_unplotted += 1
            continue
        plotted.append(
            {
                "accuracy": accuracy,
                "violation": violation,
                "lambda": _field(r, "lambda"),
                "satisfied": satisfied,
            }
        )

    # A frontier point needs no verdict (the dashed line is not a status
    # colour), but it still needs both coordinates.
    pareto_plotted: List[Dict[str, float]] = []
    n_pareto_unplotted = 0
    for p in pareto_frontier:
        accuracy = _coord(_field(p, "accuracy"))
        violation = _coord(_field(p, "violation"))
        if accuracy is None or violation is None:
            n_pareto_unplotted += 1
            continue
        pareto_plotted.append({"accuracy": accuracy, "violation": violation})

    best_fair_raw = _optimum(tradeoff_data.get("best_fair"))
    best_accurate_raw = _optimum(tradeoff_data.get("best_accurate"))
    n_optima_supplied = sum(
        1 for key in ("best_fair", "best_accurate") if tradeoff_data.get(key) is not None
    )
    n_optima_unmarked = n_optima_supplied - sum(
        1 for opt in (best_fair_raw, best_accurate_raw) if opt is not None
    )

    # Compute tight axis ranges. Only from coordinates that were reported: a
    # defaulted 0 in here silently rescales the axis under every real point.
    # The population is unchanged (the plotted configurations and the marked
    # optima, never the frontier), so a fully reported sweep keeps the axes it
    # has always had.
    all_x = [r["accuracy"] for r in plotted]
    all_y = [r["violation"] for r in plotted]
    if best_fair_raw:
        all_x.append(best_fair_raw["accuracy"])
        all_y.append(best_fair_raw["violation"])
    if best_accurate_raw:
        all_x.append(best_accurate_raw["accuracy"])
        all_y.append(best_accurate_raw["violation"])

    x_min, x_range, y_max = _axis_ranges(all_x, all_y)
    _xr = x_range if x_range else 1.0
    _ym = y_max if y_max else 1.0

    # Normalise coordinates to 0-1 within the truncated range
    def _nx(v: float) -> float:
        return max(0.0, min(1.0, (v - x_min) / _xr))

    def _ny(v: float) -> float:
        return max(0.0, min(1.0, v / _ym))

    # The headline rule, part (c): the withheld count goes on the CANVAS, in the
    # headline band, not at the foot and not only in the accessible <desc>. This
    # template's subtitle is fixed prose, so the title carries it, exactly as
    # drift_report_to_svg names its partial scales in the one header line it
    # controls. A fully reported sweep keeps the title it has always had.
    n_withheld = n_unplotted + n_pareto_unplotted + n_optima_unmarked
    n_supplied = len(all_results) + len(pareto_frontier) + n_optima_supplied
    title = "Accuracy-Fairness Trade-off Analysis"
    if n_withheld:
        title = f"Accuracy-Fairness Trade-off: {n_withheld} of {n_supplied} points not plotted"

    data = {
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "x_min": x_min,
        "x_range": x_range,
        "y_max": y_max,
        "points": [
            {
                "x": _nx(r["accuracy"]),
                "y": _ny(r["violation"]),
                "lambda": r["lambda"],
                "satisfied": r["satisfied"],
            }
            for r in plotted
        ],
        "pareto_points": [
            {
                "x": _nx(p["accuracy"]),
                "y": _ny(p["violation"]),
            }
            for p in pareto_plotted
        ],
        "best_fair": {
            "accuracy": _nx(best_fair_raw["accuracy"]),
            "violation": _ny(best_fair_raw["violation"]),
        }
        if best_fair_raw
        else None,
        "best_accurate": {
            "accuracy": _nx(best_accurate_raw["accuracy"]),
            "violation": _ny(best_accurate_raw["violation"]),
        }
        if best_accurate_raw
        else None,
        # Counted for the record, and read by the tests that hold this rule.
        "n_withheld_points": n_withheld,
        "n_supplied_points": n_supplied,
    }

    # No point, no frontier and no marked optimum means the trade-off was never
    # searched: nothing was trained at any constraint strength. The axes still
    # drew 0.00 to 1.00 on both sides, which reads as a completed sweep whose
    # points happen to be off-frame. See _mark_not_assessable.
    if not data["points"] and not data["pareto_points"] and not best_fair_raw:
        if n_supplied:
            # The rows exist and not one of them can be drawn. "No configuration
            # was supplied" would be false here, and an empty plot under the
            # ordinary subtitle would say the sweep ran and found nothing.
            _mark_not_assessable(
                data,
                f"None of the {n_supplied} supplied trade-off row(s) reported a complete "
                "accuracy-violation pair with a constraint verdict, so no configuration "
                "could be placed on either axis and no frontier was found.",
                "No configuration could be plotted",
            )
        else:
            _mark_not_assessable(
                data,
                "No trade-off configuration was supplied, so no constraint strength was "
                "trained, no accuracy-fairness pair was measured and no frontier was found.",
                "No configuration was evaluated",
            )

    try:
        data["explanation"] = explanation
        svg = render_svg("tradeoff_analysis", data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        # NEVER return "" here. A zero-length string is the worst of the three
        # states: a caller that writes the return value gets a 0-byte .svg that
        # opens as a broken image, and a caller that passed save_path keeps
        # whatever the PREVIOUS render left on disk, with success reported
        # either way. Only a UserWarning said otherwise, and warnings are
        # filtered by default in exactly the batch pipelines that call this.
        # Same failure card and same writer as method_comparison_to_svg and
        # adapters_post_processing, so the paths cannot drift.
        warnings.warn(f"SVG rendering failed: {e}")
        svg = _could_not_render_svg(
            "Accuracy-Fairness Trade-off Analysis",
            f"The chart could not be drawn ({type(e).__name__} while rendering the template).",
            str(data.get("timestamp", "")),
        )
        _write_svg(save_path, svg)
        return svg


def training_analysis_report_to_svg(
    report: "FairnessTrainingReport",
    *,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a comprehensive full-page analysis report for FairnessTrainingAnalyzer.

    This produces a polished 900×940 SVG dashboard that visualises every
    section of a ``FairnessTrainingReport``:

    * **Header**: title, timestamp, task type, constraint type
    * **Summary cards**: data summary, baseline performance, recommendation
    * **Group fairness analysis**: per-group positive-rate bars
    * **Method comparison chart**: accuracy / violation bars with status
    * **Trade-off mini scatter**: Pareto frontier overlay
    * **Critical issues**: severity-tagged issue list
    * **Action items**: numbered checklist

    Args:
        report: A ``FairnessTrainingReport`` produced by
            ``FairnessTrainingAnalyzer.full_analysis()``.
        save_path: Optional filesystem path; when provided the SVG
            string is also written to this file.

    Returns:
        The rendered SVG markup string.

    Notes
    -----
    CRITICAL, three states, never two. Until 2026-08-27 an empty report rendered
    "Accuracy 0.0000 | Violation 0.0000 | Constraint VIOLATED | Base rate
    disparity: 0.000" at accessible severity HIGH. Every one of those numbers
    came from a ``.get`` default, and VIOLATED is what the False default of
    ``constraint_satisfied`` draws. A constraint nobody evaluated must never be
    drawn as a constraint that failed: see ``_baseline_state``.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    # Unpack data_info
    data_info = report.data_info or {}

    # Unpack baseline_metrics
    baseline = report.baseline_metrics or {}
    baseline_state = _baseline_state(baseline)

    # Unpack recommendation
    recommendation = _recommendation_state(report.recommendation)

    # Unpack fairness_analysis → group_stats
    fa = report.fairness_analysis or {}
    group_statistics = fa.get("group_statistics", {})
    #
    # CRITICAL, per ROW, and the FABRICATED-BREACH direction. Until 2026-08-28
    # all three cells were read through a `.get(key, 0)`, so a group entry
    # carrying only its name rendered a complete measured row: "ZZ_BARE | 0 |
    # 0.0% | 0.0%" with a drawn rate bar. A POSITIVE RATE of 0.0% is the worst
    # finding this panel can show, a group that never received a single positive
    # outcome, and it was invented from an absent key; a SIZE of 0 says the same
    # group has no members, which contradicts its own presence in the table.
    # Executed on this repo against a report whose other two groups reported
    # 72% and 58%, the bare row drew a 3px bar at 0.0% and read as the most
    # disadvantaged group on the page.
    #
    # A default is not a measurement. Each cell is None unless the analyser
    # reported it, and the template says so in words per cell, because a group
    # may report a size with no rate beside it. See _coord and _count.
    group_stats: List[Dict[str, Any]] = []
    for grp_name, stats in group_statistics.items():
        size = _count(stats.get("size"))
        proportion = _coord(stats.get("proportion"))
        positive_rate = _coord(stats.get("positive_rate"))
        group_stats.append(
            {
                "name": str(grp_name),
                "size": size,
                "proportion": proportion,
                "positive_rate": positive_rate,
                # `rate_measured` gates the NUMBER and the bar together: the bar
                # width is `positive_rate * 100` in the template, so a None here
                # raised TypeError inside the render rather than drawing wrong.
                "rate_measured": positive_rate is not None,
            }
        )
    n_groups_unreported = sum(
        1
        for g in group_stats
        if g["size"] is None or g["proportion"] is None or not g["rate_measured"]
    )

    # Unpack method_comparisons.
    #
    # CRITICAL, three states, never two, and this is the FABRICATED-FAILURE
    # direction. This row was built by hand here rather than through
    # ``_method_state``, so the hardening that closed the same defect on
    # ``training_report`` and ``method_comparison`` two lines away never reached
    # this page: ``constraint_satisfied`` arrived raw, and every branch of the
    # methods table in ``training_analysis_report.svg`` is written
    # ``{% if m.satisfied %} ... {% else %}`` (rows 196, 203 and 205). None
    # takes the else arm, so a method whose constraint was NEVER EVALUATED was
    # drawn with the red row stripe, the red violation number and a red FAIL
    # chip. Executed on this repo, ``MethodComparison("adversarial", 0.77, 0.05,
    # None)`` produced one PASS and one FAIL, and no reader can tell that
    # invented failure from a measured one: it sends someone after a violation
    # that does not exist.
    #
    # RESIDUE CLOSED (2026-08-28). The row was withheld from the table while
    # this template had no third branch to draw it honestly, which meant a
    # method that WAS trained and simply never graded vanished from the page
    # altogether. Silence is its own fabrication: a table listing two of three
    # trained methods reads as the whole run. ``training_analysis_report.svg``
    # now carries the same ``m.graded`` branches ``training_report.svg`` has
    # (row stripe, violation colour and a slate NOT CHECKED chip), so every
    # supplied method is drawn, and an ungraded one is drawn as ungraded.
    #
    # The count is still named in the subtitle, in the headline band, because a
    # reader who takes the header and stops reading must not be left with a
    # methods table that looks fully evaluated.
    method_states = [_method_state(m) for m in (report.method_comparisons or [])]
    methods: List[Dict[str, Any]] = method_states
    n_methods_ungraded = sum(1 for m in method_states if not m["graded"])
    # Inside `n_methods_ungraded`, not beside it: a vacuous verdict is ungraded.
    # Counted apart only so the subtitle can say WHICH kind of ungraded it is.
    n_methods_vacuous = sum(1 for m in method_states if m["vacuous"])

    # Unpack tradeoff_analysis (with axis truncation).
    #
    # The same rule as tradeoff_analysis_to_svg, on the same data, because this
    # is the same scatter drawn small: `.get("accuracy", 0)` PLACED a
    # configuration that reported no coordinate at the left edge of the axis and
    # `.get("satisfied", False)` painted it in the red "constraint violated"
    # colour, so an unfinished row appeared as a measured breach at a measured
    # position. Both defaults also moved the axis under every real point, since
    # _axis_ranges reads this same list. See _coord and _field.
    ta = report.tradeoff_analysis or {}
    all_results = ta.get("all_results") or []
    pareto_frontier = ta.get("pareto_frontier") or []

    mini_plotted: List[Dict[str, Any]] = []
    n_mini_unplotted = 0
    for r in all_results:
        _acc = _coord(_field(r, "accuracy"))
        _viol = _coord(_field(r, "violation"))
        _sat = _field(r, "satisfied")
        if _acc is None or _viol is None or not isinstance(_sat, bool):
            n_mini_unplotted += 1
            continue
        mini_plotted.append({"accuracy": _acc, "violation": _viol, "satisfied": _sat})

    mini_pareto: List[Dict[str, float]] = []
    n_mini_pareto_unplotted = 0
    for p in pareto_frontier:
        _acc = _coord(_field(p, "accuracy"))
        _viol = _coord(_field(p, "violation"))
        if _acc is None or _viol is None:
            n_mini_pareto_unplotted += 1
            continue
        mini_pareto.append({"accuracy": _acc, "violation": _viol})

    # Collect raw values for axis range computation
    raw_x = [r["accuracy"] for r in mini_plotted]
    raw_y = [r["violation"] for r in mini_plotted]
    best_fair_raw = _optimum(ta.get("best_fair"))
    best_accurate_raw = _optimum(ta.get("best_accurate"))
    n_mini_optima_supplied = sum(
        1 for key in ("best_fair", "best_accurate") if ta.get(key) is not None
    )
    n_mini_optima_unmarked = n_mini_optima_supplied - sum(
        1 for opt in (best_fair_raw, best_accurate_raw) if opt is not None
    )
    if best_fair_raw:
        raw_x.append(best_fair_raw["accuracy"])
        raw_y.append(best_fair_raw["violation"])
    if best_accurate_raw:
        raw_x.append(best_accurate_raw["accuracy"])
        raw_y.append(best_accurate_raw["violation"])

    t_x_min, t_x_range, t_y_max = _axis_ranges(raw_x, raw_y)
    _txr = t_x_range if t_x_range else 1.0
    _tym = t_y_max if t_y_max else 1.0

    def _tnx(v: float) -> float:
        return max(0.0, min(1.0, (v - t_x_min) / _txr))

    def _tny(v: float) -> float:
        return max(0.0, min(1.0, v / _tym))

    tradeoff_points = [
        {
            "x": _tnx(r["accuracy"]),
            "y": _tny(r["violation"]),
            "satisfied": r["satisfied"],
        }
        for r in mini_plotted
    ]
    pareto_points = [{"x": _tnx(p["accuracy"]), "y": _tny(p["violation"])} for p in mini_pareto]

    if best_fair_raw:
        best_fair = {
            "x": _tnx(best_fair_raw["accuracy"]),
            "y": _tny(best_fair_raw["violation"]),
        }
    else:
        best_fair = None
    if best_accurate_raw:
        best_accurate = {
            "x": _tnx(best_accurate_raw["accuracy"]),
            "y": _tny(best_accurate_raw["violation"]),
        }
    else:
        best_accurate = None

    # Unpack critical_issues
    issues: List[Dict[str, Any]] = []
    if report.critical_issues:
        for iss in report.critical_issues[:5]:
            if isinstance(iss, dict):
                issues.append(iss)
            else:
                # No severity was reported with the string, so it is labelled
                # "unrated" rather than graded. W-19, closed 2026-09-07.
                # See _UNGRADED_ISSUE_SEVERITY.
                issues.append(
                    {
                        "type": "issue",
                        "description": str(iss),
                        "severity": _UNGRADED_ISSUE_SEVERITY,
                    }
                )

    # Unpack action_items
    actions: List[str] = []
    if report.action_items:
        actions = [str(a) for a in report.action_items[:5]]

    # The headline rule, part (c): the ungraded count goes on the CANVAS, in the
    # HEADLINE BAND, not at the foot. `constraint_type` is the one field of the
    # subtitle at y=122 that appears nowhere else on this page (task_type has a
    # stat card, attribute_name is repeated in the baseline panel), so it
    # carries the notice, exactly as drift_report_to_svg names its partial
    # scales in the one header line it controls. A run in which every method
    # reported its constraint keeps the subtitle it has always had.
    constraint_label = str(fa.get("constraint_type", "N/A"))
    n_points_withheld = n_mini_unplotted + n_mini_pareto_unplotted + n_mini_optima_unmarked
    if n_methods_ungraded or n_points_withheld or n_groups_unreported:
        # Kept short on purpose: this subtitle is ONE unwrapped <text> at
        # font-size 10 from x=36, and a longer sentence runs off the 680-wide
        # canvas, taking "constraint - gender" with it. Measured by rasterising,
        # not by eye on the markup.
        _notes = []
        if n_methods_ungraded:
            # A vacuous verdict is named as vacuous rather than folded into
            # "ungraded": a reader who sees "ungraded" looks for a method that
            # did not finish, and this one finished and measured nothing.
            if n_methods_vacuous == n_methods_ungraded:
                _notes.append(f"{n_methods_vacuous} of {len(method_states)} method(s) vacuous")
            elif n_methods_vacuous:
                _notes.append(
                    f"{n_methods_ungraded} of {len(method_states)} method(s) ungraded "
                    f"({n_methods_vacuous} vacuous)"
                )
            else:
                _notes.append(f"{n_methods_ungraded} of {len(method_states)} method(s) ungraded")
        if n_groups_unreported:
            # The GROUPS stat card above prints a bare count, and a count reads
            # as that many measured groups. A group whose size, share or
            # positive rate was never reported is named here, in the headline
            # band, and not left for a reader to spot in the table.
            _notes.append(f"{n_groups_unreported} group(s) partly unreported")
        if n_points_withheld:
            _notes.append(f"{n_points_withheld} trade-off point(s) not plotted")
        constraint_label = f"{constraint_label} ({', '.join(_notes)})"

    # Build template data dict
    data = {
        "title": "Fairness Training: Comprehensive Analysis Report",
        "timestamp": _fmt_ts(report.timestamp),
        "task_type": report.task_type or "N/A",
        "constraint_type": constraint_label,
        # Data info
        "n_samples": data_info.get("n_samples", "N/A"),
        "n_groups": data_info.get("n_groups", "N/A"),
        "n_features": data_info.get("n_features", "N/A"),
        "attribute_name": data_info.get("attribute_name", "sensitive_attr"),
        # Baseline: numbers and verdict, each withheld unless measured
        **baseline_state,
        # Recommendation
        "recommendation": recommendation,
        # Group fairness
        "group_stats": group_stats,
        # None, not 0: a disparity of 0.000 is the canvas for perfect parity
        # between groups, which is a strong claim to make about a report that
        # carried no group statistics at all.
        # Through `_coord`, like the per-group positive rate twenty lines above,
        # because this row's sibling was swept and this one was not: the raw
        # value reached the subtitle and a NaN rendered "Base rate disparity:
        # nan" (executed 2026-09-30). The analyser reports None for a rate it
        # could not measure, but this page also renders hand-built and
        # third-party `fairness_analysis` dicts, and `|f3` prints a NaN straight
        # through as the string "nan".
        "base_rate_disparity": _coord(fa.get("base_rate_disparity")),
        "base_rate_measured": _coord(fa.get("base_rate_disparity")) is not None,
        # The analyser's own WHY, which was produced and consumed nowhere.
        "base_rate_reason": str(fa.get("base_rate_not_measured_reason") or "").strip(),
        # Methods
        "methods": methods,
        # Trade-off (normalised, with axis metadata)
        "tradeoff_points": tradeoff_points,
        "pareto_points": pareto_points,
        "best_fair": best_fair,
        "best_accurate": best_accurate,
        "x_min": t_x_min,
        "x_range": t_x_range,
        "y_max": t_y_max,
        # Issues / Actions
        "issues": issues,
        "actions": actions,
    }

    # The constraint verdict is this report's spine: it is the red VIOLATED chip
    # on the canvas AND the "(constraint violated)" clause of the accessible
    # description, both taken from the same field. Without it there is no
    # verdict to draw, so the whole page becomes could-not-check rather than
    # printing a baseline of zeros and a 0.000 base-rate disparity under a
    # fabricated failure. See _baseline_state for why this is chart-level.
    if not baseline_state["constraint_evaluated"]:
        nothing_at_all = not any(
            [
                baseline_state["baseline_measured"],
                data["base_rate_measured"],
                group_stats,
                # Every SUPPLIED method, graded or not: "the report carries no
                # trained method" would be false about a run that trained two of
                # them and graded neither.
                method_states,
                all_results,
                tradeoff_points,
                pareto_points,
                best_fair,
                best_accurate,
                issues,
                actions,
                recommendation["given"],
            ]
        )
        _mark_not_assessable(
            data,
            "The report carries no baseline metric, no group statistic, no trained "
            "method and no trade-off configuration, so no accuracy was recorded, no "
            "fairness violation was measured and no constraint was evaluated."
            if nothing_at_all
            else "The baseline carries no constraint result, so whether the fairness "
            "constraint holds was never evaluated on this run.",
            "No training run was analysed"
            if nothing_at_all
            else "The fairness constraint was not evaluated",
        )

    try:
        data["explanation"] = explanation
        svg = render_svg("training_analysis_report", data)

        if save_path:
            with open(save_path, "w") as f:
                f.write(svg)

        return svg
    except Exception as e:
        warnings.warn(f"SVG rendering failed: {e}")
        return _generate_fallback_svg(data)


def _generate_fallback_svg(data: Dict[str, Any]) -> str:
    """Generate a simple fallback SVG when templates are not available."""
    methods_text = ""
    y_offset = 200
    for method in data.get("methods", []):
        # CRITICAL, the same three states the main template carries, and the
        # same crash-safety as the baseline lines above. This row was written
        # `{method["accuracy"]:.3f}` with a `"✓" if satisfied else "✗"` status,
        # so on a partial row it did BOTH wrong things at once: `_method_state`
        # withholds an unreported accuracy as None, which `:.3f` cannot format,
        # so this card raised TypeError inside the very branch whose job is to
        # survive a template failure; and a method whose constraint was never
        # evaluated was stamped ✗, the fabricated failure this wave exists to
        # remove, in the one place a reader looks when the chart did not draw.
        graded = method.get("graded", method.get("satisfied") is not None)
        status = ("✓" if method.get("satisfied") else "✗") if graded else "not evaluated"
        methods_text += f"""
        <text x="50" y="{y_offset}" font-size="14">
            {method["name"]}: Acc={_fmt3(method.get("accuracy"))}, Viol={_fmt3(method.get("violation"))} [{status}]
        </text>
        """
        y_offset += 25

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
    <rect width="800" height="600" fill="#f8f9fa"/>

    <!-- Title -->
    <text x="400" y="40" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">
        {data.get("title", "Fairness Training Report")}
    </text>

    <!-- Data Info -->
    <text x="50" y="80" font-size="14" fill="#666">
        Task: {data.get("task_type", "N/A")} | Samples: {data.get("n_samples", "N/A")} | Groups: {data.get("n_groups", "N/A")}
    </text>

    <!-- Baseline -->
    <text x="50" y="120" font-size="16" font-weight="bold" fill="#333">Baseline Performance</text>
    <text x="50" y="145" font-size="14">
        Accuracy: {_fmt4(data.get("baseline_accuracy"))} | Violation: {_fmt4(data.get("baseline_violation"))}
    </text>

    <!-- Methods -->
    <text x="50" y="180" font-size="16" font-weight="bold" fill="#333">Method Comparison</text>
    {methods_text}

    <!-- Recommendation -->
    <text x="50" y="{y_offset + 30}" font-size="16" font-weight="bold" fill="#333">Recommendation</text>
    <text x="50" y="{y_offset + 55}" font-size="14">
        {data.get("recommendation", {}).get("method", "N/A")}:
        {data.get("recommendation", {}).get("rationale", "N/A")[:60]}...
    </text>

    <!-- Footer -->
    <text x="400" y="580" font-size="10" text-anchor="middle" fill="#999">
        Generated by vfairness | {data.get("timestamp", "")}
    </text>
</svg>
"""
