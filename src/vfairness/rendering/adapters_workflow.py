"""
Adapters for workflow integration SVG templates.

Transforms vfairness workflow integration objects into flat dictionaries
suitable for SVG template rendering.

Templates:
    workflow_overview: Development workflow integration pipeline overview
    hierarchical_gate: Three-level hierarchical fairness gate visualization
    report_card: PR-ready fairness report card
"""

import textwrap
from datetime import datetime
from typing import Dict, List, Optional

from .engine import render_svg


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _fmt_ts(ts) -> str:
    """Normalise any timestamp to '%Y-%m-%d %H:%M'."""
    if ts is None:
        return _now()
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return _now()


# 1.  Workflow Overview  →  workflow_overview.svg


def workflow_overview_to_svg(
    *,
    title: str = "Workflow Integration Overview",
    integrations: Optional[List[Dict[str, str]]] = None,
    vcs_tools: Optional[List[Dict[str, str]]] = None,
    test_types: Optional[List[Dict[str, str]]] = None,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a workflow integration overview as an SVG dashboard.

    Parameters
    ----------
    title : str
        Dashboard title.
    integrations : list of dict
        Experiment tracking integrations wired into the caller's own pipeline.
        Each dict has keys ``name``, ``description``, optionally ``status``
        ('active'|'optional') and ``color``.
    vcs_tools : list of dict
        Version control tools. Each dict has ``name``, ``description``.
    test_types : list of dict
        Testing capabilities. Each dict has ``name``, ``description``.
    example : bool
        Render the built-in reference set of integrations, watermarked EXAMPLE
        on the canvas.  Off by default and never reached implicitly: see the
        note below.
    explanation : str
        Optional educational explanation text.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old defaults. Until 2026-08-27 calling this with
    no arguments substituted three module-level fixture lists and returned a
    complete pipeline map, three ACTIVE badges and all, whose accessible ``<desc>``
    read "Workflow overview: 3 tracking integration(s), 2 VCS tool(s), 3 test
    type(s)". Those are vfairness's own reference integrations, not a reading of
    the caller's pipeline, and the render was indistinguishable from one built
    from a caller's real ``integrations=`` list. An SVG is an export format: it
    leaves the building and outlives the code that made it, and "MLflow: ACTIVE"
    on a page about someone's ML lifecycle is a claim about their setup. Missing
    input is COULD NOT CHECK, stated on the canvas. The reference set survives
    only behind the explicit ``example=True``, watermarked.
    """
    supplied = integrations is not None or vcs_tools is not None or test_types is not None
    if not supplied and not example:
        return _workflow_not_assessed(title=title, explanation=explanation, save_path=save_path)

    # A caller who named their OWN integrations must not have the other two
    # sections silently filled from the fixture: that is the same substitution
    # at a finer grain, and it is harder to spot because two thirds of the page
    # would then be real. Outside example mode an unnamed section is empty.
    if supplied:
        integrations = integrations if integrations is not None else []
        vcs_tools = vcs_tools if vcs_tools is not None else []
        test_types = test_types if test_types is not None else []

    if integrations is None:
        integrations = [
            {
                "name": "MLflow",
                "description": "log_fairness_to_mlflow(): metrics, CIs, artifacts",
                "status": "active",
                "color": "#3b82f6",
            },
            {
                "name": "W&B",
                "description": "log_fairness_to_wandb(): batch logging, report artifacts",
                "status": "active",
                "color": "#f59e0b",
            },
            {
                "name": "Auto-logging",
                "description": "@auto_log_fairness: decorator for transparent tracking",
                "status": "active",
                "color": "#8b5cf6",
            },
        ]

    if vcs_tools is None:
        vcs_tools = [
            {
                "name": "Pre-commit hooks",
                "description": "check_fairness_config() + check_model_card()",
            },
            {"name": "PR template", "description": "Fairness checklist for code reviews"},
        ]

    if test_types is None:
        test_types = [
            {
                "name": "pytest plugin",
                "description": "@pytest.mark.fairness + fairness_gate fixture",
            },
            {
                "name": "assert_fairness()",
                "description": "Threshold-based fairness assertions in CI",
            },
            {"name": "FairnessTestSuite", "description": "Parametrized test suites for models"},
        ]

    data = {
        "title": title,
        "timestamp": _now(),
        "integrations": integrations,
        "vcs_tools": vcs_tools,
        "test_types": test_types,
        "explanation": explanation,
    }
    if not supplied:
        # DO NOT REMOVE. Every row below comes from the built-in reference set,
        # not from the caller's pipeline. This flag is what puts the EXAMPLE band
        # and watermark on the canvas and the EXAMPLE ONLY marker at the front of
        # the accessible description, and it is the only thing that separates a
        # sample from a map of somebody's real workflow.
        data["is_example"] = True

    svg = render_svg("workflow_overview", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _workflow_not_assessed(*, title, explanation, save_path):
    """The third state: an overview for a call that described no pipeline.

    Every section is withheld rather than defaulted: no integration row, no VCS
    row, no test row, and therefore no count. ``not_assessable`` carries the same
    fact into the accessible ``<desc>`` through
    ``rendering.explain._not_assessable``, so the canvas and the screen reader
    tell one story.
    """
    reason = (
        "No integration, version-control tool or test type was supplied to "
        "workflow_overview_to_svg, so no part of a pipeline was described here."
    )
    data = {
        "title": title,
        "timestamp": _now(),
        "not_assessable": True,
        "not_assessable_reason": reason,
        "na_headline": "No pipeline was described",
        "integrations": [],
        "vcs_tools": [],
        "test_types": [],
        "explanation": explanation,
    }
    svg = render_svg("workflow_overview", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 2.  Hierarchical Gate  →  hierarchical_gate.svg


def hierarchical_gate_to_svg(
    decision=None,
    *,
    title: str = "Hierarchical Fairness Gate",
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a hierarchical (intersectional) gate decision as an SVG dashboard.

    Parameters
    ----------
    decision : IntersectionalGateDecision, optional
        Result from ``ModelFairnessGate.evaluate_hierarchical()``.
        If None, renders a could-not-check canvas that states plainly that no
        level was evaluated (unless *example* is set).
    title : str
        Dashboard title.
    example : bool
        Render the built-in demonstration decision, watermarked EXAMPLE on the
        canvas.  Off by default and never reached implicitly: see the note
        below.
    explanation : str
        Optional educational explanation text.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old default. Until 2026-08-27 a missing
    *decision* fell straight through to ``_hierarchical_from_demo`` and this
    function returned a gate marked APPROVED, with PASS on all three levels and
    2/2, 4/4 and 2/2 metric counts, for a model that had never been gated. The
    caller could not tell that apart from a real evaluation, and an exported SVG
    outlives the code that made it. Missing input is COULD NOT CHECK, stated on
    the canvas, never an approval.
    """
    if decision is not None:
        return _hierarchical_from_decision(
            decision, title=title, explanation=explanation, save_path=save_path
        )
    if example:
        return _hierarchical_from_demo(title=title, explanation=explanation, save_path=save_path)
    return _hierarchical_not_assessed(title=title, explanation=explanation, save_path=save_path)


def _hierarchical_not_assessed(*, title, explanation, save_path):
    """The third state: a gate canvas for a run that evaluated no level.

    ``approved`` is omitted rather than set to False. The template's verdict
    branch keys on it being absent or None, so the badge reads NOT CHECKED
    instead of BLOCKED: a gate that was never run has not rejected anything
    either. ``not_assessable`` carries the same fact into the accessible
    ``<desc>`` through ``rendering.explain._not_assessable``.
    """
    reason = (
        "No gate decision was supplied to hierarchical_gate_to_svg, so no level "
        "was evaluated: not overall, not per attribute, not intersectional."
    )
    data = {
        "title": title,
        "timestamp": _now(),
        "not_assessable": True,
        "not_assessable_reason": reason,
        "levels": [],
        "warnings": [],
        # No decision reached this canvas, so there is no small-sample answer to
        # disclose and no panel: the not_assessable band above already states that
        # nothing was evaluated, and a "NOT RUN" panel here would imply a gate ran
        # and skipped one check. Set explicitly rather than left undefined.
        "min_group_size": None,
        "small_sample_state": "",
        "small_sample_headline": "",
        "small_sample_notes": [],
        "explanation": explanation,
    }
    svg = render_svg("hierarchical_gate", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _small_sample_disclosure(ran, unsized, vacuous_minimum, warnings_data, min_group_size):
    """The small-sample panel's state, heading and notes. Three states, never two.

    F13, 2026-09-30. ``RAN`` / ``NOT RUN`` / ``NOT RECORDED`` map one to one onto
    ``IntersectionalGateDecision.small_sample_check_ran``'s True / False / None,
    which is the same vocabulary the markdown report and the report card already
    print, so the canvas cannot disagree with them. A could-not-check is never
    collapsed into either of the others, and a ``RAN`` that does not cover
    everything says what it leaves out.

    Returns ``(state, headline, notes)``. ``state`` is falsy only when there is
    genuinely nothing to disclose, which cannot happen for a real decision: the
    three states are exhaustive over ``ran``.
    """
    n = len(warnings_data)
    floor = f"the minimum of {min_group_size}" if min_group_size is not None else "the minimum"
    notes = []
    if n:
        state = "BELOW MINIMUM"
        headline = (
            f"{n} group(s) or rate denominator(s) below {floor}"
            if min_group_size is not None
            else f"{n} group(s) or rate denominator(s) below the minimum (not recorded here)"
        )
        notes.append(
            "A rate denominator is a LABELLED SUBSET of its group, so a group above the "
            "minimum can still hold a rate measured over a handful of rows. Each row below "
            "carries the count that rate was divided by and the metric it belongs to."
        )
    elif ran is True:
        state = "RAN"
        headline = f"Sizes were compared against {floor}, and nothing fell below it"
        notes.append(
            "This is a measurement, not an absence of one: the check ran and found no "
            "group and no rate denominator under the minimum."
        )
    elif ran is False:
        state = "NOT RUN"
        headline = "No size was compared to a minimum, so nothing here is a clean reading"
        if vacuous_minimum is not None:
            notes.append(
                f"The configured minimum is {vacuous_minimum}, which no group count can "
                f"fall below, so the check could not have failed."
            )
        else:
            notes.append(
                "An empty list of warnings here means NOT CHECKED, not 'no group was too "
                "small'. Nothing on this canvas establishes that any group was large "
                "enough to certify a comparison across it."
            )
    else:
        state = "NOT RECORDED"
        headline = "This decision does not say whether any size was checked"
        notes.append(
            "Could not check: the decision carries no record either way, so the absence of "
            "a warning below establishes nothing. It is not the same as a check that ran "
            "and came back clean."
        )
    if unsized:
        notes.append("Never sized: " + "; ".join(str(u) for u in unsized))
    # One note per rendered line, wrapped here rather than in the template so the
    # panel height computed at the top of the template matches what is drawn.
    lines = []
    for note in notes:
        lines.extend(textwrap.wrap(note, width=104) or [""])
    return state, headline, lines


def _hierarchical_from_decision(decision, *, title, explanation, save_path):
    """Build template data from a real IntersectionalGateDecision."""
    levels = []
    overall_pass = 0
    overall_total = 0
    single_pass = 0
    single_total = 0
    intersect_pass = 0
    intersect_total = 0

    for level_name, gate_dec in decision.level_results.items():
        metrics_data = []
        for ev in gate_dec.metric_evaluations:
            metrics_data.append(
                {
                    "name": ev.metric_name,
                    "value": ev.value,
                    "threshold": ev.threshold,
                    "passed": ev.passed,
                }
            )

        if level_name == "overall":
            level_type = "overall"
            display_name = "Overall (all groups)"
            overall_total = len(metrics_data)
            overall_pass = sum(1 for m in metrics_data if m["passed"])
        elif level_name.startswith("attr:"):
            level_type = "single attribute"
            display_name = level_name.replace("attr:", "Attribute: ")
            single_total += len(metrics_data)
            single_pass += sum(1 for m in metrics_data if m["passed"])
        else:
            level_type = "intersectional"
            display_name = level_name.replace("intersection:", "Intersection: ").replace(
                "_x_", " x "
            )
            intersect_total += len(metrics_data)
            intersect_pass += sum(1 for m in metrics_data if m["passed"])

        levels.append(
            {
                "name": display_name,
                "type": level_type,
                "approved": gate_dec.approved,
                "metrics": metrics_data,
            }
        )

    # A FLOOR COUNTING ROWS WHERE THE QUANTITY NEEDS LABELS, AT THE RENDER
    # BOUNDARY (F13, 2026-09-30). Two separate collapses lived on the nine lines
    # this replaces, and both made a canvas that says nothing about the
    # small-sample check indistinguishable from one that checked and found
    # nothing.
    #
    # ONE: the label-conditional denominators never arrived. ``evaluate`` records
    # a SmallSampleWarning for a rate leg whose own rows are below the floor (a
    # true-positive rate lives on the rows whose label is 1, so it can rest on one
    # person inside a group of a hundred) and it puts it on the LEVEL decision.
    # ``IntersectionalGateDecision.small_sample_warnings`` is filled from the
    # hierarchy's own ROW-COUNT scan only, and this loop read that list alone.
    # Measured 2026-09-30, two groups of 100 rows, group f holding ONE
    # positive-label row, min_group_size=30, equalized_odds_difference gated::
    #
    #     level 'overall'     leg warning ('f, the true-positive rate leg, rows
    #                         whose true label is 1', n=1, min=30) and a blocking
    #                         reason quoting it
    #     level 'attr:gender' the same
    #     decision.small_sample_warnings                          []
    #     'true-positive rate leg' anywhere on the canvas       False
    #     'SMALL-SAMPLE' panel on the canvas                    False
    #
    # so the gate BLOCKED on a rate certified over one row and the canvas drew no
    # small-sample panel at all.
    #
    # TWO: the three-state answer was dropped. ``small_sample_check_ran`` is
    # True / False / None and ``small_sample_unsized`` names what a True does not
    # cover; the other four graded surfaces carry both. Measured on one real
    # decision with the level results and the (empty) warning list held fixed and
    # only the field varied::
    #
    #     small_sample_check_ran=True  -> sha256 c47334584ab93abd  len 19073
    #     small_sample_check_ran=False -> sha256 c47334584ab93abd  len 19073
    #     small_sample_check_ran=None  -> sha256 c47334584ab93abd  len 19073
    #     small_sample_vacuous_minimum=1 -> identical to the True canvas
    #
    # Byte-identical: a hierarchy whose group-size check never ran, and one run
    # against a minimum no count can fall below, rendered as a clean gate.
    #
    # Read through getattr because the adapter is fed duck-typed decisions (see
    # tests/test_adapters_row_level_regression.py) and a stand-in that carries
    # neither field must come out as "nobody recorded either way", never as a
    # check that ran.
    warnings_data = []
    _seen_warnings = set()

    def _collect(source) -> None:
        for w in getattr(source, "small_sample_warnings", None) or []:
            metric = getattr(w, "metric_name", None) or "all"
            key = (str(w.group_name), int(w.sample_size), metric)
            if key in _seen_warnings:
                # The same leg is recorded on every level that evaluated the
                # metric, so it would otherwise be printed once per level.
                continue
            _seen_warnings.add(key)
            warnings_data.append(
                {
                    "group": w.group_name,
                    "size": w.sample_size,
                    "min_recommended": w.minimum_recommended,
                    # Printed on the row, because "f, the true-positive rate leg"
                    # is a statement about ONE metric and the panel heading
                    # cannot say which.
                    "metric": metric,
                }
            )

    _collect(decision)
    for _lvl in (getattr(decision, "level_results", None) or {}).values():
        _collect(_lvl)

    # NEVER 30 BECAUSE 30 IS THE LIBRARY DEFAULT. The old fallback asserted a
    # floor that nothing on this decision had applied: measured 2026-09-30 on a
    # decision with hierarchical_config=None carrying one warning whose own
    # minimum is 200, the panel heading read "Groups below minimum sample size
    # (30)" directly above the row "tiny: 3 samples (min 200)". None means the
    # heading says the minimum was not recorded; every row still carries its own.
    min_group_size = None
    _hconfig = getattr(decision, "hierarchical_config", None)
    if _hconfig is not None:
        min_group_size = getattr(_hconfig, "min_group_size", None)

    ss_ran = getattr(decision, "small_sample_check_ran", None)
    ss_unsized = list(getattr(decision, "small_sample_unsized", None) or [])
    ss_vacuous = getattr(decision, "small_sample_vacuous_minimum", None)
    ss_state, ss_headline, ss_notes = _small_sample_disclosure(
        ss_ran, ss_unsized, ss_vacuous, warnings_data, min_group_size
    )

    data = {
        "title": title,
        "timestamp": _now(),
        "approved": decision.approved,
        "levels": levels,
        "warnings": warnings_data,
        "min_group_size": min_group_size,
        # Three states on the small-sample panel, never two. See
        # _small_sample_disclosure.
        "small_sample_state": ss_state,
        "small_sample_headline": ss_headline,
        "small_sample_notes": ss_notes,
        "overall_status": "approved"
        if overall_pass == overall_total and overall_total > 0
        else "blocked",
        "n_overall_pass": overall_pass,
        "n_overall_total": overall_total,
        "single_status": "approved"
        if single_pass == single_total and single_total > 0
        else ("blocked" if single_total > 0 else None),
        "n_single_pass": single_pass,
        "n_single_total": single_total,
        "intersect_status": "approved"
        if intersect_pass == intersect_total and intersect_total > 0
        else ("blocked" if intersect_total > 0 else None),
        "n_intersect_pass": intersect_pass,
        "n_intersect_total": intersect_total,
        "explanation": explanation,
    }

    svg = render_svg("hierarchical_gate", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _hierarchical_from_demo(*, title, explanation, save_path):
    """Build template data with demo values. Reachable only via example=True."""
    data = {
        "title": title,
        "timestamp": _now(),
        # DO NOT REMOVE. Every value below is invented. This flag is what puts
        # the EXAMPLE band and watermark on the canvas, and it is the only thing
        # that stops this render being read as a gate result for a real model.
        "is_example": True,
        "approved": True,
        "levels": [
            {
                "name": "Overall (all groups)",
                "type": "overall",
                "approved": True,
                "metrics": [
                    {
                        "name": "demographic_parity_difference",
                        "value": 0.042,
                        "threshold": 0.100,
                        "passed": True,
                    },
                    {
                        "name": "equalized_odds_difference",
                        "value": 0.065,
                        "threshold": 0.100,
                        "passed": True,
                    },
                ],
            },
            {
                "name": "Attribute: gender",
                "type": "single attribute",
                "approved": True,
                "metrics": [
                    {
                        "name": "demographic_parity_difference",
                        "value": 0.038,
                        "threshold": 0.100,
                        "passed": True,
                    },
                    {
                        "name": "equalized_odds_difference",
                        "value": 0.071,
                        "threshold": 0.100,
                        "passed": True,
                    },
                ],
            },
            {
                "name": "Attribute: race",
                "type": "single attribute",
                "approved": True,
                "metrics": [
                    {
                        "name": "demographic_parity_difference",
                        "value": 0.055,
                        "threshold": 0.100,
                        "passed": True,
                    },
                    {
                        "name": "equalized_odds_difference",
                        "value": 0.082,
                        "threshold": 0.100,
                        "passed": True,
                    },
                ],
            },
            {
                "name": "Intersection: gender x race",
                "type": "intersectional",
                "approved": True,
                "metrics": [
                    {
                        "name": "demographic_parity_difference",
                        "value": 0.089,
                        "threshold": 0.120,
                        "passed": True,
                    },
                    {
                        "name": "equalized_odds_difference",
                        "value": 0.098,
                        "threshold": 0.120,
                        "passed": True,
                    },
                ],
            },
        ],
        "warnings": [
            {
                "group": "Female_Asian",
                "size": 22,
                "min_recommended": 30,
                "metric": "all",
            },
        ],
        "min_group_size": 30,
        # Invented like everything else here, and it has to be present or the
        # EXAMPLE canvas would be the one surface that still hides the state.
        "small_sample_state": "BELOW MINIMUM",
        "small_sample_headline": "1 group(s) or rate denominator(s) below the minimum of 30",
        "small_sample_notes": [
            "A rate denominator is a LABELLED SUBSET of its group, so a group above the",
            "minimum can still hold a rate measured over a handful of rows.",
        ],
        "overall_status": "approved",
        "n_overall_pass": 2,
        "n_overall_total": 2,
        "single_status": "approved",
        "n_single_pass": 4,
        "n_single_total": 4,
        "intersect_status": "approved",
        "n_intersect_pass": 2,
        "n_intersect_total": 2,
        "explanation": explanation,
    }

    svg = render_svg("hierarchical_gate", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 3.  Report Card  →  report_card.svg


def report_card_to_svg(
    report_card=None,
    *,
    decision=None,
    model_name: str = "model",
    title: str = "Fairness Report Card",
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a fairness report card as an SVG dashboard.

    Parameters
    ----------
    report_card : FairnessReportCard, optional
        A pre-built report card object.
    decision : GateDecision, optional
        Gate decision to convert (alternative to report_card).
    model_name : str
        Name of the model being evaluated.
    title : str
        Dashboard title.
    example : bool
        Render the built-in demonstration card, watermarked EXAMPLE on the
        canvas.  Off by default and never reached implicitly: see the note
        below.
    explanation : str
        Optional educational explanation text.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old default. Until 2026-08-27, calling this
    with neither a *report_card* nor a *decision* returned the demo card, and
    the demo card renders DEPLOYMENT APPROVED beside a 3/3 PASS RATE and the
    line "Model meets fairness requirements and is approved for deployment".
    That is a deployment approval fabricated out of no input at all, on a file
    format built to be exported, attached to a pull request and shown to a
    regulator. Missing input is COULD NOT CHECK, stated on the canvas.
    """
    if report_card is not None:
        return _report_card_from_object(
            report_card, title=title, explanation=explanation, save_path=save_path
        )
    elif decision is not None:
        return _report_card_from_decision(
            decision,
            model_name=model_name,
            title=title,
            explanation=explanation,
            save_path=save_path,
        )
    elif example:
        return _report_card_demo(
            model_name=model_name, title=title, explanation=explanation, save_path=save_path
        )
    else:
        return _report_card_not_assessed(
            model_name=model_name, title=title, explanation=explanation, save_path=save_path
        )


def _report_card_not_assessed(*, model_name, title, explanation, save_path):
    """The third state: a card for a run with no gate decision behind it.

    ``approved`` is omitted rather than set to False. The template's verdict
    branch keys on it being absent or None, so the badge reads NOT CHECKED and
    the pass rate reads "not measured" instead of a 0/0 that looks like a
    measurement. ``not_assessable`` carries the same fact into the accessible
    ``<desc>`` through ``rendering.explain._not_assessable``.
    """
    reason = (
        "No gate decision was supplied to report_card_to_svg, so no metric was "
        "evaluated and no threshold was applied."
    )
    data = {
        "title": title,
        "timestamp": _now(),
        "model_name": model_name,
        "not_assessable": True,
        "not_assessable_reason": reason,
        "metrics": [],
        "has_baseline": False,
        "blocking_reasons": [],
        "card_warnings": [],
        "explanation": explanation,
    }
    svg = render_svg("report_card", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _report_card_from_object(report_card, *, title, explanation, save_path):
    """Build from a FairnessReportCard object."""
    decision = report_card.decision
    model_name = report_card.model_name

    return _report_card_from_decision(
        decision,
        model_name=model_name,
        title=title,
        explanation=explanation,
        save_path=save_path,
    )


def _report_card_from_decision(decision, *, model_name, title, explanation, save_path):
    """Build from a GateDecision.

    CRITICAL, and this is the ONE boundary the card's three-state machinery
    depends on. G13, measured 2026-09-30 on
    ``ModelFairnessGate(thresholds={'demographic_parity_difference': 0.1})
    .evaluate(y, y, ['a'] * 200)`` -- one group, so no disparity is definable::

        producer:  value=nan  passed=False  _metric_row_state -> 'unmeasured'
        markdown:  | ... | not measured (NaN: insufficient evidence) | 0.1000 | Could not check |
        THIS CARD: RESULT cells ['FAIL'],  desc "0/1 metrics pass"

    ``report_card.svg`` badges NOT GRADED in slate, prints "not measured" in
    the value cell and keeps the row out of the PASS RATE denominator; and
    ``explain._fr_report_card`` writes the same third state into the
    accessible ``<desc>``. BOTH key on ``passed is none``, and this function
    copied ``ev.passed`` straight through -- and the gate sets ``passed``
    False for a could-not-check as well as for a measured breach, so that
    entire third state was UNREACHABLE from the only producer that feeds it.
    A red FAIL against a threshold of 0.1000 beside a value cell reading N/A,
    on the surface that gets pasted into a pull request.

    The row verdict is therefore taken from the gate's own
    ``_metric_row_state`` rather than re-derived here: it is the function the
    other five renderers of a MetricEvaluation were moved onto, so the
    producer and the renderer cannot disagree about which rows were graded.
    Its other states map onto what the template already draws (pass ->
    ev.passed, warn/fail -> the same ``is_blocking`` split this template makes,
    refused -> a blocking refusal whose reason is in ``blocking_reasons``).
    """
    # Imported in the function body: `rendering` is imported by parts of
    # `operations`, so a module-level import here would close a cycle.
    from ..operations.cicd.gate import _ROW_UNMEASURED, _metric_row_state

    metrics_data = []
    has_baseline = False
    for ev in decision.metric_evaluations:
        if ev.baseline_value is not None:
            has_baseline = True
        ungraded = _metric_row_state(ev) == _ROW_UNMEASURED
        metrics_data.append(
            {
                "name": ev.metric_name,
                "value": ev.value,
                "threshold": ev.threshold,
                "passed": None if ungraded else ev.passed,
                "is_blocking": ev.is_blocking,
                "baseline": ev.baseline_value,
            }
        )

    status_val = (
        decision.status.value if hasattr(decision.status, "value") else str(decision.status)
    )

    data = {
        "title": title,
        "timestamp": _now(),
        "model_name": model_name,
        "approved": decision.approved,
        "status": status_val,
        "metrics": metrics_data,
        "has_baseline": has_baseline,
        "blocking_reasons": decision.blocking_reasons,
        "card_warnings": decision.warnings,
        "explanation": explanation,
    }

    svg = render_svg("report_card", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


def _report_card_demo(*, model_name, title, explanation, save_path):
    """Render demo report card with sample data. Reachable only via example=True."""
    data = {
        "title": title,
        "timestamp": _now(),
        "model_name": model_name,
        # DO NOT REMOVE. Every value below is invented, including the approval.
        # This flag is what puts the EXAMPLE band and watermark on the canvas,
        # and it is the only thing that stops this render being read as a real
        # deployment approval.
        "is_example": True,
        "approved": True,
        "status": "approved",
        "metrics": [
            {
                "name": "demographic_parity_difference",
                "value": 0.042,
                "threshold": 0.100,
                "passed": True,
                "is_blocking": True,
                "baseline": 0.078,
            },
            {
                "name": "equalized_odds_difference",
                "value": 0.065,
                "threshold": 0.100,
                "passed": True,
                "is_blocking": True,
                "baseline": 0.091,
            },
            {
                "name": "equal_opportunity_difference",
                "value": 0.031,
                "threshold": 0.100,
                "passed": True,
                "is_blocking": False,
                "baseline": None,
            },
        ],
        "has_baseline": True,
        "blocking_reasons": [],
        "card_warnings": [],
        "n_groups": 4,
        "explanation": explanation,
    }

    svg = render_svg("report_card", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg
