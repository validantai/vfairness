"""
SVG Rendering Adapter for Data Validation Module.

Transforms DataValidationResult and ValidationIssue objects into flat
dictionaries suitable for the ``data_validation.svg`` template.
"""

import math
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

_PASS = "#059669"
_FAIL = "#dc2626"
_WARN = "#f59e0b"
_INFO = "#3b82f6"
# Slate, the could-not-check tone. Same value and same reason as the NaN branch
# of engine._risk_color: unknown is neither green nor a benign informational
# blue, because both of those read to a reviewer as "nothing to see here".
_UNKNOWN = "#64748b"
_UNKNOWN_BG = "#f1f5f9"

_SEVERITY_COLORS = {
    "CRITICAL": (_FAIL, "#fef2f2"),
    "ERROR": (_FAIL, "#fef2f2"),
    "WARNING": (_WARN, "#fffbeb"),
    "INFO": (_INFO, "#eff6ff"),
    "UNKNOWN": (_UNKNOWN, _UNKNOWN_BG),
}

_SEVERITY_ICONS = {
    "CRITICAL": "!!",
    "ERROR": "!",
    "WARNING": "~",
    "INFO": "i",
    "UNKNOWN": "?",
}

# The verdict banner has three states and never two. PASS and FAIL are assessed
# results; UNKNOWN means this adapter was not given enough to assess anything
# and must say so rather than defaulting to either side.
_VERDICT_STYLE = {
    "PASS": (_PASS, "#ecfdf5"),
    "FAIL": (_FAIL, "#fef2f2"),
    "UNKNOWN": (_UNKNOWN, _UNKNOWN_BG),
}

_MISSING = object()

#: The values ``_validation_coverage`` can return, matching the vocabulary
#: ``BiasAuditReport.execution_coverage`` already established for the bias audit
#: (see ``rendering.adapters._COVERAGE_STATES``): one word, one meaning, across
#: every page that has to say how much of an analysis actually executed.
_COVERAGE_STATES = ("complete", "partial", "none", "unrecorded")


def _validation_coverage(d: Dict[str, Any], result: Any) -> str:
    """How much of the validation actually executed: complete/partial/none/unrecorded.

    ROOT CAUSE, now closed on both sides; the history is kept because the defect
    is easy to reintroduce by "simplifying" the dataclass.
    ``DataValidationResult`` (``operations.cicd.validator``) used to carry
    ``passed``, ``issues``, ``metrics`` and ``summary``, and NOTHING that said
    which checks ran. ``DataBiasValidator.validate`` gates all five of its checks
    on config flags, so a config with every check switched off returned
    ``passed=True``, ``issues=[]`` and the summary "Validation passed - no bias
    issues detected". That result was indistinguishable, in the object itself,
    from a dataset that passed every check, and this page rendered it as a green
    PASS reading "The data is ready for fairness analysis" (reproduced
    2026-08-27). No amount of work in this adapter could separate the two,
    because the distinction was never recorded.

    The source-side half landed 2026-08-28: a ``checks_run`` field plus an
    ``execution_coverage()`` method on ``DataValidationResult``, exactly as
    ``BiasAuditReport.modules_run`` / ``execution_coverage()`` was added for the
    bias audit, recorded per check that ACTUALLY ran rather than per check that
    was enabled. This function is the reader for it and works against the object
    OR against its ``to_dict()``.

    ``"unrecorded"`` is still a first-class answer and still changes nothing: it
    neither grants a verdict nor refuses one. A ``DataValidationResult`` built by
    hand, or unpickled from a version before the field existed, does not record
    its coverage, and must not be read as having recorded either answer. Turning
    every such PASS into COULD NOT CHECK would be a false alarm on the whole
    install base rather than a fix.
    """
    fn = getattr(result, "execution_coverage", None)
    if callable(fn):
        try:
            state = str(fn())
        except Exception as e:  # noqa: BLE001 - a render must not die on a stale result
            warnings.warn(f"result.execution_coverage() failed: {e}")
            return "unrecorded"
        if state in _COVERAGE_STATES:
            return state
        return "unrecorded"

    reported = d.get("execution_coverage")
    if isinstance(reported, str) and reported in _COVERAGE_STATES:
        return reported

    # The raw record, for a dict that carries the field without the derived
    # word. None (or absent) is unknown, NOT "none": an empty list is the
    # positive statement "it is recorded that no check ran", and a result that
    # never recorded anything must not be read as making that statement.
    checks_run = d.get("checks_run", _MISSING)
    if checks_run is _MISSING or checks_run is None:
        return "unrecorded"
    try:
        names = {str(c) for c in checks_run}
    except TypeError:
        return "unrecorded"
    if not names:
        return "none"
    # Whether a non-empty set is complete or partial is the source's judgement,
    # not this renderer's: it does not know the full check list and must not
    # invent one. Both are treated as "some checks ran", which is what the
    # verdict logic below actually needs.
    return "partial"


def _classify_severity(raw: Any) -> str:
    """Map a reported severity onto one of the buckets this page can count.

    Returns CRITICAL, ERROR, WARNING, INFO, or UNKNOWN for anything else.
    Colour, icon and counter all go through this one function, so an
    unrecognised severity can no longer be coloured as one thing and counted as
    another. It used to be coloured INFO-blue and counted in NO bucket at all,
    which printed "0 critical, 0 warn, 0 info" directly above a listed issue.
    """
    if hasattr(raw, "value"):
        raw = raw.value
    name = str(raw).strip().upper()
    if name in _SEVERITY_COLORS and name != "UNKNOWN":
        return name
    return "UNKNOWN"


def _coerce_passed(raw: Any) -> Optional[bool]:
    """Read a reported pass flag, or None when it does not carry a verdict.

    Only real booleans (including numpy's) and the ints 0/1 are accepted. A
    string, a NaN or any other object is NOT truthiness-tested: bool("False")
    and bool(float("nan")) are both True, so guessing here is how an unchecked
    input renders as a green PASS.
    """
    if raw is _MISSING or raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    item = getattr(raw, "item", None)  # numpy.bool_ and friends
    if callable(item):
        try:
            unwrapped = item()
        except Exception:
            return None
        if isinstance(unwrapped, bool):
            return unwrapped
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    return None


#  data_validation.svg


def data_validation_to_svg(
    result: Optional[Any] = None,
    *,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render data validation results as an SVG report.

    Parameters
    ----------
    result : DataValidationResult or dict
        If None, renders a could-not-check canvas that states plainly that
        nothing was validated (unless *example* is set).
    example : bool
        Render the built-in demonstration fixture, watermarked EXAMPLE on the
        canvas. Off by default and never reached implicitly: see the note below.
    explanation : str
    save_path : str, optional

    Returns
    -------
    str  SVG markup.

    Notes
    -----
    CRITICAL, do not restore the old default. Until 2026-08-27 a missing
    *result* silently dispatched to ``_demo`` and this function returned a
    complete, confident FAIL verdict for a dataset nobody had validated: five
    issues, "1 critical, 2 warnings, 2 info", "Dataset contains 5,230 records
    across 3 groups", "Feature 'zipcode' correlates 0.72 with protected
    attribute", a "15% outcome disparity" and six data-quality figures. None of
    it was measured from anything the caller supplied, and a caller could not
    tell that output apart from a real report. An SVG is an export format: it
    leaves the building, an auditor reads it, and it outlives the version that
    produced it, so a fabricated finding about someone's data is exactly the
    artifact this module must never emit.

    A FAIL is not a "safe" thing to invent either: it names groups, features and
    correlations that do not exist, and it is acted on. Missing input is COULD
    NOT CHECK, and it is stated on the canvas. The demo fixture survives only
    behind the explicit ``example=True``, watermarked.

    CLOSED 2026-08-28, and the shape of the fix must be preserved. A
    ``DataValidationResult`` that ran NO check used to render a green PASS here,
    because the object did not record which checks ran: ``DataBiasValidator``
    gates all five of its checks on config flags, and with every flag off it
    returned ``passed=True`` with the summary "Validation passed - no bias issues
    detected" (reproduced 2026-08-27). The source-side half is now a
    ``checks_run`` field and an ``execution_coverage()`` method on
    ``DataValidationResult`` (``operations.cicd.validator``), mirroring
    ``BiasAuditReport.modules_run``; ``_validation_coverage`` below is the reader
    and is wired into the verdict. A run that records ``checks_run == []`` is
    COULD NOT CHECK with its counters withheld, and a recorded failure still
    renders FAIL. A result that does NOT record its coverage answers
    "unrecorded" and renders exactly as before, which is deliberate: turning
    every PASS on the install base into COULD NOT CHECK would be a false alarm,
    not a fix.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    if result is None:
        if example:
            return _demo(explanation=explanation, save_path=save_path)
        return _not_validated(explanation=explanation, save_path=save_path)

    unreadable_reason: Optional[str] = None
    if isinstance(result, dict):
        d: Dict[str, Any] = result
    else:
        extracted = _obj_to_dict(result)
        if extracted is None:
            # Nothing could be read off the object, so nothing was validated as
            # far as this page can tell. It used to fall back to an empty dict,
            # which then took the old `passed` default of True and rendered a
            # green PASS for an input the adapter never understood.
            d = {}
            # Short enough to fit the banner strip; the technical detail goes to
            # the caller as a warning rather than off the edge of the canvas.
            unreadable_reason = f"the result object ({type(result).__name__}) could not be read"
            warnings.warn(
                f"{type(result).__name__} exposes no to_dict() and no dataclass fields; "
                "rendering the validation verdict as UNKNOWN"
            )
        else:
            d = extracted

    # Issues
    issues_raw = d.get("issues", [])
    issues: List[Dict[str, Any]] = []
    for iss in issues_raw:
        if hasattr(iss, "__dataclass_fields__"):
            iss = {k: getattr(iss, k) for k in iss.__dataclass_fields__}
        if not isinstance(iss, dict):
            continue
        # `_MISSING`, never "INFO". G13 2026-09-30, two defects in one line:
        #
        # * `.get(key, default)` does NOT fire when the key is PRESENT holding
        #   None, so `{"severity": None}` skipped the default and reached
        #   `str(None).upper()`. Measured: a severity badge reading "NONE", and
        #   `{"severity": nan}` one reading "NAN" -- content MINTED by `str()`
        #   out of the absence of a severity.
        # * the "INFO" default was itself a defaulted VERDICT, applied ABOVE the
        #   classifier that exists to answer this. An issue carrying no severity
        #   at all was badged INFO in informational blue and counted in
        #   `n_info`, i.e. filed as advisory, when `_classify_severity` would
        #   have called it UNKNOWN, coloured it slate and counted it with the
        #   warnings where the summary line names it. This is the same shape
        #   `adapters_discovery._severity_or_none` documents for
        #   `d.get("severity", "medium")`.
        raw_sev = iss.get("severity", _MISSING)
        if hasattr(raw_sev, "value"):
            raw_sev = raw_sev.value
        # ONE classification drives colour, icon, the badge TEXT and the header
        # counters. The badge text used to be a separate `str(raw).upper()`,
        # which is how it could read "NONE" while the colour and the counter
        # both said UNKNOWN; for every recognised severity the two strings are
        # identical, so nothing a reader can grade changes here.
        bucket = _classify_severity(raw_sev)
        sev = bucket
        color, bg = _SEVERITY_COLORS[bucket]
        icon = _SEVERITY_ICONS[bucket]
        affected = iss.get("affected_groups", [])
        if isinstance(affected, (list, tuple)):
            affected_str = ", ".join(str(g) for g in affected[:3])
        else:
            affected_str = str(affected)
        issues.append(
            {
                "severity": sev,
                "bucket": bucket,
                "color": color,
                "bg": bg,
                "icon": icon,
                "type": str(iss.get("issue_type") or "unspecified"),
                # `or ""`, not a get() default: ValidationIssue.recommendation is
                # Optional and defaults to None, and str(None) put the literal
                # word "None" in the Recommendations block of every report whose
                # first issues carried no recommendation.
                "message": str(iss.get("message") or "")[:90],
                "affected": affected_str,
                "recommendation": str(iss.get("recommendation") or "")[:80],
            }
        )

    # Metrics cards (only scalar and simple values, skip deeply nested dicts)
    metrics_raw = d.get("metrics", {})
    metric_cards: List[Dict[str, Any]] = []
    if isinstance(metrics_raw, dict):
        for k, v in list(metrics_raw.items())[:8]:
            if isinstance(v, (int, float, bool, str)):
                metric_cards.append(
                    {"label": str(k).replace("_", " ").title(), "value": _fmt_val(v)}
                )
            elif isinstance(v, dict):
                # Only include scalar sub-values (skip nested dicts/lists like counts, ratios)
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float, bool, str)):
                        metric_cards.append(
                            {
                                "label": f"{k}.{sub_k}".replace("_", " ").title(),
                                "value": _fmt_val(sub_v),
                            }
                        )
            elif isinstance(v, list) and len(v) <= 5 and all(isinstance(x, str) for x in v):
                metric_cards.append(
                    {"label": str(k).replace("_", " ").title(), "value": ", ".join(v)}
                )

    # Summary
    summary = d.get("summary", "")
    n_critical = sum(1 for i in issues if i["bucket"] in ("CRITICAL", "ERROR"))
    # The template carries three counters, so an ungraded issue is counted with
    # the warnings: it is the only bucket that neither claims a severity the
    # report cannot support nor files the issue away as advisory. The count is
    # named in the summary line below so the reader is not left guessing.
    n_unknown = sum(1 for i in issues if i["bucket"] == "UNKNOWN")
    n_warning = sum(1 for i in issues if i["bucket"] == "WARNING") + n_unknown
    n_info = sum(1 for i in issues if i["bucket"] == "INFO")

    coverage = _validation_coverage(d, result)
    verdict, verdict_reason = _verdict(d, issues, unreadable_reason, coverage)
    verdict_color, verdict_bg = _VERDICT_STYLE[verdict]

    recs: List[str] = []
    # Ahead of the per-issue advice, because a verdict drawn over a partial run
    # is the thing a reader most needs told before they act on it.
    if coverage == "partial":
        recs.append(
            "Only some validation checks ran; this verdict covers those and "
            "says nothing about the rest."
        )
    for iss in issues[:3]:
        if iss["recommendation"]:
            recs.append(iss["recommendation"])
    if n_unknown:
        recs.append(
            f"{n_unknown} issue(s) carry an unrecognised severity and could not be "
            "graded; read them in the table above."
        )
    if not recs:
        if verdict == "PASS":
            recs.append("All validation checks passed. The data is ready for fairness analysis.")
        elif verdict == "FAIL":
            recs.append("Resolve the issues listed above before using this data.")
        else:
            recs.append(f"No verdict could be established: {verdict_reason}.")

    summary_text = str(summary) if summary else f"{len(issues)} issue(s) found"
    if verdict == "UNKNOWN":
        summary_text = f"Verdict could not be established: {verdict_reason}"
    elif n_unknown:
        summary_text = f"{summary_text} ({n_unknown} of unrecognised severity)"
    # The template draws this line as a single unwrapped run ending at the
    # verdict banner, so an over-long summary slides underneath the banner and
    # is clipped mid-word. Truncate here instead, visibly.
    if len(summary_text) > 100:
        summary_text = summary_text[:97] + "..."

    template_data = {
        "title": "Data Validation Report",
        "timestamp": d.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
        # None, not a bool, for could-not-check: downstream readers of this key
        # must not be able to read an unassessed report as a passing one.
        "passed": True if verdict == "PASS" else (False if verdict == "FAIL" else None),
        "pass_label": verdict,
        "pass_color": verdict_color,
        "pass_bg": verdict_bg,
        "summary": summary_text,
        "issues": issues[:8],
        "n_critical": n_critical,
        "n_warning": n_warning,
        "n_info": n_info,
        "metric_cards": metric_cards[:6],
        "recommendations": recs[:4],
        "explanation": explanation,
    }

    # A run that records no executed check gets the same treatment as no result
    # at all: counters absent rather than zero, and the flag that keeps the
    # canvas and the accessible <desc> telling one story. "0 critical, 0 warn,
    # 0 info" is the same line a clean validation prints, and this run counted
    # nothing because it checked nothing.
    if verdict == "UNKNOWN" and coverage == "none":
        template_data["n_critical"] = None
        template_data["n_warning"] = None
        template_data["n_info"] = None
        template_data["not_assessable"] = True
        template_data["not_assessable_reason"] = (
            "This validation result records that no check ran, so its pass flag "
            "reports the absence of a failure rather than the presence of a pass."
        )

    return _render_and_save("data_validation", template_data, save_path)


def _not_validated(*, explanation: Optional[str] = None, save_path: Optional[str] = None) -> str:
    """The third state: a report for a run that validated nothing.

    Every field that could read as a finding is withheld rather than defaulted.
    ``not_assessable`` drives the template's withheld counter line, its "Nothing
    was validated" panels and (through ``rendering.explain.build_explanation``)
    the accessible ``<desc>`` and the action line, so the canvas and the screen
    reader tell one story. There is deliberately no verdict, no issue row, no
    severity tally and no data-quality figure: a reader must not be able to take
    a number off this canvas.

    Note the counters are absent, not zero. "0 critical, 0 warn, 0 info" is the
    same sentence as a clean bill of health, and this run counted nothing at all.
    """
    reason = (
        "No validation result was supplied to data_validation_to_svg, so no check "
        "ran and no issue was counted on this run."
    )
    template_data = {
        "title": "Data Validation Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # None, not a bool: a downstream reader of this key must not be able to
        # read an unvalidated report as a passing one.
        "passed": None,
        "pass_label": "NOT CHECKED",
        "pass_color": _UNKNOWN,
        "pass_bg": _UNKNOWN_BG,
        "summary": "Nothing was validated: no validation result was supplied",
        "issues": [],
        "n_critical": None,
        "n_warning": None,
        "n_info": None,
        "metric_cards": [],
        "not_assessable": True,
        "not_assessable_reason": reason,
        "recommendations": [
            "COULD NOT CHECK: no validation result was supplied, so no check ran.",
            "This report is not a pass and not a failure: it certifies nothing about any dataset.",
            "Pass a DataValidationResult (or a compatible dict) to obtain a verdict.",
        ],
        "explanation": explanation,
    }
    return _render_and_save("data_validation", template_data, save_path)


def _fmt_val(v) -> str:
    """Format a metric value for SVG display. Handles nested dicts/lists gracefully."""
    if isinstance(v, float):
        # G13 2026-09-30: `f"{float('nan'):.3f}"` is the string "nan", and this
        # formatter runs in Python, so it never passes the engine's `_to_float`
        # net that closed the same hole for every Jinja numeric filter. Measured:
        # a metric card on this canvas printed "nan" as a measured value, in the
        # font and weight of a measurement. "N/A" is the spelling those filters
        # already use for the third state, so there is one across the library.
        if not math.isfinite(v):
            return "N/A"
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, dict):
        # Summarize dict: show count of entries or key highlights
        if len(v) <= 3:
            return ", ".join(f"{k}: {_fmt_val(val)}" for k, val in v.items())
        return f"{len(v)} entries"
    if isinstance(v, (list, tuple)):
        if len(v) <= 3:
            return ", ".join(str(item) for item in v)
        return f"{len(v)} items"
    s = str(v)
    # Truncate overly long strings (raw repr leaking)
    if len(s) > 60:
        return s[:57] + "..."
    return s


def _verdict(
    d: Dict[str, Any],
    issues: List[Dict[str, Any]],
    unreadable_reason: Optional[str],
    coverage: str = "unrecorded",
) -> tuple:
    """Decide the headline verdict: PASS, FAIL or UNKNOWN, plus the reason.

    UNKNOWN is a first-class outcome. The banner used to read PASS whenever the
    `passed` key was absent, so an unreadable object and a dict that simply
    never carried a verdict both rendered green. Nothing here defaults to pass.
    """
    if unreadable_reason:
        return "UNKNOWN", unreadable_reason

    # A run that records it executed NO check has no verdict to report: its
    # `passed=True` is the absence of a failure, not the presence of a pass.
    # A recorded FAIL still stands, because an issue was raised by something.
    if coverage == "none" and not any(i["bucket"] in ("CRITICAL", "ERROR") for i in issues):
        return "UNKNOWN", "the result records that no validation check ran"

    raw = d.get("passed", _MISSING)
    reported = _coerce_passed(raw)
    blocking = sum(1 for i in issues if i["bucket"] in ("CRITICAL", "ERROR"))

    if reported is None:
        if blocking:
            # DataBiasValidator's own rule (validator.py: has_errors -> not
            # passed), so this is the library's definition of failure applied to
            # the issues in hand, not a verdict invented by the renderer.
            return "FAIL", ""
        if raw is _MISSING or raw is None:
            return "UNKNOWN", "the result carries no pass/fail flag"
        return "UNKNOWN", f"the pass/fail flag is not a boolean but a {type(raw).__name__}"

    if reported and blocking:
        return (
            "UNKNOWN",
            f"a pass is reported above {blocking} listed critical/error issue(s)",
        )
    return ("PASS" if reported else "FAIL"), ""


def _obj_to_dict(obj) -> Optional[dict]:
    """Read a result object into a dict, or None when it cannot be read.

    None, never an empty dict: an empty dict is indistinguishable from a result
    that genuinely validated nothing, and it used to be handed straight to the
    old `passed` default of True.
    """
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception as e:  # a result that cannot serialise itself
            warnings.warn(f"result.to_dict() failed: {e}")
            return None
        return converted if isinstance(converted, dict) else None
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return None


def _render_and_save(template: str, data: dict, save_path: Optional[str]) -> str:
    try:
        svg = render_svg(template, data)
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


def _demo(*, explanation: Optional[str] = None, save_path: Optional[str] = None) -> str:
    """Build template data with demo values. Reachable only via example=True."""
    issues = [
        {
            "severity": "CRITICAL",
            "color": _FAIL,
            "bg": "#fef2f2",
            "icon": "!!",
            "type": "imbalanced_groups",
            "message": "Group 'age>65' has only 12 samples (minimum: 30)",
            "affected": "age>65",
            "recommendation": "Collect more data for underrepresented group",
        },
        {
            "severity": "WARNING",
            "color": _WARN,
            "bg": "#fffbeb",
            "icon": "~",
            "type": "missing_values",
            "message": "Feature 'income' has 8.3% missing values for group 'minority'",
            "affected": "minority",
            "recommendation": "Investigate missing data pattern for bias",
        },
        {
            "severity": "WARNING",
            "color": _WARN,
            "bg": "#fffbeb",
            "icon": "~",
            "type": "outcome_disparity",
            "message": "Positive outcome rate differs by 15% across groups",
            "affected": "female, minority",
            "recommendation": "Review labelling process for potential bias",
        },
        {
            "severity": "INFO",
            "color": _INFO,
            "bg": "#eff6ff",
            "icon": "i",
            "type": "feature_correlation",
            "message": "Feature 'zipcode' correlates 0.72 with protected attribute",
            "affected": "race",
            "recommendation": "Consider proxy variable analysis",
        },
        {
            "severity": "INFO",
            "color": _INFO,
            "bg": "#eff6ff",
            "icon": "i",
            "type": "data_quality",
            "message": "Dataset contains 5,230 records across 3 groups",
            "affected": "",
            "recommendation": "",
        },
    ]
    metric_cards = [
        {"label": "total_records", "value": "5,230"},
        {"label": "n_groups", "value": "3"},
        {"label": "min_group_size", "value": "12"},
        {"label": "missing_rate", "value": "3.2%"},
        {"label": "outcome_rate", "value": "0.340"},
        {"label": "max_disparity", "value": "0.150"},
    ]
    recs = [
        "Collect more data for underrepresented group (age>65, n=12).",
        "Investigate missing data pattern for bias in 'income' feature.",
        "Review labelling process: 15% outcome disparity across groups.",
    ]

    template_data = {
        "title": "Data Validation Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # DO NOT REMOVE. Every issue, record count and correlation below is
        # invented. This flag is what puts the EXAMPLE band and watermark on the
        # canvas and the EXAMPLE ONLY marker at the head of the accessible
        # description, and it is the only thing that stops this render being
        # read as a finding about a real dataset.
        "is_example": True,
        "passed": False,
        "pass_label": "FAIL",
        "pass_color": _FAIL,
        "pass_bg": "#fef2f2",
        "summary": "5 issues found (1 critical, 2 warnings, 2 info)",
        "issues": issues,
        "n_critical": 1,
        "n_warning": 2,
        "n_info": 2,
        "metric_cards": metric_cards,
        "recommendations": recs,
        "explanation": explanation,
    }

    return _render_and_save("data_validation", template_data, save_path)
