"""
Adapters that transform vfairness report objects into flat dictionaries
suitable for SVG template rendering.

Each adapter:
    1. Accepts a report object (dataclass or dict).
    2. Extracts and pre-computes all values the SVG template needs.
    data["explanation"] = explanation
    3. Returns a plain dict that is passed to ``render_svg()``.
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..evaluation.vfairness_metrics._metric_direction import is_ratio_metric
from .adapters_fairness import (
    _UNKNOWN_BG,
    _UNKNOWN_COLOR,
    COULD_NOT_CHECK,
    FAIL,
    NOT_ASSESSABLE_TEXT,
    PASS,
    _fit_subtitle,
    _is_finite,
    _is_missing_level,
    _metric_state,
    _state_style,
    _to_finite_float,
    report_assessability,
)
from .engine import _risk_bg, _risk_color, _risk_label, render_svg


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _fmt_ts(ts) -> str:
    """Normalise any timestamp to the standard '%Y-%m-%d %H:%M' format."""
    if ts is None:
        return _now()
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    # ISO string or similar: parse then format
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return _now()


# 1.  BiasAuditReport  →  bias_audit.svg


#: Printed in a module tile's detail line when some of its findings carried no
#: gradable severity. It has to survive the template's truncate_text(18), so it
#: is short; the same count is spelled out on the headline line.
_UNGRADED_FINDING_TEXT = "ungraded"


def _module_score(
    findings: list, score_attr: str = "risk_score", *, ran: Optional[bool] = None
) -> Tuple[Optional[float], int]:
    """Average risk score for a list of finding objects, and the ungraded count.

    Returns ``(score, n_ungraded)``. ``score`` is None when nothing in the list
    could be graded; ``n_ungraded`` counts the findings that reported no usable
    severity and were therefore kept OUT of the average.

    Falls back to mapping risk_level/severity enums or using
    confidence_score / correlation as a numeric proxy.

    Returns None, NEVER 0.0, for an empty list of unknown provenance. A module
    that returned no finding did not necessarily measure a risk of zero: the
    same empty list arrives when the module ran and cleared the data and when
    it was never run at all (``BiasDetector.full_audit(include_proxies=False)``
    and friends), and the list alone cannot tell those apart. 0.0 lit the
    emerald tile through ``engine._risk_color``, so a module that did nothing
    showed up green. ``_risk_color``/``_risk_bg`` already map None to slate,
    and the |f2 filter prints it as "N/A".

    *ran* is that missing provenance, and the ONLY thing that turns an empty
    list back into a measured 0.0: pass True only when the report records that
    this module executed (``BiasAuditReport.modules_run``). ``None`` and
    ``False`` both stay withheld, because "not recorded" is not "did not run"
    and neither of them is "ran and found nothing".

    PER ROW, and this is the half the earlier waves left open. A finding whose
    severity could not be read used to be scored anyway, in two different ways,
    and BOTH of them were then averaged into the tile the reader grades the
    module by:

    * ``risk_level=None`` reached ``str(None).lower()``, which is the string
      "none", which is a KEY of ``_enum_map``, so an absent severity scored
      0.05 (negligible) and pulled the module's risk DOWN. A finding that
      reported no severity is not a finding of negligible risk, and this is the
      quiet direction of the bug: absence lowering a risk.
    * a severity string the map does not know, and a finding object carrying
      none of the numeric or enum attributes at all, both scored a flat 0.5,
      i.e. a MEDIUM risk invented out of an unreadable row. That is the mirror
      direction, a fabricated breach, and it is no safer than the first.

    Neither is graded now. An unreadable finding is counted and returned, and
    the caller states the count on the canvas; the average is taken over the
    findings that WERE graded, which is the headline rule applied per module.
    ``risk_level=RiskLevel.NONE`` still scores 0.05: that is a severity the
    detector actually reported.
    """
    if not findings:
        return (0.0 if ran else None), 0

    _enum_map = {
        "critical": 0.95,
        "high": 0.75,
        "medium": 0.50,
        "low": 0.25,
        "none": 0.05,
        "negligible": 0.05,
        "adequate": 0.10,
        "overrepresented": 0.30,
    }

    def _enum_score(raw) -> Optional[float]:
        """The mapped score for one severity/risk_level, or None if unreadable.

        ``raw is None`` is tested BEFORE the str() fall-back on purpose: without
        it "none" is a live key of ``_enum_map`` and an absent severity scored
        0.05. A value the map does not know returns None as well; it is NOT
        0.5, because a midpoint is a measurement.
        """
        if raw is None:
            return None
        val = raw.value if hasattr(raw, "value") else raw
        if val is None:
            return None
        return _enum_map.get(str(val).lower())

    scores = []
    n_ungraded = 0
    for f in findings:
        # Try explicit numeric score fields first
        if hasattr(f, "confidence_score") and f.confidence_score is not None:
            scores.append(float(f.confidence_score))
        elif hasattr(f, "correlation") and f.correlation is not None:
            scores.append(min(1.0, abs(float(f.correlation))))
        elif hasattr(f, "effect_size") and f.effect_size is not None:
            scores.append(min(1.0, abs(float(f.effect_size))))
        # Fall back to enum mapping
        elif hasattr(f, "risk_level") or hasattr(f, "severity"):
            mapped = _enum_score(getattr(f, "risk_level", None))
            if mapped is None:
                mapped = _enum_score(getattr(f, "severity", None))
            if mapped is None:
                n_ungraded += 1
            else:
                scores.append(mapped)
        else:
            # No numeric proxy and no severity of any kind: nothing to grade.
            n_ungraded += 1
    return (sum(scores) / len(scores) if scores else None), n_ungraded


# The four detection modules of BiasDetector.full_audit, in report order and
# under the names full_audit records in BiasAuditReport.modules_run. Pinned
# against detector.AUDIT_MODULES by tests/test_bias_audit_ran_nothing.py: if a
# name here ever stopped matching the detector's, the mismatch degrades this
# canvas to could-not-check (a module whose name is not in the record reads as
# "did not run"), never to a false all-clear.
_AUDIT_MODULES = ("historical", "representation", "disparities", "proxies")

# Every state ``BiasAuditReport.execution_coverage`` can return. It returns
# FIVE and this tuple listed four, so "ran_but_assessed_nothing" -- the detector
# saying every module executed and not one of them could assess anything -- fell
# through to "unrecorded" and the canvas rendered "OVERALL RISK 0% MINIMAL" over
# it. The report knew; the renderer had no word for what it was told.
# tests/test_bias_audit_svg_coverage.py pins this tuple against the detector's
# own constants, so a sixth state cannot be added on one side alone.
_COVERAGE_STATES = ("complete", "partial", "none", "ran_but_assessed_nothing", "unrecorded")


def _modules_that_ran(report) -> Optional[set]:
    """The set of audit modules the report records as executed, or None.

    None means the report does not record it (an older or hand-built report),
    which is unknown, not empty.
    """
    modules_run = getattr(report, "modules_run", None)
    if modules_run is None:
        return None
    try:
        return {str(m) for m in modules_run}
    except TypeError:
        return None


def _audit_coverage(report) -> str:
    """'complete' | 'partial' | 'none' | 'unrecorded' for a bias audit report.

    Asks the report itself (``BiasAuditReport.execution_coverage``) so the
    definition of a complete audit lives with the detector that runs it, and
    falls back to "unrecorded" for anything that cannot answer. Unrecorded is
    the safe fallback in both directions: it withholds the verdict rather than
    granting or refusing one.
    """
    fn = getattr(report, "execution_coverage", None)
    if callable(fn):
        try:
            state = str(fn())
        except Exception:  # noqa: BLE001 - a render must not die on a stale report
            return "unrecorded"
        if state in _COVERAGE_STATES:
            return state
    return "unrecorded"


def _risk_style(score: Optional[float]) -> Dict[str, str]:
    """Tile colours for a module score that may be absent.

    ``engine._risk_color`` / ``_risk_bg`` already map a non-number to slate
    (their ``_is_nan`` guard, added because a NaN score fell through to the
    MINIMAL branch and rendered as a pass). This wrapper carries the Optional
    through the type checker without widening their signatures, and it must
    keep mapping None to the slate pair: a module that returned no finding is
    not a module that measured no risk.
    """
    value = float("nan") if score is None else score
    return {"color": _risk_color(value), "bg": _risk_bg(value)}


def _module_tile(
    name: str, noun: str, findings: list, score: Optional[float], n_ungraded: int
) -> Dict[str, Any]:
    """One module tile, whose detail line names its ungraded findings.

    The tile carries a number a reader grades the module by, so an ungraded
    finding must be visible ON the tile and not only in the aggregate. The
    template truncates this line at 18 characters, which is why the wording is
    "N groups · M ungraded" rather than a sentence.
    """
    detail = f"{len(findings)} {noun}"
    if n_ungraded:
        # "2 patterns · 1 ungraded" is 23 characters and the template truncates
        # at 18, so it rendered as "2 patterns · 1 ung…" and the reader lost the
        # word that carries the meaning. Rasterised, then reworded: the graded
        # fraction says the same thing in 13.
        detail = f"{len(findings) - n_ungraded} of {len(findings)} graded"
    return {
        "name": name,
        "score": score,
        "count": len(findings),
        "n_ungraded": n_ungraded,
        "detail": detail,
        **_risk_style(score),
    }


def _extract_issues(report) -> List[Dict[str, str]]:
    """Build a uniform list of {type, title, detail, severity} dicts."""
    issues: List[Dict[str, str]] = []

    # From critical_issues (already dicts)
    for ci in (report.critical_issues or [])[:6]:
        issues.append(
            {
                "type": ci.get("type", "Issue"),
                "title": ci.get("description", "")[:60],
                "detail": ci.get("details", ci.get("description", ""))[:80],
                "severity": ci.get("severity", "high"),
            }
        )

    return issues


def _representation_bars(report) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build bar-chart data from representation findings, and name the ungraded.

    Returns ``(bars, groups_without_benchmark)``.

    CRITICAL, do not restore the ``, 1.0)`` default on the ratio lookup. The
    benchmark bar is derived as ``dataset_pct / ratio``, so a ratio of 1.0 makes
    the benchmark EQUAL the dataset share, and the panel then draws two bars of
    identical length under the legend "Dataset / Population benchmark": a
    picture of perfect representation, for a group whose representation ratio
    was never in the finding at all. That is a fabricated all-clear on the one
    panel a reader scans for under-representation.

    A ratio of 0, a negative ratio and a non-finite one are refused for the same
    reason: the old ``if ratio and ratio > 0 else dataset_pct`` fell back to the
    dataset share, which draws the same identical pair. The benchmark is left as
    None, the template draws no benchmark bar for that row and says so, and the
    group's name is returned so the caller can put the count on the canvas.

    The returned names are read back off the TRUNCATED bar list rather than
    collected as the rows are built. Both lists used to be capped at six
    independently, so a report with more groups than fit could hand the canvas a
    count taken from rows the canvas does not show: "6 of 6 groups had no
    population benchmark" printed over six rows that visibly carry one. A count
    on a canvas has to be a count OF that canvas.
    """
    bars = []
    for r in (report.representation_findings or [])[:4]:
        # RepresentationBiasResult has group_distributions and representation_ratios.
        #
        # DO NOT lower-case `group_name` here. The two dicts are keyed
        # differently by design (labels as the data spells them / labels folded
        # for benchmark matching), and this lookup used to MISS for every
        # dataset whose labels are not already lower-case, which is essentially
        # all of them: this gallery's own audit reported "no population
        # benchmark was reported for this group" on all six rows of a report
        # that supplied a benchmark for each. The join is fixed at the SOURCE,
        # in representation._GroupRatios, so the raw label is the right thing to
        # pass and every other caller gets the fix too. Normalising here again
        # would only hide the next mismatch.
        ratios = getattr(r, "representation_ratios", None) or {}
        for group_name, dataset_pct in list(r.group_distributions.items())[:6]:
            ratio = _to_finite_float(ratios.get(group_name))
            # Estimate benchmark proportion from dataset_pct / ratio. Both ends
            # must be real numbers before a benchmark exists at all.
            share = _to_finite_float(dataset_pct)
            if ratio is None or ratio <= 0 or share is None:
                benchmark_pct = None
            else:
                benchmark_pct = share / ratio
            bars.append(
                {
                    "group": str(group_name)[:20],
                    "attribute": str(r.attribute)[:15],
                    "dataset_pct": share,
                    "benchmark_pct": benchmark_pct,
                    "severity": r.severity.value
                    if hasattr(r.severity, "value")
                    else str(r.severity),
                }
            )
    bars = bars[:6]
    no_benchmark: List[str] = [str(b["group"]) for b in bars if b["benchmark_pct"] is None]
    return bars, no_benchmark


def bias_audit_to_svg(
    report, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """
    Render a BiasAuditReport as a polished SVG dashboard.

    Parameters
    ----------
    report : BiasAuditReport
        The report object returned by ``BiasDetector.full_audit()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Raises
    ------
    TypeError
        If *report* is not a bias-audit report. The report is a REQUIRED
        argument, so its absence is a caller mistake, not a run that measured
        nothing: there is no timestamp, no protected attribute and no dataset
        to draw a canvas about. Rendering a could-not-check canvas here would
        manufacture an audit-shaped export out of a programming error, and that
        file then gets attached to a ticket. The could-not-check canvas below
        is for a report that EXISTS and found nothing; this is not that.
        Previously any such call died on ``'NoneType' object has no attribute
        'historical_findings'``, which names nothing a caller can act on.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    Three states, never two, and the third one is decided by the RECORD of what
    executed rather than by the score. ``_calculate_overall_risk`` returns 0.0
    both for a dataset every module cleared and for a run where no module
    executed (``full_audit(include_historical=False, ...)``), so an earlier
    wave had to withhold the verdict from both, which cost a real clean result
    its green badge. ``BiasAuditReport.modules_run`` now records which modules
    ran, so a zero-finding audit renders:

    * a genuine "OVERALL RISK 0% MINIMAL" when the record shows all four
      modules executed against at least one protected attribute,
    * COULD NOT CHECK when the record shows some or none of them executed, or
      when there is no record at all (an older or hand-built report), or when
      no protected attribute was audited.

    Could-not-check is not a rejection either: the canvas says the audit
    certifies nothing, not that the data failed.
    """
    missing = [
        attr
        for attr in (
            "historical_findings",
            "representation_findings",
            "disparity_findings",
            "proxy_findings",
            "overall_risk_score",
            "critical_issues",
            "recommendations",
        )
        if not hasattr(report, attr)
    ]
    if missing:
        raise TypeError(
            f"bias_audit_to_svg() needs a BiasAuditReport (as returned by "
            f"BiasDetector.full_audit()); got {type(report).__name__}, which is missing: "
            f"{', '.join(missing)}."
        )

    # What actually executed. `ran` is None when the report does not record it,
    # and `m in None` is not askable, so each tile is told False in that case:
    # unknown and did-not-run both withhold the tile score, only a recorded run
    # turns an empty finding list into a measured 0.00.
    coverage = _audit_coverage(report)
    # The report's own answer to "could anything be assessed at all". Computed
    # here, beside `coverage`, because BOTH the headline badge and the four
    # module tiles need it: a module that executed over a frame where no group
    # was large enough to compare did not measure a risk of 0.00, and a tile
    # reading "0.00 / 0 patterns" in emerald is the same false all-clear as the
    # headline, one level down and four times over.
    assessed_nothing = coverage in ("none", "ran_but_assessed_nothing")
    # BGL6 AUDIT, 2026-09-29. The TILES need a stronger fact than
    # `assessed_nothing`. "unrecorded" is the state where the report lists all
    # four modules and says NOTHING about whether any protected attribute was
    # assessed, so `module in ran` is True for all four and each tile turned an
    # empty finding list into a measured 0.00. Measured before this flag existed,
    # on a report with modules_run = all four audit modules and no assessment
    # record: execution_coverage() 'unrecorded', headline NOT ASSESSABLE, and
    # FOUR '>0.00<' tiles still on the canvas, which the batch's own pin for a
    # 2-row report asserts must be zero. A correct headline with fabricated tiles
    # underneath it is still a fabricated measurement; the reader reads both.
    #
    # `assessed_nothing` is kept as it was, because the HEADLINE sentence below
    # names that state specifically and must go on distinguishing "all four ran
    # and assessed nothing" from "we do not know".
    # The same question asked one state further out. `execution_coverage` returns
    # "partial" BEFORE it consults the assessment half, so a partial report whose
    # assessment record is missing reached the tiles with the flag True and the
    # module that ran rendered a measured 0.00 from an empty list. Measured
    # 2026-09-29: modules_run=['historical'] with no assessment record rendered
    # one '>0.00<' tile. Asked of the report directly, so the flag depends on the
    # record rather than on which branch of the execution word it arrived through.
    # "complete" is unaffected by construction: execution_coverage only says
    # complete when the assessment half is neither none nor unrecorded.
    _assessment = None
    _ac = getattr(report, "assessment_coverage", None)
    if callable(_ac):
        try:
            _assessment = str(_ac())
        except Exception:  # noqa: BLE001
            _assessment = None
    assessment_recorded = (
        coverage not in ("none", "ran_but_assessed_nothing", "unrecorded")
        and _assessment is not None
        and _assessment not in ("none", "unrecorded")
    )
    modules_recorded = _modules_that_ran(report) is not None
    ran = _modules_that_ran(report) or set()
    # How many of the FOUR audit modules ran. `modules_run` is a free-form list,
    # so `len(ran)` counted whatever names the report supplied, which is not the
    # same question: see the note on the partial-coverage sentence below.
    ran_recognised = [m for m in _AUDIT_MODULES if m in ran]

    # Module scores. None where a module returned no finding and nothing says
    # it ran, see _module_score.
    def _ran(module: str) -> bool:
        """Did this module both execute AND have something it could assess?

        `_module_score` turns an empty finding list into a measured 0.00 on the
        strength of `ran`, and executing is only half of the provenance it needs.
        On a 2-row single-group frame all four modules execute, none of them can
        reach a verdict, and three tiles rendered "0.00" in emerald.

        BGL6 2026-09-29: the second half is `assessment_recorded`, not
        `not assessed_nothing`. A report that does not SAY whether anything was
        assessable establishes the 0.00 no better than one that says nothing was,
        and a tile is a measurement claim either way.
        """
        return module in ran and assessment_recorded

    hist_score, hist_ungraded = _module_score(report.historical_findings, ran=_ran("historical"))
    repr_score, repr_ungraded = _module_score(
        report.representation_findings, ran=_ran("representation")
    )
    stat_score, stat_ungraded = _module_score(report.disparity_findings, ran=_ran("disparities"))
    proxy_score, proxy_ungraded = _module_score(report.proxy_findings, ran=_ran("proxies"))
    n_ungraded_findings = hist_ungraded + repr_ungraded + stat_ungraded + proxy_ungraded

    issues = _extract_issues(report)
    bars, groups_without_benchmark = _representation_bars(report)
    # ── Could-not-check, decided HERE and not in the template ──────────────
    # DO NOT collapse this into the score branch. The template bands
    # overall_risk_score through engine._risk_label, and 0.0 lands in MINIMAL
    # with an emerald 0% beside it. On a report with nothing in any of the four
    # finding lists that badge certifies a clean dataset out of four empty
    # lists, which is the fabricated-all-clear shape closed across this surface.
    n_findings = sum(
        len(getattr(report, attr, None) or [])
        for attr in (
            "historical_findings",
            "representation_findings",
            "disparity_findings",
            "proxy_findings",
        )
    )
    # A zero-finding audit is a MEASURED all-clear only when the record shows
    # all four modules executed and there was at least one protected attribute
    # for them to compare across. Anything less is an absence: a partial run
    # scores what it looked at, not the four axes the badge is read as covering,
    # and an audit with no protected attribute had no group to compare at all.
    protected = list(getattr(report, "protected_attributes", None) or [])
    measured_clean = coverage == "complete" and bool(protected)
    # THE HEADLINE NUMBER ITSELF. `not_assessable` used to key ONLY on the
    # finding count, which is the wrong thing to ask about the OVERALL RISK
    # badge: the badge renders `overall_risk_score`, and that is a separate
    # field a report can simply not carry. A report with REAL findings and no
    # score therefore took the assessable branch, handed the template None, and
    # the template's `(overall_score * 100)|int` raised TypeError, which
    # engine.render_svg re-raises as ValueError. Executed on this repo: one
    # historical finding with risk_level "high" and overall_risk_score=None
    # killed the render outright. A partial row must never crash the canvas, and
    # a score nobody reported is could-not-check, so the badge is WITHHELD.
    #
    # It is withheld through its OWN flag, ``score_assessable``, and NOT by
    # raising ``not_assessable``. That distinction is the whole point, and it
    # was found by rasterising rather than by reading: ``not_assessable`` is
    # what ``rendering.explain`` keys on, and it replaces the accessible <desc>
    # with "COULD NOT CHECK: nothing on this chart was assessed". On a report
    # carrying a HIGH critical issue that sentence is false, and it is false in
    # the dangerous direction: the sighted reader sees the issue table and the
    # screen-reader user is told nothing was found. The description must agree
    # with the canvas. So: the BADGE is withheld, the findings stay, and the
    # <desc> keeps reporting them.
    #
    # The value is NORMALISED here as well, not only tested. `_is_finite`
    # answers True for the string "0.4", so the raw attribute reached the
    # template and `overall_score * 100` multiplied a STRING (Jinja repeats it),
    # and `_risk_color` compared a str to a float and raised TypeError before
    # that. One coercion feeds the badge, the band and the three colour helpers,
    # so they cannot disagree about what the score is.
    overall_value = _to_finite_float(getattr(report, "overall_risk_score", None))
    _score_or_nan = float("nan") if overall_value is None else overall_value
    score_reported = overall_value is not None
    # The report's OWN verdict on whether anything was assessable, which
    # `n_findings` cannot see. An INSUFFICIENT_DATA representation row IS a
    # finding by count, so a 2-row single-group frame arrived here with
    # n_findings == 1, took the assessable branch, and rendered "OVERALL RISK
    # 0% MINIMAL" over a report whose own recommendation read "Every audit
    # module executed and none of them assessed anything ... Nothing here clears
    # the data". Measured by execution 2026-09-25, BGL to_svg.
    #
    # `not issues` keeps the asymmetry the surrounding code is built on: a report
    # carrying a graded critical issue must keep its findings and its <desc>,
    # whatever the coverage says. Withholding is for the all-clear, never for a
    # finding.
    nothing_measured = (n_findings == 0 and not issues and not measured_clean) or (
        assessed_nothing and not issues
    )
    not_assessable = nothing_measured or (not score_reported and n_findings == 0 and not issues)
    not_assessable_reason = ""
    if not score_reported and not nothing_measured:
        # Stated in the score's own terms, and it must NOT read as an all-clear
        # nor as a breach: the findings below are untouched by it.
        not_assessable_reason = (
            "this report carries no overall risk score, so no aggregate risk was computed "
            "and none is shown here."
        )
        if n_findings or issues:
            not_assessable_reason += (
                f" The {n_findings} finding(s) listed below still stand and are not cleared "
                f"by this canvas."
            )
    elif not_assessable and not score_reported:
        not_assessable_reason = (
            "this report carries no overall risk score and no module returned a finding, so "
            "nothing was aggregated and nothing was found."
        )
    elif not_assessable:
        if coverage == "none":
            not_assessable_reason = (
                "no module of this audit executed, so the 0.0 risk score aggregates nothing "
                "and no finding was possible."
            )
        elif coverage == "ran_but_assessed_nothing":
            not_assessable_reason = (
                "every audit module executed and not one of them could assess anything: no "
                "protected attribute carried enough observations for any module to reach a "
                "verdict, so the risk score aggregates nothing. This canvas does not clear "
                "the dataset."
            )
        elif coverage == "partial":
            # Count the modules RECOGNISED as having run, not the names the
            # report happened to supply. `modules_run` is a free-form list, so a
            # renamed, misspelled or third-party entry counted towards the
            # numerator while still appearing in `did_not_run`, and the sentence
            # contradicted itself while overstating coverage: a report recording
            # ["historical", "bogus_a", "bogus_b"] read "only 3 of the 4 audit
            # modules executed (representation, disparities, proxies did not
            # run)", three plus three out of four, when exactly ONE of the four
            # axes had been looked at. Overstated coverage is the fabricated
            # all-clear in miniature: it credits the audit with checks it never
            # made. `_AUDIT_MODULES` is the only list that decides, and it is the
            # same list `did_not_run` is built from, so the two halves can no
            # longer disagree.
            did_not_run = ", ".join(m for m in _AUDIT_MODULES if m not in ran)
            not_assessable_reason = (
                f"only {len(ran_recognised)} of the {len(_AUDIT_MODULES)} audit modules "
                f"executed ({did_not_run} did not run), so no finding here covers the axes "
                "they would have checked."
            )
        elif coverage == "complete":
            not_assessable_reason = (
                "every audit module executed and none returned a finding, but no protected "
                "attribute was audited, so no module had a group to compare."
            )
        elif modules_recorded:
            # BGL6 AUDIT, 2026-09-29. "unrecorded" has TWO routes and they are
            # DIFFERENT SILENCES, exactly as detector.empty_is_not_a_measurement
            # says in its own comment: the report may record no modules_run at
            # all, or it may record all four and say nothing about whether any
            # protected attribute was assessed. The single else branch printed
            # "this report does not record which modules executed" for both, and
            # measured on a report listing all four audit modules that sentence
            # is FALSE, printed on the canvas a reader looks at, while the
            # detector's own reason sentence had already been corrected to the
            # true one. A reason a reader can check has to be the true one.
            not_assessable_reason = (
                "every audit module executed and this report does not record whether any "
                "requested protected attribute was assessed, so this canvas cannot tell a "
                "dataset every module cleared from an audit that had nothing to look at."
            )
        else:
            not_assessable_reason = (
                "no module of this audit returned a finding, and this report does not record "
                "which modules executed, so this canvas cannot tell a dataset every module "
                "cleared from a run where no module ran."
            )
        if not protected and coverage != "complete":
            not_assessable_reason += " No protected attribute was audited."

    # The qualifier line under the subtitle, at y=136, inside the headline band.
    # The template used to hard-code "this audit neither clears the dataset nor
    # faults it", which is right for an audit that found nothing and WRONG for a
    # report that carries findings and no aggregate score: that one does fault
    # the dataset, it just cannot total it up. The wording is decided here, where
    # it is testable, and the template prints whatever this says.
    parts = []
    if not score_reported:
        parts.append(
            f"no overall risk score was reported; the {n_findings} finding(s) below stand"
            if (n_findings or issues)
            else "no overall risk score was reported, so nothing here is totalled"
        )
    elif not_assessable:
        parts.append("this audit neither clears the dataset nor faults it")
    # Part (c) of the headline rule: the ungraded count goes ON THE CANVAS, in
    # the headline band, whatever the verdict. A finding that reported no
    # severity is in no module average and in no badge, so the reader has to be
    # told it exists or the badge reads as covering it.
    if n_ungraded_findings:
        parts.append(f"{n_ungraded_findings} finding(s) reported no severity, none graded")
    # ONE line, and TRIMMED to the canvas. Found by rasterising: the two clauses
    # together ran to roughly x=900 on a 680-wide canvas, so the half saying the
    # findings are ungraded was off the page entirely. A qualifier that does not
    # fit is a qualifier that was not made. `_fit_subtitle` measures the 10px
    # subtitle line and this one is 9px, so it is a conservative budget.
    na_line = _fit_subtitle(" · ".join(parts)) if parts else ""

    data = {
        "title": "BiasDetector: Audit Report",
        "overall_score": overall_value,
        # NaN, not the raw Optional, for the same reason _risk_style already
        # does it: the three helpers are annotated `float` and already map a
        # non-number to the slate / "N/A" trio through their own _is_nan guard
        # (added when a NaN score fell through to the MINIMAL branch and
        # rendered as a pass). Widening their signatures here would be three
        # real mypy errors and no behaviour change.
        "overall_color": _risk_style(overall_value)["color"],
        "overall_bg": _risk_style(overall_value)["bg"],
        "overall_label": _risk_label(_score_or_nan),
        "not_assessable": not_assessable,
        "not_assessable_text": NOT_ASSESSABLE_TEXT,
        "not_assessable_reason": not_assessable_reason,
        "unknown_color": _UNKNOWN_COLOR,
        "unknown_bg": _UNKNOWN_BG,
        # Coverage travels to the canvas so the subtitle can say when a score
        # came from fewer than the four modules the reader sees tiles for.
        "coverage": coverage,
        # The same recognised count the coverage sentence uses. The subtitle and
        # the sentence render the SAME number on the SAME canvas, so they cannot
        # be allowed to disagree about how much of the audit ran.
        "n_modules_run": len(ran_recognised),
        "n_modules": len(_AUDIT_MODULES),
        "n_critical": len(report.critical_issues),
        "n_ungraded_findings": n_ungraded_findings,
        "modules": [
            _module_tile(
                "Historical", "patterns", report.historical_findings, hist_score, hist_ungraded
            ),
            _module_tile(
                "Representation",
                "groups",
                report.representation_findings,
                repr_score,
                repr_ungraded,
            ),
            _module_tile(
                "Statistical", "disparities", report.disparity_findings, stat_score, stat_ungraded
            ),
            _module_tile("Proxy", "proxies", report.proxy_findings, proxy_score, proxy_ungraded),
        ],
        "na_line": na_line,
        # The badge's OWN third state, separate from the canvas-level one above.
        # See the note at `score_reported`: raising `not_assessable` here would
        # tell every screen-reader user that nothing was assessed while the
        # issue table beside it lists a HIGH finding.
        "score_assessable": score_reported,
        "issues": issues,
        "bars": bars,
        # Named, not silently dropped: a benchmark bar that is simply absent
        # from a row reads as a row nobody had a benchmark question about.
        "groups_without_benchmark": groups_without_benchmark,
        "n_groups_without_benchmark": len(groups_without_benchmark),
        "recommendations": (report.recommendations or [])[:3],
        "timestamp": _fmt_ts(report.timestamp),
    }

    data["explanation"] = explanation
    svg = render_svg("bias_audit", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 2.  CalibrationReport  →  calibration_report.svg


def _calibration_curves(report) -> Tuple[List[Dict[str, Any]], int]:
    """Build the per-group table rows, and count the keys with no group label.

    Returns ``(curves, n_keys_with_no_group_label)``.

    A metric the report never supplied comes back as None, NEVER as 0. The old
    ``metrics.get("ece", 0)`` default printed "0.000" in the ECE column and lit
    the green GOOD badge beside it (the template bands on ``g.ece < 0.05``) for
    a group whose calibration error was simply absent from the dict. A missing
    key is not a measured zero, and the |f3 filter renders None as "N/A", which
    is the only honest thing to print for a number nobody measured. NaN and inf
    are dropped to None for the same reason: they print as "nan"/"inf", which a
    reader takes for a value.
    """
    colors = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]
    curves = []
    group_metrics = report.group_metrics or {}
    # `or {}` rescues a FALSY wrong shape (None, [], "") and nothing else. A
    # NON-EMPTY list is truthy, sails straight through, and raises
    # "'list' object has no attribute 'items'" on the next line. That is the
    # same blind spot as `.get(key, {})` on a key that is present holding the
    # wrong type, and this is the second copy of it in the calibration
    # rendering path (the first was calibration_disparity_to_svg, 2026-09-09).
    # Note the inner value on the line below was ALREADY guarded with
    # isinstance, which is what made the outer one look safe.
    #
    # An unrecognised shape is COULD NOT CHECK: it is named, and then no rows
    # are produced, so the table shows nothing rather than inventing groups.
    if not isinstance(group_metrics, dict):
        warnings.warn(
            "calibration_report_to_svg: report.group_metrics is a "
            f"{type(group_metrics).__name__}, not a mapping of group to metrics, "
            "so the per-group calibration table was NOT MEASURED and is empty. "
            "CalibrationAnalyzer.full_analysis() produces a dict; check whatever "
            "built this report.",
            UserWarning,
            stacklevel=3,
        )
        group_metrics = {}
    # A KEY THAT IS THE ABSENCE OF A GROUP is not a group. `str(group)[:20]`
    # below mints a name from any of absence's six doors, and this table then
    # bands it and prints its ECE. Measured 2026-09-30 with a real report whose
    # group_metrics gained a ``None`` key: the canvas drew a row labelled "None"
    # with a real-looking ECE and a GOOD badge, and with ``pd.NA`` one labelled
    # "<NA>". Dropped, counted, and the count is named on the canvas by the
    # caller, because a row that silently disappears is the same omission as one
    # invented.
    n_missing_group_keys = sum(1 for k in group_metrics if _is_missing_level(k))
    for i, (group, metrics) in enumerate(
        (g, m) for g, m in group_metrics.items() if not _is_missing_level(g)
    ):
        m = metrics if isinstance(metrics, dict) else {}
        curves.append(
            {
                "group": str(group)[:20],
                "ece": _to_finite_float(m.get("ece")),
                "mce": _to_finite_float(m.get("mce")),
                "brier": _to_finite_float(m.get("brier")),
                "color": colors[i % len(colors)],
            }
        )
    return curves, n_missing_group_keys


def calibration_report_to_svg(
    report, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """
    Render a CalibrationReport as a polished SVG dashboard.

    Parameters
    ----------
    report : CalibrationReport
        The report returned by ``CalibrationAnalyzer.full_analysis()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.

    Notes
    -----
    Three states, never two. The canvas carries TWO independent verdicts, and
    each one is withheld on its own evidence: the CALIBRATION badge needs a
    finite overall ECE, and the disparity panel needs at least two groups whose
    calibration error was actually measured, exactly as
    ``calibration_disparity_to_svg`` requires. Neither falls back to a number.

    Raises
    ------
    TypeError
        If *report* is not a calibration report. Same reasoning as
        ``bias_audit_to_svg``: a required argument that was not supplied is a
        caller mistake with no canvas behind it, and the old failure named
        nothing (``'NoneType' object has no attribute 'overall_metrics'``).
    """
    missing = [
        attr
        for attr in ("overall_metrics", "group_metrics", "n_groups", "is_well_calibrated")
        if not hasattr(report, attr)
    ]
    if missing:
        raise TypeError(
            f"calibration_report_to_svg() needs a CalibrationReport (as returned by "
            f"CalibrationAnalyzer.full_analysis()); got {type(report).__name__}, which is "
            f"missing: {', '.join(missing)}."
        )

    overall = report.overall_metrics or {}
    disparity = report.disparity_analysis
    curves, n_missing_group_keys = _calibration_curves(report)

    # ── Could-not-check, decided HERE and not in the template ──────────────
    # DO NOT reinstate the `overall.get("ece", 0)` / `ece_disparity = ... else 0`
    # seeds below. On an empty CalibrationReport (n_groups=0, no overall
    # metrics, no group metrics) this adapter used to hand the template
    # ECE 0.000, MCE 0.000, BRIER 0.000 and an ECE disparity of 0.000, and the
    # template then printed a red MISCALIBRATED badge (from a defaulted
    # `is_well_calibrated=False`) beside a green NO DISPARITY panel reading
    # "All groups are similarly well-calibrated", across ZERO groups. One
    # canvas asserting a fabricated failure and a fabricated all-clear from the
    # same absent data, and the accessible <desc> agreed with both: "ECE 0.000
    # (miscalibrated); ECE disparity 0.000 across groups."
    # A sentinel is not a measurement, and a count of zero is not a finding of
    # zero. Same shape as reliability_diagram_to_svg in adapters_calibration.
    ece = _to_finite_float(overall.get("ece"))
    mce = _to_finite_float(overall.get("mce"))
    brier = _to_finite_float(overall.get("brier"))
    overall_assessable = ece is not None

    measured = [c for c in curves if c["ece"] is not None]
    raw_disparity = (
        _to_finite_float(getattr(disparity, "ece_disparity", None))
        if disparity is not None
        else None
    )
    # A spread needs TWO measured groups before it exists at all, and the value
    # itself has to be a real number. "Best" and "worst" are comparative and
    # need the same two: naming one group both the best and the most
    # miscalibrated compares it to nothing.
    disparity_assessable = len(measured) >= 2 and raw_disparity is not None
    ece_disparity = raw_disparity if disparity_assessable else None
    has_disparity = bool(report.has_significant_disparity) and disparity_assessable
    best_group = (
        str(getattr(disparity, "least_miscalibrated_group", "N/A"))
        if disparity_assessable
        else "N/A"
    )
    worst_group = (
        str(getattr(disparity, "most_miscalibrated_group", "N/A"))
        if disparity_assessable
        else "N/A"
    )

    overall_reason = ""
    if not overall_assessable:
        overall_reason = (
            "no overall calibration metric was supplied, so the expected calibration "
            "error was never measured."
            if not overall
            else "the overall expected calibration error came out undefined on this data."
        )

    disparity_reason = ""
    if not disparity_assessable:
        if not curves:
            disparity_reason = "no group was supplied, so no two groups were compared."
        elif not measured:
            disparity_reason = "no group's calibration error was supplied."
        elif len(measured) < 2:
            disparity_reason = "only one group's calibration error was supplied."
        else:
            disparity_reason = "no calibration disparity was supplied for these groups."

    # The canvas-level flag drives the accessible <desc> through
    # rendering.explain._not_assessable, which replaces the finding sentence
    # with a COULD NOT CHECK clause. It keys on the headline badge: an overall
    # ECE that WAS measured is still a real finding even when the groups could
    # not be compared, and the disparity panel says so on its own.
    not_assessable = not overall_assessable
    not_assessable_reason = overall_reason
    if not_assessable and not disparity_assessable:
        not_assessable_reason = f"{overall_reason} No group comparison either: {disparity_reason}"

    data = {
        "title": "CalibrationAnalyzer: Report",
        "ece": ece,
        "mce": mce,
        "brier": brier,
        # A verdict is only carried when something was measured to support it.
        #
        # THREE STATES, because the PRODUCER reports three. The field is
        # declared ``Optional[bool]`` and the analyzer sets it to None
        # deliberately, with its own comment saying why: "`nan < 0.05` is False,
        # so an ECE that could not be computed reported 'not well calibrated':
        # fail-closed, but still a verdict nobody measured". Its text summary
        # renders None through ``_yes_no_unknown`` as "Not assessable (not
        # measured)". This canvas ran it through ``bool()``, where None is
        # False, and the template's two-way branch then painted the red
        # MISCALIBRATED badge. Measured 2026-09-30 on a real report with
        # ``is_well_calibrated=None`` and a finite ECE of 0.047: canvas
        # "MISCALIBRATED", desc "ECE 0.047 (miscalibrated)" at severity MEDIUM.
        # That is a fabricated BREACH, which costs the same as a fabricated
        # all-clear: someone is sent after a miscalibration that was never
        # found.
        #
        # THE REACHABLE DOOR is a report that went through ``to_dict()`` and
        # back (which is what a task consumer and a stored artifact do): the
        # analyzer itself only emits None when the ECE is non-finite, i.e. when
        # ``overall_assessable`` is False anyway, so on a live in-process report
        # the two coincide and this branch is masked. A deserialised report
        # carries ``is_well_calibrated: null`` beside a finite ece, and this
        # adapter accepts any object with the four attributes it checks for.
        "is_well_calibrated": (
            None
            if report.is_well_calibrated is None
            else (bool(report.is_well_calibrated) and overall_assessable)
        ),
        "overall_assessable": overall_assessable,
        "overall_reason": overall_reason,
        "has_disparity": has_disparity,
        "disparity_assessable": disparity_assessable,
        "disparity_reason": disparity_reason,
        "ece_disparity": ece_disparity,
        "best_group": best_group,
        "worst_group": worst_group,
        "groups": curves,
        "n_groups": report.n_groups,
        "n_measured_groups": len(measured),
        # Carried out so a caller or a test can count them without parsing prose.
        "n_missing_group_keys": n_missing_group_keys,
        "not_assessable": not_assessable,
        "not_assessable_text": NOT_ASSESSABLE_TEXT,
        "not_assessable_reason": not_assessable_reason,
        "unknown_color": _UNKNOWN_COLOR,
        "unknown_bg": _UNKNOWN_BG,
        "protected_attribute": report.protected_attribute or "N/A",
        "issues": report.critical_issues[:4] if report.critical_issues else [],
        "recommendations": report.recommendations[:3] if report.recommendations else [],
        "recommendation_strategy": (
            report.recommendation.strategy if report.recommendation else "N/A"
        ),
        "timestamp": _fmt_ts(report.timestamp),
    }

    data["explanation"] = explanation
    svg = render_svg("calibration_report", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 3.  FairnessAnalyzer report dict  →  fairness_report.svg

_METRIC_LABELS = {
    "demographic_parity_difference": "Demographic Parity",
    "demographic_parity_ratio": "Disparate Impact",
    "equal_opportunity_difference": "Equal Opportunity",
    "equalized_odds_difference": "Equalized Odds",
    "predictive_parity_difference": "Predictive Parity",
    "calibration_difference": "Calibration Diff",
    "mae_parity_difference": "MAE Parity",
    "rmse_parity_difference": "RMSE Parity",
    "mean_prediction_difference": "Mean Pred Diff",
}

_DEFAULT_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "demographic_parity_ratio": 0.80,  # ≥ threshold
    "equal_opportunity_difference": 0.10,
    "equalized_odds_difference": 0.15,
    "predictive_parity_difference": 0.10,
    "calibration_difference": 0.05,
    "mae_parity_difference": 0.10,
    "rmse_parity_difference": 0.10,
    "mean_prediction_difference": 0.10,
}


def _metric_cards(report_dict: dict) -> List[Dict[str, Any]]:
    """Build metric card data from a FairnessAnalyzer report dict.

    Every card carries one of THREE states in ``card["state"]``: ``pass``,
    ``fail`` or ``could_not_check``. Collapsing the third into either of the
    other two is the defect this function used to embody: with no threshold
    configured, or with a threshold of 0.0 (falsy, a zero-tolerance policy),
    ``passed = abs(value) <= threshold if threshold else True`` rendered a green
    FAIR badge for a metric that was never compared to anything.
    """
    metrics = report_dict.get("metrics", {})
    thresholds = report_dict.get("thresholds_used", _DEFAULT_THRESHOLDS)
    assess = report_assessability(report_dict)
    cards = []

    for key, value in metrics.items():
        if value is None:
            continue
        label = _METRIC_LABELS.get(key, key.replace("_", " ").title())
        threshold = thresholds.get(key, _DEFAULT_THRESHOLDS.get(key))

        # For ratio metrics, "pass" means ≥ threshold. The card only uses this
        # to decide how the bound is DISPLAYED, but it is still a direction
        # question, so it goes to the shared helper and is never re-derived from
        # the name here: a local exact "_ratio" suffix test answered False for
        # `disparate_impact`, which is a ratio, and printed its four-fifths
        # floor as a ceiling.
        is_ratio = is_ratio_metric(key)
        state, reason = _metric_state(key, value, threshold)
        # A report that compared no groups makes every per-metric verdict
        # vacuous, even for a metric that still produced a finite number.
        if not assess["assessable"]:
            state = COULD_NOT_CHECK
            reason = assess["reason"]

        style = _state_style(state)
        status = "UNFAIR" if state == FAIL else style["status"]

        cards.append(
            {
                "key": key,
                "label": label,
                "value": value if _is_finite(value) else None,
                "threshold": threshold,
                "passed": state == PASS,
                "state": state,
                "reason": reason,
                "status": status,
                "color": style["color"],
                "bg": style["bg"],
                "icon": style["icon"],
                "is_ratio": is_ratio,
            }
        )

    return cards[:6]  # max 6 cards


def _group_bars(report_dict: dict) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build group comparison bar data, and name the groups it could not plot.

    Returns ``(bars, unmeasured_group_names)``. The ARITY IS DELIBERATELY
    UNCHANGED: the count of keys with no group label is recomputed by the caller
    through the same ``_is_missing_level`` predicate rather than threaded out of
    here, because tests/test_pattern_core_adapters.py unpacks this pair and a
    private helper's shape is not worth breaking for one integer. The two are
    pinned to agree in
    tests/test_bgl_grade1_g11_render_monitor_regularizers.py.

    CRITICAL, do not restore the ``, 0)`` default on the rate lookup. A group
    that reported no positive rate was plotted at 0 and labelled "0.0%" beside
    "n=0", which on this panel reads as a group the model never selects: the
    worst finding the panel can express, invented from an absent key, under a
    correct 90/100 headline that lends it credibility. Absent and zero are
    different claims.

    The unmeasured group is DROPPED from the plot rather than drawn in a third
    state, and its name is returned so the caller can put it on the canvas. The
    template renders a bar's width through the ``bar_width`` filter, which
    multiplies the rate, so a None would raise there; keeping the row would mean
    changing ``templates/fairness_report.svg``, which is outside this change.
    Dropping and NAMING follows the house pattern already used for groups
    excluded for insufficient evidence (``report_assessability``). A silently
    dropped group would be its own fabrication, so the name must not be lost.
    """
    colors = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]
    group_stats = report_dict.get("group_stats", {})
    bars = []
    unmeasured: List[str] = []
    # A GROUP KEY THAT IS THE ABSENCE OF A GROUP is not a stratum. `str(group)`
    # below mints a name from any of absence's six doors, and this panel then
    # draws it a bar and a colour. Measured 2026-09-30 with
    # ``{None: {"positive_rate": 0.4, "size": 10}, "b": {...}}``: the dashboard
    # drew a bar labelled "None" at 40% beside "n=10", and with ``pd.NA`` one
    # labelled "<NA>". The count is named on the canvas below, in the same
    # sentence as the groups that reported no rate, because a key that silently
    # disappears from a fairness dashboard is the same omission as a bar
    # invented for one.
    for i, (group, stats) in enumerate(
        (g, s) for g, s in group_stats.items() if not _is_missing_level(g)
    ):
        rate = _to_finite_float(stats.get("positive_rate", stats.get("mean_prediction", None)))
        if rate is None:
            unmeasured.append(str(group))
            continue
        bars.append(
            {
                "group": str(group)[:20],
                "rate": rate,
                # "size" is the key the engine emits (group_statistics in
                # evaluation/vfairness_metrics/classification.py). Reading only
                # "count"/"n" printed "n=0" beside every populated group; the
                # older keys stay as fallbacks for legacy report shapes.
                #
                # The last default was `0`, and "n=0" is a MEASURED count: it
                # says this stratum is empty, which is exactly what an auditor
                # looks for, and it was printed for any row carrying none of the
                # three size keys. Absent is not zero. The sibling panel
                # (adapters_fairness.group_comparison_to_svg) already ends this
                # chain in None and its template already draws
                # "n not reported"; this one was the copy that never got the
                # fix. None it is, and fairness_report.svg renders it the same
                # way.
                "n": stats.get("size", stats.get("count", stats.get("n", None))),
                "color": colors[i % len(colors)],
            }
        )
    return bars[:6], unmeasured


def fairness_report_to_svg(
    report_dict: dict, *, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """
    Render a FairnessAnalyzer report dict as a polished SVG dashboard.

    Parameters
    ----------
    report_dict : dict
        The dictionary returned by ``FairnessAnalyzer.get_report()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    assessment = report_dict.get("assessment", {})
    assess = report_assessability(report_dict)
    cards = _metric_cards(report_dict)
    bars, unmeasured_groups = _group_bars(report_dict)
    # Same predicate _group_bars filters on, so the count and the filter agree.
    n_missing_group_keys = sum(
        1 for k in (report_dict.get("group_stats") or {}) if _is_missing_level(k)
    )

    n_passed = sum(1 for c in cards if c["state"] == PASS)
    n_failed = sum(1 for c in cards if c["state"] == FAIL)
    n_unknown = sum(1 for c in cards if c["state"] == COULD_NOT_CHECK)

    # No default. The old `, 0)` was a fabricated BREACH, the direction this
    # campaign keeps being told is not the safer one: a report whose assessment
    # block simply omits `fairness_score` (a hand-built dict, an older producer)
    # took 0, which is finite, so it passed the guard below untouched and the
    # panel printed a confident "0/100" in #dc2626 under an UNFAIR band, on a
    # dashboard whose own metric cards could all be passing. `None` is what the
    # engine writes for "could not check", and it is what an absent key means
    # too, so both now land in the suppression branch below.
    # COERCED ONCE, and the arithmetic below reads the coerced number.
    # `_is_finite` accepts a numeric string deliberately (JSON and CSV rows),
    # and `raw_score <= 1` on the raw field then raised
    # ``TypeError: '<=' not supported between instances of 'str' and 'int'``
    # for a report whose assessment block had been through json.dumps, which is
    # the guard-coerces/arithmetic-does-not split that killed three of the
    # adapters_fairness canvases on the same input. `None` and an absent key
    # both still land in the suppression branch below.
    raw_score = _to_finite_float(assessment.get("fairness_score"))
    # The score is the third state's home too: the engine reports None when no
    # metric produced a threshold verdict. Rendering that as 100/100 next to the
    # report's own NOT ASSESSABLE caveat is a self-contradicting certificate, so
    # the score panel is SUPPRESSED instead (the template renders None as
    # "NOT ASSESSABLE"). `raw_score <= 1` also raised TypeError on None, which is
    # the crash this replaces.
    not_assessable = (not assess["assessable"]) or (n_passed + n_failed) == 0
    if raw_score is None or not _is_finite(raw_score) or not_assessable:
        fairness_score = None
        not_assessable = True
    else:
        # Normalise: if the score is a 0-1 float, scale to 0-100
        fairness_score = int(round(raw_score * 100)) if raw_score <= 1 else int(round(raw_score))

    summary = assessment.get("summary", "")
    if not_assessable and not summary:
        summary = assess["reason"] or (
            f"{NOT_ASSESSABLE_TEXT}: not one metric on this dashboard could be "
            f"checked, so nothing here certifies fairness."
        )
    # A group that reported no rate is not on the plot, and a group that
    # silently vanishes from a fairness dashboard is the same omission the
    # invented 0.0% bar was: the reader is left believing every stratum was
    # looked at. Name it where a reader looks, on the canvas.
    if unmeasured_groups:
        shown = ", ".join(unmeasured_groups[:4])
        head = summary.strip()
        if head and head[-1] not in ".!?":
            head += "."
        summary = (
            f"{head} " if head else ""
        ) + f"Not plotted, no rate reported: {shown}. These groups were not measured."
    # The same rule for a key that carried no group label at all. It cannot be
    # NAMED (there is no name; that is the point), so it is counted. Silently
    # removing it would leave the reader believing every stratum in the data is
    # on the panel, which is the omission the invented 0.0% bar was.
    if n_missing_group_keys:
        head = summary.strip()
        if head and head[-1] not in ".!?":
            head += "."
        summary = (f"{head} " if head else "") + (
            f"{n_missing_group_keys} group key(s) carried no group label and are not on this panel."
        )

    data = {
        "title": "FairnessAnalyzer: Metric Dashboard",
        "task_type": report_dict.get("task_type", "classification"),
        "cards": cards,
        "bars": bars,
        "unmeasured_groups": unmeasured_groups,
        "n_unmeasured_groups": len(unmeasured_groups),
        # Carried out so a caller or a test can count them without parsing prose.
        "n_missing_group_keys": n_missing_group_keys,
        "fairness_score": fairness_score,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_unknown": n_unknown,
        "not_assessable": not_assessable,
        "not_assessable_reason": assess["reason"],
        "excluded_groups": assess["excluded"],
        "summary": summary,
        "explanations": report_dict.get("explanations", {}),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    data["explanation"] = explanation
    svg = render_svg("fairness_report", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg


# 4.  CI/CD pipeline results  →  cicd_pipeline.svg


def cicd_pipeline_to_svg(
    *,
    validation_result=None,
    test_results: Optional[list] = None,
    gate_decision=None,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render a combined CI/CD pipeline report as an SVG dashboard.

    Parameters
    ----------
    validation_result : DataValidationResult, optional
        Result from ``DataBiasValidator.validate()``.
    test_results : list of TestResult, optional
        Results from ``FairnessTestSuite.test_model()``.
    gate_decision : GateDecision, optional
        Decision from ``ModelFairnessGate.evaluate()``.
    save_path : str, optional
        If provided, write the SVG to this file.

    Returns
    -------
    str
        SVG markup.
    """
    # Validation. A stage that was not supplied did not PASS, it did not RUN.
    # The old default of {"passed": True, "n_errors": 0, "n_warnings": 0}
    # rendered a green "DATA VALIDATION PASS / 0 errors, 0 warnings" for a
    # validator that never executed, and the empty issues list rendered "All
    # validation checks passed successfully" underneath it. Zero of zero is not
    # success. `state` carries the third state ("not_run") so the template
    # branches on a declared fact rather than inferring it from a count that
    # reads the same either way; `passed` is None, because False is the
    # representation of "it ran and it failed".
    val_data: Dict[str, Any] = {
        "state": "not_run",
        "passed": None,
        "issues": [],
    }
    if validation_result is not None:
        val_errors = getattr(validation_result, "errors", [])
        val_warnings = getattr(validation_result, "warnings", [])
        val_data = {
            "state": "ran",
            "passed": validation_result.passed,
            "n_errors": len(val_errors),
            "n_warnings": len(val_warnings),
            "issues": [
                {
                    "message": str(iss.message)[:60],
                    "severity": iss.severity.value
                    if hasattr(iss.severity, "value")
                    else str(iss.severity),
                }
                for iss in (val_errors + val_warnings)[:5]
            ],
        }

    # Test results. THREE states, never two. ``passed`` was
    # ``status == "passed"``, so every status that is not a pass became a
    # FAILURE: a check whose status is None, and a check the suite SKIPPED, were
    # badged FAIL in red, counted into "1 passed, 2 failed", turned the tests
    # tile red and dragged ``all_passed`` down with them. A gate that blocks on
    # a check which never ran is the same class of defect as a gate that
    # approves one, and it is the kind a reader eventually catches: they chase
    # the failure, find no such breach, and stop believing the report.
    #
    # "failed" and "error" stay failures on purpose. Both are facts about a run
    # that HAPPENED, and blocking on them is the gate doing its job. It is
    # absence, not badness, that is being separated out here.
    tests_data = []
    for t in test_results[:8] if test_results else []:
        raw_status = getattr(t, "status", None)
        status_text = getattr(raw_status, "value", raw_status)
        status_text = str(status_text).lower() if status_text is not None else None
        if status_text == "passed":
            passed = True
        elif status_text in ("failed", "error"):
            passed = False
        else:
            # None, "skipped", or a status this renderer cannot grade. Absent is
            # not zero, and a check that did not run did not fail.
            passed = None
        tests_data.append(
            {
                "name": str(t.test_name)[:45],
                "passed": passed,
                "status": status_text,
                "metric": getattr(t, "metric_name", ""),
                "value": getattr(t, "metric_value", None),
                "threshold": getattr(t, "threshold", None),
            }
        )

    # A suite that produced no test result executed no test. "0 passed, 0
    # failed" is not "ALL PASS", and neither is a table of rows that all report
    # nothing: the stage counts as run only when at least one check was graded.
    n_tests_graded = sum(1 for t in tests_data if t["passed"] is not None)
    n_tests_ungraded = len(tests_data) - n_tests_graded
    tests_state = "ran" if n_tests_graded else "not_run"

    # Gate decision
    gate_data: Dict[str, Any] = {
        "state": "not_run",
        "status": "not supplied",
        "approved": None,
        "reasons": [],
    }
    if gate_decision is not None:
        gate_data = {
            "state": "ran",
            "status": gate_decision.status.value
            if hasattr(gate_decision.status, "value")
            else str(gate_decision.status),
            "approved": gate_decision.approved,
            "reasons": [str(r)[:60] for r in (gate_decision.blocking_reasons or [])[:4]],
            "warnings": [str(w)[:60] for w in (gate_decision.warnings or [])[:4]],
        }

    # A gate report certifies the pipeline only when every stage of it actually
    # ran. A stage that did not run is named on the canvas, and until all three
    # have run the report is could-not-check: it neither approves nor blocks.
    stages_missing = [
        name
        for name, ran in (
            ("data validation", val_data["state"] == "ran"),
            ("the fairness tests", tests_state == "ran"),
            ("the fairness gate", gate_data["state"] == "ran"),
        )
        if not ran
    ]

    data = {
        "title": "CI/CD Fairness Pipeline: Gate Report",
        "timestamp": _now(),
        "validation": val_data,
        "tests": tests_data,
        "tests_state": tests_state,
        "n_tests_graded": n_tests_graded,
        "n_tests_ungraded": n_tests_ungraded,
        "gate": gate_data,
        # An ungraded check keeps ``passed`` None, which is falsy, so a pipeline
        # carrying one can never read as an all-clear. That is the rule, in this
        # direction as well: "everything passed" may only be said when the whole
        # population was graded.
        "all_passed": (
            not stages_missing
            and not n_tests_ungraded
            and bool(val_data["passed"])
            and all(t["passed"] for t in tests_data)
            and (gate_data.get("approved") is not False)
        ),
    }
    if stages_missing:
        data["not_assessable"] = True
        data["not_assessable_reason"] = f"Did not run: {', '.join(stages_missing)}."

    data["explanation"] = explanation
    svg = render_svg("cicd_pipeline", data)
    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
    return svg
