"""
SVG Rendering Adapter for Ranking Fairness Module.

Transforms RankingFairnessResult and group metrics into flat dictionaries
suitable for the ``ranking_fairness.svg`` template.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .._triage import is_flag
from ..evaluation.vfairness_metrics._metric_direction import ThresholdOutcome, check_threshold
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

_PASS = "#059669"
_FAIL = "#dc2626"
_WARN = "#f59e0b"
_UNKNOWN = "#64748b"  # slate-500, never green
_GROUP_COLORS = ["#3b82f6", "#ec4899", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"]

# Headline states. Three, never two: a ranking whose metrics all passed is not
# the same page as a ranking where nothing was graded at all.
_VERDICT_FAIR = "FAIR"
_VERDICT_UNFAIR = "UNFAIR"
_VERDICT_NOT_ASSESSED = "NOT ASSESSED"

# Badge label for a metric whose threshold was never applied: the direction of
# the metric could not be resolved, so the comparison did not happen. It is
# neither PASS nor FAIL and is drawn in the neutral slate, never in green.
_NOT_CHECKED_LABEL = "NOT CHECKED"
_NOT_CHECKED_BG = "#f1f5f9"  # slate-100

# Badge label for a metric ROW that reported nothing to grade: no value, or no
# threshold to grade it against, and no verdict of its own either. Distinct from
# NOT CHECKED, which is a row that DID report a number the resolver could not
# place a direction on. Both are ungraded and both are drawn in the neutral
# slate; only the reason on the recommendation panel differs.
_NOT_MEASURED_LABEL = "NOT MEASURED"

# Every count on this page reads the `graded` flag each badge carries, and that
# flag is set from the PASS / FAIL allow-list rather than from a deny-list of
# neutral labels: a fourth label added later is ungraded until someone decides
# otherwise, instead of silently joining the pass rate.

# There is deliberately NO stand-in metric name here any more.
#
# CRITICAL. Until 2026-09-10 an unnamed row was graded under the name
# "disparity", on the reasoning that it is a real violation-magnitude token so
# the resolver, not this module, still decided the direction. That reasoning is
# wrong in the way that matters: the resolver was answering about a name this
# module invented, not about the caller's metric. Measured on this repo,
# ``_pf(0.62, 0.80, "")`` returned "PASS" with the #059669 pass green, and the
# rendered SVG carried a green PASS badge over 0.620 -- while EVERY other
# unresolvable name (``_pf(0.62, 0.80, "mystery_metric")``, and the
# ``metric_name`` default "metric") correctly returned NOT CHECKED in slate.
# 0.62 against a 0.80 bound is the exposure-parity shape this module ships, and
# on a four-fifths floor it is a FAIL, so an empty name was the one input that
# could turn a breach into an all-clear.
#
# A missing name is missing EVIDENCE about the direction, which is exactly the
# could-not-check state. It is passed through to the resolver unchanged, where
# an empty name already normalises to UNKNOWN.


def _num(value: Any) -> Optional[float]:
    """A number this page may DRAW, or None when the row reported none.

    CRITICAL, three states, never two. The group panels below print exposures,
    positions and counts, and every one of them used to be read through a
    ``.get(key, 0)``. Zero is not a neutral filler on any of those scales: zero
    exposure is a group shut out of the ranking entirely, and it is the single
    worst reading that panel can produce. A default is not a measurement.

    A bool is refused because ``True`` is an ``int`` in Python and would print
    as an exposure of 1.0, the best value on that scale. NaN and the infinities
    are refused because they are sentinels rather than measurements, and because
    an infinite exposure divides through the bar-width arithmetic below.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _count(value: Any) -> Optional[int]:
    """A reported whole-number count, or None when the row reported none.

    Same rule as :func:`_num`, and for the same reason: "n=0" on the group panel
    reads as a group with no members in the ranking, which is a finding, not a
    blank.
    """
    num = _num(value)
    return int(num) if num is not None else None


def _gradable(value: Any) -> Optional[float]:
    """The number the badge loop may GRADE, or None when the row reported none.

    Deliberately NOT ``_num``: this page fails closed on a NaN or infinite
    metric value (pinned by ``test_adapters_ranking_verdicts.py::
    test_an_unmeasurable_metric_fails_closed``), so a non-finite number has to
    survive coercion and be recognised as non-finite further down. ``_num``
    would fold it into None, i.e. into the neutral not-measured state, which is
    the one relaxation that pin forbids.

    What it does refuse is what ``float()`` alone accepted. G13, 2026-09-30, on
    the badge loop, which read ``float(raw) if raw is not None``:

    * ``value=True`` became 1.0 and rendered a GREEN PASS badge for
      ``exposure_parity_ratio`` against its 0.8 floor -- 1.0 is the best value
      on that scale, and the row carried no measurement at all. ``_num``'s own
      docstring states this rule ten lines up; the badge loop bypassed it.
      ``is_flag`` catches ``np.bool_`` too, which a DataFrame column gives.
    * ``value="N/A"`` raised ValueError and took the WHOLE export down. One
      unparseable cell in one row is a not-measured row, not a dead page.
    """
    if value is None or is_flag(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _outcome(val: float, threshold: float, metric_name: str) -> ThresholdOutcome:
    """Grade one value against one bound IN THAT METRIC'S OWN DIRECTION.

    The direction is resolved by
    :mod:`vfairness.evaluation.vfairness_metrics._metric_direction`, never by a
    name test written here. Until 2026-08-27 this module compared every metric
    as ``val <= threshold``, i.e. it assumed lower is always better. That
    inverts the verdict for the whole ratio family, and the ranking module ships
    one: ``exposure_parity_ratio`` is a four-fifths-style floor where HIGHER is
    better. Executed on this repo,
    ``check_threshold("exposure_parity_ratio", 0.62, 0.80)`` returned FAIL
    ("is below the required minimum") while this adapter rendered a green PASS
    for the same two numbers, so a group receiving 62 percent of the exposure of
    the best-served group was certified as fair on the canvas an auditor reads.
    That is the fourth appearance of one bug class, and every appearance was a
    SECOND COPY of a rule that already existed. Do not reintroduce a local copy
    here, however small: extend ``_metric_direction``, where the precedence
    table is pinned by ``tests/test_metric_direction.py``.
    """
    outcome, _ = check_threshold(metric_name, val, threshold)
    return outcome


def _pf(val: float, threshold: float, metric_name: str = "") -> str:
    """PASS / FAIL / NOT CHECKED for one metric against its threshold."""
    outcome = _outcome(val, threshold, metric_name)
    if outcome is ThresholdOutcome.PASS:
        return "PASS"
    if outcome is ThresholdOutcome.FAIL:
        return "FAIL"
    return _NOT_CHECKED_LABEL


def _pf_color(val: float, threshold: float, metric_name: str = "") -> str:
    """The badge colour for the same three states. Could-not-check is NEVER green."""
    outcome = _outcome(val, threshold, metric_name)
    if outcome is ThresholdOutcome.PASS:
        return _PASS
    if outcome is ThresholdOutcome.FAIL:
        return _FAIL
    return _UNKNOWN


def _headline(n_metrics: int, n_fail: int, n_unchecked: int = 0) -> Dict[str, str]:
    """Grade the page headline into one of THREE states.

    The headline used to be decided in the template as ``n_pass == n_total``,
    and that comparison is ALSO true when nothing was measured. A call carrying
    group exposures with a 2:1 gap and an empty results list rendered a green
    FAIR out of ``0 == 0``: a ranking nobody graded, presented as a ranking that
    passed. Nothing measured is not a pass, so it gets its own state and says so
    on the page rather than borrowing either verdict.

    ``n_unchecked`` closes the same hole one metric at a time. A measured breach
    is a finding whatever else is missing, so UNFAIR still wins; but FAIR is an
    all-clear over the WHOLE page, and it is only claimable when every metric on
    it was actually compared to its threshold.
    """
    if n_metrics <= 0:
        return {
            "label": _VERDICT_NOT_ASSESSED,
            "color": _UNKNOWN,
            "caption": "no metric computed",
            "font_size": "13",
        }
    if n_fail:
        return {
            "label": _VERDICT_UNFAIR,
            "color": _FAIL,
            "caption": "",
            "font_size": "20",
        }
    if n_unchecked:
        return {
            "label": _VERDICT_NOT_ASSESSED,
            "color": _UNKNOWN,
            "caption": f"{n_unchecked} of {n_metrics} metric(s) not graded",
            "font_size": "13",
        }
    return {
        "label": _VERDICT_FAIR,
        "color": _PASS,
        "caption": "",
        "font_size": "20",
    }


#  ranking_fairness.svg


def ranking_fairness_to_svg(
    results: Optional[List] = None,
    group_metrics: Optional[Dict] = None,
    *,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render ranking fairness results as an SVG dashboard.

    Parameters
    ----------
    results : list of RankingFairnessResult or dicts
        Each with metric_name, value, is_fair, threshold.
    group_metrics : dict
        Per-group ranking metrics from get_ranking_group_metrics().
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
    CRITICAL, do not restore the old default. Until 2026-08-27 a call carrying
    no data at all dispatched to ``_demo`` and this function returned a
    complete, confident ranking assessment: three named metrics over four named
    groups, two PASS badges, one FAIL, a ``2/3 metrics pass`` rate, an UNFAIR
    headline and the line "NDKL exceeds threshold (0.12 > 0.10)". None of it was
    measured from anything the caller supplied, and a caller could not tell that
    output apart from a real evaluation. An SVG is an export format: it leaves
    the building, an auditor reads it, and it outlives the version that produced
    it.

    Missing input is now routed through the SAME could-not-check path that
    ``ranking_fairness_to_svg([], {})`` already took, which withholds the rate,
    prints NOT ASSESSED on the badge and says on the canvas that no metric was
    supplied. Deliberately not a second could-not-check canvas of its own: two
    codepaths saying the same thing is how one of them drifts. The demo fixture
    survives only behind the explicit ``example=True``, watermarked.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    if example and results is None and group_metrics is None:
        return _demo(explanation=explanation, save_path=save_path)

    # Metric badges
    #
    # CRITICAL, per ROW. Until 2026-08-27 this loop read the two numbers it
    # grades out of `d.get("value", 0)` and `d.get("threshold", 0.1)`, so a row
    # carrying ONLY a metric_name was graded against 0 versus an invented 0.1
    # bound. Executed on this repo, one row `{"metric_name": "ndkl"}` rendered a
    # green PASS badge reading 0.000, a green OVERALL FAIR headline, "1/1
    # metrics pass" and the sentence "All ranking fairness metrics pass their
    # configured thresholds": a complete all-clear produced from a row that
    # reported nothing. Confirmed identically for exposure_parity_difference and
    # ndcg_parity_difference, and for a higher-is-better name
    # (exposure_parity_ratio) the same defaults rendered a fabricated FAIL, which
    # is equally invented.
    #
    # A default is not a measurement and a sentinel is not a measurement. A row
    # with no value gets no number, no badge colour and no place in the pass
    # rate, and it holds the headline back from the all-clear through
    # `_headline(..., n_unchecked=...)`.
    badges: List[Dict[str, Any]] = []
    for r in results or []:
        d = r if isinstance(r, dict) else _obj_to_dict(r)
        # See the CRITICAL note above on stand-in metric names: an absent name is
        # passed to the resolver as the "metric" stand-in, which normalises to
        # UNKNOWN. A key PRESENT holding None skipped that default and reached
        # `str(None)`, so the badge read "NONE". G13 2026-09-30.
        raw_name = d.get("metric_name")
        name = str(raw_name) if raw_name is not None else "metric"
        raw_val = d.get("value")
        raw_thr = d.get("threshold")
        val = _gradable(raw_val)
        thr = _gradable(raw_thr)
        verdict_reported = d.get("is_fair")
        if verdict_reported is not None:
            # The metric module's own verdict wins: it knows things a name and
            # two numbers cannot (pinned by test_adapters_ranking_verdicts.py::
            # test_the_verdict_follows_is_fair_when_the_caller_supplies_it). A
            # reported verdict is a measurement even when the value beside it is
            # missing, so the badge keeps the verdict and withholds the number.
            label = "PASS" if verdict_reported else "FAIL"
        elif "is_fair" in d:
            # CRITICAL: is_fair PRESENT and holding None is the metric module
            # DECLINING to grade, and it is not the same as is_fair absent. The
            # ranking metrics returned is_fair=True whenever fewer than two
            # groups met min_group_size, and this branch turned that default
            # into a green PASS badge on an exported SVG: on the incident data
            # (12 items, group B holding the bottom four slots and receiving 12
            # percent of the attention) the page read FAIR, 1/1 metrics pass.
            # The metrics answer None there now, and None is could-not-check: it
            # gets the ungraded label, the neutral slate, no place in the pass
            # rate, and it holds the headline back from the all-clear.
            #
            # Read with `in`, not with `.get(..., default)`: a key that is
            # PRESENT holding None does not trigger a get-default, so a default
            # here would silently rejoin the graded path.
            label = _NOT_MEASURED_LABEL
        elif val is None or thr is None:
            # Nothing to grade: no value, or no bound to grade it against. The
            # 0 / 0.1 pair that used to stand in here is where the false
            # all-clear came from.
            label = _NOT_MEASURED_LABEL
        else:
            label = _pf(val, thr, name)
            if label == _NOT_CHECKED_LABEL and not math.isfinite(val):
                # An unmeasurable VALUE (NaN, inf) is a different failure from an
                # unresolvable DIRECTION: the metric was requested and could not
                # be produced, and this page has always failed closed on it.
                # Pinned by test_adapters_ranking_verdicts.py::
                # test_an_unmeasurable_metric_fails_closed. Do NOT relax it into
                # the neutral state; see the note in issues for the third-state
                # follow-up on this row.
                label = "FAIL"
        color = {"PASS": _PASS, "FAIL": _FAIL}.get(label, _UNKNOWN)
        badges.append(
            {
                "name": name,
                "value": val,
                "threshold": thr,
                "label": label,
                "color": color,
                "bg": {"PASS": "#ecfdf5", "FAIL": "#fef2f2"}.get(label, _NOT_CHECKED_BG),
                # `measured` gates the NUMBER on the badge, `graded` gates the
                # VERDICT and every count taken over the badges. They are
                # independent: a caller-supplied is_fair with no value is graded
                # but not measured.
                #
                # Non-finite is not measured, by the same rule _num already
                # states for the group panel: NaN and the infinities are
                # sentinels, not readings. This half of the page bypassed _num,
                # so a NaN value printed the literal "nan" as its number, and
                # after the is_fair=None fix above it would have printed that
                # "nan" directly underneath a NOT MEASURED label.
                "measured": val is not None and math.isfinite(val),
                "graded": label in ("PASS", "FAIL"),
            }
        )

    # Group exposure bars
    #
    # CRITICAL, per ROW. This is the SAME defect the badge loop above was
    # hardened against, in a block that sits directly beneath it and was left
    # carrying the old defaults. Until 2026-08-28 every cell here was read
    # through `.get(key, 0)`, so a group entry reporting only its NAME rendered
    # a complete measured row: "n=0", a drawn exposure bar, "0.000" in the
    # group's own colour, and 0.0 / 0.0 / 0 / 0 / 0 across the position table.
    # Executed on this repo, `{"Female": {}}` beside a measured `{"Male":
    # {"mean_exposure": 0.28, ...}}` drew Female at exposure 0.000 with n=0 and
    # then tripped the `wide_gap` test (0.28 / 1 > 2 is False, but the pair
    # 0.28 / 0.0 divides through the `or 1` guard), so the recommendation panel
    # advised a re-ranking strategy over a gap invented from an absent number.
    # Zero exposure is not a neutral filler: it is a group shut out of the
    # ranking entirely, which is the worst finding this panel can carry.
    #
    # A default is not a measurement. A group that reported no exposure gets no
    # number, no bar and no place in the max-exposure scale or the gap test, and
    # it is counted into the subtitle so the reader can see the population the
    # panel actually measured. See _num and _count.
    groups: List[Dict[str, Any]] = []
    if group_metrics:
        reported = []
        for gm in group_metrics.values():
            e = _num(gm.get("mean_exposure", gm.get("exposure")))
            if e is not None:
                reported.append(e)
        # Scaled over the REPORTED exposures only: a defaulted 0 in here cannot
        # move the maximum, but a defaulted value on any other scale could, and
        # this list is what every bar width on the panel is measured against.
        max_exp = max(reported, default=0) or 1
        for i, (name, gm) in enumerate(sorted(group_metrics.items())):
            exp = _num(gm.get("mean_exposure", gm.get("exposure")))
            groups.append(
                {
                    "name": str(name),
                    "exposure": exp,
                    # `measured` gates the bar AND the number, exactly as it does
                    # on the metric badges above. The position cells carry their
                    # own None and the template says so per cell, because a group
                    # may report an exposure with no positions beside it.
                    "measured": exp is not None,
                    "bar_w": max(4, int(exp / max_exp * 260)) if exp is not None else 0,
                    "color": _GROUP_COLORS[i % len(_GROUP_COLORS)],
                    "avg_position": _num(gm.get("avg_position")),
                    "median_position": _num(gm.get("median_position")),
                    "min_position": _count(gm.get("min_position")),
                    "max_position": _count(gm.get("max_position")),
                    "count": _count(gm.get("count", gm.get("size"))),
                }
            )

    # THE COUNT, and the decided headline rule (a) and (d).
    #
    # Wave 12 fixed the row STATE here and left the counts alone, so a page with
    # one measured breach and six rows that reported nothing still printed
    # "UNFAIR 0/7 metrics pass". Every one of those six sat in the DENOMINATOR
    # of a rate whose subject is passing a threshold, which says they were
    # compared to one and did not pass. They were never compared to anything.
    #
    # The rate is taken over the GRADED subset, so a partially-measured run still
    # gives a useful verdict on what WAS measured (rule a), and an ungraded row
    # enters neither the numerator nor the denominator (rule d). The count of
    # ungraded rows is not dropped, it is printed beside the headline on the
    # canvas (rule c), because withholding it would hide the very fact that the
    # denominator shrank.
    n_pass = sum(1 for b in badges if b["label"] == "PASS")
    n_graded = sum(1 for b in badges if b["graded"])
    failing = [b for b in badges if b["label"] == "FAIL"]
    # Both ungraded states, read off `graded` rather than off one label, so a row
    # that reported nothing cannot slip back into the pass rate or the headline
    # by being given a new neutral label later.
    unchecked = [b for b in badges if not b["graded"]]
    not_checked = [b for b in badges if b["label"] == _NOT_CHECKED_LABEL]
    not_measured = [b for b in badges if b["label"] == _NOT_MEASURED_LABEL]
    # The exposure gap is a FINDING, so it is taken over the groups that
    # reported an exposure and no others. A group with no exposure used to enter
    # this ratio as a defaulted 0, and `or 1` then turned that absence into a
    # denominator of 1, so any group above 2.0 raised a gap warning against a
    # number nobody measured. Withheld rows enter neither side.
    measured_exposures = [g["exposure"] for g in groups if g["measured"]]
    ungraded_groups = [g for g in groups if not g["measured"]]
    # CRITICAL, and the SECOND half of the same line. The wave that added the
    # `measured` gate above fixed the UNMEASURED half of `min(...) or 1` and
    # left the MEASURED half live, because `0.0 or 1` cannot tell an absent
    # exposure from a reported one. A group whose exposure is a MEASURED 0.0 is
    # shut out of the ranking entirely, which _num calls "the single worst
    # reading that panel can produce", and it was still handed a denominator of
    # 1: the ratio collapsed to `max`, `wide_gap` went False, and the panel
    # printed "Measured exposure is well-distributed across the groups shown".
    # Executed on this repo before the fix, one metric row and three groups:
    #
    #   {A: 1.8, B: 1.2, C: 0.0}  ->  "Measured exposure is well-distributed..."
    #   {A: 1.8, B: 1.2, C: 0.5}  ->  "Large exposure gap between groups..."
    #
    # The MORE severe input produced the MORE reassuring sentence. A zero
    # denominator is not a one, it is an UNBOUNDED ratio, so the shut-out rows
    # are named in their own right and carry the strongest exposure finding this
    # panel has. `_spread` is Optional now, because a page whose measured
    # exposures are all zero has no ratio at all (0/0), which is a third state
    # and not a small gap: `shut_out` is what reports that page.
    shut_out = [g for g in groups if g["measured"] and g["exposure"] == 0.0]
    _spread: Optional[float] = None
    if len(measured_exposures) > 1:
        _lo, _hi = min(measured_exposures), max(measured_exposures)
        if _lo != 0.0:
            _spread = _hi / _lo
        elif _hi > 0.0:
            # Unbounded, not undefined: one group received exposure and another
            # received none at all. No truthiness test decides this, because a
            # truthiness test is what produced the defect.
            _spread = math.inf
        # else every measured exposure is 0.0: the ratio is 0/0 and does not
        # exist. It stays None, and `shut_out` is what reports that page.
    wide_gap = _spread is not None and _spread > 2

    # The recommendations carried the same 0 == 0 claim as the headline: with no
    # metric results the panel printed "All ranking fairness metrics pass". The
    # metric verdict and the exposure observation are separate sentences now, so
    # a descriptive exposure reading is never dressed up as a passed metric.
    recs: List[str] = []
    if not badges:
        recs.append(
            "No ranking fairness metric was supplied, so no metric verdict was graded here."
        )
    if failing:
        recs.append(f"{len(failing)} metric(s) fail threshold, review the ranking algorithm.")
    if not_checked:
        recs.append(
            f"{len(not_checked)} metric(s) could not be checked "
            f"({', '.join(b['name'] for b in not_checked)}); this report certifies nothing "
            f"about them."
        )
    if not_measured:
        recs.append(
            f"{len(not_measured)} metric(s) reported no value to grade "
            f"({', '.join(b['name'] for b in not_measured)}); they were not measured and "
            f"this report certifies nothing about them."
        )
    if ungraded_groups:
        recs.append(
            f"{len(ungraded_groups)} group(s) reported no exposure "
            f"({', '.join(g['name'] for g in ungraded_groups)}); they are drawn without a "
            f"bar and are excluded from the exposure gap."
        )
    if shut_out:
        # The strongest exposure finding this panel has, and it supersedes the
        # milder "large gap" sentence rather than sitting beside it: a group
        # that received NONE of the exposure is not a group at the far end of a
        # ratio, and recs is truncated to four, so the strongest reading must
        # not be the one that falls off the end.
        #
        # The ratio clause is read off `_spread` and not asserted: it is
        # unbounded only when some OTHER group did receive exposure. When every
        # measured exposure is 0.0 there is no ratio to take, and calling that
        # one "unbounded" would invent a second number on the same panel.
        _ratio_clause = (
            "the exposure ratio against them is unbounded"
            if _spread is not None and math.isinf(_spread)
            else "no group that reported an exposure received any, so no ratio can be taken"
        )
        recs.append(
            f"{len(shut_out)} group(s) received ZERO measured exposure "
            f"({', '.join(g['name'] for g in shut_out)}); they are shut out of the ranking "
            f"entirely, {_ratio_clause}, and re-ranking is required."
        )
    elif wide_gap:
        recs.append("Large exposure gap between groups, consider a re-ranking strategy.")
    elif measured_exposures:
        # "well-distributed across the groups shown" is an all-clear over the
        # whole panel, and the panel now shows rows that reported nothing, so
        # the claim is narrowed to the groups it is actually made from.
        recs.append(
            "Measured exposure is well-distributed across the groups shown."
            if not ungraded_groups
            else f"Measured exposure is well-distributed across the "
            f"{len(measured_exposures)} group(s) that reported one."
        )
    if badges and not failing and not unchecked:
        recs.append("All ranking fairness metrics pass their configured thresholds.")

    # ONE state feeds the headline, the recommendation panel and the accessible
    # <desc>. `not_assessable` is what rendering.explain._fr_ranking reads, so
    # the paragraph beside the badge cannot say "all pass" while the badge says
    # NOT ASSESSED. See the note on rendering/explain.build_explanation.
    verdict = _headline(len(badges), len(failing), len(unchecked))
    not_assessable = verdict["label"] == _VERDICT_NOT_ASSESSED

    # "0 metrics · 0 groups" is a count, and a count reads as a measurement. A
    # page that graded nothing has no counts to print, so the subtitle says what
    # happened instead. A page that graded SOME metrics keeps its real counts:
    # the unchecked ones are named in the recommendation panel.
    subtitle = f"{len(badges)} metrics · {len(groups)} groups"
    if ungraded_groups:
        # Rule (c) for the group half of the page: "4 groups" is a count, and a
        # count reads as four measured groups. The metric half prints its
        # ungraded count at y=136; this one goes in the subtitle at y=122, which
        # is the only header line the group panel controls. Both sit inside the
        # headline band, and a fully reported run keeps the subtitle it has
        # always had.
        subtitle += f" ({len(ungraded_groups)} reported no exposure)"
    if not badges and not groups:
        subtitle = "Nothing was assessed: no ranking metric and no group was supplied"

    template_data = {
        "title": "Ranking Fairness Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": subtitle,
        "badges": badges[:6],
        "groups": groups[:8],
        "n_pass": n_pass,
        # The denominator is the GRADED population, never every supplied row.
        # `rendering.explain._fr_ranking` writes the same two numbers into the
        # accessible <desc> off these keys, so the canvas and the description
        # cannot disagree about what was counted.
        "n_total": n_graded,
        # Printed on the canvas beside the headline: rule (c). Withholding it
        # would leave a reader with a shrunken denominator and no way to see it.
        "n_ungraded": len(unchecked),
        "n_supplied": len(badges),
        "verdict": verdict,
        "not_assessable": not_assessable,
        "not_assessable_reason": (
            f"{len(not_measured)} of {len(badges)} ranking metric(s) reported no value, "
            f"so they were not graded."
            if not_measured and not not_checked
            else (
                f"{len(unchecked)} of {len(badges)} ranking metric(s) could not be checked."
                if unchecked
                else ("no ranking fairness metric was supplied." if not badges else "")
            )
        ),
        "recommendations": recs[:4],
        "explanation": explanation,
    }

    return _render_and_save("ranking_fairness", template_data, save_path)


def _obj_to_dict(obj) -> dict:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {}


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
    # `measured` is required by the template: a badge without it renders "no
    # value reported" instead of its number. Every value below is invented, so
    # this says only that the FIXTURE states a value, never that anything was
    # measured; the EXAMPLE band and watermark are what carry that.
    badges = [
        {
            "name": "Exposure Parity",
            "value": 0.08,
            "threshold": 0.10,
            "label": "PASS",
            "color": _PASS,
            "bg": "#ecfdf5",
            "measured": True,
            "graded": True,
        },
        {
            "name": "NDKL",
            "value": 0.12,
            "threshold": 0.10,
            "label": "FAIL",
            "color": _FAIL,
            "bg": "#fef2f2",
            "measured": True,
            "graded": True,
        },
        {
            "name": "Attention Fairness",
            "value": 0.05,
            "threshold": 0.10,
            "label": "PASS",
            "color": _PASS,
            "bg": "#ecfdf5",
            "measured": True,
            "graded": True,
        },
    ]
    groups = [
        {
            "name": "Male",
            "exposure": 0.28,
            "measured": True,
            "bar_w": 260,
            "color": "#3b82f6",
            "avg_position": 4.2,
            "median_position": 3.0,
            "min_position": 1,
            "max_position": 15,
            "count": 450,
        },
        {
            "name": "Female",
            "exposure": 0.24,
            "measured": True,
            "bar_w": 223,
            "color": "#ec4899",
            "avg_position": 5.1,
            "median_position": 4.0,
            "min_position": 1,
            "max_position": 18,
            "count": 380,
        },
        {
            "name": "Non-binary",
            "exposure": 0.26,
            "measured": True,
            "bar_w": 241,
            "color": "#f59e0b",
            "avg_position": 4.8,
            "median_position": 4.0,
            "min_position": 1,
            "max_position": 12,
            "count": 120,
        },
        {
            "name": "Other",
            "exposure": 0.22,
            "measured": True,
            "bar_w": 204,
            "color": "#10b981",
            "avg_position": 5.5,
            "median_position": 5.0,
            "min_position": 2,
            "max_position": 20,
            "count": 50,
        },
    ]
    recs = [
        "NDKL exceeds threshold (0.12 > 0.10), review the ranking distribution.",
        "Exposure parity and attention fairness pass, group coverage is balanced.",
    ]

    template_data = {
        "title": "Ranking Fairness Report",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "subtitle": "3 metrics · 4 groups",
        # DO NOT REMOVE. Every metric, group and count below is invented. This
        # flag is what puts the EXAMPLE band and watermark on the canvas and the
        # EXAMPLE ONLY marker at the head of the accessible description, and it
        # is the only thing that stops this render being read as a finding about
        # a real ranking.
        "is_example": True,
        # The demo shows a measured, failing report: NDKL breaches its
        # threshold, so the headline is a finding and not a withheld verdict.
        "not_assessable": False,
        "not_assessable_reason": "",
        "badges": badges,
        "groups": groups,
        "n_pass": 2,
        # All three fixture badges are graded, so the graded denominator is 3
        # and nothing is withheld from the count.
        "n_total": 3,
        "n_ungraded": 0,
        "n_supplied": 3,
        "verdict": _headline(3, 1),
        "recommendations": recs,
        "explanation": explanation,
    }

    return _render_and_save("ranking_fairness", template_data, save_path)
