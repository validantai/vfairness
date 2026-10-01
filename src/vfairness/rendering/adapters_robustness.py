"""
SVG Rendering Adapter for Robustness Testing Module.

Transforms PermutationTestResult, SensitivityResult, and SubgroupAuditResult
into flat dictionaries suitable for the ``robustness_testing.svg`` template.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from .._triage import is_flag
from .engine import JINJA2_AVAILABLE, raise_jinja2_missing, render_svg

# colour helpers
_PASS = "#059669"
_WARN = "#f59e0b"
_FAIL = "#dc2626"
#: slate-500. The colour of a state that is neither a pass nor a failure. It is
#: deliberately outside the three-colour verdict scale above, so a reader cannot
#: take a grade off a canvas that graded nothing.
_UNKNOWN = "#64748b"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return _PASS
    if score >= 0.5:
        return _WARN
    return _FAIL


def _score_label(score: float) -> str:
    if score >= 0.8:
        return "PASS"
    if score >= 0.5:
        return "MARGINAL"
    return "FAIL"


#  robustness_testing.svg


def robustness_testing_to_svg(
    permutation_results: Optional[List] = None,
    sensitivity_results: Optional[List] = None,
    subgroup_audit: Optional[Any] = None,
    *,
    example: bool = False,
    explanation: Optional[str] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Render robustness testing results as an SVG dashboard.

    Parameters
    ----------
    permutation_results : list of PermutationTestResult or dicts
    sensitivity_results : list of SensitivityResult or dicts
    subgroup_audit : SubgroupAuditResult or dict
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
    CRITICAL, do not restore the old defaults. Until 2026-08-27 this function
    had two ways of certifying a model nobody had tested.

    ``robustness_testing_to_svg(None)`` silently dispatched to ``_demo`` and
    returned a complete dashboard: verdict ROBUST, score 0.72, three named
    permutation tests with p-values, three sensitivity bars and a six-subgroup
    audit, none of it measured from anything the caller supplied. A caller could
    not tell that output apart from a real evaluation, and an SVG is an export
    format: it leaves the building, an auditor reads it, and it outlives the
    version that produced it.

    ``robustness_testing_to_svg([], [], None)`` was worse, because it looked
    like it had run: the canvas said "0 permutation tests, 0 sensitivity checks,
    0 subgroups" and still printed a banded MARGINAL verdict with "score: 0.50"
    beside it, over a Recommendations panel reading "All robustness checks
    passed". The 0.50 was the neutral SENTINEL chosen for the no-scored-test
    branch, not a measurement, and the amber band gave it a grade.

    Nothing tested is COULD NOT CHECK, and it is stated on the canvas. A score
    is printed only when a permutation test or a sensitivity check was actually
    supplied to compute it from. The demo fixture survives only behind the
    explicit ``example=True``, watermarked.
    """
    if not JINJA2_AVAILABLE:
        raise_jinja2_missing()

    if permutation_results is None and sensitivity_results is None and subgroup_audit is None:
        if example:
            return _demo(explanation=explanation, save_path=save_path)
        return _not_tested(
            reason=(
                "No permutation test, no sensitivity check and no subgroup audit was "
                "supplied to robustness_testing_to_svg, so nothing was tested on this run."
            ),
            explanation=explanation,
            save_path=save_path,
        )

    # Permutation tests
    #
    # CRITICAL, per ROW. Until 2026-08-27 the significance of a row was derived
    # from `d.get("p_value", 1)`, and the p-value cell printed the same default.
    # A test that reported no p-value therefore rendered "P-VALUE 1.0000" and a
    # green STABLE badge: 1.0 is the most reassuring p-value obtainable and it
    # was a SENTINEL, not a measurement. Executed on this repo, the single row
    # `{"method": "demographic_parity"}` rendered STABLE at 1.0000 AND carried
    # that invented stability into the overall score, printing "ROBUSTNESS PASS,
    # score 1.00" for a run in which nothing was measured.
    #
    # A row with no p-value and no verdict of its own reports nothing: it gets no
    # number, no badge colour, and no place in the stability count or the score.
    perm_rows: List[Dict[str, Any]] = []
    for pr in permutation_results or []:
        d = pr if isinstance(pr, dict) else _obj_to_dict(pr)
        raw_p = d.get("p_value")
        # READINESS-6, 2026-09-10. `raw_p is not None` lets a NaN through, and
        # NaN is how this library says a test REFUSED to run: robustness.py's
        # permutation_test answers p=nan when the null collapsed. It was then
        # graded by the `elif p_value is not None` branch below and counted as a
        # PASS in the 0.4-weighted permutation component, because `nan < 0.05`
        # is False. The comment fifteen lines above says this module exists to
        # stop exactly that; the absent-p path was closed and the NaN path was
        # not.
        #
        # Measured that day: a single refused permutation row rendered
        # "stability score 1.00", a perfect score for a test that never ran, and
        # a refused row beside a real breach rendered 0.50, halving the visible
        # severity.
        p_value = _finite_or_none(raw_p)
        raw_sig = d.get("significant_at_05")
        if raw_sig is not None:
            # The test's own verdict wins, exactly as is_fair does on the ranking
            # canvas: it may know things a p-value alone does not (a correction
            # for multiplicity, a one-sided alternative).
            sig, graded = bool(raw_sig), True
        elif p_value is not None:
            sig, graded = p_value < 0.05, True
        else:
            # Not significant and not stable: ungraded. `significant` stays False
            # only so the red finding band is not painted; `graded` is what the
            # template and every count below read.
            sig, graded = False, False
        raw_obs = d.get("observed_statistic")
        # A name is not a measurement, but `str(None)` MINTS one: `.get(key, default)`
        # does not fire when the key is PRESENT holding None, which is what a producer
        # dataclass with an Optional field defaulting to None hands this loop, so the
        # canvas printed the literal word "None" where a name belongs. G13 2026-09-30.
        perm_rows.append(
            {
                "method": str(d.get("method") or "permutation"),
                # `raw_obs is not None` is not a measurement test: a NaN is not
                # None. `_finite_or_none` is the predicate the p-value one line
                # below already uses, and it also refuses a bool and an
                # unparseable cell rather than raising. G13 2026-09-30.
                "observed": _finite_or_none(raw_obs),
                "p_value": p_value,
                "significant": sig,
                "graded": graded,
                "direction": str(d.get("effect_direction") or "n/a"),
            }
        )

    # Sensitivity bars
    #
    # CRITICAL, per ROW, and it is the FABRICATED-BREACH direction of the same
    # defect the permutation loop above carries. Until 2026-08-28 the score was
    # read as `d.get("robustness_score", 0)`, and 0 is the WORST score this scale
    # has: a perturbation that reported nothing rendered a red WEAK badge at
    # 0.00 with an empty bar and "max dev: 0.000", and then that invented
    # weakness was averaged into the 0.6-weighted sensitivity component and
    # dragged the whole dashboard down. Executed on this repo, one measured
    # perturbation at 0.92 beside the bare row `{"perturbation_type":
    # "feature_dropout_10%"}` rendered ROBUSTNESS MARGINAL at 0.68, and a second
    # bare row takes it to FAIL. That sends a reader after a fragility that was
    # never measured, which discredits the tool exactly as fast as a false
    # all-clear does.
    #
    # A row with no score and no robustness verdict of its own reports nothing:
    # no number, no bar, no badge colour, and no place in the score.
    sens_rows: List[Dict[str, Any]] = []
    for sr in sensitivity_results or []:
        d = sr if isinstance(sr, dict) else _obj_to_dict(sr)
        raw_score = d.get("robustness_score")
        # G13 2026-09-30. This was `float(raw_score) if raw_score is not None`,
        # and `_finite_or_none` -- the guard written in THIS file for exactly
        # this, and used by the permutation loop above -- was not applied to the
        # score. Three measured consequences, all on the cell the headline is
        # built from:
        #   * `robustness_score=np.True_` (a boolean column read out of a
        #     DataFrame, the arrival path `_finite_or_none`'s own docstring
        #     names) became 1.0 and rendered ROBUST / PASS at score 1.00 -- the
        #     "score 1.00 for a run in which nothing was measured" that the
        #     CRITICAL note above says this module exists to prevent;
        #   * `robustness_score=nan` raised ValueError out of `int(score * 200)`
        #     in the bar width, so a refused check took the whole export down;
        #   * a text cell ("x", "n/a") raised ValueError the same way.
        # NaN is how this library says a check REFUSED; it is ungraded here now,
        # exactly as the refused p-value is.
        score = _finite_or_none(raw_score)
        raw_robust = d.get("is_robust")
        if raw_robust is not None:
            # The check's own verdict wins, exactly as `significant_at_05` does
            # on the permutation table and `is_fair` does on the ranking canvas.
            # A reported verdict is a measurement even when no score came with
            # it, so the badge keeps the verdict and withholds the number.
            is_robust, graded = bool(raw_robust), True
        elif score is not None:
            is_robust, graded = score >= 0.8, True
        else:
            # Ungraded. `is_robust` stays False only so no green ROBUST chip is
            # painted; `graded` is what the template and every count below read.
            is_robust, graded = False, False
        raw_dev = d.get("max_deviation")
        sens_rows.append(
            {
                "type": str(d.get("perturbation_type") or "unknown"),
                "score": score,
                "bar_w": max(4, min(200, int(score * 200))) if score is not None else 0,
                "color": _score_color(score) if score is not None else _UNKNOWN,
                "is_robust": is_robust,
                # `measured` gates the NUMBER and the bar, `graded` gates the
                # VERDICT badge and the score. They are independent: a supplied
                # is_robust with no score is graded but not measured.
                "measured": score is not None,
                "graded": graded,
                "max_deviation": _finite_or_none(raw_dev),
            }
        )

    # Subgroup audit
    #
    # CRITICAL, per ROW, the FABRICATED-ALL-CLEAR direction. Until 2026-08-28
    # every field here carried a reassuring default: `n_subgroups_flagged`
    # defaulted to 0, `worst_disparity` to 0.0 and `gerrymandering_detected` to
    # False. Executed on this repo, the audit `{"n_subgroups_analyzed": 6}`
    # rendered "6 analysed - 0 flagged - worst: n/a (0.000)" with no
    # gerrymandering chip: a complete clean bill of health for six subgroups
    # whose disparities the audit never reported. "0 flagged" is the identical
    # cell a genuinely clean audit prints, and zero disparity is perfect parity,
    # which is the strongest claim this panel can make.
    #
    # Absent and zero are different claims. Each field is None unless the audit
    # actually reported it, and the template says so in words where it did not.
    sg_data: Dict[str, Any] = {}
    flagged: List[Dict[str, Any]] = []
    if subgroup_audit is not None:
        sa = subgroup_audit if isinstance(subgroup_audit, dict) else _obj_to_dict(subgroup_audit)
        for sg in sa.get("flagged_subgroups") or []:
            if isinstance(sg, dict):
                flagged.append(sg)
            elif isinstance(sg, (tuple, list)) and len(sg) >= 2:
                # `float(sg[1])` raised on a non-numeric second element and let
                # a NaN through to be drawn as "N/A" in the flagged row's red
                # 600-weight number, which reads as a measured disparity whose
                # digits were lost. G13 2026-09-30.
                flagged.append({"name": str(sg[0]), "disparity": _finite_or_none(sg[1])})
        raw_flagged = sa.get("n_subgroups_flagged")
        if raw_flagged is None and sa.get("flagged_subgroups") is not None:
            # An audit that enumerated its flagged subgroups DID report the
            # count: the list is the report. An audit that reported neither has
            # reported nothing, and no number is invented for it.
            raw_flagged = len(flagged)
        raw_worst = sa.get("worst_subgroup")
        raw_disparity = sa.get("worst_disparity")
        # The LAST default on this panel, and it was the one that decided
        # whether the panel was drawn at all. `n_subgroups_analyzed` defaulted
        # to 0, and robustness_testing.svg gates the whole subgroup section on
        # `subgroup.n_analyzed`, so an audit that reported its FINDINGS but not
        # its denominator was erased from the canvas: executed on this repo,
        # `{"flagged_subgroups": [("age<25 x female", 0.31)],
        # "worst_disparity": 0.31}` with no permutation test and no sensitivity
        # check took the `_not_tested` branch below and rendered "Nothing was
        # tested ... no subgroup was analysed", over a supplied audit naming a
        # 0.31 disparity. That is a fabricated all-clear by DELETION, which is
        # worse than a wrong number because there is nothing on the page for a
        # reader to disbelieve.
        raw_analyzed = sa.get("n_subgroups_analyzed")
        # All three through `_finite_or_none` (G13 2026-09-30): `int(nan)` raises
        # ValueError, `int(True)` is 1, and a NaN `worst_disparity` passed the
        # template's `is not none` gate and printed "worst: <name> (N/A)" where
        # the third state "worst disparity not reported" is what that panel has
        # for a number the audit did not report.
        n_analyzed = _count_or_none(raw_analyzed)
        n_flagged = _count_or_none(raw_flagged)
        worst_disparity = _finite_or_none(raw_disparity)
        gerrymandering = sa.get("gerrymandering_detected")
        sg_data = {
            "n_analyzed": n_analyzed,
            "n_flagged": n_flagged,
            "worst_subgroup": str(raw_worst) if raw_worst is not None else None,
            "worst_disparity": worst_disparity,
            # None is a third state here and the template draws it: an audit
            # that did not run the gerrymandering check is not an audit that
            # cleared it, and the absence of the red chip reads as cleared.
            "gerrymandering": gerrymandering,
            # What the template gates the panel on now, instead of a count that
            # may never have been reported. An audit that reported ANY of its
            # cells is an audit the reader has to see, denominator or not.
            "reported": bool(
                flagged
                or n_analyzed
                or n_flagged is not None
                or raw_worst is not None
                or worst_disparity is not None
                or gerrymandering is not None
            ),
        }

    # A run with no test AND no audited subgroup measured nothing at all, so it
    # gets the could-not-check canvas rather than an empty table shell under a
    # graded headline. This is the `robustness_testing_to_svg([], [], None)`
    # case recorded in the CRITICAL note above; `[], [], {}` reaches it too,
    # because an audit that analysed no subgroup is an audit that did not run.
    #
    # Gated on `reported`, never on the analysed COUNT alone: the count is one
    # cell of the audit and it may be the one cell that is missing, so keying
    # this branch on it discarded audits that had reported a flagged subgroup
    # and its disparity. See the note beside `sg_data` above.
    if not perm_rows and not sens_rows and not sg_data.get("reported"):
        return _not_tested(
            reason=(
                "No permutation test and no sensitivity check was supplied, and no "
                "subgroup was analysed, so nothing was tested on this run."
            ),
            explanation=explanation,
            save_path=save_path,
        )

    # Overall score, computed only over SUPPLIED components. An absent
    # section is unknown, not a zero: counting missing permutation tests
    # as failures rendered MARGINAL for a run whose every supplied test
    # was perfect.
    parts = []  # (component score, weight)
    # Only SCORED sensitivity rows are averaged. Dividing by every supplied row
    # scored an unmeasured one as a total failure, which is how one bare
    # perturbation row turned a perfect 0.92 into MARGINAL 0.68.
    scored_sens = [s for s in sens_rows if s["score"] is not None]
    ungraded_sens = [s for s in sens_rows if not s["graded"]]
    if scored_sens:
        parts.append((sum(s["score"] for s in scored_sens) / len(scored_sens), 0.6))
    # Only GRADED permutation rows are counted. Dividing by every supplied row
    # scored an ungraded one as a pass, which is how a defaulted p-value of 1.0
    # became a perfect permutation component.
    graded_perm = [p for p in perm_rows if p["graded"]]
    ungraded_perm = [p for p in perm_rows if not p["graded"]]
    if graded_perm:
        perm_pass = sum(1 for p in graded_perm if not p["significant"])
        parts.append((perm_pass / len(graded_perm), 0.4))
    # Three states, never two. Only the subgroup audit was supplied: there is no
    # scored test to aggregate, so there is no score. The old neutral 0.50 was a
    # SENTINEL, and the template graded it like a measurement, printing
    # "MARGINAL" in amber with "score: 0.50" underneath for a model whose
    # stability nobody had tested. A midpoint is not a middling result.
    score_known = bool(parts)
    overall: Optional[float] = None
    if score_known:
        overall = max(0, min(1, sum(v * w for v, w in parts) / sum(w for _, w in parts)))

    # Recommendations
    recs: List[str] = []
    if any(p["significant"] for p in perm_rows):
        recs.append("Significant permutation test(s) detected: review metric stability.")
    # Graded rows only: an ungraded perturbation is not a low-robustness finding,
    # and this line is what sends someone to augment their data.
    if any(s["graded"] and not s["is_robust"] for s in sens_rows):
        recs.append("Some perturbation types show low robustness: consider data augmentation.")
    if sg_data.get("gerrymandering"):
        recs.append("Gerrymandering risk detected: audit subgroup definitions carefully.")
    elif sg_data and sg_data.get("gerrymandering") is None:
        recs.append("The subgroup audit reported no gerrymandering result: it was not checked.")
    # `is not None` and never truthiness: an audit that really flagged nothing
    # keeps its honest silence here, while one that reported no count says so.
    if sg_data.get("n_flagged") is not None:
        if sg_data["n_flagged"] > 0:
            recs.append(f"{sg_data['n_flagged']} subgroup(s) flagged: investigate worst disparity.")
    elif sg_data.get("reported"):
        # "The audit of None subgroup(s)" is what an f-string makes of a
        # withheld denominator, and a reader cannot tell that from a rendering
        # fault. The sentence drops the number rather than inventing one; an
        # audit that DID report its denominator keeps the line it always had.
        recs.append(
            f"The audit of {sg_data['n_analyzed']} subgroup(s) reported no flagged count: "
            f"nothing here says they were clear."
            if sg_data.get("n_analyzed") is not None
            else "The subgroup audit reported neither a subgroup count nor a flagged "
            "count: nothing here says they were clear."
        )
    # Named on the canvas, and placed BEFORE the all-clear below so it also
    # suppresses it: "All robustness checks passed" is a claim over every
    # supplied test, and a test that reported no p-value did not pass anything.
    if ungraded_perm:
        recs.append(
            f"{len(ungraded_perm)} permutation test(s) reported no p-value "
            f"({', '.join(p['method'] for p in ungraded_perm)}); they are not graded and are "
            f"excluded from the robustness score."
        )
    # The same sentence for the sensitivity half, and for the same reason: it has
    # to land BEFORE the all-clear so it suppresses it. A perturbation that
    # reported no score neither passed nor failed, so "All robustness checks
    # passed" is not a claim this run gets to make.
    if ungraded_sens:
        # Rasterised and read: this panel does not wrap, and the longer phrasing
        # used for the permutation line above ran off the right edge once a
        # perturbation name as ordinary as "feature_dropout_10%" was in it.
        recs.append(
            f"{len(ungraded_sens)} sensitivity check(s) reported no score "
            f"({', '.join(s['type'] for s in ungraded_sens)}); not graded, and excluded "
            f"from the score."
        )
    # Why there is no score, in the reader's terms. A run whose permutation tests
    # were ALL ungraded did supply tests, so the older sentence ("none was
    # supplied") would have been false about it, and each account below has to
    # stay true of its own branch. The same is now possible on the sensitivity
    # side, so "no sensitivity check was supplied" may no longer be said when
    # unscored sensitivity rows are on the canvas.
    if not perm_rows and not sens_rows:
        no_score_reason = "No permutation test and no sensitivity check was supplied"
    elif not sens_rows:
        no_score_reason = (
            "No permutation test could be graded and no sensitivity check was supplied"
        )
    elif not perm_rows:
        no_score_reason = (
            "No sensitivity check reported a score and no permutation test was supplied"
        )
    else:
        no_score_reason = "No permutation test could be graded and no sensitivity check was scored"
    # The all-clear is written only when a check ran to pass. It used to be the
    # `else` of every finding above, so "All robustness checks passed" was the
    # headline sentence on a canvas reporting zero checks. Nothing failed and
    # nothing ran are different claims, and this panel is where a reader looks
    # for the second one, so the absence leads the list.
    if not score_known:
        # This panel draws one line per entry and does not wrap: past roughly 130
        # characters the sentence runs off the right edge of the canvas, which is
        # why the second branch stops short. The <desc> wraps and carries the
        # fuller account.
        recs.insert(
            0,
            f"{no_score_reason}: no robustness score was computed, and the audit is not graded."
            if not perm_rows and not sens_rows
            else f"{no_score_reason}: no robustness score was computed.",
        )
    elif not recs:
        recs.append("All robustness checks passed: model shows stable fairness behaviour.")

    template_data = {
        "title": "Robustness Testing Dashboard",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score_known": score_known,
        "overall_score": overall,
        "overall_color": _score_color(overall) if overall is not None else _UNKNOWN,
        "overall_label": _score_label(overall) if overall is not None else "NOT SCORED",
        # Read by rendering.explain.build_explanation, which replaces the
        # accessible <desc> and the action line whenever it is set. Without it
        # the <desc> read "Robustness NOT SCORED (score 0.00)", i.e. the number
        # withheld from the canvas came back as a measurement for every reader
        # who cannot see the canvas.
        "not_assessable": not score_known,
        "not_assessable_reason": (
            ""
            if score_known
            else (
                f"{no_score_reason}, so no robustness score was computed. The subgroup "
                f"audit on this canvas is shown exactly as it was supplied and is not "
                f"graded into any verdict."
            )
        ),
        "nothing_tested": False,
        "is_example": False,
        # The subtitle counts SUPPLIED tests, and a count reads as a measurement,
        # so the ungraded ones are named in it rather than left to be assumed
        # measured. This is the decided headline rule (c): the ungraded count
        # goes on the CANVAS beside the headline, not only into the <desc>.
        "n_perm_ungraded": len(ungraded_perm),
        "n_sens_ungraded": len(ungraded_sens),
        "perm_tests": perm_rows[:8],
        "sens_bars": sens_rows[:6],
        "subgroup": sg_data,
        "flagged": flagged[:6],
        "recommendations": recs[:4],
        "explanation": explanation,
    }

    return _render_and_save("robustness_testing", template_data, save_path)


# Helpers


def _finite_or_none(raw: Any) -> Optional[float]:
    """A p-value only when it is a real number. NaN is a refusal, not a result.

    READINESS-6, 2026-09-10. A BOOL is refused, which the sentence above always
    claimed and the code never did: ``float(True)`` is 1.0, a finite number that
    passes every check here. numpy's ``np.bool_`` is the half that gets missed,
    because it is not a Python bool, so a boolean column read out of a DataFrame
    walked straight through. A numeric STRING is still accepted, deliberately:
    these rows arrive from JSON and CSV where "0.5" is a real measurement that
    was serialised, and refusing it would discard evidence.
    """
    if raw is None or is_flag(raw):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _count_or_none(raw: Any) -> Optional[int]:
    """A reported whole-number count, or None. Same rule as `_finite_or_none`.

    `int(raw)` raised ValueError on a NaN and returned 1 for ``True``, and this
    count is what the template gates the whole subgroup panel on.
    """
    value = _finite_or_none(raw)
    return int(value) if value is not None else None


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


def _not_tested(
    *, reason: str, explanation: Optional[str] = None, save_path: Optional[str] = None
) -> str:
    """The third state: a dashboard for a run that tested nothing.

    Every field that could read as a finding is withheld rather than defaulted.
    ``nothing_tested`` drives the template's single "Nothing was tested" panel
    and suppresses the permutation, sensitivity, subgroup and recommendation
    panels, which would otherwise render as empty table shells complete with
    column headings. ``not_assessable`` drives the accessible ``<desc>`` and the
    action line (through ``rendering.explain.build_explanation``), so the canvas
    and the screen reader tell one story.

    There is deliberately no score, no verdict label and no status colour beyond
    slate: a reader must not be able to take a grade off this canvas. It is not
    a rejection either. Nothing here says the model is fragile, only that its
    robustness was not measured.
    """
    template_data = {
        "title": "Robustness Testing Dashboard",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "score_known": False,
        "overall_score": None,
        "overall_color": _UNKNOWN,
        "overall_label": "NOT SCORED",
        "not_assessable": True,
        "not_assessable_reason": reason,
        "nothing_tested": True,
        "nothing_tested_reason": reason,
        "is_example": False,
        "n_perm_ungraded": 0,
        "n_sens_ungraded": 0,
        "perm_tests": [],
        "sens_bars": [],
        "subgroup": {},
        "flagged": [],
        "recommendations": [],
        "explanation": explanation,
    }
    return _render_and_save("robustness_testing", template_data, save_path)


def _demo(*, explanation: Optional[str] = None, save_path: Optional[str] = None) -> str:
    """Generate demo SVG with synthetic robustness data.

    Reachable ONLY through ``robustness_testing_to_svg(example=True)``. Every
    number below is invented, and ``is_example`` is what marks the canvas as a
    demonstration: it draws the full-width EXAMPLE band under the header and
    leads the accessible description with the same marker.
    """
    perm_tests = [
        {
            "method": "demographic_parity",
            "observed": 0.042,
            "p_value": 0.312,
            "significant": False,
            "graded": True,
            "direction": "positive",
        },
        {
            "method": "equalized_odds",
            "observed": 0.087,
            "p_value": 0.003,
            "significant": True,
            "graded": True,
            "direction": "negative",
        },
        {
            "method": "calibration_diff",
            "observed": 0.021,
            "p_value": 0.578,
            "significant": False,
            "graded": True,
            "direction": "neutral",
        },
    ]
    # `measured` and `graded` are required by the template: a bar without them
    # renders "no score" beside a NOT CHECKED chip. Every number below is
    # invented, so they say only that the FIXTURE states a score, never that
    # anything was measured; the EXAMPLE band is what carries that.
    sens_bars = [
        {
            "type": "label_noise_5%",
            "score": 0.92,
            "bar_w": 184,
            "color": _PASS,
            "is_robust": True,
            "measured": True,
            "graded": True,
            "max_deviation": 0.008,
        },
        {
            "type": "feature_dropout_10%",
            "score": 0.74,
            "bar_w": 148,
            "color": _WARN,
            "is_robust": False,
            "measured": True,
            "graded": True,
            "max_deviation": 0.031,
        },
        {
            "type": "sample_bootstrap",
            "score": 0.88,
            "bar_w": 176,
            "color": _PASS,
            "is_robust": True,
            "measured": True,
            "graded": True,
            "max_deviation": 0.015,
        },
    ]
    subgroup = {
        "n_analyzed": 6,
        "n_flagged": 2,
        "worst_subgroup": "age<25 × female",
        "worst_disparity": 0.142,
        "gerrymandering": False,
        # Required by the template: the subgroup panel is gated on `reported`,
        # so a fixture without it would silently lose the whole section. Every
        # number above is invented; the EXAMPLE band is what carries that.
        "reported": True,
    }
    flagged = [
        {"name": "age<25 × female", "disparity": 0.142},
        {"name": "age>65 × minority", "disparity": 0.098},
    ]
    recs = [
        "Equalized odds permutation test is significant: review metric stability.",
        "Feature dropout robustness is marginal: consider data augmentation.",
        "2 subgroups flagged: investigate worst disparity (0.142).",
    ]

    overall = 0.72
    template_data = {
        "title": "Robustness Testing Dashboard",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # DO NOT REMOVE: the band on the canvas and the marker on the <desc> are
        # the only things separating this fixture from an audit artifact once
        # the SVG is exported.
        "is_example": True,
        "nothing_tested": False,
        "n_perm_ungraded": 0,
        "n_sens_ungraded": 0,
        "score_known": True,
        "overall_score": overall,
        "overall_color": _score_color(overall),
        "overall_label": _score_label(overall),
        "perm_tests": perm_tests,
        "sens_bars": sens_bars,
        "subgroup": subgroup,
        "flagged": flagged,
        "recommendations": recs,
        "explanation": explanation,
    }

    return _render_and_save("robustness_testing", template_data, save_path)
