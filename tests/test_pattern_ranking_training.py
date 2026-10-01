"""The absent-value pattern, swept out of the ranking / training / robustness adapters.

Wave 15 was scoped by FILE rather than by a list of reported sites, because the
previous waves kept hardening the block they were pointed at while an identical
block a few lines below kept its ``.get(key, 0)`` defaults. Two of the three
files here were in exactly that state: ``adapters_ranking`` had a hardened metric
badge loop sitting directly above a group panel that still read every cell
through a zero default, and ``adapters_training`` had ``_method_state`` and
``_coord`` in the same module as a group-statistics loop that used neither.

THE RULE, per row: a row that reported nothing gets no number, no badge, no
colour, no bar, no plot point, and no place in any count, sort or headline
implying it was measured. Both directions are wrong. A fabricated all-clear
("0 flagged", "well distributed") and a fabricated breach ("0.0% positive rate",
"FAIL") are the same defect, and the second is not the safer error.

Every test below states the FABRICATION it forbids, so a future reader can tell
what would break if the guard were removed.
"""

import ast
import math
import pathlib
import re
import warnings
from types import SimpleNamespace

import pytest

from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg
from vfairness.rendering.adapters_robustness import robustness_testing_to_svg
from vfairness.rendering.adapters_training import (
    method_comparison_to_svg,
    training_analysis_report_to_svg,
    training_report_to_svg,
)

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vfairness" / "rendering"
_TEMPLATES = _SRC / "templates"

_MY_ADAPTERS = [
    _SRC / "adapters_ranking.py",
    _SRC / "adapters_training.py",
    _SRC / "adapters_robustness.py",
]
_MY_TEMPLATES = [
    _TEMPLATES / "ranking_fairness.svg",
    _TEMPLATES / "robustness_testing.svg",
    _TEMPLATES / "training_analysis_report.svg",
    _TEMPLATES / "training_report.svg",
    _TEMPLATES / "method_comparison.svg",
    _TEMPLATES / "tradeoff_analysis.svg",
]

# The headline band. Rule (c) requires the ungraded count to be drawn HERE, where
# a reader who takes the header and stops reading will see it, not at the foot of
# a canvas that is often two thousand units tall.
_HEADLINE_BAND_Y = 200

# The methods-table row stripe, matched by GEOMETRY. Colour is not usable in an
# assertion here: `rendering.engine` remaps every hex the template writes to the
# Blanco palette after the render, so `#dc2626` never survives to the output and
# a hex assertion would pin the palette rather than the rule.
_STRIPE = 'width="2.5" height="32"'

_TEXT_RE = re.compile(r"<text\b([^>]*)>(.*?)</text>", re.DOTALL)
_Y_RE = re.compile(r'\by="(-?\d+(?:\.\d+)?)"')


def _headline_band_text(svg: str) -> str:
    """Every <text> drawn at y <= 200, concatenated. Rule (c)'s search space."""
    out = []
    for attrs, body in _TEXT_RE.findall(svg):
        m = _Y_RE.search(attrs)
        if m and float(m.group(1)) <= _HEADLINE_BAND_Y:
            out.append(re.sub(r"<[^>]+>", "", body))
    return " ".join(out)


def _canvas_text(svg: str) -> str:
    """Every <text> on the canvas, tags stripped. Excludes <desc> and <metadata>."""
    return " ".join(re.sub(r"<[^>]+>", "", body) for _, body in _TEXT_RE.findall(svg))


def _desc(svg: str) -> str:
    m = re.search(r"<desc>(.*?)</desc>", svg, re.DOTALL)
    return m.group(1) if m else ""


#  ── ranking fixtures ────────────────────────────────────────────────────────

_MEASURED_METRICS = [
    {"metric_name": "ndkl", "value": 0.08, "threshold": 0.10},
    {"metric_name": "exposure_parity_difference", "value": 0.12, "threshold": 0.10},
]
_MEASURED_GROUPS = {
    "Male": {
        "mean_exposure": 0.28,
        "avg_position": 4.2,
        "median_position": 3.0,
        "min_position": 1,
        "max_position": 15,
        "count": 450,
    },
    "Female": {
        "mean_exposure": 0.24,
        "avg_position": 5.1,
        "median_position": 4.0,
        "min_position": 1,
        "max_position": 18,
        "count": 380,
    },
}


def _report(**kw):
    base = dict(
        timestamp="2026-08-28 12:00",
        task_type="binary_classification",
        data_info={"n_samples": 5000, "n_groups": 2, "n_features": 12, "attribute_name": "gender"},
        baseline_metrics={
            "accuracy": 0.852,
            "fairness_violation": 0.185,
            "constraint_satisfied": False,
        },
        method_comparisons=[],
        recommendation=None,
        critical_issues=[],
        action_items=[],
        tradeoff_analysis={},
        fairness_analysis={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _method(name, accuracy, violation, satisfied, time=1.0):
    return SimpleNamespace(
        method_name=name,
        accuracy=accuracy,
        fairness_violation=violation,
        constraint_satisfied=satisfied,
        training_time=time,
    )


#  ── ranking: the group exposure panel ───────────────────────────────────────


class TestRankingGroupPanel:
    """adapters_ranking, the block that sat directly under a hardened one.

    THE FABRICATION: a group entry carrying only its NAME rendered a complete
    measured row, "n=0" beside an exposure of "0.000" drawn in the group's own
    colour, with 0.0 / 0.0 / 0 / 0 / 0 across the position table. Zero exposure
    is a group shut out of the ranking entirely, which is the strongest finding
    this panel can carry, and every digit of it came from a `.get(key, 0)`.
    """

    def test_a_group_that_reported_no_exposure_gets_no_number_and_no_bar(self):
        svg = ranking_fairness_to_svg(_MEASURED_METRICS, dict(_MEASURED_GROUPS, ZZ_BARE={}))
        text = _canvas_text(svg)

        assert "ZZ_BARE" in text, "the group itself must still be listed"
        assert "no exposure reported" in text
        assert "n not reported" in text
        # The fabricated cells, in the exact form the old defaults produced.
        assert "0.000" not in text, "an exposure of 0.000 is a finding, not a blank"
        assert "n=0" not in text
        # The measured rows are untouched.
        assert "0.280" in text and "0.240" in text

    def test_an_unreported_position_cell_is_withheld_not_drawn_as_rank_zero(self):
        svg = ranking_fairness_to_svg(_MEASURED_METRICS, dict(_MEASURED_GROUPS, ZZ_BARE={}))
        text = _canvas_text(svg)
        # The bare group's POSITION-TABLE row only. The name occurs three times
        # on this canvas (exposure panel, position table, recommendation panel),
        # so `split(...)[-1]` lands in the prose and asserts nothing: it was
        # written that way first and the sabotage pass caught it green.
        row = text.split("ZZ_BARE")[2].split("Recommendations")[0]

        # "0.0" in an AVERAGE POSITION column reads as rank zero, better than
        # first place; "0" in MIN / MAX / COUNT reads the same way.
        assert "0.0" not in row, "a defaulted average position reads as rank zero"
        assert "0" not in row, "a defaulted min, max or count reads the same way"
        assert row.split().count("N/A") == 5, "all five cells of the row say so"

    def test_a_measured_zero_exposure_still_draws_its_number(self):
        """The over-correction control, and the point of `is not None`.

        A group that really WAS given no exposure is a finding and must keep its
        0.000. Withholding it would be the same fabrication pointed the other
        way.
        """
        svg = ranking_fairness_to_svg(
            _MEASURED_METRICS, {"Shut_Out": {"mean_exposure": 0.0, "count": 0}}
        )
        text = _canvas_text(svg)

        assert "0.000" in text
        assert "n=0" in text
        assert "no exposure reported" not in text

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "high", None])
    def test_a_sentinel_exposure_is_not_a_measurement(self, bad):
        """`True` is an int in Python and would print as exposure 1.0, the best
        value on the scale; NaN and the infinities are sentinels, and an infinite
        exposure divides through the bar-width arithmetic."""
        svg = ranking_fairness_to_svg([], {"A": {"mean_exposure": bad}})

        assert svg, "a sentinel must never collapse the render to an empty string"
        assert "no exposure reported" in _canvas_text(svg)

    def test_an_unreported_exposure_never_triggers_the_gap_finding(self):
        """THE FABRICATED BREACH. The gap test read the absent exposure as 0 and
        `or 1` turned that into a denominator of 1, so any measured group above
        2.0 raised "Large exposure gap" against a number nobody measured."""
        svg = ranking_fairness_to_svg(
            _MEASURED_METRICS, {"Big": {"mean_exposure": 3.0}, "Bare": {}}
        )
        text = _canvas_text(svg)

        assert "Large exposure gap" not in text
        assert "1 group(s) reported no exposure" in text

    def test_a_real_gap_between_two_measured_groups_is_still_reported(self):
        """The other direction of the same control: the finding must survive."""
        svg = ranking_fairness_to_svg(
            _MEASURED_METRICS, {"Big": {"mean_exposure": 0.9}, "Small": {"mean_exposure": 0.1}}
        )

        assert "Large exposure gap" in _canvas_text(svg)

    def test_the_all_clear_is_narrowed_to_the_groups_that_reported_one(self):
        """ "Well-distributed across the groups shown" is a claim over the whole
        panel, and the panel shows rows that reported nothing."""
        svg = ranking_fairness_to_svg(_MEASURED_METRICS, dict(_MEASURED_GROUPS, ZZ_BARE={}))
        text = _canvas_text(svg)

        assert "well-distributed across the groups shown" not in text
        assert "well-distributed across the 2 group(s) that reported one" in text

    def test_the_ungraded_group_count_reaches_the_headline_band(self):
        """Rule (c). "3 groups" is a count, and a count reads as three measured
        groups; the qualifier has to sit where the count does."""
        svg = ranking_fairness_to_svg(_MEASURED_METRICS, dict(_MEASURED_GROUPS, ZZ_BARE={}))

        assert "1 reported no exposure" in _headline_band_text(svg)

    def test_a_fully_reported_run_keeps_the_subtitle_it_always_had(self):
        """The over-correction control for rule (c): no notice when none is due."""
        svg = ranking_fairness_to_svg(_MEASURED_METRICS, _MEASURED_GROUPS)
        band = _headline_band_text(svg)

        assert "2 metrics · 2 groups" in band
        assert "reported no exposure" not in band

    def test_the_max_exposure_scale_is_taken_over_reported_groups_only(self):
        """A defaulted 0 could not move the maximum, but it could move the bar
        widths through the `or 1` fallback once every group was bare."""
        svg = ranking_fairness_to_svg([], {"A": {}, "B": {}})

        assert svg
        assert _canvas_text(svg).count("no exposure reported") == 2


#  ── training: the group statistics panel ────────────────────────────────────


class TestTrainingGroupStatistics:
    """adapters_training, the known site.

    THE FABRICATION: a group carrying only its name rendered "ZZ_BARE | 0 |
    0.0% | 0.0%" with a rate bar. A POSITIVE RATE of 0.0% is the worst finding
    this panel can show, a group that never received a single positive outcome,
    and a SIZE of 0 says the same group has no members, contradicting its own
    presence in the table.
    """

    _GOOD = {
        "Male": {"size": 2750, "proportion": 0.55, "positive_rate": 0.72},
        "Female": {"size": 2000, "proportion": 0.40, "positive_rate": 0.58},
    }

    def _svg(self, groups):
        return training_analysis_report_to_svg(
            _report(
                fairness_analysis={
                    "constraint_type": "demographic_parity",
                    "base_rate_disparity": 0.12,
                    "group_statistics": groups,
                }
            )
        )

    def test_a_bare_group_row_reports_nothing_rather_than_zero(self):
        text = _canvas_text(self._svg(dict(self._GOOD, ZZ_BARE={})))
        # Only the bare row's own cells, so a measured "40.0%" elsewhere on the
        # canvas cannot satisfy a substring test for "0.0%".
        row = text.split("ZZ_BARE", 1)[1].split("METHOD", 1)[0]

        assert "ZZ_BARE" in text
        assert row.count("not reported") == 3, "size, proportion and rate each say so"
        assert "%" not in row, "0.0% positive rate is the worst finding this panel draws"
        assert "0" not in row, "a size of 0 says the group has no members"
        # The measured rows are untouched.
        assert "72.0%" in text and "58.0%" in text and "2750" in text

    def test_a_bare_group_draws_no_rate_bar(self):
        """The bar has a 3px minimum-width clamp so a genuinely tiny rate stays
        visible. An unmeasured rate would have inherited that same stub, which is
        indistinguishable from a rate close to zero."""
        # Counted by GEOMETRY, never by fill: the engine remaps every hex in the
        # template to the Blanco palette after rendering (and drops `rx`), so a
        # colour assertion here would pin the palette rather than the rule. The
        # RATE BAR column starts at x=560 and holds two rects when a rate was
        # measured, the empty track and the filled bar, and only the track when
        # it was not.
        rects = lambda svg: re.findall(r'<rect[^>]*\bx="560"[^>]*>', svg)  # noqa: E731
        bare = self._svg({"ZZ_BARE": {}})
        measured = self._svg({"Tiny": {"size": 1, "proportion": 0.01, "positive_rate": 0.001}})

        assert len(rects(bare)) == 1, f"the bare row draws its track and no bar: {rects(bare)}"
        assert len(rects(measured)) == 2, "a measured row draws both"
        assert 'width="3"' in rects(measured)[1], "the 3px clamp still protects a real rate"

    def test_a_measured_zero_positive_rate_still_draws_its_number_and_bar(self):
        """The over-correction control. A group that really received no positive
        outcome is a finding and keeps its 0.0%."""
        text = _canvas_text(
            self._svg({"Excluded": {"size": 120, "proportion": 0.02, "positive_rate": 0.0}})
        )

        assert "0.0%" in text
        assert "120" in text
        assert "not reported" not in text

    @pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "many", None])
    def test_a_sentinel_statistic_is_not_a_measurement(self, bad):
        svg = self._svg({"A": {"size": bad, "proportion": bad, "positive_rate": bad}})

        assert svg, "a sentinel must never collapse the render to an empty string"
        assert "not reported" in _canvas_text(svg)

    def test_the_partly_unreported_group_count_reaches_the_headline_band(self):
        """Rule (c). The GROUPS stat card prints a bare count and reads as that
        many measured groups."""
        band = _headline_band_text(self._svg(dict(self._GOOD, ZZ_BARE={})))

        assert "1 group(s) partly unreported" in band

    def test_a_fully_reported_run_keeps_the_subtitle_it_always_had(self):
        band = _headline_band_text(self._svg(self._GOOD))

        assert "demographic_parity constraint" in band
        assert "unreported" not in band


#  ── training: the method table, and the residue this wave closed ────────────


class TestTrainingMethodTable:
    """THE FABRICATION, and then its mirror image.

    ``constraint_satisfied`` reached this template raw and every branch was
    ``{% if m.satisfied %} ... {% else %}``, so a method whose constraint was
    NEVER EVALUATED drew a red row stripe, a red violation number and a red FAIL
    chip. The first fix WITHHELD such rows from the table because the template
    had no third branch, which traded a fabricated failure for a silent deletion:
    a table listing two of three trained methods reads as the whole run.
    """

    _M = [_method("Baseline", 0.852, 0.185, False), _method("Constrained", 0.828, 0.041, True)]
    _UNGRADED = _method("Adversarial", 0.77, 0.05, None, time=8.7)

    def _svg(self, methods):
        return training_analysis_report_to_svg(_report(method_comparisons=methods))

    def test_an_ungraded_method_is_drawn_and_is_drawn_as_ungraded(self):
        text = _canvas_text(self._svg(self._M + [self._UNGRADED]))

        assert "Adversarial" in text, "a trained method must not vanish from the table"
        assert "NOT CHECKED" in text
        assert text.count("FAIL") == 1, "only the one MEASURED failure may show a FAIL chip"

    def test_an_ungraded_method_gets_no_row_stripe_and_no_verdict_chip(self):
        """Asserted by GEOMETRY, not by fill: the engine remaps every hex to the
        Blanco palette after rendering, so a colour assertion would pin the
        palette rather than the rule. The methods-table stripe is the only
        `width="2.5" height="32"` rect on this page."""
        svg = self._svg([self._UNGRADED])
        text = _canvas_text(svg)

        assert _STRIPE not in svg, "no part of an ungraded row may be painted as a verdict"
        assert "NOT CHECKED" in text
        assert "PASS" not in text and "FAIL" not in text

    def test_a_measured_failure_keeps_its_row_stripe(self):
        """The over-correction control: the finding must survive the fix."""
        svg = self._svg([_method("Baseline", 0.852, 0.185, False)])

        assert _STRIPE in svg
        assert "FAIL" in _canvas_text(svg)

    def test_the_ungraded_method_count_reaches_the_headline_band(self):
        band = _headline_band_text(self._svg(self._M + [self._UNGRADED]))

        assert "1 of 3 method(s) ungraded" in band

    def test_a_fully_graded_run_carries_no_ungraded_notice(self):
        band = _headline_band_text(self._svg(self._M))

        assert "ungraded" not in band


#  ── robustness: the subgroup audit ──────────────────────────────────────────


class TestRobustnessSubgroupAudit:
    """THE FABRICATION, in its worst shape: an all-clear by DELETION.

    ``n_subgroups_analyzed`` defaulted to 0 and the template gated the whole
    subgroup section on it, so an audit that reported its FINDINGS but not its
    denominator was erased from the canvas. Executed on this repo, an audit
    naming a 0.31 disparity rendered "Nothing was tested ... no subgroup was
    analysed". There is nothing on that page for a reader to disbelieve.
    """

    _FINDINGS = {
        "flagged_subgroups": [("age 18-25 x female", 0.31)],
        "worst_subgroup": "age 18-25 x female",
        "worst_disparity": 0.31,
    }

    def test_an_audit_with_findings_and_no_denominator_is_not_erased(self):
        svg = robustness_testing_to_svg([], [], self._FINDINGS)
        text = _canvas_text(svg)

        assert "age 18-25 x female" in text, "a reported finding must reach the canvas"
        assert "0.310" in text
        assert "Nothing was tested" not in text

    def test_the_withheld_denominator_is_named_rather_than_invented(self):
        text = _canvas_text(robustness_testing_to_svg([], [], self._FINDINGS))

        assert "subgroup count not reported" in text
        assert "0 analysed" not in text, "0 analysed is the cell an audit of nobody prints"
        assert "None" not in text, "a withheld count must never reach the canvas as 'None'"

    def test_the_description_agrees_with_the_canvas(self):
        """The accessible <desc> is the whole artifact for a screen-reader user.
        It used to say "no subgroup was analysed" over a canvas showing one."""
        svg = robustness_testing_to_svg([], [], self._FINDINGS)

        assert "no subgroup was analysed" not in _desc(svg)
        assert "not graded" in _desc(svg)

    def test_findings_alone_never_produce_a_robustness_score(self):
        """Rule (a) and (d): an audit is not a scored test, so it grades nothing."""
        svg = robustness_testing_to_svg([], [], self._FINDINGS)

        assert "NOT SCORED" in _canvas_text(svg)

    def test_an_audit_that_reported_nothing_at_all_still_reaches_nothing_tested(self):
        """The over-correction control. `[], [], {}` must not now draw an empty
        subgroup shell under a headline; it is a run that tested nothing."""
        for empty in ({}, {"n_subgroups_analyzed": 0}):
            text = _canvas_text(robustness_testing_to_svg([], [], empty))
            assert "Nothing was tested" in text, empty

    def test_a_reported_denominator_keeps_the_sentence_it_always_had(self):
        text = _canvas_text(robustness_testing_to_svg([], [], {"n_subgroups_analyzed": 6}))

        assert "6 analysed" in text
        assert "The audit of 6 subgroup(s) reported no flagged count" in text

    def test_the_example_fixture_still_draws_its_subgroup_panel(self):
        """The demo dict must carry `reported`, or the panel it exists to show
        disappears. Pinned because a fixture key is easy to forget."""
        text = _canvas_text(robustness_testing_to_svg(example=True))

        assert "Subgroup Audit" in text
        assert "6 analysed" in text
        assert "EXAMPLE ONLY" in text


#  ── the file-level sweeps: this wave is scoped by FILE, not by site ─────────


def _scanner_hits(path: pathlib.Path):
    """The wave scanner: `.get(key, <numeric or boolean literal>)`."""
    hits = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
        ):
            continue
        default = node.args[1]
        if isinstance(default, ast.Constant) and isinstance(default.value, (int, float, bool)):
            hits.append(f"{path.name}:{node.lineno}: {ast.unparse(node)[:90]}")
    return hits


@pytest.mark.parametrize("path", _MY_ADAPTERS, ids=lambda p: p.name)
def test_the_scanner_pattern_is_gone_from_the_whole_file(path):
    """Scoped by FILE. Fourteen waves fixed the block they were pointed at while
    an identical block a few lines away kept its defaults, so the target is the
    file's count, not a named site. Twelve hits stood here on 2026-08-28.

    A LAYOUT default (an x offset, a row height) would be legitimate; none of
    these three files needs one, so the bar is zero. If a genuine layout default
    is added later, exempt it explicitly here rather than raising the number.
    """
    assert _scanner_hits(path) == []


_SELF_SET_RE = re.compile(r"\{%-?\s*set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^%]*\b\1\b")


@pytest.mark.parametrize("path", _MY_TEMPLATES, ids=lambda p: p.name)
def test_no_template_accumulates_a_counter_across_a_jinja_loop(path):
    """The template shape the AST scan cannot see.

    ``{% set c = c + 1 %}`` inside ``{% for %}`` is discarded by Jinja's loop
    scoping, so every item draws at the same coordinate and the last one painted
    covers the rest. That shipped a false green "LOW: 2" over a HIGH-risk proxy
    in the published gallery, and a substring assertion passed straight over it
    because the markup was correct. Executed here: `{% set c = 0 %}{% for x in
    [1,2,3] %}{% set c = c + 1 %}{{ c }}{% endfor %}` renders "111", not "123".

    A ``{% set %}`` inside an ``{% if %}`` inside a ``{% for %}`` is fine and is
    deliberately not flagged: ``if`` is not a scoping construct in Jinja, so the
    bar-width clamps in these templates do work. Verified by rendering.
    """
    offenders = [
        f"{path.name}:{i}: {line.strip()[:100]}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _SELF_SET_RE.search(line)
    ]
    assert offenders == []


_PARTIAL_ROW_CASES = [
    ("ranking, empty", lambda: ranking_fairness_to_svg([], {})),
    ("ranking, bare group", lambda: ranking_fairness_to_svg([{"metric_name": "ndkl"}], {"A": {}})),
    (
        "ranking, inf exposure",
        lambda: ranking_fairness_to_svg([], {"A": {"mean_exposure": math.inf}}),
    ),
    ("robustness, bare audit", lambda: robustness_testing_to_svg([], [], {})),
    (
        "robustness, findings only",
        lambda: robustness_testing_to_svg([], [], {"flagged_subgroups": [("x", 0.3)]}),
    ),
    ("robustness, bare perm row", lambda: robustness_testing_to_svg([{"method": "dp"}], [], None)),
    (
        "training, empty report",
        lambda: training_analysis_report_to_svg(_report(baseline_metrics={})),
    ),
    (
        "training, bare group",
        lambda: training_analysis_report_to_svg(
            _report(fairness_analysis={"group_statistics": {"A": {}}})
        ),
    ),
    (
        "training, bare method",
        lambda: training_analysis_report_to_svg(
            _report(method_comparisons=[_method("m", None, None, None, None)])
        ),
    ),
    (
        "training_report, bare method",
        lambda: training_report_to_svg(
            _report(method_comparisons=[_method("m", None, None, None, None)])
        ),
    ),
    (
        "method_comparison, bare method",
        lambda: method_comparison_to_svg([_method("m", None, None, None, None)]),
    ),
]


@pytest.mark.parametrize("name,call", _PARTIAL_ROW_CASES, ids=[c[0] for c in _PARTIAL_ROW_CASES])
def test_a_partial_row_never_crashes_and_never_returns_an_empty_string(name, call):
    """NEVER CRASH, NEVER RETURN "".

    A zero-length string is the worst of the three states, because it cannot even
    be read as could-not-check: a caller that writes it gets a 0-byte .svg that
    opens as a broken image, and a caller that passed save_path keeps whatever
    the PREVIOUS render left on disk, with success reported either way.

    The warning check is what catches a SILENT fallback: `_render_and_save`
    swallows a template exception into a UserWarning, and warnings are filtered
    by default in exactly the batch pipelines that call these adapters.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = call()

    assert svg, "an empty string is not one of the three states"
    assert svg.lstrip().startswith(("<?xml", "<svg", "\n<svg")) or "<svg" in svg
    failures = [str(w.message) for w in caught if "rendering failed" in str(w.message)]
    assert not failures, f"the template raised and the adapter swallowed it: {failures}"


def test_a_plotted_tradeoff_point_always_carries_a_real_boolean_verdict():
    """The two-state branch that is defended IN THE ADAPTER, on purpose.

    Both scatter templates write the dot colour as
    ``fill="{% if p.satisfied %}#059669{% else %}#dc2626{% endif %}"``, and None
    takes the red arm, so a configuration whose constraint was never evaluated
    would be drawn as a measured breach at a measured position. There is no
    honest third colour for a point, so the guard is that such a row is never
    plotted at all. This pins the guard where it is testable.
    """
    from vfairness.rendering.adapters_training import _coord, _field

    rows = [
        {"accuracy": 0.85, "violation": 0.18, "satisfied": True},
        {"accuracy": 0.83, "violation": 0.08, "satisfied": None},
        {"accuracy": 0.81, "violation": 0.04},
        {"accuracy": None, "violation": 0.02, "satisfied": False},
    ]
    plotted = [
        r
        for r in rows
        if _coord(_field(r, "accuracy")) is not None
        and _coord(_field(r, "violation")) is not None
        and isinstance(_field(r, "satisfied"), bool)
    ]

    assert len(plotted) == 1
    svg = training_analysis_report_to_svg(
        _report(tradeoff_analysis={"all_results": rows, "pareto_frontier": []})
    )
    assert svg.count('r="4"') == 1, "exactly the one fully reported configuration is drawn"
    assert "3 trade-off point(s) not plotted" in _headline_band_text(svg)
