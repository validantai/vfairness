"""Surface grade g021: the unified bias detector's public read surfaces.

Batch g021 graded every public member of
``vfairness.preprocessing.bias_detection.detector``. The coverage record itself
(``modules_run``, ``attribute_observations``, ``attribute_assessed``,
``execution_coverage``, ``assessment_coverage``, ``summary``, ``to_dict``) was
already pinned by ``tests/test_bias_audit_ran_nothing.py`` and
``tests/test_bgl_stage2_s2g09.py``. What this file adds are the three surfaces
that were reading a 0.0 out of that record and grading it anyway, plus the cache
that was answering a question nobody asked.

1. ``BiasAuditReport.get_critical_count`` answered ``0`` on a 2-row frame whose
   own report said ``execution_coverage() == 'ran_but_assessed_nothing'``, whose
   recommendation said "Nothing here clears the data", and where four
   sub-modules had each refused in a warning. Same shape,
   ``get_high_risk_features`` answered ``[]`` while the proxy module had refused
   for ``n < min_sample_size``, which is the empty list that means "no proxies
   found" when the scan never ran.

2. ``BiasDetector.get_explanation`` came back graded off that same 0.0:
   severity 'info' on an item reading "The overall risk score of 0.00 is in the
   MINIMAL band. The dataset appears suitable for training with standard
   monitoring." The refusal was in the same object, in the recommendations,
   and the explanation contradicted it.

3. The four detection methods memoized into ONE slot each, so the first call's
   arguments answered every later call. Measured on a 600-row frame carrying two
   large, significant disparities: an exploratory
   ``analyze_disparities(min_group_size=400)`` made the following
   ``full_audit()`` report 0 disparity findings, 2 critical issues instead of 4,
   and drop the "Investigate significant disparities" recommendation, while
   ``execution_coverage()`` still said 'complete'. A stale cache deleted real
   findings from a real audit. The same defect was fixed for
   ``post_processing.calibration.analyzer`` by audit wave 4
   (``tests/test_audit_wave4_postproc.py::TestAnalyzerCacheKeys``).

Every refusal below is paired with a healthy control asserting the real value is
still measured exactly, because a refusal that fires on healthy data is a worse
defect than the one it replaced and passes any test that only checks the
degenerate case.

CLOSED 2026-09-25, and the diagnosis recorded below was right on both counts.
``BiasAuditReport.to_svg`` rendered "OVERALL RISK 0% MINIMAL" for that same 2-row
audit; the fix landed in ``rendering/adapters.py`` and the two strict xfails here
turned into the failures they were designed to become, which is what asked for
these markers to be removed. A third cause the note did not have: the module
tiles rendered a measured 0.00 each, on the strength of the module having run.
See ``tests/test_bias_audit_svg_coverage.py``.
"""

from __future__ import annotations

import html
import re
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.detector import (
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    COVERAGE_UNASSESSED,
    COVERAGE_UNRECORDED,
    BiasDetector,
)
from vfairness.preprocessing.bias_detection.historical import HistoricalRiskLevel
from vfairness.preprocessing.bias_detection.proxy import ProxyRiskLevel
from vfairness.preprocessing.bias_detection.representation import RepresentationSeverity
from vfairness.preprocessing.bias_detection.statistical import (
    EffectSizeInterpretation,
    SignificanceLevel,
)

# ── fixtures ────────────────────────────────────────────────────────────────


def _caught(fn, *a, **kw) -> Tuple[Any, List[str]]:
    """Call *fn* with warnings ENABLED and return (result, [messages]).

    A refusal carried only in a warning is invisible if warnings are filtered,
    and the default filter hides a repeat of the same message.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(w.message) for w in caught]


def _silent(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _two_row_frame() -> pd.DataFrame:
    """Two rows: one per group. Every module refuses on this."""
    return pd.DataFrame({"gender": ["M", "F"], "approved": [1, 0], "income": [50000, 52000]})


def _all_null_frame(n: int = 500) -> pd.DataFrame:
    """500 rows, and the protected column holds not one observation."""
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "gender": [None] * n,
            "approved": rng.integers(0, 2, n),
            "income": rng.normal(50_000, 5_000, n),
        }
    )


def _clean_frame(n: int = 900) -> pd.DataFrame:
    """900 rows, two well populated groups, nothing wrong with it."""
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "cohort": rng.choice(["a", "b"], size=n),
            "score_x": rng.normal(0, 1, n),
            "approved": rng.integers(0, 2, n),
        }
    )


def _biased_frame(n: int = 600) -> pd.DataFrame:
    """600 rows carrying a real, findable disparity and a real proxy.

    Approval is 85% for one group and 15% for the other, and ``zipcode`` is
    drawn from two non-overlapping clusters by group, so the proxy screen has
    something true to find.
    """
    rng = np.random.default_rng(3)
    g = rng.choice(["male", "female"], size=n)
    approved = np.where(g == "male", rng.random(n) < 0.85, rng.random(n) < 0.15).astype(int)
    zipcode = np.where(g == "male", rng.normal(1000, 5, n), rng.normal(9000, 5, n))
    return pd.DataFrame(
        {
            "gender": g,
            "approved": approved,
            "income": rng.normal(50_000, 5_000, n),
            "zipcode": zipcode,
        }
    )


#: A historical pattern this test invents, in the shape of HISTORICAL_RISK_PATTERNS.
#: It matches the ``income`` column, which no built-in pattern does, so a call
#: that was served a cached default answer instead of being computed is visible.
_INVENTED_PATTERN = {
    "invented": {
        "keywords": ["income"],
        "pattern_type": "Invented",
        "historical_context": "A pattern this test invented.",
        "affected_groups": ["nobody"],
        "risk_level": HistoricalRiskLevel.CRITICAL,
        "recommendations": ["Ignore this, it is a fixture."],
    }
}


def _audit(df: pd.DataFrame, attrs: List[str], **kw):
    detector = BiasDetector(df, protected_attributes=attrs, outcome_column="approved")
    return _silent(detector.full_audit, **kw)


# ── independent expectations, recomputed from the raw findings ──────────────


def _expected_critical_count(report) -> int:
    """The critical rule, applied to the finding lists by this test.

    Recomputed here rather than read back from ``critical_issues``, so the count
    the getter returns is checked against the rule and not against itself.
    """
    n = sum(1 for h in report.historical_findings if h.risk_level == HistoricalRiskLevel.CRITICAL)
    n += sum(
        1 for r in report.representation_findings if r.severity == RepresentationSeverity.CRITICAL
    )
    n += sum(
        1
        for d in report.disparity_findings
        if d.significance == SignificanceLevel.HIGHLY_SIGNIFICANT
        and d.effect_interpretation == EffectSizeInterpretation.LARGE
    )
    n += sum(1 for p in report.proxy_findings if p.risk_level == ProxyRiskLevel.CRITICAL)
    return n


def _expected_high_risk_features(report) -> List[str]:
    """Historical and proxy findings at CRITICAL or HIGH, by enum identity."""
    bad_hist = {HistoricalRiskLevel.CRITICAL, HistoricalRiskLevel.HIGH}
    bad_proxy = {ProxyRiskLevel.CRITICAL, ProxyRiskLevel.HIGH}
    features = {h.feature for h in report.historical_findings if h.risk_level in bad_hist}
    features |= {p.feature for p in report.proxy_findings if p.risk_level in bad_proxy}
    return sorted(features)


# ── 1. get_critical_count ───────────────────────────────────────────────────


class TestGetCriticalCount:
    def test_the_fixture_reaches_the_branch_it_is_meant_to(self):
        """The 2-row audit really does run every module and assess nothing."""
        report = _audit(_two_row_frame(), ["gender"])

        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert report.assessment_coverage() == COVERAGE_NONE
        assert report.critical_issues == []

    def test_an_audit_that_assessed_nothing_reports_no_count(self):
        report = _audit(_two_row_frame(), ["gender"])

        count, messages = _caught(report.get_critical_count)

        assert count is None
        assert any("get_critical_count" in m for m in messages)
        assert any("none of them assessed any requested protected attribute" in m for m in messages)

    def test_an_all_null_protected_column_reports_no_count(self):
        report = _audit(_all_null_frame(), ["gender"])

        assert _caught(report.get_critical_count)[0] is None

    def test_an_audit_where_no_module_ran_reports_no_count(self):
        report = _audit(
            _clean_frame(),
            ["cohort"],
            include_historical=False,
            include_representation=False,
            include_disparities=False,
            include_proxies=False,
        )
        count, messages = _caught(report.get_critical_count)

        assert report.execution_coverage() == COVERAGE_NONE
        assert count is None
        assert any("no audit module executed" in m for m in messages)

    def test_a_partial_run_reports_no_count_and_names_what_ran(self):
        report = _audit(
            _clean_frame(), ["cohort"], include_disparities=False, include_proxies=False
        )
        count, messages = _caught(report.get_critical_count)

        assert report.execution_coverage() == COVERAGE_PARTIAL
        assert count is None
        assert any("only these audit modules executed" in m for m in messages)

    # ---- over-correction controls ----

    def test_a_complete_audit_of_a_clean_frame_measures_zero(self):
        """The control. Withholding here would be the worse defect."""
        report = _audit(_clean_frame(), ["cohort"])
        count, messages = _caught(report.get_critical_count)

        assert report.execution_coverage() == COVERAGE_COMPLETE
        assert count == 0
        assert _expected_critical_count(report) == 0
        assert not any("get_critical_count" in m for m in messages)

    def test_a_real_audit_still_counts_its_critical_issues(self):
        report = _audit(_biased_frame(), ["gender"])
        expected = _expected_critical_count(report)
        count, messages = _caught(report.get_critical_count)

        assert expected >= 2, "fixture must carry real critical issues"
        assert count == expected
        assert not any("get_critical_count" in m for m in messages)

    def test_a_finding_is_never_withheld_however_thin_the_coverage(self):
        """Evidence is returned even under coverage that withholds a zero."""
        report = _audit(_biased_frame(), ["gender"], include_proxies=False)

        assert report.execution_coverage() == COVERAGE_PARTIAL
        assert report.critical_issues, "fixture must carry a finding under partial coverage"
        assert _caught(report.get_critical_count)[0] == len(report.critical_issues)


# ── 2. get_high_risk_features ───────────────────────────────────────────────


class TestGetHighRiskFeatures:
    def test_an_audit_that_screened_nothing_reports_no_list(self):
        report = _audit(_two_row_frame(), ["gender"])
        features, messages = _caught(report.get_high_risk_features)

        assert features is None
        assert any("get_high_risk_features" in m for m in messages)
        assert any("would not be a finding that no feature is high risk" in m for m in messages)

    def test_an_all_null_protected_column_reports_no_list(self):
        report = _audit(_all_null_frame(), ["gender"])

        assert _caught(report.get_high_risk_features)[0] is None

    def test_a_misspelled_attribute_reports_no_list(self):
        """Nothing was screened because the column requested does not exist."""
        report = _audit(_clean_frame(), ["cohortt"])

        assert report.attribute_observations == {"cohortt": 0}
        assert _caught(report.get_high_risk_features)[0] is None

    # ---- over-correction controls ----

    def test_a_complete_audit_of_a_clean_frame_measures_an_empty_list(self):
        report = _audit(_clean_frame(), ["cohort"])
        features, messages = _caught(report.get_high_risk_features)

        assert features == []
        assert _expected_high_risk_features(report) == []
        assert not any("get_high_risk_features" in m for m in messages)

    def test_a_real_audit_still_names_its_high_risk_features(self):
        report = _audit(_biased_frame(), ["gender"])
        expected = _expected_high_risk_features(report)
        features, messages = _caught(report.get_high_risk_features)

        assert "zipcode" in expected, "fixture must carry a real proxy"
        assert features == expected
        assert not any("get_high_risk_features" in m for m in messages)


# ── 3. get_explanation ──────────────────────────────────────────────────────


def _risk_item(explanation):
    for item in explanation.explanations:
        if item.metric_name == "Overall Bias Risk Score":
            return item
    raise AssertionError("the bias audit explanation must carry its overall risk item")


class TestGetExplanation:
    def _unassessable(self):
        detector = BiasDetector(
            _two_row_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        return _silent(detector.get_explanation)

    def _healthy(self):
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        return _silent(detector.get_explanation)

    def test_it_does_not_band_a_score_that_aggregates_nothing(self):
        item = _risk_item(self._unassessable())

        assert item.severity == "could_not_check"
        assert "MINIMAL band" not in item.evaluation
        assert "appears suitable for training" not in item.evaluation

    def test_it_says_why_at_the_top_of_the_summary(self):
        explanation = self._unassessable()

        assert explanation.summary.startswith("COULD NOT CHECK")
        assert "is not a measurement of low risk" in explanation.summary

    def test_it_is_never_the_bottom_of_the_severity_scale(self):
        """'info' is what a graded, genuinely benign audit gets."""
        assert self._unassessable().severity != "info"

    def test_it_recommends_re_running_rather_than_monitoring(self):
        item = _risk_item(self._unassessable())

        assert "Re-run the audit" in item.recommendation

    def test_the_audit_recommendations_are_not_swallowed(self):
        """The guard must not delete what the audit did say."""
        recommendations = " ".join(self._unassessable().recommendations)

        assert "none of them assessed anything" in recommendations

    # ---- over-correction controls ----

    def test_a_real_audit_keeps_its_grade_and_its_wording(self):
        explanation = self._healthy()
        item = _risk_item(explanation)

        assert "COULD NOT CHECK" not in explanation.summary
        assert explanation.severity in ("medium", "high", "critical")
        assert item.severity != "could_not_check"
        assert "MEDIUM band" in item.evaluation

    def test_a_real_audit_still_reports_its_score_in_the_summary(self):
        explanation = self._healthy()

        assert "0.50" in explanation.summary


# ── 4. the caches answer the question they were asked ───────────────────────


class TestCachesAreKeyedByTheirArguments:
    def test_analyze_disparities_min_group_size_is_not_ignored(self):
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        loose = _silent(detector.analyze_disparities, min_group_size=30)
        strict = _silent(detector.analyze_disparities, min_group_size=400)

        fresh = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        expected = _silent(fresh.analyze_disparities, min_group_size=400)

        assert len(loose) == 2, "fixture must carry real disparities at the default size"
        assert len(strict) == len(expected) == 0

    def test_identify_proxies_threshold_is_not_ignored(self):
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        permissive = _silent(detector.identify_proxies, correlation_threshold=0.05)
        strict = _silent(detector.identify_proxies, correlation_threshold=0.99)

        fresh = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        expected = _silent(fresh.identify_proxies, correlation_threshold=0.99)

        assert len(permissive) > len(strict)
        assert [p.feature for p in strict] == [p.feature for p in expected]

    def test_an_exploratory_call_no_longer_empties_the_audit(self):
        """The regression this cache defect actually caused.

        One exploratory call at a group size nothing can meet used to answer the
        audit's own call, so ``full_audit`` reported zero disparities over a
        frame carrying two large significant ones, and dropped two critical
        issues with them.
        """
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        exploratory = _silent(detector.analyze_disparities, min_group_size=400)
        report = _silent(detector.full_audit)

        assert exploratory == []
        assert len(report.disparity_findings) == 2
        assert any("significant disparities" in r for r in report.recommendations)

    def test_a_repeat_call_with_the_same_arguments_is_still_memoized(self):
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        first = _silent(detector.identify_proxies)

        assert _silent(detector.identify_proxies) is first

    def test_clear_cache_empties_all_four_caches_and_returns_nothing(self):
        """All four, because clearing three of them looks identical from one."""
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        methods = (
            detector.detect_historical_patterns,
            detector.detect_representation_bias,
            detector.analyze_disparities,
            detector.identify_proxies,
        )
        first = [_silent(m) for m in methods]

        assert detector.clear_cache() is None

        for method, before in zip(methods, first):
            assert _silent(method) is not before, f"{method.__name__} was not cleared"

    def test_custom_patterns_take_effect_and_do_not_poison_the_default(self):
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        default_first = _silent(detector.detect_historical_patterns)
        custom = _silent(
            detector.detect_historical_patterns,
            custom_patterns=_INVENTED_PATTERN,
        )
        default_again = _silent(detector.detect_historical_patterns)

        assert [h.feature for h in default_first] == ["zipcode"]
        # The custom call must be ANSWERED, not served the default result.
        assert "income" in [h.feature for h in custom]
        assert [h.feature for h in default_again] == [h.feature for h in default_first]

    def test_min_confidence_is_not_ignored(self):
        detector = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        permissive = _silent(detector.detect_historical_patterns, min_confidence=0.3)
        strict = _silent(detector.detect_historical_patterns, min_confidence=0.9)

        fresh = BiasDetector(
            _biased_frame(), protected_attributes=["gender"], outcome_column="approved"
        )
        expected = _silent(fresh.detect_historical_patterns, min_confidence=0.9)

        assert len(permissive) == 1
        assert len(strict) == len(expected) == 0


# ── 4b. the record every other surface reads ────────────────────────────────


class TestTheCoverageRecordItself:
    """The three-state record ``get_critical_count`` and the canvas read.

    Pinned here as well as in ``test_bgl_stage2_s2g09.py`` because this file's
    refusals are derived from it: a coverage method that answered 'complete'
    for everything would make every guard above inert while leaving them green
    on their own terms.
    """

    def test_an_unassessable_audit_reports_none_of_its_attributes_assessed(self):
        report = _audit(_two_row_frame(), ["gender"])

        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert report.assessment_coverage() == COVERAGE_NONE
        assert report.unassessable_attributes() == ["gender"]

    def test_a_clean_audit_reports_complete_coverage_and_no_gaps(self):
        report = _audit(_clean_frame(), ["cohort"])

        assert report.execution_coverage() == COVERAGE_COMPLETE
        assert report.assessment_coverage() == COVERAGE_COMPLETE
        assert report.unassessable_attributes() == []

    def test_one_assessable_and_one_starved_attribute_is_partial(self):
        frame = _clean_frame()
        region = [None] * len(frame)
        region[:5] = ["north", "north", "north", "south", "south"]
        frame["region"] = pd.Series(region, dtype=object)
        report = _audit(frame, ["cohort", "region"])

        assert report.assessment_coverage() == COVERAGE_PARTIAL
        assert report.unassessable_attributes() == ["region"]

    def test_the_text_summary_carries_the_refusal_to_a_reader(self):
        """The sub-module warnings are gone by the time anyone prints this."""
        text = _audit(_two_row_frame(), ["gender"]).summary()

        assert "Overall Risk Score: 0.0%" in text
        assert "none of them assessed anything" in text
        assert "not a measurement of low risk" in text

    def test_the_text_summary_of_a_clean_audit_carries_no_such_line(self):
        text = _audit(_clean_frame(), ["cohort"]).summary()

        assert "Overall Risk Score: 0.0%" in text
        assert "assessed anything" not in text
        assert "does not record" not in text

    def test_the_json_export_carries_both_coverage_words(self):
        import json

        exported = json.loads(_audit(_two_row_frame(), ["gender"]).to_json())

        assert exported["execution_coverage"] == COVERAGE_UNASSESSED
        assert exported["assessment_coverage"] == COVERAGE_NONE
        assert exported["attribute_assessed"] == {"gender": False}
        assert exported["overall_risk_score"] == 0.0

    def test_the_json_export_of_a_clean_audit_says_complete(self):
        import json

        exported = json.loads(_audit(_clean_frame(), ["cohort"]).to_json())

        assert exported["execution_coverage"] == COVERAGE_COMPLETE
        assert exported["assessment_coverage"] == COVERAGE_COMPLETE
        assert exported["attribute_assessed"] == {"cohort": True}

    def test_an_audit_with_no_protected_attribute_requested_assessed_none(self):
        """The empty-record branch: nothing was requested, so nothing was assessed."""
        detector = _silent(
            BiasDetector,
            _clean_frame(),
            protected_attributes=[],
            outcome_column="approved",
        )
        report = _silent(detector.full_audit)

        assert report.attribute_assessed == {}
        assert report.assessment_coverage() == COVERAGE_NONE
        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert _caught(report.get_critical_count)[0] is None

    def test_a_hand_built_report_says_unrecorded_rather_than_either_state(self):
        from vfairness.preprocessing.bias_detection.detector import BiasAuditReport

        report = BiasAuditReport(
            timestamp="2026-09-17T00:00:00",
            dataset_info={},
            protected_attributes=["gender"],
            historical_findings=[],
            representation_findings=[],
            disparity_findings=[],
            proxy_findings=[],
            overall_risk_score=0.0,
            critical_issues=[],
            recommendations=[],
        )

        assert report.execution_coverage() == COVERAGE_UNRECORDED
        assert report.assessment_coverage() == COVERAGE_UNRECORDED
        assert report.unassessable_attributes() is None
        assert _caught(report.get_critical_count)[0] is None
        assert _caught(report.get_high_risk_features)[0] is None


# ── 5. get_dataset_summary describes the frame and invents nothing ──────────


class TestGetDatasetSummary:
    def test_an_empty_frame_has_no_missing_rate_rather_than_a_zero_one(self):
        detector = _silent(
            BiasDetector,
            pd.DataFrame({"gender": pd.Series(dtype=object), "approved": pd.Series(dtype=float)}),
            protected_attributes=["gender"],
        )
        summary = detector.get_dataset_summary()

        assert summary["n_rows"] == 0
        assert np.isnan(summary["missing_rates"]["gender"])
        assert np.isnan(summary["missing_rates"]["approved"])

    def test_an_all_null_column_has_no_distribution_and_a_full_missing_rate(self):
        summary = BiasDetector(
            _all_null_frame(), protected_attributes=["gender"]
        ).get_dataset_summary()

        assert summary["protected_attribute_distributions"]["gender"] == {}
        assert summary["missing_rates"]["gender"] == 1.0

    def test_a_populated_column_reports_its_real_proportions(self):
        frame = pd.DataFrame({"gender": ["M"] * 30 + ["F"] * 70, "approved": [1] * 100})
        summary = BiasDetector(frame, protected_attributes=["gender"]).get_dataset_summary()

        assert summary["protected_attribute_distributions"]["gender"] == {"F": 0.70, "M": 0.30}
        assert summary["missing_rates"]["gender"] == 0.0
        assert summary["n_rows"] == 100


# ── 6. STILL OPEN: the canvas grades what the report refuses ────────────────


def _visible(svg: str) -> str:
    body = svg.split("</metadata>")[-1]
    return " | ".join(html.unescape(t) for t in re.findall(r">([^<>]+)<", body) if t.strip())


class TestTheCanvasStillGradesAnAuditThatAssessedNothing:
    """DEFECT OPEN, batch g021, 2026-09-17. The fix is in rendering/adapters.py.

    ``BiasDetector(2-row frame).full_audit().to_svg()`` renders "OVERALL RISK 0%
    MINIMAL" in emerald, over a report whose ``execution_coverage()`` is
    'ran_but_assessed_nothing'. Two independent causes in
    ``rendering/adapters.py``:

    * ``_COVERAGE_STATES`` lists four words and the detector can return five, so
      ``_audit_coverage`` reads the fifth as 'unrecorded'. That much is
      fail-safe on its own.
    * ``nothing_measured = n_findings == 0 and ...`` counts the one
      ``RepresentationSeverity.INSUFFICIENT_DATA`` result as a finding, which
      disarms the guard and lets the 0.0 reach the badge. This is exactly the
      shape ``_calculate_overall_risk`` fixed one layer down, where a single
      INSUFFICIENT_DATA append disarmed ``if not risk_components``.

    Both were marked xfail(strict) so the day the canvas withholds that badge,
    they would turn into failures asking for the markers to be removed. That day
    was 2026-09-25 and the markers are gone; the diagnosis above is kept verbatim
    because it is the record of what was wrong, and both clauses were correct.
    """

    def _svg(self) -> str:
        report = _audit(_two_row_frame(), ["gender"])
        assert report.execution_coverage() == COVERAGE_UNASSESSED
        return _silent(report.to_svg)

    # xfail(strict) removed 2026-09-25: the canvas now withholds the badge.
    def test_the_canvas_withholds_the_badge_when_nothing_was_assessed(self):
        pytest.importorskip("jinja2", reason="SVG rendering requires jinja2")
        seen = _visible(self._svg())

        assert "MINIMAL" not in seen
        assert "NOT ASSESSABLE" in seen

    # xfail(strict) removed 2026-09-25: _COVERAGE_STATES holds all five words.
    def test_the_adapter_knows_every_coverage_word_the_detector_returns(self):
        pytest.importorskip("jinja2", reason="SVG rendering requires jinja2")
        from vfairness.rendering import adapters

        detector_words = {
            COVERAGE_COMPLETE,
            COVERAGE_PARTIAL,
            COVERAGE_NONE,
            COVERAGE_UNRECORDED,
            COVERAGE_UNASSESSED,
        }
        assert detector_words <= set(adapters._COVERAGE_STATES)
