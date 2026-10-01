"""BGL wave-4 PINS, batch T1-F5-preproc (2026-09-30): preprocessing/bias_detection.

Seven grades an independent auditor overturned, all seven reproduced by execution
at HEAD 21e07b5 before anything was changed. Every class below pins ONE of them
plus the sibling door found in the same unit, and every class carries a CONTROL
asserting the healthy case's REAL number so a guard that refuses everything
cannot pass.

The units: ``statistical.detect_temporal_drift``,
``representation.calculate_representation_ratio``,
``representation.compare_to_benchmark``,
``historical.attribute_historical_pattern``,
``historical.domain_historical_context``,
``historical.check_geographic_redlining_risk``,
``geographic_data.fetch_holc_data``.
"""

from __future__ import annotations

import warnings
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection import geographic_data as gd
from vfairness.preprocessing.bias_detection.historical import (
    attribute_historical_pattern,
    check_geographic_redlining_risk,
    domain_historical_context,
)
from vfairness.preprocessing.bias_detection.representation import (
    calculate_representation_ratio,
    compare_to_benchmark,
)
from vfairness.preprocessing.bias_detection.statistical import detect_temporal_drift


def _caught(fn, *a, **kw):
    """Run ``fn`` and return ``(result, [warning messages])``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(w.message) for w in rec]


# ── ROW 6. statistical.detect_temporal_drift ──────────────────────────────────
#
# MEASURED BEFORE (OMP_NUM_THREADS=1, 200 rows over two instants, gender 50/50 in
# BOTH cohorts so the composition PSI is 0.00 and the selection rate is the only
# thing reported):
#
#   decision recorded early at 0.50, NEVER RECORDED late
#     -> severity 'warn', selectionRateFirst 0.5, selectionRateLast 0.0,
#        selectionRateShift -0.5, "the selection rate moved -50 points",
#        warnings [], and no key anywhere counting the 100 unreadable rows.
#   decision NEVER RECORDED early, 0.90 late
#     -> selectionRateFirst 0.0, shift +0.9, "+90 points", severity 'warn'.
#   the SIBLING door, a TEXT-typed score column (an ordinary CSV read):
#     '0.90'/'0.10' -> selectionRateFirst 0.0, selectionRateLast 0.0, shift 0.0,
#        severity 'pass', "the selection rate moved +0 points", ZERO warnings,
#        for the IDENTICAL values that as float64 publish 0.5 -> 0.9 and 'warn'.

_N = 100
_GENDER = ["m"] * 50 + ["f"] * 50
_DATES = ["2024-01-01"] * _N + ["2024-06-01"] * _N
_DEC_HALF = [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25  # rate 0.50
_DEC_NINE = [1] * 45 + [0] * 5 + [1] * 45 + [0] * 5  # rate 0.90


def _drift(decision: List[Any]):
    df = pd.DataFrame({"app_date": _DATES, "gender": _GENDER + _GENDER, "decision": decision})
    return _caught(
        detect_temporal_drift, df, ["gender"], time_column="app_date", prediction="decision"
    )


class TestTemporalDriftSelectionRateNeedsReadableOutcomes:
    """An unreadable prediction is not a rejection, and a text column is readable."""

    @pytest.mark.parametrize(
        "decision,label",
        [
            (_DEC_HALF + [None] * _N, "never recorded late"),
            ([None] * _N + _DEC_NINE, "never recorded early"),
        ],
    )
    def test_a_cohort_with_no_readable_outcome_publishes_no_rate(self, decision, label):
        res, warns = _drift(decision)
        entry = res["drift"][0]
        # The three published numbers are None, not the imputed 0.0 / -0.5.
        assert entry["selectionRateFirst"] is None, label
        assert entry["selectionRateLast"] is None, label
        assert entry["selectionRateShift"] is None, label
        assert entry["selectionRateStatus"] == "could_not_measure"
        assert "could-not-check" in entry["selectionRateNotMeasuredReason"]
        # The ROW count is where a reader looks, and it is the real 100.
        assert res["nRowsWithUnreadablePrediction"] == _N
        assert {entry["nPredictionsReadFirst"], entry["nPredictionsReadLast"]} == {0, _N}
        # The severity was NOT escalated off the fabricated half-point swing.
        assert entry["severity"] == "pass"
        # And it is visible in the prose a reader actually reads.
        assert "could NOT be measured" in entry["plain"]
        assert "points" not in entry["plain"].split("could NOT be measured")[0]
        assert any("no readable number" in w for w in warns), warns
        assert any("could-not-check" in w for w in warns), warns

    def test_the_sibling_door_a_text_typed_score_column_is_read_as_scores(self):
        """The dtype must not decide the verdict. Same values, str and float64."""
        as_text = [f"{v:.2f}" for v in ([0.9] * 25 + [0.1] * 25) * 2] + [
            f"{v:.2f}" for v in ([0.95] * 45 + [0.05] * 5) * 2
        ]
        as_float = [float(v) for v in as_text]
        text_res, text_warns = _drift(as_text)
        float_res, _ = _drift(as_float)
        t_entry, f_entry = text_res["drift"][0], float_res["drift"][0]
        # The REAL numbers, not the all-rejections 0.0 / 0.0 / +0.
        assert t_entry["selectionRateFirst"] == 0.5
        assert t_entry["selectionRateLast"] == 0.9
        assert t_entry["selectionRateShift"] == 0.4
        assert t_entry["severity"] == "warn"
        assert text_res["nRowsWithUnreadablePrediction"] == 0
        assert text_warns == []
        # And they agree with the float column value for value.
        for key in (
            "selectionRateFirst",
            "selectionRateLast",
            "selectionRateShift",
            "severity",
            "selectionRateStatus",
        ):
            assert t_entry[key] == f_entry[key], key

    @pytest.mark.parametrize(
        "absent",
        [None, float("nan"), pd.NA, pd.NaT, "", "   ", "None"],
        ids=["none", "nan", "pdNA", "pdNaT", "blank", "space", "literal-None"],
    )
    def test_every_absence_door_in_a_token_column_is_unreadable_not_a_rejection(self, absent):
        """Six doors plus whitespace. 'none' is falsy, the rest are unreadable."""
        decision = ["approved"] * 25 + ["denied"] * 25 + ["approved"] * 25 + ["denied"] * 25
        res, _ = _drift(decision + [absent] * _N)
        entry = res["drift"][0]
        if str(absent).strip().lower() == "none" and isinstance(absent, str):
            # A reader writing 'none' in a decision column means "not selected".
            assert entry["selectionRateStatus"] == "measured"
            assert entry["selectionRateLast"] == 0.0
        else:
            assert entry["selectionRateStatus"] == "could_not_measure", absent
            assert res["nRowsWithUnreadablePrediction"] == _N

    # ── CONTROLS: the healthy cases keep their REAL numbers ────────────────────

    def test_control_a_measured_flat_rate_still_publishes_its_real_zero_shift(self):
        res, warns = _drift(_DEC_HALF + _DEC_HALF)
        entry = res["drift"][0]
        assert entry["selectionRateFirst"] == 0.5
        assert entry["selectionRateLast"] == 0.5
        assert entry["selectionRateShift"] == 0.0
        assert entry["selectionRateStatus"] == "measured"
        assert entry["severity"] == "pass"
        assert res["nRowsWithUnreadablePrediction"] == 0
        assert warns == []

    def test_control_a_real_swing_still_publishes_its_real_forty_points(self):
        res, warns = _drift(_DEC_HALF + _DEC_NINE)
        entry = res["drift"][0]
        assert entry["selectionRateFirst"] == 0.5
        assert entry["selectionRateLast"] == 0.9
        assert entry["selectionRateShift"] == 0.4
        assert entry["selectionRateStatus"] == "measured"
        assert entry["severity"] == "warn"
        assert "the selection rate moved +40 points." in entry["plain"]
        assert warns == []

    def test_control_a_partly_unreadable_column_still_measures_and_says_so(self):
        """Enough readable outcomes on both sides: a REAL rate, plus the count."""
        decision = (_DEC_HALF[:80] + [None] * 20) + (_DEC_NINE[:80] + [None] * 20)
        res, warns = _drift(decision)
        entry = res["drift"][0]
        assert entry["selectionRateStatus"] == "measured"
        assert entry["nPredictionsReadFirst"] == 80
        assert entry["nPredictionsReadLast"] == 80
        assert res["nRowsWithUnreadablePrediction"] == 40
        assert entry["selectionRateFirst"] == pytest.approx(
            float(np.mean(_DEC_HALF[:80])), abs=1e-4
        )
        assert "whose outcome could be read" in entry["plain"]
        assert any("no readable number" in w for w in warns), warns


# ── ROWS 8 and 9. representation.calculate_representation_ratio /
#    representation.compare_to_benchmark ─────────────────────────────────────────
#
# MEASURED BEFORE, 300 rows with 'm' holding 150 of them (actual_proportion 0.50),
# against the docstring's own "Expected proportion (0-1)":
#
#   bp=2.0   -> representation_ratio 0.25,  deficit_count 450,   'measured', warn 0
#   bp=5.0   -> 0.1,                        deficit_count 1350
#   bp=100.0 -> 0.005,                      deficit_count 29850
#   bp=1e9   -> 5e-10,                      deficit_count 299999999850
#
# A deficit of 450 people in a 300-row frame. And the verdict could not have
# disagreed: actual_proportion cannot exceed 1.0, so `ratio < 0.8` is True for
# EVERY possible dataset once bp > 1.25 and `ratio > 1.2` can never fire once
# bp > 0.8333. Secondary door, same unit: min_group_size=nan silently disabled the
# sample floor (`n_total < nan` is False) and reported 'measured' with no refusal.
#
# compare_to_benchmark, the sibling entry point, same 300 rows:
#   {'m': 2.0,  'f': 0.5}  -> ratios {'f': 1.0,  'm': 0.25}, gaps {'f': 0.0, 'm': -1.5}
#   {'m': 50.0, 'f': 50.0} -> ratios {'f': 0.01, 'm': 0.01}, gaps {'f': -49.5, 'm': -49.5}
# both 'measured' with groups_not_measured [], and the one warning emitted stated a
# reason FALSE for the input: "0.0% of the rows fall in groups the benchmark never
# mentions" when 0.0% is the number it printed. The THIRD entry point,
# detect_representation_bias, already renormalised with a warning.

_DF_300 = pd.DataFrame({"gender": ["m"] * 150 + ["f"] * 150})


class TestRepresentationBenchmarkMustBeOnTheShareScale:
    """A benchmark above the 0-1 scale is not an expected proportion."""

    @pytest.mark.parametrize("bp", [1.25, 2.0, 5.0, 100.0, 1e9])
    def test_a_benchmark_above_one_is_refused_not_divided_by(self, bp):
        res, warns = _caught(calculate_representation_ratio, _DF_300, "gender", "m", bp)
        assert res["representation_ratio"] is None, bp
        assert res["deficit_count"] is None, bp
        assert res["is_underrepresented"] is None, bp
        assert res["is_overrepresented"] is None, bp
        assert res["measurement_status"] == "could_not_measure"
        assert "above the 0-1 share scale" in res["not_measured_reason"]
        # The observed share is still a measurement and is still reported.
        assert res["actual_proportion"] == 0.5
        assert any("above the 0-1 share scale" in w for w in warns), warns

    def test_a_nan_floor_cannot_switch_the_sample_gate_off_in_silence(self):
        res, warns = _caught(
            calculate_representation_ratio,
            _DF_300,
            "gender",
            "m",
            0.5,
            min_group_size=float("nan"),
        )
        assert res["measurement_status"] == "could_not_measure"
        assert "not a comparable number" in res["not_measured_reason"]
        assert res["representation_ratio"] is None
        assert any("sample-size floor could not be applied" in w for w in warns), warns

    @pytest.mark.parametrize(
        "benchmark,expected_ratios",
        [
            ({"m": 50.0, "f": 50.0}, {"m": 1.0, "f": 1.0}),
            ({"m": 2.0, "f": 0.5}, {"m": 0.625, "f": 2.5}),
        ],
    )
    def test_compare_to_benchmark_renormalises_an_off_scale_benchmark(
        self, benchmark, expected_ratios
    ):
        res, warns = _caught(compare_to_benchmark, _DF_300, "gender", benchmark)
        for group, want in expected_ratios.items():
            assert res["representation_ratios"][group] == pytest.approx(want, abs=1e-9), group
        # A gap between two proportions lives in [-1, 1]. -49.5 did not.
        for group, gap in dict(res["gaps"]).items():
            assert -1.0 <= gap <= 1.0, (group, gap)
        assert res["benchmark_renormalised_from"] == pytest.approx(sum(benchmark.values()))
        # The caller's own numbers are kept, and what was divided by is published.
        assert res["benchmark_distribution"] == benchmark
        assert sum(res["benchmark_distribution_used"].values()) == pytest.approx(1.0)
        assert any("not a population distribution" in w for w in warns), warns

    def test_the_chi_squared_reason_names_the_real_cause(self):
        """0.0% uncovered rows was published as the reason 0.0% of the time."""
        # A benchmark that is off-scale AND has a group the data does not hold, so
        # the totals still disagree after renormalisation is not the story: the
        # helper is exercised directly for the off-scale shape it used to misreport.
        from vfairness.preprocessing.bias_detection.representation import (
            _chi_squared_representation_test,
        )

        stat, pval, reason = _chi_squared_representation_test(
            {"m": 0.5, "f": 0.5}, {"m": 50.0, "f": 50.0}, 300
        )
        assert stat is None and pval is None
        assert "not 1.0, so it is not a population distribution" in reason
        assert "every observed row IS covered" in reason
        assert "fall in groups the benchmark never mentions" not in reason

    # ── CONTROLS: the healthy cases keep their REAL numbers ────────────────────

    @pytest.mark.parametrize(
        "bp,ratio,deficit,under",
        [(0.5, 1.0, 0, False), (0.8, 0.625, 90, True), (1.0, 0.5, 150, True)],
    )
    def test_control_a_real_share_still_publishes_its_real_ratio(self, bp, ratio, deficit, under):
        res, warns = _caught(calculate_representation_ratio, _DF_300, "gender", "m", bp)
        assert res["representation_ratio"] == pytest.approx(ratio)
        assert res["deficit_count"] == deficit
        assert res["is_underrepresented"] is under
        assert res["measurement_status"] == "measured"
        assert warns == []

    @pytest.mark.parametrize("mgs", [30, 0, -5])
    def test_control_a_deliberately_low_floor_is_not_refused(self, mgs):
        res, warns = _caught(
            calculate_representation_ratio, _DF_300, "gender", "m", 0.5, min_group_size=mgs
        )
        assert res["measurement_status"] == "measured"
        assert res["representation_ratio"] == 1.0
        assert warns == []

    def test_control_a_real_distribution_is_not_renormalised(self):
        res, warns = _caught(compare_to_benchmark, _DF_300, "gender", {"m": 0.5, "f": 0.5})
        assert dict(res["representation_ratios"]) == {"m": 1.0, "f": 1.0}
        assert dict(res["gaps"]) == {"m": 0.0, "f": 0.0}
        assert res["benchmark_renormalised_from"] is None
        assert res["measurement_status"] == "measured"
        assert res["pvalue"] == pytest.approx(1.0)
        assert res["statistically_significant"] is False
        assert warns == []

    def test_control_a_zero_expected_share_is_still_named_not_renormalised_away(self):
        """The BGL5 per-group refusal must survive the new renormalisation."""
        res, warns = _caught(
            compare_to_benchmark, _DF_300, "gender", {"m": 0.5, "f": 0.5, "x": 0.0}
        )
        assert res["groups_not_measured"] == ["x"]
        assert res["measurement_status"] == "partial"
        assert res["benchmark_renormalised_from"] is None
        assert dict(res["representation_ratios"]) == {"m": 1.0, "f": 1.0}
        assert any("no representation ratio or gap exists" in w for w in warns), warns


# ── ROWS 13 and 14. historical.attribute_historical_pattern /
#    historical.domain_historical_context ────────────────────────────────────────
#
# MEASURED BEFORE. An attribute name carrying TWO protected classes resolved to
# ONE by alias length and token order, with no warning and no disclosure, and the
# caller turned that pick into a citation-backed legal precedent:
#
#   'race_age_group'         -> class 'age',      'hiring_age',  citing the ADEA
#                               (RACE silently discarded)
#   'age_race'               -> class 'race',     'hiring_race', citing
#                               Bertrand & Mullainathan (2004)
#   'disability_race'        -> race;  'race_disability_status' -> disability (ADA)
#   'gender_age'/'age_gender'-> gender (the Amazon recruiting tool)
#   'race_zip_code'          -> 'geographic', which has no 'hiring' entry, so
#                               attribute_historical_pattern returned None: NO
#                               precedent at all for a column naming race in a
#                               hiring audit, which the docstring tells a reader
#                               to read as "no documented precedent".
#
# On the domain side, two DOCUMENTED domains matching both published one of them,
# picked by character length, with warnings 0:
#   'university hospital clinical trials'      -> 'education', citing Buolamwini &
#       Gebru's 'Gender Shades' (a facial-recognition paper) as the documented
#       precedent for a clinical-trial audit ('university' 10 chars beats
#       'clinical' 8). Word order does not change it, so it is the tiebreak.
#   'criminal justice education program' -> education, not justice
#   'medical school admissions'          -> education
#   'employment credit screening'        -> hiring, dropping lending


class TestHistoricalPrecedentDisclosesAnAmbiguousMatch:
    """Two protected classes is a could-not-check; two domains is a disclosure."""

    @pytest.mark.parametrize(
        "column,classes",
        [
            ("race_age_group", {"race", "age"}),
            ("age_race", {"race", "age"}),
            ("disability_race", {"disability", "race"}),
            ("race_disability_status", {"disability", "race"}),
            ("gender_age", {"gender", "age"}),
            ("age_gender", {"gender", "age"}),
            ("race_zip_code", {"race", "geographic"}),
        ],
    )
    def test_two_protected_classes_yields_a_could_not_check_not_a_precedent(self, column, classes):
        out, warns = _caught(attribute_historical_pattern, column, "hiring")
        assert out is not None, column
        # No precedent and no legal citation is asserted.
        assert out["pattern_id"] is None
        assert out["citations"] == []
        assert out["attribute_class"] is None
        assert out["measurement_status"] == "could_not_check"
        assert set(out["attribute_classes_matched"]) == classes
        # The reason is on the envelope a reader renders, not only in a warning.
        assert "could not be determined" in out["not_matched_reason"]
        assert "could not be determined" in out["summary"]
        # Every key the success shape carries is present, so a consumer indexing
        # them cannot break on the third state.
        for key in ("pattern_id", "label", "summary", "citations", "domain", "jurisdiction"):
            assert key in out, key
        assert any("carries 2 protected attribute classes" in w for w in warns), warns

    @pytest.mark.parametrize(
        "phrase,picked,also",
        [
            ("university hospital clinical trials", "education", "healthcare"),
            ("clinical trials at a university hospital", "education", "healthcare"),
            ("criminal justice education program", "education", "justice"),
            ("medical school admissions", "education", "healthcare"),
            ("employment credit screening", "hiring", "lending"),
            ("credit screening for employment", "hiring", "lending"),
        ],
    )
    def test_two_documented_domains_are_all_named_and_warned(self, phrase, picked, also):
        out, warns = _caught(domain_historical_context, phrase)
        assert out is not None, phrase
        assert out["domain"] == picked
        assert out["domain_match_is_ambiguous"] is True
        assert also in out["domains_matched"]
        assert picked in out["domains_matched"]
        # The banner is declared one candidate of several, where a reader looks.
        assert "one candidate" in out["disclosure"]
        assert "could-not-check" in out["disclosure"]
        assert any("documented domains" in w for w in warns), warns

    # ── CONTROLS: the healthy cases still resolve, to their REAL precedent ──────

    @pytest.mark.parametrize(
        "column,cls,pattern",
        [
            ("age_group", "age", "lending_age"),
            ("customerAge", "age", "lending_age"),
            ("race_ethnicity", "race", "lending_race_redlining"),
            ("census_tract_id", "geographic", "lending_geographic_redlining"),
            ("AGE", "age", "lending_age"),
            ("zip_code", "geographic", "lending_geographic_redlining"),
            ("Gender", "gender", "lending_gender"),
        ],
    )
    def test_control_one_protected_class_still_earns_its_real_precedent(self, column, cls, pattern):
        out, warns = _caught(attribute_historical_pattern, column, "lending")
        assert out["pattern_id"] == pattern
        assert out["attribute_class"] == cls
        assert out["measurement_status"] == "measured"
        assert out["citations"], out
        assert warns == []

    @pytest.mark.parametrize(
        "column",
        [
            "primary_language",
            "average_salary",
            "message_count",
            "percentage_complete",
            "usage_tier",
            "package_weight",
            "storage_class",
            "triage_score",
            "contract_type",
        ],
    )
    def test_control_the_nine_false_classifications_stay_silent(self, column):
        assert attribute_historical_pattern(column, "hiring") is None

    @pytest.mark.parametrize(
        "phrase,domain",
        [
            ("hr analytics", "hiring"),
            ("online hiring platform", "hiring"),
            ("bail", "justice"),
            ("mortgage bail-in", "lending"),
            ("health insurance", "insurance"),
            ("hiring", "hiring"),
            ("lending", "lending"),
        ],
    )
    def test_control_a_single_documented_domain_resolves_and_is_not_flagged(self, phrase, domain):
        out, warns = _caught(domain_historical_context, phrase)
        assert out["domain"] == domain
        assert out["citations"], out
        if not out["domain_match_is_ambiguous"]:
            assert warns == []
            assert out["domains_matched"] == [domain]
        else:
            # 'mortgage bail-in' and 'health insurance' are the pinned longest-match
            # cases: still resolved to the SAME domain, now with the other match named.
            assert domain in out["domains_matched"]
            assert out["domain"] == domain

    @pytest.mark.parametrize(
        "phrase",
        [
            "chronic disease management",
            "threat detection",
            "anthropology research",
            "shrinkage analytics",
            "xhrx",
            "bailout underwriting",
        ],
    )
    def test_control_the_six_false_domain_resolutions_stay_silent(self, phrase):
        assert domain_historical_context(phrase) is None


# ── ROW 7. historical.check_geographic_redlining_risk ─────────────────────────
#
# MEASURED BEFORE, 60 rows over 6 ZIPs at a flat 0.50 approval rate: score 0.0,
# assessment_coverage 'complete', outcome_analysis n_geographic_units 6 and
# n_geographic_units_with_a_finite_mean 6, not_assessed [], partial [], warnings
# [], and `[k for k in r if 'row' in k.lower()]` EMPTY.
#
#   (a) add 60 MORE rows whose zip is None and which are 100 percent DENIED (half
#       the file, and the whole disparity) -> the published result is
#       BYTE-IDENTICAL to the 60-row control.
#   (b) the mirror door: make those 60 rows' zip the literal string 'None' (or ''
#       or '   ') and they become a SEVENTH geographic unit -> score 0.4,
#       n_geographic_units 7, 7 of 7 with a finite mean, outcome_range 0.5,
#       coverage 'complete', findings ['Large outcome disparity across geographic
#       units (range: 50.00%). Potential redlining pattern.'], ZERO warnings. A
#       redlining finding against a place that is not a place.

_ZIPS = [f"z{i}" for i in range(6) for _ in range(10)]
_FLAT = [1, 0] * 30


class TestRedliningDisclosesRowsWithNoGeography:
    """Absence is not a place, and it is not invisible either."""

    @pytest.mark.parametrize(
        "absent",
        [None, float("nan"), pd.NA, pd.NaT, "", "   ", "None"],
        ids=["none", "nan", "pdNA", "pdNaT", "blank", "space", "literal-None"],
    )
    @pytest.mark.parametrize("outcome", [0, 1], ids=["all-denied", "all-approved"])
    def test_geography_less_rows_are_counted_and_never_become_a_unit(self, absent, outcome):
        df = pd.concat(
            [
                pd.DataFrame({"zip": _ZIPS, "approved": _FLAT}),
                pd.DataFrame({"zip": [absent] * 60, "approved": [outcome] * 60}),
            ],
            ignore_index=True,
        )
        res, warns = _caught(check_geographic_redlining_risk, df, "zip", outcome_column="approved")
        # The ROW axis, where a reader looks, with the real counts.
        assert res["n_rows_with_no_readable_geography"] == 60
        assert res["n_rows_with_a_readable_geography"] == 60
        # Absence never became a geographic unit, in either direction.
        assert res["outcome_analysis"]["n_geographic_units"] == 6
        assert res["outcome_analysis"]["n_geographic_units_with_a_finite_mean"] == 6
        assert res["outcome_analysis"]["outcome_range"] == pytest.approx(0.0)
        # No fabricated redlining finding off a place that is not a place.
        assert res["redlining_risk_score"] == pytest.approx(0.0)
        assert res["findings"] == []
        # And the verdict cannot read "complete" over half a file.
        assert res["assessment_coverage"] == "partial"
        assert [e["component"] for e in res["components_partially_assessed"]] == ["all"]
        assert "could-not-check" in res["components_partially_assessed"][0]["reason"]
        assert any("carry no readable value" in w for w in warns), warns

    # ── CONTROLS: the healthy cases keep their REAL numbers ────────────────────

    def test_control_a_whole_file_is_still_complete_with_its_real_zero(self):
        res, warns = _caught(
            check_geographic_redlining_risk,
            pd.DataFrame({"zip": _ZIPS, "approved": _FLAT}),
            "zip",
            outcome_column="approved",
        )
        assert res["assessment_coverage"] == "complete"
        assert res["n_rows_with_no_readable_geography"] == 0
        assert res["n_rows_with_a_readable_geography"] == 60
        assert res["redlining_risk_score"] == pytest.approx(0.0)
        assert res["outcome_analysis"]["outcome_range"] == pytest.approx(0.0)
        assert res["components_partially_assessed"] == []
        assert warns == []

    def test_control_a_real_full_range_disparity_still_scores_and_finds(self):
        approved = [1] * 30 + [0] * 30
        res, warns = _caught(
            check_geographic_redlining_risk,
            pd.DataFrame({"zip": _ZIPS, "approved": approved}),
            "zip",
            outcome_column="approved",
        )
        assert res["redlining_risk_score"] == pytest.approx(0.4)
        assert res["outcome_analysis"]["outcome_range"] == pytest.approx(1.0)
        assert "range: 100.00%" in res["findings"][0]
        assert res["assessment_coverage"] == "complete"
        assert warns == []

    def test_control_a_real_disparity_survives_geography_less_rows(self):
        """The finding must still fire on the rows that DO have a geography."""
        approved = [1] * 30 + [0] * 30
        df = pd.concat(
            [
                pd.DataFrame({"zip": _ZIPS, "approved": approved}),
                pd.DataFrame({"zip": [None] * 20, "approved": [1] * 20}),
            ],
            ignore_index=True,
        )
        res, _ = _caught(check_geographic_redlining_risk, df, "zip", outcome_column="approved")
        assert res["redlining_risk_score"] == pytest.approx(0.4)
        assert res["outcome_analysis"]["outcome_range"] == pytest.approx(1.0)
        assert "range: 100.00%" in res["findings"][0]
        assert res["assessment_coverage"] == "partial"
        assert res["n_rows_with_no_readable_geography"] == 20


# ── ROW 12. geographic_data.fetch_holc_data ───────────────────────────────────
#
# MEASURED BEFORE. The guard is `isinstance(payload.get(key), list)` plus a
# non-empty test: TYPE AND LENGTH, with no check on the list's CONTENTS, while the
# docstring's contract is "a body that is not a HOLC document is REFUSED". All
# three returned the document with ZERO warnings, counted through this function's
# OWN documented reader len(data.get('areas', [])) / len(features):
#   {'features': ['Not Found']}                  -> returned dict, count 1
#   {'features': [{'foo': 1}, {'bar': 2}, None]} -> returned dict, count 3
#   {'areas': [0, 0, 0, 0, 0]}                   -> returned dict, count 5
# So the earlier fix closed the door producing a measured ZERO from a body never
# read and left open the door producing a measured NON-ZERO count of HOLC graded
# areas from a body never read, in a redlining module, in silence, without even
# the disclosure the empty case gets. parse_holc_grade(None) and
# parse_holc_grade(1) raise AttributeError, so it crashes the next step too.


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeRequests:
    def __init__(self, payload):
        self.payload = payload
        self.exceptions = gd.requests.exceptions

    def get(self, url, timeout=None):
        return _FakeResp(self.payload)


def _holc(payload):
    """Drive the real function against a stubbed transport. No repo file touched."""
    original = gd.requests
    gd.requests = _FakeRequests(payload)
    try:
        return _caught(gd.fetch_holc_data, "Detroit", "MI")
    finally:
        gd.requests = original


class TestHolcDocumentContentsAreChecked:
    """A list of non-records is not a HOLC document, whatever its length."""

    @pytest.mark.parametrize(
        "payload,count",
        [
            ({"features": ["Not Found"]}, 1),
            ({"features": [{"foo": 1}, {"bar": 2}, None]}, 3),
            ({"areas": [0, 0, 0, 0, 0]}, 5),
            ({"features": [["a"], ["b"]]}, 2),
        ],
    )
    def test_a_list_of_non_records_is_refused_not_counted(self, payload, count):
        out, warns = _holc(payload)
        assert out is None, payload
        assert any("are not records" in w for w in warns), warns
        assert any(f"measured count of {count} HOLC graded areas" in w for w in warns), warns

    def test_a_well_shaped_document_with_no_grade_anywhere_is_disclosed(self):
        out, warns = _holc({"features": [{"foo": 1}, {"properties": {"grade": None}}]})
        # Returned, because the shape is right and refusing would reject a real
        # response spelling its grade another way. But NOT silently.
        assert out is not None
        assert any("NOT ONE of them holds a readable HOLC grade" in w for w in warns), warns

    # ── CONTROLS: the real document and the documented refusals ────────────────

    def test_control_a_real_geojson_feature_collection_is_returned_silently(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"grade": "D"}, "geometry": {}},
                {"type": "Feature", "properties": {"grade": "A"}, "geometry": {}},
            ],
        }
        out, warns = _holc(payload)
        assert out is payload
        assert len(out["features"]) == 2
        assert warns == []

    def test_control_the_documented_empty_third_state_is_still_returned(self):
        out, warns = _holc({"features": []})
        assert out == {"features": []}
        assert any("carries ZERO areas" in w for w in warns), warns

    @pytest.mark.parametrize(
        "payload",
        [{"message": "Not Found"}, {}, [1, 2, 3], "not json data", None, {"features": {}}],
    )
    def test_control_the_earlier_refusals_still_refuse(self, payload):
        out, warns = _holc(payload)
        assert out is None
        assert warns and "fetch_holc_data" in warns[0]
