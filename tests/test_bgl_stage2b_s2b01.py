"""Beta Go-Live Stage 2b, group s2b01.

Every test here pins a THREE-STATE disclosure at a PUBLIC entry point, and
every one is paired with a control proving the measured path still returns its
real measurement. A fix that refuses everything passes any test that only
exercises the degenerate case, so the control is not optional.

Findings closed in this file (each reproduced by execution first, before/after
values in the docstrings):

- proxy_feature_score  (xai/decomposition/shap_fairness.py)
  The denominator ``sum(abs(v))`` overflowed to ``inf`` for disparities near
  the float ceiling, and ``abs(v) / inf`` is 0.0 for EVERY feature. Measured:
  ``proxy_score({"f1": 1e308, "f2": 1e308, "f3": 0.5})`` returned
  ``{f1: 0.0, f2: 0.0, f3: 0.0}`` with NO warning, i.e. two features carrying a
  1e308 disparity published as "0 percent of the total" and ``flagged_proxies``
  read ``[]``. Now ``{f1: 0.5, f2: 0.5, f3: 2.5e-309}``: a MEASUREMENT, not a
  refusal, because scaling by the max before summing leaves the ratio alone.

- intersectional_calibration  (post_processing/calibration/group_calibrator.py)
  Four holes: the transform disclosure omitted unknown intersections from its
  own count (1 of 951 for a true 51 of 951); a single-class intersection above
  ``min_group_size`` was labelled ``intersectional`` while its own fitter had
  refused; the disclosure was a bare ``UserWarning`` that left ``is_fitted``
  False under ``-W error``; and an object fitted by 0.1.0 died with
  ``AttributeError`` in transform().

- report_generation  (operations/reporting/reports.py, store.py)
  Three sibling report methods skipped the drift-coverage correction and
  rendered "| Drift Stability | 100.0 |" on a store with zero drift rows; a
  store whose only content was an unresolved CRITICAL alert published
  "97/100 (GREEN)" over "| Metric Compliance | 100.0 |" with no warning.

- distribution_matching  (post_processing/reweighting/reweighter.py)
  PINS ONLY. The refusals are already present in that file; the auditor's
  highest-priority finding was that NOTHING in the suite reaches them, so
  reverting either guard stays green. This file reaches both. The source is
  owned by another group in this wave and is not edited here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# proxy_feature_score
# ---------------------------------------------------------------------------
from vfairness.xai.decomposition.shap_fairness import (  # noqa: E402
    lundberg_fairness_decomposition,
    proxy_score,
)


class TestProxyScoreOverflowDenominator:
    def test_a_disparity_at_the_float_ceiling_is_measured_not_published_as_zero(self):
        """REFUSAL-OF-FABRICATION PIN, at the public entry ``proxy_score``.

        BEFORE: {'f1': 0.0, 'f2': 0.0, 'f3': 0.0}, zero warnings.
        AFTER:  {'f1': 0.5, 'f2': 0.5, 'f3': 2.499...e-309}, zero warnings.

        0.0 is the single most reassuring value this function can return: it
        says the feature carries none of the disparity. Two features holding a
        1e308 disparity got it, and ``flagged_proxies`` therefore read "no proxy
        features detected" beside them.
        """
        per_feature = {"f1": 1e308, "f2": 1e308, "f3": 0.5}
        # The fixture must actually reach the overflow: assert the raw sum
        # really is non-finite, so this cannot pass by never exercising it.
        assert not math.isfinite(sum(abs(v) for v in per_feature.values()))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores = proxy_score(per_feature)

        assert scores["f1"] == pytest.approx(0.5), scores
        assert scores["f2"] == pytest.approx(0.5), scores
        # f3 is genuinely negligible beside 1e308, but it is a MEASURED share.
        assert scores["f3"] is not None and scores["f3"] >= 0.0
        assert scores["f1"] != 0.0 and scores["f2"] != 0.0, (
            f"a 1e308 disparity was published as 0.0 of the total: {scores}"
        )
        # A measurement, so no could-not-check warning is appropriate here.
        assert [str(w.message) for w in caught] == []

    def test_control_ordinary_disparities_keep_their_exact_shares(self):
        """OVER-CORRECTION CONTROL. The scaling must not move a normal answer."""
        scores = proxy_score({"a": 0.6, "b": 0.2, "c": 0.2})
        assert scores == {"a": pytest.approx(0.6), "b": pytest.approx(0.2), "c": pytest.approx(0.2)}
        assert sum(v for v in scores.values() if v is not None) == pytest.approx(1.0)

    def test_control_all_zero_disparities_still_mean_no_feature_is_a_proxy(self):
        """OVER-CORRECTION CONTROL. A finite denominator of 0 means every
        disparity WAS measured and every one is exactly 0.0. That is a real
        finding ("no feature is a proxy") and must not become a refusal."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scores = proxy_score({"a": 0.0, "b": 0.0})
        assert scores == {"a": 0.0, "b": 0.0}
        assert [str(w.message) for w in caught] == []

    def test_control_one_unmeasurable_feature_does_not_silence_the_measured_ones(self):
        """OVER-CORRECTION CONTROL for the pre-existing None path."""
        with pytest.warns(UserWarning, match="COULD NOT BE MEASURED"):
            scores = proxy_score({"a": 0.6, "b": float("nan"), "c": 0.2})
        assert scores["b"] is None
        assert scores["a"] == pytest.approx(0.75)
        assert scores["c"] == pytest.approx(0.25)

    def test_an_empty_request_is_not_a_report_that_nothing_was_measurable(self):
        """``proxy_score({})`` warned "not one per-feature disparity is finite",
        which is a claim about data that does not exist. BEFORE: 1 warning.
        AFTER: 0 warnings, ``{}``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert proxy_score({}) == {}
        assert [str(w.message) for w in caught] == []


class TestDecompositionUnscoredGuardIsReachable:
    def test_an_all_finite_shap_matrix_can_still_have_a_non_finite_disparity(self):
        """The comment at the ``unscored`` guard said "Unreachable from here".
        It is reachable: ``_finite_col`` tests the SHAP CELLS, this tests the
        between-group MEAN DIFFERENCE, and a column holding +1e308 for group A
        and -1e308 for group B is finite cell by cell with a disparity of +inf.

        This is the auditor's finding 1 for this capability: sabotage S5 stayed
        green across 175 tests because the only fixture spoiled a CELL and so
        never reached this arm.
        """
        sv = np.zeros((4, 2), dtype=float)
        sv[:2, 0] = 1e308
        sv[2:, 0] = -1e308
        sv[:, 1] = [0.1, 0.2, 0.3, 0.4]
        groups = np.array(["A", "A", "B", "B"])

        # The fixture reaches THIS guard and not the cell guard above it.
        assert np.isfinite(sv).all(), "every SHAP cell must be finite, or the wrong guard fires"
        assert not math.isfinite(float(sv[:2, 0].mean() - sv[2:, 0].mean()))

        with pytest.raises(ValueError, match=r"proxy_score could not be computed for"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lundberg_fairness_decomposition(
                    shap_values=sv,
                    group_labels=groups,
                    feature_names=["f1", "f2"],
                    metric="demographic_parity",
                    protected_attribute="race",
                    subject_id="s",
                    audit_artifact_id="a",
                )

    def test_control_a_healthy_matrix_still_decomposes_and_flags_its_proxy(self):
        """OVER-CORRECTION CONTROL."""
        rng = np.random.default_rng(11)
        n = 40
        groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        sv = rng.normal(0, 0.01, size=(n, 3))
        sv[: n // 2, 0] += 0.5  # f1 is the proxy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = lundberg_fairness_decomposition(
                shap_values=sv,
                group_labels=groups,
                feature_names=["f1", "f2", "f3"],
                metric="demographic_parity",
                protected_attribute="race",
                subject_id="s",
                audit_artifact_id="a",
            )
        assert "f1" in dec.flagged_proxies, dec.proxy_scores
        assert all(v is not None for v in dec.proxy_scores.values())
        assert dec.total_disparity > 0.3


# ---------------------------------------------------------------------------
# intersectional_calibration
# ---------------------------------------------------------------------------

from vfairness.post_processing.calibration.group_calibrator import (  # noqa: E402
    PROVENANCE_INTERSECTIONAL,
    PROVENANCE_INTERSECTIONAL_DEGENERATE,
    PROVENANCE_UNKNOWN_GLOBAL,
    IntersectionalCalibrator,
    IntersectionalProvenanceWarning,
)


def _sparse_frame(seed: int = 0):
    """300/300/300/1 across gender x race: F_B is below min_group_size."""
    rng = np.random.default_rng(seed)
    rows = []
    for pair, k in zip([("M", "A"), ("M", "B"), ("F", "A"), ("F", "B")], [300, 300, 300, 1]):
        rows += [pair] * k
    df = pd.DataFrame(rows, columns=["gender", "race"])
    y_prob = rng.uniform(0.05, 0.95, len(df))
    y_true = (rng.uniform(size=len(df)) < y_prob).astype(int)
    return df, y_prob, y_true


def _healthy_frame(seed: int = 5):
    rng = np.random.default_rng(seed)
    rows = []
    for pair in [("M", "A"), ("M", "B"), ("F", "A"), ("F", "B")]:
        rows += [pair] * 200
    df = pd.DataFrame(rows, columns=["gender", "race"])
    y_prob = rng.uniform(0.05, 0.95, len(df))
    y_true = (rng.uniform(size=len(df)) < y_prob).astype(int)
    return df, y_prob, y_true


class TestIntersectionalTransformCountsEveryUncalibratedRow:
    def test_unknown_intersections_are_counted_in_the_transform_disclosure(self):
        """PUBLIC ENTRY: ``IntersectionalCalibrator.transform``.

        BEFORE: "1 of 951 returned row(s) are NOT intersectionally calibrated",
                provenance detail ``'F_B': borrowed_marginal`` only.
        AFTER:  "51 of 951 ...", detail also ``'X_Z': unknown_intersection_global``.

        The 50 rows of an unseen intersection were handed the POOLED global
        curve a few lines up and then excluded from the count of rows that are
        not intersectionally calibrated, which is the one group that certainly
        is not.
        """
        df, y_prob, y_true = _sparse_frame()
        cal = IntersectionalCalibrator(min_group_size=30, borrowing_strategy="hierarchical")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal.fit(y_true, y_prob, df)

        unseen = pd.DataFrame([("X", "Z")] * 50, columns=["gender", "race"])
        df2 = pd.concat([df, unseen], ignore_index=True)
        y2 = np.concatenate([y_prob, np.linspace(0.1, 0.9, 50)])

        # The fixture really does carry an intersection fit() never saw.
        assert "X_Z" not in cal.calibrators_

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.transform(y2, df2)

        msgs = [
            str(w.message) for w in caught if isinstance(w.message, IntersectionalProvenanceWarning)
        ]
        assert len(msgs) == 1, msgs
        assert "51 of 951 returned row(s) are NOT intersectionally calibrated" in msgs[0], msgs[0]
        assert PROVENANCE_UNKNOWN_GLOBAL in msgs[0], msgs[0]
        assert "'X_Z'" in msgs[0], msgs[0]

    def test_control_an_all_intersectional_run_says_nothing(self):
        """OVER-CORRECTION CONTROL. Every intersection big enough and learnable:
        no provenance warning at all, and the output is a real calibration."""
        df, y_prob, y_true = _healthy_frame()
        cal = IntersectionalCalibrator(min_group_size=30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y_true, y_prob, df)
            out = cal.transform(y_prob, df)
        assert set(cal.group_provenance_.values()) == {PROVENANCE_INTERSECTIONAL}
        assert [w for w in caught if isinstance(w.message, IntersectionalProvenanceWarning)] == []
        assert np.unique(out).size > 2, "a healthy calibration must still vary"


class TestMinGroupSizeIsNotAMeasurabilityGate:
    @staticmethod
    def _single_class_frame():
        """M_A: 100 rows, ONE observed class. Clears min_group_size=30 easily
        and supports no probability-to-outcome map whatsoever."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame([("M", "A")] * 100 + [("F", "A")] * 100, columns=["gender", "race"])
        y_prob = rng.uniform(0.05, 0.95, 200)
        y_true = np.concatenate(
            [np.ones(100, dtype=int), (rng.uniform(size=100) < y_prob[100:]).astype(int)]
        )
        return df, y_prob, y_true

    def test_a_group_its_own_fitter_refused_is_not_labelled_intersectional(self):
        """PUBLIC ENTRY: ``fit`` then ``get_group_provenance``.

        BEFORE: group_provenance_['M_A'] == 'intersectional', ZERO transform
                warnings, and transform returned a flat 1.0 for inputs spanning
                0.074 to 0.947.
        AFTER:  'intersectional_degenerate', disclosed at fit AND at transform.

        ``min_group_size`` counts rows. It cannot see that those rows carry one
        class, which is what the base calibrator itself warned about.
        """
        df, y_prob, y_true = self._single_class_frame()
        # The fixture exercises the branch it claims to: M_A is ABOVE the gate.
        assert (df["gender"] == "M").sum() == 100
        assert np.unique(y_true[:100]).size == 1
        assert np.unique(y_prob[:100]).size > 2, "the INPUT must vary, or a flat map says nothing"

        cal = IntersectionalCalibrator(min_group_size=30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y_true, y_prob, df)

        prov = cal.get_group_provenance()
        assert prov["M_A"] == PROVENANCE_INTERSECTIONAL_DEGENERATE, prov
        assert prov["F_A"] == PROVENANCE_INTERSECTIONAL, prov
        assert cal.group_sample_counts_["M_A"] >= cal.min_group_size

        fit_msgs = [
            str(w.message) for w in caught if isinstance(w.message, IntersectionalProvenanceWarning)
        ]
        assert len(fit_msgs) == 1, fit_msgs
        assert "could not learn a probability-to-outcome map" in fit_msgs[0], fit_msgs[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.transform(y_prob, df)
        tmsgs = [
            str(w.message) for w in caught if isinstance(w.message, IntersectionalProvenanceWarning)
        ]
        assert len(tmsgs) == 1, tmsgs
        assert "100 of 200 returned row(s) are NOT intersectionally calibrated" in tmsgs[0]
        assert PROVENANCE_INTERSECTIONAL_DEGENERATE in tmsgs[0]

    def test_control_a_learnable_group_of_the_same_size_keeps_intersectional(self):
        """OVER-CORRECTION CONTROL. The degenerate label must require BOTH a
        fit note AND an information-free map, so a group of the same size whose
        labels carry both classes is untouched."""
        rng = np.random.default_rng(3)
        df = pd.DataFrame([("M", "A")] * 100 + [("F", "A")] * 100, columns=["gender", "race"])
        y_prob = rng.uniform(0.05, 0.95, 200)
        y_true = (rng.uniform(size=200) < y_prob).astype(int)
        cal = IntersectionalCalibrator(min_group_size=30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.fit(y_true, y_prob, df)
        assert set(cal.get_group_provenance().values()) == {PROVENANCE_INTERSECTIONAL}
        assert [w for w in caught if isinstance(w.message, IntersectionalProvenanceWarning)] == []


class TestProvenanceWarningIsSelectableAndDoesNotUnfitTheObject:
    def test_escalating_the_disclosure_does_not_destroy_the_fit(self):
        """BEFORE: ``simplefilter("error", UserWarning)`` -> fit raised and left
        ``is_fitted=False`` on an ordinary 300/300/300/1 dataset.
        AFTER: it still raises (the caller asked for that), but the object is
        fitted, because a disclosure must not also undo what it describes."""
        df, y_prob, y_true = _sparse_frame()
        cal = IntersectionalCalibrator(min_group_size=30)
        with pytest.raises(IntersectionalProvenanceWarning):
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntersectionalProvenanceWarning)
                cal.fit(y_true, y_prob, df)
        assert cal.is_fitted is True
        assert cal.get_group_provenance()["F_B"] != PROVENANCE_INTERSECTIONAL

    def test_the_category_is_selectable_without_escalating_every_userwarning(self):
        """A bare UserWarning could not be singled out. This one can, and it
        still subclasses UserWarning so existing filters keep matching."""
        assert issubclass(IntersectionalProvenanceWarning, UserWarning)
        df, y_prob, y_true = _sparse_frame()
        cal = IntersectionalCalibrator(min_group_size=30)
        with pytest.warns(IntersectionalProvenanceWarning):
            cal.fit(y_true, y_prob, df)


class TestObjectsFittedByTheReleasedVersionStillTransform:
    def test_a_missing_group_provenance_attribute_degrades_instead_of_crashing(self):
        """BEFORE: AttributeError: 'IntersectionalCalibrator' object has no
        attribute 'group_provenance_'. AFTER: transform returns, and discloses
        that it cannot vouch for any row's provenance."""
        df, y_prob, y_true = _sparse_frame()
        cal = IntersectionalCalibrator(min_group_size=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal.fit(y_true, y_prob, df)
        del cal.group_provenance_  # what a 0.1.0-pickled object looks like
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = cal.transform(y_prob, df)
        assert len(out) == len(y_prob)
        msgs = [
            str(w.message) for w in caught if isinstance(w.message, IntersectionalProvenanceWarning)
        ]
        assert msgs and "901 of 901" in msgs[0], msgs
        assert cal.get_group_provenance() == {}


# ---------------------------------------------------------------------------
# report_generation
# ---------------------------------------------------------------------------

from vfairness.operations.monitoring import (  # noqa: E402
    FairnessMonitor,
    FairnessMonitorConfig,
)
from vfairness.operations.reporting import MetricsStore  # noqa: E402
from vfairness.operations.reporting.reports import (  # noqa: E402
    OutputFormat,
    ReportGenerator,
)


def _breaching_store():
    """Group A selected 90 percent, group B 10 percent: both built-ins breach.
    ZERO drift rows, which is the default state of every MetricsStore."""
    batch = pd.DataFrame(
        {
            "prediction": np.r_[np.repeat([1, 0], [90, 10]), np.repeat([1, 0], [10, 90])],
            "label": np.r_[np.ones(100, dtype=int), np.zeros(100, dtype=int)],
            "group_gender": ["A"] * 100 + ["B"] * 100,
        }
    )
    mon = FairnessMonitor(
        config=FairnessMonitorConfig(metrics_to_track=["disparate_impact", "demographic_parity"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snap = mon.update_and_check(batch)
    store = MetricsStore()
    store.ingest_window_metrics(snap)
    return store


class TestEventReportsGetTheSameDriftCoverageAsTheTieredOnes:
    @pytest.mark.parametrize("which", ["alert", "breach"])
    def test_a_store_with_no_drift_rows_does_not_publish_a_drift_stability_of_100(self, which):
        """PUBLIC ENTRY: ``generate_alert_report`` / ``generate_threshold_breach_report``.

        BEFORE: "| Drift Stability | 100.0 |", no coverage note, on a store
                whose drift table is empty, while generate_executive_report on
                the SAME store rendered "not assessed".
        AFTER:  "| Drift Stability | not assessed |" and a Coverage section.
        """
        store = _breaching_store()
        assert len(store.get_drift_history()) == 0, "the fixture must have no drift rows"
        gen = ReportGenerator(store)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if which == "alert":
                rep = gen.generate_alert_report(
                    {"severity": "HIGH", "metric_name": "demographic_parity_difference"},
                    output_format=OutputFormat.MARKDOWN,
                )
            else:
                rep = gen.generate_threshold_breach_report(
                    "demographic_parity_difference", 0.35, 0.10, OutputFormat.MARKDOWN
                )
        content = rep.content
        assert "| Drift Stability | 100.0 |" not in content, content
        assert "| Drift Stability | not assessed |" in content, content
        assert "Drift stability: NOT ASSESSED" in content, content

    def test_the_published_band_is_the_one_the_measured_components_give(self):
        """The unmeasured 20 percent did not merely blur the score, it moved the BAND.

        Measured before: 0.50*0.0 + 0.30*100.0 + 0.20*100.0 published
        "50/100 (YELLOW)", while the two measured components renormalise to 37.5, which
        is RED. This test asserted that pair, 50 YELLOW beside 37.5 RED, and the
        sentence naming them as different bands.

        RENAMED AND REPOINTED 2026-09-27, subject strengthened rather than dropped. The
        producer no longer publishes the fabricated composite at all: it drops the drift
        component and renormalises over the two that were measured, so the HEADLINE is
        now the evidence-based band. There is no second reading to contrast, and a note
        claiming the score above "includes the unmeasured drift weight" would now be
        false. What this test exists to protect is that a reader is shown the band that
        rests on evidence, and that is asserted directly.
        """
        store = _breaching_store()
        gen = ReportGenerator(store)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = gen.generate_executive_report(output_format=OutputFormat.MARKDOWN)
        content = rep.content
        assert "(RED)" in content, content
        assert "(YELLOW)" not in content, (
            "YELLOW was the band the fabricated drift weight produced; the measured "
            f"components give RED:\n{content}"
        )
        assert "| Drift Stability | not assessed |" in content, content
        assert "Drift stability: NOT ASSESSED" in content, content
        assert "| Drift Stability | 100.0 |" not in content, content

    def test_control_a_clean_store_reports_the_same_band_both_ways(self):
        """OVER-CORRECTION CONTROL. When the unmeasured component changes
        nothing, the note must say so rather than manufacture an alarm."""
        batch = pd.DataFrame(
            {
                "prediction": [1, 0] * 100,
                "label": [1, 0] * 100,
                "group_gender": ["A"] * 100 + ["B"] * 100,
            }
        )
        mon = FairnessMonitor(
            config=FairnessMonitorConfig(
                metrics_to_track=["disparate_impact", "demographic_parity"]
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snap = mon.update_and_check(batch)
        store = MetricsStore()
        store.ingest_window_metrics(snap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ReportGenerator(store).generate_executive_report(
                output_format=OutputFormat.MARKDOWN
            )
        assert "## Health Score: 100/100 (GREEN)" in rep.content
        # The two-reading sentences are gone with the second reading: once the producer
        # renormalises, there is one score and it is the measured one. What this control
        # is FOR survives and is asserted here: the disclosure must not manufacture an
        # alarm when the missing component would not have changed the band.
        assert "Drift stability: NOT ASSESSED" in rep.content, rep.content
        assert "DIFFERENT BANDS" not in rep.content, rep.content
        assert "(RED)" not in rep.content and "(YELLOW)" not in rep.content, rep.content


class TestMetricComplianceNeedsEvidence:
    def test_a_store_holding_only_an_alert_does_not_publish_97_green(self):
        """PUBLIC ENTRY: ``MetricsStore.compute_health_score`` and the report.

        BEFORE: score 97.0, status 'green', components
                {'metric_compliance': 100.0, 'alert_frequency': 90.0,
                 'drift_stability': 100.0}, ZERO warnings, and the page read
                "## Health Score: 97/100 (GREEN)" over
                "| Metric Compliance | 100.0 |" beside an unresolved CRITICAL.
        AFTER:  score None, status 'not_assessed', components {}, one warning.

        The only metric record in that store is the mirrored alert row, which
        is excluded from the compliance frame by design and already counted by
        the alert component. Nothing was ever compared with a threshold, so
        100.0 is the perfect default for no evidence.
        """
        store = MetricsStore()
        store.ingest_alert(
            {
                "severity": "CRITICAL",
                "metric_name": "demographic_parity_difference",
                "priority_score": 9.8,
                "message": "unresolved critical",
            }
        )
        # The fixture reaches the branch it claims to: one record, and it is
        # the mirrored alert row rather than an unmeasurable or descriptive one.
        raw = store.get_metrics()
        assert len(raw) == 1
        assert raw["source"].tolist() == ["FairnessAlertPrioritizer"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hs = store.compute_health_score()
        assert hs.score is None, hs
        assert hs.status == "not_assessed", hs
        assert "metric_compliance" not in hs.components, hs.components
        joined = " ".join(str(w.message) for w in caught)
        assert "metric compliance has NO evidence" in joined, joined
        assert "not a metric compliance of 100" in joined, joined

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = ReportGenerator(store).generate_executive_report(
                output_format=OutputFormat.MARKDOWN
            )
        assert "| Metric Compliance | 100.0 |" not in rep.content, rep.content
        assert "(GREEN)" not in rep.content, rep.content

    def test_control_real_metric_rows_still_produce_their_measured_compliance(self):
        """OVER-CORRECTION CONTROL. Two determined built-ins, both breaching:
        metric compliance is a measured 0.0, not a refusal."""
        store = _breaching_store()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hs = store.compute_health_score()
        assert hs.score is not None
        assert hs.components["metric_compliance"] == pytest.approx(0.0)
        # BGL5 A-operations-3, 2026-09-27: alert_frequency was asserted at 100.0
        # here, on a fixture that never feeds the alert channel, so that 100.0 was
        # the same no-evidence default this class exists to refuse one component
        # over. It is now absent. The subject, a MEASURED compliance of 0.0 rather
        # than a refusal, is the assertion above.
        assert "alert_frequency" not in hs.components, hs.components


# ---------------------------------------------------------------------------
# distribution_matching: PINS ONLY (the source belongs to another group)
# ---------------------------------------------------------------------------

from vfairness.post_processing.reweighting.reweighter import DistributionMatcher  # noqa: E402


class TestDistributionMatcherRefusalsAreReached:
    def test_a_group_with_one_distinct_score_is_refused_not_mapped_to_the_maximum(self):
        """HIGHEST-PRIORITY PIN from the audit: nothing in the suite reached
        this arm, so flipping ``np.unique(...).size < 2`` to ``< 1`` left all
        32 tests green while those rows came back 0.9419, the pooled maximum,
        with disparity_reduction +0.168 and zero warnings.

        A percentile grid needs at least two distinct scores. Forty rows all at
        0.20 give one, so the group's distribution COULD NOT BE ESTIMATED.
        """
        rng = np.random.default_rng(7)
        y_prob = np.concatenate([np.full(40, 0.20), rng.uniform(0.05, 0.95, 160)])
        groups = np.array(["A"] * 40 + ["B"] * 160)
        y_true = (rng.uniform(size=200) < y_prob).astype(int)

        # The fixture reaches THIS guard, not the small-sample one: 40 >= the
        # default min_group_size of 30, and the group has exactly one value.
        assert (groups == "A").sum() >= DistributionMatcher().min_group_size
        assert np.unique(y_prob[:40]).size == 1

        dm = DistributionMatcher()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
            out = dm.transform(y_prob, groups)

        assert np.isnan(out[:40]).all(), (
            f"an unestimable distribution returned values: {np.unique(out[:40])[:5]}"
        )
        assert not np.isnan(out[40:]).any(), "group B was measurable and must keep its values"
        joined = " ".join(str(w.message) for w in caught)
        assert "single distinct score" in joined, joined

    def test_an_unfittable_reference_group_refuses_instead_of_raising(self):
        """PIN. Without the reference-group refusal propagating, this input
        raised an uncaught ValueError out of np.interp. B is the reference and
        holds 5 rows; A holds 100 varied ones and is perfectly measurable."""
        rng = np.random.default_rng(9)
        y_prob = np.concatenate([rng.uniform(0.05, 0.95, 100), np.array([0.3, 0.4, 0.5, 0.6, 0.7])])
        groups = np.array(["A"] * 100 + ["B"] * 5)
        y_true = (rng.uniform(size=105) < y_prob).astype(int)

        dm = DistributionMatcher(reference_group="B")
        assert (groups == "B").sum() < dm.min_group_size
        assert np.unique(y_prob[:100]).size > 2, "the non-reference group must be fittable"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
            out = dm.transform(y_prob, groups)

        assert len(out) == len(y_prob)
        joined = " ".join(str(w.message) for w in caught)
        assert "reference group 'B'" in joined, joined

    def test_control_two_healthy_groups_are_still_matched_and_measured(self):
        """OVER-CORRECTION CONTROL. The refusals above must not swallow a run
        in which both groups are estimable."""
        rng = np.random.default_rng(13)
        y_prob = np.concatenate([rng.uniform(0.05, 0.45, 100), rng.uniform(0.55, 0.95, 100)])
        groups = np.array(["A"] * 100 + ["B"] * 100)
        y_true = (rng.uniform(size=200) < y_prob).astype(int)
        dm = DistributionMatcher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=groups)
            out = dm.transform(y_prob, groups)
        assert not np.isnan(out).any(), "a fully measurable run must not refuse"
        before = dm.result_.original_metrics["mean_disparity"]
        after = dm.result_.adjusted_metrics["mean_disparity"]
        assert math.isfinite(float(before)) and math.isfinite(float(after))
        assert after < before, (before, after)
