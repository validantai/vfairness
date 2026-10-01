"""BGL grade-1, batch G08: xai.schemas, preprocessing.bias_detection, operations.reporting.

Forty-three units, all first-time graded, executed on a healthy input and on the
degenerate inputs where the thing each one publishes does not exist. Most of them
are containers and records and their state is NOT A MEASUREMENT; the pins below
cover the seven places where execution found a value being minted, a disclosure
being dropped, or a reader surface saying nothing.

Each class names the measurement taken BEFORE the fix, so the subject of the
test is the defect and not today's numbers. Every class carries an
over-correction control asserting the healthy case's real value.
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.reporting.compliance import AdverseActionReasons
from vfairness.operations.reporting.dashboard import FairnessDashboard
from vfairness.operations.reporting.interactive import InteractiveDashboard
from vfairness.operations.reporting.store import (
    HealthScore,
    MetricsStore,
    MetricsStoreConfig,
)
from vfairness.preprocessing.bias_detection.proxy import (
    LEAKAGE_NOT_ASSESSED,
    MultivariateProxyResult,
)
from vfairness.preprocessing.bias_detection.representation import (
    detect_representation_bias,
)
from vfairness.xai.schemas import (
    Attribution,
    CounterfactualExplanation,
    CounterfactualInstance,
    Explanation,
)

_SENTINEL = object()


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return fn(*a, **kw)


def _caught(fn, *a, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(x.message) for x in w]


def _store(base: datetime, *, values=(0.02, 0.03, 0.01, 0.04), alert=False) -> MetricsStore:
    store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
    rows = [
        {
            "timestamp": base - timedelta(hours=i),
            "metric": "demographic_parity",
            "value": v,
            "group": "overall",
            "group_size": 200,
            "alert": alert,
            "alert_determined": True,
        }
        for i, v in enumerate(values)
    ]
    _quiet(
        store.ingest_dataframe,
        pd.DataFrame(rows),
        source="monitor",
        group_col="group",
        alert_col="alert",
        group_size_col="group_size",
        alert_determined_col="alert_determined",
    )
    return store


def _figure_texts(fig) -> list[str]:
    payload = json.loads(fig.to_json())
    return [a.get("text", "") for a in payload.get("layout", {}).get("annotations", [])]


# ===========================================================================
# 1. MultivariateProxyResult.was_assessed
# ===========================================================================


class TestWasAssessedAsksTheMeasurementQuestion:
    """MEASURED BEFORE THE FIX, on a record carrying auc=nan, severity='low',
    systemic_leakage=False: ``was_assessed()`` returned True. It is the one call
    the docstring tells a consumer to make before reading ``auc``, and it said
    the leakage test had run. ``inf`` and ``True`` passed it too
    (``float(True) == 1.0``, a perfect reconstruction score)."""

    @staticmethod
    def _record(auc):
        return MultivariateProxyResult(
            protected_attribute="race",
            auc=auc,
            chance_auc=0.5,
            severity="low",
            systemic_leakage=False,
            n_features_used=3,
            n_samples=50,
            n_classes=2,
            method="hist_gradient_boosting",
            top_contributors=[],
            interpretation="i",
        )

    @pytest.mark.parametrize(
        "auc",
        [
            float("nan"),
            np.float64("nan"),
            np.float32("nan"),
            float("inf"),
            float("-inf"),
            True,
            None,
        ],
        ids=["nan", "np64nan", "np32nan", "inf", "neg_inf", "bool_true", "none"],
    )
    def test_a_value_that_is_not_a_measurement_is_not_assessed(self, auc):
        assert self._record(auc).was_assessed() is False

    def test_the_not_assessed_record_the_producer_builds_says_so(self):
        rec = MultivariateProxyResult(
            protected_attribute="race",
            auc=None,
            chance_auc=0.5,
            severity=LEAKAGE_NOT_ASSESSED,
            systemic_leakage=None,
            n_features_used=0,
            n_samples=0,
            n_classes=0,
            method="error",
            top_contributors=[],
            interpretation="x",
        )
        assert rec.was_assessed() is False
        assert rec.to_dict()["auc"] is None

    # ---- OVER-CORRECTION CONTROLS: a real measurement still reads as one ----

    @pytest.mark.parametrize("auc", [0.5, 0.91, np.float64(0.73), np.float32(0.62), 0.0, 1.0, 1])
    def test_control_a_real_finite_auc_is_assessed(self, auc):
        rec = self._record(auc)
        assert rec.was_assessed() is True
        assert rec.to_dict()["auc"] == pytest.approx(float(auc))


# ===========================================================================
# 2. _camel_dict: a key collision dropped a field from the JSONB payload
# ===========================================================================


class TestToDbRowNeverDropsAFieldSilently:
    """MEASURED BEFORE THE FIX: CounterfactualExplanation.to_db_row on a
    counterfactual whose flipped_features held {"orig_value": 1,
    "origValue": 999} wrote {"origValue": 999}. One of the two values was simply
    not in the row, with no error and nothing recording the loss."""

    @staticmethod
    def _explanation(flipped):
        return CounterfactualExplanation(
            method="dice",
            instance_id="r1",
            subject_id="s",
            model_hash="m",
            data_hash="d",
            base_value=0.1,
            prediction=0.9,
            attributions=[Attribution(feature="income", contribution=0.4)],
            units="probability",
            scope="local",
            counterfactuals=[
                CounterfactualInstance(
                    flipped_features=flipped,
                    predicted_outcome="approved",
                    proximity=0.2,
                    feasibility=0.9,
                )
            ],
        )

    @pytest.mark.parametrize(
        "flipped",
        [
            [{"orig_value": 1, "origValue": 999}],
            [{"a_b": 1, "a__b": 2}],
            [{"new_value": 1, "newValue": 2}],
        ],
        ids=["orig_value", "double_underscore", "new_value"],
    )
    def test_two_keys_that_converge_are_refused_not_merged(self, flipped):
        with pytest.raises(ValueError, match="both become"):
            self._explanation(flipped).to_db_row("owner")

    # ---- OVER-CORRECTION CONTROL ----

    def test_control_the_shape_the_dice_adapter_produces_still_serialises(self):
        row = self._explanation([{"feature": "income", "from": 1, "to": 2}]).to_db_row("owner")
        payload = row["params"]["counterfactuals"][0]
        assert payload["flippedFeatures"] == [{"feature": "income", "from": 1, "to": 2}]
        assert payload["predictedOutcome"] == "approved"
        assert row["attributions"] == [
            {"feature": "income", "contribution": 0.4, "baseValue": None}
        ]


# ===========================================================================
# 3. Explanation.to_db_row: a GUESSED scope was silent
# ===========================================================================


class TestAGuessedScopeSaysSoInsteadOfPassingAsARecord:
    """MEASURED BEFORE THE FIX: TreeShapExplainer.explain_global stamps
    instance_id='<subject>#row-<i>' on every row of a global batch and sets no
    scope, so each of those rows serialised with scope='local' and nothing said
    the column had been guessed. The column is NOT NULL on the frozen contract,
    so the guess stays; it is now audible."""

    @staticmethod
    def _explanation(**overrides):
        kwargs = dict(
            method="shap.TreeExplainer",
            instance_id="subj#row-3",
            subject_id="subj",
            model_hash="m",
            data_hash="d",
            base_value=0.0,
            prediction=1.0,
            attributions=[],
            units="raw",
        )
        kwargs.update(overrides)
        return Explanation(**kwargs)

    def test_an_unset_scope_warns_and_names_the_id_it_guessed_from(self):
        row, messages = _caught(self._explanation().to_db_row, "owner")
        assert row["scope"] == "local"
        guessed = [m for m in messages if "GUESSED from instance_id" in m]
        assert guessed, messages
        assert "subj#row-3" in guessed[0]

    def test_an_empty_instance_id_is_also_a_guess(self):
        row, messages = _caught(self._explanation(instance_id="").to_db_row, "owner")
        assert row["scope"] == "global"
        assert any("GUESSED from instance_id" in m for m in messages)

    # ---- OVER-CORRECTION CONTROLS ----

    @pytest.mark.parametrize("scope", ["local", "global"])
    def test_control_an_explicit_scope_is_written_and_says_nothing(self, scope):
        row, messages = _caught(self._explanation(scope=scope).to_db_row, "owner")
        assert row["scope"] == scope
        assert not [m for m in messages if "GUESSED" in m], messages


# ===========================================================================
# 4. The health gauge a reader cannot read
# ===========================================================================


class TestAWithheldHealthScoreIsDisclosedOnEveryTierThatDrawsIt:
    """MEASURED BEFORE THE FIX, on an EMPTY MetricsStore (score=None,
    status='not_assessed', components={}): create_operational_view and
    create_technical_view each rendered an Indicator with value=None inside a
    red/amber/green banded dial titled "Health Score", the word "assessed"
    appeared nowhere on either figure, and the store's own sentence ("No metric
    records were found in the evaluation window ... certifies nothing") was on
    neither. create_health_score_gauge and create_executive_view both carried it
    for that identical store. The only caption the two views did carry asserted
    that "this score covers metric compliance and alert frequency", about a
    score that did not exist."""

    TIERS = ("operational", "technical")

    @pytest.mark.parametrize("tier", TIERS)
    def test_the_figure_says_the_gauge_is_empty_because_nothing_was_scored(self, tier):
        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        dash = FairnessDashboard(store)
        hs = _quiet(store.compute_health_score)
        assert hs.score is None and hs.status == "not_assessed"

        fig = _quiet(getattr(dash, f"create_{tier}_view"))
        texts = _figure_texts(fig)

        assert [t for t in texts if "Fairness Health Score: NOT ASSESSED" in t], texts
        assert [t for t in texts if hs.explanation in t], texts
        assert [t.value for t in fig.data if t.type == "indicator"] == [None]

    @pytest.mark.parametrize("tier", TIERS)
    def test_the_html_a_reader_opens_carries_it_too(self, tier):
        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        html = _quiet(FairnessDashboard(store).to_html, tier=tier)
        assert "Fairness Health Score: NOT ASSESSED" in html

    def test_the_drift_note_does_not_claim_coverage_for_a_score_that_was_withheld(self):
        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        texts = _figure_texts(_quiet(FairnessDashboard(store).create_operational_view))
        drift = [t for t in texts if t.startswith("Drift stability:")]
        assert drift, texts
        assert "covers metric compliance and alert frequency" not in drift[0]
        assert "no health score was computed" in drift[0]

    # ---- OVER-CORRECTION CONTROLS ----

    @pytest.mark.parametrize("tier", ("operational", "technical", "executive"))
    def test_control_a_measured_score_adds_no_withheld_note(self, tier):
        store = _store(datetime.now())
        hs = _quiet(store.compute_health_score)
        assert hs.score == pytest.approx(100.0)

        texts = _figure_texts(_quiet(getattr(FairnessDashboard(store), f"create_{tier}_view")))
        assert not [t for t in texts if "NOT ASSESSED. The gauge above is EMPTY" in t], texts

    def test_control_a_measured_score_keeps_the_original_drift_sentence(self):
        store = _store(datetime.now())
        texts = _figure_texts(_quiet(FairnessDashboard(store).create_operational_view))
        drift = [t for t in texts if t.startswith("Drift stability:")]
        assert drift, texts
        assert "covers metric compliance and alert frequency" in drift[0]


# ===========================================================================
# 5. to_standalone_html: a figure that CRASHED left a blank panel and no word
# ===========================================================================


class TestTheStandalonePageNamesEveryFigureThatFailedToBuild:
    """MEASURED BEFORE THE FIX, with create_trend_analysis and
    create_health_score_gauge raising RuntimeError('trend builder exploded') on a
    store holding four clean determined records: the page came back at 4,884,285
    characters with ``var gaugeData = {};``, the page's own renderPlot returns at
    ``if (!figData || !figData.data) return;`` so those panels were blank, and
    "exploded", "could not render" and "failed" appeared nowhere in it. The
    health badge still read 100/100 beside them. ReportGenerator._render_html
    already carried exactly this fix for the document surface; this was the
    second rendering of the same page and had neither the fix nor a pin."""

    @staticmethod
    def _boom(self, *a, **kw):
        raise RuntimeError("trend builder exploded")

    def test_a_crashed_figure_is_not_rendered_as_nothing_to_plot(self, monkeypatch):
        store = _store(datetime.now())
        monkeypatch.setattr(FairnessDashboard, "create_trend_analysis", self._boom)
        monkeypatch.setattr(FairnessDashboard, "create_health_score_gauge", self._boom)

        html = _quiet(InteractiveDashboard(store).to_standalone_html)

        assert "COULD NOT RENDER" in html
        assert "not because there was nothing to plot" in html
        assert "RuntimeError: trend builder exploded" in html
        assert "Fairness Health Score gauge" in html
        assert "Trend: demographic_parity" in html

    def test_the_count_of_failed_panels_is_the_real_one(self, monkeypatch):
        store = _store(datetime.now())
        monkeypatch.setattr(FairnessDashboard, "create_drift_timeline", self._boom)
        html = _quiet(InteractiveDashboard(store).to_standalone_html)
        assert "1 figure(s) on this page" in html
        assert html.count("<li><strong>") == 1

    def test_an_empty_store_reports_zero_metrics_not_the_placeholder(self):
        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        html = _quiet(InteractiveDashboard(store).to_standalone_html)
        line = re.search(r"Metrics: ([^|]*)\|", html).group(1).strip()
        assert line.startswith("0"), line
        assert "placeholder" in line
        assert re.search(r'<span class="health-badge">([^<]*)</span>', html).group(1) == (
            "not assessed"
        )

    @pytest.mark.parametrize(
        "score", [float("nan"), float("inf"), None], ids=["nan", "inf", "none"]
    )
    def test_a_score_that_is_not_a_measurement_reads_not_assessed(self, monkeypatch, score):
        store = _store(datetime.now())
        withheld = HealthScore(
            score=score,
            status="green",
            trend="stable",
            trend_slope=0.0,
            components={"metric_compliance": score},
            timestamp=datetime.now(),
            explanation="e",
            n_metrics=4,
        )
        monkeypatch.setattr(MetricsStore, "compute_health_score", lambda self, *a, **k: withheld)
        html = _quiet(InteractiveDashboard(store).to_standalone_html)
        badge = re.search(r'<span class="health-badge">([^<]*)</span>', html).group(1)
        assert badge == "not assessed", badge

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_working_page_says_nothing_and_embeds_its_figures(self):
        store = _store(datetime.now())
        html = _quiet(InteractiveDashboard(store).to_standalone_html)
        assert "COULD NOT RENDER" not in html
        assert "var gaugeData = {};" not in html
        assert re.search(r'<span class="health-badge">([^<]*)</span>', html).group(1) == ("100/100")
        assert re.search(r"Metrics: ([^|]*)\|", html).group(1).strip() == "1"


# ===========================================================================
# 6. A tz-aware store took every reader surface down
# ===========================================================================


class TestAStoreWithTimezoneAwareTimestampsIsStillReadable:
    """MEASURED BEFORE THE FIX, on a store ingested with
    datetime.now(timezone.utc) timestamps (what a timestamptz column and any
    ISO-8601 'Z' string give): get_metrics, get_summary and get_alerts answered
    correctly, then compute_health_score raised "TypeError: Invalid comparison
    between dtype=datetime64[ns, UTC] and Timestamp", because the window bound
    was built from a tz-naive datetime.now(). Everything above it went with it:
    the three view builders, FairnessDashboard.to_html, get_explanation,
    InteractiveDashboard.to_standalone_html and every ReportGenerator tier."""

    BASES = {
        "utc_aware": lambda: datetime.now(timezone.utc),
        "naive_local": datetime.now,
    }

    @pytest.mark.parametrize("base_id", list(BASES))
    def test_the_health_score_is_the_same_in_either_time_base(self, base_id):
        store = _store(self.BASES[base_id]())
        hs = _quiet(store.compute_health_score)
        assert hs.n_metrics == 4
        assert hs.score == pytest.approx(100.0)

    @pytest.mark.parametrize("base_id", list(BASES))
    @pytest.mark.parametrize("tier", ("executive", "operational", "technical"))
    def test_every_reader_surface_renders_in_either_time_base(self, base_id, tier):
        store = _store(self.BASES[base_id]())
        dash = FairnessDashboard(store)
        assert len(_quiet(dash.to_html, tier=tier)) > 1000
        assert "/100" in _quiet(dash.get_explanation)
        assert len(_quiet(InteractiveDashboard(store).to_standalone_html)) > 1000

    def test_window_now_adds_no_offset_and_leaves_a_naive_store_naive(self):
        aware = _store(datetime.now(timezone.utc))
        naive = _store(datetime.now())
        assert aware.window_now().tzinfo is not None
        assert naive.window_now().tzinfo is None
        # Same instant in both bases, so the two windows start together.
        assert abs(aware.window_now().timestamp() - naive.window_now().timestamp()) < 5.0

    def test_control_an_empty_store_still_withholds_the_score(self):
        store = MetricsStore(config=MetricsStoreConfig(enable_privacy=False))
        hs = _quiet(store.compute_health_score)
        assert hs.score is None and hs.status == "not_assessed"


# ===========================================================================
# 7. Representation: the doors dropna does not close
# ===========================================================================


class TestALabelThatIsASpellingOfAbsenceIsDisclosed:
    """MEASURED BEFORE THE FIX, 300 rows of White/Black/Hispanic with the first
    60 race values set to the string 'None': group_distributions came back
    {'White': 0.463, 'Black': 0.217, 'None': 0.2, 'Hispanic': 0.12} with
    sample_size 300, so 'None' was published as a demographic group holding a
    fifth of the population and every real group's share was diluted by 60 rows
    carrying no group (White is 0.579 among the 240 labelled rows). The same 60
    rows written as real pd.NA gave sample_size 240 and three groups, which is
    this function's own policy. The shares are deliberately NOT recomputed: for
    some protected attributes 'None' is a real category, and deleting a fifth of
    a real population would be the worse error. The reader is told instead."""

    @staticmethod
    def _frame(inject):
        rng = np.random.default_rng(3)
        col = pd.Series(
            rng.choice(["White", "Black", "Hispanic"], 300, p=[0.6, 0.25, 0.15]),
            dtype=object,
        )
        if inject is not _SENTINEL:
            col.iloc[:60] = inject
        return pd.DataFrame({"race": col, "y": rng.integers(0, 2, 300)})

    @pytest.mark.parametrize(
        "inject",
        ["None", "", "nan", "<NA>", "null", "  ", "NaT"],
        ids=["None", "blank", "nan", "NA_brackets", "null", "whitespace", "NaT"],
    )
    def test_the_caveat_names_the_count_and_the_spelling(self, inject):
        _, messages = _caught(detect_representation_bias, self._frame(inject), ["race"])
        said = [m for m in messages if "SPELLING OF ABSENCE" in m]
        assert said, messages
        assert "60 of 300" in said[0]
        assert "diluted" in said[0]

    def test_a_real_null_is_still_excluded_from_the_denominator(self):
        result, messages = _caught(detect_representation_bias, self._frame(pd.NA), ["race"])
        assert result[0].sample_size == 240
        assert sorted(result[0].group_distributions) == ["Black", "Hispanic", "White"]
        assert not [m for m in messages if "SPELLING OF ABSENCE" in m]

    # ---- OVER-CORRECTION CONTROLS ----

    def test_control_a_clean_column_says_nothing_and_keeps_its_real_shares(self):
        result, messages = _caught(detect_representation_bias, self._frame(_SENTINEL), ["race"])
        assert not [m for m in messages if "SPELLING OF ABSENCE" in m], messages
        assert result[0].sample_size == 300
        assert sorted(result[0].group_distributions) == ["Black", "Hispanic", "White"]
        assert sum(result[0].group_distributions.values()) == pytest.approx(1.0)

    def test_control_no_row_is_dropped_and_the_shares_are_untouched(self):
        result, _ = _caught(detect_representation_bias, self._frame("None"), ["race"])
        assert result[0].sample_size == 300
        assert result[0].group_distributions["None"] == pytest.approx(0.2)


# ===========================================================================
# 8. AdverseActionReasons.complete: the nothing-examined clause had no pin
# ===========================================================================


class TestTheNothingExaminedClauseCanActuallyFail:
    """SABOTAGE FINDING, BGL grade-1 G08, 2026-09-30. ``complete`` has three
    clauses and the FIRST one had nothing that could redden it. Replacing
    ``self.features_examined > 0`` with ``True`` left 249 tests across the six
    files that exercise this class green, because the only test of the
    nothing-examined case goes through
    ``compute_adverse_action_reasons(_SHAP, [], [])``, where all five adverse
    attributions are also absent from feature_names, so the THIRD clause
    carries the refusal on its own. A clause with no reachable pin is a clause a
    refactor can delete, and this one is the difference between an ECOA notice
    built on nothing and one built on five fully attributed features. The clause
    was correct; only the evidence was missing, so nothing in the library
    changed here."""

    def test_zero_features_examined_is_incomplete_on_its_own(self):
        notice = AdverseActionReasons(
            [],
            unattributed_features=[],
            features_examined=0,
            adverse_attributions_not_ranked=[],
        )
        assert notice.complete is False
        assert "no feature was examined" in repr(notice)
        assert notice.to_dict() == {
            "reasons": [],
            "complete": False,
            "features_examined": 0,
            "unattributed_features": [],
            "adverse_attributions_not_ranked": [],
        }

    def test_a_notice_carrying_codes_but_no_examination_count_is_incomplete(self):
        """The other side of the same clause: entries present, count 0. Nothing
        else in the object is wrong, so only the first clause can refuse it."""
        notice = AdverseActionReasons(
            [{"code": "RC01", "feature": "income"}],
            unattributed_features=[],
            features_examined=0,
            adverse_attributions_not_ranked=[],
        )
        assert notice.complete is False

    def test_json_dumps_on_the_bare_object_still_loses_the_disclosure(self):
        """Not a defect to fix, a property to keep visible: this IS a list
        subclass, so the bare array is what json.dumps writes. to_dict is the
        documented boundary and the pin is here so the two cannot drift."""
        notice = AdverseActionReasons([], features_examined=0)
        assert json.loads(json.dumps(notice)) == []
        assert json.loads(json.dumps(notice.to_dict()))["complete"] is False

    # ---- OVER-CORRECTION CONTROL ----

    def test_control_a_whole_model_notice_is_complete(self):
        notice = AdverseActionReasons(
            [{"code": "RC01", "feature": "income"}],
            unattributed_features=[],
            features_examined=5,
            adverse_attributions_not_ranked=[],
        )
        assert notice.complete is True
        assert repr(notice) == "[{'code': 'RC01', 'feature': 'income'}]"
        assert notice.to_dict()["features_examined"] == 5
