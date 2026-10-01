"""Readiness wave 3: a downstream surface must not undo an upstream fix.

TWO defects, both of them one layer BELOW the code that was fixed.

RE-COLLAPSE (src/vfairness/operations/experimentation/analysis.py). The
producer was changed so that a heterogeneity test that could not run reports
``heterogeneity_detected=None`` and ``heterogeneity_p_value=nan``, and a
per-intersection test that could not run reports ``significant=None``. Five
consumer sites in analysis.py then read those three states through a bare
truthiness test or a NaN comparison and flattened them back into two:

  1. ``if r.heterogeneity_detected:``  (decision_recommendation)
     None took the else branch, so no reasoning line, no trade-off, no caveat
     and no confidence penalty: the recommendation went out at 0.85 as though
     between-group consistency had been established.
  2. ``'Detected' if ... else 'Not detected'``  (to_report_sections)
     printed a finding of consistency for a test nobody ran, beside "(p=nan)".
     Measured on a REAL experiment whose two intersection effects were 1.0 and
     4.0: the producer's repr said "heterogeneity=not assessed" and warned,
     while the report section said "Heterogeneity: Not detected (p=nan)".
  3. ``e.effect < 0 and e.significant``  (decision_recommendation)
     dropped an intersection with effect -3.0 and no verdict out of
     harmed_groups, and the recommendation read "with no harmed groups".
  4. ``r.overall_p_value < 0.05``  (decision_recommendation)
     ``nan < 0.05`` is False, so an untestable overall effect was reported as
     "is not statistically significant (p=nan)".
  5. ``cate_df["significant"].sum()``  (to_report_sections)
     counted an ungraded intersection as not significant.

LEDGER AUTO-CLEARING (the verdict-ledger builder under scripts/). The rules
recognise SHAPES, not sites, so a site nobody has read can match one. A brand
new ``payload.get("disparity", 0.0)`` planted in the tree was written into
the verdict ledger as "legitimate-fail-closed" by a
plain write run: the ledger answered for the reviewer it exists to require.
Only the two per-site tables (FIXED, HAND) and the entries the ledger already
holds may CLEAR a site now; a rule proposal that clears is refused, printed,
and exits 1, so the fabricated-verdict gate keeps failing on
it. That gate's own per-key BUDGET covers the complementary hole (a second site
inside an already-reviewed function, which produces no new key); this file does
not re-test that, it only stays out of its way.

Every refusal pin below is paired with an over-correction control asserting
MEASURED values, because a surface that answers "could not check" to everything
is as useless as one that answers "fine" to everything.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types
import warnings

import pandas as pd
import pytest

from vfairness.operations.experimentation.analysis import ExperimentAnalysis
from vfairness.operations.experimentation.experiment import (
    ExperimentConfig,
    ExperimentResult,
    FairnessExperiment,
    IntersectionEffect,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "audits" / "fabricated-verdict-ledger.json"
BUILDER = ROOT / "scripts" / "build_verdict_ledger.py"

NAN = float("nan")


# ---------------------------------------------------------------------------
# Fixtures: an ExperimentResult with exactly the states under test
# ---------------------------------------------------------------------------


def _effect(name: str, effect: float, p_value: float, significant) -> IntersectionEffect:
    return IntersectionEffect(
        intersection=(name,),
        control_mean=0.0,
        treatment_mean=effect,
        effect=effect,
        ci_lower=effect - 1.0,
        ci_upper=effect + 1.0,
        p_value=p_value,
        effect_size_d=0.3,
        n_control=50,
        n_treatment=50,
        significant=significant,
        powered=True,
    )


def _result(effects, het_detected, het_p, overall_p=0.001, overall_effect=0.5):
    return ExperimentResult(
        overall_effect=overall_effect,
        overall_ci=(overall_effect - 0.3, overall_effect + 0.3),
        overall_p_value=overall_p,
        intersection_effects=list(effects),
        heterogeneity_detected=het_detected,
        heterogeneity_p_value=het_p,
        # Every intersection well powered, so the power modifier never fires
        # and the confidence numbers asserted below are the ones under test.
        power_results={str(e.intersection): 0.9 for e in effects},
        n_intersections=len(effects),
    )


def _recommend(result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExperimentAnalysis(result).decision_recommendation()


def _effects_section(result) -> str:
    for section in ExperimentAnalysis(result).to_report_sections():
        if section["title"] == "Intersectional Treatment Effects":
            return section["content"]
    raise AssertionError("the per-intersection section was not emitted at all")


TWO_CLEAN = [_effect("a", 1.0, 0.01, True), _effect("b", 4.0, 0.02, True)]


# ---------------------------------------------------------------------------
# 1 + 2. Heterogeneity: None is could-not-check on BOTH consumer surfaces
# ---------------------------------------------------------------------------


class TestHeterogeneityThreeStateSurvivesTheConsumer:
    def test_refusal_the_report_section_does_not_report_a_finding(self):
        """The worst of the five: the surface a reader actually reads."""
        content = _effects_section(_result(TWO_CLEAN, None, NAN))
        assert "Not detected" not in content
        assert "nan" not in content
        assert content == (
            "2 intersections analysed, 2 statistically significant.\n\n"
            "Heterogeneity: COULD NOT CHECK (no heterogeneity test was reported "
            "for this experiment)."
        )

    def test_refusal_the_recommendation_says_so_and_pays_for_it(self):
        rec = _recommend(_result(TWO_CLEAN, None, NAN))
        assert any("COULD NOT BE RUN" in line for line in rec.reasoning)
        assert any("was not assessed" in c for c in rec.caveats)
        assert rec.trade_offs["heterogeneity"].startswith("Unknown:")
        # 0.85 for the deploy branch, times the 0.85 heterogeneity penalty.
        assert rec.confidence == 0.7225

    def test_control_a_measured_absence_still_reads_as_an_absence(self):
        """Over-correction control. A test that RAN and found no heterogeneity
        must still say so, with its p-value, and pay no penalty for it."""
        result = _result(TWO_CLEAN, False, 0.4)
        content = _effects_section(result)
        assert content == (
            "2 intersections analysed, 2 statistically significant.\n\n"
            "Heterogeneity: Not detected (p=0.4000)."
        )
        rec = _recommend(result)
        assert "heterogeneity" not in rec.trade_offs
        assert rec.confidence == 0.85
        assert not any("COULD NOT" in line for line in rec.reasoning)

    def test_control_a_detected_finding_is_still_detected(self):
        result = _result(TWO_CLEAN, True, 0.021)
        assert _effects_section(result).endswith("Heterogeneity: Detected (p=0.0210).")
        rec = _recommend(result)
        assert "Significant heterogeneity detected (p=0.0210)." in " ".join(rec.reasoning)
        assert rec.confidence == 0.7225

    def test_the_three_states_are_three_different_answers_not_sentences(self):
        """Not three different sentences: three different VERDICTS. Comparing
        whole lines is not enough, because the defect rendered None and False
        with the same words and only the p-value differed, and a reader reads
        the words. So the verdict is extracted and the three are compared."""
        verdicts = {}
        for state, p_value in ((True, 0.021), (False, 0.4), (None, NAN)):
            line = _effects_section(_result(TWO_CLEAN, state, p_value)).split("\n\n")[1]
            verdicts[state] = line.split(":", 1)[1].split("(")[0].strip()
        assert len(set(verdicts.values())) == 3, verdicts
        assert verdicts[True] == "Detected"
        assert verdicts[False] == "Not detected"
        assert verdicts[None] == "COULD NOT CHECK"


# ---------------------------------------------------------------------------
# 3. A negative effect with no significance verdict is not safety
# ---------------------------------------------------------------------------


class TestUnassessedHarmIsNotBankedAsSafety:
    HARMED_UNKNOWN = [_effect("winners", 2.0, 0.01, True), _effect("harmed", -3.0, NAN, None)]

    def test_refusal_the_recommendation_does_not_claim_no_harmed_groups(self):
        rec = _recommend(_result(self.HARMED_UNKNOWN, False, 0.4))
        assert "with no harmed groups" not in " ".join(rec.reasoning)
        assert any("COULD NOT BE DETERMINED" in line for line in rec.reasoning)
        assert any("unknown, not as safety" in c for c in rec.caveats)
        # A deploy on the back of an unanswered harm question is downgraded.
        assert rec.decision.value == "extend_experiment"
        assert rec.confidence == 0.595  # 0.85 * 0.7

    def test_refusal_the_ungraded_intersection_is_counted_as_ungraded(self):
        assert _effects_section(_result(self.HARMED_UNKNOWN, False, 0.4)) == (
            "2 intersections analysed, 1 of 1 graded statistically significant, "
            "1 could not be graded.\n\nHeterogeneity: Not detected (p=0.4000)."
        )

    def test_control_a_graded_harmless_group_still_deploys(self):
        """Over-correction control. A negative effect that was TESTED and found
        non-significant is exactly what 'no harmed groups' means."""
        graded = [_effect("winners", 2.0, 0.01, True), _effect("noise", -3.0, 0.9, False)]
        rec = _recommend(_result(graded, False, 0.4))
        assert rec.decision.value == "deploy_treatment"
        assert "with no harmed groups" in " ".join(rec.reasoning)
        assert rec.confidence == 0.85
        assert not any("COULD NOT BE DETERMINED" in line for line in rec.reasoning)
        assert _effects_section(_result(graded, False, 0.4)).startswith(
            "2 intersections analysed, 1 statistically significant."
        )

    def test_control_measured_harm_is_still_measured_harm(self):
        harmed = [_effect("winners", 2.0, 0.01, True), _effect("harmed", -3.0, 0.001, True)]
        rec = _recommend(_result(harmed, False, 0.4))
        assert rec.decision.value == "investigate_further"
        assert "show significant harm (max |effect|=3.0000)" in " ".join(rec.reasoning)
        assert rec.confidence == 0.5


# ---------------------------------------------------------------------------
# 4. An overall test that could not run is not a negative result
# ---------------------------------------------------------------------------


class TestOverallSignificanceThreeState:
    def test_refusal_nan_is_not_reported_as_not_significant(self):
        rec = _recommend(_result(TWO_CLEAN, False, 0.4, overall_p=NAN, overall_effect=0.0))
        joined = " ".join(rec.reasoning)
        assert "not statistically significant" not in joined
        assert "nan" not in joined
        assert "COULD NOT BE TESTED" in joined
        assert any("unknown, not as a negative result" in c for c in rec.caveats)
        assert rec.decision.value == "extend_experiment"

    def test_control_a_measured_null_result_still_reads_as_one(self):
        rec = _recommend(_result(TWO_CLEAN, False, 0.4, overall_p=0.9, overall_effect=0.0))
        assert "is not statistically significant (p=0.9000)." in " ".join(rec.reasoning)
        assert not any("COULD NOT BE TESTED" in line for line in rec.reasoning)
        assert rec.decision.value == "extend_experiment"
        assert rec.confidence == 0.3


# ---------------------------------------------------------------------------
# 5. End to end: the producer and the consumer describe the same run
# ---------------------------------------------------------------------------


def test_producer_and_report_surface_agree_on_a_real_experiment():
    """Not a hand-built result. A REAL FairnessExperiment whose intersections
    are constant within each arm, so every bootstrap standard error is 0,
    Cochran's Q has nothing to weight, and the producer withholds the verdict
    while the effects (1.0 and 4.0) plainly differ."""
    n = 60
    control = pd.DataFrame({"g": ["A"] * n + ["B"] * n, "y": [10.0] * n + [20.0] * n})
    treatment = pd.DataFrame({"g": ["A"] * n + ["B"] * n, "y": [11.0] * n + [24.0] * n})
    experiment = FairnessExperiment(
        control_data=control,
        treatment_data=treatment,
        protected_attributes=["g"],
        outcome_column="y",
        config=ExperimentConfig(n_bootstrap=200, min_group_size=10, random_state=7),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = experiment.run_full_analysis()

    # The producer's own three states, so the pin fails loudly if the upstream
    # fix is what regressed rather than this file's.
    assert result.heterogeneity_detected is None
    assert result.heterogeneity_p_value != result.heterogeneity_p_value  # NaN
    assert "heterogeneity=not assessed" in repr(result)
    assert any("could not be" in str(w.message) for w in caught)
    assert [e.effect for e in result.intersection_effects] == [1.0, 4.0]

    analysis = ExperimentAnalysis(result, experiment=experiment)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = analysis.decision_recommendation()
    content = _effects_section(result)

    assert "COULD NOT CHECK" in content
    assert "Not detected" not in content
    assert "nan" not in content
    assert any("COULD NOT BE RUN" in line for line in rec.reasoning)
    assert rec.trade_offs["heterogeneity"].startswith("Unknown:")


# ---------------------------------------------------------------------------
# 6. The ledger cannot clear a site nobody read
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder():
    if not BUILDER.exists():
        pytest.skip("build_verdict_ledger.py not present in this tree")
    sys.path.insert(0, str(BUILDER.parent))
    try:
        import build_verdict_ledger as module
    finally:
        sys.path.pop(0)
    return module


def _site(file, function, kind, snippet, line=1):
    return {
        "file": file,
        "line": line,
        "function": function,
        "kind": kind,
        "snippet": snippet,
        "why": "planted by tests/test_readiness3_recollapse.py",
    }


# A site whose only support is a SHAPE rule: brand new, read by nobody.
NEW_CLEARABLE = _site(
    "src/vfairness/operations/experimentation/analysis.py",
    "_probe_disparity",
    "neutral-dict-default",
    'return payload.get("disparity", 0.0)',
)
# The two per-site tables, i.e. verdicts a person actually wrote.
IN_FIXED_TABLE = _site(
    "src/vfairness/agents/temporal.py",
    "TemporalTracker.detect_drift",
    "neutral-return-under-emptiness-guard",
    "return False",
)
IN_HAND_TABLE = _site(
    "src/vfairness/llm/scorers.py",
    "KeywordToxicityScorer.score",
    "neutral-return-under-emptiness-guard",
    "return 0.0",
)
# A shape rule that proposes "open" clears nothing, so it is still written.
RULE_SAYS_OPEN = _site(
    "src/vfairness/operations/pulse/pipeline.py",
    "_some_new_helper",
    "neutral-ternary-fallback",
    "share = observed_share if flagged else 0.0",
)


def _run_builder(builder, monkeypatch, capsys, sites, tmp_ledger, seed=None):
    """Run the REAL main() over a supplied scan, writing to a throwaway path."""
    if seed is not None:
        tmp_ledger.write_text(json.dumps({"reviewed": seed}, indent=2), encoding="utf-8")
    fake = types.SimpleNamespace(
        run=lambda *a, **k: types.SimpleNamespace(stdout=json.dumps(sites))
    )
    monkeypatch.setattr(builder, "subprocess", fake)
    monkeypatch.setattr(builder, "LEDGER", tmp_ledger)
    monkeypatch.setattr(sys, "argv", ["build_verdict_ledger.py", "--write"])
    code = builder.main()
    written = json.loads(tmp_ledger.read_text(encoding="utf-8"))["reviewed"]
    return code, written, capsys.readouterr().out


def _key(site):
    return f"{site['file']}:{site['function']}:{site['kind']}"


class TestTheLedgerCannotClearASiteNobodyRead:
    def test_refusal_a_new_site_a_rule_would_clear_stays_out_of_the_ledger(
        self, builder, monkeypatch, capsys, tmp_path
    ):
        code, written, out = _run_builder(
            builder,
            monkeypatch,
            capsys,
            [NEW_CLEARABLE, IN_FIXED_TABLE],
            tmp_path / "ledger.json",
        )
        assert _key(NEW_CLEARABLE) not in written, (
            "a site nobody has read acquired a verdict, which is the ledger "
            "answering for the reviewer it exists to require"
        )
        assert code == 1, "the run must refuse, not just mention it"
        assert "NEEDS A HAND VERDICT" in out
        assert "_probe_disparity" in out, "a refusal nobody can see is not a refusal"
        assert "legitimate-fail-closed" in out, "the rule's proposal is shown for the reviewer"

    def test_refusal_the_rule_that_would_have_cleared_it_is_still_intact(self, builder):
        """The fix is a PLACEMENT change, not a deleted rule. If the rule had
        simply been removed, the pin above would pass for the wrong reason and
        every legitimate site would need rewriting by hand."""
        assert builder._hand_verdict(NEW_CLEARABLE) is None
        verdict, reason = builder._rule(NEW_CLEARABLE)
        assert verdict == "legitimate-fail-closed"
        assert "compared against" in reason

    def test_control_hand_written_verdicts_still_place_their_sites(
        self, builder, monkeypatch, capsys, tmp_path
    ):
        """Over-correction control, the half that matters most: reviewed sites
        must not be dragged back to open by a fix aimed at new ones.

        WIDENED 2026-09-10 (READINESS-5) to separate the two cases this used to
        conflate. An existing entry is preserved against a RULE, which is the
        property worth keeping: a mechanical shape match must never overwrite a
        decision a person made. It is NOT preserved against a HAND entry, which
        is a person writing now, because otherwise a verdict that turns out to
        be wrong cannot be corrected at all. That happened: a
        `representation_severity` verdict recorded a reason that was true and
        described exactly the case that was broken; the corrected entry was
        written, the ledger regenerated, and the old reason stayed. The audit
        record was append-only-wrong.

        So the seeded site here carries a HAND entry and must end up with the
        HAND verdict, and a second seeded site with NO hand entry must be
        preserved verbatim.
        """
        seeded_key = "src/vfairness/vision/__init__.py:ndkl:neutral-ternary-fallback"
        no_hand_key = _key(RULE_SAYS_OPEN)
        seeded = {
            seeded_key: {"verdict": "legitimate", "reason": "seeded by this test"},
            no_hand_key: {"verdict": "legitimate", "reason": "a person read this one"},
        }
        code, written, _ = _run_builder(
            builder,
            monkeypatch,
            capsys,
            [
                IN_FIXED_TABLE,
                IN_HAND_TABLE,
                RULE_SAYS_OPEN,
                _site(
                    "src/vfairness/vision/__init__.py",
                    "ndkl",
                    "neutral-ternary-fallback",
                    "z = total if entries else 1.0",
                ),
            ],
            tmp_path / "ledger.json",
            seed=seeded,
        )
        assert code == 0, "nothing here needs a person, so nothing may be refused"
        assert written[_key(IN_FIXED_TABLE)]["verdict"] == "fixed"
        assert written[_key(IN_HAND_TABLE)]["verdict"] == "legitimate"

        # Preserved against a RULE. This site's rule says `open`, and a seeded
        # human verdict must survive it: that is the whole point of the
        # preserve branch.
        assert written[no_hand_key] == seeded[no_hand_key], (
            "a rule dragged a reviewed site back to open"
        )

        # Superseded by a HAND entry, because a person can correct a person.
        # Never back to `open`: the site stays reviewed either way.
        assert written[seeded_key]["verdict"] != "open"
        assert written[seeded_key] == {
            "verdict": builder.HAND["vision/__init__.py:ndkl"][0],
            "reason": builder.HAND["vision/__init__.py:ndkl"][1],
        }, (
            "the hand table is the reviewed source of truth and did not reach "
            "the ledger, so a wrong verdict cannot be corrected"
        )

    def test_control_the_reviewed_corpus_is_reproduced_from_the_live_tree(
        self, builder, monkeypatch, capsys, tmp_path
    ):
        """The same control against the REAL scanner output: every verdict the
        committed ledger holds is written back unchanged. Subset, not equality,
        because another lane's new site may legitimately appear as open."""
        if not LEDGER.exists():
            pytest.skip("ledger not present in this tree")
        committed = json.loads(LEDGER.read_text(encoding="utf-8"))["reviewed"]
        import subprocess

        scan = json.loads(
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "scan_fabricated_verdicts.py"), "--json"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        _, written, _ = _run_builder(
            builder, monkeypatch, capsys, scan, tmp_path / "ledger.json", seed=committed
        )
        # A floor, not a target: it exists so a mass clear cannot pass as a
        # clean corpus. It moves DOWN only when a fix genuinely removes a
        # neutral default from the code, and each decrement is recorded here so
        # the next reader can tell that apart from an accidental wipe.
        #
        # 137 -> 135 on 2026-09-10 (READINESS-5). Two sites stopped existing:
        #   scorers.py KeywordSentimentScorer.score, whose neutral 0.0 under
        #     the emptiness guard became a NaN plus an UnscorableTextWarning;
        #   llm_probe.py _evidence_from_cells, whose `.get("pValue", 1.0)`
        #     became an explicit None check.
        # 135 -> 134 later the same day: gate.py _compute_default_metrics._gap,
        #   whose unreachable `else 0.0` on a deployment-BLOCKING surface became
        #   `else float("nan")`, so the branch can no longer fail OPEN if its
        #   reachability ever changes.
        # 134 -> 132 at the end of the READINESS-6 fix waves. FIVE sites stopped
        #   existing and THREE new ones were reviewed, so the net is -2:
        #     gone: collusion.analyze (nan-swallowing), emergent.analyze
        #       (the `1e6 if system_bias > 0 else 1.0` sentinel that made a
        #       0.001 noise gap a CRITICAL finding), groupthink
        #       .analyze_convergence (nan-swallowing), and both sequential.py
        #       neutral ternaries (cusum_drift, sequential_fairness_drift).
        #     new: embedding_bias.weat, validator.to_junit_xml, and
        #       detector.full_audit, each with a hand verdict and its reason.
        # 133 -> 132 on 2026-09-17 (BGL final sweep, group u02). One site
        #   stopped existing: proxy.py compute_proxy_correlations, whose
        #   `correlations.get(<statistic>, 0)` became `float("nan")`. Its
        #   standing verdict was legitimate-fail-closed on the reasoning that
        #   `nan > t` suppresses a finding, but nothing compared this value to a
        #   threshold: it went to _determine_risk_level, which GRADES it, and 0
        #   grades NEGLIGIBLE. Measured on a constant protected attribute, where
        #   the correlation ratio is never computed and its key is therefore
        #   absent, the function returned {'primary_correlation': 0,
        #   'risk_level': 'negligible'} for a pair nothing was measured about.
        #   Pinned in tests/test_bgl_final_u02.py.
        # 132 -> 130 on 2026-09-17 (surface grading, wave 1). FOUR sites stopped
        #   existing and ONE new one was reviewed, so the net is -2. The four
        #   were not fixed away one by one: wave 1 split three fabricating
        #   functions into a `_score_or_none` / `_missing_column_nan` shape, so
        #   the neutral value moved to a differently named function and the old
        #   key stopped matching any code:
        #     scorers.py RepresentationScorer.score, whose neutral 0.0 for a
        #       text the tokeniser could not read is now None in
        #       _score_or_none and NaN plus an UnscorableTextWarning at the
        #       surface. Measured before: eight Chinese generations against
        #       eight English ones reported delta -0.4, p 0.000138,
        #       is_significant True, assessed True, a significant erasure
        #       finding manufactured out of the scorer being unable to read one
        #       group's script.
        #     tracker.py TemporalFairnessAnalyzer.detect_trend and
        #       .forecast_metric, whose neutral ternaries were replaced.
        #     store.py MetricsStore.ingest_alert, whose neutral dict default
        #       was replaced.
        #   new: tracker.py FairnessMonitor._missing_column_nan, legitimate:
        #     it answers "is a configured column missing" and False means none
        #     is, so the neutral value IS the measurement.
        #   Also removed in the same change, and NOT counted above because the
        #   scanner never had a verdict for it: the dead
        #   `if n_categories > 0 else 0` in RepresentationScorer, unreachable
        #   because coverage_score divides by the same value three lines above.
        # 130 -> 129 on 2026-09-17 (surface grading, batch g005). ONE site
        #   stopped existing: analyzer.py CalibrationReport.summary, whose
        #   `self.overall_metrics.get('ece', 0)` (and the same default for mce,
        #   brier and the per-group ECE table) became an explicit three-state
        #   render. Its standing verdict was legitimate-fail-closed on the
        #   reasoning that `nan > t` suppresses a finding, but summary() does
        #   not compare this value to a threshold, it PRINTS it, and on the ECE
        #   / MCE / Brier scale 0.0000 is the best score there is. Measured on a
        #   report with overall_metrics={} the text read "Expected Calibration
        #   Error (ECE): 0.0000 / Maximum Calibration Error (MCE): 0.0000 /
        #   Brier Score: 0.0000" beside "Well Calibrated: Not assessable (not
        #   measured)": three fabricated perfect scores next to an honest
        #   refusal, from one report. Pinned in
        #   tests/test_surface_grade_g005.py.
        # 129 -> 126 on 2026-09-25 (sidecar outage wave). THREE sites stopped
        #   existing, all in llm/scorers.py and all the same change: the outage
        #   branches of SidecarSentimentScorer.score, SidecarToxicityScorer.score
        #   and _sidecar_stereotype_score returned 0.0 under an emptiness guard
        #   and now return nan, so there is no neutral return left to review. On
        #   toxicity 0.0 means NOT TOXIC and on sentiment NEUTRAL, so a fairness
        #   audit run against a dead sidecar reported "no toxicity difference
        #   between these groups", and the warning that was supposed to cover it
        #   fires ONCE PER PROCESS. Their standing verdicts are kept under
        #   `_retired` in the ledger rather than deleted, so a reader can see the
        #   sites were removed by a fix and not dropped as inconvenient. Pinned in
        #   tests/test_audit_wave4_llm.py and
        #   tests/test_no_aggregator_fabricates_a_verdict.py, whose tripwire
        #   assertion message was "the wave-4 API-compatibility decision changed
        #   silently".
        # 126 -> 124 on 2026-09-25 (B1 gap wave). TWO sites stopped existing, the
        #   two copies of `_correlation_ratio_with_pvalue` in
        #   preprocessing/bias_detection/proxy.py and
        #   preprocessing/feature_engineering/correlation.py. Their `ss_total <= 0`
        #   branch returned `0.0, 1.0` and now returns `nan, nan`. This one
        #   OVERTURNS a standing "legitimate" verdict rather than removing a site
        #   nobody had judged, so the superseded verdict and its reasoning are both
        #   kept verbatim under `_retired`. The verdict's argument ("no categorical
        #   can explain variation that does not exist") supports eta = 0; it cannot
        #   support pvalue = 1.0, which reports that an ANOVA found no significance
        #   when scipy's f_oneway returns nan for all-identical input and the ANOVA
        #   never ran. What settled it was an inconsistency inside the library: the
        #   same constant feature against a TWO-level attribute goes to
        #   _point_biserial and came back nan, so one undefined quantity had two
        #   different published answers depending on the attribute's cardinality.
        #   Pinned in tests/test_bgl2_analysis_surfaces.py and, for the genuinely
        #   measured zero that must survive, tests/test_readiness6_names.py, whose
        #   over-correction control had used a CONSTANT variable as its example of a
        #   real zero and was corrected in the same change.
        # 124 -> 122 on 2026-09-27 (BGL3 campaign). SEVEN sites stopped existing and
        #   FIVE new ones were verdicted, so the corpus fell by two. The seven are four
        #   in in_processing/analyzer.py (_generate_action_items,
        #   _identify_critical_issues, FairnessTrainingReport.summary and
        #   baseline_comparison_summary._overall_accuracy), GroupFairnessRegularizer
        #   .forward in in_processing/regularizers/fairness_regularizers.py,
        #   ece_confidence_intervals in post_processing/calibration/metrics.py, and
        #   calibration_vs_error_parity in post_processing/calibration/tradeoffs.py.
        #   Each had its neutral default replaced by a three-state answer, so there is
        #   no neutral return left to review. Their standing verdicts are kept under
        #   `_retired` with the reason each one recorded, because that reason is the
        #   history of the defect.
        #
        #   Two of those removals matter beyond the arithmetic. summary() printed
        #   "Accuracy: 0.0000" directly above two lines that refuse to invent anything,
        #   and baseline_comparison_summary reported "reduced disparity by 0.250 at no
        #   accuracy cost" where both sides were unmeasured, so the difference between
        #   them was arithmetic on two absences.
        #
        #   The five ADDED sites are the residue of the same campaign's fixes rather
        #   than new defects: two `X is not None` three-state tests in
        #   MetricsStore.compute_health_score, a clamp lower bound in
        #   ContextualStereotypeScorer, a percentage inside a warning sentence in
        #   streaming_demographic_parity, and a `correlation_matrix is not None` test in
        #   create_analysis_dashboard. The sixth, _score_ethics_symmetry, is verdicted
        #   `fixed`: its Jaccard fallback is gone and the surviving ternary is the
        #   measurement.
        assert len(committed) >= 122
        missing = sorted(set(committed) - set(written))
        assert not missing, f"reviewed sites lost their verdict: {missing[:5]}"
        changed = sorted(k for k in committed if written[k] != committed[k])
        assert not changed, f"reviewed sites had their verdict rewritten: {changed[:5]}"
