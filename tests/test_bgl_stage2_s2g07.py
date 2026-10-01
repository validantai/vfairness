"""Beta Go-Live Stage 2, group s2g07: nine fabricated measurements, closed.

Every test here asserts at the PUBLIC entry point named in the finding, and
every one is paired with a CONTROL on healthy data: a fix that makes
everything refuse is a worse defect than the one it replaces, and it passes
any test that only exercises the degenerate case.

The before/after values in the docstrings were MEASURED by running the
finding's own repro script against this checkout, not copied from the filing.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 1. groupthink_detector: an UNMEASURABLE agent was an independent singleton
# ---------------------------------------------------------------------------


def _rounds_with_an_unmeasurable_agent():
    """a and b diverge, c emits a zero-length vector every round.

    Cosine similarity against a zero vector is undefined, so c's agreement
    with everyone (and with itself) is nan.
    """
    return [
        {"a": [1 - 0.1 * t, 0.1 * t], "b": [0.1 * t, 1 - 0.1 * t], "c": [0.0, 0.0]}
        for t in range(6)
    ]


def _healthy_rounds():
    """Three agents converging on real, non-degenerate output vectors."""
    return [
        {
            "a": [1 - 0.1 * t, 0.1 * t],
            "b": [0.95 - 0.1 * t, 0.05 + 0.1 * t],
            "c": [0.9 - 0.1 * t, 0.1 + 0.1 * t],
        }
        for t in range(6)
    ]


def test_the_fixture_really_does_make_one_agent_unmeasurable():
    """Assert the fixture exercises the branch, before asserting the branch.

    In Stage 1 a green sabotage turned out to be a fixture that never reached
    the code under test.
    """
    from vfairness import GroupthinkDetector

    final = _rounds_with_an_unmeasurable_agent()[-1]
    sim = GroupthinkDetector._cosine_similarity(
        np.array(final["c"], dtype=float), np.array(final["a"], dtype=float)
    )
    assert math.isnan(sim), "the fixture must produce an undefined agreement, or it proves nothing"


def test_an_unmeasurable_agent_is_not_reported_as_an_independent_coalition():
    """BEFORE: coalition_structure=[['a','b'], ['c']] -- c presented as a
    measured independent, the one two-state field on a result whose six other
    fields all refused correctly.
    AFTER: coalition_structure=[['a','b']], unplaced_agents=['c'],
    coalitions_detectable=False, coalition_note starts 'COULD NOT CHECK'.
    """
    from vfairness import GroupthinkDetector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = GroupthinkDetector().analyze_convergence(_rounds_with_an_unmeasurable_agent())

    flat = [name for coalition in result.coalition_structure for name in coalition]
    assert "c" not in flat, "an agent nobody could measure was placed in a coalition"
    assert ["c"] == result.unplaced_agents
    assert result.coalitions_detectable is not True
    assert "COULD NOT CHECK" in result.coalition_note
    # and the reader can tell the third state WITHOUT reading the source
    assert result.coalition_note != ""


def test_detect_coalitions_refuses_the_singleton_directly_too():
    """The public method, called on the matrix the finding used."""
    from vfairness import GroupthinkDetector

    matrix = np.array([[1.0, 0.95, np.nan], [0.95, 1.0, np.nan], [np.nan, np.nan, np.nan]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coalitions = GroupthinkDetector().detect_coalitions(matrix)
    assert coalitions == [{0, 1}], f"got {coalitions!r}; index 2 must not be its own coalition"
    assert any("COULD NOT BE CHECKED" in str(w.message) for w in caught)


def test_control_healthy_agents_still_form_a_measured_coalition():
    """OVER-CORRECTION CONTROL. Measured: [['a','b','c']], no unplaced agent,
    coalitions_detectable True, and ZERO warnings."""
    from vfairness import GroupthinkDetector

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = GroupthinkDetector().analyze_convergence(_healthy_rounds())

    assert result.coalition_structure == [["a", "b", "c"]]
    assert result.unplaced_agents == []
    assert result.coalitions_detectable is True
    assert result.coalition_note == ""
    assert [str(w.message) for w in caught] == []
    assert math.isfinite(result.p_value)


# ---------------------------------------------------------------------------
# 2. cot_faithfulness_analyzer: an accusation issued on zero reasoning text
# ---------------------------------------------------------------------------

_REAL_COT = "The candidate has ten years of experience and strong references."


def test_an_empty_chain_of_thought_is_not_graded_unfaithful_silent():
    """BEFORE: cot_similarity=1.0, cot_changed=False,
    classification='unfaithful_silent' -- a positive accusation of hidden
    demographic influence, for a model that supplied no reasoning at all, and
    nothing in the 11 result fields separated it from the healthy run.
    AFTER: cot_similarity=nan, cot_changed=None,
    classification='not_assessed', assessed=False, not_assessed_reason set.
    """
    from vfairness import CoTFaithfulnessAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = CoTFaithfulnessAnalyzer().analyze_pair(
            original_output="Recommended salary: $120,000",
            variant_output="Recommended salary: $95,000",
            original_cot="",
            variant_cot="",
            demographic_cue="gender",
        )

    assert r.classification == "not_assessed"
    assert r.assessed is False
    assert math.isnan(r.cot_similarity)
    assert r.cot_changed is None
    assert "COULD NOT CHECK" in r.not_assessed_reason
    # THE REAL FINDING SURVIVES: the outputs genuinely differ, and that is
    # still reported. Only the verdict about the reasoning is withheld.
    assert r.output_changed is True


def test_the_batch_rates_exclude_what_could_not_be_assessed():
    """BEFORE: five empty scenarios -> n_consistent=5, faithfulness_score=0.0,
    silent_influence_rate=0.0, warnings=[].
    AFTER: n_not_assessed=5, both rates nan.
    """
    from vfairness import CoTFaithfulnessAnalyzer

    scenarios = [
        {
            "scenario_id": f"s{i}",
            "original_output": "",
            "variant_output": "",
            "original_cot": "",
            "variant_cot": "",
            "demographic_cue": "gender",
        }
        for i in range(5)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = CoTFaithfulnessAnalyzer().analyze_batch(scenarios)

    assert report.n_not_assessed == 5
    assert report.n_consistent == 0, "an unassessable scenario is not a consistent one"
    assert math.isnan(report.faithfulness_score)
    assert math.isnan(report.silent_influence_rate)


def test_control_real_reasoning_is_still_graded_unfaithful_silent():
    """OVER-CORRECTION CONTROL, and the module's whole purpose: an output that
    moves $25,000 with reasoning that never mentions the cue IS a finding.
    Measured: classification='unfaithful_silent', cot_similarity 0.9999...,
    assessed=True, silent_influence_rate=1.0, zero warnings."""
    from vfairness import CoTFaithfulnessAnalyzer

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = CoTFaithfulnessAnalyzer().analyze_pair(
            original_output="Recommended salary: $120,000",
            variant_output="Recommended salary: $95,000",
            original_cot=_REAL_COT,
            variant_cot=_REAL_COT,
            demographic_cue="gender",
        )
    assert r.classification == "unfaithful_silent"
    assert r.assessed is True
    assert r.cot_similarity == pytest.approx(1.0)
    assert r.cot_changed is False
    assert [str(w.message) for w in caught] == []


def test_control_one_empty_side_is_still_a_measured_total_change():
    """A model that reasoned in one condition and not in the other is a REAL
    observation, not an absence of evidence. Refusing there too would delete a
    finding."""
    from vfairness import CoTFaithfulnessAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = CoTFaithfulnessAnalyzer().analyze_pair(
            original_output="Recommended salary: $120,000",
            variant_output="Recommended salary: $95,000",
            original_cot=_REAL_COT,
            variant_cot="",
            demographic_cue="gender",
        )
    assert r.assessed is True
    assert r.cot_similarity == 0.0
    assert r.cot_changed is True


# ---------------------------------------------------------------------------
# 3. proxy_feature_score: NaN read as "below the proxy threshold"
# ---------------------------------------------------------------------------


def _shap_fixture(spoil: bool):
    """zipcode carries a blatant, fully measured disparity of 1.0; income is
    flat. With spoil=True one cell of the UNRELATED income column is NaN."""
    groups = np.array(["A"] * 50 + ["B"] * 50)
    sv = np.zeros((100, 2))
    sv[:50, 1] = 0.5
    sv[50:, 1] = -0.5
    if spoil:
        sv[3, 0] = np.nan
    return sv, groups


def test_proxy_score_returns_none_for_a_feature_it_could_not_measure():
    """BEFORE: proxy_score({'income': nan, 'zipcode': 1.0}) ->
    {'income': nan, 'zipcode': nan}: one bad column poisoned the shared
    denominator, and `nan >= 0.15` is False, so the MEASURED proxy went
    unflagged.
    AFTER: {'income': None, 'zipcode': 1.0}. The measured feature keeps its
    measured share, and None cannot be silently compared against a threshold.
    """
    from vfairness.xai.decomposition.shap_fairness import proxy_score

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = proxy_score({"income": float("nan"), "zipcode": 1.0})

    assert scores["income"] is None
    assert scores["zipcode"] == pytest.approx(1.0)
    assert any("COULD NOT BE" in str(w.message) for w in caught)
    flagged = [n for n, s in scores.items() if s is not None and s >= 0.15]
    assert flagged == ["zipcode"], "the measured proxy must still be flagged"


def test_proxy_score_refuses_everything_only_when_nothing_is_finite():
    from vfairness.xai.decomposition.shap_fairness import proxy_score

    nan = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert proxy_score({"a": nan, "b": nan}) == {"a": None, "b": None}


def test_control_a_measured_zero_denominator_is_still_a_measured_zero():
    """NOT a could-not-check, and refusing here would delete a real finding.
    denom is a sum of ABSOLUTE values, so a finite denom of 0 means every
    per-feature disparity was measured and every one is exactly 0.0: "no
    feature is a proxy" is the right answer. Pinned in two existing test
    modules; re-pinned here so a later widening of the refusal is caught."""
    from vfairness.xai.decomposition.shap_fairness import proxy_score

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert proxy_score({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    assert [str(w.message) for w in caught] == []


def test_the_public_decomposition_refuses_a_non_finite_column():
    """The only public call that reaches proxy_score. It must not answer
    'no proxy features detected' for a frame it could not decompose."""
    from vfairness.xai import lundberg_fairness_decomposition

    sv, groups = _shap_fixture(spoil=True)
    with pytest.raises(ValueError, match="COULD NOT BE MEASURED"):
        lundberg_fairness_decomposition(
            shap_values=sv,
            group_labels=groups,
            feature_names=["income", "zipcode"],
            metric="demographic_parity",
            protected_attribute="race",
            subject_id="s1",
            audit_artifact_id="a1",
        )


def test_control_the_decomposition_still_flags_the_real_proxy():
    """OVER-CORRECTION CONTROL. Measured: proxy_scores={'income': 0.0,
    'zipcode': 1.0}, flagged_proxies=['zipcode']."""
    from vfairness.xai import lundberg_fairness_decomposition

    sv, groups = _shap_fixture(spoil=False)
    d = lundberg_fairness_decomposition(
        shap_values=sv,
        group_labels=groups,
        feature_names=["income", "zipcode"],
        metric="demographic_parity",
        protected_attribute="race",
        subject_id="s1",
        audit_artifact_id="a1",
    )
    assert d.flagged_proxies == ["zipcode"]
    assert d.proxy_scores["zipcode"] == pytest.approx(1.0)
    d.assert_identity()


# ---------------------------------------------------------------------------
# 4. pipeline_tracker: the strongest stage lost the ranking in silence
# ---------------------------------------------------------------------------


def _pipeline(spoil: bool):
    """Three stages. 'action' carries the blatant disparity. With spoil=True,
    ONE outcome out of forty in the PRECEDING stage is not finite."""
    from vfairness import PipelineTracker

    rng = np.random.default_rng(1)
    tracker = PipelineTracker(["retrieval", "reasoning", "action"])
    for stage, (ma, mb) in {
        "retrieval": (0.55, 0.50),
        "reasoning": (0.80, 0.39),
        "action": (0.80, 0.39),
    }.items():
        a, b = rng.normal(ma, 0.1, 40), rng.normal(mb, 0.1, 40)
        if spoil and stage == "reasoning":
            a = a.copy()
            a[0] = np.nan
        tracker.record_stage(stage, a, b)
    return tracker


def test_the_fixture_leaves_the_strongest_stage_testable_but_unrankable():
    """Assert the shape the defect needs, before asserting the defect: the
    'action' stage must pass the p-value filter and still have no ranking key.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = {r.stage_name: r for r in _pipeline(spoil=True).compute_cumulative()}
    action = rows["action"]
    assert math.isfinite(action.bias_metrics["p_value"]), "must survive the `testable` filter"
    assert action.bias_metrics["p_value"] < 1e-10
    assert action.bias_metrics["abs_disparity"] > rows["retrieval"].bias_metrics["abs_disparity"]
    assert not math.isfinite(action.stage_contribution), "its ranking key must be unmeasurable"


def test_a_ranking_that_could_not_be_completed_names_no_stage():
    """BEFORE: identify_bias_source() -> 'retrieval' (abs_disparity 0.065,
    p=0.125) while 'action' carried abs_disparity 0.434 at p=1.8e-14, and the
    warning named only 'reasoning', which was affirmatively false about
    'action'.
    AFTER: None, with a warning naming 'action'.
    """
    tracker = _pipeline(spoil=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source = tracker.identify_bias_source()

    assert source is None, f"named {source!r} as the bias source of a ranking nobody finished"
    messages = " ".join(str(w.message) for w in caught)
    assert "action" in messages
    assert "could not check" in messages.lower()


def test_compute_cumulative_says_which_stages_it_could_not_measure():
    """BEFORE: compute_cumulative emitted NO warning at all; `untestable` was
    populated only by the n<=1 branch, never by a NaN mean."""
    tracker = _pipeline(spoil=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = tracker.compute_cumulative()

    messages = " ".join(str(w.message) for w in caught)
    assert "COULD NOT BE MEASURED" in messages
    assert "reasoning" in messages
    by_name = {r.stage_name: r for r in rows}
    assert by_name["reasoning"].bias_metrics["n_a_nonfinite"] == 1
    assert by_name["action"].bias_metrics["n_a_nonfinite"] == 0


def test_control_a_clean_pipeline_still_names_its_bias_source():
    """OVER-CORRECTION CONTROL. Measured on the same fixture with no NaN:
    retrieval contrib +0.0649, reasoning +0.3680, action +0.0015, so
    'reasoning' is the stage that ADDS the most bias and is still named, with
    no warning.

    Which sharpens the defect test above: 'reasoning' is exactly the stage the
    NaN erases, so the spoiled run lost the true source AND named 'retrieval',
    a stage with a twentieth of the marginal contribution.
    """
    tracker = _pipeline(spoil=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source = tracker.identify_bias_source()
    assert source == "reasoning"
    assert [str(w.message) for w in caught] == []
    contributions = {r.stage_name: r.stage_contribution for r in tracker.compute_cumulative()}
    assert contributions["reasoning"] > contributions["retrieval"] > 0
    assert all(math.isfinite(v) for v in contributions.values())


def test_control_an_untestable_but_measured_stage_is_still_only_excluded():
    """The refusal is deliberately NARROWER than the existing p-value
    exclusion: a stage with one observation per arm has no significance test
    but DOES have a measured contribution, so the largest testable contributor
    is still named. Pinned in tests/test_readiness6_remainder.py; re-pinned
    here because widening my guard would silently overturn it."""
    from vfairness import PipelineTracker

    rng = np.random.default_rng(7)
    tracker = PipelineTracker(["screen", "interview"])
    tracker.record_stage("screen", rng.normal(0.8, 0.1, 40), rng.normal(0.4, 0.1, 40))
    tracker.record_stage("interview", np.array([1.0]), np.array([0.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tracker.identify_bias_source() == "screen"


# ---------------------------------------------------------------------------
# 5. compute_feature_correlations: eta = 1.000 produced by arithmetic
# ---------------------------------------------------------------------------


def _frame_with_a_unique_id_column(seed: int):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "customer_id": [f"id{i}" for i in range(n)],
            "city": np.resize(np.array(["A", "B", "C"]), n),
        }
    )


@pytest.mark.parametrize("seed", [0, 7])
def test_a_unique_per_row_column_does_not_correlate_perfectly_with_anything(seed):
    """BEFORE: correlations.loc['customer_id','age'] = 1.000000 on two
    different random age vectors (seeds 0 and 7), against 60 unique ids. One
    row per level means ss_between == ss_total identically, so eta = sqrt(1)
    for ANY numbers.
    AFTER: nan, with a warning naming the column.
    """
    from vfairness.preprocessing.feature_engineering import compute_feature_correlations

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_feature_correlations(
            _frame_with_a_unique_id_column(seed), protected_attributes=["age"]
        )

    cell = result.correlations.loc["customer_id", "age"]
    assert not np.isfinite(cell), f"a correlation of {cell} was reported from one row per level"
    assert any("COULD NOT BE MEASURED" in str(w.message) for w in caught)
    assert any("customer_id" in str(w.message) for w in caught)


@pytest.mark.parametrize("seed", [0, 7])
def test_control_the_genuine_categorical_column_is_still_measured(seed):
    """OVER-CORRECTION CONTROL. 'city' has three levels over sixty rows and is
    a real measurement; RE-MEASURED 2026-09-17 as 0.095829 (seed 0) and
    0.312773 (seed 7) on this exact fixture. The figures written here first,
    0.334933 and 0.134669, were not the values this fixture produces, and a
    stated measurement that nobody can reproduce is the defect this campaign
    removes, sitting in the control that is supposed to catch it. The tight
    assertion on these numbers lives in tests/test_bgl_final_d02.py."""
    from vfairness.preprocessing.feature_engineering import compute_feature_correlations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_feature_correlations(
            _frame_with_a_unique_id_column(seed), protected_attributes=["age"]
        )
    cell = result.correlations.loc["city", "age"]
    assert np.isfinite(cell), "a genuine 3-level column must still be measured"
    assert 0.0 <= cell < 1.0


def test_control_a_strong_real_association_still_reads_as_strong():
    """The guard must not flatten a genuine continuous-vs-categorical proxy."""
    from vfairness.preprocessing.feature_engineering import compute_feature_correlations

    rng = np.random.default_rng(3)
    city = np.resize(np.array(["A", "B", "C"]), 90)
    offsets = {"A": 0.0, "B": 20.0, "C": 40.0}
    df = pd.DataFrame(
        {"age": np.array([offsets[c] for c in city]) + rng.normal(0, 1.0, 90), "city": city}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_feature_correlations(df, protected_attributes=["age"])
    assert result.correlations.loc["city", "age"] > 0.9


# ---------------------------------------------------------------------------
# 6. find_proxy_chains: [] meant both "none found" and "could not look"
# ---------------------------------------------------------------------------


def _chain_frame(n: int, seed: int):
    """zipcode -> income -> race, a real chain at every n."""
    rng = np.random.default_rng(seed)
    race = (rng.random(n) < 0.5).astype(float)
    income = race * 2 + rng.normal(0, 1, n)
    residual = income - race * 2
    return pd.DataFrame(
        {"race": race, "income": income, "zipcode": residual * 3 + rng.normal(0, 0.6, n)}
    )


@pytest.mark.parametrize("n,seed", [(29, 4), (25, 4)])
def test_a_chain_that_could_not_be_examined_is_not_reported_as_absent(n, seed):
    """BEFORE: [] with warnings=[] at n=25 and n=29, where the chain is
    STRONGER (0.7947 / 0.7721) than in the passing n=400 control.
    AFTER: still an empty chain list, but complete=False and every skipped
    pair named with its reason.
    """
    from vfairness import find_proxy_chains

    frame = _chain_frame(n, seed)
    # the fixture really does hold the chain
    assert abs(frame["zipcode"].corr(frame["income"])) > 0.3
    assert abs(frame["income"].corr(frame["race"])) > 0.3

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_proxy_chains(frame, "race")

    assert result.complete is False
    assert result.pairs_not_computed, "the skipped pairs must be named, not silently dropped"
    assert all("insufficient_samples" in why for _a, _b, why in result.pairs_not_computed)
    assert any("COULD NOT BE CORRELATED" in str(w.message) for w in caught)


def test_a_protected_attribute_that_is_not_in_the_frame_is_not_a_clean_result():
    """BEFORE: [] silently, for a column that is not there at all."""
    from vfairness import find_proxy_chains

    frame = _chain_frame(400, 1).drop(columns=["race"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_proxy_chains(frame, "race")

    assert result.protected_attribute_present is False
    assert result.complete is False
    assert any("COULD NOT CHECK" in str(w.message) for w in caught)


def test_control_the_healthy_chain_is_still_found_and_still_complete():
    """OVER-CORRECTION CONTROL. Measured at n=400: one chain
    zipcode -> income -> race, indirect_correlation 0.5094, complete=True,
    zero warnings."""
    from vfairness import find_proxy_chains

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_proxy_chains(_chain_frame(400, 1), "race")

    assert result.complete is True
    assert result.pairs_not_computed == []
    assert [str(w.message) for w in caught] == []
    assert len(result) == 1
    assert result[0]["chain"] == ["zipcode", "income", "race"]
    assert result[0]["indirect_correlation"] > 0.4


def test_the_result_is_still_a_list_for_every_existing_consumer():
    """The MCP tool counts with `isinstance(v, list)`, the analyzer calls
    `.extend`, and the pulse orchestrator relies on `... or []` being falsy
    when empty. None of them may break."""
    import json

    from vfairness import find_proxy_chains

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = find_proxy_chains(_chain_frame(400, 1), "race")
        empty = find_proxy_chains(_chain_frame(400, 1).drop(columns=["race"]), "race")

    assert isinstance(found, list) and isinstance(empty, list)
    assert not empty, "an empty result must stay falsy for the `or []` consumers"
    accumulator: list = []
    accumulator.extend(found)
    assert len(accumulator) == len(found) == 1
    assert json.loads(json.dumps(found))[0]["chain_length"] == 2


# ---------------------------------------------------------------------------
# 7. generate_model_card: an all-clear printed on zero evidence
# ---------------------------------------------------------------------------

_PROFILE = {
    "name": "M",
    "purpose": "p",
    "domain": "d",
    "version": "1",
    "protected_attributes": ["race"],
}


def _ethics_block(card: str) -> str:
    return card[card.find("## Ethical Considerations") :]


@pytest.mark.parametrize("payload", [{"metrics": []}, {}])
def test_a_card_with_no_metrics_makes_no_all_clear(payload):
    """BEFORE: 'This model has been evaluated using the vfairness fairness
    assessment pipeline.' followed by 'All evaluated fairness metrics met
    their thresholds at the time of assessment.' on ZERO metric rows, and
    byte-identical whether 'metrics' was [] or absent.
    AFTER: a Not assessed arm, and no claim that an evaluation happened.
    """
    from vfairness.operations.reporting import generate_model_card

    block = _ethics_block(generate_model_card(_PROFILE, payload))
    assert "All evaluated fairness metrics met their thresholds" not in block
    assert "has been evaluated using the vfairness" not in block
    assert "**Not assessed:** no fairness metric was recorded" in block


def test_control_a_card_whose_metrics_all_passed_still_says_so():
    """OVER-CORRECTION CONTROL. The all-clear is a real finding when there is
    something to clear."""
    from vfairness.operations.reporting import generate_model_card

    block = _ethics_block(
        generate_model_card(
            _PROFILE,
            {
                "metrics": [
                    {"name": "demographic_parity", "value": 0.02, "threshold": 0.1, "passed": True}
                ]
            },
        )
    )
    assert "All evaluated fairness metrics met their thresholds" in block
    assert "This model has been evaluated using the vfairness" in block


def test_control_an_ungradable_row_still_reaches_the_existing_not_assessed_arm():
    """The pre-existing three-state arm must keep firing: the new `not
    metrics` branch sits ABOVE it and must not shadow it."""
    from vfairness.operations.reporting import generate_model_card

    block = _ethics_block(
        generate_model_card(_PROFILE, {"metrics": [{"name": "dp", "value": 0.1}]})
    )
    assert "recorded fairness metric(s) carry" in block
    assert "All evaluated fairness metrics met their thresholds" not in block


# ---------------------------------------------------------------------------
# 8. simulate_threshold_change: a what-if quantified over an empty window
# ---------------------------------------------------------------------------


def _alerting_store(hours_ago: float):
    from vfairness import MetricsStore

    now = datetime.now()
    df = pd.DataFrame(
        [
            {
                "timestamp": now - timedelta(hours=hours_ago + i * 0.1),
                "metric": "demographic_parity",
                "value": 0.30 if i % 2 == 0 else 0.03,
                "group": "female" if i % 2 == 0 else "male",
                "alert": i % 2 == 0,
            }
            for i in range(24)
        ]
    )
    store = MetricsStore()
    store.ingest_dataframe(df, group_col="group", alert_col="alert")
    return store


_REFUSED_FIELDS = (
    "current_alerts",
    "projected_alerts",
    "groups_impacted",
    "change_abs",
    "change_pct",
)


def test_an_empty_window_simulates_nothing_and_says_so():
    """BEFORE: a store holding 24 graded readings of which 12 ARE alerts,
    queried on a window that misses them, returned current_alerts=0,
    projected_alerts=0, groups_impacted=[], change_abs=0, change_pct=0.0, with
    'direction' and 'not_simulated_reason' absent entirely. 'Alerts 0 -> 0,
    change 0 (0.0%)' is flatly false of that store.
    AFTER: all five None, plus a COULD NOT CHECK sentence.
    """
    from vfairness import InteractiveDashboard

    store = _alerting_store(hours_ago=24 * 60)
    result = InteractiveDashboard(store).simulate_threshold_change("demographic_parity", 0.5)

    for field in _REFUSED_FIELDS:
        assert result[field] is None, f"{field} was quantified for a window holding no record"
    assert "COULD NOT CHECK" in result["not_simulated_reason"]
    assert "direction" in result, "the key must not vanish on the empty path"


def test_a_fresh_store_and_a_typo_are_not_a_zero_simulation_either():
    """Three distinct causes used to return the identical quantified zero."""
    from vfairness import InteractiveDashboard, MetricsStore

    fresh = InteractiveDashboard(MetricsStore()).simulate_threshold_change(
        "demographic_parity", 0.5
    )
    typo = InteractiveDashboard(_alerting_store(hours_ago=1)).simulate_threshold_change(
        "demogrpahic_parity", 0.5
    )
    for result in (fresh, typo):
        for field in _REFUSED_FIELDS:
            assert result[field] is None
        assert result["not_simulated_reason"]
    # the fail-closed direction guard is no longer bypassed by an empty window
    assert typo["direction"] == "unknown"


def test_control_a_populated_window_still_simulates():
    """OVER-CORRECTION CONTROL. Measured: current_alerts=12,
    projected_alerts=0, change_abs=-12, change_pct=-100.0,
    records_simulated=24."""
    from vfairness import InteractiveDashboard

    result = InteractiveDashboard(_alerting_store(hours_ago=1)).simulate_threshold_change(
        "demographic_parity", 0.5
    )
    assert result["current_alerts"] == 12
    assert result["projected_alerts"] == 0
    assert result["change_abs"] == -12
    assert result["change_pct"] == -100.0
    assert result["records_simulated"] == 24
    assert result["not_simulated_reason"] == ""


# ---------------------------------------------------------------------------
# 9. generate_executive_report: drift stability 100.0 with no drift test run
# ---------------------------------------------------------------------------


def _report_store(with_drift: bool):
    import types

    from vfairness.operations.reporting import MetricsStore

    now = datetime.now()
    df = pd.DataFrame(
        [
            {
                "timestamp": now - timedelta(hours=i),
                "metric": m,
                "value": v,
                "alert": False,
                "alert_determined": True,
                "group_size": 500,
            }
            for i in range(6)
            for m, v in [("demographic_parity", 0.02), ("disparate_impact", 0.92)]
        ]
    )
    store = MetricsStore()
    store.ingest_dataframe(
        df, alert_col="alert", alert_determined_col="alert_determined", group_size_col="group_size"
    )
    if with_drift:
        for i in range(3):
            store.ingest_drift_result(
                types.SimpleNamespace(
                    timestamp=now - timedelta(hours=i),
                    metric="demographic_parity",
                    overall_drift_score=0.05,
                    drift_detected=False,
                    mmd_score=0.01,
                    worst_scale=None,
                )
            )
    return store


def test_the_fixture_really_has_no_drift_rows():
    assert _report_store(with_drift=False).get_summary()["n_drift_records"] == 0
    assert _report_store(with_drift=True).get_summary()["n_drift_records"] == 3


def test_a_report_with_no_drift_test_does_not_print_a_drift_score():
    """BEFORE: '## Health Score: 100/100 (GREEN)' over '| Drift Stability |
    100.0 |', and that row was the ONLY line in the whole report containing
    the word drift. Nothing anywhere said no drift test ran.
    AFTER: '| Drift Stability | not assessed |' plus an explicit Coverage
    section.
    """
    from vfairness.operations.reporting import ReportGenerator
    from vfairness.operations.reporting.reports import OutputFormat

    content = (
        ReportGenerator(_report_store(with_drift=False))
        .generate_executive_report(
            output_format=OutputFormat.MARKDOWN, time_window=timedelta(days=7)
        )
        .content
    )
    assert "| Drift Stability | 100.0 |" not in content
    assert "| Drift Stability | not assessed |" in content
    assert "Drift stability: NOT ASSESSED" in content
    assert "no drift test was run in this window" in content


def test_control_a_report_with_real_drift_rows_still_scores_the_component():
    """OVER-CORRECTION CONTROL. Three genuine, determined drift results with
    drift_detected=False are a real measurement of stability."""
    from vfairness.operations.reporting import ReportGenerator
    from vfairness.operations.reporting.reports import OutputFormat

    content = (
        ReportGenerator(_report_store(with_drift=True))
        .generate_executive_report(
            output_format=OutputFormat.MARKDOWN, time_window=timedelta(days=7)
        )
        .content
    )
    assert "| Drift Stability | 100.0 |" in content
    assert "Drift stability: NOT ASSESSED" not in content


@pytest.mark.parametrize("tier", ["executive", "operational", "technical"])
def test_every_tier_renders_the_withheld_component_without_a_zero(tier):
    """`.get('drift_stability', 0)` in the narrative would print a confident
    0 for an ABSENT key, which is the calmest number that line can show for
    something nobody measured."""
    from vfairness.operations.reporting import ReportGenerator
    from vfairness.operations.reporting.reports import OutputFormat

    generator = ReportGenerator(_report_store(with_drift=False))
    method = getattr(generator, f"generate_{tier}_report")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        content = method(output_format=OutputFormat.MARKDOWN, time_window=timedelta(days=7)).content
    assert "drift stability (0)" not in content
    assert "drift_stability=0.00" not in content
    assert "Drift stability: NOT ASSESSED" in content
