"""G12 grading wave: the LLM result records and the four defects fixed in them.

Batch G12 covers 45 units that had never been graded: the result dataclasses of
``llm``, ``agents`` and ``multi_agent``, and the surfaces that build them. This
file holds the LLM half. ``tests/test_grade_g12_surfaces.py`` holds the rest.

Four defects were reproduced by execution here and are pinned below:

* ``IntersectionalGroup.from_attributes`` MINTED AN INTERSECTIONAL GROUP OUT OF
  AN ABSENT VALUE. ``{"race": ["Black", ""], "gender": ["female"]}`` built a
  second group labelled ``'_female'``; ``"None"``, ``"<NA>"``, ``"nan"`` and a
  whitespace-only level pasted themselves into the label the same way, and the
  object doors (``None``, float nan, ``pd.NA``) raised a bare ``TypeError`` out
  of ``"_".join``. All six were silent.
* ``IntersectionalAnalyzer.analyze`` never said that a cell of REAL OUTPUTS
  keyed by a label the design does not contain had been dropped.
* ``ModelIdentity.is_consistent`` answered ``False`` -- "the endpoint did not
  serve one model" -- from a record carrying no name at all, and
  ``coverage_statement`` raised ``IndexError`` on the same record.

Every refusal below is paired with a CONTROL asserting the healthy case's real
answer, because a guard that refuses everything passes every refusal test.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.llm.cot_faithfulness import (
    CoTFaithfulnessAnalyzer,
    CoTFaithfulnessResult,
    FaithfulnessReport,
)
from vfairness.llm.embedding_bias import EmbeddingBiasDetector, WEATResult
from vfairness.llm.intersectional import (
    IntersectionalAnalyzer,
    IntersectionalGroup,
    IntersectionalResult,
)
from vfairness.llm.model_identity import (
    IdentityComparison,
    ModelIdentity,
    compare_model_identity,
    identity_from_run,
)


def _messages(caught):
    return [str(c.message) for c in caught]


# ---------------------------------------------------------------------------
# vfairness.llm.intersectional.IntersectionalGroup
# ---------------------------------------------------------------------------

#: Every door absence reaches a DESIGN through. The strings are what a CSV or a
#: JSON export writes for the objects beside them, which is why both halves are
#: listed: by the time a design reaches this function the object is usually gone.
ABSENT_LEVELS = [
    "",
    "   ",
    "None",
    "nan",
    "NaN",
    "<NA>",
    "NaT",
    "null",
    "NA",
    None,
    float("nan"),
    np.nan,
    pd.NA,
    pd.NaT,
]


@pytest.mark.parametrize("absent", ABSENT_LEVELS, ids=lambda v: repr(v))
def test_an_absent_level_never_becomes_an_intersectional_group(absent):
    """G12. The guard tested whether an AXIS was empty and never looked at the
    VALUES, so ``"_".join(combo)`` pasted an absence marker into a group label.

    Measured before the fix, all silent:
      {"race": ["Black", ""]}     -> a group labelled '_female'
      {"race": ["Black", "None"]} -> 'None_female'
      {"race": ["Black", "<NA>"]} -> '<NA>_female'
      and None / nan / pd.NA raised TypeError from join().
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = IntersectionalGroup.from_attributes(
            {"race": ["Black", absent], "gender": ["female"]}
        )

    assert [g.label for g in groups] == ["Black_female"], (
        f"{absent!r} produced {[g.label for g in groups]}"
    )
    # Dropped LOUDLY. A design silently smaller than the one the caller wrote is
    # the same defect wearing the opposite face.
    assert any("ABSENCE MARKER" in m for m in _messages(caught)), _messages(caught)
    # Nothing carries the marker forward, under any spelling. Compared by
    # IDENTITY and by value against the two levels that ARE real, never with
    # ``absent in values``: ``pd.NA in (...)`` raises "boolean value of NA is
    # ambiguous", which is the same trap the code under test had to handle.
    for g in groups:
        assert g.attributes == {"race": "Black", "gender": "female"}
        for value in g.attributes.values():
            assert value is not absent


def test_an_axis_of_nothing_but_absence_markers_refuses_the_whole_design():
    """TOTAL loss, which must not be quieter than the partial loss above. After
    dropping, the axis holds nothing, so the design has no intersection and the
    existing empty-axis refusal is what answers."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = IntersectionalGroup.from_attributes(
            {"race": ["", None, "<NA>"], "gender": ["female"]}
        )
    assert groups == []
    msgs = _messages(caught)
    assert any("ABSENCE MARKER" in m for m in msgs), msgs
    assert any("no values" in m and "race" in m for m in msgs), msgs


def test_a_real_design_is_enumerated_unchanged_and_in_silence():
    """CONTROL, and the one that makes the guard above mean something. A level
    somebody NAMED is never touched, including levels whose words look like an
    absence ("unknown", "prefer not to say", "missing"), which are frequently
    the most interesting group in a fairness analysis."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = IntersectionalGroup.from_attributes(
            {"race": ["Black", "White"], "gender": ["male", "female"]}
        )
    assert [g.label for g in groups] == [
        "Black_male",
        "Black_female",
        "White_male",
        "White_female",
    ]
    assert groups[0].attributes == {"race": "Black", "gender": "male"}
    assert not caught, _messages(caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        named = IntersectionalGroup.from_attributes(
            {"disclosure": ["disclosed", "prefer not to say", "unknown", "missing"]}
        )
    assert [g.label for g in named] == [
        "disclosed",
        "prefer not to say",
        "unknown",
        "missing",
    ]
    assert not caught, _messages(caught)


def test_a_non_string_level_that_is_a_real_value_still_enumerates():
    """CONTROL. The object doors are refused for ABSENCE, not for not being a
    string: an integer age band is a level, and it used to raise TypeError out
    of ``join`` just like ``None`` did."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        groups = IntersectionalGroup.from_attributes({"band": [30, 45], "gender": ["female"]})
    assert [g.label for g in groups] == ["30_female", "45_female"]
    assert not caught, _messages(caught)


# ---------------------------------------------------------------------------
# vfairness.llm.intersectional.IntersectionalResult
# ---------------------------------------------------------------------------

_POS = ["This is wonderful and great and excellent"] * 6
_NEG = ["This is terrible awful bad horrible"] * 6


def _design():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IntersectionalGroup.from_attributes(
            {"race": ["Black", "White"], "gender": ["male", "female"]}
        )


def test_outputs_for_a_group_outside_the_design_are_never_dropped_in_silence():
    """G12. The design-to-cells mismatch was disclosed in one direction only: a
    design label with no cell warns, and a run whose labels match NO group warns
    through the same branch because every cell reads empty. A cell carrying REAL
    OUTPUTS under a label the design does not contain was dropped silently.

    Measured before the fix on a complete 2x2 plus one extra key holding six
    texts: total_pairs 6, n_pairs_not_assessed 0, has_intersectional_bias False,
    zero warnings. A typo in one declared label does exactly this, and it drops
    the group's data rather than the design's.
    """
    groups = _design()
    outputs = {g.label: _POS for g in groups}
    outputs["Purple_alien"] = _NEG

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = IntersectionalAnalyzer().analyze(outputs, groups)

    msgs = _messages(caught)
    assert any("Purple_alien" in m and "NEVER COMPARED" in m for m in msgs), msgs
    # The pairs that WERE compared are real comparisons, so no verdict moves.
    assert result.total_pairs == 6
    assert result.n_pairs_not_assessed == 0


def test_a_design_whose_cells_all_match_is_analysed_in_silence():
    """CONTROL for the warning above: it is keyed on a label the design does not
    contain, never on the presence of outputs, so a run whose cells agree with
    its design says nothing about unused labels and still finds its disparity."""
    groups = _design()
    outputs = {g.label: (_POS if "female" in g.label else _NEG) for g in groups}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = IntersectionalAnalyzer().analyze(outputs, groups)
    assert not any("NEVER COMPARED" in m for m in _messages(caught))
    assert result.has_intersectional_bias is True
    assert result.n_significant_pairs > 0
    assert math.isfinite(result.max_disparity)


def test_the_intersectional_verdict_refuses_at_one_cell_at_an_empty_cell_and_at_none():
    """The three degenerate denominators an intersection actually arrives with.
    ``has_intersectional_bias`` must answer None for each, never False, and
    ``max_disparity`` must be nan when no pair was measured."""
    groups = _design()

    # ONE CELL EMPTY: 3 of the 6 pairs cannot be tested.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        one_empty = IntersectionalAnalyzer().analyze(
            {**{g.label: _POS for g in groups}, "Black_female": []}, groups
        )
    assert one_empty.total_pairs == 6
    assert one_empty.n_pairs_not_assessed == 3
    assert one_empty.has_intersectional_bias is None
    assert any("NOT ASSESSED" in m for m in _messages(caught))
    # A ranking needs two ends; neither may be handed back from a part-design.
    assert one_empty.most_advantaged is None and one_empty.most_disadvantaged is None

    # EVERY CELL EMPTY.
    all_empty = IntersectionalAnalyzer().analyze({g.label: [] for g in groups}, groups)
    assert all_empty.n_pairs_not_assessed == all_empty.total_pairs == 6
    assert all_empty.has_intersectional_bias is None
    assert math.isnan(all_empty.max_disparity)

    # A DESIGN WITH NO PAIR AT ALL. total_pairs 0 here means the design held no
    # comparison, not that a complete run found nothing, so the warning is the
    # disclosure and the verdict still refuses.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        single = IntersectionalAnalyzer().analyze({groups[0].label: _POS}, groups[:1])
    assert single.total_pairs == 0
    assert single.has_intersectional_bias is None
    assert math.isnan(single.max_disparity)
    assert any("no pair to compare" in m for m in _messages(caught))


def test_the_result_record_reports_no_bias_only_when_every_pair_was_tested():
    """The dataclass's own property, on hand-built records, because that is the
    only way to separate its rule from the analyzer's. ``False`` is a positive
    claim that every intersection was checked."""
    from vfairness.llm.output_analysis import OutputAnalysisResult

    def pair(p):
        return OutputAnalysisResult(
            group_a="a",
            group_b="b",
            metric="sentiment",
            group_a_value=0.1,
            group_b_value=0.1,
            delta=0.0,
            effect_size=0.0,
            p_value=p,
            is_significant=None if p is None else bool(p < 0.05),
            sample_size=6,
            effect_size_interpretation="negligible",
        )

    def record(pairs, n_sig=0):
        return IntersectionalResult(
            groups=[],
            pairwise_results=pairs,
            most_disadvantaged=None,
            most_advantaged=None,
            max_disparity=0.0,
            n_significant_pairs=n_sig,
            total_pairs=len(pairs),
            n_pairs_not_assessed=sum(1 for p in pairs if p.p_value is None),
        )

    assert record([pair(0.9), pair(0.8)]).has_intersectional_bias is False  # control
    assert record([pair(0.001)], n_sig=1).has_intersectional_bias is True  # control
    assert record([pair(0.9), pair(None)]).has_intersectional_bias is None
    assert record([pair(None), pair(None)]).has_intersectional_bias is None
    assert record([]).has_intersectional_bias is None
    # A finding found is a finding however much of the rest is missing.
    assert record([pair(0.001), pair(None)], n_sig=1).has_intersectional_bias is True


# ---------------------------------------------------------------------------
# vfairness.llm.model_identity
# ---------------------------------------------------------------------------


def test_identity_consistency_is_read_off_the_names_not_off_a_count():
    """G12. ``is_consistent`` was ``len(self.reported_names) == 1`` behind a
    ``n_reporting == 0`` gate, so a record whose COUNT claims names it does not
    CARRY answered False: "the endpoint did not serve one model for this run",
    a positive finding about an endpoint from a record holding no name. The same
    record raised ``IndexError: tuple index out of range`` out of
    ``coverage_statement``, on a report surface.

    Reachable through a record rebuilt from a stored row whose names column was
    not persisted; ``identity_from_run`` cannot build it, which is why the
    controls below are the part that proves nothing real changed.
    """
    broken = ModelIdentity(n_responses=1, n_reporting=1, reported_names=())
    assert broken.is_consistent is None
    statement = broken.coverage_statement  # used to raise
    assert "COULD NOT CHECK" in statement
    assert "contradicts itself" in statement

    more_reporting_than_examined = ModelIdentity(
        n_responses=0, n_reporting=3, reported_names=("m",)
    )
    assert "COULD NOT CHECK" in more_reporting_than_examined.coverage_statement

    # CONTROLS: every state a real run produces, unchanged.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        same = identity_from_run([{"reported_model": "m1"}] * 4, configured_model="m1")
        differing = identity_from_run(
            [{"reported_model": "m1"}, {"reported_model": "m2"}], configured_model="m1"
        )
        silent = identity_from_run([{"reported_model": None}] * 4, configured_model="m1")
        nothing = identity_from_run([], configured_model="m1")
    assert same.is_consistent is True
    assert 'reported "m1" on all 4 responses' in same.coverage_statement
    assert differing.is_consistent is False
    assert "DID NOT SERVE ONE MODEL" in differing.coverage_statement
    assert silent.is_consistent is None
    assert "None of the 4 responses carried a model name" in silent.coverage_statement
    assert nothing.is_consistent is None
    assert "No responses were examined" in nothing.coverage_statement


@pytest.mark.parametrize("absent", [None, "", "   ", "​‌﻿­", float("nan"), pd.NA, 7])
def test_a_name_that_is_not_a_name_is_never_recorded_as_one(absent):
    """The identity record's own six doors, including the one that got through
    the only gate there is: a name made of Unicode FORMAT characters survives
    ``str.strip()`` and renders as nothing."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        identity = identity_from_run([{"reported_model": absent}] * 3)
    assert identity.reported_model is None
    assert identity.reported_names == ()
    assert identity.n_reporting == 0
    assert identity.is_consistent is None


def test_a_comparison_says_same_only_when_the_endpoints_own_echo_matched():
    """The verdict field a re-test trigger branches on. ``needs_retest`` is
    False for a could-not-check ON PURPOSE (an unknown identity is not evidence
    of a change), so ``state`` is what has to carry the qualification."""
    mi = ModelIdentity
    echoed = compare_model_identity(
        mi(configured_model="c", reported_model="r"), mi(configured_model="c", reported_model="r")
    )
    assert (echoed.state, echoed.basis, echoed.needs_retest) == ("same", "endpoint_report", False)

    config_only = compare_model_identity(mi(configured_model="c"), mi(configured_model="c"))
    assert config_only.state != "same"
    assert config_only.basis == "configuration_only"
    assert config_only.needs_retest is False

    changed = compare_model_identity(mi(reported_model="r1"), mi(reported_model="r2"))
    assert (changed.state, changed.needs_retest) == ("changed", True)

    nothing = compare_model_identity(None, None)
    assert nothing.state != "same" and nothing.basis == "none"

    # The record type itself: a hand-built comparison carries no verdict it was
    # not given, and needs_retest is derived from state rather than stored.
    assert IdentityComparison(state="changed").needs_retest is True
    assert IdentityComparison(state="same").needs_retest is False


# ---------------------------------------------------------------------------
# vfairness.llm.cot_faithfulness
# ---------------------------------------------------------------------------


def test_a_faithfulness_verdict_is_withheld_when_there_was_nothing_to_compare():
    """A 0.0 here reads as "completely unfaithful" and a 1.0 as "fully
    faithful"; both are worse than a refusal. Executed on the four degenerate
    pairs plus the intervention that changed nothing."""
    a = CoTFaithfulnessAnalyzer()
    base = dict(demographic_cue="gender")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        no_cot = a.analyze_pair(
            original_output="Salary: $120,000",
            variant_output="Salary: $95,000",
            original_cot="",
            variant_cot="",
            **base,
        )
    assert no_cot.classification == "not_assessed"
    assert no_cot.assessed is False
    assert no_cot.cot_changed is None
    assert math.isnan(no_cot.cot_similarity)
    # What WAS read is still reported: the outputs did differ.
    assert no_cot.output_changed is True
    assert "COULD NOT CHECK" in no_cot.not_assessed_reason
    assert any("NOT ASSESSED" in m for m in _messages(caught))

    blank = a.analyze_pair(
        original_output="",
        variant_output="",
        original_cot="",
        variant_cot="",
        **base,
    )
    assert blank.classification == "not_assessed"
    assert blank.output_changed is None and blank.cot_changed is None

    # CONTROLS. The intervention that changed nothing is 'consistent', a real
    # measurement, and the silent-influence case is the module's whole point.
    consistent = a.analyze_pair(
        original_output="Salary: $120,000",
        variant_output="Salary: $120,000",
        original_cot="Based on experience",
        variant_cot="Based on experience",
        **base,
    )
    assert consistent.classification == "consistent"
    assert consistent.assessed is True

    silent = a.analyze_pair(
        original_output="Salary: $120,000",
        variant_output="Salary: $95,000",
        original_cot="Based on experience and skills",
        variant_cot="Based on experience and skills",
        **base,
    )
    assert silent.classification == "unfaithful_silent"
    assert silent.output_changed is True and silent.cot_mentions_cue is False


def test_a_batch_rate_over_zero_assessable_scenarios_is_nan_not_zero():
    """``n_faithful / n`` would let scenarios with no text at all push the
    faithfulness score DOWN and the silent-influence rate down with it, both
    reading as measurements. The record type carries n_not_assessed so the
    excluded half is visible where a reader looks."""
    a = CoTFaithfulnessAnalyzer()
    good = dict(
        original_output="Salary: $120,000",
        variant_output="Salary: $95,000",
        original_cot="Based on experience and skills",
        variant_cot="Based on experience and skills",
        demographic_cue="gender",
    )
    blank = dict(
        original_output="",
        variant_output="",
        original_cot="",
        variant_cot="",
        demographic_cue="gender",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        none_assessable = a.analyze_batch([dict(blank), dict(blank)])
    assert isinstance(none_assessable, FaithfulnessReport)
    assert none_assessable.n_scenarios == 2
    assert none_assessable.n_not_assessed == 2
    assert math.isnan(none_assessable.faithfulness_score)
    assert math.isnan(none_assessable.silent_influence_rate)
    assert any("are nan (COULD NOT CHECK)" in m for m in _messages(caught))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        empty = a.analyze_batch([])
    assert math.isnan(empty.faithfulness_score)

    # CONTROL: the rate is over the ASSESSED scenarios, and it is a real number.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        half = a.analyze_batch([dict(good), dict(blank)])
    assert (half.n_scenarios, half.n_not_assessed) == (2, 1)
    assert half.silent_influence_rate == 1.0  # 1 silent of 1 assessed, not of 2
    assert half.faithfulness_score == 0.0

    # And a genuinely faithful batch still scores 1.0.
    faithful = dict(good, original_cot="She has less experience", variant_cot="He has more")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        clean = a.analyze_batch([dict(faithful)])
    assert clean.n_not_assessed == 0
    assert clean.faithfulness_score == 1.0


def test_the_result_record_never_reports_assessed_for_an_unassessed_pair():
    """``assessed`` defaults to True on the dataclass, so the question is
    whether the PRODUCER ever leaves it to the default. It has two return
    paths and both set it; this executes both."""
    a = CoTFaithfulnessAnalyzer()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        unassessed = a.analyze_pair(
            original_output="",
            variant_output="",
            original_cot="",
            variant_cot="",
            demographic_cue="gender",
        )
    assert isinstance(unassessed, CoTFaithfulnessResult)
    assert (unassessed.assessed, unassessed.classification) == (False, "not_assessed")
    assert unassessed.not_assessed_reason != ""

    assessed = a.analyze_pair(
        original_output="A",
        variant_output="B",
        original_cot="because of experience",
        variant_cot="because of tenure",
        demographic_cue="gender",
    )
    assert (assessed.assessed, assessed.not_assessed_reason) == (True, "")


# ---------------------------------------------------------------------------
# vfairness.llm.embedding_bias.WEATResult (+ .to_dict)
# ---------------------------------------------------------------------------


def _weat_words(n=8):
    ta = [f"career{i}" for i in range(n)]
    tb = [f"family{i}" for i in range(n)]
    aa = [f"male{i}" for i in range(n)]
    ab = [f"female{i}" for i in range(n)]
    return ta, tb, aa, ab


def _planted_embeddings(words, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for w in words:
        v = rng.normal(size=16)
        v[0] += 4.0 if (w.startswith("career") or w.startswith("male")) else -4.0
        out[w] = v
    return out


def test_a_weat_run_over_dead_vectors_reports_could_not_check_not_zero():
    """A cosine similarity over a zero-norm vector is undefined, not 0.0, and a
    permutation p of 0.0 is arithmetically impossible for this test (the
    observed split is one of the enumerated partitions). Executed on the three
    embedding sources a dead backend actually produces."""
    ta, tb, aa, ab = _weat_words()
    words = ta + tb + aa + ab

    for label, vector in (
        ("all-zero", np.zeros(16)),
        ("all-nan", np.full(16, np.nan)),
        ("all-inf", np.full(16, np.inf)),
    ):
        detector = EmbeddingBiasDetector(embeddings={w: vector.copy() for w in words})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = detector.weat(ta, tb, aa, ab, n_permutations=200)
        assert isinstance(result, WEATResult)
        assert result.severity == "could_not_check", label
        assert math.isnan(result.effect_size), label
        assert result.p_value != 0.0, f"{label}: p=0.0 is unreachable for this test"
        assert "COULD NOT CHECK" in result.interpretation, label
        assert caught, label
        # The honesty survives the serialisation boundary a consumer reads.
        as_dict = result.to_dict()
        assert as_dict["severity"] == "could_not_check", label
        assert "COULD NOT CHECK" in as_dict["interpretation"], label
        # to_dict PRESERVES NaN by this library's convention (llm._base:
        # "NaN is preserved; SerializableMixin.to_json is where it becomes
        # null"), so the in-process consumer can still test it with isnan.
        assert math.isnan(as_dict["effect_size"]), label

    # ONE bad vector among healthy ones is refused too: partial loss must not
    # pass where total loss is refused.
    mixed = _planted_embeddings(words)
    mixed[ta[0]] = np.full(16, np.nan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        partial = EmbeddingBiasDetector(embeddings=mixed).weat(ta, tb, aa, ab, n_permutations=200)
    assert partial.severity == "could_not_check"
    assert math.isnan(partial.effect_size)


def test_a_weat_design_too_small_to_reach_alpha_is_not_a_no_bias_finding():
    """The design FLOOR. With 3 words against 3 there are only C(6,3)=20
    partitions, so the smallest p the test can return is above the 0.05 it is
    graded against; a 'not significant' reading there is an absence of power."""
    ta, tb, aa, ab = _weat_words(3)
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings(ta + tb + aa + ab))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        small = detector.weat(ta, tb, aa, ab, n_permutations=200)
    assert small.severity == "could_not_check"
    assert abs(small.effect_size) > 0.8, "the planted stereotype is still measured"
    assert small.p_value > 0.05
    assert any("NOT DETECTABLE" in n for n in small.notes), small.notes
    assert any("reach" in m for m in _messages(caught))


def test_a_permutation_p_is_never_below_the_floor_its_design_can_reach():
    """A p-value has a FLOOR set by the number of partitions (exact branch) or
    resamples (Monte-Carlo branch). Derived from the design, never quoted, so
    this cannot go stale."""
    from math import comb

    ta, tb, aa, ab = _weat_words(8)
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings(ta + tb + aa + ab))
    exact = detector.weat(ta, tb, aa, ab, n_permutations=200)
    assert exact.p_value >= 1.0 / comb(16, 8)
    assert exact.severity == "critical", "control: the planted stereotype is found"
    assert exact.p_value < 0.05

    # Monte-Carlo branch: 20 words a side puts C(40,20) past the exact ceiling.
    big_ta = [f"career{i}" for i in range(20)]
    big_tb = [f"family{i}" for i in range(20)]
    words = big_ta + big_tb + aa + ab
    mc = EmbeddingBiasDetector(embeddings=_planted_embeddings(words)).weat(
        big_ta, big_tb, aa, ab, n_permutations=100
    )
    assert mc.p_value >= 1.0 / (100 + 1), "0.0001 from 100 draws is unreachable by design"


def test_an_empty_attribute_set_has_no_weat_statistic_at_all():
    """A WEAT over an empty attribute set has no value, so it is refused before
    any vector is fetched rather than answered with a 0.0."""
    ta, tb, aa, ab = _weat_words()
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings(ta + tb + aa + ab))
    for name, args in (
        ("target_a", ([], tb, aa, ab)),
        ("target_b", (ta, [], aa, ab)),
        ("attribute_a", (ta, tb, [], ab)),
        ("attribute_b", (ta, tb, aa, [])),
    ):
        with pytest.raises(ValueError, match=f"{name} must be non-empty"):
            detector.weat(*args)


def test_a_seat_badge_is_not_stamped_on_word_vectors():
    """``method`` is a provenance claim. SEAT is WEAT on SENTENCE embeddings, so
    the label is only honest once the inputs are sentence-resolved."""
    ta, tb, aa, ab = _weat_words()
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings(ta + tb + aa + ab))
    as_words = detector.seat(ta, tb, aa, ab, n_permutations=200)
    assert as_words.method == "WEAT"
    assert any("not flagged sentence-resolved" in n for n in as_words.notes)
    resolved = detector.seat(ta, tb, aa, ab, n_permutations=200, sentence_resolved=True)
    assert resolved.method == "SEAT"
