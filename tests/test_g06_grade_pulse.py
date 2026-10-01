"""Grade wave G06: ``vfairness.operations.pulse``, executed on undefined input.

THREE DEFECTS THIS WAVE FOUND, and all three are ONE mechanism: this package
held its own copy of a judgement the library had already made, and answered it
the other way. ``_validation.handle_missing_values`` defaults to
``missing_strategy='exclude'``, names its opt-in level ``__missing__`` rather
than a ``str()``-minted one, and its own comment says a phantom group made out
of absent values "could drive the reported disparity".
``selection_rate_disparity_matrix`` was fixed for exactly that on 2026-09-09 and
``validate_inputs``'s intersectional branch on 2026-09-30. Meanwhile:

1. ``orchestrator.run_pulse``. Twenty-odd sections each wrote
   ``.astype("string").fillna("missing")``, so rows carrying NO protected
   attribute became a demographic group named "missing". Measured on 400 rows
   labelled "A" and 400 whose protected value is absent, identically for pd.NA,
   np.nan, None, a blank string and a pd.NaT date of birth:

       verdict.headline 'Not fit to deploy as-is: material disparity for
                         "missing".'
       perVariable      worstGroup 'missing', gap 0.355, pValue 0.002
       and the minted level also reached disparityMatrix, assurance,
       individualFairness, causal and regulatoryExports, with NO warning.

   In the SAME payload, ``dataPreparation.binning`` carried the binner's own
   note: '"group" has only one distinct value (\'A\') ... No between-group
   fairness comparison exists for it.' The binner counts levels with
   ``dropna=True`` and was right.

2. ``traces._lookup``, the trace-attribute reader. Its test was
   ``v is not None and str(v).strip() != ""``, and ``str()`` on an absent value
   MINTS CONTENT: ``str(float('nan'))`` is 'nan', ``str(pd.NA)`` is '<NA>',
   ``str(pd.NaT)`` is 'NaT'.

3. ``agent_probe._probe``, which returns from ``run_pulse`` BEFORE the
   orchestrator's own preparation, so it needed the rule where it stands.
   Measured through ``ingest_and_run_pulse`` on 60 episodes, half labelled "A"
   and half with no acting group recorded:

       "Adverse opinion: material fairness defects make this system unfit to
        deploy as-is. Trajectory gap on 'outcome': A averages 0.80 vs 0.30 for
        nan (outcome rate; permutation test, BH-adjusted p 0.001998, Cohen's d
        1.14)."

   byte for byte the sentence a genuine second group "B" produces on the same
   data.

A FOURTH, in the family-correction machinery: ``llm_probe.apply_bh`` asked
``it["pValue"] is not None`` and nothing else, so a NaN p-value joined the
Benjamini-Hochberg family and came back ``familyWiseSignificant: False``, which
reads as "tested, no disparity", while also raising the rank-1 threshold for
every member that did measure something. This file's own ``_pair_stats``
docstring states the rule it missed. Every producer inside that module already
filters, so I could not reach it from a real probe run; it is fixed with the
canonical ``_triage.is_measured`` anyway, because a correction routine is the
wrong place to leave the narrow question.

EVERY REFUSAL PIN BELOW IS PAIRED WITH A CONTROL asserting the REAL number on
complete data. A guard that refuses everything passes every refusal test and
destroys the unit.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import agent_probe as AP
from vfairness.operations.pulse import causal_skeleton as CS
from vfairness.operations.pulse import traces as TR
from vfairness.operations.pulse.llm_probe import apply_bh, correct_families, llm_probe_pulse
from vfairness.operations.pulse.orchestrator import (
    build_legal_framework_block,
    build_scope_block,
    run_pulse,
)
from vfairness.operations.pulse.pipeline import PulsePipelineInputs, run_pulse_pipeline
from vfairness.operations.pulse.recommend import (
    recommend_fairness_definition,
    recommend_interventions,
)
from vfairness.operations.pulse.scoring import score_model_over_frame

#: Every door absence arrives through, as a value for a protected column.
#: ``str()`` on each of the first five mints a readable label.
ABSENCE_DOORS = {
    "None": None,
    "float nan": float("nan"),
    "np.nan": np.nan,
    "pd.NA": pd.NA,
    "blank string": "   ",
}


def _tabular(n: int = 800) -> pd.DataFrame:
    """A REAL disparity: A selected at ~0.6, B at ~0.3."""
    rng = np.random.default_rng(11)
    half = n // 2
    return pd.DataFrame(
        {
            "group": np.array(["A"] * half + ["B"] * half),
            "prediction": np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.3, half)]),
            "feature": rng.normal(size=n),
        }
    )


def _per_variable(data):
    rows = data.get("perVariable") or []
    return rows[0] if rows else {}


def _run(df, inputs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pulse(df, inputs).get("data") or {}


# ==========================================================================
# 1. run_pulse: an absent protected value is not a protected group.
# ==========================================================================


@pytest.mark.parametrize("door", sorted(ABSENCE_DOORS))
def test_run_pulse_never_builds_a_protected_group_out_of_an_absence(door):
    value = ABSENCE_DOORS[door]
    df = _tabular()
    df["group"] = np.array(["A"] * 400 + [value] * 400, dtype=object)
    data = _run(df, {"protected_attributes": ["group"]})

    pv = _per_variable(data)
    labels = [g["label"] for g in pv.get("groups", [])]
    assert labels == ["A"], f"{door} minted the group {labels!r}"
    # The verdict cannot be an adverse-impact finding about a group that is the
    # absence of a group.
    headline = (data.get("verdict") or {}).get("headline") or ""
    assert "material disparity" not in headline, headline
    assert pv.get("gap") is None and pv.get("pValue") is None

    # AND THE COULD-NOT-CHECK IS WHERE A READER LOOKS. "Excluded" is not a fact
    # a reader can be expected to infer from a smaller n.
    prep = data.get("dataPreparation") or {}
    assert prep.get("rowsExcludedNoProtectedValue") == 400, door
    details = " ".join(str(b.get("detail")) for b in prep.get("binning") or [])
    assert "no recorded value" in details, door
    assert "not a demographic group" in details, door


def test_run_pulse_catches_the_absence_the_binner_itself_minted():
    """A date of birth half full of pd.NaT.

    The PREPARED column cannot be tested for this: ``protected_binning`` bands
    the dates into age bands through ``.fillna("missing")``, so the NaT arrives
    at the orchestrator as the readable string "missing" and ``isna()`` is False.
    The raw column at the same row is the only place the absence is still
    visible. Measured before: groups [under_18 n=400, missing n=400], headline
    'material disparity for "missing"'.
    """
    rng = np.random.default_rng(11)
    df = _tabular()
    df["group"] = pd.array([pd.Timestamp("1990-01-01")] * 400 + [pd.NaT] * 400)
    # The fixture reaches the branch: the binner does mint the level.
    from vfairness.preprocessing import prepare_protected_attributes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_protected_attributes(df, ["group"])
    assert "missing" in set(prepared.frame["group"].astype("string").dropna()), (
        "the binner no longer mints the level, so this test pins nothing"
    )
    assert int(prepared.frame["group"].isna().sum()) == 0, "and isna() cannot see it"

    data = _run(df, {"protected_attributes": ["group"]})
    labels = [g["label"] for g in _per_variable(data).get("groups", [])]
    assert "missing" not in labels, labels
    assert (data.get("dataPreparation") or {}).get("rowsExcludedNoProtectedValue") == 400
    assert rng is not None  # keep the seed import honest


def test_a_literal_level_somebody_chose_is_still_a_real_group():
    """The OTHER over-correction control, and the reason the rule is pd.isna.

    A protected value spelled "None" is indistinguishable from a category
    somebody named, and this library already made that call in the same
    direction: its ``__missing__`` sentinel is spelled that way so it can never
    swallow a real level called "missing".
    """
    for chosen in ("None", "missing", "unknown", "other"):
        df = _tabular()
        df["group"] = np.array(["A"] * 400 + [chosen] * 400, dtype=object)
        data = _run(df, {"protected_attributes": ["group"]})
        labels = sorted(g["label"] for g in _per_variable(data).get("groups", []))
        assert labels == sorted(["A", chosen]), (chosen, labels)
        assert (data.get("dataPreparation") or {}).get("rowsExcludedNoProtectedValue") == 0


def test_control_a_complete_frame_is_untouched_and_keeps_its_real_disparity():
    data = _run(_tabular(), {"protected_attributes": ["group"]})
    pv = _per_variable(data)
    groups = {g["label"]: (g["n"], g["rate"]) for g in pv["groups"]}
    assert set(groups) == {"A", "B"}
    assert groups["A"][0] == 400 and groups["B"][0] == 400, "no row was dropped"
    # DERIVED from the frame, not quoted from a previous run.
    df = _tabular()
    for label in ("A", "B"):
        expected = float(df.loc[df["group"] == label, "prediction"].mean())
        assert groups[label][1] == pytest.approx(expected, abs=1e-9)
    assert pv["worstGroup"] == "B"
    assert pv["gap"] == pytest.approx(groups["A"][1] - groups["B"][1], abs=1e-9)
    assert pv["pValue"] is not None and pv["significant"] is True
    assert (data.get("dataPreparation") or {}).get("rowsExcludedNoProtectedValue") == 0
    assert "material disparity" in ((data.get("verdict") or {}).get("headline") or "")


def test_partial_loss_is_not_refused_the_way_total_loss_is():
    """PARTIAL loss must still MEASURE the real groups, and say what it dropped.

    The over-correction this pairs with is real: refusing any frame with one
    blank protected cell would refuse most real uploads.
    """
    rng = np.random.default_rng(3)
    n = 900
    df = pd.DataFrame(
        {
            "group": np.array(["A"] * 300 + ["B"] * 300 + ["C"] * 300, dtype=object),
            "prediction": np.concatenate(
                [rng.binomial(1, 0.7, 300), rng.binomial(1, 0.4, 300), rng.binomial(1, 0.55, 300)]
            ),
            "feature": rng.normal(size=n),
        }
    )
    holed = df.copy()
    holed.loc[rng.choice(n, 45, replace=False), "group"] = None
    data = _run(holed, {"protected_attributes": ["group"]})
    pv = _per_variable(data)
    labels = sorted(g["label"] for g in pv.get("groups", []))
    assert labels == ["A", "B", "C"], labels
    assert pv.get("gap") is not None and pv.get("pValue") is not None
    assert pv.get("worstGroup") == "B"
    assert (data.get("dataPreparation") or {}).get("rowsExcludedNoProtectedValue") == 45
    # Every surviving n is below its clean value and above zero: the rows went,
    # the groups did not.
    for g in pv["groups"]:
        assert 0 < g["n"] < 300, g


def test_the_exclusion_never_empties_the_frame_and_names_why_it_refused():
    """Two attributes absent on DISJOINT halves. Excluding every incomplete row
    would leave nothing, so the axes leave the assessment instead and the
    refusal states the real reason rather than the generic one about
    identifiers."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "group": np.array(["A"] * 400 + [None] * 400, dtype=object),
            "g2": np.array([None] * 400 + ["X"] * 400, dtype=object),
            "prediction": rng.binomial(1, 0.5, 800),
            "feature": rng.normal(size=800),
        }
    )
    data = _run(df, {"protected_attributes": ["group", "g2"]})
    verdict = data.get("verdict") or {}
    assert verdict.get("headline") == "No assessable protected attribute."
    assert "every row in the frame is missing" in (verdict.get("summary") or "").lower()
    assert "identifier" not in (verdict.get("summary") or "").lower(), (
        "the generic sentence is untrue of an attribute that WAS groupable"
    )
    excluded = {e["column"] for e in (data.get("dataPreparation") or {}).get("excluded") or []}
    assert excluded == {"group", "g2"}
    # It did not publish a phantom finding instead.
    assert "material disparity" not in (verdict.get("headline") or "")


def test_a_text_typed_prediction_column_measures_the_same_as_a_numeric_one():
    """``"1" == 1`` is False and a text column is what an ordinary CSV read
    gives you, so the two readings are compared rather than assumed."""
    numeric = _run(_tabular(), {"protected_attributes": ["group"]})
    textual = _tabular()
    textual["prediction"] = textual["prediction"].astype(str)
    textual = _run(textual, {"protected_attributes": ["group"]})
    for data in (numeric, textual):
        assert _per_variable(data)["gap"] is not None
    assert _per_variable(numeric)["gap"] == pytest.approx(_per_variable(textual)["gap"])
    assert _per_variable(numeric)["worstGroup"] == _per_variable(textual)["worstGroup"] == "B"


def test_a_constant_prediction_discloses_the_constant_column_beside_its_zero_gap():
    """Approving EVERYBODY is trivially equal, so a 0.0 gap here is a true
    measurement AND a degenerate one. The reader has to be able to see which."""
    df = _tabular()
    df["prediction"] = 1
    data = _run(df, {"protected_attributes": ["group"]})
    pv = _per_variable(data)
    assert all(g["rate"] == 1.0 for g in pv["groups"]), "the gap really is 0.0"
    assert pv["gap"] == 0.0
    checks = {c["key"]: c for c in (data.get("dataQuality") or {}).get("checks") or []}
    assert checks["vf_constant_columns"]["status"] == "warn"
    assert "prediction" in checks["vf_constant_columns"]["detail"]


# ==========================================================================
# 2. traces: str() on an absent value mints a group label.
# ==========================================================================


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), np.nan, pd.NA, pd.NaT, "", "   ", "<NA>", "NaT", "nan", "None", "null"],
    ids=repr,
)
def test_trace_attribute_lookup_refuses_every_door_of_absence(value):
    assert TR._lookup({"user.group": value}, TR.GROUP_KEYS) is None


def test_trace_attribute_lookup_still_reads_every_real_value():
    """The over-correction control. The serialised-repr half of the rule is
    pandas' own ``STR_NA_VALUES``, which every CSV read in this library already
    applies, so a level a person chose must survive it."""
    for real in ("A", "missing", "unknown", "other", "female", "0", 0, False, ["a"], 3.5):
        assert TR._lookup({"user.group": real}, TR.GROUP_KEYS) == real or (
            TR._lookup({"user.group": real}, TR.GROUP_KEYS) is real
        ), real
    # An absent value no longer BLOCKS a present sibling key either.
    assert TR._lookup({"group": float("nan"), "metadata.group": "A"}, TR.GROUP_KEYS) == "A"
    # pd.isna on a list raises, so the container guard is load-bearing.
    assert TR._lookup({"user.group": np.array([1, 2])}, TR.GROUP_KEYS) is not None


def _span_export(group_value, n=60):
    """Half "A", half ``group_value``; the second half fails far more often, so
    there IS a real trajectory gap for the probe to find."""
    records = []
    for i in range(n):
        first_half = i < n // 2
        ok = (i % 10) < (8 if first_half else 3)
        records.append(
            {
                "trace_id": f"t{i}",
                "name": "llm.call",
                "attributes": {
                    "user.group": "A" if first_half else group_value,
                    "tool.name": "search",
                    "status": "OK" if ok else "ERROR",
                },
                "status": {"code": "OK" if ok else "ERROR"},
            }
        )
    return records


@pytest.mark.parametrize("door", [float("nan"), pd.NA, pd.NaT, None, "   "], ids=repr)
def test_parse_trace_export_leaves_an_absent_group_absent(door):
    table = TR.parse_trace_export(_span_export(door))
    groups = set(table["group"].dropna().tolist())
    assert groups == {"A"}, groups
    assert int(table["group"].isna().sum()) == 30


def test_parse_trace_export_control_a_real_second_group_survives():
    table = TR.parse_trace_export(_span_export("B"))
    assert sorted(set(table["group"].tolist())) == ["A", "B"]
    assert len(table) == 60


def test_parse_trace_export_refuses_junk_without_inventing_episodes():
    for junk in (None, [], {"a": 1}, [1, "x", None], [{"name": "x", "attributes": {}}]):
        table = TR.parse_trace_export(junk)
        assert len(table) == 0
        notes = str((table.attrs.get("ingestion") or {}).get("notes") or "")
        assert "skipped" in notes, junk


def test_the_malformed_export_disclosure_reaches_the_report_not_only_a_warning():
    spans = _span_export("B", n=4)
    truncated = pd.DataFrame({"span": [json.dumps(s)[:-8] for s in spans]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = TR.ingest_and_run_pulse(
            truncated, {"source_kind": "agent_traces", "protected_attributes": ["group"]}
        )
    assert any(issubclass(w.category, TR.MalformedTraceExportWarning) for w in caught)
    data = result.get("data") or {}
    ingestion = data.get("traceIngestion") or {}
    assert ingestion.get("malformedJsonCells") == 4
    assert "could NOT be read as a trace export" in str(ingestion.get("note"))
    # And as a degradation, so the verdict machinery sees it too.
    assert any(d.get("stage") == "trace_ingestion" for d in data.get("degradations") or [])

    # Control: an INTACT export is flattened and carries no malformed count.
    intact = pd.DataFrame({"span": [json.dumps(s) for s in _span_export("B")]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = TR.ingest_and_run_pulse(
            intact, {"source_kind": "agent_traces", "protected_attributes": ["group"]}
        )
    assert (ok.get("data") or {}).get("traceIngestion") in (None, {})
    assert ok["success"] is True


def test_malformed_trace_export_warning_is_a_user_warning_and_carries_no_value():
    assert issubclass(TR.MalformedTraceExportWarning, UserWarning)
    assert not issubclass(TR.MalformedTraceExportWarning, RuntimeWarning)
    assert str(TR.MalformedTraceExportWarning("a message")) == "a message"


# ==========================================================================
# 3. agent_probe: the same judgement, on the pathway that returns early.
# ==========================================================================


@pytest.mark.parametrize("door", [float("nan"), pd.NA, None, "   "], ids=repr)
def test_agent_probe_excludes_episodes_with_no_acting_group(door):
    table = TR.parse_trace_export(_span_export(door))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = AP.agent_probe_pulse(
            table, {"protected_attributes": ["group"]}, "tool", "group", "hiring", "US"
        )
    agent = (result.get("data") or {}).get("agent") or {}
    adequacy = agent.get("sampleAdequacy") or {}
    assert set(adequacy.get("perGroup") or {}) == {"A"}, adequacy
    assert adequacy.get("episodesExcludedNoGroup") == 30
    assert "no recorded acting group" in str(adequacy.get("note"))
    assert "not a demographic group" in str(adequacy.get("note"))
    # No adverse opinion about the absence.
    headline = ((result.get("data") or {}).get("verdict") or {}).get("headline") or ""
    assert "Adverse opinion" not in headline, headline


def test_agent_probe_control_a_real_group_still_gets_its_adverse_opinion():
    """The over-correction control, and it has to be the SAME data shape: the
    only difference from the test above is that the second group has a name."""
    table = TR.parse_trace_export(_span_export("B"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = AP.agent_probe_pulse(
            table, {"protected_attributes": ["group"]}, "tool", "group", "hiring", "US"
        )
    data = result.get("data") or {}
    adequacy = (data.get("agent") or {}).get("sampleAdequacy") or {}
    assert set(adequacy.get("perGroup") or {}) == {"A", "B"}
    assert adequacy.get("episodesExcludedNoGroup") == 0
    assert (data.get("agent") or {}).get("available") is True
    assert "Adverse opinion" in ((data.get("verdict") or {}).get("headline") or "")


def test_agent_probe_pulse_refuses_a_table_with_nothing_to_compare():
    empty = TR.parse_trace_export([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = AP.agent_probe_pulse(
            empty, {"protected_attributes": ["group"]}, "tool", "group", "", ""
        )
    agent = (result.get("data") or {}).get("agent") or {}
    assert agent.get("available") is False
    assert result["success"] is True


# ==========================================================================
# 4. llm_probe: the family-correction machinery.
# ==========================================================================


def test_apply_bh_keeps_an_unmeasurable_comparison_out_of_the_family():
    items = [
        {"pValue": None},
        {"pValue": float("nan")},
        {"pValue": -1.0},
        {"pValue": 5.0},
        {"pValue": True},
        {"pValue": 0.001},
    ]
    apply_bh(items)
    # Five could-not-checks, reported as such and NOT as tested negatives.
    assert [it["familyWiseSignificant"] for it in items[:5]] == [None] * 5
    # And the one real measurement is still a discovery: m is 1, not 6.
    assert items[-1]["familyWiseSignificant"] is True


def test_apply_bh_control_real_p_values_are_corrected_exactly_as_before():
    items = [{"pValue": 0.001}, {"pValue": 0.4}, {"pValue": 0.9}]
    apply_bh(items)
    assert [it["familyWiseSignificant"] for it in items] == [True, False, False]
    # 0.0 and 1.0 are legitimate p-values on this probe's deterministic branch.
    edges = [{"pValue": 0.0}, {"pValue": 1.0}]
    apply_bh(edges)
    assert [it["familyWiseSignificant"] for it in edges] == [True, False]
    # np.float32 is a measurement, not a could-not-check.
    numpy_scalars = [{"pValue": np.float32(0.001)}, {"pValue": np.float64(0.9)}]
    apply_bh(numpy_scalars)
    assert [it["familyWiseSignificant"] for it in numpy_scalars] == [True, False]
    # An empty family does not raise and invents nothing.
    apply_bh([])


def test_apply_bh_does_not_spend_one_familys_alpha_on_another():
    """The family boundary, as a property a test can call.

    The counterfactual is in the same test: the identical three p-values POOLED
    into one family of three would not all fire, so this is not a tautology.
    """
    cells = [{"pValue": 0.02}]
    refusals = [{"pValue": 0.02}]
    trends = [{"pValue": 0.02}]
    correct_families(cells, refusals, trends)
    assert cells[0]["familyWiseSignificant"] is True
    assert refusals[0]["familyWiseSignificant"] is True
    assert trends[0]["familyWiseSignificant"] is True

    # The measured harm READINESS-5 removed: three separate questions in one
    # family raise each other's bar. At m=3 the rank-1 threshold is 0.05/3.
    pooled = [{"pValue": 0.02}, {"pValue": 0.3}, {"pValue": 0.4}]
    apply_bh(pooled)
    assert pooled[0]["familyWiseSignificant"] is False, (
        "pooling must actually cost something, or the test above proves nothing"
    )


def test_llm_probe_pulse_refuses_without_an_endpoint_instead_of_inventing_one():
    for config in (None, {}, {"endpoint_url": None}, {"endpoint_url": ""}):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = llm_probe_pulse(llm_config=config)
        data = result.get("data") or {}
        assert result["success"] is True
        assert (data.get("generative") or {}).get("available") is False
        assert (data.get("verdict") or {}).get("tone") == "disclaimer"
        assert (data.get("assurance") or {}).get("overall") == "Disclaimer"
        assert "Nothing was assessed" in " ".join((data.get("scope") or {}).get("notCovered") or [])
        # No fabricated numbers anywhere in the refusal.
        assert data.get("metrics") in (None, [], {})


# ==========================================================================
# 5. causal_skeleton: an unmeasurable association is not an absent one.
# ==========================================================================


def _proxy_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sex = rng.choice(["m", "f"], n)
    zip_ = np.where(
        sex == "m",
        rng.choice(["A", "B"], n, p=[0.8, 0.2]),
        rng.choice(["A", "B"], n, p=[0.2, 0.8]),
    )
    return pd.DataFrame({"sex": sex, "zip": zip_, "y": (zip_ == "A").astype(int)})


def test_build_causal_skeleton_finds_the_real_proxy_path():
    skeleton = CS.build_causal_skeleton(_proxy_frame(), ["sex"], "y").to_dict()
    assert [n["id"] for n in skeleton["nodes"]] == ["sex", "y", "zip"]
    kinds = sorted({e["kind"] for e in skeleton["edges"]})
    assert kinds == ["direct", "indirect"]
    assert skeleton["unassessed"] == []
    assert "proxy path(s) found" in skeleton["overview"]
    for edge in skeleton["edges"]:
        assert 0.0 <= edge["strength"] <= 1.0


@pytest.mark.parametrize("shape", ["constant", "all null", "eight rows", "one row"])
def test_build_causal_skeleton_says_it_measured_nothing_rather_than_nothing_found(shape):
    """0.0 in ``_assoc`` means "measured, and these two are UNRELATED", and that
    is the finding that decides whether a proxy edge is drawn. An unmeasurable
    candidate must not be silently absent from the graph."""
    df = _proxy_frame()
    if shape == "constant":
        df["zip"] = "A"
    elif shape == "all null":
        df["zip"] = np.nan
    elif shape == "eight rows":
        df = df.head(8)
    else:
        df = df.head(1)
    skeleton = CS.build_causal_skeleton(df, ["sex"], "y").to_dict()
    assert "zip" in skeleton["unassessed"], shape
    assert not any(e["kind"] == "indirect" for e in skeleton["edges"]), shape
    # The sentence must NOT be the positive conclusion about where a disparity
    # comes from, which is what it used to be.
    assert "No proxy path could be assessed" in skeleton["overview"], shape
    assert "any disparity is a more direct effect" not in skeleton["overview"], shape


def test_build_causal_skeleton_refuses_a_frame_with_no_structure():
    for df, protected, outcome in (
        (pd.DataFrame(), ["sex"], "y"),
        (_proxy_frame(), ["nope"], "y"),
        (_proxy_frame(), ["sex"], "nope"),
    ):
        skeleton = CS.build_causal_skeleton(df, protected, outcome).to_dict()
        assert skeleton["nodes"] == [] and skeleton["edges"] == []
        assert skeleton["overview"] == "Not enough structure to draw a causal overview."


def test_causal_skeleton_to_dict_carries_the_coverage_disclosure_to_the_consumer():
    """A field-by-field rebuild is how a three-state disclosure gets dropped at
    the boundary, and ``to_dict`` is the only shape ``run_pulse`` consumes."""
    empty = CS.CausalSkeleton().to_dict()
    assert empty == {
        "nodes": [],
        "edges": [],
        "paths": [],
        "overview": "",
        "identified": None,
        "unassessed": [],
    }
    df = _proxy_frame()
    df["zip"] = "A"
    payload = CS.build_causal_skeleton(df, ["sex"], "y").to_dict()
    assert payload["unassessed"] == ["zip"]
    # And it survives the orchestrator, which is the surface a reader reads.
    frame = pd.DataFrame(
        {"sex": df["sex"], "zip": df["zip"], "prediction": df["y"], "feature": 1.0}
    )
    data = _run(frame, {"protected_attributes": ["sex"]})
    assert "unassessed" in (data.get("causal") or {})


def test_domain_dag_template_refuses_rather_than_instantiating_an_empty_dag():
    roles = {
        "protected": ["sex"],
        "proxy_candidates": ["zip"],
        "job_relevant": ["years_exp"],
        "model_output": ["hired"],
    }
    ok = CS.domain_dag_template("hiring", roles)
    assert ok["available"] is True
    assert {n["kind"] for n in ok["nodes"]} == {"decision", "protected", "legitimate", "proxy"}
    assert {e["kind"] for e in ok["edges"]} == {"direct", "legitimate", "proxy", "indirect"}
    assert "not learned" in ok["note"]
    # An alias resolves to the same template rather than falling through.
    assert CS.domain_dag_template("recruitment", roles)["domain"] == "hiring"

    for domain, schema, expect in (
        ("astrology", roles, "No domain template"),
        ("", roles, "No domain template"),
        (None, roles, "No domain template"),
        ("hiring", None, "No domain template"),
        ("hiring", "not a dict", "No domain template"),
        ("hiring", {}, "Not enough classified columns"),
        ("hiring", {"proxy_candidates": ["zip"]}, "Not enough classified columns"),
        (
            "hiring",
            {"protected": None, "job_relevant": None},
            "Not enough classified columns",
        ),
    ):
        out = CS.domain_dag_template(domain, schema)
        assert out["available"] is False, (domain, schema)
        assert expect in out["reason"], (domain, schema, out["reason"])
        assert "nodes" not in out and "edges" not in out


# ==========================================================================
# 6. recommend: three states on the impossibility tension.
# ==========================================================================


def test_recommend_fairness_definition_keeps_unmeasured_base_rates_distinguishable():
    base = dict(domain="lending", jurisdiction="EU", has_ground_truth=True, scores_exposed=True)
    differ = recommend_fairness_definition(**base, base_rates_differ=True)["tradeoff"]
    same = recommend_fairness_definition(**base, base_rates_differ=False)["tradeoff"]
    unknown = recommend_fairness_definition(**base, base_rates_differ=None)["tradeoff"]
    assert len({differ, same, unknown}) == 3, "a could-not-check collapsed into a negative"
    assert "shows differing base rates" in differ
    assert "no material base-rate difference was confirmed" in same
    assert "could not be measured on this run" in unknown
    assert "Kleinberg" in differ, "the impossibility is named where it applies"


def test_recommend_fairness_definition_separates_the_objective_from_computability():
    computable = recommend_fairness_definition("lending", "EU", has_ground_truth=True)
    not_computable = recommend_fairness_definition("lending", "EU", has_ground_truth=False)
    # The normative objective does not move with what happens to be computable.
    assert computable["primary"] == not_computable["primary"] == "Equal opportunity"
    assert computable["primaryComputableHere"] is True
    assert not_computable["primaryComputableHere"] is False
    assert "cannot be computed directly here" in not_computable["howToApply"]
    # An unrecognised domain falls back rather than guessing a normative choice.
    generic = recommend_fairness_definition("", "", has_ground_truth=True)
    assert generic["primary"] == "Demographic parity / four-fifths"
    assert generic["headlineMetric"] == "selection_rate"


def test_recommend_interventions_reports_an_unmapped_bias_type_instead_of_dropping_it():
    mapped = recommend_interventions(["representation", "proxy", "calibration"])
    assert [row["biasType"] for row in mapped] == [
        "representation",
        "proxy",
        "calibration",
        "disparity",
    ]
    assert all(row["matched"] == "true" for row in mapped)
    # Always at least a path.
    assert [row["biasType"] for row in recommend_interventions([])] == ["disparity"]
    # A type with no control family is REPORTED, never silently dropped.
    for odd in ("astrology_bias", None, float("nan"), 7):
        out = recommend_interventions([odd])
        unmatched = [row for row in out if row["matched"] == "false"]
        assert len(unmatched) == 1, odd
        assert unmatched[0]["control"] == "Not mapped"
        assert unmatched[0]["stage"] == "unmapped"
    # The cali-BRATIO-n trap: matching is whole-word, so "ratio" must NOT match
    # "calibration".
    ratio = recommend_interventions(["ratio"])
    assert [(r["biasType"], r["matched"]) for r in ratio] == [
        ("disparity", "true"),
        ("ratio", "false"),
    ]


# ==========================================================================
# 7. The remaining units: refusals, formatters and the quarantined pipeline.
# ==========================================================================


def test_build_legal_framework_block_is_one_shape_for_every_pathway():
    eeoc = build_legal_framework_block("hiring", "US")
    assert eeoc["mode"] == "ratio_drives_severity"
    assert "4/5ths" in eeoc["framework"]
    generic = build_legal_framework_block("", "")
    assert generic["mode"] == "ratio_is_heuristic"
    assert "Generic adverse-impact heuristic" in generic["framework"]
    # An unmatched domain says so rather than claiming a framework.
    unmatched = build_legal_framework_block("nonsense", "nowhere")
    assert "no specific framework matched" in unmatched["framework"]
    assert unmatched["mode"] == "ratio_is_heuristic"
    # The shape is fixed, and the same keys are present in every case.
    for block in (eeoc, generic, unmatched, build_legal_framework_block(None, None)):
        assert set(block) == {
            "framework",
            "ratioWarn",
            "ratioCritical",
            "mode",
            "domain",
            "jurisdiction",
        }
        assert 0.0 < block["ratioCritical"] <= block["ratioWarn"] <= 1.0


def test_build_scope_block_always_carries_the_absence_is_not_clearance_sentence():
    block = build_scope_block("tabular", ["a"], ["b"], "a power note")
    assert block["level"] == "triage"
    assert block["covered"] == ["a"] and block["notCovered"] == ["b"]
    assert block["powerNote"] == "a power note"
    assert any("is not clearance" in d for d in block["disclosures"]), (
        "the sentence a reader needs most is the one the engine owns"
    )
    assert any("never legal thresholds" in d for d in block["disclosures"])
    # The lists are COPIES: a caller mutating one block must not change the next.
    block["covered"].append("MUTATED")
    block["disclosures"].append("MUTATED")
    fresh = build_scope_block("tabular", ["a"], ["b"])
    assert fresh["covered"] == ["a"]
    assert "MUTATED" not in fresh["disclosures"]
    assert len(fresh["disclosures"]) == 3
    # An empty scope is still a scope, and still carries the disclosures.
    empty = build_scope_block("tabular", [], [])
    assert empty["covered"] == [] and len(empty["disclosures"]) == 3


def test_score_model_over_frame_is_a_no_op_without_a_model_or_an_endpoint():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "sex": ["m", "f"] * 100,
            "age": rng.integers(20, 70, 200),
            "income": rng.normal(5e4, 1e4, 200),
        }
    )

    def never_called(payload, cols):  # pragma: no cover - must not run
        raise AssertionError("build_predict was called with no model and no endpoint")

    out, features, notes, degradations = score_model_over_frame(
        df.copy(), {"protected_attributes": ["sex"]}, {}, never_called
    )
    assert features is None and notes == [] and degradations == []
    pd.testing.assert_frame_equal(out, df)


def test_score_model_over_frame_refuses_a_model_that_scored_the_wrong_rows():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "sex": ["m", "f"] * 100,
            "age": rng.integers(20, 70, 200),
            "income": rng.normal(5e4, 1e4, 200),
        }
    )

    def short(payload, cols):
        return (lambda X: np.ones(5)), "stub"

    with pytest.raises(ValueError, match="model returned 5 scores for 200 rows"):
        score_model_over_frame(
            df.copy(), {"protected_attributes": ["sex"]}, {"model_base64": "x"}, short
        )

    # CONTROL: a model that scored every row is used, and the protected column
    # is NOT among the features it scored on.
    scores = rng.binomial(1, 0.5, 200)

    def full(payload, cols):
        return (lambda X: scores[: len(X)]), "stub"

    out, features, notes, _ = score_model_over_frame(
        df.copy(), {"protected_attributes": ["sex"]}, {"model_base64": "x"}, full
    )
    assert "sex" not in features
    assert sorted(features) == ["age", "income"]
    assert "prediction" in out.columns
    np.testing.assert_array_equal(out["prediction"].to_numpy(), scores)


def test_score_model_over_frame_does_not_turn_an_unscored_row_into_a_decision():
    """A model answering NaN for every row must not become a decision column of
    zeros; the rows have to reach ``run_pulse`` as unscored."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "sex": ["m", "f"] * 100,
            "age": rng.integers(20, 70, 200),
            "income": rng.normal(5e4, 1e4, 200),
        }
    )

    def all_nan(payload, cols):
        return (lambda X: np.full(len(X), np.nan)), "stub"

    out, _, _, _ = score_model_over_frame(
        df.copy(), {"protected_attributes": ["sex"]}, {"model_base64": "x"}, all_nan
    )
    assert int(out["prediction"].isna().sum()) == 200
    assert not (out["prediction"] == 0).any()
    # And the orchestrator then refuses rather than reporting parity.
    data = _run(out, {"protected_attributes": ["sex"]})
    assert "Could not compute" in ((data.get("verdict") or {}).get("headline") or "")


def test_run_pulse_pipeline_is_quarantined_and_fails_closed():
    """NOT A MEASUREMENT: it refuses every input, including healthy input,
    because it fabricated metrics when labels were absent. The refusal names
    what it fabricated and where to go instead."""
    inputs = PulsePipelineInputs(
        df=_tabular(),
        system_type="classifier",
        domain="hiring",
        jurisdiction="US",
        artifact_label="a",
        protected_attributes=["group"],
    )
    with pytest.raises(RuntimeError) as excinfo:
        run_pulse_pipeline(inputs)
    message = str(excinfo.value)
    assert "quarantined" in message
    assert "fabricates metrics when labels are absent" in message
    assert "orchestrator.run_pulse" in message
    # Healthy input too: there is no door through it at all.
    with pytest.raises(RuntimeError):
        run_pulse_pipeline(
            PulsePipelineInputs(
                df=_tabular(),
                system_type="",
                domain="",
                jurisdiction="",
                artifact_label="",
                protected_attributes=[],
            )
        )


def test_pulse_pipeline_inputs_is_a_carrier_and_mints_nothing():
    """NOT A MEASUREMENT. Every field is required, so it cannot fill a missing
    protected-attribute list or an absent frame with a default that then reads
    as data."""
    fields = PulsePipelineInputs.__dataclass_fields__
    assert list(fields) == [
        "df",
        "system_type",
        "domain",
        "jurisdiction",
        "artifact_label",
        "protected_attributes",
    ]
    for name, field in fields.items():
        assert field.default.__class__.__name__ == "_MISSING_TYPE", name
        assert field.default_factory.__class__.__name__ == "_MISSING_TYPE", name
    with pytest.raises(TypeError):
        PulsePipelineInputs()  # type: ignore[call-arg]
    frame = _tabular()
    carrier = PulsePipelineInputs(frame, "t", "hiring", "US", "a", ["group"])
    assert carrier.df is frame and carrier.protected_attributes == ["group"]


def test_handle_pulse_run_refuses_a_payload_it_cannot_read():
    """The task entry point. A refusal must be a refusal, never an envelope
    carrying a verdict about nothing."""
    from vfairness.operations.pulse import task_handlers as TH

    for payload, expect in (
        ({}, "No artifact provided"),
        ({"artifact": {}}, "No artifact provided"),
        ({"inputs": {"protected_attributes": ["group"]}}, "No artifact provided"),
        ({"artifact": {"inline_csv": "a,b\n"}}, "contained no rows"),
        ({"artifact": {"inline_json": []}}, "No artifact provided"),
        ({"artifact": {"inline_json": [{}]}}, "contained no rows"),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = TH.handle_pulse_run(payload)
        assert out["success"] is False, payload
        assert expect in out["error"], (payload, out["error"])
        assert "data" not in out, "a refusal must not carry a payload to read"
        assert out["task_type"] == "vfairness_pulse_run"


def test_handle_pulse_run_control_a_real_upload_still_reaches_a_real_verdict():
    """The over-correction control for the refusals above, and the end-to-end
    proof that the exclusion this wave added travels through the task shim."""
    from vfairness.operations.pulse import task_handlers as TH

    clean = _tabular().to_csv(index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = TH.handle_pulse_run(
            {
                "artifact": {"inline_csv": clean, "filename": "d.csv"},
                "inputs": {"protected_attributes": ["group"]},
            }
        )
    assert out["success"] is True
    data = out["data"]
    assert "material disparity" in data["verdict"]["headline"]
    assert (data.get("dataPreparation") or {}).get("rowsExcludedNoProtectedValue") == 0

    # And the same upload with the protected column blank on half the rows
    # refuses instead of publishing a disparity for the blanks. A CSV is the
    # realistic door: pandas turns a blank cell into NaN on read.
    holed = _tabular()
    holed.loc[holed.index[400:], "group"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out2 = TH.handle_pulse_run(
            {
                "artifact": {"inline_csv": holed.to_csv(index=False), "filename": "d.csv"},
                "inputs": {"protected_attributes": ["group"]},
            }
        )
    assert out2["success"] is True
    assert "material disparity" not in out2["data"]["verdict"]["headline"]
    assert out2["data"]["dataPreparation"]["rowsExcludedNoProtectedValue"] == 400


def test_task_handlers_main_fails_closed_on_a_payload_it_cannot_parse(monkeypatch, capsys):
    """``main`` exits 0 for a successful ENVELOPE even when that envelope wraps
    ``success: false`` (its docstring says so), so the two cases that must exit
    non-zero are the ones where no envelope could be produced at all."""
    import io as _io

    from vfairness.operations.pulse import task_handlers as TH

    monkeypatch.setattr("sys.argv", ["pulse", "vfairness_pulse_run"])
    monkeypatch.setattr("sys.stdin", _io.StringIO("{not json"))
    assert TH.main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["success"] is False
    assert "invalid payload JSON" in printed["error"]

    monkeypatch.setattr("sys.argv", ["pulse", "some_other_task"])
    monkeypatch.setattr("sys.stdin", _io.StringIO("{}"))
    assert TH.main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["success"] is False
    assert "unknown task type" in printed["error"]

    # CONTROL: a real payload writes a real envelope and exits 0.
    monkeypatch.setattr("sys.argv", ["pulse", "vfairness_pulse_run"])
    monkeypatch.setattr(
        "sys.stdin",
        _io.StringIO(
            json.dumps(
                {
                    "artifact": {"inline_csv": _tabular().to_csv(index=False), "filename": "d.csv"},
                    "inputs": {"protected_attributes": ["group"]},
                }
            )
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert TH.main() == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["success"] is True
    assert "material disparity" in envelope["data"]["verdict"]["headline"]


def test_the_helper_this_wave_added_is_the_only_absence_rule_in_the_orchestrator():
    """PUT THE GUARD ABOVE THE DISPATCH, and prove there is only one of it.

    The sections below the guard still write ``.fillna("missing")`` for their own
    reasons (a crosstab over a non-protected column may legitimately treat
    missingness as a level). What must not exist again is a SECOND answer to
    "is this row's protected attribute recorded", so this asserts the single
    helper is what the orchestrator asks, and that asking it is what produced
    the count the payload discloses.
    """
    from vfairness.operations.pulse import orchestrator as ORCH

    assert callable(ORCH._rows_missing_a_protected_value)
    frame = pd.DataFrame(
        {"g": ["A", None, "  ", "missing", float("nan"), pd.NA, "B"]},
        index=[5, 6, 7, 8, 9, 10, 11],
    )
    mask = ORCH._rows_missing_a_protected_value(frame, frame, ["g"])
    assert list(mask) == [False, True, True, False, True, True, False]
    assert list(mask.index) == [5, 6, 7, 8, 9, 10, 11], "the index must survive"
    # Nothing to check is not the same as everything absent.
    assert not ORCH._rows_missing_a_protected_value(frame, frame, []).any()
    assert not ORCH._rows_missing_a_protected_value(frame, frame, ["absent_col"]).any()
    # The raw half: a column the preparation has already overwritten.
    raw = pd.DataFrame({"g": [pd.NaT, pd.Timestamp("2000-01-01")]}, index=[0, 1])
    prepared = pd.DataFrame({"g": ["missing", "18-24"]}, index=[0, 1])
    assert list(ORCH._rows_missing_a_protected_value(raw, prepared, ["g"])) == [True, False]
    assert math.isclose(1.0, 1.0)


# ==========================================================================
# 8. vision_probe: the all-clear sentence was said when skew had REFUSED.
# ==========================================================================


def _vision(df, inputs=None):
    from vfairness.operations.pulse.vision_probe import vision_probe_pulse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vision_probe_pulse(df, inputs or {}, "detected_race", "hiring", "US")
    return (result.get("data") or {}).get("vision") or {}


def test_vision_probe_never_says_within_tolerance_when_skew_refused():
    """PARTIAL loss passing where TOTAL loss is refused, inside one payload.

    ``findings`` is appended to only inside ``if skew.available``, so a REFUSED
    skew panel left the list empty and the summary read the all-clear. Measured
    before: 50 labelled + 50 unlabelled images gave skew ``available: False``
    with the reason "... so representation skew is undefined. This is NOT a
    pass.", ndkl ``available: False``, and summary "Within representation
    tolerance on the perceived labels". The panel said it was not a pass and the
    sentence beside it said it was.
    """
    cases = {
        "half unlabelled": ["white"] * 50 + [np.nan] * 50,
        "one image": ["white"],
        "one group only": ["white"] * 100,
    }
    for label, column in cases.items():
        vision = _vision(pd.DataFrame({"detected_race": column}))
        assert vision["skew"]["available"] is False, label
        summary = vision["summary"]
        assert "Within representation tolerance" not in summary, (label, summary)
        assert summary.startswith("COULD NOT CHECK"), (label, summary)
        assert "NOT a finding of balanced representation" in summary, label
        # The panel's own reason travels with the refusal.
        assert "undefined" in summary.lower() or "refused" in summary.lower(), label


def test_vision_probe_control_a_measured_clean_set_keeps_its_all_clear():
    """The over-correction control. Refusing every clean image set would be
    worse than the defect, so the exact wording has to survive where skew
    actually ran and found nothing."""
    balanced = ["white"] * 34 + ["black"] * 33 + ["asian"] * 33
    vision = _vision(pd.DataFrame({"detected_race": balanced}))
    assert vision["skew"]["available"] is True
    assert vision["summary"].startswith("Within representation tolerance")
    assert vision["labelCoverage"] == {
        "images": 100,
        "labelled": 100,
        "unlabelled": 0,
        "complete": True,
    }
    # And a genuinely skewed set still produces its finding count.
    skewed = ["white"] * 60 + ["black"] * 20 + ["asian"] * 20
    vision = _vision(pd.DataFrame({"detected_race": skewed}))
    assert vision["skew"]["available"] is True
    assert vision["skew"]["maxSkew"] > 0.5
    assert "image-representation finding(s)" in vision["summary"]


@pytest.mark.parametrize(
    "value",
    [np.nan, None, pd.NA, pd.NaT, "nan", "<NA>", "None", "null", "n/a", "", "   "],
    ids=repr,
)
def test_vision_probe_refuses_an_image_set_carrying_no_demographic_label(value):
    """``astype(str)`` on a missing demographic renders the literal string 'nan',
    which every metric downstream reads as one ordinary group, and MaxSkew 0.0 /
    NDKL 0.0 are the BEST attainable score on both scales."""
    vision = _vision(pd.DataFrame({"detected_race": [value] * 100}))
    assert vision["skew"]["available"] is False
    assert "COULD NOT CHECK, not balanced representation" in vision["skew"]["reason"]
    assert vision["ndkl"]["available"] is False
    assert vision["summary"].startswith("NOT ASSESSED")
    assert vision["labelCoverage"] == {
        "images": 100,
        "labelled": 0,
        "unlabelled": 100,
        "complete": False,
    }
    assert vision.get("amplification", {}).get("available") is False


def test_vision_probe_discloses_partial_label_coverage_as_its_own_number():
    vision = _vision(pd.DataFrame({"detected_race": ["white"] * 40 + ["black"] * 40 + [None] * 20}))
    assert vision["labelCoverage"] == {
        "images": 100,
        "labelled": 80,
        "unlabelled": 20,
        "complete": False,
    }
    assert "20 of 100 image(s) carry no demographic label" in vision["summary"]
    assert "UNASSESSED, not evenly distributed" in vision["summary"]
    # The measurement over the 80 that ARE labelled is still real.
    assert vision["skew"]["available"] is True
    assert vision["skew"]["observed"] == {"white": 0.5, "black": 0.5}
