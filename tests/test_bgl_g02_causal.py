"""BGL-G02 (2026-09-30): the causal ops, graded by execution on undefined input.

Every causal estimate rests on an assumption the data cannot verify, so the
failure mode here is not a wrong number: it is a number published without the
assumption it depends on, or a neutral value published where nothing was
measured. Each test below pins one such case that WAS reachable on 2026-09-30,
and each group carries a control asserting the healthy case's real number, so a
guard that refuses everything cannot pass this file.

The defects pinned:

* ``attribute_distribution_change`` scored EVERY node 0.0, silently, for a
  binary outcome whose base rate went 0.330 -> 0.657 (DoWhy's default divergence
  estimator answers 0 for an integer-coded column), and published shares of a
  drift of exactly zero when the two windows were the identical frame.
* ``compute_counterfactual`` could not compute a counterfactual at all (it fitted
  a non-invertible SCM), and minted a NaN ``effect`` from an absent factual value.
* ``identify_paths`` returned one verdict, "we cannot cleanly separate this
  effect from confounding", for five structurally different situations, and
  asserted identifiability while assuming every variable in the graph is
  measured.
* ``run_refutation_suite`` scored robustness 0.75 on three rows and published a
  minimum-norm base estimate for a constant treatment.
* ``CausalFairnessGraph`` accepted NaN / pd.NA / pd.NaT / '' as variable NAMES
  (so an absent label satisfied the "no protected variable" refusal), and a
  re-declaration silently cleared a proxy role, downgrading the published
  severity from high to medium.
* the task-handler envelope wrote the bare token ``NaN``, which is not JSON, and
  reported an empty mediation analysis as a success.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.causal import task_handlers as th
from vfairness.operations.causal.attribute import attribute_distribution_change
from vfairness.operations.causal.counterfactual import compute_counterfactual
from vfairness.operations.causal.graph import CausalFairnessGraph
from vfairness.operations.causal.identify import identify_paths
from vfairness.operations.causal.refute import run_refutation_suite

pytest.importorskip("dowhy")

CONFOUNDING_SENTENCE = "We cannot cleanly separate this effect from confounding"


def _gml(nodes, edges):
    parts = ["graph [", "  directed 1"]
    for i, n in enumerate(nodes):
        parts.append(f'  node [ id {i} label "{n}" ]')
    idx = {n: i for i, n in enumerate(nodes)}
    for a, b in edges:
        parts.append(f"  edge [ source {idx[a]} target {idx[b]} ]")
    parts.append("]")
    return "\n".join(parts)


ABY_GML = _gml(["A", "B", "Y"], [("A", "Y"), ("B", "Y")])
N = 150


def _binary_window(p_a: float, seed: int) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    a = (r.random(N) < p_a).astype(int)
    b = (r.random(N) < 0.5).astype(int)
    y = (r.random(N) < np.clip(0.15 + 0.55 * a + 0.1 * b, 0.0, 1.0)).astype(int)
    return pd.DataFrame({"A": a, "B": b, "Y": y})


def _continuous_window(shift_a: float, seed: int) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    a = r.normal(shift_a, 1.0, N)
    b = r.normal(0.0, 1.0, N)
    return pd.DataFrame({"A": a, "B": b, "Y": 2.0 * a + 0.5 * b + r.normal(0.0, 0.3, N)})


# ---------------------------------------------------------------- attribution


@pytest.mark.slow
def test_a_binary_outcome_shift_is_attributed_to_the_input_that_moved():
    """CONTROL and pin in one: the answer used to be 0.0 for every node."""
    baseline = _binary_window(0.2, 1)
    current = _binary_window(0.8, 2)
    # The premise of the test, asserted rather than assumed: the outcome's base
    # rate really did move, and A is the only input that was changed.
    assert abs(current["Y"].mean() - baseline["Y"].mean()) > 0.15
    assert abs(current["B"].mean() - baseline["B"].mean()) < 0.15

    result = attribute_distribution_change(ABY_GML, baseline, current, "Y")

    assert result.not_assessable == []
    assert result.contributions, "a measured drift must be attributed to something"
    top = result.contributions[0]
    assert top.node == "A"
    assert top.share > 0.5, top.share
    # The defect was a NEUTRAL ZERO, so the pin is that the magnitude is not one.
    assert top.contribution is not None and abs(top.contribution) > 0.0
    assert result.outcome_change is not None and result.outcome_change > 0.0
    assert result.outcome_change_p_value is not None and result.outcome_change_p_value < 0.05
    shares = {c.node: c.share for c in result.contributions}
    assert abs(shares["A"]) > abs(shares["B"])


@pytest.mark.slow
def test_a_continuous_drift_still_gets_its_real_attribution():
    """CONTROL: the assessability gate must not refuse a genuine shift."""
    result = attribute_distribution_change(
        ABY_GML, _continuous_window(0.0, 1), _continuous_window(1.0, 2), "Y"
    )
    assert result.not_assessable == []
    assert result.contributions[0].node == "A"
    assert result.contributions[0].share > 0.5


@pytest.mark.parametrize(
    "label, current_of",
    [
        ("the identical frame", lambda base: base.copy()),
        ("a fresh draw from the unchanged DGP", lambda base: _continuous_window(0.0, 7)),
    ],
)
def test_a_drift_that_was_never_measured_publishes_no_share(label, current_of):
    base = _continuous_window(0.0, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = attribute_distribution_change(ABY_GML, base, current_of(base), "Y")

    assert result.contributions == [], label
    assert result.not_assessable, label
    reason = " ".join(result.not_assessable)
    assert "no change" in reason and "nothing to attribute" in reason
    # Visible to a reader who never looks at the return value, too.
    assert any("did not attribute anything" in str(w.message) for w in caught)
    assert result.outcome_change_p_value is not None
    assert result.outcome_change_p_value >= 0.05
    assert reason in " ".join(result.warnings)


def test_a_window_mostly_unreadable_is_refused_like_a_wholly_unreadable_one():
    """Partial loss must not pass where total loss is refused."""
    base = _continuous_window(0.0, 1)
    holed = _continuous_window(1.0, 2)
    holed.loc[holed.index[: int(0.5 * len(holed))], "Y"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = attribute_distribution_change(ABY_GML, base, holed, "Y")
    assert result.contributions == []
    assert any("ceiling" in r for r in result.not_assessable)
    assert result.outcome_change is None
    assert caught


def test_a_window_with_a_few_holes_is_not_refused_for_that():
    """CONTROL: the ceiling is a ceiling, not a ban on any missing value."""
    base = _continuous_window(0.0, 1)
    holed = _continuous_window(1.0, 2)
    holed.loc[holed.index[: int(0.05 * len(holed))], "Y"] = np.nan
    result = attribute_distribution_change(ABY_GML, base, holed, "Y")
    assert not any("ceiling" in r for r in result.not_assessable)


@pytest.mark.slow
@pytest.mark.parametrize(
    "returned",
    [
        {"A": float("nan"), "B": float("nan"), "Y": float("nan")},
        {"A": 0.0, "B": 0.0, "Y": 0.0},
        {"A": float("nan"), "B": 0.0, "Y": 0.0},
    ],
)
def test_an_attribution_that_read_nothing_is_not_a_finding_that_nobody_contributed(
    monkeypatch, returned
):
    """The measured change is real; the Shapley step returned nothing usable.

    Reached by substituting DoWhy's own step, because a fitted GCM does not
    produce this on demand. Without the guard, `sum(abs(v)) or 1.0` makes the
    denominator 1.0 (or NaN) and every node is published at 0.0 (or NaN), which
    reads as "no input caused the shift" for a shift that WAS measured.
    """
    from dowhy import gcm as real_gcm

    monkeypatch.setattr(real_gcm, "distribution_change", lambda *a, **k: dict(returned))
    result = attribute_distribution_change(
        ABY_GML, _continuous_window(0.0, 1), _continuous_window(1.0, 2), "Y"
    )
    assert result.contributions == []
    assert result.not_assessable
    assert "could-not-check" in " ".join(result.not_assessable)
    assert result.outcome_change_p_value is not None and result.outcome_change_p_value < 0.05


def test_a_missing_outcome_column_is_a_refusal_not_an_empty_ranking():
    base = _continuous_window(0.0, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = attribute_distribution_change(ABY_GML, base, base.copy(), "not_a_column")
    assert result.contributions == []
    assert len(result.not_assessable) == 2  # named per window
    assert all("not a column" in r for r in result.not_assessable)
    assert caught
    assert "not_assessable" in result.to_dict()


# ------------------------------------------------------------- counterfactual

CF_GML = _gml(["T", "Y"], [("T", "Y")])


@pytest.fixture(scope="module")
def cf_data():
    r = np.random.default_rng(0)
    t = r.normal(0.0, 1.0, 200)
    return pd.DataFrame({"T": t, "Y": 3.0 * t + r.normal(0.0, 0.2, 200)})


def test_a_counterfactual_is_actually_computed(cf_data):
    """The capability was DEAD: every input returned all-None with one note.

    Y = 3T with additive noise, so for an individual at T=0, Y=0 (noise 0) the
    counterfactual under T=1 is the slope itself. Derived from the DGP, not
    quoted from a run.
    """
    slope = 3.0
    result = compute_counterfactual(CF_GML, cf_data, "T", "Y", {"T": 0.0, "Y": 0.0}, 1.0, "i1")
    assert result.notes == [], result.notes
    assert result.counterfactual_value is not None
    assert result.counterfactual_value == pytest.approx(slope, abs=0.25)
    assert result.effect == pytest.approx(slope, abs=0.25)

    # The other direction, to pin that the noise is recovered rather than the
    # population mean returned: the same individual's factual outcome moved.
    back = compute_counterfactual(CF_GML, cf_data, "T", "Y", {"T": 1.0, "Y": slope}, 0.0, "i1")
    assert back.counterfactual_value == pytest.approx(0.0, abs=0.25)
    assert back.effect == pytest.approx(-slope, abs=0.25)


@pytest.mark.parametrize(
    "door, value",
    [
        ("float nan", float("nan")),
        ("None", None),
        ("blank string", ""),
        ("pd.NA", pd.NA),
        ("pd.NaT", pd.NaT),
        ("the literal string 'None'", "None"),
    ],
)
def test_an_absent_factual_outcome_mints_no_effect(cf_data, door, value):
    result = compute_counterfactual(CF_GML, cf_data, "T", "Y", {"T": 0.0, "Y": value}, 1.0, "i1")
    assert result.factual_value is None, door
    assert result.effect is None, door
    assert result.counterfactual_value is None, door
    joined = " ".join(result.notes)
    assert "'Y'" in joined
    assert "No counterfactual was computed" in joined


def test_a_missing_factual_outcome_key_mints_no_effect(cf_data):
    result = compute_counterfactual(CF_GML, cf_data, "T", "Y", {"T": 0.0}, 1.0, "i1")
    assert result.factual_value is None
    assert result.effect is None
    assert "carries no value for the outcome 'Y'" in " ".join(result.notes)


@pytest.mark.parametrize("intervention", [None, float("nan")])
def test_an_absent_intervention_value_is_refused(cf_data, intervention):
    result = compute_counterfactual(
        CF_GML, cf_data, "T", "Y", {"T": 0.0, "Y": 0.0}, intervention, "i1"
    )
    assert result.counterfactual_value is None
    assert result.effect is None
    assert "no alternative treatment" in " ".join(result.notes)


def test_a_categorical_outcome_says_why_it_cannot_be_inverted(cf_data):
    r = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "T": cf_data["T"].to_numpy(),
            "Y": np.where(r.random(len(cf_data)) < 0.4, "hired", "rejected"),
        }
    )
    result = compute_counterfactual(CF_GML, data, "T", "Y", {"T": 0.0, "Y": "hired"}, 1.0)
    assert result.counterfactual_value is None
    assert result.effect is None
    assert "not a number" in " ".join(result.notes)


# ---------------------------------------------------------------- identify

LATENT_GML = _gml(["X", "Y", "U"], [("X", "Y"), ("U", "X"), ("U", "Y")])
HEALTHY_GML = _gml(["X", "Y", "C"], [("X", "Y"), ("C", "X"), ("C", "Y")])


def test_a_real_adjustment_set_is_still_reported():
    """CONTROL: the healthy identification must keep its real answer."""
    result = identify_paths(HEALTHY_GML, ["X"], ["Y"], dataset_columns=["X", "Y", "C"])
    (path,) = result.paths
    assert path.is_identifiable is True
    assert path.backdoor_adjustment_set == ["C"]
    assert path.no_causal_path is False
    assert path.assumes_all_graph_nodes_observed is False
    assert "controlling for C" in path.verdict
    assert "assumption nobody checked" not in path.verdict
    assert result.warnings == []


def test_an_unobserved_confounder_is_not_silently_assumed_observed():
    assumed = identify_paths(LATENT_GML, ["X"], ["Y"])
    (path,) = assumed.paths
    # DoWhy still answers with U as the adjustment set, because the placeholder
    # frame is built from the graph. What changed is that it SAYS SO.
    assert path.assumes_all_graph_nodes_observed is True
    assert "assumption nobody checked" in path.verdict
    assert any("assumes EVERY" in w for w in assumed.warnings)

    told = identify_paths(LATENT_GML, ["X"], ["Y"], dataset_columns=["X", "Y"])
    (path2,) = told.paths
    assert path2.is_identifiable is False
    assert path2.assumes_all_graph_nodes_observed is False
    assert "DoWhy found no identification strategy" in " ".join(path2.notes)


@pytest.mark.parametrize(
    "label, gml, treatments, outcomes, expect_no_path",
    [
        ("treatment == outcome", _gml(["X", "Y"], [("X", "Y")]), ["X"], ["X"], False),
        ("disconnected", _gml(["X", "Y", "Z"], [("Z", "Y")]), ["X"], ["Y"], True),
        ("a cycle", _gml(["X", "Y"], [("X", "Y"), ("Y", "X")]), ["X"], ["Y"], False),
        ("no edges at all", _gml(["X", "Y"], []), ["X"], ["Y"], True),
        ("the edge reversed", _gml(["X", "Y"], [("Y", "X")]), ["X"], ["Y"], True),
    ],
)
def test_the_five_collapsed_cases_no_longer_share_the_confounding_verdict(
    label, gml, treatments, outcomes, expect_no_path
):
    result = identify_paths(gml, treatments, outcomes, dataset_columns=["X", "Y", "Z"])
    (path,) = result.paths
    assert path.is_identifiable is None, label
    assert CONFOUNDING_SENTENCE not in path.verdict, label
    assert path.notes, label
    assert path.no_causal_path is expect_no_path, label
    if expect_no_path:
        assert "no cause-and-effect pathway" in path.verdict, label
    else:
        assert "could not decide" in path.verdict, label


def test_the_five_collapsed_cases_are_distinguishable_from_each_other():
    cases = {
        "self": (_gml(["X", "Y"], [("X", "Y")]), ["X"], ["X"]),
        "disconnected": (_gml(["X", "Y", "Z"], [("Z", "Y")]), ["X"], ["Y"]),
        "cycle": (_gml(["X", "Y"], [("X", "Y"), ("Y", "X")]), ["X"], ["Y"]),
    }
    notes = set()
    for gml, t, y in cases.values():
        (path,) = identify_paths(gml, t, y, dataset_columns=["X", "Y", "Z"]).paths
        notes.add(" ".join(path.notes))
    assert len(notes) == len(cases), notes


def test_an_identification_crash_is_a_could_not_check_not_a_confounding_finding(monkeypatch):
    import vfairness.operations.causal.identify as ident

    class Boom:
        def __init__(self, *a, **k):
            pass

        def identify_effect(self, *a, **k):
            raise RuntimeError("solver unavailable")

    monkeypatch.setattr(ident, "_import_dowhy", lambda: Boom)
    (path,) = identify_paths(HEALTHY_GML, ["X"], ["Y"], dataset_columns=["X", "Y", "C"]).paths
    assert path.is_identifiable is None
    assert CONFOUNDING_SENTENCE not in path.verdict
    assert "solver unavailable" in " ".join(path.notes)


# ----------------------------------------------------------------- refutation

REFUTE_GML = _gml(["T", "Y", "C"], [("T", "Y"), ("C", "T"), ("C", "Y")])


def _refute_frame(effect: float, n: int = 300, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    c = r.normal(0, 1, n)
    t = 0.5 * c + r.normal(0, 1, n)
    y = effect * t + 0.8 * c + r.normal(0, 0.5, n)
    return pd.DataFrame({"T": t, "Y": y, "C": c})


@pytest.mark.slow
def test_a_real_effect_still_scores_its_real_robustness():
    """CONTROL: the suite must still grade, and on the DGP's own coefficient."""
    true_effect = 0.30
    result = run_refutation_suite(REFUTE_GML, _refute_frame(true_effect), "T", "Y")
    assert result.base_estimate is not None
    assert result.base_estimate == pytest.approx(true_effect, abs=0.1)
    assert result.robustness_score == 1.0
    assert [o.passed for o in result.outcomes] == [True, True, True, True]


def test_three_rows_earn_no_robustness_score():
    result = run_refutation_suite(REFUTE_GML, _refute_frame(0.30).head(3), "T", "Y")
    assert result.robustness_score is None
    assert result.base_estimate is None
    assert result.outcomes == []
    assert any("below the 5" in w for w in result.warnings)


def test_a_constant_treatment_publishes_no_base_estimate():
    frame = _refute_frame(0.30)
    frame["T"] = 1.0
    result = run_refutation_suite(REFUTE_GML, frame, "T", "Y")
    assert result.base_estimate is None
    assert result.robustness_score is None
    assert any("single value in every row" in w for w in result.warnings)
    assert any("rank-deficient" in w for w in result.warnings)


# --------------------------------------------------------------------- graph


def _proxy_graph() -> CausalFairnessGraph:
    g = CausalFairnessGraph()
    g.add_variable("gender", protected=True)
    g.add_variable("zip", proxy=True)
    g.add_variable("hiring", outcome=True)
    g.add_edge("gender", "zip").add_edge("zip", "hiring")
    return g


def test_a_real_proxy_pathway_is_still_graded_high():
    """CONTROL for both graph pins below."""
    summary = _proxy_graph().summary()
    assert summary["severity"] == "high"
    assert summary["counts"]["proxy"] == 1
    assert summary["cleared_roles"] == []
    assert summary["assessable"] is True


@pytest.mark.parametrize(
    "door, name",
    [
        ("None", None),
        ("float nan", float("nan")),
        ("blank string", ""),
        ("whitespace", "   "),
        ("pd.NA", pd.NA),
        ("pd.NaT", pd.NaT),
    ],
)
def test_an_absent_name_cannot_declare_a_variable(door, name):
    with pytest.raises(ValueError):
        CausalFairnessGraph().add_variable(name, protected=True)
    with pytest.raises(ValueError):
        CausalFairnessGraph().add_edge(name, "hiring")
    with pytest.raises(ValueError):
        CausalFairnessGraph().add_edge("gender", name)


def test_an_absent_name_cannot_make_a_graph_assessable():
    """The reachable harm: the missing protected role is the refusal's trigger."""
    g = CausalFairnessGraph()
    g.add_variable("hiring", outcome=True)
    g.add_edge("feature", "hiring")
    assert g.not_assessable(), "premise: with no protected role this is not assessable"
    with pytest.raises(ValueError):
        g.add_variable(float("nan"), protected=True)
    assert g.not_assessable(), "an absent name must not satisfy the protected-role check"
    assert g.summary()["severity"] == "unknown"


def test_a_stringified_absence_used_as_a_name_is_disclosed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CausalFairnessGraph().add_variable("None", protected=True)
    assert any("str() of an absent value" in str(w.message) for w in caught)


def test_a_cleared_proxy_role_is_named_where_the_severity_is_read():
    g = _proxy_graph()
    assert g.summary()["severity"] == "high"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.add_variable("zip", mediator=True)
    assert any("cleared the role(s) proxy" in str(w.message) for w in caught)
    summary = g.summary()
    # The downgrade is real (the declaration replaced the role), so what this
    # pins is that the downgrade is no longer silent.
    assert summary["severity"] == "medium"
    assert summary["cleared_roles"] == [
        "re-declaring 'zip' cleared the role(s) proxy it already carried"
    ]
    json.dumps(summary)  # the disclosure has to survive the JSON boundary


def test_roles_declared_in_one_call_clear_nothing():
    """CONTROL: the warning must not fire for ordinary authoring."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = _proxy_graph()
        g.add_variable("zip", proxy=True, mediator=True)
    assert [str(w.message) for w in caught if "cleared" in str(w.message)] == []
    assert g.summary()["cleared_roles"] == []
    assert g.summary()["severity"] == "high"


# ------------------------------------------------------------------ handlers

HANDLER_GML = _gml(
    ["race", "education", "hired"],
    [("race", "education"), ("race", "hired"), ("education", "hired")],
)
CLEAN_CSV = "race,education,hired\n" + "\n".join(
    f"{i % 2},{i % 5},{(i * 3) % 7}" for i in range(24)
)
INF_CSV = "race,education,hired\n" + "\n".join(
    (f"{i % 2},{i % 5},inf" if i == 3 else f"{i % 2},{i % 5},{(i * 3) % 7}") for i in range(24)
)


def _mediate_payload(csv: str) -> dict:
    return {
        "serialized": {
            "gml": HANDLER_GML,
            "treatments": ["race"],
            "outcomes": ["hired"],
            "mediators": ["education"],
        },
        "dataset_ref": {"inline_csv": csv},
    }


def test_identify_publishes_no_clean_path_when_the_graph_could_not_be_read():
    out = th.handle_identify(
        {"serialized": {"gml": "not gml", "treatments": ["race"], "outcomes": ["hired"]}}
    )
    assert out["data"]["paths"] == []
    assert any("Could not parse DAG" in w for w in out["data"]["warnings"])


def test_identify_refuses_a_payload_with_no_graph():
    for payload in ({}, {"serialized": {}}, {"serialized": {"gml": ""}}):
        out = th.handle_identify(payload)
        assert out["success"] is False
        assert "gml is required" in out["error"]


def test_identify_names_the_assumption_it_could_not_check():
    """The handler passes no dataset_columns, so every path carries the caveat."""
    out = th.handle_identify(
        {"serialized": {"gml": HANDLER_GML, "treatments": ["race"], "outcomes": ["hired"]}}
    )
    assert out["success"] is True
    assert any("assumes EVERY" in w for w in out["data"]["warnings"])
    assert all(p["assumes_all_graph_nodes_observed"] for p in out["data"]["paths"])


@pytest.mark.parametrize(
    "options, phrase",
    [
        ({}, "options.treatment"),
        ({"treatment": "race"}, "options.outcome"),
        ({"treatment": "race", "outcome": ""}, "options.outcome"),
    ],
)
def test_refute_refuses_a_request_that_names_no_pair(options, phrase):
    out = th.handle_refute({"serialized": {"gml": HANDLER_GML}, "options": options})
    assert out["success"] is False
    assert phrase in out["error"]


def test_refute_and_counterfactual_refuse_when_no_dataset_was_attached():
    base = {"serialized": {"gml": HANDLER_GML}}
    refute = th.handle_refute({**base, "options": {"treatment": "race", "outcome": "hired"}})
    assert refute["success"] is False and "dataset" in refute["error"]
    cf = th.handle_counterfactual(
        {
            **base,
            "options": {"treatment": "race", "outcome": "hired", "factual": {"race": 1}},
        }
    )
    assert cf["success"] is False and "dataset" in cf["error"]


@pytest.mark.parametrize(
    "options, phrase",
    [
        ({}, "options.outcome"),
        ({"outcome": None}, "options.outcome"),
    ],
)
def test_attribute_refuses_a_request_with_no_outcome(options, phrase):
    out = th.handle_attribute({"serialized": {"gml": HANDLER_GML}, "options": options})
    assert out["success"] is False
    assert phrase in out["error"]


@pytest.mark.parametrize(
    "ref",
    [
        {},
        {"inline_json_baseline": [], "inline_json_current": [{"race": 1}]},
        {"inline_json_baseline": [{"race": 1}]},
    ],
)
def test_attribute_refuses_a_request_with_only_one_window(ref):
    out = th.handle_attribute(
        {"serialized": {"gml": HANDLER_GML}, "options": {"outcome": "hired"}, "dataset_ref": ref}
    )
    assert out["success"] is False
    assert "baseline + current" in out["error"]


@pytest.mark.parametrize("missing", ["treatments", "outcomes", "mediators"])
def test_a_mediation_request_naming_nothing_is_a_refusal_not_an_empty_success(missing):
    payload = _mediate_payload(CLEAN_CSV)
    payload["serialized"][missing] = []
    out = th.handle_mediate(payload)
    assert out["success"] is False
    assert missing in out["error"]


def test_a_non_finite_field_is_nulled_and_named_so_the_envelope_stays_json():
    out = th.handle_mediate(_mediate_payload(INF_CSV))
    assert out["success"] is True
    assert out["unmeasured_fields"] == ["decompositions[0].total_effect"]
    assert out["data"]["decompositions"][0]["total_effect"] is None
    # The defect: json.dumps wrote the bare token NaN, which JSON.parse rejects.
    text = json.dumps(out, default=str, allow_nan=False)
    assert json.loads(text)["unmeasured_fields"]


def test_a_clean_mediation_envelope_reports_its_real_numbers():
    """CONTROL: the sanitiser must not null a measurement."""
    out = th.handle_mediate(_mediate_payload(CLEAN_CSV))
    assert out["success"] is True
    assert "unmeasured_fields" not in out
    dec = out["data"]["decompositions"][0]
    assert isinstance(dec["total_effect"], float)
    assert dec["natural_direct_effect"] is not None
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("csv, expect_named", [(INF_CSV, True), (CLEAN_CSV, False)])
def test_the_cli_prints_valid_json_for_both(csv, expect_named):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vfairness.operations.causal.task_handlers",
            "vfairness_causal_mediate",
        ],
        input=json.dumps(_mediate_payload(csv)),
        capture_output=True,
        text=True,
    )
    envelope = json.loads(proc.stdout)  # strict: a bare NaN token would raise here
    assert envelope["success"] is True
    assert ("unmeasured_fields" in envelope) is expect_named


def test_the_cli_refuses_an_unknown_task_type():
    proc = subprocess.run(
        [sys.executable, "-m", "vfairness.operations.causal.task_handlers", "nope"],
        input="{}",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["success"] is False
