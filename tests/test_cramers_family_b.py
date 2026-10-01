"""The NAIVE (uncorrected) Cramer's V family in the Pulse engine: three states.

Two sites, both reached from a PUBLIC entry (``run_pulse``):

* ``operations/pulse/agent_probe.py::_cramers_v`` -- the per-comparison tool /
  action effect size of the agent-trace probe, additionally wrapped in
  ``_safe(..., 0.0)`` at its call site, so a raise became a measurement too.
* ``operations/pulse/orchestrator.py::_proxies::_cramers_v`` -- the strong
  proxy crosswalk's (feature, protected attribute) association.

Both answered 0.0 for a table where V is mathematically undefined, and on this
scale 0.0 means "measured, no association": a clean bill of health for a pair
nobody could measure. Reproduced at the public entry on 2026-09-17:

    run_pulse(pd.DataFrame({"group": ["A"] * 30 + ["B"] * 30,
                            "tool": ["search"] * 60}),
              {"source_kind": "agent", "protected_attributes": ["group"]})
    -> agent.perComparison[0] == {..., "cramersV": 0.0, "severity": "info"}

with zero warnings, for a 2x1 contingency table on which k = min(r, c) - 1 is
0, so V = sqrt(chi2 / (n * 0)).

Every expected value below is computed in the test from the fixture counts,
never copied out of the engine, and every refusal test is paired with a
healthy-data CONTROL proving a genuine association is still measured exactly.
"""

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse import run_pulse
from vfairness.operations.pulse.agent_probe import _cramers_v, _severity_from_v, _v_clause

# ── independent reference implementation (the test's own arithmetic) ────────


def _reference_cramers_v(counts) -> float:
    """Uncorrected Cramer's V, written out here so the assertions never
    quote the implementation they are checking."""
    t = np.asarray(counts, dtype=float)
    n = t.sum()
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    chi2 = (((t - expected) ** 2) / expected).sum()
    k = min(t.shape) - 1
    return float(math.sqrt(chi2 / (n * k)))


def _agent_pulse(df):
    return run_pulse(df, {"source_kind": "agent", "protected_attributes": ["group"]})["data"]


def _comparison(agent, ga, gb):
    for c in agent["perComparison"]:
        if {c["groupA"], c["groupB"]} == {ga, gb}:
            return c
    raise AssertionError(f"missing comparison {ga} vs {gb}")


# ── fixtures ────────────────────────────────────────────────────────────────

_SINGLE_TOOL = pd.DataFrame({"group": ["A"] * 30 + ["B"] * 30, "tool": ["search"] * 60})

# A vs C is a 2x3 table of [[84, 24, 12], [36, 72, 12]].
_BASE_DIST = {"search": 84, "escalate": 24, "respond": 12}
_SKEW_DIST = {"search": 36, "escalate": 72, "respond": 12}


def _three_group_df() -> pd.DataFrame:
    rows = []
    for g, dist in (("A", _BASE_DIST), ("B", _BASE_DIST), ("C", _SKEW_DIST)):
        for tool, k in dist.items():
            rows.extend({"group": g, "tool": tool} for _ in range(k))
    return pd.DataFrame(rows)


def _delegation_df() -> pd.DataFrame:
    """Identical tool use; routing depends on the group (senior 48/60 for A
    vs 12/60 for B), so the route x group table is [[48, 12], [12, 48]]."""
    rows = []
    for g, n_senior in (("A", 48), ("B", 12)):
        for i in range(60):
            rows.append(
                {
                    "group": g,
                    "tool": "search",
                    "delegate": "senior_agent" if i < n_senior else "junior_agent",
                }
            )
    return pd.DataFrame(rows)


def _tabular_df() -> pd.DataFrame:
    """One protected attribute, one categorical feature weakly associated
    with it (the crosswalk's own measurement), and one column too sparse for
    any association to be computed at all."""
    rng = np.random.default_rng(11)
    n = 400
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    pred = np.concatenate([rng.binomial(1, 0.6, n // 2), rng.binomial(1, 0.3, n // 2)])
    dept = np.where(rng.random(n) < np.where(group == "A", 0.58, 0.42), "sales", "ops")
    sparse = np.array([None] * n, dtype=object)
    sparse[:3] = "x"
    sparse[n // 2 : n // 2 + 3] = "y"
    return pd.DataFrame({"group": group, "prediction": pred, "dept": dept, "sparse_city": sparse})


# ── site 1a: agent_probe._cramers_v, the helper ─────────────────────────────


@pytest.mark.parametrize(
    "table, shape_words",
    [
        ([[1, 2, 3]], "1x3"),
        ([[30], [30]], "2x1"),
        ([[0, 0], [0, 0]], "2x2"),
    ],
)
def test_helper_refuses_a_table_whose_v_is_undefined(table, shape_words):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = _cramers_v(table)
    assert math.isnan(v), f"expected nan (could not check), got {v!r}"
    assert v != 0.0
    messages = " ".join(str(w.message) for w in caught)
    assert caught, "a refusal carried in silence is invisible to the caller"
    assert shape_words in messages
    assert "undefined" in messages
    assert "never 0.0" in messages


def test_helper_refuses_an_empty_table():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = _cramers_v([])
    assert math.isnan(v)
    assert any("two dimensions" in str(w.message) for w in caught)


def test_helper_still_measures_a_real_table_exactly():
    """CONTROL. The refusal must not cost a measurement: the A vs C table of
    the planted-skew fixture has a chi-square of 43.2 on n = 240, so
    V = sqrt(43.2 / 240) = 0.4242640687..."""
    table = [
        [_BASE_DIST["search"], _BASE_DIST["escalate"], _BASE_DIST["respond"]],
        [_SKEW_DIST["search"], _SKEW_DIST["escalate"], _SKEW_DIST["respond"]],
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = _cramers_v(table)
    assert v == pytest.approx(_reference_cramers_v(table), rel=1e-12)
    assert v == pytest.approx(math.sqrt(43.2 / 240), rel=1e-12)
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


def test_severity_from_v_is_three_state():
    assert _severity_from_v(float("nan")) is None
    assert _severity_from_v(None) is None
    assert _severity_from_v(float("inf")) is None
    # CONTROL: the measured bands are untouched.
    assert _severity_from_v(0.60) == "critical"
    assert _severity_from_v(0.50) == "critical"
    assert _severity_from_v(0.35) == "warn"
    assert _severity_from_v(0.30) == "warn"
    assert _severity_from_v(0.10) == "info"
    assert _severity_from_v(0.0) == "info"


def test_v_clause_never_formats_an_unmeasured_effect_size():
    """_probe is wrapped in _safe, so a TypeError from "{None:.2f}" would
    swallow the whole probe into a Disclaimer and delete every finding in it."""
    assert "undefined" in _v_clause({"cramersV": None, "cramersVMeasured": False})
    assert _v_clause({"cramersV": 0.4242640687, "cramersVMeasured": True}) == "Cramér's V 0.42"


# ── site 1b: agent_probe, at the public entry ───────────────────────────────


def test_public_entry_single_tool_pair_is_could_not_check():
    """THE REPRODUCTION. Before: cramersV 0.0, severity "info", no warning."""
    agent = _agent_pulse(_SINGLE_TOOL)["agent"]
    comp = _comparison(agent, "A", "B")
    assert comp["cramersV"] is None, "an unmeasured effect size is null, never 0.0"
    assert comp["cramersVMeasured"] is False
    assert comp["severity"] == "insufficient_data"
    assert comp["severity"] != "info"
    # The reason is on the payload, so a reader can tell the third state from a
    # measurement without reading the source.
    reasons = " ".join(comp.get("couldNotCheck") or [])
    assert "could not be computed" in reasons
    assert "not a small one" in reasons
    notes = " ".join(comp.get("effectSizeNotes") or [])
    assert "undefined" in notes and "2x1" in notes


def test_public_entry_single_tool_still_raises_what_it_could_raise():
    """A refusal must not delete a finding. Nothing is findable on a frame
    where every episode of both groups called the same tool (both rates are
    1.0 and the omnibus test cannot be significant), and that was already
    true before the refusal: the assurance opinion is unchanged."""
    d = _agent_pulse(_SINGLE_TOOL)
    assert d["assurance"]["overall"] == "Disclaimer"
    assert [f for f in d["bias"] if f["type"] == "agent_tool_bias"] == []
    assert [f for f in d["bias"] if f["type"] == "agent_action_bias"] == []


def test_public_entry_measures_the_planted_skew_exactly():
    """CONTROL. A genuine, findable association is still measured to its exact
    value and still fires its findings at the severity that value implies."""
    d = _agent_pulse(_three_group_df())
    expected = _reference_cramers_v(
        [
            [_BASE_DIST["search"], _BASE_DIST["escalate"], _BASE_DIST["respond"]],
            [_SKEW_DIST["search"], _SKEW_DIST["escalate"], _SKEW_DIST["respond"]],
        ]
    )
    comp = _comparison(d["agent"], "A", "C")
    assert comp["cramersV"] == pytest.approx(expected, rel=1e-12)
    assert comp["cramersVMeasured"] is True
    assert comp["severity"] == "warn"
    assert "couldNotCheck" not in comp

    tool_findings = [f for f in d["bias"] if f["type"] == "agent_tool_bias"]
    assert tool_findings, "the planted skew must still be found"
    assert all(f["severity"] == "warn" for f in tool_findings)
    stat = tool_findings[0]["statisticalTest"]["effectSize"]
    assert stat["cramersV"] == pytest.approx(expected, rel=1e-12)
    assert stat["cramersVMeasured"] is True
    assert any("Cramér's V 0.42" in f["plain"] for f in tool_findings)


# ── site 1c: the delegation section shares _severity_from_v ─────────────────


def test_delegation_measures_its_routing_table_exactly():
    """CONTROL. Route x group is [[48, 12], [12, 48]]: chi-square 43.2 on
    n = 120, so V = sqrt(43.2 / 120) = 0.6 exactly."""
    d = _agent_pulse(_delegation_df())
    dg = d["agent"]["delegation"]
    assert dg["available"] is True
    assert dg["cramersV"] == pytest.approx(_reference_cramers_v([[48, 12], [12, 48]]), rel=1e-12)
    assert dg["cramersV"] == pytest.approx(0.6, rel=1e-12)
    assert dg["cramersVMeasured"] is True
    assert dg["couldNotCheck"] == []
    assert dg["severity"] == "critical"
    findings = [f for f in d["bias"] if f["type"] == "delegation_routing"]
    assert findings and findings[0]["severity"] == "critical"
    assert "Cramér's V 0.60" in findings[0]["plain"]


def test_delegation_unmeasured_v_is_never_graded_info(monkeypatch):
    """multi_agent/delegation.py still answers 0.0 for k = 0, so this pins the
    Pulse side against the day it reports the honest nan: nan >= 0.5 and
    nan >= 0.3 are both False, which used to land on "info", the weakest word
    the scale has, for an effect nobody measured."""
    from vfairness.multi_agent import DelegationRoutingAuditor

    original = DelegationRoutingAuditor.analyze

    def _nan_v(self, routes, demographics, **kwargs):
        return dataclasses.replace(
            original(self, routes, demographics, **kwargs), cramers_v=float("nan")
        )

    monkeypatch.setattr(DelegationRoutingAuditor, "analyze", _nan_v)
    d = _agent_pulse(_delegation_df())
    dg = d["agent"]["delegation"]
    assert dg["cramersV"] is None
    assert dg["cramersVMeasured"] is False
    assert dg["severity"] == "insufficient_data"
    assert dg["severity"] != "info"
    assert dg["couldNotCheck"] and "could not be computed" in dg["couldNotCheck"][0]
    # The routing test itself is still significant, so the finding survives the
    # refusal: only its effect-size word is withheld.
    assert dg["significant"] is True
    findings = [f for f in d["bias"] if f["type"] == "delegation_routing"]
    assert findings, "an unmeasured effect size must not delete a significant finding"
    assert findings[0]["severity"] == "insufficient_data"
    assert "undefined" in findings[0]["plain"]
    assert "nan" not in findings[0]["plain"]


# ── site 2: the proxy crosswalk, at the public entry ────────────────────────


def test_crosswalk_discloses_the_pair_it_could_not_measure():
    """THE REPRODUCTION. Before: the pair scored 0.0, fell under the 0.015
    floor, and vanished. Absence from the crosswalk is the engine's way of
    saying "measured, not a proxy", so a pair that was never measured must
    not be reported the same way."""
    proxies = run_pulse(_tabular_df(), {"protected_attributes": ["group"]})["data"]["proxies"]
    not_assessed = proxies["crosswalkNotAssessed"]
    assert proxies["crosswalkNotAssessedCount"] == len(not_assessed) == 1
    entry = not_assessed[0]
    assert entry["feature"] == "sparse_city"
    assert entry["protectedAttribute"] == "group"
    assert entry["method"] == "cramers_v"
    assert entry["strength"] is None
    assert "6 aligned non-null rows" in entry["reason"]
    assert "never 0.0" in entry["reason"]
    # The summary is the sentence a reader acts on, so it must not read as a
    # clean bill of health for a pair nobody could measure.
    assert "COULD NOT BE MEASURED" in proxies["summary"]
    assert "not a pair without proxy risk" in proxies["summary"]
    # The unmeasured pair never enters the proxy list as a number.
    assert all(p["feature"] != "sparse_city" for p in proxies["proxies"])


def test_crosswalk_still_measures_a_real_association_exactly():
    """CONTROL, in the SAME run as the refusal above: 'dept' is weakly but
    genuinely associated with 'group' and the crosswalk must still report its
    exact Cramer's V, above its own 0.015 floor."""
    df = _tabular_df()
    proxies = run_pulse(df, {"protected_attributes": ["group"]})["data"]["proxies"]
    links = [p for p in proxies["proxies"] if p.get("linkSource") == "crosswalk"]
    assert links, "the crosswalk must still create the links it can measure"
    dept = [p for p in links if p["feature"] == "dept"]
    assert len(dept) == 1
    expected = _reference_cramers_v(pd.crosstab(df["dept"], df["group"]).to_numpy())
    assert dept[0]["method"] == "cramers_v"
    assert dept[0]["strength"] == pytest.approx(expected, rel=1e-12)
    assert dept[0]["strength"] > 0.015
    assert dept[0]["severity"] == "medium"


# ── the sibling default at the same call site ───────────────────────────────


def test_bootstrap_ci_default_is_not_a_measurement(monkeypatch):
    """The _safe default beside the Cramer's V one. (0.0, 0.0) is the
    TIGHTEST POSSIBLE interval around no difference, published for a
    resampling that never ran. The bootstrap cannot raise at this call site
    (both groups clear the n >= 20 floor), so this pins the guard by making
    it raise."""
    from vfairness.operations.pulse import agent_probe

    def _boom(tools_a, tools_b, tool):
        raise RuntimeError("bootstrap unavailable")

    monkeypatch.setattr(agent_probe, "_bootstrap_rate_diff_ci", _boom)
    d = _agent_pulse(_three_group_df())
    comp = _comparison(d["agent"], "A", "C")
    largest = comp["largestRateDiff"]
    assert largest["ciLow"] is None and largest["ciHigh"] is None
    # 'escalate' is the argmax by absolute gap: 24/120 for A vs 72/120 for C.
    assert largest["tool"] == "escalate"
    assert largest["observed"] == pytest.approx(24 / 120 - 72 / 120, rel=1e-12)
    # A refusal on the interval must not delete the finding beside it.
    assert [f for f in d["bias"] if f["type"] == "agent_tool_bias"]


def test_bootstrap_ci_is_measured_on_healthy_data():
    """CONTROL. The real interval is still reported, and it excludes zero for
    the planted skew."""
    d = _agent_pulse(_three_group_df())
    largest = _comparison(d["agent"], "A", "C")["largestRateDiff"]
    assert isinstance(largest["ciLow"], float)
    assert isinstance(largest["ciHigh"], float)
    assert largest["ciLow"] > 0.0 or largest["ciHigh"] < 0.0
