"""BGL grade wave, batch G09: vfairness.mcp.server.

The eleven units here are the FastMCP layer: seven tool wrappers, two
resources and two prompt templates. The fairness arithmetic lives in
vfairness.mcp.tools and is graded there, so what these pins establish is the
two things that are true of this layer ALONE and of nothing underneath it:

1. THE MOUNT. A wrapper that computes correctly and is not registered on the
   FastMCP instance is unreachable by every MCP client, and no test of the
   function would notice. So each unit is exercised BOTH as a Python callable
   and through the server's own list_tools / read_resource / get_prompt.
2. THE FORWARDING. Each wrapper's job is to pass its arguments on, and a
   dropped keyword is invisible unless the test makes the two values produce
   DIFFERENT answers. `min_group_size` is the one with a recorded incident (the
   underlying tool hardwired 2 until 2026-09-10, so a group of TWO people
   returned a demographic parity difference of 0.010 with zero warnings), and
   it is checked by measuring both settings on the same 8-row frame.

Every wrapper is also run on the degenerate inputs where the thing it measures
does not exist: no data source, a remote data_path, a missing column, zero
rows, one group, an all-missing protected attribute, and groups below the size
floor.
"""

import asyncio
import json
import warnings

import numpy as np
import pytest

mcp_server = pytest.importorskip(
    "vfairness.mcp.server",
    reason="the MCP server needs the optional 'mcp' extra (mcp>=1.2.0,<2)",
)

server = mcp_server

TOOLS = [
    ("triage_dataset", dict(protected_attributes=["grp"], target_column="target")),
    (
        "measure_fairness",
        dict(protected_attributes=["grp"], prediction_column="pred", target_column="target"),
    ),
    ("detect_proxies", dict(protected_attributes=["grp"])),
    (
        "analyze_intersectional",
        dict(protected_attributes=["grp", "g2"], prediction_column="pred", target_column="target"),
    ),
    ("suggest_mitigation", dict(protected_attributes=["grp"], target_column="target")),
    ("explain_decision", dict(prediction_column="pred")),
    ("audit_agent", dict(group_column="grp", action_column="pred")),
]
RESOURCES = ["glossary", "legal_framing"]
PROMPTS = ["fairness_triage", "explain_in_plain_language"]


def _healthy_csv():
    return "\n".join(
        ["grp,g2,pred,target,score,feat"]
        + [
            f"a,{'m' if i % 2 else 'f'},{i % 2},{(i + 1) % 2},{0.1 + 0.008 * i:.3f},{i % 7}"
            for i in range(60)
        ]
        + [
            f"b,{'m' if i % 2 else 'f'},{(i + 1) % 2},{i % 2},{0.2 + 0.008 * i:.3f},{(i + 3) % 7}"
            for i in range(60)
        ]
    )


def _small_csv():
    """Two groups of four rows: measurable at min_group_size=2, not at 30."""
    return "grp,g2,pred,target,score\n" + "\n".join(
        f"{'a' if i < 4 else 'b'},{'m' if i % 2 else 'f'},{i % 2},{(i + 1) % 2},{0.1 + 0.1 * i:.2f}"
        for i in range(8)
    )


def _quiet(fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(**kw)


# ---------------------------------------------------------------------------
# 1. THE MOUNT
# ---------------------------------------------------------------------------


def test_every_tool_is_registered_on_the_server_under_its_own_name():
    registered = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert set(registered) == {name for name, _ in TOOLS}
    assert server.mcp.name == "vfairness"
    for name, _ in TOOLS:
        tool = registered[name]
        # A registered tool with no description is unusable by a model that has
        # to choose between seven of them.
        assert tool.description and len(tool.description) > 40, name
        # Both data sources are offered, and neither is required, because
        # exactly one of the two must be given.
        props = tool.inputSchema.get("properties", {})
        required = set(tool.inputSchema.get("required", []))
        assert {"data_path", "csv"} <= set(props), name
        assert not ({"data_path", "csv"} & required), name


@pytest.mark.parametrize(
    "name,keyword",
    [
        ("measure_fairness", "min_group_size"),
        ("analyze_intersectional", "min_group_size"),
        ("detect_proxies", "correlation_threshold"),
        ("suggest_mitigation", "score_column"),
        ("explain_decision", "feature_columns"),
    ],
)
def test_the_optional_knobs_reach_the_published_schema(name, keyword):
    """A knob a client cannot see is a knob no client will set."""
    registered = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert keyword in registered[name].inputSchema["properties"]


def test_both_resources_are_registered_and_readable_through_the_server():
    listed = {str(r.uri): r for r in asyncio.run(server.mcp.list_resources())}
    assert set(listed) == {"vfairness://glossary", "vfairness://legal-framing"}
    for uri in listed:
        body = list(asyncio.run(server.mcp.read_resource(uri)))[0].content
        assert isinstance(body, str) and body.lstrip().startswith("#")
        assert len(body) > 200, uri


def test_both_prompts_are_registered_with_their_arguments():
    listed = {p.name: p for p in asyncio.run(server.mcp.list_prompts())}
    assert set(listed) == set(PROMPTS)
    triage_args = {a.name: a.required for a in listed["fairness_triage"].arguments}
    assert triage_args == {
        "data_path": True,
        "protected_attributes": True,
        "target_column": True,
    }
    explain_args = {a.name: a.required for a in listed["explain_in_plain_language"].arguments}
    assert explain_args == {"audience": False}


def test_the_prompts_render_through_the_server_with_the_arguments_in_them():
    rendered = asyncio.run(
        server.mcp.get_prompt(
            "fairness_triage",
            {
                "data_path": "/tmp/loans.csv",
                "protected_attributes": "race,gender",
                "target_column": "approved",
            },
        )
    )
    text = rendered.messages[0].content.text
    for token in ("/tmp/loans.csv", "race,gender", "approved"):
        assert token in text, token
    # It names the tools it tells the model to run, and they are real tools.
    registered = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {n for n in registered if n in text} >= {
        "triage_dataset",
        "measure_fairness",
        "detect_proxies",
        "analyze_intersectional",
    }

    default = asyncio.run(server.mcp.get_prompt("explain_in_plain_language", {}))
    custom = asyncio.run(
        server.mcp.get_prompt("explain_in_plain_language", {"audience": "a banking regulator"})
    )
    assert "a non-technical stakeholder" in default.messages[0].content.text
    assert "a banking regulator" in custom.messages[0].content.text
    # The audience is really interpolated, not decoration on a fixed string.
    assert default.messages[0].content.text != custom.messages[0].content.text


# ---------------------------------------------------------------------------
# 2. THE FORWARDING, measured as a difference in the answer
# ---------------------------------------------------------------------------


def test_measure_fairness_forwards_min_group_size_and_it_changes_the_answer():
    """The recorded incident: the underlying tool hardwired 2, so a group of
    two returned a demographic parity difference of 0.010 with zero warnings
    where the library default returns NaN with six warnings."""
    csv = _small_csv()

    def run(min_group_size):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = server.measure_fairness(
                protected_attributes=["grp"],
                prediction_column="pred",
                target_column="target",
                csv=csv,
                min_group_size=min_group_size,
            )
        per = out["per_attribute"]["grp"]
        return (
            per["metrics"]["demographic_parity_difference"],
            per["assessment"]["assessable"],
            len(caught),
        )

    permissive = run(2)
    strict = run(30)
    # Below the floor the metric is NOT a number and the run is NOT assessable.
    assert strict[0] is None and strict[1] is False and strict[2] > 0
    # Above it, a real measurement and silence.
    assert isinstance(permissive[0], float) and permissive[1] is True
    # The default is the library's own 30, not the tool's old 2.
    import inspect

    assert inspect.signature(server.measure_fairness).parameters["min_group_size"].default == 30
    assert run(30) == strict


def test_analyze_intersectional_forwards_min_group_size_too():
    csv = _small_csv()
    permissive = _quiet(
        server.analyze_intersectional,
        protected_attributes=["grp", "g2"],
        prediction_column="pred",
        target_column="target",
        csv=csv,
        min_group_size=2,
    )
    strict = _quiet(
        server.analyze_intersectional,
        protected_attributes=["grp", "g2"],
        prediction_column="pred",
        target_column="target",
        csv=csv,
        min_group_size=30,
    )
    assert permissive["coverage"]["min_group_size"] == 2
    assert permissive["coverage"]["max_disparity_measured"] is True
    assert strict["coverage"]["max_disparity_measured"] is False
    assert "NOT ASSESSED" in strict["summary"]
    assert "not a finding" in strict["summary"]
    import inspect

    assert (
        inspect.signature(server.analyze_intersectional).parameters["min_group_size"].default == 30
    )


def test_detect_proxies_forwards_correlation_threshold_and_it_changes_the_answer():
    """A frame with a real two-hop chain: grp -> zipc -> income, where income
    is NOT directly correlated with grp (it is the noise term of zipc), so the
    chain search reaches it."""
    rng = np.random.default_rng(11)
    n = 200
    grp = np.array([1] * 100 + [0] * 100)
    noise = rng.normal(0, 1, n)
    zipc = grp + noise
    income = noise
    pred = (zipc > np.median(zipc)).astype(int)
    csv = "\n".join(
        ["grp,zipc,income,pred"]
        + [f"{grp[i]},{zipc[i]:.4f},{income[i]:.4f},{pred[i]}" for i in range(n)]
    )
    low = _quiet(
        server.detect_proxies, protected_attributes=["grp"], csv=csv, correlation_threshold=0.3
    )
    high = _quiet(
        server.detect_proxies, protected_attributes=["grp"], csv=csv, correlation_threshold=0.95
    )
    assert low["proxy_chains"]["grp"], "the fixture no longer carries a chain to find"
    assert low["proxy_chains"]["grp"][0]["chain"] == ["income", "zipc", "grp"]
    assert high["proxy_chains"]["grp"] == []
    assert low["summary"] != high["summary"]


def test_suggest_mitigation_forwards_score_column_and_it_changes_the_answer():
    csv = _healthy_csv()
    without = _quiet(
        server.suggest_mitigation, protected_attributes=["grp"], target_column="target", csv=csv
    )
    with_score = _quiet(
        server.suggest_mitigation,
        protected_attributes=["grp"],
        target_column="target",
        csv=csv,
        score_column="score",
    )
    # The post-processing trade-off only exists when a score column is given.
    assert "postprocessing_pareto" not in without
    assert "postprocessing_pareto" in with_score


def test_explain_decision_forwards_feature_columns_and_it_changes_the_answer():
    csv = _healthy_csv()
    everything = _quiet(server.explain_decision, prediction_column="pred", csv=csv)
    subset = _quiet(
        server.explain_decision, prediction_column="pred", csv=csv, feature_columns=["score"]
    )
    all_named = {d["feature"] for d in everything["drivers"]} | {
        f["feature"] for f in everything.get("features_not_measured", [])
    }
    subset_named = {d["feature"] for d in subset["drivers"]} | {
        f["feature"] for f in subset.get("features_not_measured", [])
    }
    assert subset_named == {"score"}
    assert all_named > subset_named


# ---------------------------------------------------------------------------
# 3. THE DEGENERATE INPUTS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kwargs", TOOLS, ids=[n for n, _ in TOOLS])
def test_every_tool_refuses_a_missing_data_source(name, kwargs):
    with pytest.raises(ValueError, match="Provide either"):
        getattr(server, name)(**kwargs)


@pytest.fixture
def no_sockets(monkeypatch):
    """Make ANY socket creation fail loudly, so "refused" means refused BEFORE
    the network and a broken guard fails instantly instead of hanging CI on a
    connect timeout to 169.254.169.254."""
    import socket as _socket

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a socket was opened: the data_path guard did not refuse before egress"
        )

    monkeypatch.setattr(_socket, "socket", refuse)
    monkeypatch.setattr(_socket, "create_connection", refuse)
    monkeypatch.setattr(_socket, "getaddrinfo", refuse)
    return refuse


@pytest.mark.parametrize("name,kwargs", TOOLS, ids=[n for n, _ in TOOLS])
@pytest.mark.parametrize(
    "hostile",
    [
        "http://169.254.169.254/latest/meta-data/x.csv",
        "https://evil.example/x.csv",
        "s3://bucket/x.csv",
        "gs://bucket/x.csv",
        "ftp://host/x.csv",
        "smb://host/share/x.csv",
    ],
)
def test_every_tool_refuses_a_remote_data_path(name, kwargs, hostile, no_sockets):
    """pandas.read_csv fetches these schemes, so an agent-supplied data_path
    would otherwise be network egress driven by an LLM, and a route to a cloud
    metadata endpoint. The server is local-first.

    Asserted with every socket entry point disabled, so the pass means the URL
    was rejected before anything was dialled and not that the dial happened to
    fail.
    """
    with pytest.raises(ValueError, match="must be a local file path"):
        getattr(server, name)(data_path=hostile, **kwargs)


@pytest.mark.parametrize("name,kwargs", TOOLS, ids=[n for n, _ in TOOLS])
def test_every_tool_names_the_columns_it_could_not_find(name, kwargs):
    with pytest.raises(ValueError, match="Column\\(s\\) not found"):
        getattr(server, name)(csv="w,x\n1,2\n", **kwargs)


# Per tool, the EXACT field or phrase that says nothing was measured. Derived
# by executing each one on an empty frame, not by matching a loose substring: a
# check for "no " or "could not" anywhere in a JSON blob cannot disagree with
# anything, which is the shape that makes a green suite worthless.
_EMPTY_DISCLOSURE = {
    "triage_dataset": lambda out: (
        out["coverage"]["assessment_coverage"] == "none"
        and out["coverage"]["execution_coverage"] == "ran_but_assessed_nothing"
        and out["coverage"]["attributes_not_assessed"] == ["grp"]
        and "NOT A CLEAN BILL OF HEALTH" in out["summary"]
    ),
    "measure_fairness": lambda out: (
        out["per_attribute"]["grp"]["assessment"]["assessable"] is False
        and all(v is None for v in out["per_attribute"]["grp"]["metrics"].values())
    ),
    "detect_proxies": lambda out: (
        out["coverage"]["chains_are_a_measurement"] is False
        and out["coverage"]["attributes_not_fully_searched"] == ["grp"]
        and "not evidence that no chain exists" in out["summary"]
    ),
    "analyze_intersectional": lambda out: (
        out["coverage"]["max_disparity_measured"] is False
        and "NOT ASSESSED" in out["summary"]
        and "not a finding that no intersectional disparity exists" in out["summary"]
    ),
    # NOT `feature_screen_complete is False`: that is ALSO False on the healthy
    # frame, where one object-dtype column is legitimately never a candidate, so
    # the predicate could not have disagreed. The empty-specific facts are that
    # NOTHING was screened at all and the tool says so.
    "suggest_mitigation": lambda out: (
        out["feature_analysis"]["n_samples"] == 0
        and out["feature_analysis"]["n_features"] == 0
        and out["feature_analysis"]["not_screened_reason"] is not None
        and "could-not-check" in out["feature_analysis"]["not_screened_reason"]
        and out["coverage"]["recommendations_are_a_measurement"] is False
        and "COULD NOT CHECK" in out["summary"]
    ),
    "explain_decision": lambda out: (
        out["drivers"] == []
        and len(out["features_not_measured"]) == 5
        and all(f["association"] is None for f in out["features_not_measured"])
        and "not because they drive nothing" in out["summary"]
    ),
    "audit_agent": lambda out: (
        out["coverage"]["gap_measured"] is False
        and out["coverage"]["gap_is_interpretable"] is None
        and "not a finding that the agent treats groups alike" in out["summary"]
    ),
}


@pytest.mark.parametrize("name,kwargs", TOOLS, ids=[n for n, _ in TOOLS])
def test_every_tool_discloses_that_an_empty_frame_measured_nothing(name, kwargs):
    """Zero rows is the degenerate input every one of these has to answer for,
    and a fairness number computed over no rows is the fabrication. What is
    asserted is the specific machine-readable coverage field each tool
    publishes, plus the sentence a reader sees."""
    empty = "grp,g2,pred,target,score,feat\n"
    out = _quiet(getattr(server, name), csv=empty, **kwargs)
    assert isinstance(out, dict)
    json.dumps(out, default=str)
    assert _EMPTY_DISCLOSURE[name](out), json.dumps(out, default=str)[:600]


@pytest.mark.parametrize("name,kwargs", TOOLS, ids=[n for n, _ in TOOLS])
def test_the_empty_frame_disclosure_is_not_what_healthy_data_says(name, kwargs):
    """The control that makes the predicate above able to disagree: on the
    healthy 120-row frame the same predicate must be FALSE."""
    out = _quiet(getattr(server, name), csv=_healthy_csv(), **kwargs)
    assert not _EMPTY_DISCLOSURE[name](out), json.dumps(out, default=str)[:400]


def test_measure_fairness_on_one_group_withholds_the_comparison():
    one_group = "grp,pred,target\n" + "\n".join(f"a,{i % 2},{(i + 1) % 2}" for i in range(40))
    out = _quiet(
        server.measure_fairness,
        protected_attributes=["grp"],
        prediction_column="pred",
        target_column="target",
        csv=one_group,
    )
    per = out["per_attribute"]["grp"]
    assert per["metrics"]["demographic_parity_difference"] is None
    assert per["assessment"]["assessable"] is False


def test_measure_fairness_with_every_group_value_missing_does_not_invent_a_group():
    """The recorded incident one layer down: series.astype(str) rendered an
    attribute-less row as the literal 'None'/'nan'/'<NA>' and every metric read
    it as a real group. The wrapper must not resurrect that."""
    all_missing = "grp,pred,target\n" + "\n".join(f",{i % 2},{(i + 1) % 2}" for i in range(40))
    out = _quiet(
        server.measure_fairness,
        protected_attributes=["grp"],
        prediction_column="pred",
        target_column="target",
        csv=all_missing,
    )
    encoded = json.dumps(out, default=str)
    for phantom in ('"None"', '"nan"', '"<NA>"', '"NaN"'):
        assert phantom not in encoded, f"{phantom} became a group: {encoded[:300]}"
    assert out["per_attribute"]["grp"]["assessment"]["assessable"] is False


def test_analyze_intersectional_says_it_was_skipped_rather_than_answering_zero():
    out = _quiet(
        server.analyze_intersectional,
        protected_attributes=["grp"],
        prediction_column="pred",
        target_column="target",
        csv=_healthy_csv(),
    )
    assert out["skipped"] is True
    assert "at least two protected attributes" in out["summary"]
    # No disparity number is published beside the skip.
    assert "max_disparity" not in json.dumps(out)


def test_audit_agent_on_one_group_says_the_absence_is_not_a_finding():
    one_group = "grp,pred\n" + "\n".join(f"a,{'approve' if i % 2 else 'deny'}" for i in range(40))
    out = _quiet(server.audit_agent, group_column="grp", action_column="pred", csv=one_group)
    assert out["coverage"]["gap_measured"] is False
    assert "not a finding" in out["summary"]


# ---------------------------------------------------------------------------
# 4. THE HEALTHY CONTROL. Every refusal above would pass for a server that
#    refused everything.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kwargs", TOOLS, ids=[n for n, _ in TOOLS])
def test_every_tool_returns_a_json_encodable_answer_on_healthy_data(name, kwargs):
    out = _quiet(getattr(server, name), csv=_healthy_csv(), **kwargs)
    assert isinstance(out, dict) and out
    json.dumps(out, default=str)
    assert out.get("summary"), name


def test_measure_fairness_publishes_real_numbers_on_healthy_data():
    out = _quiet(
        server.measure_fairness,
        protected_attributes=["grp"],
        prediction_column="pred",
        target_column="target",
        csv=_healthy_csv(),
    )
    per = out["per_attribute"]["grp"]
    assert per["assessment"]["assessable"] is True
    value = per["metrics"]["demographic_parity_difference"]
    assert isinstance(value, float) and 0.0 <= value <= 1.0
    assert per["group_stats"], "a measured run published no per-group statistics"


def test_the_two_resources_are_documents_and_the_legal_one_disclaims():
    glossary = server.glossary()
    legal = server.legal_framing()
    assert glossary.lstrip().startswith("#") and "demographic parity" in glossary.lower()
    assert legal.lstrip().startswith("#")
    # The docstring promises "Informational legal framing ... Not legal advice",
    # and a reader only ever sees the document.
    assert "not legal advice" in legal.lower()
    for token in ("4/5", "eu ai act"):
        assert token in legal.lower(), token
