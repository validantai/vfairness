"""CausalFairnessGraph test suite, Workstream F.

Builds the canonical hiring DAG (gender -> education -> hiring legitimate,
gender -> hiring direct, gender -> zip -> hiring proxy) and asserts the graph
classifies each pathway correctly.
"""

import pytest

from vfairness.operations.causal.graph import (
    CausalFairnessGraph,
    DiscriminationPath,
)


def _hiring_graph():
    g = CausalFairnessGraph()
    g.add_variable("gender", protected=True)
    g.add_variable("education", mediator=True)
    g.add_variable("zip", proxy=True)
    g.add_variable("hiring", outcome=True)
    g.add_edge("gender", "education")  # historical
    g.add_edge("education", "hiring")  # legitimate -> indirect path
    g.add_edge("gender", "hiring")  # direct discrimination
    g.add_edge("gender", "zip")  # gender determines zip
    g.add_edge("zip", "hiring")  # proxy path
    return g


def test_cycle_rejected():
    g = CausalFairnessGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    with pytest.raises(ValueError):
        g.add_edge("c", "a")


def test_classifies_all_three_path_kinds():
    g = _hiring_graph()
    paths = g.discrimination_paths()
    kinds = {tuple(p.path): p.kind for p in paths}
    assert kinds[("gender", "hiring")] == "direct"
    assert kinds[("gender", "education", "hiring")] == "indirect"
    assert kinds[("gender", "zip", "hiring")] == "proxy"
    assert all(isinstance(p, DiscriminationPath) and p.verdict for p in paths)


def test_has_direct_discrimination():
    assert _hiring_graph().has_direct_discrimination() is True


def test_no_direct_when_only_mediated():
    g = CausalFairnessGraph()
    g.add_variable("race", protected=True)
    g.add_variable("score", mediator=True)
    g.add_variable("loan", outcome=True)
    g.add_edge("race", "score")
    g.add_edge("score", "loan")
    assert g.has_direct_discrimination() is False
    paths = g.discrimination_paths()
    assert len(paths) == 1 and paths[0].kind == "indirect"


def test_summary_severity_and_shape():
    s = _hiring_graph().summary()
    assert s["n_paths"] == 3
    assert s["counts"] == {"direct": 1, "indirect": 1, "proxy": 1}
    assert s["severity"] == "high"  # direct/proxy present
    assert s["has_direct_discrimination"] is True
    import json

    json.dumps(s)  # JSON-friendly


def test_indirect_only_is_medium():
    g = CausalFairnessGraph()
    g.add_variable("age", protected=True)
    g.add_variable("tenure", mediator=True)
    g.add_variable("promo", outcome=True)
    g.add_edge("age", "tenure")
    g.add_edge("tenure", "promo")
    assert g.summary()["severity"] == "medium"


def test_to_gml_roundtrips_nodes():
    gml = _hiring_graph().to_gml()
    assert "gender" in gml and "hiring" in gml
    import networkx as nx

    parsed = nx.parse_gml(gml)
    assert parsed.number_of_nodes() == 4
