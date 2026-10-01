"""
CausalFairnessGraph, DAG specification for causal fairness (Workstream F).

A lightweight, networkx-backed directed acyclic graph for declaring the
cause-and-effect structure of a decision system and classifying the pathways
from a protected attribute to an outcome as DIRECT discrimination, INDIRECT
discrimination (mediated by a legitimate variable), or PROXY (via a variable
that merely stands in for the protected attribute).

This is the ROADMAP "Causal Fairness Framework, Phase 1" deliverable and the
graph-theoretic complement to the DoWhy-backed estimation handlers
(identify / mediate / refute / counterfactual / attribute) in this package.
Unlike those, it needs only `networkx` (already a core dep), no DoWhy, so the
graph can be authored and reasoned about even where DoWhy is unavailable.

Example::

    g = CausalFairnessGraph()
    g.add_variable('gender', protected=True)
    g.add_variable('education', mediator=True)
    g.add_variable('hiring', outcome=True)
    g.add_edge('gender', 'education')      # historical effect
    g.add_edge('education', 'hiring')      # legitimate
    g.add_edge('gender', 'hiring')         # direct discrimination
    paths = g.discrimination_paths()       # -> [DiscriminationPath(...), ...]
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class DiscriminationPath:
    """One protected -> outcome pathway and its fairness classification."""

    source: str
    target: str
    path: List[str]
    kind: str  # 'direct' | 'indirect' | 'proxy'
    mediators: List[str] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _PathScanResult(list):
    """A list of pathways that can also say the graph could not be assessed.

    BGL-C (2026-09-17). ``discrimination_paths`` returned a bare list, and an
    empty one was the answer to two different questions: "the graph was read and
    no protected -> outcome pathway exists" (a measurement) and "no variable was
    ever marked protected=True, so no pathway could exist by construction"
    (nothing was measured). This stays a list for every existing caller and
    carries ``not_assessable`` for the ones that ask, exactly as ``_ScanResult``
    does for the tabular scan in ``evaluation.vfairness_metrics.discovery``.
    """

    not_assessable: List[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.not_assessable = []


#: What ``str()`` of an absent value prints. A variable NAME equal to one of
#: these almost certainly came from stringifying an absence rather than from an
#: author, so it is warned about; the absent objects themselves are refused
#: outright by :func:`_checked_name`. Deliberately case-sensitive and
#: deliberately WITHOUT bare 'NA', which is a legitimate name (North America).
_ABSENT_LITERALS = {"None", "nan", "NaN", "NaT", "<NA>", "null", "NULL"}


def _checked_name(name: Any, role: str = "variable") -> Any:
    """Refuse a variable name that is an ABSENT VALUE rather than a name.

    BGL-G02 (2026-09-30). networkx refuses ``None`` ("None cannot be a node")
    and nothing else, so every other door minted a declared variable. Measured
    that day, each with no warning of any kind::

        add_variable(float('nan'), protected=True) -> protected_nodes() [nan]
        add_variable(pd.NA,        protected=True) -> protected_nodes() [<NA>]
        add_variable(pd.NaT,       protected=True) -> protected_nodes() [NaT]
        add_variable('',           protected=True) -> protected_nodes() ['']
        add_edge(pd.NA, 'hiring')                  -> nodes [<NA>, 'hiring']

    That is not only a nameless node in a report. ``not_assessable`` refuses to
    grade a graph in which no variable is marked protected, and an absent name
    SATISFIES that precondition: a graph whose only protected "variable" is a
    NaN reads as assessable and gets a real severity. The third state is
    defeated by a missing label, which is exactly how ``pd.NA`` once minted a
    demographic group called '<NA>' elsewhere in this repo.
    """
    if name is None:
        raise ValueError(
            f"The {role} name cannot be None. An absent name is not a variable: it would be "
            f"counted as a declared role and make the graph look assessable."
        )
    if isinstance(name, float) and name != name:  # NaN, without importing math
        raise ValueError(
            f"The {role} name cannot be NaN. An absent name is not a variable: it would be "
            f"counted as a declared role and make the graph look assessable."
        )
    if not isinstance(name, str) and repr(name) in ("<NA>", "NaT", "nan", "None"):
        # pd.NA / pd.NaT / a numpy NaN scalar, recognised by repr so this
        # networkx-only module does not have to import pandas to do it.
        raise ValueError(
            f"The {role} name cannot be {name!r} (an absent pandas value). An absent name is "
            f"not a variable: it would be counted as a declared role and make the graph "
            f"look assessable."
        )
    if isinstance(name, str):
        if not name.strip():
            raise ValueError(
                f"The {role} name cannot be blank. An absent name is not a variable: it would "
                f"be counted as a declared role and make the graph look assessable."
            )
        if name in _ABSENT_LITERALS:
            warnings.warn(
                f"CausalFairnessGraph: the {role} name {name!r} is what str() of an absent "
                f"value prints. If it came from a missing label rather than from you, this "
                f"graph declares a variable that does not exist, and its role counts towards "
                f"making the graph look assessable.",
                UserWarning,
                stacklevel=3,
            )
    return name


def _import_nx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - networkx is a core dep
        raise ImportError(
            "CausalFairnessGraph requires networkx. Install with: pip install networkx"
        ) from exc
    return nx


class CausalFairnessGraph:
    """A DAG of variables with fairness roles.

    Roles (set per variable):
      protected : a protected/sensitive attribute (race, gender, age, ...).
      outcome   : the decision/target.
      mediator  : a legitimate variable on a path from protected to outcome
                  (its effect is considered explainable, not discriminatory).
      proxy     : a variable that stands in for a protected attribute (its
                  effect on the outcome is treated as discriminatory).

    BGL-U00 (2026-09-17). This class HAS now been reached and run, on six graphs
    (tests/test_bgl_final_u00.py), and it FABRICATED. Four of the six cannot be
    traced at all, because no variable carries protected=True, or none carries
    outcome=True, or the only protected variable is also the only outcome. Each
    returned discrimination_paths() == [], has_direct_discrimination() is False
    and summary()['severity'] == 'info': byte for byte what a graph WITH both
    roles and genuinely no pathway returns. The realistic one is a graph built
    only from add_edge, since add_edge creates missing variables with every role
    False, so add_edge('gender', 'hiring') alone yields a graph containing literal
    direct discrimination and grades it "info, no direct discrimination". Fixed
    below by `not_assessable()`, a None verdict and an "unknown" severity; an
    assessable graph still reports its real pathways and its real clean verdict.

    The machine-generated stamp below is derived from src/vfairness/_proof_status.py
    and is STALE until that ledger, its evidence file and the published tables are
    regenerated together. Read this paragraph, not the stamp, for what is known.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: causal_fairness_graph. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self):
        nx = _import_nx()
        self._nx = nx
        self.graph = nx.DiGraph()
        #: Roles a later ``add_variable`` call CLEARED. Surfaced in
        #: :meth:`summary` because it changes the published verdict; see
        #: :meth:`add_variable`.
        self.cleared_roles: List[str] = []

    # construction

    def add_variable(
        self,
        name: str,
        protected: bool = False,
        outcome: bool = False,
        mediator: bool = False,
        proxy: bool = False,
    ) -> "CausalFairnessGraph":
        """Declare a variable and its fairness roles. The LAST call wins.

        BGL-G02 (2026-09-30). ``add_node`` overwrites the attributes it is
        given, and this method always passes all four, so re-declaring a
        variable to ADD a role silently CLEARS the ones not repeated. Measured
        that day on gender -> zip -> hiring with zip declared ``proxy=True``::

            discrimination_paths() kinds ['proxy'],  summary() severity 'high'
            g.add_variable('zip', mediator=True)     # author means "also a mediator"
            discrimination_paths() kinds ['indirect'], summary() severity 'medium'
            warnings: none

        The published verdict went from PROXY discrimination ("treat as
        discriminatory") to an INDIRECT effect ("may be explainable") because a
        second declaration dropped the first, and the only record of it was the
        absence of a word. The semantics are unchanged (a re-declaration still
        replaces, so no existing caller moves), but the clearing is now named in
        a warning and in ``summary()['cleared_roles']``.
        """
        name = _checked_name(name)
        if name in self.graph:
            requested = {
                "protected": protected,
                "outcome": outcome,
                "mediator": mediator,
                "proxy": proxy,
            }
            cleared = [
                role for role, wanted in requested.items() if self._role(name, role) and not wanted
            ]
            if cleared:
                note = (
                    f"re-declaring '{name}' cleared the role(s) "
                    f"{', '.join(sorted(cleared))} it already carried"
                )
                self.cleared_roles.append(note)
                warnings.warn(
                    f"CausalFairnessGraph: {note}. add_variable() replaces every role, it "
                    f"does not add one, so a pathway's classification and the graph's "
                    f"severity can change on this call. Declare every role the variable "
                    f"has in ONE call.",
                    UserWarning,
                    stacklevel=2,
                )
        self.graph.add_node(
            name,
            protected=protected,
            outcome=outcome,
            mediator=mediator,
            proxy=proxy,
        )
        return self

    def add_edge(self, src: str, dst: str) -> "CausalFairnessGraph":
        src = _checked_name(src, "edge source")
        dst = _checked_name(dst, "edge target")
        for n in (src, dst):
            if n not in self.graph:
                self.graph.add_node(n, protected=False, outcome=False, mediator=False, proxy=False)
        self.graph.add_edge(src, dst)
        if not self._nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(src, dst)
            raise ValueError(f"Adding edge {src}->{dst} would create a cycle; DAG required.")
        return self

    # queries

    def _role(self, node: str, role: str) -> bool:
        return bool(self.graph.nodes[node].get(role, False))

    def protected_nodes(self) -> List[str]:
        return [n for n in self.graph if self._role(n, "protected")]

    def outcome_nodes(self) -> List[str]:
        return [n for n in self.graph if self._role(n, "outcome")]

    def not_assessable(self) -> List[str]:
        """Why this graph cannot be assessed for discrimination pathways.

        Empty means it CAN be assessed, so an empty pathway list from an
        assessable graph is a real measurement: the structure was read and no
        protected -> outcome pathway exists in it.

        Five reasons, each one a by-construction impossibility rather than a
        judgement: an empty graph, a graph with no edges at all, no variable
        marked protected, no variable marked outcome, the only protected variable
        also being the only outcome, and no edge pointing into any declared
        outcome. See the measured cases in the comments below.

        BGL-C (2026-09-17). Measured on four graphs that carry no protected or
        no outcome role, including one built entirely with ``add_edge`` (which
        creates nodes with every role defaulting to False, so
        ``add_edge('gender', 'hiring')`` alone declares nothing): each returned
        ``discrimination_paths() == []``, ``has_direct_discrimination() is
        False`` and ``summary()['severity'] == 'info'``, which is byte for byte
        what a graph with both roles declared and genuinely no pathway returns.
        The most reassuring verdict this class emits was the default for a graph
        nobody had told it how to read.
        """
        reasons: List[str] = []
        n_nodes = self.graph.number_of_nodes()
        if n_nodes == 0:
            reasons.append(
                "the graph is empty: no variable has been declared, so no "
                "protected -> outcome pathway can exist by construction"
            )
            return reasons

        # BGL5 AUDIT, 2026-09-27. The EDGE analogue of the node guard above was
        # missing, and the sentence it prints is true of it word for word.
        # Measured on a graph declaring gender=protected and hiring=outcome and
        # NO edges at all:
        #   has_direct_discrimination() -> False
        #   summary() -> severity 'info', assessable True, not_assessable [],
        #                n_paths 0, warnings []
        # i.e. byte-identical to the traced graph in
        # tests/test_bgl_final_u00.py::_traced_but_clean_graph, which HAS a
        # structure. Declaring the roles and never authoring the causal structure
        # is the commonest half-finished state of this object (the five shapes the
        # previous fix pinned all have a MISSING ROLE instead). Now it refuses:
        # has_direct_discrimination() None, severity 'unknown', reason printed.
        if self.graph.number_of_edges() == 0:
            reasons.append(
                f"the graph has no edges: {n_nodes} variable(s) are declared but no "
                "causal structure was authored, so no protected -> outcome pathway can "
                "exist by construction"
            )
            return reasons

        protected = self.protected_nodes()
        outcome = self.outcome_nodes()

        # Same by-construction argument, one step in: a path that ENDS at an
        # outcome needs an edge INTO that outcome. Measured 2026-09-27 on two
        # authoring errors the previous fix did not cover, both of which returned
        # the cleared verdict False at severity 'info' with no warning:
        #   add_edge("gender", "hirng")  -> the declared outcome 'hiring' is
        #       isolated, so the typo silently produced a clean bill of health
        #   add_edge("hiring", "gender") -> the edge points the wrong way, so
        #       nothing in the graph determines the outcome at all
        # Deliberately NOT mirrored on the protected side: a declared protected
        # variable with no outgoing edge inside an AUTHORED graph is a positive
        # statement that the attribute does not enter the model, and
        # tests/test_bgl_final_u00.py::test_a_traced_graph_with_no_pathway_keeps_
        # its_clean_verdict pins that reading (its 'gender' has no out-edge). The
        # outcome is the target of every question this class asks, so a decision
        # with no modelled parents leaves all of them unanswerable.
        if outcome and not any(self.graph.in_degree(y) for y in outcome):
            reasons.append(
                f"no edge points INTO any declared outcome variable {sorted(outcome)}, so "
                "nothing in this graph determines the outcome and no pathway ending there "
                "can exist by construction (check for a misspelt edge target or an edge "
                "pointing the wrong way)"
            )

        if not protected:
            reasons.append(
                f"no variable is marked protected=True ({n_nodes} variable(s) declared), "
                "so no protected -> outcome pathway can exist by construction. Note that "
                "add_edge() creates missing variables with every role set to False, so a "
                "graph built only from edges declares no roles at all"
            )
        if not outcome:
            reasons.append(
                f"no variable is marked outcome=True ({n_nodes} variable(s) declared), "
                "so no protected -> outcome pathway can exist by construction"
            )
        if protected and outcome and not [(p, y) for p in protected for y in outcome if p != y]:
            reasons.append(
                f"the only protected variable(s) {sorted(protected)} are also the only "
                "outcome variable(s), so there is no distinct source/target pair to trace"
            )
        return reasons

    def discrimination_paths(self) -> List[DiscriminationPath]:
        """Enumerate and classify every protected -> outcome path.

        Returns a list that also carries ``not_assessable``: a non-empty
        ``not_assessable`` with an empty list means NOTHING WAS TRACED, which is
        not the same answer as a traced graph with no pathway in it.
        """
        nx = self._nx
        results = _PathScanResult()
        reasons = self.not_assessable()
        results.not_assessable = list(reasons)
        if reasons:
            warnings.warn(
                "CausalFairnessGraph could not be assessed: "
                + "; ".join(reasons)
                + ". An empty pathway list here means no pathway was traced, not that "
                "the system is free of discrimination pathways.",
                UserWarning,
                stacklevel=2,
            )
            return results
        for p in self.protected_nodes():
            for y in self.outcome_nodes():
                if p == y or not nx.has_path(self.graph, p, y):
                    continue
                for path in nx.all_simple_paths(self.graph, p, y):
                    results.append(self._classify(path))
        return results

    def _classify(self, path: Sequence[str]) -> DiscriminationPath:
        src, target = path[0], path[-1]
        intermediates = list(path[1:-1])
        if not intermediates:
            kind, verdict = (
                "direct",
                (
                    f"DIRECT discrimination: '{target}' depends on the protected "
                    f"attribute '{src}' with no intervening variable. This path is "
                    f"prima facie unfair and should be removed or justified."
                ),
            )
        elif any(self._role(n, "proxy") for n in intermediates):
            proxies = [n for n in intermediates if self._role(n, "proxy")]
            kind, verdict = (
                "proxy",
                (
                    f"PROXY discrimination: the effect of '{src}' on '{target}' flows "
                    f"through proxy variable(s) {proxies}, which stand in for the "
                    f"protected attribute. Treat as discriminatory unless the proxy "
                    f"has an independent, justified business meaning."
                ),
            )
        else:
            kind, verdict = (
                "indirect",
                (
                    f"INDIRECT effect: '{src}' influences '{target}' via "
                    f"{intermediates}. If those are legitimate mediators, this path "
                    f"may be explainable; confirm each mediator is not itself a proxy."
                ),
            )
        return DiscriminationPath(
            source=src,
            target=target,
            path=list(path),
            kind=kind,
            mediators=intermediates,
            verdict=verdict,
        )

    def has_direct_discrimination(self) -> Optional[bool]:
        """True / False when the graph can be read, None when it cannot.

        None is the third state and is never False: a graph with no protected
        role declared has not been cleared of direct discrimination, it has not
        been examined for it.
        """
        paths = self.discrimination_paths()
        if getattr(paths, "not_assessable", []):
            return None
        return any(d.kind == "direct" for d in paths)

    def summary(self) -> Dict[str, Any]:
        paths = self.discrimination_paths()
        reasons = list(getattr(paths, "not_assessable", []))
        by_kind: Dict[str, int] = {"direct": 0, "indirect": 0, "proxy": 0}
        for d in paths:
            by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
        if reasons:
            # Not "info". "info" is the grade a graph earns by being read and
            # found free of pathways; this one was not read.
            severity = "unknown"
        elif by_kind["direct"] or by_kind["proxy"]:
            severity = "high"
        elif by_kind["indirect"]:
            severity = "medium"
        else:
            severity = "info"
        return {
            "n_paths": len(paths),
            "counts": by_kind,
            "severity": severity,
            # None, never False, when nothing was traced. Both fields below are
            # explicit so a reader can tell the third state from a measurement
            # without opening this file.
            "has_direct_discrimination": None if reasons else by_kind["direct"] > 0,
            "assessable": not reasons,
            "not_assessable": reasons,
            # Roles a later add_variable() call cleared. A cleared proxy role
            # DOWNGRADES this very severity, so it belongs where the severity
            # is read, not only in a Python warning nobody kept. See
            # add_variable().
            "cleared_roles": list(self.cleared_roles),
            "paths": [d.to_dict() for d in paths],
        }

    # serialisation

    def to_gml(self) -> str:
        """Serialise to GML (the same format the causal estimation handlers
        and the frontend DAG serialiser use)."""
        return "\n".join(self._nx.generate_gml(self.graph))
