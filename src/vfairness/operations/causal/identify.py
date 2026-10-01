"""
Causal effect identification via DoWhy.

Given a DAG (as GML) plus a list of treatments and outcomes, return for every
(treatment, outcome) pair:
    * whether the causal effect is identifiable from observational data
    * which adjustment set closes the backdoor (if any)
    * which instruments are available (if any)
    * the frontdoor adjustment set (if any)
    * a plain-language verdict the UI can show without exposing library jargon

This is Phase 1 of the DoWhy integration. No estimation, no refutation; only
graph-theoretic identification, which is fast and side-effect free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence


@dataclass
class PathIdentification:
    treatment: str
    outcome: str
    #: THREE states, never two. ``True`` identifiable, ``False`` a directed path
    #: exists and observational data cannot separate it from confounding,
    #: ``None`` nothing was decided: no directed path in the graph, treatment
    #: and outcome the same variable, the graph not a DAG, or identification
    #: raised. ``notes`` names which, and ``verdict`` says it in words.
    #: BGL-G02 (2026-09-30); see :func:`identify_paths`.
    is_identifiable: Optional[bool]
    backdoor_adjustment_set: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    frontdoor_adjustment_set: List[str] = field(default_factory=list)
    estimand_expression: Optional[str] = None
    verdict: str = ""
    notes: List[str] = field(default_factory=list)
    #: True when the graph contains no directed path from treatment to outcome.
    #: That is a statement ABOUT THE GRAPH, not a failure of identification, and
    #: it used to be reported as "we cannot separate this effect from
    #: confounding".
    no_causal_path: bool = False
    #: True when identifiability was decided WITHOUT being told which variables
    #: the dataset actually holds, so every node in the graph was assumed to be
    #: measured. An unobserved confounder in the graph is then offered as an
    #: adjustment variable. See :func:`identify_paths`.
    assumes_all_graph_nodes_observed: bool = False


@dataclass
class IdentificationResult:
    treatments: List[str]
    outcomes: List[str]
    paths: List[PathIdentification] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "treatments": self.treatments,
            "outcomes": self.outcomes,
            "paths": [asdict(p) for p in self.paths],
            "warnings": list(self.warnings),
        }


def _verdict_for(p: PathIdentification) -> str:
    """User-facing sentence. No 'backdoor', 'estimand', or library names.

    BGL-G02 (2026-09-30). The "we cannot cleanly separate this effect from
    confounding" sentence was returned for FIVE structurally different
    situations, four of which it describes wrongly, all with ``notes == []`` so
    nothing distinguished them. Measured that day, each one word for word
    identical::

        treatment == outcome                     -> the confounding sentence
        treatment and outcome DISCONNECTED       -> the confounding sentence
        the graph contains a CYCLE               -> the confounding sentence
        the graph has no edges at all            -> the confounding sentence
        the outcome is an ANCESTOR of the        -> the confounding sentence
            treatment (the model authored
            backwards)

    In the disconnected case the truth is the OPPOSITE of the sentence: there is
    no confounding to separate because the graph asserts no effect at all, and
    "add the missing variables" sends the reader looking for a confounder that
    does not exist instead of at the edge they forgot to draw. Each now has its
    own state and its own sentence.
    """
    if p.is_identifiable is None:
        if p.no_causal_path:
            return (
                f"This graph contains no cause-and-effect pathway from '{p.treatment}' to "
                f"'{p.outcome}', so it asserts there is nothing to estimate. Nothing was "
                f"checked about confounding. If an effect is expected here, the graph is "
                f"missing an edge, an edge points the wrong way, or a name is misspelt."
            )
        return (
            "We could not decide whether this effect is identifiable. This is not a "
            "finding about confounding and it is not a clean bill of health: see the "
            "notes for what stopped the check."
        )
    if not p.is_identifiable:
        return (
            "We cannot cleanly separate this effect from confounding using the "
            "current graph. Either add the missing variables or treat results "
            "from this path as suggestive only."
        )
    parts: List[str] = []
    if p.backdoor_adjustment_set:
        parts.append(
            "we can isolate this effect by controlling for " + ", ".join(p.backdoor_adjustment_set)
        )
    if p.instruments:
        parts.append("an instrument is available (" + ", ".join(p.instruments) + ")")
    if p.frontdoor_adjustment_set:
        parts.append(
            "an indirect-pathway adjustment exists via " + ", ".join(p.frontdoor_adjustment_set)
        )
    caveat = (
        (
            " This rests on an assumption nobody checked: the dataset was not named, so "
            "every variable in the graph was taken to be measured in it. If any variable "
            "above is not in the data, this effect is NOT identifiable from it."
        )
        if p.assumes_all_graph_nodes_observed
        else ""
    )
    if not parts:
        return "This effect is identifiable directly from the graph." + caveat
    return "This effect is identifiable: " + "; ".join(parts) + "." + caveat


def _import_dowhy():
    try:
        import networkx as nx  # noqa: F401
        from dowhy import CausalModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vfairness.operations.causal.identify requires the optional "
            "dependencies 'dowhy' and 'networkx'. Install with: "
            "pip install dowhy networkx"
        ) from exc
    return CausalModel


def identify_paths(
    gml: str,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    dataset_columns: Optional[Sequence[str]] = None,
) -> IdentificationResult:
    """
    Run DoWhy identification for every (treatment, outcome) pair.

    Parameters
    ----------
    gml : str
        Serialized DAG in GML format (produced by the frontend DAG serialiser).
    treatments : sequence of str
        GML node labels treated as protected attributes / treatments.
    outcomes : sequence of str
        GML node labels treated as outcomes.
    dataset_columns : sequence of str, optional
        The column names the DATASET actually holds. Identification is
        graph-theoretic but not graph-only: which variables are MEASURED decides
        whether a backdoor set is available. If absent, every node in the graph
        is assumed to be in the data, the result says so on every path
        (``assumes_all_graph_nodes_observed``), and the verdict carries the
        caveat. Pass the real columns whenever they are known.

    Returns
    -------
    IdentificationResult
        Aggregate result with one PathIdentification per (treatment, outcome) pair.

    BGL-G02 (2026-09-30). Two defects, both measured on this repo that day.

    1. IDENTIFIABILITY WAS ASSERTED ON AN UNCHECKABLE ASSUMPTION, silently. With
       ``dataset_columns`` left out, the placeholder frame is built from the
       GRAPH's own labels, so a latent confounder is treated as a measured
       variable. On X <- U -> Y with X -> Y::

           identify_paths(gml, ["X"], ["Y"])
               is_identifiable True, backdoor ['U'], warnings []
               verdict "we can isolate this effect by controlling for U"
           identify_paths(gml, ["X"], ["Y"], dataset_columns=["X", "Y"])
               is_identifiable False

       U is unobserved by construction, which is what makes the graph a
       confounded one worth drawing. ``handle_identify`` never passes
       ``dataset_columns``, so in production every effect with a graph-side
       adjustment set came back identifiable. The assumption is now named on the
       path and in the verdict.

    2. FIVE DIFFERENT SITUATIONS SHARED ONE VERDICT. See :func:`_verdict_for`.
    """
    CausalModel = _import_dowhy()
    import networkx as nx
    import pandas as pd

    result = IdentificationResult(treatments=list(treatments), outcomes=list(outcomes))

    if not treatments or not outcomes:
        result.warnings.append("No treatments or no outcomes provided.")
        return result

    try:
        graph = nx.parse_gml(gml)
    except Exception as exc:
        result.warnings.append(f"Could not parse DAG: {exc}")
        return result

    node_labels = {str(n) for n in graph.nodes()}

    # ABOVE the pair loop: a graph that is not a DAG makes every pair
    # unanswerable for the same reason, and the reason is a property of the
    # graph, not of any pair. Measured 2026-09-30 on X -> Y -> X: DoWhy returned
    # an estimand whose every strategy was None and the suite reported
    # "is_identifiable False" with empty notes for a graph the whole framework
    # does not apply to.
    not_a_dag: Optional[str] = None
    if not nx.is_directed_acyclic_graph(graph):
        cycle = ""
        try:
            cycle = " (e.g. " + " -> ".join(str(n) for n, _ in nx.find_cycle(graph)) + ")"
        except Exception:  # noqa: BLE001 - the cycle is illustrative, not required
            cycle = ""
        not_a_dag = (
            f"The graph is not acyclic{cycle}, so it is not a causal DAG and no effect in "
            f"it can be identified. Nothing was checked about confounding; this is a "
            f"could-not-check, not a finding."
        )
        result.warnings.append(not_a_dag)

    assumed_observed = dataset_columns is None
    columns: List[str] = [str(c) for c in (dataset_columns or [])]
    if assumed_observed:
        columns = sorted(node_labels)
        result.warnings.append(
            "No dataset columns were given, so identifiability below assumes EVERY "
            "variable in the graph is measured in the dataset. A variable drawn as an "
            "unobserved confounder is offered as an adjustment variable under that "
            "assumption. Pass dataset_columns to have it checked."
        )
    placeholder = pd.DataFrame({c: [0] for c in columns})
    available = set(columns)

    for t in treatments:
        if t not in node_labels:
            result.warnings.append(f"Treatment '{t}' not present in DAG. Skipped.")
            continue
        for y in outcomes:
            if y not in node_labels:
                result.warnings.append(f"Outcome '{y}' not present in DAG. Skipped.")
                continue

            path = PathIdentification(treatment=t, outcome=y, is_identifiable=None)
            path.assumes_all_graph_nodes_observed = assumed_observed

            if not_a_dag is not None:
                path.notes.append(not_a_dag)
                path.verdict = _verdict_for(path)
                result.paths.append(path)
                continue
            if t == y:
                path.notes.append(
                    f"The treatment and the outcome are the same variable ('{t}'), so there "
                    f"is no effect of one on the other to identify. Nothing was checked."
                )
                path.verdict = _verdict_for(path)
                result.paths.append(path)
                continue
            try:
                model = CausalModel(
                    data=placeholder,
                    treatment=t,
                    outcome=y,
                    graph=gml,
                )
                estimand = model.identify_effect(proceed_when_unidentifiable=True)

                backdoor = list(estimand.get_backdoor_variables() or [])
                instruments = list(estimand.get_instrumental_variables() or [])
                frontdoor = list(estimand.get_frontdoor_variables() or [])

                path.backdoor_adjustment_set = backdoor
                path.instruments = instruments
                path.frontdoor_adjustment_set = frontdoor
                path.estimand_expression = str(estimand)
                # Identifiability must come from whether DoWhy produced a valid
                # estimand, NOT from non-emptiness of the adjustment sets: an
                # unconfounded effect is identifiable with an EMPTY backdoor
                # set (no adjustment needed). The estimands dict holds a
                # non-None entry per available identification strategy.
                estimand_types = getattr(estimand, "estimands", None) or {}
                has_valid_estimand = any(v is not None for v in estimand_types.values())

                # DoWhy states this itself, and it is the discriminator the
                # five collapsed cases needed: no directed treatment -> outcome
                # path in the graph. Verified 2026-09-30, no_directed_path is
                # True for the disconnected, self-pair and reversed-edge graphs
                # and False for the cyclic and healthy ones.
                if getattr(estimand, "no_directed_path", False):
                    path.no_causal_path = True
                    path.is_identifiable = None
                    path.notes.append(
                        f"The graph holds no directed path from '{t}' to '{y}', so it "
                        f"asserts there is no effect to identify. This is a reading of the "
                        f"graph, not a failure to separate an effect from confounding."
                    )
                else:
                    path.is_identifiable = bool(
                        has_valid_estimand or backdoor or instruments or frontdoor
                    )
                    if not path.is_identifiable:
                        path.notes.append(
                            f"A directed path from '{t}' to '{y}' exists, and DoWhy found no "
                            f"identification strategy for it with the variables available."
                        )

                unavailable = sorted(
                    {str(v) for v in backdoor + instruments + frontdoor} - available
                )
                if unavailable:
                    path.notes.append(
                        f"The adjustment set names variable(s) the dataset was not said to "
                        f"hold: {unavailable}. Without them this effect is not identifiable "
                        f"from that dataset."
                    )
            except Exception as exc:
                # A crash is a could-not-check, never "not identifiable": the
                # old False sent the reader the confounding verdict, which is a
                # substantive causal claim nobody had established.
                path.notes.append(f"Identification raised: {exc}")
                path.is_identifiable = None

            path.verdict = _verdict_for(path)
            result.paths.append(path)

    return result
