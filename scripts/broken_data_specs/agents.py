"""Check 2 specs for the agent and multi-agent capabilities.

Loaded by scripts/broken_data_check.py (``all_specs``). Each spec maps the shared
World into the capability's real input, one agent decision per row:

  * the row's group ``w.s`` is the demographic (or, for GroupthinkDetector, the
    agent identity, since convergence is a property of agents, not of rows);
  * the row's decision ``w.p`` is the action / route / tool / callback, and
    ``w.prob`` is the score an agent produced before deciding;
  * a row whose decision is NaN carries NO decision: it becomes a missing
    outcome (a record without the outcome, a None, a retrieval with no document,
    a blank completion, or no tool call), never a 0.

Two-group methods take the first two sorted labels as A and B; a label that is
absent gives that side an empty input, which is what a caller would pass.

Methods that need TURNS or ROUNDS (TemporalTracker, NegotiationFairnessTracker,
GroupthinkDetector) split each group's decisions, in row order, into
``min(5, smallest group)`` consecutive chunks, so a turn exists only where every
group has a row for it. The number of turns a group can supply is therefore its
row count, and ``min_group_size`` on those specs is the method's OWN minimum
number of turns (3 for Kendall, CUSUM and EWMA, 2 for drift and for a round
pair), not a choice made here.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

_TURNS = 5


def _labels(w) -> List[str]:
    return sorted(set(w.s.tolist()))


def _pair(w):
    labs = _labels(w)
    return (labs + [None, None])[:2]


def _decisions(w, g) -> np.ndarray:
    """The recorded decisions of group g, in row order, missing ones dropped."""
    if g is None:
        return np.zeros(0)
    p = np.asarray(w.p, dtype=float)[w.s == g]
    return p[np.isfinite(p)]


def _raw(w, g, attr="p") -> np.ndarray:
    """Group g's values in row order, NaN kept (the caller decides what NaN is)."""
    if g is None:
        return np.zeros(0)
    return np.asarray(getattr(w, attr), dtype=float)[w.s == g]


def _turn_chunks(w) -> Dict[str, List[np.ndarray]]:
    """Each group's decisions split into the same number of consecutive turns."""
    per = {g: _decisions(w, g) for g in _labels(w)}
    if not per:
        return {}
    n_turns = min([_TURNS] + [len(v) for v in per.values()])
    if n_turns <= 0:
        return {g: [] for g in per}
    return {g: [c for c in np.array_split(v, n_turns)] for g, v in per.items()}


def _gap(values: np.ndarray, groups: np.ndarray) -> float:
    """Widest gap between group means; undefined (nan) without two measured groups."""
    v = np.asarray(values, dtype=float)
    means = []
    for g in sorted(set(groups.tolist())):
        x = v[groups == g]
        x = x[np.isfinite(x)]
        if len(x):
            means.append(float(np.mean(x)))
    if len(means) < 2:
        return float("nan")
    return float(max(means) - min(means))


def _bool_or_refused(v, REFUSED):
    return REFUSED if v is None else v


# ===========================================================================
# agents/
# ===========================================================================


def _action_records(w, g):
    out = []
    for v in _raw(w, g):
        rec = {"action": "approve" if v == 1 else ("review" if v == 0 else "unknown")}
        if np.isfinite(v):
            rec["approved"] = float(v)  # a missing decision has no outcome field at all
        out.append(rec)
    return out


def _routes(w, g, yes="approve", no="manual_review"):
    return [yes if v == 1 else no for v in _decisions(w, g)]


def _turn_tracker(w, cls):
    t = cls()
    chunks = _turn_chunks(w)
    a, b = _pair(w)
    if a is None or b is None or not chunks.get(a) or not chunks.get(b):
        return t
    for i, (ca, cb) in enumerate(zip(chunks[a], chunks[b])):
        t.record_turn(i, ca, cb)
    return t


def _pipeline(w):
    from vfairness.agents.pipeline_tracker import PipelineTracker

    a, b = _pair(w)
    t = PipelineTracker(stages=["scoring", "decision"])
    # NaN is kept: this tracker's own contract is that a non-finite outcome makes
    # the stage unmeasurable, rather than being dropped from the denominator.
    t.record_stage("scoring", _raw(w, a, "prob"), _raw(w, b, "prob"))
    t.record_stage("decision", _raw(w, a), _raw(w, b))
    return t


def _rag_parts(w, g):
    docs, outs, queries = [], [], []
    for i, v in enumerate(_raw(w, g)):
        queries.append(f"query {g} {i}")
        if v == 1:
            docs.append({"id": "policy_approve"})
            outs.append("Your application meets the criteria and is approved.")
        elif v == 0:
            docs.append({"id": "policy_decline"})
            outs.append("Your application does not meet the criteria at this time.")
        else:
            outs.append("")  # nothing was retrieved and nothing was generated
    return queries, docs, outs


def _agent_specs(Spec, REFUSED) -> list:
    from vfairness.agents.action_bias import ActionBiasAnalyzer
    from vfairness.agents.correspondence import CorrespondenceTester
    from vfairness.agents.rag_bias import RAGBiasAnalyzer
    from vfairness.agents.temporal import TemporalTracker
    from vfairness.agents.tool_bias import ToolBiasAuditor

    specs = []

    def ab_outcomes(w):
        a, b = _pair(w)
        return ActionBiasAnalyzer().analyze_outcomes(
            _action_records(w, a), _action_records(w, b), "approved"
        )

    specs.append(
        Spec(
            "ActionBiasAnalyzer",
            "ActionBiasAnalyzer.analyze_outcomes",
            ab_outcomes,
            lambda r: r.disparity,
            min_group_size=1,  # a mean gap needs one outcome a side; n<30 only warns
        )
    )

    def ab_delegation(w):
        a, b = _pair(w)
        return ActionBiasAnalyzer().analyze_delegation(_routes(w, a), _routes(w, b))

    specs.append(
        Spec(
            "ActionBiasAnalyzer",
            "ActionBiasAnalyzer.analyze_delegation",
            ab_delegation,
            lambda r: {t: v["rate_a"] - v["rate_b"] for t, v in r["per_target"].items()},
            min_group_size=1,
        )
    )

    def corr(w):
        a, b = _pair(w)

        def outs(g):
            return [None if not np.isfinite(v) else float(v) for v in _raw(w, g)]

        return CorrespondenceTester().analyze_outcomes(outs(a), outs(b))

    specs.append(
        Spec(
            "CorrespondenceTester",
            "CorrespondenceTester.analyze_outcomes",
            corr,
            lambda r: r.disparity_metric,
            min_group_size=1,
        )
    )

    def four_fifths(w):
        a, b = _pair(w)
        ra, rb = _decisions(w, a), _decisions(w, b)
        rate_a = float(np.mean(ra)) if len(ra) else float("nan")
        rate_b = float(np.mean(rb)) if len(rb) else float("nan")
        return CorrespondenceTester().four_fifths_rule(rate_a, rate_b)

    specs.append(
        Spec(
            "CorrespondenceTester",
            "CorrespondenceTester.four_fifths_rule",
            four_fifths,
            lambda r: r["ratio"],
            min_group_size=1,
            skip={
                "n_equals_2": (
                    "both one-row groups were selected at rate 0, so the impact ratio "
                    "is 0/0 and its refusal is correct; the counted needs are per group "
                    "and cannot say 'at least ONE group selected', which is what this "
                    "ratio needs (a per-group 'ppos' would wrongly call the "
                    "one_row_minority 0.0 a fabrication)"
                )
            },
        )
    )

    specs.append(
        Spec(
            "PipelineTracker",
            "PipelineTracker.compute_cumulative",
            lambda w: _pipeline(w).compute_cumulative(),
            lambda r: [s.bias_metrics["abs_disparity"] for s in r],
            consumes=("p", "prob"),
            min_group_size=1,
        )
    )
    specs.append(
        Spec(
            "PipelineTracker",
            "PipelineTracker.identify_bias_source",
            lambda w: _pipeline(w).identify_bias_source(),
            # A stage NAME is the verdict; 1.0 stands for "a stage was named".
            lambda r: REFUSED if r is None else 1.0,
            consumes=("p", "prob"),
            min_group_size=2,  # its ranking admits only stages a test could run on
        )
    )

    _rag_bare_skip = {
        "single_group": (
            "the bare method receives only the retrieved items, so it cannot tell a "
            "group that was never queried from a group that was queried and got "
            "nothing back; the latter is a real total disparity. full_analysis, which "
            "does receive the queries, is judged on this case"
        )
    }

    def rag_retrieval(w):
        a, b = _pair(w)
        return RAGBiasAnalyzer().analyze_retrieval(_rag_parts(w, a)[1], _rag_parts(w, b)[1])

    def rag_output(w):
        a, b = _pair(w)
        return RAGBiasAnalyzer().analyze_output(_rag_parts(w, a)[2], _rag_parts(w, b)[2])

    def rag_full(w):
        a, b = _pair(w)
        qa, da, oa = _rag_parts(w, a)
        qb, db, ob = _rag_parts(w, b)
        return RAGBiasAnalyzer().full_analysis(qa, qb, da, db, oa, ob)

    specs.append(
        Spec(
            "RAGBiasAnalyzer",
            "RAGBiasAnalyzer.analyze_retrieval",
            rag_retrieval,
            min_group_size=1,
            skip=_rag_bare_skip,
        )
    )
    specs.append(
        Spec(
            "RAGBiasAnalyzer",
            "RAGBiasAnalyzer.analyze_output",
            rag_output,
            min_group_size=1,
            skip=_rag_bare_skip,
        )
    )
    specs.append(
        Spec(
            "RAGBiasAnalyzer",
            "RAGBiasAnalyzer.full_analysis",
            rag_full,
            lambda r: [r.retrieval_disparity, r.output_disparity],
            min_group_size=1,
        )
    )

    specs.append(
        Spec(
            "TemporalTracker",
            "TemporalTracker.compute_trajectory",
            lambda w: _turn_tracker(w, TemporalTracker).compute_trajectory(),
            lambda r: [t.value for t in r],
            min_group_size=1,
        )
    )
    specs.append(
        Spec(
            "TemporalTracker",
            "TemporalTracker.detect_drift",
            lambda w: _turn_tracker(w, TemporalTracker).detect_drift(),
            lambda r: _bool_or_refused(r, REFUSED),
            min_group_size=2,  # drift is a change between at least 2 turns
        )
    )
    specs.append(
        Spec(
            "TemporalTracker",
            "TemporalTracker.detect_feedback_loop",
            lambda w: _turn_tracker(w, TemporalTracker).detect_feedback_loop(),
            lambda r: _bool_or_refused(r["has_feedback_loop"], REFUSED),
            min_group_size=3,  # its _MIN_TURNS_FOR_TREND
        )
    )
    specs.append(
        Spec(
            "TemporalTracker",
            "TemporalTracker.detect_drift_cusum",
            lambda w: _turn_tracker(w, TemporalTracker).detect_drift_cusum(),
            lambda r: _bool_or_refused(r["has_drift"], REFUSED),
            min_group_size=3,
        )
    )
    specs.append(
        Spec(
            "TemporalTracker",
            "TemporalTracker.detect_drift_ewma",
            lambda w: _turn_tracker(w, TemporalTracker).detect_drift_ewma(),
            lambda r: _bool_or_refused(r["has_drift"], REFUSED),
            min_group_size=3,
        )
    )

    def tool_calls(w):
        a, b = _pair(w)
        ta = [{"tool": t} for t in _routes(w, a)]
        tb = [{"tool": t} for t in _routes(w, b)]
        return ToolBiasAuditor().analyze_tool_calls(ta, tb)

    def tool_disparity(w):
        a, b = _pair(w)
        return ToolBiasAuditor().compute_selection_disparity(_routes(w, a), _routes(w, b))

    specs.append(
        Spec(
            "ToolBiasAuditor",
            "ToolBiasAuditor.analyze_tool_calls",
            tool_calls,
            lambda r: [t.invocation_rate_a - t.invocation_rate_b for t in r],
            min_group_size=1,
        )
    )
    specs.append(
        Spec(
            "ToolBiasAuditor",
            "ToolBiasAuditor.compute_selection_disparity",
            tool_disparity,
            lambda r: {t: v["rate_a"] - v["rate_b"] for t, v in r.items()},
            min_group_size=1,
        )
    )
    return specs


# ===========================================================================
# multi_agent/
# ===========================================================================


def _groupthink_rounds(w):
    """One agent per group label; each round is that agent's [approve, decline] mix."""
    chunks = _turn_chunks(w)
    if not chunks:
        return []
    n_rounds = min(len(c) for c in chunks.values())
    rounds = []
    for r in range(n_rounds):
        rounds.append(
            {g: [float(np.mean(c[r])), float(1.0 - np.mean(c[r]))] for g, c in chunks.items()}
        )
    return rounds


def _multi_specs(Spec, REFUSED) -> list:
    from vfairness.multi_agent.collusion import AdversarialCollusionDetector
    from vfairness.multi_agent.compositionality import CompositionalityAnalyzer
    from vfairness.multi_agent.delegation import DelegationRoutingAuditor
    from vfairness.multi_agent.emergent import EmergentBiasDetector
    from vfairness.multi_agent.groupthink import GroupthinkDetector
    from vfairness.multi_agent.negotiation import NegotiationFairnessTracker

    specs = []

    specs.append(
        Spec(
            "AdversarialCollusionDetector",
            "AdversarialCollusionDetector.analyze",
            # One agent: its score before the interaction, its decision after.
            lambda w: AdversarialCollusionDetector(n_permutations=100).analyze(
                {"agent": np.asarray(w.prob, float)}, {"agent": np.asarray(w.p, float)}, w.s
            ),
            lambda r: r.collusion_score,
            consumes=("p", "prob"),
            min_group_size=1,
        )
    )

    specs.append(
        Spec(
            "CompositionalityAnalyzer",
            "CompositionalityAnalyzer.analyze",
            lambda w: CompositionalityAnalyzer().analyze(
                {"scorer": _gap(w.prob, w.s)}, _gap(w.p, w.s)
            ),
            lambda r: REFUSED if r.scenario == "not_assessed" else r.divergence,
            consumes=("p", "prob"),
            min_group_size=1,
        )
    )

    def delegation(w):
        p = np.asarray(w.p, dtype=float)
        keep = np.isfinite(p)
        routes = ["senior_agent" if v == 1 else "junior_agent" for v in p[keep]]
        return DelegationRoutingAuditor(n_permutations=200).analyze(routes, w.s[keep].tolist())

    _one_route = (
        "every decision went to ONE route, so the routing association (chi-square, "
        "Cramer's V) has no second category and is undefined; refusing is "
        "defensible, and so would be reporting identical routing"
    )
    specs.append(
        Spec(
            "DelegationRoutingAuditor",
            "DelegationRoutingAuditor.analyze",
            delegation,
            lambda r: r.cramers_v,
            min_group_size=1,
            skip={
                "one_label_only": _one_route,
                "constant_scores": _one_route,
                "n_equals_2": _one_route,
            },
        )
    )

    specs.append(
        Spec(
            "EmergentBiasDetector",
            "EmergentBiasDetector.analyze",
            lambda w: EmergentBiasDetector().analyze(
                {"scorer": np.asarray(w.prob, float)}, np.asarray(w.p, float), w.s
            ),
            lambda r: _bool_or_refused(r.is_emergent, REFUSED),
            consumes=("p", "prob"),
            min_group_size=1,
        )
    )

    specs.append(
        Spec(
            "GroupthinkDetector",
            "GroupthinkDetector.analyze_convergence",
            lambda w: GroupthinkDetector().analyze_convergence(_groupthink_rounds(w)),
            lambda r: r.echo_chamber_score,
            min_group_size=2,  # the method's own minimum of 2 rounds
        )
    )

    def coalitions(w):
        d = GroupthinkDetector()
        labs = _labels(w)
        vecs = []
        for g in labs:
            x = _decisions(w, g)
            m = float(np.mean(x)) if len(x) else float("nan")
            vecs.append(np.array([m, 1.0 - m]))
        n = len(vecs)
        mat = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if np.all(np.isfinite(vecs[i])) and np.all(np.isfinite(vecs[j])):
                    mat[i, j] = d._cosine_similarity(vecs[i], vecs[j])
        return d.detect_coalitions(mat)

    specs.append(
        Spec(
            "GroupthinkDetector",
            "GroupthinkDetector.detect_coalitions",
            coalitions,
            # The returned partition is the verdict; its size is what is read.
            lambda r: float(len(r)) if r else REFUSED,
            min_group_size=1,
        )
    )

    def negotiation(w):
        chunks = _turn_chunks(w)
        a, b = _pair(w)
        ca = chunks.get(a, []) if a is not None else []
        cb = chunks.get(b, []) if b is not None else []
        return NegotiationFairnessTracker().analyze(
            [float(np.mean(c)) for c in ca], [float(np.mean(c)) for c in cb]
        )

    specs.append(
        Spec(
            "NegotiationFairnessTracker",
            "NegotiationFairnessTracker.analyze",
            negotiation,
            lambda r: r.mean_disparity,
            min_group_size=3,  # the method's own minimum of 3 turns
        )
    )
    return specs


def specs(Spec, REFUSED, flags_absence) -> list:
    return _agent_specs(Spec, REFUSED) + _multi_specs(Spec, REFUSED)
