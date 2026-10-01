"""Check 2 specs: ranking, vision representation, discovery, intersectional, causal.

Loaded by scripts/broken_data_check.py (``all_specs``). Each spec maps the shared
World onto the capability's REAL input, so the harness's counted expectation is
the true answer for that input:

* Ranking metrics take a score column and group labels: ``w.prob`` is the score
  that orders the items, ``w.s`` the group of each item. The order needs scores
  that differ, hence ``needs=("var_prob",)``; ``min_group_size`` is their own
  default of 5.
* The vision representation measures take a list of demographic LABELS, in the
  order the system returned them: ``w.s`` in row order. A label is one row, so
  any observed group counts (``min_group_size=1``), and with no reference the
  comparison needs two observed groups.
* Discovery and intersectional analyses take a frame and binary predictions:
  ``w.s`` becomes a protected column, ``w.p`` the decisions.
* Counterfactual fairness, decomposition and recourse take a model. The model
  here is an in-process fake whose factual score IS ``w.prob``, so a missing
  score is a missing prediction and nothing reaches the network.
* CausalFairnessGraph is authored, not estimated: its input is the structure of
  the frame's columns, which every world shares.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def _m(path: str):
    return importlib.import_module("vfairness." + path)


_SEX = {"a": "female", "b": "male"}
_REGION = {"a": "north", "b": "south"}
_BAND = {"pass": 0.0, "warn": 1.0, "critical": 2.0}


def _labels(w):
    return [str(x) for x in w.s]


def _frame(w) -> pd.DataFrame:
    """A realistic applicant frame: ``w.s`` is the protected column."""
    return pd.DataFrame(
        {
            # object dtype even at zero rows, where pandas would infer float64
            # and the column would be skipped as numeric before it was read.
            "sex": pd.Series([_SEX[str(g)] for g in w.s], dtype=object),
            "income": np.asarray(w.yr, dtype=float),
            "score": np.asarray(w.prob, dtype=float),
            "approved": np.asarray(w.y),
        }
    )


def _is_a(w) -> np.ndarray:
    return (np.asarray(w.s) == "a").astype(float)


def _counterfactual_scores(w) -> np.ndarray:
    """The fake model re-queried with the protected attribute flipped.

    The world's score carries +0.08 for group 'a'; flipping a row's group moves
    that term to the other side, so the counterfactual is a real model output
    and NaN stays NaN.
    """
    prob = np.asarray(w.prob, dtype=float)
    is_a = _is_a(w)
    return np.clip(prob - 0.08 * is_a + 0.08 * (1.0 - is_a), 0.01, 0.99)


def _recourse_inputs(w):
    X = np.column_stack([np.asarray(w.prob, dtype=float), np.asarray(w.yr, dtype=float), _is_a(w)])

    def predict(Z):
        Z = np.asarray(Z, dtype=float)
        return np.clip(Z[:, 0] + 0.02 * (Z[:, 1] - 10.0), 0.0, 1.0)

    return predict, X


def _graph(w):
    """The DAG an analyst authors over this frame's columns."""
    G = _m("operations.causal.graph").CausalFairnessGraph()
    df = w.df
    G.add_variable("group", protected=True)
    G.add_variable("income", mediator=True)
    G.add_variable("y_pred", outcome=True)
    assert {"group", "income", "y_pred"} <= set(df.columns)
    G.add_edge("group", "income")
    G.add_edge("income", "y_pred")
    G.add_edge("group", "y_pred")
    return G


def specs(Spec, REFUSED, flags_absence) -> list:
    R = _m("evaluation.vfairness_metrics.ranking")
    V = _m("vision")
    D = _m("evaluation.vfairness_metrics.discovery")
    I = _m("evaluation.vfairness_metrics.intersectional")  # noqa: E741
    CF = _m("evaluation.vfairness_metrics.counterfactual_metric")
    FD = _m("evaluation.vfairness_metrics.fairness_decomposition")
    RC = _m("evaluation.vfairness_metrics.recourse")
    out = []

    # Ranking: scores order the items, groups own them. Own default floor is 5.
    for name in (
        "exposure_parity_difference",
        "exposure_parity_ratio",
        "normalized_discounted_kl_divergence",
    ):
        out.append(
            Spec(
                name,
                name,
                lambda w, f=getattr(R, name): f(w.prob, w.s),
                needs=("var_prob",),
                consumes=("prob",),
                min_group_size=5,
            )
        )
    out.append(
        Spec(
            "attention_weighted_rank_fairness",
            "attention_weighted_rank_fairness",
            lambda w: R.attention_weighted_rank_fairness(w.prob, w.s),
            # The verdict is read as well as the number: an is_fair of True or
            # False beside a NaN value would be a verdict nobody measured.
            lambda r: [r.value] + ([float(r.is_fair)] if r.is_fair is not None else []),
            needs=("var_prob",),
            consumes=("prob",),
            min_group_size=5,
        )
    )

    # Vision representation over labels in the order returned. No reference:
    # the comparison is between the observed groups, so two must be observed.
    out.append(
        Spec(
            "skew",
            "skew",
            lambda w: V.skew(_labels(w)),
            lambda r: [r.get("maxSkew"), r.get("minSkew")],
            consumes=(),
            min_group_size=1,
        )
    )
    # With a reference the comparison is against the reference, so any
    # non-empty label set is measurable, including one group only.
    out.append(
        Spec(
            "skew",
            "skew[reference]",
            lambda w: V.skew(_labels(w), reference={"a": 0.5, "b": 0.5}),
            lambda r: [r.get("maxSkew"), r.get("minSkew")],
            consumes=(),
            grouped_min=0,
        )
    )
    out.append(
        Spec(
            "ndkl",
            "ndkl",
            lambda w: V.ndkl(_labels(w)),
            consumes=(),
            min_group_size=1,
        )
    )
    out.append(
        Spec(
            "ndkl",
            "ndkl[reference]",
            lambda w: V.ndkl(_labels(w), reference={"a": 0.5, "b": 0.5}),
            consumes=(),
            grouped_min=0,
        )
    )
    out.append(
        Spec(
            "bias_amplification",
            "bias_amplification",
            lambda w: V.bias_amplification(_labels(w), {"a": 0.5, "b": 0.5}),
            lambda r: (
                [r.get("maxAmplification")]
                + [v for v in (r.get("perGroup") or {}).values() if v is not None]
            ),
            consumes=(),
            grouped_min=0,
        )
    )
    out.append(
        Spec(
            "representation_severity",
            "representation_severity",
            lambda w: V.representation_severity(V.skew(_labels(w))),
            # The band is the headline. 'not_assessed' is its documented refusal.
            lambda r: _BAND.get(r, REFUSED),
            consumes=(),
            min_group_size=1,
        )
    )

    # Discovery.
    out.append(
        Spec(
            "detect_protected_attributes",
            "detect_protected_attributes",
            lambda w: D.detect_protected_attributes(_frame(w)),
            # The finding a reader acts on: is the protected column found.
            lambda r: [c.confidence for c in r if c.column == "sex"],
            consumes=(),
            grouped_min=0,
            skip={
                "empty": (
                    "a zero-row frame still has a column named 'sex'; detection by "
                    "name alone is a true statement about the schema, and refusing "
                    "for want of values is defensible too"
                )
            },
        )
    )

    def _discover(w):
        df = pd.DataFrame(
            {
                "sex": [_SEX[str(g)] for g in w.s],
                "region": [_REGION[str(g)] for g in w.s],
            }
        )
        # min_disparity=0 so every computed disparity produces a dict; an
        # empty list then means no combination was measured.
        return D.discover_intersectional_groups(df, ["sex", "region"], w.p, min_disparity=0.0)

    out.append(
        Spec(
            "discover_intersectional_groups",
            "discover_intersectional_groups",
            # 'region' is determined by 'sex' here, so the intersectional cells
            # ARE the w.s groups and the harness's count over w.s is the count
            # over cells.
            _discover,
            lambda r: [d["disparity"] for d in r],
            min_group_size=30,
        )
    )
    out.append(
        Spec(
            "identify_privileged_groups",
            "identify_privileged_groups",
            lambda w: I.identify_privileged_groups(w.y, w.p, w.s),
            lambda r: r["max_disparity"],
            min_group_size=30,
        )
    )
    out.append(
        Spec(
            "intersectional_disparity_analysis",
            "intersectional_disparity_analysis",
            lambda w: I.intersectional_disparity_analysis(
                w.y, w.p, w.s, single_attributes={"sex": w.s}
            ),
            lambda r: [
                r["intersectional_analysis"]["max_disparity"],
                r["comparison"]["hidden_disparity"],
            ],
            min_group_size=30,
        )
    )

    # Individual-level measurements over a model. Not group comparisons.
    out.append(
        Spec(
            "counterfactual_fairness",
            "counterfactual_fairness",
            lambda w: CF.counterfactual_fairness(w.prob, _counterfactual_scores(w)),
            lambda r: REFUSED if r.flip_rate is None else [r.flip_rate, r.mean_abs_diff],
            consumes=("prob",),
            grouped_min=0,
        )
    )
    out.append(
        Spec(
            "fairness_decomposition",
            "fairness_decomposition",
            lambda w: FD.fairness_decomposition(
                lambda X: np.asarray(X, dtype=float)[:, 0],
                np.column_stack([np.asarray(w.prob, float), np.asarray(w.yr, float)]),
                w.s,
                feature_names=["score", "income"],
            ),
            lambda r: r.total_disparity,
            consumes=("prob",),
            # No floor in its signature: a group of one row is compared.
            min_group_size=1,
        )
    )

    def _recourse(w):
        predict, X = _recourse_inputs(w)
        return RC.generate_recourse(
            predict,
            X[0],
            X,
            feature_names=["score", "income", "group"],
            immutable_features=["group"],
            n_candidates=600,
        )

    out.append(
        Spec(
            "generate_recourse",
            "generate_recourse",
            _recourse,
            # found is the verdict an applicant is given; None is could-not-check.
            lambda r: (
                REFUSED
                if r.found is None
                else [float(r.found)] + [c.proximity for c in r.counterfactuals]
            ),
            consumes=("prob",),
            grouped_min=0,
        )
    )

    # The causal graph is authored over the frame's columns, not estimated.
    _authored = {
        "empty": (
            "the DAG is authored over the column structure, which a zero-row frame "
            "still has; tracing it is a reading of the authored graph, not of rows"
        )
    }
    out.append(
        Spec(
            "CausalFairnessGraph",
            "CausalFairnessGraph.discrimination_paths",
            lambda w: _graph(w).discrimination_paths(),
            lambda r: REFUSED if getattr(r, "not_assessable", []) else float(len(r)),
            consumes=(),
            grouped_min=0,
            skip=_authored,
        )
    )
    out.append(
        Spec(
            "CausalFairnessGraph",
            "CausalFairnessGraph.has_direct_discrimination",
            lambda w: _graph(w).has_direct_discrimination(),
            lambda r: REFUSED if r is None else float(r),
            consumes=(),
            grouped_min=0,
            skip=_authored,
        )
    )
    out.append(
        Spec(
            "CausalFairnessGraph",
            "CausalFairnessGraph.summary",
            lambda w: _graph(w).summary(),
            lambda r: (
                REFUSED
                if not r.get("assessable")
                else {"info": 0.0, "medium": 1.0, "high": 2.0}.get(r["severity"], REFUSED)
            ),
            consumes=(),
            grouped_min=0,
            skip=_authored,
        )
    )
    return out
