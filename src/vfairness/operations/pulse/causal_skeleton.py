"""Build a causal skeleton from a raw dataset for the Pulse causal view.

Produces an OVERVIEW graph (not every edge): protected attribute(s) ->
suspected proxy mediators -> outcome, classifying each path as

  * direct      : protected -> outcome
  * indirect    : protected -> proxy -> outcome (the fairness-mediation case)

No 'confounding' path class is produced here: distinguishing a confounder
from a mediator needs causal structure this observational overview does
not have. Confounder-style reasoning lives in the DoWhy identifiability
enrichment (``identified``) and in the domain-template DAG below.

Pure pandas/numpy so the overview ALWAYS renders. If `dowhy`+`networkx`
are available the GML is additionally handed to
``vfairness.operations.causal.identify.identify_paths`` for a formal
identifiability verdict; otherwise we fall back to the structural reading.

Strength uses Cramer's V (categorical) / |corr| (numeric), the same family
the proxy detector uses. Edges below a small floor are dropped so the graph
stays an overview, per the product spec ("not all of them listed").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROXY_MIN = 0.20  # Cramer's V / |r| at/above => meaningful mediator
_DIRECT_MIN = 0.10  # protected-to-outcome association floor for showing
# a DIRECT path edge (deliberately lower than
# _PROXY_MIN so a weak but real direct signal still
# renders in the overview)
_MAX_PROXIES = 4  # overview, not every column
#: Fewer overlapping non-null rows than this and no association can be
#: measured at all. Named rather than inline so the refusal is greppable.
_MIN_OVERLAP = 10


def _assoc(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Association strength in [0,1], or None when it could not be measured.

    Cramer's V, or |corr| if both numeric.

    READINESS-6, 2026-09-10. Every one of the five refusals below returned
    ``0.0``, and 0.0 in this function means "measured, and these two are
    UNRELATED". That is a finding, and it is the finding that decides whether a
    proxy edge is drawn at all:

        if _assoc(df[p], df[f]) >= _PROXY_MIN:   # draw the edge

    So a candidate proxy that could not be assessed, because it is constant, or
    all-null, or has fewer than ten overlapping rows, was silently absent from
    the causal graph, indistinguishable from one that was measured and found
    unrelated. Measured on this repo before the change: a constant column, an
    all-null column and an 8-row overlap all returned exactly 0.0, while two
    genuinely independent normals returned 0.0649. The unmeasurable cases scored
    BETTER than the honest one.

    The overview sentence made it worse. With no proxies found it read "No
    strong proxy mediators detected; any disparity is a more direct effect of
    the protected attribute(s)", which asserts a positive conclusion about where
    a disparity comes from. Over columns that were never assessed, that sentence
    is built out of nothing.

    None is the third state. Callers must exclude it from edge decisions AND
    count it, so the overview can say what it did not look at.
    """
    a2, b2 = a.dropna(), b.dropna()
    idx = a2.index.intersection(b2.index)
    if len(idx) < _MIN_OVERLAP:
        return None
    a2, b2 = a2.loc[idx], b2.loc[idx]
    if pd.api.types.is_numeric_dtype(a2) and pd.api.types.is_numeric_dtype(b2):
        if a2.nunique() < 2 or b2.nunique() < 2:
            # A constant column has no variance, so correlation is undefined.
            # It is NOT a measured absence of association.
            return None
        r = np.corrcoef(a2.astype(float), b2.astype(float))[0, 1]
        return float(abs(r)) if np.isfinite(r) else None
    ct = pd.crosstab(a2.astype("string"), b2.astype("string"))
    if ct.size == 0 or min(ct.shape) < 2:
        return None
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    if n == 0:
        return None
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / np.where(exp == 0, np.nan, exp))
    v = math.sqrt((chi2 / n) / max(1, min(ct.shape) - 1))
    return float(min(1.0, v))


@dataclass
class CausalSkeleton:
    nodes: List[Dict[str, str]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Dict[str, str]] = field(default_factory=list)
    gml: str = ""
    overview: str = ""
    identified: Optional[List[Dict[str, Any]]] = None  # DoWhy verdicts if avail
    #: Candidate features whose association could NOT be measured (constant,
    #: all-null, or fewer than `_MIN_OVERLAP` overlapping rows). They are absent
    #: from the graph because nothing was measured about them, which is a
    #: different claim from being absent because they were measured and found
    #: unrelated. READINESS-6, 2026-09-10.
    unassessed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "paths": self.paths,
            "overview": self.overview,
            "identified": self.identified,
            "unassessed": self.unassessed,
        }


def build_causal_skeleton(
    df: pd.DataFrame,
    protected: List[str],
    outcome: str,
    candidate_features: Optional[List[str]] = None,
) -> CausalSkeleton:
    """Construct the overview causal graph + path classification."""
    sk = CausalSkeleton()
    prot = [p for p in protected if p in df.columns]
    if not prot or outcome not in df.columns:
        sk.overview = "Not enough structure to draw a causal overview."
        return sk

    # Identifiers (ids, names, raw DOB, near-unique columns) are never causal
    # mediators -- reuse the Phase-1 canonical identifier logic so the causal
    # view and the protected-attribute prep agree.
    from vfairness.preprocessing.protected_binning import (
        _MAX_GROUP_CARDINALITY,
        _is_dob,
        _is_identifier_name,
    )

    def _is_idish(c: str) -> bool:
        if _is_identifier_name(c) or _is_dob(c):
            return True
        nun = df[c].nunique(dropna=True)
        return nun > _MAX_GROUP_CARDINALITY and nun / max(1, len(df)) >= 0.5

    feats = candidate_features or [
        c for c in df.columns if c not in set(prot) | {outcome} and not _is_idish(c)
    ]
    y = df[outcome]

    for p in prot:
        sk.nodes.append({"id": p, "kind": "protected", "label": p})
    sk.nodes.append({"id": outcome, "kind": "outcome", "label": outcome})

    proxies: List[Tuple[str, float, float]] = []
    for f in feats:
        # Association with ANY protected attr and with the outcome.
        #
        # READINESS-6: `max(..., default=0.0)` over values that could be None.
        # An unmeasurable pair used to score 0.0 and be silently dropped, so a
        # candidate proxy nobody could assess looked exactly like one that was
        # assessed and cleared. Measure over what was MEASURED, and record the
        # rest instead of discarding it.
        prot_values = [v for v in (_assoc(df[p], df[f]) for p in prot) if v is not None]
        a_out = _assoc(df[f], y)
        if not prot_values or a_out is None:
            sk.unassessed.append(f)
            continue
        a_prot = max(prot_values)
        if a_prot >= _PROXY_MIN and a_out >= _PROXY_MIN:
            proxies.append((f, a_prot, a_out))
    proxies.sort(key=lambda t: t[1] * t[2], reverse=True)
    proxies = proxies[:_MAX_PROXIES]

    for f, ap, ao in proxies:
        sk.nodes.append({"id": f, "kind": "proxy", "label": f})
        for p in prot:
            strength = _assoc(df[p], df[f])
            if strength is not None and strength >= _PROXY_MIN:
                sk.edges.append(
                    {
                        "source": p,
                        "target": f,
                        "kind": "indirect",
                        "strength": round(strength, 2),
                    }
                )
        sk.edges.append(
            {"source": f, "target": outcome, "kind": "indirect", "strength": round(ao, 2)}
        )

    # Direct path: protected still associates with outcome on its own.
    for p in prot:
        d = _assoc(df[p], y)
        if d is not None and d >= _DIRECT_MIN:
            sk.edges.append(
                {"source": p, "target": outcome, "kind": "direct", "strength": round(d, 2)}
            )
            sk.paths.append(
                {
                    "path": f"{p} -> {outcome}",
                    "kind": "direct",
                    "reading": f'"{p}" still relates to the outcome even setting '
                    "proxies aside: a direct disparity signal.",
                }
            )
    for f, ap, ao in proxies:
        # `max(key=...)` over a key that can be None raises on comparison. Every
        # feature in `proxies` has at least one measured protected association by
        # construction, so rank on those and never on a stand-in value.
        src = max(prot, key=lambda p: _assoc(df[p], df[f]) or -1.0)
        sk.paths.append(
            {
                "path": f"{src} -> {f} -> {outcome}",
                "kind": "indirect",
                "reading": f'"{f}" carries "{src}" into the outcome; removing '
                f'"{src}" alone would not remove this path.',
            }
        )

    if proxies:
        sk.overview = (
            f"{len(proxies)} proxy path(s) found: the outcome is reachable "
            f"from {', '.join(p for p in prot)} through "
            f"{', '.join(f for f, _, _ in proxies)}. "
            "Indirect (proxy) paths are why dropping a protected column does "
            "not make a model fair."
        )
        if sk.unassessed:
            sk.overview += (
                f" {len(sk.unassessed)} further candidate feature(s) could not be "
                f"assessed and may carry paths this overview does not show: "
                f"{', '.join(sk.unassessed[:_MAX_PROXIES])}"
                f"{', ...' if len(sk.unassessed) > _MAX_PROXIES else ''}."
            )
    elif sk.unassessed and not proxies and len(sk.unassessed) == len(feats):
        # READINESS-6, 2026-09-10. NOT the sentence below. Every candidate
        # feature was unmeasurable, so "no strong proxy mediators detected"
        # would report a search that never happened, and the clause after the
        # semicolon asserts positively where a disparity COMES FROM. That is a
        # conclusion drawn from zero measurements, on the surface a reader
        # trusts most.
        sk.overview = (
            f"No proxy path could be assessed: association could not be measured "
            f"for any of the {len(sk.unassessed)} candidate feature(s) "
            f"({', '.join(sk.unassessed[:_MAX_PROXIES])}"
            f"{', ...' if len(sk.unassessed) > _MAX_PROXIES else ''}), because "
            f"each is constant, entirely missing, or overlaps the outcome on "
            f"fewer than {_MIN_OVERLAP} rows. This is NOT a finding that no "
            f"proxy exists, and it says nothing about where a disparity comes "
            f"from."
        )
    else:
        sk.overview = (
            "No strong proxy mediators detected; any disparity is a more "
            "direct effect of the protected attribute(s)."
        )
        if sk.unassessed:
            sk.overview += (
                f" {len(sk.unassessed)} of {len(feats)} candidate feature(s) could "
                f"not be assessed at all and are excluded from that reading: "
                f"{', '.join(sk.unassessed[:_MAX_PROXIES])}"
                f"{', ...' if len(sk.unassessed) > _MAX_PROXIES else ''}."
            )

    # GML for the formal identifier (optional enrichment).
    lines = ["graph [", "  directed 1"]
    ids = {n["id"]: i for i, n in enumerate(sk.nodes)}
    for nid, i in ids.items():
        lines.append(f'  node [ id {i} label "{nid}" ]')
    for e in sk.edges:
        if e["source"] in ids and e["target"] in ids:
            lines.append(f"  edge [ source {ids[e['source']]} target {ids[e['target']]} ]")
    lines.append("]")
    sk.gml = "\n".join(lines)

    try:
        from vfairness.operations.causal.identify import identify_paths

        res = identify_paths(sk.gml, treatments=prot, outcomes=[outcome])
        sk.identified = res.to_dict().get("paths") if hasattr(res, "to_dict") else None
    except Exception:
        sk.identified = None  # DoWhy/networkx absent -> structural reading only

    return sk


# DOMAIN-TEMPLATE DAG (read-only, defensible)
#
# Causal discovery from observational data is unreliable (Spirtes et al.
# 2000). The spec asks for a TEMPLATE that is instantiated with the user's
# actual columns -- something to edit (in the Navigator), not argue with.
# Edges are TYPED: legitimate (job-relevant -> decision), direct (protected
# -> decision, problematic), proxy (protected -> proxy -> decision),
# indirect (protected -> legitimate mediator -> decision). Nodes are placed
# from the column-role typology (E2) -- one source of truth.

_DAG_DOMAINS = {
    "hiring": {"qual": "qualifications & skills", "decision": "hiring decision"},
    "lending": {"qual": "creditworthiness signals", "decision": "credit decision"},
    "healthcare": {"qual": "clinical need", "decision": "care / risk decision"},
    "justice": {"qual": "case facts", "decision": "risk decision"},
    "education": {"qual": "academic record", "decision": "admission decision"},
    "insurance": {"qual": "actuarial risk", "decision": "underwriting decision"},
}
_DAG_ALIASES = {
    "employment": "hiring",
    "recruitment": "hiring",
    "recruiting": "hiring",
    "credit": "lending",
    "loan": "lending",
    "finance": "lending",
    "health": "healthcare",
    "medical": "healthcare",
    "recidivism": "justice",
    "criminal": "justice",
    "admissions": "education",
}


def domain_dag_template(
    domain: str,
    schema_roles: Dict[str, Any],
) -> Dict[str, Any]:
    """Template-instantiated, typed, read-only DAG for the domain.

    Args:
        domain: domain tag (hiring / lending / ...)
        schema_roles: the dict from classify_column_roles (E2). Uses its
            buckets: protected, proxy_candidates, job_relevant, model_output.

    Returns {available, domain, nodes, edges, legend, note}. nodes carry
    `kind` (protected|proxy|legitimate|decision); edges carry typed `kind`
    (legitimate|direct|proxy|indirect) + a plain `reading`. Never raises.
    """
    try:
        key = str(domain or "").strip().lower()
        key = _DAG_ALIASES.get(key, key)
        meta = _DAG_DOMAINS.get(key)
        if not meta or not isinstance(schema_roles, dict):
            return {"available": False, "reason": "No domain template for this domain."}
        protected = list(schema_roles.get("protected") or [])[:6]
        proxies = list(schema_roles.get("proxy_candidates") or [])[:6]
        legit = list(schema_roles.get("job_relevant") or [])[:6]
        outs = list(schema_roles.get("model_output") or [])
        decision = outs[0] if outs else meta["decision"]
        if not protected and not legit:
            return {
                "available": False,
                "reason": "Not enough classified columns to instantiate the domain template.",
            }
        nodes, edges = [], []
        nodes.append({"id": decision, "kind": "decision", "label": decision})
        for p in protected:
            nodes.append({"id": p, "kind": "protected", "label": p})
            edges.append(
                {
                    "source": p,
                    "target": decision,
                    "kind": "direct",
                    "reading": f"Direct discrimination path: {p} "
                    f"influencing the {meta['decision']} on its own "
                    f"is not legally defensible.",
                }
            )
        for f in legit:
            nodes.append({"id": f, "kind": "legitimate", "label": f})
            edges.append(
                {
                    "source": f,
                    "target": decision,
                    "kind": "legitimate",
                    "reading": f"Legitimate: {f} is a "
                    f"{meta['qual']} factor and may justifiably drive "
                    f"the decision.",
                }
            )
        for x in proxies:
            nodes.append({"id": x, "kind": "proxy", "label": x})
            edges.append(
                {
                    "source": x,
                    "target": decision,
                    "kind": "proxy",
                    "reading": f"Proxy path: {x} can stand in for a "
                    f"protected attribute, so it carries discrimination "
                    f"indirectly even if the protected column is "
                    f"removed.",
                }
            )
            if protected:
                edges.append(
                    {
                        "source": protected[0],
                        "target": x,
                        "kind": "indirect",
                        "reading": f"{protected[0]} is encoded in {x}.",
                    }
                )
        return {
            "available": True,
            "domain": key,
            "nodes": nodes,
            "edges": edges,
            "legend": {
                "legitimate": "job-relevant -> decision (acceptable)",
                "direct": "protected -> decision (problematic)",
                "proxy": "protected -> stand-in -> decision (hidden)",
                "indirect": "protected -> legitimate mediator -> decision",
            },
            "note": (
                "Template instantiated with your columns, not learned "
                "from the data (observational causal discovery is "
                "unreliable). Edit it in the full assessment."
            ),
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"DAG template error: {e}"}
