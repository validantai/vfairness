"""
Individual-level counterfactuals via DoWhy's GCM module.

Answers "What would Y have been for this individual under treatment = a' instead
of a?" -- the canonical counterfactual fairness query (Kusner et al., 2017).

CAVEAT: GCM fits an additive-noise structural causal model. If the true mechanism
is multiplicative or involves heavy-tailed noise, counterfactuals will be biased
and there is no warning at runtime. Always pair with refutation results before
acting on individual counterfactuals.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..._triage import is_measured

#: What ``str()`` of an absent value mints. A factual record that reached this
#: function through a CSV, a JSON round trip or a DataFrame ``.to_dict()`` can
#: carry the WORD rather than the absence, and ``float('nan')`` of it raises
#: while ``float(x)`` of a real absence yields NaN: both used to arrive as the
#: same vague "Counterfactual raised" note.
_ABSENT_LITERALS = {"None", "nan", "NaN", "NaT", "<NA>", "NA", "null", "NULL"}


@dataclass
class CounterfactualResult:
    treatment: str
    outcome: str
    individual_id: Optional[str] = None
    factual_value: Optional[float] = None
    counterfactual_value: Optional[float] = None
    effect: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _import_gcm():
    try:
        import networkx as nx  # noqa: F401
        from dowhy import gcm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vfairness.operations.causal.counterfactual needs dowhy's gcm module "
            "(plus networkx). It needs dowhy>=0.13 (older releases call a networkx "
            "function networkx has removed), which currently supports "
            "Python <=3.13 only; on Python 3.14 the available dowhy (0.8) has a gcm "
            "that is incompatible with numpy>=1.25. Run this op on a Python 3.13 "
            "worker, or use the graph-based CausalFairnessGraph / counterfactual_fairness() "
            "metric instead, which need no gcm. "
            f"(underlying error: {exc})"
        ) from exc
    return gcm


def _read_number(
    record: Dict[str, Any], key: str, label: str
) -> Tuple[Optional[float], Optional[str]]:
    """Read one number out of the factual record, or say why there is none.

    Absence has more than one door and they all used to end in the same place.
    ``float(record.get(key, float('nan')))`` turned a MISSING key into NaN and
    published it as ``factual_value``, and ``effect`` was then NaN minus a real
    counterfactual: a number in the field a reader takes for the measurement.
    ``float(None)`` and ``float('')`` instead raised inside the blanket except
    and became "Counterfactual raised: float() argument must be ...", which
    names neither the field nor the individual.
    """
    if key not in record:
        return None, (
            f"The factual record carries no value for the {label} '{key}', so the "
            f"{label} value is not known and no effect is derived from it."
        )
    value = record[key]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, f"The {label} '{key}' is recorded as absent in the factual record."
    if value is pd.NaT or value is pd.NA:
        return None, f"The {label} '{key}' is recorded as absent ({value!r}) in the factual record."
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, f"The {label} '{key}' is blank in the factual record."
        if text in _ABSENT_LITERALS:
            return None, (
                f"The {label} '{key}' holds the literal string {value!r}, which is what "
                f"str() of an absent value prints, not a recorded value."
            )
        try:
            value = float(text)
        except ValueError:
            return None, (
                f"The {label} '{key}' holds {value!r}, which is not a number, so no "
                f"numeric {label} value could be read."
            )
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, (f"The {label} '{key}' holds a {type(value).__name__}, which is not a number.")
    if not is_measured(number):
        return None, (
            f"The {label} '{key}' is {value!r}, which is not a finite number, so it "
            f"carries no measurement."
        )
    return number, None


def compute_counterfactual(
    gml: str,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    factual: Dict[str, Any],
    intervention_value: Any,
    individual_id: Optional[str] = None,
) -> CounterfactualResult:
    """
    Fit a structural causal model from data, then ask the counterfactual
    question for a single individual.

    Parameters
    ----------
    gml : DAG in GML format.
    data : training frame used to fit functional mechanisms.
    treatment, outcome : node labels.
    factual : observed feature dict for the individual (must include all parents).
    intervention_value : the alternative treatment value.
    individual_id : optional pass-through label.

    BGL-G02 (2026-09-30). THIS FUNCTION COULD NOT PRODUCE A COUNTERFACTUAL AT
    ALL. It fitted a ``gcm.StructuralCausalModel``, and ``counterfactual_samples``
    with ``observed_data`` has to invert each mechanism to recover the
    individual's noise, which only an ``InvertibleStructuralCausalModel`` can do.
    Measured on nine inputs, healthy and degenerate alike, every one of them
    identical::

        factual_value None, counterfactual_value None, effect None
        notes: ['Counterfactual raised: Since no noise_data is given, this has to
                 be estimated from the given observed_data. This can only be done
                 with InvertibleStructuralCausalModel.']

    It refused honestly, so it never fabricated, but the capability was dead. With
    the invertible model the same call on Y = 3T (factual T=0, Y=0, intervene
    T=1) answers counterfactual_value 2.986 against a true 3.0.
    """
    gcm = _import_gcm()
    import networkx as nx

    out = CounterfactualResult(treatment=treatment, outcome=outcome, individual_id=individual_id)

    try:
        graph = nx.parse_gml(gml)
    except Exception as exc:
        out.notes.append(f"Could not parse the DAG, so nothing was fitted: {exc}")
        return out

    # ABOVE the fit: every branch below needs the two nodes to exist in the
    # graph and in the training frame, and fitting an SCM first only reaches
    # the same refusal through a library traceback.
    labels = {str(n) for n in graph.nodes()}
    for role, name in (("treatment", treatment), ("outcome", outcome)):
        if name not in labels:
            out.notes.append(
                f"The {role} '{name}' is not a node of the DAG "
                f"({sorted(labels)[:8]}), so no counterfactual was computed."
            )
    if isinstance(data, pd.DataFrame):
        for role, name in (("treatment", treatment), ("outcome", outcome)):
            if name not in data.columns:
                out.notes.append(
                    f"The {role} '{name}' is not a column of the training data, so no "
                    f"mechanism could be fitted for it and no counterfactual was computed."
                )
        if data.empty:
            out.notes.append("The training data holds no rows, so no mechanism could be fitted.")
    else:
        out.notes.append(
            f"The training data is a {type(data).__name__}, not a DataFrame, so no "
            f"mechanism could be fitted."
        )
    if not isinstance(factual, dict) or not factual:
        out.notes.append(
            "The factual record is empty, so there is no individual to ask the "
            "counterfactual question about."
        )
    if intervention_value is None or (
        isinstance(intervention_value, float) and math.isnan(intervention_value)
    ):
        out.notes.append(
            f"The intervention value is {intervention_value!r}, so there is no alternative "
            f"treatment to set '{treatment}' to and no counterfactual was computed."
        )
    if out.notes:
        return out

    # Read the factual outcome BEFORE fitting, so its absence is reported as
    # itself rather than as a float() traceback from inside the try below. An
    # unreadable factual outcome is a REFUSAL, not merely a missing effect: this
    # individual's counterfactual is the alternative-treatment prediction plus
    # the noise recovered from their observed outcome, so without that value
    # there is no individual left in the question.
    factual_value, factual_reason = _read_number(factual, outcome, "outcome")
    out.factual_value = factual_value
    if factual_reason:
        out.notes.append(factual_reason)
        out.notes.append(
            f"No counterfactual was computed: this individual's noise is recovered from "
            f"their observed '{outcome}', and no value for it could be read (see above). "
            f"Without it the answer would be the population prediction for the "
            f"alternative treatment, which is not this individual's counterfactual."
        )
        return out

    try:
        scm = gcm.InvertibleStructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(scm, data)
        gcm.fit(scm, data)

        factual_frame = pd.DataFrame([factual])
        if isinstance(factual.get(outcome), str):
            # The value WAS read, as a number, and the note says so; the frame
            # has to carry that number or the mechanism inversion subtracts a
            # string and reports a library TypeError instead.
            factual_frame[outcome] = [factual_value]
            out.notes.append(
                f"The factual '{outcome}' arrived as text ({factual[outcome]!r}) and was "
                f"read as the number {factual_value}."
            )
        missing_nodes = sorted(labels - set(factual_frame.columns))
        if missing_nodes:
            out.notes.append(
                f"The factual record carries no value for {missing_nodes}, which the "
                f"noise of each mechanism is recovered from, so no counterfactual was "
                f"computed."
            )
            return out

        cf_samples = gcm.counterfactual_samples(
            scm,
            {treatment: lambda _x: intervention_value},
            observed_data=factual_frame,
        )
        counterfactual_value = float(cf_samples[outcome].iloc[0])
        out.counterfactual_value = (
            counterfactual_value if is_measured(counterfactual_value) else None
        )
        if out.counterfactual_value is None:
            out.notes.append(
                f"The fitted model returned {counterfactual_value!r} for the counterfactual "
                f"outcome, which is not a finite number, so nothing was measured."
            )
    except AttributeError as exc:
        if "estimate_noise" in str(exc):
            # A categorical mechanism (ClassifierFCM) has no inverse, so the
            # individual's noise cannot be recovered. Reported as the modelling
            # limit it is, rather than as a bare AttributeError.
            out.notes.append(
                "At least one variable on the path is CATEGORICAL, and an additive-noise "
                "inversion does not exist for a categorical mechanism, so this individual's "
                "noise could not be recovered and no counterfactual was computed. Use the "
                "graph-based counterfactual_fairness() metric for categorical outcomes. "
                f"(underlying error: {exc})"
            )
        else:
            out.notes.append(f"Counterfactual raised: {exc}")
        return out
    except Exception as exc:
        out.notes.append(f"Counterfactual raised: {exc}")
        return out

    # is_measured on BOTH, never `is not None` ALONE: the old test was
    # `factual_value is not None and counterfactual_value is not None`, which
    # could not be false, because both fields had just been assigned floats one
    # line above. A NaN factual value therefore published a NaN effect. Both
    # halves are reachable here: either field is None when it was refused, and
    # is_measured is what rejects a non-finite one.
    cf_value = out.counterfactual_value
    factual_measured = out.factual_value
    if (
        cf_value is not None
        and factual_measured is not None
        and is_measured(cf_value)
        and is_measured(factual_measured)
    ):
        out.effect = cf_value - factual_measured
    else:
        out.notes.append(
            "No effect is reported: an effect is the counterfactual outcome minus the "
            "factual one, and one of the two was not measured (see the note above)."
        )

    return out
