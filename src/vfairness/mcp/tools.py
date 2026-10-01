"""
vfairness MCP tools (pure logic).

This module contains the tool implementations for the local-first vfairness MCP
server. It deliberately does NOT import the `mcp` SDK, so the logic can be unit
tested and reused without the optional MCP dependency installed. ``server.py``
wraps these functions with FastMCP.

Every tool is a thin wrapper over the SAME vfairness functions the platform's
Navigator and Pulse use. No fairness math is re-implemented here.
"""

from __future__ import annotations

import enum
import io
import math
import warnings
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# pandas.read_csv transparently fetches these schemes over the network (or a
# metadata endpoint). The MCP server is local-first, so a data_path using one of
# them is rejected rather than silently turned into egress / an SSRF primitive.
_REMOTE_SCHEMES = frozenset(
    {
        "http",
        "https",
        "ftp",
        "ftps",
        "sftp",
        "ssh",
        "s3",
        "s3a",
        "s3n",
        "gs",
        "gcs",
        "az",
        "abfs",
        "abfss",
        "adl",
        "hdfs",
        "webhdfs",
        "smb",
    }
)

# Serialization


# What a non-finite float becomes on the way out to an agent.
#
# NaN and INFINITY MEAN OPPOSITE THINGS IN THIS LIBRARY and both used to become
# null here, which made them indistinguishable at the only boundary an MCP caller
# can see.
#
#   NaN      no measurement was made. null is the honest mapping: JSON has no NaN,
#            and an absent value is what this is.
#   +/-inf   a MEASUREMENT, and the most serious one the ratio metrics produce.
#            explainer.py carries the carve-out in full: "risk_ratio and odds_ratio
#            return inf only for a table that WAS measured and in which one arm
#            received no positive outcomes at all", reported as "Total exclusion:
#            ... This is the strongest disparate impact reading available, not a
#            missing measurement", at severity critical.
#
# So a total exclusion arrived at an agent as null, reading exactly like a field
# nobody computed, and the caller here is a model acting on what it is given rather
# than a person who might find a null suspicious. A quoted string keeps the document
# valid JSON, survives every parser, and says what it is.
_NON_FINITE_JSON = {"inf": "Infinity", "-inf": "-Infinity"}


def _non_finite_to_json(v: float) -> Any:
    """null for NaN, a named string for an infinity. See _NON_FINITE_JSON."""
    if math.isnan(v):
        return None
    return _NON_FINITE_JSON["inf" if v > 0 else "-inf"]


def jsonify(obj: Any) -> Any:
    """Recursively convert numpy / dataclass / enum / pandas objects into
    JSON-safe Python types.

    NaN becomes null, because no measurement was made. An infinity becomes the
    string "Infinity" or "-Infinity", because it IS a measurement: see
    _NON_FINITE_JSON above for why collapsing the two was a defect.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else _non_finite_to_json(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return round(v, 6) if math.isfinite(v) else _non_finite_to_json(v)
    if isinstance(obj, np.ndarray):
        return [jsonify(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, enum.Enum):
        # Must come before the __dict__ branch: the enum metaclass is EnumType
        # on Python 3.11+ (not EnumMeta), and enum members have a __dict__ whose
        # public keys are empty, so the generic branch would serialize them to {}.
        return jsonify(obj.value)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        try:
            return jsonify(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: jsonify(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


# Input loading


def load_dataframe(*, data_path: Optional[str] = None, csv: Optional[str] = None) -> pd.DataFrame:
    """Load the dataset from a local CSV path or an inline CSV string.

    Local-first: the file is read in the caller's own process; nothing is sent
    anywhere. A ``data_path`` that is a remote URL (http(s)://, s3://, gs://,
    ftp://, ...) is rejected, because ``pandas.read_csv`` would otherwise fetch
    it and this tool is driven by an LLM/agent whose ``data_path`` must not be
    turned into network egress or an SSRF/metadata-endpoint request."""
    if csv:
        return pd.read_csv(io.StringIO(csv))
    if data_path:
        scheme = urlparse(str(data_path)).scheme.lower()
        if scheme in _REMOTE_SCHEMES:
            raise ValueError(
                f"data_path must be a local file path, not a {scheme!r} URL. "
                "The vfairness MCP server is local-first and does not fetch remote data."
            )
        return pd.read_csv(data_path)
    raise ValueError("Provide either `data_path` (a local CSV path) or `csv` (inline CSV text).")


def _require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {missing}. Available columns: {list(df.columns)}")


def _group_labels(series: pd.Series) -> np.ndarray:
    """Group labels as strings, with MISSING VALUES LEFT MISSING.

    ``series.astype(str)`` renders a row that carries NO protected attribute as
    the literal string "None" / "nan" / "<NA>", and every metric downstream can
    only read that as a real group. It also makes the library's
    ``missing_strategy`` unreachable, because by the time the array arrives
    there is nothing missing left to exclude.

    Measured 2026-09-17 on 40 A + 40 B + 40 attribute-less rows whose only real
    comparison, A against B, is 0.000: ``audit_agent`` published "Largest
    cross-group action-rate gap: 'escalate' (0.450)" for the pair (A, "None"),
    with that phantom group marked ``interpretable: true``. On the same shape
    of frame ``analyze_intersectional`` published "max subgroup disparity
    0.500" against the cell ``None_M``, and ``measure_fairness`` reported
    ``n_excluded: 0`` under ``missing_strategy='exclude'``, which is an
    affirmative claim that nothing was dropped.

    ``selection_rate_disparity_matrix`` carries the identical incident in its
    own comment (a ``fillna("missing")`` that let an attribute-less record take
    the reference-group slot); stringifying here re-created it one layer up,
    where the library's fix could not reach. Keeping the missing value as None
    lets every downstream ``missing_strategy`` fire, so those rows are excluded
    and COUNTED rather than measured.
    """
    missing = pd.isna(series).to_numpy()
    values = series.to_numpy(dtype=object, copy=True)
    return np.array(
        [None if m else str(v) for v, m in zip(values, missing)],
        dtype=object,
    )


def _numeric_labels(series: pd.Series) -> np.ndarray:
    """Return a numeric label array. Non-numeric columns (for example "yes"/"no"
    or "approve"/"deny") are encoded to integer category codes, matching the
    platform's consumer handler. Without this, FairnessAnalyzer raises
    "could not convert string to float" on string-labelled datasets.

    A MISSING label stays missing. ``pd.Categorical(...).codes`` uses -1 as its
    sentinel for "not in any category", and returning that sentinel as a label
    turned a row with NO recorded outcome into a THIRD outcome class that every
    metric then measured. Measured 2026-09-17 on 300 rows whose prediction
    column was "approve"/"deny" with 50 missing: the codes came out
    ``[-1, 0, 1]``, three distinct whole numbers made FairnessAnalyzer
    auto-detect REGRESSION, and ``measure_fairness`` answered with a regression
    battery (mae_parity_difference 0.333, r2_parity_difference 2.667,
    mean_prediction_difference 0.200) over a class nobody recorded, reporting
    ``n_excluded: 0``. In the y_true position the same -1 raised InvalidDataError
    instead, so the two columns disagreed about what a missing label means.
    NaN is the value the library's own missing_strategy is written to handle.

    The codes of the values that ARE present are untouched: a category is built
    from the non-null values either way, so a complete column encodes exactly as
    before.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy()
    codes = pd.Categorical(series).codes.astype(float)
    codes[codes < 0] = np.nan
    return codes


# A WARNING IS THE LIBRARY'S COULD-NOT-CHECK CHANNEL, AND IT DOES NOT CROSS THIS
# BOUNDARY (BGL5, 2026-09-27). The MCP client sees a dict and nothing else: no
# stderr, no warnings module, no logger. Every one of these modules already says
# out loud when a single comparison produced no measurement, and all of it was
# being dropped one function short of the caller.
#
# These are substrings of the library's own could-not-check sentences, matched
# case-insensitively. The list is a CLASSIFIER, not a filter: every captured
# warning is published either way (see _audit_disclosures), so a rewording that
# escapes this list still reaches the client as a sentence, it just stops being
# counted. That is the failure direction worth having.
_COULD_NOT_CHECK_MARKERS = (
    "unmeasured",
    "not measured",
    "not_measurable",
    "could not be tested",
    "could not be computed",
    "could not check",
    "could not be measured",
    "not tested",
    "unassessed",
    "unbenchmarked",
    "not defined",
    "may be unreliable",
    "too few",
    "insufficient",
)


def _audit_disclosures(caught: List[warnings.WarningMessage]) -> Dict[str, Any]:
    """Split captured warnings into could-not-checks and everything else.

    Both lists are returned verbatim and de-duplicated, because the counts inside
    the library's own sentences ("2 of 5 comparison(s) have an UNMEASURED effect
    size") are the measurement a reader needs, and re-deriving them here would be
    a second, weaker implementation of a fact the library already established.
    """
    could_not_check: List[str] = []
    other: List[str] = []
    for item in caught:
        text = str(item.message)
        target = (
            could_not_check
            if any(marker in text.lower() for marker in _COULD_NOT_CHECK_MARKERS)
            else other
        )
        if text not in target:
            target.append(text)
    return {"could_not_check": could_not_check, "other": other}


def _reemit(caught: List[warnings.WarningMessage]) -> None:
    """Raise the captured warnings again, at their original location.

    Capturing them must not SWALLOW them: a caller who does watch warnings keeps
    seeing exactly what the library raised. ``registry=None`` disables the
    per-module de-duplication registry, so a warning is not lost to a "once"
    filter that the capture above already consumed; the caller's filters still
    apply, so an "ignore" still ignores.
    """
    for item in caught:
        warnings.warn_explicit(
            item.message,
            item.category,
            item.filename,
            item.lineno,
            registry=None,
        )


# Tools


def measure_fairness(
    df: pd.DataFrame,
    protected_attributes: List[str],
    prediction_column: str,
    target_column: str,
    min_group_size: int = 30,
) -> Dict[str, Any]:
    """Compute the vfairness fairness-metric battery per protected attribute.

    ``min_group_size`` is the library default of 30, the same number
    ``analyze_intersectional`` exposes and the same number every metric in
    ``evaluation.vfairness_metrics.classification`` defaults to. It used to be
    HARDWIRED to 2 here, with no way to change it, on a module that ships as
    the console script ``vfairness-mcp``. Measured 2026-09-10 on a group of
    two people: this tool reported demographic_parity_difference 0.100 with 0
    warnings, while the library default on the identical input reported nan
    with 6 small-sample warnings. A two-person comparison is not a measurement,
    and an MCP client reading a clean number has no way to know it was drawn
    across one.
    """
    from vfairness import FairnessAnalyzer

    _require_columns(df, [prediction_column, target_column, *protected_attributes])
    y_true = _numeric_labels(df[target_column])
    y_pred = _numeric_labels(df[prediction_column])

    per_attribute: Dict[str, Any] = {}
    headline: List[str] = []
    for attr in protected_attributes:
        analyzer = FairnessAnalyzer(
            y_true,
            y_pred,
            _group_labels(df[attr]),
            min_group_size=min_group_size,
        )
        report = analyzer.get_report()
        per_attribute[attr] = jsonify(report)
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
        dpd = metrics.get("demographic_parity_difference")
        if dpd is not None and math.isfinite(float(dpd)):
            headline.append(f"{attr}: demographic parity difference {float(dpd):.3f}")

    return {
        "summary": (
            "Computed the vfairness metric battery for "
            f"{len(protected_attributes)} protected attribute(s). "
            + ("; ".join(headline) if headline else "")
        ),
        "protected_attributes": protected_attributes,
        "per_attribute": per_attribute,
    }


def triage_dataset(
    df: pd.DataFrame,
    protected_attributes: List[str],
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the vfairness bias audit over a dataset to find where to look.

    ``target_column`` is passed straight through as ``BiasDetector``'s
    ``outcome_column``. It is not cosmetic provenance: it decides which column
    the disparity analysis tests as an OUTCOME rather than as an incidental
    feature distribution, and it is the column the intersectional outcome
    analysis runs on. Left unset, the detector falls back to keyword
    auto-discovery, which can land on a different column than the caller's real
    target (or on none at all, which drops the intersectional outcome findings
    entirely), so the two paths are reported apart rather than blurred.
    """
    from vfairness import BiasDetector

    required = list(protected_attributes)
    if target_column:
        required.append(target_column)
    _require_columns(df, required)
    detector = BiasDetector(
        df,
        protected_attributes=protected_attributes,
        outcome_column=target_column,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = detector.full_audit()
    _reemit(caught)
    disclosures = _audit_disclosures(caught)
    serialized = jsonify(report)

    # Compact count summary across known finding buckets.
    counts: Dict[str, int] = {}
    if isinstance(serialized, dict):
        for key, value in serialized.items():
            if key.endswith("findings") and isinstance(value, list):
                counts[key] = len(value)
    total = sum(counts.values())

    # BiasAuditReport already separates "every module RAN" from "any module
    # reached a verdict", and full_audit warns when nothing was assessed. A
    # warning does not cross the MCP boundary: the dict below is the whole of
    # what the client sees. Measured 2026-09-17 on a 120-row frame with one
    # group, where the report said assessment_coverage 'none',
    # execution_coverage 'ran_but_assessed_nothing' and overall_risk_score 0.0:
    # this tool still answered "Bias audit complete: 1 finding(s)". Empty
    # finding lists and a 0.0 risk are a measurement only under 'complete'.
    assessment_coverage = (
        serialized.get("assessment_coverage") if isinstance(serialized, dict) else None
    )
    execution_coverage = (
        serialized.get("execution_coverage") if isinstance(serialized, dict) else None
    )
    unassessed = (
        sorted(a for a, ok in (serialized.get("attribute_assessed") or {}).items() if not ok)
        if isinstance(serialized, dict) and isinstance(serialized.get("attribute_assessed"), dict)
        else []
    )
    if assessment_coverage == "complete":
        headline = f"Bias audit complete: {total} finding(s) across "
        coverage_note = ""
    elif assessment_coverage == "none":
        headline = f"Bias audit ASSESSED NOTHING: {total} finding(s) across "
        coverage_note = (
            " NOT A CLEAN BILL OF HEALTH: no requested protected attribute reached a"
            f" verdict ({', '.join(unassessed) or 'none assessed'}), so the empty finding"
            " lists and the overall_risk_score in the report are an absence, not a"
            " measurement of low risk."
        )
    elif assessment_coverage == "partial":
        headline = f"Bias audit INCOMPLETE: {total} finding(s) across "
        coverage_note = (
            f" Not every protected attribute reached a verdict ({', '.join(unassessed)});"
            " nothing was measured for those, so they are unassessed rather than clean."
        )
    else:
        headline = f"Bias audit, coverage UNRECORDED: {total} finding(s) across "
        coverage_note = (
            " This report does not record whether any protected attribute was actually"
            " assessed, so the finding counts cannot be read as a measurement of low risk."
        )

    # THE SECOND AXIS: per COMPARISON, not per ATTRIBUTE. The block above answers
    # "did each requested attribute reach a verdict", and it was the whole of what
    # crossed the boundary. An attribute can reach a verdict while individual
    # comparisons inside it measure nothing, and those comparisons are DROPPED from
    # the finding lists rather than listed as unassessed, so their absence reads as
    # "tested and clean".
    #
    # Measured 2026-09-27 on 30 'a' + 30 'b' rows with real spread on f1/f2/score
    # and the caller's own target column y constant at 1 (a model that approves
    # everybody), triage_dataset(df, ['race'], 'y'):
    #   before -> summary "Bias audit complete: 4 finding(s) across 1 protected
    #             attribute(s). Outcome column: 'y', as named by the caller.",
    #             coverage {'assessment_coverage': 'complete', 'execution_coverage':
    #             'complete', 'attributes_not_assessed': [],
    #             'findings_are_a_measurement': True}, and the tokens
    #             'not_measurable', 'UNMEASURED' and 'could not' appearing ZERO
    #             times anywhere in the returned dict, while the library warned
    #             "2 of 5 comparison(s) have an UNMEASURED effect size
    #             (effect_size is NaN, effect_interpretation is not_measurable) ...
    #             This is not a small effect and must not be read as one". The two
    #             unmeasured comparisons were y and pred, one of them the outcome
    #             column the caller named and this tool echoes back.
    #   after  -> comparison_coverage 'partial', findings_are_a_measurement False,
    #             coverage.could_not_check carrying that sentence and the five
    #             others verbatim, and a summary that says so.
    could_not_check = disclosures["could_not_check"]
    comparison_coverage = "partial" if could_not_check else "complete"
    if could_not_check:
        comparison_note = (
            f" {len(could_not_check)} check(s) inside that audit reported a quantity they"
            " COULD NOT measure, so the finding lists are what was measurable and not the"
            " whole of what was asked; an unmeasured comparison is dropped from them"
            " rather than listed. See coverage.could_not_check for the library's own"
            " sentences."
        )
    else:
        comparison_note = ""

    if target_column:
        outcome_note = f" Outcome column: '{target_column}', as named by the caller."
    else:
        outcome_note = (
            " No target_column was given, so the outcome column was left to the"
            " detector's keyword auto-discovery; a real target it does not"
            " recognise is reported as a feature-distribution difference rather"
            " than an outcome disparity, and gets no intersectional outcome"
            " analysis."
        )

    return {
        "summary": (
            headline
            + f"{len(protected_attributes)} protected attribute(s)."
            + coverage_note
            + outcome_note
            + comparison_note
        ),
        "outcome_column": target_column,
        "finding_counts": counts,
        "coverage": {
            "assessment_coverage": assessment_coverage,
            "execution_coverage": execution_coverage,
            "attributes_not_assessed": unassessed,
            # Per COMPARISON. 'complete' means no module reported a quantity it
            # could not compute, which is the library's own disclosure channel; it
            # is not an enumeration of every comparison attempted.
            "comparison_coverage": comparison_coverage,
            "n_could_not_check": len(could_not_check),
            "could_not_check": could_not_check,
            "other_warnings": disclosures["other"],
            # BOTH axes. Either one short of complete makes the finding lists
            # something less than a measurement of this dataset, and the two are
            # kept separately above so neither answer is lost in the conjunction.
            "findings_are_a_measurement": (
                assessment_coverage == "complete" and not could_not_check
            ),
        },
        "report": serialized,
    }


def detect_proxies(
    df: pd.DataFrame,
    protected_attributes: List[str],
    correlation_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Find features that act as proxies for the protected attributes:
    statistical correlations plus multi-hop proxy chains."""
    from vfairness import compute_feature_correlations, find_proxy_chains

    _require_columns(df, protected_attributes)

    corr = compute_feature_correlations(df, protected_attributes)
    correlations = corr.to_dict() if hasattr(corr, "to_dict") else jsonify(corr)

    # Statistical, fully offline proxy signal: per-attribute correlation chains.
    # (The semantic / embedding-based proxy ranker is intentionally not used here
    # so the local server never reaches out to a model hub.)
    # BGL-S2b (2026-09-17). `find_proxy_chains` returns a ProxyChainResult,
    # which is a `list` SUBCLASS carrying pairs_not_computed / complete /
    # depths_not_searched. `jsonify` flattens it to a bare list, so every one
    # of those fields was dropped at this boundary and the tool answered
    # "Proxy analysis complete: 0 proxy chain(s)" with no could-not-check
    # anywhere in the payload. Measured on a 25-row frame: the underlying
    # result knew two column pairs had never been correlated (below the
    # 30-row floor) and the MCP client was told the analysis was complete.
    # `to_dict()` is the serialisation that keeps the coverage.
    proxy_chains: Dict[str, Any] = {}
    coverage: Dict[str, Any] = {}
    for attr in protected_attributes:
        try:
            result = find_proxy_chains(df, attr, correlation_threshold=correlation_threshold)
            proxy_chains[attr] = jsonify(list(result))
            payload = result.to_dict() if hasattr(result, "to_dict") else {}
            coverage[attr] = jsonify(
                {
                    "complete": payload.get("complete"),
                    "protected_attribute_present": payload.get("protected_attribute_present"),
                    "pairs_not_computed": payload.get("pairs_not_computed", []),
                    "n_pairs_not_computed": payload.get("n_pairs_not_computed", 0),
                    "depths_not_searched": payload.get("depths_not_searched", []),
                    "coverage_note": payload.get("coverage_note", ""),
                }
            )
        except Exception as exc:  # graceful degradation, never raise on one attr
            proxy_chains[attr] = {"error": str(exc)}
            coverage[attr] = {
                "complete": None,
                "coverage_note": (
                    f"COULD NOT CHECK: the proxy chain search for {attr!r} raised "
                    f"{exc}. No chain was searched for and none is reported."
                ),
            }

    n_chains = sum(len(v) for v in proxy_chains.values() if isinstance(v, list))
    # The headline says "complete" only when every attribute's search WAS
    # complete. `complete is not True` covers the None the except branch sets.
    incomplete = [a for a, c in coverage.items() if c.get("complete") is not True]
    if not protected_attributes:
        # ZERO SEARCHES IS NOT A COMPLETE SEARCH. With an empty attribute list the
        # loop above never runs, so `incomplete` cannot be non-empty and the
        # complete branch fired over nothing. Measured 2026-09-27 on a healthy
        # 200-row two-group frame, detect_proxies(df, []):
        #   before -> "Proxy analysis complete: 0 proxy chain(s) across 0 protected
        #             attribute(s)." with proxy_chain_coverage {}
        #   after  -> the sentence below, and chains_are_a_measurement False
        # The sibling triage_dataset already refuses the identical input
        # ("ASSESSED NOTHING ... none assessed"), so two surfaces in one file
        # disagreed about it.
        summary = (
            "Proxy analysis ASSESSED NOTHING: protected_attributes is empty, so zero "
            "searches ran. The 0 proxy chain(s) reported is the count of searches that "
            "happened, not evidence that no proxy chain exists. Name the protected "
            "attribute(s) to search."
        )
    elif incomplete:
        summary = (
            f"Proxy analysis INCOMPLETE: {n_chains} proxy chain(s) found across "
            f"{len(protected_attributes)} protected attribute(s), but the search did "
            f"not cover the whole frame for {', '.join(incomplete)}. An empty or short "
            f"result is not evidence that no chain exists. See proxy_chain_coverage."
        )
    else:
        summary = (
            f"Proxy analysis complete: {n_chains} proxy chain(s) across "
            f"{len(protected_attributes)} protected attribute(s)."
        )
    return {
        "summary": summary,
        "correlations": jsonify(correlations),
        "proxy_chains": proxy_chains,
        "proxy_chain_coverage": coverage,
        # Three states at the surface, so a client never has to infer "no chain
        # found" from "no search ran".
        "coverage": {
            "n_attributes_searched": len(protected_attributes),
            "attributes_not_fully_searched": incomplete,
            "chains_are_a_measurement": bool(protected_attributes) and not incomplete,
        },
    }


def analyze_intersectional(
    df: pd.DataFrame,
    protected_attributes: List[str],
    prediction_column: str,
    target_column: str,
    min_group_size: int = 30,
    outcome_polarity: str = "positive_favorable",
) -> Dict[str, Any]:
    """Intersectional disparity across combinations of protected attributes."""
    from vfairness import intersectional_disparity_analysis

    if len(protected_attributes) < 2:
        return {
            "skipped": True,
            "summary": "Intersectional analysis needs at least two protected attributes.",
        }
    _require_columns(df, [prediction_column, target_column, *protected_attributes])

    y_true = _numeric_labels(df[target_column])
    y_pred = _numeric_labels(df[prediction_column])
    # A cell is only a cell when every attribute in it was recorded: one
    # missing part used to be joined in as the literal "None", so "None_M" was
    # ranked beside "a_M" as a subgroup (see _group_labels).
    parts = [_group_labels(df[a]) for a in protected_attributes]
    any_part_missing = np.zeros(len(df), dtype=bool)
    for part in parts:
        any_part_missing |= pd.isna(part)
    sensitive = np.array(
        [None if miss else "_".join(combo) for combo, miss in zip(zip(*parts), any_part_missing)],
        dtype=object,
    )
    single: Dict[str, Any] = dict(zip(protected_attributes, parts))

    result = intersectional_disparity_analysis(
        y_true,
        y_pred,
        sensitive,
        single_attributes=single,
        min_group_size=min_group_size,
        outcome_polarity=outcome_polarity,
    )
    serialized = jsonify(result)
    inter = serialized.get("intersectional_analysis", {}) if isinstance(serialized, dict) else {}
    max_disp = inter.get("max_disparity") if isinstance(inter, dict) else None
    # jsonify renders the library's NaN as null, so a max_disparity that is not
    # a number is its "no cell was measurable" refusal arriving intact. The
    # word "complete" was printed either way. Measured 2026-09-17 on a 20-row
    # frame where every one of the six cells fell under min_group_size=30:
    # "Intersectional analysis complete." over zero measured groups, which an
    # MCP client can only read as an analysis that found no subgroup disparity.
    max_disparity: Optional[float] = (
        float(max_disp)
        if isinstance(max_disp, (int, float)) and not isinstance(max_disp, bool)
        else None
    )
    measured = max_disparity is not None
    excluded = serialized.get("excluded_groups") or [] if isinstance(serialized, dict) else []
    if max_disparity is not None:
        summary = f"Intersectional analysis complete; max subgroup disparity {max_disparity:.3f}."
    else:
        summary = (
            "Intersectional analysis NOT ASSESSED: no subgroup disparity could be measured"
            f" (0 cell(s) reached min_group_size={min_group_size}"
            + (f", {len(excluded)} excluded as too small" if excluded else "")
            + "). This is an absence, not a finding that no intersectional disparity"
            " exists. See result.excluded_groups and result.data_treatment."
        )
    return {
        "summary": summary,
        "coverage": {
            "max_disparity_measured": measured,
            "min_group_size": min_group_size,
            "n_groups_excluded_as_too_small": len(excluded),
        },
        "result": serialized,
    }


def suggest_mitigation(
    df: pd.DataFrame,
    protected_attributes: List[str],
    target_column: str,
    score_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommend bias mitigations for a dataset: a pre-processing feature
    analysis (which proxies to suppress/transform), plus, when a probability
    score column is given, post-processing threshold/calibration trade-offs."""
    from vfairness import FeatureEngineeringAnalyzer

    _require_columns(df, [target_column, *protected_attributes])
    analyzer = FeatureEngineeringAnalyzer(df, protected_attributes, target_column=target_column)
    report = jsonify(analyzer.full_analysis())
    recs = report.get("recommendations") if isinstance(report, dict) else None
    n_recs = len(recs) if isinstance(recs, list) else 0

    # FeatureEngineeringAnalyzer records which screens never produced a
    # measurement (screens_not_run / proxy_screen_complete) and warns about it.
    # The warning stops at this boundary. Measured 2026-09-17 on a 5-row frame
    # where all 10 screens were refused for sample size: this tool answered
    # "Suggested 0 pre-processing mitigation(s) from feature analysis", which
    # reads as a screen that ran and found nothing to fix.
    screens_not_run = report.get("screens_not_run") or [] if isinstance(report, dict) else []
    screen_complete = report.get("proxy_screen_complete") if isinstance(report, dict) else None
    if not protected_attributes:
        # ZERO SCREENS IS NOT A COMPLETE SCREEN. Measured 2026-09-27 on a healthy
        # 200-row two-group frame, suggest_mitigation(df, [], 'y'):
        #   before -> "Suggested 0 pre-processing mitigation(s) from feature analysis
        #             across 0 protected attribute(s)." with coverage
        #             {'feature_screen_complete': True, 'n_screens_not_run': 0,
        #             'recommendations_are_a_measurement': True}. The analyzer
        #             answers proxy_screen_complete True because none of its screens
        #             FAILED, and the machine-readable field then asserted that an
        #             empty list of mitigations IS a measurement, over nothing.
        #   after  -> the sentence below and
        #             recommendations_are_a_measurement False
        coverage_note = (
            " ASSESSED NOTHING: protected_attributes is empty, so zero feature screens"
            " ran. The empty list of mitigations is an absence, not a finding that"
            " nothing needs mitigating. Name the protected attribute(s) to screen."
        )
    elif screen_complete is True:
        coverage_note = ""
    elif screen_complete is False:
        coverage_note = (
            f" COULD NOT CHECK: {len(screens_not_run)} feature screen(s) produced no"
            " measurement, so this list is what was found in the part that was screened,"
            " not a finding that nothing else needs mitigating. See"
            " feature_analysis.screens_not_run."
        )
    else:
        coverage_note = (
            " Screen coverage is UNRECORDED for this run, so an empty or short list of"
            " mitigations cannot be read as an all-clear."
        )

    out: Dict[str, Any] = {
        "summary": (
            f"Suggested {n_recs} pre-processing mitigation(s) from feature analysis across "
            f"{len(protected_attributes)} protected attribute(s)." + coverage_note
        ),
        "coverage": {
            "feature_screen_complete": screen_complete,
            "n_screens_not_run": len(screens_not_run),
            "n_attributes_screened": len(protected_attributes),
            # A complete screen of NOTHING is not a measurement: see the
            # empty-attribute-list branch above.
            "recommendations_are_a_measurement": (
                screen_complete is True and bool(protected_attributes)
            ),
        },
        "feature_analysis": report,
    }

    # Post-processing trade-offs need calibrated probability scores; only run
    # when the caller supplies a score column.
    if score_column and score_column in df.columns:
        try:
            from vfairness import mitigation_pareto

            y_true = _numeric_labels(df[target_column])
            y_prob = df[score_column].to_numpy()
            out["postprocessing_pareto"] = {
                attr: jsonify(mitigation_pareto(y_true, y_prob, _group_labels(df[attr])))
                for attr in protected_attributes
            }
        except Exception as exc:  # graceful degradation
            out["postprocessing_pareto"] = {"skipped": str(exc)}

    return out


def explain_decision(
    df: pd.DataFrame,
    prediction_column: str,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Explain which features the model's decisions track, by association
    strength (Cramer's V / eta / correlation) between each feature and the
    prediction column.

    Note: this is model-free. It explains the OBSERVED predictions in the
    dataset, not a SHAP/LIME attribution of a specific model object (the local
    server only sees a CSV). Pair it with detect_proxies to see whether the top
    drivers are also proxies for protected attributes."""
    from vfairness import compute_feature_correlations

    _require_columns(df, [prediction_column])
    corr = compute_feature_correlations(df, [prediction_column], feature_columns=feature_columns)
    cd = corr.to_dict() if hasattr(corr, "to_dict") else jsonify(corr)

    by_prediction: Dict[str, Any] = {}
    if isinstance(cd, dict):
        by_prediction = (cd.get("correlations", {}) or {}).get(prediction_column, {}) or {}
    driver_rows: List[Dict[str, Any]] = []
    # A feature whose association is NaN was NOT measured: compute_feature_
    # correlations returns NaN below MIN_SAMPLE_SIZE usable rows, for a constant
    # column, and for anything it cannot encode. Those rows used to be dropped
    # by the filter below and appear nowhere, so "Top drivers" was a ranking of
    # the measurable subset presented as a ranking of the features. Measured
    # 2026-09-17 on a 200-row frame with one constant and one sparse column:
    # "Top drivers of 'prediction': real_driver (0.98)." and a two-entry
    # silence. A feature missing from `drivers` must be readable as "never
    # assessed" without opening the correlation matrix.
    not_measured: List[Dict[str, Any]] = []
    for f, v in by_prediction.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            driver_rows.append({"feature": f, "association": round(float(v), 6)})
        else:
            not_measured.append(
                {
                    "feature": f,
                    "association": None,
                    "reason": (
                        "the association could not be computed on this data (too few usable"
                        " rows, a constant column, or values that will not encode); it is"
                        " unassessed, not weak"
                    ),
                }
            )
    # A feature the caller NAMED that is not a column of the frame is dropped by
    # compute_feature_correlations before any statistic is attempted, so it
    # never reaches the matrix at all. Silence there reads as "not a driver".
    screened = set(cd.get("feature_names") or []) if isinstance(cd, dict) else set()
    for f in feature_columns or []:
        if f not in screened and f not in by_prediction:
            not_measured.append(
                {
                    "feature": f,
                    "association": None,
                    "reason": "not a column of this dataset, so it was never screened",
                }
            )
    drivers = sorted(
        driver_rows,
        key=lambda d: abs(d["association"]),
        reverse=True,
    )
    top = ", ".join(f"{d['feature']} ({d['association']:.2f})" for d in drivers[:3])
    unmeasured_note = (
        ""
        if not not_measured
        else (
            f" {len(not_measured)} feature(s) could NOT be assessed "
            f"({', '.join(str(d['feature']) for d in not_measured[:5])}"
            + (", ..." if len(not_measured) > 5 else "")
            + "); they are absent from the ranking because nothing was measured for them,"
            " not because they drive nothing. See features_not_measured."
        )
    )
    return {
        "summary": (
            (
                f"Top drivers of '{prediction_column}': {top}."
                if top
                else f"No feature associations could be computed for '{prediction_column}'."
            )
            + unmeasured_note
        ),
        "drivers": drivers,
        "features_not_measured": not_measured,
        "correlations": jsonify(cd),
    }


# Cap on distinct actions audited, so a free-text action column does not explode.
_MAX_AUDIT_ACTIONS = 20


def audit_agent(
    df: pd.DataFrame,
    group_column: str,
    action_column: str,
) -> Dict[str, Any]:
    """Audit an agent / LLM decision log for action-allocation fairness: for each
    action the agent takes, does the rate differ across groups? Expects a log
    where each row is one decision with a group column and a categorical action
    (or outcome / tool-choice) column."""
    from vfairness import selection_rate_disparity_matrix

    _require_columns(df, [group_column, action_column])
    groups = _group_labels(df[group_column])
    # Rows with NO recorded action say nothing about what the agent did, so
    # they are neither an action category of their own nor a silent "did not
    # take this action" in every denominator. They are handed to the metric as
    # a missing prediction, which is the path its own missing_strategy covers,
    # and counted here so the count reaches the caller rather than a warning.
    action_missing = pd.isna(df[action_column]).to_numpy()
    actions = df[action_column].astype(str).mask(action_missing)

    distinct = list(actions.value_counts().index[:_MAX_AUDIT_ACTIONS])
    dropped = int(actions.nunique() - len(distinct))
    n_no_action = int(action_missing.sum())

    per_action: Dict[str, Any] = {}
    worst_action: Optional[str] = None
    worst_gap: Optional[float] = None
    # How many groups the metric itself judged large enough to interpret. Its
    # worst pair falls back to ALL groups when fewer than two qualify, so a
    # headline drawn in that state is a comparison of two group(s) it has
    # already called uninterpretable. Measured 2026-09-17 on one row per group:
    # "Largest cross-group action-rate gap: 'escalate' (1.000)", both groups
    # tier 'invalid'. The number is kept (refusing it would throw away the only
    # evidence there is) and the state it was drawn in is said out loud.
    n_interpretable: Optional[int] = None
    for action in distinct:
        binary = np.where(action_missing, np.nan, (actions == action).to_numpy(dtype=float))
        matrix = jsonify(selection_rate_disparity_matrix(binary, groups))
        per_action[action] = matrix
        if not isinstance(matrix, dict):
            continue
        rates = matrix.get("rates") or {}
        n_interp_here = (
            sum(1 for r in rates.values() if isinstance(r, dict) and r.get("interpretable"))
            if isinstance(rates, dict)
            else None
        )
        gap = matrix.get("max_difference")
        # THE FALLBACK IS READ HERE, not re-published as max_difference.
        #
        # selection_rate_disparity_matrix stopped putting an uninterpretable pair's
        # number in max_difference on 2026-09-27, and correctly: with fewer than two
        # interpretable groups it answers NaN, sets headline_basis to
        # "no_interpretable_pair", and preserves what the number WOULD have been
        # under headline_over_all_groups. That is the right call for a metric whose
        # max_difference is read as a headline.
        #
        # This surface made the opposite call, on purpose, and its own comment above
        # and its pin's docstring both say so: the number is KEPT because refusing it
        # throws away the only evidence there is, and the state it was drawn in is
        # said out loud. Without this fallback the two decisions collided and the
        # WEAKER statement won. Measured on one row per group, after the upstream
        # change and before this line:
        #   summary "Audited 2 agent action(s) ... NO cross-group gap could be
        #            computed: fewer than two groups were comparable"
        #   coverage interpretable_groups null, gap_is_interpretable null
        # so a 1.000 spread between two groups the metric calls unreliable vanished
        # from the reader's view entirely, and "no gap could be computed" is a
        # different and softer claim than "here is the gap, and it means nothing".
        #
        # max_difference in per_action stays None, because that field belongs to the
        # metric and the metric has refused it. The number travels in the summary
        # and under headline_over_all_groups, where its name says what it is.
        if gap is None and matrix.get("headline_basis") == "no_interpretable_pair":
            fallback = matrix.get("headline_over_all_groups") or {}
            if isinstance(fallback, dict):
                gap = fallback.get("max_difference")
        if isinstance(gap, (int, float)) and not isinstance(gap, bool) and gap != gap:
            # NaN survived jsonify as a float rather than becoming None. Not a gap.
            gap = None
        if isinstance(gap, (int, float)) and (worst_gap is None or gap > worst_gap):
            # Recorded for THIS action, not for whichever action happened to be
            # audited last: the headline must carry its own action's coverage.
            worst_action, worst_gap, n_interpretable = action, float(gap), n_interp_here

    summary = f"Audited {len(per_action)} agent action(s) across groups in '{group_column}'."
    if worst_action is not None:
        # worst_action and worst_gap are only ever assigned together (above), so
        # a non-None worst_action guarantees a non-None numeric worst_gap.
        assert worst_gap is not None
        summary += (
            f" Largest cross-group action-rate gap: '{worst_action}' ({float(worst_gap):.3f})."
        )
        if n_interpretable is not None and n_interpretable < 2:
            summary += (
                f" NOT INTERPRETABLE: only {n_interpretable} group(s) in '{group_column}' hold"
                " enough rows to interpret, so that gap is the spread between groups the"
                " metric itself rates unreliable, not a measured disparity."
            )
    else:
        summary += (
            " NO cross-group gap could be computed: fewer than two groups were comparable,"
            " so the absence of a gap here is not a finding that the agent treats groups"
            " alike. See per_action[*].rates for what was and was not measured."
        )
    if dropped > 0:
        summary += f" ({dropped} rarer action(s) not shown.)"
    if n_no_action > 0:
        summary += (
            f" ({n_no_action} row(s) carry no recorded action and were excluded from every"
            " action rate and every denominator.)"
        )

    return {
        "summary": summary,
        "group_column": group_column,
        "action_column": action_column,
        "per_action": per_action,
        # Three states at the surface, so a caller never has to infer the
        # difference between "no gap" and "no comparison" from a missing key.
        "coverage": {
            "gap_measured": worst_action is not None,
            "interpretable_groups": n_interpretable,
            "gap_is_interpretable": (
                None if worst_action is None or n_interpretable is None else n_interpretable >= 2
            ),
            "rows_with_no_recorded_action": n_no_action,
            "rarer_actions_not_audited": dropped,
        },
    }


# Reference content (exposed as MCP resources)

GLOSSARY: Dict[str, str] = {
    "demographic_parity": "Equal selection (positive-prediction) rates across groups. The 4/5ths (80%) rule flags adverse impact when one group's rate is below 80% of another's.",
    "equalized_odds": "Equal true-positive AND false-positive rates across groups. The model makes the same kinds of errors for everyone.",
    "equal_opportunity": "Equal true-positive rates across groups: qualified individuals are equally likely to be selected.",
    "predictive_parity": "Equal precision (positive predictive value) across groups: a positive prediction means the same thing for everyone.",
    "calibration": "Predicted probabilities mean the same thing across groups (a 0.7 score implies a 70% rate for every group).",
    "proxy_variable": "A feature that correlates with a protected attribute and lets a model discriminate indirectly even when the protected attribute is excluded (for example ZIP code as a proxy for race).",
    "intersectional_fairness": "Fairness assessed across combinations of attributes (for example race x gender), which can reveal disparities that single-attribute analysis hides (fairness gerrymandering).",
}

JURISDICTIONS: Dict[str, str] = {
    "us_disparate_impact": "US (Title VII / EEOC): the 4/5ths rule treats a selection-rate ratio below 0.8 between groups as evidence of adverse impact, shifting the burden to justify the practice.",
    "eu_ai_act": "EU AI Act: high-risk AI systems require risk management, data governance addressing bias, technical documentation, and human oversight (Annex IV / Articles 9-15).",
    "general": "These framings are informational, not legal advice. Statistical disparity is necessary but not sufficient to establish unlawful discrimination; interpretation depends on context and jurisdiction.",
}


def glossary_markdown() -> str:
    lines = ["# vfairness fairness glossary", ""]
    for term, definition in GLOSSARY.items():
        lines.append(f"- **{term.replace('_', ' ')}**: {definition}")
    return "\n".join(lines)


def jurisdictions_markdown() -> str:
    lines = ["# Legal framing (informational, not legal advice)", ""]
    for key, text in JURISDICTIONS.items():
        lines.append(f"- **{key.replace('_', ' ')}**: {text}")
    return "\n".join(lines)
