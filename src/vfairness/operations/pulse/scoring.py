"""
Pulse model scoring (F-07)
==========================

Score an uploaded model (or a live scoring endpoint) over the audited
dataset to produce the ``prediction`` column the Pulse fairness battery
reads. This is the canonical home of the scoring ANALYSIS: feature
resolution, matrix building (including one-hot encoding of categorical
features), validation, and scored-frame assembly.

Canonical rule: ALL analysis lives in this library, and anything that
needs deployment-specific dependencies stays with the caller. Those
pieces are deserialising the model artifact, decrypting it if it is
stored encrypted, and calling a live endpoint. The caller supplies them
as a single ``build_predict`` callable, so this module never imports
host code and can be tested without any of it.

Categorical upgrade over the old consumer port
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The consumer's original matrix builder was numeric-only with median
imputation, which silently DROPPED every categorical feature. A model
audited without the features it actually uses can produce materially
distorted decisions, and the omission was invisible. New behavior:

* Categorical features are one-hot encoded (``pandas.get_dummies``,
  deterministic sorted column order, ``col=value`` names) whenever the
  model does not declare its own input columns. A model that declares
  ``feature_names_in_`` (passed by the caller as
  ``payload["expected_feature_names"]``) forbids the expansion, because
  the encoded column set could no longer match the declared contract.
* Every dropped or imputed feature is RECORDED as a degradation entry
  ``{"stage": "model_scoring", "detail": ...}`` so a distorted scoring
  basis can never read as a clean run.

Degradation contract
~~~~~~~~~~~~~~~~~~~~
``score_model_over_frame`` returns its degradations to the caller. The
consumer appends them to ``inputs["_scoring_degradations"]`` before it
calls ``run_pulse``; ``run_pulse`` merges that key into its own
degradations list IF present, so the verdict-downgrade machinery sees
scoring degradations exactly like stage collapses. (Until that merge is
wired in the orchestrator, the consumer also mirrors the details into
``inputs["scoring_notes"]`` and logs them, so nothing is silent.)

Counterfactual flip pass (G-25 remainder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module is the only place with predict_fn access, so the
counterfactual half of individual fairness runs here: for every
protected attribute the model actually takes as an input (it can only
enter the feature spec through an explicit ``feature_columns`` contract
or the model's own ``feature_names_in_``; derived candidates exclude
protected attributes), re-score the full frame ONCE with that attribute
flipped and report how many decisions change. Results travel on the
same inputs side-channel as degradations (``inputs["_counterfactual_flips"]``,
consumed by ``run_pulse`` into the individualFairness block), so an
older consumer shim keeps working unchanged. When the model never sees
a protected attribute, the pass reports that the direct pathway is
invariant by construction and defers to the proxy battery for indirect
pathways; it never fabricates a flip on an attribute the model cannot
react to.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Mirrors the consumer's historical name lists (kept in lockstep with the
# Pulse handler's prediction/label detection so scoring never treats the
# decision or the ground truth as a model input).
_DECISION_NAMES = [
    "invite_decision",
    "decision",
    "prediction",
    "predicted",
    "y_pred",
    "approved",
    "hired",
    "selected",
    "outcome",
    "result",
    "label",
    "target",
]
_LABEL_NAMES = [
    "true_qualification_score",
    "ground_truth",
    "y_true",
    "actual",
    "qualified",
    "label",
    "target",
    "outcome",
]

# One-hot guardrails: a categorical column is only encodable when it is a
# genuine low-cardinality category, not free text or an identifier.
_MAX_ONE_HOT_LEVELS = 20
_MAX_TEXT_MEAN_LEN = 64.0
_MIN_ROWS_FOR_DISTINCT_RULE = 5

# Counterfactual level-scan cap: an attribute with more observed levels
# than this is skipped honestly instead of spending one model call per
# level in triage (a 6-level scan already adds 6 batch calls).
_MAX_FLIP_LEVELS = 6


def _degrade(degradations: List[Dict[str, str]], detail: str) -> None:
    degradations.append({"stage": "model_scoring", "detail": str(detail)})


def _has_named_decision(df: pd.DataFrame, protected: List[Any]) -> bool:
    """True only when the dataset carries an EXPLICITLY-named decision column
    (one of _DECISION_NAMES). Deliberately does NOT fall back to 'any binary
    column', which misfires on a binary feature or sensitive column and would
    wrongly skip scoring the uploaded model.
    """
    pa = {str(p).strip().lower() for p in protected}
    names = set(_DECISION_NAMES)
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in names and cl not in pa:
            return True
    return False


def _candidate_columns(df: pd.DataFrame, protected: List[Any]) -> List[str]:
    """Non-protected, non-decision, non-label columns, in dataset order."""
    pa = {str(p).strip().lower() for p in protected}
    skip = pa | set(_DECISION_NAMES) | set(_LABEL_NAMES)
    return [c for c in df.columns if str(c).strip().lower() not in skip]


def _categorical_drop_reason(series: pd.Series) -> Optional[str]:
    """Reason string when a non-numeric column is NOT one-hot encodable,
    else None. Guards against free text, identifiers and empty columns.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "the column is entirely empty"
    as_str = non_null.astype(str)
    n_unique = int(as_str.nunique())
    if n_unique > _MAX_ONE_HOT_LEVELS:
        return (
            "it has {0} distinct values (more than {1}); it looks like free "
            "text or an identifier, not an encodable category".format(n_unique, _MAX_ONE_HOT_LEVELS)
        )
    if len(as_str) >= _MIN_ROWS_FOR_DISTINCT_RULE and n_unique >= len(as_str):
        return (
            "every value is distinct; it looks like an identifier or "
            "free text, not an encodable category"
        )
    mean_len = float(as_str.str.len().mean())
    if mean_len > _MAX_TEXT_MEAN_LEN:
        return "its values average {0:.0f} characters; it looks like free text".format(mean_len)
    return None


def _split_spec(
    df: pd.DataFrame,
    ordered_cols: List[Tuple[str, str]],
    allow_one_hot: bool,
    degradations: List[Dict[str, str]],
) -> List[Tuple[str, str, str]]:
    """Turn ordered (dataset_col, model_col) pairs into a matrix spec.

    Returns ordered entries ``(kind, dataset_col, model_col)`` where kind is
    'num' or 'cat'. Non-numeric columns are kept as 'cat' only when one-hot
    encoding is allowed AND the column passes the usability guard; otherwise
    they are dropped WITH a recorded degradation (never silently).
    """
    spec: List[Tuple[str, str, str]] = []
    for ds_col, m_col in ordered_cols:
        if pd.api.types.is_numeric_dtype(df[ds_col]):
            spec.append(("num", ds_col, m_col))
            continue
        if not allow_one_hot:
            _degrade(
                degradations,
                (
                    "Feature '{0}' was dropped from model scoring: the model "
                    "declares its own numeric input columns (feature_names_in_), "
                    "so categorical one-hot encoding is disabled.".format(ds_col)
                ),
            )
            continue
        reason = _categorical_drop_reason(df[ds_col])
        if reason is not None:
            _degrade(
                degradations,
                (
                    "Feature '{0}' was dropped from model scoring: {1}. The "
                    "model was scored without it, which can distort its "
                    "decisions.".format(ds_col, reason)
                ),
            )
            continue
        spec.append(("cat", ds_col, m_col))
    return spec


def _resolve_feature_spec(
    df: pd.DataFrame,
    inputs: Dict[str, Any],
    expected_names: List[str],
    protected: List[Any],
    degradations: List[Dict[str, str]],
) -> List[Tuple[str, str, str]]:
    """Resolve the ordered feature spec the model scores on.

    Priority: explicit inputs['feature_columns'] (positional contract for
    endpoints, order preserved) -> the model's declared feature_names_in_
    (case-insensitively matched to dataset columns, numeric only) -> derived
    candidates in dataset order. When the model declares its own input
    columns, one-hot encoding is forbidden everywhere (the encoded column
    set could not match the declared contract).
    """
    allow_one_hot = not expected_names

    explicit = [c for c in (inputs.get("feature_columns") or []) if c in df.columns]
    if explicit:
        return _split_spec(df, [(c, c) for c in explicit], allow_one_hot, degradations)

    if expected_names:
        case_map = {str(c).lower(): c for c in df.columns}
        pairs: List[Tuple[str, str]] = []
        for expected in expected_names:
            ds_col = case_map.get(str(expected).lower())
            if ds_col is None:
                _degrade(
                    degradations,
                    (
                        "The model expects feature '{0}', which is not in the "
                        "dataset; it was dropped from scoring.".format(expected)
                    ),
                )
                continue
            if not pd.api.types.is_numeric_dtype(df[ds_col]):
                _degrade(
                    degradations,
                    (
                        "The model expects feature '{0}', but dataset column "
                        "'{1}' is not numeric; it was dropped from scoring "
                        "(one-hot encoding is disabled because the model "
                        "declares its own input columns).".format(expected, ds_col)
                    ),
                )
                continue
            pairs.append((ds_col, str(expected)))
        if pairs:
            return [("num", ds, m) for ds, m in pairs]
        # Nothing matched: fall through to derived columns (mirrors the
        # historical consumer behavior; the model's own name validation
        # will surface a clean error if the contract truly cannot be met).

    candidates = [(c, c) for c in _candidate_columns(df, protected)]
    return _split_spec(df, candidates, allow_one_hot, degradations)


def _build_matrix(
    df: pd.DataFrame,
    spec: List[Tuple[str, str, str]],
    notes: List[str],
    degradations: List[Dict[str, str]],
    categories: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build the 2D float matrix for the ordered spec.

    Numeric columns: coerced to numeric, missing values imputed with the
    per-column median (0 for all-NaN columns) so sklearn-style models never
    see NaN. Categorical columns: one-hot encoded via pandas.get_dummies
    with deterministic (sorted) indicator order and 'col=value' names;
    missing categorical values become all-zero indicator rows. Every
    imputation is recorded as a degradation.

    ``categories`` optionally pins the indicator column set per
    categorical column (dataset column name -> level list). The
    counterfactual flip pass MUST pass the base frame's levels here: a
    swapped or level-assigned frame can carry fewer observed levels, and
    without pinning the encoded matrix would silently change width and
    column meaning under the model.

    Returns (X, encoded_model_column_names).
    """
    columns: List[np.ndarray] = []
    names: List[str] = []
    imputed: List[str] = []

    for kind, ds_col, m_col in spec:
        if kind == "num":
            s = pd.to_numeric(df[ds_col], errors="coerce")
            n_missing = int(s.isna().sum())
            if n_missing:
                imputed.append(
                    "'{0}' ({1} value(s) filled with the column median)".format(ds_col, n_missing)
                )
            median = s.median()
            fill = 0.0 if pd.isna(median) else float(median)
            columns.append(s.fillna(fill).to_numpy(dtype=float))
            names.append(str(m_col))
        else:
            raw = df[ds_col]
            n_missing = int(raw.isna().sum())
            if n_missing:
                imputed.append(
                    "'{0}' ({1} missing value(s) encoded as all-zero indicators)".format(
                        ds_col, n_missing
                    )
                )
            as_str = raw.map(lambda v: np.nan if pd.isna(v) else str(v))
            dummies = pd.get_dummies(as_str, prefix=str(ds_col), prefix_sep="=", dtype=float)
            # get_dummies already emits sorted category order; re-sort so the
            # deterministic-order contract is explicit, not incidental.
            pinned = (categories or {}).get(str(ds_col))
            if pinned is not None:
                wanted = ["{0}={1}".format(ds_col, lev) for lev in sorted(str(x) for x in pinned)]
                dummies = dummies.reindex(wanted, axis=1, fill_value=0.0)
            else:
                dummies = dummies.reindex(sorted(dummies.columns), axis=1)
            for enc_name in dummies.columns:
                columns.append(dummies[enc_name].to_numpy(dtype=float))
                names.append(str(enc_name))
            notes.append(
                "One-hot encoded categorical feature '{0}' into {1} indicator column(s).".format(
                    ds_col, int(dummies.shape[1])
                )
            )

    if imputed:
        _degrade(
            degradations,
            "Imputed missing feature values before scoring: " + "; ".join(imputed) + ".",
        )

    X = np.column_stack(columns) if columns else np.empty((len(df), 0), dtype=float)
    return X, names


def _flip_series(series: pd.Series) -> Optional[pd.Series]:
    """Swap the two observed levels of a binary column; None when the
    column does not have exactly two observed levels (a flip with three or
    more levels is not well-defined in triage and is skipped honestly).
    Handles numeric and string columns; missing values stay missing.
    """
    non_null = series.dropna()
    levels = pd.unique(non_null)
    if len(levels) != 2:
        return None
    a, b = levels[0], levels[1]
    return series.map(lambda v: v if pd.isna(v) else (b if v == a else a))


def _counterfactual_flip_pass(
    df: pd.DataFrame,
    spec: List[Tuple[str, str, str]],
    protected: List[Any],
    predict: Callable[[np.ndarray], Any],
    base_scores: np.ndarray,
    inputs: Dict[str, Any],
) -> None:
    """G-25 remainder: re-score the frame with each model-input protected
    attribute flipped and record decision-flip rates.

    Writes the result to ``inputs["_counterfactual_flips"]`` (the same
    side-channel as scoring degradations); never raises past itself: a
    collapse is recorded as a degradation entry inside the block so the
    base scoring run is untouched.
    """
    pa = {str(p).strip().lower() for p in protected}
    flip_cols = [ds for _k, ds, _m in spec if str(ds).strip().lower() in pa]
    if not flip_cols:
        inputs["_counterfactual_flips"] = {
            "available": False,
            "notApplicable": True,
            "reason": (
                "The model does not take any protected attribute as an "
                "input, so flipping one cannot change its output by "
                "construction (the direct pathway is invariant). Indirect "
                "pathways through correlated features are what the proxy "
                "battery screens."
            ),
        }
        return

    # Scale detection on the BASE scores decides how a 'changed decision'
    # is counted: binary outputs compare directly; probability-like scores
    # compare at the conventional 0.5 decision threshold; unbounded raw
    # scores get magnitude-only reporting (no invented threshold).
    finite = base_scores[np.isfinite(base_scores)]
    is_binary = finite.size > 0 and set(np.unique(finite)) <= {0.0, 1.0}
    is_prob = (
        not is_binary
        and finite.size > 0
        and float(finite.min()) >= 0.0
        and float(finite.max()) <= 1.0
    )
    scale = "binary" if is_binary else ("probability" if is_prob else "raw")

    def _decisions(scores: np.ndarray) -> Optional[np.ndarray]:
        if scale == "binary":
            return scores
        if scale == "probability":
            return (scores >= 0.5).astype(float)
        return None

    # Pin every categorical indicator set to the BASE frame's levels so a
    # swapped or level-assigned frame encodes to the exact matrix layout
    # the model was scored with (a collapsed level set would silently
    # change the column meaning under the model).
    base_categories: Dict[str, List[str]] = {
        ds: [str(v) for v in pd.unique(df[ds].dropna().astype(str))]
        for kind, ds, _m in spec
        if kind == "cat"
    }

    per_attribute: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    calls_added = 0
    for col in flip_cols:
        observed = pd.unique(df[col].dropna())
        n_levels = int(len(observed))
        flipped_series = _flip_series(df[col])
        if flipped_series is not None:
            # Two levels: one swap call, each row compared to itself with
            # the other level.
            df_flipped = df.copy()
            df_flipped[col] = flipped_series
            X_f, _ = _build_matrix(df_flipped, spec, [], [], categories=base_categories)
            flipped_scores = np.asarray(predict(X_f), dtype=float).ravel()
            calls_added += 1
            if len(flipped_scores) != len(base_scores):
                skipped.append(
                    {
                        "attribute": str(col),
                        "reason": "the model returned a mismatched score "
                        "count on the flipped frame",
                    }
                )
                continue
            deltas = flipped_scores - base_scores
            entry: Dict[str, Any] = {
                "attribute": str(col),
                "levels": [str(v) for v in observed],
                "nRows": int(len(base_scores)),
                "scoreScale": scale,
                "meanAbsDelta": float(np.mean(np.abs(deltas))),
                "maxAbsDelta": float(np.max(np.abs(deltas))),
                "modelCalls": 1,
            }
            base_dec = _decisions(base_scores)
            flip_dec = _decisions(flipped_scores)
            if base_dec is not None and flip_dec is not None:
                entry["flipRate"] = float(np.mean(flip_dec != base_dec))
                if scale == "probability":
                    entry["threshold"] = 0.5
            else:
                entry["flipRate"] = None
                entry["note"] = (
                    "Scores are not probabilities, so no decision "
                    "threshold is assumed; only the magnitude of the "
                    "score change is reported."
                )
            per_attribute.append(entry)
            continue

        # Multi-level scan (3..cap levels): score the frame once per
        # level with EVERY row assigned that level, then read per-row
        # invariance across assignments. One batch call per level; the
        # budget is disclosed. Beyond the cap the scan is skipped
        # honestly (the call count would grow past what triage should
        # spend on one attribute).
        if n_levels < 2:
            skipped.append(
                {
                    "attribute": str(col),
                    "reason": (
                        "the column has {0} observed level(s); a flip needs at least two".format(
                            n_levels
                        )
                    ),
                }
            )
            continue
        if n_levels > _MAX_FLIP_LEVELS:
            skipped.append(
                {
                    "attribute": str(col),
                    "reason": (
                        "the column has {0} observed levels; the level scan "
                        "is capped at {1} to bound model calls in triage".format(
                            n_levels, _MAX_FLIP_LEVELS
                        )
                    ),
                }
            )
            continue
        if scale == "raw":
            skipped.append(
                {
                    "attribute": str(col),
                    "reason": (
                        "the model returns unbounded raw scores; the "
                        "multi-level scan reads decision invariance and "
                        "would need an assumed threshold, so it is skipped "
                        "rather than invented"
                    ),
                }
            )
            continue
        level_dec: List[np.ndarray] = []
        ok = True
        for lev in observed:
            df_lev = df.copy()
            df_lev[col] = pd.Series([lev] * len(df), index=df.index)
            X_l, _ = _build_matrix(df_lev, spec, [], [], categories=base_categories)
            lev_scores = np.asarray(predict(X_l), dtype=float).ravel()
            calls_added += 1
            if len(lev_scores) != len(base_scores):
                skipped.append(
                    {
                        "attribute": str(col),
                        "reason": "the model returned a mismatched score "
                        "count on a level-assigned frame",
                    }
                )
                ok = False
                break
            dec = _decisions(lev_scores)
            if dec is None:
                ok = False
                break
            level_dec.append(dec)
        if not ok or len(level_dec) < 2:
            continue
        stack = np.vstack(level_dec)
        # A row flips when its decision is not identical across every
        # level assignment; the worst pair names where it concentrates.
        any_flip = (stack != stack[0]).any(axis=0)
        worst_pair, worst_rate = None, 0.0
        for i in range(len(level_dec)):
            for j in range(i + 1, len(level_dec)):
                rate = float(np.mean(stack[i] != stack[j]))
                if worst_pair is None or rate > worst_rate:
                    worst_pair = [str(observed[i]), str(observed[j])]
                    worst_rate = rate
        per_attribute.append(
            {
                "attribute": str(col),
                "levels": [str(v) for v in observed],
                "nRows": int(len(base_scores)),
                "scoreScale": scale,
                "flipRate": float(np.mean(any_flip)),
                "worstPair": worst_pair,
                "worstPairFlipRate": worst_rate,
                "modelCalls": n_levels,
                "note": (
                    "Multi-level scan: the frame was scored once per level "
                    "with every row assigned that level; flipRate is the "
                    "share of rows whose decision is not identical across "
                    "all assignments."
                ),
            }
        )

    inputs["_counterfactual_flips"] = {
        "available": bool(per_attribute),
        "perAttribute": per_attribute,
        "skipped": skipped,
        "modelCallsAdded": calls_added,
        "method": (
            "Each protected attribute the model takes as an input is "
            "counterfactually varied and the model re-scored: two-level "
            "attributes are swapped in one extra call; attributes with "
            "up to {0} levels get a per-level scan (one call per "
            "level). A changed decision on an otherwise identical row "
            "is direct evidence the decision depends on the protected "
            "attribute.".format(_MAX_FLIP_LEVELS)
        ),
    }
    if not per_attribute and skipped:
        inputs["_counterfactual_flips"]["reason"] = (
            "No protected model input was scannable: "
            + "; ".join(s["reason"] for s in skipped)
            + "."
        )


def score_model_over_frame(
    df: pd.DataFrame,
    inputs: Dict[str, Any],
    payload: Dict[str, Any],
    build_predict: Callable[[Dict[str, Any], List[str]], Tuple[Callable[[np.ndarray], Any], str]],
) -> Tuple[pd.DataFrame, Optional[List[str]], List[str], List[Dict[str, str]]]:
    """Score an uploaded model or live endpoint over ``df``.

    Args:
        df: The audited dataset.
        inputs: Pulse run inputs (protected_attributes, feature_columns,
            model_format, auth fields, ...).
        payload: Caller-resolved scoring inputs. Recognized keys:
            ``model_base64`` (resolved plaintext base64; envelope decryption
            is consumer-side), ``endpoint_url``, and
            ``expected_feature_names`` (the model's feature_names_in_ as a
            plain list; extracting it requires deserialising the model, a
            consumer dep, so the caller passes the list).
        build_predict: Injected factory, signature
            ``build_predict(scoring_payload, model_cols) -> (predict, source)``
            where ``predict(X_2d) -> 1D scores``. The consumer passes its
            ``_build_predict_from_payload`` (model loading and endpoint
            transport stay consumer-side).

    Returns:
        ``(scored_df, feature_columns, notes, degradations)``.
        feature_columns are DATASET-level column names the model scored on
        (categoricals listed by their original name, not their indicator
        expansions) so the in-Pulse XAI chapter dispatches on the same
        columns; None when there is no model/endpoint (no-op). notes are
        JSON-safe plain-language strings. degradations entries are
        ``{"stage": "model_scoring", "detail": str}``; the caller appends
        them to ``inputs["_scoring_degradations"]`` for run_pulse to merge.

    Raises:
        ValueError: no usable feature columns, or the model returned the
            wrong number of scores. Transport/model errors from
            ``build_predict`` propagate untouched.
    """
    inputs = inputs or {}
    payload = payload or {}
    notes: List[str] = []
    degradations: List[Dict[str, str]] = []

    model_b64 = payload.get("model_base64") or inputs.get("model_base64")
    endpoint = payload.get("endpoint_url") or inputs.get("endpoint_url")
    if not model_b64 and not endpoint:
        return df, None, notes, degradations

    protected = list(inputs.get("protected_attributes") or [])
    expected_names = [str(n) for n in (payload.get("expected_feature_names") or [])]

    spec = _resolve_feature_spec(df, inputs, expected_names, protected, degradations)
    if not spec:
        raise ValueError("no usable feature columns available to score the model on")
    feature_columns = [ds_col for _kind, ds_col, _m in spec]

    # If the dataset already carries a NAMED decision column, keep it (do
    # not rescore), but still surface the feature columns so the XAI
    # chapter can run on them.
    if _has_named_decision(df, protected):
        notes.append(
            "Dataset already carries a named decision column; the model was not rescored over it."
        )
        inputs["_counterfactual_flips"] = {
            "available": False,
            "notApplicable": True,
            "reason": (
                "The dataset already carries a named decision column, so "
                "the model was not re-scored; a flip test compares the "
                "model's own outputs and cannot run against stored "
                "decisions."
            ),
        }
        return df, feature_columns, notes, degradations

    X, model_cols = _build_matrix(df, spec, notes, degradations)
    if X.shape[1] == 0:
        raise ValueError("no usable feature columns available to score the model on")

    scoring_payload = {
        "model_base64": model_b64,
        "model_format": inputs.get("model_format", "joblib"),
        "endpoint_url": endpoint,
        "auth_type": inputs.get("auth_type", "none"),
        "auth_token": inputs.get("auth_token"),
        "response_mapping_key": inputs.get("response_mapping_key", "predictions"),
    }
    predict, source = build_predict(scoring_payload, model_cols)
    scores = np.asarray(predict(X), dtype=float).ravel()
    if len(scores) != len(df):
        raise ValueError("model returned {0} scores for {1} rows".format(len(scores), len(df)))

    out = df.copy()
    out["prediction"] = scores
    notes.append(
        "Scored {0} model over {1} rows on {2} feature column(s) "
        "({3} model input(s) after encoding).".format(
            source, int(len(out)), len(feature_columns), int(X.shape[1])
        )
    )
    log.info("[pulse-scoring] %s", notes[-1])

    # G-25 remainder: the counterfactual flip pass. A collapse here must
    # never lose the base scoring run, so it degrades instead of raising.
    try:
        _counterfactual_flip_pass(df, spec, protected, predict, scores, inputs)
    except Exception as exc:  # noqa: BLE001
        _degrade(
            degradations,
            "The counterfactual flip pass could not be completed: "
            "{0}. The base scoring run is unaffected.".format(exc),
        )
        inputs["_counterfactual_flips"] = {
            "available": False,
            "reason": "The counterfactual flip pass collapsed; see the degradations list.",
        }
    return out, feature_columns, notes, degradations
