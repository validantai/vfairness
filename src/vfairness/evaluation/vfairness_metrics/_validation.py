"""
Shared validation utilities for vfairness.

This module provides centralized input validation, coercion, and missing value
handling for all fairness metric computations.
"""

import logging
import warnings
from typing import Any, Dict, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ...exceptions import ConfigurationError, InvalidDataError

ArrayLike = Union[np.ndarray, pd.Series, list[Any]]
MissingStrategy = Literal["exclude", "as_group", "error"]

# FAIL CLOSED. An unrecognised option used to fall through to the permissive
# branch: missing_strategy='raise' (the obvious typo for 'error') ran a
# fail-OPEN audit that silently dropped every row with a missing protected
# attribute, and a typo'd task_type skipped the binary-label validation
# entirely. docs/API_STABILITY.md names both as frozen constructor surface and
# promises ConfigurationError for an unknown option. Do NOT reintroduce a
# default fall-through here; an unknown option must refuse to run.
VALID_MISSING_STRATEGIES = ("exclude", "as_group", "error")

# The one label this library gives to rows whose sensitive attribute is absent
# under missing_strategy='as_group'. Named here rather than written twice,
# because handle_missing_values (1D) and validate_inputs (intersectional) have
# to agree: two spellings would mean the same rows land in differently named
# groups depending on whether one attribute was supplied or two, and the whole
# point of an explicit label is that it is never a str()-minted 'nan'.
MISSING_GROUP_LABEL = "__missing__"
VALID_TASK_TYPES = ("classification", "regression")

# Refusals that deserve a follow-up sentence, because the caller had a good
# reason to believe the value was supported. 'ranking' is the live example: the
# published API reference listed it as a task_type while the runtime refused it
# by name (audit 2026-08-28). Ranking fairness IS implemented, just not as an
# analyzer task type, so the refusal points at what to call instead of only
# saying no. A hint NEVER widens VALID_TASK_TYPES; it is message text only.
TASK_TYPE_HINTS = {
    "ranking": (
        "Ranking fairness is not a FairnessAnalyzer task type: the analyzer "
        "compares y_true/y_pred per group, while ranking metrics compare "
        "exposure across a ranked list. Use the standalone functions instead, "
        "which take (rankings, groups): exposure_parity_difference, "
        "exposure_parity_ratio, attention_weighted_rank_fairness, "
        "normalized_discounted_kl_divergence."
    ),
}


logger = logging.getLogger(__name__)


def _check_option(
    value: Any,
    name: str,
    allowed: Tuple[str, ...],
    hints: Optional[Dict[str, str]] = None,
) -> None:
    """Refuse an unrecognised option instead of falling back to a default.

    Args:
        value: The option value supplied by the caller.
        name: The keyword-argument name, used in the error message.
        allowed: The exact accepted values.
        hints: Optional per-value follow-up sentence, appended to the error for
            a refused value the caller plausibly believed was supported. Hints
            only add text; the accepted set is ``allowed`` and nothing else.

    Raises:
        ConfigurationError: If ``value`` is not one of ``allowed``.
    """
    if value not in allowed:
        accepted = ", ".join(repr(option) for option in allowed)
        message = (
            f"Unknown {name}: {value!r}. Accepted values are {accepted}. "
            "Matching is exact and case-sensitive."
        )
        # Only a hashable, exactly-spelled value can carry a hint; an unhashable
        # one (a list, say) must still reach the refusal above, not a TypeError.
        hint = hints.get(value) if hints and isinstance(value, str) else None
        if hint:
            message = f"{message} {hint}"
        raise ConfigurationError(message)


def coerce_to_array(data: ArrayLike, name: str = "input") -> np.ndarray:
    """
    Convert input to numpy array with validation.

    Args:
        data: Input data (array, Series, or list)
        name: Name for error messages

    Returns:
        np.ndarray: Converted array

    Raises:
        TypeError: If input cannot be converted to array
    """
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pd.Series):
        # Handle Arrow-backed arrays (pandas 2.0+) and regular arrays
        return np.asarray(data)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return np.asarray(data.iloc[:, 0])
        raise TypeError(
            f"{name} is a DataFrame with {data.shape[1]} columns. "
            "Use a Series or specify a single column."
        )
    if isinstance(data, (list, tuple)):
        return np.array(data)
    # Try to convert anything else with np.asarray
    try:
        return np.asarray(data)
    except (ValueError, TypeError):
        raise TypeError(f"{name} must be array-like, got {type(data).__name__}")


def check_consistent_length(*arrays: ArrayLike) -> int:
    """
    Verify all arrays have the same length.

    Args:
        *arrays: Variable number of arrays to check

    Returns:
        int: Common length of all arrays

    Raises:
        ValueError: If arrays have inconsistent lengths
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    if len(set(lengths)) > 1:
        raise InvalidDataError(f"Inconsistent array lengths: {lengths}")
    return lengths[0] if lengths else 0


def validate_binary_labels(y: np.ndarray, name: str = "y") -> None:
    """
    Validate that labels are binary (0/1).

    Args:
        y: Array of labels
        name: Name for error messages

    Raises:
        ValueError: If labels are not binary
    """
    unique = np.unique(y[~np.isnan(y)] if np.issubdtype(y.dtype, np.floating) else y)
    if not set(unique).issubset({0, 1}):
        raise InvalidDataError(f"{name} must contain only 0 and 1, got unique values: {unique}")


def validate_probabilities(probs: np.ndarray, name: str = "probabilities") -> None:
    """
    Validate that probabilities are in [0, 1].

    Args:
        probs: Array of probabilities
        name: Name for error messages

    Raises:
        InvalidDataError: If any probability is missing (NaN, None or pd.NA) in
            ANY dtype, or if the values are outside [0, 1].
    """
    probs = np.asarray(probs)
    # NaN comparisons are always False, so NaN silently passed the range
    # check below and poisoned downstream calibration/threshold math.
    #
    # THE GUARD USED TO BE GATED ON THE DTYPE: `np.issubdtype(probs.dtype,
    # np.floating) and np.any(np.isnan(probs))`, so only a float array was ever
    # searched, and an object-dtype column of probabilities walked straight past
    # it into the range check that cannot see NaN. Measured 2026-09-27:
    #
    #   np.array([0.5, nan, 0.7])               -> InvalidDataError (refused)
    #   np.array([0.5, nan, 0.7], dtype=object) -> None (ACCEPTED, warnings [])
    #   np.array([0.5, None, 0.7], dtype=object)-> TypeError from the range check
    #                                              ("'<' not supported between
    #                                              NoneType and int")
    #
    # and at the public consumer, 200 rows of which 20 had no probability:
    # expected_calibration_error(y_true, probs.astype(object)) RETURNED a result
    # with bin counts [22, 20, 18, 22, 13, 14, 20, 20, 10, 41], the 20 unscored
    # rows swept into the last bin and an accuracy of 0.4878 reported over them,
    # with no UserWarning. Measured after: every dtype above raises
    # InvalidDataError naming the count, and that calibration call refuses.
    # pd.isna is the dtype-agnostic test (it catches float nan, None and pd.NA
    # alike) and returns all-False for an int array, so nothing else changes:
    # np.array([0, 1, 1]) and np.array([0.5, 0.6, 0.7], dtype=object) are still
    # accepted, and np.array([]) is still accepted.
    missing = np.asarray(pd.isna(probs), dtype=bool)
    if bool(missing.any()):
        n_nan = int(missing.sum())
        raise InvalidDataError(
            f"{name} contains {n_nan} NaN value(s); probabilities must be "
            f"finite values in [0, 1]. Handle missing probabilities before "
            f"validation (e.g. missing_strategy='exclude')."
        )
    if np.any(probs < 0) or np.any(probs > 1):
        raise InvalidDataError(f"{name} must be in range [0, 1]")


def handle_missing_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    strategy: MissingStrategy = "exclude",
    y_prob: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """
    Handle missing values according to specified strategy.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        sensitive_attr: Sensitive attribute values
        strategy: How to handle missing values
            - 'exclude': Remove rows with any missing values
            - 'as_group': Treat missing as a separate group (sensitive attr only)
            - 'error': Raise an error if any missing values found
        y_prob: Optional probability predictions

    Returns:
        Tuple of (y_true, y_pred, sensitive_attr, y_prob, n_excluded)

    Raises:
        ConfigurationError: If strategy is not one of the accepted values
        InvalidDataError: If strategy is 'error' and missing values exist
    """
    # FAIL CLOSED before any row is touched: an unrecognised strategy is a
    # configuration mistake, not a request for the 'exclude' default.
    _check_option(strategy, "missing_strategy", VALID_MISSING_STRATEGIES)

    # Convert to arrays for consistent handling
    y_true = coerce_to_array(y_true, "y_true").astype(float)
    y_pred = coerce_to_array(y_pred, "y_pred").astype(float)
    sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

    if y_prob is not None:
        y_prob = coerce_to_array(y_prob, "y_prob").astype(float)

    # Detect missing values
    missing_y_true = pd.isna(y_true)
    missing_y_pred = pd.isna(y_pred)

    # Handle sensitive attribute (can be any dtype)
    if np.issubdtype(sensitive_attr.dtype, np.floating):
        missing_sensitive = np.isnan(sensitive_attr)
    else:
        missing_sensitive = pd.isna(sensitive_attr)

    missing_prob = np.zeros(len(y_true), dtype=bool)
    if y_prob is not None:
        missing_prob = pd.isna(y_prob)

    # Combine missing masks
    any_missing_labels = missing_y_true | missing_y_pred | missing_prob

    if strategy == "error":
        if np.any(any_missing_labels) or np.any(missing_sensitive):
            n_missing = np.sum(any_missing_labels | missing_sensitive)
            raise InvalidDataError(
                f"Found {n_missing} rows with missing values. "
                "Use strategy='exclude' or 'as_group' to handle them."
            )
        return y_true, y_pred, sensitive_attr, y_prob, 0

    if strategy == "as_group":
        # Only handle sensitive attribute missing as separate group
        # Still exclude rows with missing labels
        mask = ~any_missing_labels
        # int() so n_excluded is JSON-serializable downstream (np.int64 is not)
        n_excluded = int(np.sum(any_missing_labels))

        y_true = y_true[mask]
        y_pred = y_pred[mask]
        sensitive_attr = sensitive_attr[mask]
        if y_prob is not None:
            y_prob = y_prob[mask]

        # Convert missing sensitive to "__missing__" group
        sensitive_attr = sensitive_attr.astype(object)
        sensitive_missing_mask = pd.isna(sensitive_attr)
        if np.any(sensitive_missing_mask):
            sensitive_attr[sensitive_missing_mask] = MISSING_GROUP_LABEL

        return y_true, y_pred, sensitive_attr, y_prob, n_excluded

    # strategy == 'exclude'
    all_missing = any_missing_labels | missing_sensitive
    mask = ~all_missing
    # int() so n_excluded is JSON-serializable downstream (np.int64 is not)
    n_excluded = int(np.sum(all_missing))

    return (
        y_true[mask],
        y_pred[mask],
        sensitive_attr[mask],
        y_prob[mask] if y_prob is not None else None,
        n_excluded,
    )


def _distinct_group_labels(sensitive_attr: Union[ArrayLike, pd.DataFrame]) -> Set[str]:
    """The set of group labels present, as strings, missing values excluded.

    Used only to compare the groups BEFORE and AFTER missing-value handling, so
    it needs to be total: an attribute column can hold anything, and a helper
    that raises here would turn a disclosure into a crash. An unreadable column
    yields the empty set, which reports no LOSS (the comparison is
    before-minus-after) rather than a false one.
    """
    try:
        if isinstance(sensitive_attr, pd.DataFrame):
            rows = sensitive_attr.astype(object).where(sensitive_attr.notna(), other=None)
            return {
                " | ".join("" if v is None else str(v) for v in row)
                for row in rows.itertuples(index=False, name=None)
                if not any(v is None for v in row)
            }
        values = np.asarray(sensitive_attr).ravel()
        return {str(v) for v, missing in zip(values, pd.isna(values)) if not missing}
    except (TypeError, ValueError):
        logger.debug(
            "_distinct_group_labels could not read the sensitive attribute, so no "
            "group-loss comparison is reported for this call",
            exc_info=True,
        )
        return set()


def validate_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: Union[ArrayLike, pd.DataFrame],
    y_prob: Optional[ArrayLike] = None,
    task_type: str = "classification",
    missing_strategy: MissingStrategy = "exclude",
) -> Tuple[
    np.ndarray, np.ndarray, Union[np.ndarray, pd.DataFrame], Optional[np.ndarray], dict[str, Any]
]:
    """
    Complete input validation pipeline.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        sensitive_attr: Sensitive attribute values (array, Series, or DataFrame for intersectional)
        y_prob: Optional probability predictions
        task_type: "classification" or "regression"
        missing_strategy: How to handle missing values

    Returns:
        Tuple of validated (y_true, y_pred, sensitive_attr, y_prob, info_dict)

    Raises:
        ConfigurationError: If task_type or missing_strategy is not recognised
        InvalidDataError: If the data itself fails validation

    **EXCLUDED ROWS ARE NOT MISSING AT RANDOM, AND THE EXCLUSION WAS SILENT.**
    BGL-3 evaluation-1, 2026-09-27. ``missing_strategy='exclude'`` drops every
    row whose sensitive attribute is missing, and it recorded the count in
    ``info['n_excluded']`` and said nothing else. Nothing warned, and nothing
    reported which GROUPS the survivors still cover, so a caller reading a
    metric had no signal at all that a comparison had been computed on part of
    the data. Measured on this repo before this change, 120 rows in three groups
    where the third group is never selected by the model::

        groups labelled          demographic_parity_difference = 0.475
                                 demographic_parity_ratio      = 0.000
        that group's attribute   demographic_parity_difference = 0.050
        missing on every row     demographic_parity_ratio      = 0.895

    The second pair is what the library reported for the SAME predictions: a
    0.895 disparate-impact ratio clears the four-fifths rule, so the starkest
    finding in the data left through the validation layer as a pass, with
    ``n_excluded: 40`` in a dict nobody was told to read. The exclusion is still
    performed (that is what the strategy asks for), but it is now disclosed: the
    count, the share of the input that qualified, and any group that vanished
    entirely, in ``info`` and in a ``UserWarning``.
    """
    # FAIL CLOSED on the options first, before any coercion. The intersectional
    # branch below has its own missing-value handling and would otherwise fall
    # through to 'exclude' for an unrecognised strategy, and a task_type that is
    # not exactly 'classification' would skip the binary-label checks in
    # silence.
    _check_option(task_type, "task_type", VALID_TASK_TYPES, hints=TASK_TYPE_HINTS)
    _check_option(missing_strategy, "missing_strategy", VALID_MISSING_STRATEGIES)

    # Convert to arrays
    y_true = coerce_to_array(y_true, "y_true")
    y_pred = coerce_to_array(y_pred, "y_pred")

    # Handle sensitive_attr - keep DataFrame for intersectional analysis
    is_intersectional = isinstance(sensitive_attr, pd.DataFrame) and sensitive_attr.shape[1] > 1
    if not is_intersectional:
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

    if y_prob is not None:
        y_prob = coerce_to_array(y_prob, "y_prob")

    # Check lengths
    sensitive_len = len(sensitive_attr)
    arrays = [y_true, y_pred]
    if y_prob is not None:
        arrays.append(y_prob)
    original_size = check_consistent_length(*arrays)
    if sensitive_len != original_size:
        raise InvalidDataError(
            f"Inconsistent array lengths: sensitive_attr has {sensitive_len}, others have {original_size}"
        )

    # Captured BEFORE the missing-value handling below reassigns it, so a group
    # that disappears with its rows can be named afterwards.
    groups_before = _distinct_group_labels(sensitive_attr)

    # Handle missing values (for intersectional, pass through as-is for now)
    if is_intersectional:
        # is_intersectional is only True when sensitive_attr is a DataFrame
        # (see above); this narrows the union for the DataFrame-only ops below.
        assert isinstance(sensitive_attr, pd.DataFrame)
        n_excluded = 0
        # Missing anywhere: the y arrays AND the sensitive-attribute columns.
        # Previously NaN attribute values were passed through, where they were
        # stringified into literal 'nan' intersection groups (phantom groups
        # that could drive the reported disparity).
        missing_y = pd.isna(y_true) | pd.isna(y_pred)
        if y_prob is not None:
            missing_y = missing_y | pd.isna(y_prob)
        missing_attr = sensitive_attr.isna().any(axis=1).to_numpy()
        missing_any = missing_y | missing_attr
        if np.any(missing_any):
            if missing_strategy == "error":
                raise InvalidDataError(
                    f"Found {int(np.sum(missing_any))} rows with missing values "
                    f"({int(np.sum(missing_attr))} in sensitive attributes)."
                )
            if missing_strategy == "as_group":
                # BGL grade-1 G01, 2026-09-30. ``as_group`` WAS SILENTLY IGNORED
                # HERE. This branch read ``missing_strategy`` only to raise on
                # 'error' and then excluded the rows whatever the caller asked,
                # while the non-intersectional branch below honours 'as_group' by
                # labelling them ``__missing__`` and dropping nothing. Measured on
                # this repo, one 120-row dataset, the same missing pattern:
                #   1d array,  as_group -> groups ['__missing__','b','w'],
                #                          120 of 120 rows assessed, 0 excluded
                #   DataFrame, as_group -> groups ['F_w','M_b'],
                #                          60 of 120 rows assessed, 60 excluded,
                #                          "missing_strategy='as_group'"
                # So half the dataset was dropped under the one strategy that
                # exists to avoid dropping it, and the provenance clause put the
                # exclusion count next to the name of the strategy that forbids
                # it: a reader reconciling the two can only conclude the data was
                # unkeepable. The rows dropped are NOT missing at random (see the
                # exclusion warning this same function raises, which records a
                # parity gap moving from 0.475 to 0.050 when one group's rows
                # went), so the verdict was computed on a biased survivor set.
                #
                # ``__missing__`` is the SAME EXPLICIT label handle_missing_values
                # uses, and deliberately not the str()-minted 'nan' / '<NA>' /
                # 'None' this branch's own comment above was written to stop: the
                # level is named by this library on purpose, it is one level
                # however many spellings of absence the frame holds, and nobody
                # can mistake it for a protected class somebody chose.
                mask = ~missing_y
                n_excluded = int(np.sum(missing_y))
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                sensitive_attr = (
                    sensitive_attr.iloc[mask]
                    .reset_index(drop=True)
                    .astype(object)
                    .where(lambda frame: frame.notna(), other=MISSING_GROUP_LABEL)
                )
                if y_prob is not None:
                    y_prob = y_prob[mask]
            else:
                mask = ~missing_any
                n_excluded = int(np.sum(missing_any))
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                sensitive_attr = sensitive_attr.iloc[mask].reset_index(drop=True)
                if y_prob is not None:
                    y_prob = y_prob[mask]
    else:
        # non-intersectional path coerced sensitive_attr to an array above.
        assert isinstance(sensitive_attr, np.ndarray)
        y_true, y_pred, sensitive_attr, y_prob, n_excluded = handle_missing_values(
            y_true, y_pred, sensitive_attr, missing_strategy, y_prob
        )

    # Task-specific validation
    if task_type == "classification":
        # y_true must be validated too: TPR/FPR condition on (y_true == 1) /
        # (y_true == 0), so non-binary encodings (e.g. 1/2 or 'yes'/'no')
        # silently dropped rows from the denominators instead of failing.
        validate_binary_labels(y_true, "y_true")
        validate_binary_labels(y_pred, "y_pred")
        if y_prob is not None:
            # y_prob was coerced to an array above and only ever re-assigned to
            # array-returning ops; narrow the union for validate_probabilities.
            assert isinstance(y_prob, np.ndarray)
            validate_probabilities(y_prob, "y_prob")

    # Coverage disclosure. n_excluded alone said how many rows went; it did not
    # say what share of the data the answer rests on, nor whether a whole group
    # went with them. See the measured 0.475 -> 0.050 swing in the docstring.
    groups_after = _distinct_group_labels(sensitive_attr)
    groups_dropped = sorted(groups_before - groups_after)
    coverage = (len(y_true) / original_size) if original_size else None

    # Compile info
    info = {
        "original_size": original_size,
        "final_size": len(y_true),
        "n_excluded": n_excluded,
        "missing_strategy": missing_strategy,
        "is_intersectional": is_intersectional,
        "n_groups_before": len(groups_before),
        "n_groups_after": len(groups_after),
        "groups_dropped": groups_dropped,
        # None, not 1.0, on an empty input: no share of nothing qualified.
        "coverage": coverage,
    }

    if n_excluded or (original_size and len(y_true) == 0):
        share = "0.0" if coverage is None else f"{coverage * 100:.1f}"
        lost = (
            f" Group(s) {', '.join(groups_dropped)} lost every row and are absent from "
            f"every comparison computed from this data."
            if groups_dropped
            else ""
        )
        if len(y_true) == 0:
            tail = (
                "NOT ONE ROW SURVIVED, so any metric computed from this is computed "
                "from nothing; a 0.0 disparity there would be perfect parity over an "
                "empty dataset."
            )
        else:
            tail = (
                "Rows with missing values are rarely missing at random, so a "
                "comparison over the survivors can differ arbitrarily from the whole: "
                "measured on this library, dropping one never-selected group's rows "
                "moved the reported demographic parity gap from 0.475 to 0.050 and the "
                "disparate-impact ratio from 0.00 to 0.89, i.e. from the starkest "
                "possible finding to a pass."
            )
        warnings.warn(
            f"validate_inputs: {n_excluded} of {original_size} row(s) were excluded by "
            f"missing_strategy={missing_strategy!r}, so {len(y_true)} row(s) "
            f"({share} percent of the input) carry every number computed from this "
            f"call.{lost} {tail}",
            UserWarning,
            stacklevel=3,
        )

    # y_prob is either None or was coerced to an ndarray on every path above.
    assert y_prob is None or isinstance(y_prob, np.ndarray)
    return y_true, y_pred, sensitive_attr, y_prob, info


def _finite_values_are_integral(values: Any) -> Optional[bool]:
    """Whether every finite value is integral. None means COULD NOT READ.

    None is a third answer and is never collapsed into False: an argument that
    cannot be coerced to numbers has not been shown to be integral OR
    continuous, and the caller has to be able to tell that apart from a
    measured answer.
    """
    try:
        arr = np.asarray(coerce_to_array(values), dtype=float)
    except (TypeError, ValueError, InvalidDataError):
        # LOGGED, not discarded. Returning None is the honest answer and the
        # caller handles it, but throwing the reason away leaves nobody able to
        # say WHY a check could not read its argument. The core no-silent-swallow
        # gate refuses a bare handler here for exactly that reason.
        logger.debug(
            "_finite_values_are_integral could not coerce its argument to "
            "numbers, so integrality is UNKNOWN rather than False",
            exc_info=True,
        )
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    # np.all returns np.bool_; this converts the type, it does not collapse a
    # missing answer into a measured one (that case returned None above).
    return bool(np.all(finite == np.rint(finite)))


def _warn_orientation_undecidable(
    y_prob: ArrayLike,
    *,
    caller: str,
    y_true_was_readable: bool,
) -> None:
    """Say so when the argument order could not be checked either way.

    Silence from ``reject_swapped_labels_and_scores`` used to cover two
    different situations: a score column that is continuous, which agrees with
    the documented order, and a hard 0/1 column, which is indistinguishable
    from a label column and so decides nothing. The second is could-not-check
    and is reported here.
    """
    prob_is_integral = _finite_values_are_integral(y_prob)
    if prob_is_integral is False and y_true_was_readable:
        return  # measured: a continuous score column, the documented order

    if prob_is_integral is None:
        detail = (
            "y_prob could not be read as numbers, so nothing about it could be "
            "compared against y_true"
        )
    elif not y_true_was_readable:
        detail = "y_true holds no finite values, so there was nothing to compare against y_prob"
    else:
        detail = (
            "y_true holds only whole numbers, which is what labels look like, "
            "and so does y_prob, which is what a hard 0/1 prediction column "
            "looks like"
        )
    warnings.warn(
        f"{caller}.fit(y_true, y_prob, sensitive_attr): the argument ORDER could not "
        f"be checked: {detail}. A reversed call is indistinguishable from a correct "
        f"one in this shape, so this is could-not-check, not a clean bill of health. "
        f"Pass a continuous probability column if you have one, and call with "
        f"keywords (y_true=..., y_prob=..., sensitive_attr=...). A reversed call "
        f"collapses every threshold and then reports a healthy fairness gap, because "
        f"accepting everybody is trivially equal.",
        UserWarning,
        stacklevel=3,
    )


def reject_swapped_labels_and_scores(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    caller: str,
) -> None:
    """Refuse a ``fit(y_prob, y_true, ...)`` call with its first two arguments reversed.

    Every ``fit`` in the post-processing API is
    ``fit(y_true, y_prob, sensitive_attr)``, and each one immediately runs
    ``coerce_to_array(y_true).astype(int)``. That cast is what makes a reversed
    call SILENT rather than noisy: a probability column becomes all zeros, so
    every threshold collapses to the floor and nearly every row is accepted.
    The fairness gap then reads HEALTHY, because accepting everybody is
    trivially equal, and so the mitigation's own scorecard reports success
    while the mitigation is doing nothing.

    Measured on the production consumer, 2026-09-09, 600 rows, two groups with
    divergent base rates. ``GroupThresholdOptimizer`` fitted as documented gave
    per-group thresholds 0.6561 and 0.4173 with 246 acceptances; fitted with
    the first two arguments reversed it gave 0.05 and 0.05 with 598
    acceptances, and 58.7 percent of decisions changed. The demographic parity
    gap afterwards was 0.007, which looks like a success.

    Labels are integral, scores are not, so a non-integral value in ``y_true``
    is the signature of the swap and is refused here by name.

    THREE STATES, NEVER TWO. Returning quietly does not mean "the arguments are
    the right way round"; it means one of two different things, and ``y_prob``
    is what tells them apart:

        refused           y_true holds non-integral values. A swap, by name.
        checked, silent   y_true is integral AND y_prob is continuous. The
                          shapes agree with the documented order.
        COULD NOT CHECK   y_true is integral and y_prob is integral too (a hard
                          0/1 prediction column), or y_prob cannot be read as
                          numbers at all. A reversed call is genuinely
                          indistinguishable from a correct one in that shape,
                          so this warns rather than passing silently.

    That third row is why ``y_prob`` is a parameter. Until 2026-09-10 it was
    accepted and never read: ``reject_swapped_labels_and_scores(labels, "not an
    array", caller="X")`` returned None, and a hard 0/1 column returned None
    with no warning, which is indistinguishable from a checked pass. The
    refusal itself is unchanged and still reads only ``y_true``; a swap cannot
    be DETECTED from ``y_prob``, only declared undecidable.

    It also does not restrict labels to {0, 1}: a multiclass integer label set
    passes. The primary defence against the swap is that every ``fit`` here
    takes keyword-only parameters, so the reversed positional call cannot be
    written at all; this value check is the second line, for a call that uses
    keywords but mislabels them.
    """
    arr = np.asarray(coerce_to_array(y_true), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        non_integral = finite[finite != np.rint(finite)]
    else:
        # Nothing finite to look at, so nothing to refuse. Not a pass either:
        # the could-not-check warning below is the honest report.
        non_integral = finite
    if non_integral.size == 0:
        _warn_orientation_undecidable(
            y_prob,
            caller=caller,
            y_true_was_readable=finite.size > 0,
        )
        return
    raise ValueError(
        f"{caller}.fit(y_true, y_prob, sensitive_attr): y_true must be labels, but it "
        f"holds non-integral values (for example {float(non_integral.flat[0])!r}). "
        f"{non_integral.size} of {finite.size} values are non-integral. This is what a "
        "reversed call looks like: the scores were passed where the labels belong. "
        "Pass the labels first, or call with keywords "
        "(y_true=..., y_prob=..., sensitive_attr=...). Refused rather than computed, "
        "because the reversed call collapses every threshold and then reports a "
        "healthy fairness gap, since accepting everybody is trivially equal."
    )
