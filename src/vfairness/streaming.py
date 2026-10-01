"""Chunked (streaming) computation for large datasets (VB-PERF-2).

Metrics that are exact aggregations over the data (per-group counts and
selection rates, and the demographic-parity spread built from them) can be
computed in fixed-size chunks with a result that is bit-identical to computing
them over the whole array at once. This lets a caller process a very large or
out-of-core dataset in pieces instead of materialising and scanning it all at
once, without changing any number.

Predictions must be binary (0/1). Rows with a NaN prediction OR a missing
sensitive value are excluded per chunk, matching the standard API's default
``missing_strategy='exclude'`` so the streamed and whole-array results agree.
Non-binary predictions raise, as they do in the validated core.

What is NOT streamed: bootstrap confidence intervals, permutation tests, and
metrics that need the full joint sample (they require all rows together). Use
the standard API in ``vfairness.evaluation`` for those.

Example::

    from vfairness.streaming import streaming_demographic_parity
    # y_pred and sensitive can be any iterables yielded in chunks
    dp = streaming_demographic_parity(y_pred, sensitive, chunk_size=100_000)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Iterator, List, Set, Tuple

import numpy as np
import pandas as pd

from .exceptions import InvalidDataError

__all__ = [
    "stream_group_counts",
    "streaming_selection_rates",
    "streaming_demographic_parity",
]


def _iter_chunks(
    y_pred: Iterable[object], sensitive_attr: Iterable[object], chunk_size: int
) -> Iterator[Tuple[Any, Any]]:
    """Yield aligned (y_pred, sensitive) chunks from arrays or iterables."""
    if chunk_size <= 0:
        raise InvalidDataError(f"chunk_size must be positive, got {chunk_size}")
    yp = np.asarray(y_pred)
    sa = np.asarray(sensitive_attr, dtype=object)
    if yp.shape[0] != sa.shape[0]:
        raise InvalidDataError(
            f"y_pred and sensitive_attr length mismatch: {yp.shape[0]} vs {sa.shape[0]}"
        )
    n = yp.shape[0]
    for start in range(0, n, chunk_size):
        end = start + chunk_size
        yield yp[start:end], sa[start:end]


def _accumulate_group_counts(
    y_pred: Iterable[object],
    sensitive_attr: Iterable[object],
    chunk_size: int,
) -> Tuple[Dict[object, Tuple[int, int]], int, int, List[object]]:
    """The chunked accumulation, plus everything the return dict cannot carry.

    Returns ``(counts, n_seen, n_qualified, groups_lost)``. ``groups_lost`` holds
    the group labels that were PRESENT in the sensitive column and contributed no
    qualifying row at all, which is the streaming mirror of the before/after group
    comparison ``_validation.validate_inputs`` makes on the whole-array path
    (``_distinct_group_labels``, groups_before minus groups_after).

    WHY THIS IS A SEPARATE FUNCTION (2026-09-27). ``streaming_demographic_parity``
    built its own drop disclosure by iterating ``counts.items()``, so it was
    STRUCTURALLY unable to name a group that never reached ``counts``: a group
    whose every row carries a NaN prediction is not a key there. Both public
    entry points now read the same accumulation, so both can say what left.
    """
    positives: Dict[object, int] = {}
    totals: Dict[object, int] = {}
    labels_seen: Set[object] = set()
    labels_readable = True
    n_seen = 0
    n_qualified = 0
    for yp_chunk, sa_chunk in _iter_chunks(y_pred, sensitive_attr, chunk_size):
        yp_arr = np.asarray(yp_chunk)
        sa_arr = np.asarray(sa_chunk, dtype=object)
        # Match the standard API's missing_strategy='exclude': rows with a NaN
        # prediction OR a missing sensitive value are dropped from BOTH the
        # numerator and the denominator, not counted. Without excluding missing
        # sensitive values, np.unique over a mix of None/NaN and strings would
        # also raise, and the streamed result would diverge from the core.
        if np.issubdtype(yp_arr.dtype, np.floating):
            valid = ~np.isnan(yp_arr)
        else:
            valid = np.ones(yp_arr.shape[0], dtype=bool)
        present_labels = ~pd.isna(sa_arr)
        valid = valid & present_labels
        # Every group the DATA holds, whether or not any of its rows qualified.
        # Missing labels are not groups, the same rule _distinct_group_labels
        # applies on the whole-array path.
        if labels_readable:
            try:
                labels_seen.update(
                    v for v, keep in zip(sa_arr.tolist(), present_labels.tolist()) if keep
                )
            except TypeError:
                # An unhashable label (a list, say) cannot be compared before and
                # after exclusion. Report NO loss rather than crash a disclosure:
                # _distinct_group_labels is deliberately total for the same reason.
                labels_readable = False
        # Counted per chunk and summed, so the disclosure below costs one integer
        # rather than a second pass over the data.
        n_seen += int(yp_arr.shape[0])
        n_qualified += int(valid.sum())
        # Reject non-binary predictions per chunk (symmetric with the validated
        # core, which raises on non-0/1 predictions), while staying streaming.
        present = np.unique(yp_arr[valid])
        if not set(present.tolist()).issubset({0, 1}):
            raise InvalidDataError(
                f"y_pred must contain only 0 and 1 (got {present.tolist()}); "
                "streaming metrics operate on binary predictions."
            )
        # Grouped over the QUALIFYING ROWS ONLY. This was
        # ``gmask = (sa_arr == g) & valid``, which compares g against the whole
        # chunk including the cells `valid` exists to exclude. With ``pd.NA`` in
        # the sensitive column that elementwise comparison yields an object
        # array holding ``pd.NA``, and the ``&`` then raises
        # ``TypeError: boolean value of NA is ambiguous``. Measured 2026-09-30 on
        # 80 rows, 40 labelled 'a' and 40 ``pd.NA``: every one of the three public
        # entry points raised, while ``None``, ``np.nan`` and ``pd.NaT`` in the
        # same position were excluded exactly as documented. ``pd.NA`` is what a
        # pandas nullable column (``string``, ``Int64``, ``boolean``) gives for a
        # missing value, so this was the ordinary arrival path for a sensitive
        # column with a gap in it, against a module whose contract is that such
        # rows are excluded. Filtering first is also equivalent for every other
        # input, because the mask could only ever select rows `valid` kept.
        labels = sa_arr[valid]
        preds = yp_arr[valid]
        for g in np.unique(labels):
            gmask = labels == g
            totals[g] = totals.get(g, 0) + int(gmask.sum())
            positives[g] = positives.get(g, 0) + int((preds[gmask] == 1).sum())
    result = {g: (positives.get(g, 0), totals[g]) for g in totals}
    groups_lost: List[object] = []
    if labels_readable:
        try:
            groups_lost = sorted(labels_seen - set(result), key=lambda v: str(v))
        except TypeError:  # pragma: no cover - set() over hashables cannot raise here
            groups_lost = []
    return result, n_seen, n_qualified, groups_lost


def _warn_group_coverage(
    caller: str,
    result: Dict[object, Tuple[int, int]],
    n_seen: int,
    n_qualified: int,
    groups_lost: List[object],
    stacklevel: int,
) -> None:
    """Emit what the counts cannot carry: rows excluded, and groups wiped out.

    Shared by ``stream_group_counts`` and ``streaming_demographic_parity`` so the
    two surfaces cannot disclose different amounts of the same exclusion.
    """
    lost = ", ".join(repr(g) for g in groups_lost)
    if not result:
        warnings.warn(
            f"{caller}: no group could be counted. {n_qualified} of "
            f"{n_seen} row(s) qualified (a row is excluded when its prediction is "
            f"NaN or its sensitive value is missing), so the empty result means "
            f"nothing was measured, NOT that the data holds no groups. Those two "
            f"are the same return value, which is why this says so."
            + (
                f" Group(s) {lost} lost every row and are absent from every "
                f"comparison computed from this data."
                if groups_lost
                else ""
            ),
            UserWarning,
            stacklevel=stacklevel,
        )
    elif n_qualified < n_seen:
        share = 100.0 * (n_seen - n_qualified) / n_seen
        warnings.warn(
            f"{caller}: {n_seen - n_qualified} of {n_seen} rows "
            f"({share:.1f} percent) were excluded for a missing prediction or a "
            f"missing sensitive value, so every count returned covers "
            f"{n_qualified} rows. Exclusion matches the standard API's "
            f"missing_strategy='exclude' default; the counts are exact over what "
            f"remained, and say nothing about what was dropped.",
            UserWarning,
            stacklevel=stacklevel,
        )
    if result and groups_lost:
        # THE SECOND ROUTE OUT OF THE COMPARISON (2026-09-27). The row-share
        # warning above never names a group, and the size-gate disclosure in
        # streaming_demographic_parity could not see this one at all. Wording
        # mirrors _validation.validate_inputs, which says the same sentence on the
        # whole-array path, so a reader comparing the two need not work out
        # whether they mean the same thing.
        warnings.warn(
            f"{caller}: group(s) {lost} lost every row and are absent from every "
            f"comparison computed from this data, because every one of their rows "
            f"carried a NaN prediction or a missing sensitive value. That is a "
            f"could not check for those group(s), NOT a finding that they match "
            f"the groups that remain.",
            UserWarning,
            stacklevel=stacklevel,
        )


def stream_group_counts(
    y_pred: Iterable[object],
    sensitive_attr: Iterable[object],
    chunk_size: int = 100_000,
) -> Dict[object, Tuple[int, int]]:
    """Accumulate ``(positive_count, total_count)`` per group in chunks.

    Positives are entries equal to 1. The result is exact: summing counts over
    chunks equals counting over the whole array.

    HOW MANY ROWS QUALIFIED IS DISCLOSED, because the return value cannot carry
    it. The whole-array path keeps that number: ``_validation.handle_missing_values``
    RETURNS ``n_excluded`` alongside the filtered arrays, so its caller can report
    what left the sample. This function returns a dict of groups and nothing else,
    so an exclusion had no channel at all and was silent. Measured 2026-09-27,
    before the warning below:

    * 60 rows, groups ``a`` and ``b`` 30 rows each, EVERY prediction NaN, returned
      ``{}`` with no warning: byte-identical to ``stream_group_counts([], [])``, so
      "nothing could be measured" and "there are no groups" were the same answer;
    * 60 rows with 30 sensitive values missing returned
      ``{'a': (8, 15), 'b': (7, 15)}``, also silently, with no way for the caller
      to learn that half the input was gone.

    The counts themselves are unchanged: exclusion still follows the standard
    API's ``missing_strategy='exclude'`` default, so the streamed and whole-array
    numbers still agree.

    A GROUP THAT LOST EVERY ROW IS NAMED (2026-09-27, second pass). The row share
    above never named a group, so the worst case it covers read as an ordinary
    partial drop. Measured before this change on 140 rows, ``a`` 50 rows and ``b``
    50 rows selected at 0.5 and ``c`` 40 rows whose predictions are ALL NaN:
    ``stream_group_counts`` returned ``{'a': (25, 50), 'b': (25, 50)}`` and said
    only "40 of 140 rows (28.6 percent) were excluded", with ``c`` named nowhere.
    It now adds "Group(s) 'c' lost every row and are absent from every comparison
    computed from this data", the same sentence ``_validation.validate_inputs``
    emits on the whole-array path.
    """
    result, n_seen, n_qualified, groups_lost = _accumulate_group_counts(
        y_pred, sensitive_attr, chunk_size
    )
    _warn_group_coverage(
        "stream_group_counts", result, n_seen, n_qualified, groups_lost, stacklevel=3
    )
    return result


def streaming_selection_rates(
    y_pred: Iterable[object],
    sensitive_attr: Iterable[object],
    chunk_size: int = 100_000,
    *,
    min_group_size: int = 30,
) -> Dict[object, float]:
    """Per-group positive (selection) rate, computed in chunks.

    THIS FUNCTION DROPS THE DENOMINATOR, so the size a rate rests on has no
    channel in the return value and must be disclosed. ``stream_group_counts``
    hands back ``(positives, total)`` and a caller can see ``(1, 1)`` for what
    it is; a ``{'b': 1.0}`` is a selection rate of 100 percent with nothing
    beside it. Measured 2026-09-30, before the warning below, on 100 rows with
    ``a`` 99 rows selected at about half and ``b`` ONE row, selected::

        streaming_selection_rates(...)   -> {'a': 0.505, 'b': 1.0}, silent
        streaming_demographic_parity(...) -> nan, and it warns

    Two surfaces of one module read the same data oppositely -- a 50-point
    selection gap against a refusal -- and only one of them said why. The rates
    themselves are UNCHANGED and no group is dropped: gating here would delete
    rows from a result whose whole contract is that it equals the whole-array
    counts, and a deletion is the worse answer. ``min_group_size`` decides only
    which denominators get NAMED, and its default matches the gate
    ``streaming_demographic_parity`` and the ``evaluation`` metrics already use.

    The row-exclusion and vanished-group disclosures are emitted under THIS
    function's name. They used to arrive via ``stream_group_counts``, so a
    caller of this function was told about a function it had not called, and the
    ``stacklevel`` pointed inside this module rather than at their own line.
    """
    counts, n_seen, n_qualified, groups_lost = _accumulate_group_counts(
        y_pred, sensitive_attr, chunk_size
    )
    _warn_group_coverage(
        "streaming_selection_rates", counts, n_seen, n_qualified, groups_lost, stacklevel=3
    )
    rates = {g: (pos / tot if tot else float("nan")) for g, (pos, tot) in counts.items()}
    undersized = {g: tot for g, (_pos, tot) in counts.items() if tot < min_group_size}
    if undersized:
        named = ", ".join(f"{g!r} (n={tot})" for g, tot in undersized.items())
        warnings.warn(
            f"streaming_selection_rates: {len(undersized)} of {len(counts)} group(s) "
            f"hold fewer than min_group_size={min_group_size} qualifying rows: "
            f"{named}. Their rate is exact over the rows they do hold and the "
            f"return value cannot carry that denominator, so read those rates as "
            f"describing that many observations, not as an estimate of the group. "
            f"streaming_demographic_parity excludes the same group(s) from its "
            f"comparison at this gate.",
            UserWarning,
            stacklevel=3,
        )
    return rates


def streaming_demographic_parity(
    y_pred: Iterable[object],
    sensitive_attr: Iterable[object],
    *,
    min_group_size: int = 30,
    chunk_size: int = 100_000,
) -> float:
    """Demographic-parity difference (max minus min selection rate) via chunks.

    Groups with fewer than ``min_group_size`` samples are excluded, matching the
    default gate of ``evaluation`` metrics. The value is identical to computing
    it over the whole array.

    Returns NaN, never 0.0, when fewer than two groups survive that gate: the
    comparison never happened, and 0.0 is indistinguishable downstream from a
    measured perfect parity. See the CONVENTION block in
    ``evaluation/vfairness_metrics/classification.py``, which this function was
    written before and did not follow. A warning is emitted with it, because a
    NaN a caller cannot explain is only half an answer.

    A PARTIAL DROP IS ALSO DISCLOSED, and that half was still missing. The core
    metric grew ``_warn_dropped_groups`` for it; this twin had not. Measured
    2026-09-27 on 110 rows, ``a`` 50 rows selected at 0.5, ``b`` 50 rows at 0.5,
    ``c`` 10 rows at 0.0, with the default gate: ``demographic_parity_difference``
    returned 0.0 AND warned "Excluding 1 group(s) below min_group_size=30:
    {'c': 10}. That leaves 10 of 110 rows (9.1 percent) out of this metric",
    while this function returned 0.0 in complete silence. 0.0 is perfect parity,
    the strongest all-clear the metric has, and the group it was hiding had been
    selected at zero percent. With ``min_group_size=10`` the same data gives 0.5.
    The VALUE stays max minus min over the eligible groups, because it must keep
    matching the whole-array result; what was missing was saying so.

    THERE ARE TWO ROUTES OUT OF THE COMPARISON AND THE FIX ABOVE COVERED ONE
    (2026-09-27, second pass). A group also leaves through MISSING DATA, and
    ``dropped_sizes`` is built by iterating ``counts.items()``, so it was
    structurally unable to name a group that never reached ``counts``. Measured on
    140 rows, ``a`` 50 rows and ``b`` 50 rows selected at 0.5 and ``c`` 40 rows
    whose predictions are ALL NaN, which is a classifier that abstained on one
    subpopulation:

    * this function returned **0.0**, perfect parity and the strongest all-clear
      the metric has, with ONE warning, a row count, and ``c`` named nowhere;
    * ``demographic_parity_difference`` on the identical data returned 0.0 AND
      warned "Group(s) c lost every row and are absent from every comparison
      computed from this data", which is the disclosure whose own comment records
      that dropping one never-selected group's rows moved a reported parity gap
      from 0.475 to 0.050, "from the starkest possible finding to a pass".

    The same 0.0 with the group NAMED is what it returns now, by the same sentence
    as the core, and the fail-closed warning below counts the groups the DATA held
    rather than the groups that reached ``counts``: it said "out of 2" for a
    dataset holding three groups. The value is still max minus min over the
    eligible groups, because it must keep matching the whole-array result.
    """
    counts, n_seen, n_qualified, groups_lost = _accumulate_group_counts(
        y_pred, sensitive_attr, chunk_size
    )
    _warn_group_coverage(
        "streaming_demographic_parity",
        counts,
        n_seen,
        n_qualified,
        groups_lost,
        stacklevel=3,
    )
    rates = {g: pos / tot for g, (pos, tot) in counts.items() if tot >= min_group_size and tot > 0}
    eligible = list(rates.values())
    if len(eligible) < 2:
        # FAIL CLOSED. This returned 0.0 until 2026-09-07, silently: a 50-strong
        # majority selected at 100 percent against a 10-strong minority selected
        # at 0 percent reported PERFECT PARITY and emitted nothing at all, which
        # is the exact incident the classification convention was written for.
        surviving = list(rates)
        # "out of {len(counts)}" UNDERCOUNTED THE DATA (2026-09-27). counts holds
        # only the groups with a qualifying row, so on 140 rows holding three
        # groups, one of them all-NaN, this said "out of 2". The denominator is now
        # every group the data held, and the vanished ones are named.
        n_groups_in_data = len(counts) + len(groups_lost)
        vanished = (
            " " + f"Group(s) {', '.join(repr(g) for g in groups_lost)} reached no "
            "comparison at all because every one of their rows was excluded."
            if groups_lost
            else ""
        )
        warnings.warn(
            f"streaming_demographic_parity: only {len(surviving)} group(s) met "
            f"min_group_size={min_group_size} out of {n_groups_in_data} group(s) in "
            f"the data, so no between-group comparison was possible; returning NaN "
            f"rather than 0.0, which would read as perfect parity." + vanished,
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    # SOME groups survived and some did not, which is the case the fail-closed
    # branch above does not cover: the spread IS measured, over whatever remained.
    # Wording mirrors _warn_dropped_groups in the core metric on purpose, down to
    # the row share, because a reader comparing the two must not have to work out
    # whether they mean the same thing.
    dropped_sizes = {g: tot for g, (_, tot) in counts.items() if g not in rates}
    if dropped_sizes:
        n_dropped = sum(dropped_sizes.values())
        n_rows = sum(tot for _, tot in counts.values())
        share = (100.0 * n_dropped / n_rows) if n_rows else 0.0
        warnings.warn(
            f"streaming_demographic_parity: excluding {len(dropped_sizes)} group(s) "
            f"below min_group_size={min_group_size}: {dropped_sizes}. That leaves "
            f"{n_dropped} of {n_rows} rows ({share:.1f} percent) out of this metric. "
            f"The value is computed over the remaining groups only, so it measures "
            f"nothing about the excluded group(s): they are could not check, not a "
            f"measured pass. To include them, lower min_group_size.",
            UserWarning,
            stacklevel=2,
        )
    return max(eligible) - min(eligible)
