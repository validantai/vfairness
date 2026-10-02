"""
Shared grouping utilities for vfairness.

This module provides centralized group computation, caching, and statistics
for all fairness metric computations.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..._triage import is_measured, unmeasurable_reason
from ...exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class GroupInfo:
    """Container for group statistics."""

    name: str
    mask: np.ndarray
    size: int
    indices: np.ndarray


class GroupManager:
    """
    Manages group computations with caching for fairness analysis.

    This class handles:
    - Unique group identification
    - Group masks and indices
    - Minimum sample size enforcement
    - Intersectional group creation
    - Caching for repeated access
    """

    def __init__(self, sensitive_attr: np.ndarray, min_group_size: int = 30):
        """
        Initialize GroupManager.

        Args:
            sensitive_attr: Sensitive attribute values (1D or 2D for intersectional)
            min_group_size: Minimum samples required per group
        """
        self.sensitive_attr = sensitive_attr
        self.min_group_size = min_group_size
        self._cache: Dict[str, GroupInfo] = {}
        self._groups: Optional[List[str]] = None
        self._is_intersectional = False

        # Handle intersectional attributes (DataFrame or 2D array)
        if isinstance(sensitive_attr, pd.DataFrame):
            self._is_intersectional = sensitive_attr.shape[1] > 1
            self._create_intersectional_groups(sensitive_attr)
        elif isinstance(sensitive_attr, np.ndarray) and sensitive_attr.ndim == 2:
            self._is_intersectional = sensitive_attr.shape[1] > 1
            df = pd.DataFrame(sensitive_attr)
            self._create_intersectional_groups(df)
        else:
            self._create_simple_groups()

    @staticmethod
    def _level_mask(attr: np.ndarray, val: Any) -> np.ndarray:
        """Boolean mask of the rows sitting at one level of the attribute.

        ``attr == nan`` is False for EVERY row, so a missing sensitive value
        produced a group whose mask selected nobody: size 0, proportion 0.0,
        and the rows carrying it in no group at all. Zero rows is not a
        measurement of a level that can hold a third of the dataset, and the
        DataFrame path in ``_create_intersectional_groups`` counts the very
        same rows correctly (it stringifies before comparing), so the two
        paths disagreed on identical data: 0 against 10 for ten missing rows.
        ``pd.isna`` matches what equality cannot.

        Missing rows are still reported under the label ``str(value)`` (i.e.
        "nan"), and ``__init__`` warns that they are being carried as their
        own level, so nobody reads them as a real protected class.
        """
        try:
            level_is_missing = bool(pd.isna(val))
        except (TypeError, ValueError):
            # A non-scalar level (pd.isna returns an array): not a missing
            # marker, so fall through to equality. LOGGED, never discarded: the
            # answer is a classification, but nobody can say why without it.
            logger.debug(
                "GroupManager._level_mask could not ask whether the level %r is a "
                "missing marker; treating it as a value and comparing by equality",
                val,
                exc_info=True,
            )
            level_is_missing = False
        if level_is_missing:
            return np.asarray(pd.isna(attr))
        return np.asarray(attr == val)

    def _create_simple_groups(self) -> None:
        """Create groups from a single attribute."""
        attr = np.asarray(self.sensitive_attr)
        unique_values = np.unique(attr)
        self._groups = [str(v) for v in unique_values]

        for val in unique_values:
            name = str(val)
            mask = self._level_mask(attr, val)
            indices = np.where(mask)[0]
            self._cache[name] = GroupInfo(
                name=name, mask=mask, size=int(np.sum(mask)), indices=indices
            )

        self._warn_on_missing_level(attr)

    @staticmethod
    def _warn_on_missing_level(attr: np.ndarray) -> None:
        """Say out loud that rows with a missing attribute are their own level.

        They are counted (see ``_level_mask``), but a level called "nan" is not
        a protected class anybody chose, and a reader has to be able to tell it
        from one. Before this, the same rows were silently absent from every
        size, proportion and metric, which is the quietest possible way to drop
        a third of a dataset.
        """
        try:
            n_missing = int(np.sum(np.asarray(pd.isna(attr))))
        except (TypeError, ValueError):
            # Cannot even ask the question on this dtype, so the number of
            # missing rows is UNKNOWN, not zero. Say nothing rather than claim
            # there are none, and log the reason: a handler that returns in
            # silence leaves nobody able to tell a clean attribute from an
            # unreadable one.
            logger.debug(
                "GroupManager could not test the sensitive attribute for missing "
                "values (dtype %r), so the missing-row count is unknown, not zero",
                getattr(attr, "dtype", None),
                exc_info=True,
            )
            return
        if n_missing:
            warnings.warn(
                f"{n_missing} of {len(attr)} rows have a MISSING sensitive attribute "
                f"value. They are carried as their own level (named 'nan'), not "
                f"dropped and not merged into another group: any metric computed "
                f"over them describes rows whose protected class is unknown, which "
                f"is not a measured result for any real group. Drop them, or use "
                f"missing_strategy='as_group' upstream, if that is not what you want.",
                UserWarning,
                # Attribute it to the line that BUILT the GroupManager (here to
                # _create_simple_groups to __init__ to the caller). Python's
                # default filter prints one warning per location, so a stacklevel
                # pointing inside this module would collapse every such warning
                # in a whole run into a single line about this file, which is the
                # trap _warn_dropped_groups in classification.py already records.
                stacklevel=4,
            )

    def _create_intersectional_groups(self, df: pd.DataFrame) -> None:
        """Create intersectional groups from multiple attributes."""
        # Create combined group labels
        if isinstance(df, pd.DataFrame):
            combined = df.apply(lambda row: "_".join(str(v) for v in row), axis=1)
        else:
            combined = pd.Series(["_".join(str(v) for v in row) for row in df])

        self.sensitive_attr = combined.values
        unique_values = combined.unique()
        self._groups = list(unique_values)

        for val in unique_values:
            mask = combined.values == val
            indices = np.where(mask)[0]
            self._cache[val] = GroupInfo(
                name=val, mask=mask, size=int(np.sum(mask)), indices=indices
            )

        self._warn_on_minted_intersection(df)

    @staticmethod
    def _warn_on_minted_intersection(df: Any) -> None:
        """Say out loud that a combined level was built out of an ABSENT value.

        BGL grade-1 G01, 2026-09-30. ``"_".join(str(v) for v in row)`` above is
        the ``str(x)`` MINTING door: applied to a missing component it does not
        fail and it does not drop the row, it manufactures a name. Measured on
        this repo, a 40-row frame of ``sex`` x ``race`` whose race column holds
        ten ``pd.NA`` and ten ``None``::

            groups      ['F_white', 'F_<NA>', 'M_black', 'M_None']
            group_sizes {'F_white': 10, 'F_<NA>': 10, 'M_black': 10, 'M_None': 10}
            warnings    NONE

        So two of the four "protected groups" in that assessment are rows whose
        protected class is UNKNOWN, carrying a third of the dataset between
        them, and nothing anywhere said so. Every metric, every per-group rate
        and every significance test is then reported against a demographic group
        that does not exist; the sibling incident recorded in CLAUDE.md is a
        ``p = 1.08e-05`` against a group named '<NA>'.

        ``_create_simple_groups`` has warned about exactly these rows since
        ``_level_mask`` was written, and its docstring records that the two paths
        "disagreed on identical data". They disagreed again one layer up: the
        simple path counts the missing rows AND discloses them, the
        intersectional path counted them and said nothing. The disclosure is the
        half that was missing here.

        Deliberately a WARNING and not a change of grouping. Dropping these rows
        silently is the quieter defect (see ``_warn_on_missing_level``), and
        merging the spellings would move rows between groups, which changes
        verdicts. So the rows stay exactly where they are and the reader is told
        what the label means.

        The distinct SPELLINGS of absence are named too, because they are a
        second consequence a reader cannot see from the labels alone: ``None``,
        ``nan`` and ``<NA>`` are the same missing state and they mint DIFFERENT
        group names, so one absent category can be split across several groups,
        each small enough to fall under ``min_group_size`` while together they
        would have qualified.
        """
        try:
            frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(np.asarray(df))
            missing_cells = pd.isna(frame)
            rows_with_missing = int(missing_cells.any(axis=1).sum())
        except (TypeError, ValueError):
            # Cannot even ask the question on these dtypes, so the count is
            # UNKNOWN and not zero. Log it rather than claiming a clean frame;
            # the same rule as _warn_on_missing_level's handler.
            logger.debug(
                "GroupManager could not test the intersectional attributes for "
                "missing values, so the count of minted levels is unknown, not zero",
                exc_info=True,
            )
            return
        if not rows_with_missing:
            return

        spellings = sorted(
            {
                str(value)
                for column in frame.columns
                for value in frame.loc[missing_cells[column], column].tolist()
            }
        )
        n_rows = int(len(frame.index))
        # Only claim a split when there IS more than one spelling. pandas 3
        # stores a text column as `str`, which folds pd.NA and None into one
        # `nan` at construction, so the same input can mint one name or several.
        split_note = (
            "Note that the spellings above are the SAME missing state and mint "
            "DIFFERENT group names, so one absent category can be split across several "
            "groups. "
            if len(spellings) > 1
            else ""
        )
        warnings.warn(
            f"{rows_with_missing} of {n_rows} rows have a MISSING value in at least "
            f"one of the intersectional attributes. The combined group label is built "
            f"with str() on each component, so those rows were given a level NAMED "
            f"after the absence ({', '.join(spellings)}): they are not dropped and not "
            f"merged, but any metric computed over such a group describes rows whose "
            f"protected class is unknown, which is not a measured result for any real "
            f"group. {split_note}Drop these rows, or impute them, if that is not what "
            f"you want.",
            UserWarning,
            # Attribute it to the line that BUILT the GroupManager (here to
            # _create_intersectional_groups to __init__ to the caller), for the
            # reason _warn_on_missing_level records: a stacklevel pointing inside
            # this module collapses every such warning in a run into one line.
            stacklevel=4,
        )

    @property
    def groups(self) -> List[str]:
        """Get list of all group names."""
        if self._groups is None:
            self._create_simple_groups()
        assert self._groups is not None  # _create_simple_groups always sets it
        return self._groups

    @property
    def n_groups(self) -> int:
        """Get number of groups."""
        return len(self.groups)

    @property
    def is_intersectional(self) -> bool:
        """Check if analysis is intersectional."""
        return self._is_intersectional

    def get_group(self, name: str) -> GroupInfo:
        """
        Get info for a specific group.

        Args:
            name: Group name

        Returns:
            GroupInfo for the group

        Raises:
            KeyError: If group not found
        """
        if name not in self._cache:
            raise KeyError(f"Group '{name}' not found. Available: {self.groups}")
        return self._cache[name]

    def get_mask(self, name: str) -> np.ndarray:
        """Get boolean mask for a group."""
        return self.get_group(name).mask

    def get_indices(self, name: str) -> np.ndarray:
        """Get indices for a group."""
        return self.get_group(name).indices

    def get_size(self, name: str) -> int:
        """Get size of a group."""
        return self.get_group(name).size

    def iter_groups(self) -> Iterator[Tuple[str, GroupInfo]]:
        """Iterate over all groups."""
        for name in self.groups:
            yield name, self._cache[name]

    def get_valid_groups(self, warn_if_empty: bool = True) -> List[str]:
        """
        Get groups meeting minimum size requirement.

        Returns:
            List of group names with size >= min_group_size

        Raises:
            Warning if all groups are below min_group_size threshold
        """
        # BGL grade-1 G01, 2026-09-30. The SAME NaN door get_invalid_groups
        # records below, on this side of the pair. ``size >= nan`` is False for
        # every group, so an unusable floor emptied this list too, and the
        # warning that fired said "All groups are below min_group_size=nan",
        # which is a claim about the SIZES. Nothing was below anything: no
        # comparison happened at all. A caller reading that message goes looking
        # for more data when the fault is in its own configuration.
        if not is_measured(self.min_group_size):
            if warn_if_empty:
                warnings.warn(
                    f"GroupManager.get_valid_groups: min_group_size is "
                    f"{self.min_group_size!r}, which is not a usable size floor "
                    f"({unmeasurable_reason(self.min_group_size)}), so no group was "
                    f"compared against it and NONE of the "
                    f"{len(self.groups)} group(s) could be admitted. This is not a "
                    f"finding that the groups are too small; the threshold was never "
                    f"applied. Group sizes: {self.get_group_sizes()}.",
                    UserWarning,
                    stacklevel=2,
                )
            return []

        valid = [name for name in self.groups if self.get_size(name) >= self.min_group_size]

        if warn_if_empty and len(valid) == 0 and len(self.groups) > 0:
            # NB: no function-local ``import warnings`` here. There was one, and
            # because a name imported inside a function is LOCAL FOR THE WHOLE
            # function body, it made every earlier ``warnings.warn`` in this
            # method an UnboundLocalError. The module already imports warnings at
            # the top; a second import can only break the guards above it.
            sizes = self.get_group_sizes()
            max_size = max(sizes.values()) if sizes else 0
            # CRITICAL: this text used to promise "default values (0.0 for
            # differences, 1.0 for ratios)". Those sentinels are gone (metrics
            # return NaN and the run is marked not assessable), and that sentence
            # was what made the old false-parity result look deliberate: a reader
            # took a documented default for a measured perfect score. Do not
            # reintroduce it, in this string or any other.
            warnings.warn(
                f"All groups are below min_group_size={self.min_group_size}. "
                f"Group sizes: {sizes}. Max group size is {max_size}. "
                f"Consider reducing min_group_size (e.g., min_group_size={max(1, max_size)}) "
                f"or using more data. Metrics will return NaN (not a number) and the "
                f"run is marked not assessable: nothing here certifies fairness.",
                UserWarning,
            )

        return valid

    def get_invalid_groups(self) -> List[str]:
        """
        Get groups not meeting minimum size requirement.

        Returns:
            List of group names with size < min_group_size

        An empty list means "every group examined is large enough", and it used to
        mean that in two cases where nothing had been examined at all. Both return
        an empty list still, because the return type is a list of names and there
        is no third value to give, so both DISCLOSE instead (measured 2026-09-25):

        * no groups at all. An empty sensitive attribute produced ``[]``, which a
          caller reads as a clean bill over zero groups.
        * ``min_group_size <= 1``, under which no group can ever be invalid. The
          check is disabled and its empty answer is vacuous, the same shape as the
          design-power problem this module already guards for significance tests:
          a check whose threshold puts it out of reach cannot fire for ANY data.
        """
        # BGL5 A-evaluation-4, 2026-09-27. THE THRESHOLD OF VACUITY IS 1, NOT 0.
        # Every group here is built from ``np.unique`` (see _create_simple_groups
        # and _create_intersectional_groups), so a group exists only because at
        # least one row carries its level: no group can ever hold 0 rows, on the
        # simple path or the intersectional one. ``size < 1`` is therefore false
        # for every group of every dataset, exactly as ``size < 0`` is, so
        # min_group_size=1 disables this check just as completely as 0 does.
        #
        # The bound was ``<= 0`` and it was off by one. Measured on
        # GroupManager(np.array(['a'] * 40 + ['b'] * 2), min_group_size=t)
        # .get_invalid_groups():
        #     t=0   -> []   1 warning (vacuous)
        #     t=1   -> []   0 warnings   <- the hole: same vacuous [] , in silence
        #     t=2   -> ['b'] 0 warnings  (the first threshold that can fire)
        # After this change t=1 returns the same [] and discloses it, and t=2 is
        # untouched and still silent. min_group_size=1 is passed by production
        # call sites (post_processing/calibration, operations/cicd/gate,
        # _statistics), which is precisely why the silence mattered.
        #
        # BGL grade-1 G01, 2026-09-30. THE GUARD ABOVE IS AN ORDER COMPARISON, AND
        # NaN LOSES EVERY ORDER COMPARISON. ``nan <= 1`` is False, so a
        # min_group_size that is not a measurement walked straight past the
        # vacuity disclosure, and then ``size < nan`` is False for every group, so
        # the answer was the same vacuous ``[]`` IN SILENCE: exactly the hole
        # t=1 had, arriving through the other door. Measured on
        # GroupManager(np.array(['a'] * 40 + ['b'] * 2), min_group_size=float('nan')):
        #     get_valid_groups()   -> []   1 warning
        #     get_invalid_groups() -> []   0 warnings   <- 'every group is big enough'
        #     get_info()           -> n_groups 2, n_valid_groups 0, n_invalid_groups 0
        # Two groups that are in NEITHER list, and nothing saying the threshold
        # was never applied. ``is_measured`` is the canonical rule and also
        # refuses None, the infinities, a bool and a string, none of which is a
        # usable size floor either.
        if not is_measured(self.min_group_size):
            warnings.warn(
                f"GroupManager.get_invalid_groups: min_group_size is "
                f"{self.min_group_size!r}, which is not a usable size floor "
                f"({unmeasurable_reason(self.min_group_size)}), so no group was "
                f"compared against anything. This empty result means the check was "
                f"NEVER APPLIED, not that every group is large enough.",
                UserWarning,
                stacklevel=2,
            )
            return []
        if self.min_group_size <= 1:
            warnings.warn(
                f"GroupManager.get_invalid_groups: min_group_size is "
                f"{self.min_group_size}, and every group is built from the levels "
                f"that actually occur, so no group can ever hold fewer than 1 row "
                f"and none can ever be too small. This empty result is vacuous: it "
                "is NOT a finding that every group is large enough. The smallest "
                "threshold that can flag anything is 2.",
                UserWarning,
                stacklevel=2,
            )
            return []
        if not list(self.groups):
            warnings.warn(
                "GroupManager.get_invalid_groups: there are no groups to examine, so "
                "the empty result means nothing was checked rather than that every "
                "group is large enough.",
                UserWarning,
                stacklevel=2,
            )
            return []
        return [name for name in self.groups if self.get_size(name) < self.min_group_size]

    def get_group_sizes(self) -> Dict[str, int]:
        """Get dictionary of group sizes."""
        return {name: self.get_size(name) for name in self.groups}

    def get_group_proportions(self) -> Dict[str, float]:
        """Get dictionary of group proportions.

        A proportion of a total of zero rows does not exist, so it is NaN and
        never 0.0. Until the NaN level was counted properly (see
        ``_level_mask``) this branch was live and answered ``{'nan': 0.0}`` for
        an attribute that was missing on every single row: 0 percent of the
        data for a level holding 100 percent of it.
        """
        total = sum(self.get_size(name) for name in self.groups)
        if total == 0:
            if self.groups:
                warnings.warn(
                    f"get_group_proportions: every group is empty (0 rows in total "
                    f"across {len(self.groups)} group(s)), so a share of the data "
                    f"does not exist. Returning NaN, not 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
            return {name: float("nan") for name in self.groups}
        return {name: self.get_size(name) / total for name in self.groups}

    def compute_group_statistic(
        self, values: np.ndarray, statistic: str = "mean"
    ) -> Dict[str, float]:
        """
        Compute a statistic for each valid group.

        Args:
            values: Array of values to compute statistic over
            statistic: One of 'mean', 'sum', 'std', 'var', 'median', 'min', 'max'

        Returns:
            Dict mapping group name to statistic value
        """
        stat_funcs: Dict[str, Callable[..., Any]] = {
            "mean": np.mean,
            "sum": np.sum,
            "std": np.std,
            "var": np.var,
            "median": np.median,
            "min": np.min,
            "max": np.max,
        }

        if statistic not in stat_funcs:
            raise ConfigurationError(
                f"Unknown statistic: {statistic}. Use one of {list(stat_funcs.keys())}"
            )

        func = stat_funcs[statistic]
        results = {}

        for name in self.get_valid_groups():
            mask = self.get_mask(name)
            group_values = values[mask]
            if len(group_values) > 0:
                results[name] = float(func(group_values))
            else:
                results[name] = np.nan

        return results

    def compute_group_rate(
        self, numerator_mask: np.ndarray, denominator_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute a rate for each valid group.

        Useful for computing positive rates, true positive rates, etc.

        Args:
            numerator_mask: Boolean mask for numerator (e.g., y_pred == 1)
            denominator_mask: Optional mask for denominator (defaults to all True)

        Returns:
            Dict mapping group name to rate
        """
        results = {}

        for name in self.get_valid_groups():
            group_mask = self.get_mask(name)

            if denominator_mask is not None:
                denom_mask = group_mask & denominator_mask
            else:
                denom_mask = group_mask

            denom = np.sum(denom_mask)
            if denom == 0:
                results[name] = np.nan
            else:
                numer = np.sum(
                    group_mask
                    & numerator_mask
                    & (denominator_mask if denominator_mask is not None else True)
                )
                results[name] = numer / denom

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get summary information about groups."""
        valid = self.get_valid_groups()
        invalid = self.get_invalid_groups()

        return {
            "n_groups": self.n_groups,
            "n_valid_groups": len(valid),
            "n_invalid_groups": len(invalid),
            "is_intersectional": self.is_intersectional,
            "min_group_size": self.min_group_size,
            "group_sizes": self.get_group_sizes(),
            "valid_groups": valid,
            "invalid_groups": invalid,
        }


def _as_float(value: Any) -> float:
    """Read one group value as a float, or NaN when it is not a number.

    NaN is the could-not-check answer here and never 0.0: a value that is None,
    or text, or otherwise unreadable, has nothing to compare, and
    ``compute_max_difference`` counts how many pairs it really compared before
    returning anything. Accepting numpy scalars matters as much as rejecting
    text: an isinstance check against (int, float) would discard np.float32 and
    np.int64, which is the same defect running backwards (real evidence thrown
    away while reading as caution).
    """
    if value is None or isinstance(value, (str, bytes)):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compute_max_difference(group_values: Dict[str, float]) -> Tuple[float, str, str]:
    """
    Compute maximum absolute difference between any two groups.

    Args:
        group_values: Dict mapping group name to value

    Returns:
        Tuple of (max_diff, higher_group, lower_group). The pair is ordered
        by VALUE (the group with the higher value first), so callers that
        interpret the names as advantaged/disadvantaged get the right labels
        regardless of dict/alphabetical iteration order.
    """
    # A DIFFERENCE BETWEEN FEWER THAN TWO GROUPS DOES NOT EXIST. 0.0 is perfect
    # equality on this scale, so returning it here handed every caller a
    # measured "no gap" for a comparison that was never made. This helper is
    # used library-wide, which is exactly why it mattered.
    if len(group_values) < 2:
        warnings.warn(
            f"compute_max_difference: {len(group_values)} group(s) supplied, so no "
            f"difference exists to compute. Returning nan, not 0.0, which reads as "
            f"perfect equality.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), "", ""

    groups = list(group_values.keys())
    values = [_as_float(v) for v in group_values.values()]

    # SAME DEFECT ONE STEP IN: the guard above counts GROUPS, this one counts
    # COMPARABLE PAIRS. Two groups whose values are NaN are two groups, so the
    # guard passed, every pair was then skipped as undefined, and the untouched
    # 0.0 initialiser below was returned as the answer. mae_parity_difference
    # returned exactly 0.0 ("identical error for every group", the PASS value)
    # on data where one group's MAE could not be computed at all. Count the
    # pairs that were actually compared and refuse when there were none.
    undefined = [g for g, v in zip(groups, values) if np.isnan(v)]

    max_diff = 0.0
    max_pair = ("", "")
    n_compared = 0

    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i < j:
                v1, v2 = values[i], values[j]
                diff = abs(v1 - v2)
                # NaN here means the pair is not comparable: either value was
                # undefined, or both are infinite (inf minus inf is NaN), which
                # is a pair of unusable values and not a measured tie.
                if np.isnan(diff):
                    continue
                n_compared += 1
                if diff > max_diff:
                    max_diff = diff
                    # Order by value, not iteration order: higher first.
                    max_pair = (g1, g2) if v1 >= v2 else (g2, g1)

    if n_compared == 0:
        reason = (
            f"undefined value for group(s) {sorted(undefined)}"
            if undefined
            else "every pair holds non-finite values whose difference is undefined (inf minus inf)"
        )
        warnings.warn(
            f"compute_max_difference: no pair of groups could be compared "
            f"({len(group_values)} group(s) supplied, {reason}), so the difference "
            f"is NOT MEASURABLE. Returning nan, not 0.0, which reads as perfect "
            f"equality.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), "", ""

    if undefined:
        # The value below is a LOWER BOUND, not the disparity: the groups whose
        # value is undefined could sit anywhere, and dropping them can only
        # narrow the spread, so the more groups fall out the fairer the number
        # reads. Callers that must fail closed on this (equal_opportunity,
        # fnr, fpr, npv, precision, r2) refuse before they get here; the rest
        # at least get told what was left out.
        warnings.warn(
            f"compute_max_difference: group(s) {sorted(undefined)} have an undefined "
            f"value and were left out of the comparison. The returned difference "
            f"covers the remaining {len(group_values) - len(undefined)} group(s) only "
            f"and is a lower bound: the excluded group(s) are could not check, not a "
            f"measured match.",
            UserWarning,
            stacklevel=2,
        )

    return max_diff, max_pair[0], max_pair[1]


def compute_max_ratio(
    group_values: Dict[str, float], epsilon: float = 1e-10
) -> Tuple[float, str, str]:
    """
    Compute maximum ratio between any two groups.

    Args:
        group_values: Dict mapping group name to value
        epsilon: Small value to avoid division by zero

    Returns:
        Tuple of (max_ratio, numerator_group, denominator_group). ``nan`` when no
        ORDERED PAIR of groups could actually be divided, which is could-not-check
        and never 1.0.
    """
    # Same as compute_max_difference above: 1.0 is perfect PARITY on a ratio
    # scale, and it was the answer for a comparison with nothing to compare.
    if len(group_values) < 2:
        warnings.warn(
            f"compute_max_ratio: {len(group_values)} group(s) supplied, so no ratio "
            f"exists to compute. Returning nan, not 1.0, which reads as perfect parity.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), "", ""

    groups = list(group_values.keys())
    # BGL grade-1 G01, 2026-09-30. ``_as_float``, exactly as the sibling does.
    # np.isnan(None) and np.isnan('0.5') both raise TypeError, so this helper
    # CRASHED on the two absence doors compute_max_difference handles (measured:
    # compute_max_ratio({'a': None, 'b': 0.5}) raised "ufunc 'isnan' not
    # supported for the input types" while the sibling returned nan and said
    # which group was undefined). Text is what an ordinary CSV read gives you.
    values = [_as_float(v) for v in group_values.values()]

    max_ratio = 1.0
    max_pair = ("", "")

    # THE SIBLING'S PAIR COUNTER, WHICH THIS FUNCTION NEVER GOT. The comment
    # above says "same as compute_max_difference", and only the group-count half
    # of that fix was ever applied here: the PAIR-count half was not, so the
    # untouched ``max_ratio = 1.0`` initialiser was still returned as a measured
    # answer whenever every pair was skipped. 1.0 is perfect parity on this
    # scale, the strongest all-clear a ratio can state. Measured on this repo
    # before this change, both in silence:
    #
    #   compute_max_ratio({'a': nan, 'b': nan}) -> (1.0, '', '')
    #       two groups, so the len < 2 guard passed; every pair then skipped as
    #       undefined. The sibling returns nan and warns on identical input.
    #   compute_max_ratio({'a': 0.5, 'b': 0.0}) -> (1.0, '', '')
    #       group 'b' is NEVER SELECTED, the most extreme disparity this
    #       statistic can express, reported as perfect parity. The pair (a, b)
    #       is skipped by the ``abs(v2) > epsilon`` denominator guard and the
    #       mirror pair (b, a) divides to ~0.0, which never exceeds 1.0.
    #
    # Count the ordered pairs actually divided, and refuse when there were none.
    # The denominator-zero skips are counted SEPARATELY from the undefined ones,
    # because they are different facts about the data and the second one is the
    # alarming one.
    n_compared = 0
    undefined = [g for g, v in zip(groups, values) if np.isnan(v)]
    zero_denominators = [g for g, v in zip(groups, values) if not np.isnan(v) and abs(v) <= epsilon]

    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i != j:
                v1, v2 = values[i], values[j]
                if not (np.isnan(v1) or np.isnan(v2)) and abs(v2) > epsilon:
                    n_compared += 1
                    ratio = v1 / (v2 + epsilon)
                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_pair = (g1, g2)

    if n_compared == 0:
        parts = []
        if undefined:
            parts.append(f"undefined value for group(s) {sorted(undefined)}")
        if zero_denominators:
            parts.append(
                f"group(s) {sorted(zero_denominators)} have a value of zero, so they "
                f"cannot be a denominator (a group that is never selected at all)"
            )
        reason = "; ".join(parts) if parts else "no ordered pair had a usable denominator"
        warnings.warn(
            f"compute_max_ratio: no ordered pair of groups could be divided "
            f"({len(group_values)} group(s) supplied, {reason}), so the ratio is NOT "
            f"MEASURABLE. Returning nan, not 1.0, which reads as perfect parity.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan"), "", ""

    if undefined or zero_denominators:
        # A LOWER BOUND, for the same reason the sibling records: the groups left
        # out could sit anywhere, and dropping them can only narrow the spread,
        # so the more groups fall out the FAIRER the number reads. A zero-valued
        # group is the sharpest case of that, since it is the one whose ratio
        # against any other group is the largest possible.
        left_out = sorted(set(undefined) | set(zero_denominators))
        warnings.warn(
            f"compute_max_ratio: group(s) {left_out} could not serve as a "
            f"denominator (undefined value, or a value of zero) and so were left out "
            f"of {len(left_out)} of the comparisons. The returned ratio is a LOWER "
            f"BOUND on the disparity: those group(s) are could not check, not a "
            f"measured match.",
            UserWarning,
            stacklevel=2,
        )

    return max_ratio, max_pair[0], max_pair[1]
