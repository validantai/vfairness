"""
Data-balancing pre-processing (T-48 feasible subset).

This module adds fairness-aware data-balancing transformers that plug into the
EXISTING preprocessing plumbing without needing a new handler. Each transformer
inherits from :class:`BaseFeatureTransformer` and reuses one of the additive
hooks the intervention handler already auto-detects:

    * ``get_resampled_data(X, y) -> (X_res, y_res)`` for row-count changing
      techniques (mirrors :class:`Resampler`).
    * ``get_sample_weights(X) -> np.ndarray`` for weighting techniques
      (mirrors :class:`ReweightingTransformer`).

Techniques implemented here (numpy / scipy / sklearn only, no torch, no
imblearn, no SDV):

    1. Synthetic resampling family (SMOTE / ADASYN / Tomek) via
       :class:`SyntheticResampler` and its three fixed-method subclasses.
    2. Propensity weighting family (Propensity Score Weighting and Inverse
       Propensity Scoring) via :func:`propensity_weights` and the
       :class:`PropensityScoreWeighter` / :class:`InversePropensityWeighter`
       transformers.
    3. Counterfactual data augmentation via :func:`counterfactual_augment`
       and :class:`CounterfactualAugmenter`.

All randomness is seeded (``random_state=42`` by default) for reproducibility.

References:
    - Chawla, Bowyer, Hall, Kegelmeyer (2002): SMOTE.
    - He, Bai, Garcia, Li (2008): ADASYN.
    - Tomek (1976): Two modifications of CNN (Tomek links).
    - Rosenbaum & Rubin (1983): The central role of the propensity score.
    - Kusner et al. (2017); Garg et al. (2019): counterfactual data augmentation
      for fairness.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .transformers import BaseFeatureTransformer, TransformationResult


def _group_key_values(df: pd.DataFrame, attrs: List[str]) -> np.ndarray:
    """Intersectional group key per row, joined across ALL protected attributes.

    With a single protected attribute this is just that column as strings; with
    several it is the joint key (for example ``"race|gender"``), so balancing is
    done over the intersectional subgroup rather than only the first attribute.
    """
    if len(attrs) == 1:
        return df[attrs[0]].astype(str).values
    return df[list(attrs)].astype(str).agg("|".join, axis=1).values


# =====================================================================
# 1. Synthetic resampling family (SMOTE / ADASYN / Tomek)
# =====================================================================


class SyntheticResampler(BaseFeatureTransformer):
    """
    Balances (group x label) cells with synthetic minority samples (SMOTE /
    ADASYN) or by boundary cleaning (Tomek links). Row counts change; the
    resampled training set is exposed through the additive
    :meth:`get_resampled_data` hook (exactly like :class:`Resampler`), so the
    existing handler rebalances with no code changes.

    Cells are defined per protected group x label so that oversampling balances
    both the class distribution AND group representation. Synthetic points are
    built only from within a cell, so a synthetic minority sample always lies in
    the convex range of that cell's real minority samples.

    Args:
        protected_attributes: Names of protected attribute columns.
        method: One of ``'smote'``, ``'adasyn'`` (oversample by synthesis) or
            ``'tomek'`` (undersample by removing majority members of Tomek
            links).
        k_neighbors: Neighbours used when interpolating synthetic points.
        strategy: ``'oversample'`` (smote / adasyn) or ``'undersample'``
            (tomek). Kept for API symmetry with :class:`Resampler`; the active
            ``method`` selects the direction.
        random_state: Seed for reproducible sampling.

    References:
        - Chawla et al. (2002): SMOTE; He et al. (2008): ADASYN;
          Tomek (1976): Tomek links.

    Example:
        >>> r = SyntheticResampler(protected_attributes=['group'], method='smote')
        >>> r.fit(X, y)
        >>> X_bal, y_bal = r.get_resampled_data(X, y)
    """

    # Above this multiple of the overall removal rate, a group has been cleaned
    # away disproportionately and the result is no longer the caller's group
    # mix. Documented threshold, not a magic number inside a branch.
    _UNEVEN_REMOVAL_FACTOR = 2.0

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        method: str = "smote",
        k_neighbors: int = 5,
        strategy: str = "oversample",
        random_state: int = 42,
    ):
        super().__init__(protected_attributes)
        if method not in ("smote", "adasyn", "tomek"):
            raise ValueError("method must be 'smote', 'adasyn' or 'tomek', got %r" % (method,))
        if strategy not in ("oversample", "undersample"):
            raise ValueError("strategy must be 'oversample' or 'undersample', got %r" % (strategy,))
        self.method = method
        self.k_neighbors = int(k_neighbors)
        self.strategy = strategy
        self.random_state = int(random_state)
        # What the LAST get_resampled_data() disclosed. Kept so a second call
        # REPLACES its predecessor's disclosures instead of stacking a second
        # copy of them onto fit_result.warnings.
        self._disclosed: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "SyntheticResampler":
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs
        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self.feature_names_out_ = feature_cols.copy()
        self.fit_result = TransformationResult(
            method="synthetic_resampling_%s" % self.method,
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            fit_metrics={
                "method": self.method,
                "strategy": self.strategy,
                "k_neighbors": self.k_neighbors,
            },
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return features unchanged. Use get_resampled_data() for balanced rows."""
        self._check_is_fitted()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)
        feature_cols = self._extract_feature_columns(df, self._fitted_protected_attributes())
        if was_dataframe:
            return df[feature_cols].copy()
        return df[feature_cols].values

    # internals

    def _numeric_feature_cols(self, df: pd.DataFrame) -> List[str]:
        feature_cols = self._extract_feature_columns(df, self._fitted_protected_attributes())
        return [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    #: Fraction of a column's recorded values that must coerce to a finite
    #: number before the column is judged to be NUMBERS STORED AS TEXT rather
    #: than a category. Half, so a column of 59 numbers and one "N/A" is caught
    #: (that single token is what makes pandas type the whole column ``object``)
    #: while a genuine category column, of which nothing coerces, is not.
    _TEXT_NUMERIC_FRACTION = 0.5

    def _numbers_stored_as_text(
        self,
        df: pd.DataFrame,
        numeric_feats: List[str],
    ) -> List[str]:
        """Feature columns holding NUMBERS AS TEXT, which this class skips.

        ``pd.api.types.is_numeric_dtype`` ROUTES; it coerces nothing. A feature
        column that arrives as text is therefore treated as a category and
        COPIED VERBATIM into every synthetic row, and it is also absent from the
        matrix the nearest-neighbour search measures distance in, so the
        interpolation happens in fewer dimensions than the caller supplied.

        G07 2026-09-30, and this is PARTIAL LOSS PASSING WHERE TOTAL LOSS IS
        REFUSED. With EVERY feature column text-typed, ``get_resampled_data``
        already refuses and says so. Measured on the same 60-row frame with ONE
        of two feature columns text-typed: 60 rows in, 80 rows out, groups
        balanced {'m': 40, 'f': 40}, ``fit_result.warnings == []``,
        ``fit_metrics`` carrying nothing and zero Python warnings, while every
        one of the 20 synthetic rows carried an ``f1`` value that was a verbatim
        copy of a real row's (``set(synthetic.f1) <= set(real.f1)``) and the
        neighbour search ran over one dimension instead of two. A single stray
        "N/A" in an otherwise numeric column reproduces it exactly, because that
        one token makes pandas type the whole column ``object``, and a plain
        ``pd.read_csv`` is how it arrives.

        A genuinely CATEGORICAL feature column is a different case and is NOT
        reported: copying it verbatim is the only honest thing to do with it,
        and warning on every run is warning on none.
        """
        feature_cols = self._extract_feature_columns(df, self._fitted_protected_attributes())
        skipped = [c for c in feature_cols if c not in set(numeric_feats)]
        as_text: List[str] = []
        for col in skipped:
            recorded = df[col].dropna()
            if not len(recorded):
                continue
            coerced = pd.to_numeric(recorded, errors="coerce")
            n_finite = int(
                np.isfinite(pd.to_numeric(coerced, errors="coerce").to_numpy(float)).sum()
            )
            if n_finite >= self._TEXT_NUMERIC_FRACTION * len(recorded):
                as_text.append(col)
        return as_text

    def _disclose_numbers_stored_as_text(self, df: pd.DataFrame, numeric_feats: List[str]) -> None:
        """Say which feature columns were skipped although they hold numbers.

        ABOVE THE METHOD DISPATCH in ``get_resampled_data``, for the reason the
        total-loss guard beside it records: every method in this class needs a
        distance between rows, so a skipped numeric column degrades smote,
        adasyn and tomek alike, and a guard below the tomek return covers two of
        the three.
        """
        as_text = self._numbers_stored_as_text(df, numeric_feats)
        if not as_text:
            return
        message = (
            f"{len(as_text)} feature column(s) hold numbers stored as TEXT and were "
            f"treated as categories: {as_text}. They are NOT part of the distance the "
            f"nearest-neighbour search measures, and every synthetic row copies its base "
            f"row's value for them verbatim instead of interpolating, so the synthesis "
            f"ran in {len(numeric_feats)} of {len(numeric_feats) + len(as_text)} numeric "
            f"dimension(s). Convert them with pd.to_numeric before fitting; a single "
            f"non-numeric token (an 'N/A') types the whole column as text."
        )
        self._warn_fit_result(message)
        warnings.warn(f"{type(self).__name__}: {message}", stacklevel=3)
        if self.fit_result is not None:
            self.fit_result.fit_metrics["feature_columns_numbers_stored_as_text"] = list(as_text)

    def _warn_fit_result(self, message: str) -> None:
        """Record a disclosure on the fit result, the channel a caller reads.

        Kept in one place so every refusal in this class reaches the same
        surface and the fitted-ness guard is not repeated per call site.
        """
        if self.fit_result is None:
            raise RuntimeError("get_resampled_data() requires fit() to have run first")
        self.fit_result.warnings.append(message)
        self._disclosed.append(message)

    def _begin_disclosures(self) -> None:
        """Drop the PREVIOUS call's disclosures before recording this call's.

        Without this, three calls to get_resampled_data() leave three copies of
        the same sentence on fit_result.warnings, and a reader counting
        warnings reads a worsening run where nothing changed.
        """
        if self.fit_result is None:
            raise RuntimeError("get_resampled_data() requires fit() to have run first")
        stale = self._disclosed
        self._disclosed = []
        if stale:
            self.fit_result.warnings = [w for w in self.fit_result.warnings if w not in stale]

    @staticmethod
    def _row_keys(M: np.ndarray) -> List[Tuple[Any, ...]]:
        """Hashable per-row key over a float matrix, NaN kept as its own value."""
        return [
            tuple("nan" if np.isnan(v) else float(v) for v in row)
            for row in np.asarray(M, dtype=float).tolist()
        ]

    @classmethod
    def _n_verbatim_copies(cls, real_M: np.ndarray, synth_M: np.ndarray) -> int:
        """How many SYNTHESISED rows are byte-identical to a real row of the cell.

        This is the honest measure of "was anything actually synthesised here",
        and it replaces a count of DISTINCT INPUT rows. `n_distinct < 2` caught a
        one-row cell and missed the far commoner shape: a cell of 9 identical
        rows plus 1 different one has two distinct rows, so it passed the old
        predicate, while every interpolation that happened to pick two of the 9
        returned one of them unchanged. Measured 2026-09-17 on exactly that
        frame: 10 of 10 added rows were verbatim copies, with no warning, under
        a closing sentence that said "Every cell that does exist was balanced to
        20 rows."
        """
        real = set(cls._row_keys(real_M))
        return int(sum(1 for key in cls._row_keys(synth_M) if key in real))

    @staticmethod
    def _neighbour_matrix(M: np.ndarray) -> np.ndarray:
        """A finite matrix for the NEIGHBOUR SEARCH only, never for output values.

        THREE STATES, NEVER TWO. ``get_resampled_data`` used to build its whole
        working matrix with ``.fillna(0.0)``, so a feature that was never
        measured entered the interpolation as a real observation of zero and
        left it as a synthetic feature value written into the returned frame
        (measured 2026-09-16: group b's f0 was entirely NaN and the resampled
        frame reported a group mean of exactly 0.000, with 30 rows outside the
        convex range of the real data the class docstring promises).

        The values are kept as NaN now, so "not measured" propagates through the
        interpolation. A distance still has to be computed to pick a neighbour,
        and sklearn refuses NaN, so THIS copy substitutes the column mean of the
        values that WERE measured and drops any column with nothing to measure.
        Choosing a neighbour is a choice, not a measurement, and the substituted
        numbers never reach the caller: only distances use them. On data with no
        missing values this returns the matrix unchanged, so healthy synthesis
        is bit-for-bit what it was.
        """
        if M.size == 0 or M.shape[1] == 0:
            return M
        missing = np.isnan(M)
        if not missing.any():
            return M
        keep = [c for c in range(M.shape[1]) if not missing[:, c].all()]
        if not keep:
            # Nothing measured anywhere in this matrix: every distance is equally
            # unknown, so make them equal rather than invent a spread.
            return np.zeros((M.shape[0], 1), dtype=float)
        D = np.asarray(M[:, keep], dtype=float).copy()
        col_mean = np.nanmean(D, axis=0)
        gaps = np.isnan(D)
        if gaps.any():
            idx = np.where(gaps)
            D[idx] = np.take(col_mean, idx[1])
        return D

    @staticmethod
    def _allocate(weights: np.ndarray, n_synth: int, rng) -> np.ndarray:
        """Turn per-point weights into a list of base positions of length n_synth.

        A point with a larger weight is chosen as a synthesis base more often.
        This is how ADASYN allocates more synthetic points to harder-to-learn
        (boundary) minority samples.
        """
        weights = np.asarray(weights, dtype=float)
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / total
        counts = np.floor(weights * n_synth).astype(int)
        remainder = int(n_synth - counts.sum())
        if remainder > 0:
            frac = weights * n_synth - counts
            extra = np.argsort(frac)[::-1][:remainder]
            counts[extra] += 1
        base_pos = np.repeat(np.arange(len(weights)), counts)
        if base_pos.size:
            rng.shuffle(base_pos)
        return base_pos

    def _adasyn_weights(
        self,
        minority_M: np.ndarray,
        other_M: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Per-minority-point density weight r_i: the fraction of each point's k
        nearest neighbours (over minority plus other-class points) that belong to
        the other class. Boundary points get a higher weight."""
        from sklearn.neighbors import NearestNeighbors

        n_min = minority_M.shape[0]
        if n_min == 0:
            return np.zeros(0)
        if other_M is None or other_M.shape[0] == 0:
            return np.ones(n_min) / n_min
        combined = self._neighbour_matrix(np.vstack([minority_M, other_M]))
        is_other = np.concatenate(
            [
                np.zeros(n_min, dtype=bool),
                np.ones(other_M.shape[0], dtype=bool),
            ]
        )
        k_eff = int(min(k, combined.shape[0] - 1))
        if k_eff < 1:
            return np.ones(n_min) / n_min
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(combined)
        neigh = nn.kneighbors(combined[:n_min], return_distance=False)[:, 1:]
        r = is_other[neigh].mean(axis=1).astype(float)
        if r.sum() <= 0:
            return np.ones(n_min) / n_min
        return r / r.sum()

    def _oversample_cell(
        self,
        cell_M: np.ndarray,
        n_synth: int,
        rng,
        other_M: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (synthetic_numeric_matrix, base_positions_within_cell)."""
        n_min, d = cell_M.shape
        if n_synth <= 0 or n_min == 0:
            return np.empty((0, d)), np.empty((0,), dtype=int)
        if n_min == 1:
            # Cannot interpolate a single point; duplicate it.
            return np.repeat(cell_M, n_synth, axis=0), np.zeros(n_synth, dtype=int)

        from sklearn.neighbors import NearestNeighbors

        k_eff = int(min(self.k_neighbors, n_min - 1))
        # Distances only: NaN feature values stay NaN in cell_M so that an
        # unmeasured value propagates into the synthetic row instead of being
        # silently replaced by 0.0. See _neighbour_matrix.
        cell_D = self._neighbour_matrix(cell_M)
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(cell_D)
        neigh = nn.kneighbors(cell_D, return_distance=False)[:, 1:]

        if self.method == "adasyn" and other_M is not None and other_M.shape[0] > 0:
            weights = self._adasyn_weights(cell_M, other_M, self.k_neighbors)
            base_pos = self._allocate(weights, n_synth, rng)
        else:
            base_pos = rng.integers(0, n_min, size=n_synth)

        synth = np.empty((n_synth, d))
        for s in range(n_synth):
            i = int(base_pos[s])
            j = int(neigh[i, rng.integers(0, neigh.shape[1])])
            gap = float(rng.random())
            synth[s] = cell_M[i] + gap * (cell_M[j] - cell_M[i])
        return synth, base_pos

    def _tomek(
        self,
        df: pd.DataFrame,
        y_arr: np.ndarray,
        numeric_feats: List[str],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Remove the majority-class member of every Tomek link (a mutual
        nearest-neighbour pair of opposite class), while keeping both classes.

        THREE STATES, NEVER TWO. This method used to have no disclosure channel
        at all: the ``n < 2`` early return, the ``.fillna(0.0)`` and the
        balancing outcome never touched ``fit_result.warnings``. Measured
        2026-09-17, before this:

        * a frame whose only numeric column was entirely unmeasured had that
          column read as 40 real observations of 0.0 while deciding which rows
          to DELETE, and the run reported nothing;
        * {'a': 6, 'b': 2} came back {'a': 5, 'b': 1} and, on other frames,
          lost a protected group outright, with ``warnings == []``. A class
          could never disappear (there is a rescue for that); a protected group
          could, which is the fairness-relevant half.
        """
        from sklearn.neighbors import NearestNeighbors

        n = len(df)
        attrs = self._fitted_protected_attributes()
        group_vals = np.asarray([str(g) for g in _group_key_values(df, attrs)])
        # AN UNRECORDED GROUP IS NOT A GROUP, and this branch never learned it
        # (BGL6 F12, 2026-09-28). ``get_resampled_data`` dispatches here BEFORE it
        # computes ``recorded``, so for ``method='tomek'`` the absence of a record
        # was still a group key, and the repair the sibling path got on 2026-09-27
        # could not reach it. ``_group_key_values`` stringifies, so a row whose
        # protected value was never recorded became the group key 'None'.
        #
        # Measured on 130 rows (60 'm', 50 'f', 20 with gender None) through fit +
        # get_resampled_data: group_totals_before {'None': 20, ...},
        # group_totals_after {'None': 19, ...}, real rows deleted and accounted to
        # that pseudo-group, with fit_result.warnings == [] and no Python warning.
        #
        # They are NOT dropped from the frame, for the same reason the sibling path
        # gives: removing real rows would be a second, larger defect. They are
        # excluded from the group TOTALS, counted under their own key, and disclosed.
        if n:
            recorded = df[list(attrs)].notna().all(axis=1).to_numpy()
        else:
            recorded = np.zeros(0, dtype=bool)
        n_unrecorded = int((~recorded).sum())
        before_counts = {
            str(g): int(((group_vals == g) & recorded).sum())
            for g in np.unique(group_vals[recorded])
        }
        if self.fit_result is not None:
            self.fit_result.fit_metrics["group_totals_before"] = dict(before_counts)
            if n_unrecorded:
                self.fit_result.fit_metrics["n_rows_without_a_recorded_group"] = n_unrecorded

        def _finish(keep_mask: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
            kept_recorded = recorded[keep_mask]
            after = {
                str(g): int(((group_vals[keep_mask] == g) & kept_recorded).sum())
                for g in before_counts
            }
            if self.fit_result is not None:
                self.fit_result.fit_metrics["group_totals_after"] = after
                self.fit_result.fit_metrics["n_rows_removed"] = int(n - int(keep_mask.sum()))
                if n_unrecorded:
                    self.fit_result.fit_metrics["n_rows_without_a_recorded_group_after"] = int(
                        (~kept_recorded).sum()
                    )
            return df.iloc[keep_mask].reset_index(drop=True), y_arr[keep_mask]

        if n_unrecorded:
            self._warn_fit_result(
                f"{n_unrecorded} of {n} row(s) have no recorded value for {list(attrs)}, "
                f"so they are NOT a protected group and are excluded from the group "
                f"totals. Boundary cleaning still reads their features, so some of them "
                f"may be removed as members of a Tomek link, and the count that remains "
                f"is reported under n_rows_without_a_recorded_group_after. They are a "
                f"could-not-check, not a group that matches any other row."
            )

        if n < 2:
            self._warn_fit_result(
                f"no boundary cleaning was performed: {n} row(s) is fewer than the two a "
                "mutual nearest-neighbour pair needs, so no Tomek link can exist. The "
                "data is returned unchanged and the groups are as unbalanced as they were."
            )
            return _finish(np.ones(n, dtype=bool))

        M = df[numeric_feats].astype(float).values
        gaps = np.isnan(M)
        if not gaps.size or bool(gaps.all(axis=0).all()):
            self._warn_fit_result(
                "no boundary cleaning was performed: not one value was measured in any of "
                f"the numeric feature column(s) {list(numeric_feats)[:8]}, so there is no "
                "distance with which to find a Tomek link. The data is returned unchanged."
            )
            return _finish(np.ones(n, dtype=bool))
        if bool(gaps.any()):
            per_col = [
                (str(c), int(gaps[:, ci].sum()))
                for ci, c in enumerate(numeric_feats)
                if bool(gaps[:, ci].any())
            ]
            detail = ", ".join(f"{c!r} in {k} of {n} row(s)" for c, k in per_col[:8])
            message = (
                f"{len(per_col)} feature column(s) hold values that were never measured: "
                f"{detail}{' ...' if len(per_col) > 8 else ''}. A value that was not "
                "measured is not an observation of 0.0, so the neighbour search uses the "
                "column mean of the values that WERE measured and drops any column with "
                "nothing to measure; a row deleted here was chosen partly on an unmeasured "
                "cell. Impute before cleaning if that matters."
            )
            self._warn_fit_result(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)
        # Distances only, and NaN never enters as a real zero. See
        # _neighbour_matrix: on data with no missing values this is the same
        # matrix and the cleaning is bit-for-bit what it was.
        D = self._neighbour_matrix(M)
        nn = NearestNeighbors(n_neighbors=2).fit(D)
        neigh = nn.kneighbors(D, return_distance=False)[:, 1]

        classes, counts = np.unique(y_arr, return_counts=True)
        majority_class = classes[int(np.argmax(counts))]

        to_remove = set()
        for i in range(n):
            j = int(neigh[i])
            if int(neigh[j]) == i and y_arr[i] != y_arr[j]:
                if y_arr[i] == majority_class:
                    to_remove.add(i)
                elif y_arr[j] == majority_class:
                    to_remove.add(j)

        keep_mask = np.ones(n, dtype=bool)
        for idx in to_remove:
            keep_mask[idx] = False
        # Never let a class disappear entirely.
        for c in classes:
            if not np.any(y_arr[keep_mask] == c):
                for idx in list(to_remove):
                    if y_arr[idx] == c:
                        keep_mask[idx] = True
        # ... and never let a protected GROUP disappear entirely, which is the
        # same rescue for the axis this module exists to protect. A group that
        # is cleaned away is absent from every fairness number computed after.
        rescued: List[str] = []
        for g in before_counts:
            if not np.any(group_vals[keep_mask] == g):
                rescued.append(g)
                for idx in list(to_remove):
                    if group_vals[idx] == g:
                        keep_mask[idx] = True
        if rescued:
            message = (
                f"boundary cleaning would have removed EVERY row of protected group(s) "
                f"{sorted(rescued)[:8]}; those rows were kept instead. A group cleaned "
                "away is absent from every fairness number computed on the result, so "
                "this run is not the cleaning that was asked for."
            )
            self._warn_fit_result(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)

        after_counts = {g: int((group_vals[keep_mask] == g).sum()) for g in before_counts}
        n_removed = int(n - int(keep_mask.sum()))
        if n_removed == 0:
            # Two different facts produce the same zero, and saying the wrong one
            # would be the defect in miniature: "no link was found" is false when
            # every selected row was handed back by a rescue.
            cause = (
                "no Tomek link was found"
                if not to_remove
                else f"all {len(to_remove)} row(s) a Tomek link selected were kept by the "
                "class / protected-group preservation rescue"
            )
            self._warn_fit_result(
                f"{cause}, so no row was removed: the data is returned UNCHANGED and the "
                "groups are as unbalanced as they were. An unchanged frame is not a "
                "cleaned one."
            )
        else:
            thin = sorted(g for g, k in after_counts.items() if k < 2)
            overall = n_removed / n
            lopsided = sorted(
                g
                for g, k in before_counts.items()
                if k and (k - after_counts[g]) / k > self._UNEVEN_REMOVAL_FACTOR * overall
            )
            if thin or lopsided:
                message = (
                    f"boundary cleaning removed {n_removed} of {n} row(s) unevenly across "
                    f"protected groups: {before_counts} became {after_counts}."
                )
                if thin:
                    message += (
                        f" Group(s) {thin[:8]} are left with fewer than two rows, so every "
                        "per-group statistic computed on the result is a could-not-check, "
                        "not a measurement."
                    )
                if lopsided:
                    message += (
                        f" Group(s) {lopsided[:8]} lost more than "
                        f"{self._UNEVEN_REMOVAL_FACTOR:g}x the overall removal rate of "
                        f"{overall:.1%}. Tomek-link cleaning removes boundary rows; it does "
                        "not balance groups, and here it moved their representation."
                    )
                self._warn_fit_result(message)
                warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)
        return _finish(keep_mask)

    def get_resampled_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Any,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Return a balanced ``(X_resampled, y_resampled)``. ``X_resampled`` keeps
        all input columns (features and protected attributes); ``y_resampled`` is
        aligned. Synthetic rows copy their base row's protected-attribute and
        non-numeric values (so the group label is preserved) and overwrite only
        the numeric feature columns with the interpolated values.

        Three states, never two, on BOTH halves of the ``(group, label)`` cell
        key. A row whose protected value was never recorded, and a row whose
        label was never recorded, are not a cell: no synthetic row is created for
        them or from them, they are returned exactly as they came in, and the
        counts reach both ``fit_result.warnings`` and
        ``fit_result.fit_metrics`` (``n_rows_without_a_recorded_group``,
        ``n_rows_without_a_recorded_label``) as well as a ``UserWarning``. The
        result is balanced over the rows that carry both.
        """
        self._check_is_fitted()
        self._begin_disclosures()
        df = self._to_dataframe(X, self.feature_names_in_).reset_index(drop=True)
        y_arr = np.asarray(y)
        attrs = self._fitted_protected_attributes()
        numeric_feats = self._numeric_feature_cols(df)
        rng = np.random.default_rng(self.random_state)

        # THREE STATES, NEVER TWO, and ABOVE the method dispatch. Every method
        # in this class needs a distance between rows, so "no numeric feature
        # column" makes every one of them a silent no-op. This guard used to sit
        # BELOW the tomek return, so TomekResampler reproduced verbatim, at the
        # same public entry, the defect SMOTE and ADASYN had just been repaired
        # for: measured 2026-09-17, 40 rows in, 40 rows out, fit_result.warnings
        # == [], while SMOTE on the identical frame disclosed. When branches
        # share a precondition the guard goes ABOVE the selection.
        if not numeric_feats:
            # Returning the frame unchanged is the only honest arithmetic, but
            # returning it SILENTLY is the no-op-reports-success shape: the
            # caller asked for balanced groups and gets back what it passed in.
            unmeasurable = (
                "no boundary cleaning was performed: none of the feature columns is "
                "numeric, so no nearest neighbour and no Tomek link can be found"
                if self.method == "tomek"
                else "no synthesis was performed: none of the feature columns is numeric, "
                "so there is nothing to interpolate between"
            )
            self._warn_fit_result(
                unmeasurable + ". The data is returned unchanged and the groups are as "
                "unbalanced as they were."
            )
            return df, y_arr

        # PARTIAL loss of the same kind, disclosed for the same reason and in the
        # same place: above the dispatch. See _numbers_stored_as_text.
        self._disclose_numbers_stored_as_text(df, numeric_feats)

        if self.method == "tomek":
            return self._tomek(df, y_arr, numeric_feats)

        group_vals = _group_key_values(df, attrs)
        # AN UNRECORDED GROUP IS NOT A GROUP, at the group key itself, exactly as
        # `_fit_propensity_model` in this module already does it (2026-09-17,
        # "counterfactual_augment in this module was repaired for exactly this
        # shape; the propensity family was not"). `_group_key_values`
        # stringifies, so a row whose protected value was never recorded became
        # the group key 'None' / 'nan' and then had SYNTHETIC ROWS INVENTED FOR
        # IT. BGL5 2026-09-27, measured on 130 rows (60 'm', 50 'f', 20 with
        # gender None) through fit + get_resampled_data: 180 rows out with
        # value_counts(dropna=False) 'm 60, f 60, None 60', so 40 synthetic rows
        # had been interpolated into the absence of a record and one third of
        # the returned training set belonged to it, with fit_result.warnings ==
        # [] and no Python warning. Those rows are now excluded from the cells,
        # so no row is invented for them, and they are disclosed.
        #
        # They are NOT dropped from the returned frame: removing real rows
        # would be a second, larger defect. They come back exactly as they went
        # in, and the warning says the result is not balanced over them.
        if len(df):
            recorded = df[list(attrs)].notna().all(axis=1).to_numpy()
        else:
            recorded = np.zeros(0, dtype=bool)
        n_unrecorded = int((~recorded).sum())
        # AN UNRECORDED LABEL IS NOT A LABEL VALUE, which is the OTHER HALF of the
        # same (group, label) cell key. The guard above closed the group half and
        # the label half stayed open, so the identical defect was still live one
        # axis across. BGL7 F14, 2026-09-29, measured on the same 130 rows with
        # gender fully recorded and y = np.array(([0, 1] * 55) + [None] * 20,
        # dtype=object): 130 rows in, 150 rows OUT, output label counts
        # {0: 60, 1: 60, None: 30}, so TEN synthetic rows were interpolated into a
        # cell whose label was never recorded. Python warnings: none at all. The
        # only trace called the absence a label VALUE: "not group-balanced: 1
        # (group, label) cell(s) have no examples to resample from ... (m,
        # label=None). Every cell that does exist was balanced to 30 rows."
        # pd.NA behaved identically. With float np.nan it was far worse, because
        # every NaN is a distinct value and therefore its own cell: the same 130
        # rows became 720, of which 600 carried no label (83 percent of the
        # returned training set), disclosed only as "20 (group, label) cell(s)
        # have no examples" plus a duplication warning about 580 rows, neither of
        # which says the label was never recorded.
        #
        # Same treatment as the group half, for the same reason: the rows are NOT
        # dropped (removing real rows would be a second, larger defect), they are
        # excluded from the cells so nothing is invented for them or from them,
        # and the count is disclosed on both channels.
        if len(df):
            labelled = ~np.asarray(pd.isna(y_arr), dtype=bool).reshape(-1)
        else:
            labelled = np.zeros(0, dtype=bool)
        n_unlabelled = int((~labelled[: len(df)]).sum())
        # Accumulate as lists, then freeze to arrays under a separate name: the two
        # stages hold different element types, so reusing one name made the dict
        # simultaneously a list-of-int map and an ndarray map.
        buckets: Dict[Any, List[int]] = {}
        for i in range(len(df)):
            if not recorded[i] or not labelled[i]:
                continue
            buckets.setdefault((group_vals[i], y_arr[i]), []).append(i)
        cells: Dict[Any, np.ndarray] = {k: np.asarray(v) for k, v in buckets.items()}
        if n_unrecorded:
            message = (
                f"{n_unrecorded} of {len(df)} row(s) have no recorded value for "
                f"{list(attrs)}. They are not a group, so NO synthetic row was created "
                f"for them and none was created from them; they are returned exactly as "
                f"they came in. The result is balanced over the recorded groups only, and "
                f"the unrecorded rows are a could-not-check, not a group that matches any "
                f"other."
            )
            self._warn_fit_result(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)
        if n_unlabelled:
            message = (
                f"{n_unlabelled} of {len(df)} row(s) have no recorded label. An absent "
                f"label is not a label VALUE, so those rows are not a (group, label) cell: "
                f"NO synthetic row was created for them and none was created from them; "
                f"they are returned exactly as they came in. The result is balanced over "
                f"the rows that carry a label only, and the unlabelled rows are a "
                f"could-not-check, not a class that matches any other."
            )
            self._warn_fit_result(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)
        if self.fit_result is not None:
            # Both counts on the metrics channel as well, because a warning can be
            # filtered and a number cannot. `_tomek` above already records
            # n_rows_without_a_recorded_group; this path recorded neither, so the
            # label axis had no counterpart to read and the group axis had none
            # either.
            if n_unrecorded:
                self.fit_result.fit_metrics["n_rows_without_a_recorded_group"] = n_unrecorded
            if n_unlabelled:
                self.fit_result.fit_metrics["n_rows_without_a_recorded_label"] = n_unlabelled
        if not cells:
            # A silent unchanged frame is the no-op-reports-success shape, and
            # this return is the one an empty input takes. Measured 2026-09-27:
            # 0 rows in, 0 rows out, fit_result.warnings == [], byte-identical
            # to a run that balanced successfully.
            #
            # The two causes are stated separately because the frame-holds-no-
            # rows sentence was published for a 60-row frame whose protected
            # column was entirely NULL (BGL5 2026-09-27; before the guard above
            # that frame reached the end and reported "every one of the 2
            # (group, label) cell(s) already holds 30 row(s), so there was
            # nothing to add", a group-balance statement about a frame in which
            # no group was recorded at all).
            #
            # And a fourth cause, for the same reason again: an entirely
            # UNRECORDED LABEL column now empties `cells` too, and saying "none
            # has a recorded value for ['gender']" about a frame whose gender is
            # fully recorded would be the same wrong sentence one axis across.
            if not len(df):
                cause = (
                    "the frame holds no rows, so there is no (group, label) cell to "
                    "balance and nothing to interpolate between"
                )
            elif not recorded.any():
                cause = (
                    f"none of the {len(df)} row(s) has a recorded value for {list(attrs)}, "
                    "so there is no (group, label) cell to balance"
                )
            elif not labelled.any():
                cause = (
                    f"none of the {len(df)} row(s) has a recorded label, so there is no "
                    "(group, label) cell to balance"
                )
            else:
                cause = (
                    f"not one of the {len(df)} row(s) carries BOTH a recorded value for "
                    f"{list(attrs)} and a recorded label, so there is no (group, label) "
                    "cell to balance"
                )
            self._warn_fit_result(
                "no synthesis was performed: "
                + cause
                + ". The data is returned UNCHANGED, so every downstream fairness number "
                "is the one the un-resampled data gives."
            )
            return df, y_arr

        target = max(len(v) for v in cells.values())
        # NaN is PRESERVED here (it used to be .fillna(0.0)): an unmeasured
        # feature must not enter the interpolation as a real observation of zero.
        # See _neighbour_matrix for how the neighbour search stays computable.
        Mnum_all = df[numeric_feats].astype(float).values

        # THREE STATES, NEVER TWO. This balances (group x label) CELLS, and an
        # ABSENT cell cannot be balanced: there is nothing to interpolate from.
        # When a group never appears with one of the labels, every cell that does
        # exist is filled to `target` and the GROUP totals still come out uneven,
        # which is correct arithmetic and looks exactly like a failed balance.
        #
        # Measured 2026-09-07 before this warning existed: 200 rows in group A
        # and 2 in group B, where B carried only label 1, returned A=218, B=109
        # with `fit_result.warnings == []`. A caller who asked a balancer to
        # balance got unbalanced groups back and had no way to find out. The
        # numbers were right; the silence was the defect.
        #
        # Reported, not repaired: inventing rows for a (group, label) pair with
        # no examples would be fabricating the very thing the caller wants
        # measured.
        observed_groups = {g for g, _ in cells}
        observed_labels = {lab for _, lab in cells}
        absent = sorted(
            (str(g), str(lab))
            for g in observed_groups
            for lab in observed_labels
            if (g, lab) not in cells
        )
        missing_bases: List[Dict[str, Any]] = []
        for (g, lab), pos in cells.items():
            if int(target - len(pos)) <= 0:
                continue
            gaps = np.isnan(Mnum_all[pos])
            for ci, col in enumerate(numeric_feats):
                n_gap = int(gaps[:, ci].sum())
                if n_gap:
                    missing_bases.append(
                        {
                            "group": str(g),
                            "label": str(lab),
                            "column": str(col),
                            "n_missing_bases": n_gap,
                            "n_bases": int(len(pos)),
                        }
                    )

        # group -> {label: positions} so ADASYN can see same-group other-class points.
        group_positions: Dict[Any, Dict[Any, np.ndarray]] = {}
        for (g, lab), pos in cells.items():
            group_positions.setdefault(g, {})[lab] = pos

        # THREE STATES, NEVER TWO, second face. A "synthetic" row that is
        # byte-identical to a real row of its cell is a COPY, and a cell filled
        # with copies is balanced by row count only. This is measured on the
        # rows that were actually produced, AFTER synthesis, because the
        # property is about the output: the earlier predicate counted DISTINCT
        # INPUT rows (`n_distinct < 2`) and so caught a one-row cell while
        # missing a cell of 9 identical rows plus 1 different one, where 10 of
        # 10 added rows came back as verbatim copies with no warning at all
        # (measured 2026-09-17). The rows are still produced (dropping them
        # would silently unbalance the result); what changes is that the caller
        # is told how many of them are copies.
        duplicated_cells: List[Dict[str, Any]] = []
        synth_frames = []
        synth_labels = []
        for (g, lab), pos in cells.items():
            n_synth = int(target - len(pos))
            if n_synth <= 0:
                continue
            cell_M = Mnum_all[pos]
            other_M = None
            if self.method == "adasyn":
                other_parts = [p for lab2, p in group_positions[g].items() if lab2 != lab]
                if other_parts:
                    other_pos = np.concatenate(other_parts)
                    other_M = Mnum_all[other_pos]
            synth_M, base_pos = self._oversample_cell(cell_M, n_synth, rng, other_M)
            if synth_M.shape[0] == 0:
                continue
            n_copies = self._n_verbatim_copies(cell_M, synth_M)
            if n_copies:
                duplicated_cells.append(
                    {
                        "group": str(g),
                        "label": str(lab),
                        "n_real_rows": int(len(pos)),
                        "n_distinct_real_rows": int(len(pd.DataFrame(cell_M).drop_duplicates())),
                        "n_synthesised_rows": int(synth_M.shape[0]),
                        "n_duplicated_rows": int(n_copies),
                    }
                )
            global_base = pos[base_pos]
            rows = df.iloc[global_base].reset_index(drop=True).copy()
            for ci, col in enumerate(numeric_feats):
                rows[col] = synth_M[:, ci]
            synth_frames.append(rows)
            synth_labels.append(np.full(synth_M.shape[0], lab))

        if absent:
            detail = ", ".join(f"({g}, label={lab})" for g, lab in absent)
            qualifier = (
                f"Every cell that does exist was filled to {target} rows, "
                f"{len(duplicated_cells)} of them by exact duplication rather than by "
                "synthesis (see the duplication warning)."
                if duplicated_cells
                else f"Every cell that does exist was balanced to {target} rows."
            )
            self._warn_fit_result(
                f"not group-balanced: {len(absent)} (group, label) cell(s) have no "
                f"examples to resample from, so they stay empty and the group totals "
                f"remain uneven: {detail}. " + qualifier
            )

        if duplicated_cells:
            detail = ", ".join(
                f"({c['group']}, label={c['label']}): {c['n_distinct_real_rows']} distinct "
                f"real row(s) in {c['n_real_rows']}, {c['n_duplicated_rows']} of "
                f"{c['n_synthesised_rows']} added row(s) are verbatim copies"
                for c in duplicated_cells[:8]
            )
            more = " ..." if len(duplicated_cells) > 8 else ""
            n_copied = sum(int(c["n_duplicated_rows"]) for c in duplicated_cells)
            message = (
                f"balanced by DUPLICATION, not by synthesis, in {len(duplicated_cells)} "
                f"(group, label) cell(s), {n_copied} row(s) in total: {detail}{more}. "
                "Interpolating between two identical real rows returns that row, so those "
                "added rows are verbatim copies of a real observation. Those cells are "
                "balanced by ROW COUNT only: the effective sample size is unchanged, and "
                "every per-group statistic computed on the result is as uncertain as it "
                "was before."
            )
            self._warn_fit_result(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)

        if missing_bases:
            detail = ", ".join(
                f"{m['column']!r} missing in {m['n_missing_bases']} of {m['n_bases']} "
                f"synthesis bases of cell ({m['group']}, label={m['label']})"
                for m in missing_bases[:8]
            )
            more = " ..." if len(missing_bases) > 8 else ""
            message = (
                f"{len(missing_bases)} (cell, column) pair(s) hold values that were never "
                f"measured: {detail}{more}. A value that was not measured cannot be "
                "interpolated, so the synthetic rows carry NaN there rather than an "
                "imputed number; neighbour distances were computed on the columns that "
                "were measured. Impute before resampling if a number is required."
            )
            self._warn_fit_result(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)

        if not synth_frames:
            # Not one synthetic row was produced, so the caller gets its own
            # frame back. The arithmetic is right (every cell already holds
            # `target` rows), and a SILENT unchanged frame is still
            # indistinguishable from a successful balance. Measured 2026-09-27
            # on 10 rows of one group carrying one label: 10 rows in, 10 rows
            # out, fit_result.warnings == []. The sibling
            # CounterfactualAugmenter discloses the same degeneracy on the same
            # shape of frame, through _warn_no_twins.
            if len(cells) < 2:
                cause = (
                    f"the {len(df)} row(s) form a single (group, label) cell, so there is no "
                    "second cell to balance against"
                )
            else:
                cause = (
                    f"every one of the {len(cells)} (group, label) cell(s) already holds "
                    f"{target} row(s), so there was nothing to add"
                )
            self._warn_fit_result(
                f"no synthesis was performed: {cause}. The data is returned UNCHANGED, so "
                "every downstream fairness number is the one the un-resampled data gives."
            )
            return df, y_arr

        out_df = pd.concat([df] + synth_frames, ignore_index=True)
        out_y = np.concatenate([y_arr] + synth_labels)
        perm = rng.permutation(len(out_df))
        return out_df.iloc[perm].reset_index(drop=True), out_y[perm]


class SMOTEResampler(SyntheticResampler):
    """SMOTE oversampling (fixed ``method='smote'``).

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: smote. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        k_neighbors: int = 5,
        strategy: str = "oversample",
        random_state: int = 42,
    ):
        super().__init__(
            protected_attributes,
            method="smote",
            k_neighbors=k_neighbors,
            strategy=strategy,
            random_state=random_state,
        )


class ADASYNResampler(SyntheticResampler):
    """ADASYN oversampling (fixed ``method='adasyn'``): density-weighted synthesis.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: adasyn. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        k_neighbors: int = 5,
        strategy: str = "oversample",
        random_state: int = 42,
    ):
        super().__init__(
            protected_attributes,
            method="adasyn",
            k_neighbors=k_neighbors,
            strategy=strategy,
            random_state=random_state,
        )


class TomekResampler(SyntheticResampler):
    """Tomek-link undersampling (fixed ``method='tomek'``): boundary cleaning.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: tomek. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        k_neighbors: int = 5,
        strategy: str = "undersample",
        random_state: int = 42,
    ):
        super().__init__(
            protected_attributes,
            method="tomek",
            k_neighbors=k_neighbors,
            strategy=strategy,
            random_state=random_state,
        )


# =====================================================================
# 2. Propensity weighting family (PSW / IPW)
# =====================================================================


# Three honest states for "was a propensity model actually estimated?", plus the
# reason. A caller reads result["propensity_model"]; nothing has to be inferred
# from the numbers, which is the whole point.
PROPENSITY_FITTED = "fitted"
PROPENSITY_SINGLE_GROUP = "single_group"
PROPENSITY_NOT_ESTIMABLE = "not_estimable"
PROPENSITY_SOLVER_FAILED = "solver_failed"
# An EMPTY frame, or a frame in which no row carries a recorded protected value,
# is its own state. It used to die inside the single-group f-string with
# ``IndexError: list index out of range`` from ``groups[0]``, which is a crash
# where a named could-not-check belongs.
PROPENSITY_NO_GROUPS = "no_groups"


@dataclass
class _PropensityModel:
    """What ``fit`` learned, and, as a first-class field, what it could not learn.

    Kept as an object rather than rebuilt per call because
    :meth:`_PropensityWeighter.get_sample_weights` has to APPLY the fitted model
    to a new batch. Refitting per batch made a row's weight depend on which
    other rows travelled with it (measured 2026-09-16: the same five rows came
    back as 1.0672 / 1.4923 / 1.3951 / 4.4149 / 1.0587 inside the fit frame and
    1.2341 / 1.6844 / 1.6191 / 2.9979 / 1.4531 on their own).
    """

    status: str
    reason: Optional[str]
    groups: List[str]
    group_codes: np.ndarray
    n_rows_fit: int
    model_columns: List[str]
    mu: np.ndarray
    sd: np.ndarray
    clf: Any
    n_imputed_feature_cells: int
    dropped_constant_columns: List[str] = field(default_factory=list)
    mean_raw_weight: float = float("nan")
    # Rows whose protected attribute was never recorded. They are NOT a group:
    # they are excluded from the model, from the shares and from the balance.
    n_rows_without_a_recorded_group: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.status == PROPENSITY_FITTED and self.clf is not None


def _coerce_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Feature columns as numbers; anything unreadable becomes NaN, not 0.0."""
    if not feature_cols:
        return pd.DataFrame(index=df.index)
    return df[feature_cols].apply(pd.to_numeric, errors="coerce")


def _fit_propensity_model(
    X: Union[pd.DataFrame, np.ndarray],
    protected_attributes: List[str],
    random_state: int = 42,
) -> _PropensityModel:
    """Estimate P(group = assigned | x), or say plainly why it could not be.

    THREE STATES, NEVER TWO. Every branch below used to end in a confident
    number. Measured 2026-09-16, before this:

    * ONE observed group returned 40 ``propensity_scores`` of exactly 1.0 for a
      model that was never fitted, and graded ``max_dev_after = 0.0``.
    * A single ``np.inf`` cell made ``LogisticRegression.fit`` raise
      ``ValueError('Input X contains NaN')``; a bare ``except`` swallowed it and
      fell back to the marginal group frequency, which by construction drives
      the weighted shares to EXACTLY uniform. So the broken run reported
      ``max_dev_after = 0.000000`` against ``0.119828`` for the working model:
      the failure graded BETTER than the success, with ``warnings == []``.
    * All-NaN features collapsed the design matrix to zeros, the solver fitted
      an intercept only, and "propensity" silently became P(group).

    The reason is now carried on the model, the scores are NaN when nothing was
    estimated, and ``max_dev_after`` is None rather than a perfect score.
    """
    from sklearn.linear_model import LogisticRegression

    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X)
    missing = [a for a in protected_attributes if a not in df.columns]
    if missing:
        raise ValueError("Protected attributes not found: %s" % (missing,))

    n = len(df)
    # THREE STATES, NEVER TWO, at the GROUP KEY itself. `_group_key_values`
    # stringifies, so a row whose protected attribute was never recorded became
    # a group literally named 'nan' (or 'None'). Measured 2026-09-17 before
    # this: 5 unrecorded rows in a 60-row frame formed a third "group", were
    # given fitted propensities and sample weights of 8.37 to 28.02, and the
    # run reported max_dev_after = 0.033784 with `warnings == []`. An
    # unrecorded group is not a group. `counterfactual_augment` in this module
    # was repaired for exactly this shape; the propensity family was not.
    if n:
        recorded = df[list(protected_attributes)].notna().all(axis=1).to_numpy()
    else:
        recorded = np.zeros(0, dtype=bool)
    n_unrecorded = int((~recorded).sum())

    group_vals = np.asarray([str(g) for g in _group_key_values(df, protected_attributes)])
    groups = sorted({str(g) for g in group_vals[recorded]})
    code_of = {g: i for i, g in enumerate(groups)}
    # -1 marks "no recorded group": a sentinel the share and balance code below
    # filters on, so an unrecorded row can never be counted into a group share.
    group_codes = np.array(
        [code_of[group_vals[i]] if recorded[i] else -1 for i in range(n)], dtype=int
    )
    n_groups = len(groups)

    feature_cols = [c for c in df.columns if c not in protected_attributes]
    numeric = _coerce_features(df, feature_cols)
    # The design matrix is the rows whose group IS recorded; nothing else can
    # contribute to estimating P(group | x).
    design = numeric.loc[recorded] if n else numeric
    n_imputed = int(design.isna().to_numpy().sum()) if design.shape[1] else 0

    # A column carries information about group membership only if it VARIES
    # across the rows the model is estimated on. np.ptp on the values that were
    # actually measured, never np.var against exact zero: an accumulated
    # statistic is not exactly 0.0 at every n.
    model_columns: List[str] = []
    dropped: List[str] = []
    for col in design.columns:
        vals = design[col].to_numpy(dtype=float)
        seen = vals[~np.isnan(vals)]
        if seen.size >= 2 and float(np.ptp(seen)) > 0.0:
            model_columns.append(str(col))
        else:
            dropped.append(str(col))

    def _unfitted(status: str, reason: str) -> _PropensityModel:
        return _PropensityModel(
            status=status,
            reason=reason,
            groups=groups,
            group_codes=group_codes,
            n_rows_fit=n,
            model_columns=model_columns,
            mu=np.zeros(0),
            sd=np.zeros(0),
            clf=None,
            n_imputed_feature_cells=n_imputed,
            dropped_constant_columns=dropped,
            n_rows_without_a_recorded_group=n_unrecorded,
        )

    if n_groups == 0:
        return _unfitted(
            PROPENSITY_NO_GROUPS,
            f"no row carries a recorded value for {list(protected_attributes)} "
            f"({n} row(s) read, {n_unrecorded} of them with no recorded group), so there "
            "is no group to compare and no propensity to estimate",
        )
    if n_groups < 2:
        return _unfitted(
            PROPENSITY_SINGLE_GROUP,
            f"only one observed group ({groups[0]!r} for {list(protected_attributes)}), so "
            "no between-group comparison exists and no propensity can be estimated",
        )
    if not model_columns:
        return _unfitted(
            PROPENSITY_NOT_ESTIMABLE,
            "no feature column has any measured variation "
            f"(dropped as constant or unreadable: {dropped[:8]}), so the design matrix "
            "carries no information about group membership and any 'propensity' would "
            "be the marginal group frequency wearing a conditional name",
        )

    Xf = design[model_columns].fillna(0.0).to_numpy(dtype=float)
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    # `sd[sd == 0] = 1.0` was two defects on one line: an exact-equality test on
    # an accumulated statistic, and a fabricated neutral scale for a column that
    # carries no information at all. A column with no spread in the design
    # matrix is DROPPED and reported instead; if that empties the matrix the
    # model is not estimable, which is a refusal, not a scale of 1.0.
    #
    # NARROWLY: only a FINITE sd that is not positive is a constant column. A
    # non-finite sd means the column holds a value that is not a measurement
    # (an inf, say), and that must still reach the solver and be REFUSED as
    # PROPENSITY_SOLVER_FAILED. Dropping it here would quietly discard the one
    # column with the problem and report a confident "fitted" run on the rest,
    # which is the swallowed-solver-failure defect wearing a repair.
    degenerate = np.isfinite(sd) & ~(sd > 0.0)
    if bool(degenerate.any()):
        flags = degenerate.tolist()
        dropped.extend(c for c, bad in zip(model_columns, flags) if bad)
        model_columns = [c for c, bad in zip(model_columns, flags) if not bad]
        if not model_columns:
            return _unfitted(
                PROPENSITY_NOT_ESTIMABLE,
                "every candidate feature column has zero spread in the design matrix "
                f"(dropped: {dropped[:8]}), so nothing distinguishes the groups and no "
                "propensity can be estimated",
            )
        Xf = design[model_columns].fillna(0.0).to_numpy(dtype=float)
        mu = Xf.mean(axis=0)
        sd = Xf.std(axis=0)
    Xs = (Xf - mu) / sd
    clf = LogisticRegression(max_iter=1000, random_state=int(random_state))
    try:
        clf.fit(Xs, group_codes[recorded])
    except Exception as exc:  # solver refused: report it, never substitute for it
        return _unfitted(
            PROPENSITY_SOLVER_FAILED,
            f"the logistic-regression solver refused the design matrix: "
            f"{type(exc).__name__}: {exc}",
        )

    return _PropensityModel(
        status=PROPENSITY_FITTED,
        reason=None,
        groups=groups,
        group_codes=group_codes,
        n_rows_fit=n,
        model_columns=model_columns,
        mu=mu,
        sd=sd,
        clf=clf,
        n_imputed_feature_cells=n_imputed,
        dropped_constant_columns=dropped,
        n_rows_without_a_recorded_group=n_unrecorded,
    )


@dataclass
class _PropensityScores:
    """Per-row assigned-group propensity, plus what could NOT be scored.

    The three refusals are separate counts on purpose: a caller that gets NaN
    back has to be able to tell "this row's group was never recorded" from
    "this row's group was never observed at fit time" from "this row's model
    feature was never measured", and none of them from a measurement.
    """

    p: np.ndarray
    unseen_groups: List[str]
    n_rows_without_a_recorded_group: int
    n_rows_with_an_unmeasured_model_feature: int


def _propensity_scores(
    model: _PropensityModel,
    df: pd.DataFrame,
    protected_attributes: List[str],
    clip: Tuple[float, float],
) -> _PropensityScores:
    """Assigned-group propensity per row of ``df`` under a FITTED ``model``.

    A row whose group was never seen at fit time gets NaN, not a neutral value:
    the model has no coefficient for it, so its propensity was not estimated.
    So does a row whose group was never RECORDED, and a row whose model feature
    was never measured. That last one was the live defect: ``.fillna(0.0)``
    here put a fabricated observation of zero into the design row and returned a
    confident weight for it (measured 2026-09-17: one and the same row weighed
    4.414860 with ``x1`` measured and 23.944332 with ``x1`` missing, silently
    both times). ``fit`` already discloses the imputation it performs; the APPLY
    path, which is the one a training loop calls, disclosed nothing.
    """
    n = len(df)
    p = np.full(n, np.nan, dtype=float)
    absent = [a for a in protected_attributes if a not in df.columns]
    if absent:
        raise ValueError("Protected attributes not found: %s" % (absent,))
    if not model.is_fitted:
        return _PropensityScores(p, [], 0, 0)

    lo, hi = float(clip[0]), float(clip[1])
    if n:
        recorded = df[list(protected_attributes)].notna().all(axis=1).to_numpy()
    else:
        recorded = np.zeros(0, dtype=bool)
    n_unrecorded = int((~recorded).sum())
    group_vals = np.asarray([str(g) for g in _group_key_values(df, protected_attributes)])
    code_for_group = {g: i for i, g in enumerate(model.groups)}
    known = recorded & np.array([g in code_for_group for g in group_vals], dtype=bool)
    unseen = sorted({str(g) for g in group_vals[recorded & ~known]})

    numeric = _coerce_features(df, list(df.columns))
    for col in model.model_columns:
        if col not in numeric.columns:
            raise ValueError(
                f"column {col!r} was used to fit the propensity model and is not "
                "present in this frame"
            )
    Xraw = numeric[model.model_columns].to_numpy(dtype=float)
    if Xraw.shape[1]:
        measured_features = ~np.isnan(Xraw).any(axis=1)
    else:
        measured_features = np.ones(n, dtype=bool)
    n_unmeasured = int((known & ~measured_features).sum())
    scorable = known & measured_features
    if not scorable.any():
        return _PropensityScores(p, unseen, n_unrecorded, n_unmeasured)

    idx = np.where(scorable)[0]
    Xs = (Xraw[idx] - model.mu) / model.sd
    proba = model.clf.predict_proba(Xs)
    col_for_code = {int(c): j for j, c in enumerate(model.clf.classes_)}
    for row, i in enumerate(idx.tolist()):
        column = col_for_code.get(int(code_for_group[group_vals[i]]))
        if column is None:
            continue  # a group present at fit but absent from the fitted classes
        p[i] = float(proba[row, column])
    measured = ~np.isnan(p)
    p[measured] = np.clip(p[measured], lo, hi)
    return _PropensityScores(p, unseen, n_unrecorded, n_unmeasured)


def _weights_from_scores(p: np.ndarray, mode: str, scale: Optional[float]) -> np.ndarray:
    """Horvitz-Thompson weights ``1 / p``, NaN preserved as NaN.

    ``scale`` is the FIT-TIME mean weight for ``mode='psw'``; passing it keeps a
    row's stabilised weight the same whatever batch it arrives in. None means
    "use this frame's own mean", which is correct only while the frame IS the
    fit frame.
    """
    weights = np.divide(
        1.0, p, out=np.full(p.shape, np.nan, dtype=float), where=~np.isnan(p) & (p != 0.0)
    )
    if mode == "psw":
        mean_w = scale if scale is not None else float(np.nanmean(weights))
        if mean_w is not None and np.isfinite(mean_w) and mean_w > 0:
            weights = weights / mean_w
    # A weight that is not strictly positive and finite was not measured; it is
    # NaN (could not check), never 1.0. Handing back the neutral "no reweighting
    # needed" value is exactly the defect this module is being repaired for.
    measured = ~np.isnan(weights)
    broken = measured & ~(np.isfinite(weights) & (weights > 0))
    if broken.any():
        weights[broken] = np.nan
        warnings.warn(
            f"propensity weighting: {int(broken.sum())} row(s) produced a weight that is "
            "not finite and positive; they are returned as NaN, not as 1.0",
            stacklevel=2,
        )
    return weights


def _propensity_result(
    model: _PropensityModel,
    mode: str,
    clip: Tuple[float, float],
    df: Optional[pd.DataFrame] = None,
    protected_attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble the public result dict for a model fitted on its own frame."""
    n = model.n_rows_fit
    group_codes = model.group_codes
    n_groups = len(model.groups)
    uniform = 1.0 / n_groups if n_groups else float("nan")
    # Rows whose protected value was never recorded are not a group and are not
    # counted into any share. `group_codes == -1` is that state.
    recorded = group_codes >= 0
    n_recorded = int(recorded.sum())

    result_warnings: List[str] = []
    n_unmeasured_rows = 0
    if model.is_fitted:
        # A bare `assert` is not a guard: `python -O` strips it, and that is the
        # build where a None here would become an AttributeError far downstream.
        if df is None or protected_attributes is None:
            raise ValueError(
                "a fitted propensity model must be scored against the frame it was "
                "fitted on; df and protected_attributes are required"
            )
        scored = _propensity_scores(model, df, protected_attributes, clip)
        p_assigned = scored.p
        n_unmeasured_rows = scored.n_rows_with_an_unmeasured_model_feature
        weights = _weights_from_scores(p_assigned, mode="ipw", scale=None)
        model.mean_raw_weight = (
            float(np.nanmean(weights)) if np.isfinite(weights).any() else float("nan")
        )
        if mode == "psw":
            weights = _weights_from_scores(p_assigned, mode="psw", scale=model.mean_raw_weight)
    elif model.status == PROPENSITY_SINGLE_GROUP:
        # The identity is the whole of the arithmetic here: with one observed
        # group there is nothing to rebalance, so every row OF THAT GROUP keeps
        # weight 1.0. What must NOT happen is reporting that as an estimated
        # propensity or as a perfect balance score, so the scores are NaN and
        # max_dev_after is None below.
        #
        # A row whose protected value was never recorded is NOT of that group,
        # and 1.0 for it is the fabricated neutral weight this whole family was
        # repaired for: it trains the model as though the row had been measured.
        # `np.ones(n)` covered every row. Measured 2026-09-27 on 15 rows of
        # group 'a' beside 5 rows with no recorded group: all 20 sample_weights
        # came back 1.0, while the warning two paragraphs below announced "their
        # sample weight is NaN" and the docstring promised the same. The
        # disclosure was correct and the array disagreed with it.
        p_assigned = np.full(n, np.nan, dtype=float)
        weights = np.where(group_codes >= 0, 1.0, np.nan) if n else np.zeros(0, dtype=float)
        result_warnings.append(f"no propensity model was estimated: {model.reason}")
    else:
        p_assigned = np.full(n, np.nan, dtype=float)
        weights = np.full(n, np.nan, dtype=float)
        result_warnings.append(
            f"no propensity model was estimated: {model.reason}. The sample weights are "
            "NaN (could not check), not 1.0: a weight that was never derived must not "
            "train a model as though it had been."
        )

    if model.n_imputed_feature_cells and model.is_fitted:
        result_warnings.append(
            f"{model.n_imputed_feature_cells} feature cell(s) were unreadable or missing "
            "and entered the design matrix as 0.0; the propensity estimate depends on "
            "that imputation"
        )
    if model.dropped_constant_columns and model.is_fitted:
        result_warnings.append(
            "column(s) with no measured variation carry no information about group "
            f"membership and were excluded from the model: {model.dropped_constant_columns[:8]}"
        )
    for message in result_warnings:
        warnings.warn(f"propensity_weights: {message}", stacklevel=2)

    share_before = {}
    for code, g in enumerate(model.groups):
        share_before[g] = (
            float((group_codes == code).sum()) / n_recorded if n_recorded else float("nan")
        )
    max_dev_before = max(abs(v - uniform) for v in share_before.values()) if share_before else None

    share_after: Optional[Dict[str, float]] = None
    max_dev_after: Optional[float] = None
    balance_reason: Optional[str] = None
    # Only a row with a RECORDED group and a MEASURED weight can contribute to a
    # weighted share. Excluding the rest is what keeps the measurement honest;
    # reporting how many were excluded is what keeps it readable.
    measurable = recorded & ~np.isnan(weights) if n else np.zeros(0, dtype=bool)
    n_excluded = int(n_recorded - int(measurable.sum()))
    if not model.is_fitted:
        balance_reason = f"no propensity model was estimated ({model.status})"
    elif not measurable.any():
        balance_reason = "no row has both a recorded group and a measured weight"
    else:
        empty_groups = [
            g
            for code, g in enumerate(model.groups)
            if not (measurable & (group_codes == code)).any()
        ]
        total_w = float(weights[measurable].sum())
        if empty_groups:
            balance_reason = (
                f"group(s) {empty_groups[:8]} have no row with a measured weight, so the "
                "weighted shares would be a comparison against a group that is not there"
            )
        elif not (np.isfinite(total_w) and total_w > 0.0):
            # `... if total_w > 0 else uniform` used to sit here: a PERFECT
            # score handed out as the fallback for a total nobody could compute.
            balance_reason = (
                f"the measured weights sum to {total_w!r}, which is not a positive finite "
                "total, so no weighted share can be derived from them"
            )
        else:
            share_after = {}
            for code, g in enumerate(model.groups):
                mask = measurable & (group_codes == code)
                share_after[g] = float(weights[mask].sum()) / total_w
            max_dev_after = max(abs(v - uniform) for v in share_after.values())

    if model.is_fitted and balance_reason is not None:
        result_warnings.append(
            "the balance after weighting was NOT measured (max_dev_after is None, not "
            f"0.0): {balance_reason}"
        )
        warnings.warn(f"propensity_weights: {result_warnings[-1]}", stacklevel=2)
    if model.n_rows_without_a_recorded_group:
        message = (
            f"{model.n_rows_without_a_recorded_group} of {n} row(s) have no recorded value "
            f"for the protected attribute(s); they are not a group, so they are excluded "
            "from the model and from the shares, and their sample weight is NaN"
        )
        result_warnings.append(message)
        warnings.warn(f"propensity_weights: {message}", stacklevel=2)
    if n_unmeasured_rows:
        message = (
            f"{n_unmeasured_rows} row(s) have no measured value in a column the propensity "
            "model was fitted on; their propensity was not estimated and their sample "
            "weight is NaN, not 1.0"
        )
        result_warnings.append(message)
        warnings.warn(f"propensity_weights: {message}", stacklevel=2)
    if model.is_fitted and n_excluded and share_after is not None:
        message = (
            f"the weighted balance was measured on {int(measurable.sum())} of {n_recorded} "
            f"row(s) with a recorded group; {n_excluded} row(s) had no measured weight"
        )
        result_warnings.append(message)
        warnings.warn(f"propensity_weights: {message}", stacklevel=2)

    group_balance: Dict[str, Any] = {
        "groups": list(model.groups),
        "n_groups": int(n_groups),
        "uniform_target": uniform,
        "share_before": share_before,
        # None, never 0.0: with no fitted model there is no weighted share to
        # measure, and 0.0 reads as a PERFECT balance to anything that grades it.
        "share_after": share_after,
        "max_dev_before": max_dev_before,
        "max_dev_after": max_dev_after,
        # Why max_dev_after is None, when it is. A caller reading None off this
        # dict should not have to guess which of the refusals produced it.
        "max_dev_after_reason": balance_reason,
        "n_rows_scored": int(measurable.sum()),
        "n_rows_without_a_recorded_group": int(model.n_rows_without_a_recorded_group),
    }

    return {
        "sample_weights": weights.tolist(),
        "propensity_scores": np.asarray(p_assigned, dtype=float).tolist(),
        "group_balance": group_balance,
        "propensity_model": model.status,
        "propensity_model_reason": model.reason,
        "n_imputed_feature_cells": model.n_imputed_feature_cells,
        "dropped_constant_columns": list(model.dropped_constant_columns),
        "n_rows_without_a_recorded_group": int(model.n_rows_without_a_recorded_group),
        "n_rows_with_an_unmeasured_model_feature": int(n_unmeasured_rows),
        "warnings": result_warnings,
    }


def propensity_weights(
    X: Union[pd.DataFrame, np.ndarray],
    protected_attributes: List[str],
    mode: str = "ipw",
    clip: Tuple[float, float] = (0.01, 0.99),
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute inverse-propensity sample weights that balance protected groups.

    A logistic-regression model estimates the propensity of the assigned group
    from the non-protected features, ``P(group = assigned | x)``. The weight is
    the inverse of the (clipped) assigned-group propensity, which up-weights the
    members of an under-represented group who look like the majority (low
    assigned-group propensity). By the Horvitz-Thompson identity this drives the
    weighted group sizes toward equality (uniform representation).

    For a two-group / assigned-group setting the "Propensity Score Weighting"
    formula ``1 / P(assigned)`` and the "Inverse Propensity" formula
    ``A/e + (1 - A)/(1 - e)`` coincide, so both modes share the same core
    weights and differ only in the reported scale:

        * ``mode='psw'`` normalises the weights to mean 1 (stabilised scale).
        * ``mode='ipw'`` returns the raw Horvitz-Thompson weights.

    The group-balance ratios (and therefore the move toward uniform
    representation) are identical for both modes because a global scale cancels.

    Args:
        X: DataFrame (or array) that CONTAINS the protected attribute columns.
        protected_attributes: Names of the protected attribute columns.
        mode: ``'ipw'`` (raw) or ``'psw'`` (mean-1 normalised).
        clip: (low, high) bounds applied to the assigned-group propensity before
            inversion, to avoid extreme weights.
        random_state: Seed for the logistic-regression solver.

    Returns:
        Dict with ``sample_weights`` (list, aligned to rows),
        ``propensity_scores`` (list, the clipped assigned-group propensity),
        ``group_balance`` (before / after shares and deviation from uniform,
        plus ``max_dev_after_reason``, ``n_rows_scored`` and
        ``n_rows_without_a_recorded_group``), ``propensity_model`` (one of
        ``'fitted'``, ``'single_group'``, ``'not_estimable'``,
        ``'solver_failed'``, ``'no_groups'``), ``propensity_model_reason``,
        ``n_imputed_feature_cells``, ``dropped_constant_columns``,
        ``n_rows_without_a_recorded_group``,
        ``n_rows_with_an_unmeasured_model_feature`` and ``warnings``.

        When ``propensity_model`` is not ``'fitted'`` NOTHING was estimated:
        ``propensity_scores`` are NaN and ``group_balance['max_dev_after']`` is
        None (could not check), never 0.0. The sample weights are NaN too,
        except in the ``'single_group'`` case, where the identity weight 1.0 is
        the complete and correct answer for a row OF that one group; a row of
        any other group, or with no recorded group at all, is still NaN.

        Per-row refusals, all NaN and all disclosed in ``warnings``: a row whose
        protected value was never recorded (it is not a group and is excluded
        from the shares), and a row with no measured value in a column the model
        was fitted on (imputing that cell to 0.0 returned a confident weight for
        a row nobody measured: 23.944332 against the 4.414860 the same row
        earns when the cell IS measured).
    """
    if mode not in ("ipw", "psw"):
        raise ValueError("mode must be 'ipw' or 'psw', got %r" % (mode,))
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(X)
    model = _fit_propensity_model(df, protected_attributes, random_state)
    return _propensity_result(model, mode, clip, df=df, protected_attributes=protected_attributes)


class _PropensityWeighter(BaseFeatureTransformer):
    """
    Base transformer for propensity-based reweighting. ``transform`` returns the
    features unchanged (mirroring :class:`ReweightingTransformer`); the value is
    in the sample weights exposed through :meth:`get_sample_weights`, which the
    existing handler already feeds into ``model.fit(..., sample_weight=...)``.
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        mode: str = "ipw",
        clip: Tuple[float, float] = (0.01, 0.99),
        random_state: int = 42,
    ):
        super().__init__(protected_attributes)
        if mode not in ("ipw", "psw"):
            raise ValueError("mode must be 'ipw' or 'psw', got %r" % (mode,))
        self.mode = mode
        self.clip = (float(clip[0]), float(clip[1]))
        self.random_state = int(random_state)
        self._weights_result: Optional[Dict[str, Any]] = None
        # The FITTED model is kept, not just its output. get_sample_weights used
        # to refit from scratch on whatever frame it was handed, which made a
        # row's weight depend on its batch mates and returned the neutral 1.0
        # for any batch that happened to hold a single group.
        self._propensity_model: Optional[_PropensityModel] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "_PropensityWeighter":
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs
        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self.feature_names_out_ = feature_cols.copy()

        model = _fit_propensity_model(df, attrs, random_state=self.random_state)
        res = _propensity_result(model, self.mode, self.clip, df=df, protected_attributes=attrs)
        self._propensity_model = model
        self._weights_result = res
        self.fit_result = TransformationResult(
            method="propensity_%s" % self.mode,
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            fit_metrics={
                "mode": self.mode,
                "clip": list(self.clip),
                "group_balance": res["group_balance"],
                # The state of the estimate, on the result object, so a reader
                # does not have to infer it from the numbers.
                "propensity_model": res["propensity_model"],
                "propensity_model_reason": res["propensity_model_reason"],
                "n_imputed_feature_cells": res["n_imputed_feature_cells"],
                "dropped_constant_columns": res["dropped_constant_columns"],
            },
            warnings=list(res["warnings"]),
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return features unchanged. Use get_sample_weights() for the weights."""
        self._check_is_fitted()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)
        feature_cols = self._extract_feature_columns(df, self._fitted_protected_attributes())
        if was_dataframe:
            return df[feature_cols].copy()
        return df[feature_cols].values

    def get_sample_weights(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Apply the FITTED propensity model to the rows of ``X``.

        ``X`` must contain the protected attribute column(s) and the feature
        columns the model was fitted on. The weight of a row is a property of
        that row and the fitted model, so it is the same whether the row arrives
        alone or inside the fit frame.

        THREE STATES, NEVER TWO. Measured 2026-09-16, before this method used
        the fitted model: it REFITTED a fresh logistic regression on whatever
        frame it was handed. A 30-row batch holding a single group took the
        degenerate branch and came back as thirty weights of exactly 1.0, the
        neutral "no reweighting needed" value, for rows the fitted model weighs
        1.0672 / 1.4923 / 1.3951; one group-B row whose fitted weight is 4.4149
        came back as ``[1.0]``; and the same five rows returned two different
        sets of weights depending on which rows travelled with them. Now: the
        stored model is applied, a row from a group the model never saw gets
        NaN, and a model that was never estimated yields NaN rather than 1.0.
        """
        self._check_is_fitted()
        df = self._to_dataframe(X, self.feature_names_in_)
        attrs = self._fitted_protected_attributes()
        # A batch without the protected column used to die as KeyError('grp')
        # out of the group key, three frames deep. Same refusal, same wording as
        # _fit_propensity_model, at the entry point the caller can see.
        absent = [a for a in attrs if a not in df.columns]
        if absent:
            raise ValueError("Protected attributes not found: %s" % (absent,))
        model = self._propensity_model
        if model is None:
            raise RuntimeError(
                f"{type(self).__name__} reports fitted but holds no propensity model; "
                "fit() must set it."
            )

        if model.status == PROPENSITY_SINGLE_GROUP:
            # The identity weight 1.0 is the complete answer for a row of THE
            # group that was observed at fit time, and only for such a row. A
            # batch may carry other groups, and for those nothing was estimated:
            # they are NaN, exactly as they would be under a fitted model. The
            # old branch returned np.ones(len(df)) for the whole batch and said
            # "there is no second group to rebalance against" about batches that
            # had several.
            fitted_group = model.groups[0]
            if len(df):
                recorded = df[list(attrs)].notna().all(axis=1).to_numpy()
            else:
                recorded = np.zeros(0, dtype=bool)
            row_groups = np.asarray([str(g) for g in _group_key_values(df, attrs)])
            is_fitted_group = recorded & (row_groups == fitted_group)
            weights = np.where(is_fitted_group, 1.0, np.nan)
            other = sorted({str(g) for g in row_groups[recorded & ~is_fitted_group]})
            n_unrecorded = int((~recorded).sum())
            message = (
                f"{type(self).__name__}.get_sample_weights: no propensity model was "
                f"estimated ({model.reason}); the {int(is_fitted_group.sum())} row(s) of "
                f"{fitted_group!r} keep weight 1.0, which is the identity, not a measured "
                "reweighting."
            )
            if other:
                message += (
                    f" {int((recorded & ~is_fitted_group).sum())} row(s) of group(s) "
                    f"{other[:8]}{' ...' if len(other) > 8 else ''} were never observed at "
                    "fit time and get NaN, not 1.0."
                )
            if n_unrecorded:
                message += f" {n_unrecorded} row(s) have no recorded group at all and get NaN."
            warnings.warn(message, stacklevel=2)
            return weights

        if not model.is_fitted:
            warnings.warn(
                f"{type(self).__name__}.get_sample_weights: no propensity model was "
                f"estimated ({model.reason}); {len(df)} row(s) get a NaN weight, not 1.0. "
                "Refit on data the model can be estimated from, or drop the step.",
                stacklevel=2,
            )
            return np.full(len(df), np.nan, dtype=float)

        scored = _propensity_scores(model, df, attrs, self.clip)
        weights = _weights_from_scores(
            scored.p,
            mode=self.mode,
            scale=model.mean_raw_weight if self.mode == "psw" else None,
        )
        if scored.unseen_groups:
            unseen = scored.unseen_groups
            warnings.warn(
                f"{type(self).__name__}.get_sample_weights: no propensity was fitted for "
                f"group(s) {unseen[:8]}{' ...' if len(unseen) > 8 else ''}; "
                f"{int(np.isnan(weights).sum())} row(s) get a NaN sample weight (not 1.0). "
                "Drop them or refit on data that covers them.",
                stacklevel=2,
            )
        if scored.n_rows_without_a_recorded_group:
            warnings.warn(
                f"{type(self).__name__}.get_sample_weights: "
                f"{scored.n_rows_without_a_recorded_group} row(s) have no recorded value "
                f"for {list(attrs)}; a row with no group has no propensity, so its weight "
                "is NaN, not 1.0.",
                stacklevel=2,
            )
        if scored.n_rows_with_an_unmeasured_model_feature:
            warnings.warn(
                f"{type(self).__name__}.get_sample_weights: "
                f"{scored.n_rows_with_an_unmeasured_model_feature} row(s) have no measured "
                f"value in a column the model was fitted on ({model.model_columns[:8]}); "
                "their propensity was not estimated and their weight is NaN. Imputing the "
                "cell to 0.0 returned a confident weight for a row nobody measured.",
                stacklevel=2,
            )
        return weights


class PropensityScoreWeighter(_PropensityWeighter):
    """Propensity Score Weighting (mean-1 normalised inverse-propensity weights).

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: propensity_weighting. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        clip: Tuple[float, float] = (0.01, 0.99),
        random_state: int = 42,
    ):
        super().__init__(protected_attributes, mode="psw", clip=clip, random_state=random_state)


class InversePropensityWeighter(_PropensityWeighter):
    """Inverse Propensity Scoring (raw Horvitz-Thompson inverse-propensity weights).

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: inverse_propensity. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        clip: Tuple[float, float] = (0.01, 0.99),
        random_state: int = 42,
    ):
        super().__init__(protected_attributes, mode="ipw", clip=clip, random_state=random_state)


# =====================================================================
# 3. Counterfactual data augmentation
# =====================================================================


def _allocate_twin_groups(
    source_groups: List[Any],
    groups: List[Any],
    counts: Dict[Any, int],
    mode: str,
) -> List[Any]:
    """Choose the group each row's counterfactual twin is sent to.

    ``groups`` is every observed group (a level, or a joint tuple), sorted so
    the choice is reproducible; ``counts`` is how many rows each group holds
    BEFORE augmentation. A row whose group is the only one observed has no
    alternative and keeps its own group, which produces no twin.

    ``mode='balanced'`` picks the alternative with the smallest RUNNING count,
    incrementing as twins are assigned. Every group shares one target, an even
    split, so the smallest running count is the largest remaining deficit; that
    is the whole of the allocation rule. Any other mode picks the first
    alternative in sorted order and ignores the counts.

    The running count is the half that was missing. Choosing against the
    ORIGINAL counts sends every twin to the same rarest group, which inverts
    the imbalance instead of removing it.
    """
    running = dict(counts)
    targets: List[Any] = []
    for cur in source_groups:
        others = [g for g in groups if g != cur]
        if not others:
            targets.append(cur)
            continue
        if mode == "balanced":
            pick = min(others, key=lambda g: (running.get(g, 0), str(g)))
            running[pick] = running.get(pick, 0) + 1
        else:
            pick = others[0]
        targets.append(pick)
    return targets


def _counterfactual_levels(df: pd.DataFrame, protected_attributes: List[str]) -> Dict[str, int]:
    """Observed level count per protected attribute, missing values excluded.

    Fewer than two levels means no counterfactual twin exists for that
    attribute: there is no alternative group to swap a row to.
    """
    return {str(a): int(df[a].dropna().nunique()) for a in protected_attributes if a in df.columns}


def _warn_no_twins(
    protected_attributes: List[str],
    levels: List[Any],
    n_twins: int,
    n_unmeasured: int,
    n_rows: int,
) -> None:
    """Say when augmentation could not augment, and how far short it fell.

    THREE STATES, NEVER TWO. Measured 2026-09-16: on a frame whose protected
    attribute had a SINGLE observed level, 60 rows went in, 60 rows came out,
    and ``fit_result.warnings``, ``fit_metrics`` and ``n_samples`` were
    byte-identical to the two-level run that correctly returned 120. A caller
    could not tell "augmented" from "there was nothing to augment", which is the
    neutered-mitigation shape: every downstream fairness number is computed on
    the unchanged rows and looks exactly as it did before. The sibling
    :class:`DisparateImpactRemover` already emits the equivalent disclosure on
    the identical frame.
    """
    if n_twins == 0:
        warnings.warn(
            f"counterfactual_augment: NO counterfactual twins were created, so the data "
            f"is returned unchanged. The protected attribute(s) {list(protected_attributes)} "
            f"have {len(levels)} observed level(s) across {n_rows} row(s)"
            + (f", and {n_unmeasured} row(s) have no recorded value" if n_unmeasured else "")
            + ". With no alternative group there is nothing to swap to; the augmentation "
            "is a no-op, not a balanced result.",
            stacklevel=3,
        )
    elif n_unmeasured:
        warnings.warn(
            f"counterfactual_augment: {n_unmeasured} of {n_rows} row(s) have no recorded "
            f"value for {list(protected_attributes)} and were NOT twinned; a row whose "
            "group is unknown has no counterfactual. "
            f"{n_twins} twin(s) were created from the {n_rows - n_unmeasured} row(s) whose "
            "group is measured.",
            stacklevel=3,
        )


def counterfactual_augment(
    df: pd.DataFrame,
    protected_attributes: List[str],
    target: Optional[str] = None,
    mode: str = "balanced",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Augment ``df`` with counterfactual twins: for each row, a duplicate whose
    protected-attribute value(s) are swapped to a different group level while all
    other columns (features and target) are preserved. This balances group
    representation without disturbing the target distribution.

    Args:
        df: Input DataFrame that CONTAINS the protected attribute columns (and,
            optionally, the target column).
        protected_attributes: Names of the protected attribute columns to swap.
        target: Optional target column name. Accepted to make the intent
            explicit and NOT read: every column of ``df`` is copied onto the
            twins, so the target is preserved whether or not it is named here.
            :class:`CounterfactualAugmenter` passes it for that reason.
        mode: ``'balanced'`` sends each twin to the alternative group that is
            currently the SMALLEST, counting the twins already assigned, so the
            augmented set moves toward an even split. Any other value sends
            every twin to the first alternative level in sorted order, which
            balances nothing and is only there to make the un-balanced
            behaviour available and explicit. Honoured in the multi-attribute
            branch too, since 2026-09-10.
        random_state: Accepted for API symmetry with the resamplers in this
            module and NOT read: every choice here is deterministic (running
            counts, ties broken by ``str``), so the same input gives the same
            output with any seed. Nothing about this function is random.

    Returns:
        A new DataFrame with the original rows followed by the counterfactual
        twins (row count increases by the number of rows that had an alternative
        group level).

    Balance is measured, not assumed. Recorded 2026-08-22 as
    [MEDIUM][CONFIRMED] and fixed on 2026-09-10: the target group used to be
    the globally least-represented alternative, computed ONCE from the original
    counts and never updated as twins accrued, so with three or more levels
    every twin landed on the same rarest level. Measured on A=60 / B=30 / C=10,
    ``mode='balanced'`` gave {C: 100, A: 60, B: 40}: the rarest group inflated
    ten-fold into the largest, while B stayed under A. In the multi-attribute
    branch it was worse: all twins went to the rarest intersectional cell and
    ``mode`` was never read at all, so the maximum deviation from a uniform
    split went from 0.2000 to 0.2500, strictly WORSE than doing nothing, under
    a docstring that claimed the opposite.
    """
    df = df.reset_index(drop=True).copy()
    if not protected_attributes:
        return df

    # A row whose protected value was never recorded has no group to be the
    # counterfactual OF, so it cannot have a twin. Measured 2026-09-16, before
    # this: a frame of 50 rows labelled 'A' and 10 rows with a MISSING race came
    # back as 70 rows, because each of the ten unmeasured rows was handed a twin
    # with race='A'. That invents a protected-attribute value for exactly the
    # rows where it is unknown, and the extra rows made the run look augmented.
    measured_mask = df[list(protected_attributes)].notna().all(axis=1).to_numpy()
    n_unmeasured = int((~measured_mask).sum())
    measured_df = df[measured_mask]

    if len(protected_attributes) == 1:
        attr = protected_attributes[0]
        levels = sorted(measured_df[attr].dropna().unique().tolist(), key=lambda v: str(v))
        counts = measured_df[attr].value_counts().to_dict()
        target_levels = _allocate_twin_groups(list(measured_df[attr].values), levels, counts, mode)
        swapped = np.array(
            [t != g for t, g in zip(target_levels, measured_df[attr].values)], dtype=bool
        )
        twin_df = measured_df[swapped].copy()
        twin_df[attr] = [t for t, keep in zip(target_levels, swapped) if keep]
        n_twins = int(len(twin_df))
        _warn_no_twins(protected_attributes, levels, n_twins, n_unmeasured, len(df))
        if n_twins == 0:
            return df
        return pd.concat([df, twin_df], ignore_index=True)

    # Multiple protected attributes: swap the joint group tuple.
    from collections import Counter

    keys = measured_df[list(protected_attributes)].astype(object).values
    group_tuples = [tuple(row) for row in keys]
    counts = Counter(group_tuples)
    observed = sorted(counts.keys(), key=str)

    targets = _allocate_twin_groups(group_tuples, observed, dict(counts), mode)
    swapped_mask = np.array([t != g for t, g in zip(targets, group_tuples)], dtype=bool)
    twin_df = measured_df[swapped_mask].copy()
    kept_targets = [targets[i] for i in range(len(group_tuples)) if swapped_mask[i]]
    for j, a in enumerate(protected_attributes):
        twin_df[a] = [t[j] for t in kept_targets]
    n_twins = int(len(twin_df))
    _warn_no_twins(protected_attributes, observed, n_twins, n_unmeasured, len(df))
    if n_twins == 0:
        return df
    return pd.concat([df, twin_df], ignore_index=True)


class CounterfactualAugmenter(BaseFeatureTransformer):
    """
    Counterfactual data augmentation as a preprocessing transformer. Row counts
    change, so the augmented training set is exposed through the additive
    :meth:`get_resampled_data` hook (exactly like :class:`Resampler`), letting the
    existing handler rebalance with no code changes. The standalone
    :func:`counterfactual_augment` does the work and remains importable for reuse.

    Args:
        protected_attributes: Names of protected attribute columns to swap.
        mode: ``'balanced'`` (swap toward the least-represented level) or another
            value (swap toward the first alternative level).
        random_state: Reserved for reproducibility.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: counterfactual_augment. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        mode: str = "balanced",
        random_state: int = 42,
    ):
        super().__init__(protected_attributes)
        self.mode = mode
        self.random_state = int(random_state)
        # What the LAST get_resampled_data() appended to fit_result.warnings.
        # Three calls used to leave three copies of the same sentence, so a
        # reader counting warnings saw a worsening run where nothing changed.
        self._disclosed: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Any] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "CounterfactualAugmenter":
        attrs = self._validate_protected_attributes(X, protected_attributes)
        df = self._to_dataframe(X)
        # _validate_protected_attributes only checks column membership when X is
        # already a DataFrame, so an ndarray used to fit "successfully" against a
        # protected attribute that is not one of its columns and only failed
        # later, with a KeyError out of get_resampled_data. Check the frame that
        # will actually be used, and raise the same ValueError either way.
        missing = [a for a in attrs if a not in df.columns]
        if missing:
            raise ValueError(f"Protected attributes not found in DataFrame: {missing}")
        self.protected_attributes = attrs
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self.feature_names_out_ = feature_cols.copy()

        # A configuration is not a capability. Record how many levels each
        # attribute actually has, so a reader can see whether a twin is even
        # possible, and say so out loud when it is not.
        #
        # Twinnability is a property of the JOINT group, because the joint group
        # tuple is what the swap actually moves. Deriving it per attribute
        # (`min(levels.values()) >= 2`) contradicted the augmentation itself:
        # measured 2026-09-17 on race=[A x 60, B x 40] with gender=[M x 100],
        # fit reported n_twins_possible=0 and warned "no counterfactual twin
        # exists ... get_resampled_data() will return the data UNCHANGED", and
        # then get_resampled_data returned 100 -> 200 rows with n_twins=100 in
        # the same fit_metrics dict. The level counts are kept, as information.
        levels = _counterfactual_levels(df, attrs)
        measured_mask = df[list(attrs)].notna().all(axis=1).to_numpy()
        n_unmeasured = int((~measured_mask).sum())
        recorded_rows = df.loc[measured_mask, list(attrs)]
        if len(recorded_rows) == 0:
            n_joint_groups = 0
        elif len(attrs) == 1:
            n_joint_groups = int(recorded_rows[attrs[0]].nunique())
        else:
            n_joint_groups = len({tuple(row) for row in recorded_rows.astype(object).values})
        twinnable = (len(df) - n_unmeasured) if n_joint_groups >= 2 else 0
        fit_warnings: List[str] = []
        thin = sorted(a for a, k in levels.items() if k < 2)
        if n_joint_groups < 2:
            message = (
                f"no counterfactual twin exists: the protected attribute(s) {list(attrs)} "
                f"take {n_joint_groups} joint group value(s) across the "
                f"{len(df) - n_unmeasured} row(s) with a recorded group (observed levels "
                f"per attribute: {levels}), so there is no alternative group to swap a row "
                "to. get_resampled_data() will return the data UNCHANGED; the augmentation "
                "is a no-op, not a balanced result."
            )
            fit_warnings.append(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)
        elif thin:
            message = (
                f"partly counterfactual: protected attribute(s) {thin} have fewer than 2 "
                f"observed levels ({levels}), so a twin never differs on them. The swap "
                f"moves the JOINT group, of which {n_joint_groups} were observed, so twins "
                "are still created; they are counterfactual only in the attribute(s) that "
                "have an alternative."
            )
            fit_warnings.append(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)
        if n_unmeasured:
            message = (
                f"{n_unmeasured} of {len(df)} row(s) have no recorded value for "
                f"{list(attrs)}; a row whose group is unknown has no counterfactual and "
                "will not be twinned."
            )
            fit_warnings.append(message)
            warnings.warn(f"{type(self).__name__}: {message}", stacklevel=2)

        self.fit_result = TransformationResult(
            method="counterfactual_augment",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            fit_metrics={
                "mode": self.mode,
                "levels_per_attribute": levels,
                "n_joint_groups": int(n_joint_groups),
                "n_rows_without_a_recorded_group": n_unmeasured,
                "n_twins_possible": int(twinnable),
            },
            warnings=fit_warnings,
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return features unchanged. Use get_resampled_data() for augmented rows."""
        self._check_is_fitted()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)
        feature_cols = self._extract_feature_columns(df, self._fitted_protected_attributes())
        if was_dataframe:
            return df[feature_cols].copy()
        return df[feature_cols].values

    def get_resampled_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Any,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Return ``(X_aug, y_aug)`` with counterfactual twins appended.
        ``X_aug`` keeps all input columns (features and protected attributes);
        ``y_aug`` is aligned and preserves the original target distribution.
        """
        self._check_is_fitted()
        if self.fit_result is not None and self._disclosed:
            stale = self._disclosed
            self.fit_result.warnings = [w for w in self.fit_result.warnings if w not in stale]
        self._disclosed = []
        df = self._to_dataframe(X, self.feature_names_in_).reset_index(drop=True)
        y_arr = np.asarray(y)

        target_col = "__cf_target__"
        while target_col in df.columns:
            target_col = target_col + "_"
        work = df.copy()
        work[target_col] = y_arr

        aug = counterfactual_augment(
            work,
            self._fitted_protected_attributes(),
            target=target_col,
            mode=self.mode,
            random_state=self.random_state,
        )
        y_res = aug[target_col].values
        X_res = aug.drop(columns=[target_col]).reset_index(drop=True)

        # Measure the augmentation instead of assuming it. n_twins == 0 means
        # the input came straight back, which is indistinguishable from a
        # successful run unless it is recorded here.
        n_twins = int(len(X_res) - len(df))
        if self.fit_result is not None:
            self.fit_result.fit_metrics["n_twins"] = n_twins
            if n_twins == 0:
                self._disclosed.append(
                    f"no counterfactual twins were created: {len(df)} row(s) in, "
                    f"{len(X_res)} row(s) out. The data is returned UNCHANGED, so every "
                    "downstream fairness number is the one the un-augmented data gives."
                )
            elif n_twins < len(df):
                self._disclosed.append(
                    f"counterfactual augmentation is PARTIAL: {n_twins} twin(s) for "
                    f"{len(df)} input row(s). Rows with no alternative group, or with no "
                    "recorded group of their own, were not twinned."
                )
            self.fit_result.warnings.extend(self._disclosed)
        return X_res, np.asarray(y_res)


__all__ = [
    "SyntheticResampler",
    "SMOTEResampler",
    "ADASYNResampler",
    "TomekResampler",
    "propensity_weights",
    "PropensityScoreWeighter",
    "InversePropensityWeighter",
    "counterfactual_augment",
    "CounterfactualAugmenter",
]
