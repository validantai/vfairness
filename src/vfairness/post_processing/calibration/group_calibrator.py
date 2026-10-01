"""
Group-Specific Calibration for vfairness.

This module provides calibration methods that operate on demographic groups
separately, ensuring probability predictions have consistent meaning across
different populations. This is essential for fair probability estimates.

Key Components:
    - GroupCalibrator: Fits separate calibrators per demographic group
    - IntersectionalCalibrator: Handles intersectional group calibration
    - Hierarchical borrowing for small sample groups

When to Use Group-Specific Calibration:
    - When calibration metrics differ significantly across groups
    - When probability thresholds are used for decisions
    - When fair interpretation of scores is required
    - When base rates differ substantially between groups

References:
    - Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.
    - Hebert-Johnson, U., et al. (2018). Multicalibration. ICML.
    - Kim, M. P., et al. (2019). Multiaccuracy. AIES.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    validate_probabilities,
)

from .methods import (
    BaseCalibrator,
    CalibrationMethod,
    create_calibrator,
)
from .metrics import (
    calibration_disparity,
    expected_calibration_error,
)


@dataclass
class GroupCalibrationResult:
    """
    Container for group calibration results.

    Attributes:
        method: Calibration method used
        n_groups: Number of groups calibrated
        group_calibrators: Dict of fitted calibrators per group
        pre_calibration_metrics: Metrics before calibration
        post_calibration_metrics: Metrics after calibration
        improvement: ECE improvement per group
        overall_improvement: Overall ECE improvement
        fallback_groups: Groups that used fallback due to small sample
        unmeasurable_groups: Groups whose calibration is a COULD-NOT-CHECK,
            mapped to the reason. See :attr:`unmeasurable_groups` below.
    """

    method: str
    n_groups: int
    group_names: List[str]
    pre_calibration_ece: Dict[str, float]
    post_calibration_ece: Dict[str, float]
    improvement: Dict[str, float]
    overall_improvement: float
    fallback_groups: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    #: Group name -> reason, for every group whose own calibrator did not
    #: produce a calibration of that group. An EMPTY dict means every group in
    #: ``group_names`` got a real, varying map fitted on its own rows; it does
    #: NOT mean "not checked". Reasons are the ``REASON_*`` constants below.
    #: BGL g015, 2026-09-17: the sibling :class:`IntersectionalCalibrator` grew
    #: ``group_provenance_`` for exactly this and this class kept none, so a
    #: group whose isotonic map collapsed to a single value was reported with
    #: ``fallback_groups=[]`` beside ``post_calibration_ece=0.0`` and
    #: ``improvement=0.51``, i.e. the best-looking row in the table.
    unmeasurable_groups: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "method": self.method,
            "n_groups": self.n_groups,
            "group_names": self.group_names,
            "pre_calibration_ece": self.pre_calibration_ece,
            "post_calibration_ece": self.post_calibration_ece,
            "improvement": self.improvement,
            "overall_improvement": self.overall_improvement,
            "fallback_groups": self.fallback_groups,
            # Serialised too, or the disclosure exists only for a reader who
            # holds the dataclass: every consumer that persists or ships this
            # result goes through to_dict().
            "unmeasurable_groups": self.unmeasurable_groups,
            "metadata": self.metadata,
        }


#: Why a group in :attr:`GroupCalibrationResult.unmeasurable_groups` is not a
#: measurement. BGL g015, 2026-09-17.
#: ``REASON_NON_FINITE``: the group's own calibrator answered NaN (or inf) on
#: the group's own rows, which is how a base calibrator reports a
#: could-not-check. Before this the NaN flowed straight into
#: expected_calibration_error and raised InvalidDataError("y_prob contains 100
#: NaN value(s)") out of fit(), destroying the OTHER groups' genuine
#: measurements and blaming the caller's y_prob for a NaN the library made.
REASON_NON_FINITE = "non_finite_output"
#: ``REASON_DEGENERATE_MAPPING``: the fitter objected AND the fitted map answers
#: with one value for inputs that genuinely varied, so it carries no
#: information about this group. The ECE of that constant predictor is still a
#: real number and is still reported; what was missing was the label.
REASON_DEGENERATE_MAPPING = "degenerate_mapping"

#: The fallback strategies GroupCalibrator actually implements.
IMPLEMENTED_FALLBACK_STRATEGIES = ("global", "none")

#: Borrowing strategies :class:`IntersectionalCalibrator` actually implements.
IMPLEMENTED_BORROWING_STRATEGIES = ("hierarchical", "global", "none")

#: Provenance labels recorded per intersectional group by
#: :meth:`IntersectionalCalibrator.fit`. ``INTERSECTIONAL`` is the only one that
#: means "this group got its own calibrator fitted on its own rows"; the other
#: three are what the caller got INSTEAD, and none of them is intersectional.
PROVENANCE_INTERSECTIONAL = "intersectional"
PROVENANCE_BORROWED_MARGINAL = "borrowed_marginal"
PROVENANCE_GLOBAL = "global"
PROVENANCE_UNCALIBRATED = "uncalibrated"
#: BGL S2b, 2026-09-17. ``min_group_size`` is a COUNT gate, not a measurability
#: gate, and the two are not the same question. Measured: an intersection of 100
#: rows carrying a single observed class cleared min_group_size=30, so it was
#: labelled ``intersectional``, i.e. "measured on this group's own rows", while
#: its OWN fitter had already warned "No probability-to-outcome mapping can be
#: learned from one class" and the fitted map returned a flat 1.0 for every
#: input in 0.074 ... 0.947. transform() said nothing, because it only looks for
#: labels that are not ``intersectional``. A group big enough to fit and too
#: degenerate to learn anything from is a could-not-check, and it now says so.
PROVENANCE_INTERSECTIONAL_DEGENERATE = "intersectional_degenerate"
#: An intersection that reached transform() but was never seen by fit(). Its
#: rows are in the returned array and they are NOT intersectionally calibrated;
#: before this they were omitted from the transform disclosure's count entirely.
PROVENANCE_UNKNOWN_GLOBAL = "unknown_intersection_global"
PROVENANCE_UNKNOWN_UNCALIBRATED = "unknown_intersection_uncalibrated"

#: Every provenance label that is NOT "this group's own intersectional curve".
#: Used by transform() so a new state cannot be added without the disclosure
#: picking it up: the disclosure tests ``!= PROVENANCE_INTERSECTIONAL`` rather
#: than enumerating, and this tuple exists for callers who want the list.
NON_INTERSECTIONAL_PROVENANCE = (
    PROVENANCE_BORROWED_MARGINAL,
    PROVENANCE_GLOBAL,
    PROVENANCE_UNCALIBRATED,
    PROVENANCE_INTERSECTIONAL_DEGENERATE,
    PROVENANCE_UNKNOWN_GLOBAL,
    PROVENANCE_UNKNOWN_UNCALIBRATED,
)


class IntersectionalProvenanceWarning(UserWarning):
    """Some returned rows are NOT this intersection's own calibration.

    BGL S2b, 2026-09-17. This was a bare ``UserWarning``, which a caller could
    neither select nor escalate on its own. Two consequences, both measured:
    under ``warnings.simplefilter("error", UserWarning)`` ``fit()`` raised and
    left the object ``is_fitted=False`` on any dataset holding a sparse
    intersection (a legitimate input), and a caller who wanted to escalate only
    THIS disclosure had to escalate every UserWarning in the process.

    Subclasses ``UserWarning`` so existing filters and ``pytest.warns`` calls
    that name ``UserWarning`` keep matching::

        warnings.simplefilter("error", IntersectionalProvenanceWarning)
    """


class UnmeasurableGroupWarning(UserWarning):
    """A group's calibrated output is NOT a calibration of that group.

    BGL g015, 2026-09-17. :class:`GroupCalibrator` had no disclosure of this at
    all, while its sibling in this file had grown ``group_provenance_`` for the
    same question. Two measured states reached the caller as clean numbers:

    * the group's own calibrator answered NaN (PlattScaling on constant scores
      correctly reports a could-not-check), which used to abort the whole fit
      with ``InvalidDataError("y_prob contains 100 NaN value(s)")``; and
    * the group's own map collapsed to a single value (isotonic on a single
      observed class), which came back as ``post_calibration_ece=0.0`` and
      ``improvement=0.51`` beside ``fallback_groups=[]``.

    Subclasses ``UserWarning`` so existing filters keep matching, and is its own
    category so a caller can fail closed on exactly this::

        warnings.simplefilter("error", UnmeasurableGroupWarning)
    """


class UnknownGroupWarning(RuntimeWarning):
    """A group reached transform() that fit() never saw, so it is NOT group calibrated.

    Custom category (like ``SidecarUnavailableWarning`` in ``vfairness.llm.scorers``)
    so a caller who wants the STRICT behaviour ``transform`` used to claim in its
    ``Raises:`` section can turn exactly this case into an exception, without
    muting or promoting every other RuntimeWarning::

        warnings.simplefilter("error", UnknownGroupWarning)
    """


def _warn_unknown_groups(
    unknown: List[str], known: List[str], globally_calibrated: bool, caller: str
) -> None:
    """Say, unconditionally, that some rows came back NOT group calibrated.

    R6-1 (2026-09-10): ``transform`` documented "ValueError: If unknown groups
    encountered" and raised nothing. Measured: fit on groups A and B, then
    transform with a group 'Z' present, returned calibrated values with NO
    exception and NO warning; the Z rows were byte-identical to
    ``global_calibrator_.transform`` on the same inputs, i.e. globally calibrated
    and not group calibrated. The lone ``warnings.warn`` on this path fired only
    when ``global_calibrator_`` was None, which ``fit()`` cannot produce (it
    always fits a global calibrator first), so the branch that warned was
    unreachable and the branch that ran was silent. A group-encoding mismatch
    ('M'/'F' vs 'male'/'female', an unseen intersection) therefore looked like a
    successful group calibration. The false ``Raises:`` claim was REMOVED and this
    warning made unconditional; strictness is available through the category.
    """
    fallback = (
        "globally calibrated (NOT group calibrated)"
        if globally_calibrated
        else "returned UNCALIBRATED"
    )
    warnings.warn(
        f"{caller}: {len(unknown)} group(s) not seen during fit(): {sorted(unknown)!r}. "
        f"Rows in those groups were {fallback}. Groups fitted: {sorted(known)!r}. "
        "This is usually a group-encoding mismatch between fit and transform; "
        'raise on it with warnings.simplefilter("error", UnknownGroupWarning).',
        UnknownGroupWarning,
        stacklevel=3,
    )


def validate_fallback_strategy(fallback_strategy: str) -> str:
    """Refuse a fallback strategy that would be accepted and not run.

    F19 (2026-09-09): 'borrow' was listed as an option, documented here as
    "not yet implemented", and silently behaved as 'global' (measured:
    byte-identical output to fallback_strategy='global' on a 20-sample
    group). CalibrationAnalyzer.fit_calibrator re-exposed the same literal
    without even that caveat. Both entry points now come through here.
    """
    if fallback_strategy == "borrow":
        raise NotImplementedError(
            "fallback_strategy='borrow' (borrowing a calibrator from similar "
            "groups) is not implemented; it used to run 'global' silently. "
            f"Implemented strategies: {IMPLEMENTED_FALLBACK_STRATEGIES}."
        )
    if fallback_strategy not in IMPLEMENTED_FALLBACK_STRATEGIES:
        raise ValueError(
            f"Unknown fallback_strategy {fallback_strategy!r}. Implemented "
            f"strategies: {IMPLEMENTED_FALLBACK_STRATEGIES}."
        )
    return fallback_strategy


def validate_borrowing_strategy(borrowing_strategy: str) -> str:
    """Refuse a borrowing strategy that would be accepted and not run.

    BGL S2 (2026-09-16): :class:`IntersectionalCalibrator` validated nothing.
    Measured: ``borrowing_strategy='borrow'`` was ACCEPTED in silence and fell
    through to the ``else`` branch, i.e. it ran 'none', so every small
    intersection came back byte-identical to its input inside an array the
    caller reads as calibrated. That is F19 in this same file, one class down,
    and it gets the same answer: refuse at construction.
    """
    if borrowing_strategy not in IMPLEMENTED_BORROWING_STRATEGIES:
        raise ValueError(
            f"Unknown borrowing_strategy {borrowing_strategy!r}. Implemented "
            f"strategies: {IMPLEMENTED_BORROWING_STRATEGIES}. It used to be "
            f"accepted and silently run 'none' (no calibration at all)."
        )
    return borrowing_strategy


def _mapping_is_degenerate(cal: BaseCalibrator, y_prob_group: np.ndarray) -> bool:
    """Did this group's OWN calibrator actually learn a mapping?

    BGL S2b, 2026-09-17. ``min_group_size`` counts rows; it cannot see
    whether those rows support a probability-to-outcome map. Both halves
    must hold before a group is downgraded, so this stays NARROW:

    1. the base calibrator recorded a fit note of its own, i.e. the fitter
       itself said something was wrong (single observed class, a singular
       Platt design, ...). A healthy fit records none, so a healthy group
       can never reach the second test; and
    2. the fitted map carries no information on this group's own scores. It
       is non-finite (the fitter's own explicit refusal, which counts on its
       own), or it answers with ONE value for inputs that genuinely varied.

    Requiring a fit note first is deliberate. A legitimately flat isotonic map
    (scores that truly do not predict the outcome) is a MEASUREMENT and records
    no fit note, so it keeps ``intersectional``; a mild caveat beside a map
    that still varies keeps ``intersectional`` too.

    The constancy test uses ``np.unique`` on the RAW outputs rather than a
    variance compared with 0.0: an accumulated statistic is not exactly
    zero at every n.
    """
    fit_result = getattr(cal, "fit_result", None)
    notes = list(getattr(fit_result, "warnings", []) or [])
    if not notes:
        return False

    scores = np.asarray(y_prob_group, dtype=float)
    try:
        out = np.asarray(cal.transform(scores), dtype=float)
    except Exception as exc:
        # A map that cannot even be applied to the rows it was fitted on
        # is certainly not this group's measured calibration. SAY SO: a bare
        # `except Exception: return True` discards the error and substitutes
        # an answer, and True is a consequential answer here.
        warnings.warn(
            f"group calibrator for this group could not be applied to the "
            f"rows it was fitted on ({type(exc).__name__}: {exc}), so its "
            f"output is treated as NOT a measured calibration.",
            UserWarning,
            stacklevel=2,
        )
        return True
    if not np.isfinite(out).all():
        # ABOVE the constant-input gate. BGL g015, 2026-09-17: these two tests
        # were the other way round, so an early return meant for "a constant
        # output says nothing here" also swallowed the fitter's own explicit
        # could-not-check. Measured on an intersection of 100 rows all scoring
        # 0.7 with method='platt': PlattScaling warned "the design is
        # rank-deficient ... transform() returns NaN", every one of the 100
        # calibrated values came back NaN, and group_provenance_['M_W'] said
        # 'intersectional', i.e. "measured on this group's own rows".
        # transform() then said nothing at all, because its disclosure only
        # looks for labels that are not 'intersectional'. A NaN is not a
        # constant output; it is the refusal, and it is never a measurement.
        return True

    finite_in = scores[np.isfinite(scores)]
    if np.unique(finite_in).size < 2:
        # The INPUT did not vary, so a FINITE constant output says nothing
        # about the map. Refusing here would refuse an input we cannot judge.
        return False
    return bool(np.unique(out).size < 2)


class GroupCalibrator:
    """
    Group-Specific Calibration Manager.

    Fits separate calibration transformations for each demographic group,
    ensuring that probability scores have consistent interpretation across
    groups. This addresses calibration disparities that may arise even in
    accurate models.

    The calibrator supports multiple underlying calibration methods and
    includes fallback strategies for groups with limited samples.

    Attributes:
        method: Calibration method ('platt', 'isotonic', 'beta', 'temperature')
        min_group_size: Minimum samples required for group-specific calibration
        fallback_strategy: How to handle small groups
        calibrators_: Fitted calibrators per group (after fit())

    Example:
        >>> calibrator = GroupCalibrator(method='isotonic', min_group_size=50)
        >>> calibrator.fit(y_true, y_prob, protected_attr)
        >>> calibrated_probs = calibrator.transform(y_prob_test, protected_attr_test)
        >>> result = calibrator.get_calibration_result()
        >>> print(f"ECE improvement: {result.overall_improvement:.3f}")

    References:
        Pleiss, G., et al. (2017). On Fairness and Calibration. NeurIPS.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: group_calibration. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        method: Union[str, CalibrationMethod] = "isotonic",
        min_group_size: int = 50,
        fallback_strategy: Literal["global", "none"] = "global",
        method_params: Optional[Dict] = None,
    ):
        """
        Initialize GroupCalibrator.

        Args:
            method: Calibration method to use
            min_group_size: Minimum samples for group-specific calibration
            fallback_strategy: Strategy for small groups
                - 'global': Use globally fitted calibrator
                - 'none': No calibration for small groups
                'borrow' (borrowing from similar groups) is not implemented
                and is refused with NotImplementedError; see
                validate_fallback_strategy.
            method_params: Parameters to pass to the calibration method
        """
        self.method = method.value if isinstance(method, CalibrationMethod) else method
        self.min_group_size = min_group_size
        self.fallback_strategy = validate_fallback_strategy(fallback_strategy)
        self.method_params = method_params or {}

        self.is_fitted = False
        # None is a valid value for the 'none' fallback strategy (no group calibrator)
        self.calibrators_: Dict[str, Optional[BaseCalibrator]] = {}
        self.global_calibrator_: Optional[BaseCalibrator] = None
        self.group_manager_: Optional[GroupManager] = None
        self.fit_result_: Optional[GroupCalibrationResult] = None

        self._pre_ece: Dict[str, float] = {}
        self._post_ece: Dict[str, float] = {}

    def fit(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        protected_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "GroupCalibrator":
        """
        Fit group-specific calibrators.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            protected_attr: Protected attribute defining groups
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator

        Notes:
            BGL S2, 2026-09-16. fit() never cleared the fitted state, so a
            second fit() KEPT every calibrator from the first. Measured: fit on
            groups {a, b}, then refit on an a-only dataset, left
            ``calibrators_`` holding ['a', 'b'] and ``fit_result_`` reporting
            ``n_groups=1`` beside ``group_names=['a', 'b']``, with ZERO
            warnings and no ECE entry for 'b'. The reporting damage is
            cosmetic; the damage is in transform(), where membership in
            ``calibrators_`` is read as proof the group was fitted on THIS
            data, so rows in 'b' were calibrated by a model fitted on a
            dataset that no longer exists (max abs difference from the correct
            output: 0.143) while the UnknownGroupWarning path built for exactly
            "these rows are not group calibrated" was never reached, even under
            ``warnings.simplefilter("error", UnknownGroupWarning)``.

            The reset is the first statement in fit(), ahead of validation, so
            every way this call can fail leaves the object honestly UNFITTED
            rather than quietly answering from the previous fit.
        """
        # Reset before anything else: see Notes.
        self.is_fitted = False
        self.calibrators_ = {}
        self.global_calibrator_ = None
        self.group_manager_ = None
        self.fit_result_ = None
        self._pre_ece = {}
        self._post_ece = {}

        y_true = coerce_to_array(y_true, "y_true")
        y_prob = coerce_to_array(y_prob, "y_prob")
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_true, y_prob, protected_attr)
        validate_probabilities(y_prob, "y_prob")

        if sample_weight is not None:
            sample_weight = coerce_to_array(sample_weight, "sample_weight")
            check_consistent_length(y_true, sample_weight)

        self.group_manager_ = GroupManager(protected_attr, min_group_size=1)

        pre_cal_result = expected_calibration_error(
            y_true, y_prob, protected_attr, min_group_size=1
        )
        self._pre_ece = pre_cal_result.group_values or {}

        # Fit global calibrator first (for fallback)
        self.global_calibrator_ = create_calibrator(self.method, **self.method_params)
        self.global_calibrator_.fit(y_true, y_prob, sample_weight)

        fallback_groups = []
        unmeasurable: Dict[str, str] = {}

        for group_name, group_info in self.group_manager_.iter_groups():
            mask = group_info.mask

            if group_info.size < self.min_group_size:
                fallback_groups.append(group_name)

                # 'borrow' no longer reaches here: validate_fallback_strategy
                # refuses it at construction (F19, 2026-09-09).
                if self.fallback_strategy == "global":
                    self.calibrators_[group_name] = self.global_calibrator_
                else:  # 'none'
                    self.calibrators_[group_name] = None
            else:
                group_cal = create_calibrator(self.method, **self.method_params)
                group_weights = sample_weight[mask] if sample_weight is not None else None
                group_cal.fit(y_true[mask], y_prob[mask], group_weights)
                self.calibrators_[group_name] = group_cal
                # BGL g015, 2026-09-17. min_group_size counts rows; it cannot
                # see whether those rows support a probability-to-outcome map.
                # Same question, same narrow test, as the sibling class.
                if _mapping_is_degenerate(group_cal, y_prob[mask]):
                    unmeasurable[group_name] = REASON_DEGENERATE_MAPPING

        calibrated_probs = self._transform_internal(y_prob, protected_attr)

        # BGL g015 / S2B08-13, 2026-09-17. A base calibrator reports a
        # could-not-check by answering NaN (PlattScaling on constant scores does
        # exactly this, correctly). Those NaNs used to be handed straight to
        # expected_calibration_error, which refused the WHOLE array with
        # InvalidDataError("y_prob contains 100 NaN value(s)"): one unmeasurable
        # group destroyed every other group's genuine measurement, and the
        # message blamed the caller's y_prob for a NaN the library produced.
        #
        # The other direction is worse, so the fix is neither "return NaN for
        # everything" nor "substitute something finite": the unmeasurable rows
        # are EXCLUDED from the post measurement, named, and reported as NaN,
        # while every group that can be measured still is.
        finite = np.isfinite(np.asarray(calibrated_probs, dtype=float))
        for group_name, group_info in self.group_manager_.iter_groups():
            if not np.all(finite[group_info.mask]):
                unmeasurable[group_name] = REASON_NON_FINITE

        if finite.all():
            measured_post = expected_calibration_error(
                y_true, calibrated_probs, protected_attr, min_group_size=1
            )
            overall_pre = pre_cal_result.overall_value
        elif int(finite.sum()) >= 2:
            measured_post = expected_calibration_error(
                y_true[finite], calibrated_probs[finite], protected_attr[finite], min_group_size=1
            )
            # Pre is re-measured on the SAME rows, or overall_improvement would
            # subtract two different populations and call the remainder an
            # improvement.
            overall_pre = expected_calibration_error(
                y_true[finite], y_prob[finite], protected_attr[finite], min_group_size=1
            ).overall_value
        else:
            measured_post = None
            overall_pre = float("nan")

        self._post_ece = dict((measured_post.group_values or {}) if measured_post else {})
        # A group whose own calibrator could not answer has no post-calibration
        # ECE. NaN, never the pre value and never 0.0: 0.0 is the best-looking
        # number in the table and would make the unmeasured group the winner.
        for group_name, reason in unmeasurable.items():
            if reason == REASON_NON_FINITE:
                self._post_ece[group_name] = float("nan")

        improvement = {}
        for group in self._pre_ece:
            # .get(..., nan) not .get(..., 0): a missing post value means the
            # group was not measured after calibration, and 0 would read as
            # "calibration changed nothing" instead.
            pre = self._pre_ece.get(group, float("nan"))
            post = self._post_ece.get(group, float("nan"))
            improvement[group] = pre - post

        overall_post = measured_post.overall_value if measured_post else float("nan")
        overall_improvement = overall_pre - overall_post

        self.fit_result_ = GroupCalibrationResult(
            method=self.method,
            n_groups=self.group_manager_.n_groups,
            group_names=list(self.calibrators_.keys()),
            pre_calibration_ece=self._pre_ece,
            post_calibration_ece=self._post_ece,
            improvement=improvement,
            overall_improvement=overall_improvement,
            fallback_groups=fallback_groups,
            unmeasurable_groups=dict(unmeasurable),
            metadata={
                "min_group_size": self.min_group_size,
                "fallback_strategy": self.fallback_strategy,
                # How many rows the post-calibration figures actually cover.
                "post_calibration_rows_measured": int(finite.sum()),
                "post_calibration_rows_total": int(finite.size),
            },
        )

        # Mark fitted BEFORE the disclosure, so a caller running
        # simplefilter("error", UnmeasurableGroupWarning) gets the exception it
        # asked for without also being left an unfitted object. Same ordering,
        # and the same reason, as IntersectionalCalibrator.fit.
        self.is_fitted = True

        if unmeasurable:
            detail = ", ".join(f"{g!r}: {r}" for g, r in sorted(unmeasurable.items()))
            warnings.warn(
                f"GroupCalibrator.fit: {len(unmeasurable)} of "
                f"{len(self.calibrators_)} group(s) did NOT get a usable "
                f"calibration of their own rows ({detail}). "
                f"'{REASON_NON_FINITE}' means that group's calibrator answered "
                f"NaN, so its post-calibration ECE is NaN and its rows are "
                f"excluded from the overall figures; '{REASON_DEGENERATE_MAPPING}' "
                f"means a map was fitted and learned nothing, so it returns one "
                f"value for every input and its ECE is the ECE of a constant. "
                f"Read unmeasurable_groups before reporting any of these rows as "
                f"a calibration result.",
                UnmeasurableGroupWarning,
                stacklevel=2,
            )

        return self

    def _transform_internal(self, y_prob: np.ndarray, protected_attr: np.ndarray) -> np.ndarray:
        """Internal transform without validation."""
        # Only called after fit() has assigned group_manager_, so it is non-None.
        assert self.group_manager_ is not None
        # float, NEVER zeros_like. BGL g015, 2026-09-17: a probability
        # vector of exact 0/1 arrives as int64 (coerce_to_array preserves
        # it and validate_probabilities accepts it), zeros_like inherited
        # that dtype, and every calibrated value was TRUNCATED on
        # assignment. Measured on the same inputs: floats calibrated to
        # [0.0, 0.8, 1.0], ints came back [0, 1], so 0.8 was handed over
        # as 0 inside an array the caller reads as calibrated.
        calibrated = np.zeros(np.shape(y_prob), dtype=float)

        for group_name, group_info in self.group_manager_.iter_groups():
            mask = group_info.mask
            calibrator = self.calibrators_.get(group_name)

            if calibrator is None:
                # No calibration
                calibrated[mask] = y_prob[mask]
            else:
                calibrated[mask] = calibrator.transform(y_prob[mask])

        return calibrated

    def transform(self, y_prob: ArrayLike, protected_attr: ArrayLike) -> np.ndarray:
        """
        Apply group-specific calibration.

        Args:
            y_prob: Predicted probabilities to calibrate
            protected_attr: Protected attribute for group assignment

        Returns:
            Calibrated probabilities

        Raises:
            RuntimeError: If not fitted

        Warns:
            UnknownGroupWarning: If a group here was not seen by fit(). Those
                rows fall through the ladder below, so they come back globally
                calibrated (or uncalibrated), never group calibrated. This is
                a WARNING, not the ValueError this docstring used to claim and
                never raised; see _warn_unknown_groups (R6-1, 2026-09-10) and
                promote it with warnings.simplefilter("error",
                UnknownGroupWarning) if you want the strict behaviour.

        The per-group ladder, in order: the group's own fitted calibrator ->
        None (fallback_strategy='none': pass through, no warning, this was the
        caller's choice at fit time) -> for a group fit() never saw, the global
        calibrator, else pass through, both under UnknownGroupWarning.
        """
        if not self.is_fitted:
            raise RuntimeError("GroupCalibrator not fitted. Call fit() first.")

        y_prob = coerce_to_array(y_prob, "y_prob")
        protected_attr = coerce_to_array(protected_attr, "protected_attr")
        check_consistent_length(y_prob, protected_attr)
        validate_probabilities(y_prob, "y_prob")

        # float, NEVER zeros_like. BGL g015, 2026-09-17: a probability
        # vector of exact 0/1 arrives as int64 (coerce_to_array preserves
        # it and validate_probabilities accepts it), zeros_like inherited
        # that dtype, and every calibrated value was TRUNCATED on
        # assignment. Measured on the same inputs: floats calibrated to
        # [0.0, 0.8, 1.0], ints came back [0, 1], so 0.8 was handed over
        # as 0 inside an array the caller reads as calibrated.
        calibrated = np.zeros(np.shape(y_prob), dtype=float)
        unique_groups = np.unique(protected_attr)
        unknown_groups: List[str] = []

        for group in unique_groups:
            mask = protected_attr == group
            group_str = str(group)

            if group_str in self.calibrators_:
                calibrator = self.calibrators_[group_str]
                if calibrator is None:
                    calibrated[mask] = y_prob[mask]
                else:
                    calibrated[mask] = calibrator.transform(y_prob[mask])
            else:
                # Unknown group - use global calibrator or pass through. Collect
                # them ALL and warn once after the loop (R6-1): the previous form
                # warned only in the pass-through branch, which fit() cannot
                # produce, so the fallback that actually ran was silent.
                unknown_groups.append(group_str)
                if self.global_calibrator_ is not None:
                    calibrated[mask] = self.global_calibrator_.transform(y_prob[mask])
                else:
                    calibrated[mask] = y_prob[mask]

        if unknown_groups:
            _warn_unknown_groups(
                unknown_groups,
                list(self.calibrators_.keys()),
                self.global_calibrator_ is not None,
                "GroupCalibrator.transform",
            )

        # BGL g015, 2026-09-17. fit() warns once, at fit time; this is the call
        # that HANDS OVER the numbers. Measured before this: 100 of 300 returned
        # values were NaN and transform() said nothing at all, so a caller who
        # fits in one place and transforms in another got them with no tell. The
        # NaN is honest on its own (it cannot be mistaken for a probability the
        # way a 0.0 can), but it deserves the reason beside it.
        non_finite = ~np.isfinite(calibrated)
        if non_finite.any():
            affected = sorted({str(g) for g in np.unique(protected_attr[non_finite])})
            warnings.warn(
                f"GroupCalibrator.transform: {int(non_finite.sum())} of "
                f"{calibrated.size} returned value(s) are NaN because their "
                f"group's calibrator could not learn a map from that group's "
                f"rows. Groups affected: {affected!r}. NaN here means COULD NOT "
                f"CALIBRATE, not a probability of zero; see "
                f"get_calibration_result().unmeasurable_groups.",
                UnmeasurableGroupWarning,
                stacklevel=2,
            )

        return np.clip(calibrated, 0, 1)

    def fit_transform(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        protected_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            protected_attr: Protected attribute
            sample_weight: Optional sample weights

        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, y_prob, protected_attr, sample_weight)
        return self.transform(y_prob, protected_attr)

    def get_calibration_result(self) -> GroupCalibrationResult:
        """
        Get detailed calibration results.

        Returns:
            GroupCalibrationResult with pre/post metrics and improvements
        """
        if self.fit_result_ is None:
            raise RuntimeError("No results available. Call fit() first.")
        return self.fit_result_

    def get_group_calibrator(self, group_name: str) -> Optional[BaseCalibrator]:
        """
        Get the fitted calibrator for a specific group.

        Args:
            group_name: Name of the group

        Returns:
            The group's fitted calibrator, or None. None has TWO meanings and
            the warning below is what separates them: the group was fitted with
            ``fallback_strategy='none'`` and deliberately has no calibrator
            (silent), or fit() never saw this group at all
            (UnknownGroupWarning).

        Warns:
            UnknownGroupWarning: If ``group_name`` was not seen by fit().

        Notes:
            BGL g015, 2026-09-17. Both states returned a bare None, so a typo or
            a group-encoding mismatch read as the fact "this group is not
            calibrated". Measured: on a calibrator fitted with groups 'a' and
            'b', ``get_group_calibrator('ZZZ')`` returned None in silence, the
            same answer as a real group that the caller chose to leave
            uncalibrated.
        """
        if not self.is_fitted:
            raise RuntimeError("GroupCalibrator not fitted.")
        key = str(group_name)
        if key not in self.calibrators_:
            warnings.warn(
                f"GroupCalibrator.get_group_calibrator: group {key!r} was not "
                f"seen during fit(), so this None means UNKNOWN GROUP, not "
                f"'this group has no calibrator'. Groups fitted: "
                f"{sorted(self.calibrators_)!r}. This is usually a "
                f"group-encoding mismatch; raise on it with "
                'warnings.simplefilter("error", UnknownGroupWarning).',
                UnknownGroupWarning,
                stacklevel=2,
            )
            return None
        return self.calibrators_[key]

    def evaluate(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        protected_attr: ArrayLike,
        metric_min_group_size: int = 30,
    ) -> Dict[str, Any]:
        """
        Evaluate calibration quality before and after calibration.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            protected_attr: Protected attribute
            metric_min_group_size: Minimum rows a group needs before its ECE is
                measured and before it counts towards the disparity. This is
                the metric's own gate and is INDEPENDENT of the calibrator's
                ``min_group_size``: a group can be large enough to get its own
                calibrator and still be too small to be assessed here. Groups
                below it are named in the returned ``coverage`` block.

        Returns:
            Dictionary with evaluation metrics, plus a ``coverage`` block
            saying which groups the figures above it actually cover.

        Notes:
            BGL g015, 2026-09-17. The returned dict carried no coverage at all,
            while the metrics it calls compute one and discard it here. Measured
            on 400/400/20 rows across three groups with
            ``GroupCalibrator(min_group_size=10)``: the returned
            ``ece_disparity`` was 0.166 over groups a and b, group 'c' was
            absent from ``group_ece`` with no key anywhere saying so, and c's
            own pre-calibration ECE at fit time was 0.529, three times the worst
            group in the figure a reader sees. The smallest group is usually
            the one the analysis exists for.
        """
        if not self.is_fitted:
            raise RuntimeError("GroupCalibrator not fitted.")

        y_true = coerce_to_array(y_true, "y_true")
        y_prob = coerce_to_array(y_prob, "y_prob")
        protected_attr = coerce_to_array(protected_attr, "protected_attr")

        calibrated = self.transform(y_prob, protected_attr)

        # Same could-not-check as in fit(), one method over. Measured before
        # this: evaluate() on a calibrator holding one group whose calibrator
        # answers NaN raised InvalidDataError("y_prob contains 100 NaN
        # value(s)"), so the two healthy groups' real figures were destroyed by
        # the third and the message blamed the caller's y_prob. Those rows are
        # dropped from BOTH sides, so pre, post and improvement still describe
        # one population, and the groups are named in coverage.
        finite = np.isfinite(np.asarray(calibrated, dtype=float))
        not_calibrated = sorted({str(g) for g in np.unique(protected_attr[~finite])})
        all_groups = {str(g) for g in np.unique(protected_attr)}

        if int(finite.sum()) < 2:
            warnings.warn(
                f"GroupCalibrator.evaluate: {int((~finite).sum())} of "
                f"{finite.size} calibrated value(s) are not finite, leaving too "
                f"few rows to measure anything. Every figure below is NaN, which "
                f"means COULD NOT CHECK, not zero error. Groups affected: "
                f"{not_calibrated!r}.",
                UnmeasurableGroupWarning,
                stacklevel=2,
            )
            nan = float("nan")
            empty: Dict[str, float] = {}
            return {
                "pre_calibration": {"overall_ece": nan, "group_ece": empty, "ece_disparity": nan},
                "post_calibration": {"overall_ece": nan, "group_ece": empty, "ece_disparity": nan},
                "improvement": {"overall_ece": nan, "ece_disparity": nan},
                "coverage": {
                    "metric_min_group_size": metric_min_group_size,
                    "groups_assessed": [],
                    "groups_not_assessed": sorted(all_groups),
                    "groups_not_calibrated": not_calibrated,
                    "rows_measured": 0,
                    "rows_total": int(finite.size),
                },
            }

        if not_calibrated:
            # fit() warned once, at fit time, and this is the call that HANDS
            # OVER the numbers: a caller who fits in one place and evaluates in
            # another saw nothing at all.
            warnings.warn(
                f"GroupCalibrator.evaluate: {int((~finite).sum())} of "
                f"{finite.size} row(s) have no finite calibrated value and are "
                f"EXCLUDED from every figure returned. Groups affected: "
                f"{not_calibrated!r}. See the coverage block.",
                UnmeasurableGroupWarning,
                stacklevel=2,
            )

        y_true_m = y_true[finite]
        y_prob_m = y_prob[finite]
        attr_m = protected_attr[finite]
        calibrated_m = calibrated[finite]

        pre_ece = expected_calibration_error(
            y_true_m, y_prob_m, attr_m, min_group_size=metric_min_group_size
        )
        pre_disparity = calibration_disparity(
            y_true_m, y_prob_m, attr_m, min_group_size=metric_min_group_size
        )
        post_ece = expected_calibration_error(
            y_true_m, calibrated_m, attr_m, min_group_size=metric_min_group_size
        )
        post_disparity = calibration_disparity(
            y_true_m, calibrated_m, attr_m, min_group_size=metric_min_group_size
        )

        assessed = sorted(pre_ece.group_values or {})
        not_assessed = sorted(all_groups - set(assessed))

        return {
            "pre_calibration": {
                "overall_ece": pre_ece.overall_value,
                "group_ece": pre_ece.group_values,
                "ece_disparity": pre_disparity.ece_disparity,
            },
            "post_calibration": {
                "overall_ece": post_ece.overall_value,
                "group_ece": post_ece.group_values,
                "ece_disparity": post_disparity.ece_disparity,
            },
            "improvement": {
                "overall_ece": pre_ece.overall_value - post_ece.overall_value,
                "ece_disparity": pre_disparity.ece_disparity - post_disparity.ece_disparity,
            },
            # Not decoration: every number above is computed over
            # groups_assessed only, and an absent group is indistinguishable
            # from a group that was never there.
            "coverage": {
                "metric_min_group_size": metric_min_group_size,
                "groups_assessed": assessed,
                "groups_not_assessed": not_assessed,
                # Groups holding at least one non-finite calibrated value.
                # Those rows are excluded from every figure above. Named
                # separately from groups_not_assessed because "too small to
                # measure" and "could not be calibrated at all" are different
                # answers a reader should not have to guess between.
                "groups_not_calibrated": not_calibrated,
                "rows_measured": int(finite.sum()),
                "rows_total": int(finite.size),
            },
        }


class IntersectionalCalibrator:
    """
    Intersectional Group Calibration.

    Extends group-specific calibration to handle intersectional groups
    (combinations of multiple protected attributes). Includes hierarchical
    borrowing for groups with insufficient samples.

    For example, with gender and race, this calibrator can fit separate
    models for each combination (male+white, female+black, etc.), with
    fallback strategies for sparse intersections.

    Attributes:
        method: Calibration method
        min_group_size: Minimum samples per intersectional group
        borrowing_strategy: How to handle small intersectional groups

    Example:
        >>> # Create intersectional groups
        >>> intersectional_attr = pd.DataFrame({
        ...     'gender': gender,
        ...     'race': race
        ... })
        >>> calibrator = IntersectionalCalibrator(method='isotonic')
        >>> calibrator.fit(y_true, y_prob, intersectional_attr)
        >>> calibrated = calibrator.transform(y_prob_test, intersectional_attr_test)

    References:
        Hebert-Johnson, U., et al. (2018). Multicalibration. ICML.
        Kim, M. P., et al. (2019). Multiaccuracy. AIES.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: intersectional_calibration. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        method: Union[str, CalibrationMethod] = "isotonic",
        min_group_size: int = 30,
        borrowing_strategy: Literal["hierarchical", "global", "none"] = "hierarchical",
        method_params: Optional[Dict] = None,
    ):
        """
        Initialize IntersectionalCalibrator.

        Args:
            method: Calibration method
            min_group_size: Minimum samples per intersectional group
            borrowing_strategy: How to handle small groups
                - 'hierarchical': Borrow from parent groups
                - 'global': Use global calibrator
                - 'none': No calibration for small groups
            method_params: Parameters for calibration method
        """
        self.method = method.value if isinstance(method, CalibrationMethod) else method
        self.min_group_size = min_group_size
        self.borrowing_strategy = validate_borrowing_strategy(borrowing_strategy)
        self.method_params = method_params or {}

        self.is_fitted = False
        # None is a valid value for the 'none' fallback strategy (no group calibrator)
        self.calibrators_: Dict[str, Optional[BaseCalibrator]] = {}
        self.global_calibrator_: Optional[BaseCalibrator] = None
        self.parent_calibrators_: Dict[str, Dict[str, BaseCalibrator]] = {}
        self.attribute_names_: List[str] = []
        # Sample counts per intersectional group, recorded during fit so
        # get_group_statistics can return them (it used to return None).
        self.group_sample_counts_: Dict[str, int] = {}
        # BGL S2, 2026-09-16. What each group's output ACTUALLY is. A raw
        # sample count tells a reader the group was small; it never tells them
        # the group's curve came from somewhere else, or from nowhere. See
        # fit() Notes. Keys are group keys, values are the PROVENANCE_* labels.
        self.group_provenance_: Dict[str, str] = {}
        # Groups that did NOT get their own intersectional calibrator, matching
        # GroupCalibrationResult.fallback_groups on the sibling class.
        self.fallback_groups_: List[str] = []

    @staticmethod
    def _escape_key_part(value) -> str:
        """Escape the join character inside a single attribute value.

        Without escaping, distinct value tuples collide: ('a_b', 'c') and
        ('a', 'b_c') both joined to 'a_b_c' and silently shared one
        calibrator. Values without underscores keep their old keys.
        """
        return str(value).replace("\\", "\\\\").replace("_", "\\_")

    def _create_group_key(self, row) -> str:
        """Create a string key from attribute values (collision safe)."""
        return "_".join(self._escape_key_part(v) for v in row)

    @staticmethod
    def _mapping_is_degenerate(cal: BaseCalibrator, y_prob_group: np.ndarray) -> bool:
        """Delegate to the module-level :func:`_mapping_is_degenerate`.

        BGL g015, 2026-09-17: the body moved to module scope unchanged so
        :class:`GroupCalibrator` can ask the same question. It had no such
        check, which is how a group whose isotonic map collapsed to one
        value was reported with the best ECE in the table. This staticmethod
        stays because tests and callers already name it.
        """
        return _mapping_is_degenerate(cal, y_prob_group)

    def fit(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        protected_attrs,  # DataFrame or dict of arrays
        sample_weight: Optional[ArrayLike] = None,
    ) -> "IntersectionalCalibrator":
        """
        Fit intersectional calibrators.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            protected_attrs: DataFrame with multiple protected attributes
                or dict mapping attribute names to arrays
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator

        Warns:
            UserWarning: If any intersection did not get its OWN calibrator
                fitted on its OWN rows. Those groups are named, together with
                what they got instead, and are listed in ``fallback_groups_``
                and ``group_provenance_``.

        Notes:
            BGL S2, 2026-09-16. Two holes, both silent.

            (1) fit() never cleared ``calibrators_``, so a refit kept every
            calibrator from the previous fit and transform() then treated that
            stale key as proof the group was fitted on THIS data. Same defect
            as :meth:`GroupCalibrator.fit`, same reset.

            (2) A sparse intersection was handed a calibrator fitted on
            something else, with nothing recorded and nothing said. Measured on
            300/300/300/1 rows across gender x race at min_group_size=30 with
            borrowing_strategy='hierarchical': warnings at fit AND transform
            were ``[]``, and ``calibrators_['F_B'] is
            parent_calibrators_['gender']['F']`` was True, i.e. the intersection
            silently received the gender-MARGINAL curve, which is the pooled
            answer a caller asks for intersectional calibration to avoid. With
            borrowing_strategy='none' the same row came back byte-identical to
            its input, never calibrated at all, inside the array the caller
            reads as calibrated. ``get_group_statistics()`` returned only raw
            counts, and a count is not a status: it says the group was small,
            never that its output was borrowed or untouched.

            Three states per group now, in ``group_provenance_``:
            ``intersectional`` (measured on this group's own rows),
            ``borrowed_marginal`` / ``global`` (a real curve, but NOT this
            group's), ``uncalibrated`` (nothing was applied).
        """
        import pandas as pd

        # Reset before anything else, so every failure path leaves the object
        # honestly UNFITTED instead of answering from the previous fit.
        self.is_fitted = False
        self.calibrators_ = {}
        self.global_calibrator_ = None
        self.parent_calibrators_ = {}
        self.attribute_names_ = []
        self.group_sample_counts_ = {}
        self.group_provenance_ = {}
        self.fallback_groups_ = []

        y_true = coerce_to_array(y_true, "y_true")
        y_prob = coerce_to_array(y_prob, "y_prob")
        check_consistent_length(y_true, y_prob)
        validate_probabilities(y_prob, "y_prob")

        if isinstance(protected_attrs, dict):
            protected_attrs = pd.DataFrame(protected_attrs)

        if not isinstance(protected_attrs, pd.DataFrame):
            raise TypeError("protected_attrs must be a DataFrame or dict")

        if len(protected_attrs) != len(y_true):
            raise ValueError("protected_attrs length must match y_true")

        self.attribute_names_ = list(protected_attrs.columns)

        if sample_weight is not None:
            sample_weight = coerce_to_array(sample_weight, "sample_weight")

        self.global_calibrator_ = create_calibrator(self.method, **self.method_params)
        self.global_calibrator_.fit(y_true, y_prob, sample_weight)

        # Fit parent calibrators (one attribute at a time)
        self.parent_calibrators_ = {}
        for attr in self.attribute_names_:
            self.parent_calibrators_[attr] = {}
            unique_vals = protected_attrs[attr].unique()

            for val in unique_vals:
                mask = protected_attrs[attr] == val
                if np.sum(mask) >= self.min_group_size:
                    cal = create_calibrator(self.method, **self.method_params)
                    weights = sample_weight[mask] if sample_weight is not None else None
                    cal.fit(y_true[mask], y_prob[mask], weights)
                    self.parent_calibrators_[attr][str(val)] = cal

        group_keys = protected_attrs.apply(self._create_group_key, axis=1)
        # Iterate unique attribute rows (not keys) so the original values
        # stay available for hierarchical borrowing; re-splitting the key
        # on '_' mangled values that themselves contain underscores.
        unique_rows = protected_attrs.drop_duplicates()

        self.group_sample_counts_ = {}
        borrowed_from: Dict[str, str] = {}
        for _, row in unique_rows.iterrows():
            group_key = self._create_group_key(row)
            mask = group_keys == group_key
            group_size = np.sum(mask)
            self.group_sample_counts_[group_key] = int(group_size)

            if group_size >= self.min_group_size:
                cal = create_calibrator(self.method, **self.method_params)
                weights = sample_weight[mask] if sample_weight is not None else None
                cal.fit(y_true[mask], y_prob[mask], weights)
                self.calibrators_[group_key] = cal
                if self._mapping_is_degenerate(cal, y_prob[mask]):
                    self.group_provenance_[group_key] = PROVENANCE_INTERSECTIONAL_DEGENERATE
                    self.fallback_groups_.append(group_key)
                else:
                    self.group_provenance_[group_key] = PROVENANCE_INTERSECTIONAL
            else:
                # Every branch below produces an output that is NOT this
                # intersection's own calibration, so every branch records what
                # it IS. 'borrow' and any other unknown string no longer reach
                # here at all: validate_borrowing_strategy refuses them at
                # construction, where they used to run 'none' in silence.
                self.fallback_groups_.append(group_key)
                if self.borrowing_strategy == "global":
                    self.calibrators_[group_key] = self.global_calibrator_
                    self.group_provenance_[group_key] = PROVENANCE_GLOBAL
                elif self.borrowing_strategy == "hierarchical":
                    # Find a parent calibrator to borrow from
                    borrowed = False
                    for attr in self.attribute_names_:
                        val = str(row[attr])
                        if val in self.parent_calibrators_.get(attr, {}):
                            self.calibrators_[group_key] = self.parent_calibrators_[attr][val]
                            self.group_provenance_[group_key] = PROVENANCE_BORROWED_MARGINAL
                            borrowed_from[group_key] = f"{attr}={val}"
                            borrowed = True
                            break
                    if not borrowed:
                        self.calibrators_[group_key] = self.global_calibrator_
                        self.group_provenance_[group_key] = PROVENANCE_GLOBAL
                else:  # 'none'
                    self.calibrators_[group_key] = None
                    self.group_provenance_[group_key] = PROVENANCE_UNCALIBRATED

        # Mark the object fitted BEFORE the disclosure. BGL S2b, 2026-09-17:
        # the warning used to come first, so a caller running
        # `warnings.simplefilter("error", ...)` got an exception out of fit()
        # and an object left at is_fitted=False on a perfectly ordinary dataset
        # (measured: "fit raised: UserWarning | is_fitted: False" on 300/300/300/1).
        # Escalating a disclosure must not also destroy the fit it describes.
        self.is_fitted = True

        if self.fallback_groups_:
            detail = ", ".join(
                f"{key!r} (n={self.group_sample_counts_[key]}, "
                f"{self.group_provenance_[key]}"
                + (f" from {borrowed_from[key]}" if key in borrowed_from else "")
                + ")"
                for key in self.fallback_groups_
            )
            n_small = sum(
                1
                for key in self.fallback_groups_
                if self.group_sample_counts_[key] < self.min_group_size
            )
            n_degenerate = len(self.fallback_groups_) - n_small
            reason = (
                f"{n_small} held fewer than min_group_size={self.min_group_size} rows"
                if n_small
                else ""
            )
            if n_degenerate:
                reason += (
                    (" and " if reason else "")
                    + f"{n_degenerate} cleared min_group_size but their own calibrator "
                    f"could not learn a probability-to-outcome map from their rows"
                )
            warnings.warn(
                f"IntersectionalCalibrator.fit: {len(self.fallback_groups_)} of "
                f"{len(self.group_sample_counts_)} intersection(s) were NOT "
                f"intersectionally calibrated ({reason}): {detail}. A borrowed marginal "
                f"curve is the pooled answer, not this intersection's; "
                f"'{PROVENANCE_UNCALIBRATED}' means nothing was applied at all; "
                f"'{PROVENANCE_INTERSECTIONAL_DEGENERATE}' means a curve was fitted on "
                f"this group's own rows and learned nothing from them. Read "
                f"group_provenance_ before treating any of these outputs as an "
                f"intersectional measurement.",
                IntersectionalProvenanceWarning,
                stacklevel=2,
            )

        return self

    def transform(self, y_prob: ArrayLike, protected_attrs) -> np.ndarray:
        """
        Apply intersectional calibration.

        Args:
            y_prob: Predicted probabilities
            protected_attrs: DataFrame with protected attributes

        Returns:
            Calibrated probabilities

        Warns:
            UnknownGroupWarning: If an intersection here was not seen by fit().
                Same silent fallback as GroupCalibrator.transform had, in the
                same file (R6-1, 2026-09-10); an unseen intersection is the
                LIKELY case for this class, not the exotic one.
            UserWarning: If any row in the RETURNED array was not
                intersectionally calibrated, because its group borrowed a
                marginal or global curve at fit time, or got none at all. The
                array cannot carry that distinction: genuinely intersectionally
                calibrated rows, borrowed rows and untouched rows all come back
                as plain floats in one vector. Cross-reference
                ``group_provenance_`` to see which is which.
        """
        import pandas as pd

        if not self.is_fitted:
            raise RuntimeError("IntersectionalCalibrator not fitted.")

        y_prob = coerce_to_array(y_prob, "y_prob")
        validate_probabilities(y_prob, "y_prob")

        if isinstance(protected_attrs, dict):
            protected_attrs = pd.DataFrame(protected_attrs)

        if len(protected_attrs) != len(y_prob):
            raise ValueError("protected_attrs length must match y_prob")

        # float, NEVER zeros_like. BGL g015, 2026-09-17: a probability
        # vector of exact 0/1 arrives as int64 (coerce_to_array preserves
        # it and validate_probabilities accepts it), zeros_like inherited
        # that dtype, and every calibrated value was TRUNCATED on
        # assignment. Measured on the same inputs: floats calibrated to
        # [0.0, 0.8, 1.0], ints came back [0, 1], so 0.8 was handed over
        # as 0 inside an array the caller reads as calibrated.
        calibrated = np.zeros(np.shape(y_prob), dtype=float)
        group_keys = protected_attrs.apply(self._create_group_key, axis=1)
        unknown_groups: List[str] = []

        for group_key in group_keys.unique():
            mask = group_keys == group_key

            if group_key in self.calibrators_:
                cal = self.calibrators_[group_key]
                if cal is not None:
                    calibrated[mask] = cal.transform(y_prob[mask])
                else:
                    calibrated[mask] = y_prob[mask]
            else:
                # Unknown intersection - use global. Silent before R6-1
                # (2026-09-10), which is worse here than in GroupCalibrator:
                # an unseen intersection is ordinary, and the caller asked for
                # INTERSECTIONAL calibration precisely to not get the pooled one.
                unknown_groups.append(str(group_key))
                if self.global_calibrator_ is not None:
                    calibrated[mask] = self.global_calibrator_.transform(y_prob[mask])
                else:
                    calibrated[mask] = y_prob[mask]

        if unknown_groups:
            _warn_unknown_groups(
                unknown_groups,
                list(self.calibrators_.keys()),
                self.global_calibrator_ is not None,
                "IntersectionalCalibrator.transform",
            )

        # BGL S2, 2026-09-16. fit() warns once, at fit time. This is the call
        # that HANDS OVER the numbers, and the fit warning is long gone by
        # then: a caller who fits in one place and transforms in another, or
        # who transforms many times, saw nothing at all. Measured before this:
        # warnings at fit AND transform were both [] while an intersection was
        # being served the gender-marginal curve.
        # BGL S2b, 2026-09-17. Two defects in the dict this count was built from.
        #
        # (a) `if str(key) in self.calibrators_` EXCLUDED every unknown
        #     intersection, which is the one group that certainly did not get
        #     its own curve: it was handed the pooled global one a few lines up.
        #     Measured on 901 fitted rows plus 50 rows of an unseen ('X','Z'):
        #     "1 of 951 returned row(s) are NOT intersectionally calibrated",
        #     when 51 of 951 was the truth, and the 50 rows were missing from
        #     the provenance detail too. Unknown groups now carry their own
        #     label rather than being omitted.
        # (b) `.get(str(key), PROVENANCE_GLOBAL)` supplied the DEFAULT for a key
        #     that was present holding something else only by accident; the
        #     default is now only reached for keys fit() never recorded.
        #
        # `getattr(..., {})` because an object pickled by the released 0.1.0 has
        # no group_provenance_ attribute at all, and transform() died with
        # AttributeError on it (measured) instead of degrading to a disclosure.
        provenance = getattr(self, "group_provenance_", {}) or {}
        has_global = getattr(self, "global_calibrator_", None) is not None
        unknown_label = PROVENANCE_UNKNOWN_GLOBAL if has_global else PROVENANCE_UNKNOWN_UNCALIBRATED
        not_intersectional = {}
        for key in group_keys.unique():
            skey = str(key)
            if skey not in self.calibrators_:
                not_intersectional[skey] = unknown_label
            elif provenance.get(skey, PROVENANCE_GLOBAL) != PROVENANCE_INTERSECTIONAL:
                not_intersectional[skey] = provenance.get(skey, PROVENANCE_GLOBAL)
        if not_intersectional:
            n_rows = int(
                sum(int((group_keys.astype(str) == key).sum()) for key in not_intersectional)
            )
            detail = ", ".join(f"{k!r}: {v}" for k, v in sorted(not_intersectional.items()))
            warnings.warn(
                f"IntersectionalCalibrator.transform: {n_rows} of {len(y_prob)} returned "
                f"row(s) are NOT intersectionally calibrated. Provenance by group: "
                f"{detail}. The returned array cannot show this, so read "
                f"group_provenance_ before reporting any of these values as an "
                f"intersectional result.",
                IntersectionalProvenanceWarning,
                stacklevel=2,
            )

        return np.clip(calibrated, 0, 1)

    def fit_transform(
        self,
        y_true: ArrayLike,
        y_prob: ArrayLike,
        protected_attrs,
        sample_weight: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_prob, protected_attrs, sample_weight)
        return self.transform(y_prob, protected_attrs)

    def get_calibrator_for_group(self, group_values: Dict[str, Any]) -> Optional[BaseCalibrator]:
        """
        Get the calibrator for a specific intersectional group.

        Args:
            group_values: Dict mapping attribute names to values. Every
                attribute in ``attribute_names_`` must be present.

        Returns:
            The intersection's fitted calibrator, or None. None has TWO
            meanings, separated by the warning below: the intersection was
            fitted with ``borrowing_strategy='none'`` and deliberately has no
            calibrator (silent), or fit() never saw this intersection
            (UnknownGroupWarning).

        Raises:
            ValueError: If ``group_values`` omits an attribute the calibrator
                was fitted on. An omitted attribute used to be substituted with
                the empty string, which builds a key for an intersection that
                does not exist and then reports its absence as a fact about the
                caller's group.

        Warns:
            UnknownGroupWarning: If the intersection was not seen by fit().

        Notes:
            BGL g015, 2026-09-17. Measured on a calibrator fitted over gender x
            race: ``get_calibrator_for_group({'gender': 'M'})`` returned None in
            silence (key ``'M_'``), and so did
            ``{'gender': 'Q', 'race': 'Z'}``, the same answer as a real
            intersection the caller chose to leave uncalibrated.
        """
        if not self.is_fitted:
            raise RuntimeError("IntersectionalCalibrator not fitted.")

        missing = [attr for attr in self.attribute_names_ if attr not in group_values]
        if missing:
            raise ValueError(
                f"group_values is missing attribute(s) {missing!r}; this "
                f"calibrator was fitted on {self.attribute_names_!r}. Every "
                f"attribute is needed to name an intersection."
            )

        key = "_".join(self._escape_key_part(group_values[attr]) for attr in self.attribute_names_)
        if key not in self.calibrators_:
            warnings.warn(
                f"IntersectionalCalibrator.get_calibrator_for_group: "
                f"intersection {key!r} was not seen during fit(), so this None "
                f"means UNKNOWN INTERSECTION, not 'this intersection has no "
                f"calibrator'. Intersections fitted: "
                f"{sorted(self.calibrators_)!r}. Raise on it with "
                'warnings.simplefilter("error", UnknownGroupWarning).',
                UnknownGroupWarning,
                stacklevel=2,
            )
            return None
        return self.calibrators_[key]

    def get_group_statistics(self) -> Dict[str, int]:
        """
        Get sample counts per intersectional group.

        A count is not a status: it says a group was small, never that its
        output was borrowed from a marginal curve or left uncalibrated. For
        that, use :meth:`get_group_provenance`.

        Returns:
            Dict mapping group keys to sample counts (as seen during fit)
        """
        if not self.is_fitted:
            raise RuntimeError("IntersectionalCalibrator not fitted.")
        return dict(self.group_sample_counts_)

    def get_group_provenance(self) -> Dict[str, str]:
        """
        Say, per intersectional group, what its calibrated output ACTUALLY is.

        Returns:
            Dict mapping group keys to one of ``'intersectional'`` (a
            calibrator fitted on this group's own rows: the thing the caller
            asked for), ``'borrowed_marginal'`` (a curve fitted on one PARENT
            attribute, i.e. the pooled answer for that attribute value),
            ``'global'`` (the curve fitted on everybody),
            ``'uncalibrated'`` (nothing was applied; the values came back as
            they went in), or ``'intersectional_degenerate'`` (a curve WAS
            fitted on this group's own rows and learned nothing from them, so
            it answers with one value for every input: a could-not-check, not
            a calibration).

            Only ``'intersectional'`` is an intersectional measurement. Every
            other label in ``NON_INTERSECTIONAL_PROVENANCE`` is what was
            returned INSTEAD of one, and they are indistinguishable in the
            transform() output array.
        """
        if not self.is_fitted:
            raise RuntimeError("IntersectionalCalibrator not fitted.")
        # getattr: an object fitted by the released 0.1.0 has no such attribute.
        return dict(getattr(self, "group_provenance_", {}) or {})
