"""
Data Bias Validator for CI/CD Pipelines.

This module provides the DataBiasValidator class for validating data pipelines
against bias and fairness requirements before model training.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd


def _to_numeric_binary(series: "pd.Series") -> "pd.Series":
    """Coerce an outcome column to a numeric 0/1 (float) series for rate maths.

    Real datasets carry the outcome as "Yes"/"No", "true"/"false",
    pandas StringDtype, etc. Calling `.mean()` on those raises
    `dtype 'str' does not support operation 'mean'`. Any 2-class column is
    mapped to 0/1 (larger / truthy class -> 1); already-numeric columns pass
    through. Non-binary non-numeric columns return NaN so callers degrade
    gracefully instead of crashing.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip().str.lower()
    truthy = {
        "yes",
        "true",
        "1",
        "y",
        "t",
        "approved",
        "hired",
        "selected",
        "positive",
        "pass",
        "accept",
        "accepted",
    }
    falsy = {
        "no",
        "false",
        "0",
        "n",
        "f",
        "denied",
        "rejected",
        "negative",
        "fail",
        "failed",
        "decline",
        "declined",
    }
    known = set(s.dropna().unique())
    if known and known <= (truthy | falsy):
        return s.map(lambda v: 1.0 if v in truthy else (0.0 if v in falsy else np.nan))
    uniq = sorted(known)
    if len(uniq) == 2:
        return s.map({uniq[0]: 0.0, uniq[1]: 1.0})
    return pd.Series(np.nan, index=series.index, dtype="float64")


#: The five checks ``DataBiasValidator.validate`` can run, in the order they are
#: attempted. Each is gated on its own ``DataValidationConfig`` flag (and two of
#: them additionally on an outcome column being present and usable), so this
#: tuple is the POPULATION a coverage answer is measured against. It is what
#: ``DataValidationResult.checks_run`` records against, and what the rendering
#: adapter's reader (``rendering.adapters_validation._validation_coverage``)
#: expects to be given.
VALIDATION_CHECKS: tuple = (
    "representation",
    "outcome_disparity",
    "missing_patterns",
    "label_quality",
    "data_hygiene",
)

#: Values returned by ``DataValidationResult.execution_coverage``. Same four
#: words, same meanings, as ``BiasAuditReport.execution_coverage``
#: (``preprocessing.bias_detection.detector``) and as the renderer's
#: ``_COVERAGE_STATES``: one vocabulary across every surface that has to say how
#: much of an analysis actually executed.
COVERAGE_COMPLETE = "complete"
COVERAGE_PARTIAL = "partial"
COVERAGE_NONE = "none"
COVERAGE_UNRECORDED = "unrecorded"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue found during data validation."""

    issue_type: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_groups: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "affected_groups": self.affected_groups,
            "recommendation": self.recommendation,
        }


@dataclass
class DataValidationConfig:
    """Configuration for data bias validation.

    Attributes:
        min_group_fraction: Minimum fraction of data each group should represent.
        max_outcome_ratio: Maximum allowed ratio between group outcome rates.
        min_samples_per_group: Minimum number of samples required per group.
        missing_value_threshold: Maximum allowed fraction of missing values.
        fail_on_warning: Whether to fail validation on warnings (not just errors).
        check_representation: Whether to check group representation balance.
        check_outcome_disparity: Whether to check outcome rate disparities.
        check_missing_patterns: Whether to check for biased missing value patterns.
        check_label_quality: Whether to check for label quality issues.
        check_data_hygiene: Whether to run general data-hygiene checks
            (total sample adequacy, duplicate rows, zero-variance columns).
            Turing AI Fairness Module 4 frames this as extending the
            traditional completeness/accuracy/consistency triad with a
            fourth dimension (equity); these structural checks are the
            substrate that fairness conclusions rest on, so they belong in
            the canonical validator rather than in any single consumer.
        min_total_samples: Below this row count, fairness estimates are
            flagged as indicative (CIs too wide to be conclusive). Module 4
            requires "confidence intervals or warnings about sample size".
        max_duplicate_fraction: Duplicate-row fraction above which the
            dataset is flagged (duplicates silently reweight groups and
            bias every disparity estimate).
    """

    min_group_fraction: float = 0.05
    max_outcome_ratio: float = 2.0
    # float (not int): the public ``representation_thresholds`` mapping is
    # typed Dict[str, float], so this threshold may be set from a float. It is
    # only ever used in numeric comparisons, where int/float behave identically.
    min_samples_per_group: float = 30
    missing_value_threshold: float = 0.1
    fail_on_warning: bool = False
    check_representation: bool = True
    check_outcome_disparity: bool = True
    check_missing_patterns: bool = True
    check_label_quality: bool = True
    check_data_hygiene: bool = True
    min_total_samples: int = 200
    max_duplicate_fraction: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "min_group_fraction": self.min_group_fraction,
            "max_outcome_ratio": self.max_outcome_ratio,
            "min_samples_per_group": self.min_samples_per_group,
            "missing_value_threshold": self.missing_value_threshold,
            "fail_on_warning": self.fail_on_warning,
            "check_representation": self.check_representation,
            "check_outcome_disparity": self.check_outcome_disparity,
            "check_missing_patterns": self.check_missing_patterns,
            "check_label_quality": self.check_label_quality,
            "check_data_hygiene": self.check_data_hygiene,
            "min_total_samples": self.min_total_samples,
            "max_duplicate_fraction": self.max_duplicate_fraction,
        }


@dataclass
class DataValidationResult:
    """Result of data bias validation.

    Attributes:
        passed: Whether the validation passed all checks.
        issues: List of validation issues found.
        metrics: Dictionary of computed metrics.
        summary: Human-readable summary of the validation.
        timestamp: When the validation was performed.
        config: Configuration used for validation.
        checks_run: Which of ``VALIDATION_CHECKS`` actually executed for this
            result. This is the record that separates a dataset every check
            cleared from a run where no check executed. Reproduced 2026-08-27:
            a config with all five check flags off rendered a green PASS on
            ``data_validation.svg`` reading "The data is ready for fairness
            analysis". Three states, never two: a non-empty list means those
            checks ran, ``[]`` means it is recorded that none ran, and ``None``
            means this result does not record it at all (built by hand, or by a
            version before this field existed), which is unknown and must not
            be read as either.

            A ``validate()`` run in which NO check executed now also sets
            ``passed=False`` and carries a ``no_validation_check_ran`` ERROR
            issue, because until 2026-09-10 it still returned ``passed=True``
            while the summary read "neither a pass nor a failure" and
            ``__repr__`` read "PASSED": the object contradicted itself and the
            boolean, which is what CI reads, was the half that said yes. The
            third state survives in ``execution_coverage()`` (``"none"``) and
            in ``__repr__`` (``COULD NOT CHECK``); it is not collapsed into a
            measured failure.
    """

    passed: bool
    issues: List[ValidationIssue]
    metrics: Dict[str, Any]
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[DataValidationConfig] = None
    # Defaults to None, NOT to []. An empty list is the positive statement "it
    # is recorded that no check ran"; a result that never recorded anything must
    # not make that statement, nor the opposite one. Keeping the default at None
    # is also what makes this field backward compatible: every
    # DataValidationResult built elsewhere keeps answering "unrecorded" and
    # renders exactly as it did before.
    checks_run: Optional[List[str]] = None

    def execution_coverage(self) -> str:
        """How much of the validation actually executed.

        Returns one of ``"complete"`` (every check in ``VALIDATION_CHECKS``
        ran), ``"partial"`` (some ran), ``"none"`` (it is recorded that none
        ran) or ``"unrecorded"`` (this result does not say).

        A zero-issue validation is a measurement only under ``"complete"``.
        Under every other value the empty issue list is an absence, and an
        absence is not a finding of no bias.
        """
        if self.checks_run is None:
            return COVERAGE_UNRECORDED
        ran = {str(c) for c in self.checks_run}
        if not ran:
            return COVERAGE_NONE
        if set(VALIDATION_CHECKS).issubset(ran):
            return COVERAGE_COMPLETE
        return COVERAGE_PARTIAL

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [
            i
            for i in self.issues
            if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        ]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def has_critical(self) -> bool:
        """Check if any critical issues were found."""
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config.to_dict() if self.config else None,
            # Both the raw record and the derived word travel with the result.
            # A consumer that only ever sees the dict (the SVG adapter reads
            # to_dict() before it reads the object) must be able to tell a clean
            # validation from one that checked nothing, and `passed` cannot.
            "checks_run": list(self.checks_run) if self.checks_run is not None else None,
            "execution_coverage": self.execution_coverage(),
        }

    def to_junit_xml(self) -> str:
        """Export results in JUnit XML format for CI/CD integration.

        EVERY run emits a ``validation_coverage`` testcase, whatever
        :meth:`execution_coverage` says, because the CI artifact is where the
        coverage question is decided and it is the one surface that cannot ask
        the object a follow-up.

        A run that records executing NO check emits that testcase as a FAILURE.
        Without it this method produced ``tests="0" failures="0"``, which every
        CI UI paints green, for a validation that examined nothing -- the same
        false all-clear ``passed`` used to give. The failing testcase is
        generated HERE, at the CI boundary, and is deliberately NOT pushed into
        ``self.issues``: the issue list is what the SVG adapter reads to choose
        between UNKNOWN and FAIL, and a fabricated issue there would turn a
        could-not-check banner into a measured failure.

        FOUR STATES, NOT TWO (G05, 2026-09-30). That guard covered only
        ``COVERAGE_NONE``, and :meth:`execution_coverage`'s own docstring states
        the rule it has to satisfy: "A zero-issue validation is a measurement
        only under complete. Under every other value the empty issue list is an
        absence, and an absence is not a finding of no bias." Measured on a
        zero-issue result in each of the four states::

            complete    tests="0" failures="0"   0 testcases
            partial     tests="0" failures="0"   0 testcases   <- 4 of 5 checks never ran
            unrecorded  tests="0" failures="0"   0 testcases   <- the DEFAULT
            none        tests="1" failures="1"   1 testcase

        so the three green rows were byte-identical, and ``checks_run=None`` is
        the default of this dataclass, i.e. every result built anywhere other
        than ``validate()``. ``partial`` and ``unrecorded`` are emitted as
        SKIPPED rather than as failures, because neither one measured a problem:
        skipped is JUnit's own could-not-check and no CI UI reads it as a pass.
        Collapsing them into the failure would be the over-correction the
        paragraph above refuses for ``issues``.
        """
        coverage = self.execution_coverage()
        no_check_ran = coverage == COVERAGE_NONE
        unverified = coverage in (COVERAGE_PARTIAL, COVERAGE_UNRECORDED)
        failures = len(self.errors) + (1 if no_check_ran else 0)
        skipped = 1 if unverified else 0
        # +1 for the coverage testcase, which is now unconditional.
        total = len(self.issues) + 1

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="DataBiasValidation" tests="{total}" '
            f'failures="{failures}" skipped="{skipped}">',
        ]

        # Built here, appended AFTER the issue rows, so an existing consumer
        # that reads the first testcase still reads the first ISSUE.
        coverage_lines = ['  <testcase name="validation_coverage">']
        if no_check_ran:
            coverage_lines.append(
                '    <failure message="No validation check ran, so the data was never '
                'examined; this result cannot certify it">'
            )
            coverage_lines.append(
                f"      Enable at least one check on DataValidationConfig "
                f"({escape(', '.join(VALIDATION_CHECKS))})."
            )
            coverage_lines.append("    </failure>")
        elif coverage == COVERAGE_UNRECORDED:
            coverage_lines.append(
                "    <skipped message="
                + quoteattr(
                    "This result does not record which validation checks ran, so an "
                    "empty issue list cannot be read as a clean result: it is a "
                    "could-not-check, not a pass."
                )
                + "/>"
            )
        elif coverage == COVERAGE_PARTIAL:
            ran = [str(c) for c in self.checks_run]
            missing = [c for c in VALIDATION_CHECKS if c not in set(ran)]
            coverage_lines.append(
                "    <skipped message="
                + quoteattr(
                    f"Only part of the validation ran, so nothing here is a finding "
                    f"about what was not examined: {', '.join(ran)} ran, "
                    f"{', '.join(missing)} did not."
                )
                + "/>"
            )
        else:
            coverage_lines.append(
                f"    <system-out>Every validation check ran "
                f"({escape(', '.join(VALIDATION_CHECKS))}), so the issue list is a "
                f"measurement and not an absence of one.</system-out>"
            )
        coverage_lines.append("  </testcase>")

        for issue in self.issues:
            status = (
                "failure"
                if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                else "passed"
            )
            # Escape user-derived text so column names / messages containing
            # <, &, or quotes cannot produce malformed XML that breaks the
            # CI consumer. quoteattr() wraps attribute values in quotes.
            xml_lines.append(f"  <testcase name={quoteattr(str(issue.issue_type))}>")
            if status == "failure":
                xml_lines.append(f"    <failure message={quoteattr(str(issue.message))}>")
                xml_lines.append(f"      {escape(str(issue.details))}")
                xml_lines.append("    </failure>")
            xml_lines.append("  </testcase>")

        xml_lines.extend(coverage_lines)
        xml_lines.append("</testsuite>")
        return "\n".join(xml_lines)

    def __repr__(self) -> str:
        # Three states, never two. It is recorded that no check ran, so this is
        # neither PASSED nor FAILED, and printing "PASSED" here was half of the
        # contradiction described on `checks_run` (the summary already said
        # nothing had been checked).
        if self.execution_coverage() == COVERAGE_NONE:
            status = "COULD NOT CHECK"
        else:
            status = "PASSED" if self.passed else "FAILED"
        return f"DataValidationResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


class DataBiasValidator:
    """Validates data pipelines for bias before model training.

    This class provides comprehensive data validation to detect potential
    bias issues before they propagate into trained models.

    Attributes:
        protected_attributes: List of column names for protected attributes.
        config: Validation configuration.

    Example:
        >>> validator = DataBiasValidator(
        ...     protected_attributes=['gender', 'race'],
        ...     representation_thresholds={'min_group_fraction': 0.05},
        ...     disparity_thresholds={'max_outcome_ratio': 2.0}
        ... )
        >>> result = validator.validate(df, outcome_column='approved')
        >>> if not result.passed:
        ...     print(f"Validation failed: {result.summary}")
        ...     for issue in result.errors:
        ...         print(f"  - {issue.message}")

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: data_bias_validation. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: List[str],
        representation_thresholds: Optional[Dict[str, float]] = None,
        disparity_thresholds: Optional[Dict[str, float]] = None,
        config: Optional[DataValidationConfig] = None,
    ):
        """Initialize the DataBiasValidator.

        Args:
            protected_attributes: List of column names for protected attributes.
            representation_thresholds: Thresholds for representation checks.
                Keys: 'min_group_fraction', 'min_samples_per_group'
            disparity_thresholds: Thresholds for disparity checks.
                Keys: 'max_outcome_ratio'
            config: Full configuration object (overrides other threshold args).
        """
        self.protected_attributes = protected_attributes

        if config is not None:
            self.config = config
        else:
            self.config = DataValidationConfig()
            if representation_thresholds:
                if "min_group_fraction" in representation_thresholds:
                    self.config.min_group_fraction = representation_thresholds["min_group_fraction"]
                if "min_samples_per_group" in representation_thresholds:
                    self.config.min_samples_per_group = representation_thresholds[
                        "min_samples_per_group"
                    ]
            if disparity_thresholds:
                if "max_outcome_ratio" in disparity_thresholds:
                    self.config.max_outcome_ratio = disparity_thresholds["max_outcome_ratio"]

    def validate(
        self,
        df: pd.DataFrame,
        outcome_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> DataValidationResult:
        """Validate a DataFrame for bias issues.

        Args:
            df: DataFrame to validate.
            outcome_column: Name of the outcome/target column (if available).
            feature_columns: Restrict the column-level checks (missing-value
                patterns, constant columns, duplicate rows, and the
                ``n_features`` count) to these columns, plus the protected
                attributes and the outcome column, which are always in scope.
                ``None`` (the default) checks every column of ``df``. Every
                name must be a column of ``df``: a name that is not raises
                ``ValueError``, because a check scoped to a column that does
                not exist has checked nothing while claiming a scope.
                ``metrics["feature_columns"]`` records the scope that ran.

        Returns:
            DataValidationResult with validation status and details.

        Raises:
            ValueError: If ``feature_columns`` names a column ``df`` lacks.
        """
        # F8, 2026-09-09. ``feature_columns`` was documented as "List of feature
        # columns to check" and never referenced: ``feature_columns=
        # ['DOES_NOT_EXIST']`` was accepted and produced metrics identical to
        # an unscoped run. A documented parameter that does nothing is a
        # false claim about what was checked, so it now does exactly what it
        # says or refuses.
        if feature_columns is not None:
            requested = [str(c) for c in feature_columns]
            missing_features = [c for c in requested if c not in df.columns]
            if missing_features:
                raise ValueError(
                    f"feature_columns names column(s) not in the data: {missing_features}. "
                    "A scope that names a missing column would check nothing while "
                    "claiming to have checked it."
                )
            # The protected attributes and the outcome column are always in
            # scope. ``outcome_column`` is Optional: absent means there is no
            # outcome to scope, which is a state of its own, not an empty
            # name. A given-but-missing outcome column is left out here and
            # reported as ``missing_outcome_column`` by the disparity check
            # below, exactly as on the unscoped path.
            scope: List[str] = [str(a) for a in self.protected_attributes if a in df.columns]
            if outcome_column is not None and outcome_column in df.columns:
                scope.append(outcome_column)
            for c in requested:
                if c not in scope:
                    scope.append(c)
            df = df[scope]

        issues: List[ValidationIssue] = []
        metrics: Dict[str, Any] = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "protected_attributes": self.protected_attributes,
        }
        if feature_columns is not None:
            # Two different facts, recorded separately because they differ.
            # feature_columns is what the CALLER ASKED to scope to.
            # columns_measured is what was actually checked, which always also
            # includes the protected attributes and the outcome column: they
            # are never optional, so a scoped run still measures them. Recording
            # only the request would tell a reader the run covered one column
            # when three were measured.
            metrics["feature_columns"] = list(requested)
            metrics["columns_measured"] = list(df.columns)
        # Appended to only where a check has ACTUALLY executed, never where it
        # was merely enabled. Two of the five are gated on an outcome column as
        # well as on their config flag, so "enabled" and "ran" are different
        # facts and only the second one may be recorded here.
        checks_run: List[str] = []

        # Validate protected attributes exist
        missing_attrs = [attr for attr in self.protected_attributes if attr not in df.columns]
        if missing_attrs:
            issues.append(
                ValidationIssue(
                    issue_type="missing_protected_attributes",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Protected attributes not found in data: {missing_attrs}",
                    details={"missing": missing_attrs},
                )
            )
            # An empty list, not None: this run is RECORDED as having executed
            # no check. It is a CRITICAL failure either way, but the record has
            # to be true, and a later reader must not take the empty issue tail
            # for a set of checks that ran and found nothing.
            return self._create_result(issues, metrics, passed=False, checks_run=checks_run)

        # Check representation
        if self.config.check_representation:
            rep_issues, rep_metrics = self._check_representation(df)
            issues.extend(rep_issues)
            metrics["representation"] = rep_metrics
            checks_run.append("representation")

        # Check outcome disparity
        if self.config.check_outcome_disparity and outcome_column:
            if outcome_column not in df.columns:
                issues.append(
                    ValidationIssue(
                        issue_type="missing_outcome_column",
                        severity=ValidationSeverity.ERROR,
                        message=f"Outcome column '{outcome_column}' not found in data",
                    )
                )
            else:
                disp_issues, disp_metrics = self._check_outcome_disparity(df, outcome_column)
                issues.extend(disp_issues)
                metrics["outcome_disparity"] = disp_metrics
                checks_run.append("outcome_disparity")

        # Check missing value patterns
        if self.config.check_missing_patterns:
            missing_issues, missing_metrics = self._check_missing_patterns(df)
            issues.extend(missing_issues)
            metrics["missing_patterns"] = missing_metrics
            checks_run.append("missing_patterns")

        # Check label quality
        if self.config.check_label_quality and outcome_column:
            if outcome_column in df.columns:
                label_issues, label_metrics = self._check_label_quality(df, outcome_column)
                issues.extend(label_issues)
                metrics["label_quality"] = label_metrics
                checks_run.append("label_quality")

        # Check general data hygiene (sample size, duplicates, constant cols)
        if self.config.check_data_hygiene:
            hygiene_issues, hygiene_metrics = self._check_data_hygiene(df)
            issues.extend(hygiene_issues)
            metrics["data_hygiene"] = hygiene_metrics
            checks_run.append("data_hygiene")

        # Determine pass/fail
        has_errors = any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues
        )
        has_warnings = any(i.severity == ValidationSeverity.WARNING for i in issues)

        if has_errors:
            passed = False
        elif has_warnings and self.config.fail_on_warning:
            passed = False
        else:
            passed = True

        return self._create_result(issues, metrics, passed, checks_run=checks_run)

    def _check_representation(
        self, df: pd.DataFrame
    ) -> tuple[List[ValidationIssue], Dict[str, Any]]:
        """Check group representation balance."""
        issues = []
        metrics = {}

        for attr in self.protected_attributes:
            group_counts = df[attr].value_counts()
            group_fractions = group_counts / len(df)

            metrics[attr] = {
                "counts": group_counts.to_dict(),
                "fractions": group_fractions.to_dict(),
            }

            # BGL3 operations-4, 2026-09-27. Three states, never two. When NO
            # group of *attr* carries a value (every row blank, or a frame with
            # no rows) ``value_counts`` is EMPTY, so both threshold tests below
            # iterate over nothing and this check returns zero issues, while
            # validate() goes on to record 'representation' in checks_run.
            # Measured on 300 rows whose only protected attribute was entirely
            # blank:
            #   passed=True, coverage='partial', issues=[('constant_columns',
            #   'warning')], metrics['representation'] =
            #   {'gender': {'counts': {}, 'fractions': {}}}
            # and build_quality_report rendered, off that same result,
            #   [pass] Group representation: Every group of "gender" is large
            #          enough for a reliable read.
            # An absence of small groups among NO groups at all is not a finding
            # that the groups are large enough.
            #
            # ERROR, not WARNING, and modelled on `disparity_not_measurable`
            # below: `passed` is what a CI step branches on and it must not be
            # True for a dimension nobody measured. The issue_type, the message
            # and `representation_measured` all say could-not-check, so no
            # reader is told a representation problem was found. ONE group is
            # still measurable (its count and its share of the frame are real
            # numbers), so the refusal is scoped to zero.
            n_blank = int(df[attr].isna().sum())
            if len(group_counts) == 0:
                metrics[attr]["representation_measured"] = False
                issues.append(
                    ValidationIssue(
                        issue_type="representation_not_measurable",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Group representation for '{attr}' could not be measured: "
                            f"none of the {len(df)} row(s) carry a value for it "
                            f"({n_blank} blank), so there is no group to size. This is a "
                            f"could-not-check result, not a finding that every group is "
                            f"large enough."
                        ),
                        details={
                            "attribute": attr,
                            "n_rows": int(len(df)),
                            "n_rows_without_value": n_blank,
                            "n_groups": 0,
                            "representation_measured": False,
                        },
                        recommendation=(
                            "Supply data in which this attribute has values, or drop it "
                            "from protected_attributes; do not read this run as evidence "
                            "that its groups are adequately represented."
                        ),
                    )
                )
                continue
            metrics[attr]["representation_measured"] = True

            # Check minimum samples per group
            small_groups = group_counts[group_counts < self.config.min_samples_per_group]
            if len(small_groups) > 0:
                issues.append(
                    ValidationIssue(
                        issue_type="insufficient_group_samples",
                        severity=ValidationSeverity.ERROR,
                        message=f"Groups in '{attr}' have fewer than {self.config.min_samples_per_group} samples",
                        details={"group_counts": small_groups.to_dict()},
                        affected_groups=list(small_groups.index),
                        recommendation="Collect more data for underrepresented groups or consider grouping rare categories",
                    )
                )

            # Check minimum fraction per group
            small_fractions = group_fractions[group_fractions < self.config.min_group_fraction]
            if len(small_fractions) > 0:
                issues.append(
                    ValidationIssue(
                        issue_type="underrepresented_groups",
                        severity=ValidationSeverity.WARNING,
                        message=f"Groups in '{attr}' represent less than {self.config.min_group_fraction:.1%} of data",
                        details={"group_fractions": small_fractions.to_dict()},
                        affected_groups=list(small_fractions.index),
                        recommendation="Consider oversampling underrepresented groups or adjusting sampling strategy",
                    )
                )

        return issues, metrics

    def _check_outcome_disparity(
        self, df: pd.DataFrame, outcome_column: str
    ) -> tuple[List[ValidationIssue], Dict[str, Any]]:
        """Check for outcome rate disparities between groups."""
        issues = []
        metrics = {}

        # Coerce string / yes-no / StringDtype outcomes to numeric 0/1 so
        # per-group means are computable (real outcome columns are rarely
        # already numeric).
        _num_outcome = _to_numeric_binary(df[outcome_column])

        for attr in self.protected_attributes:
            # Calculate outcome rates per group
            group_rates = _num_outcome.groupby(df[attr]).mean()
            metrics[attr] = {
                "outcome_rates": group_rates.to_dict(),
            }

            # A group whose rate is NaN has NO DEFINED rate (every outcome in
            # it was blank or unreadable), which is a different fact from a
            # measured rate of zero. NaN fails `min_rate > 0` exactly as 0.0
            # does, so leaving them in wrote an attribute nobody could measure
            # up as a CRITICAL "zero positive outcome rate" finding whose
            # `zero_rate_groups` list was EMPTY. Reproduced 2026-09-11.
            defined_rates = group_rates[group_rates.notna()]
            undefined_groups = [str(g) for g in group_rates.index[group_rates.isna()]]

            if len(defined_rates) < 2:
                # BGL-D, three states never two. Fewer than two groups with a
                # defined outcome rate means no disparity EXISTS to be
                # measured. The bare `continue` that used to stand here left
                # metrics[attr] holding a one-entry `outcome_rates` with no
                # `outcome_ratio` key and raised nothing, while validate() went
                # on to record 'outcome_disparity' in checks_run: coverage read
                # "complete", passed read True and the summary read "no bias
                # issues detected", over data that cannot support any fairness
                # comparison at all. Measured 2026-09-11 on 200 rows with a
                # single gender level: passed=True with zero errors.
                #
                # `outcome_ratio=None` and `disparity_measured=False` are the
                # machine-readable half; the issue is the half a CI gate reads.
                # ERROR, not WARNING: `passed` must not be True for a dimension
                # nobody measured. It is deliberately NOT the `outcome_disparity`
                # issue_type and NOT CRITICAL either -- the issue_type, the
                # message and `disparity_measured` all say could-not-check, so a
                # reader is never told a disparity was found.
                metrics[attr]["outcome_ratio"] = None
                metrics[attr]["disparity_measured"] = False
                if undefined_groups:
                    metrics[attr]["groups_without_defined_rate"] = undefined_groups
                levels = [str(g) for g in defined_rates.index]
                issues.append(
                    ValidationIssue(
                        issue_type="disparity_not_measurable",
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Outcome disparity for '{attr}' could not be measured: "
                            f"{len(defined_rates)} of {len(group_rates)} group(s) have a "
                            f"defined outcome rate, and a comparison needs two. This is a "
                            f"could-not-check result, not a finding of no disparity."
                        ),
                        details={
                            "groups_with_defined_rate": levels,
                            "groups_without_defined_rate": undefined_groups,
                            "outcome_rates": defined_rates.to_dict(),
                        },
                        affected_groups=levels,
                        recommendation=(
                            "Collect data covering at least two groups of this attribute "
                            "with readable outcomes, or drop it from protected_attributes; "
                            "do not read this run as evidence of no disparity."
                        ),
                    )
                )
                # A MEASURED zero is still a measurement, and it does not need a
                # second group. Beta Go-Live Stage 1 audit, 2026-09-11: the
                # `continue` above swallowed the CRITICAL zero_outcome_rate
                # finding that the pre-fix code correctly raised when exactly one
                # group had a defined rate and that rate was a real 0.0. Removing
                # a fabricated disparity must never take a real finding with it:
                # "no group in this data received a single positive outcome" is a
                # fact about the rows that WERE read, and it stands whether or not
                # a comparison group exists.
                zero_groups = [str(g) for g in defined_rates.index[defined_rates == 0]]
                if zero_groups:
                    issues.append(
                        ValidationIssue(
                            issue_type="zero_outcome_rate",
                            severity=ValidationSeverity.CRITICAL,
                            message=(
                                f"Some groups in '{attr}' have zero positive outcome rate "
                                f"(measured, on the {len(defined_rates)} group(s) with a "
                                f"defined rate; the disparity itself could not be measured)"
                            ),
                            details={
                                "zero_rate_groups": zero_groups,
                                "disparity_measured": False,
                            },
                            affected_groups=zero_groups,
                            recommendation=(
                                "Investigate why certain groups have no positive outcomes. "
                                "This finding is independent of the disparity comparison, "
                                "which could not be made on this data."
                            ),
                        )
                    )
                continue

            metrics[attr]["disparity_measured"] = True
            if undefined_groups:
                # Measured over the groups that HAVE a rate, and it says which
                # ones it could not include rather than quietly averaging them
                # away.
                metrics[attr]["groups_without_defined_rate"] = undefined_groups

            # Check outcome ratio
            max_rate = defined_rates.max()
            min_rate = defined_rates.min()

            if min_rate > 0:
                outcome_ratio = max_rate / min_rate
                metrics[attr]["outcome_ratio"] = outcome_ratio

                if outcome_ratio > self.config.max_outcome_ratio:
                    # defined_rates, not group_rates: the named groups must be
                    # ones whose rate was actually measured.
                    max_group = defined_rates.idxmax()
                    min_group = defined_rates.idxmin()
                    issues.append(
                        ValidationIssue(
                            issue_type="outcome_disparity",
                            severity=ValidationSeverity.ERROR,
                            message=f"Outcome ratio for '{attr}' is {outcome_ratio:.2f}x (threshold: {self.config.max_outcome_ratio}x)",
                            details={
                                "max_group": max_group,
                                "max_rate": max_rate,
                                "min_group": min_group,
                                "min_rate": min_rate,
                                "ratio": outcome_ratio,
                            },
                            affected_groups=[str(max_group), str(min_group)],
                            recommendation="Investigate root causes of outcome disparity and consider bias mitigation techniques",
                        )
                    )
            else:
                metrics[attr]["outcome_ratio"] = float("inf")
                issues.append(
                    ValidationIssue(
                        issue_type="zero_outcome_rate",
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Some groups in '{attr}' have zero positive outcome rate",
                        details={
                            "zero_rate_groups": [
                                str(g) for g in defined_rates.index[defined_rates == 0]
                            ]
                        },
                        affected_groups=[str(g) for g in defined_rates.index[defined_rates == 0]],
                        recommendation="Investigate why certain groups have no positive outcomes",
                    )
                )

        return issues, metrics

    def _check_missing_patterns(
        self, df: pd.DataFrame
    ) -> tuple[List[ValidationIssue], Dict[str, Any]]:
        """Check for biased missing value patterns."""
        issues = []
        metrics = {
            "overall_missing_rate": df.isnull().mean().to_dict(),
        }

        for attr in self.protected_attributes:
            group_missing = df.groupby(attr).apply(
                lambda x: x.isnull().mean(), include_groups=False
            )
            metrics[f"{attr}_missing_by_group"] = group_missing.to_dict()

            # Check if missing patterns differ significantly by group
            for col in df.columns:
                if col in self.protected_attributes:
                    continue

                col_missing_by_group = df.groupby(attr)[col].apply(
                    lambda x: x.isnull().mean(), include_groups=False
                )

                if (
                    col_missing_by_group.max() - col_missing_by_group.min()
                    > self.config.missing_value_threshold
                ):
                    issues.append(
                        ValidationIssue(
                            issue_type="biased_missing_pattern",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{col}' has different missing rates across '{attr}' groups",
                            details={
                                "column": col,
                                "missing_rates": col_missing_by_group.to_dict(),
                            },
                            affected_groups=list(col_missing_by_group.index),
                            recommendation="Investigate missing data mechanism and consider group-aware imputation",
                        )
                    )

        return issues, metrics

    def _check_label_quality(
        self, df: pd.DataFrame, outcome_column: str
    ) -> tuple[List[ValidationIssue], Dict[str, Any]]:
        """Check for label quality issues."""
        issues = []
        metrics = {}

        outcome = df[outcome_column]

        # Check for missing labels
        missing_rate = outcome.isnull().mean()
        metrics["missing_label_rate"] = missing_rate

        if missing_rate > self.config.missing_value_threshold:
            issues.append(
                ValidationIssue(
                    issue_type="missing_labels",
                    severity=ValidationSeverity.ERROR,
                    message=f"Outcome column has {missing_rate:.1%} missing values",
                    details={"missing_rate": missing_rate},
                    recommendation="Address missing labels before training",
                )
            )

        # Check for extreme class imbalance (binary outcomes only). A
        # "positive rate" computed over a continuous or multi-class outcome
        # (e.g. the mean of raw incomes) is meaningless, so those are
        # skipped with an explicit note instead of reporting nonsense.
        numeric_outcome = _to_numeric_binary(outcome)
        observed_values = set(numeric_outcome.dropna().unique())
        is_binary = bool(observed_values) and observed_values <= {0.0, 1.0}
        if not is_binary:
            metrics["class_imbalance_check"] = (
                "skipped: outcome is not binary, positive rate is undefined"
            )
        else:
            # Coerce non-numeric binary ("Yes"/"No", StringDtype, ...) before
            # the mean -- `.mean()` on a string column raises.
            positive_rate = numeric_outcome.mean()
            metrics["positive_rate"] = positive_rate

            if positive_rate < 0.01 or positive_rate > 0.99:
                issues.append(
                    ValidationIssue(
                        issue_type="extreme_class_imbalance",
                        severity=ValidationSeverity.WARNING,
                        message=f"Extreme class imbalance detected: {positive_rate:.1%} positive rate",
                        details={"positive_rate": positive_rate},
                        recommendation="Consider resampling techniques or class weighting",
                    )
                )

        return issues, metrics

    def _check_data_hygiene(self, df: pd.DataFrame) -> tuple[List[ValidationIssue], Dict[str, Any]]:
        """General structural data-quality checks.

        Turing AI Fairness Module 4: "Traditional data validation focuses on
        completeness, accuracy, and consistency. We add a crucial fourth
        dimension: equity." Fairness conclusions are only as trustworthy as
        the structural integrity of the data they rest on, so total-sample
        adequacy, duplicate rows, and zero-variance columns are validated
        here as part of the canonical contract.
        """
        issues: List[ValidationIssue] = []
        n_rows = len(df)
        n_dup = int(df.duplicated().sum())
        dup_frac = n_dup / n_rows if n_rows else 0.0
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        metrics = {
            "n_samples": int(n_rows),
            "n_duplicate_rows": n_dup,
            "duplicate_fraction": dup_frac,
            "constant_columns": constant_cols,
        }

        # Total sample adequacy -- Module 4 requires an explicit sample-size
        # warning when estimates become unreliable.
        if n_rows < self.config.min_total_samples:
            issues.append(
                ValidationIssue(
                    issue_type="insufficient_total_samples",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Dataset has only {n_rows} rows "
                        f"(< {self.config.min_total_samples}); fairness estimates "
                        "are indicative, not conclusive"
                    ),
                    details={"n_samples": int(n_rows)},
                    recommendation=(
                        "Collect more data, or report results with explicit "
                        "confidence intervals and a sample-size caveat."
                    ),
                )
            )

        # Duplicate rows silently reweight groups and bias every disparity.
        if dup_frac > self.config.max_duplicate_fraction:
            issues.append(
                ValidationIssue(
                    issue_type="duplicate_rows",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"{n_dup} duplicate rows ({dup_frac:.1%}) exceed the "
                        f"{self.config.max_duplicate_fraction:.0%} threshold"
                    ),
                    details={"n_duplicate_rows": n_dup, "duplicate_fraction": dup_frac},
                    recommendation="De-duplicate before measuring fairness.",
                )
            )
        elif n_dup > 0:
            issues.append(
                ValidationIssue(
                    issue_type="duplicate_rows",
                    severity=ValidationSeverity.WARNING,
                    message=f"{n_dup} duplicate rows ({dup_frac:.1%}) present",
                    details={"n_duplicate_rows": n_dup, "duplicate_fraction": dup_frac},
                    recommendation="Review and de-duplicate if unintended.",
                )
            )

        # Zero-variance columns carry no information and can mask issues.
        if constant_cols:
            issues.append(
                ValidationIssue(
                    issue_type="constant_columns",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"{len(constant_cols)} column(s) have a single constant "
                        f"value: {constant_cols[:5]}"
                    ),
                    details={"constant_columns": constant_cols},
                    recommendation="Drop constant columns; they add no signal.",
                )
            )

        return issues, metrics

    def _create_result(
        self,
        issues: List[ValidationIssue],
        metrics: Dict[str, Any],
        passed: bool,
        checks_run: Optional[List[str]] = None,
    ) -> DataValidationResult:
        """Create a validation result object.

        *checks_run* defaults to None so an external caller of this private
        helper keeps the old behaviour ("this result does not record its
        coverage") rather than silently claiming that no check ran.
        """
        n_errors = len(
            [
                i
                for i in issues
                if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            ]
        )
        n_warnings = len([i for i in issues if i.severity == ValidationSeverity.WARNING])

        if passed:
            if n_warnings > 0:
                summary = f"Validation passed with {n_warnings} warning(s)"
            else:
                summary = "Validation passed - no bias issues detected"
        else:
            summary = f"Validation failed with {n_errors} error(s) and {n_warnings} warning(s)"

        # The summary is the sentence a human reads, and "no bias issues
        # detected" over a population that was never looked at is the same false
        # all-clear the SVG used to paint. Grade over the subset that ran, never
        # withhold the verdict entirely, and state the count that did NOT run
        # right beside the verdict rather than leaving it to be inferred.
        if checks_run is not None:
            ran = [c for c in VALIDATION_CHECKS if c in set(checks_run)]
            not_run = [c for c in VALIDATION_CHECKS if c not in set(checks_run)]
            if not ran:
                # CRITICAL (fail closed): the summary said one thing and
                # ``passed`` said the other. Measured 2026-09-10 with every
                # check flag off, on a frame with 12 rows and a 2-person group:
                #   passed=True  checks_run=[]  coverage='none'
                #   summary="No validation check ran ... neither a pass nor a
                #            failure"
                #   repr="DataValidationResult(PASSED, 0 errors, 0 warnings)"
                # so this class's OWN docstring example, `if not result.passed:
                # print(...)`, printed nothing and the caller went on to
                # train/deploy. Every consumer that reads the boolean -- and the
                # boolean is what a CI step reads -- got a clean bill of health
                # from a run that looked at nothing.
                #
                # A verdict nobody computed is not a pass, so the boolean fails
                # closed. NOTHING is added to ``issues``: an invented ERROR
                # would collapse could-not-check into failed-verification, and
                # ``rendering/adapters_validation._verdict`` reads exactly that
                # list to decide between UNKNOWN and FAIL ("a recorded FAIL
                # still stands, because an issue was raised by something"). The
                # first attempt at this fix did add one, and
                # tests/test_rowlevel_validator_reporting.py caught it: the SVG
                # flipped from a grey UNKNOWN to a red FAIL, which is a
                # different lie about the same run.
                #
                # So all three states survive: ``execution_coverage()`` answers
                # "none", ``__repr__`` says COULD NOT CHECK, the SVG banner
                # says UNKNOWN, ``issues`` stays empty, and only the BOOLEAN --
                # the thing a CI step branches on -- moves to the safe side.
                # ``to_junit_xml()`` says so too, for the same reason.
                #
                # Scoped deliberately to "NO check ran". Partial coverage keeps
                # grading over the subset that ran, exactly as the paragraph
                # above intends: a validator that refused every partial run
                # would block every legitimate one.
                passed = False
                summary = (
                    "No validation check ran, so nothing was checked: this is a "
                    "could-not-check result, not a pass"
                )
            elif not_run:
                summary = (
                    f"{summary} over {len(ran)} of {len(VALIDATION_CHECKS)} checks; "
                    f"{len(not_run)} not run ({', '.join(not_run)})"
                )

        return DataValidationResult(
            passed=passed,
            issues=issues,
            metrics=metrics,
            summary=summary,
            config=self.config,
            checks_run=checks_run,
        )

    def get_explanation(self, result):
        """Generate educational explanations for a validation result.

        The execution coverage of ``result`` travels into the report. Anything
        other than ``"complete"`` adds a coverage explanation, an action item and
        (for ``"none"`` / ``"unrecorded"``) a floor on the report severity, so
        the surface a person reads cannot be more confident than the object it
        explains.

        Parameters
        ----------
        result : DataValidationResult
            The result from :meth:`validate`.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        report = FairnessExplainer.explain(result)

        # BGL5 A-operations-3, 2026-09-27. The four-state coverage record was
        # dropped at the only surface written for a human. `_explain_validation_result`
        # reads `passed` and `issues` and nothing else, and `checks_run` DEFAULTS to
        # None, so every result built outside validate() was explained as a clean
        # bill. Measured on DataValidationResult(passed=True, issues=[], metrics={},
        # summary='built elsewhere', checks_run=None):
        #   before  execution_coverage() 'unrecorded', repr 'PASSED', and the 1150
        #           character report read "Validation PASSED. 0 issue(s) found" plus
        #           "All checks passed; proceed to training with confidence"; the
        #           words 'could not', 'unrecorded', 'coverage' and 'no check'
        #           appeared NOWHERE in it.
        #   after   the same report carries "COULD NOT CHECK on coverage", a
        #           'Validation Coverage' explanation at severity could_not_check,
        #           the action item "Record which validation checks ran ...", and
        #           the confident recommendation is replaced.
        # And with checks_run=[] (it is recorded that nothing ran) the report used
        # to read '[HIGH] Validation FAILED. 0 issue(s) found' with 'Fix all
        # error-level issues', which its own interpretation guide says cannot
        # happen; it now says no check ran, so there is no issue to fix.
        # A 'complete' run is returned byte-identical: the whole block is skipped.
        #
        # The root fix belongs in explainer._explain_validation_result, which is
        # outside this batch's files; handed back as needs_another_batch. This
        # boundary is where DataBiasValidator hands the report to its reader, so
        # the disclosure is applied here for every caller of this method.
        # Scoped to a DataValidationResult by TYPE, not by "has an
        # execution_coverage method": BiasAuditReport has one too, over a
        # DIFFERENT population of checks, and describing its coverage against
        # VALIDATION_CHECKS would be a statement about checks that never applied
        # to it. This method is documented as taking the result of validate().
        if not isinstance(result, DataValidationResult):
            return report
        coverage = result.execution_coverage()
        if coverage == COVERAGE_COMPLETE:
            return report

        from vfairness.evaluation.vfairness_metrics.explainer import MetricExplanation

        ran = [str(c) for c in (getattr(result, "checks_run", None) or [])]
        not_run = [c for c in VALIDATION_CHECKS if c not in set(ran)]
        if coverage == COVERAGE_UNRECORDED:
            sentence = (
                "This result does not record which validation checks ran "
                "(execution_coverage 'unrecorded'), so how much of the data was "
                "examined is a COULD NOT CHECK. The verdict above is not a "
                "statement that the five checks ran."
            )
            action = (
                "Record which validation checks ran (DataBiasValidator.validate "
                "does) before treating this verdict as coverage."
            )
            item_severity = "could_not_check"
        elif coverage == COVERAGE_NONE:
            sentence = (
                "It is recorded that NO validation check ran "
                "(execution_coverage 'none'), so nothing was examined: this is a "
                "COULD NOT CHECK and the empty issue list is an absence, not a "
                "finding of no bias. It is not a measured failure either, so "
                "there is no error-level issue to fix."
            )
            action = (
                "Enable at least one check on DataValidationConfig "
                f"({', '.join(VALIDATION_CHECKS)}) and re-run; nothing was checked."
            )
            item_severity = "could_not_check"
        else:
            sentence = (
                f"Only {len(ran)} of {len(VALIDATION_CHECKS)} validation checks ran "
                f"(execution_coverage 'partial'); not run: {', '.join(not_run)}. The "
                "verdict above covers the checks that ran and says nothing about "
                "the others."
            )
            action = (
                f"{len(not_run)} check(s) did not run ({', '.join(not_run)}); enable "
                "them for full coverage before relying on this verdict."
            )
            item_severity = "low"

        report.summary = f"{report.summary} {sentence}"
        kept = list(report.recommendations or [])
        if coverage == COVERAGE_NONE and not list(getattr(result, "errors", []) or []):
            # "Fix all error-level issues before proceeding to model training" is
            # an instruction about issues this run does not have: the verdict is
            # False because nothing ran, not because something was measured to be
            # wrong. Dropped ONLY in that exact case, so a real error-level
            # finding keeps its action item.
            kept = [r for r in kept if "Fix all error-level issues" not in str(r)]
        report.recommendations = [action] + kept
        if item_severity == "could_not_check":
            # Floor, never a ceiling: a report the explainer already graded
            # 'high' keeps that grade. _UNKNOWN_SEV in the explainer and the
            # renderer is 'medium' for the same reason: an unmeasured verdict is
            # not benign news and not a finding either.
            order = ("info", "low", "medium", "high", "critical")
            current = report.severity if report.severity in order else "info"
            if order.index(current) < order.index("medium"):
                report.severity = "medium"
            # The confident line belongs to a measured pass only.
            for item in report.explanations:
                if "proceed to training with confidence" in str(item.recommendation):
                    item.recommendation = "Do NOT proceed on this result alone: " + sentence
        report.explanations.append(
            MetricExplanation(
                metric_name="Validation Coverage",
                definition=(
                    "How much of the configured validation actually executed, in the "
                    "same four words every other surface uses: complete, partial, "
                    "none, unrecorded."
                ),
                interpretation_guide=(
                    "A zero-issue validation is a measurement only under 'complete'. "
                    "Under every other value the empty issue list is an absence."
                ),
                value=coverage,
                evaluation=sentence,
                benchmark_context=(
                    "checks: " + ", ".join(VALIDATION_CHECKS) + f"; ran: {ran or 'none recorded'}"
                ),
                recommendation=action,
                severity=item_severity,
                related_metrics=[],
            )
        )
        return report

    def validate_batch(
        self,
        dataframes: List[pd.DataFrame],
        outcome_column: Optional[str] = None,
        names: Optional[List[str]] = None,
    ) -> Dict[str, DataValidationResult]:
        """Validate multiple DataFrames.

        Args:
            dataframes: List of DataFrames to validate.
            outcome_column: Name of the outcome column.
            names: Optional names for each DataFrame. There must be exactly one
                name per DataFrame and the names must be distinct: either
                failure would drop frames from the report in silence.

        Returns:
            Dictionary mapping names to validation results, one entry per
            DataFrame.

        Raises:
            ValueError: If ``names`` does not hold exactly one distinct name per
                DataFrame.
        """
        # BGL5 A-operations-3, 2026-09-27. The body was
        # `{name: self.validate(df, outcome_column) for name, df in zip(names, dataframes)}`
        # and zip() TRUNCATES, so a mismatched or repeated name list dropped
        # frames from the batch report without a word. Measured on 3 identical
        # 120-row frames:
        #   before  names=['only_one']      -> keys ['only_one'], 1 result, 0 warnings
        #                                      (2 of 3 frames never validated)
        #           names=['same','same']   -> keys ['same'], 1 result, 0 warnings
        #                                      (the FIRST frame's result overwritten and lost)
        #           dataframes=[]           -> {}, 0 warnings ("every dataset is clean")
        #   after   both name cases raise ValueError naming the count/duplicate,
        #           and the empty batch warns "validated 0 dataset(s)".
        # validate() already refuses the analogous misconfiguration
        # (feature_columns naming a missing column: "A scope that names a missing
        # column would check nothing while claiming to have checked it"), so this
        # is the same class held to the same standard inside one class.
        # Control: names=['a','b'] over 2 frames still returns both results, and
        # names=None still auto-names dataset_0..n, silently.
        if names is None:
            names = [f"dataset_{i}" for i in range(len(dataframes))]
        else:
            names = [str(n) for n in names]
            if len(names) != len(dataframes):
                raise ValueError(
                    f"validate_batch got {len(names)} name(s) for {len(dataframes)} "
                    "dataframe(s). One name per dataframe is required: zip() would "
                    "silently drop the extras, and a batch report short of a dataset "
                    "reads as a dataset that was validated and found clean."
                )
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(
                    f"validate_batch got duplicate name(s) {duplicates}. Results are "
                    "keyed by name, so a repeated name overwrites the earlier "
                    "frame's result and loses it without a trace."
                )

        if not dataframes:
            warnings.warn(
                "validate_batch validated 0 dataset(s) because the batch was empty. "
                "An empty batch report is not a clean bill of health for anything: "
                "nothing was examined.",
                UserWarning,
                stacklevel=2,
            )

        return {name: self.validate(df, outcome_column) for name, df in zip(names, dataframes)}
