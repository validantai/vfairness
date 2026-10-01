"""
Unified Bias Detector

Provides a comprehensive, unified interface for all bias detection capabilities.
Combines historical pattern detection, representation analysis, statistical
disparity testing, and proxy identification into a single workflow.

The BiasDetector class serves as the main entry point for bias detection,
producing standardized audit reports suitable for documentation and compliance.

Example:
    >>> from vfairness.preprocessing.bias_detection import BiasDetector
    >>>
    >>> detector = BiasDetector(
    ...     df,
    ...     protected_attributes=['gender', 'race', 'age'],
    ...     outcome_column='approved'
    ... )
    >>>
    >>> # Full audit
    >>> report = detector.full_audit()
    >>> print(report.summary())
    >>>
    >>> # Or run specific analyses
    >>> historical = detector.detect_historical_patterns()
    >>> representation = detector.detect_representation_bias()
    >>> disparities = detector.analyze_disparities()
    >>> proxies = detector.identify_proxies()
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .historical import (
    HistoricalPatternResult,
    HistoricalRiskLevel,
    detect_historical_patterns,
)
from .proxy import (
    ProxyRiskLevel,
    ProxyVariableResult,
    identify_proxy_variables,
)
from .representation import (
    RepresentationBiasResult,
    RepresentationSeverity,
    detect_representation_bias,
)
from .statistical import (
    EffectSizeInterpretation,
    SignificanceLevel,
    StatisticalDisparityResult,
    analyze_statistical_disparities,
)

#: The four detection modules ``BiasDetector.full_audit`` can run, in the order
#: they appear in the report. These names are what ``full_audit`` records in
#: ``BiasAuditReport.modules_run``, and what the rendering adapter matches its
#: four module tiles against (see ``rendering.adapters._AUDIT_MODULES``, pinned
#: against this tuple by ``tests/test_bias_audit_ran_nothing.py`` so the two
#: cannot drift apart).
AUDIT_MODULES: tuple = ("historical", "representation", "disparities", "proxies")

#: Values returned by ``BiasAuditReport.execution_coverage``.
COVERAGE_COMPLETE = "complete"
COVERAGE_PARTIAL = "partial"
COVERAGE_NONE = "none"
COVERAGE_UNRECORDED = "unrecorded"
#: Every module was INVOKED and none of them had anything to look at, because
#: no protected attribute carried a single observation. R-11, 2026-09-10:
#: ``modules_run`` records INVOCATION, and the four names went in unchanged for
#: an audit over a 500-row frame whose only protected column was entirely NULL.
#: Measured on that frame: execution_coverage() 'complete',
#: overall_risk_score 0.0, recommendations ['No critical bias issues detected.
#: Continue monitoring and perform periodic audits.'], and the rendered SVG
#: "OVERALL RISK 0% MINIMAL". Three sub-modules had each warned UNASSESSED and
#: every one of those statements died in ``warnings``. A reader keying on
#: ``== "complete"`` (which is every reader this library ships, including
#: ``rendering.adapters._audit_coverage``) got the licence to read four empty
#: finding lists as a measurement. This word is what those readers see instead;
#: it is not in their whitelist, so they fall back to withholding the verdict.
COVERAGE_UNASSESSED = "ran_but_assessed_nothing"

#: The smallest per-group row count this audit will read a comparison from.
#: It is the default ``BiasDetector.analyze_disparities`` passes down, and it
#: is named here because ``full_audit`` needs the SAME number to decide whether
#: a protected attribute was assessable at all. BGL-S2 2026-09-16: a 2-row
#: frame reported execution_coverage() 'complete' while all four sub-modules
#: warned UNASSESSED, because the only question anyone asked of the record was
#: "did this attribute have at least one non-null row".
MIN_ASSESSABLE_GROUP_SIZE = 30


def _observation_count(value: Any) -> int:
    """A recorded observation count as a number, or -1 when it cannot be read.

    ``attribute_observations`` is a free-form mapping on a hand-built or older
    report, so the value can be anything: ``None``, a string, a dict. -1 is
    returned for those, which compares below ``MIN_ASSESSABLE_GROUP_SIZE`` and
    therefore reads as "nothing was assessed for this attribute". That is the
    only safe direction: an unreadable count is a could-not-check, and the one
    thing it must never do is buy the word every reader treats as a licence.
    """
    if isinstance(value, bool):
        # A flag is not a count. True would otherwise read as 1.
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


@dataclass
class BiasAuditReport:
    """
    Comprehensive bias audit report combining all detection results.

    Attributes:
        timestamp: When the audit was performed
        dataset_info: Basic information about the analyzed dataset
        protected_attributes: List of protected attributes analyzed
        historical_findings: Results from historical pattern detection
        representation_findings: Results from representation bias detection
        disparity_findings: Results from statistical disparity analysis
        proxy_findings: Results from proxy variable identification
        overall_risk_score: Aggregate risk score (0-1)
        critical_issues: List of critical issues requiring immediate attention
        recommendations: Prioritized list of recommendations
        metadata: Additional metadata about the audit
        modules_run: Which of ``AUDIT_MODULES`` actually executed for this
            report. This is the record that separates a dataset every module
            cleared from a run where no module executed: both end up with four
            empty finding lists and ``overall_risk_score == 0.0``, so the score
            alone cannot tell them apart and any reader of the score alone
            reports the second as a clean bill of health. Three states, never
            two: a non-empty list means those modules ran, ``[]`` means it is
            recorded that none ran, and ``None`` means this report does not
            record it at all (a report built by hand or by an older version),
            which is unknown and must not be read as either.
        attribute_observations: How many non-null rows each REQUESTED protected
            attribute had, keyed by the name the caller asked for. This is the
            second half of the record ``modules_run`` alone could not carry: a
            module that was invoked against an attribute with zero observations
            executed, but it assessed nothing, and its empty finding list is an
            absence exactly as if it had never run. A requested attribute that
            is not a column of the DataFrame at all (a misspelling) is recorded
            here as 0, because that is what the audit had to work with. Same
            three states as ``modules_run``: a mapping is the record, ``{}``
            means it is recorded that no attribute was requested, and ``None``
            means this report does not record it (hand-built, or an older
            version), which is unknown and must not be read as either.
    """

    timestamp: str
    dataset_info: Dict[str, Any]
    protected_attributes: List[str]
    historical_findings: List[HistoricalPatternResult]
    representation_findings: List[RepresentationBiasResult]
    disparity_findings: List[StatisticalDisparityResult]
    proxy_findings: List[ProxyVariableResult]
    overall_risk_score: float
    critical_issues: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Defaults to None, NOT to []. An empty list is the positive statement
    # "it is recorded that no module ran"; a report that never recorded
    # anything must not make that statement, nor the opposite one.
    modules_run: Optional[List[str]] = None
    # Same three-state discipline as modules_run, and defaulting to None for
    # the same reason: {} is the positive statement "no attribute was
    # requested", which a report that never recorded anything must not make.
    attribute_observations: Optional[Dict[str, int]] = None
    # Whether each REQUESTED protected attribute was actually assessed, as
    # opposed to merely having rows. BGL-S2 2026-09-16: a 2-row frame carried
    # attribute_observations {'gender': 2}, and one non-null row was the whole
    # test assessment_coverage() applied, so it answered "complete" while
    # representation had returned INSUFFICIENT_DATA, the disparity module had
    # EXCLUDED both comparisons for holding fewer than min_group_size rows,
    # the proxy module had refused for n < min_sample_size and
    # _calculate_overall_risk had warned that the 0.0 measures nothing. Having
    # a row and being assessed are two different facts and only the first was
    # ever recorded. Same three states: a mapping is the record, {} means it is
    # recorded that no attribute was requested, and None means this report does
    # not record it (hand-built, or an older version), which is unknown and
    # falls back to the observation counts rather than inventing a verdict.
    attribute_assessed: Optional[Dict[str, bool]] = None

    def unassessable_attributes(self) -> Optional[List[str]]:
        """The requested protected attributes nothing was assessed for.

        Read from ``attribute_assessed`` when the report records it, which is
        the fact this question is actually about; an attribute with two rows
        was not assessed by anything even though it "carried observations".
        Falls back to ``attribute_observations`` (zero rows) for a report that
        predates that record, because a count is the only thing such a report
        holds.

        ``None`` when this report records neither, which is unknown and is not
        the same as the empty list (that one says every requested attribute was
        assessed).
        """
        if self.attribute_assessed is not None:
            return sorted(str(a) for a, ok in self.attribute_assessed.items() if not ok)
        if self.attribute_observations is None:
            return None
        # Same bound as the fallback in assessment_coverage, and for the same
        # reason: the two must never contradict each other. BGL5 2026-09-27,
        # measured on attribute_observations={'gender': 2} after
        # assessment_coverage was corrected: this method still answered [], so
        # empty_is_not_a_measurement() read "none of them assessed any requested
        # protected attribute (none were requested)" about a report that
        # requested gender. It now answers ['gender'] and the sentence names it.
        return sorted(
            str(a)
            for a, n in self.attribute_observations.items()
            if _observation_count(n) < MIN_ASSESSABLE_GROUP_SIZE
        )

    def assessment_coverage(self) -> str:
        """How much of the audit had anything to assess.

        Separate from :meth:`execution_coverage`, which answers whether a
        module was INVOKED. Every module can be invoked against a protected
        attribute holding no observation at all, and it then returns the same
        empty finding list a clean attribute produces.

        Returns ``"complete"`` (every requested protected attribute was
        assessed), ``"partial"`` (some were, some were not), ``"none"`` (none
        was, or none was requested) or ``"unrecorded"`` (this report records
        neither assessment nor observation counts).

        The question is whether a module reached a VERDICT about the attribute,
        not whether the column held a row. Those were the same test until
        BGL-S2 2026-09-16, when a 2-row frame answered ``"complete"`` here:
        ``attribute_observations`` said ``{'gender': 2}``, ``if n`` was true,
        and the word licensed four empty finding lists and a 0.0 risk score as
        a clean bill of health while every sub-module had refused in a warning.
        ``full_audit`` now records the verdict itself in ``attribute_assessed``
        and this method reads that; the count is the fallback for reports that
        predate the record and hold nothing better.
        """
        if self.attribute_assessed is not None:
            flags = list(self.attribute_assessed.values())
        elif self.attribute_observations is not None:
            # The fallback reads the count against MIN_ASSESSABLE_GROUP_SIZE,
            # NOT against zero. BGL5 2026-09-27, measured: the BGL-S2 defect was
            # still live here verbatim. BiasAuditReport(dataset_info={'n_rows':
            # 2}, modules_run=list(AUDIT_MODULES),
            # attribute_observations={'gender': 2}) returned
            # assessment_coverage() 'complete', execution_coverage()
            # 'complete', empty_is_not_a_measurement() None,
            # get_critical_count() 0 and an SVG carrying 'MINIMAL', four
            # '>0.00<' tiles and no 'NOT ASSESSABLE' anywhere. It now returns
            # 'none', 'ran_but_assessed_nothing', the refusal sentence, None
            # with a warning, and an SVG carrying 'NOT ASSESSABLE'.
            # `if n` asked "did this attribute have at least one non-null row",
            # which is the exact question the BGL-S2 note above says is not the
            # question: below the minimum group size NO module of this audit can
            # reach a verdict, so a count under it is a could-not-check and not
            # an assessment. 40 and 0 (the two counts the tier1 pin exercises)
            # are unchanged at complete and none, because 40 >= 30 and 0 < 30.
            flags = [
                _observation_count(n) >= MIN_ASSESSABLE_GROUP_SIZE
                for n in self.attribute_observations.values()
            ]
        else:
            return COVERAGE_UNRECORDED
        if not flags:
            return COVERAGE_NONE
        assessed = [f for f in flags if f]
        if not assessed:
            return COVERAGE_NONE
        if len(assessed) == len(flags):
            return COVERAGE_COMPLETE
        return COVERAGE_PARTIAL

    def execution_coverage(self) -> str:
        """How much of the audit executed, and whether it had anything to run on.

        Returns one of ``"complete"`` (every module in ``AUDIT_MODULES`` ran
        AND at least one protected attribute was actually assessed),
        ``"partial"`` (some modules ran), ``"none"`` (it is recorded that none
        ran), ``"ran_but_assessed_nothing"`` (they all ran and not one
        requested protected attribute was assessed, whether because it carried
        no observation at all or because it carried too few for any module to
        reach a verdict) or ``"unrecorded"`` (this report does not say, either
        because it records no ``modules_run`` or because it records no
        assessment half at all, which are two different silences and neither is
        a licence).

        A zero-finding audit is a measurement only under ``"complete"``, and
        that word now requires both halves of the record. It used to require
        only ``modules_run``, which records INVOCATION: an audit over a column
        that was entirely NULL invoked all four modules, returned "complete",
        and licensed four empty finding lists as a clean bill of health. See
        ``COVERAGE_UNASSESSED`` for the measurement. Under every value other
        than ``"complete"`` the empty finding lists are an absence, and an
        absence is not a finding of no bias.
        """
        if self.modules_run is None:
            return COVERAGE_UNRECORDED
        ran = {str(m) for m in self.modules_run}
        if not ran:
            return COVERAGE_NONE
        if not set(AUDIT_MODULES).issubset(ran):
            return COVERAGE_PARTIAL
        # Every module ran. Whether that amounted to an assessment is the
        # second record, and BOTH halves have to be readable before the word
        # "complete" is spoken: this method's own docstring defines it as every
        # module ran AND at least one protected attribute was actually assessed,
        # and a report that does not record the second half establishes no such
        # thing.
        #
        # BGL5 2026-09-27, measured: BiasAuditReport(...,
        # modules_run=list(AUDIT_MODULES)) with attribute_assessed=None and
        # attribute_observations=None returned assessment_coverage()
        # 'unrecorded' and execution_coverage() 'complete' in the same breath,
        # so empty_is_not_a_measurement() was None, get_critical_count() was 0
        # with no warning, and rendering.adapters (which keys on
        # `coverage == "complete" and bool(protected)`) rendered 'OVERALL RISK
        # 0% MINIMAL' with four '0.00' tiles and no could-not-check token on the
        # canvas. It now returns 'unrecorded', which the adapter already has a
        # word for, and the same report renders 'NOT ASSESSABLE'.
        #
        # The earlier comment here read "an unrecorded count leaves the
        # historical answer alone rather than inventing a verdict about a report
        # that does not say". Answering 'complete' IS inventing that verdict;
        # 'unrecorded' is the one that does not say, and it is already in the
        # vocabulary above and in adapters._COVERAGE_STATES.
        assessment = self.assessment_coverage()
        if assessment == COVERAGE_NONE:
            return COVERAGE_UNASSESSED
        if assessment == COVERAGE_UNRECORDED:
            return COVERAGE_UNRECORDED
        return COVERAGE_COMPLETE

    def summary(self) -> str:
        """Generate human-readable summary of the audit."""
        lines = [
            "=" * 70,
            "BIAS DETECTION AUDIT REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Dataset: {self.dataset_info.get('n_rows', 'N/A')} rows, "
            f"{self.dataset_info.get('n_columns', 'N/A')} columns",
            f"Protected Attributes: {', '.join(self.protected_attributes)}",
            "",
            f"Overall Risk Score: {self.overall_risk_score:.1%}",
        ]

        # The score is an aggregate of the modules that ran, so it means
        # nothing on its own when they did not all run: 0.0% here reads as a
        # clean dataset whether four modules cleared it or none looked at it.
        # Say which happened, in the text report as well as on the SVG.
        coverage = self.execution_coverage()
        if coverage == COVERAGE_NONE:
            lines.append("  (no audit module executed, so this score measures nothing)")
        elif coverage == COVERAGE_PARTIAL:
            ran = ", ".join(m for m in AUDIT_MODULES if m in set(self.modules_run or []))
            lines.append(f"  (only these audit modules executed: {ran})")
        elif coverage == COVERAGE_UNASSESSED:
            # The all-null-column case. Every module ran and none of them had a
            # single row to read, so this line has to appear on the TEXT report
            # too: the warnings the sub-modules raised are gone by the time
            # anyone prints this.
            lines.append(
                "  (every audit module executed and none of them assessed anything about "
                "a requested protected attribute: not one of them carried enough "
                "observations for any module to reach a verdict, so this score is not a "
                "measurement of low risk)"
            )
        elif coverage == COVERAGE_UNRECORDED:
            lines.append("  (this report does not record which audit modules executed)")
        elif self.assessment_coverage() == COVERAGE_PARTIAL:
            blank = ", ".join(self.unassessable_attributes() or [])
            lines.append(
                f"  (nothing was assessed for these protected attributes, which carried "
                f"too few observations for any module to reach a verdict: {blank})"
            )

        lines += [
            "",
            "-" * 70,
            "SUMMARY OF FINDINGS",
            "-" * 70,
            f"Historical Pattern Issues: {len(self.historical_findings)}",
            f"Representation Bias Issues: {len(self.representation_findings)}",
            f"Statistical Disparities: {len(self.disparity_findings)}",
            f"Proxy Variables Identified: {len(self.proxy_findings)}",
            "",
        ]

        if self.critical_issues:
            lines.extend(
                [
                    "-" * 70,
                    "CRITICAL ISSUES (Require Immediate Attention)",
                    "-" * 70,
                ]
            )
            for i, issue in enumerate(self.critical_issues[:5], 1):
                lines.append(f"{i}. [{issue['type']}] {issue['description']}")
            lines.append("")

        if self.recommendations:
            lines.extend(
                [
                    "-" * 70,
                    "TOP RECOMMENDATIONS",
                    "-" * 70,
                ]
            )
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "dataset_info": self.dataset_info,
            "protected_attributes": self.protected_attributes,
            "historical_findings": [f.to_dict() for f in self.historical_findings],
            "representation_findings": [f.to_dict() for f in self.representation_findings],
            "disparity_findings": [f.to_dict() for f in self.disparity_findings],
            "proxy_findings": [f.to_dict() for f in self.proxy_findings],
            "overall_risk_score": self.overall_risk_score,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
            # Serialised alongside the score, never instead of it: a consumer
            # reading the JSON needs the same means to tell a cleared dataset
            # from an audit that ran nothing. None stays None (unknown), it is
            # not flattened to [].
            "modules_run": self.modules_run,
            # Serialised for the same reason as modules_run and beside it: a
            # JSON consumer cannot otherwise tell a module that cleared an
            # attribute from one that was handed an empty column. None stays
            # None (unknown), it is not flattened to {}.
            "attribute_observations": self.attribute_observations,
            # Beside the counts, never instead of them: the count says how much
            # data there was, this says whether any module got a verdict out of
            # it. None stays None (unknown), it is not flattened to {}.
            "attribute_assessed": self.attribute_assessed,
            "execution_coverage": self.execution_coverage(),
            "assessment_coverage": self.assessment_coverage(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def empty_is_not_a_measurement(self) -> Optional[str]:
        """Why a count of zero from this audit is not a measurement of zero.

        ``None`` means it IS one: every module in ``AUDIT_MODULES`` executed and
        at least one requested protected attribute was assessed, which is the
        single state :meth:`execution_coverage` reserves for a real result. Any
        other state returns the sentence saying what was missing, in the words
        the summary and the recommendations already use.

        This is the test the two aggregate getters below apply. They return a
        number and a list that a reader grades the dataset by, and neither can
        carry the coverage record on its own: ``0`` and ``[]`` are exactly what
        a four-module audit of a clean dataset produces.
        """
        coverage = self.execution_coverage()
        if coverage == COVERAGE_COMPLETE:
            return None
        if coverage == COVERAGE_NONE:
            return "no audit module executed"
        if coverage == COVERAGE_UNASSESSED:
            blank = ", ".join(self.unassessable_attributes() or []) or "none were requested"
            return (
                "every audit module executed and none of them assessed any requested "
                f"protected attribute ({blank})"
            )
        if coverage == COVERAGE_PARTIAL:
            ran = ", ".join(m for m in AUDIT_MODULES if m in set(self.modules_run or []))
            return f"only these audit modules executed: {ran}"
        # COVERAGE_UNRECORDED, which has TWO routes since BGL5 2026-09-27, and
        # they are different silences. Saying "does not record which modules
        # executed" about a report that lists all four of them would be false,
        # and a reason sentence a reader can check has to be the true one.
        if self.modules_run is not None:
            return (
                "every audit module executed and this report does not record whether any "
                "requested protected attribute was assessed, so it cannot be told apart "
                "from an audit that had nothing to look at"
            )
        return "this report does not record which audit modules executed"

    def get_critical_count(self) -> Optional[int]:
        """How many critical issues this audit FOUND, or None if it could not tell.

        Three states, never two:

        * a positive count is always returned, whatever the coverage. A finding
          is evidence and withholding it would throw the evidence away, which is
          the worse defect of the two.
        * ``0`` means every audit module executed, something was assessed, and
          not one critical issue was found. That is a measurement.
        * ``None`` means no critical issue was found by an audit that was in no
          position to find one (see :meth:`empty_is_not_a_measurement`), so the
          zero is an absence of measurement and is not reported as a count. The
          raw ``critical_issues`` list is still there for a caller that wants
          what the modules which did run returned.

        BGL g021 2026-09-17: measured on a 2-row frame, this answered ``0``
        while the report's own ``execution_coverage()`` said
        ``'ran_but_assessed_nothing'``, its recommendation said "Nothing here
        clears the data" and four sub-modules had each refused in a warning. A
        caller printing "Critical issues: 0" beside those had no way to tell it
        from a clean audit of a million rows.
        """
        if self.critical_issues:
            return len(self.critical_issues)
        reason = self.empty_is_not_a_measurement()
        if reason is None:
            return 0
        warnings.warn(
            f"BiasAuditReport.get_critical_count: {reason}, so no count of critical "
            "issues is reported. A count of 0 here would not be a measurement of no "
            "critical issue. Read execution_coverage() and assessment_coverage() for "
            "what was examined.",
            UserWarning,
            stacklevel=2,
        )
        return None

    def get_high_risk_features(self) -> Optional[List[str]]:
        """Features this audit FOUND to be high risk, or None if it could not tell.

        Checks historical and proxy findings for CRITICAL or HIGH risk levels.
        Uses string value comparison to avoid enum identity issues that can
        occur when modules are reloaded (e.g. in notebooks or hot-reload).

        Same three states as :meth:`get_critical_count`, and for the same
        reason: an empty list is what a clean dataset produces AND what an audit
        that screened nothing produces. A non-empty list is always returned; an
        empty one is returned only when the audit was in a position to find
        something (see :meth:`empty_is_not_a_measurement`), and is ``None``
        otherwise. The proxy half is the one this matters most for: the proxy
        module refuses below its own ``min_sample_size`` and its refusal reaches
        this list as nothing at all.
        """
        high_risk_values = {"critical", "high"}
        features = set()

        for h in self.historical_findings:
            if h.risk_level.value in high_risk_values:
                features.add(h.feature)

        for p in self.proxy_findings:
            if p.risk_level.value in high_risk_values:
                features.add(p.feature)

        if features:
            return sorted(features)
        reason = self.empty_is_not_a_measurement()
        if reason is None:
            return []
        warnings.warn(
            f"BiasAuditReport.get_high_risk_features: {reason}, so no high-risk feature "
            "list is reported. An empty list here would not be a finding that no feature "
            "is high risk. Read execution_coverage() and assessment_coverage() for what "
            "was examined.",
            UserWarning,
            stacklevel=2,
        )
        return None

    def to_svg(self, save_path: Optional[str] = None) -> str:
        """Render this report as a polished SVG dashboard.

        Requires ``jinja2`` (install with ``pip install jinja2``).

        Parameters
        ----------
        save_path : str, optional
            If provided, also write the SVG to this file.

        Returns
        -------
        str
            Complete SVG markup.
        """
        from vfairness.rendering import bias_audit_to_svg

        return bias_audit_to_svg(self, save_path=save_path)


#: The one item in a bias-audit ``ExplanationReport`` that grades the aggregate
#: score. It is named here because it is the item whose verdict is fabricated
#: when the audit had nothing to aggregate.
_OVERALL_RISK_EXPLANATION = "Overall Bias Risk Score"

#: Severity for an explanation that was never graded, spelled the way
#: ``evaluation.vfairness_metrics.explainer.MetricExplanation`` documents it: a
#: THIRD state, never collapsed into 'info', which is what a genuinely graded
#: and genuinely benign metric receives.
_COULD_NOT_CHECK_SEV = "could_not_check"

#: Report-level floor for an explanation that could not check. ``ExplanationReport.severity``
#: has no could-not-check member, so an unmeasured audit takes the same floor band
#: ``explainer._UNKNOWN_SEV`` and ``rendering.explain._UNKNOWN_SEV`` use. It is only
#: ever raised from 'info', never lowered: a real finding keeps its own severity.
_UNKNOWN_REPORT_SEV = "medium"


def _qualify_explanation(explanation: Any, report: Any) -> Any:
    """Carry the audit's coverage into the explanation a reader is handed.

    ``explainer._explain_bias_audit`` bands ``overall_risk_score`` and nothing
    else, so an audit that assessed nothing came back graded: measured on a
    2-row frame (BGL g021, 2026-09-17) it returned severity 'info' on an item
    reading "The overall risk score of 0.00 is in the MINIMAL band. The dataset
    appears suitable for training with standard monitoring", over a report whose
    own ``execution_coverage()`` said 'ran_but_assessed_nothing'. The refusal
    was present in that same object, in the recommendations, and one line above
    the sentence that contradicted it.

    Nothing is removed here. Findings, recommendations and per-module items are
    untouched, the score stays on the item as its value, and the severity is
    only ever raised, so a real CRITICAL finding under partial coverage keeps
    its own grade. Only the verdict ON the aggregate is withdrawn, because that
    is the one nothing measured.

    A report that cannot answer ``empty_is_not_a_measurement`` (anything that is
    not a :class:`BiasAuditReport`) is returned untouched: this function states
    a coverage record, and it must not invent one.
    """
    if not isinstance(report, BiasAuditReport):
        return explanation

    reason = report.empty_is_not_a_measurement()
    if reason is None:
        return explanation

    statement = (
        f"COULD NOT CHECK: {reason}, so the overall risk score of "
        f"{report.overall_risk_score:.2f} aggregates nothing and is not a measurement "
        "of low risk. An empty finding list from this audit is an absence, not a "
        "finding of no bias."
    )

    summary = str(getattr(explanation, "summary", "") or "")
    explanation.summary = f"{statement} {summary}".strip()

    for item in getattr(explanation, "explanations", None) or []:
        if getattr(item, "metric_name", None) != _OVERALL_RISK_EXPLANATION:
            continue
        item.evaluation = statement
        item.severity = _COULD_NOT_CHECK_SEV
        item.recommendation = (
            "Re-run the audit with every module enabled, over a populated protected "
            "attribute carrying enough rows per group for a comparison, before reading "
            "any verdict from this score."
        )

    if getattr(explanation, "severity", None) == "info":
        explanation.severity = _UNKNOWN_REPORT_SEV

    return explanation


class BiasDetector:
    """
    Unified bias detection system for comprehensive data auditing.

    Combines four core detection capabilities:
    A. Historical Pattern Detection
    B. Representation Bias Detection
    C. Statistical Disparity Analysis
    D. Proxy Variable Identification

    The detector can be used for individual analyses or a full comprehensive
    audit that produces a standardized report.

    Args:
        df: DataFrame to analyze
        protected_attributes: List of protected attribute column names
        outcome_column: Optional column representing the target/outcome
        feature_columns: Optional list of feature columns to analyze
        benchmarks: Optional population benchmarks for representation analysis
        config: Optional configuration dictionary

    Example:
        >>> detector = BiasDetector(
        ...     df,
        ...     protected_attributes=['gender', 'race'],
        ...     outcome_column='loan_approved'
        ... )
        >>>
        >>> # Full audit
        >>> report = detector.full_audit()
        >>>
        >>> # Individual analyses
        >>> detector.detect_historical_patterns()
        >>> detector.detect_representation_bias()
        >>> detector.analyze_disparities()
        >>> detector.identify_proxies()

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: bias_audit. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protected_attributes: List[str],
        *,
        outcome_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.df = df
        # The names the CALLER asked for, kept beside the filtered list. A
        # misspelled attribute is dropped by the filter below and then exists
        # nowhere on the report, so the audit reported the same four empty
        # finding lists and the same 0.0 as a dataset every module cleared.
        # full_audit records an observation count of 0 against these names.
        self.requested_protected_attributes = list(protected_attributes)
        self.protected_attributes = [a for a in protected_attributes if a in df.columns]
        self.outcome_column = outcome_column
        self.feature_columns = feature_columns
        self.benchmarks = benchmarks or {}
        self.config = config or {}

        if not self.protected_attributes:
            warnings.warn(
                "No valid protected attributes found in DataFrame. "
                "Bias detection capabilities will be limited.",
                UserWarning,
            )

        # Memoization caches keyed by the arguments that change the result.
        # A single slot per module returned the FIRST call's result to every
        # later call whatever it asked for, which is a measurement nobody made
        # at the parameters the caller supplied. Measured on a 600-row frame
        # (BGL g021, 2026-09-17): identify_proxies(correlation_threshold=0.05)
        # then identify_proxies(correlation_threshold=0.99) returned the same
        # two proxies, the second call's true answer being one; and
        # analyze_disparities(min_group_size=100000) returned the 2 comparisons
        # found at 30, silently, over a frame where every group is too small.
        # The same shape was fixed in post_processing.calibration.analyzer by
        # audit wave 4 (tests/test_audit_wave4_postproc.py::TestAnalyzerCacheKeys).
        self._historical_results: Dict[Any, List[HistoricalPatternResult]] = {}
        self._representation_results: Dict[Any, List[RepresentationBiasResult]] = {}
        self._disparity_results: Dict[Any, List[StatisticalDisparityResult]] = {}
        self._proxy_results: Dict[Any, List[ProxyVariableResult]] = {}

    def full_audit(
        self,
        *,
        include_historical: bool = True,
        include_representation: bool = True,
        include_disparities: bool = True,
        include_proxies: bool = True,
        include_intersectional: bool = True,
    ) -> BiasAuditReport:
        """
        Perform comprehensive bias audit.

        Runs all four detection modules and produces a unified audit report.

        Args:
            include_historical: Whether to include historical pattern detection
            include_representation: Whether to include representation analysis
            include_disparities: Whether to include statistical disparity analysis
            include_proxies: Whether to include proxy variable identification
            include_intersectional: Whether to include intersectional analysis

        Returns:
            BiasAuditReport with comprehensive findings
        """
        historical = []
        representation = []
        disparities = []
        proxies = []

        # Which modules EXECUTED, recorded as they execute rather than derived
        # afterwards from the findings. An empty finding list is produced both
        # by a module that ran and cleared the data and by a module that was
        # switched off, so the lists cannot be read backwards into this record.
        modules_run: List[str] = []

        # How much there was to run ON, recorded from the frame itself before
        # any module is called. Invocation and assessment are two different
        # facts and only the first one was being kept: a 500-row frame whose
        # only protected column was entirely NULL invoked all four modules,
        # so modules_run held all four names and execution_coverage() said
        # "complete" over an audit that had read zero observations.
        attribute_observations: Dict[str, int] = {
            attr: (int(self.df[attr].notna().sum()) if attr in self.df.columns else 0)
            for attr in self.requested_protected_attributes
        }

        if include_historical:
            historical = self.detect_historical_patterns()
            modules_run.append("historical")

        if include_representation:
            representation = self.detect_representation_bias(
                include_intersectional=include_intersectional
            )
            modules_run.append("representation")

        if include_disparities:
            disparities = self.analyze_disparities(include_intersectional=include_intersectional)
            modules_run.append("disparities")

        if include_proxies:
            proxies = self.identify_proxies()
            modules_run.append("proxies")

        # Whether each requested attribute was ASSESSED, decided after the
        # modules have run and from what they returned. Recorded beside the row
        # counts rather than derived from them: two rows is a count, and it is
        # not an assessment.
        attribute_assessed: Dict[str, bool] = {
            attr: self._was_assessed(attr, representation, disparities, proxies)
            for attr in self.requested_protected_attributes
        }

        risk_score = self._calculate_overall_risk(historical, representation, disparities, proxies)

        critical_issues = self._identify_critical_issues(
            historical, representation, disparities, proxies
        )

        recommendations = self._generate_recommendations(
            historical,
            representation,
            disparities,
            proxies,
            modules_run=modules_run,
            attribute_observations=attribute_observations,
            attribute_assessed=attribute_assessed,
        )

        report = BiasAuditReport(
            timestamp=datetime.now().isoformat(),
            dataset_info={
                "n_rows": len(self.df),
                "n_columns": len(self.df.columns),
                "columns": list(self.df.columns),
                "protected_attributes": self.protected_attributes,
                "outcome_column": self.outcome_column,
            },
            protected_attributes=self.protected_attributes,
            historical_findings=historical,
            representation_findings=representation,
            disparity_findings=disparities,
            proxy_findings=proxies,
            overall_risk_score=risk_score,
            critical_issues=critical_issues,
            recommendations=recommendations,
            metadata={
                "config": self.config,
                "include_intersectional": include_intersectional,
            },
            modules_run=modules_run,
            attribute_observations=attribute_observations,
            attribute_assessed=attribute_assessed,
        )

        return report

    def _was_assessed(
        self,
        attribute: str,
        representation: List[RepresentationBiasResult],
        disparities: List[StatisticalDisparityResult],
        proxies: List[ProxyVariableResult],
    ) -> bool:
        """Did anything in this audit actually assess *attribute*?

        Two kinds of evidence, and either one is enough:

        1. A module RETURNED a graded verdict naming this attribute. A
           representation result is a verdict unless its severity is that
           module's own refusal, ``INSUFFICIENT_DATA``; a disparity finding or
           a proxy finding naming the attribute only exists because a test ran
           on its rows.
        2. The attribute's own group structure could support a comparison at
           all: at least two distinct non-null groups, each holding at least
           ``MIN_ASSESSABLE_GROUP_SIZE`` rows, which is the threshold this
           detector hands to ``analyze_disparities``. This second limb matters
           because a module that TESTED and found nothing worth reporting
           returns the same empty list as a module that could not test, and
           without it a genuinely clean 900-row frame would be reported as
           assessing nothing, which is the opposite defect.

        Historical patterns are deliberately not evidence here. That module
        reads column NAMES against known patterns of discrimination and never
        reads a row of this attribute, so a redlining hit on ``zipcode`` says
        nothing about whether ``gender`` was assessed.
        """
        for r in representation:
            if (
                str(getattr(r, "attribute", "")) == attribute
                and getattr(r, "severity", None) != RepresentationSeverity.INSUFFICIENT_DATA
            ):
                return True
        for d in disparities:
            if str(getattr(d, "protected_attribute", "")) == attribute:
                return True
        for pr in proxies:
            if str(getattr(pr, "protected_attribute", "")) == attribute:
                return True

        if attribute not in self.df.columns:
            return False
        # dropna is pandas' default for value_counts, so an all-NULL column
        # yields an empty series and falls through to False, and a column with
        # two rows yields two counts of 1.
        counts = self.df[attribute].value_counts()
        testable_groups = int((counts >= MIN_ASSESSABLE_GROUP_SIZE).sum())
        return testable_groups >= 2

    def detect_historical_patterns(
        self,
        *,
        min_confidence: float = 0.3,
        custom_patterns: Optional[Dict] = None,
    ) -> List[HistoricalPatternResult]:
        """
        Detect historical bias patterns in dataset features.

        Analyzes column names and values against known patterns of
        historical discrimination (redlining, segregation, etc.).

        Args:
            min_confidence: Minimum confidence threshold
            custom_patterns: Additional custom patterns to check

        Returns:
            List of HistoricalPatternResult objects
        """
        key = (min_confidence, repr(custom_patterns))
        if key not in self._historical_results:
            self._historical_results[key] = detect_historical_patterns(
                self.df,
                protected_attributes=self.protected_attributes,
                custom_patterns=custom_patterns,
                min_confidence=min_confidence,
            )
        return self._historical_results[key]

    def detect_representation_bias(
        self,
        *,
        underrepresentation_threshold: float = 0.8,
        include_intersectional: bool = True,
    ) -> List[RepresentationBiasResult]:
        """
        Detect representation bias in protected attribute distributions.

        Compares dataset distributions against population benchmarks
        to identify under/overrepresentation.

        Args:
            underrepresentation_threshold: Threshold for underrepresentation
            include_intersectional: Whether to analyze intersections

        Returns:
            List of RepresentationBiasResult objects
        """
        key = (underrepresentation_threshold, include_intersectional)
        if key not in self._representation_results:
            self._representation_results[key] = detect_representation_bias(
                self.df,
                self.protected_attributes,
                benchmarks=self.benchmarks,
                underrepresentation_threshold=underrepresentation_threshold,
                include_intersectional=include_intersectional,
            )
        return self._representation_results[key]

    def analyze_disparities(
        self,
        *,
        significance_level: float = 0.05,
        min_group_size: int = MIN_ASSESSABLE_GROUP_SIZE,
        include_intersectional: bool = True,
    ) -> List[StatisticalDisparityResult]:
        """
        Analyze statistical disparities across protected groups.

        Performs hypothesis tests to identify significant differences
        in outcomes and features between groups.

        Args:
            significance_level: Alpha level for tests
            min_group_size: Minimum group size for analysis
            include_intersectional: Whether to include intersectional analysis

        Returns:
            List of StatisticalDisparityResult objects
        """
        key = (significance_level, min_group_size, include_intersectional)
        if key not in self._disparity_results:
            outcome_columns = [self.outcome_column] if self.outcome_column else None

            self._disparity_results[key] = analyze_statistical_disparities(
                self.df,
                self.protected_attributes,
                outcome_columns=outcome_columns,
                feature_columns=self.feature_columns,
                significance_level=significance_level,
                min_group_size=min_group_size,
                include_intersectional=include_intersectional,
            )
        return self._disparity_results[key]

    def identify_proxies(
        self,
        *,
        correlation_threshold: float = 0.3,
        include_mutual_information: bool = True,
    ) -> List[ProxyVariableResult]:
        """
        Identify proxy variables for protected attributes.

        Detects features that may enable indirect discrimination
        by correlating with protected attributes.

        Args:
            correlation_threshold: Minimum correlation to flag
            include_mutual_information: Whether to compute MI

        Returns:
            List of ProxyVariableResult objects
        """
        key = (correlation_threshold, include_mutual_information)
        if key not in self._proxy_results:
            self._proxy_results[key] = identify_proxy_variables(
                self.df,
                self.protected_attributes,
                feature_columns=self.feature_columns,
                correlation_threshold=correlation_threshold,
                include_mutual_information=include_mutual_information,
            )
        return self._proxy_results[key]

    def _calculate_overall_risk(
        self,
        historical: List[HistoricalPatternResult],
        representation: List[RepresentationBiasResult],
        disparities: List[StatisticalDisparityResult],
        proxies: List[ProxyVariableResult],
    ) -> float:
        """Calculate aggregate risk score from all findings.

        The return value is a float in 0-1 and stays one: it is read by
        ``explainer``, by the SVG adapter and by anything holding an existing
        ``BiasAuditReport``, so it is not widened to Optional here.

        What it CANNOT express is the difference between "every module ran and
        found nothing" and "no module ran": both arrive as four empty lists and
        leave as 0.0 (see the ``if not risk_components`` branch below). That
        distinction is not derivable from the findings at all, so it is
        recorded where it is known instead, by ``full_audit`` in
        ``BiasAuditReport.modules_run`` / ``execution_coverage()``. A caller
        reading this score for a verdict MUST read that record with it; a 0.0
        under any coverage other than "complete" is an absence of measurement,
        not a measurement of no risk.
        """
        risk_components = []

        if historical:
            critical = sum(1 for h in historical if h.risk_level == HistoricalRiskLevel.CRITICAL)
            high = sum(1 for h in historical if h.risk_level == HistoricalRiskLevel.HIGH)
            hist_risk = min(1.0, (critical * 0.3 + high * 0.15))
            risk_components.append(hist_risk)

        # Only results that reached a severity VERDICT contribute a component.
        # RepresentationSeverity.INSUFFICIENT_DATA is that module's own word for
        # "I could not assess this" and it is neither CRITICAL nor HIGH, so it
        # used to add rep_risk = 0.0 to risk_components, a measured risk of
        # zero from a result that says nothing was measured. That single append
        # DISARMED the `if not risk_components` guard below, which is the one
        # thing standing between an unassessable audit and a 0.0 nobody warns
        # about. Measured on a 500-row frame with an all-NULL gender column:
        # one INSUFFICIENT_DATA result in, risk_components == [0.0], guard
        # skipped, overall_risk_score 0.0, no warning at all.
        graded_representation = [
            r for r in representation if r.severity != RepresentationSeverity.INSUFFICIENT_DATA
        ]
        if graded_representation:
            critical = sum(
                1 for r in graded_representation if r.severity == RepresentationSeverity.CRITICAL
            )
            high = sum(
                1 for r in graded_representation if r.severity == RepresentationSeverity.HIGH
            )
            rep_risk = min(1.0, (critical * 0.25 + high * 0.12))
            risk_components.append(rep_risk)

        if disparities:
            large_effects = sum(
                1
                for d in disparities
                if d.effect_interpretation
                in [EffectSizeInterpretation.LARGE, EffectSizeInterpretation.MEDIUM]
            )
            disp_risk = min(1.0, large_effects * 0.1)
            risk_components.append(disp_risk)

        if proxies:
            critical = sum(1 for p in proxies if p.risk_level == ProxyRiskLevel.CRITICAL)
            high = sum(1 for p in proxies if p.risk_level == ProxyRiskLevel.HIGH)
            proxy_risk = min(1.0, (critical * 0.25 + high * 0.1))
            risk_components.append(proxy_risk)

        if not risk_components:
            # Ambiguous by construction, and deliberately left so: read it
            # together with BiasAuditReport.execution_coverage(), which is the
            # only thing that says whether anything looked.
            #
            # That prior decision stands, because the return type is a float
            # every consumer already indexes and banding. What does NOT stand is
            # doing it silently: a reader who never calls execution_coverage()
            # sees a measured risk of 0.0. The warning is the part that was
            # missing, not the value.
            warnings.warn(
                "BiasDetector: no risk component could be computed, so the overall "
                "risk score of 0.0 is NOT a measurement of low risk. Call "
                "BiasAuditReport.execution_coverage() to see whether anything was "
                "actually examined.",
                UserWarning,
                stacklevel=2,
            )
            return 0.0

        score = min(1.0, sum(risk_components) / len(risk_components) * 1.5)

        # CRITICAL findings must not be averaged away by clean axes: a
        # dataset with e.g. 3 critical issues previously scored 0.225 and
        # rendered a green MINIMAL badge. Floor the score into the band the
        # criticals demand (badge bands: 0.25 LOW / 0.50 MEDIUM / 0.75 HIGH).
        n_critical = (
            sum(1 for h in historical if h.risk_level == HistoricalRiskLevel.CRITICAL)
            + sum(1 for r in representation if r.severity == RepresentationSeverity.CRITICAL)
            + sum(1 for p in proxies if p.risk_level == ProxyRiskLevel.CRITICAL)
        )
        if n_critical >= 3:
            score = max(score, 0.75)  # at least HIGH
        elif n_critical >= 1:
            score = max(score, 0.50)  # at least MEDIUM

        return min(1.0, score)

    def _identify_critical_issues(
        self,
        historical: List[HistoricalPatternResult],
        representation: List[RepresentationBiasResult],
        disparities: List[StatisticalDisparityResult],
        proxies: List[ProxyVariableResult],
    ) -> List[Dict[str, Any]]:
        """Identify critical issues requiring immediate attention."""
        critical = []

        for h in historical:
            if h.risk_level == HistoricalRiskLevel.CRITICAL:
                critical.append(
                    {
                        "type": "Historical Pattern",
                        "feature": h.feature,
                        "description": f"Feature '{h.feature}' encodes historical discrimination "
                        f"pattern: {h.pattern_type}",
                        "affected_groups": h.affected_groups,
                        "recommendations": h.recommendations[:2],
                    }
                )

        for r in representation:
            if r.severity == RepresentationSeverity.CRITICAL:
                critical.append(
                    {
                        "type": "Representation Bias",
                        "attribute": r.attribute,
                        "description": f"Severe underrepresentation in '{r.attribute}': "
                        f"{[g['group'] for g in r.underrepresented_groups[:3]]}",
                        "recommendations": r.recommendations[:2],
                    }
                )

        for d in disparities:
            if (
                d.significance == SignificanceLevel.HIGHLY_SIGNIFICANT
                and d.effect_interpretation == EffectSizeInterpretation.LARGE
            ):
                critical.append(
                    {
                        "type": "Statistical Disparity",
                        "feature": d.feature,
                        "attribute": d.protected_attribute,
                        "description": f"Large, significant disparity in '{d.feature}' "
                        f"across '{d.protected_attribute}' "
                        f"(effect size: {d.effect_size:.2f})",
                        "recommendations": d.recommendations[:2],
                    }
                )

        for p in proxies:
            if p.risk_level == ProxyRiskLevel.CRITICAL:
                critical.append(
                    {
                        "type": "Proxy Variable",
                        "feature": p.feature,
                        "protected_attribute": p.protected_attribute,
                        "description": f"Feature '{p.feature}' is a strong proxy for "
                        f"'{p.protected_attribute}' (correlation: {p.correlation:.2f})",
                        "affected_groups": p.affected_groups,
                        "recommendations": p.recommendations[:2],
                    }
                )

        return critical

    def _generate_recommendations(
        self,
        historical: List[HistoricalPatternResult],
        representation: List[RepresentationBiasResult],
        disparities: List[StatisticalDisparityResult],
        proxies: List[ProxyVariableResult],
        *,
        modules_run: Optional[List[str]] = None,
        attribute_observations: Optional[Dict[str, int]] = None,
        attribute_assessed: Optional[Dict[str, bool]] = None,
    ) -> List[str]:
        """Generate prioritized recommendations from all findings.

        *modules_run* is the same record ``full_audit`` puts on the report, and
        it is here for the same reason: the fallback recommendation below is a
        verdict ("no critical bias issues detected"), and with no finding to
        base it on it was issued just as readily by a run where no module
        executed. Keyword with a default so an existing caller keeps working;
        None means unrecorded and keeps the original wording.

        *attribute_observations* closes the other half of the same hole, and it
        had to be closed separately because ``modules_run`` was FULL: over a
        column that was entirely NULL all four modules were invoked, so the
        wording chosen below was the all-clear. It is the same three states.

        *attribute_assessed* is what the wording is actually chosen on now.
        Observations were the proxy for it and the proxy was wrong in the one
        direction that matters: two rows are observations, no module could
        reach a verdict from them, and the all-clear went out anyway. None
        means unrecorded and falls back to the counts.
        """
        recommendations = []

        critical_features = set()
        for h in historical:
            if h.risk_level in [HistoricalRiskLevel.CRITICAL, HistoricalRiskLevel.HIGH]:
                critical_features.add(h.feature)
        for p in proxies:
            if p.risk_level in [ProxyRiskLevel.CRITICAL, ProxyRiskLevel.HIGH]:
                critical_features.add(p.feature)

        if critical_features:
            recommendations.append(
                f"CRITICAL: Review and consider removing high-risk features: "
                f"{', '.join(list(critical_features)[:5])}"
            )

        severe_underrep = []
        for r in representation:
            if r.severity in [RepresentationSeverity.CRITICAL, RepresentationSeverity.HIGH]:
                severe_underrep.extend([g["group"] for g in r.underrepresented_groups[:2]])
        if severe_underrep:
            recommendations.append(
                f"Address underrepresentation of: {', '.join(severe_underrep[:5])}. "
                "Consider targeted data collection or resampling."
            )

        sig_disparities = [
            d
            for d in disparities
            if d.significance
            in [SignificanceLevel.HIGHLY_SIGNIFICANT, SignificanceLevel.SIGNIFICANT]
            and d.effect_interpretation
            in [EffectSizeInterpretation.LARGE, EffectSizeInterpretation.MEDIUM]
        ]
        if sig_disparities:
            recommendations.append(
                f"Investigate significant disparities in: "
                f"{', '.join([d.feature for d in sig_disparities[:3]])}. "
                "Consider fairness constraints or post-processing."
            )

        if historical:
            recommendations.append(
                "Document historical context for flagged features and implement "
                "monitoring for disparate impact."
            )

        if proxies:
            recommendations.append(
                "Evaluate necessity of proxy variables and consider fairness-aware "
                "feature selection."
            )

        if not recommendations:
            ran = (
                None if modules_run is None else [m for m in AUDIT_MODULES if m in set(modules_run)]
            )
            if attribute_assessed is not None:
                blank = sorted(str(a) for a, ok in attribute_assessed.items() if not ok)
                nothing_to_assess = bool(attribute_assessed) and not any(
                    attribute_assessed.values()
                )
            elif attribute_observations is not None:
                blank = sorted(str(a) for a, n in attribute_observations.items() if not n)
                nothing_to_assess = not any(attribute_observations.values())
            else:
                blank = None
                nothing_to_assess = False
            if ran is not None and not ran:
                recommendations.append(
                    "No audit module executed, so nothing was examined and nothing here "
                    "clears the data."
                )
            elif nothing_to_assess:
                named = ", ".join(blank or []) or "none were requested"
                recommendations.append(
                    f"Every audit module executed and none of them assessed anything: no "
                    f"requested protected attribute carried enough observations for any "
                    f"module to reach a verdict ({named}). Nothing here clears the data. "
                    f"Supply a populated protected attribute with enough rows per group, "
                    f"or check the column name, and re-run."
                )
            elif ran is not None and len(ran) < len(AUDIT_MODULES):
                recommendations.append(
                    "No critical issue was detected by the modules that ran; the others did "
                    "not run and cleared nothing."
                )
            elif blank:
                recommendations.append(
                    f"No critical bias issues detected for the protected attributes that "
                    f"were assessed. Nothing was assessed for {', '.join(blank)}, which "
                    f"carried too few observations for any module to reach a verdict, and "
                    f"this audit clears them of nothing."
                )
            else:
                recommendations.append(
                    "No critical bias issues detected. Continue monitoring and perform "
                    "periodic audits."
                )

        return recommendations

    def get_explanation(self, report=None):
        """Generate educational explanations for the bias audit results.

        Parameters
        ----------
        report : BiasAuditReport, optional
            Pre-computed audit report. If *None*, :meth:`full_audit` is run.

        Returns
        -------
        ExplanationReport
            Container with per-finding explanations, severity, and
            actionable recommendations. When the audit was in no position to
            measure anything, the explanation says so where a reader meets it
            (see :func:`_qualify_explanation`) instead of grading the 0.0.
        """
        from vfairness.explainer import FairnessExplainer

        if report is None:
            report = self.full_audit()
        return _qualify_explanation(FairnessExplainer.explain(report), report)

    def clear_cache(self) -> None:
        """Clear cached results to force re-computation.

        Empties every keyed cache, so the next call to any detection method
        recomputes whatever arguments it is given. Returns nothing.
        """
        self._historical_results = {}
        self._representation_results = {}
        self._disparity_results = {}
        self._proxy_results = {}

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset."""
        return {
            "n_rows": len(self.df),
            "n_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "protected_attributes": self.protected_attributes,
            "protected_attribute_distributions": {
                attr: self.df[attr].value_counts(normalize=True).to_dict()
                for attr in self.protected_attributes
                if attr in self.df.columns
            },
            "missing_rates": {col: self.df[col].isna().mean() for col in self.df.columns},
        }
