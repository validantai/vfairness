"""
Feature Engineering Analyzer - Unified Interface for Fairness-Aware Feature Engineering.

This module provides the FeatureEngineeringAnalyzer class, which serves as the
central interface for analyzing and transforming features to promote fairness.

Key Capabilities:
    1. Comprehensive proxy variable detection
    2. Feature correlation analysis with protected attributes
    3. Fairness-aware feature transformation
    4. Feature importance analysis for fairness
    5. Actionable recommendations for feature engineering

The analyzer integrates all feature engineering components into a cohesive
workflow for both analysis and transformation.

Example:
    >>> from vfairness.preprocessing.feature_engineering import FeatureEngineeringAnalyzer
    >>> analyzer = FeatureEngineeringAnalyzer(
    ...     df=my_data,
    ...     protected_attributes=['gender', 'race'],
    ...     target_column='outcome'
    ... )
    >>> report = analyzer.full_analysis()
    >>> X_fair = analyzer.transform(method='correlation_reduction')

References:
    - Zemel et al. (2013): Learning Fair Representations
    - Feldman et al. (2015): Certifying and Removing Disparate Impact
    - Barocas & Selbst (2016): Big Data's Disparate Impact
"""

import json
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

from vfairness._not_assessed import NOT_ASSESSED
from vfairness._triage import is_measured, measured_values, unmeasurable_reason

from .correlation import (
    FeatureCorrelationMatrix,
    ProxyRiskLevel,
    ProxyScreenResult,
    ProxyVariableResult,
    _encode_for_correlation,
    analyze_intersectional_correlations,
    compute_feature_correlations,
    compute_pearson_correlation_matrix,
    find_proxy_chains,
    identify_proxy_variables,
)
from .transformers import (
    BaseFeatureTransformer,
    CorrelationReducer,
    FeatureSuppressor,
    IntersectionalTransformer,
    ResidualTransformer,
    ReweightingTransformer,
    TransformationResult,
)

# Supported transformation method identifiers (see FeatureEngineeringAnalyzer.transform).
TransformMethod = Literal[
    "correlation_reduction",
    "feature_suppression",
    "residualization",
    "reweighting",
    "intersectional",
]


class ProxyScreenFindings(ProxyScreenResult):
    """What the proxy screen GRADED, plus both kinds of thing it did not.

    A subclass of :class:`ProxyScreenResult` so every existing consumer of
    ``analyze_proxies`` keeps working unchanged: it is still a ``list`` of
    :class:`ProxyVariableResult`, still iterable, still ``len``-able, still
    passes ``isinstance(x, list)``.

    BGL-S3 (2026-09-17). ``full_analysis`` learned to split a screened pair
    into "measured and gradeable" and "ran but produced no measurement"
    (see :meth:`FeatureEngineeringAnalyzer._partition_gradeable`).
    ``analyze_proxies``, the other public entry onto the same screen, did not,
    so the fix was one method wide. Measured on a constant protected attribute
    over 500 rows, ``analyze_proxies`` returned two ProxyVariableResult rows
    with ``correlation=nan``, ``pvalue=nan``, ``risk_level=NEGLIGIBLE``,
    ``evidence['pvalue_status']='computed'`` and ``complete=True``: a full
    all-clear, with a provenance field asserting a p-value that does not exist,
    over a frame where nothing whatsoever was measurable. NEGLIGIBLE is the
    safest band and its action is "Minimal risk. No action needed.", so the
    fabrication pointed in the reassuring direction.

    Attributes:
        screens_not_run: pairs the screen never looked at, inherited.
        not_screened_reason: why NOTHING was screened, when that is the case
            (a request holding no feature/attribute pair at all), inherited.
        ungraded_screens: pairs the screen DID look at whose headline
            correlation was NaN or infinite, so no risk band applies. They are
            kept rather than dropped, because their evidence dict still names
            which individual measures were computed and a deleted row is a
            finding a reader can never get back.
        complete: True only when both of those lists are empty AND at least one
            pair was requested.
    """

    def __init__(
        self,
        results: Optional[List[ProxyVariableResult]] = None,
        *,
        screens_not_run: Optional[List[Dict[str, Any]]] = None,
        ungraded_screens: Optional[List[ProxyVariableResult]] = None,
        not_screened_reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            results,
            screens_not_run=screens_not_run,
            not_screened_reason=not_screened_reason,
        )
        self.ungraded_screens: List[ProxyVariableResult] = list(ungraded_screens or [])

    @property
    def complete(self) -> bool:
        """True only when every requested pair was screened AND graded.

        BGL7 F16 2026-09-29: this override rebuilt the predicate field by field
        and so dropped the term the base class had just gained, which is how a
        correct measurement reaches no reader. Measured through this entry with
        ``feature_columns=[]``: 0 results, ``complete`` True, while
        ``identify_proxy_variables`` underneath had already recorded that it
        screened zero pairs.
        """
        return (
            not self.screens_not_run
            and not self.ungraded_screens
            and self.not_screened_reason is None
        )


@dataclass
class FeatureAnalysisReport:
    """
    Comprehensive report from feature engineering analysis.

    Attributes:
        n_features: Number of features analyzed
        n_protected_attributes: Number of protected attributes
        n_samples: Number of samples
        proxy_variables: List of identified proxy variables
        correlation_matrix: Feature-attribute correlation matrix
        high_risk_features: Features flagged as high risk
        recommendations: Prioritized recommendations
        intersectional_analysis: Intersectional correlation results
        proxy_chains: Indirect proxy chains
        summary: Human-readable summary
        screens_not_run: Proxy screens that never ran, one entry per
            (protected attribute) or (protected attribute, feature) pair that
            was skipped. An EMPTY ``proxy_variables`` list means "no proxy was
            found" only when this list is also empty; otherwise it means
            "not looked at".
        ungraded_proxy_screens: Screens that ran but whose headline correlation
            was not a measurement (NaN or infinite), so no risk level could be
            assigned. These are held OUT of ``proxy_variables`` because the
            grader returns NEGLIGIBLE for them, which is the safest band and is
            indistinguishable from a measured all-clear.

    BGL-S2 (2026-09-16). Both lists are new. Before them, two different
    could-not-check states were reported as clean findings at the public entry:

    1. An attribute with fewer than ``min_sample_size`` usable rows was dropped
       by a bare ``continue`` in ``identify_proxy_variables``. Measured on
       n=50 rows whose corr(income, race) was 0.9998: ``proxy_variables=[]``,
       ``risk_summary`` all zeros, zero warnings, and the summary a reader sees
       ended "No significant proxy variables detected." The identical generator
       at n=500 reports that same feature CRITICAL.
    2. A constant protected attribute makes every correlation NaN. Each feature
       still came back as a ProxyVariableResult graded NEGLIGIBLE and counted in
       ``risk_summary['negligible']``, carrying ``pvalue_status='computed'``
       over ``pvalue=nan``: a status actively asserting a measurement that did
       not happen.
    """

    n_features: int
    n_protected_attributes: int
    n_samples: int
    proxy_variables: List[ProxyVariableResult]
    correlation_matrix: Optional[FeatureCorrelationMatrix]
    high_risk_features: List[str]
    recommendations: List[Dict[str, Any]]
    intersectional_analysis: Optional[Dict[str, Any]] = None
    proxy_chains: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    screens_not_run: List[Dict[str, Any]] = field(default_factory=list)
    ungraded_proxy_screens: List[ProxyVariableResult] = field(default_factory=list)
    #: Why NOTHING was screened, when that is the case: a request holding no
    #: feature/attribute pair at all. None when at least one pair was requested.
    #: BGL7 F16, 2026-09-29: without it `proxy_screen_complete` read True over a
    #: screen that looked at zero pairs, on the surface full_analysis publishes.
    not_screened_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "n_features": self.n_features,
            "n_protected_attributes": self.n_protected_attributes,
            "n_samples": self.n_samples,
            "proxy_variables": [p.to_dict() for p in self.proxy_variables],
            "correlation_matrix": self.correlation_matrix.to_dict()
            if self.correlation_matrix
            else None,
            "high_risk_features": self.high_risk_features,
            "recommendations": self.recommendations,
            "intersectional_analysis": self.intersectional_analysis,
            "proxy_chains": self.proxy_chains,
            "summary": self.summary,
            "screens_not_run": self.screens_not_run,
            "ungraded_proxy_screens": [p.to_dict() for p in self.ungraded_proxy_screens],
            "not_screened_reason": self.not_screened_reason,
            "proxy_screen_complete": self.proxy_screen_complete,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def proxy_screen_complete(self) -> bool:
        """True only when every feature/attribute pair was actually screened.

        Read this BEFORE reading ``proxy_variables``. False means the empty or
        short list below it is a statement about what was looked at, not about
        what is there.

        BGL7 F16 2026-09-29: over a screen asked for ZERO feature/attribute
        pairs both lists are empty, so this read True and ``risk_summary``
        counted 0 not_assessed. Complete coverage of nothing is the reassuring
        half of a two-state answer, and this property is what a reader is told
        to read FIRST.
        """
        return (
            not self.screens_not_run
            and not self.ungraded_proxy_screens
            and self.not_screened_reason is None
        )

    @property
    def risk_summary(self) -> Dict[str, int]:
        """Get count of features by risk level.

        The ``not_assessed`` key is the third state and is ALWAYS present, so a
        caller reading the dict cannot miss it by looking only at the bands it
        expected. It counts screens that did not run plus screens that ran and
        produced no measurable correlation; neither is graded into a band.
        """
        counts = {level.value: 0 for level in ProxyRiskLevel}
        for proxy in self.proxy_variables:
            counts[proxy.risk_level.value] += 1
        counts[NOT_ASSESSED] = len(self.screens_not_run) + len(self.ungraded_proxy_screens)
        return counts


#: Severity for an explanation whose aggregate nothing graded. A THIRD state,
#: never collapsed into 'info', which is what a genuinely graded and genuinely
#: clean analysis receives. Spelled the way
#: ``evaluation.vfairness_metrics.explainer.MetricExplanation`` documents it.
_COULD_NOT_CHECK_SEV = "could_not_check"

#: Report-level floor for an explanation that could not check.
#: ``ExplanationReport.severity`` has no could-not-check member, so an
#: unmeasured analysis takes the same floor band the bias-audit path uses. Only
#: ever raised FROM 'info', never lowered: a real finding keeps its own severity.
_UNKNOWN_REPORT_SEV = "medium"


def _qualify_feature_explanation(explanation, report):
    """Carry the analysis's coverage into the explanation a reader is handed.

    THE DEFECT, measured 2026-09-25 on a 2-row single-group frame. The report was
    honest: ``screens_not_run`` listed three screens with the reason "only 2
    non-null row(s), and the proxy screen needs min_sample_size=100". The
    explanation built from it read "Analysed 1 features across 1 protected
    attribute(s). 0 proxy variable(s) and 0 high-risk feature(s)" at severity
    'info'. Nothing ran, and the reader was handed a clean result.

    This is the same shape, and the same fix, as
    ``preprocessing.bias_detection.detector._qualify_explanation``, which closed
    it for the bias audit under BGL g021. The discriminator here is the report's
    own record rather than a new rule: a screen that did not run is listed in
    ``screens_not_run``, and a proxy screen whose result could not be graded is
    listed in ``ungraded_proxy_screens``. An analysis with both lists empty
    reached a verdict on everything it was asked about, and its 'info' is a
    measurement that must survive untouched, which the control test asserts.

    NOTHING IS REMOVED. Findings, recommendations and the per-item values are
    left alone and the severity is only ever RAISED, so a real high-risk feature
    under partial coverage keeps its own grade. Only the verdict on the aggregate
    is withdrawn, because that is the one nothing measured.
    """
    not_run = list(getattr(report, "screens_not_run", None) or [])
    ungraded = list(getattr(report, "ungraded_proxy_screens", None) or [])
    if not not_run and not ungraded:
        return explanation

    parts = []
    if not_run:
        parts.append(f"{len(not_run)} screen(s) did not run")
    if ungraded:
        parts.append(f"{len(ungraded)} proxy screen(s) returned no gradeable result")
    statement = (
        "COULD NOT CHECK: " + " and ".join(parts) + ", so a count of 0 proxy variables "
        "or 0 high-risk features from this analysis is an absence rather than a finding "
        "that none exist. Read `screens_not_run` for what was skipped and why."
    )

    summary = str(getattr(explanation, "summary", "") or "")
    explanation.summary = f"{statement} {summary}".strip()

    for item in getattr(explanation, "explanations", None) or []:
        if getattr(item, "severity", None) in (None, "info"):
            item.severity = _COULD_NOT_CHECK_SEV
            item.evaluation = statement

    if getattr(explanation, "severity", None) == "info":
        explanation.severity = _UNKNOWN_REPORT_SEV
    return explanation


class FeatureEngineeringAnalyzer:
    """
    Unified interface for fairness-aware feature engineering.

    This analyzer provides a comprehensive toolkit for:
    - Identifying proxy variables and discriminatory features
    - Analyzing correlations between features and protected attributes
    - Transforming features to reduce discrimination
    - Generating actionable recommendations

    The analyzer supports multiple analysis modes and transformation methods,
    allowing users to choose the appropriate approach for their use case.

    Attributes:
        df: DataFrame containing features and protected attributes
        protected_attributes: List of protected attribute column names
        target_column: Optional target/outcome column name
        feature_columns: List of feature column names to analyze
        analysis_report: Results from the most recent analysis

    Example:
        >>> analyzer = FeatureEngineeringAnalyzer(
        ...     df=loan_data,
        ...     protected_attributes=['race', 'gender'],
        ...     target_column='approved'
        ... )
        >>>
        >>> # Run comprehensive analysis
        >>> report = analyzer.full_analysis()
        >>> print(f"Found {len(report.proxy_variables)} potential proxies")
        >>>
        >>> # Transform features to reduce correlation
        >>> X_fair = analyzer.transform(method='correlation_reduction')

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: feature_engineering_analysis. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protected_attributes: List[str],
        *,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        correlation_threshold: float = 0.3,
        min_sample_size: int = 100,
    ):
        """
        Initialize the Feature Engineering Analyzer.

        Args:
            df: DataFrame containing all data
            protected_attributes: List of protected attribute column names
            target_column: Optional target/outcome column for supervised analysis
            feature_columns: Specific columns to analyze (auto-detect if None)
            correlation_threshold: Threshold for flagging correlations
            min_sample_size: Minimum samples required for analysis

        Raises:
            ValueError: If protected attributes are not in DataFrame
        """
        self.df = df.copy()
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.min_sample_size = min_sample_size

        missing = [a for a in protected_attributes if a not in df.columns]
        if missing:
            raise ValueError(f"Protected attributes not found: {missing}")

        exclude = set(protected_attributes)
        if target_column:
            exclude.add(target_column)

        # BGL stage 4 (2026-09-27): the auto-detection keeps only numeric columns and
        # used to record the drop NOWHERE, so the report's two coverage lists
        # (screens_not_run and ungraded_proxy_screens, which are the only
        # discriminator _qualify_feature_explanation has) were empty because nothing
        # was ever a candidate rather than because everything was screened. Measured
        # on 300 rows where 'zip_code' is a PERFECT proxy for gender (F to '48201', M
        # to '90210') and every feature is categorical: feature_columns [],
        # screens_not_run [], proxy_screen_complete True, risk_summary all zeros
        # including not_assessed 0, zero warnings, and the explanation read "Analysed
        # 0 features across 1 protected attribute(s). 0 proxy variable(s) and 0
        # high-risk feature(s) identified." at severity 'info', with 'zip_code' named
        # on no channel. The excluded columns are kept here and reported as screens
        # that did not run, so every consumer of that record (the qualifier, the
        # not_assessed band, the UserWarning) sees them with no new rule.
        self._features_excluded_at_construction: List[Tuple[str, str]] = []
        if feature_columns is None:
            self.feature_columns = [
                c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            self._features_excluded_at_construction = [
                (c, str(df[c].dtype))
                for c in df.columns
                if c not in exclude and c not in self.feature_columns
            ]
        else:
            self.feature_columns = [c for c in feature_columns if c not in exclude]

        self.analysis_report: Optional[FeatureAnalysisReport] = None
        self._transformers: Dict[str, BaseFeatureTransformer] = {}

    def full_analysis(
        self,
        *,
        include_proxy_chains: bool = True,
        include_intersectional: bool = True,
        include_mutual_information: bool = True,
    ) -> FeatureAnalysisReport:
        """
        Perform comprehensive feature engineering analysis.

        Combines all analysis methods to provide a complete picture of
        fairness concerns in the feature set.

        Args:
            include_proxy_chains: Whether to find indirect proxy chains
            include_intersectional: Whether to analyze intersectional effects
            include_mutual_information: Whether to compute mutual information

        Returns:
            FeatureAnalysisReport with comprehensive analysis results

        Example:
            >>> report = analyzer.full_analysis()
            >>> print(report.summary)
            >>> for rec in report.recommendations:
            ...     print(f"- {rec['priority']}: {rec['action']}")
        """
        proxy_vars = identify_proxy_variables(
            self.df,
            self.protected_attributes,
            feature_columns=self.feature_columns,
            correlation_threshold=self.correlation_threshold,
            include_mutual_information=include_mutual_information,
            min_sample_size=self.min_sample_size,
        )

        corr_matrix = compute_feature_correlations(
            self.df,
            self.protected_attributes,
            feature_columns=self.feature_columns,
        )

        proxy_chains: List[Dict[str, Any]] = []
        chain_screens_not_run: List[Dict[str, Any]] = []
        if include_proxy_chains:
            for attr in self.protected_attributes:
                chains = find_proxy_chains(
                    self.df,
                    attr,
                    correlation_threshold=self.correlation_threshold,
                )
                proxy_chains.extend(chains)
                # `find_proxy_chains` may return a result that carries its own
                # coverage alongside the chains. `extend` keeps only the chains,
                # so the refusal it computed would be dropped at THIS boundary
                # and reach no reader: a correct measurement nobody can see.
                # Read defensively (getattr) so this holds whether the callee
                # returns a bare list or the richer result.
                chain_screens_not_run.extend(self._describe_chain_coverage(attr, chains))

        intersectional = None
        if include_intersectional and len(self.protected_attributes) > 1:
            intersectional = analyze_intersectional_correlations(
                self.df,
                self.protected_attributes,
                feature_columns=self.feature_columns,
            )

        # BGL-S2: split the three states BEFORE anything grades, counts or
        # summarises. `proxy_vars` reaching here is only "what was measured and
        # gradeable"; the other two channels are the screens that never ran and
        # the screens whose correlation is not a number.
        # `identify_proxy_variables` returns a ProxyScreenResult, a list subclass
        # that CARRIES the pairs it never screened. Rebinding `proxy_vars` to a
        # plain list here threw that away: the four gates inside the screen (the
        # attribute is absent, too few non-null rows, will not encode,
        # high-cardinality) each end in a bare `continue`, and a pair stopped by
        # one of them left no trace at all. The high-cardinality gate is the
        # dangerous one, because an identifier-shaped column is exactly what a
        # strong proxy often looks like.
        screened = proxy_vars
        graded, ungraded = self._partition_gradeable(list(screened))
        screens_not_run = self._find_screens_not_run(graded + ungraded)
        # The screen records its own skips and this analyser derives the same
        # ones from what came back, so merging them naively reported every skip
        # TWICE and doubled `not_assessed`. De-duplicate on the pair the entry
        # is about; the richer entry (the one the screen wrote, which carries the
        # counts) wins.
        _seen = {(e.get("protected_attribute"), e.get("feature")) for e in screens_not_run}
        for entry in getattr(screened, "screens_not_run", []):
            pair = (entry.get("protected_attribute"), entry.get("feature"))
            if pair in _seen:
                for i, existing in enumerate(screens_not_run):
                    if (existing.get("protected_attribute"), existing.get("feature")) == pair:
                        if len(entry) > len(existing):
                            screens_not_run[i] = entry
                        break
                continue
            _seen.add(pair)
            screens_not_run.append(entry)
        screens_not_run.extend(chain_screens_not_run)

        high_risk = [
            p.feature
            for p in graded
            if p.risk_level in [ProxyRiskLevel.CRITICAL, ProxyRiskLevel.HIGH]
        ]

        recommendations = self._generate_recommendations(
            graded,
            corr_matrix,
            proxy_chains,
            screens_not_run=screens_not_run,
            ungraded=ungraded,
        )

        summary = self._generate_summary(
            graded,
            high_risk,
            corr_matrix,
            screens_not_run=screens_not_run,
            ungraded=ungraded,
        )

        self._warn_screens_not_run(screens_not_run, ungraded)

        self.analysis_report = FeatureAnalysisReport(
            n_features=len(self.feature_columns),
            n_protected_attributes=len(self.protected_attributes),
            n_samples=len(self.df),
            proxy_variables=graded,
            correlation_matrix=corr_matrix,
            high_risk_features=high_risk,
            recommendations=recommendations,
            intersectional_analysis=intersectional,
            proxy_chains=proxy_chains,
            summary=summary,
            screens_not_run=screens_not_run,
            ungraded_proxy_screens=ungraded,
            not_screened_reason=getattr(screened, "not_screened_reason", None),
        )

        return self.analysis_report

    @staticmethod
    def _describe_chain_coverage(attr: str, chains: Any) -> List[Dict[str, Any]]:
        """Carry forward any could-not-check the chain search reported.

        BGL-S2, found while fixing the proxy screen. ``find_proxy_chains`` can
        return a result object that says which pairs it could not compute and
        whether the protected attribute was in the frame at all, and
        ``proxy_chains.extend(chains)`` above keeps ONLY the chains. An empty
        ``report.proxy_chains`` therefore meant "looked and found nothing" and
        "could not look" alike, which is the same defect as the proxy screen's,
        one layer along.

        Read with ``getattr`` rather than an import so this works against a
        plain ``list`` return too, and never raises if the callee's shape moves.
        """
        entries: List[Dict[str, Any]] = []
        if getattr(chains, "protected_attribute_present", True) is False:
            entries.append(
                {
                    "protected_attribute": attr,
                    "feature": None,
                    "reason": "proxy-chain search: attribute is not a column of the frame",
                    "n_usable": 0,
                    "min_sample_size": None,
                }
            )
            return entries
        for pair in getattr(chains, "pairs_not_computed", ()) or ():
            try:
                col_a, col_b, reason = pair
            except (TypeError, ValueError):
                continue
            entries.append(
                {
                    "protected_attribute": attr,
                    "feature": f"{col_a} -> {col_b}",
                    "reason": f"proxy-chain search: {reason}",
                    "n_usable": None,
                    "min_sample_size": None,
                }
            )
        return entries

    def _partition_gradeable(
        self, proxy_vars: List[ProxyVariableResult]
    ) -> Tuple[List[ProxyVariableResult], List[ProxyVariableResult]]:
        """Split screened pairs into graded measurements and ungradeable ones.

        A ProxyVariableResult whose headline ``correlation`` is NaN or infinite
        was not measured, but it still arrives carrying a risk level, because
        every ``>=`` comparison against NaN is False and the grader falls
        through to its lowest band. NEGLIGIBLE is the SAFEST band and its action
        is "Minimal risk. No action needed.", so the fabrication points in the
        reassuring direction.

        The ungradeable rows are returned rather than dropped: their evidence
        dict can still name which individual measures were computed, and a
        deleted row is a finding a reader can never get back. What is corrected
        on the way out is the provenance stamped on them, which asserted
        ``pvalue_status='computed'`` for a NaN p-value.
        """
        graded: List[ProxyVariableResult] = []
        ungraded: List[ProxyVariableResult] = []

        for proxy in proxy_vars:
            corr_ok = is_measured(proxy.correlation)
            evidence = dict(proxy.evidence)
            # A NaN p-value is not a computed p-value. The `is not None` test in
            # correlation.py is True for NaN, so the status read "computed".
            if not is_measured(proxy.pvalue):
                evidence["pvalue_status"] = "could_not_compute"
            if corr_ok:
                graded.append(replace(proxy, evidence=evidence))
                continue
            evidence["risk_level_status"] = NOT_ASSESSED
            evidence["correlation_status"] = unmeasurable_reason(proxy.correlation)
            ungraded.append(replace(proxy, evidence=evidence))

        return graded, ungraded

    def _find_screens_not_run(self, screened: List[ProxyVariableResult]) -> List[Dict[str, Any]]:
        """List every feature/attribute pair the proxy screen never looked at.

        Mirrors the two sample-size gates inside ``identify_proxy_variables``:
        an attribute with fewer than ``min_sample_size`` non-null rows is
        skipped entirely, and so is any pair whose row-wise intersection falls
        below the same floor. Both were bare ``continue`` / ``return None``
        statements, which is why an unscreened attribute was indistinguishable
        from a clean one.

        Pairs the screen DID reach are excluded by identity of the
        (attribute, feature) key, so a pair dropped for a legitimate reason
        (below the correlation threshold, not significant) is not miscounted as
        unscreened: that pair was looked at and found nothing.
        """
        reached = {(p.protected_attribute, p.feature) for p in screened}
        not_run: List[Dict[str, Any]] = []

        for attr in self.protected_attributes:
            if attr not in self.df.columns:
                not_run.append(
                    {
                        "protected_attribute": attr,
                        "feature": None,
                        "reason": "column not present in the DataFrame",
                        "n_usable": 0,
                        "min_sample_size": self.min_sample_size,
                    }
                )
                continue

            attr_series = self.df[attr].dropna()
            if len(attr_series) < self.min_sample_size:
                not_run.append(
                    {
                        "protected_attribute": attr,
                        "feature": None,
                        "reason": (
                            f"only {len(attr_series)} non-null row(s), and the proxy "
                            f"screen needs min_sample_size={self.min_sample_size}"
                        ),
                        "n_usable": int(len(attr_series)),
                        "min_sample_size": self.min_sample_size,
                    }
                )
                continue

            if _encode_for_correlation(attr_series) is None:
                not_run.append(
                    {
                        "protected_attribute": attr,
                        "feature": None,
                        "reason": "attribute could not be encoded for correlation",
                        "n_usable": int(len(attr_series)),
                        "min_sample_size": self.min_sample_size,
                    }
                )
                continue

            # A column dropped by the numeric-only auto-detection in __init__ was
            # never a candidate, so the screen could not have reached it and no
            # other channel mentions it. It is the same state as a pair the screen
            # skipped: not looked at, therefore not assessed.
            for feature, dtype in self._features_excluded_at_construction:
                if feature == attr or (attr, feature) in reached:
                    continue
                not_run.append(
                    {
                        "protected_attribute": attr,
                        "feature": feature,
                        "reason": (
                            f"never a candidate: the column is dtype {dtype} and the "
                            "auto-detected feature set keeps only numeric columns, so "
                            "the proxy screen never looked at it. Pass "
                            "feature_columns=[...] to screen it."
                        ),
                        "n_usable": None,
                        "min_sample_size": self.min_sample_size,
                    }
                )

            for feature in self.feature_columns:
                if feature == attr or feature not in self.df.columns:
                    continue
                if (attr, feature) in reached:
                    continue
                n_pair = int(len(self.df[[feature, attr]].dropna()))
                if n_pair < self.min_sample_size:
                    not_run.append(
                        {
                            "protected_attribute": attr,
                            "feature": feature,
                            "reason": (
                                f"only {n_pair} row(s) have both values, and the proxy "
                                f"screen needs min_sample_size={self.min_sample_size}"
                            ),
                            "n_usable": n_pair,
                            "min_sample_size": self.min_sample_size,
                        }
                    )

        return not_run

    def _warn_screens_not_run(
        self,
        screens_not_run: List[Dict[str, Any]],
        ungraded: List[ProxyVariableResult],
        *,
        site: str = "full_analysis",
    ) -> None:
        """Warn, with counts, that the proxy screen is incomplete.

        ``site`` names the public entry the reader called, because both
        ``full_analysis`` and ``analyze_proxies`` reach the same screen and a
        warning naming the wrong one sends an operator to the wrong method.
        """
        if not screens_not_run and not ungraded:
            return
        parts = []
        if screens_not_run:
            parts.append(f"{len(screens_not_run)} screen(s) did not run")
        if ungraded:
            parts.append(f"{len(ungraded)} screen(s) produced no measurable correlation")
        warnings.warn(
            f"FeatureEngineeringAnalyzer.{site}: "
            + " and ".join(parts)
            + ". Those pairs are reported as not_assessed, NOT as negligible risk. "
            "An empty or short result here means 'not looked at', "
            "not 'no proxies present'.",
            UserWarning,
            stacklevel=3,
        )

    def analyze_proxies(
        self,
        *,
        correlation_threshold: Optional[float] = None,
        include_known_patterns: bool = True,
    ) -> ProxyScreenFindings:
        """
        Analyze features for proxy discrimination.

        Args:
            correlation_threshold: Override default threshold. ``0.0`` is a
                legitimate override meaning "report every pair", and is now
                honoured; it used to fall back to the constructor default
                because the test was ``or``, not ``is not None``.
            include_known_patterns: Check against known proxy patterns

        Returns:
            :class:`ProxyScreenFindings`, a list of the pairs that were
            measured AND gradeable. Read ``complete`` before reading the list:
            when it is False the list is a statement about what was looked at,
            not about what is there, and ``screens_not_run`` and
            ``ungraded_screens`` say which pairs are missing and why.

        Warns:
            UserWarning: naming the counts, whenever any pair went unscreened or
            produced no measurable correlation.

        Example:
            >>> proxies = analyzer.analyze_proxies(correlation_threshold=0.2)
            >>> if not proxies.complete:
            ...     print("incomplete screen:", proxies.screens_not_run)
            >>> for p in proxies:
            ...     print(f"{p.feature}: {p.risk_level.value}")
        """
        threshold = (
            self.correlation_threshold if correlation_threshold is None else correlation_threshold
        )

        screened = identify_proxy_variables(
            self.df,
            self.protected_attributes,
            feature_columns=self.feature_columns,
            correlation_threshold=threshold,
            include_known_patterns=include_known_patterns,
            min_sample_size=self.min_sample_size,
        )

        # BGL-S3: the SAME partition `full_analysis` performs. Without it this
        # entry graded an unmeasurable correlation NEGLIGIBLE, which is the
        # safest band, and stamped pvalue_status='computed' over a NaN p-value.
        graded, ungraded = self._partition_gradeable(list(screened))
        screens_not_run = list(getattr(screened, "screens_not_run", []))
        self._warn_screens_not_run(screens_not_run, ungraded, site="analyze_proxies")

        return ProxyScreenFindings(
            graded,
            screens_not_run=screens_not_run,
            ungraded_screens=ungraded,
            not_screened_reason=getattr(screened, "not_screened_reason", None),
        )

    def get_correlation_matrix(self) -> FeatureCorrelationMatrix:
        """
        Get correlation matrix between features and protected attributes.

        Returns:
            FeatureCorrelationMatrix with correlations and p-values

        Example:
            >>> matrix = analyzer.get_correlation_matrix()
            >>> print(matrix.correlations)
        """
        return compute_feature_correlations(
            self.df,
            self.protected_attributes,
            feature_columns=self.feature_columns,
        )

    def get_feature_correlation_matrix(
        self,
        *,
        return_pvalues: bool = False,
        min_periods: int = 10,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get Pearson correlation matrix between numeric features.

        This computes feature-to-feature correlations (not feature-to-protected-attribute).
        Useful for identifying multicollinearity and redundant features.

        Args:
            return_pvalues: If True, also return p-value matrix
            min_periods: Minimum observations per pair

        Returns:
            DataFrame correlation matrix. If return_pvalues=True, tuple of
            (correlations, pvalues).

        Example:
            >>> # Get feature correlations
            >>> corr = analyzer.get_feature_correlation_matrix()
            >>> print(corr)

            >>> # With significance testing
            >>> corr, pvals = analyzer.get_feature_correlation_matrix(return_pvalues=True)
            >>> highly_correlated = corr[corr.abs() > 0.7]
        """
        return compute_pearson_correlation_matrix(
            self.df,
            feature_columns=self.feature_columns,
            min_periods=min_periods,
            return_pvalues=return_pvalues,
        )

    def transform(
        self,
        method: TransformMethod = "correlation_reduction",
        *,
        return_transformer: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, BaseFeatureTransformer]]:
        """
        Transform features using the specified method.

        Available methods:
        - 'correlation_reduction': Reduce correlation with protected attributes
        - 'feature_suppression': Remove or mask discriminatory features
        - 'residualization': Compute residuals controlling for protected attributes
        - 'reweighting': Compute sample weights for fairness
        - 'intersectional': Handle intersectional fairness concerns

        Args:
            method: Transformation method to use
            return_transformer: If True, also return the fitted transformer
            **kwargs: Additional arguments passed to the transformer

        Returns:
            Transformed DataFrame (and optionally the transformer)

        Example:
            >>> X_fair = analyzer.transform(method='correlation_reduction')
            >>> X_fair, transformer = analyzer.transform(
            ...     method='feature_suppression',
            ...     return_transformer=True,
            ...     strategy='remove'
            ... )
        """
        transformer: BaseFeatureTransformer
        if method == "correlation_reduction":
            transformer = CorrelationReducer(
                protected_attributes=self.protected_attributes, **kwargs
            )
        elif method == "feature_suppression":
            transformer = FeatureSuppressor(
                protected_attributes=self.protected_attributes,
                correlation_threshold=self.correlation_threshold,
                **kwargs,
            )
        elif method == "residualization":
            transformer = ResidualTransformer(
                protected_attributes=self.protected_attributes, **kwargs
            )
        elif method == "reweighting":
            transformer = ReweightingTransformer(
                protected_attributes=self.protected_attributes, **kwargs
            )
        elif method == "intersectional":
            transformer = IntersectionalTransformer(
                protected_attributes=self.protected_attributes, **kwargs
            )
        else:
            raise ValueError(f"Unknown transformation method: {method}")

        # Fit and transform using only numeric feature columns + protected attributes
        transform_cols = list(self.feature_columns) + [
            a for a in self.protected_attributes if a not in self.feature_columns
        ]
        df_transform = self.df[transform_cols]
        X_fair = transformer.fit_transform(df_transform)

        self._transformers[method] = transformer

        if return_transformer:
            return X_fair, transformer
        return X_fair

    def get_feature_recommendations(
        self,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get prioritized recommendations for feature handling.

        Returns recommendations sorted by priority, with specific actions
        for each feature based on its fairness risk.

        Args:
            top_n: Number of top recommendations to return

        Returns:
            List of recommendation dictionaries

        Example:
            >>> recs = analyzer.get_feature_recommendations(top_n=5)
            >>> for rec in recs:
            ...     print(f"[{rec['priority']}] {rec['feature']}: {rec['action']}")
        """
        if self.analysis_report is None:
            self.full_analysis()

        # full_analysis() always assigns a non-None report.
        assert self.analysis_report is not None
        return self.analysis_report.recommendations[:top_n]

    def get_transformation_report(
        self,
        method: str = "correlation_reduction",
    ) -> Optional[TransformationResult]:
        """
        Get the transformation report for a previously applied method.

        Args:
            method: Name of the transformation method

        Returns:
            TransformationResult if method was applied, None otherwise
        """
        transformer = self._transformers.get(method)
        if transformer and transformer.is_fitted:
            return transformer.fit_result
        return None

    def compare_transformations(
        self,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple transformation methods.

        Applies multiple methods and compares their effect on correlations
        with protected attributes.

        Args:
            methods: List of methods to compare (defaults to all)

        Returns:
            DataFrame with one row per method. Besides the per-attribute
            ``corr_<attr>`` columns and ``avg_correlation`` it carries three
            columns a reader needs BEFORE ranking anything:

            - ``status``: ``measured``, ``partial``, ``not_measured`` or
              ``failed: <error>``.
            - ``pairs_measured`` / ``pairs_not_measured``: how many
              feature/attribute pairs the row's number rests on, and how many
              could not be correlated at all.

            An unmeasurable correlation is ``nan``, never 0.0.

        Warns:
            UserWarning: when a method raises, and when an attribute's average
            correlation could not be computed on a frame.

        Example:
            >>> comparison = analyzer.compare_transformations()
            >>> print(comparison)
            >>> usable = comparison[comparison["status"] != "not_measured"]
        """
        if methods is None:
            methods = [
                "correlation_reduction",
                "feature_suppression",
                "residualization",
            ]

        results = []

        orig_corrs, orig_ok, orig_bad = self._avg_correlation_with_coverage(
            self.df, label="original:"
        )
        results.append(self._comparison_row("original", orig_corrs, orig_ok, orig_bad))

        for method in methods:
            try:
                # transform() validates the method at runtime and raises
                # ValueError for unknown values (caught below); return_transformer
                # is not passed, so the result is always a DataFrame.
                X_fair = cast(
                    pd.DataFrame,
                    self.transform(method=cast(TransformMethod, method)),
                )
                # Add protected attributes back for correlation computation
                df_fair = X_fair.copy()
                for attr in self.protected_attributes:
                    df_fair[attr] = self.df[attr].values

                corrs, ok, bad = self._avg_correlation_with_coverage(df_fair, label=f"{method}:")
                results.append(self._comparison_row(method, corrs, ok, bad))
            except Exception as e:
                # BGL-S3: a method that RAISED used to vanish from the table
                # entirely, so a reader comparing three methods saw two rows and
                # could not tell a method that blew up from one never requested.
                # It gets a row, with nan correlations and the error in `status`.
                warnings.warn(
                    f"FeatureEngineeringAnalyzer.compare_transformations: "
                    f"'{method}' raised {type(e).__name__}: {e}. Its row reports "
                    f"status='failed' with nan correlations; it is NOT an absence "
                    f"of correlation.",
                    UserWarning,
                    stacklevel=2,
                )
                results.append(
                    {
                        "method": method,
                        "avg_correlation": float("nan"),
                        "status": f"failed: {type(e).__name__}: {e}",
                    }
                )

        return pd.DataFrame(results)

    def _comparison_row(
        self,
        method: str,
        corrs: Dict[str, float],
        n_measured: int,
        n_not_measured: int,
    ) -> Dict[str, Any]:
        """One row of the ``compare_transformations`` table, coverage included.

        ``avg_correlation`` averages the attributes that WERE measured and is
        nan when none was, so it never reports 0.0 (perfect independence) for a
        frame on which nothing could be correlated.
        """
        usable = measured_values(list(corrs.values()))
        if usable:
            status = "measured" if n_not_measured == 0 else "partial"
            average = float(np.mean(usable))
        else:
            status = "not_measured"
            average = float("nan")
        return {
            "method": method,
            **{f"corr_{attr}": corr for attr, corr in corrs.items()},
            "avg_correlation": average,
            "pairs_measured": n_measured,
            "pairs_not_measured": n_not_measured,
            "status": status,
        }

    def _compute_avg_correlation(
        self,
        df: pd.DataFrame,
        *,
        label: str = "",
    ) -> Dict[str, float]:
        """Average absolute correlation with each protected attribute.

        A value is NaN when this frame supported no correlation at all for that
        attribute. See :meth:`_avg_correlation_with_coverage` for why that is
        not 0.0.
        """
        correlations, _measured, _not_measured = self._avg_correlation_with_coverage(
            df, label=label
        )
        return correlations

    def _avg_correlation_with_coverage(
        self,
        df: pd.DataFrame,
        *,
        label: str = "",
    ) -> Tuple[Dict[str, float], int, int]:
        """Average |correlation| per attribute, plus how many pairs it rests on.

        Returns ``(per_attribute_mean, n_pairs_measured, n_pairs_not_measured)``.
        A mean is NaN when NOT ONE feature/attribute pair could be correlated on
        this frame.

        BGL-S3 (2026-09-17). This used to end ``float(np.mean(corrs)) if corrs
        else 0.0``, and 0.0 on this scale is the BEST attainable answer: zero
        association with the protected attribute, the exact result a successful
        mitigation is trying to produce. Measured before the fix, on a frame
        whose two feature columns were entirely NaN::

            FeatureEngineeringAnalyzer(df, ["race"]).compare_transformations()
                              method  corr_race  avg_correlation
            0               original        0.0              0.0
            1  correlation_reduction        0.0              0.0
            2    feature_suppression        0.0              0.0
            3        residualization        0.0              0.0

        Every row a perfect score, over a frame carrying no measurable number
        anywhere, and the ``original`` row carried no warning of its own. Five
        rows and zero rows produced the same table. Sorting that table to pick
        the best method ranks the fabrication FIRST, because 0.0 is the minimum.
        NaN sorts last under ``sort_values`` and is skipped by ``idxmin``.

        The mean is taken over the pairs that WERE measurable rather than
        poisoned by the ones that were not: one constant column among ten used
        to turn the whole attribute NaN, which throws away nine real
        measurements. The two counts returned beside it say how much of the
        frame the number rests on, so partial coverage is visible without
        reading this source.
        """
        correlations: Dict[str, float] = {}
        n_measured = 0
        n_not_measured = 0
        feature_cols = [c for c in df.columns if c not in self.protected_attributes]
        where = f"{label} " if label else ""

        for attr in self.protected_attributes:
            if attr not in df.columns:
                # Absent from the frame is not "uncorrelated with the frame".
                correlations[attr] = float("nan")
                n_not_measured += len(feature_cols)
                warnings.warn(
                    f"FeatureEngineeringAnalyzer.compare_transformations: {where}"
                    f"protected attribute '{attr}' is not a column of this frame, so "
                    f"its average correlation is nan (could not check), NOT 0.0.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            pair_values: List[float] = []
            attr_encoded = self._encode_column(df[attr])

            for col in feature_cols:
                try:
                    feature_vals = df[col].values.astype(float)
                    mask = ~np.isnan(feature_vals) & ~np.isnan(attr_encoded)
                    if mask.sum() >= 10:
                        corr = np.corrcoef(feature_vals[mask], attr_encoded[mask])[0, 1]
                        # np.corrcoef returns NaN for a constant column; that is
                        # an unmeasurable pair, not a zero association.
                        pair_values.append(abs(float(corr)))
                    else:
                        pair_values.append(float("nan"))
                except (ValueError, TypeError):
                    pair_values.append(float("nan"))

            usable = measured_values(pair_values)
            n_measured += len(usable)
            n_not_measured += len(pair_values) - len(usable)

            if usable:
                correlations[attr] = float(np.mean(usable))
                continue

            correlations[attr] = float("nan")
            warnings.warn(
                f"FeatureEngineeringAnalyzer.compare_transformations: {where}"
                f"not one of {len(pair_values)} feature(s) could be correlated with "
                f"'{attr}' (every pair had fewer than 10 paired non-missing rows, a "
                f"constant column, or a non-numeric value). Reporting nan "
                f"(could not check), NOT 0.0, which on this scale would read as "
                f"perfect independence from the protected attribute.",
                UserWarning,
                stacklevel=3,
            )

        return correlations, n_measured, n_not_measured

    def _encode_column(self, series: pd.Series) -> np.ndarray:
        """Encode a column for correlation computation."""
        # is_numeric_dtype is robust to the pandas-3 string dtype (a str column is
        # NOT object and its dtype.name is not "category"), which otherwise fell to
        # the float cast and crashed on a string protected attribute.
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Categorical(series).codes.astype(float)
        return series.to_numpy().astype(float)

    def _generate_recommendations(
        self,
        proxy_vars: List[ProxyVariableResult],
        corr_matrix: FeatureCorrelationMatrix,
        proxy_chains: List[Dict[str, Any]],
        *,
        screens_not_run: Optional[List[Dict[str, Any]]] = None,
        ungraded: Optional[List[ProxyVariableResult]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations.

        BGL-S3: the recommendation list is what ``get_feature_recommendations``
        hands a reader and what the explainer prints as ACTION ITEMS, and it was
        built from the GRADED proxies alone. On a frame the screen never ran on
        it therefore came back ``[]``, which reads as "nothing to do" and is
        indistinguishable from a genuinely clean frame. The could-not-check now
        gets a row of its own, sorted above CRITICAL so it cannot be missed.
        """
        screens_not_run = screens_not_run or []
        ungraded = ungraded or []
        recommendations: List[Dict[str, Any]] = []

        n_unchecked = len(screens_not_run) + len(ungraded)
        if n_unchecked:
            reasons = [str(e.get("reason", "")) for e in screens_not_run[:3]]
            reasons += [
                f"{p.feature} vs {p.protected_attribute}: correlation is "
                f"{p.evidence.get('correlation_status', 'not a measurement')}"
                for p in ungraded[:3]
            ]
            recommendations.append(
                {
                    "priority": NOT_ASSESSED.upper(),
                    "feature": None,
                    "protected_attribute": None,
                    "correlation": float("nan"),
                    "action": (
                        f"COULD NOT CHECK: {n_unchecked} proxy screen(s) produced no "
                        f"measurement. Nothing below is an all-clear for the pairs "
                        f"they cover. Raise the sample size or supply a varying "
                        f"protected attribute, then re-run."
                    ),
                    "rationale": "; ".join(r for r in reasons if r),
                }
            )

        for proxy in proxy_vars:
            priority = {
                ProxyRiskLevel.CRITICAL: "CRITICAL",
                ProxyRiskLevel.HIGH: "HIGH",
                ProxyRiskLevel.MEDIUM: "MEDIUM",
                ProxyRiskLevel.LOW: "LOW",
                ProxyRiskLevel.NEGLIGIBLE: "INFO",
            }[proxy.risk_level]

            action = self._get_action_for_risk(proxy)

            recommendations.append(
                {
                    "priority": priority,
                    "feature": proxy.feature,
                    "protected_attribute": proxy.protected_attribute,
                    "correlation": proxy.correlation,
                    "action": action,
                    "rationale": proxy.recommendations[0] if proxy.recommendations else "",
                }
            )

        for chain in proxy_chains[:5]:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "feature": chain["chain"][0],
                    "protected_attribute": chain["chain"][-1],
                    "correlation": chain["indirect_correlation"],
                    "action": f"Indirect proxy via {chain['chain'][1]}. Consider removing or monitoring.",
                    "rationale": f"Chain: {' -> '.join(chain['chain'])}",
                }
            )

        priority_order = {
            NOT_ASSESSED.upper(): -1,
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
            "INFO": 4,
        }

        def _magnitude(rec: Dict[str, Any]) -> float:
            # A NaN correlation must not enter the sort key: NaN compares False
            # both ways, so its position becomes whatever the sort happens to do.
            value = rec.get("correlation")
            if not is_measured(value):
                return 0.0
            # is_measured has already established a real, finite, non-bool
            # number; cast tells mypy that, without a bare assert (which
            # python -O strips) and without silencing a genuine Optional.
            return -abs(float(cast(float, value)))

        recommendations.sort(key=lambda x: (priority_order.get(x["priority"], 5), _magnitude(x)))

        return recommendations

    def _get_action_for_risk(self, proxy: ProxyVariableResult) -> str:
        """Get recommended action for a proxy variable."""
        if proxy.risk_level == ProxyRiskLevel.CRITICAL:
            return "REMOVE this feature or apply strict fairness constraints."
        elif proxy.risk_level == ProxyRiskLevel.HIGH:
            return "Consider removing or transforming this feature."
        elif proxy.risk_level == ProxyRiskLevel.MEDIUM:
            return "Monitor for disparate impact. Consider correlation reduction."
        elif proxy.risk_level == ProxyRiskLevel.LOW:
            return "Low risk. Continue monitoring."
        else:
            return "Minimal risk. No action needed."

    def _generate_summary(
        self,
        proxy_vars: List[ProxyVariableResult],
        high_risk: List[str],
        corr_matrix: FeatureCorrelationMatrix,
        *,
        screens_not_run: Optional[List[Dict[str, Any]]] = None,
        ungraded: Optional[List[ProxyVariableResult]] = None,
    ) -> str:
        """Generate human-readable summary.

        BGL-S2: the summary is the surface a person actually reads, so the
        could-not-check state has to appear HERE, not only on a field. The
        all-clear sentence is now conditional on the screen having been run;
        printing "No significant proxy variables detected." over an attribute
        nobody screened was the worst line in this file.
        """
        screens_not_run = screens_not_run or []
        ungraded = ungraded or []
        lines = [
            "=" * 60,
            "FEATURE ENGINEERING ANALYSIS SUMMARY",
            "=" * 60,
            "",
            f"Analyzed {len(self.feature_columns)} features against {len(self.protected_attributes)} protected attribute(s).",
            f"Sample size: {len(self.df):,}",
            "",
        ]

        risk_counts: Dict[str, int] = {}
        for proxy in proxy_vars:
            risk = proxy.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        if risk_counts:
            lines.append("RISK LEVEL BREAKDOWN:")
            for risk in ["critical", "high", "medium", "low", "negligible"]:
                if risk in risk_counts:
                    lines.append(f"  - {risk.upper()}: {risk_counts[risk]} feature(s)")
            lines.append("")

        n_unchecked = len(screens_not_run) + len(ungraded)
        if n_unchecked:
            lines.append(f"COULD NOT CHECK: {n_unchecked} proxy screen(s) produced no measurement.")
            lines.append(
                "  An empty or short proxy list below means 'not looked at', "
                "NOT 'no proxies present'."
            )
            for entry in screens_not_run[:5]:
                target = entry["protected_attribute"]
                if entry.get("feature"):
                    target = f"{entry['feature']} vs {target}"
                lines.append(f"  - NOT SCREENED {target}: {entry['reason']}")
            if len(screens_not_run) > 5:
                lines.append(f"  ... and {len(screens_not_run) - 5} more not screened")
            for proxy in ungraded[:5]:
                lines.append(
                    f"  - NOT GRADED {proxy.feature} vs {proxy.protected_attribute}: "
                    f"{proxy.correlation_type} correlation is "
                    f"{proxy.evidence.get('correlation_status', 'not a measurement')}"
                )
            if len(ungraded) > 5:
                lines.append(f"  ... and {len(ungraded) - 5} more not graded")
            lines.append("")

        if high_risk:
            lines.append("HIGH-RISK FEATURES (require attention):")
            for feature in high_risk[:5]:
                match = next((p for p in proxy_vars if p.feature == feature), None)
                if match:
                    lines.append(
                        f"  - {feature}: corr={match.correlation:.3f} with {match.protected_attribute}"
                    )
            if len(high_risk) > 5:
                lines.append(f"  ... and {len(high_risk) - 5} more")
            lines.append("")

        lines.append("KEY RECOMMENDATIONS:")
        if not proxy_vars:
            if n_unchecked:
                lines.append(
                    f"  No all-clear can be given: {n_unchecked} screen(s) did not produce "
                    "a measurement. Raise the sample size or supply a varying protected "
                    "attribute, then re-run."
                )
            else:
                lines.append("  No significant proxy variables detected.")
        else:
            critical_count = risk_counts.get("critical", 0)
            high_count = risk_counts.get("high", 0)

            if critical_count > 0:
                lines.append(
                    f"  1. Address {critical_count} CRITICAL proxy variable(s) immediately"
                )
            if high_count > 0:
                lines.append(f"  2. Review {high_count} HIGH-risk feature(s) for potential removal")
            lines.append("  3. Consider applying correlation reduction transformation")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_explanation(self, report=None):
        """Generate educational explanations for the feature analysis results.

        Parameters
        ----------
        report : FeatureAnalysisReport, optional
            Pre-computed report. If *None*, :meth:`full_analysis` is run.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        if report is None:
            report = self.full_analysis()
        # The aggregate verdict is withheld when the report's own coverage record
        # says screens did not run; see _qualify_feature_explanation.
        return _qualify_feature_explanation(FairnessExplainer.explain(report), report)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureEngineeringAnalyzer("
            f"n_features={len(self.feature_columns)}, "
            f"protected_attrs={self.protected_attributes})"
        )
