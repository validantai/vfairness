"""
vfairness.preprocessing - Data & Preprocessing Module

This module contains tools for pre-training data analysis and transformation:

Submodules:
    - feature_engineering: Fairness-aware feature transformations
    - bias_detection: Pre-training bias auditing and detection

Feature Engineering:
    - Proxy variable detection and analysis
    - Feature correlation analysis with protected attributes
    - Fairness-aware feature transformations
    - Intersectional feature engineering
    - Correlation reduction and residualization

Bias Detection:
    - Historical pattern detection
    - Representation bias analysis
    - Statistical disparity analysis
    - Proxy variable identification

Example:
    >>> from vfairness.preprocessing import FeatureEngineeringAnalyzer, BiasDetector
    >>>
    >>> # Analyze features for proxy correlations
    >>> fe_analyzer = FeatureEngineeringAnalyzer(
    ...     df=data,
    ...     protected_attributes=['race', 'gender'],
    ...     target_column='approved'
    ... )
    >>> fe_report = fe_analyzer.full_analysis()
    >>>
    >>> # Detect pre-existing biases in data
    >>> detector = BiasDetector(
    ...     df,
    ...     protected_attributes=['gender', 'race'],
    ...     outcome_column='approved',
    ... )
    >>> bias_report = detector.full_audit()
"""

from .bias_detection import (
    HISTORICAL_RISK_PATTERNS,
    BiasAuditReport,
    BiasDetector,
    HistoricalPatternResult,
    ProxyVariableResult,
    RepresentationBiasResult,
    StatisticalDisparityResult,
    analyze_statistical_disparities,
    compare_to_benchmark,
    compute_proxy_correlations,
    detect_historical_patterns,
    detect_representation_bias,
    identify_proxy_variables,
    run_disparity_tests,
)
from .bias_detection import (
    compute_effect_sizes as compute_disparity_effect_sizes,
)
from .feature_engineering import (
    KNOWN_PROXY_PATTERNS,
    BaseFeatureTransformer,
    CorrelationReducer,
    FeatureAnalysisReport,
    FeatureCorrelationMatrix,
    FeatureEngineeringAnalyzer,
    FeatureSuppressor,
    IntersectionalTransformer,
    ProxyRiskLevel,
    ProxyType,
    ResidualTransformer,
    ReweightingTransformer,
    TransformationResult,
    analyze_intersectional_correlations,
    compute_feature_correlations,
    find_proxy_chains,
)
from .feature_engineering import (
    ProxyVariableResult as FeatureProxyResult,
)
from .feature_engineering import (
    identify_proxy_variables as identify_feature_proxies,
)

# Canonical protected-attribute preparation (DOB->age bands, identifier
# exclusion, quantile binning, binned-sibling preference). Single source of
# truth -- used by quality_report, the Pulse pipeline and the task consumer.
from .protected_binning import (
    PreparedProtected,
    prepare_protected_attributes,
)

__all__ = [
    "prepare_protected_attributes",
    "PreparedProtected",
    "FeatureEngineeringAnalyzer",
    "FeatureAnalysisReport",
    "BaseFeatureTransformer",
    "CorrelationReducer",
    "FeatureSuppressor",
    "ResidualTransformer",
    "IntersectionalTransformer",
    "ReweightingTransformer",
    "TransformationResult",
    "compute_feature_correlations",
    "identify_feature_proxies",
    "find_proxy_chains",
    "analyze_intersectional_correlations",
    "FeatureCorrelationMatrix",
    "FeatureProxyResult",
    "ProxyRiskLevel",
    "ProxyType",
    "KNOWN_PROXY_PATTERNS",
    "BiasDetector",
    "BiasAuditReport",
    "detect_historical_patterns",
    "HistoricalPatternResult",
    "HISTORICAL_RISK_PATTERNS",
    "detect_representation_bias",
    "RepresentationBiasResult",
    "compare_to_benchmark",
    "analyze_statistical_disparities",
    "StatisticalDisparityResult",
    "run_disparity_tests",
    "compute_disparity_effect_sizes",
    "identify_proxy_variables",
    "ProxyVariableResult",
    "compute_proxy_correlations",
]
