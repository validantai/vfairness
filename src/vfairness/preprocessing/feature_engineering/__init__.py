"""
Feature Engineering Module for vfairness.

This module provides comprehensive tools for fairness-aware feature engineering,
including proxy variable detection, feature transformation, and correlation analysis.

Main Components:
    1. **FeatureEngineeringAnalyzer**: Unified interface for analysis and transformation
    2. **Transformers**: Fairness-aware feature transformation methods
    3. **Correlation Analysis**: Tools for analyzing feature-protected attribute relationships
    4. **Visualization**: Plotting tools for feature engineering insights

Quick Start:
    >>> from vfairness.preprocessing.feature_engineering import FeatureEngineeringAnalyzer
    >>>
    >>> # Create analyzer
    >>> analyzer = FeatureEngineeringAnalyzer(
    ...     df=my_data,
    ...     protected_attributes=['gender', 'race'],
    ...     target_column='outcome'
    ... )
    >>>
    >>> # Run comprehensive analysis
    >>> report = analyzer.full_analysis()
    >>> print(report.summary)
    >>>
    >>> # Transform features to reduce correlation
    >>> X_fair = analyzer.transform(method='correlation_reduction')

Key Concepts:
    - **Proxy Variables**: Features that indirectly encode protected attributes
    - **Correlation Reduction**: Techniques to reduce discriminatory signals
    - **Fair Representation**: Creating feature representations that promote fairness
    - **Intersectional Fairness**: Addressing interactions between protected attributes

References:
    - Zemel et al. (2013): Learning Fair Representations
    - Feldman et al. (2015): Certifying and Removing Disparate Impact
    - Barocas & Selbst (2016): Big Data's Disparate Impact
    - Datta et al. (2017): Proxy Discrimination in Data-Driven Systems
"""

from .analyzer import (
    FeatureAnalysisReport,
    FeatureEngineeringAnalyzer,
)
from .correlation import (
    CHAIN_MIN_SAMPLES,
    CORRELATION_THRESHOLD_CRITICAL,
    CORRELATION_THRESHOLD_HIGH,
    CORRELATION_THRESHOLD_LOW,
    CORRELATION_THRESHOLD_MEDIUM,
    KNOWN_PROXY_PATTERNS,
    MAX_CARDINALITY_RATIO,
    MI_BINS,
    MIN_SAMPLE_SIZE,
    CorrelationResult,
    CorrelationType,
    FeatureCorrelationMatrix,
    ProxyRiskLevel,
    ProxyType,
    ProxyVariableResult,
    analyze_intersectional_correlations,
    compute_feature_correlations,
    compute_pearson_correlation_matrix,
    find_proxy_chains,
    identify_proxy_variables,
)
from .data_balancing import (
    ADASYNResampler,
    CounterfactualAugmenter,
    InversePropensityWeighter,
    PropensityScoreWeighter,
    SMOTEResampler,
    SyntheticResampler,
    TomekResampler,
    counterfactual_augment,
    propensity_weights,
)
from .significance import paired_metric_significance
from .transformers import (
    BaseFeatureTransformer,
    CorrelationReducer,
    DisparateImpactRemover,
    FairnessObjective,
    FairRepresentationTransformer,
    FeatureImportanceResult,
    FeatureSuppressor,
    IntersectionalTransformer,
    LabelMassager,
    Resampler,
    ResidualTransformer,
    ReweightingTransformer,
    TransformationMethod,
    TransformationResult,
)
from .visualization import (
    create_analysis_dashboard,
    plot_correlation_bars,
    plot_correlation_heatmap,
    plot_feature_correlation_matrix,
    plot_feature_transformation_effect,
    plot_intersectional_heatmap,
    plot_proxy_chains,
    plot_proxy_risk_chart,
    plot_risk_distribution,
    plot_transformation_comparison,
)

__all__ = [
    "FeatureEngineeringAnalyzer",
    "FeatureAnalysisReport",
    "BaseFeatureTransformer",
    "CorrelationReducer",
    "FeatureSuppressor",
    "ResidualTransformer",
    "IntersectionalTransformer",
    "ReweightingTransformer",
    "DisparateImpactRemover",
    "LabelMassager",
    "Resampler",
    "FairRepresentationTransformer",
    # Preprocessing catalog (synthetic resampling, propensity weighting, counterfactual augmentation)
    "SyntheticResampler",
    "SMOTEResampler",
    "ADASYNResampler",
    "TomekResampler",
    "PropensityScoreWeighter",
    "InversePropensityWeighter",
    "CounterfactualAugmenter",
    "propensity_weights",
    "counterfactual_augment",
    "TransformationResult",
    "FeatureImportanceResult",
    "FeatureCorrelationMatrix",
    "ProxyVariableResult",
    "CorrelationResult",
    "TransformationMethod",
    "FairnessObjective",
    "ProxyRiskLevel",
    "ProxyType",
    "CorrelationType",
    "compute_pearson_correlation_matrix",
    "compute_feature_correlations",
    "identify_proxy_variables",
    "find_proxy_chains",
    "analyze_intersectional_correlations",
    "paired_metric_significance",
    "MIN_SAMPLE_SIZE",
    "CHAIN_MIN_SAMPLES",
    "MAX_CARDINALITY_RATIO",
    "MI_BINS",
    "CORRELATION_THRESHOLD_CRITICAL",
    "CORRELATION_THRESHOLD_HIGH",
    "CORRELATION_THRESHOLD_MEDIUM",
    "CORRELATION_THRESHOLD_LOW",
    "KNOWN_PROXY_PATTERNS",
    "plot_correlation_heatmap",
    "plot_correlation_bars",
    "plot_feature_correlation_matrix",
    "plot_proxy_risk_chart",
    "plot_risk_distribution",
    "plot_proxy_chains",
    "plot_transformation_comparison",
    "plot_feature_transformation_effect",
    "plot_intersectional_heatmap",
    "create_analysis_dashboard",
]
