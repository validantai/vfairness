"""
Bias Detection Module for vfairness.

This module provides comprehensive bias detection capabilities for raw data analysis,
implementing four core building blocks:

A. Historical Pattern Detection
   - Detect bias patterns rooted in historical discrimination
   - Cross-reference features with known discriminatory patterns (redlining, etc.)
   - Identify temporal bias trends

B. Representation Bias Detection
   - Detect underrepresentation/overrepresentation of demographic groups
   - Compare against population benchmarks (census data, etc.)
   - Quantify representation ratios and statistical significance

C. Statistical Disparity Analysis
   - Rigorous hypothesis testing for group differences
   - T-tests, chi-squared tests, proportion tests
   - Effect size estimation and confidence intervals
   - Intersectional subgroup analysis

D. Proxy Variable Identification
   - Detect features correlated with protected attributes
   - Multiple correlation measures (Pearson, Cramér's V, mutual information)
   - Risk scoring for potential proxy discrimination

The module supports:
- Automated detection without manual bias specification
- Intersectional analysis across multiple protected attributes
- Configurable thresholds for different risk contexts
- Comprehensive reporting with statistical confidence measures

Example:
    >>> from vfairness.preprocessing.bias_detection import BiasDetector
    >>>
    >>> detector = BiasDetector(df, protected_attributes=['gender', 'race'])
    >>> report = detector.full_audit()
    >>>
    >>> # Or use individual components
    >>> from vfairness.preprocessing.bias_detection import (
    ...     detect_historical_patterns,
    ...     detect_representation_bias,
    ...     analyze_statistical_disparities,
    ...     identify_proxy_variables
    ... )

References:
    - Barocas, S., Hardt, M., & Narayanan, A. (2023). Fairness and ML. MIT Press.
    - Buolamwini, J., & Gebru, T. (2018). Gender Shades. FAT* Conference.
    - Mehrabi, N., et al. (2021). Survey on Bias and Fairness in ML. ACM Computing Surveys.
"""

from .detector import (
    BiasAuditReport,
    BiasDetector,
)
from .geographic_data import (
    SVI_THEMES,
    GeographicRiskAssessment,
    HOLCAreaInfo,
    HOLCGrade,
    assess_geographic_feature_risk,
    fetch_holc_data,
    get_available_holc_cities,
    get_svi_data_url,
    lookup_holc_grade_by_zip,
    print_data_quality_notes,
)
from .historical import (
    HISTORICAL_RISK_PATTERNS,
    HistoricalPatternResult,
    attribute_historical_pattern,
    detect_historical_patterns,
    domain_historical_context,
)
from .proxy import (
    MultivariateProxyResult,
    ProxyVariableResult,
    compute_proxy_correlations,
    identify_proxy_variables,
    multivariate_proxy_leakage,
)
from .representation import (
    RepresentationBiasResult,
    compare_to_benchmark,
    detect_representation_bias,
)
from .statistical import (
    StatisticalDisparityResult,
    analyze_statistical_disparities,
    compute_effect_sizes,
    detect_specification_bias,
    detect_temporal_drift,
    run_disparity_tests,
)

__all__ = [
    "BiasDetector",
    "BiasAuditReport",
    "detect_historical_patterns",
    "HistoricalPatternResult",
    "HISTORICAL_RISK_PATTERNS",
    "domain_historical_context",
    "attribute_historical_pattern",
    "detect_representation_bias",
    "RepresentationBiasResult",
    "compare_to_benchmark",
    "analyze_statistical_disparities",
    "StatisticalDisparityResult",
    "detect_temporal_drift",
    "detect_specification_bias",
    "run_disparity_tests",
    "compute_effect_sizes",
    "identify_proxy_variables",
    "ProxyVariableResult",
    "compute_proxy_correlations",
    "multivariate_proxy_leakage",
    "MultivariateProxyResult",
    "HOLCGrade",
    "HOLCAreaInfo",
    "GeographicRiskAssessment",
    "get_available_holc_cities",
    "fetch_holc_data",
    "lookup_holc_grade_by_zip",
    "assess_geographic_feature_risk",
    "SVI_THEMES",
    "get_svi_data_url",
    "print_data_quality_notes",
]
