"""
vfairness.in_processing.calibrators - Group-Specific Training Calibrators

This module provides calibration methods that can be integrated into the
training process to ensure group-specific probability calibration.

Calibrators Implemented:
    1. TemperatureScalingCalibrator: Group-specific temperature scaling
    2. PlattScalingCalibrator: Group-specific Platt scaling
    3. BetaCalibrator: Group-specific beta calibration
    4. FocalCalibrator: Group-specific focal calibration
    5. TrainableGroupCalibrator: Unified trainable calibrator

Quick Start:
    >>> from vfairness.in_processing.calibrators import (
    ...     TrainableGroupCalibrator,
    ...     create_group_calibrator,
    ... )
    >>>
    >>> # Create calibrator
    >>> calibrator = create_group_calibrator(n_groups=2, method='temperature')
    >>>
    >>> # Use in training loop
    >>> logits = model(x)
    >>> calibrated_logits = calibrator(logits, group_ids)
    >>> loss = F.cross_entropy(calibrated_logits, y)
    >>> cal_loss = calibrator.calibration_loss(calibrated_logits, y, group_ids)
    >>> total_loss = loss + cal_loss

References:
    - Guo et al. (2017): On Calibration of Modern Neural Networks
    - Platt (2000): Probabilities for Support Vectors
    - Kull et al. (2019): Beyond Temperature Scaling
    - Pleiss et al. (2017): On Fairness and Calibration
"""

from .group_calibrators import (
    TORCH_AVAILABLE,
    BetaCalibrator,
    # Training helper
    CalibrationAwareTrainer,
    # Enums
    CalibrationMethodType,
    # Data containers
    CalibrationState,
    FocalCalibrator,
    PlattScalingCalibrator,
    # Individual calibrators
    TemperatureScalingCalibrator,
    # Unified calibrator
    TrainableGroupCalibrator,
    # Base utilities
    check_torch_available,
    # Factory
    create_group_calibrator,
)

__all__ = [
    # Utilities
    "check_torch_available",
    "TORCH_AVAILABLE",
    # Enums
    "CalibrationMethodType",
    # Data containers
    "CalibrationState",
    # Calibrators
    "TemperatureScalingCalibrator",
    "PlattScalingCalibrator",
    "BetaCalibrator",
    "FocalCalibrator",
    "TrainableGroupCalibrator",
    # Training
    "CalibrationAwareTrainer",
    # Factory
    "create_group_calibrator",
]
