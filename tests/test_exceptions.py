"""Public exception hierarchy tests (VB-API-5)."""

import numpy as np
import pytest

import vfairness
from vfairness import (
    ConfigurationError,
    InsufficientDataError,
    InvalidDataError,
    ProtectedAttributeError,
    VfairnessError,
)
from vfairness.evaluation.vfairness_metrics._validation import validate_binary_labels

SPECIFIC = [InvalidDataError, InsufficientDataError, ProtectedAttributeError, ConfigurationError]


@pytest.mark.parametrize("exc", SPECIFIC)
def test_specific_errors_derive_from_base(exc):
    assert issubclass(exc, VfairnessError)


@pytest.mark.parametrize("exc", SPECIFIC)
def test_specific_errors_stay_backward_compatible_with_valueerror(exc):
    # Existing `except ValueError` handlers must keep catching these.
    assert issubclass(exc, ValueError)


def test_base_is_exported_and_is_exception():
    assert issubclass(VfairnessError, Exception)
    assert VfairnessError.__module__ == "vfairness.exceptions"
    for name in (
        "VfairnessError",
        "InvalidDataError",
        "InsufficientDataError",
        "ProtectedAttributeError",
        "ConfigurationError",
    ):
        assert name in vfairness.__all__


def test_validation_raises_the_hierarchy_and_is_valueerror():
    bad = np.array([0, 1, 2])  # not binary
    with pytest.raises(InvalidDataError):
        validate_binary_labels(bad)
    # Same call is still catchable as ValueError (backward compatible) and as base.
    with pytest.raises(ValueError):
        validate_binary_labels(bad)
    with pytest.raises(VfairnessError):
        validate_binary_labels(bad)
