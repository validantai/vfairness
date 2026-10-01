"""Deprecation helper tests (VB-API-2)."""

import warnings

import pytest

from vfairness._deprecation import deprecated, warn_deprecated


def test_deprecated_decorator_warns_and_delegates():
    @deprecated("use new_fn()", removed_in="0.1.0")
    def old_fn(a, b):
        return a + b

    with pytest.warns(DeprecationWarning) as record:
        result = old_fn(2, 3)
    assert result == 5
    msg = str(record[0].message)
    assert "old_fn is deprecated" in msg
    assert "0.1.0" in msg
    assert old_fn.__name__ == "old_fn"  # functools.wraps preserved
    assert hasattr(old_fn, "__deprecated__")


def test_warn_deprecated_imperative():
    with pytest.warns(DeprecationWarning, match="OldAlias is deprecated"):
        warn_deprecated("OldAlias", "use NewThing", removed_in="0.1.0")


def test_no_warning_without_removed_in_still_warns():
    @deprecated("gone soon")
    def f():
        return 1

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f()
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
