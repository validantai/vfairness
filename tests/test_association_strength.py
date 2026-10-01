"""Direct tests for association_strength and the proxy scan's incompleteness reporting.

Register #20. ``association_strength`` documents itself as "ONE source of truth
for how strongly does X track Y across the library" and, until 2026-08-27, had
no test at all: `grep -rn association_strength tests/` returned nothing. It
ended `except Exception: return 0.0`, and a returned zero is indistinguishable
from a genuine measurement of no association. Executed on a column that is a
PERFECT proxy for race but stored as list-valued objects (a JSON column,
entirely realistic), the function returned 0.0 and
``identify_proxy_features`` returned ``[]`` with no warning: the tool that
exists to say "this feature is a stand-in for the protected attribute" reported
no proxies on a dataset containing a perfect one.

Covered here, per the audit's recommendation: numeric x numeric, numeric x
nominal, nominal x nominal, and the unhashable-object failure path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    ProxyScanIncompleteWarning,
    association_strength,
    identify_proxy_features,
)

N = 300


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def race(rng):
    return pd.Series(rng.choice(["White", "Black", "Asian"], N), name="race")


# --- the three measurement paths --------------------------------------------


def test_numeric_x_numeric_is_absolute_pearson(rng):
    x = pd.Series(rng.normal(size=N), name="x")
    y = pd.Series(-2.0 * x + rng.normal(scale=1e-9, size=N), name="y")
    assert association_strength(x, y) == pytest.approx(1.0, abs=1e-6)


def test_numeric_x_numeric_independent_is_near_zero(rng):
    x = pd.Series(rng.normal(size=N), name="x")
    y = pd.Series(rng.normal(size=N), name="y")
    assert association_strength(x, y) < 0.25


def test_nominal_x_nominal_perfect_proxy_is_one(race):
    zipc = race.map({"White": "A", "Black": "B", "Asian": "C"})
    zipc.name = "zipc"
    assert association_strength(race, zipc) == pytest.approx(1.0, abs=1e-6)


def test_numeric_x_nominal_uses_the_correlation_ratio(race):
    # Group means far apart, tiny within-group noise -> eta near 1.
    income = race.map({"White": 750.0, "Black": 580.0, "Asian": 760.0}) + 1e-6
    income.name = "income"
    assert association_strength(income, race) > 0.99


def test_result_is_bounded_in_the_unit_interval(rng, race):
    x = pd.Series(rng.normal(size=N), name="x")
    for a, b in [(x, x), (race, race), (x, race), (race, x)]:
        v = association_strength(a, b)
        assert 0.0 <= v <= 1.0 or v != v


# --- the failure path: NaN and a warning, never a fabricated 0.0 -------------


def test_unmeasurable_pair_returns_nan_not_zero(race):
    """The defect. List-valued objects raise inside the nominal path."""
    zipbox = race.map({"White": ["A"], "Black": ["B"], "Asian": ["C"]})
    zipbox.name = "zipbox"
    with pytest.warns(ProxyScanIncompleteWarning):
        value = association_strength(race, zipbox)
    assert value != value, "an unmeasurable association must be NaN, never 0.0"


def test_the_same_proxy_as_plain_strings_scores_one(race):
    """Anchors the severity: it is the STORAGE that defeated the measurement."""
    zipbox = race.map({"White": ["A"], "Black": ["B"], "Asian": ["C"]})
    zipstr = race.map({"White": "A", "Black": "B", "Asian": "C"})
    zipbox.name = zipstr.name = "zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProxyScanIncompleteWarning)
        unmeasurable = association_strength(race, zipbox)
    assert unmeasurable != unmeasurable
    assert association_strength(race, zipstr) == pytest.approx(1.0, abs=1e-6)


def test_insufficient_overlap_returns_nan_and_warns():
    a = pd.Series([1.0, 2.0, 3.0], name="a")
    b = pd.Series([1.0, 2.0, 3.0], name="b")
    with pytest.warns(ProxyScanIncompleteWarning, match="aligned non-null rows"):
        value = association_strength(a, b)
    assert value != value, "too few rows is could-not-measure, not zero association"


def test_a_measurable_pair_raises_no_warning(rng):
    x = pd.Series(rng.normal(size=N), name="x")
    y = pd.Series(rng.normal(size=N), name="y")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProxyScanIncompleteWarning)
        association_strength(x, y)  # must not warn


# --- the scan must say when its answer is incomplete -------------------------


def test_proxy_scan_warns_when_a_column_could_not_be_assessed(race):
    zipbox = race.map({"White": ["A"], "Black": ["B"], "Asian": ["C"]})
    df = pd.DataFrame({"race": race, "zipbox": zipbox})
    with pytest.warns(ProxyScanIncompleteWarning, match="could NOT be assessed"):
        proxies = identify_proxy_features(df, "race")
    assert proxies == [], "precondition: the perfect proxy is still not detectable"
    # The point is not that it found it. The point is that the empty answer is
    # no longer readable as "no proxies exist".


def test_the_warning_names_the_column_it_could_not_assess(race):
    zipbox = race.map({"White": ["A"], "Black": ["B"], "Asian": ["C"]})
    df = pd.DataFrame({"race": race, "zipbox": zipbox})
    with pytest.warns(ProxyScanIncompleteWarning) as rec:
        identify_proxy_features(df, "race")
    assert "zipbox" in str(rec[0].message)


def test_a_fully_assessable_scan_is_silent_and_still_finds_the_proxy(rng, race):
    """Control. The fix must not warn on every scan, nor lose a real detection."""
    zipstr = race.map({"White": "A", "Black": "B", "Asian": "C"})
    noise = pd.Series(rng.normal(size=N))
    df = pd.DataFrame({"race": race, "zipstr": zipstr, "noise": noise})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProxyScanIncompleteWarning)
        proxies = identify_proxy_features(df, "race")
    assert [p["column"] for p in proxies] == ["zipstr"]
    assert proxies[0]["risk_level"] == "high"
    assert proxies[0]["abs_correlation"] == pytest.approx(1.0, abs=1e-6)


def test_an_incomplete_scan_still_reports_the_proxies_it_could_measure(rng, race):
    """A dropped column must not suppress a real finding beside it."""
    zipstr = race.map({"White": "A", "Black": "B", "Asian": "C"})
    zipbox = race.map({"White": ["A"], "Black": ["B"], "Asian": ["C"]})
    df = pd.DataFrame({"race": race, "zipstr": zipstr, "zipbox": zipbox})
    with pytest.warns(ProxyScanIncompleteWarning):
        proxies = identify_proxy_features(df, "race")
    assert [p["column"] for p in proxies] == ["zipstr"]
