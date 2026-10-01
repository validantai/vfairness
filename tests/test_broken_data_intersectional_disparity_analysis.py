"""Pin: a supplied single attribute is never dropped in silence.

intersectional_disparity_analysis compared each single attribute's length with
the rows that survived validate_inputs, so ONE excluded row made every single
attribute fail the test and vanish, and the comparison then reported "no
single_attributes were supplied". Measured before the fix on 400 rows with 10
predictions NaN: intersectional disparity 0.2055 measured, single disparities
{}, hidden_disparity NaN, comparison_not_run "no single_attributes were
supplied". The MCP intersectional tool reaches it on every frame with one
missing attribute value, because it marks those rows None.

Expected values are recomputed here from the raw arrays, never copied from the
function under test.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.intersectional import (
    intersectional_disparity_analysis,
)


def _data():
    rng = np.random.default_rng(7)
    n = 400
    s = np.array(["a"] * 200 + ["b"] * 200, dtype=object)
    p = (rng.random(n) < np.where(s == "a", 0.6, 0.35)).astype(float)
    y = (rng.random(n) < 0.5).astype(int)
    return s, y, p


def _gap(p, s):
    return abs(p[s == "a"].mean() - p[s == "b"].mean())


def _run(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return intersectional_disparity_analysis(*args, **kw)


def test_control_healthy_single_attribute_measured_exactly():
    s, y, p = _data()
    comp = _run(y, p, s, single_attributes={"sex": s})["comparison"]
    assert comp["single_attribute_disparities"]["sex"] == pytest.approx(_gap(p, s), abs=1e-12)
    assert comp["hidden_disparity"] == pytest.approx(0.0, abs=1e-12)
    assert comp["intersectional_reveals_more"] is False


def test_excluded_prediction_rows_do_not_drop_the_single_attribute():
    s, y, p = _data()
    p = p.copy()
    p[:10] = np.nan
    comp = _run(y, p, s, single_attributes={"sex": s})["comparison"]
    keep = ~np.isnan(p)
    want = _gap(p[keep], s[keep])
    assert "comparison_not_run" not in comp, comp
    assert comp["single_attribute_disparities"]["sex"] == pytest.approx(want, abs=1e-12)
    assert math.isfinite(comp["hidden_disparity"])
    assert comp["intersectional_reveals_more"] is False


def test_missing_sensitive_value_rows_are_aligned_like_the_mcp_caller():
    # The MCP tool marks a row None when any part of its cell is missing and
    # passes the per-attribute parts at the caller's full length.
    s, y, p = _data()
    cell = s.copy()
    cell[[0, 1, 250]] = None
    comp = _run(y, p, cell, single_attributes={"sex": s})["comparison"]
    keep = np.array([c is not None for c in cell])
    assert comp["single_attribute_disparities"]["sex"] == pytest.approx(
        _gap(p[keep], s[keep]), abs=1e-12
    )


def test_an_unalignable_attribute_is_named_not_reported_as_never_supplied():
    s, y, p = _data()
    p = p.copy()
    p[:10] = np.nan
    comp = _run(y, p, s, single_attributes={"sex": s[:7]})["comparison"]
    assert "sex" in comp["single_attributes_unusable"]
    assert "no single_attributes were supplied" not in comp["comparison_not_run"]
    assert math.isnan(comp["hidden_disparity"])
    assert comp["intersectional_reveals_more"] is None
