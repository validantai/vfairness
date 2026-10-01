"""Beta Go-Live final sweep, group u02: the two capabilities the 2026-09-11
census never REACHED.

``identify_proxy_variables`` and ``predictive_parity_difference`` were recorded
BGL-C UNPROVEN, meaning no generic fixture called either of them with data, so
nothing was known about whether they fabricate. Unknown is a positive statement
that nothing is known, never a clean bill. This file is the execution evidence
that closes both rows, and the pins that keep the answer true.

Measured 2026-09-17 on the real functions, warnings ENABLED (a refusal carried
only in a warning is invisible when warnings are suppressed, and two of the
three defects below were exactly that):

* ``identify_proxy_variables`` MEASURES. On 600 rows where a continuous feature
  encodes a 3-level categorical protected attribute it returns eta=0.984
  CRITICAL, and it leaves a genuinely independent feature unflagged. Its
  refusals were already loud for the sample-size floor and for an association
  that could not be computed.
* Two silent skips at the TOP of its loop were not. A protected attribute that
  is not a column of the frame, and one whose column cannot be encoded, were
  each skipped with ``continue`` and no warning at all, so a renamed or mistyped
  column produced a proxy report that is byte-identical to a clean one: the
  function returns a LIST, and "screened, found nothing" and "never looked" are
  both ``[]``.
* ``compute_proxy_correlations``, the single-pair convenience wrapper in the
  same module, read its decision statistic out of the correlations dict with a
  ``0`` default. A statistic that is absent was never measured, so an
  unmeasurable pair came back as ``{'primary_correlation': 0, 'risk_level':
  'negligible'}``, a confident clean bill. ``_analyze_proxy_relationship``
  already refused that case behind an ``np.isfinite`` guard; this sibling still
  carried it.
* ``predictive_parity_difference`` MEASURES, and every unmeasurable input it was
  given returned NaN, never the 0.0 that reads as perfect parity. But its
  insufficient-evidence gate counted the groups with a DEFINED precision rather
  than the QUALIFYING groups its siblings count, which made its own warning
  unreachable in the case its own comment calls canonical: two groups, one the
  model never predicts positive for. That input returned a bare NaN with no
  warning of any kind.

Every value-bearing assertion here is paired with a healthy-data control, so a
guard that starts refusing real measurements fails this file rather than
quietly making it greener.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics import classification as C
from vfairness.preprocessing.bias_detection.proxy import (
    compute_proxy_correlations,
    identify_proxy_variables,
)

# ---------------------------------------------------------------------------
# Fixtures. min_sample_size defaults to 100 for the proxy screen and
# min_group_size to 30 for the metric, so every frame below clears its own
# gate on purpose: three times in this campaign a capability looked honest
# only because the fixture was too small to reach the measurement.
# ---------------------------------------------------------------------------

N_PROXY = 600
_GROUP_MEANS = {"White": 750.0, "Black": 580.0, "Asian": 760.0}
_GROUP_ZIPS = {"White": "10001", "Black": "10002", "Asian": "10003"}


def _proxy_frame(n: int = N_PROXY, seed: int = 7) -> pd.DataFrame:
    """race is a 3-level nominal attribute.

    ``credit_rating`` is a near-perfect CONTINUOUS proxy for it (group mean plus
    small noise), ``zip_code`` a near-perfect CATEGORICAL one, and ``unrelated``
    is genuinely independent. The healthy control is that the first two are
    flagged and the third is not.
    """
    rng = np.random.default_rng(seed)
    groups = rng.choice(list(_GROUP_MEANS), size=n, p=[0.4, 0.3, 0.3])
    return pd.DataFrame(
        {
            "race": groups,
            "credit_rating": np.array([_GROUP_MEANS[g] for g in groups])
            + rng.normal(0, 15, size=n),
            "zip_code": np.array([_GROUP_ZIPS[g] for g in groups]),
            "unrelated": rng.normal(500, 100, size=n),
        }
    )


def _ppv_gap_data(n_per_group: int = 100):
    """Two groups of 100. Group A precision 45/50 = 0.90, group B 30/50 = 0.60,
    so the true predictive-parity difference is exactly 0.30."""
    half = n_per_group // 2
    y_true_a = [1] * 45 + [0] * 5 + [1] * 10 + [0] * 40
    y_true_b = [1] * 30 + [0] * 20 + [1] * 10 + [0] * 40
    y_pred_half = [1] * half + [0] * half
    return (
        np.array(y_true_a + y_true_b),
        np.array(y_pred_half + y_pred_half),
        np.array(["A"] * n_per_group + ["B"] * n_per_group),
    )


def _by_feature(results):
    return {r.feature: r for r in results}


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _messages(recorded) -> str:
    return " || ".join(str(w.message) for w in recorded)


# ---------------------------------------------------------------------------
# identify_proxy_variables: it measures (healthy control first)
# ---------------------------------------------------------------------------


def test_proxy_screen_measures_a_real_disparity() -> None:
    """The control for everything below: on healthy data carrying a real,
    findable proxy this capability returns a measurement, not a refusal."""
    results = _quiet(
        identify_proxy_variables,
        _proxy_frame(),
        protected_attributes=["race"],
        include_known_patterns=False,
    )
    by_feature = _by_feature(results)

    assert "credit_rating" in by_feature, "the continuous proxy was not measured"
    cont = by_feature["credit_rating"]
    assert cont.correlation_type == "Correlation ratio (eta)"
    assert cont.correlation > 0.9
    assert cont.risk_level.value == "critical"
    assert cont.sample_size == N_PROXY
    assert cont.pvalue is not None and cont.pvalue < 0.01
    assert cont.evidence["pvalue_status"] == "computed"

    assert "zip_code" in by_feature, "the categorical proxy was not measured"
    assert by_feature["zip_code"].correlation_type == "Cramér's V"
    assert by_feature["zip_code"].correlation > 0.9

    assert "unrelated" not in by_feature, (
        "an independent feature was flagged as a proxy; the screen is not discriminating"
    )


# ---------------------------------------------------------------------------
# identify_proxy_variables: the two silent skips
# ---------------------------------------------------------------------------


def test_absent_protected_attribute_is_disclosed_not_silently_skipped() -> None:
    """DEFECT PIN. ``if attr not in df.columns: continue`` returned the same
    empty list as a clean screen. A caller asking for 'gender' on a frame whose
    column is spelled 'sex' got a proxy report saying nothing was found about an
    attribute nothing was looked at."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            _proxy_frame(),
            protected_attributes=["gender", "age"],
            include_known_patterns=False,
        )

    assert results == [], "no attribute was screenable, so there is nothing to report"
    text = _messages(caught)
    assert "UNASSESSED" in text
    for attr in ("gender", "age"):
        assert f"'{attr}'" in text, f"the empty report never says {attr} was not screened"


def test_absent_attribute_does_not_suppress_the_attributes_that_are_present() -> None:
    """The guard above must not delete a real finding: one missing attribute in
    the list cannot cost the screen its measurement on the present one."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            _proxy_frame(),
            protected_attributes=["race", "gender"],
            include_known_patterns=False,
        )

    by_feature = _by_feature(results)
    assert "credit_rating" in by_feature, "the real proxy finding was lost"
    assert by_feature["credit_rating"].correlation > 0.9
    assert "'gender'" in _messages(caught)


def test_unencodable_protected_attribute_is_disclosed_not_silently_skipped() -> None:
    """DEFECT PIN, same class one branch down: ``_encode_for_correlation``
    returns None for a column it cannot turn into numbers, and the skip was
    silent, so every feature went unscreened behind an ordinary empty list."""
    df = _proxy_frame()
    df["race_list"] = [[g] for g in df["race"]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            df,
            protected_attributes=["race_list"],
            feature_columns=["credit_rating"],
            include_known_patterns=False,
        )

    assert results == []
    text = _messages(caught)
    assert "could not be encoded" in text
    assert "UNASSESSED" in text
    assert "'race_list'" in text


# ---------------------------------------------------------------------------
# identify_proxy_variables: degenerate inputs relevant to a proxy CLAIM
# ---------------------------------------------------------------------------


def test_single_group_attribute_is_not_screened_and_says_so() -> None:
    """One group only: with no variation in the protected attribute there is no
    association to measure for ANY feature. The empty list must be accompanied
    by the statement that the pairs were NOT SCREENED."""
    df = _proxy_frame()
    df["race"] = "White"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )

    assert results == []
    text = _messages(caught)
    assert "NOT SCREENED" in text
    assert "could-not-check" in text
    assert "credit_rating" in text


@pytest.mark.parametrize("n_rows", [0, 2, 99])
def test_below_the_sample_floor_is_unassessed_not_clean(n_rows: int) -> None:
    """Zero rows, two rows, and one row short of min_sample_size all land on the
    same floor. The empty list must never read as 'no proxies'."""
    df = _proxy_frame().iloc[:n_rows]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )

    assert results == []
    text = _messages(caught)
    assert "min_sample_size=100" in text
    assert "UNASSESSED for this attribute, not clean" in text


def test_all_nan_feature_is_reported_unassessed_while_the_others_still_measure() -> None:
    """An unreadable feature column is a could-not-check for THAT feature only.
    The control half of this test is that the measurable features still land."""
    df = _proxy_frame()
    df["credit_rating"] = np.nan

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )

    by_feature = _by_feature(results)
    assert "credit_rating" not in by_feature
    assert "zip_code" in by_feature, "a NaN column cost the screen an unrelated finding"
    assert by_feature["zip_code"].correlation > 0.9

    text = _messages(caught)
    assert "credit_rating" in text
    assert "UNASSESSED, not clean" in text


def test_high_cardinality_free_text_does_not_cost_the_real_findings() -> None:
    """Unreadable text: a per-row free-text column is dropped by the cardinality
    filter. It must not disturb the features that CAN be screened."""
    df = _proxy_frame()
    df["notes"] = [f"free text {i}" for i in range(len(df))]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            df, protected_attributes=["race"], include_known_patterns=False
        )

    by_feature = _by_feature(results)
    assert "notes" not in by_feature
    assert {"credit_rating", "zip_code"} <= set(by_feature)

    # The drop is deliberate and correct, but it leaves the column with NO
    # proxy verdict, which is not the same as a verdict of clean. Before this
    # was collected the skip was silent and the column simply vanished from the
    # report, indistinguishable from one measured below the threshold.
    text = _messages(caught)
    assert "notes" in text
    assert "max_cardinality=50" in text
    assert "UNASSESSED, not clean" in text


def test_a_low_cardinality_frame_raises_no_cardinality_warning() -> None:
    """Control: the disclosure above must not fire for an ordinary frame."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identify_proxy_variables(
            _proxy_frame(), protected_attributes=["race"], include_known_patterns=False
        )

    assert "max_cardinality" not in _messages(caught)


# ---------------------------------------------------------------------------
# compute_proxy_correlations: the sibling that defaulted an unmeasured
# statistic to 0 and then GRADED it
# ---------------------------------------------------------------------------


def test_single_pair_correlation_measures_on_healthy_data() -> None:
    """Healthy control for the pin below."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_proxy_correlations(_proxy_frame(), "credit_rating", "race")

    assert out["measurement_status"] == "measured"
    assert out["primary_correlation_type"] == "Correlation ratio (eta)"
    assert out["primary_correlation"] > 0.9
    assert out["risk_level"] == "critical"
    assert "NOT SCREENED" not in _messages(caught)


def test_single_pair_correlation_refuses_an_unmeasurable_pair() -> None:
    """DEFECT PIN. With a constant protected attribute the correlation ratio is
    never computed at all, so its key is absent from the dict. The ``0`` default
    published that as a measured zero and ``_determine_risk_level(0.0, ...)``
    graded it NEGLIGIBLE."""
    df = _proxy_frame()
    df["race"] = "White"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_proxy_correlations(df, "credit_rating", "race")

    assert "correlation_ratio" not in out["all_correlations"], (
        "fixture no longer reproduces the unmeasurable case"
    )
    assert math.isnan(out["primary_correlation"]), (
        f"an unmeasured association came back as {out['primary_correlation']!r}"
    )
    assert out["primary_correlation"] != 0, "0 is the value of no association at all"
    assert out["measurement_status"] == "could_not_measure"
    assert out["risk_level"] is None, "an unmeasured pair was graded"
    assert "NOT SCREENED" in _messages(caught)


# ---------------------------------------------------------------------------
# predictive_parity_difference: it measures (healthy control first)
# ---------------------------------------------------------------------------


def test_predictive_parity_measures_a_real_gap() -> None:
    """The control for every refusal below. Precision 0.90 against 0.60 is a
    real, hand-checkable disparity and must come back as 0.30."""
    y_true, y_pred, sens = _ppv_gap_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert got == pytest.approx(0.30, abs=1e-9)
    assert "NOT MEASURABLE" not in _messages(caught)


def test_a_genuinely_measured_zero_survives_the_refusals() -> None:
    """A fix can DELETE A REAL FINDING. Here 0.0 is the honest answer: every
    label is positive, so both groups' precision is a defined 1.0 and the gap
    really is zero. It must not be turned into NaN by the guards above it."""
    n = 100
    y_true = np.ones(2 * n, dtype=int)
    y_pred = np.array(([1] * (n // 2) + [0] * (n // 2)) * 2)
    sens = np.array(["A"] * n + ["B"] * n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert got == 0.0
    assert not math.isnan(got), "a measured 0.0 was refused as unmeasurable"
    assert "NOT MEASURABLE" not in _messages(caught)


def test_predictive_parity_still_measures_when_a_tiny_group_is_dropped() -> None:
    """The size gate excludes a group, it does not make the metric unassessable.
    The survivors' gap is still measured and the exclusion is disclosed."""
    y_true, y_pred, sens = _ppv_gap_data()
    y_true = np.concatenate([y_true, np.ones(5, dtype=int)])
    y_pred = np.concatenate([y_pred, np.ones(5, dtype=int)])
    sens = np.concatenate([sens, np.array(["C"] * 5)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert got == pytest.approx(0.30, abs=1e-9)
    assert "min_group_size" in _messages(caught)


# ---------------------------------------------------------------------------
# predictive_parity_difference: degenerate inputs relevant to a PRECISION claim
# ---------------------------------------------------------------------------


def test_group_never_predicted_positive_returns_nan_and_names_the_group() -> None:
    """DEFECT PIN. This is the case the function's own comment calls canonical,
    a model that never approves one group, and it was the case its own warning
    could not reach: the gate counted DEFINED precisions (1 here) rather than
    QUALIFYING groups (2), so it returned a bare NaN with nothing said."""
    y_true, y_pred, sens = _ppv_gap_data()
    y_pred = np.concatenate([y_pred[:100], np.zeros(100, dtype=int)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert math.isnan(got)
    assert got != 0.0
    text = _messages(caught)
    assert "NOT MEASURABLE" in text
    assert "'B'" in text, "the refusal does not name the group it could not measure"


def test_no_positive_predictions_anywhere_returns_nan_and_names_both_groups() -> None:
    """Every score identical, at the value that makes precision undefined for
    everyone: no group has a denominator, so nothing is comparable."""
    y_true, _y_pred, sens = _ppv_gap_data()
    y_pred = np.zeros(len(y_true), dtype=int)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert math.isnan(got)
    text = _messages(caught)
    assert "NOT MEASURABLE" in text
    assert "'A'" in text and "'B'" in text


def test_three_groups_with_one_unmeasurable_arm_refuses_the_whole_metric() -> None:
    """Reporting the two measurable groups' spread would be a LOWER BOUND
    presented as a measurement: the dropped arm's precision could be anything."""
    y_true, y_pred, sens = _ppv_gap_data()
    y_true = np.concatenate([y_true, np.array([1] * 50 + [0] * 50)])
    y_pred = np.concatenate([y_pred, np.zeros(100, dtype=int)])
    sens = np.concatenate([sens, np.array(["C"] * 100)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert math.isnan(got)
    assert got != pytest.approx(0.30), "the survivors' gap was published as the answer"
    assert "'C'" in _messages(caught)


@pytest.mark.parametrize(
    "label, n_a, n_b",
    [("one group only", 100, 0), ("one group has a single row", 100, 1), ("zero rows", 0, 0)],
)
def test_too_few_qualifying_groups_returns_nan_and_says_why(label: str, n_a: int, n_b: int) -> None:
    """Insufficient evidence, never the 0.0 that reads as perfect parity, and
    never in silence."""
    y_true = np.array([1, 0] * ((n_a + n_b) // 2) + [1] * ((n_a + n_b) % 2))
    y_pred = np.ones(n_a + n_b, dtype=int)
    sens = np.array(["A"] * n_a + ["B"] * n_b)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(y_true, y_pred, sens)

    assert math.isnan(got), f"{label}: returned {got!r} for a comparison that never ran"
    assert got != 0.0
    text = _messages(caught)
    assert "NOT MEASURABLE" in text, f"{label}: the refusal was silent"
    assert "insufficient evidence" in text


def test_every_row_missing_is_insufficient_evidence_not_parity() -> None:
    """All labels NaN: the exclude strategy empties the frame, so no group
    qualifies and nothing is comparable."""
    y_true, y_pred, sens = _ppv_gap_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = C.predictive_parity_difference(
            np.full(len(y_true), np.nan), y_pred.astype(float), sens
        )

    assert math.isnan(got)
    assert "NOT MEASURABLE" in _messages(caught)


def test_an_unknown_feature_column_is_disclosed_not_silently_dropped() -> None:
    """The feature-level twin of the attribute pin above. A caller who names a
    column that is not in the frame had it dropped by a silent ``continue``, so
    a config listing four features to screen could screen three and hand back a
    result that reads as if it had screened four."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = identify_proxy_variables(
            _proxy_frame(),
            protected_attributes=["race"],
            feature_columns=["credit_rating", "TYPO_RATING"],
            include_known_patterns=False,
        )

    by_feature = _by_feature(results)
    assert "credit_rating" in by_feature, "the guard cost the screen a real finding"
    assert by_feature["credit_rating"].correlation > 0.9

    text = _messages(caught)
    assert "TYPO_RATING" in text
    assert "NOT screened" in text


def test_a_fully_present_feature_list_raises_no_unknown_column_warning() -> None:
    """Control: the disclosure above must not fire for an ordinary scoped call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identify_proxy_variables(
            _proxy_frame(),
            protected_attributes=["race"],
            feature_columns=["credit_rating", "unrelated"],
            include_known_patterns=False,
        )

    assert "not in the frame" not in _messages(caught)
