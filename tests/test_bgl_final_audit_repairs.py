"""The three items the Stage 2 final audit refused to sign off.

Each is a defect the FIX introduced or left behind, which is the reason the audit
pass exists: a fix that is real, correct and well-pinned can still be wrong about
the thing it claims to have closed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. The design floor was switched off above 1000 rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [900, 1002, 1200, 2000, 20000])
def test_the_design_floor_is_computed_at_any_number_of_rows(n: int) -> None:
    """A row cap belonging to a SIBLING function had been copied here.

    _label_arrangement_floor computes math.factorial(n) exactly and needs the
    cap: 20,504 ms at a million rows, and OverflowError on conversion from 2,000
    rows up. This one spends O(n log n) in np.unique and then a flat 0.8 us of
    lgamma arithmetic, so the cap bought back 0.8 us and cost the refusal. With
    it, the floor returned None above 1000 rows, so the refusal it exists to
    produce worked only on inputs too small to matter. The audit measured the
    consequence at n=1002/1200/2000: significant_at_05=False against a floor
    274x below anything reachable through comprehensive_fairness_test, 544x
    through permutation_test directly.
    """
    from vfairness.evaluation.vfairness_metrics.robustness import _effective_design_floor

    sensitive = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    floor = _effective_design_floor(sensitive, 4)
    assert floor is not None, f"the design floor switched itself off at n={n}"
    assert 0.0 < floor < 1.0


def test_the_floor_is_stable_across_the_old_cap() -> None:
    """Anti-vacuity: the value must not jump at the boundary, or the cap was
    hiding a discontinuity rather than only a cost."""
    from vfairness.evaluation.vfairness_metrics.robustness import _effective_design_floor

    def f(n: int) -> float:
        sa = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
        return _effective_design_floor(sa, 4)

    assert f(1000) == pytest.approx(f(1002), rel=1e-3)


def test_the_public_entry_refuses_a_powerless_metric_above_the_old_cap() -> None:
    """The shape the defect was actually about, at the PUBLIC entry.

    The two pins above address the two halves separately: the cap inside
    _effective_design_floor, and (in test_bgl_final_d04.py) the n_design_rows
    plumbing through comprehensive_fairness_test at a 200-row fixture. The audit
    of 2026-09-17 found that reverting EITHER half is detected and reverting the
    combination at real audit size is not, because no test ran the public entry
    above 1000 rows. That is precisely the regime the defect lived in.

    equal_opportunity reads only the true-positive rows. With four of them in
    1200, no arrangement of the labels can put the p-value under 0.05, so a
    graded significant_at_05=False would be a verdict the data could never have
    contradicted. It must refuse instead.
    """
    from vfairness.evaluation.vfairness_metrics.robustness import (
        comprehensive_fairness_test,
    )

    n = 1200
    rng = np.random.default_rng(20250917)
    sensitive = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    y_true = np.zeros(n, dtype=int)
    y_true[[0, 1, n // 2, n // 2 + 1]] = 1  # four positives, two per group
    y_pred = np.zeros(n, dtype=int)
    # A real, findable demographic-parity gap: 30% of group a, 4% of group b.
    y_pred[: n // 2] = (rng.random(n // 2) < 0.30).astype(int)
    y_pred[n // 2 :] = (rng.random(n - n // 2) < 0.04).astype(int)

    out = comprehensive_fairness_test(y_true, y_pred, sensitive, n_permutations=200, random_state=7)
    eo = out["metric_tests"]["equal_opportunity"]

    assert eo["significant_at_05"] is None, (
        "the public entry graded a metric that could not have fired at n=1200; "
        "the design floor is switched off again above the old 1000-row cap"
    )
    assert math.isnan(eo["p_value"])
    assert eo["detectable_at_05"] is False
    assert "equal_opportunity" in out["not_assessable"]
    assert eo["min_attainable_p_value"] > 0.05

    # Control, same run: the metric that DOES read every row still measures the
    # real gap, so the refusal above is about power and not a blanket refusal.
    dp = out["metric_tests"]["demographic_parity"]
    assert dp["significant_at_05"] is True
    assert "demographic_parity" in out["significant_metrics"]
    assert out["any_significant"] is True
    assert out["n_tests"] == 2  # the powerless metric left the family


# ---------------------------------------------------------------------------
# 2. A cancelling IG explanation bought its own tolerance.
# ---------------------------------------------------------------------------


def test_completeness_is_scaled_by_movement_not_by_attribution_mass() -> None:
    """Completeness is sum(attributions) == prediction - base_value.

    Attribution mass used to sit in the denominator, so an explanation whose
    parts are large and OPPOSITE had huge mass, explained nothing, and passed:
    any run whose mass exceeded 20x the residual was graded complete. Measured by
    the audit at amplitude 50 and 100, with 100% of the movement unexplained and
    zero warnings.
    """
    from vfairness.xai.explainers.ig_adapter import completeness_scale

    # The model moved by 1.0; the explanation is two huge cancelling terms.
    cancelling = np.array([500.0, -500.0])
    scale = completeness_scale(1.0, 0.0, cancelling)
    assert scale == pytest.approx(1.0), (
        f"scale={scale}; attribution mass is buying tolerance for an explanation "
        f"that accounts for none of the movement"
    )

    # CONTROL: an honest explanation of the same movement is unaffected.
    honest = np.array([0.6, 0.4])
    assert completeness_scale(1.0, 0.0, honest) == pytest.approx(1.0)


def test_completeness_scale_refuses_a_non_finite_movement() -> None:
    from vfairness.xai.explainers.ig_adapter import completeness_scale

    assert math.isnan(completeness_scale(float("nan"), 0.0, np.array([1.0])))


# ---------------------------------------------------------------------------
# 3. The third copy of Cramer's V published 0.0 for an undefined statistic.
# ---------------------------------------------------------------------------


def test_the_proxy_screens_cramers_v_refuses_an_undefined_table() -> None:
    """`else: cramers_v = 0.0` said NO ASSOCIATION for a table the bias-corrected
    statistic is not defined on. The guarded twin in feature_engineering already
    refused; this copy had drifted from it, and it is the third place this exact
    shape has appeared."""
    import pandas as pd

    from vfairness.preprocessing.bias_detection.proxy import compute_proxy_correlations

    # EVERY ROW ITS OWN CATEGORY in both dimensions. Then r_corr = k_corr = 1
    # and the corrected minimum dimension is exactly 0, which is the branch under
    # test. The first fixture here used a single-level attribute, which never
    # reached it: chi2_contingency raises on that table and the surrounding
    # `except` swallowed it, so the sabotage came back green and the pin proved
    # nothing. An identifier-shaped column is also the realistic case, and it is
    # exactly the shape a strong proxy often takes.
    n = 40
    df = pd.DataFrame(
        {
            "feature": [f"f{i}" for i in range(n)],
            "attr": [f"a{i}" for i in range(n)],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = compute_proxy_correlations(df, "feature", "attr")
    v = (out.get("all_correlations") or {}).get("cramers_v")
    assert v is not None, "the fixture never reached the branch; this pins nothing"
    assert math.isnan(v), f"cramers_v={v!r} for a table the corrected statistic is not defined on"
    assert any("not defined" in str(c.message) for c in caught), "the refusal was silent"


def test_control_a_real_categorical_association_still_measures() -> None:
    """Over-correction control: the refusal must not swallow a genuine one."""
    import pandas as pd

    from vfairness.preprocessing.bias_detection.proxy import compute_proxy_correlations

    grp = ["a"] * 60 + ["b"] * 60
    df = pd.DataFrame({"feature": grp, "attr": grp})  # perfectly associated
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_proxy_correlations(df, "feature", "attr")
    v = (out.get("all_correlations") or {}).get("cramers_v")
    assert v is not None and not math.isnan(v) and v > 0.5, (
        f"a perfect categorical association measured {v!r}"
    )
