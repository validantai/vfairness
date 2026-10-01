"""The two entry points a first-time user of this library actually calls.

`FairnessAnalyzer` is the class the README, the quickstart and the API reference
all lead with, and on 2026-09-25 its badge read NOT CHECKED because seven of its
methods had never been graded. Two of those seven are the ones a user reaches
first, and they turned out to sit on opposite sides of the defect this library
exists to remove:

  compute_all_metrics     ALREADY HONEST. On a single-group frame, on one label
                          only, on an empty frame and at n=2 it returns nan for
                          every metric and warns. Nothing is changed here; what
                          was missing is a test holding it there.
  compare_with_fairlearn  FABRICATED. It handed back fairlearn's conventional 0.0
                          with no warning for a single-group frame, where the
                          SAME object's compute_all_metrics returns nan. A number
                          carrying another project's name reads as independent
                          corroboration of parity, which makes this the worse of
                          the two to get wrong.

Every refusal test below is paired with a control on measurable data, because a
method that refuses everything passes every refusal test ever written.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer

N = 60


def _frame(kind: str):
    rng = np.random.default_rng(3)
    y = (rng.random(N) < 0.5).astype(int)
    pred = y.copy()
    groups = np.array(["a"] * (N // 2) + ["b"] * (N - N // 2))
    if kind == "single_group":
        groups = np.array(["a"] * N)
    elif kind == "one_label":
        y = np.zeros(N, dtype=int)
        pred = np.zeros(N, dtype=int)
    elif kind == "n_equals_2":
        return y[:2], pred[:2], groups[:2]
    elif kind == "empty":
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=object),
        )
    elif kind == "one_tiny_group":
        groups = np.array(["a"] * (N - 2) + ["b"] * 2)
    elif kind == "real_gap":
        # b is selected far less often: a disparity both libraries must find.
        y = np.array([1] * (N // 2) + [0] * (N - N // 2))
        pred = y.copy()
        groups = np.array(["a"] * (N // 2) + ["b"] * (N - N // 2))
    return y, pred, groups


def _analyzer(kind: str) -> FairnessAnalyzer:
    y, pred, groups = _frame(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y, pred, groups)


def _caught(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in rec]


# ── compute_all_metrics ───────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["single_group", "one_label", "n_equals_2", "empty"])
def test_compute_all_metrics_refuses_where_nothing_is_measurable(kind):
    metrics, msgs = _caught(lambda: _analyzer(kind).compute_all_metrics())
    assert metrics, "no metric keys at all, so this test would pass vacuously"
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    assert numeric, f"{kind} returned no numeric metric to check"
    measured = [v for v in numeric if not (isinstance(v, float) and math.isnan(v))]
    # demographic parity on a single-label frame is a legitimate measured 0.0
    # (every rate is identical and observed), so the rule is that a degenerate
    # frame may not come back with EVERY metric measured and no warning at all.
    assert msgs or not measured, (
        f"{kind}: {len(measured)} of {len(numeric)} metrics came back measured with no warning"
    )


def test_control_compute_all_metrics_still_measures_a_real_disparity():
    metrics, _msgs = _caught(lambda: _analyzer("real_gap").compute_all_metrics())
    dp = metrics.get("demographic_parity_difference")
    assert dp is not None and not math.isnan(dp), "a total selection-rate gap was refused"
    assert dp == pytest.approx(1.0, abs=1e-9), (
        f"group a is selected always and group b never, so the gap is 1.0, got {dp}"
    )


def test_control_compute_all_metrics_is_exact_on_a_known_frame():
    """Recomputed independently rather than copied from what the code returned."""
    y = np.array([1, 1, 0, 0] * 15)
    pred = np.array([1, 1, 1, 0] * 15)  # group a selected 3/4, group b 3/4 too
    groups = np.array(["a", "a", "b", "b"] * 15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = FairnessAnalyzer(y, pred, groups).compute_all_metrics()
    rate_a = pred[groups == "a"].mean()
    rate_b = pred[groups == "b"].mean()
    assert metrics["demographic_parity_difference"] == pytest.approx(abs(rate_a - rate_b))


# ── compare_with_fairlearn ────────────────────────────────────────────────────

fairlearn = pytest.importorskip("fairlearn", reason="the comparison needs fairlearn")


@pytest.mark.parametrize("kind", ["single_group", "one_tiny_group"])
def test_a_cross_library_comparison_is_never_reported_where_none_is_defined(kind):
    out, msgs = _caught(lambda: _analyzer(kind).compare_with_fairlearn())
    assert out is not None
    assert math.isnan(out["fairlearn_demographic_parity_difference"]), (
        f"{kind} reported {out['fairlearn_demographic_parity_difference']} as a parity comparison"
    )
    assert math.isnan(out["fairlearn_equalized_odds_difference"])
    assert "not_comparable" in out
    assert any("could not check" in m for m in msgs), f"{kind} refused in silence"


def test_fairlearns_own_convention_is_kept_rather_than_dropped(kind="single_group"):
    """Withholding the verdict must not destroy the evidence: the point of this
    method is to show what the reference library says."""
    out, _msgs = _caught(lambda: _analyzer(kind).compare_with_fairlearn())
    assert "fairlearn_reported_demographic_parity_difference" in out
    assert out["fairlearn_reported_demographic_parity_difference"] == pytest.approx(0.0)


def test_control_a_real_comparison_still_comes_back_measured():
    out, msgs = _caught(lambda: _analyzer("healthy").compare_with_fairlearn())
    assert not math.isnan(out["fairlearn_demographic_parity_difference"])
    assert "not_comparable" not in out
    assert not msgs, f"a comparable frame emitted {msgs}"


def test_control_the_comparison_agrees_with_our_own_metric_on_a_real_gap():
    """The whole purpose of the method, and the strongest control there is: on data
    where a disparity exists, the two libraries must land on the same number."""
    analyzer = _analyzer("real_gap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = analyzer.compute_all_metrics()["demographic_parity_difference"]
        theirs = analyzer.compare_with_fairlearn()["fairlearn_demographic_parity_difference"]
    assert theirs == pytest.approx(ours, abs=1e-9), (
        f"vfairness says {ours} and fairlearn says {theirs} about the same frame"
    )


def test_the_refusal_uses_the_analyzers_own_definition_of_comparable():
    """Not a fresh rule: raising min_group_size must move the boundary, so the two
    halves of the object can never disagree about what is comparable."""
    y, pred, groups = _frame("healthy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict = FairnessAnalyzer(y, pred, groups, min_group_size=1000)
        loose = FairnessAnalyzer(y, pred, groups, min_group_size=5)
    assert math.isnan(
        _caught(strict.compare_with_fairlearn)[0]["fairlearn_demographic_parity_difference"]
    )
    assert not math.isnan(
        _caught(loose.compare_with_fairlearn)[0]["fairlearn_demographic_parity_difference"]
    )


# ── compare_with_aequitas ─────────────────────────────────────────────────────
#
# THE SAME DEFECT AS THE TWIN ABOVE, LEFT BEHIND. The guard
# compare_with_fairlearn gained on 2026-09-25 was never carried across, and on
# 2026-09-29 the before-state was measured at this entry point with the stub
# below (the guard needs no aequitas installed):
#
#   FairnessAnalyzer(y, y, np.array(['a'] * 60)).compare_with_aequitas()
#   -> {'attribute_value': {0: 'a'}, 'ppr_disparity': {0: 1.0},
#       'fpr_disparity': {0: 1.0}}
#
# zero warnings, no not_comparable key, while the same object's own
# demographic_parity_difference() is nan and the twin returned nan + one warning.
# Worse than the twin's before-state, because ref_groups_dict is self.groups[0]:
# on a one-group frame the only row is a group compared with ITSELF, whose
# disparity is 1.0 by construction and reads as full four-fifths compliance under
# another project's name.
#
# The stub is deliberately faithful on the one point that matters: it computes
# ppr_disparity as a REAL selection-rate ratio out of the frame it is handed, so
# the control below asserts a number aequitas would actually produce, and the
# refusal cannot be passed by a stub that returns nothing.


class _StubGroup:
    def get_crosstabs(self, df, attr_cols=None):
        import pandas as pd

        return pd.DataFrame({"attribute_value": sorted(df["sensitive_attr"].unique())}), None


class _StubBias:
    def get_disparity_predefined_groups(
        self, xtab, original_df=None, ref_groups_dict=None, alpha=0.05
    ):
        import pandas as pd

        ref = ref_groups_dict["sensitive_attr"]
        rates = original_df.groupby("sensitive_attr")["score"].mean()
        ref_rate = rates[ref]
        values = sorted(original_df["sensitive_attr"].unique())
        return pd.DataFrame(
            {
                "attribute_value": values,
                "ppr_disparity": [
                    (rates[v] / ref_rate if ref_rate else float("nan")) for v in values
                ],
            }
        )


@pytest.fixture
def stub_aequitas(monkeypatch):
    """Make `import aequitas` succeed so the real code path runs unchanged."""
    import sys
    import types

    aequitas = types.ModuleType("aequitas")
    group = types.ModuleType("aequitas.group")
    bias = types.ModuleType("aequitas.bias")
    group.Group = _StubGroup
    bias.Bias = _StubBias
    aequitas.group = group
    aequitas.bias = bias
    monkeypatch.setitem(sys.modules, "aequitas", aequitas)
    monkeypatch.setitem(sys.modules, "aequitas.group", group)
    monkeypatch.setitem(sys.modules, "aequitas.bias", bias)


@pytest.mark.parametrize("kind", ["single_group", "one_tiny_group"])
def test_an_aequitas_disparity_is_never_reported_where_none_is_defined(stub_aequitas, kind):
    out, msgs = _caught(lambda: _analyzer(kind).compare_with_aequitas())
    assert out is not None and "error" not in out, f"{kind}: {out}"
    disparities = [c for c in out if str(c).endswith("_disparity")]
    assert disparities, "no disparity column at all, so this test would pass vacuously"
    for column in disparities:
        for index, value in out[column].items():
            assert math.isnan(value), (
                f"{kind}: {column}[{index}] reported {value} as a between-group disparity"
            )
    assert "not_comparable" in out
    assert any("could not check" in m for m in msgs), f"{kind} refused in silence"


def test_the_self_comparison_is_named_on_every_return(stub_aequitas):
    """ref_groups_dict is self.groups[0], so one row of the table is always a
    group compared with itself. A reader cannot tell which row unless it says."""
    for kind in ("single_group", "healthy"):
        out, _msgs = _caught(lambda: _analyzer(kind).compare_with_aequitas())
        assert out["reference_group"] == "a", kind
        assert "itself" in out["reference_group_note"], kind


def test_aequitas_own_output_is_kept_rather_than_dropped(stub_aequitas):
    """Withholding the verdict must not destroy the evidence."""
    out, _msgs = _caught(lambda: _analyzer("single_group").compare_with_aequitas())
    kept = out["aequitas_reported_disparities"]
    assert kept["ppr_disparity"][0] == pytest.approx(1.0), kept
    assert out["disparity_columns_withheld"] == ["ppr_disparity"]


def test_control_a_comparable_frame_still_comes_back_with_its_real_ratio(stub_aequitas):
    """OVER-CORRECTION CONTROL: the real number, recomputed here from the frame.
    Group a is selected 4/5 of the time and group b 2/5, so the ratio aequitas is
    handed back for b is 0.5 exactly, and nothing is withheld."""
    pred = np.array([1, 1, 1, 1, 0] * 6 + [1, 1, 0, 0, 0] * 6)
    y = pred.copy()
    groups = np.array(["a"] * 30 + ["b"] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = FairnessAnalyzer(y, pred, groups)
    out, msgs = _caught(analyzer.compare_with_aequitas)
    rate_a = pred[groups == "a"].mean()
    rate_b = pred[groups == "b"].mean()
    assert out["ppr_disparity"][1] == pytest.approx(rate_b / rate_a)
    assert out["ppr_disparity"][1] == pytest.approx(0.5)
    assert out["ppr_disparity"][0] == pytest.approx(1.0), "the reference row is kept verbatim"
    assert "not_comparable" not in out
    assert "aequitas_reported_disparities" not in out
    assert not msgs, f"a comparable frame emitted {msgs}"


def test_the_aequitas_refusal_uses_the_analyzers_own_definition_of_comparable(stub_aequitas):
    """The same rule as the twin: raising min_group_size must move the boundary.

    One boundary out, and this is how the guard-above-the-dispatch was found:
    with NO group meeting min_group_size, `self.groups` is empty, `self.groups[0]`
    raised IndexError and the broad except returned
    `{'error': 'list index out of range'}` where the twin says "0 group(s) meet
    min_group_size=1000". An opaque error string is not one of the three states.
    """
    y, pred, groups = _frame("healthy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict = FairnessAnalyzer(y, pred, groups, min_group_size=1000)
        loose = FairnessAnalyzer(y, pred, groups, min_group_size=5)
    refused, msgs = _caught(strict.compare_with_aequitas)
    assert "error" not in refused, f"an opaque failure instead of a refusal: {refused}"
    assert "not_comparable" in refused
    assert refused["reference_group"] is None
    assert any("could not check" in m for m in msgs), "refused in silence"
    assert "not_comparable" not in _caught(loose.compare_with_aequitas)[0]
