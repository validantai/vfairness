"""fairness_detailed_report_to_svg against the shape the engine actually emits.

Audit finding #8 had two halves.

**The named half:** the executive dashboard rendered
"Disparate Impact Ratio | 0.0000 | <= 0.80 | PASS" - a green verdict for a
selection ratio of zero, and a printed operator that states the four-fifths rule
as a ceiling when it is a floor.

**The half that made the first one moot:** the adapter read a shape the engine
does not produce. It expected ``metrics[name]`` to be a dict with
``value``/``threshold``, group statistics under ``group_statistics`` and a
top-level ``n_samples``; ``FairnessAnalyzer.get_report()`` emits
``{name: float}``, ``group_stats``, ``thresholds_used`` and ``data_info``. A real
report therefore rendered "0 metrics evaluated", GROUPS 0 and SAMPLES N/A: an
empty table under a confident score. Grading the rows correctly is worth nothing
while no row ever reaches the page, so both halves are pinned here.

Fail closed throughout: a metric with no usable value, or no threshold to
compare it against, is COULD NOT CHECK. It is never graded against an invented
bound, because a metric nobody graded, reported as a metric that passed, is the
same false certificate in a different costume.
"""

import re
import warnings

import numpy as np
import pytest

from vfairness.rendering import adapters_post_processing as ad

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _texts(svg: str) -> list[str]:
    """Every non-empty text node in the rendered SVG, in document order."""
    return [t.strip() for t in re.findall(r">([^<>]+)<", svg) if t.strip()]


def _render(report) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ad.fairness_detailed_report_to_svg(report)


def _template_data(monkeypatch, report) -> dict:
    """Capture what the adapter hands the template, without rendering."""
    captured = {}

    def fake_render(_template, data):
        captured.update(data)
        return "<svg/>"

    monkeypatch.setattr(ad, "render_svg", fake_render)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad.fairness_detailed_report_to_svg(report)
    return captured


def _real_report():
    """A genuine FairnessAnalyzer.get_report() with a large, real disparity."""
    from vfairness import FairnessAnalyzer

    rng = np.random.default_rng(7)
    n = 600
    groups = np.array(["male" if i % 2 else "female" for i in range(n)], dtype=object)
    y_true = rng.integers(0, 2, n)
    y_pred = np.where(groups == "male", rng.random(n) < 0.6, rng.random(n) < 0.15).astype(int)
    return FairnessAnalyzer(y_true, y_pred, groups).get_report()


# --------------------------------------------------------------------------
# The adapter must consume the shape the engine actually emits
# --------------------------------------------------------------------------


def test_real_report_metrics_reach_the_metrics_table():
    report = _real_report()
    assert isinstance(next(iter(report["metrics"].values())), float), (
        "the engine's report shape changed; this pin is now testing the wrong thing"
    )
    svg = _render(report)
    assert "0 metrics evaluated" not in svg
    assert f"{len(report['metrics'])} metrics evaluated" in svg
    # every metric the engine computed is named on the canvas
    for name in report["metrics"]:
        assert name.replace("_", " ").title()[:28] in svg, f"{name} never reached the table"


def test_real_report_group_statistics_reach_the_groups_table():
    report = _real_report()
    svg = _render(report)
    texts = _texts(svg)
    assert "female" in texts and "male" in texts, "group_stats never reached the page"
    groups_index = texts.index("GROUPS")
    assert texts[groups_index + 1] == "2", "GROUPS still counted zero on a real report"


def test_real_report_sample_count_is_not_na():
    report = _real_report()
    svg = _render(report)
    texts = _texts(svg)
    samples_index = texts.index("SAMPLES")
    assert texts[samples_index + 1] == "600", "SAMPLES read from a key the engine never emits"


def test_real_report_ratio_row_fails_and_prints_the_floor():
    """demographic_parity_ratio 0.28 against the four-fifths rule is a failure."""
    report = _real_report()
    ratio = report["metrics"]["demographic_parity_ratio"]
    assert ratio < 0.8, "fixture no longer produces a failing ratio"
    svg = _render(report)
    assert "≥ 0.80" in svg, "the four-fifths floor was not printed as a floor"
    assert "≤ 0.80" not in svg, "the four-fifths floor was printed as a ceiling"


def test_real_regression_report_also_renders():
    """The regression report uses the same keys and must not come out empty."""
    from vfairness import FairnessAnalyzer

    rng = np.random.default_rng(3)
    n = 400
    groups = np.array(["a" if i % 2 else "b" for i in range(n)], dtype=object)
    y_true = rng.random(n) * 10
    y_pred = y_true + np.where(groups == "a", 1.5, 0.0) + rng.random(n)
    report = FairnessAnalyzer(y_true, y_pred, groups, task_type="regression").get_report()
    svg = _render(report)
    assert "0 metrics evaluated" not in svg
    assert "Mae Parity Difference" in svg


# --------------------------------------------------------------------------
# Direction: the verdict and the printed operator, on the real shape
# --------------------------------------------------------------------------


def _flat(metric, value, threshold, score=0.9):
    """A report in the engine's own shape, with one metric."""
    return {
        "task_type": "classification",
        "metrics": {metric: value},
        "thresholds_used": {metric: threshold},
        "group_stats": {"f": {"size": 300, "positive_rate": 0.0, "tpr": 0.0, "fpr": 0.0}},
        "assessment": {"fairness_score": score},
        "data_info": {"n_samples": 300},
    }


def test_selection_ratio_of_zero_is_not_a_pass():
    """The named defect, on the shape the engine emits."""
    svg = _render(_flat("disparate_impact_ratio", 0.0, 0.8))
    assert ">PASS<" not in svg
    assert ">FAIL<" in svg
    assert "≥ 0.80" in svg
    assert "≤ 0.80" not in svg


def test_perfect_parity_ratio_is_a_pass():
    """Over-correction control: the ratio family must still be able to pass."""
    svg = _render(_flat("disparate_impact_ratio", 1.0, 0.8))
    assert ">PASS<" in svg
    assert ">FAIL<" not in svg
    assert "≥ 0.80" in svg


def test_difference_metric_keeps_its_ceiling():
    svg = _render(_flat("demographic_parity_difference", 0.04, 0.10))
    assert ">PASS<" in svg
    assert "≤ 0.10" in svg
    assert "≥ 0.10" not in svg


def test_adapter_states_the_direction_rather_than_leaving_it_to_be_inferred(monkeypatch):
    """The operator must come from the metric's real direction, not from which
    inequality happens to hold in this row."""
    data = _template_data(monkeypatch, _flat("disparate_impact_ratio", 0.0, 0.8))
    row = data["metrics"][0]
    assert row["direction"] == "higher_is_better"
    assert row["passed"] is False

    data = _template_data(monkeypatch, _flat("demographic_parity_difference", 0.04, 0.10))
    assert data["metrics"][0]["direction"] == "lower_is_better"


# --------------------------------------------------------------------------
# Fail closed: never grade against an invented bound or an invented value
# --------------------------------------------------------------------------


def test_metric_with_no_threshold_is_could_not_check(monkeypatch):
    """A flat metric with no entry in thresholds_used used to be graded against
    a hardcoded 0.1, which turns a 0.28 four-fifths ratio into a PASS."""
    report = {
        "metrics": {"disparate_impact_ratio": 0.28},
        "assessment": {"fairness_score": 0.9},
    }
    with pytest.warns(UserWarning, match="threshold"):
        svg = ad.fairness_detailed_report_to_svg(report)
    assert ">PASS<" not in svg
    assert "NO DATA" in svg
    assert "COULD NOT CHECK" in svg

    data = _template_data(monkeypatch, report)
    assert data["metrics"][0]["passed"] is None
    assert data["metrics"][0]["state"] == "could_not_check"


def test_metric_dict_without_a_value_is_could_not_check(monkeypatch):
    """`metric_info.get("value", 0)` invented a zero, and zero passes every
    ceiling. An absent value is not a measurement."""
    report = {
        "metrics": {"demographic_parity_difference": {"threshold": 0.1}},
        "assessment": {"fairness_score": 0.9},
    }
    with pytest.warns(UserWarning):
        svg = ad.fairness_detailed_report_to_svg(report)
    assert ">PASS<" not in svg
    assert "NO DATA" in svg

    data = _template_data(monkeypatch, report)
    assert data["metrics"][0]["passed"] is None


def test_nan_metric_value_is_could_not_check():
    svg = _render(_flat("demographic_parity_difference", float("nan"), 0.1))
    assert ">PASS<" not in svg
    assert "NO DATA" in svg


def test_unknown_direction_is_still_could_not_check():
    with pytest.warns(UserWarning, match="no known better-direction"):
        svg = ad.fairness_detailed_report_to_svg(_flat("mystery_house_metric", 0.05, 0.1))
    assert ">PASS<" not in svg
    assert "NO DATA" in svg


# --------------------------------------------------------------------------
# Over-correction control: the older hand-built shape still renders
# --------------------------------------------------------------------------


def _legacy(metric, value, threshold, score=0.9):
    return {
        "title": "Fairness Analysis Report",
        "metrics": {metric: {"value": value, "threshold": threshold}},
        "group_statistics": {
            "female": {"size": 300, "positive_rate": 0.0, "tpr": 0.0, "fpr": 0.0},
            "male": {"size": 300, "positive_rate": 0.54, "tpr": 1.0, "fpr": 0.0},
        },
        "n_samples": 600,
        "assessment": {"fairness_score": score},
    }


def test_legacy_dict_shape_still_renders_its_metrics():
    svg = _render(_legacy("demographic_parity_difference", 0.04, 0.10))
    assert "1 metrics evaluated" in svg
    assert ">PASS<" in svg
    assert "≤ 0.10" in svg


def test_legacy_group_statistics_key_still_populates_the_groups_table():
    svg = _render(_legacy("demographic_parity_difference", 0.04, 0.10))
    texts = _texts(svg)
    assert "female" in texts and "male" in texts
    assert texts[texts.index("GROUPS") + 1] == "2"
    assert texts[texts.index("SAMPLES") + 1] == "600"


def test_legacy_ratio_row_is_graded_in_its_own_direction():
    assert ">PASS<" not in _render(_legacy("disparate_impact_ratio", 0.0, 0.8))
    assert ">PASS<" in _render(_legacy("disparate_impact_ratio", 1.0, 0.8))
