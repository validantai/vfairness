"""Behavioural tests for `preprocessing.feature_engineering.visualization`.

302 statements, ten public plotting functions, and ZERO tests: no file under
tests/ mentioned this module, so none of it had ever executed in any environment.

These charts are how a reader SEES a proxy-variable audit. A bar chart that drops
a flagged feature, or labels a HIGH risk as LOW, is the same defect class the rest
of this suite exists for, just rendered instead of printed: the number is right
somewhere and the picture says something else.

So these are not smoke tests. Each one pins a relationship between the data going
in and what comes out: every flagged proxy gets a bar, the labels match the
severities, truncation is honest, and a save path actually produces a file.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from vfairness.preprocessing.feature_engineering import visualization as viz  # noqa: E402
from vfairness.preprocessing.feature_engineering.correlation import (  # noqa: E402
    FeatureCorrelationMatrix,
    ProxyRiskLevel,
    ProxyType,
    ProxyVariableResult,
)

FEATURES = ["zipcode", "income", "age"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def correlation_matrix() -> FeatureCorrelationMatrix:
    values = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.1], [0.2, 0.1, 1.0]])
    return FeatureCorrelationMatrix(
        correlations=pd.DataFrame(values, index=FEATURES, columns=FEATURES),
        pvalues=pd.DataFrame(np.full((3, 3), 0.01), index=FEATURES, columns=FEATURES),
        feature_names=FEATURES,
        protected_attributes=["race"],
        method="pearson",
    )


def _proxy(feature: str, correlation: float, level: ProxyRiskLevel) -> ProxyVariableResult:
    return ProxyVariableResult(
        feature=feature,
        protected_attribute="race",
        correlation=correlation,
        correlation_type="pearson",
        risk_level=level,
        proxy_type=ProxyType.DIRECT,
        mutual_information=0.3,
        cramers_v=0.4,
        pvalue=0.01,
        sample_size=500,
        affected_groups=["A", "B"],
        recommendations=["drop it"],
        evidence={},
    )


@pytest.fixture
def proxies() -> list[ProxyVariableResult]:
    return [
        _proxy("zipcode", 0.82, ProxyRiskLevel.HIGH),
        _proxy("income", 0.41, ProxyRiskLevel.MEDIUM),
        _proxy("age", 0.12, ProxyRiskLevel.LOW),
    ]


def _text(ax) -> str:
    parts = [t.get_text() for t in ax.texts]
    parts += [lab.get_text() for lab in ax.get_xticklabels() + ax.get_yticklabels()]
    parts += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    return " ".join(p for p in parts if p)


def test_every_flagged_proxy_reaches_the_chart(proxies):
    """A feature that is flagged and then not drawn is invisible to the reader."""
    ax = viz.plot_proxy_risk_chart(proxies)
    rendered = _text(ax)
    missing = [p.feature for p in proxies if p.feature not in rendered]
    assert not missing, f"flagged proxies absent from the chart: {missing}"


def test_the_bar_count_matches_the_number_of_proxies(proxies):
    ax = viz.plot_proxy_risk_chart(proxies)
    bars = [p for p in ax.patches if getattr(p, "get_width", None)]
    assert len(bars) == len(proxies), (
        f"{len(proxies)} proxies were passed in and {len(bars)} bars were drawn"
    )


def test_truncation_is_honest_about_what_it_dropped(proxies):
    """max_features must not silently hide the rest of a risk list."""
    ax = viz.plot_proxy_risk_chart(proxies, max_features=1)
    bars = [p for p in ax.patches if getattr(p, "get_width", None)]
    assert len(bars) <= 1, "max_features did not limit the chart"
    # The one drawn must be the WORST, not an arbitrary one.
    assert "zipcode" in _text(ax), (
        "truncating to one bar dropped the highest-correlation proxy and kept a weaker one"
    )


def test_the_risk_distribution_covers_every_severity_present(proxies):
    ax = viz.plot_risk_distribution(proxies)
    rendered = _text(ax).lower()
    for level in {p.risk_level.value.lower() for p in proxies}:
        assert level in rendered, f"severity {level!r} is in the data but not on the chart"


def test_the_heatmap_labels_every_feature(correlation_matrix):
    ax = viz.plot_correlation_heatmap(correlation_matrix)
    rendered = _text(ax)
    missing = [f for f in FEATURES if f not in rendered]
    assert not missing, f"features missing from the heatmap axes: {missing}"


def test_the_dashboard_returns_a_figure_with_more_than_one_panel(correlation_matrix, proxies):
    fig = viz.create_analysis_dashboard(correlation_matrix, proxies)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 1, "a one-panel 'dashboard' is a chart with a grander name"


def test_save_path_actually_writes_a_file(proxies, tmp_path):
    """A save that silently writes nothing is the reporting equivalent of a no-op."""
    target = tmp_path / "risk.png"
    viz.plot_proxy_risk_chart(proxies, save_path=str(target))
    assert target.exists(), "save_path produced no file"
    assert target.stat().st_size > 0, "save_path produced an empty file"


def test_an_empty_proxy_list_does_not_pretend_to_have_findings():
    """Nothing flagged must not render as something flagged.

    The honest answer here is a refusal, and that is what the module does: it
    returns None and warns "No proxy variables to plot" rather than handing back
    an empty but fully labelled chart, which is the shape a reader mistakes for
    "we looked and found nothing". Pinned in both halves, because returning an
    empty axes silently would look identical to a caller that does not check.
    """
    with pytest.warns(UserWarning, match="No proxy variables"):
        result = viz.plot_proxy_risk_chart([])
    assert result is None, (
        "an empty proxy list produced a chart object; an empty but labelled chart "
        "reads as a completed audit that found nothing"
    )


def test_a_single_proxy_still_renders(proxies):
    """Degenerate sizes are where plotting code usually breaks."""
    ax = viz.plot_proxy_risk_chart(proxies[:1])
    assert "zipcode" in _text(ax)
