"""Zero and empty input must render COULD NOT CHECK, never a verdict.

"Nothing failed" and "nothing ran" must never render the same. A sentinel is not
a measurement. A count of zero is not a finding of zero.

An SVG is an export format: it leaves the building, an auditor reads it, and it
outlives the version that produced it. Every adapter covered here used to hand a
caller an artifact that could not be told apart from a real evaluation:

* ``transformation_comparison_to_svg({}, {})`` printed the verdict word
  "Minimal" beside AVG REDUCTION 0% and FEATURES IMPROVED 0 / 0, across zero
  features, at accessible severity HIGH.
* ``intersectional_disparity_to_svg({})`` printed a banded "DISPARITY: MEDIUM"
  and "0.0pp between N/A and N/A" across zero subgroups. A band is a verdict
  about a measured gap; N/A on both sides says the gap had no operands.
* ``intersectional_analysis_to_svg({}, "")`` printed MAX DISPARITY 0.000, the
  canvas for perfect parity across every intersection.
* ``correlation_matrix_to_svg`` and ``correlation_heatmap_to_svg`` printed the
  reassuring "No high correlations detected" for a scan over zero features.
* ``reweighting_comparison_to_svg({})`` asserted "an original disparity of 0.000
  across 0 methods" in its explanation and its accessible description.
* ``workflow_overview_to_svg()`` substituted vfairness's own reference
  integrations, ACTIVE badges and all, as though they were the caller's.
* Five more (alert timeline, live dashboard, drift report, temporal trends,
  accuracy-fairness trade-off) printed 0-of-0 counts with advisory prose at
  severity INFO or LOW: "0 of 0 windows raised alerts (0 clean)" is the canvas
  for a monitored period that stayed clean.
* ``method_comparison_to_svg([])`` and ``disparity_heatmap_to_svg({})`` returned
  a zero-length string, so a caller passing ``save_path`` wrote a 0-byte SVG and
  got no error at all: the worst of the three states, because the artifact
  cannot even be read as could-not-check.

Every test below asserts on the RENDERED artifact (its drawn text and its
accessible description), never on an intermediate dict, because the artifact is
what a reader receives. The CONTROLS at the bottom are as load-bearing as the
guards: a fix that silenced healthy verdicts too would be a different bug, so
healthy input must still render its number, its band and its verdict word, and
must never render the could-not-check panel.
"""

import json
import re
from types import SimpleNamespace

import pytest

from vfairness.rendering.adapters_fairness import disparity_heatmap_to_svg
from vfairness.rendering.adapters_feature_engineering import (
    correlation_heatmap_to_svg,
    correlation_matrix_to_svg,
    intersectional_analysis_to_svg,
    intersectional_disparity_to_svg,
    proxy_risk_to_svg,
    transformation_comparison_to_svg,
)
from vfairness.rendering.adapters_monitoring import (
    alert_timeline_to_svg,
    drift_report_to_svg,
    monitoring_dashboard_to_svg,
    temporal_analysis_to_svg,
)
from vfairness.rendering.adapters_post_processing import reweighting_comparison_to_svg
from vfairness.rendering.adapters_training import (
    method_comparison_to_svg,
    tradeoff_analysis_to_svg,
)
from vfairness.rendering.adapters_workflow import workflow_overview_to_svg

pd = pytest.importorskip("pandas")


# ── helpers ─────────────────────────────────────────────────────────────────

_TEXT_RE = re.compile(r">([^<>]+)<")
_META_RE = re.compile(
    r'<metadata id="vfairness-explanation"><!\[CDATA\[(.*?)\]\]></metadata>', re.S
)
_A11Y_RE = re.compile(
    r"<title>.*?</title>|<desc>.*?</desc>"
    r'|<metadata id="vfairness-explanation">.*?</metadata>',
    re.S,
)


def drawn_text(svg: str) -> str:
    """Everything a sighted reader sees on the canvas, as one string.

    Deliberately NOT the raw markup: a token that appears only inside an
    attribute (a colour, an id) is not drawn, and matching it would make these
    assertions pass or fail for the wrong reason.

    The injected ``<title>``/``<desc>``/``<metadata>`` block is stripped, for
    the same reason. It is the accessible layer, not the picture, it is asserted
    on separately through :func:`explanation_meta`, and leaving it in makes the
    canvas assertions lie in both directions: the caption ends "(severity:
    MEDIUM)", which matched a search for the disparity band label MEDIUM on a
    canvas that no longer draws one.
    """
    return " ".join(t.strip() for t in _TEXT_RE.findall(_A11Y_RE.sub("", svg)) if t.strip())


def explanation_meta(svg: str) -> dict:
    """The machine-readable explanation block injected by rendering.explain."""
    match = _META_RE.search(svg)
    assert match, "every rendered chart carries a vfairness-explanation metadata block"
    return json.loads(match.group(1))


# ── empty-input fixtures ────────────────────────────────────────────────────


class _EmptyCorrelationMatrix:
    correlations: dict = {}
    features: list = []
    protected_attributes: list = []


class _EmptyWindow:
    timestamp = None
    metrics: dict = {}
    alerts: dict = {}
    group_rates: dict = {}
    mmd_scores: dict = {}
    sample_count = 0
    any_alert = False


class _EmptyTemporal:
    def to_dataframe(self):
        return pd.DataFrame({"metric_name": []})


_EMPTY_DRIFT = SimpleNamespace(
    drift_detected=False,
    overall_drift_score=0.0,
    metric="demographic_parity_diff",
    scales={},
)


# Each case: the render, and the tokens that must NOT survive on the canvas
# because each one is a claim the input never established.
EMPTY_CASES = {
    "transformation_comparison": (
        lambda: transformation_comparison_to_svg({}, {}),
        ["Minimal", "Moderate", "Effective", "AVG REDUCTION", "FEATURES IMPROVED", "0 / 0"],
    ),
    "intersectional_disparity": (
        lambda: intersectional_disparity_to_svg({}),
        ["DISPARITY", "MEDIUM", "N/A", "0.0pp", "MOST ADVANTAGED", "MOST DISADVANTAGED"],
    ),
    "intersectional_analysis": (
        lambda: intersectional_analysis_to_svg({}, ""),
        ["MAX DISPARITY", "0.000", "Max intersectional disparity"],
    ),
    "correlation_matrix_none": (
        lambda: correlation_matrix_to_svg(None),
        ["No high correlations detected", "HIGH CORRELATIONS", "FEATURES"],
    ),
    "correlation_matrix_empty": (
        lambda: correlation_matrix_to_svg({}),
        ["No high correlations detected", "HIGH CORRELATIONS", "FEATURES"],
    ),
    "correlation_heatmap": (
        lambda: correlation_heatmap_to_svg(_EmptyCorrelationMatrix()),
        ["No high correlations detected", "HIGH CORRELATIONS", "FEATURES"],
    ),
    "proxy_risk": (
        lambda: proxy_risk_to_svg([]),
        ["LOW", "MEDIUM", "HIGH", "CRITICAL", "CORRELATION", "CONFIDENCE"],
    ),
    "reweighting_comparison": (
        lambda: reweighting_comparison_to_svg({}),
        ["BEST METHOD", "N/A", "original disparity of 0.000", "TRADE-OFF"],
    ),
    "workflow_overview": (
        lambda: workflow_overview_to_svg(),
        ["MLflow", "ACTIVE", "W&B", "pytest plugin", "Experiment"],
    ),
    "alert_timeline": (
        lambda: alert_timeline_to_svg([]),
        ["TOTAL EVENTS", "WITH ALERTS", "CLEAN", "0 of 0", "monitoring events"],
    ),
    "monitoring_dashboard": (
        lambda: monitoring_dashboard_to_svg(_EmptyWindow()),
        ["0 alerts", "metrics monitored", "Group Positive Rates", "VALUE"],
    ),
    "drift_report": (
        lambda: drift_report_to_svg(_EMPTY_DRIFT),
        ["STABLE", "DETECTED", "DRIFT STATUS", "overall score", "scales analysed"],
    ),
    "temporal_analysis": (
        lambda: temporal_analysis_to_svg(_EmptyTemporal()),
        ["temporal trend and pattern analysis", "DEGRADED", "STABLE"],
    ),
    "tradeoff_analysis": (
        lambda: tradeoff_analysis_to_svg({}),
        # "Pareto frontier" is deliberately absent from this list: the phrase
        # also appears in the chart's static concept paragraph, which explains
        # what the chart WOULD show and is not a claim about this run.
        ["Constraint satisfied", "Constraint violated", "Best fair", "Best accurate"],
    ),
    "method_comparison": (
        lambda: method_comparison_to_svg([]),
        ["PASS", "FAIL", "ACCURACY", "VIOLATION", "methods evaluated"],
    ),
    "disparity_heatmap": (
        lambda: disparity_heatmap_to_svg({}),
        [
            "Low disparity across groups",
            "Moderate disparity detected",
            "High disparity detected",
        ],
    ),
    "disparity_heatmap_groups_no_metric": (
        lambda: disparity_heatmap_to_svg({"group_stats": {"Male": {"positive_rate": 0.5}}}),
        [
            "Low disparity across groups",
            "Moderate disparity detected",
            "High disparity detected",
        ],
    ),
}


# ── the guards ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", sorted(EMPTY_CASES))
def test_empty_input_renders_a_real_artifact(case):
    """No adapter may answer empty input with an empty string.

    ``method_comparison_to_svg([])`` and ``disparity_heatmap_to_svg({})`` both
    returned "", so a caller passing ``save_path`` wrote a 0-byte file, saw no
    exception, and later opened a broken image with nothing to explain it.
    """
    svg = EMPTY_CASES[case][0]()
    assert svg, f"{case}: empty input returned an empty string, which writes a 0-byte SVG"
    assert svg.lstrip().startswith("<svg"), f"{case}: not a rendered SVG"
    assert svg.rstrip().endswith("</svg>"), f"{case}: truncated SVG"


@pytest.mark.parametrize("case", sorted(EMPTY_CASES))
def test_empty_input_says_not_checked_on_the_canvas(case):
    """The absence must be stated where a SIGHTED reader sees it.

    Saying it only in the accessible ``<desc>`` is not enough: the exported
    picture is what gets attached to a pull request and shown to a regulator.
    """
    text = drawn_text(EMPTY_CASES[case][0]())
    assert "NOT CHECKED" in text, f"{case}: canvas carries no could-not-check state"
    assert "not a pass and not a failure" in text, (
        f"{case}: canvas does not say that could-not-check is neither verdict"
    )


@pytest.mark.parametrize("case", sorted(EMPTY_CASES))
def test_empty_input_withholds_every_verdict_and_number(case):
    """Could-not-check withholds the numbers, it does not merely annotate them.

    A NOT CHECKED badge beside a 0-of-0 tile row is still a canvas a reader
    takes a number off.
    """
    text = drawn_text(EMPTY_CASES[case][0]())
    survivors = [tok for tok in EMPTY_CASES[case][1] if tok in text]
    assert not survivors, (
        f"{case}: {survivors} still drawn on a chart that measured nothing. "
        f"Each is a verdict, a score, a rate or a count the input never established."
    )


@pytest.mark.parametrize("case", sorted(EMPTY_CASES))
def test_empty_input_description_leads_with_could_not_check(case):
    """The accessible description is the whole artifact for a screen reader."""
    meta = explanation_meta(EMPTY_CASES[case][0]())
    finding = str(meta["finding"])
    assert finding.upper().startswith(("COULD NOT CHECK", "NOT ASSESSABLE")), (
        f"{case}: <desc> finding reads as a result, not as could-not-check: {finding[:120]}"
    )


@pytest.mark.parametrize("case", sorted(EMPTY_CASES))
def test_empty_input_is_never_graded_info_or_low(case):
    """Severity must not read as benign news, and must not read as a finding.

    This is the exact shape of the whole tier-C bug: five adapters reported
    0-of-0 counts with advisory prose at severity INFO or LOW, which is how a
    healthy chart is graded. An unsupported claim is not benign; nothing was
    measured to be wrong either, so it is not HIGH.
    """
    severity = str(explanation_meta(EMPTY_CASES[case][0]())["severity"]).lower()
    assert severity == "medium", (
        f"{case}: could-not-check graded {severity!r}; INFO and LOW read as a clean bill of "
        f"health, HIGH and CRITICAL read as a finding, and neither happened here"
    )


@pytest.mark.parametrize("case", sorted(EMPTY_CASES))
def test_empty_input_action_does_not_imply_a_measurement(case):
    """The curated per-chart action tells the reader to fix failing metrics.

    On a run that measured nothing that instruction is not merely useless, it
    asserts that a measurement happened.
    """
    meta = explanation_meta(EMPTY_CASES[case][0]())
    assert "certifies nothing" in str(meta["recommendation"]), (
        f"{case}: recommendation still assumes a verdict was produced"
    )


def test_save_path_never_writes_a_zero_byte_file(tmp_path):
    """The tier-D failure, checked on the FILESYSTEM rather than the return value.

    Reading the returned string proves the function; a caller uses ``save_path``
    and never sees the string at all.
    """
    for case, (render, _) in sorted(EMPTY_CASES.items()):
        target = tmp_path / f"{case}.svg"
        render_with_path = {
            "method_comparison": lambda p: method_comparison_to_svg([], save_path=str(p)),
            "disparity_heatmap": lambda p: disparity_heatmap_to_svg({}, save_path=str(p)),
        }.get(case)
        if render_with_path is None:
            continue
        render_with_path(target)
        assert target.exists(), f"{case}: save_path wrote no file"
        assert target.stat().st_size > 0, f"{case}: save_path wrote a 0-byte SVG"
        assert "NOT CHECKED" in drawn_text(target.read_text()), (
            f"{case}: the file on disk does not carry the could-not-check state"
        )


def test_workflow_overview_fixture_needs_an_explicit_example_flag():
    """The built-in reference integrations may not be reached implicitly.

    They are vfairness's own, not a reading of the caller's pipeline, and an
    unmarked render of them is indistinguishable from one built from a real
    ``integrations=`` list.
    """
    implicit = drawn_text(workflow_overview_to_svg())
    assert "MLflow" not in implicit
    assert "NOT CHECKED" in implicit

    explicit = drawn_text(workflow_overview_to_svg(example=True))
    assert "MLflow" in explicit, "example=True must still render the reference set"
    assert "EXAMPLE ONLY" in explicit, "an example render must be watermarked on the canvas"
    finding = str(explanation_meta(workflow_overview_to_svg(example=True))["finding"])
    assert finding.startswith("EXAMPLE ONLY"), (
        "the example marker must LEAD the accessible description, so it survives truncation"
    )


def test_workflow_overview_does_not_backfill_unnamed_sections():
    """Naming one section must not fill the other two from the fixture.

    That is the same substitution at a finer grain, and harder to spot because
    two thirds of the page would then be the caller's real data.
    """
    text = drawn_text(
        workflow_overview_to_svg(
            integrations=[{"name": "OurTracker", "description": "in-house run log"}]
        )
    )
    assert "OurTracker" in text
    assert "MLflow" not in text
    assert "pytest plugin" not in text
    assert "PR template" not in text


# ── CONTROLS: healthy input must be untouched ───────────────────────────────
#
# These are not decoration. A "fix" that withheld verdicts from real data too
# would satisfy every guard above while destroying the product, so each control
# asserts the measured number, the band and the verdict word are still drawn,
# and that the could-not-check panel is NOT.

_HEALTHY_PROXY_VARS = [
    SimpleNamespace(
        feature="zip_code",
        protected_attribute="race",
        correlation=0.72,
        risk_level=SimpleNamespace(value="high"),
        confidence_score=0.91,
    ),
    SimpleNamespace(
        feature="income",
        protected_attribute="age",
        correlation=0.45,
        risk_level=SimpleNamespace(value="medium"),
        confidence_score=0.78,
    ),
]

_HEALTHY_INTERSECTIONAL = {
    "privilegedGroup": {"group": "Asian Male", "positiveRate": 0.71, "size": 150},
    "disadvantagedGroup": {"group": "Black Female", "positiveRate": 0.34, "size": 195},
    "allGroups": [
        {"group": "Asian Male", "positiveRate": 0.71, "size": 150, "severity": "low"},
        {"group": "Black Female", "positiveRate": 0.34, "size": 195, "severity": "high"},
    ],
    "maxDisparity": 0.37,
    "disparitySeverity": "high",
    "insights": ["Black Female shows a 37pp approval gap"],
    "findings": [{"severity": "high"}],
}

_HEALTHY_METHODS = [
    SimpleNamespace(
        method_name="Baseline",
        accuracy=0.852,
        fairness_violation=0.185,
        constraint_satisfied=False,
        training_time=2.1,
    ),
    SimpleNamespace(
        method_name="Constrained",
        accuracy=0.828,
        fairness_violation=0.041,
        constraint_satisfied=True,
        training_time=5.2,
    ),
]

_HEALTHY_WINDOW = SimpleNamespace(
    timestamp=None,
    metrics={"demographic_parity_diff": 0.14},
    alerts={"demographic_parity_diff": True},
    any_alert=True,
    sample_count=1000,
    group_rates={"gender": {"Male": 0.72, "Female": 0.58}},
    mmd_scores={"gender": 0.03},
)

_HEALTHY_DRIFT = SimpleNamespace(
    drift_detected=True,
    overall_drift_score=0.45,
    metric="demographic_parity_diff",
    scales={
        "medium_term": SimpleNamespace(
            drift_score=0.45,
            ks_statistic=0.22,
            p_value=0.04,
            mean_shift=0.05,
            reference_mean=0.08,
            current_mean=0.13,
            drift_detected=True,
        )
    },
)

CONTROL_CASES = {
    # name: (render, tokens that MUST still be drawn)
    "transformation_comparison": (
        lambda: transformation_comparison_to_svg({"zip_code": 0.72}, {"zip_code": 0.28}),
        ["AVG REDUCTION", "FEATURES IMPROVED", "Effective"],
    ),
    "intersectional_disparity": (
        lambda: intersectional_disparity_to_svg(_HEALTHY_INTERSECTIONAL),
        ["DISPARITY", "HIGH", "Black Female"],
    ),
    "intersectional_analysis": (
        lambda: intersectional_analysis_to_svg(
            {"matrix": {"White": {"Male": 0.65}, "Black": {"Male": 0.42}}}, "approval"
        ),
        ["MAX DISPARITY", "0.230"],
    ),
    "correlation_matrix": (
        lambda: correlation_matrix_to_svg({"a": {"a": 1.0, "b": 0.8}, "b": {"a": 0.8, "b": 1.0}}),
        ["HIGH CORRELATIONS", "FEATURES"],
    ),
    "correlation_heatmap": (
        lambda: correlation_heatmap_to_svg(
            SimpleNamespace(
                features=["zip_code"],
                protected_attributes=["race"],
                correlations={"zip_code": {"race": 0.72}},
            )
        ),
        ["HIGH CORRELATIONS", "potential proxy variable"],
    ),
    "proxy_risk": (
        lambda: proxy_risk_to_svg(_HEALTHY_PROXY_VARS),
        ["zip_code", "HIGH", "CORRELATION"],
    ),
    "reweighting_comparison": (
        lambda: reweighting_comparison_to_svg(
            {
                "method_results": [
                    {
                        "method": "group_specific",
                        "original_fairness": {"demographic_parity_diff": 0.18},
                        "adjusted_fairness": {"demographic_parity_diff": 0.04},
                        "original_performance": {"accuracy": 0.85},
                        "adjusted_performance": {"accuracy": 0.82},
                        "calibration_metrics": {"ece_change": 0.02},
                        "trade_off_score": 0.85,
                    }
                ],
                "best_method": "group_specific",
            }
        ),
        ["BEST METHOD", "group_specific", "TRADE-OFF"],
    ),
    "workflow_overview": (
        lambda: workflow_overview_to_svg(
            integrations=[{"name": "MLflow", "description": "run log", "status": "active"}]
        ),
        ["MLflow", "ACTIVE", "Experiment"],
    ),
    "alert_timeline": (
        lambda: alert_timeline_to_svg(
            [
                SimpleNamespace(
                    timestamp=None,
                    alerts={"demographic_parity_diff": True},
                    sample_count=1000,
                )
            ]
        ),
        ["TOTAL EVENTS", "WITH ALERTS", "CLEAN", "ALERT"],
    ),
    "monitoring_dashboard": (
        lambda: monitoring_dashboard_to_svg(_HEALTHY_WINDOW),
        ["demographic_parity_diff", "ALERT", "metrics monitored"],
    ),
    "drift_report": (
        lambda: drift_report_to_svg(_HEALTHY_DRIFT),
        ["DRIFT STATUS", "DETECTED", "overall score"],
    ),
    "tradeoff_analysis": (
        lambda: tradeoff_analysis_to_svg(
            {
                "all_results": [
                    {"accuracy": 0.85, "violation": 0.18, "lambda": 0.0, "satisfied": False},
                    {"accuracy": 0.82, "violation": 0.04, "lambda": 0.5, "satisfied": True},
                ],
                "pareto_frontier": [{"accuracy": 0.82, "violation": 0.04}],
                "best_fair": {"accuracy": 0.82, "violation": 0.04},
            }
        ),
        ["Constraint satisfied", "Pareto frontier", "Best fair"],
    ),
    "method_comparison": (
        lambda: method_comparison_to_svg(_HEALTHY_METHODS),
        ["PASS", "FAIL", "Baseline", "Constrained"],
    ),
    "disparity_heatmap": (
        lambda: disparity_heatmap_to_svg(
            {
                "protected_attribute": "gender",
                "metrics": {"demographic_parity_difference": 0.14},
                "group_stats": {
                    "Male": {"size": 550, "positive_rate": 0.72},
                    "Female": {"size": 450, "positive_rate": 0.58},
                },
            }
        ),
        ["Moderate disparity detected", "Male", "Female"],
    ),
}


@pytest.mark.parametrize("case", sorted(CONTROL_CASES))
def test_healthy_input_still_renders_its_verdict(case):
    """CONTROL. Real data keeps every number, band and verdict word it had."""
    text = drawn_text(CONTROL_CASES[case][0]())
    missing = [tok for tok in CONTROL_CASES[case][1] if tok not in text]
    assert not missing, (
        f"{case}: healthy input lost {missing} from the canvas. The could-not-check "
        f"branch must fire on ABSENT input only, never on a measured result."
    )


@pytest.mark.parametrize("case", sorted(CONTROL_CASES))
def test_healthy_input_never_renders_the_could_not_check_panel(case):
    """CONTROL. The third state must not leak onto a chart that measured things."""
    text = drawn_text(CONTROL_CASES[case][0]())
    assert "NOT CHECKED" not in text, f"{case}: healthy render shows the could-not-check badge"
    meta = explanation_meta(CONTROL_CASES[case][0]())
    assert not str(meta["finding"]).upper().startswith("COULD NOT CHECK"), (
        f"{case}: healthy render's <desc> claims nothing was assessed"
    )


def test_healthy_disparity_heatmap_keeps_its_rates_when_it_cannot_band():
    """CONTROL for the narrower of the two heatmap states.

    A grid that HAS per-group rates but cannot band a disparity keeps its cells:
    the rates are real, and only the colour gradient and the verdict band are
    withdrawn. Collapsing that case to the empty panel would throw away measured
    data, which is the mirror-image failure of inventing it.
    """
    svg = disparity_heatmap_to_svg(
        {
            "protected_attribute": "gender",
            "metrics": {"some_unknown_metric": 0.4},
            "group_stats": {
                "Male": {"size": 550, "positive_rate": 0.72},
                "Female": {"size": 450, "positive_rate": 0.58},
            },
        }
    )
    text = drawn_text(svg)
    assert "Male" in text and "Female" in text, "measured per-group rates were discarded"
    assert "0.72" in text and "0.58" in text, "the rates themselves must stay on the grid"
    assert "Low disparity across groups" not in text, "an unbanded grid must not print an all-clear"
