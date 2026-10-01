"""Four more adapters answered "no data" with a fabricated finding.

Wave 8 closed this for the three reporting adapters (report_card,
hierarchical_gate, reporting_dashboard). These four fell into exactly the same
hole, reached by exactly the same route: a module-level ``_demo()`` fixture
dispatched as a RUNTIME FALLBACK when the caller supplied nothing, so the
output of a call that measured nothing was indistinguishable from the output of
a real evaluation.

* ``auto_discovery_to_svg()`` rendered a scan of a dataset that was never
  opened: three named candidate attributes with confidences, four violations of
  which two were high severity, six named intersectional groups with sample
  sizes, and "2 high-severity violations, immediate review needed".
* ``data_validation_to_svg(None)`` rendered a FAIL verdict with five issues,
  "1 critical, 2 warnings, 2 info", "Dataset contains 5,230 records across 3
  groups", "Feature 'zipcode' correlates 0.72 with protected attribute" and a
  "15% outcome disparity".
* ``ranking_fairness_to_svg()`` rendered three metrics over four groups, two
  PASS badges, one FAIL, a "2/3 metrics pass" rate and "failing: NDKL".
* ``regression_fairness_to_svg()`` rendered four disparity metrics over three
  groups, a "2/4 metrics pass" rate, a BIASED headline and per-group MAE, RMSE
  and R-squared figures.

An SVG is an export format. It leaves the building, an auditor reads it, and it
outlives the version of the library that produced it, so a fabricated finding is
the most damaging artifact this package can emit. A fabricated FAIL is not the
safe direction either: it names groups, features and correlations that do not
exist, and it gets acted on.

The rule these tests pin: no adapter may render a verdict, a score, a rate, a
count or a reassuring phrase that the supplied data did not establish. Missing
input is COULD NOT CHECK, stated ON THE CANVAS, never only in the accessible
``<desc>``. Three states, never two, and never a demo fixture at runtime.

The demo fixtures still exist, because the gallery needs them, but they are
reachable only through an explicit ``example=True`` and every one of them is
watermarked EXAMPLE on the canvas and in the ``<desc>``.

The asymmetries are asserted at the bottom, because a third state that swallows
the other two is its own defect: a REAL violation must still be listed, a REAL
breach must still print its FAIL badge, and a REAL all-clear must still print
its all-clear.
"""

import re

import pytest

pytest.importorskip("jinja2", reason="SVG rendering needs jinja2")

from vfairness.rendering.adapters_discovery import auto_discovery_to_svg  # noqa: E402
from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg  # noqa: E402
from vfairness.rendering.adapters_regression import regression_fairness_to_svg  # noqa: E402
from vfairness.rendering.adapters_validation import data_validation_to_svg  # noqa: E402

# Badge and cell labels that ARE a verdict. Matched as WHOLE drawn text nodes,
# never as substrings of prose: a could-not-check canvas has to be free to write
# "this report is not a pass and not a failure", and a substring test would read
# that sentence as the verdict it denies.
VERDICT_LABELS = frozenset(
    {
        "PASS",
        "FAIL",
        "WARN",
        "UNFAIR",
        "FAIR",
        "BIASED",
        "EQUITABLE",
        "HIGH",
        "MEDIUM",
        "LOW",
        "CRITICAL",
    }
)

# A drawn "2/3" is a measurement on the canvas. So is a bare number in a badge.
RATE_RE = re.compile(r"^\d+\s*/\s*\d+$")

_DESC_RE = re.compile(r"<desc>(.*?)</desc>", re.S)
_TEXT_RE = re.compile(r">([^<>]+)<")


def _body(svg: str) -> str:
    """The drawn markup, with the accessible layer removed.

    A could-not-check state that is visible only to a screen reader is not
    visible. These helpers keep the two readings apart so a test cannot pass on
    the strength of the other one.
    """
    body = svg
    for tag in ("desc", "title", "metadata"):
        body = re.sub(rf"<{tag}\b.*?</{tag}>", " ", body, flags=re.S)
    return body


def canvas_nodes(svg: str) -> list:
    """Every drawn text node, whole, in document order."""
    return [t.strip() for t in _TEXT_RE.findall(_body(svg)) if t.strip()]


def canvas_text(svg: str) -> str:
    """All drawn text joined, for testing that a SENTENCE is present."""
    return " ".join(canvas_nodes(svg))


def desc_text(svg: str) -> str:
    m = _DESC_RE.search(svg)
    assert m, "every rendered chart carries an accessible <desc>"
    return m.group(1)


# The four adapters, called exactly as a caller with no data would call them.
NO_DATA_CASES = {
    "auto_discovery": lambda: auto_discovery_to_svg(),
    "data_validation": lambda: data_validation_to_svg(None),
    "ranking_fairness": lambda: ranking_fairness_to_svg(),
    "regression_fairness": lambda: regression_fairness_to_svg(),
}

EXAMPLE_CASES = {
    "auto_discovery": lambda: auto_discovery_to_svg(example=True),
    "data_validation": lambda: data_validation_to_svg(example=True),
    "ranking_fairness": lambda: ranking_fairness_to_svg(example=True),
    "regression_fairness": lambda: regression_fairness_to_svg(example=True),
}

# Strings that could only come from the fixture. Each one is asserted PRESENT in
# the example render first, so a tell that no longer matches anything cannot
# quietly turn this test into a no-op. Deliberately not words that also occur in
# the static per-chart explanation prose (the ranking concept text says "NDKL",
# the regression one says "Cohen's d"), which is why a first pass of this test
# went red against a correct render.
FIXTURE_TELLS = {
    "auto_discovery": ("age_bucket", "female × minority", "non-binary × white"),
    "data_validation": ("5,230", "zipcode", "8.3%"),
    "ranking_fairness": ("ATTENTION FAIRNESS", "Non-binary", "n=450"),
    "regression_fairness": ("0.823", "MEAN PRED DIFF", "Male vs Female"),
}


# No data in, no analysis out


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_renders_no_verdict_on_the_canvas(name):
    """The labels a reader sees may not be verdicts the data never gave."""
    found = sorted({n.upper() for n in canvas_nodes(NO_DATA_CASES[name]())} & VERDICT_LABELS)
    assert found == [], f"{name} rendered the verdict label(s) {found} from no input at all"


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_renders_no_rate_on_the_canvas(name):
    """No pass rate: 0/0 is not a measurement, and neither is 2/3 from a fixture."""
    rates = [n for n in canvas_nodes(NO_DATA_CASES[name]()) if RATE_RE.match(n)]
    assert rates == [], f"{name} drew the rate(s) {rates} over data it never had"


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_says_so_on_the_canvas(name):
    """Could-not-check must be legible to a sighted reader, not only in <desc>."""
    drawn = canvas_text(NO_DATA_CASES[name]()).upper()
    assert any(
        marker in drawn
        for marker in ("NOT CHECKED", "NOT SCANNED", "NOT ASSESSED", "NOT ASSESSABLE")
    ), f"{name} carries no could-not-check badge on the canvas"
    assert "NOTHING WAS" in drawn, f"{name} never states on the canvas that nothing was done"


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_says_so_in_the_accessible_description(name):
    """And the screen-reader sentence must tell the same story as the canvas."""
    desc = desc_text(NO_DATA_CASES[name]())
    # The marker must LEAD the finding, not appear somewhere inside it: the
    # finding is the first thing a screen reader speaks and the first thing a
    # truncating consumer keeps.
    finding = desc.split(": ", 1)[1] if ": " in desc else desc
    assert finding.upper().startswith("COULD NOT CHECK"), (
        f"{name} <desc> does not lead with the third state: {desc}"
    )


@pytest.mark.parametrize("name", sorted(NO_DATA_CASES))
def test_no_data_never_leaks_the_demo_fixture(name):
    """The fixture's own words leaking through is how the fabrication was spotted."""
    example = EXAMPLE_CASES[name]()
    missing = [tell for tell in FIXTURE_TELLS[name] if tell not in example]
    assert missing == [], f"{missing} no longer marks the {name} fixture; this test is a no-op"

    svg = NO_DATA_CASES[name]()
    leaked = [tell for tell in FIXTURE_TELLS[name] if tell in svg]
    assert leaked == [], f"{name} rendered demo fixture content {leaked} from no input"


def test_no_data_is_not_a_rejection_either():
    """Three states, never two: could-not-check must not collapse into a failure."""
    for name in ("data_validation", "regression_fairness"):
        nodes = {n.upper() for n in canvas_nodes(NO_DATA_CASES[name]())}
        assert "FAIL" not in nodes
        assert "BIASED" not in nodes


def test_no_data_validation_withholds_the_issue_tally():
    """ "0 critical, 0 warn, 0 info" is the same line a clean dataset prints."""
    nodes = canvas_nodes(data_validation_to_svg(None))
    # Matched on the SEPARATOR, not on the zeroes: whatever the adapter leaves in
    # those three keys, drawing the tally line at all asserts a count happened.
    # A version of this test that only looked for "0 critical" stayed green
    # against the tally rendering "None critical · None warn · None info".
    tally = [n for n in nodes if "critical ·" in n or "warn ·" in n]
    assert tally == [], f"the banner drew a severity tally over nothing counted: {tally}"
    assert "no issue was counted" in nodes


def test_no_data_discovery_says_which_parts_were_not_scanned():
    """Each empty panel names the scan that did not run, where a reader looks."""
    drawn = canvas_text(auto_discovery_to_svg())
    for phrase in (
        "No attribute scan was supplied",
        "No violation scan was supplied",
        "No group profile was supplied",
    ):
        assert phrase in drawn, f"the auto-discovery canvas never states: {phrase}"


def test_no_data_discovery_withholds_the_violation_count():
    """A green zero cannot answer both "found none" and "never looked"."""
    nodes = canvas_nodes(auto_discovery_to_svg())
    assert "NOT SCANNED" in nodes
    assert "0 high severity" not in nodes


def test_no_data_regression_withholds_the_pass_rate():
    """The rate sat directly under the NOT ASSESSABLE badge, where a reader looks."""
    nodes = canvas_nodes(regression_fairness_to_svg())
    assert "0/0" not in nodes
    assert "not measured" in nodes


def test_no_data_ranking_withholds_the_metric_count():
    """ "0 metrics, 0 groups" is a count, and a count reads as a measurement."""
    drawn = canvas_text(ranking_fairness_to_svg())
    assert "0 metrics · 0 groups" not in drawn
    assert "Nothing was assessed" in drawn


def test_no_data_and_empty_containers_agree():
    """None and an empty container both mean "nothing was measured" here.

    The fix routes the None path into the empty-container path rather than
    building a second could-not-check canvas, because two codepaths saying the
    same thing is how one of them drifts. This is what pins that.
    """
    assert ranking_fairness_to_svg() == ranking_fairness_to_svg([], {})
    assert regression_fairness_to_svg() == regression_fairness_to_svg({}, {}, [])


# The demo fixtures survive only behind an explicit argument, watermarked


@pytest.mark.parametrize("name", sorted(EXAMPLE_CASES))
def test_example_is_watermarked_on_the_canvas(name):
    nodes = canvas_nodes(EXAMPLE_CASES[name]())
    banner = [n for n in nodes if n.upper().startswith("EXAMPLE ONLY")]
    assert banner, f"{name} demo carries no EXAMPLE band"
    assert "NOT A REAL EVALUATION" in banner[0].upper()
    assert "EXAMPLE" in nodes, f"{name} demo carries no EXAMPLE watermark"


@pytest.mark.parametrize("name", sorted(EXAMPLE_CASES))
def test_example_is_marked_in_the_accessible_description(name):
    """The <desc> is the whole artifact for a screen-reader user."""
    desc = desc_text(EXAMPLE_CASES[name]()).upper()
    assert desc.split(": ", 1)[1].startswith("EXAMPLE ONLY"), (
        f"{name} <desc> reads as a measurement: {desc}"
    )


@pytest.mark.parametrize("name", sorted(EXAMPLE_CASES))
def test_example_and_no_data_are_different_renders(name):
    """The whole defect was that these two were the same artifact."""
    assert EXAMPLE_CASES[name]() != NO_DATA_CASES[name]()


def test_example_must_be_asked_for_by_keyword():
    """A positional argument must not be able to switch the demo on."""
    with pytest.raises(TypeError):
        auto_discovery_to_svg(None, None, None, True)
    with pytest.raises(TypeError):
        data_validation_to_svg(None, True)


# The asymmetries: could-not-check withdraws an all-clear, it never hides a finding


REAL_VIOLATION = {
    "attribute": "gender",
    "metric": "demographic_parity",
    "value": 0.15,
    "threshold": 0.10,
    "severity": "high",
}
REAL_GROUPS = [
    {
        "group": "male",
        "positive_rate": 0.45,
        "size": 1200,
        "relative_to_overall": 1.32,
        "severity": "high",
    },
    {
        "group": "female",
        "positive_rate": 0.22,
        "size": 620,
        "relative_to_overall": 0.65,
        "severity": "high",
    },
]


def test_a_real_violation_is_still_rendered():
    svg = auto_discovery_to_svg(
        [{"name": "gender", "confidence": 0.95}], [REAL_VIOLATION], REAL_GROUPS
    )
    nodes = canvas_nodes(svg)

    assert "HIGH" in nodes
    assert "1 high severity" in nodes
    assert "NOT SCANNED" not in nodes
    assert "EXAMPLE" not in nodes, "a real scan must not be watermarked"


def test_a_partly_supplied_scan_says_which_part_did_not_run():
    """The half that never ran must say so, beside the half that did.

    This is the identify_proxy_features trap in miniature: an empty panel under
    a heading reads as "we looked here and found nothing", and for two of these
    three panels nothing looked at all.
    """
    drawn = canvas_text(auto_discovery_to_svg(None, [REAL_VIOLATION], None))

    assert "No attribute scan was supplied" in drawn
    assert "No group profile was supplied" in drawn
    # and the part that DID run is still reported in full
    assert "demographic_parity" in drawn
    assert "No violation scan was supplied" not in drawn


def test_a_real_validation_failure_is_still_rendered():
    result = {
        "passed": False,
        "issues": [
            {
                "severity": "CRITICAL",
                "issue_type": "imbalanced_groups",
                "message": "Group 'age>65' has only 12 samples",
                "affected_groups": ["age>65"],
                "recommendation": "Collect more data",
            }
        ],
        "metrics": {"total_records": 4000},
    }
    nodes = canvas_nodes(data_validation_to_svg(result))

    assert "FAIL" in nodes
    assert "NOT CHECKED" not in nodes
    assert "1 critical · 0 warn · 0 info" in nodes


def test_a_real_validation_pass_is_still_rendered():
    """The other direction: the third state must not eat a genuine all-clear."""
    nodes = canvas_nodes(data_validation_to_svg({"passed": True, "issues": [], "metrics": {}}))

    assert "PASS" in nodes
    assert "NOT CHECKED" not in nodes


def test_a_real_ranking_breach_is_still_rendered():
    svg = ranking_fairness_to_svg(
        [{"metric_name": "ndkl", "value": 0.12, "threshold": 0.10}],
        {"Male": {"mean_exposure": 0.28, "count": 450}},
    )
    nodes = canvas_nodes(svg)

    assert "UNFAIR" in nodes
    assert "FAIL" in nodes
    assert "NOT ASSESSED" not in nodes


def test_a_real_regression_all_clear_is_still_rendered():
    """A measured all-clear is a result, and withholding it would be the mirror bug."""
    svg = regression_fairness_to_svg(
        {"Male": {"size": 100, "mae": 0.1, "rmse": 0.2, "r2": 0.8, "mean_residual": 0.0}},
        {"mae_parity": 0.02, "rmse_parity": 0.03},
    )
    nodes = canvas_nodes(svg)

    assert "EQUITABLE" in nodes
    assert "2/2" in nodes
    assert "NOT ASSESSABLE" not in nodes
    assert "not measured" not in nodes
