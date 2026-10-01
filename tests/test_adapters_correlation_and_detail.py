"""Two adapters the empty-input sweep could not reach, for two different reasons.

**correlation_heatmap_to_svg had never worked with the type its own docstring
names.** It advertised ``FeatureCorrelationMatrix`` and then read
``correlation_matrix.features``, while that class exposes ``.feature_names``.
Every real matrix raised ``AttributeError`` inside the adapter, populated ones
included, so the failure was not an empty-input edge case at all: the advertised
path had never once produced a picture. The empty-input sweep did not catch it
because its fixture was a hand-written stand-in carrying ``.features``, which is
the shape the BUG expects. That is why the tests below build a genuine
``FeatureCorrelationMatrix`` from the class the library ships, and why the
stand-in shape is kept only as a compatibility control beside it.

**fairness_detailed_report_to_svg was the last report off the pattern.** A report
with no fairness score short-circuited to a bespoke 956-byte card built by string
formatting: it never reached ``render_svg``, so it carried no COULD NOT CHECK
wording on the canvas, no severity, and none of the machine-readable explanation
metadata every other chart emits. Anything parsing the SVG could not tell it from
an ordinary evaluation. Behind it sat a bare ``return ""`` on the render-failure
path, which hands a caller passing ``save_path`` a zero-byte file, no exception,
and a broken image with nothing to explain it: the worst of the three states,
because it cannot even be READ as could-not-check.

Everything is asserted on the RENDERED artifact, its drawn text and its
accessible description, never on an intermediate dict, because the artifact is
what leaves the building. The CONTROLS at the bottom are as load-bearing as the
guards: healthy input must still render its numbers, its badges and its verdict
word, and must never render the could-not-check panel.
"""

import json
import re
import warnings
from types import SimpleNamespace

import pytest

from vfairness.rendering.adapters_feature_engineering import correlation_heatmap_to_svg
from vfairness.rendering.adapters_post_processing import fairness_detailed_report_to_svg

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

    The accessible ``<title>``/``<desc>``/``<metadata>`` layer is stripped: it is
    asserted on separately through :func:`explanation_meta`, and leaving it in
    makes the canvas assertions pass for the wrong reason (the caption ends
    "(severity: MEDIUM)", which matches a search for a band label MEDIUM on a
    canvas that draws no band at all).
    """
    return " ".join(t.strip() for t in _TEXT_RE.findall(_A11Y_RE.sub("", svg)) if t.strip())


def explanation_meta(svg: str) -> dict:
    """The machine-readable explanation block injected by rendering.explain."""
    match = _META_RE.search(svg)
    assert match, "every rendered chart carries a vfairness-explanation metadata block"
    return json.loads(match.group(1))


def quietly(fn, *args, **kwargs):
    """Render without letting the deliberate could-not-check warnings fail the run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ── fixtures ────────────────────────────────────────────────────────────────


def real_matrix():
    """A genuine FeatureCorrelationMatrix, built exactly as the engine builds one.

    ``analyze_feature_correlations`` fills a DataFrame with ``index=features`` and
    ``columns=protected_attributes`` and hands it over with ``feature_names``.
    Nothing here is hand-shaped to suit the adapter; that is the whole point.
    """
    from vfairness.preprocessing.feature_engineering.correlation import (
        FeatureCorrelationMatrix,
    )

    features = ["zip_code", "income"]
    attrs = ["race", "gender"]
    return FeatureCorrelationMatrix(
        correlations=pd.DataFrame(
            [[0.81, 0.12], [0.22, 0.55]], index=features, columns=attrs, dtype=float
        ),
        pvalues=pd.DataFrame(
            [[0.001, 0.400], [0.300, 0.020]], index=features, columns=attrs, dtype=float
        ),
        feature_names=features,
        protected_attributes=attrs,
        method="pearson",
    )


def empty_real_matrix():
    from vfairness.preprocessing.feature_engineering.correlation import (
        FeatureCorrelationMatrix,
    )

    return FeatureCorrelationMatrix(
        correlations=pd.DataFrame(),
        pvalues=pd.DataFrame(),
        feature_names=[],
        protected_attributes=[],
        method="pearson",
    )


#: The pre-existing hand-written shape, exposing `.features` rather than
#: `.feature_names`. The gallery fixture still uses it, so the fix had to widen
#: what the adapter accepts, not swap one accepted shape for another.
LEGACY_SHAPE = SimpleNamespace(
    features=["zip_code", "income"],
    protected_attributes=["race", "gender"],
    correlations={
        "zip_code": {"race": 0.81, "gender": 0.12},
        "income": {"race": 0.22, "gender": 0.55},
    },
)


def healthy_report():
    """A detailed report that measured something: score, metric, group."""
    return {
        "assessment": {"fairness_score": 0.9},
        "metrics": {"demographic_parity_difference": {"value": 0.02, "threshold": 0.10}},
        "group_stats": {
            "male": {"size": 300, "positive_rate": 0.40, "tpr": 0.70, "fpr": 0.20},
            "female": {"size": 300, "positive_rate": 0.38, "tpr": 0.68, "fpr": 0.21},
        },
        "data_info": {"n_samples": 600},
    }


# ── 1. the adapter must work with the type it advertises ────────────────────


def test_a_real_feature_correlation_matrix_renders_at_all():
    """The advertised type must not raise inside the adapter.

    ``correlation_matrix.features`` raised AttributeError for every genuine
    matrix, so this adapter had never produced a picture from the object its
    docstring names. Populated input, not empty: the defect was never an
    edge case.
    """
    svg = correlation_heatmap_to_svg(real_matrix(), threshold=0.3)
    assert svg.lstrip().startswith("<svg"), "not a rendered SVG"
    assert svg.rstrip().endswith("</svg>"), "truncated SVG"


def test_a_real_matrix_puts_its_own_numbers_on_the_canvas():
    """Rendering without raising is not enough: the CELLS must be the real ones.

    A normaliser that silently read no value would still return a full-size SVG,
    with every cell defaulted to 0.00 and a green "No high correlations
    detected" pill over a matrix whose strongest proxy is 0.81.
    """
    text = drawn_text(correlation_heatmap_to_svg(real_matrix(), threshold=0.3))
    for label in ("zip_code", "income", "race", "gender"):
        assert label in text, f"{label!r} missing: the row/column labels did not reach the canvas"
    for value in ("0.81", "0.12", "0.22", "0.55"):
        assert value in text, f"correlation {value} missing from the heatmap cells"
    assert "No high correlations detected" not in text, (
        "a proxy-risk all-clear over a matrix carrying |r| = 0.81"
    )
    assert "2 potential proxy variables detected" in text


def test_a_real_matrix_and_the_legacy_shape_render_the_same_canvas():
    """Widening the accepted shapes must not change what either one draws."""
    from_real = drawn_text(correlation_heatmap_to_svg(real_matrix(), threshold=0.3))
    from_legacy = drawn_text(correlation_heatmap_to_svg(LEGACY_SHAPE, threshold=0.3))
    assert from_real == from_legacy


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            pd.DataFrame(
                [[0.81, 0.12], [0.22, 0.55]],
                index=["zip_code", "income"],
                columns=["race", "gender"],
                dtype=float,
            ),
            id="dataframe",
        ),
        pytest.param(
            {
                "zip_code": {"race": 0.81, "gender": 0.12},
                "income": {"race": 0.22, "gender": 0.55},
            },
            id="dict_of_dicts",
        ),
    ],
)
def test_the_other_documented_shapes_render_the_same_canvas(shape):
    """A bare DataFrame and a bare dict-of-dicts carry the same matrix."""
    assert drawn_text(correlation_heatmap_to_svg(shape, threshold=0.3)) == drawn_text(
        correlation_heatmap_to_svg(real_matrix(), threshold=0.3)
    )


# ── 2. an empty REAL matrix is could-not-check, not an all-clear ────────────


@pytest.mark.parametrize(
    "case",
    ["real", "legacy", "none"],
)
def test_an_empty_matrix_says_could_not_check_on_the_canvas(case):
    """A scan over zero features computed no correlation to be clear of.

    The empty-input sweep only ever exercised the ``legacy`` fixture here,
    because the ``real`` one could not be constructed without hitting the
    AttributeError first.
    """
    source = {
        "real": empty_real_matrix,
        "legacy": lambda: SimpleNamespace(correlations={}, features=[], protected_attributes=[]),
        "none": lambda: None,
    }[case]
    text = drawn_text(correlation_heatmap_to_svg(source()))
    assert "NOT CHECKED" in text, "canvas carries no could-not-check state"
    assert "not a pass and not a failure" in text
    assert "No high correlations detected" not in text, (
        "an emerald proxy-risk all-clear from a scan that computed no correlation"
    )
    assert "HIGH CORRELATIONS" not in text, "a count of zero is not a finding of zero"


def test_an_empty_real_matrix_description_is_not_more_confident_than_the_badge():
    meta = explanation_meta(correlation_heatmap_to_svg(empty_real_matrix()))
    assert str(meta["finding"]).upper().startswith(("COULD NOT CHECK", "NOT ASSESSABLE"))
    assert str(meta["severity"]).lower() == "medium", (
        "INFO and LOW read as a clean bill of health, HIGH and CRITICAL read as a "
        "finding, and neither happened here"
    )


# ── 3. the detailed report joins the pattern ────────────────────────────────


def test_an_empty_detailed_report_says_could_not_check_on_the_canvas():
    """``{}`` established nothing: no score, no metric, no group.

    It used to draw a bespoke card whose strongest wording was "N/A", with the
    metric count and the group count printed under their own headings.
    """
    text = drawn_text(quietly(fairness_detailed_report_to_svg, {}))
    assert "NOT CHECKED" in text
    assert "not a pass and not a failure" in text


def test_an_empty_detailed_report_withholds_every_count_and_verdict():
    text = drawn_text(quietly(fairness_detailed_report_to_svg, {}))
    survivors = [
        token
        for token in (
            "FAIRNESS SCORE",
            "/100",
            "FAIR",
            "MARGINAL",
            "UNFAIR",
            "GROUPS",
            "SAMPLES",
            "METRIC",
            "THRESHOLD",
            "STATUS",
            "0 metrics evaluated",
        )
        if token in text
    ]
    assert not survivors, (
        f"{survivors} still drawn on a report that measured nothing. Each is a score, "
        f"a heading over an empty table, or a count the input never established."
    )


def test_an_empty_detailed_report_is_machine_readable_as_could_not_check():
    """The bespoke card emitted no explanation metadata at all.

    An SVG consumer had no field to read, so the one report in the library that
    certified nothing was the one report that said nothing about it.
    """
    meta = explanation_meta(quietly(fairness_detailed_report_to_svg, {}))
    assert str(meta["finding"]).upper().startswith(("COULD NOT CHECK", "NOT ASSESSABLE"))
    assert str(meta["severity"]).lower() == "medium"
    assert "certifies nothing" in str(meta["recommendation"])


def test_a_missing_score_withholds_the_score_and_keeps_the_measured_rows():
    """Withholding is not deleting.

    The score is absent, so no number and no FAIR/MARGINAL/UNFAIR band may be
    drawn. The metric row and the group rows WERE measured, and dropping them
    (as the bespoke card did) is the opposite error: it destroys real findings
    to report a missing aggregate.
    """
    report = dict(healthy_report())
    report.pop("assessment")
    with pytest.warns(UserWarning, match="fairness_score"):
        svg = fairness_detailed_report_to_svg(report)
    text = drawn_text(svg)
    assert "NOT ASSESSABLE" in text, "the score badge does not state the third state"
    assert "/100" not in text, "a score was printed for a report that carries none"
    for band in ("FAIR", "MARGINAL", "UNFAIR"):
        assert band not in text.replace("FAIRNESS SCORE", ""), (
            f"{band!r} banded a score that was never produced"
        )
    # The template truncates the metric name at 28 characters, so match the
    # stem rather than the full title.
    assert "Demographic Parity" in text, "a measured metric row was dropped"
    assert "0.0200" in text, "the measured value on that row was dropped"
    assert "PASS" in text, "a measured verdict on that row was dropped"
    assert "male" in text and "female" in text, "measured group rows were dropped"
    meta = explanation_meta(svg)
    assert str(meta["finding"]).upper().startswith("COULD NOT CHECK")


def test_a_render_failure_never_leaves_a_zero_byte_svg(tmp_path, monkeypatch):
    """The tier-D failure, checked on the FILESYSTEM rather than the return value.

    ``open(save_path, "w")`` truncates before it writes, so the old ``return ""``
    left a zero-byte file on disk, raised nothing, and gave the reader a broken
    image with no way to tell a failed render from a failed model.
    """
    from vfairness.rendering import adapters_post_processing as ap

    def boom(_template, _data):
        raise RuntimeError("template exploded")

    monkeypatch.setattr(ap, "render_svg", boom)
    target = tmp_path / "detailed.svg"
    target.write_text("<svg>previous run</svg>")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = ap.fairness_detailed_report_to_svg(healthy_report(), save_path=str(target))

    assert svg, "a failed render returned an empty string"
    assert target.exists() and target.stat().st_size > 0, "save_path wrote a 0-byte SVG"
    assert target.read_text() == svg, "the file on disk is not what the caller was returned"
    on_disk = target.read_text()
    assert on_disk.lstrip().startswith("<svg") and on_disk.rstrip().endswith("</svg>")
    text = drawn_text(on_disk)
    assert "NOT CHECKED" in text
    assert "not a pass and not a failure" in text
    assert "90" not in text, "a score reached a canvas the template never drew"


def test_a_render_failure_card_survives_a_title_that_needs_escaping():
    """The fallback is built by string formatting, so nothing autoescapes it.

    A report titled ``A & B`` produced markup no XML parser would accept, which
    turns a could-not-check artifact back into an unreadable one.
    """
    from xml.etree import ElementTree

    from vfairness.rendering import adapters_post_processing as ap

    svg = ap._generate_render_failure_detailed_report_svg(
        {"title": "Fairness <A & B>", "timestamp": "2026-01-01 00:00"},
        RuntimeError("boom"),
    )
    ElementTree.fromstring(svg)  # raises ParseError on unescaped markup


# ── CONTROLS: healthy input must be untouched ───────────────────────────────


def test_control_healthy_detailed_report_still_states_its_verdict():
    """A fix that silenced real verdicts too would be a different bug."""
    text = drawn_text(fairness_detailed_report_to_svg(healthy_report()))
    assert "NOT CHECKED" not in text, "could-not-check panel drawn over a measured report"
    assert "NOT ASSESSABLE" not in text
    assert "90" in text and "/100" in text, "the measured score was withheld"
    assert "FAIR" in text, "the measured verdict word was withheld"
    assert "PASS" in text
    assert "600" in text, "the sample count was withheld"
    meta = explanation_meta(fairness_detailed_report_to_svg(healthy_report()))
    assert not str(meta["finding"]).upper().startswith("COULD NOT CHECK")
    assert str(meta["severity"]).lower() == "low"


def test_control_healthy_heatmap_still_states_its_finding():
    svg = correlation_heatmap_to_svg(real_matrix(), threshold=0.3)
    text = drawn_text(svg)
    assert "NOT CHECKED" not in text, "could-not-check panel drawn over a real matrix"
    assert "HIGH CORRELATIONS" in text and "FEATURES" in text
    assert "2 potential proxy variables detected" in text
    assert str(explanation_meta(svg)["severity"]).lower() != "medium" or "0.81" in text
