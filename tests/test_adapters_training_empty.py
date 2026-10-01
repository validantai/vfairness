"""A constraint nobody evaluated may never be drawn as a constraint that FAILED.

Given an empty ``FairnessTrainingReport``, both ``training_analysis_report_to_svg``
and ``training_report_to_svg`` used to render

    Accuracy 0.0000 | Violation 0.0000 | Constraint VIOLATED
    Base rate disparity: 0.000

with the accessible description graded HIGH. Every number came from a ``.get``
default, and VIOLATED is simply what the ``False`` default of
``constraint_satisfied`` paints. That is a fabricated FAILURE: the mirror image
of the fabricated all-clears closed in the other waves, and just as wrong. It
sends a reader to investigate a violation that does not exist, HIGH is the grade
a real breach gets, and the first time someone works out why, the tool is
discredited. ``training_report_to_svg`` additionally raised AttributeError when
*recommendation* was None, so the two renders of one report disagreed: one
crashed, the other invented.

A count of zero is not a finding of zero. A sentinel is not a measurement. An
absence and a measured zero are different claims and must render differently,
which is why the CONTROLS below are as load-bearing as the guards: a baseline
that really came out 0.0000 keeps its numbers, its verdict and its colour.

Every assertion here reads the RENDERED artifact, its drawn text and its
accessible description, never an intermediate dict, because the artifact is what
a reader receives and what leaves the building.
"""

import json
import re
import warnings
from types import SimpleNamespace

import pytest

from vfairness.rendering import adapters_training
from vfairness.rendering.adapters_training import (
    training_analysis_report_to_svg,
    training_report_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")


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

    Deliberately not the raw markup: a token that appears only inside an
    attribute (a colour, an id) is not drawn, and matching it would make these
    assertions pass or fail for the wrong reason. The injected accessibility
    block is stripped because it is asserted on separately, and because its
    caption ends "(severity: HIGH)", which would match a search for the chip
    words on a canvas that no longer draws one.
    """
    return " ".join(t.strip() for t in _TEXT_RE.findall(_A11Y_RE.sub("", svg)) if t.strip())


def drawn_tokens(svg: str) -> list:
    """The drawn text as separate strings, so a chip word can be matched exactly.

    ``"VIOLATED" in drawn_text(...)`` is not safe here: the on-canvas
    explanation paragraph legitimately contains the word "violated" in its
    static how-to-read sentence, which is a description of what the chart WOULD
    show and not a claim about this run.
    """
    return [t.strip() for t in _TEXT_RE.findall(_A11Y_RE.sub("", svg)) if t.strip()]


def explanation_meta(svg: str) -> dict:
    """The machine-readable explanation block injected by rendering.explain."""
    match = _META_RE.search(svg)
    assert match, "every rendered chart carries a vfairness-explanation metadata block"
    return json.loads(match.group(1))


FAIR_METHOD = SimpleNamespace(
    method_name="reweighting",
    accuracy=0.842,
    fairness_violation=0.031,
    constraint_satisfied=True,
    training_time=1.2,
)
UNFAIR_METHOD = SimpleNamespace(
    method_name="adversarial",
    accuracy=0.871,
    fairness_violation=0.180,
    constraint_satisfied=False,
    training_time=9.4,
)
BLANK_RECOMMENDATION = SimpleNamespace(
    recommended_method="",
    priority="low",
    rationale="",
    alternative_methods=[],
)
REAL_RECOMMENDATION = SimpleNamespace(
    recommended_method="reweighting",
    priority="high",
    rationale="Best fairness at a small accuracy cost",
    alternative_methods=["exp_gradient"],
)


def report(**overrides):
    """An EMPTY report by default: the shape the exhaustive sweep called with.

    A light stand-in rather than the real dataclass, because the adapters read
    plain attributes and the stand-in exercises exactly that contract.
    """
    base = dict(
        timestamp="2026-08-27T10:00:00",
        task_type="",
        data_info={},
        baseline_metrics={},
        fairness_analysis={},
        method_comparisons=[],
        recommendation=None,
        tradeoff_analysis={},
        critical_issues=[],
        action_items=[],
        metadata={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def healthy(**overrides):
    """A report that really was analysed. The controls render this one."""
    fields = dict(
        task_type="classification",
        data_info={
            "n_samples": 1000,
            "n_groups": 2,
            "n_features": 12,
            "attribute_name": "gender",
        },
        baseline_metrics={
            "accuracy": 0.88,
            "fairness_violation": 0.22,
            "constraint_satisfied": False,
        },
        method_comparisons=[FAIR_METHOD, UNFAIR_METHOD],
        recommendation=REAL_RECOMMENDATION,
        critical_issues=["Baseline violates the constraint"],
        action_items=["Retrain with reweighting"],
        fairness_analysis={
            "constraint_type": "demographic_parity",
            "base_rate_disparity": 0.19,
            "group_statistics": {
                "male": {"size": 600, "proportion": 0.6, "positive_rate": 0.51},
                "female": {"size": 400, "proportion": 0.4, "positive_rate": 0.32},
            },
        },
    )
    fields.update(overrides)
    return report(**fields)


# Each case: the render, and why the constraint verdict is not available.
NOT_EVALUATED_CASES = {
    # The exhaustive sweep's own input: an empty report, no recommendation.
    "analysis_empty_no_recommendation": lambda: training_analysis_report_to_svg(report()),
    "report_empty_no_recommendation": lambda: training_report_to_svg(report()),
    # The same report carrying a recommendation OBJECT. The old code crashed on
    # the None path and fabricated on this one, so both are pinned.
    "analysis_empty_blank_recommendation": lambda: training_analysis_report_to_svg(
        report(recommendation=BLANK_RECOMMENDATION)
    ),
    "report_empty_blank_recommendation": lambda: training_report_to_svg(
        report(recommendation=BLANK_RECOMMENDATION)
    ),
    # Methods WERE trained, but the baseline carries no constraint result. The
    # verdict is still fabricated if it is drawn, and the accessible sentence
    # takes it from the same field, so the whole page is could-not-check.
    "analysis_methods_without_constraint": lambda: training_analysis_report_to_svg(
        report(task_type="classification", method_comparisons=[FAIR_METHOD])
    ),
    "report_methods_without_constraint": lambda: training_report_to_svg(
        report(task_type="classification", method_comparisons=[FAIR_METHOD])
    ),
}

# Words a reader takes a verdict, a score or a rate off. Matched as whole drawn
# strings, never as substrings of the static explanatory prose.
FORBIDDEN_CHIPS = ("VIOLATED", "SATISFIED", "PASS", "FAIL", "HIGH", "MEDIUM", "LOW")


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_renders_a_real_artifact(case):
    """No adapter may answer this input with an empty string or an exception.

    ``training_report_to_svg`` raised AttributeError outright when
    *recommendation* was None, so a caller passing ``save_path`` got no file at
    all and a traceback from a rendering call.
    """
    svg = NOT_EVALUATED_CASES[case]()
    assert svg, f"{case}: returned an empty string, which writes a 0-byte SVG"
    assert svg.lstrip().startswith("<svg"), f"{case}: not a rendered SVG"
    assert svg.rstrip().endswith("</svg>"), f"{case}: truncated SVG"


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_says_not_checked_on_the_canvas(case):
    """The absence must be stated where a SIGHTED reader sees it.

    Saying it only in the accessible description is not enough: the exported
    picture is what gets attached to a pull request and shown to an auditor.
    """
    text = drawn_text(NOT_EVALUATED_CASES[case]())
    assert "NOT CHECKED" in text, f"{case}: canvas carries no could-not-check state"
    assert "not a pass and not a failure" in text, (
        f"{case}: canvas does not say that could-not-check is neither verdict"
    )


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_is_never_drawn_as_a_failure(case):
    """The named defect, checked on the picture.

    VIOLATED is the exact word the False default painted, in the red chip, and
    it is the one a reader acts on.
    """
    tokens = drawn_tokens(NOT_EVALUATED_CASES[case]())
    survivors = [tok for tok in tokens if tok in FORBIDDEN_CHIPS]
    assert not survivors, (
        f"{case}: {survivors} drawn as a verdict on a run whose constraint was never "
        f"evaluated. VIOLATED here is a fabricated failure, not a finding."
    )


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_withholds_every_baseline_number(case):
    """Could-not-check withholds the numbers, it does not merely annotate them.

    A NOT CHECKED badge above an "Accuracy 0.0000" row is still a canvas a
    reader takes a number off, and 0.0000 beside a 0.000 base-rate disparity is
    the picture of a model with no accuracy and perfect parity between groups.
    """
    text = drawn_text(NOT_EVALUATED_CASES[case]())
    for token in ("0.0000", "0.000", "Accuracy", "Violation", "Base rate disparity"):
        assert token not in text, f"{case}: {token!r} still drawn on a chart that measured nothing"


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_description_leads_with_could_not_check(case):
    """The accessible description is the whole artifact for a screen reader.

    It must never be more confident than the visible badge, and this is the
    layer where the old wording said "(constraint violated)" outright.
    """
    finding = str(explanation_meta(NOT_EVALUATED_CASES[case]())["finding"])
    assert finding.upper().startswith(("COULD NOT CHECK", "NOT ASSESSABLE")), (
        f"{case}: description reads as a result, not as could-not-check: {finding[:120]}"
    )
    assert "constraint violated" not in finding.lower(), (
        f"{case}: description still asserts a violation that was never evaluated"
    )


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_is_never_graded_high(case):
    """HIGH is the grade a real breach gets, and nothing here was measured.

    INFO and LOW are refused for the same reason in the other direction: they
    read as a clean bill of health.
    """
    severity = str(explanation_meta(NOT_EVALUATED_CASES[case]())["severity"]).lower()
    assert severity == "medium", (
        f"{case}: could-not-check graded {severity!r}; HIGH and CRITICAL read as a "
        f"finding, INFO and LOW read as a clean bill of health, and neither happened"
    )


@pytest.mark.parametrize("case", sorted(NOT_EVALUATED_CASES))
def test_an_unevaluated_constraint_action_does_not_imply_a_measurement(case):
    """The curated action tells the reader to adopt the recommended method.

    On a run that evaluated nothing that instruction is not merely useless, it
    asserts that an evaluation happened.
    """
    meta = explanation_meta(NOT_EVALUATED_CASES[case]())
    assert "certifies nothing" in str(meta["recommendation"]), (
        f"{case}: recommendation still assumes a verdict was produced"
    )


def test_a_report_with_no_recommendation_object_does_not_raise():
    """The other half of the assignment, pinned on its own.

    ``training_report_to_svg`` reached straight for
    ``report.recommendation.recommended_method``. A None recommendation is a
    third state, not an exception thrown out of a rendering call.
    """
    svg = training_report_to_svg(report(recommendation=None))

    assert svg.rstrip().endswith("</svg>")
    assert "NOT CHECKED" in drawn_text(svg)


def test_save_path_writes_the_could_not_check_state_to_disk():
    """Checked on the FILESYSTEM, not on the return value.

    A caller using ``save_path`` never sees the returned string at all, and a
    0-byte SVG opens as a broken image with nothing anywhere to explain it.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        for name, render in (
            ("analysis.svg", training_analysis_report_to_svg),
            ("report.svg", training_report_to_svg),
        ):
            target = Path(tmp) / name
            render(report(), save_path=str(target))

            assert target.exists(), f"{name}: save_path wrote no file"
            assert target.stat().st_size > 0, f"{name}: save_path wrote a 0-byte SVG"
            assert "NOT CHECKED" in drawn_text(target.read_text()), (
                f"{name}: the file on disk does not carry the could-not-check state"
            )


# ── the partial states: measured, but not everything ────────────────────────


def test_a_missing_base_rate_disparity_is_not_drawn_as_perfect_parity():
    """0.000 is the canvas for identical base rates across every group.

    The constraint here WAS evaluated, so the page keeps its verdict; only the
    disparity that was never computed is withheld.
    """
    svg = training_analysis_report_to_svg(
        healthy(fairness_analysis={"constraint_type": "demographic_parity"})
    )
    text = drawn_text(svg)

    assert "Base rate disparity: 0.000" not in text
    assert "Base rate disparity was not computed" in text
    # The verdict that WAS reached still stands.
    assert "VIOLATED" in drawn_tokens(svg)


@pytest.mark.parametrize("render", [training_analysis_report_to_svg, training_report_to_svg])
def test_a_constraint_verdict_without_numbers_does_not_invent_the_numbers(render):
    """A verdict can arrive without the metrics behind it.

    The accuracy and the violation are then withheld, the verdict is kept, and
    the real template still renders: reading ``baseline_violation`` as a number
    when it is absent used to throw inside the template and silently drop the
    caller onto the plain-text fallback canvas.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = render(report(baseline_metrics={"constraint_satisfied": True}))

    assert not [w for w in caught if "SVG rendering failed" in str(w.message)], (
        "the template raised and the render fell through to the fallback canvas"
    )
    text = drawn_text(svg)
    assert "BASELINE PERFORMANCE" in text, "this is the fallback canvas, not the template"
    assert "0.0000" not in text
    assert "Accuracy and fairness violation were not measured on this run." in text
    assert "SATISFIED" in drawn_tokens(svg), "the verdict that WAS reached must survive"


@pytest.mark.parametrize("render", [training_analysis_report_to_svg, training_report_to_svg])
def test_no_priority_chip_is_drawn_for_advice_that_was_never_given(render):
    """A LOW chip beside "N/A" is a grade invented for a recommendation that
    does not exist. The constraint verdict is untouched by this."""
    svg = render(healthy(recommendation=None))

    tokens = drawn_tokens(svg)
    assert "N/A" in tokens, "the absent recommendation is still named on the canvas"
    assert "LOW" not in tokens, "a priority chip was drawn for advice nobody gave"
    assert "No mitigation method was recommended for this run." in drawn_text(svg)


# ── CONTROLS: healthy input must be untouched ───────────────────────────────
#
# Not decoration. A "fix" that withheld verdicts from real data too would
# satisfy every guard above while destroying the product, so each control
# asserts the measured number, the verdict word and the grade are still there,
# and that the could-not-check panel is NOT.


@pytest.mark.parametrize("render", [training_analysis_report_to_svg, training_report_to_svg])
def test_control_a_measured_violation_still_renders_its_verdict(render):
    svg = render(healthy())
    text = drawn_text(svg)
    tokens = drawn_tokens(svg)

    assert "NOT CHECKED" not in text, "healthy input was withheld from the reader"
    assert "VIOLATED" in tokens
    assert "0.8800" in text and "0.2200" in text
    assert "PASS" in tokens and "FAIL" in tokens
    assert explanation_meta(svg)["severity"] == "high", (
        "a real violated constraint is still a HIGH finding"
    )


@pytest.mark.parametrize("render", [training_analysis_report_to_svg, training_report_to_svg])
def test_control_a_measured_pass_still_renders_its_verdict(render):
    svg = render(
        healthy(
            baseline_metrics={
                "accuracy": 0.90,
                "fairness_violation": 0.01,
                "constraint_satisfied": True,
            }
        )
    )
    tokens = drawn_tokens(svg)

    assert "SATISFIED" in tokens
    assert "VIOLATED" not in tokens
    assert "NOT CHECKED" not in drawn_text(svg)


@pytest.mark.parametrize("render", [training_analysis_report_to_svg, training_report_to_svg])
def test_control_a_measured_zero_is_not_an_absence(render):
    """The whole rule, stated as a test.

    A baseline whose accuracy really came out 0.0000 with the constraint
    satisfied keeps every number, its green verdict and its grade. An absence
    and a measured zero are different claims and render differently.
    """
    svg = render(
        healthy(
            baseline_metrics={
                "accuracy": 0.0,
                "fairness_violation": 0.0,
                "constraint_satisfied": True,
            }
        )
    )
    text = drawn_text(svg)

    assert "NOT CHECKED" not in text, "a measured zero was mistaken for a missing value"
    assert "0.0000" in text
    assert "SATISFIED" in drawn_tokens(svg)


def test_control_a_measured_base_rate_disparity_still_reaches_the_page():
    """Pinned at three decimals, the same value the verdict suite pins."""
    text = drawn_text(training_analysis_report_to_svg(healthy()))

    assert "Base rate disparity: 0.190" in text
    assert "Base rate disparity was not computed" not in text


@pytest.mark.parametrize("render", [training_analysis_report_to_svg, training_report_to_svg])
def test_control_a_real_recommendation_keeps_its_priority_chip(render):
    tokens = drawn_tokens(render(healthy()))

    assert "reweighting" in tokens
    assert "HIGH" in tokens, "the priority chip of a real recommendation was dropped"


def _explode_the_template(monkeypatch):
    monkeypatch.setattr(
        adapters_training,
        "render_svg",
        lambda template, data: (_ for _ in ()).throw(RuntimeError("template exploded")),
    )


def test_the_fallback_canvas_does_not_print_an_absent_baseline_as_zero(monkeypatch):
    """The fallback formats the baseline itself, so it needs the same rule.

    It read ``data.get("baseline_accuracy", 0)`` straight into a ``:.4f``, so a
    baseline that was never evaluated printed "Accuracy: 0.0000" on the very
    canvas whose whole job is to survive a template failure, and a withheld
    None would have raised TypeError inside it.
    """
    _explode_the_template(monkeypatch)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        absent = training_analysis_report_to_svg(report())

    assert "0.0000" not in absent
    assert "not measured" in absent


def test_control_the_fallback_canvas_still_prints_measured_numbers(monkeypatch):
    """A real baseline keeps its four decimals on the fallback canvas too."""
    _explode_the_template(monkeypatch)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        measured = training_analysis_report_to_svg(healthy())

    assert "Accuracy: 0.8800 | Violation: 0.2200" in measured
