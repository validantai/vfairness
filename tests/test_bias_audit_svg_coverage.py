"""The bias-audit canvas must not certify a dataset nobody could assess.

THE DEFECT, reproduced by execution on 2026-09-25 and not by reading. A 2-row
single-group frame produces a report that is honest in every field: it records
``attribute_assessed={'gender': False}``, its own recommendation reads "Every
audit module executed and none of them assessed anything ... Nothing here clears
the data", and it warns that "the overall risk score of 0.0 is NOT a measurement
of low risk". Rendered to SVG, the same report showed:

    OVERALL RISK | 0% | MINIMAL     and three module tiles reading 0.00

and no trace of could-not-check anywhere on the canvas. The measurement was
correct in the object and dropped at the rendering boundary, which is this
library's standing failure shape, and an SVG is an export: it outlives the run,
gets attached to a ticket, and nobody who opens it can see the warning.

THREE causes, all of them live at once, which is why the canvas looked fine to
whoever checked one of them:

  1. ``BiasAuditReport.execution_coverage`` can return FIVE states and
     ``adapters._COVERAGE_STATES`` whitelisted four, so the one that says "ran
     and assessed nothing" fell through to "unrecorded".
  2. ``nothing_measured`` keyed on the finding COUNT, and an insufficient-data
     representation row is a finding by count, so the assessable branch was taken.
  3. ``_module_score`` turns an empty list into a measured 0.00 on the strength
     of the module having RUN, and all four modules do run on a 2-row frame.

Each test below fails if one of those is reverted. The control tests fail if the
fix over-corrects, which is the other way this goes wrong: refusing a canvas that
did measure something throws away a real result.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection import BiasDetector
from vfairness.rendering.adapters import _COVERAGE_STATES


def _visible(svg: str) -> str:
    return " | ".join(t.strip() for t in re.findall(r">([^<>]+)<", svg) if t.strip())


def _audit(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BiasDetector(
            df, protected_attributes=["gender"], outcome_column="hired"
        ).full_audit()


@pytest.fixture(scope="module")
def nothing_assessable():
    """One group, two rows: no module can reach a verdict about a disparity."""
    return _audit(pd.DataFrame({"gender": ["F", "F"], "income": [50000, 52000], "hired": [1, 0]}))


@pytest.fixture(scope="module")
def really_measured():
    """400 rows, two groups, a real and findable hiring gap."""
    rng = np.random.default_rng(20260925)
    n = 400
    gender = np.where(rng.random(n) < 0.5, "F", "M")
    hired = np.where(gender == "M", rng.random(n) < 0.7, rng.random(n) < 0.3).astype(int)
    return _audit(
        pd.DataFrame(
            {"gender": gender, "income": rng.normal(50000, 8000, n).round(), "hired": hired}
        )
    )


def test_the_report_itself_knows_nothing_was_assessed(nothing_assessable):
    """The premise. If this fails the defect has moved and the rest means nothing."""
    assert nothing_assessable.execution_coverage() == "ran_but_assessed_nothing"
    assert nothing_assessable.assessment_coverage() == "none"
    assert nothing_assessable.overall_risk_score == 0.0


def test_the_canvas_never_bands_an_unassessed_audit_as_minimal(nothing_assessable):
    svg = nothing_assessable.to_svg()
    assert "MINIMAL" not in svg, "a dataset nobody could assess was banded MINIMAL"
    assert "NOT ASSESSABLE" in _visible(svg)


def test_no_module_tile_claims_a_measured_zero(nothing_assessable):
    """Cause 3. All four modules execute on this frame and none could assess."""
    svg = nothing_assessable.to_svg()
    assert svg.count(">0.00<") == 0, "a module that assessed nothing rendered a measured 0.00"
    assert _visible(svg).count("N/A") >= 4


def test_the_canvas_says_why_in_words_that_are_true_of_this_report(nothing_assessable):
    """Cause 2, and the hardcoded-sentence defect with it.

    The template used to print "no module of this audit returned a finding",
    which is FALSE here: the representation module returned an insufficient-data
    finding. The sentence now comes from the adapter, where it is computed.
    """
    seen = _visible(nothing_assessable.to_svg())
    assert "every audit module executed and not one of them could assess anything" in seen
    assert "does not clear the dataset" in seen.lower()
    assert "no module of this audit returned a finding" not in seen


def test_the_renderer_has_a_word_for_every_state_the_detector_can_return():
    """Cause 1, pinned as a vocabulary drift test rather than as one string.

    A sixth coverage state added to the detector alone would silently degrade to
    "unrecorded" again. This fails the moment the two sides disagree.
    """
    from vfairness.preprocessing.bias_detection import detector as det

    states = {
        value
        for name, value in vars(det).items()
        if name.startswith("COVERAGE_") and isinstance(value, str)
    }
    assert states, "the detector's coverage constants are no longer named COVERAGE_*"
    missing = states - set(_COVERAGE_STATES)
    assert not missing, (
        f"the renderer has no word for {missing}, so it will read them as unrecorded"
    )


# ── controls: the fix must not refuse what WAS measured ────────────────────────


def test_control_a_real_audit_is_still_assessed(really_measured):
    assert really_measured.execution_coverage() == "complete"
    assert really_measured.assessment_coverage() != "none"


def test_control_a_real_audit_keeps_its_measured_badge(really_measured):
    svg = really_measured.to_svg()
    seen = _visible(svg)
    assert "NOT ASSESSABLE" not in seen, "an audit that measured a real gap was withheld"
    assert re.search(r">\d+%<", svg), "the measured risk percentage disappeared"


def test_control_a_real_audit_still_scores_its_modules(really_measured):
    """The tile gate must not withhold a module that genuinely assessed."""
    svg = really_measured.to_svg()
    assert _visible(svg).count("N/A") < 4, "every tile was withheld on a measured audit"
