"""A guessed positive class does not fail closed: it INVERTS the finding.

``_test_categorical_disparity`` ranks groups by their rate of the FAVOURABLE
outcome. For a binary outcome it resolves that label from a whitelist, and for
anything outside the whitelist it used to fall back to ``cols[-1]``, the last
crosstab column, which is alphabetical order.

WHY THIS FILE EXISTS. Measured 2026-09-30 with the labels ``admit`` and
``waitlist``, neither of them exotic for an admissions audit. ``waitlist`` sorts
last, so the rate was read as the waitlist rate and the result named group B as
PRIVILEGED and group A as DISADVANTAGED, while A was admitted 90% of the time
and B 10%. Zero warnings. With ``offer`` and ``decline`` the identical code is
correct by luck, which is why it survived.

That is worse than a fabricated magnitude. A fabricated magnitude understates a
finding; this points a real finding AT THE WRONG GROUP, and a reader acting on
it would investigate the group that was favoured. The comment above the
whitelist in the source already said so, and the whitelist was added to fix
exactly this, leaving the original defect in place as the fallback.

What is withheld is only the DIRECTION. The gap, the p-value and Cramer's V are
direction invariant and stay measured, because suppressing them would delete a
real finding in order to avoid naming a direction.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection import statistical as S

DISPARITY_TYPE = list(S.DisparityType)[0]


def _run(positive: str, negative: str):
    """Group A gets ``positive`` 90% of the time, so A is FAVOURED. Always."""
    df = pd.DataFrame(
        {
            "grp": ["A"] * 100 + ["B"] * 100,
            "out": [positive] * 90 + [negative] * 10 + [positive] * 10 + [negative] * 90,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = S._test_categorical_disparity(
            df, "out", "grp", DISPARITY_TYPE, {"A": 100, "B": 100}, 30
        )
    assert result is not None, "the fixture is meant to produce a result"
    return result, [str(w.message) for w in caught]


# Labels outside every whitelist. The second element of each pair is the one
# that sorts LAST, which is what the old fallback would have called positive.
@pytest.mark.parametrize(
    "positive,negative",
    [("admit", "waitlist"), ("offer", "shortlist"), ("granted", "returned")],
    ids=["admit_waitlist", "offer_shortlist", "granted_returned"],
)
def test_an_unresolvable_positive_class_withholds_the_direction(positive, negative):
    result, caught = _run(positive, negative)

    assert result.privileged_group is None, (
        f"the direction was asserted from sort order: privileged="
        f"{result.privileged_group!r}. Group A receives {positive!r} 90% of the time."
    )
    assert result.disadvantaged_group is None, result.disadvantaged_group
    assert result.direction_not_determined_reason, "the could-not-check carries no reason"
    assert result.positive_class_used == negative, (
        "the reference label must be NAMED so a reader can settle the direction: "
        f"{result.positive_class_used!r}"
    )
    assert len(caught) == 1, caught


@pytest.mark.parametrize(
    "positive,negative",
    [("admit", "waitlist"), ("offer", "shortlist")],
    ids=["admit_waitlist", "offer_shortlist"],
)
def test_the_gap_and_the_significance_are_still_measured(positive, negative):
    """Withholding a direction must not delete the finding."""
    result, _ = _run(positive, negative)
    assert result.disparity_magnitude == pytest.approx(0.8), result.disparity_magnitude
    assert result.pvalue < 1e-20, result.pvalue


@pytest.mark.parametrize(
    "positive,negative",
    [("admit", "waitlist"), ("offer", "shortlist")],
    ids=["admit_waitlist", "offer_shortlist"],
)
def test_the_reader_facing_prose_says_could_not_check_and_does_not_invent_a_group(
    positive, negative
):
    """The pin is on what a person READS, not on the dataclass field.

    ``f"{None}"`` is the readable word "None", so a withheld direction printed
    through an unguarded f-string becomes a recommendation about a group called
    None: the absent value MINTING CONTENT.
    """
    result, _ = _run(positive, negative)
    prose = " ".join(result.recommendations)
    assert "could NOT be determined" in prose, result.recommendations
    assert "'None'" not in prose and "None'" not in prose, result.recommendations
    assert "statistically significant" in prose, (
        "an undetermined direction must not read as an absence of harm"
    )
    published = result.to_dict()
    assert published["privileged_group"] is None
    assert published["direction_not_determined_reason"]
    assert published["positive_class_used"] == negative


# ---------------------------------------------------------------------------
# Over-correction controls. A guard that refuses every label passes every
# refusal test above and destroys the unit, so each of these asserts the REAL
# number and the REAL named group.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "positive,negative",
    [("approved", "denied"), ("hired", "rejected"), ("1", "0"), ("yes", "no")],
    ids=["approved_denied", "hired_rejected", "one_zero", "yes_no"],
)
def test_a_resolvable_positive_class_still_names_the_direction(positive, negative):
    result, caught = _run(positive, negative)
    assert result.privileged_group == "A", result.privileged_group
    assert result.disadvantaged_group == "B", result.disadvantaged_group
    assert result.disparity_magnitude == pytest.approx(0.8), result.disparity_magnitude
    assert result.positive_class_used == positive, result.positive_class_used
    assert result.direction_not_determined_reason is None
    assert caught == [], caught
    prose = " ".join(result.recommendations)
    assert "'A'" in prose and "'B'" in prose, result.recommendations
    assert "could NOT be determined" not in prose


def test_a_multiclass_outcome_withholds_the_direction_and_keeps_the_strength():
    """It used to name the first and last group by DICT ORDER, which is not a
    measurement of anything."""
    df = pd.DataFrame(
        {
            "grp": ["A"] * 100 + ["B"] * 100,
            "out": ["a"] * 50 + ["b"] * 30 + ["c"] * 20 + ["a"] * 20 + ["b"] * 30 + ["c"] * 50,
        }
    )
    result = S._test_categorical_disparity(
        df, "out", "grp", DISPARITY_TYPE, {"A": 100, "B": 100}, 30
    )
    assert result is not None
    assert result.privileged_group is None
    assert result.disadvantaged_group is None
    assert "3 classes" in result.direction_not_determined_reason
    assert result.effect_size > 0.3, result.effect_size
    assert result.pvalue < 1e-4, result.pvalue
