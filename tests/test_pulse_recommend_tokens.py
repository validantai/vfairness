"""Wave 4 pins for the Pulse recommender's label matching.

Three waves closed one bug class: a metric's DIRECTION decided by a SUBSTRING of
its name, which graded a maximal violation as a pass ("ratio" matches
"cali[bratio]n_difference"; CLAUDE.md, reconciled in 47f1e09f8). Two sites in
``operations/pulse/recommend.py`` were the last ones keeping the repo's
reintroduction guard red on a clean tree:

    secondaries and "parity" in secondaries[0].lower()
    any("calibration" in a["definition"].lower() for a in applied)

Neither inverts a verdict. Both steer user-facing RECOMMENDATION PROSE: the
first names the proxy a label-free run should optimise against, the second
decides whether the reader is shown the Kleinberg/Chouldechova impossibility
trade-off. And both are wrong in the way the class always is: "parity" is a
substring of "disparity", and "calibration" is a substring of
"multicalibration" and "recalibration", so a label that merely MENTIONS the
family is read as being of it.

A red guard on a clean tree is worse than no guard, because it teaches everyone
to scroll past it. These are fixed by whole-TOKEN matching, not by narrowing the
guard.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib

import pytest

from vfairness.operations.pulse.recommend import (
    _label_has,
    _label_words,
    recommend_fairness_definition,
)

# 1.  The guard the two sites kept red


def test_recommend_holds_no_substring_direction_test_any_more():
    """The WIDENED repo guard, run on this module.

    Loaded from ``test_audit_final_release`` rather than copied: a private copy
    here would go stale exactly the way the narrow marker list did, and would
    then certify this file as clean while the same defect sat in it.
    """
    import vfairness.operations.pulse.recommend as recommend

    guard_path = pathlib.Path(__file__).with_name("test_audit_final_release.py")
    spec = importlib.util.spec_from_file_location("_wave4_guard", guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    path = pathlib.Path(recommend.__file__)
    assert not guard._substring_direction_hits(path), (
        "a substring test on a metric/definition label is back in recommend.py"
    )

    # The module must still EXPLAIN the rule in prose, so the next author does
    # not delete the token matcher as unexplained indirection. Deleting the
    # explanation is how this understanding was lost the first time.
    src = path.read_text(encoding="utf-8")
    assert "multicalibration" in src
    assert "Whole-token, never substring" in src


# 2.  What whole-token matching means, one case per way the substring test lied


@pytest.mark.parametrize(
    "label,phrase",
    [
        # The exact labels this module emits.
        ("Calibration within groups", "calibration"),
        ("Demographic parity / four-fifths", "parity"),
        ("False-positive parity / predictive parity", "predictive parity"),
        ("False-positive parity / predictive parity", "false-positive"),
        ("Equal opportunity (true-positive-rate parity)", "opportunity"),
        ("Equalized odds", "odds"),
        # Case and punctuation are normalised away, the word boundary is not.
        ("CALIBRATION WITHIN GROUPS", "calibration"),
        ("Calibration-within-groups", "calibration"),
    ],
)
def test_a_label_of_the_family_matches(label, phrase):
    assert _label_has(label, phrase) is True


@pytest.mark.parametrize(
    "label,phrase",
    [
        # THE DEFECT, in both tokens. Each of these is True under
        # `phrase in label.lower()` and must be False here.
        ("Multicalibration across subgroups", "calibration"),
        ("Recalibration error", "calibration"),
        ("Disparity headline", "parity"),
        ("Disparity screen (four-fifths)", "parity"),
        # A multi-word phrase must match as consecutive words, not as two words
        # that happen to appear somewhere in the label.
        ("Predictive value, reported with parity elsewhere", "predictive parity"),
        # A different family entirely.
        ("Demographic parity / four-fifths", "calibration"),
    ],
)
def test_a_label_that_merely_mentions_the_family_does_not_match(label, phrase):
    assert _label_has(label, phrase) is False, (
        f"{phrase!r} matched {label!r}: this is the substring test again, and it "
        "puts the wrong recommendation paragraph in front of a reader"
    )


def test_the_substring_test_really_would_have_said_yes():
    """Positive control for the cases above.

    Without this, a typo in a fixture would make the negative cases vacuous:
    they would pass because nothing matches anything, not because whole-token
    matching works.
    """
    for label, phrase in (
        ("Multicalibration across subgroups", "calibration"),
        ("Recalibration error", "calibration"),
        ("Disparity headline", "parity"),
    ):
        assert phrase in label.lower(), "fixture no longer exercises the substring trap"
        assert not _label_has(label, phrase)


def test_label_words_splits_on_every_non_alphanumeric_run():
    assert _label_words("False-positive parity / predictive parity") == [
        "false",
        "positive",
        "parity",
        "predictive",
        "parity",
    ]
    assert _label_words("") == []
    assert _label_words(None) == ["none"]  # str() first, never raises


# 3.  The prose the two sites drive, end to end


def test_a_label_free_run_names_the_parity_proxy_it_can_actually_compute():
    """Site 1. Hiring has no ground truth here, so the primary stays the
    normative objective and the proxy must be the parity secondary."""
    out = recommend_fairness_definition("hiring", "EU", has_ground_truth=False)
    how = out["howToApply"]
    assert "demographic parity / four-fifths" in how
    assert "the selection-rate (four-fifths) gap" not in how
    assert out["primary"] == "Equal opportunity"


def test_a_label_free_run_with_no_parity_secondary_falls_back_to_the_gap():
    """Site 1, the other branch: clinical secondaries lead with calibration, so
    there is no parity label to name and the generic gap is used."""
    out = recommend_fairness_definition("clinical triage", "EU", has_ground_truth=False)
    assert out["secondaries"][0] == "Calibration within groups"
    assert "the selection-rate (four-fifths) gap" in out["howToApply"]


def test_the_impossibility_tradeoff_appears_only_with_both_families_applied():
    """Site 2. Calibration is applied when scores are exposed; the error-rate
    family comes from the domain prior. With differing base rates measured, the
    Kleinberg/Chouldechova result is the honest thing to say."""
    out = recommend_fairness_definition(
        "clinical triage",
        "EU",
        has_ground_truth=True,
        scores_exposed=True,
        base_rates_differ=True,
    )
    assert "Kleinberg" in out["tradeoff"]
    assert any(_label_has(a["definition"], "calibration") for a in out["definitionsApplied"])


def test_no_impossibility_claim_when_calibration_was_never_applied():
    """NEGATIVE case: hiring without exposed scores applies no calibration
    definition, so the trade-off paragraph must not be raised at all."""
    out = recommend_fairness_definition(
        "hiring",
        "EU",
        has_ground_truth=True,
        scores_exposed=False,
        base_rates_differ=True,
    )
    assert not any(_label_has(a["definition"], "calibration") for a in out["definitionsApplied"])
    assert "Kleinberg" not in out["tradeoff"]
    assert "Headline emphasis is" in out["tradeoff"]


def test_base_rates_that_could_not_be_measured_are_stated_as_such():
    """The third state in this module's prose: base_rates_differ=None is "not
    measured", never "no difference"."""
    out = recommend_fairness_definition(
        "clinical triage",
        "EU",
        has_ground_truth=False,
        scores_exposed=True,
        base_rates_differ=None,
    )
    assert "could not be measured" in out["tradeoff"]
    assert "Kleinberg" not in out["tradeoff"]


# 4.  The shipped label vocabulary, swept


def _every_applied_label():
    """Every definition label this module can put in ``definitionsApplied``."""
    domains = ("lending", "hiring", "clinical triage", "recidivism", "streaming ads")
    labels = set()
    for domain, gt, hd, exposed in itertools.product(
        domains, (True, False), (None, "punitive", "assistive"), (True, False)
    ):
        out = recommend_fairness_definition(
            domain, "EU", has_ground_truth=gt, harm_direction=hd, scores_exposed=exposed
        )
        labels.update(a["definition"] for a in out["definitionsApplied"])
        labels.update(r["definition"] for r in out["rejected"])
    return labels


def test_exactly_the_calibration_labels_are_read_as_calibration():
    """The sweep the unit tests cannot do: run the real function over its whole
    input space and check the classification of every label it actually emits.

    A future label such as "Multicalibration within groups" would be classified
    by the substring test as a calibration definition and would silently raise
    the impossibility trade-off; this test is what would catch that.
    """
    labels = _every_applied_label()
    assert labels, "the sweep produced no labels; it is no longer exercising anything"
    matched = {label for label in labels if _label_has(label, "calibration")}
    assert matched == {"Calibration within groups"}, (
        f"calibration classification drifted across the label vocabulary: {matched}"
    )
