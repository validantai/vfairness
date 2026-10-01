"""A set containing ONE demographic scored better than a balanced one.

Found 2026-09-11 by a census that ran every registered capability against
degenerate input. This is the sharpest instance of the defect class in the
library, because it is not merely a false all-clear: it is INVERTED. The most
homogeneous possible input received the best possible grade.

Measured before the fix, through `representation_severity`, which is the
verdict a caller is actually shown::

    skew(["White_male"] * 100)            -> 'pass'
    skew(["a"]*40 + ["b"]*35 + ["c"]*25)  -> 'warn'
    skew(["a"]*90 + ["b"]*5  + ["c"]*5)   -> 'critical'

100 images of one demographic scored BETTER than a genuinely balanced set, with
no warning anywhere.

THE CAUSE IS THE DEFAULT, NOT THE MEASURE. Skew is ln(observed / desired), and
the default desired distribution is uniform over the groups that were OBSERVED.
With one observed group that reference is {g: 1.0}, the observed share is also
1.0, and the skew is exactly ln(1) = 0. The measure faithfully reports that the
set matches a reference which says nothing.

Supplying a real reference has always worked and still does: the same 100 images
against a four-group reference grade 'critical' and name the three absent
groups. So the fix is to refuse the vacuous default, not to change the maths.
"""

from __future__ import annotations

import math
import warnings

import pytest

from vfairness.vision import bias_amplification, ndkl, representation_severity, skew


class TestTheInvertedVerdict:
    def test_one_demographic_is_not_a_pass(self):
        result = skew(["White_male"] * 100)
        assert result["available"] is False, (
            f"a set containing one demographic reported available={result['available']}, "
            f"maxSkew={result.get('maxSkew')}. Nothing was compared."
        )
        assert representation_severity(result) == "not_assessed"

    def test_it_no_longer_outscores_a_balanced_set(self):
        """The inversion, stated as the comparison that exposed it."""
        homogeneous = representation_severity(skew(["White_male"] * 100))
        balanced = representation_severity(skew(["a"] * 40 + ["b"] * 35 + ["c"] * 25))
        assert homogeneous != "pass", "one demographic is still graded a pass"
        assert not (homogeneous == "pass" and balanced != "pass"), (
            f"homogeneous={homogeneous!r} still outranks balanced={balanced!r}"
        )

    def test_the_refusal_says_why(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = skew(["White_male"] * 100)
        assert "not a pass" in result["reason"].lower()
        assert any("nothing to compare against" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]

    def test_the_observed_distribution_is_still_reported(self):
        """Refusing must not throw away what WAS seen. The caller still needs to
        know the set was 100% one group; that fact is the finding."""
        result = skew(["White_male"] * 100)
        assert result["observed"] == {"White_male": 1.0}


class TestTheSiblings:
    def test_ndkl_refuses_a_single_group_with_no_reference(self):
        """0.0 is the BEST attainable NDKL and it was returned silently."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = ndkl(["a"] * 10)
        assert math.isnan(value), f"a single-group ranking scored {value}"
        assert any("undefined" in str(w.message) for w in caught)

    def test_amplification_refuses_an_empty_reference(self):
        """Amplification is generated share MINUS real-world share. With no
        real-world share there is nothing to subtract."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = bias_amplification(["a"] * 10, {})
        assert result["available"] is False, f"reported {result}"
        assert result["maxAmplification"] is None
        assert any("nothing to subtract" in str(w.message) for w in caught)


class TestOverCorrectionControls:
    """The measure must keep working. A fix that refuses everything would be
    worse than the defect, because it would remove the capability entirely."""

    def test_a_real_multi_group_set_is_still_measured(self):
        result = skew(["a"] * 40 + ["b"] * 35 + ["c"] * 25)
        assert result["available"] is True
        assert result["maxSkew"] > 0
        assert representation_severity(result) in ("pass", "warn", "critical")

    def test_a_genuinely_skewed_set_is_still_critical(self):
        assert representation_severity(skew(["a"] * 90 + ["b"] * 5 + ["c"] * 5)) == "critical"

    def test_one_group_with_a_reference_still_detects_total_exclusion(self):
        """The case the refusal must not swallow: a caller who supplies the
        desired distribution is telling us what was expected, and a set missing
        three of four groups is the strongest finding this measure produces."""
        reference = {g: 0.25 for g in ("White_male", "White_female", "Black_male", "Black_female")}
        result = skew(["White_male"] * 100, reference)
        assert result["available"] is True
        assert sorted(result["absentGroups"]) == ["Black_female", "Black_male", "White_female"]
        assert representation_severity(result) == "critical"

    def test_ndkl_still_measures_with_a_reference_or_several_groups(self):
        assert ndkl(["a", "a", "a", "b"]) == pytest.approx(0.5986, abs=1e-3)
        assert ndkl(["a"] * 10, {"a": 0.5, "b": 0.5}) == pytest.approx(0.6931, abs=1e-3)

    def test_amplification_still_measures_against_a_real_reference(self):
        result = bias_amplification(["a"] * 8 + ["b"] * 2, {"a": 0.5, "b": 0.5})
        assert result["available"] is True
        assert result["maxAmplification"] == pytest.approx(0.3)
