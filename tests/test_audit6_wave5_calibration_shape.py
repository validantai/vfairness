"""A result field that is PRESENT holding the wrong shape sails past its default.

Found from the outside: the session running the deployed consumer reported an
intermittent ``AttributeError: 'list' object has no attribute 'items'`` from a
calibration task, could not reproduce it in three further runs, and recorded it
UNEXPLAINED rather than dismissing it. It reproduces exactly here with
``{"group_metrics": []}``.

The mechanism is the one already recorded in this repo one shape along:
``dict.get(key, default)`` and ``getattr(obj, key, default)`` fall back only
when the key is ABSENT. A key that is PRESENT holding a list is returned as-is,
and the ``.items()`` call downstream raises.

Why it renders rather than raises now: an unrecognised shape is COULD NOT CHECK.
With no measured groups the adapter's own ``assessable`` gate is False, so the
panel reports "not assessable" instead of the green all-clear that the same
function's comment block records as a previous defect of this exact kind.
"""

from __future__ import annotations

import warnings

import pytest

from vfairness.rendering.adapters_calibration import calibration_disparity_to_svg

BAD_SHAPES = [
    pytest.param([], id="empty-list"),
    pytest.param([{"ece": 0.1}], id="list-of-dicts"),
    pytest.param("a,b", id="string"),
    pytest.param(7, id="int"),
]


class TestGroupMetricsOfTheWrongShape:
    @pytest.mark.parametrize("bad", BAD_SHAPES)
    def test_it_does_not_raise(self, bad):
        """The reported crash. A rendering adapter must not take down the task
        that called it because an upstream field had an unexpected type."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert calibration_disparity_to_svg({"group_metrics": bad})

    @pytest.mark.parametrize("bad", BAD_SHAPES)
    def test_the_bad_shape_is_named_not_swallowed(self, bad):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibration_disparity_to_svg({"group_metrics": bad})
        named = [str(w.message) for w in caught if "NOT MEASURED" in str(w.message)]
        assert named, [str(w.message) for w in caught]
        assert type(bad).__name__ in named[0], named[0]

    def test_the_same_hole_on_an_object_result(self):
        """The adapter reads dicts via .get and objects via getattr. Both
        defaults have the same blind spot, so both need the guard."""

        class Result:
            group_metrics = []
            has_significant_disparity = None
            overall_ece = None
            strategy = "platt"
            recommendation = ""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert calibration_disparity_to_svg(Result())
        assert [w for w in caught if "NOT MEASURED" in str(w.message)]

    def test_control_a_real_result_renders_and_is_silent(self):
        """Over-correction control: the measured path must be untouched, and
        must NOT emit the not-measured warning."""
        good = {
            "group_metrics": {"a": {"ece": 0.10}, "b": {"ece": 0.40}},
            "has_significant_disparity": True,
            "overall_ece": 0.25,
            "strategy": "platt",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = calibration_disparity_to_svg(good)
        assert svg and "<svg" in svg
        assert not [w for w in caught if "NOT MEASURED" in str(w.message)]

    def test_control_an_absent_key_is_still_the_ordinary_empty_case(self):
        """An ABSENT key is a different thing from a present-but-wrong one and
        keeps its existing behaviour: the {} default fires normally."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert calibration_disparity_to_svg({})
        assert not [w for w in caught if "NOT MEASURED" in str(w.message)]


class TestTheSecondCopyOnTheProductionPath:
    """The first guard went on ``calibration_disparity_to_svg``, which turned
    out to have NO caller in src at all. The production handler reaches
    ``calibration_report_to_svg`` instead, and that one delegates to
    ``_calibration_curves``, which had the SAME hole in a slightly worse form:

        group_metrics = report.group_metrics or {}

    ``or {}`` rescues a FALSY wrong shape and nothing else, so an EMPTY list is
    fine and a NON-EMPTY list sails through to ``.items()``. As in the first
    copy, the inner value on the next line was already isinstance-guarded,
    which is exactly what made the outer one look safe.
    """

    @staticmethod
    def _report(group_metrics):
        class R:
            timestamp = "2026-09-09T00:00:00"
            data_info = {"n_samples": 100}
            protected_attribute = "gender"
            n_groups = 2
            overall_metrics = {"ece": 0.1, "mce": 0.2, "brier": 0.3}
            disparity_analysis = None
            brier_decomposition = None
            tradeoff_analysis = None
            recommendation = None
            is_well_calibrated = None
            has_significant_disparity = None
            critical_issues: list = []
            recommendations: list = []

        R.group_metrics = group_metrics
        return R()

    def test_a_non_empty_list_does_not_raise(self):
        from vfairness.rendering.adapters import calibration_report_to_svg

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert calibration_report_to_svg(self._report([{"ece": 0.1}]))

    def test_the_shape_is_named(self):
        from vfairness.rendering.adapters import calibration_report_to_svg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibration_report_to_svg(self._report([{"ece": 0.1}]))
        named = [str(w.message) for w in caught if "NOT MEASURED" in str(w.message)]
        assert named, [str(w.message) for w in caught]
        assert "list" in named[0]

    def test_control_an_empty_list_was_always_fine_and_stays_silent(self):
        """`or {}` already handled the falsy case. That behaviour is unchanged,
        which is why an empty list must NOT start warning."""
        from vfairness.rendering.adapters import calibration_report_to_svg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert calibration_report_to_svg(self._report([]))
        assert not [w for w in caught if "NOT MEASURED" in str(w.message)]

    def test_control_a_real_report_renders_its_groups(self):
        from vfairness.rendering.adapters import calibration_report_to_svg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = calibration_report_to_svg(self._report({"a": {"ece": 0.10}, "b": {"ece": 0.40}}))
        assert svg and "<svg" in svg
        assert not [w for w in caught if "NOT MEASURED" in str(w.message)]


def test_the_analyzer_has_no_analyze_method_only_full_analysis():
    """Recorded because a consumer guarded on it with hasattr and therefore
    silently produced no calibration report at all. A hasattr on a method that
    does not exist is a guard that is always False: the feature is skipped and
    nothing says so. The adapter's own docstring names full_analysis()."""
    from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer

    assert not hasattr(CalibrationAnalyzer, "analyze")
    assert hasattr(CalibrationAnalyzer, "full_analysis")
