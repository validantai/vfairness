"""Three states for ``subgroup_robustness_audit`` (Beta Go-Live, Stage 1, g08).

The defect
----------
``subgroup_robustness_audit`` dropped every subgroup it could not measure with a
bare ``continue`` and no tally: cells below ``min_subgroup_size``, and cells
whose metric came back NaN. When every subgroup was dropped the function still
returned a complete verdict::

    gerrymandering_detected=False, worst_disparity=0.0, worst_subgroup=None,
    n_subgroups_analyzed=0, n_subgroups_flagged=0, subgroup_metrics={}

with no warning. That is byte-identical to the result of a real audit that
examined its subgroups and cleared them, and ``worst_disparity=0.0`` is perfect
parity, the strongest all-clear the result can state. Executed on this repo the
library's own renderer turned it into "All robustness checks passed: model shows
stable fairness behaviour."

The coupling is what makes it critical: Kearns gerrymandering lives precisely in
the small intersectional cells the ``min_subgroup_size`` filter discards, so the
filter removed exactly what the capability exists to find and then certified its
absence.

The rule pinned here
--------------------
Three states, never two: detected / not detected / could not check.

* ``gerrymandering_detected is None`` when no subgroup was measured, when some
  were dropped unmeasured and the survivors cleared, or when only one subgroup
  survived (there is no second subgroup to gerrymander against).
* ``worst_disparity`` is NaN, not 0.0, when nothing was measured.
* The drops are counted on the result (``n_subgroups_too_small``,
  ``n_subgroups_unmeasurable``) and named in a ``UserWarning``, so a caller can
  tell a could-not-check from a measurement without reading the source.
* A real finding is NEVER withdrawn by a caveat: if the measured subgroups
  already breach the threshold, ``gerrymandering_detected`` stays True and the
  dropped cells are reported alongside it.

Every assertion below goes through the public entry point ``vfairness
.subgroup_robustness_audit``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness


def _audit(*args, **kwargs):
    """Call the PUBLIC entry point, returning (result, list-of-warning-messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vfairness.subgroup_robustness_audit(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def _healthy_frame(seed=42, n=600):
    """600 rows, four intersectional cells, a planted gap in M_young."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], n)
    age = rng.choice(["young", "old"], n)
    base = rng.random(n)
    y_pred = (base < 0.3).astype(int)
    gap = (sex == "M") & (age == "young")
    y_pred[gap] = (base[gap] < 0.95).astype(int)
    return y_pred, pd.DataFrame({"sex": sex, "age": age})


class TestCouldNotCheck:
    """The unmeasurable inputs must refuse, not clear."""

    def test_every_cell_below_min_size_is_could_not_check(self):
        # The reproduction from the finding: 20 rows, four cells of five, all
        # below min_subgroup_size=30. Before the fix this returned
        # gerrymandering_detected=False, worst_disparity=0.0, no warning.
        sex = np.array(["M", "F"] * 10)
        age = np.array((["young"] * 5 + ["old"] * 5) * 2)
        y_pred = np.random.default_rng(1).integers(0, 2, 20)

        result, messages = _audit(y_pred, pd.DataFrame({"sex": sex, "age": age}))

        assert result.gerrymandering_detected is None, (
            "no subgroup was examined, so the gerrymandering question was not "
            f"answered; got {result.gerrymandering_detected!r}"
        )
        assert result.gerrymandering_detected is not False
        assert np.isnan(result.worst_disparity), (
            "0.0 is perfect parity across every subgroup, and none was measured; "
            f"got {result.worst_disparity!r}"
        )
        assert result.n_subgroups_analyzed == 0
        # The drops are counted, not silent.
        assert result.n_subgroups_too_small == 4
        assert result.n_subgroups_unmeasurable == 0
        assert any("could not check" in m for m in messages), messages
        assert any("min_subgroup_size" in m for m in messages), messages

    def test_the_could_not_check_reaches_a_caller_through_to_dict(self):
        """A dict consumer (the renderer, the Pulse orchestrator) sees it too."""
        sex = np.array(["M", "F"] * 10)
        y_pred = np.zeros(20, dtype=int)

        result, _ = _audit(y_pred, pd.DataFrame({"sex": sex}))
        payload = result.to_dict()

        assert payload["gerrymandering_detected"] is None
        assert np.isnan(payload["worst_disparity"])
        assert payload["n_subgroups_too_small"] == 2
        assert "n_subgroups_unmeasurable" in payload

    def test_fpr_with_no_negative_labels_is_could_not_check(self):
        """The metric has no value for any cell, so the audit has no verdict."""
        n = 200
        sex = np.array(["M", "F"] * (n // 2))
        y_true = np.ones(n, dtype=int)  # not one negative label anywhere
        y_pred = np.random.default_rng(0).integers(0, 2, n)

        with warnings.catch_warnings():
            # numpy's own "Mean of empty slice" is pre-existing noise here.
            warnings.simplefilter("ignore", RuntimeWarning)
            result, messages = _audit(
                y_pred, pd.DataFrame({"sex": sex}), y_true=y_true, metric="fpr"
            )

        assert result.gerrymandering_detected is None
        assert np.isnan(result.worst_disparity)
        assert result.n_subgroups_unmeasurable == 2
        assert any("non-finite 'fpr'" in m for m in messages), messages

    def test_empty_input_is_could_not_check(self):
        empty = np.array([], dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result, messages = _audit(np.array([]), pd.DataFrame({"sex": empty, "age": empty}))

        assert result.gerrymandering_detected is None
        assert np.isnan(result.worst_disparity)
        assert any("could not check" in m for m in messages), messages

    def test_dropped_cells_with_clean_survivors_is_could_not_check(self):
        """The dropped cells are exactly where gerrymandering hides."""
        grp = np.array(["big"] * 390 + ["tiny"] * 10)
        y_pred = (np.random.default_rng(7).random(400) < 0.5).astype(int)

        result, messages = _audit(y_pred, pd.DataFrame({"g": grp}))

        assert result.n_subgroups_flagged == 0
        assert result.n_subgroups_too_small == 1
        assert result.gerrymandering_detected is None, (
            "one cell was never examined, so a clean sweep of the rest is not a clean sweep"
        )
        assert any("dropped unmeasured" in m for m in messages), messages

    def test_a_single_surviving_subgroup_is_could_not_check(self):
        """One cell is the whole population: there is no partition to gerrymander."""
        y_pred = (np.random.default_rng(7).random(100) < 0.4).astype(int)

        result, messages = _audit(y_pred, pd.DataFrame({"g": np.array(["only"] * 100)}))

        assert result.n_subgroups_analyzed == 1
        assert result.gerrymandering_detected is None
        assert any("only one subgroup" in m for m in messages), messages


class TestTheFindingIsNeverDeleted:
    """Removing a fabricated all-clear must not remove a real alarm."""

    def test_a_breach_survives_dropped_subgroups(self):
        grp = np.array(["a"] * 200 + ["b"] * 190 + ["tiny"] * 10)
        y_pred = np.concatenate([np.ones(200, int), np.zeros(190, int), np.ones(10, int)])

        result, messages = _audit(y_pred, pd.DataFrame({"g": grp}))

        assert result.gerrymandering_detected is True, (
            "the measured cells already breach the threshold and the dropped one "
            "could only add more, so the detection stands"
        )
        assert result.n_subgroups_flagged == 2
        assert result.n_subgroups_too_small == 1
        assert not np.isnan(result.worst_disparity)
        assert any("the detection stands" in m for m in messages), messages


class TestControlHealthyDataStillMeasures:
    """A fix that makes everything refuse is a worse defect than the one fixed."""

    def test_healthy_data_still_finds_the_planted_gap(self):
        y_pred, attrs = _healthy_frame()

        result, messages = _audit(y_pred, attrs)

        assert result.gerrymandering_detected is True
        assert result.worst_subgroup == "M_young"
        assert result.worst_disparity == pytest.approx(0.4934, abs=0.01)
        assert result.n_subgroups_analyzed == 4
        assert result.n_subgroups_flagged == 4
        assert result.n_subgroups_too_small == 0
        assert result.n_subgroups_unmeasurable == 0
        assert messages == [], f"a fully measured audit warns about nothing: {messages}"

    def test_a_genuinely_clean_audit_still_returns_false(self):
        """The False state must stay reachable, or None means nothing."""
        # Four cells of 100, each with exactly 40 positives: every disparity is
        # far inside the 0.10 threshold, nothing is dropped.
        cells = [("M", "young"), ("M", "old"), ("F", "young"), ("F", "old")]
        sex, age, y_pred = [], [], []
        for s, a in cells:
            sex += [s] * 100
            age += [a] * 100
            y_pred += [1] * 40 + [0] * 60

        result, messages = _audit(
            np.array(y_pred), pd.DataFrame({"sex": np.array(sex), "age": np.array(age)})
        )

        assert result.gerrymandering_detected is False, (
            "four fully measured, equal-rate cells are a real pass, not a refusal"
        )
        assert result.n_subgroups_analyzed == 4
        assert result.n_subgroups_flagged == 0
        # Never an exact-zero comparison on an accumulated statistic.
        assert abs(result.worst_disparity) < 0.10
        assert messages == [], messages

    def test_relative_disparity_is_none_not_zero_without_a_base_rate(self):
        """A ratio against a zero overall rate is not 0.0, it is unavailable."""
        sex = np.array(["M", "F"] * 100)
        result, _ = _audit(np.zeros(200, dtype=int), pd.DataFrame({"sex": sex}))

        assert set(result.subgroup_metrics) == {"M", "F"}
        for cell in result.subgroup_metrics.values():
            assert cell["relative_disparity"] is None


class TestTheReaderSeesIt:
    """The value is measured honestly AND survives to the surface a person reads."""

    def test_the_renderer_no_longer_prints_the_all_clear(self):
        from dataclasses import asdict

        from vfairness.rendering import adapters_robustness as ar

        sex = np.array(["M", "F"] * 10)
        age = np.array((["young"] * 5 + ["old"] * 5) * 2)
        y_pred = np.random.default_rng(1).integers(0, 2, 20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vfairness.subgroup_robustness_audit(
                y_pred, pd.DataFrame({"sex": sex, "age": age})
            )

        svg = ar.robustness_testing_to_svg(
            [
                {
                    "p_value": 0.9,
                    "is_significant": False,
                    "metric_name": "dpd",
                    "n_permutations": 100,
                }
            ],
            [
                {
                    "perturbation_type": "label_noise",
                    "robustness_score": 1.0,
                    "is_robust": True,
                    "max_deviation": 0.0,
                }
            ],
            asdict(result),
        )

        assert "All robustness checks passed" not in svg, (
            "the subgroup audit checked nothing, so this run does not get the all-clear sentence"
        )
        assert "not checked" in svg
