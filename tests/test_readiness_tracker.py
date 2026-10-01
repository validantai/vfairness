"""R-3: the monitor that monitors nothing, and the monotone inversion.

Two blocking findings in ``operations/monitoring/tracker.py``, both measured by
execution on 2026-09-09 before any edit.

R-3a. ``FairnessMonitor._detect_protected_cols`` returned only columns starting
with ``group_``. There was no auto-detection, although ``update_and_check``
promised protected columns "should start with ``group_`` or be auto-detected",
the class docstring promised metrics "for every protected-attribute column in
the batch", and the library exports ``detect_protected_attributes``. Measured on
one batch of 2946 rows in which one group is never approved::

    column            n_metrics   any_alert   n_warnings
    race                      0       False            0
    gender                    0       False            0
    sex                       0       False            0
    group                     0       False            0
    protected_race            0       False            0
    group_race                2        True            0

``any_alert=False`` is the monitoring product's VERDICT, and five of those six
runs produced it without computing a single fairness metric. Nothing in the
snapshot separated "clean" from "never looked".

R-3b. ``min_samples`` produced a MONOTONE INVERSION. ``_group_positive_rates``
drops groups below the floor and ``compute_disparate_impact`` refused only when
fewer than TWO groups survived, so with two survivors the excluded group simply
vanished and the ratio was taken over the survivors. Measured on 485 white and
485 black at ~0.47 plus a ``native`` group approved 0 of N, min_samples=30,
alert_threshold=0.8, on the correctly named ``group_race`` column::

    native n   true DI   reported DI   alert   any_alert   warnings
           5    0.0000        0.9781   False       False          0
          10    0.0000        0.9781   False       False          0
          12    0.0000        0.9781   False       False          0
          31    0.0000        0.0000    True        True          0

The smaller the victim group, the better the reported number and the quieter the
monitor, and the group that is never selected at all is exactly the one the floor
is most likely to swallow.

Every test below is paired: a refusal pin, and an over-correction control that
asserts MEASURED values exactly, so a fix that simply refuses everything fails
here too.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    WindowMetrics,
)

# ===========================================================================
# Fixtures. Deterministic frames, no randomness, so every number below is exact.
# ===========================================================================


def _unfair_batch(group_col: str, n_victim: int = 46) -> pd.DataFrame:
    """A window in which one group is NEVER selected.

    600 rows in 'white' selected 1-in-2, 600 in 'black' selected 1-in-2, and
    ``n_victim`` rows in 'native' selected never. The true disparate impact over
    the whole window is 0.0 and the true demographic parity gap is 0.5.
    """
    rows = []
    for grp in ("white", "black"):
        rows.append(
            pd.DataFrame(
                {
                    "prediction": [1, 0] * 300,
                    "label": [1, 0] * 300,
                    group_col: grp,
                }
            )
        )
    rows.append(
        pd.DataFrame(
            {
                "prediction": [0] * n_victim,
                "label": [1, 0] * (n_victim // 2) + [1] * (n_victim % 2),
                group_col: "native",
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


def _clean_batch(group_col: str = "group_gender") -> pd.DataFrame:
    """Both groups selected exactly 1-in-2: DI 1.0, DP gap 0.0, nothing breaches."""
    return pd.DataFrame(
        {
            "prediction": [1, 0] * 100,
            "label": [1, 0] * 100,
            group_col: ["A"] * 100 + ["B"] * 100,
        }
    )


def _monitor(**cfg) -> FairnessMonitor:
    return FairnessMonitor(config=FairnessMonitorConfig(window_size=5000, **cfg))


def _run(monitor: FairnessMonitor, batch: pd.DataFrame):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        snap = monitor.update_and_check(batch)
    return snap, [str(w.message) for w in caught]


# ===========================================================================
# R-3a. A run that computed nothing is not a clean run.
# ===========================================================================


class TestR3aProtectedColumnsAreDetected:
    NAMES = ["race", "gender", "sex", "group", "protected_race", "group_race"]

    @pytest.mark.parametrize("group_col", NAMES)
    def test_the_column_name_does_not_decide_whether_anyone_looks(self, group_col):
        """REFUSAL PIN. The measured defect: only 'group_race' was examined.

        The same unfairness under six honest column names must produce the same
        finding, not silence under five of them.
        """
        snap, _ = _run(_monitor(), _unfair_batch(group_col))

        assert snap.metrics, f"'{group_col}' produced no metric at all"
        di = snap.metrics[f"disparate_impact_{group_col}"]
        dp = snap.metrics[f"demographic_parity_{group_col}"]
        # Exact, not "in a plausible band": 'native' is selected 0 of 46 and
        # both other groups exactly 1-in-2.
        assert di == pytest.approx(0.0)
        assert dp == pytest.approx(0.5)
        assert snap.alerts[f"disparate_impact_{group_col}"] is True
        assert snap.any_alert is True
        assert snap.group_rates[group_col] == {
            "black": pytest.approx(0.5),
            "native": pytest.approx(0.0),
            "white": pytest.approx(0.5),
        }

    def test_a_window_that_computed_nothing_reports_none_not_false(self):
        """REFUSAL PIN. ``any_alert`` was ``any({}.values())``, i.e. ``False``.

        A batch with no protected attribute in it computes nothing, and the
        headline verdict for that run must be could-not-check.
        """
        featureless = pd.DataFrame(
            {
                "prediction": [1, 0] * 100,
                "label": [1, 0] * 100,
                "loan_amount": np.arange(200),
            }
        )
        snap, caught = _run(_monitor(), featureless)

        assert snap.metrics == {}
        assert snap.alerts == {}
        assert snap.any_alert is None, (
            f"a window that computed zero fairness metrics reported any_alert={snap.any_alert!r}"
        )
        assert any("no protected-attribute column was found" in w for w in caught), caught
        assert any("never False" in w for w in caught), caught

    def test_none_is_not_reached_by_a_falsy_read(self):
        """``None`` and ``False`` are both falsy, so identity is the assertion.

        A reader doing ``if snap.any_alert:`` cannot tell them apart, which is
        why the pin above uses ``is None`` and this one states the difference.
        """
        nothing = WindowMetrics(
            batch_id="b",
            timestamp=pd.Timestamp("2026-09-09").to_pydatetime(),
            sample_count=0,
            metrics={},
            group_rates={},
            alerts={},
        )
        clean = WindowMetrics(
            batch_id="b",
            timestamp=pd.Timestamp("2026-09-09").to_pydatetime(),
            sample_count=200,
            metrics={"disparate_impact_group_g": 1.0},
            group_rates={},
            alerts={"disparate_impact_group_g": False},
        )
        assert nothing.any_alert is None
        assert clean.any_alert is False
        assert nothing.any_alert is not clean.any_alert

    def test_over_correction_control_a_clean_batch_still_reports_false(self):
        """OVER-CORRECTION CONTROL. Exact values, not a membership test.

        A fix that turned every verdict into ``None``, or that started warning
        on healthy windows, fails here.
        """
        snap, caught = _run(_monitor(), _clean_batch())

        assert snap.metrics == {
            "disparate_impact_group_gender": pytest.approx(1.0),
            "demographic_parity_group_gender": pytest.approx(0.0),
        }
        assert snap.alerts == {
            "disparate_impact_group_gender": False,
            "demographic_parity_group_gender": False,
        }
        assert snap.any_alert is False
        assert snap.excluded_groups == {}
        assert caught == []

    def test_predictions_and_labels_are_never_taken_as_protected_columns(self):
        """OVER-CORRECTION CONTROL. Detection must not monitor the model's own
        output columns, which are binary and low-cardinality and would otherwise
        be prime candidates."""
        snap, _ = _run(_monitor(), _clean_batch())
        monitored = set(snap.group_rates)
        assert monitored == {"group_gender"}, monitored


# ===========================================================================
# R-3b. An aggregate over a subset of the groups carries that fact.
# ===========================================================================


class TestR3bUndersampledGroupsAreNotSilentlyDropped:
    @pytest.mark.parametrize("n_victim", [5, 10, 12, 29])
    def test_a_dropped_group_makes_the_ratio_unmeasurable_not_flattering(self, n_victim):
        """REFUSAL PIN, and the inversion itself.

        Before the fix every one of these sizes reported disparate_impact=0.9781
        with alert=False while the true ratio over the window was 0.0.
        """
        snap, caught = _run(
            _monitor(min_samples=30, alert_threshold=0.8),
            _unfair_batch("group_race", n_victim=n_victim),
        )

        di = snap.metrics["disparate_impact_group_race"]
        dp = snap.metrics["demographic_parity_group_race"]
        assert math.isnan(di), f"reported {di!r} with a group excluded from it"
        assert math.isnan(dp), f"reported {dp!r} with a group excluded from it"

        # No alert determination was possible, so there is no alerts entry and
        # no verdict for the window.
        assert "disparate_impact_group_race" not in snap.alerts
        assert snap.alerts == {}
        assert snap.any_alert is None

        # The excluded group is NAMED and COUNTED on the window.
        assert snap.excluded_groups == {"group_race": {"native": n_victim}}
        assert snap.to_dict()["excluded_groups"] == {"group_race": {"native": n_victim}}

        # And named, counted and called out as never selected in the warning.
        text = " ".join(caught)
        assert "native" in text and f"n={n_victim}" in text, caught
        assert "NEVER selected" in text, caught
        assert "min_samples" in text, caught

    def test_the_reported_number_no_longer_improves_as_the_victim_shrinks(self):
        """REFUSAL PIN on the MONOTONICITY, which is the heart of the defect.

        Shrinking the group that is never selected must never make the reported
        disparate impact better. It used to move 0.0 to 0.9781.
        """
        reported = {}
        for n_victim in (5, 12, 29, 30, 46):
            snap, _ = _run(
                _monitor(min_samples=30, alert_threshold=0.8),
                _unfair_batch("group_race", n_victim=n_victim),
            )
            reported[n_victim] = snap.metrics["disparate_impact_group_race"]

        # Below the floor: unmeasurable. At or above it: measured, and 0.0.
        assert math.isnan(reported[5]) and math.isnan(reported[12]) and math.isnan(reported[29])
        assert reported[30] == pytest.approx(0.0)
        assert reported[46] == pytest.approx(0.0)
        # Nothing in the family reads better than the measured truth.
        measured = [v for v in reported.values() if not math.isnan(v)]
        assert max(measured) == pytest.approx(0.0), reported

    def test_label_metrics_carry_the_same_exclusion(self):
        """REFUSAL PIN. equalized_odds and equal_opportunity skip groups under
        the same floor, so the same lower-bound-as-measurement applies to them.
        """
        snap, _ = _run(
            _monitor(
                min_samples=30,
                metrics_to_track=["equalized_odds", "equal_opportunity"],
            ),
            _unfair_batch("group_race", n_victim=10),
        )
        assert math.isnan(snap.metrics["equalized_odds_group_race"])
        assert math.isnan(snap.metrics["equal_opportunity_group_race"])
        assert snap.alerts == {}
        assert snap.any_alert is None

    def test_over_correction_control_a_fully_sampled_window_is_measured(self):
        """OVER-CORRECTION CONTROL. Exact numbers, and silence.

        Every group clears the floor here, so all four metrics are real values,
        the window carries no exclusions and nothing warns. A fix that refused
        whenever groups differ in size, or that warned on every window, fails.
        """
        snap, caught = _run(
            _monitor(
                min_samples=30,
                metrics_to_track=[
                    "disparate_impact",
                    "demographic_parity",
                    "equalized_odds",
                    "equal_opportunity",
                ],
            ),
            _unfair_batch("group_race", n_victim=46),
        )

        assert snap.metrics["disparate_impact_group_race"] == pytest.approx(0.0)
        assert snap.metrics["demographic_parity_group_race"] == pytest.approx(0.5)
        # 'native' is never predicted positive, so its TPR is 0.0 against 1.0
        # for the other two groups: the gap is exactly 1.0 on both arms.
        assert snap.metrics["equalized_odds_group_race"] == pytest.approx(1.0)
        assert snap.metrics["equal_opportunity_group_race"] == pytest.approx(1.0)
        assert snap.alerts == {
            "disparate_impact_group_race": True,
            "demographic_parity_group_race": True,
            "equalized_odds_group_race": True,
            "equal_opportunity_group_race": True,
        }
        assert snap.any_alert is True
        assert snap.excluded_groups == {}
        assert caught == []

    def test_over_correction_control_a_small_group_that_clears_the_floor(self):
        """OVER-CORRECTION CONTROL. min_samples=5 measures the n=5 group.

        The floor is the only thing that should decide, so the identical frame
        that refuses at min_samples=30 must produce the exact number at 5.
        """
        snap, caught = _run(
            _monitor(min_samples=5, alert_threshold=0.8),
            _unfair_batch("group_race", n_victim=5),
        )
        assert snap.metrics["disparate_impact_group_race"] == pytest.approx(0.0)
        assert snap.metrics["demographic_parity_group_race"] == pytest.approx(0.5)
        assert snap.alerts["disparate_impact_group_race"] is True
        assert snap.any_alert is True
        assert snap.excluded_groups == {}
        assert caught == []


# ===========================================================================
# The two findings meet: the honest column name AND the tiny victim group.
# ===========================================================================


def test_the_two_defects_together_produced_the_quietest_possible_run():
    """A batch named 'race' with a 5-row group that is never selected hit both
    defects at once and reported metrics={}, alerts={}, any_alert=False.

    It now computes the column, refuses the aggregate, names the group and
    reports no verdict.
    """
    snap, caught = _run(
        _monitor(min_samples=30, alert_threshold=0.8),
        _unfair_batch("race", n_victim=5),
    )
    assert set(snap.metrics) == {"disparate_impact_race", "demographic_parity_race"}
    assert all(math.isnan(v) for v in snap.metrics.values())
    assert snap.any_alert is None
    assert snap.excluded_groups == {"race": {"native": 5}}
    assert any("native" in w and "NEVER selected" in w for w in caught), caught


# ===========================================================================
# R-3c. Sibling of the same shape, found while measuring the two above: the
# cooldown rewrote a MEASURED breach as a clean reading.
# ===========================================================================


def _breaching_batch() -> pd.DataFrame:
    """Group A selected 90 of 100, group B 10 of 100: DI 0.1111, DP gap 0.8."""
    return pd.DataFrame(
        {
            "prediction": [1] * 90 + [0] * 10 + [1] * 10 + [0] * 90,
            "label": [1] * 100 + [0] * 100,
            "group_g": ["A"] * 100 + ["B"] * 100,
        }
    )


class TestR3cCooldownSuppressesNotificationNotTheReading:
    """Measured 2026-09-09, one batch pushed through the same monitor 3 times::

        window 1: DI=0.1111 alert=True  any_alert=True
        window 2: DI=0.1111 alert=False any_alert=False
        window 3: DI=0.1111 alert=False any_alert=False

    ``alerts[key] = breached and self._should_alert(key)`` folded the cooldown
    into the reading, so a disparity that never moved reported clean for the
    whole cooldown hour.
    """

    KEY = "disparate_impact_group_g"

    def test_a_repeat_breach_is_still_a_breach(self):
        """REFUSAL PIN."""
        monitor = _monitor(metrics_to_track=["disparate_impact"])
        batch = _breaching_batch()
        seen = []
        for _ in range(3):
            snap, _ = _run(monitor, batch)
            seen.append((snap.metrics[self.KEY], snap.alerts[self.KEY], snap.any_alert))

        for di, alert, any_alert in seen:
            assert di == pytest.approx(0.1111, abs=1e-4)
            assert alert is True, seen
            assert any_alert is True, seen
        assert monitor.get_alert_summary() == {self.KEY: 3}

    def test_the_cooldown_still_works_it_just_says_so_out_loud(self):
        """OVER-CORRECTION CONTROL. The rate limiter must not be deleted.

        The first window notifies, the next two are marked suppressed, and a
        zero cooldown suppresses nothing.
        """
        monitor = _monitor(metrics_to_track=["disparate_impact"])
        batch = _breaching_batch()
        first, _ = _run(monitor, batch)
        second, _ = _run(monitor, batch)
        third, _ = _run(monitor, batch)

        assert first.suppressed_alerts == []
        assert second.suppressed_alerts == [self.KEY]
        assert third.suppressed_alerts == [self.KEY]
        assert third.to_dict()["suppressed_alerts"] == [self.KEY]

        eager = _monitor(metrics_to_track=["disparate_impact"], alert_cooldown_seconds=0.0)
        for _ in range(3):
            snap, _ = _run(eager, batch)
            assert snap.alerts[self.KEY] is True
            assert snap.suppressed_alerts == []

    def test_over_correction_control_repeats_of_a_clean_batch_stay_clean(self):
        """OVER-CORRECTION CONTROL. Exact values on every repeat."""
        monitor = _monitor(metrics_to_track=["disparate_impact", "demographic_parity"])
        for _ in range(3):
            snap, caught = _run(monitor, _clean_batch())
            assert snap.metrics["disparate_impact_group_gender"] == pytest.approx(1.0)
            assert snap.alerts == {
                "disparate_impact_group_gender": False,
                "demographic_parity_group_gender": False,
            }
            assert snap.any_alert is False
            assert snap.suppressed_alerts == []
            assert caught == []
        assert monitor.get_alert_summary() == {}
